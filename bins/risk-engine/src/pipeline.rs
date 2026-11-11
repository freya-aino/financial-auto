//! The I/O half of the automatic strategy lifecycle (spec 35 §6.4) and of holdout locking
//! (spec 41 §6.4). Decisions are pure (`fa_risk::pipeline`, `fa_backtest::holdout`); this
//! module reads Garage, Postgres, VictoriaMetrics and the bus, and applies them.
//!
//! One round (every `pipeline.interval_secs`, on `ctl.risk.pipeline {action: run}` and at start):
//!
//! 1. **Holdouts**: every configured timescale (and every candidate's) gets a locked holdout:
//!    `plan_lock` → the segment must hold `holdout.min_bars` bars in VictoriaMetrics → an
//!    idempotent insert (advisory-locked, overlap-checked) audited as `validation.holdout.lock`.
//!    Missing ones are reported in the status and the `holdout.missing` gauge.
//! 2. **Discovery**: the newest `generations_scanned` strategy generations in
//!    `fa-artifacts/strategies/` (written by the orchestrator's promote hook) are read once;
//!    their members get selection evidence from `validation.trial` / `generation_stats` and
//!    pass `select_candidates` → new `candidate` rows (`ledger.strategy_state`).
//! 3. **Per strategy**: `decide` on the evidence (holdout trial, gate checklist, live slots,
//!    strategy-runtime auto-pauses, live consistency); `RequestHoldout` publishes one
//!    `evo.eval.strategy.holdout` job on `JOBS_EVO`; transitions are applied and audited
//!    (`strategy.<transition>`, actor `system:pipeline`).
//! 4. **Live set**: when membership changed (or the stored file differs) the managed live set
//!    `fa-artifacts/live/strategies.toml` is rewritten and strategy-runtime reloaded with
//!    `ctl.strategy.reload {artifact_key: managed}`.
//!
//! 5. **Snapshot changes** (spec 35 §6.5): strategy-runtime reports per strategy whether its
//!    region literals were remapped onto a new encoder snapshot (`snapshot.remapped_ms`) or are
//!    unmappable (pause code 4); an **active** strategy promoted before that remap, or suspended,
//!    goes back to `shadow` (`strategy.snapshot_shadow`) and must pass the gate again. A
//!    suspended shadow strategy stays suspended (flat) — not a runtime auto-pause, so it is not
//!    retired for it.
//! 6. **Helix mirror** (spec 35 §5): every evolved strategy's status (incl. `retired`) is
//!    mirrored onto its `Genome` node (`fa_hub::set_genome_status`, created when missing);
//!    best-effort — Helix down is counted (`helix.mirror_failed`) and retried next round, since
//!    the round reconciles every row against what was last mirrored. Hand-written baselines are
//!    not genomes (D-040) and are not mirrored.
//!
//! Failed holdout jobs (`jobs.result.evo.eval.strategy.holdout` with a non-OK status) are noted
//! on the strategy and retried next round up to `holdout_max_attempts`.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Duration,
};

use fa_backtest::holdout::{LockPlan, plan_lock};
use fa_blob::{BlobStore, ObjectKey, S3BlobStore, buckets};
use fa_bus::Bus;
use fa_core::Timescale;
use fa_proto::{
    Message,
    v1::{CtlReply, CtlRequest, Job, JobResult, job_result},
};
use fa_risk::{
    pipeline::{Decision, FrontMember, Observation, PipelinePolicy, Selection, decide, select_candidates},
    strategy::{StrategyAction, StrategyStatus},
};
use fa_strategies::{
    genome::StrategyGenome,
    population::{LIVE_SET_KEY, StrategyFile},
};
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::{
    Shared,
    engine::Evidence,
    persist::Persist,
    settings::{HoldoutSettings, Settings},
};

/// Audit actor of pipeline actions.
pub const ACTOR: &str = "system:pipeline";
/// Job kind of a promotion replay (evo-worker `holdout::KIND`).
pub const HOLDOUT_JOB_KIND: &str = "evo.eval.strategy.holdout";
const STRATEGIES_PREFIX: &str = "strategies/";

pub struct Pipeline {
    pub bus: Bus,
    pub store: Arc<Persist>,
    pub blob: Option<S3BlobStore>,
    pub tsdb: Option<fa_tsdb::Tsdb>,
    pub engine: Shared,
    pub policy: PipelinePolicy,
    pub holdout: HoldoutSettings,
    /// Generation keys already read (in memory; a restart re-reads the newest ones, idempotently).
    seen: Mutex<BTreeSet<String>>,
    /// Live-set text last written (None = not yet compared with Garage).
    live_written: Mutex<Option<String>>,
    /// Last round's status (`ctl.risk.pipeline {action: status}`).
    status: Mutex<Value>,
    /// Serialises rounds (timer, ctl, start).
    round_lock: Mutex<()>,
    /// Latest rejected front members across rounds (a generation is read once).
    rejected_log: Mutex<std::collections::VecDeque<Value>>,
    /// Helix client for the genome status mirror (None = not mirrored).
    hub: Option<fa_hub::HubClient>,
    /// `strategy id → (status, since_ms)` last mirrored successfully.
    mirrored: Mutex<BTreeMap<String, (String, i64)>>,
    /// Promotion reviews of shadow strategies: on gate change, else hourly (spec 40 §5).
    reviews: Mutex<crate::engine::history::ReviewDedup>,
    review_every_ms: i64,
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

/// `{ id = "gp-…"` → `gp-…`.
pub fn genome_id_of(candidate: &str) -> Option<&str> {
    let rest = candidate.trim_start().strip_prefix('{')?.trim_start().strip_prefix("id")?.trim_start();
    let rest = rest.strip_prefix('=')?.trim_start().strip_prefix('"')?;
    rest.split('"').next()
}

impl Pipeline {
    pub fn new(
        bus: Bus,
        store: Arc<Persist>,
        blob: Option<S3BlobStore>,
        tsdb: Option<fa_tsdb::Tsdb>,
        engine: Shared,
        settings: &Settings,
    ) -> Self {
        Self {
            bus,
            store,
            blob,
            tsdb,
            engine,
            policy: settings.pipeline.clone(),
            holdout: settings.holdout.clone(),
            seen: Mutex::new(BTreeSet::new()),
            live_written: Mutex::new(None),
            status: Mutex::new(json!({"rounds": 0})),
            round_lock: Mutex::new(()),
            rejected_log: Mutex::new(std::collections::VecDeque::new()),
            hub: None,
            mirrored: Mutex::new(BTreeMap::new()),
            reviews: Mutex::new(Default::default()),
            review_every_ms: settings.history.review_unchanged_secs as i64 * 1000,
        }
    }

    /// Enables the Helix genome status mirror (spec 35 §5).
    pub fn with_hub(mut self, hub: Option<fa_hub::HubClient>) -> Self {
        self.hub = hub;
        self
    }

    pub async fn status(&self) -> Value {
        self.status.lock().await.clone()
    }

    // ---------------------------------------------------------------- holdouts (spec 41 §6.4)

    /// Bars of `ts` the best-covered instrument has in `[start, end]` (VictoriaMetrics).
    async fn coverage(&self, ts: Timescale, start_ms: i64, end_ms: i64) -> Result<usize, String> {
        let Some(vm) = &self.tsdb else { return Err("no VictoriaMetrics configured".into()) };
        let series = vm
            .trade_prices(None, start_ms / 1000, end_ms / 1000, fa_tsdb::step_of(ts))
            .await
            .map_err(|e| format!("victoriametrics: {e}"))?;
        Ok(series.iter().map(|s| s.points.len()).max().unwrap_or(0))
    }

    /// Ensures a locked holdout per configured timescale plus `extra`; returns the status per
    /// timescale and the id of the latest holdout per timescale.
    pub async fn ensure_holdouts(&self, extra: &[Timescale]) -> (Value, Vec<(String, fa_backtest::replay::Holdout)>) {
        let mut want: Vec<Timescale> = self.holdout.timescales();
        for t in extra {
            if !want.contains(t) {
                want.push(*t);
            }
        }
        let mut status = serde_json::Map::new();
        let mut all: Vec<(String, fa_backtest::replay::Holdout)> = Vec::new();
        let rows = match self.store.holdouts().await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "holdouts not readable");
                return (json!({"error": e.to_string()}), all);
            }
        };
        let existing: Vec<fa_backtest::replay::Holdout> = rows.iter().map(|(_, h)| h.clone()).collect();
        let mut missing = 0usize;
        for ts in want {
            let policy = self.holdout.policy(ts);
            let plan = plan_lock(ts, &policy, now_ms(), &existing);
            let entry = match plan {
                LockPlan::Locked(h) => {
                    json!({"locked": true, "start_ms": h.start_ms, "end_ms": h.end_ms, "embargo_ms": h.embargo_ms.unwrap_or(policy.embargo_ms)})
                }
                LockPlan::Lock { start_ms, end_ms, relock, embargo_ms } if self.holdout.auto_lock => {
                    match self.coverage(ts, start_ms, end_ms).await {
                        Ok(bars) if bars >= policy.min_bars => {
                            let note = format!(
                                "auto-lock by risk-engine (spec 41 §6.4): latest completed {} of {ts}, {bars} bars{}",
                                humanize(end_ms - start_ms),
                                if relock { ", re-lock" } else { "" }
                            );
                            match self.store.lock_holdout(ts.as_str(), start_ms, end_ms, embargo_ms, &note).await {
                                Ok(Some(id)) => {
                                    tracing::warn!(%id, %ts, start_ms, end_ms, bars, "holdout locked automatically");
                                    self.store
                                        .audit_as(
                                            ACTOR,
                                            "validation.holdout.lock",
                                            Some(&id),
                                            json!({"timescale": ts.as_str(), "start_ms": start_ms, "end_ms": end_ms, "embargo_ms": embargo_ms, "bars": bars, "relock": relock, "note": note}),
                                        )
                                        .await;
                                    json!({"locked": true, "new": true, "id": id, "start_ms": start_ms, "end_ms": end_ms, "embargo_ms": embargo_ms, "bars": bars})
                                }
                                Ok(None) => {
                                    json!({"locked": false, "reason": "overlaps a concurrently locked row; re-read next round"})
                                }
                                Err(e) => {
                                    missing += 1;
                                    json!({"locked": false, "reason": format!("insert failed: {e}")})
                                }
                            }
                        }
                        Ok(bars) => {
                            missing += 1;
                            json!({"locked": false, "reason": format!("only {bars} of {} {ts} bars in [{start_ms}, {end_ms}]", policy.min_bars)})
                        }
                        Err(e) => {
                            missing += 1;
                            json!({"locked": false, "reason": e})
                        }
                    }
                }
                LockPlan::Lock { .. } => {
                    missing += 1;
                    json!({"locked": false, "reason": "holdout.auto_lock = false: lock it via POST /api/validation/holdout"})
                }
                LockPlan::Wait(r) => {
                    missing += usize::from(!existing.iter().any(|h| h.timescale == ts));
                    json!({"locked": existing.iter().any(|h| h.timescale == ts), "reason": r})
                }
            };
            if entry["locked"] == false {
                tracing::warn!(%ts, reason = %entry["reason"], "no locked holdout");
            }
            status.insert(ts.as_str().to_owned(), entry);
        }
        // every row after any insert (ordered by timescale, start)
        if let Ok(rows) = self.store.holdouts().await {
            all = rows;
        }
        self.engine.lock().await.counters.insert("holdout.missing", missing as f64);
        (Value::Object(status), all)
    }

    // ------------------------------------------------------------ discovery (spec 35 §6.4)

    async fn discover(&self, rejected: &mut Vec<Value>) -> usize {
        let Some(blob) = &self.blob else { return 0 };
        let list = match blob.list(buckets::ARTIFACTS, STRATEGIES_PREFIX).await {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!(error = %e, "listing strategy generations failed");
                return 0;
            }
        };
        let mut keys: Vec<String> = list.into_iter().map(|m| m.key).filter(|k| k.ends_with(".toml")).collect();
        keys.sort();
        let keys: Vec<String> = keys.into_iter().rev().take(self.policy.generations_scanned.max(1)).collect();
        let mut admitted = 0usize;
        for key in keys {
            if self.seen.lock().await.contains(&key) {
                continue;
            }
            let generation = key.trim_start_matches(STRATEGIES_PREFIX).trim_end_matches(".toml").to_owned();
            let bytes = match blob.get(&ObjectKey { bucket: buckets::ARTIFACTS, key: key.clone() }).await {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(%key, error = %e, "generation file not readable");
                    continue;
                }
            };
            let file = match String::from_utf8(bytes)
                .map_err(|e| e.to_string())
                .and_then(|s| StrategyFile::parse(&s).map_err(|e| e.to_string()))
            {
                Ok(f) => f,
                Err(e) => {
                    tracing::warn!(%key, error = %e, "generation file invalid; skipped");
                    self.seen.lock().await.insert(key);
                    continue;
                }
            };
            let genomes: Vec<StrategyGenome> =
                file.strategy.iter().filter_map(|d| StrategyGenome::from_def(d).ok()).collect();
            let selection = self.store.front_selection(&generation).await;
            let pbo = self.store.generation_pbo(&generation).await;
            let members: Vec<FrontMember> = genomes
                .iter()
                .map(|g| {
                    let def = g.def();
                    FrontMember {
                        strategy_id: def.id.clone(),
                        genome: g.to_string(),
                        generation: generation.clone(),
                        timescale: def.timescale.as_str().to_owned(),
                        step_secs: def.step_secs() as f64,
                        score: None,
                        selection: selection
                            .get(&def.id)
                            .cloned()
                            .map_or_else(|| Selection { pbo, ..Default::default() }, |s| Selection { pbo, ..s }),
                    }
                })
                .collect();
            let chosen = {
                let e = self.engine.lock().await;
                let (chosen, rej) = select_candidates(&self.policy, &members, pbo, |id| e.known_strategy(id));
                for (id, why) in rej {
                    tracing::info!(strategy = %id, %generation, reason = %why, "front member not selected");
                    rejected.push(json!({"generation": generation, "strategy_id": id, "reason": why}));
                }
                chosen
            };
            for m in chosen {
                let added = self.engine.lock().await.admit_candidate(&m, now_ms());
                if added {
                    admitted += 1;
                    tracing::info!(strategy = %m.strategy_id, %generation, dsr = ?m.selection.deflated_sharpe, pbo = ?pbo, "strategy candidate selected");
                    self.store
                        .audit_as(
                            ACTOR,
                            "strategy.candidate",
                            Some(&m.strategy_id),
                            json!({"generation": generation, "selection": m.selection, "genome": m.genome}),
                        )
                        .await;
                }
            }
            self.seen.lock().await.insert(key);
        }
        admitted
    }

    // ------------------------------------------------------------------- runtime status

    /// strategy-runtime's `ctl.strategy.status` JSON.
    async fn runtime_status(&self) -> Option<Value> {
        let req = CtlRequest { verb: "status".into(), args: Default::default() };
        let msg = tokio::time::timeout(
            Duration::from_secs(3),
            self.bus.client().request(fa_bus::subjects::ctl("strategy", "status"), req.encode_to_vec().into()),
        )
        .await
        .ok()?
        .ok()?;
        let reply = CtlReply::decode(msg.payload.as_ref()).ok()?;
        serde_json::from_str(&reply.message).ok()
    }

    /// `strategy_id → auto-pause reason` (`budget` / `nan`) from `ctl.strategy.status`.
    fn runtime_pauses(v: &Value) -> Option<BTreeMap<String, String>> {
        Some(
            v["strategies"]
                .as_array()?
                .iter()
                .filter_map(|s| {
                    let code = s["paused_reason_code"].as_f64().unwrap_or(0.0);
                    let why = match code as i64 {
                        2 => "budget",
                        3 => "nan",
                        _ => return None,
                    };
                    Some((s["id"].as_str()?.to_owned(), why.to_owned()))
                })
                .collect(),
        )
    }

    // ---------------------------------------------------------------------- holdout job

    async fn request_holdout(&self, id: &str, genome: &str, holdout_id: &str) -> Result<String, String> {
        let job_id = fa_core::JobId::new().0.to_string();
        let spec = json!({"candidate": genome, "holdout_id": holdout_id, "snapshot": Value::Null});
        let job = Job {
            job_id: job_id.clone(),
            kind: HOLDOUT_JOB_KIND.into(),
            parent_job_id: format!("holdout-{holdout_id}"),
            priority: 0,
            snapshot_id: String::new(),
            genome_id: id.to_owned(),
            spec: serde_json::to_vec(&spec).unwrap_or_default(),
            input_refs: vec![],
            deadline: None,
            budget_gpu_ms: 0,
            checkpoint_ref: String::new(),
            seed: 0,
        };
        let subject = fa_bus::subjects::jobs(HOLDOUT_JOB_KIND);
        let js = self.bus.jetstream();
        let ack = js
            .publish(subject.clone(), job.encode_to_vec().into())
            .await
            .map_err(|e| format!("publish {subject}: {e}"))?;
        ack.await.map_err(|e| format!("ack {subject}: {e}"))?;
        Ok(job_id)
    }

    /// A finished holdout job: a non-OK result is noted on its strategy and frees the retry.
    pub async fn on_job_result(&self, r: &JobResult) {
        let ok = r.status == job_result::Status::Ok as i32;
        let mut e = self.engine.lock().await;
        let Some(id) = e.pipeline_ids().into_iter().find(|id| {
            e.strategy(id).and_then(|s| s.pipeline.as_ref()).and_then(|p| p.holdout_job_id.as_deref())
                == Some(r.job_id.as_str())
        }) else {
            return;
        };
        if !ok {
            let err = if r.error.is_empty() { format!("status {}", r.status) } else { r.error.clone() };
            tracing::warn!(strategy = %id, job = %r.job_id, error = %err, "holdout evaluation failed");
            e.holdout_job_failed(&id, &err);
        }
    }

    // ------------------------------------------------------------------------- the round

    /// One pipeline round; returns its status.
    pub async fn round(&self) -> Value {
        let _guard = self.round_lock.lock().await;
        let started = now_ms();
        let mut rejected = Vec::new();
        // timescales of strategies waiting for a holdout
        let extra: Vec<Timescale> = {
            let e = self.engine.lock().await;
            e.pipeline_ids()
                .iter()
                .filter_map(|id| e.strategy(id))
                .filter(|s| s.status == StrategyStatus::Candidate)
                .filter_map(|s| s.pipeline.as_ref().and_then(|p| p.timescale.parse().ok()))
                .collect()
        };
        let (holdouts, rows) = self.ensure_holdouts(&extra).await;
        if !self.policy.enabled {
            let st = json!({"enabled": false, "holdouts": holdouts, "at_ms": started});
            *self.status.lock().await = st.clone();
            return st;
        }
        let discovered = self.discover(&mut rejected).await;
        let rejected_recent: Vec<Value> = {
            let mut log = self.rejected_log.lock().await;
            log.extend(rejected.iter().cloned());
            while log.len() > 100 {
                log.pop_front();
            }
            log.iter().rev().take(50).cloned().collect()
        };
        let runtime = self.runtime_status().await;
        let pauses = runtime.as_ref().and_then(Self::runtime_pauses);
        let mut actions = Vec::new();
        if let Some(v) = &runtime {
            actions.extend(self.snapshot_shadow(v).await);
        }
        let mut waiting = serde_json::Map::new();
        let ids = self.engine.lock().await.pipeline_ids();
        for id in ids {
            let now = now_ms();
            let Some(st) = self.engine.lock().await.strategy(&id).cloned() else { continue };
            let Some(rec) = st.pipeline.clone() else { continue };
            let ts: Option<Timescale> = rec.timescale.parse().ok();
            // the holdout already requested, else the newest one of its timescale that its
            // selection never touched (spec 41 §6.4: used only for promotion)
            let clean: Option<String> = rows
                .iter()
                .rev()
                .find(|(_, h)| Some(h.timescale) == ts && rec.selection.clean_for(h.start_ms, h.end_ms))
                .map(|(id, _)| id.clone());
            let holdout_id = rec.holdout_id.clone().or_else(|| clean.clone());
            let holdout_result = match (&holdout_id, st.status) {
                (Some(h), StrategyStatus::Candidate) => self.store.holdout_outcome(&id, &rec.genome, h).await,
                _ => None,
            };
            let ev: Evidence = if matches!(st.status, StrategyStatus::Shadow | StrategyStatus::Active) {
                self.store.evidence(&[&id], Some(&rec.genome), Some(&rec.generation)).await
            } else {
                Evidence::default()
            };
            let (gate, live_slot_free, live_consistent) = {
                let e = self.engine.lock().await;
                let gate = (st.status == StrategyStatus::Shadow).then(|| e.checklist(&id, &ev, now)).flatten();
                (gate, e.live_evolved().len() < self.policy.max_live_evolved, e.live_consistent(&id))
            };
            if let Some(g) = gate.as_ref() {
                let due = self.reviews.lock().await.due(&id, g, now, self.review_every_ms);
                if let Some(trigger) = due {
                    let evidence = json!({"deflated_sharpe": ev.deflated_sharpe, "pbo": ev.pbo});
                    let status = Some(st.status.as_str());
                    crate::risk_io::review(&self.store, &id, trigger, ACTOR, status, status, g, evidence).await;
                }
            }
            let paused = pauses.as_ref().and_then(|p| p.get(&id)).map(String::as_str);
            let obs = Observation {
                now_ms: now,
                holdout_id: holdout_id.as_deref(),
                holdout_result,
                live_slot_free,
                gate: gate.as_ref(),
                runtime_auto_paused: paused,
                live_consistent,
            };
            let d = decide(&self.policy, &st, &obs);
            match &d {
                Decision::Nothing => {}
                Decision::Wait(r) => {
                    self.engine.lock().await.pipeline_note(&id, r);
                    waiting.insert(id.clone(), json!(r));
                }
                Decision::RequestHoldout { holdout_id } => {
                    match self.request_holdout(&id, &rec.genome, holdout_id).await {
                        Ok(job_id) => {
                            self.engine.lock().await.record_holdout_request(&id, holdout_id, &job_id, now);
                            self.store
                            .audit_as(
                                ACTOR,
                                "validation.holdout.evaluate",
                                Some(&job_id),
                                json!({"strategy_id": id, "holdout_id": holdout_id, "candidate": rec.genome, "kind": HOLDOUT_JOB_KIND}),
                            )
                            .await;
                            tracing::info!(strategy = %id, %holdout_id, %job_id, "holdout evaluation requested");
                            actions.push(json!({"strategy_id": id, "action": "holdout_requested", "holdout_id": holdout_id, "job_id": job_id}));
                        }
                        Err(e) => {
                            tracing::warn!(strategy = %id, error = %e, "holdout job not published");
                            self.engine.lock().await.pipeline_note(&id, &e);
                        }
                    }
                }
                _ => {
                    let applied = self.engine.lock().await.apply_decision(&id, &d, &ev, now);
                    match applied {
                        Ok(Some((action, payload))) => {
                            tracing::warn!(strategy = %id, %action, %payload, "pipeline transition");
                            if matches!(d, Decision::Promote)
                                && let Some(g) = gate.as_ref()
                            {
                                let evidence = json!({"deflated_sharpe": ev.deflated_sharpe, "pbo": ev.pbo});
                                let after = payload["to"].as_str();
                                crate::risk_io::review(&self.store, &id, "pipeline", ACTOR, Some(st.status.as_str()), after, g, evidence).await;
                            }
                            let row = self.engine.lock().await.strategy_row_pub(&id);
                            if let Some((s, book)) = row {
                                self.store.strategy_state(&s, Some(&book), gate.as_ref()).await;
                            }
                            self.store.audit_as(ACTOR, &action, Some(&id), payload.clone()).await;
                            actions.push(json!({"strategy_id": id, "action": action, "detail": payload}));
                        }
                        Ok(None) => {}
                        Err(e) => {
                            tracing::warn!(strategy = %id, error = %e, "pipeline transition refused");
                            self.engine.lock().await.pipeline_note(&id, &e);
                        }
                    }
                }
            }
        }
        let live = self.sync_live_set().await;
        let helix = self.mirror_genomes().await;
        let summary = {
            let e = self.engine.lock().await;
            let mut by_status: BTreeMap<&str, usize> = BTreeMap::new();
            for id in e.strategy_ids() {
                if let Some(s) = e.strategy(&id).filter(|s| s.evolved()) {
                    *by_status.entry(s.status.as_str()).or_default() += 1;
                }
            }
            json!(by_status)
        };
        let st = json!({
            "enabled": true,
            "at_ms": started,
            "took_ms": now_ms() - started,
            "holdouts": holdouts,
            "discovered": discovered,
            "rejected": rejected_recent,
            "rejected_this_round": rejected.len(),
            "actions": actions,
            "waiting": waiting,
            "evolved_by_status": summary,
            "live_set": live,
            "helix_mirror": helix,
            "runtime_reachable": pauses.is_some(),
            "policy": self.policy,
        });
        {
            let mut s = self.status.lock().await;
            let rounds = s["rounds"].as_u64().unwrap_or(0) + 1;
            *s = st.clone();
            s["rounds"] = json!(rounds);
        }
        st
    }

    /// Spec 35 §6.5: an active strategy whose region literals were remapped onto a new snapshot
    /// after it was promoted, or that strategy-runtime suspended because they are unmappable
    /// (pause code 4), re-enters shadow. Returns the actions taken.
    async fn snapshot_shadow(&self, runtime: &Value) -> Vec<Value> {
        let Some(rows) = runtime["strategies"].as_array() else { return Vec::new() };
        let mut out = Vec::new();
        for r in rows {
            let Some(id) = r["id"].as_str() else { continue };
            let remapped_ms = r["snapshot"]["remapped_ms"].as_i64();
            let suspended = r["paused_reason_code"].as_f64() == Some(4.0);
            if remapped_ms.is_none() && !suspended {
                continue;
            }
            let now = now_ms();
            let reason = {
                let mut e = self.engine.lock().await;
                let Some(st) = e.strategy(id) else { continue };
                if st.status != StrategyStatus::Active {
                    continue;
                }
                let reason = match (suspended, remapped_ms) {
                    (true, _) => format!(
                        "snapshot change: region lineage breaks ({})",
                        r["paused_reason"].as_str().unwrap_or("unmappable literal")
                    ),
                    (false, Some(at)) if at > st.since_ms => format!(
                        "snapshot change: region literals remapped {} -> {}; revalidate in shadow",
                        r["snapshot"]["remap"]["from"].as_str().unwrap_or("?"),
                        r["snapshot"]["remap"]["to"].as_str().unwrap_or("?")
                    ),
                    _ => continue,
                };
                if let Err(err) = e.strategy_action(id, StrategyAction::Shadow, &Evidence::default(), now) {
                    tracing::warn!(strategy = %id, error = %err, "snapshot shadow refused");
                    continue;
                }
                e.pipeline_note(id, &reason);
                reason
            };
            tracing::warn!(strategy = %id, %reason, "strategy back to shadow (spec 35 §6.5)");
            let row = self.engine.lock().await.strategy_row_pub(id);
            if let Some((s, book)) = row {
                self.store.strategy_state(&s, Some(&book), None).await;
            }
            let payload = json!({"reason": reason, "snapshot": r["snapshot"].clone()});
            self.store.audit_as(ACTOR, "strategy.snapshot_shadow", Some(id), payload.clone()).await;
            out.push(json!({"strategy_id": id, "action": "strategy.snapshot_shadow", "detail": payload}));
        }
        out
    }

    /// Mirrors every evolved strategy's lifecycle status onto its Helix `Genome` (spec 35 §5).
    /// Rows whose `(status, since)` differ from the last successful mirror are written; failures
    /// stay pending and are retried next round.
    async fn mirror_genomes(&self) -> Value {
        let Some(hub) = &self.hub else { return json!({"enabled": false}) };
        let want: Vec<(String, fa_hub::GenomeStatus)> = {
            let e = self.engine.lock().await;
            e.strategy_ids()
                .into_iter()
                .filter_map(|id| {
                    let st = e.strategy(&id)?;
                    let p = st.pipeline.as_ref()?;
                    let genome_id = genome_id_of(&p.genome).unwrap_or(&id).to_owned();
                    let reason = if st.last_reason.is_empty() { p.note.clone() } else { st.last_reason.clone() };
                    Some((
                        id.clone(),
                        fa_hub::GenomeStatus {
                            genome_id,
                            kind: "strategy".into(),
                            family: "gp".into(),
                            params: p.genome.clone(),
                            strategy_id: id,
                            status: st.status.as_str().to_owned(),
                            reason,
                            since_ms: st.since_ms,
                        },
                    ))
                })
                .collect()
        };
        let mut mirrored = self.mirrored.lock().await;
        let (mut written, mut failed, mut last_error) = (0usize, 0usize, None::<String>);
        for (id, g) in &want {
            if mirrored.get(id).is_some_and(|(s, since)| *s == g.status && *since == g.since_ms) {
                continue;
            }
            if failed > 0 && last_error.as_deref().is_some_and(|e| e.contains("unreachable")) {
                failed += 1; // Helix is down: do not hammer it this round
                continue;
            }
            match hub.set_genome_status(g).await {
                Ok(_) => {
                    mirrored.insert(id.clone(), (g.status.clone(), g.since_ms));
                    written += 1;
                }
                Err(err) => {
                    failed += 1;
                    let unreachable = err.is_unreachable();
                    last_error = Some(if unreachable { format!("unreachable: {err}") } else { err.to_string() });
                }
            }
        }
        if let Some(err) = &last_error {
            tracing::warn!(failed, error = %err, "helix genome status mirror incomplete; retried next round");
        }
        {
            let mut e = self.engine.lock().await;
            e.counters.insert("helix.mirror_written", written as f64);
            e.counters.insert("helix.mirror_failed", failed as f64);
            e.counters.insert("helix.mirrored", mirrored.len() as f64);
        }
        json!({"enabled": true, "evolved": want.len(), "written": written, "failed": failed, "in_sync": mirrored.len(), "last_error": last_error})
    }

    /// Rewrites the managed live set when it differs from Garage and reloads strategy-runtime.
    async fn sync_live_set(&self) -> Value {
        let genomes: Vec<String> = self.engine.lock().await.live_set_genomes();
        let parsed: Vec<StrategyGenome> = genomes.iter().filter_map(|g| StrategyGenome::parse(g).ok()).collect();
        let text = StrategyGenome::population_toml(
            &parsed,
            "Managed live set (spec 35 §6.4): evolved strategies past the locked holdout, written by the\nrisk-engine pipeline; strategy-runtime runs the baselines plus these (`managed`). Do not edit.",
        );
        let Some(blob) = &self.blob else {
            return json!({"written": false, "reason": "Garage not configured", "evolved": parsed.len()});
        };
        let key = ObjectKey { bucket: buckets::ARTIFACTS, key: LIVE_SET_KEY.into() };
        let mut written = self.live_written.lock().await;
        if written.is_none() {
            // first round: compare with what Garage holds (nothing stored = empty set)
            let stored = match blob.get(&key).await {
                Ok(b) => String::from_utf8(b).unwrap_or_default(),
                Err(_) => StrategyGenome::population_toml(&[], ""),
            };
            let stored_ids: BTreeSet<String> = StrategyFile::parse(&stored)
                .map(|f| f.strategy.into_iter().map(|s| s.id).collect())
                .unwrap_or_default();
            let want_ids: BTreeSet<String> = parsed.iter().map(StrategyGenome::id).collect();
            if stored_ids == want_ids {
                *written = Some(text.clone());
            }
        }
        if written.as_deref() == Some(text.as_str()) {
            return json!({"written": false, "unchanged": true, "evolved": parsed.len(), "key": format!("{}/{LIVE_SET_KEY}", buckets::ARTIFACTS)});
        }
        if let Err(e) = blob.put(&key, text.clone().into_bytes(), "application/toml").await {
            tracing::warn!(error = %e, "live set not written");
            return json!({"written": false, "reason": e.to_string()});
        }
        // reload strategy-runtime (managed population); retried next round when it fails
        let mut args = std::collections::HashMap::new();
        args.insert("artifact_key".to_owned(), "managed".to_owned());
        args.insert("actor".to_owned(), ACTOR.to_owned());
        let req = CtlRequest { verb: "reload".into(), args };
        let reloaded = tokio::time::timeout(
            Duration::from_secs(10),
            self.bus.client().request(fa_bus::subjects::ctl("strategy", "reload"), req.encode_to_vec().into()),
        )
        .await;
        let reply = match reloaded {
            Ok(Ok(m)) => CtlReply::decode(m.payload.as_ref()).ok(),
            _ => None,
        };
        let ok = reply.as_ref().is_some_and(|r| r.ok);
        if ok {
            *written = Some(text);
        }
        let ids: Vec<String> = parsed.iter().map(StrategyGenome::id).collect();
        tracing::warn!(evolved = ?ids, reloaded = ok, "managed live set written");
        self.store
            .audit_as(
                ACTOR,
                "strategy.live_set",
                Some(LIVE_SET_KEY),
                json!({"strategies": ids, "reloaded": ok, "reply": reply.map(|r| r.message)}),
            )
            .await;
        json!({"written": true, "reloaded": ok, "evolved": ids.len(), "strategies": ids})
    }
}

fn humanize(ms: i64) -> String {
    let s = ms / 1000;
    if s % 86_400 == 0 {
        format!("{}d", s / 86_400)
    } else if s % 3600 == 0 {
        format!("{}h", s / 3600)
    } else if s % 60 == 0 {
        format!("{}m", s / 60)
    } else {
        format!("{s}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genome_ids_are_read_from_the_inline_table() {
        assert_eq!(genome_id_of(r#"{ id = "gp-6433eb2ddeeddf72", genome_id = "x" }"#), Some("gp-6433eb2ddeeddf72"));
        assert_eq!(genome_id_of(r#"{id="gp-1"}"#), Some("gp-1"));
        assert_eq!(genome_id_of("alloc:name=x"), None);
        assert_eq!(humanize(7_200_000), "2h");
        assert_eq!(humanize(86_400_000), "1d");
        assert_eq!(humanize(90_000), "90s");
    }
}
