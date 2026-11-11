//! `evo.eval.strategy.holdout` (spec 41 §3 "locked holdout — used only for promotion gates,
//! never in selection", §6.4; spec 40 §6.6 gate item 2; D-059): the **only** way to run a
//! `ReplayPurpose::Promotion` replay. The job names one candidate and one locked holdout
//! (`validation.holdout.id`, locked through `POST /api/validation/holdout`); the replay window is
//! exactly the holdout, the candidate's timescale must match it, and the result is written to
//! `validation.trial` with `kind = holdout` and `holdout_id` set — rows that never enter the
//! generation statistics (`TrialStore::trials` excludes them) and that
//! `GET /api/validation/reports/{candidate}` shows separately.
//!
//! `Job.spec` (JSON): `{"candidate": "<genome text or gp-… id>", "holdout_id": "<uuid>",
//! "snapshot": "<encoder snapshot>"?, "generation": n?}`. A `gp-…` id resolves to the latest
//! genome text evaluated under it. Refused (`STATUS_FAILED`, `rejected: …`): no registry, no such
//! holdout, unknown candidate, timescale mismatch, or latents of another snapshot.
//!
//! Result payload (protocol `evo.eval.strategy.holdout/v0`): as `evo.eval.strategy/v0` plus
//! `"holdout": {"id", "timescale", "start_ms", "end_ms", "locked_at_ms", "note"}` and
//! `"trial": {"kind": "holdout", "recorded": bool}`; `generation_stats` is always `null`.

use std::time::Instant;

use fa_backtest::replay::{self, ReplayPurpose, ReplaySpec};
use fa_proto::v1::Job;
use fa_strategies::genome::StrategyGenome;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    data::{self, DataWindow},
    job::{Outcome, Worker},
    strategy::{CAPITAL, data_json, directions_json, objectives_json, ok_result, replay_json},
    trials::{self, HoldoutRow, TrialRow},
};

pub const KIND: &str = "evo.eval.strategy.holdout";
/// `audit.log` actor of this worker.
pub const AUDIT_ACTOR: &str = "system:evo-worker";
/// `audit.log` action of a selection job refused for touching a locked holdout (spec 41 §8
/// "holdout accessed by selection job → job rejected, audit event").
pub const AUDIT_ACCESS: &str = "validation.holdout.access_rejected";

/// Writes the audit event of a refused holdout access (best effort: a missing registry or a
/// failed insert is logged, the job is rejected either way). `subject` is the job id.
pub async fn audit_access(
    store: Option<&trials::TrialStore>,
    job: &Job,
    candidate: &str,
    spec: &ReplaySpec,
    error: &str,
) -> bool {
    let payload = json!({
        "job_id": job.job_id, "kind": job.kind, "generation": job.parent_job_id, "candidate": candidate,
        "purpose": spec.purpose, "timescale": spec.timescale.to_string(),
        "window_start_ms": spec.window_ms.0, "window_end_ms": spec.window_ms.1, "error": error,
    });
    let Some(store) = store else {
        tracing::warn!(job = %job.job_id, %error, "holdout access rejected; no registry to audit it in");
        return false;
    };
    match store.audit(AUDIT_ACCESS, Some(&job.job_id), &payload).await {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!(job = %job.job_id, error = %e, "holdout access audit failed");
            false
        }
    }
}
pub const PROTOCOL: &str = "evo.eval.strategy.holdout/v0";
/// `validation.trial.kind` of holdout evaluations.
pub const TRIAL_KIND: &str = "holdout";

/// Parsed job spec.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct HoldoutJobSpec {
    pub candidate: String,
    pub holdout_id: String,
    #[serde(default)]
    pub snapshot: Option<String>,
    #[serde(default)]
    pub generation: i32,
}

/// Parses `Job.spec`; both `candidate` and `holdout_id` are required.
pub fn parse_spec(job: &Job) -> Result<HoldoutJobSpec, String> {
    let spec: HoldoutJobSpec = serde_json::from_slice(&job.spec)
        .map_err(|e| format!("holdout spec must be JSON {{candidate, holdout_id}}: {e}"))?;
    if spec.candidate.trim().is_empty() {
        return Err("holdout spec: `candidate` is empty".into());
    }
    if trials::uuid_like(spec.holdout_id.trim()).is_none() {
        return Err(format!("holdout spec: `holdout_id` `{}` is not a UUID", spec.holdout_id));
    }
    Ok(HoldoutJobSpec { holdout_id: spec.holdout_id.trim().to_owned(), ..spec })
}

/// A candidate may only be promoted on a holdout of its own timescale.
pub fn check_candidate(genome: &StrategyGenome, h: &HoldoutRow) -> Result<(), String> {
    if genome.timescale != h.holdout.timescale {
        return Err(format!(
            "candidate timescale {} does not match holdout {} ({})",
            genome.timescale, h.id, h.holdout.timescale
        ));
    }
    if h.holdout.end_ms <= h.holdout.start_ms {
        return Err(format!("holdout {} is empty", h.id));
    }
    Ok(())
}

fn holdout_json(h: &HoldoutRow) -> Value {
    json!({
        "id": h.id, "timescale": h.holdout.timescale, "start_ms": h.holdout.start_ms, "end_ms": h.holdout.end_ms,
        "locked_at_ms": h.holdout.locked_at_ms, "note": h.note,
    })
}

impl Worker {
    /// `evo.eval.strategy.holdout`.
    pub async fn evaluate_holdout(&self, job: &Job, run: &crate::run::JobRun) -> Outcome {
        let started = Instant::now();
        let spec = match parse_spec(job) {
            Ok(s) => s,
            Err(e) => return self.failed(job, format!("rejected: {e}"), 0),
        };
        let Some(store) = &self.trials else {
            return self.failed(
                job,
                "rejected: holdout evaluations need the Postgres trial registry (validation.holdout)".into(),
                0,
            );
        };
        // 1. the locked holdout
        let holdout = match store.holdout_by_id(&spec.holdout_id).await {
            Ok(Some(h)) => h,
            Ok(None) => {
                return self.failed(
                    job,
                    format!(
                        "rejected: holdout {} is not locked; promotion replays run only inside a locked holdout (POST /api/validation/holdout)",
                        spec.holdout_id
                    ),
                    0,
                );
            }
            Err(e) => return self.failed(job, format!("holdout lookup: {e}"), 0),
        };
        // 2. the candidate (text or registry id)
        let text = match store.candidate_text(&spec.candidate).await {
            Ok(Some(t)) => t,
            Ok(None) => {
                return self.failed(job, format!("rejected: candidate `{}` was never evaluated", spec.candidate), 0);
            }
            Err(e) => return self.failed(job, format!("candidate lookup: {e}"), 0),
        };
        let genome = match StrategyGenome::parse(&text) {
            Ok(g) => g,
            Err(e) => return self.failed(job, format!("strategy genome: {e}"), 0),
        };
        if let Err(e) = check_candidate(&genome, &holdout) {
            return self.failed(job, format!("rejected: {e}"), 0);
        }
        let def = genome.def();
        run.progress(0.0, format!("holdout {} for {}", holdout.id, def.id)).await;

        // 3. bars over exactly the holdout window
        let need = genome.lookback_bars + genome.steps as usize + 2;
        let window =
            DataWindow { hours: None, start_ms: Some(holdout.holdout.start_ms), end_ms: Some(holdout.holdout.end_ms) };
        let loaded =
            data::load_bars(self.tsdb.as_ref(), &self.market, genome.timescale, None, &window, 0.0, need).await;
        if loaded.bars.is_empty() {
            return self.failed(
                job,
                format!(
                    "insufficient market data in holdout {}: need ≥ {need} {} bars for `{}`, longest series has {} (source: {}; vm: {})",
                    holdout.id,
                    genome.timescale,
                    def.id,
                    loaded.longest,
                    loaded.source.source,
                    loaded.source.vm_error.as_deref().unwrap_or("ok"),
                ),
                started.elapsed().as_millis() as u64,
            );
        }
        let rspec = ReplaySpec {
            window_ms: (
                loaded.source.window_start_ms.max(holdout.holdout.start_ms),
                loaded.source.window_end_ms.min(holdout.holdout.end_ms),
            ),
            purpose: ReplayPurpose::Promotion,
            // the candidate's asset-set gene (spec 35 §6.3), as in selection
            instruments: genome.assets.clone(),
            timescale: genome.timescale,
            snapshot: spec
                .snapshot
                .clone()
                .or_else(|| genome.snapshot.clone())
                .unwrap_or_else(|| job.snapshot_id.clone()),
            cost_model: self.cost.id.clone(),
            seed: job.seed,
            capital: CAPITAL,
            n_trials: 1,
            trial_sharpe_var: 0.0,
            // the gate judges the trades the live engine would send (same rule as selection)
            trade_rule: self.trade_rule.clone(),
            ..ReplaySpec::selection(genome.timescale, 0, 0)
        };
        debug_assert!(replay::check_holdout(&rspec, std::slice::from_ref(&holdout.holdout)).is_ok());
        let data_block = data_json(&loaded);
        let run = match self
            .replay_within_budget(job, run, &def, &rspec, loaded.bars, vec![holdout.holdout.clone()], started)
            .await
        {
            Ok(r) => r,
            Err(outcome) => return *outcome,
        };
        let report = &run.report;

        // 4. holdout trial row (never feeds selection statistics)
        let objectives = objectives_json(report);
        let recorded = trials::register_holdout(
            Some(store),
            &TrialRow {
                generation: if job.parent_job_id.is_empty() { "holdout".into() } else { job.parent_job_id.clone() },
                gen_index: spec.generation,
                rung: 0,
                job_id: job.job_id.clone(),
                candidate: genome.to_string(),
                kind: TRIAL_KIND.into(),
                window_start_ms: rspec.window_ms.0,
                window_end_ms: rspec.window_ms.1,
                sharpe: report.sharpe,
                n_obs: report.n_obs,
                objectives: Value::Object(objectives.clone()),
                returns: report.returns.clone(),
                holdout_id: Some(holdout.id.clone()),
            },
        )
        .await;
        let report_key = fa_blob::eval_report(&job.parent_job_id, &job.job_id);
        let mut payload = json!({
            "protocol": PROTOCOL,
            "score": report.score(),
            "objectives": objectives,
            "directions": directions_json(),
            "genome": {"text": genome.to_string(), "id": def.id, "timescale": def.timescale, "steps": def.steps,
                       "lookback_bars": def.lookback_bars, "signal": def.signal, "uncertainty": def.uncertainty,
                       "nodes": genome.size()},
            "holdout": holdout_json(&holdout),
            "trial": {"kind": TRIAL_KIND, "recorded": recorded},
            "replay": replay_json(&rspec, report, 1, true),
            "latents": self.recorded.json(&run.latents),
            "data": data_block,
            "generation_stats": Value::Null,
            "budget": {"budget_ms": run.budget_ms, "elapsed_ms": run.elapsed_ms, "truncated": run.truncated},
            "stop": report.stop,
            "report_ref": {"planned": format!("{}/{}", report_key.bucket, report_key.key), "persisted": self.blob.is_some()},
            "worker_id": self.worker_id,
        });
        let mut stored = payload.clone();
        stored["equity"] = json!(report.equity);
        stored["trades"] = json!(report.trades);
        stored["returns"] = json!(report.returns);
        if !self.persist_report(&report_key, &stored).await {
            payload["report_ref"]["persisted"] = json!(false);
        }
        tracing::info!(job = %job.job_id, strategy = %def.id, holdout = %holdout.id, sharpe = report.sharpe,
            dsr = report.deflated_sharpe, steps = report.n_steps, sources = ?report.sources, "holdout evaluated");
        Outcome { result: ok_result(&self.worker_id, job, &payload, report, run.elapsed_ms), outputs: Vec::new() }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use fa_core::Timescale;
    use fa_proto::v1::job_result::Status;

    use super::*;

    #[test]
    fn spec_and_candidate_checks() {
        let job = |spec: &str| Job { spec: spec.as_bytes().to_vec(), ..Default::default() };
        let id = "0190a1b2-c3d4-7e5f-8a6b-7c8d9e0f1a2b";
        let ok =
            parse_spec(&job(&format!(r#"{{"candidate": "gp-1", "holdout_id": " {id} ", "generation": 2}}"#))).unwrap();
        assert_eq!(
            ok,
            HoldoutJobSpec { candidate: "gp-1".into(), holdout_id: id.into(), snapshot: None, generation: 2 }
        );
        assert!(parse_spec(&job("")).unwrap_err().contains("must be JSON"));
        assert!(parse_spec(&job(r#"{"candidate": "gp-1"}"#)).unwrap_err().contains("must be JSON"));
        assert!(parse_spec(&job(r#"{"candidate": "", "holdout_id": "x"}"#)).unwrap_err().contains("empty"));
        assert!(
            parse_spec(&job(r#"{"candidate": "gp-1", "holdout_id": "not-a-uuid"}"#))
                .unwrap_err()
                .contains("not a UUID")
        );
        let genome = StrategyGenome::parse(r#"{ id = "gp-1", timescale = "1m", signal = "1" }"#).unwrap();
        let row = |ts: Timescale, end: i64| HoldoutRow {
            id: id.into(),
            holdout: replay::Holdout { timescale: ts, start_ms: 1_000, end_ms: end, locked_at_ms: 5, embargo_ms: None },
            note: None,
        };
        assert!(check_candidate(&genome, &row(Timescale::Minute, 2_000)).is_ok());
        assert!(check_candidate(&genome, &row(Timescale::Second, 2_000)).unwrap_err().contains("does not match"));
        assert!(check_candidate(&genome, &row(Timescale::Minute, 1_000)).unwrap_err().contains("empty"));
        assert_eq!(holdout_json(&row(Timescale::Minute, 2_000))["timescale"], "1m");
    }

    /// A worker over the compose NATS with no registry, no VictoriaMetrics, no Helix, no Garage.
    pub(crate) async fn offline_worker() -> Worker {
        let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
        let bus = fa_bus::Bus::connect(&url).await.expect("NATS at FA_NATS_URL");
        let (notices, _) = tokio::sync::broadcast::channel(4);
        Worker {
            bus,
            worker_id: "test-worker".into(),
            market: crate::market::MarketBuffer::default(),
            notices,
            gpu_lease_mb: 0,
            tsdb: None,
            data_window_hours: 1.0,
            latents: crate::latent::LatentBuffer::default(),
            hub: None,
            blob: None,
            trials: None,
            cost: std::sync::Arc::new(fa_backtest::cost::CostModel::default()),
            trade_rule: fa_backtest::trade_rule::TradeRule::live_default(),
            recorded: crate::recorded::RecordedStore::disabled(),
            news_refs: crate::news::NewsRefs::default(),
        }
    }

    /// `FA_NATS_TEST=1 cargo test -p evo-worker -- holdout_job`: without a registry, with a
    /// malformed spec and with an unknown holdout the job is refused before any replay.
    #[tokio::test]
    async fn holdout_job_is_refused_outside_a_locked_holdout() {
        if std::env::var("FA_NATS_TEST").is_err() {
            eprintln!("skipping (set FA_NATS_TEST=1)");
            return;
        }
        let mut worker = offline_worker().await;
        let id = "0190a1b2-c3d4-7e5f-8a6b-7c8d9e0f1a2b";
        let job = |spec: String| Job {
            job_id: "h1".into(),
            kind: KIND.into(),
            spec: spec.into_bytes(),
            ..Default::default()
        };
        let out = worker.handle(&job(r#"{"candidate": "gp-1"}"#.into())).await;
        assert_eq!(out.result.status, Status::Failed as i32);
        assert!(out.result.error.contains("rejected: holdout spec"), "{}", out.result.error);
        let good = format!(
            r#"{{"candidate": "{{ id = \"gp-1\", timescale = \"1s\", signal = \"1\" }}", "holdout_id": "{id}"}}"#
        );
        let out = worker.handle(&job(good.clone())).await;
        assert!(out.result.error.contains("need the Postgres trial registry"), "{}", out.result.error);
        if std::env::var("FA_PG_TEST").ok().as_deref() == Some("1") {
            let url = std::env::var("FA_PG_URL").unwrap_or_else(|_| "postgres://fa:fa@127.0.0.1:5432/fa".into());
            worker.trials = trials::TrialStore::connect(&url).await;
            assert!(worker.trials.is_some());
            let out = worker.handle(&job(good)).await;
            assert_eq!(out.result.status, Status::Failed as i32);
            assert!(out.result.error.contains("is not locked"), "{}", out.result.error);
            eprintln!("unknown holdout → {}", out.result.error);
        }
        // the dispatcher routes the kind
        let out = worker.handle(&job("{}".into())).await;
        assert!(out.result.error.starts_with("rejected: holdout spec"), "{}", out.result.error);
    }
    /// A refused holdout access leaves an `audit.log` row (spec 41 §8). Without a registry it is
    /// only logged; with `FA_PG_TEST=1` the row is written, read back and removed again.
    #[tokio::test]
    async fn holdout_access_is_audited() {
        let job =
            Job { job_id: format!("test-{}", ulid::Ulid::generate()), kind: "evo.eval.strategy".into(), ..Default::default() };
        let spec = ReplaySpec::selection(fa_core::Timescale::Second, 1_000, 2_000);
        assert!(!audit_access(None, &job, "gp-1", &spec, "intersects").await);
        if std::env::var("FA_PG_TEST").ok().as_deref() != Some("1") {
            eprintln!("skipped the Postgres half: set FA_PG_TEST=1");
            return;
        }
        let url = std::env::var("FA_PG_URL").unwrap_or_else(|_| "postgres://fa:fa@127.0.0.1:5432/fa".into());
        let store = trials::TrialStore::connect(&url).await.expect("postgres");
        assert!(audit_access(Some(&store), &job, "gp-1", &spec, "intersects").await);
        let pool = sqlx::postgres::PgPoolOptions::new().max_connections(1).connect(&url).await.unwrap();
        let (id, actor, action, payload): (i64, String, String, Value) =
            sqlx::query_as("SELECT id, actor, action, payload FROM audit.log WHERE subject = $1")
                .bind(&job.job_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!((actor.as_str(), action.as_str()), (AUDIT_ACTOR, AUDIT_ACCESS));
        assert_eq!(payload["candidate"], "gp-1");
        assert_eq!(payload["window_end_ms"], 2_000);
        sqlx::query("DELETE FROM audit.log WHERE id = $1").bind(id).execute(&pool).await.unwrap();
    }
}
