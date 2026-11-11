//! Encoder generation state machine (spec 36 §4.1, §6.2, §6.3, §8) and the end-to-end gen-0
//! bootstrap (acceptance 1).
//!
//! A generation of one timescale walks
//! `Planned → DataFreeze → EncoderTrain → EncoderSelect → Reembed → RegionFit → McmcInit →
//! DecoderEvolve → StrategyEvolve → Validate → (awaiting promotion | Rejected)`;
//! `Promoted` is set by control-api when the selected snapshot is promoted (promotion stays a
//! human / API decision behind the §6.4 checklist and shadow serving; nothing here promotes).
//!
//! * **DataFreeze**: the bars are read once (locked holdouts excluded, [`crate::holdout`]) with
//!   the window end pinned; every candidate then trains on exactly these bars
//!   ([`crate::run::Services::frozen`]). Train / validation boundaries and excluded holdouts go
//!   to `validation.data_window`.
//! * **EncoderTrain**: one fit per family candidate (neural families allowed at the timescale;
//!   `latent_diffusion` only at day/week, D-010) through [`crate::run::run_fit`] — the same path
//!   as a `pretrain` / `generation` job, run manifests included. `generation` cycles warm-start
//!   from the parent (active) snapshot, with anchors and elite feedback, as a job would.
//! * **EncoderSelect**: Pareto over the candidates ([`select`]): next-step loss
//!   (`val_pred_loss`, the per-expert next-step objective all families share — the families'
//!   own SSL losses measure different things and are not comparable), −mixture skill, the
//!   expert-space health shortfall against the checklist targets, and cost (parameters).
//!   Front 0 by NSGA-II sort ([`fa_evolution::non_dominated_sort`]); ties by health shortfall,
//!   skill, cost.
//! * **Reembed**: the selected snapshot (reloaded from what was written) embeds every window of
//!   the frozen bars; each window is assigned to its nearest fitted region per expert.
//! * **RegionFit**: the snapshot's regions (fitted with the family's declared discovery in the
//!   fit) and their occupancy on the re-embedded history.
//! * **McmcInit**: one Dirichlet Markov graph per expert with the geometric prior of spec 33
//!   §6.1 (centroids), fed with the re-embedded transitions; persisted as `mcmc_init.json`
//!   (model-runtime transfers its live kernels through the region map on swap, D-062).
//! * **DecoderEvolve** (seed): one default genome per decoder family plus seeded random
//!   variants (spec 36 §6.2 step 4); evolution itself runs in evo-orchestrator / evo-worker on
//!   the served snapshot.
//! * **StrategyEvolve** (seed): a seeded random GP strategy population
//!   ([`fa_strategies::random::random_population`]).
//! * **Validate**: the checklist items the manifest can decide offline (spec 36 §6.4 items 1–5;
//!   shadow serving is checked at promotion). All pass → `awaiting_promotion`; otherwise
//!   `Rejected` (§8: the next generation proceeds from the current active snapshot).
//!
//! Failures retry the phase up to [`MAX_ATTEMPTS`] times, then the generation is rejected and
//! the active snapshot is unchanged (§8). The record is persisted after every transition
//! (Postgres `validation.encoder_generation`, a local `record.json`, Garage
//! `fa-models/generations/<id>/`) and published on `lifecycle.generation`. Aborts
//! (`POST /api/generations/{id}/abort` → status `aborting`, or a job cancel) are honoured between
//! phases and between candidates. Generations of different timescales are independent
//! (spec 36 Q3, decided).

use std::{collections::BTreeMap, path::PathBuf, sync::atomic::Ordering};

use fa_core::Timescale;
use fa_encoders::{EncoderFamily, REGIONS_FILE, SnapshotManifest, WEIGHTS_FILE, neural::NeuralEncoder};
use fa_patterns::FittedRegion;
use fa_proto::v1::job_result::Status;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqlx::{PgPool, postgres::PgPoolOptions};

use crate::{
    data::{self, BarHistory, Instruments},
    fit::FitSpec,
    run::{CANCEL, Control, Kind, Outcome, PREEMPT, Services},
};

pub const PROTOCOL: &str = "train.cycle/v0";
/// NATS subject of phase events (core NATS; consumers subscribe, nothing is retained).
pub const SUBJECT: &str = "lifecycle.generation";
pub const MAX_ATTEMPTS: u32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenPhase {
    Planned,
    DataFreeze,
    EncoderTrain,
    EncoderSelect,
    Reembed,
    RegionFit,
    McmcInit,
    DecoderEvolve,
    StrategyEvolve,
    Validate,
    Promoted,
    Rejected,
}

impl GenPhase {
    pub const WORK: [Self; 9] = [
        Self::DataFreeze,
        Self::EncoderTrain,
        Self::EncoderSelect,
        Self::Reembed,
        Self::RegionFit,
        Self::McmcInit,
        Self::DecoderEvolve,
        Self::StrategyEvolve,
        Self::Validate,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Planned => "planned",
            Self::DataFreeze => "data_freeze",
            Self::EncoderTrain => "encoder_train",
            Self::EncoderSelect => "encoder_select",
            Self::Reembed => "reembed",
            Self::RegionFit => "region_fit",
            Self::McmcInit => "mcmc_init",
            Self::DecoderEvolve => "decoder_evolve",
            Self::StrategyEvolve => "strategy_evolve",
            Self::Validate => "validate",
            Self::Promoted => "promoted",
            Self::Rejected => "rejected",
        }
    }

    /// The phase after a successful one (`None` after `Validate` and for terminal phases).
    pub fn next(self) -> Option<Self> {
        match self {
            Self::Planned => Some(Self::DataFreeze),
            _ => Self::WORK.iter().position(|p| *p == self).and_then(|i| Self::WORK.get(i + 1).copied()),
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Promoted | Self::Rejected)
    }
}

/// One transition / attempt of the record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseEvent {
    pub phase: GenPhase,
    pub attempt: u32,
    pub ok: bool,
    pub at_ms: i64,
    pub detail: Value,
}

/// The persisted generation (`validation.encoder_generation.record`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationRecord {
    pub generation_id: String,
    pub timescale: Timescale,
    /// `bootstrap` (gen 0, every family, downstream seeds) | `cycle`.
    pub kind: String,
    pub phase: GenPhase,
    /// `running | awaiting_promotion | promoted | rejected | failed | aborted`.
    pub status: String,
    pub job_id: String,
    pub started_ms: i64,
    pub updated_ms: i64,
    pub spec: Value,
    pub attempts: BTreeMap<String, u32>,
    pub events: Vec<PhaseEvent>,
    pub data: Value,
    pub candidates: Vec<Value>,
    pub selection: Value,
    pub selected_snapshot: Option<String>,
    pub artefacts: BTreeMap<String, Value>,
    pub validation: Value,
    pub error: Option<String>,
}

impl GenerationRecord {
    pub fn new(id: String, ts: Timescale, kind: &str, job_id: &str, spec: Value) -> Self {
        let now = data::now_ms();
        Self {
            generation_id: id,
            timescale: ts,
            kind: kind.into(),
            phase: GenPhase::Planned,
            status: "running".into(),
            job_id: job_id.into(),
            started_ms: now,
            updated_ms: now,
            spec,
            attempts: BTreeMap::new(),
            events: Vec::new(),
            data: Value::Null,
            candidates: Vec::new(),
            selection: Value::Null,
            selected_snapshot: None,
            artefacts: BTreeMap::new(),
            validation: Value::Null,
            error: None,
        }
    }

    /// Records the outcome of the current phase. Ok → the next phase; failure → the same phase
    /// again while attempts remain, else `Rejected` (`status = failed`). Returns whether to go on.
    pub fn complete(&mut self, result: Result<Value, String>) -> bool {
        let phase = self.phase;
        let attempt = self.attempts.get(phase.as_str()).copied().unwrap_or(0);
        let now = data::now_ms();
        self.updated_ms = now;
        match result {
            Ok(detail) => {
                self.events.push(PhaseEvent { phase, attempt, ok: true, at_ms: now, detail });
                match phase.next() {
                    Some(n) => {
                        self.phase = n;
                        *self.attempts.entry(n.as_str().into()).or_default() += 1;
                        true
                    }
                    None => false,
                }
            }
            Err(e) => {
                self.events.push(PhaseEvent { phase, attempt, ok: false, at_ms: now, detail: json!({ "error": e }) });
                if attempt < MAX_ATTEMPTS {
                    *self.attempts.entry(phase.as_str().into()).or_default() += 1;
                    true
                } else {
                    self.error = Some(format!("{} failed {attempt} times: {e}", phase.as_str()));
                    self.phase = GenPhase::Rejected;
                    self.status = "failed".into();
                    false
                }
            }
        }
    }

    /// Enters `DataFreeze` from `Planned`.
    pub fn start(&mut self) {
        self.phase = GenPhase::DataFreeze;
        *self.attempts.entry(GenPhase::DataFreeze.as_str().into()).or_default() += 1;
    }

    pub fn abort(&mut self, why: &str) {
        self.status = "aborted".into();
        self.phase = GenPhase::Rejected;
        self.error = Some(why.into());
        self.updated_ms = data::now_ms();
    }
}

/// `train.cycle` job spec / `trainer cycle|bootstrap` arguments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CycleSpec {
    pub timescale: Timescale,
    /// `bootstrap` (gen 0: `pretrain` fits, no parent) or `cycle` (gen ≥ 1: `generation` fits
    /// from the parent / active snapshot; a timescale without one falls back to `pretrain`).
    #[serde(default = "default_kind")]
    pub kind: String,
    /// Neural families to train (default: every family allowed at the timescale).
    #[serde(default)]
    pub families: Vec<String>,
    #[serde(default = "default_hours")]
    pub window_hours: f64,
    #[serde(default)]
    pub instruments: Option<Instruments>,
    #[serde(default)]
    pub seed: u64,
    /// Genome overrides shared by every candidate (e.g. `{"steps": 300}`).
    #[serde(default)]
    pub neural: Option<Value>,
    #[serde(default)]
    pub parent_snapshot_id: Option<String>,
    #[serde(default)]
    pub end_ms: Option<i64>,
    #[serde(default = "default_decoders")]
    pub decoder_seeds: usize,
    #[serde(default = "default_strategies")]
    pub strategy_seeds: usize,
    #[serde(default)]
    pub generation_id: Option<String>,
    #[serde(default = "default_probes")]
    pub probe_max: usize,
}

fn default_kind() -> String {
    "cycle".into()
}
fn default_hours() -> f64 {
    12.0
}
fn default_decoders() -> usize {
    8
}
fn default_strategies() -> usize {
    16
}
fn default_probes() -> usize {
    512
}

impl CycleSpec {
    pub fn new(ts: Timescale, kind: &str) -> Self {
        serde_json::from_value(json!({ "timescale": ts, "kind": kind })).expect("defaults")
    }

    pub fn families(&self) -> anyhow::Result<Vec<fa_encoders::neural::NeuralFamily>> {
        use fa_encoders::neural::NeuralFamily;
        let fams: Vec<NeuralFamily> = if self.families.is_empty() {
            NeuralFamily::ALL.iter().copied().filter(|f| f.allows(self.timescale)).collect()
        } else {
            self.families
                .iter()
                .map(|f| NeuralFamily::parse(f).ok_or_else(|| anyhow::anyhow!("unknown neural family `{f}`")))
                .collect::<anyhow::Result<_>>()?
        };
        let allowed: Vec<NeuralFamily> = fams.into_iter().filter(|f| f.allows(self.timescale)).collect();
        if allowed.is_empty() {
            anyhow::bail!("no neural family is allowed at {}", self.timescale);
        }
        Ok(allowed)
    }
}

/// Where records go.
pub struct GenStore {
    pub pg: Option<PgPool>,
    pub bus: Option<fa_bus::Bus>,
}

impl GenStore {
    pub async fn connect(pg_url: Option<&str>, bus: Option<fa_bus::Bus>) -> Self {
        let pg = match pg_url {
            Some(u) => match PgPoolOptions::new()
                .max_connections(2)
                .acquire_timeout(std::time::Duration::from_secs(5))
                .connect(u)
                .await
            {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "generation records: Postgres unavailable (local / Garage record only)");
                    None
                }
            },
            None => None,
        };
        Self { pg, bus }
    }

    pub async fn save(&self, rec: &GenerationRecord, svc: &Services<'_>) {
        let bytes = serde_json::to_vec_pretty(rec).unwrap_or_default();
        if let Some(dir) = &svc.stores.out_dir {
            let d = dir.join("generations").join(&rec.generation_id);
            if std::fs::create_dir_all(&d).is_ok() {
                let _ = std::fs::write(d.join("record.json"), &bytes);
            }
        }
        if let Some(pg) = &self.pg {
            let r = sqlx::query(
                "insert into validation.encoder_generation \
                 (generation_id, timescale, kind, phase, status, job_id, selected_snapshot, record, started_at, updated_at) \
                 values ($1, $2, $3, $4, $5, $6, $7, $8, to_timestamp($9 / 1000.0), now()) \
                 on conflict (generation_id) do update set phase = excluded.phase, \
                 status = case when validation.encoder_generation.status = 'aborting' and excluded.status = 'running' \
                          then 'aborting' else excluded.status end, \
                 selected_snapshot = excluded.selected_snapshot, record = excluded.record, updated_at = now()",
            )
            .bind(&rec.generation_id)
            .bind(rec.timescale.as_str())
            .bind(&rec.kind)
            .bind(rec.phase.as_str())
            .bind(&rec.status)
            .bind(&rec.job_id)
            .bind(&rec.selected_snapshot)
            .bind(serde_json::to_value(rec).unwrap_or(Value::Null))
            .bind(rec.started_ms as f64)
            .execute(pg)
            .await;
            if let Err(e) = r {
                tracing::warn!(error = %e, generation = %rec.generation_id, "generation record not written to Postgres");
            }
        }
        if let Some(bus) = &self.bus {
            let ev = json!({
                "generation_id": rec.generation_id, "timescale": rec.timescale, "kind": rec.kind,
                "phase": rec.phase.as_str(), "status": rec.status, "selected_snapshot": rec.selected_snapshot,
                "attempts": rec.attempts, "error": rec.error, "at_ms": rec.updated_ms,
            });
            let _ = bus.client().publish(SUBJECT, serde_json::to_vec(&ev).unwrap_or_default().into()).await;
        }
    }

    /// `POST /api/generations/{id}/abort` sets `aborting`.
    pub async fn abort_requested(&self, id: &str) -> bool {
        let Some(pg) = &self.pg else { return false };
        sqlx::query_scalar::<_, String>("select status from validation.encoder_generation where generation_id = $1")
            .bind(id)
            .fetch_optional(pg)
            .await
            .ok()
            .flatten()
            .is_some_and(|s| s == "aborting")
    }

    async fn data_windows(&self, rec: &GenerationRecord, rows: &[(String, i64, i64, Value)]) {
        let Some(pg) = &self.pg else { return };
        for (kind, a, b, detail) in rows {
            let _ = sqlx::query(
                "insert into validation.data_window (generation_id, kind, timescale, start_at, end_at, detail) \
                 values ($1, $2, $3, to_timestamp($4 / 1000.0), to_timestamp($5 / 1000.0), $6) on conflict do nothing",
            )
            .bind(&rec.generation_id)
            .bind(kind)
            .bind(rec.timescale.as_str())
            .bind(*a as f64)
            .bind(*b as f64)
            .bind(detail)
            .execute(pg)
            .await;
        }
    }
}

async fn put_artefact(svc: &Services<'_>, rec: &mut GenerationRecord, name: &str, v: &Value) {
    let bytes = serde_json::to_vec_pretty(v).unwrap_or_default();
    let mut loc = json!({});
    if let Some(dir) = &svc.stores.out_dir {
        let d = dir.join("generations").join(&rec.generation_id);
        if std::fs::create_dir_all(&d).is_ok() && std::fs::write(d.join(name), &bytes).is_ok() {
            loc["local"] = json!(d.join(name).display().to_string());
        }
    }
    if let Some(blob) = &svc.stores.blob {
        let key = fa_blob::ObjectKey {
            bucket: fa_blob::buckets::MODELS,
            key: format!("generations/{}/{name}", rec.generation_id),
        };
        if fa_blob::BlobStore::put(blob, &key, bytes, fa_blob::JSON).await.is_ok() {
            loc["garage"] = json!(key.key);
        }
    }
    rec.artefacts.insert(name.into(), loc);
}

// ── selection ────────────────────────────────────────────────────────────

/// Health shortfall of a candidate against the checklist targets (0 = all met).
pub fn health_shortfall(m: &Value) -> f64 {
    let g = |k: &str, d: f64| m[k].as_f64().unwrap_or(d);
    (2.0 - g("k_eff", 0.0)).max(0.0) / 2.0
        + (g("resid_corr_max", 1.0) - 0.1).max(0.0)
        + (0.02 - g("skill_min", -1.0)).max(0.0)
}

/// Minimised objectives of a candidate's metrics.
pub fn objectives(m: &Value) -> Vec<f64> {
    let g = |k: &str, d: f64| m[k].as_f64().filter(|v| v.is_finite()).unwrap_or(d);
    // a candidate with collapsed experts (spec 31 §8) is unfit: dominated by every fit one
    if g("collapsed", 0.0) > 0.0 {
        return vec![f64::INFINITY; 4];
    }
    // cost: parameters, doubled per OOM retry (batch halvings, spec 31 §8)
    let cost = g("n_params", f64::INFINITY) * 2f64.powf(g("oom_retries", 0.0));
    vec![g("val_pred_loss", f64::INFINITY), -g("mix_skill", f64::NEG_INFINITY), health_shortfall(m), cost]
}

/// Pareto selection over candidate results (`{family, snapshot_id, metrics}`; failed ones carry
/// no `snapshot_id`). Returns the chosen index and the table.
pub fn select(candidates: &[Value]) -> Option<(usize, Value)> {
    let ok: Vec<usize> = (0..candidates.len()).filter(|&i| candidates[i]["snapshot_id"].is_string()).collect();
    if ok.is_empty() {
        return None;
    }
    let objs: Vec<Vec<f64>> = ok.iter().map(|&i| objectives(&candidates[i]["metrics"])).collect();
    let fronts = fa_evolution::non_dominated_sort(&objs);
    let mut rank = vec![usize::MAX; ok.len()];
    for (r, f) in fronts.iter().enumerate() {
        for &i in f {
            rank[i] = r;
        }
    }
    let best = (0..ok.len())
        .min_by(|&a, &b| {
            rank[a]
                .cmp(&rank[b])
                .then(objs[a][2].total_cmp(&objs[b][2]))
                .then(objs[a][1].total_cmp(&objs[b][1]))
                .then(objs[a][3].total_cmp(&objs[b][3]))
                .then(candidates[ok[a]]["family"].as_str().cmp(&candidates[ok[b]]["family"].as_str()))
        })
        .expect("non-empty");
    let table: Vec<Value> = ok
        .iter()
        .enumerate()
        .map(|(j, &i)| {
            json!({
                "family": candidates[i]["family"], "snapshot_id": candidates[i]["snapshot_id"], "front": rank[j],
                "objectives": { "next_step_loss": objs[j][0], "neg_mix_skill": objs[j][1], "health_shortfall": objs[j][2], "n_params": objs[j][3] },
            })
        })
        .collect();
    Some((
        ok[best],
        json!({
            "method": "NSGA-II front 0 (fa_evolution::non_dominated_sort); ties: health shortfall, mixture skill, cost",
            "objectives": ["val_pred_loss (min)", "mix_skill (max)", "health shortfall vs K_eff ≥ 2 / corr ≤ 0.1 / skill ≥ 0.02 (min)", "n_params × 2^oom_retries (min)"],
            "unfit": "metrics.collapsed > 0 (collapsed experts, spec 31 §8): every objective +∞",
            "table": table,
            "selected": candidates[ok[best]]["snapshot_id"],
        }),
    ))
}

// ── validation (offline part of the §6.4 checklist) ─────────────────────────

fn env_f(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).filter(|v: &f64| v.is_finite()).unwrap_or(d)
}

/// The checklist items decidable from the manifest (shadow serving is checked at promotion).
pub fn offline_checklist(manifest: &Value) -> Value {
    let m = &manifest["metrics"];
    let mut checks = Vec::new();
    let mut push = |name: &str, v: Option<f64>, th: f64, at_least: bool| {
        if let Some(v) = v {
            checks.push(
                json!({ "name": name, "value": v, "threshold": th, "ok": if at_least { v >= th } else { v <= th } }),
            );
        }
    };
    push(
        "carried_mass",
        m["carried_mass"].as_f64().filter(|_| !manifest["parent_snapshot_id"].is_null()),
        env_f("FA_PROMOTE_MIN_CARRIED_MASS", 0.5),
        true,
    );
    push("ssl_regression", m["ssl_regression"].as_f64(), env_f("FA_PROMOTE_REGRESS_TOL", 0.02), false);
    push("expert_collapse", m["collapsed"].as_f64(), 0.0, false);
    push("expert_k_eff", m["k_eff"].as_f64(), env_f("FA_PROMOTE_MIN_K_EFF", 2.0), true);
    push("expert_resid_corr", m["resid_corr_max"].as_f64(), env_f("FA_PROMOTE_MAX_RESID_CORR", 0.1), false);
    push("expert_min_skill", m["skill_min"].as_f64(), env_f("FA_PROMOTE_MIN_SKILL", 0.02), true);
    push("mcmc_ece", m["mcmc_ece_regression"].as_f64(), env_f("FA_PROMOTE_ECE_TOL", 0.02), false);
    push("elite_transfer", m["elite_transfer"].as_f64(), env_f("FA_PROMOTE_TRANSFER_TOL", 0.10), false);
    let failing: Vec<Value> = checks.iter().filter(|c| c["ok"] == json!(false)).map(|c| c["name"].clone()).collect();
    json!({ "checks": checks, "failing": failing, "passed": failing.is_empty(),
            "pending": ["shadow_served", "shadow_minutes", "shadow_region_agreement (checked by POST /api/latent/snapshots/{id}/promote)"] })
}

// ── loading the selected snapshot back ─────────────────────────────────────

async fn load_written(
    svc: &Services<'_>,
    cand: &Value,
) -> anyhow::Result<(SnapshotManifest, Vec<u8>, Vec<Vec<FittedRegion>>)> {
    let id = cand["snapshot_id"].as_str().unwrap_or_default();
    if let Some(dir) = cand["local_dir"].as_str() {
        let d = PathBuf::from(dir);
        let m = SnapshotManifest::from_json(&std::fs::read(d.join(fa_encoders::MANIFEST_FILE))?)?;
        let w = std::fs::read(d.join(WEIGHTS_FILE))?;
        let r = serde_json::from_slice(&std::fs::read(d.join(REGIONS_FILE))?)?;
        return Ok((m, w, r));
    }
    let blob = svc.stores.blob.as_ref().ok_or_else(|| anyhow::anyhow!("snapshot {id} was written nowhere readable"))?;
    let (m, w, _) = crate::artifact::load_snapshot(blob, id).await?;
    let r = serde_json::from_slice(&fa_blob::BlobStore::get(blob, &fa_blob::snapshot_file(id, REGIONS_FILE)).await?)?;
    Ok((m, w, r))
}

/// Per instrument, per expert: the region of every window of the frozen bars.
type Sequences = Vec<Vec<Vec<Option<String>>>>;

fn reembed(
    enc: &NeuralEncoder,
    regions: &[Vec<FittedRegion>],
    history: &[BarHistory],
    max_windows: usize,
) -> anyhow::Result<(Sequences, usize)> {
    let ctx = enc.sizing().context_steps as usize;
    let mut out = Vec::with_capacity(history.len());
    let mut n = 0;
    for h in history {
        let ends: Vec<usize> = (ctx.min(h.bars.len())..=h.bars.len()).collect();
        let stride = (ends.len() / max_windows.max(1)).max(1);
        let ends: Vec<usize> = ends.into_iter().step_by(stride).collect();
        let encoded = tokio::task::block_in_place(|| enc.encode_windows(&h.bars, &ends))?;
        let k = regions.len();
        let mut seq = vec![Vec::with_capacity(encoded.len()); k];
        for e in &encoded {
            for (x, s) in seq.iter_mut().enumerate() {
                let rs = &regions[x];
                let lab =
                    e.mu.get(x)
                        .and_then(|mu| crate::promotion::assign_nearest(std::slice::from_ref(mu), rs).pop().flatten());
                s.push(lab);
            }
        }
        n += encoded.len();
        out.push(seq);
    }
    Ok((out, n))
}

fn stop_requested(ctl: &Control) -> Option<&'static str> {
    match ctl.stop.load(Ordering::Relaxed) {
        CANCEL => Some("cancelled"),
        PREEMPT => Some("preempted"),
        _ => None,
    }
}

/// Runs one generation of `spec.timescale` through every phase. Never promotes.
pub async fn run_cycle(spec: CycleSpec, svc: &Services<'_>, ctl: &Control, store: &GenStore) -> Outcome {
    let started = std::time::Instant::now();
    let id = spec.generation_id.clone().unwrap_or_else(|| {
        format!("gen-{}-{}-{}", spec.timescale, spec.kind, ulid::Ulid::generate().to_string().to_ascii_lowercase())
    });
    let mut rec = GenerationRecord::new(
        id.clone(),
        spec.timescale,
        &spec.kind,
        &ctl.job_id,
        serde_json::to_value(&spec).unwrap_or_default(),
    );
    let families = match spec.families() {
        Ok(f) => f,
        Err(e) => {
            rec.complete(Err(e.to_string()));
            rec.error = Some(e.to_string());
            rec.status = "failed".into();
            rec.phase = GenPhase::Rejected;
            store.save(&rec, svc).await;
            return outcome(&rec, Status::Failed, started);
        }
    };
    store.save(&rec, svc).await;
    rec.start();
    let ts = spec.timescale;
    let mut frozen: Vec<BarHistory> = Vec::new();
    let mut seqs: Sequences = Vec::new();
    let mut selected: Option<(Value, SnapshotManifest, Vec<Vec<FittedRegion>>)> = None;
    while !rec.phase.is_terminal() {
        if let Some(why) = stop_requested(ctl) {
            rec.abort(why);
            break;
        }
        if store.abort_requested(&rec.generation_id).await {
            rec.abort("aborted through the API");
            break;
        }
        store.save(&rec, svc).await;
        let phase = rec.phase;
        (ctl.progress)(
            GenPhase::WORK.iter().position(|p| *p == phase).unwrap_or(0) as f32 / GenPhase::WORK.len() as f32,
            format!("generation {id}: {}", phase.as_str()),
        );
        let result: Result<Value, String> = match phase {
            GenPhase::DataFreeze => {
                let horizon =
                    families.iter().map(|f| fa_encoders::neural::NeuralConfig::new(*f).horizon()).max().unwrap_or(1);
                let end = spec.end_ms.unwrap_or_else(data::now_ms);
                let (a, b) = data::window(ts, spec.window_hours, end);
                let slot = crate::holdout::AuditSlot::default();
                let s2 = Services { audit: Some(&slot), ..*svc };
                match crate::holdout::load_fit_set(&s2, ts, spec.instruments.as_ref(), None, a, b, None, horizon).await
                {
                    Ok(h) => {
                        frozen = h;
                        let audit = slot.lock().ok().and_then(|g| g.clone()).unwrap_or_default();
                        let val_cut = a + ((b - a) as f64 * (1.0 - 0.15)) as i64;
                        let mut rows = vec![
                            (
                                "train".to_owned(),
                                a,
                                val_cut,
                                json!({ "note": "per-series time-ordered split; validation = last 15 % of each series after the embargo" }),
                            ),
                            ("validation".to_owned(), val_cut, b, json!({ "val_fraction": 0.15 })),
                        ];
                        for (x, y) in &audit.excluded_ranges {
                            rows.push(("holdout_excluded".to_owned(), *x, *y, json!({ "embargo_included": true })));
                        }
                        store.data_windows(&rec, &rows).await;
                        rec.data = json!({
                            "start_ms": a, "end_ms": b, "audit": audit,
                            "instruments": frozen.iter().map(BarHistory::data_ref).collect::<Vec<_>>(),
                        });
                        Ok(
                            json!({ "bars": audit.n_bars, "instruments": frozen.len(), "excluded_bars": audit.excluded_bars, "bars_sha256": audit.bars_sha256 }),
                        )
                    }
                    Err(e) => Err(e.to_string()),
                }
            }
            GenPhase::EncoderTrain => {
                let parent = spec.parent_snapshot_id.clone();
                let fsvc = Services { frozen: Some(&frozen), audit: None, ..*svc };
                rec.candidates.clear();
                let mut aborted = None;
                for (i, fam) in families.iter().enumerate() {
                    if let Some(why) = stop_requested(ctl) {
                        aborted = Some(why);
                        break;
                    }
                    let kind = if spec.kind == "bootstrap" { Kind::Pretrain } else { Kind::Generation };
                    let fs = FitSpec {
                        family: Some(fam.as_str().into()),
                        neural: spec.neural.clone(),
                        window_hours: spec.window_hours,
                        instruments: Some(Instruments::List(frozen.iter().map(|h| h.instrument.clone()).collect())),
                        seed: spec.seed.wrapping_add(i as u64),
                        end_ms: rec.data["end_ms"].as_i64(),
                        parent_snapshot_id: parent.clone(),
                        probe_max: spec.probe_max,
                        ..FitSpec::new(ts)
                    };
                    let child = Control { job_id: format!("{}-{}", ctl.job_id, fam.as_str()), ..Control::local("") };
                    let mut out = crate::run::run_fit(kind, fs.clone(), &fsvc, &child).await;
                    if out.status != Status::Ok && kind == Kind::Generation && parent.is_none() {
                        // no active snapshot of the timescale yet: gen 0 for this family
                        out = crate::run::run_fit(Kind::Pretrain, fs, &fsvc, &child).await;
                    }
                    rec.candidates.push(if out.status == Status::Ok {
                        json!({
                            "family": fam.as_str(), "snapshot_id": out.result["snapshot_id"], "metrics": out.result["metrics"],
                            "local_dir": out.result["persisted"]["local_dir"], "manifest_key": out.result["manifest_key"],
                            "run_manifest": out.result["run_manifest"], "generation": out.result["generation"],
                        })
                    } else {
                        json!({ "family": fam.as_str(), "error": out.error })
                    });
                    store.save(&rec, svc).await;
                }
                match aborted {
                    Some(why) => {
                        rec.abort(why);
                        break;
                    }
                    None if rec.candidates.iter().any(|c| c["snapshot_id"].is_string()) => Ok(json!({
                        "trained": rec.candidates.iter().filter(|c| c["snapshot_id"].is_string()).count(),
                        "failed": rec.candidates.iter().filter(|c| c["error"].is_string()).map(|c| c["family"].clone()).collect::<Vec<_>>(),
                    })),
                    None => Err(format!(
                        "no candidate trained: {:?}",
                        rec.candidates.iter().map(|c| c["error"].clone()).collect::<Vec<_>>()
                    )),
                }
            }
            GenPhase::EncoderSelect => match select(&rec.candidates) {
                Some((i, table)) => match load_written(svc, &rec.candidates[i]).await {
                    Ok((m, _, r)) => {
                        rec.selection = table.clone();
                        rec.selected_snapshot = rec.candidates[i]["snapshot_id"].as_str().map(str::to_owned);
                        selected = Some((rec.candidates[i].clone(), m, r));
                        Ok(table)
                    }
                    Err(e) => Err(format!("selected snapshot unreadable: {e}")),
                },
                None => Err("no candidate to select".into()),
            },
            GenPhase::Reembed => {
                let (cand, m, regions) = selected.as_ref().expect("selected");
                match load_written(svc, cand).await.and_then(|(_, w, _)| Ok(NeuralEncoder::from_manifest(m, &w)?)) {
                    Ok(enc) => match reembed(&enc, regions, &frozen, 4096) {
                        Ok((s, n)) => {
                            seqs = s;
                            Ok(json!({ "windows": n, "instruments": frozen.len(), "experts": regions.len() }))
                        }
                        Err(e) => Err(e.to_string()),
                    },
                    Err(e) => Err(e.to_string()),
                }
            }
            GenPhase::RegionFit => {
                let (_, m, regions) = selected.as_ref().expect("selected");
                let occupancy: Vec<BTreeMap<String, usize>> = (0..regions.len())
                    .map(|k| {
                        let mut c = BTreeMap::new();
                        for inst in &seqs {
                            for r in inst.get(k).into_iter().flatten().flatten() {
                                *c.entry(r.clone()).or_default() += 1;
                            }
                        }
                        c
                    })
                    .collect();
                let v = json!({
                    "discovery": m.params["region_discovery"], "regions_per_expert": regions.iter().map(Vec::len).collect::<Vec<_>>(),
                    "carried_mass": m.metrics.get("carried_mass"), "occupancy": occupancy,
                });
                put_artefact(
                    svc,
                    &mut rec,
                    "regions.json",
                    &json!({ "snapshot_id": m.snapshot_id, "regions": regions, "occupancy": v["occupancy"] }),
                )
                .await;
                Ok(v)
            }
            GenPhase::McmcInit => {
                let (_, m, regions) = selected.as_ref().expect("selected");
                let mut graphs = Vec::new();
                let mut summary = Vec::new();
                for (k, rs) in regions.iter().enumerate() {
                    let mut g = fa_mcmc::MarkovGraph::new(0.5, 1.0);
                    for r in rs {
                        g.register_region(&r.region_id);
                        g.set_centroid(&r.region_id, &r.centroid);
                    }
                    let mut n = 0usize;
                    for inst in &seqs {
                        for w in inst[k].windows(2) {
                            if let (Some(a), Some(b)) = (&w[0], &w[1]) {
                                g.observe(a, b);
                                n += 1;
                            }
                        }
                    }
                    summary.push(json!({ "expert": k, "regions": rs.len(), "transitions": n, "mass": g.total_mass(), "geometric_prior": g.has_geometry() }));
                    graphs.push(g);
                }
                put_artefact(
                    svc,
                    &mut rec,
                    "mcmc_init.json",
                    &json!({ "snapshot_id": m.snapshot_id, "graphs": graphs, "ece": m.metrics.get("mcmc_ece") }),
                )
                .await;
                Ok(json!({ "experts": summary, "mcmc_ece": m.metrics.get("mcmc_ece") }))
            }
            GenPhase::DecoderEvolve => {
                let (_, m, _) = selected.as_ref().expect("selected");
                let pop = decoder_seeds(ts, &m.snapshot_id, spec.decoder_seeds, spec.seed);
                let n = pop.len();
                put_artefact(svc, &mut rec, "decoders.json", &json!({ "snapshot_id": m.snapshot_id, "population": pop,
                    "note": "seed population (spec 36 §6.2 step 4); evolution runs in evo-orchestrator / evo-worker on the served snapshot" })).await;
                Ok(json!({ "seeded": n }))
            }
            GenPhase::StrategyEvolve => {
                let rspec = fa_strategies::random::RandomSpec {
                    experts: selected.as_ref().map_or(1, |s| s.2.len() as u32),
                    ..Default::default()
                };
                let pop: Vec<String> =
                    fa_strategies::random::random_population(spec.seed, spec.strategy_seeds, ts, &rspec)
                        .iter()
                        .map(ToString::to_string)
                        .collect();
                let n = pop.len();
                put_artefact(svc, &mut rec, "strategies.json", &json!({ "population": pop,
                    "note": "seed population; strategies are evolved and gated by evo-orchestrator / risk-engine (spec 35, 40, 41) — no strategy is promoted from gen 0 without the full gate" })).await;
                Ok(json!({ "seeded": n }))
            }
            GenPhase::Validate => {
                let (_, m, _) = selected.as_ref().expect("selected");
                let v = offline_checklist(&serde_json::to_value(m).unwrap_or_default());
                rec.validation = v.clone();
                if v["passed"] == json!(true) {
                    rec.status = "awaiting_promotion".into();
                } else {
                    rec.status = "rejected".into();
                    rec.phase = GenPhase::Rejected;
                    rec.error = Some(format!("promotion checks failed: {}", v["failing"]));
                }
                Ok(v)
            }
            GenPhase::Planned | GenPhase::Promoted | GenPhase::Rejected => break,
        };
        let validated = phase == GenPhase::Validate;
        let rejected_in_validate = validated && rec.phase == GenPhase::Rejected;
        if rejected_in_validate {
            rec.events.push(PhaseEvent {
                phase,
                attempt: 1,
                ok: true,
                at_ms: data::now_ms(),
                detail: result.unwrap_or_default(),
            });
            break;
        }
        if !rec.complete(result) {
            break;
        }
    }
    rec.updated_ms = data::now_ms();
    store.save(&rec, svc).await;
    let status = match rec.status.as_str() {
        "failed" => Status::Failed,
        "aborted" => Status::Cancelled,
        _ => Status::Ok,
    };
    outcome(&rec, status, started)
}

fn outcome(rec: &GenerationRecord, status: Status, started: std::time::Instant) -> Outcome {
    Outcome {
        status,
        result: json!({
            "protocol": PROTOCOL, "kind": "train.cycle", "generation_id": rec.generation_id, "timescale": rec.timescale,
            "phase": rec.phase.as_str(), "status": rec.status, "selected_snapshot": rec.selected_snapshot,
            "selection": rec.selection, "validation": rec.validation, "artefacts": rec.artefacts,
            "candidates": rec.candidates, "error": rec.error, "elapsed_ms": started.elapsed().as_millis() as u64,
        }),
        checkpoint_ref: String::new(),
        output_refs: rec.candidates.iter().filter_map(|c| c["manifest_key"].as_str().map(str::to_owned)).collect(),
        error: rec.error.clone().unwrap_or_default(),
        snapshot_id: rec.selected_snapshot.clone(),
    }
}

/// Synthetic regime-switching bars (not market data) for offline bootstraps: `n_inst`
/// instruments of `n_bars` bars ending at `end_ms`, calm / volatile blocks with a volume signature.
pub fn synthetic_bars(ts: Timescale, n_inst: usize, n_bars: usize, end_ms: i64, seed: u64) -> Vec<BarHistory> {
    let step = data::step_ms(ts);
    (0..n_inst)
        .map(|k| {
            let mut rng = fa_core::rng::Rng::seed(seed ^ (k as u64 + 1).wrapping_mul(0x9E37_79B9));
            let mut p = 100.0;
            let mut regime = 0usize;
            let bars: Vec<fa_encoders::MarketBar> = (0..n_bars)
                .map(|i| {
                    if i % 40 == 0 {
                        regime = rng.below(2);
                    }
                    let (vol, drift) = if regime == 0 { (0.001, 0.0003) } else { (0.006, -0.0005) };
                    let open = p;
                    p *= 1.0 + drift + vol * rng.normal();
                    let volume = if regime == 0 { 100.0 } else { 400.0 } * (0.5 + rng.uniform());
                    fa_encoders::MarketBar {
                        open,
                        high: open.max(p) * (1.0 + vol * rng.uniform()),
                        low: open.min(p) * (1.0 - vol * rng.uniform()),
                        close: p,
                        volume,
                        trade_count: (volume / 10.0) as u64,
                    }
                })
                .collect();
            let first = end_ms - step * (n_bars as i64 - 1);
            BarHistory {
                instrument: format!("synthetic{k:02}"),
                timescale: ts,
                source: data::Source::Bars,
                start_ms: first - step,
                end_ms,
                ends_ms: (0..n_bars as i64).map(|i| first + i * step).collect(),
                bars,
            }
        })
        .collect()
}

/// One default genome per decoder family plus seeded jittered variants.
pub fn decoder_seeds(ts: Timescale, snapshot_id: &str, extra: usize, seed: u64) -> Vec<Value> {
    use fa_decoders::genome::{BaselineFamily, DecoderGenome};
    let mut out = Vec::new();
    let mut rng = fa_core::rng::Rng::seed(seed ^ 0xDEC0);
    let base: Vec<DecoderGenome> = BaselineFamily::ALL
        .iter()
        .map(|f| DecoderGenome { timescale: ts, ..DecoderGenome::default_for(&format!("seed-{}", f.as_str()), *f) })
        .collect();
    for g in &base {
        out.push(json!({ "id": g.id, "family": g.family.as_str(), "genome": g, "origin": "default", "snapshot_id": snapshot_id }));
    }
    for i in 0..extra {
        let g = &base[rng.below(base.len())];
        let mut p = g.params();
        for v in p.values_mut() {
            *v *= 1.0 + 0.3 * rng.normal();
            if *v < 0.0 {
                *v = -*v;
            }
        }
        if let Ok(mut v) = g.with_params(&p) {
            v.id = format!("seed-{}-{i}", g.family.as_str());
            out.push(json!({ "id": v.id, "family": v.family.as_str(), "genome": v, "origin": "jittered", "snapshot_id": snapshot_id }));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{artifact::Stores, holdout::tests::history};

    #[test]
    fn phases_advance_and_retry_then_reject() {
        let mut r = GenerationRecord::new("g".into(), Timescale::Minute, "cycle", "", Value::Null);
        r.start();
        assert_eq!(r.phase, GenPhase::DataFreeze);
        assert!(r.complete(Ok(json!({}))));
        assert_eq!(r.phase, GenPhase::EncoderTrain);
        for _ in 0..MAX_ATTEMPTS - 1 {
            assert!(r.complete(Err("boom".into())), "retried");
            assert_eq!(r.phase, GenPhase::EncoderTrain);
        }
        assert!(!r.complete(Err("boom".into())));
        assert_eq!((r.phase, r.status.as_str()), (GenPhase::Rejected, "failed"));
        assert_eq!(GenPhase::Validate.next(), None);
        assert_eq!(GenPhase::Planned.next(), Some(GenPhase::DataFreeze));
    }

    #[test]
    fn pareto_selection_prefers_the_front_then_health() {
        let c = |f: &str, loss: f64, skill: f64, k: f64, corr: f64, smin: f64, np: f64| {
            json!({ "family": f, "snapshot_id": format!("s-{f}"), "metrics": {
                "val_pred_loss": loss, "mix_skill": skill, "k_eff": k, "resid_corr_max": corr, "skill_min": smin, "n_params": np } })
        };
        let cands = vec![
            c("a", 1.0, 0.3, 1.5, 0.5, 0.05, 1000.0),
            c("b", 1.2, 0.2, 1.4, 0.6, 0.01, 2000.0),   // dominated by a
            c("c", 1.1, 0.25, 2.5, 0.08, 0.05, 1500.0), // on the front, healthiest
            json!({ "family": "d", "error": "failed" }),
        ];
        // collapsed experts make a candidate unfit however good its other numbers
        let mut bad = c("z", 0.1, 0.9, 3.0, 0.0, 0.5, 10.0);
        bad["metrics"]["collapsed"] = json!(1.0);
        assert_eq!(objectives(&bad["metrics"]), vec![f64::INFINITY; 4]);
        let mut oom = c("o", 1.0, 0.3, 1.5, 0.5, 0.05, 1000.0);
        oom["metrics"]["oom_retries"] = json!(2.0);
        assert_eq!(objectives(&oom["metrics"])[3], 4000.0);
        let (i, t) = select(&cands).unwrap();
        assert_eq!(cands[i]["family"], "c", "{t}");
        let fronts: Vec<u64> = t["table"].as_array().unwrap().iter().map(|r| r["front"].as_u64().unwrap()).collect();
        assert_eq!(fronts, vec![0, 1, 0]);
    }

    /// Acceptance 1 on synthetic bars: encoders (two families) → selection → re-embedding →
    /// regions → MCMC init → decoder seeds → strategy seeds, every artefact persisted.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bootstrap_runs_end_to_end() {
        let ts = Timescale::Minute;
        let bars = vec![history("aaa", ts, 420, 0), history("bbbbb", ts, 400, 20 * 60_000)];
        let dir = std::env::temp_dir().join(format!("fa-bootstrap-{}", std::process::id()));
        let stores = Stores { blob: None, hub: None, out_dir: Some(dir.clone()) };
        let tsdb = fa_tsdb::Tsdb::new("http://127.0.0.1:9");
        let svc = Services { frozen: Some(&bars), ..Services::new(&stores, &tsdb, None) };
        let spec = CycleSpec {
            families: vec!["fsq_hybrid".into(), "jepa_vae".into()],
            neural: Some(json!({ "steps": 15, "context_steps": 16, "d_model": 8, "layers": 2, "batch": 16,
                                  "moe": { "max_experts": 2, "expert_dim": 4 } })),
            end_ms: Some(420 * 60_000),
            window_hours: 24.0,
            probe_max: 60,
            decoder_seeds: 3,
            strategy_seeds: 4,
            ..CycleSpec::new(ts, "bootstrap")
        };
        let store = GenStore { pg: None, bus: None };
        let out = run_cycle(spec, &svc, &Control::local("boot"), &store).await;
        let id = out.result["generation_id"].as_str().unwrap().to_owned();
        let rec: GenerationRecord =
            serde_json::from_slice(&std::fs::read(dir.join("generations").join(&id).join("record.json")).unwrap())
                .unwrap();
        let visited: Vec<GenPhase> = rec.events.iter().filter(|e| e.ok).map(|e| e.phase).collect();
        assert_eq!(visited, GenPhase::WORK.to_vec(), "{:?} {:?}", rec.events, rec.error);
        assert!(matches!(rec.status.as_str(), "awaiting_promotion" | "rejected"), "{}", rec.status);
        assert_eq!(rec.candidates.iter().filter(|c| c["snapshot_id"].is_string()).count(), 2);
        assert!(rec.selected_snapshot.is_some());
        for f in ["regions.json", "mcmc_init.json", "decoders.json", "strategies.json", "record.json"] {
            assert!(dir.join("generations").join(&id).join(f).exists(), "{f}");
        }
        let mcmc: Value =
            serde_json::from_slice(&std::fs::read(dir.join("generations").join(&id).join("mcmc_init.json")).unwrap())
                .unwrap();
        assert!(!mcmc["graphs"].as_array().unwrap().is_empty());
        let strategies: Value =
            serde_json::from_slice(&std::fs::read(dir.join("generations").join(&id).join("strategies.json")).unwrap())
                .unwrap();
        assert_eq!(strategies["population"].as_array().unwrap().len(), 4);
        // every candidate left a run manifest
        assert_eq!(std::fs::read_dir(dir.join("runs")).unwrap().count(), 2);
        let _ = std::fs::remove_dir_all(dir);
    }
}
