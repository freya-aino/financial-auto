//! One training run, shared by the CLI and the job worker: `pretrain` / `generation` fit a
//! snapshot ([`crate::fit`]), `adapter` re-fits statistics on the last hour. The caller passes
//! the control surface a job needs — a stop flag (preemption / cancel), a budget, a checkpoint
//! interval and a progress callback — and gets an [`Outcome`] in `JobResult` terms.

use std::{
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
    time::{Duration, Instant},
};

use fa_core::Timescale;
use fa_encoders::{EncoderFamily, FitConfig, NormStats, STATISTICAL_FAMILY, StatisticalEncoder};
use fa_proto::v1::job_result::Status;
use fa_tsdb::Tsdb;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    artifact::{self, Checkpoint, Stores},
    data::{self, BarHistory},
    fit::{Fit, FitSpec},
    search::Candidate,
};

pub const PROTOCOL: &str = "train/v0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Pretrain,
    Generation,
    Adapter,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pretrain => "pretrain",
            Self::Generation => "generation",
            Self::Adapter => "adapter",
        }
    }

    /// `Job.kind` / result subject suffix: `train.<kind>`.
    pub fn job_kind(self) -> String {
        format!("train.{}", self.as_str())
    }

    pub fn parse_job_kind(kind: &str) -> Option<Self> {
        match kind.strip_prefix("train.")? {
            "pretrain" => Some(Self::Pretrain),
            "generation" => Some(Self::Generation),
            "adapter" => Some(Self::Adapter),
            _ => None,
        }
    }

    /// GPU priority class (spec 36 §6.7).
    pub fn gpu_class(self) -> &'static str {
        match self {
            Self::Adapter => crate::gpu::ONLINE_ADAPTER,
            _ => crate::gpu::GENERATION_TRAINING,
        }
    }
}

/// `Job.spec` of an `adapter` job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterSpec {
    /// Snapshot to adapt; default: the active statistical snapshot of `timescale`.
    #[serde(default)]
    pub snapshot_id: Option<String>,
    #[serde(default)]
    pub timescale: Option<Timescale>,
    #[serde(default = "one_hour")]
    pub window_hours: f64,
    #[serde(default)]
    pub instrument: Option<String>,
    /// Mean |Δmean|/σ over features beyond which the adapter is frozen (spec 36 §6.5 guard);
    /// default: the snapshot's `manifest.params.adapter_guard.max_drift`, else 1.0.
    #[serde(default)]
    pub max_drift: Option<f64>,
    #[serde(default)]
    pub fit: FitConfig,
    /// Neural snapshots: skip the weight-level LoRA adapter (input normalisation only).
    #[serde(default)]
    pub no_lora: bool,
    /// Neural snapshots: LoRA hyper-parameters (default rank 8, 150 steps).
    #[serde(default)]
    pub lora: Option<fa_encoders::neural::adapter::LoraConfig>,
}

fn one_hour() -> f64 {
    1.0
}

impl Default for AdapterSpec {
    fn default() -> Self {
        Self {
            snapshot_id: None,
            timescale: None,
            window_hours: 1.0,
            instrument: None,
            max_drift: None,
            fit: FitConfig::default(),
            no_lora: false,
            lora: None,
        }
    }
}

/// Stop requests from outside the run.
pub const RUN: u8 = 0;
pub const PREEMPT: u8 = 1;
pub const CANCEL: u8 = 2;

/// Control surface of a run.
pub struct Control {
    pub job_id: String,
    pub stop: Arc<AtomicU8>,
    /// `None` = unlimited.
    pub budget: Option<Duration>,
    pub checkpoint_every: Option<Duration>,
    /// Resume from this checkpoint key (`Job.checkpoint_ref`); the job's own key is tried too.
    pub checkpoint_ref: Option<String>,
    pub progress: Box<dyn Fn(f32, String) + Send + Sync>,
}

impl Control {
    pub fn local(job_id: impl Into<String>) -> Self {
        Self {
            job_id: job_id.into(),
            stop: Arc::default(),
            budget: None,
            checkpoint_every: None,
            checkpoint_ref: None,
            progress: Box::new(|f, m| tracing::info!(progress = format!("{:.0}%", f * 100.0), "{m}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Outcome {
    pub status: Status,
    pub result: Value,
    pub checkpoint_ref: String,
    pub output_refs: Vec<String>,
    pub error: String,
    pub snapshot_id: Option<String>,
}

impl Outcome {
    fn failed(e: impl std::fmt::Display, kind: Kind) -> Self {
        Self {
            status: Status::Failed,
            result: json!({ "protocol": PROTOCOL, "kind": kind.job_kind(), "error": e.to_string() }),
            checkpoint_ref: String::new(),
            output_refs: vec![],
            error: e.to_string(),
            snapshot_id: None,
        }
    }
}

/// Shared services of a run.
pub struct Services<'a> {
    pub stores: &'a Stores,
    pub tsdb: &'a Tsdb,
    pub pg_url: Option<&'a str>,
}

pub async fn run_fit(kind: Kind, spec: FitSpec, svc: &Services<'_>, ctl: &Control) -> Outcome {
    match fit_inner(kind, spec, svc, ctl).await {
        Ok(o) => o,
        Err(e) => Outcome::failed(e, kind),
    }
}

async fn fit_inner(kind: Kind, spec: FitSpec, svc: &Services<'_>, ctl: &Control) -> anyhow::Result<Outcome> {
    if let Some(family) = crate::neural_fit::family_of(&spec)? {
        return crate::neural_fit::run(kind, family, spec, svc, ctl).await;
    }
    let started = Instant::now();
    let now = data::now_ms();
    let hub = svc.stores.hub.as_ref();
    let ts = spec.timescale;

    // 1. parent (explicit, else the active snapshot of the timescale)
    let parent_id = match &spec.parent_snapshot_id {
        Some(p) => Some(p.clone()),
        None => match hub {
            Some(h) => h.active_snapshot(ts.as_str(), STATISTICAL_FAMILY).await.ok().flatten().map(|s| s.snapshot_id),
            None => None,
        },
    };
    let parent_graph = artifact::parent_graph(hub, parent_id.as_deref()).await;
    if spec.parent_snapshot_id.is_some() && parent_graph.is_none() {
        anyhow::bail!("parent snapshot {} not found in Helix", parent_id.unwrap_or_default());
    }
    let generation = match kind {
        Kind::Pretrain => 0,
        _ => parent_graph.as_ref().map_or(1, |g| g.snapshot.generation + 1),
    };

    // 2. resume or start
    let mut resumed_from = None;
    let existing = match (&svc.stores.blob, &ctl.checkpoint_ref) {
        (Some(blob), Some(key)) => artifact::load_checkpoint(blob, key).await?,
        (Some(blob), None) if !ctl.job_id.is_empty() => {
            artifact::load_checkpoint(blob, &artifact::checkpoint_key(&ctl.job_id).key).await?
        }
        _ => None,
    };
    let (history, mut fit) = match existing.filter(|c| c.fit.spec == spec && c.kind == kind.job_kind()) {
        Some(c) => {
            tracing::info!(job = %ctl.job_id, phase = ?c.fit.phase, "resuming from checkpoint");
            resumed_from = Some(artifact::checkpoint_key(&ctl.job_id).key);
            let h = data::load_set(svc.tsdb, ts, None, None, c.start_ms, c.end_ms, Some(&c.pins)).await?;
            (h, c.fit)
        }
        None => {
            let (start_ms, end_ms) = data::window(ts, spec.window_hours, now);
            let h = data::load_set(
                svc.tsdb,
                ts,
                spec.instruments.as_ref(),
                spec.instrument.as_deref(),
                start_ms,
                end_ms,
                None,
            )
            .await?;
            let id = crate::fit::snapshot_id(ts, generation, now);
            let f = Fit::new(spec.clone(), id, generation, parent_id.clone(), &h, now);
            (h, f)
        }
    };
    (ctl.progress)(
        fit.fraction(),
        format!(
            "{} bars of {} instrument(s) at {ts} ({}); {} windows, {} probes",
            history.iter().map(|h| h.bars.len()).sum::<usize>(),
            history.len(),
            history
                .iter()
                .map(|h| format!("{}={}", &h.instrument[..h.instrument.len().min(8)], h.source.as_str()))
                .collect::<Vec<_>>()
                .join(","),
            fit.n_windows(),
            fit.probes.len()
        ),
    );

    // 3. statistics phase: checkpoint / preempt / budget between chunks
    let mut last_ckpt = Instant::now();
    let mut ckpt_key = resumed_from.clone();
    loop {
        let stop = ctl.stop.load(Ordering::Relaxed);
        let over_budget = ctl.budget.is_some_and(|b| started.elapsed() > b);
        if stop != RUN || over_budget {
            let key = save(svc, ctl, kind, &history, &fit).await;
            let (status, why) = match (stop, over_budget) {
                (CANCEL, _) => (Status::Cancelled, "cancelled"),
                (PREEMPT, _) => (Status::Preempted, "preempted by the gpu arbiter"),
                _ => (Status::BudgetExceeded, "budget exhausted"),
            };
            return Ok(Outcome {
                status,
                result: json!({
                    "protocol": PROTOCOL, "kind": kind.job_kind(), "snapshot_id": fit.snapshot_id,
                    "phase": fit.phase_name(), "fraction": fit.fraction(), "reason": why,
                    "checkpoint_ref": key, "elapsed_ms": started.elapsed().as_millis() as u64,
                }),
                checkpoint_ref: key.unwrap_or_default(),
                output_refs: vec![],
                error: why.into(),
                snapshot_id: Some(fit.snapshot_id.clone()),
            });
        }
        if !fit.step(&history)? {
            break;
        }
        (ctl.progress)(fit.fraction(), format!("statistics {:.0}%", fit.fraction() / 0.9 * 100.0));
        if let Some(every) = ctl.checkpoint_every
            && last_ckpt.elapsed() >= every
        {
            ckpt_key = save(svc, ctl, kind, &history, &fit).await;
            last_ckpt = Instant::now();
        }
        tokio::task::yield_now().await;
    }

    if let Some(o) = cancelled(ctl, kind, &fit, started) {
        return Ok(o);
    }
    // 4. decoder feedback (generation only): the latest evolution statistics, the parent's
    //    decoder skill and the hyper-parameter search scored by the in-process proxy.
    if kind == Kind::Generation {
        let stats = match svc.pg_url {
            Some(url) => crate::feedback::latest_generation_stats(url).await,
            None => None,
        };
        fit.feedback = Some(json!({ "decoder_stats": stats }));
        if fit.spec.search.candidates > 1 {
            (ctl.progress)(0.8, format!("feedback search over {} candidates", fit.spec.search.candidates));
            let anchor = parent_candidate(svc, parent_graph.as_ref(), &fit.spec).await;
            let skill = crate::feedback::parent_skill(svc.pg_url, parent_id.as_deref()).await;
            fit.apply_search(&history, &anchor, &skill)?;
        }
    }
    if let Some(o) = cancelled(ctl, kind, &fit, started) {
        return Ok(o);
    }
    (ctl.progress)(0.9, "regions and alignment".into());
    let parent = artifact::parent_assignment(hub, parent_graph.as_ref(), &history, &fit.probe_keys(&history)).await;
    let bundle = fit.finish(&history, &parent, chrono::Utc::now().to_rfc3339())?;
    if let Some(o) = cancelled(ctl, kind, &fit, started) {
        return Ok(o);
    }
    let written = artifact::write_bundle(svc.stores, &bundle, parent_graph.as_ref(), kind.as_str()).await?;
    if let (Some(blob), Some(key)) = (&svc.stores.blob, &ckpt_key) {
        let _ =
            fa_blob::BlobStore::delete(blob, &fa_blob::ObjectKey { bucket: fa_blob::buckets::CKPT, key: key.clone() })
                .await;
    }
    (ctl.progress)(1.0, format!("snapshot {} written", bundle.manifest.snapshot_id));
    let m = &bundle.manifest;
    Ok(Outcome {
        status: Status::Ok,
        result: json!({
            "protocol": PROTOCOL,
            "kind": kind.job_kind(),
            "snapshot_id": m.snapshot_id,
            "timescale": m.timescale,
            "generation": m.generation,
            "parent_snapshot_id": m.parent_snapshot_id,
            "status": "candidate",
            "manifest_key": written.manifest_key,
            "persisted": written,
            "metrics": m.metrics,
            "probe_set": m.probe_set,
            "data": m.data,
            "params": m.params,
            "region_map": {
                "regions": bundle.region_map.n_regions(),
                "matched": bundle.region_map.n_matched(),
                "carried_mass": bundle.region_map.carried_mass(),
                "parent_coverage": parent.covered as f64 / fit.probes.len().max(1) as f64,
                "note": bundle.region_map.note,
            },
            "feedback": m.feedback,
            "resumed_from": resumed_from,
            "elapsed_ms": started.elapsed().as_millis() as u64,
        }),
        checkpoint_ref: String::new(),
        output_refs: written.manifest_key.iter().cloned().collect(),
        error: String::new(),
        snapshot_id: Some(m.snapshot_id.clone()),
    })
}

/// A cancel that arrived after the statistics loop (search, regions, before persisting): the
/// run stops without writing a snapshot. Preemption after the statistics phase is not honoured
/// here (the remaining work is short and not checkpointed); it finishes.
fn cancelled(ctl: &Control, kind: Kind, fit: &Fit, started: Instant) -> Option<Outcome> {
    (ctl.stop.load(Ordering::Relaxed) == CANCEL).then(|| Outcome {
        status: Status::Cancelled,
        result: json!({
            "protocol": PROTOCOL, "kind": kind.job_kind(), "snapshot_id": fit.snapshot_id,
            "phase": fit.phase_name(), "reason": "cancelled", "elapsed_ms": started.elapsed().as_millis() as u64,
        }),
        checkpoint_ref: String::new(),
        output_refs: vec![],
        error: "cancelled".into(),
        snapshot_id: Some(fit.snapshot_id.clone()),
    })
}

/// The parent's search parameters (candidate 0 of the search): from its Garage manifest when it
/// has one, else the spec defaults.
async fn parent_candidate(svc: &Services<'_>, parent: Option<&fa_hub::SnapshotGraph>, spec: &FitSpec) -> Candidate {
    let fallback = spec.default_candidate();
    let (Some(blob), Some(g)) = (&svc.stores.blob, parent) else { return fallback };
    if g.snapshot.garage_key.is_empty() {
        return fallback;
    }
    match artifact::load_snapshot(blob, &g.snapshot.snapshot_id).await {
        Ok((m, _, _)) => Candidate::from_manifest(&m, &fallback),
        Err(e) => {
            tracing::warn!(parent = %g.snapshot.snapshot_id, error = %e, "parent manifest unreadable; search anchors on defaults");
            fallback
        }
    }
}

async fn save(svc: &Services<'_>, ctl: &Control, kind: Kind, h: &[BarHistory], fit: &Fit) -> Option<String> {
    let blob = svc.stores.blob.as_ref()?;
    if ctl.job_id.is_empty() {
        return None;
    }
    let ck = Checkpoint {
        job_id: ctl.job_id.clone(),
        kind: kind.job_kind(),
        pins: h.iter().map(|x| (x.instrument.clone(), x.source)).collect(),
        start_ms: h.iter().map(|x| x.start_ms).min().unwrap_or_default(),
        end_ms: h.iter().map(|x| x.end_ms).max().unwrap_or_default(),
        fit: fit.clone(),
        saved_ms: data::now_ms(),
    };
    match artifact::save_checkpoint(blob, &ctl.job_id, &ck).await {
        Ok(key) => {
            tracing::info!(job = %ctl.job_id, key, fraction = fit.fraction(), "checkpoint saved");
            Some(key)
        }
        Err(e) => {
            tracing::warn!(job = %ctl.job_id, error = %e, "checkpoint save failed");
            None
        }
    }
}

impl Fit {
    pub fn phase_name(&self) -> &'static str {
        match self.phase {
            crate::fit::Phase::Stats { .. } => "stats",
            crate::fit::Phase::Regions { .. } => "regions",
            crate::fit::Phase::Done => "done",
        }
    }
}

// ── adapter ─────────────────────────────────────────────────────────────────

/// Mean over features of `|mean_new − mean_ref| / std_ref` (features with `std_ref ≈ 0` skipped).
pub fn drift(reference: &NormStats, new: &NormStats) -> Option<f64> {
    let mut sum = 0.0;
    let mut n = 0usize;
    for (e, row) in reference.mean.iter().enumerate() {
        for (d, m) in row.iter().enumerate() {
            let s = reference.std.get(e).and_then(|r| r.get(d)).copied().unwrap_or(0.0);
            let Some(nm) = new.mean.get(e).and_then(|r| r.get(d)) else { continue };
            if s > 1e-6 {
                sum += f64::from((nm - m).abs() / s);
                n += 1;
            }
        }
    }
    (n > 0).then(|| sum / n as f64)
}

pub async fn run_adapter(spec: AdapterSpec, svc: &Services<'_>, ctl: &Control) -> Outcome {
    match adapter_inner(spec, svc, ctl).await {
        Ok(o) => o,
        Err(e) => Outcome::failed(e, Kind::Adapter),
    }
}

async fn adapter_inner(spec: AdapterSpec, svc: &Services<'_>, ctl: &Control) -> anyhow::Result<Outcome> {
    let started = Instant::now();
    let hub = svc.stores.hub.as_ref();
    // Which snapshot: explicit id, else the active one of the timescale.
    let snap = match (&spec.snapshot_id, hub) {
        (Some(id), Some(h)) => h.snapshot_by_id(id).await?,
        (None, Some(h)) => {
            let ts = spec.timescale.ok_or_else(|| anyhow::anyhow!("adapter needs `snapshot_id` or `timescale`"))?;
            crate::neural_fit::active_any(h, ts).await
        }
        (_, None) => None,
    };
    let snapshot_id = snap
        .as_ref()
        .map(|s| s.snapshot_id.clone())
        .or_else(|| spec.snapshot_id.clone())
        .or_else(|| spec.timescale.map(StatisticalEncoder::baseline_id))
        .ok_or_else(|| anyhow::anyhow!("no snapshot to adapt (Helix unavailable and no `snapshot_id`)"))?;
    let ts = match (&snap, spec.timescale) {
        (Some(s), _) => s.timescale.parse::<Timescale>()?,
        (None, Some(ts)) => ts,
        (None, None) => anyhow::bail!("snapshot {snapshot_id} not in Helix; pass `timescale`"),
    };
    // Reference encoder: fitted statistics from Garage when the snapshot has an artifact.
    let has_artifact = snap.as_ref().is_some_and(|s| !s.garage_key.is_empty());
    let (encoder, reference, guard) = match (&svc.stores.blob, has_artifact) {
        (Some(blob), true) => {
            let (m, w, _) = artifact::load_snapshot(blob, &snapshot_id).await?;
            if fa_encoders::neural::is_neural_family(&m.encoder_family) {
                return crate::neural_fit::run_adapter(spec, svc, ctl, m, w, started).await;
            }
            let enc = StatisticalEncoder::from_manifest(&m, &w)?;
            let norm = enc.norm().cloned();
            (enc, norm, m.adapter_guard())
        }
        _ => (StatisticalEncoder::new(ts), None, fa_encoders::AdapterGuard::default()),
    };
    let max_drift = spec.max_drift.unwrap_or(guard.max_drift);
    (ctl.progress)(0.1, format!("adapting {snapshot_id} on the last {:.2} h", spec.window_hours));
    let now = data::now_ms();
    let (start_ms, end_ms) = data::window(ts, spec.window_hours, now);
    let history = data::load_one(svc.tsdb, ts, spec.instrument.as_deref(), start_ms, end_ms, None).await?;
    if ctl.stop.load(Ordering::Relaxed) == CANCEL {
        return Ok(Outcome {
            status: Status::Cancelled,
            result: json!({ "protocol": PROTOCOL, "kind": "train.adapter", "snapshot_id": snapshot_id, "reason": "cancelled" }),
            checkpoint_ref: String::new(),
            output_refs: vec![],
            error: "cancelled".into(),
            snapshot_id: Some(snapshot_id),
        });
    }
    if ctl.stop.load(Ordering::Relaxed) == PREEMPT {
        return Ok(Outcome {
            status: Status::Preempted,
            result: json!({ "protocol": PROTOCOL, "kind": "train.adapter", "snapshot_id": snapshot_id, "reason": "preempted" }),
            checkpoint_ref: String::new(),
            output_refs: vec![],
            error: "preempted by the gpu arbiter".into(),
            snapshot_id: Some(snapshot_id),
        });
    }
    let stats = StatisticalEncoder::fit(ts, &history.bars, &spec.fit)?;
    let d = reference.as_ref().and_then(|r| drift(r, &stats));
    let frozen = d.is_some_and(|d| d > max_drift);
    // Occupancy under the snapshot's encoder over the window (how the adapter would move latents).
    let last_view = encoder.encode(&fa_encoders::EncoderInput::from_bars(history.end_ms, &history.bars))?;
    let artifact = json!({
        "protocol": PROTOCOL,
        "snapshot_id": snapshot_id,
        "timescale": ts,
        "ts_ms": now,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "window": { "start_ms": history.start_ms, "end_ms": history.end_ms, "instrument": history.instrument, "source": history.source.as_str(), "n_bars": history.bars.len() },
        "n_windows": stats.n_windows,
        "stats": stats,
        "reference": if reference.is_some() { "snapshot weights" } else { "baseline (no fitted statistics)" },
        "drift": d,
        "max_drift": max_drift,
        "guard": guard,
        "frozen": frozen,
        "last_latent_rms": last_view.experts.iter().map(|e| (e.mean.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / e.mean.len().max(1) as f64).sqrt()).collect::<Vec<_>>(),
    });
    let key = fa_blob::adapter_state(&snapshot_id, now);
    let mut persisted = json!({ "garage": false, "local": Value::Null });
    if let Some(blob) = &svc.stores.blob {
        fa_blob::BlobStore::put(blob, &key, serde_json::to_vec_pretty(&artifact)?, fa_blob::JSON).await?;
        persisted["garage"] = json!(true);
    }
    if let Some(dir) = &svc.stores.out_dir {
        let path = dir.join("adapters").join(&snapshot_id);
        std::fs::create_dir_all(&path)?;
        let file = path.join(format!("{now}.json"));
        std::fs::write(&file, serde_json::to_vec_pretty(&artifact)?)?;
        persisted["local"] = json!(file.display().to_string());
    }
    // AdapterState node next to the artifact (spec 36 §5).
    let adapter_state = match svc.stores.hub.as_ref() {
        Some(hub) => {
            let row = fa_hub::NewAdapterState {
                snapshot_id: snapshot_id.clone(),
                timescale: ts.to_string(),
                version: u64::try_from(now).unwrap_or_default(),
                garage_key: if persisted["garage"] == json!(true) { key.key.clone() } else { String::new() },
                drift: d,
                max_drift,
                frozen,
                n_windows: stats.n_windows,
                window_start_ms: history.start_ms,
                window_end_ms: history.end_ms,
            };
            match artifact::write_adapter_state(hub, &row).await {
                Ok(Some(a)) => json!({ "adapter_id": a.adapter_id, "helix_id": a.id }),
                Ok(None) => json!({ "skipped": format!("snapshot {snapshot_id} not in Helix") }),
                Err(e) => json!({ "error": e.to_string() }),
            }
        }
        None => Value::Null,
    };
    (ctl.progress)(1.0, format!("adapter {} drift={:?} frozen={frozen}", key.key, d));
    Ok(Outcome {
        status: Status::Ok,
        result: json!({
            "protocol": PROTOCOL, "kind": "train.adapter", "snapshot_id": snapshot_id, "timescale": ts,
            "artifact_key": key.key, "persisted": persisted, "n_bars": history.bars.len(),
            "source": history.source.as_str(), "adapter_state": adapter_state,
            "n_windows": stats.n_windows, "drift": d, "frozen": frozen, "max_drift": max_drift,
            "elapsed_ms": started.elapsed().as_millis() as u64,
        }),
        checkpoint_ref: String::new(),
        output_refs: vec![key.key],
        error: String::new(),
        snapshot_id: Some(snapshot_id),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_and_drift() {
        assert_eq!(Kind::parse_job_kind("train.adapter"), Some(Kind::Adapter));
        assert_eq!(Kind::parse_job_kind("evo.eval.decoder"), None);
        assert_eq!(Kind::Pretrain.job_kind(), "train.pretrain");
        assert_eq!(Kind::Generation.gpu_class(), "generation_training");
        assert_eq!(Kind::Adapter.gpu_class(), "online_adapter");
        let r = NormStats { n_windows: 10, mean: vec![vec![0.0, 1.0]], std: vec![vec![1.0, 0.0]], gain: vec![] };
        let n = NormStats { n_windows: 3, mean: vec![vec![0.5, 9.0]], std: vec![vec![1.0, 1.0]], gain: vec![] };
        assert_eq!(drift(&r, &n), Some(0.5), "zero-variance features are skipped");
        let s: AdapterSpec = serde_json::from_str(r#"{"snapshot_id":"x"}"#).unwrap();
        assert_eq!(s.window_hours, 1.0);
        assert_eq!(s.max_drift, None, "default comes from the snapshot's adapter guard");
    }
}
