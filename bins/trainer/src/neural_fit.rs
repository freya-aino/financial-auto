//! Neural snapshot fits (`family = fsq_hybrid | hier_vae_flow | jepa_vae | latent_diffusion`, spec 31 §6.2, spec 32 §6.3, spec 36
//! §6.2–§6.4): the same job kinds, data, Garage layout (D-058) and Helix mirror as the
//! statistical fit, with a candle model trained end to end on CPU (or CUDA with the `cuda`
//! feature).
//!
//! Steps of a run
//! 1. **Parent**: explicit, else the active snapshot of the timescale (any family). A neural
//!    parent of the same family and genome **warm-starts** the child (spec 36 §6.3); its region
//!    centroids become the **D-007 anchor** of the matching experts. The **D-009 hook** is fed by
//!    the elite decoders in Garage ([`crate::feedback::EliteFeedback`], D-073) when cards of the
//!    fit's step and latent width exist, else by frozen linear probes fitted on the parent's expert
//!    latents ([`crate::feedback::FrozenProbeFeedback`]).
//! 2. **Train** in chunks of [`CHUNK_STEPS`] SGD steps; between chunks the run honours cancel,
//!    preemption (checkpoint + `STATUS_PREEMPTED`) and the budget, and checkpoints every
//!    `checkpoint_every` to `fa-ckpt/<job_id>/neural.safetensors` (+ `state.json` with the data
//!    pins), so a redelivered job resumes bit for bit.
//! 3. **Validate**: optional refit of the next-step heads (`head_refit_steps` > 0; `head_solver`
//!    picks the penalised or the constrained solver), spec 32 §6.4 health on the validation windows (residual correlation,
//!    `K_eff`, per-expert skill, usage), then **prune** experts below `prune_usage_below`
//!    (emergent count, ≥ 2 kept) and evaluate again; a neural parent is evaluated on the same
//!    windows for the §6.4 item 1 regression check.
//! 4. **Expert identity** (spec 32 §6.5): linear CKA of child vs parent experts on the probe
//!    windows, greedy matching, matched experts keep the parent's index.
//! 5. **Regions + alignment**: the family's declared discovery per expert on the probe latents
//!    (`fsq_hybrid` codebook with the residual merge, `jepa_vae` mapper, families 2/4 density;
//!    D-080), then Jaccard alignment against the parent's stored observations as in the
//!    statistical fit (D-007 fallback).
//! 6. **Bundle**: `manifest.json` (genome, device, health and checklist metrics), `weights.bin`
//!    (safetensors), `regions.json`, `region_map.json` → Garage + Helix `candidate`.

use std::{collections::BTreeMap, sync::atomic::Ordering, time::Instant};

use fa_core::Timescale;
use fa_encoders::{
    DataRef, EncoderFamily, ProbeSetRef, STATISTICAL_FAMILY, SnapshotManifest,
    neural::{Dataset, NeuralConfig, NeuralEncoder, NeuralFamily, NeuralTrainer, Series},
};
use fa_expert_space::{
    health::{linear_cka, match_experts},
    nn::{device_from_str, device_label},
};
use fa_patterns::{AlignInput, ParentIds, RegionMap, align_expert};
use fa_proto::v1::job_result::Status;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    artifact,
    data::{self, BarHistory, Source},
    fit::{Bundle, FitSpec, ProbeKey, probe_version, probe_windows},
    run::{CANCEL, Control, Kind, Outcome, PREEMPT, PROTOCOL, RUN, Services},
};

/// SGD steps between control checks.
pub const CHUNK_STEPS: usize = 20;
/// CKA at or above which an expert keeps its parent's index (spec 32 §9.4 placeholder 0.7;
/// lower here because probe sets are small).
pub const EXPERT_MATCH_CKA: f64 = 0.5;
/// Training windows used to fit the frozen probe decoders.
const PROBE_FIT_WINDOWS: usize = 2048;

/// The neural family of a fit spec (`None` = statistical).
pub fn family_of(spec: &FitSpec) -> anyhow::Result<Option<NeuralFamily>> {
    match spec.family.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        None => Ok(None),
        Some(f) if f.eq_ignore_ascii_case(STATISTICAL_FAMILY) || f.eq_ignore_ascii_case("stat_baseline") => Ok(None),
        Some(f) => NeuralFamily::parse(f).map(Some).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown encoder family `{f}` (statistical | fsq_hybrid | hier_vae_flow | jepa_vae | latent_diffusion)"
            )
        }),
    }
}

/// Per-timescale genome defaults (spec 31 §6.1 ladder scaled to CPU), then the job's overrides
/// (`spec.neural`, merged into the config JSON; `moe` merges as an object).
pub fn config_for(family: NeuralFamily, ts: Timescale, overrides: Option<&Value>) -> anyhow::Result<NeuralConfig> {
    if !family.allows(ts) {
        anyhow::bail!("{} is only enabled at the day/week timescales (D-010), not {ts}", family.as_str());
    }
    let mut cfg = NeuralConfig::new(family);
    cfg.context_steps = match ts {
        Timescale::Second | Timescale::Minute => 64,
        Timescale::Hour => 48,
        Timescale::Day | Timescale::Week => 32,
    };
    let Some(o) = overrides.filter(|o| o.is_object()) else { return Ok(cfg) };
    let mut base = serde_json::to_value(&cfg)?;
    merge(&mut base, o);
    let mut cfg: NeuralConfig = serde_json::from_value(base).map_err(|e| anyhow::anyhow!("neural config: {e}"))?;
    cfg.family = family;
    Ok(cfg)
}

fn merge(base: &mut Value, over: &Value) {
    match (base, over) {
        (Value::Object(b), Value::Object(o)) => {
            for (k, v) in o {
                match b.get_mut(k) {
                    Some(slot) if slot.is_object() && v.is_object() => merge(slot, v),
                    _ => {
                        b.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (b, o) => *b = o.clone(),
    }
}

/// Snapshot id: `<fsq|hvf|jepa|ldm>-v1-<ts>-g<gen>-<unix seconds>`.
pub fn snapshot_id(family: NeuralFamily, ts: Timescale, generation: u32, now_ms: i64) -> String {
    format!("{}-{ts}-g{generation}-{}", family.id_prefix(), now_ms / 1000)
}

/// Rough memory of a fit (spec 60 device side): parameters + Adam moments + gradients, plus
/// activations of one batch through the backbone; used as the GPU lease request.
pub fn mem_estimate_mb(cfg: &NeuralConfig, n_params: usize) -> u32 {
    let params = n_params as f64 * 4.0 * 4.0;
    let acts = (cfg.batch * cfg.context_steps * cfg.d_model * (cfg.layers + 2)) as f64 * 4.0 * 3.0;
    (((params + acts) / 1e6).ceil() as u32).max(16)
}

/// `fa-ckpt/<job_id>/state.json` of a neural run: what the resumed job must reload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralCheckpointMeta {
    pub job_id: String,
    pub kind: String,
    pub spec: FitSpec,
    pub snapshot_id: String,
    pub generation: u32,
    pub parent_snapshot_id: Option<String>,
    pub pins: Vec<(String, Source)>,
    pub start_ms: i64,
    pub end_ms: i64,
    pub started_ms: i64,
    pub step: u64,
    pub saved_ms: i64,
}

fn weights_key(job_id: &str) -> fa_blob::ObjectKey {
    fa_blob::ObjectKey { bucket: fa_blob::buckets::CKPT, key: format!("{job_id}/neural.safetensors") }
}

/// The active snapshot of a timescale, whatever its family.
pub async fn active_any(hub: &fa_hub::HubClient, ts: Timescale) -> Option<fa_hub::Snapshot> {
    let mut families = vec![STATISTICAL_FAMILY];
    families.extend(NeuralFamily::ALL.iter().map(|f| f.as_str()));
    for f in families {
        if let Ok(Some(s)) = hub.active_snapshot(ts.as_str(), f).await {
            return Some(s);
        }
    }
    None
}

/// Loaded neural parent: encoder, weights, regions per expert.
struct NeuralParent {
    manifest: SnapshotManifest,
    weights: Vec<u8>,
    encoder: NeuralEncoder,
    centroids: Vec<Vec<Vec<f32>>>,
}

async fn load_neural_parent(
    svc: &Services<'_>,
    g: Option<&fa_hub::SnapshotGraph>,
    dev: &candle_core::Device,
) -> Option<NeuralParent> {
    let (blob, g) = (svc.stores.blob.as_ref()?, g?);
    if !fa_encoders::neural::is_neural_family(&g.snapshot.encoder_family) || g.snapshot.garage_key.is_empty() {
        return None;
    }
    let (manifest, weights, _) = match artifact::load_snapshot(blob, &g.snapshot.snapshot_id).await {
        Ok(x) => x,
        Err(e) => {
            tracing::warn!(parent = %g.snapshot.snapshot_id, error = %e, "neural parent unreadable; training without warm start / anchors");
            return None;
        }
    };
    let encoder = NeuralEncoder::from_manifest_on(&manifest, &weights, dev).ok()?;
    let regions: Vec<Vec<fa_patterns::FittedRegion>> = match fa_blob::BlobStore::get(
        blob,
        &fa_blob::snapshot_file(&g.snapshot.snapshot_id, fa_encoders::REGIONS_FILE),
    )
    .await
    {
        Ok(b) => serde_json::from_slice(&b).unwrap_or_default(),
        Err(_) => Vec::new(),
    };
    let centroids = regions.iter().map(|rs| rs.iter().map(|r| r.centroid.clone()).collect()).collect();
    Some(NeuralParent { manifest, weights, encoder, centroids })
}

fn stop_outcome(
    kind: Kind,
    snapshot_id: &str,
    status: Status,
    why: &str,
    key: Option<String>,
    step: u64,
    started: Instant,
) -> Outcome {
    Outcome {
        status,
        result: json!({
            "protocol": PROTOCOL, "kind": kind.job_kind(), "snapshot_id": snapshot_id, "phase": "train",
            "step": step, "reason": why, "checkpoint_ref": key, "elapsed_ms": started.elapsed().as_millis() as u64,
        }),
        checkpoint_ref: key.unwrap_or_default(),
        output_refs: vec![],
        error: why.into(),
        snapshot_id: Some(snapshot_id.to_owned()),
    }
}

async fn save_checkpoint(
    svc: &Services<'_>,
    ctl: &Control,
    meta: &NeuralCheckpointMeta,
    tr: &NeuralTrainer,
) -> Option<String> {
    let blob = svc.stores.blob.as_ref()?;
    if ctl.job_id.is_empty() {
        return None;
    }
    let bytes = match tr.checkpoint() {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(error = %e, "neural checkpoint serialisation failed");
            return None;
        }
    };
    let mut meta = meta.clone();
    meta.step = tr.step;
    meta.saved_ms = data::now_ms();
    let key = artifact::checkpoint_key(&ctl.job_id);
    let ok = async {
        fa_blob::BlobStore::put(blob, &weights_key(&ctl.job_id), bytes, fa_blob::OCTET_STREAM).await?;
        fa_blob::BlobStore::put(blob, &key, serde_json::to_vec(&meta)?, fa_blob::JSON).await?;
        anyhow::Ok(())
    }
    .await;
    match ok {
        Ok(()) => {
            tracing::info!(job = %ctl.job_id, step = tr.step, "neural checkpoint saved");
            Some(key.key)
        }
        Err(e) => {
            tracing::warn!(job = %ctl.job_id, error = %e, "neural checkpoint save failed");
            None
        }
    }
}

async fn load_checkpoint(svc: &Services<'_>, ctl: &Control) -> Option<(NeuralCheckpointMeta, Vec<u8>)> {
    let blob = svc.stores.blob.as_ref()?;
    if ctl.job_id.is_empty() {
        return None;
    }
    let key = ctl.checkpoint_ref.clone().unwrap_or_else(|| artifact::checkpoint_key(&ctl.job_id).key);
    let meta_bytes =
        fa_blob::BlobStore::get(blob, &fa_blob::ObjectKey { bucket: fa_blob::buckets::CKPT, key }).await.ok()?;
    let meta: NeuralCheckpointMeta = serde_json::from_slice(&meta_bytes).ok()?;
    let w = fa_blob::BlobStore::get(blob, &weights_key(&meta.job_id)).await.ok()?;
    Some((meta, w))
}

pub async fn run(
    kind: Kind,
    family: NeuralFamily,
    spec: FitSpec,
    svc: &Services<'_>,
    ctl: &Control,
) -> anyhow::Result<Outcome> {
    let started = Instant::now();
    let now = data::now_ms();
    let hub = svc.stores.hub.as_ref();
    let ts = spec.timescale;
    let (dev, dev_note) =
        device_from_str(spec.device.as_deref().unwrap_or(&std::env::var("FA_ML_DEVICE").unwrap_or_default()));
    if let Some(n) = &dev_note {
        tracing::warn!("{n}");
    }
    let cfg = config_for(family, ts, spec.neural.as_ref())?;

    // 1. parent
    let parent_id = match &spec.parent_snapshot_id {
        Some(p) => Some(p.clone()),
        None => match hub {
            Some(h) => active_any(h, ts).await.map(|s| s.snapshot_id),
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
    let resumed = load_checkpoint(svc, ctl).await.filter(|(m, _)| m.spec == spec && m.kind == kind.job_kind());
    let (history, mut meta, mut tr, resumed_from) = match resumed {
        Some((m, w)) => {
            tracing::info!(job = %ctl.job_id, step = m.step, "resuming neural fit from checkpoint");
            let h = data::load_set(svc.tsdb, ts, None, None, m.start_ms, m.end_ms, Some(&m.pins)).await?;
            let tr = NeuralTrainer::from_checkpoint(&w, &dev)?;
            let key = artifact::checkpoint_key(&ctl.job_id).key;
            (h, m, tr, Some(key))
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
            let ds = dataset(&h, &cfg);
            if ds.train.is_empty() {
                anyhow::bail!(
                    "{} bars over {} instrument(s) give no training windows (min_window {} + horizon {}) at {ts}",
                    ds.n_bars(),
                    h.len(),
                    cfg.min_window,
                    cfg.horizon()
                );
            }
            let tr = NeuralTrainer::new(cfg.clone(), &ds, spec.seed, &dev)?;
            let m = NeuralCheckpointMeta {
                job_id: ctl.job_id.clone(),
                kind: kind.job_kind(),
                spec: spec.clone(),
                snapshot_id: snapshot_id(family, ts, generation, now),
                generation,
                parent_snapshot_id: parent_id.clone(),
                pins: h.iter().map(|x| (x.instrument.clone(), x.source)).collect(),
                start_ms: h.iter().map(|x| x.start_ms).min().unwrap_or(start_ms),
                end_ms: h.iter().map(|x| x.end_ms).max().unwrap_or(end_ms),
                started_ms: now,
                step: 0,
                saved_ms: now,
            };
            (h, m, tr, None)
        }
    };
    let ds = dataset(&history, &tr.model.cfg);
    let cfg = tr.model.cfg.clone();

    // parent-derived terms: warm start (fresh runs only), anchors, frozen-probe feedback
    let parent = load_neural_parent(svc, parent_graph.as_ref(), &dev).await;
    let mut warm = json!(null);
    if let Some(p) = &parent {
        if resumed_from.is_none() && p.encoder.family() == family {
            match tr.warm_start(&p.weights) {
                Ok(n) => {
                    // the parent's surviving experts keep their positions; pruned ones may revive
                    let mut order = p.encoder.model().moe.active.clone();
                    let revived: Vec<usize> = (0..cfg.moe.max_experts).filter(|e| !order.contains(e)).collect();
                    order.extend(revived);
                    tr.model.moe.select(order);
                    warm = json!({ "from": p.manifest.snapshot_id, "tensors": n });
                }
                Err(e) => warm = json!({ "from": p.manifest.snapshot_id, "error": e.to_string() }),
            }
        }
        if kind == Kind::Generation && cfg.moe.expert_dim == p.encoder.model().cfg.moe.expert_dim {
            let n_parent = p.encoder.model().moe.n_active();
            let per: Vec<Option<Vec<Vec<f32>>>> = (0..tr.model.moe.n_active())
                .map(|pos| (pos < n_parent).then(|| p.centroids.get(pos).cloned()).flatten())
                .collect();
            tr.anchors = fa_encoders::neural::Anchors::from_vecs(&per, &dev)?;
            if tr.model.cfg.lambda_anchor <= 0.0 && tr.anchors.n_anchored() > 0 {
                tr.model.cfg.lambda_anchor = 0.1;
            }
        }
    }
    // D-009: the real elite decoders (D-068) when elites reading this latent width exist, else
    // the frozen-probe stand-in of a neural parent
    let mut feedback_source = "none";
    if kind == Kind::Generation {
        let latent_dim = tr.model.moe.n_active() * cfg.moe.expert_dim;
        let elites = match &svc.stores.blob {
            Some(blob) => {
                match crate::feedback::load_elites(blob, &history, data::step_ms(ts), latent_dim, &dev).await {
                    Ok(e) => e,
                    Err(e) => {
                        tracing::warn!(error = %e, "decoder elites unavailable; falling back to the probe stand-in");
                        None
                    }
                }
            }
            None => None,
        };
        if let Some(fb) = elites {
            tracing::info!(elites = fb.n_elites(), latent_dim, "D-009 feedback from elite decoders");
            tr.feedback.push(Box::new(fb));
            feedback_source = "elite_decoders";
        } else if let Some(p) = parent.as_ref().filter(|p| cfg.moe.expert_dim == p.encoder.model().cfg.moe.expert_dim) {
            match crate::feedback::FrozenProbeFeedback::fit(&p.encoder, &tr.model, &ds, PROBE_FIT_WINDOWS) {
                Ok(fb) => {
                    tr.feedback.push(Box::new(fb));
                    feedback_source = "frozen_linear_probe";
                }
                Err(e) => tracing::warn!(error = %e, "frozen probe feedback unavailable"),
            }
        }
        if !tr.feedback.is_empty() && tr.model.cfg.lambda_feedback <= 0.0 {
            tr.model.cfg.lambda_feedback = 0.5;
        }
    }
    let n_params = tr.model.ps.n_params();
    (ctl.progress)(
        0.02,
        format!(
            "{family:?} on {}: {} bars of {} instrument(s) at {ts}; {} train / {} val windows; {n_params} parameters; step {}/{}",
            device_label(&dev),
            ds.n_bars(),
            history.len(),
            ds.train.len(),
            ds.val.len(),
            tr.step,
            cfg.steps
        ),
    );

    // 3. training loop
    let mut last_ckpt = Instant::now();
    let mut ckpt_key = resumed_from.clone();
    while !tr.done() {
        let stop = ctl.stop.load(Ordering::Relaxed);
        let over_budget = ctl.budget.is_some_and(|b| started.elapsed() > b);
        if stop != RUN || over_budget {
            let key = save_checkpoint(svc, ctl, &meta, &tr).await;
            let (status, why) = match (stop, over_budget) {
                (CANCEL, _) => (Status::Cancelled, "cancelled"),
                (PREEMPT, _) => (Status::Preempted, "preempted by the gpu arbiter"),
                _ => (Status::BudgetExceeded, "budget exhausted"),
            };
            return Ok(stop_outcome(kind, &meta.snapshot_id, status, why, key, tr.step, started));
        }
        let n = CHUNK_STEPS.min(cfg.steps - tr.step as usize);
        // CPU-bound steps off the async executor
        let log = tokio::task::block_in_place(|| tr.train_steps(&ds, n))?;
        let frac = 0.05 + 0.8 * tr.step as f32 / cfg.steps.max(1) as f32;
        (ctl.progress)(
            frac,
            format!(
                "step {}/{} loss {:.4} (decor {:.3}, ssl {:.3})",
                tr.step, cfg.steps, log.total, log.decor, log.ssl
            ),
        );
        if let Some(every) = ctl.checkpoint_every
            && last_ckpt.elapsed() >= every
        {
            meta.step = tr.step;
            ckpt_key = save_checkpoint(svc, ctl, &meta, &tr).await.or(ckpt_key);
            last_ckpt = Instant::now();
        }
        tokio::task::yield_now().await;
    }
    if ctl.stop.load(Ordering::Relaxed) == CANCEL {
        return Ok(stop_outcome(kind, &meta.snapshot_id, Status::Cancelled, "cancelled", None, tr.step, started));
    }

    // 3b. optional full-batch refit of the next-step heads (spec 32 §6.3 decorrelation)
    if cfg.head_refit_steps > 0 {
        (ctl.progress)(0.85, format!("head refit ({} full-batch steps)", cfg.head_refit_steps));
        tokio::task::block_in_place(|| tr.refit_heads(&ds, cfg.head_refit_steps, 4096))?;
    }
    // 4. validation, pruning, parent regression
    (ctl.progress)(0.86, "validation and expert pruning".into());
    let val = if ds.val.is_empty() { ds.train.clone() } else { ds.val.clone() };
    let pre = tokio::task::block_in_place(|| tr.evaluate(&ds, &val))?;
    let before = tr.model.moe.active.clone();
    let kept = tr.prune(&pre.health.usage, 2);
    let post = tokio::task::block_in_place(|| tr.evaluate(&ds, &val))?;
    let parent_eval = match &parent {
        Some(p) => {
            let pt = NeuralTrainer::from_model(p.encoder.model().clone(), 0);
            tokio::task::block_in_place(|| pt.evaluate(&ds, &val)).ok()
        }
        None => None,
    };

    // 5. probes, expert identity, regions, alignment
    let ends: Vec<Vec<usize>> = history
        .iter()
        .map(|h| {
            let n = h.bars.len();
            (cfg.min_window.max(2)..=n).collect()
        })
        .collect();
    let probes = probe_windows(&ends, spec.probe_max);
    let child = NeuralEncoder::new(tr.model.clone(), ts, meta.snapshot_id.clone());
    let child_lat = encode_probes(&child, &history, &ends, &probes)?;
    let mut alignment = json!(null);
    // (final position, parent position, cka) of every expert that continues a parent expert
    let mut aligned: Vec<(usize, usize, f64)> = Vec::new();
    if let Some(p) = &parent {
        let parent_lat = encode_probes(&p.encoder, &history, &ends, &probes)?;
        let cka: Vec<Vec<f64>> =
            child_lat.iter().map(|c| parent_lat.iter().map(|q| linear_cka(c, q)).collect()).collect();
        let matches = match_experts(&cka, EXPERT_MATCH_CKA);
        let order_pos = reorder_by_parent(&matches, child_lat.len());
        let active = tr.model.moe.active.clone();
        tr.model.moe.select(order_pos.iter().map(|&p| active[p]).collect());
        for (pos, &c) in order_pos.iter().enumerate() {
            if let Some(m) = matches.get(c)
                && let Some(j) = m.parent_expert
            {
                aligned.push((pos, j, m.cka));
            }
        }
        alignment = json!({
            "parent": p.manifest.snapshot_id,
            "threshold": EXPERT_MATCH_CKA,
            "cka": cka,
            "matches": matches,
            "order": order_pos,
        });
    }
    let child_final = NeuralEncoder::new(tr.model.clone(), ts, meta.snapshot_id.clone());
    let (latents, residuals) = encode_probes_with_residuals(&child_final, &history, &ends, &probes)?;
    let probe_keys: Vec<ProbeKey> =
        probes.iter().map(|&(i, p)| (history[i].instrument.clone(), history[i].ends_ms[ends[i][p] - 1])).collect();
    let parent_assign = artifact::parent_assignment(hub, parent_graph.as_ref(), &history, &probe_keys).await;
    // the region discovery the family declares (spec 30 §6.2: codebook for fsq_hybrid with the
    // residual merge criterion, mapper for jepa_vae, density modes for families 2 and 4)
    let disc = fa_encoders::neural::regions::discovery_for(family, spec.region_radius, spec.max_regions);
    let mut experts = Vec::new();
    let mut regions = Vec::new();
    for (k, points) in latents.iter().enumerate() {
        let centroids = match residuals.get(k) {
            Some(r) => disc.discover_with_residuals(points, r),
            None => disc.discover(points),
        };
        let new_assign: Vec<Option<usize>> =
            points.iter().map(|p| disc.assign(&centroids, p).or_else(|| fa_patterns::nearest(&centroids, p))).collect();
        let empty = Vec::new();
        let no_ids = ParentIds::new();
        let m = align_expert(AlignInput {
            snapshot_id: &meta.snapshot_id,
            expert_index: k as u32,
            new_centroids: &centroids,
            radius: spec.region_radius,
            new_assign: &new_assign,
            old_assign: parent_assign.assign.get(k).unwrap_or(&empty),
            old_ids: parent_assign.ids.get(k).unwrap_or(&no_ids),
            carry_threshold: spec.carry_threshold,
        });
        regions.push(m.regions.clone());
        experts.push(m);
    }
    let probe_ts: Vec<i64> = probe_keys.iter().map(|(_, t)| *t).collect();
    let region_map = RegionMap {
        snapshot_id: meta.snapshot_id.clone(),
        parent_snapshot_id: meta.parent_snapshot_id.clone(),
        probe_set_version: probe_version(ts, &probe_ts),
        n_probes: probes.len(),
        carry_threshold: spec.carry_threshold,
        experts,
        note: parent_assign.note.clone(),
    };

    // 6. manifest + bundle
    let h = &post.health;
    let mut metrics: BTreeMap<String, f64> = BTreeMap::new();
    let n_bars = ds.n_bars();
    for (k, v) in [
        ("n_bars", n_bars as f64),
        ("n_windows", ds.train.len() as f64),
        ("n_val_windows", val.len() as f64),
        ("n_probes", probes.len() as f64),
        ("n_instruments", history.len() as f64),
        ("n_params", n_params as f64),
        ("train_steps", tr.step as f64),
        ("n_experts", tr.model.moe.n_active() as f64),
        ("experts_pruned", (before.len() - kept.len()) as f64),
        ("k_eff", h.k_eff),
        ("resid_corr_max", h.resid_corr_max),
        ("skill_min", h.skill_min()),
        ("mix_skill", post.mix_skill),
        ("val_loss", post.loss.total),
        ("val_pred_loss", post.loss.pred),
        ("val_ssl_loss", post.loss.ssl),
        ("val_decor", post.loss.decor),
        ("pre_prune_k_eff", pre.health.k_eff),
        ("parent_probe_coverage", parent_assign.covered as f64 / probes.len().max(1) as f64),
        ("carried_mass", region_map.carried_mass()),
        ("regions", region_map.n_regions() as f64),
        ("regions_matched", region_map.n_matched() as f64),
        ("mem_estimate_mb", f64::from(mem_estimate_mb(&cfg, n_params))),
        ("train_seconds", started.elapsed().as_secs_f64()),
    ] {
        metrics.insert(k.into(), v);
    }
    if let Some(last) = tr.log.last() {
        metrics.insert("train_loss".into(), last.total);
    }
    // order of h.skill / usage follows the pre-identity output order; report per final position
    let pos_of: Vec<usize> =
        tr.model.moe.active.iter().map(|e| kept.iter().position(|k| k == e).unwrap_or(0)).collect();
    for (pos, &src) in pos_of.iter().enumerate() {
        metrics.insert(format!("skill_e{pos}"), h.skill.get(src).copied().unwrap_or(0.0));
        metrics.insert(format!("usage_e{pos}"), h.usage.get(src).copied().unwrap_or(0.0));
        metrics.insert(format!("regions_e{pos}"), regions.get(pos).map_or(0, Vec::len) as f64);
    }
    if let Some(pe) = &parent_eval {
        metrics.insert("parent_mix_skill".into(), pe.mix_skill);
        metrics.insert("ssl_regression".into(), pe.mix_skill - post.mix_skill);
    }
    if let Some(a) = alignment.get("matches").and_then(Value::as_array) {
        metrics.insert("experts_matched".into(), a.iter().filter(|m| !m["parent_expert"].is_null()).count() as f64);
    }

    let decoder_stats = match (kind, svc.pg_url) {
        (Kind::Generation, Some(url)) => crate::feedback::latest_generation_stats(url).await,
        _ => None,
    };
    let feedback = json!({
        "decoder_stats": decoder_stats,
        "decoders": tr.feedback.iter().map(|f| f.summary()).collect::<Vec<_>>(),
        "source": feedback_source,
        "lambda_feedback": tr.model.cfg.lambda_feedback,
        "anchors": { "experts": tr.anchors.n_anchored(), "lambda_anchor": tr.model.cfg.lambda_anchor, "source": parent.as_ref().map(|p| p.manifest.snapshot_id.clone()) },
        "warm_start": warm,
        "expert_alignment": alignment,
    });
    let mut params = serde_json::to_value(&tr.model.cfg)?;
    params["context_steps"] = json!(cfg.context_steps);
    params["expert_dim"] = json!(cfg.moe.expert_dim);
    params["active_experts"] = json!(tr.model.moe.active);
    params["device"] = json!(device_label(&dev));
    params["device_note"] = json!(dev_note);
    params["region_radius"] = json!(spec.region_radius);
    params["max_regions"] = json!(spec.max_regions);
    params["region_discovery"] = json!(child.descriptor().region_discovery);
    params["health"] = serde_json::to_value(h)?;
    params["loss_log"] = serde_json::to_value(&tr.log)?;
    spec.adapter_guard.clone().unwrap_or_default().write_params(&mut params);

    let n_from = |src: Source| history.iter().filter(|h| h.source == src).count();
    let source = match (n_from(Source::Bars) > 0, n_from(Source::Trades) > 0) {
        (true, false) => "victoriametrics:fa_bar",
        (false, true) => "victoriametrics:fa_trade_price",
        _ => "victoriametrics:fa_bar+fa_trade_price",
    };
    let label = if history.len() == 1 { history[0].instrument.clone() } else { format!("multi:{}", history.len()) };
    let manifest = SnapshotManifest {
        snapshot_id: meta.snapshot_id.clone(),
        encoder_family: family.as_str().into(),
        timescale: ts,
        generation: meta.generation,
        parent_snapshot_id: meta.parent_snapshot_id.clone(),
        created_at: chrono::Utc::now().to_rfc3339(),
        params,
        probe_set: ProbeSetRef {
            version: region_map.probe_set_version.clone(),
            instrument: label.clone(),
            n_windows: probe_keys.len(),
            start_ms: probe_ts.iter().copied().min().unwrap_or_default(),
            end_ms: probe_ts.iter().copied().max().unwrap_or_default(),
            hash: crate::fit::fnv_keys(&probe_keys),
        },
        data: DataRef {
            source: source.into(),
            instrument: label,
            start_ms: history.iter().map(|h| h.start_ms).min().unwrap_or_default(),
            end_ms: history.iter().map(|h| h.end_ms).max().unwrap_or_default(),
            n_bars,
            instruments: history.iter().map(BarHistory::data_ref).collect(),
        },
        metrics,
        feedback: Some(feedback),
        seed: spec.seed,
    };
    let bundle = Bundle {
        manifest,
        stats: fa_encoders::NormStats { n_windows: 0, mean: vec![], std: vec![], gain: vec![] },
        regions,
        region_map,
        neural_weights: Some(tr.weights()?),
    };
    if ctl.stop.load(Ordering::Relaxed) == CANCEL {
        return Ok(stop_outcome(kind, &meta.snapshot_id, Status::Cancelled, "cancelled", None, tr.step, started));
    }
    (ctl.progress)(0.95, "writing snapshot".into());
    let written = artifact::write_bundle(svc.stores, &bundle, parent_graph.as_ref(), kind.as_str()).await?;
    // spec 32 §5: Expert nodes, ExpertOf, AlignedTo (identity across generations)
    let expert_nodes = match hub {
        Some(h) if written.helix => {
            let spec_by_expert = specialisation(&child_final, &history, &ends, &probes)?;
            write_expert_nodes(
                h,
                &bundle.manifest,
                &tr.model,
                &post.health,
                &pos_of,
                &aligned,
                parent.as_ref().map(|p| (p.manifest.snapshot_id.as_str(), p.encoder.model().moe.n_active())),
                &spec_by_expert,
            )
            .await
        }
        _ => json!(null),
    };
    if let (Some(blob), Some(_)) = (&svc.stores.blob, &ckpt_key) {
        for key in [artifact::checkpoint_key(&ctl.job_id), weights_key(&ctl.job_id)] {
            let _ = fa_blob::BlobStore::delete(blob, &key).await;
        }
    }
    let m = &bundle.manifest;
    (ctl.progress)(1.0, format!("snapshot {} written", m.snapshot_id));
    Ok(Outcome {
        status: Status::Ok,
        result: json!({
            "protocol": PROTOCOL,
            "kind": kind.job_kind(),
            "family": family.as_str(),
            "snapshot_id": m.snapshot_id,
            "timescale": m.timescale,
            "generation": m.generation,
            "parent_snapshot_id": m.parent_snapshot_id,
            "status": "candidate",
            "manifest_key": written.manifest_key,
            "persisted": written,
            "metrics": m.metrics,
            "health": h,
            "device": device_label(&dev),
            "probe_set": m.probe_set,
            "data": m.data,
            "region_map": {
                "regions": bundle.region_map.n_regions(),
                "matched": bundle.region_map.n_matched(),
                "carried_mass": bundle.region_map.carried_mass(),
                "note": bundle.region_map.note,
            },
            "feedback": m.feedback,
            "experts": expert_nodes,
            "resumed_from": resumed_from,
            "elapsed_ms": started.elapsed().as_millis() as u64,
        }),
        checkpoint_ref: String::new(),
        output_refs: written.manifest_key.iter().cloned().collect(),
        error: String::new(),
        snapshot_id: Some(m.snapshot_id.clone()),
    })
}

/// Gate mass per instrument of every expert (output position) on the probe windows: the
/// specialisation summary of the `Expert` node (spec 32 §5).
fn specialisation(
    enc: &NeuralEncoder,
    history: &[BarHistory],
    ends: &[Vec<usize>],
    probes: &[(usize, usize)],
) -> anyhow::Result<Vec<Value>> {
    let k = enc.sizing().max_experts as usize;
    let mut mass: Vec<BTreeMap<String, f64>> = vec![BTreeMap::new(); k];
    let mut i = 0;
    while i < probes.len() {
        let inst = probes[i].0;
        let j = probes[i..].iter().position(|p| p.0 != inst).map_or(probes.len(), |d| i + d);
        let win_ends: Vec<usize> = probes[i..j].iter().map(|&(s, p)| ends[s][p]).collect();
        for e in tokio::task::block_in_place(|| enc.encode_windows(&history[inst].bars, &win_ends))? {
            for (pos, g) in e.gates.iter().enumerate().take(k) {
                *mass[pos].entry(history[inst].instrument.clone()).or_default() += f64::from(*g);
            }
        }
        i = j;
    }
    Ok(mass
        .into_iter()
        .map(|m| {
            let total: f64 = m.values().sum();
            let mut top: Vec<(String, f64)> =
                m.into_iter().map(|(i, v)| (i, if total > 0.0 { v / total } else { 0.0 })).collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            top.truncate(5);
            json!({ "gate_mass": total, "top_instruments": top })
        })
        .collect())
}

/// Writes the `Expert` nodes of a neural snapshot with `ExpertOf`, the `AlignedTo` edges of
/// matched experts and `died_gen` on parent experts no child continues.
#[allow(clippy::too_many_arguments)]
async fn write_expert_nodes(
    hub: &fa_hub::HubClient,
    m: &SnapshotManifest,
    model: &fa_encoders::neural::NeuralModel,
    h: &fa_expert_space::health::HealthReport,
    pos_of: &[usize],
    aligned: &[(usize, usize, f64)],
    parent: Option<(&str, usize)>,
    spec_by_expert: &[Value],
) -> Value {
    let mut written = 0usize;
    let mut errors = Vec::new();
    for (pos, &src) in pos_of.iter().enumerate() {
        let continued = aligned.iter().find(|a| a.0 == pos);
        let born_gen = match (continued, parent) {
            (Some(a), Some((pid, _))) => match hub.snapshot_experts(pid).await {
                Ok(rows) => rows
                    .iter()
                    .find(|r| r.expert_id as usize == a.1)
                    .map_or(m.generation.saturating_sub(1), |r| r.born_gen),
                Err(_) => m.generation.saturating_sub(1),
            },
            _ => m.generation,
        };
        let e = fa_hub::NewExpert {
            snapshot_id: m.snapshot_id.clone(),
            timescale: m.timescale.to_string(),
            expert_id: pos as u32,
            dim: model.cfg.moe.expert_dim as u32,
            usage: h.usage.get(src).copied().unwrap_or(0.0),
            skill: h.skill.get(src).copied().unwrap_or(0.0),
            born_gen,
            specialisation: spec_by_expert.get(pos).map(Value::to_string).unwrap_or_default(),
        };
        match hub.upsert_expert(&e).await {
            Ok(()) => written += 1,
            Err(err) => errors.push(err.to_string()),
        }
    }
    let mut edges = 0usize;
    let mut retired = Vec::new();
    if let Some((pid, n_parent)) = parent {
        for &(pos, j, cka) in aligned {
            let a = fa_hub::NewAlignment {
                child_snapshot_id: m.snapshot_id.clone(),
                child_expert_id: pos as u32,
                parent_snapshot_id: pid.to_owned(),
                parent_expert_id: j as u32,
                cka,
                kind: "carry".into(),
            };
            match hub.align_expert(&a).await {
                Ok(()) => edges += 1,
                Err(err) => errors.push(err.to_string()),
            }
        }
        for j in (0..n_parent).filter(|j| !aligned.iter().any(|a| a.1 == *j)) {
            if hub.retire_expert(pid, j as u32, m.generation).await.is_ok() {
                retired.push(j);
            }
        }
    }
    json!({ "written": written, "aligned_to": edges, "parent_experts_retired": retired, "errors": errors })
}

fn dataset(history: &[BarHistory], cfg: &NeuralConfig) -> Dataset {
    let series = history.iter().map(|h| Series::new(h.instrument.clone(), h.bars.clone())).collect();
    Dataset::build(series, cfg.min_window, cfg.horizon(), cfg.val_fraction)
}

/// Probe latents per expert: `out[k][w]`.
fn encode_probes(
    enc: &NeuralEncoder,
    history: &[BarHistory],
    ends: &[Vec<usize>],
    probes: &[(usize, usize)],
) -> anyhow::Result<Vec<Vec<Vec<f32>>>> {
    Ok(encode_probes_with_residuals(enc, history, ends, probes)?.0)
}

/// Per expert the probe latents and the expert's next-step residual on `next_ret` (normalised
/// target − expert prediction; NaN for a probe without a next bar), the residuals the codebook
/// merge criterion of spec 30 §6.2 compares.
#[allow(clippy::type_complexity)]
fn encode_probes_with_residuals(
    enc: &NeuralEncoder,
    history: &[BarHistory],
    ends: &[Vec<usize>],
    probes: &[(usize, usize)],
) -> anyhow::Result<(Vec<Vec<Vec<f32>>>, Vec<Vec<f64>>)> {
    let k = enc.sizing().max_experts as usize;
    let norm = &enc.model().norm;
    let mut out: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(probes.len()); k];
    let mut res: Vec<Vec<f64>> = vec![Vec::with_capacity(probes.len()); k];
    let mut i = 0;
    while i < probes.len() {
        let inst = probes[i].0;
        let j = probes[i..].iter().position(|p| p.0 != inst).map_or(probes.len(), |d| i + d);
        let win_ends: Vec<usize> = probes[i..j].iter().map(|&(s, p)| ends[s][p]).collect();
        let enc_w = tokio::task::block_in_place(|| enc.encode_windows(&history[inst].bars, &win_ends))?;
        let feats = fa_encoders::neural::data::bar_features(&history[inst].bars);
        for (e, end) in enc_w.into_iter().zip(&win_ends) {
            let target = fa_encoders::neural::data::targets_at(&feats, *end).map(|t| norm.target(0, t[0]));
            for (k, mu) in e.mu.into_iter().enumerate() {
                out[k].push(mu);
                let r = match (target, e.pred.get(k).and_then(|p| p.first())) {
                    (Some(t), Some(p)) => f64::from(t - p),
                    _ => f64::NAN,
                };
                res[k].push(r);
            }
        }
        i = j;
    }
    Ok((out, res))
}

/// Output order after expert matching (spec 32 §6.5): a child expert matched to parent index
/// `j < n` takes position `j`; the others fill the remaining positions in their own order.
/// Returns child positions in output order.
pub fn reorder_by_parent(matches: &[fa_expert_space::health::ExpertMatch], n: usize) -> Vec<usize> {
    let mut slots: Vec<Option<usize>> = vec![None; n];
    for m in matches {
        if let Some(j) = m.parent_expert
            && j < n
            && slots[j].is_none()
        {
            slots[j] = Some(m.expert);
        }
    }
    let placed: Vec<usize> = slots.iter().flatten().copied().collect();
    let mut rest = (0..n).filter(|c| !placed.contains(c));
    slots.into_iter().map(|s| s.or_else(|| rest.next()).unwrap_or(0)).collect()
}

// ── genome sweep (no writes) ────────────────────────────────────────────────

/// Spec 32 §6.3 feasibility bound of the promotion checklist for two experts sharing windows.
/// Write each expert's prediction as `ŷ_k = f + d_k` (`f` = the conditional mean, `d_k` any
/// function of the window, so `d_k ⟂ ε` out of sample). The scalarised residual is `ē − d̄_k`,
/// its correlation is smallest for `d̄_j = −d̄_i`: `(V − D) / (V + D)` with `V = var(ē)` and
/// `D = var(d̄)` ≤ the mean per-target deviation energy, while skill is `R̄² − D` at best. Both
/// `|corr| ≤ c` and `skill ≥ s` are reachable only if `R̄² − s ≥ V·(1 − c)/(1 + c)`. Returns
/// `R̄² − s − V·(1 − c)/(1 + c)` with `R̄²` and `V` estimated by the trained mixture (best
/// predictor at hand: it under-estimates `R̄²` and over-estimates `V` by the same deficit).
pub fn feasibility_margin(r2: f64, v: f64, min_skill: f64, max_corr: f64) -> f64 {
    r2 - min_skill - v * (1.0 - max_corr) / (1.0 + max_corr)
}

/// Trains every genome override of `grid` on the same history and reports the spec 32 §6.4
/// health and the promotion-checklist verdict of each (`trainer sweep`). Nothing is written:
/// this is how a genome that passes the checklist is searched for before a real `pretrain`.
pub fn sweep(
    family: NeuralFamily,
    ts: Timescale,
    history: &[BarHistory],
    grid: &[Value],
    seed: u64,
    dev: &candle_core::Device,
    mut report: impl FnMut(&Value),
) -> anyhow::Result<Vec<Value>> {
    let mut out = Vec::with_capacity(grid.len());
    for (i, over) in grid.iter().enumerate() {
        let started = Instant::now();
        let cfg = config_for(family, ts, Some(over))?;
        let ds = dataset(history, &cfg);
        if ds.train.is_empty() {
            anyhow::bail!("no training windows");
        }
        let mut tr = NeuralTrainer::new(cfg.clone(), &ds, seed, dev)?;
        tr.train_steps(&ds, cfg.steps)?;
        tr.refit_heads(&ds, cfg.head_refit_steps, 4096)?;
        let val = if ds.val.is_empty() { ds.train.clone() } else { ds.val.clone() };
        let pre = tr.evaluate(&ds, &val)?;
        tr.prune(&pre.health.usage, 2);
        let post = tr.evaluate(&ds, &val)?;
        let h = &post.health;
        let pass = h.k_eff >= 2.0 && h.resid_corr_max <= 0.1 && h.skill_min() >= 0.02;
        // the same measures in sample (generalisation gap vs optimisation failure)
        let train_sub: Vec<(usize, usize)> =
            crate::fit::evenly(ds.train.len(), 2000).into_iter().map(|i| ds.train[i]).collect();
        let tr_eval = tr.evaluate(&ds, &train_sub)?;
        let row = json!({
            "i": i,
            "overrides": over,
            "pass": pass,
            "k_eff": h.k_eff,
            "resid_corr_max": h.resid_corr_max,
            "skill_min": h.skill_min(),
            "skill": h.skill,
            "usage": h.usage,
            "mix_skill": post.mix_skill,
            "target_skill": post.target_skill,
            "mix_resid_var": post.mix_resid_var,
            "feasibility_margin": feasibility_margin(post.mix_skill, post.mix_resid_var, 0.02, 0.1),
            "pre_prune": { "k_eff": pre.health.k_eff, "resid_corr_max": pre.health.resid_corr_max, "skill": pre.health.skill },
            "n_experts": tr.model.moe.n_active(),
            "train_health": {
                "k_eff": tr_eval.health.k_eff, "resid_corr_max": tr_eval.health.resid_corr_max,
                "skill": tr_eval.health.skill, "mix_skill": tr_eval.mix_skill,
            },
            "val_loss": post.loss,
            "train_windows": ds.train.len(),
            "val_windows": val.len(),
            "seconds": started.elapsed().as_secs_f64(),
        });
        report(&row);
        out.push(row);
    }
    Ok(out)
}

// ── online adapter of a neural snapshot ─────────────────────────────────────

/// Mean over features of `|mean_new − mean_ref| / std_ref` of two input normalisations.
pub fn norm_drift(reference: &fa_encoders::neural::InputNorm, new: &fa_encoders::neural::InputNorm) -> f64 {
    let n = reference.mean.len().min(new.mean.len()).max(1);
    reference
        .mean
        .iter()
        .zip(&reference.std)
        .zip(&new.mean)
        .map(|((m, s), nm)| f64::from((nm - m).abs() / s.max(1e-6)))
        .sum::<f64>()
        / n as f64
}

/// Online adapter of a neural snapshot (spec 36 §6.5), two layers:
/// 1. the **input adapter** — the market front-end's normalisation re-fitted on the recent
///    window (drift guard `max_drift` on the mean |Δμ|/σ of the inputs);
/// 2. a **LoRA adapter** on `bb.out` and the expert projectors trained on the recent windows
///    with the SSL next-step objective ([`fa_encoders::neural::adapter`]), stored as
///    `adapters/<snapshot>/<ms>.lora.safetensors`; frozen (not served) beyond
///    `max_delta_norm` / `max_latent_shift` of the snapshot's adapter guard.
///
/// Same artifact path, `AdapterState` node and runtime application path as the statistical
/// adapter; model-runtime applies `input_norm` and folds a non-frozen LoRA into the weights.
pub async fn run_adapter(
    spec: crate::run::AdapterSpec,
    svc: &Services<'_>,
    ctl: &Control,
    manifest: SnapshotManifest,
    weights: Vec<u8>,
    started: Instant,
) -> anyhow::Result<Outcome> {
    let ts = manifest.timescale;
    let snapshot_id = manifest.snapshot_id.clone();
    let (dev, _) = fa_encoders::neural::device_from_env();
    let enc = NeuralEncoder::from_manifest_on(&manifest, &weights, &dev)?;
    let guard = manifest.adapter_guard();
    let max_drift = spec.max_drift.unwrap_or(guard.max_drift);
    (ctl.progress)(0.1, format!("adapting {snapshot_id} (input normalisation) on the last {:.2} h", spec.window_hours));
    let (start_ms, end_ms) = data::window(ts, spec.window_hours, data::now_ms());
    let history = data::load_one(svc.tsdb, ts, spec.instrument.as_deref(), start_ms, end_ms, None).await?;
    match ctl.stop.load(Ordering::Relaxed) {
        CANCEL => {
            return Ok(stop_outcome(Kind::Adapter, &snapshot_id, Status::Cancelled, "cancelled", None, 0, started));
        }
        PREEMPT => {
            return Ok(stop_outcome(
                Kind::Adapter,
                &snapshot_id,
                Status::Preempted,
                "preempted by the gpu arbiter",
                None,
                0,
                started,
            ));
        }
        _ => {}
    }
    let cfg = &enc.model().cfg;
    let ds =
        Dataset::build(vec![Series::new(history.instrument.clone(), history.bars.clone())], cfg.min_window, 1, 0.0);
    let norm = fa_encoders::neural::InputNorm::fit(&ds);
    let drift = norm_drift(&enc.model().norm, &norm);
    let frozen = drift > max_drift;
    // weight-level LoRA on top of the re-fitted input normalisation (spec 36 §6.5)
    let now = data::now_ms();
    let mut lora = Value::Null;
    if !spec.no_lora && !frozen && !ds.train.is_empty() {
        let mut model = enc.model().clone();
        model.norm = norm.clone();
        let lcfg = spec.lora.clone().unwrap_or_default();
        let ads = Dataset::build(
            vec![Series::new(history.instrument.clone(), history.bars.clone())],
            cfg.min_window,
            cfg.horizon(),
            0.0,
        );
        (ctl.progress)(0.3, format!("LoRA rank {} on {} windows ({} steps)", lcfg.rank, ads.train.len(), lcfg.steps));
        let fitted = tokio::task::block_in_place(|| {
            fa_encoders::neural::adapter::train_lora(&model, &ads, &lcfg, now as u64, &dev)
        });
        lora = match fitted {
            Ok((set, rep)) => {
                let lora_frozen = rep.delta_norm > guard.max_delta_norm || rep.latent_shift > guard.max_latent_shift;
                let key = fa_blob::ObjectKey {
                    bucket: fa_blob::buckets::MODELS,
                    key: format!("adapters/{snapshot_id}/{now}.lora.safetensors"),
                };
                let bytes = set.to_bytes(
                    [("snapshot_id".to_owned(), snapshot_id.clone()), ("ts_ms".to_owned(), now.to_string())].into(),
                )?;
                let mut stored = false;
                if let Some(blob) = &svc.stores.blob {
                    fa_blob::BlobStore::put(blob, &key, bytes.clone(), fa_blob::OCTET_STREAM).await?;
                    stored = true;
                }
                if let Some(dir) = &svc.stores.out_dir {
                    let path = dir.join("adapters").join(&snapshot_id);
                    std::fs::create_dir_all(&path)?;
                    std::fs::write(path.join(format!("{now}.lora.safetensors")), &bytes)?;
                }
                json!({
                    "key": if stored { Value::String(key.key) } else { Value::Null },
                    "rank": set.rank, "alpha": set.alpha, "config": lcfg, "seed": now, "report": rep,
                    "frozen": lora_frozen,
                    "guard": { "max_delta_norm": guard.max_delta_norm, "max_latent_shift": guard.max_latent_shift },
                })
            }
            Err(e) => json!({ "error": e.to_string() }),
        };
    }
    let artifact = json!({
        "protocol": PROTOCOL,
        "snapshot_id": snapshot_id,
        "family": manifest.encoder_family,
        "timescale": ts,
        "ts_ms": now,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "window": { "start_ms": history.start_ms, "end_ms": history.end_ms, "instrument": history.instrument, "source": history.source.as_str(), "n_bars": history.bars.len() },
        "n_windows": ds.train.len(),
        "input_norm": norm,
        "lora": lora,
        "reference": "snapshot input normalisation",
        "drift": drift,
        "max_drift": max_drift,
        "guard": guard,
        "frozen": frozen,
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
    let adapter_state = match svc.stores.hub.as_ref() {
        Some(hub) => {
            let row = fa_hub::NewAdapterState {
                snapshot_id: snapshot_id.clone(),
                timescale: ts.to_string(),
                version: u64::try_from(now).unwrap_or_default(),
                garage_key: if persisted["garage"] == json!(true) { key.key.clone() } else { String::new() },
                drift: Some(drift),
                max_drift,
                frozen,
                n_windows: ds.train.len() as u64,
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
    (ctl.progress)(1.0, format!("adapter {} drift={drift:.3} frozen={frozen}", key.key));
    Ok(Outcome {
        status: Status::Ok,
        result: json!({
            "protocol": PROTOCOL, "kind": "train.adapter", "family": manifest.encoder_family, "snapshot_id": snapshot_id,
            "timescale": ts, "artifact_key": key.key, "persisted": persisted, "n_bars": history.bars.len(),
            "source": history.source.as_str(), "adapter_state": adapter_state,
            "adapter": if lora.get("key").is_some_and(|k| !k.is_null()) { "input_norm+lora" } else { "input_norm" },
            "lora": lora,
            "n_windows": ds.train.len(), "drift": drift, "frozen": frozen, "max_drift": max_drift,
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
    fn family_and_config_overrides() {
        let mut spec = FitSpec::new(Timescale::Minute);
        assert_eq!(family_of(&spec).unwrap(), None);
        spec.family = Some("statistical".into());
        assert_eq!(family_of(&spec).unwrap(), None);
        spec.family = Some("fsq_hybrid".into());
        assert_eq!(family_of(&spec).unwrap(), Some(NeuralFamily::FsqHybrid));
        spec.family = Some("nope".into());
        assert!(family_of(&spec).is_err());
        let cfg = config_for(
            NeuralFamily::JepaVae,
            Timescale::Hour,
            Some(&json!({ "steps": 50, "backbone": "gru", "moe": { "max_experts": 6 } })),
        )
        .unwrap();
        assert_eq!(
            (cfg.steps, cfg.backbone, cfg.moe.max_experts, cfg.context_steps),
            (50, fa_encoders::neural::Backbone::Gru, 6, 48)
        );
        assert_eq!(cfg.moe.expert_dim, 8, "unspecified nested fields keep their defaults");
        assert!(config_for(NeuralFamily::JepaVae, Timescale::Hour, Some(&json!({ "steps": "x" }))).is_err());
        assert_eq!(
            snapshot_id(NeuralFamily::FsqHybrid, Timescale::Minute, 2, 1_700_000_000_000),
            "fsq-v1-1m-g2-1700000000"
        );
        assert!(mem_estimate_mb(&cfg, 50_000) >= 16);
    }

    #[test]
    fn matched_experts_take_their_parent_index() {
        use fa_expert_space::health::ExpertMatch;
        let m = |e, p: Option<usize>| ExpertMatch { expert: e, parent_expert: p, cka: 0.9 };
        // child 0 continues parent 2, child 2 continues parent 0, child 1 is new
        assert_eq!(reorder_by_parent(&[m(0, Some(2)), m(1, None), m(2, Some(0))], 3), vec![2, 1, 0]);
        // parent index beyond the child's count: the expert fills a free slot
        assert_eq!(reorder_by_parent(&[m(0, Some(5)), m(1, Some(0))], 2), vec![1, 0]);
        assert_eq!(reorder_by_parent(&[], 2), vec![0, 1]);
    }
}
