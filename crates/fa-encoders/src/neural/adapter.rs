//! Weight-level online adapters of a neural snapshot (spec 36 §6.5): LoRA factors on the
//! backbone's output projection (`bb.out`) and on every active expert projector
//! (`moe.proj<e>`), trained on the recent windows (the replay buffer) with the **SSL next-step
//! objective only** — per-expert next-step loss, mixture loss and the family's SSL term; no
//! decorrelation, balancing, KL, prior, anchor or decoder feedback — plus an L2 anchor of the
//! factors toward the identity adapter. The snapshot's weights stay frozen.
//!
//! Guards: `delta_norm = ‖ΔW‖_F / ‖W‖_F` and `latent_shift` = mean ‖Δμ_k‖ over the probe windows
//! relative to the spread of the snapshot's own latents. The trainer freezes (does not publish
//! for serving) an adapter beyond [`crate::AdapterGuard::max_delta_norm`] /
//! [`crate::AdapterGuard::max_latent_shift`].

use candle_core::{Device, Result};
use fa_core::rng::Rng;
use fa_expert_space::{
    lora::{LoraSet, LoraView},
    nn::{Adam, ParamStore},
};
use serde::{Deserialize, Serialize};

use super::{data::Dataset, model::NeuralModel, train::NeuralTrainer};

/// Hyper-parameters of a LoRA adapter (spec 36 §7 `adapter_rank`, `adapter_lr`; CPU-sized).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f64,
    pub lr: f64,
    pub steps: usize,
    pub batch: usize,
    /// Weight of `Σ‖A‖² + ‖B‖²`.
    pub l2: f64,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self { rank: 8, alpha: 8.0, lr: 1e-4, steps: 100, batch: 64, l2: 1e-3 }
    }
}

/// What an adapter fit measured.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LoraReport {
    pub targets: Vec<String>,
    pub n_windows: usize,
    pub steps: usize,
    /// SSL next-step objective on the windows before / after.
    pub loss_before: f64,
    pub loss_after: f64,
    pub delta_norm: f64,
    pub latent_shift: f64,
}

/// The weights an adapter acts on: `bb.out.w` and the active experts' projectors.
pub fn lora_targets(model: &NeuralModel) -> Vec<String> {
    let mut t = vec!["bb.out.w".to_owned()];
    t.extend(model.moe.projector_weights());
    t
}

/// A trainer whose objective is the SSL next-step part only.
fn ssl_only(model: &NeuralModel) -> NeuralTrainer {
    let mut m = model.clone();
    let c = &mut m.cfg;
    c.moe.lambda_dec = 0.0;
    c.moe.lambda_bal = 0.0;
    c.moe.beta_kl = 0.0;
    c.moe.router_noise = 0.0;
    c.lambda_prior = 0.0;
    c.lambda_anchor = 0.0;
    c.lambda_feedback = 0.0;
    c.lambda_skill = 0.0;
    c.decor_dual_lr = 0.0;
    NeuralTrainer::from_model(m, 0)
}

fn objective(
    tr: &NeuralTrainer,
    w: &dyn fa_expert_space::nn::Weights,
    ds: &Dataset,
    idx: &[(usize, usize)],
) -> Result<f64> {
    let mut acc = 0.0;
    let mut n = 0.0;
    for chunk in idx.chunks(256) {
        let b = tr.batch(ds, chunk)?;
        let (_, log, _) = tr.losses_w(w, &b, None)?;
        acc += log.total * chunk.len() as f64;
        n += chunk.len() as f64;
    }
    Ok(if n > 0.0 { acc / n } else { 0.0 })
}

/// Mean ‖μ_adapted − μ_base‖ per expert over `idx`, relative to the mean distance of the base
/// latents to their centroid.
fn latent_shift(model: &NeuralModel, lora: &LoraSet, ds: &Dataset, idx: &[(usize, usize)]) -> Result<f64> {
    let tr = NeuralTrainer::from_model(model.clone(), 0);
    let view = LoraView { base: &model.ps, lora };
    let (mut shift, mut spread, mut n) = (0.0f64, 0.0f64, 0usize);
    for chunk in idx.chunks(256) {
        let b = tr.batch(ds, chunk)?;
        let base = model.forward_w(&model.ps, &b.x, None, None)?.out.mu.to_vec3::<f32>()?;
        let ad = model.forward_w(&view, &b.x, None, None)?.out.mu.to_vec3::<f32>()?;
        let k = base.first().map_or(0, Vec::len);
        let d = base.first().and_then(|r| r.first()).map_or(0, Vec::len);
        let mut centre = vec![vec![0.0f64; d]; k];
        for row in &base {
            for (e, v) in row.iter().enumerate() {
                for (j, x) in v.iter().enumerate() {
                    centre[e][j] += f64::from(*x) / base.len() as f64;
                }
            }
        }
        for (rb, ra) in base.iter().zip(&ad) {
            for e in 0..k {
                let dist = |a: &[f32], b: &[f64]| {
                    a.iter().zip(b).map(|(x, y)| (f64::from(*x) - y).powi(2)).sum::<f64>().sqrt()
                };
                let delta: f64 = rb[e].iter().zip(&ra[e]).map(|(x, y)| f64::from(x - y).powi(2)).sum::<f64>().sqrt();
                shift += delta;
                spread += dist(&rb[e], &centre[e]);
                n += 1;
            }
        }
    }
    Ok(if spread > 0.0 && n > 0 { shift / spread } else { 0.0 })
}

/// Trains a LoRA adapter of `model` on every window of `ds` (train ∪ val: the recent replay
/// buffer). Deterministic for a seed.
pub fn train_lora(
    model: &NeuralModel,
    ds: &Dataset,
    cfg: &LoraConfig,
    seed: u64,
    dev: &Device,
) -> Result<(LoraSet, LoraReport)> {
    let windows: Vec<(usize, usize)> = ds.train.iter().chain(&ds.val).copied().collect();
    if windows.is_empty() {
        candle_core::bail!("no adapter windows");
    }
    let targets = lora_targets(model);
    let lora = LoraSet::new(&model.ps, &targets, cfg.rank, cfg.alpha, seed, dev)?;
    let tr = ssl_only(model);
    let eval_idx: Vec<(usize, usize)> = {
        let step = (windows.len() / 512).max(1);
        windows.iter().step_by(step).copied().collect()
    };
    let loss_before = objective(&tr, &model.ps, ds, &eval_idx)?;
    let mut opt = Adam::new(cfg.lr, 0.0);
    for s in 0..cfg.steps {
        let mut rng = Rng::seed(seed ^ (s as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let bsz = cfg.batch.min(windows.len()).max(1);
        let idx: Vec<(usize, usize)> = (0..bsz).map(|_| windows[rng.below(windows.len())]).collect();
        let b = tr.batch(ds, &idx)?;
        let view = LoraView { base: &model.ps, lora: &lora };
        let (loss, _, _) = tr.losses_w(&view, &b, None)?;
        let loss = (loss + lora.l2()?.affine(cfg.l2, 0.0)?)?;
        opt.step(&lora.ps, &loss)?;
    }
    let view = LoraView { base: &model.ps, lora: &lora };
    let loss_after = objective(&tr, &view, ds, &eval_idx)?;
    let report = LoraReport {
        targets: lora.targets.clone(),
        n_windows: windows.len(),
        steps: cfg.steps,
        loss_before,
        loss_after,
        delta_norm: lora.delta_norm(&model.ps)?,
        latent_shift: latent_shift(model, &lora, ds, &eval_idx)?,
    };
    Ok((lora, report))
}

/// The model with an adapter folded into an independent copy of its weights (serving).
pub fn merged(model: &NeuralModel, lora: &LoraSet) -> Result<NeuralModel> {
    let mut ps = ParamStore::new(0, model.device());
    ps.insert_all(&model.ps.snapshot()?.into_iter().collect(), "")?;
    lora.merge_into(&mut ps)?;
    Ok(NeuralModel { ps, ..model.clone() })
}
