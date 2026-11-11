//! Neural sequence encoders (spec 31 §6.2 families 1 and 3) on candle.
//!
//! A window of `context_steps` bars → market front-end ([`super::data`]) → **backbone** → pooled
//! hidden `h ∈ R^d_model` → learned expert space ([`fa_expert_space::moe`]: router, per-expert
//! `(mu, logvar)`, per-expert next-step predictors).
//!
//! Backbones (a family gene, [`Backbone`]):
//! * `tcn`: causal dilated temporal convolutions (kernel 3, dilation `2^l`), residual + RMS norm,
//!   pooled as `[last step, mean over time]`;
//! * `gru`: a GRU over the window, pooled the same way.
//!
//! Families ([`NeuralFamily`]):
//! * `fsq_hybrid` (family 1): FSQ-quantised leading dims + continuous residual; self-supervised
//!   by reconstructing the last `recon_steps` bars of the window from the gated expert latents;
//! * `jepa_vae` (family 3): Gaussian (VAE) expert heads; self-supervised JEPA-style by predicting
//!   the **EMA target encoder's** embedding of the window `jepa_horizon` steps ahead from the
//!   gated expert latents (no reconstruction of noisy inputs).
//!
//! Both add the spec 32 terms (per-expert next-step loss, D-008 residual decorrelation, router
//! load balancing, KL) and the mixture next-step loss that trains the router.

use std::collections::BTreeMap;

use candle_core::{D, Device, Result, Tensor};
use fa_expert_space::{
    moe::{MoeConfig, MoeExpertSpace},
    nn::{Init, Linear, ParamStore},
};
use serde::{Deserialize, Serialize};

use super::data::{InputNorm, N_FEATURES, N_TARGETS};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuralFamily {
    FsqHybrid,
    JepaVae,
}

impl NeuralFamily {
    /// `manifest.encoder_family` / Helix `Snapshot.encoder_family`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FsqHybrid => crate::FamilyId::FsqHybrid.as_str(),
            Self::JepaVae => crate::FamilyId::JepaVae.as_str(),
        }
    }

    /// Short prefix of snapshot ids (`fsq-v1-<ts>-g<gen>-<secs>`).
    pub fn id_prefix(self) -> &'static str {
        match self {
            Self::FsqHybrid => "fsq-v1",
            Self::JepaVae => "jepa-v1",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fsq_hybrid" | "fsq" => Some(Self::FsqHybrid),
            "jepa_vae" | "jepa" => Some(Self::JepaVae),
            _ => None,
        }
    }

    pub const ALL: [NeuralFamily; 2] = [Self::FsqHybrid, Self::JepaVae];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backbone {
    Tcn,
    Gru,
}

/// Genome of a neural encoder (spec 31 §4.1 `Config`). Small CPU-sized defaults: a fit over a few
/// thousand windows runs in about a minute on 12 cores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuralConfig {
    pub family: NeuralFamily,
    pub backbone: Backbone,
    pub context_steps: usize,
    pub d_model: usize,
    pub layers: usize,
    pub moe: MoeConfig,
    /// Bars reconstructed by `fsq_hybrid`.
    pub recon_steps: usize,
    /// Steps ahead the `jepa_vae` target window ends.
    pub jepa_horizon: usize,
    /// EMA decay of the JEPA target encoder.
    pub ema: f64,
    pub lambda_next: f64,
    /// Share of the per-expert next-step loss that is gate-weighted (0 = every window counts
    /// equally for every expert, 1 = experts answer only for the windows routed to them).
    #[serde(default = "default_gated_pred")]
    pub gated_pred: f64,
    pub lambda_mix: f64,
    pub lambda_ssl: f64,
    /// D-007 anchor toward the parent's region centroids (0 = off; only used with a parent).
    pub lambda_anchor: f64,
    /// Temperature of the anchor's soft-min over centroids.
    pub anchor_tau: f64,
    /// D-009 decoder-feedback weight.
    pub lambda_feedback: f64,
    pub lr: f64,
    pub weight_decay: f64,
    pub batch: usize,
    pub steps: usize,
    pub min_window: usize,
    pub val_fraction: f64,
}

fn default_gated_pred() -> f64 {
    0.5
}

impl NeuralConfig {
    pub fn new(family: NeuralFamily) -> Self {
        Self {
            family,
            backbone: Backbone::Tcn,
            context_steps: 64,
            d_model: 32,
            layers: 4,
            moe: MoeConfig {
                fsq_levels: if family == NeuralFamily::FsqHybrid { vec![8, 5, 5, 5] } else { Vec::new() },
                ..MoeConfig::default()
            },
            recon_steps: 8,
            jepa_horizon: 4,
            ema: 0.99,
            lambda_next: 1.0,
            gated_pred: default_gated_pred(),
            lambda_mix: 1.0,
            lambda_ssl: 1.0,
            lambda_anchor: 0.0,
            anchor_tau: 0.1,
            lambda_feedback: 0.0,
            lr: 3e-3,
            weight_decay: 1e-4,
            batch: 64,
            steps: 600,
            min_window: 16,
            val_fraction: 0.15,
        }
    }

    /// The ahead-steps a window needs after its end (target bar / JEPA target window).
    pub fn horizon(&self) -> usize {
        match self.family {
            NeuralFamily::FsqHybrid => 1,
            NeuralFamily::JepaVae => self.jepa_horizon.max(1),
        }
    }
}

/// Where a forward pass reads its parameters (the trainable store, an EMA copy, a frozen parent).
pub trait Weights {
    fn w(&self, name: &str) -> Result<Tensor>;
}

impl Weights for ParamStore {
    fn w(&self, name: &str) -> Result<Tensor> {
        self.tensor(name).ok_or_else(|| candle_core::Error::Msg(format!("missing parameter {name}")))
    }
}

impl Weights for BTreeMap<String, Tensor> {
    fn w(&self, name: &str) -> Result<Tensor> {
        self.get(name).cloned().ok_or_else(|| candle_core::Error::Msg(format!("missing parameter {name}")))
    }
}

const KERNEL: usize = 3;

fn rms_norm_channels(x: &Tensor) -> Result<Tensor> {
    // x [B, d, L]: normalise over channels
    let ms = (x.sqr()?.mean_keepdim(1)? + 1e-5)?.sqrt()?;
    x.broadcast_div(&ms)
}

fn sigmoid(x: &Tensor) -> Result<Tensor> {
    (x.neg()?.exp()? + 1.0)?.recip()
}

fn lin(w: &dyn Weights, name: &str, x: &Tensor) -> Result<Tensor> {
    x.broadcast_matmul(&w.w(&format!("{name}.w"))?)?.broadcast_add(&w.w(&format!("{name}.b"))?)
}

/// Creates the backbone parameters.
fn init_backbone(ps: &mut ParamStore, cfg: &NeuralConfig) -> Result<()> {
    let d = cfg.d_model;
    Linear::new(ps, "bb.in", N_FEATURES, d, true)?;
    match cfg.backbone {
        Backbone::Tcn => {
            for l in 0..cfg.layers {
                ps.get(&format!("bb.conv{l}.w"), &[d, d, KERNEL], Init::Uniform { fan_in: d * KERNEL, gain: 1.0 })?;
                ps.get(&format!("bb.conv{l}.b"), &[d], Init::Zeros)?;
            }
        }
        Backbone::Gru => {
            for g in ["z", "r", "h"] {
                Linear::new(ps, &format!("bb.gru.{g}"), 2 * d, d, true)?;
            }
        }
    }
    Linear::new(ps, "bb.out", 2 * d, d, true)?;
    Ok(())
}

/// Backbone forward: `x [B, L, F]` → `h [B, d]`.
pub fn backbone_forward(w: &dyn Weights, cfg: &NeuralConfig, x: &Tensor) -> Result<Tensor> {
    let u = lin(w, "bb.in", x)?; // [B, L, d]
    let (b, l, d) = u.dims3()?;
    let (last, mean) = match cfg.backbone {
        Backbone::Tcn => {
            let mut u = u.transpose(1, 2)?.contiguous()?; // [B, d, L]
            for layer in 0..cfg.layers {
                let dil = 1usize << layer.min(8);
                let pad = (KERNEL - 1) * dil;
                let y = u.pad_with_zeros(2, pad, 0)?.conv1d(&w.w(&format!("bb.conv{layer}.w"))?, 0, 1, dil, 1)?;
                let y = y.broadcast_add(&w.w(&format!("bb.conv{layer}.b"))?.reshape((1, d, 1))?)?.gelu()?;
                u = rms_norm_channels(&(u + y)?)?;
            }
            (u.narrow(2, l - 1, 1)?.squeeze(2)?, u.mean(2)?)
        }
        Backbone::Gru => {
            let mut h = Tensor::zeros((b, d), u.dtype(), u.device())?;
            let mut acc = h.clone();
            for t in 0..l {
                let xt = u.narrow(1, t, 1)?.squeeze(1)?;
                let xh = Tensor::cat(&[&xt, &h], 1)?;
                let z = sigmoid(&lin(w, "bb.gru.z", &xh)?)?;
                let r = sigmoid(&lin(w, "bb.gru.r", &xh)?)?;
                let cand = lin(w, "bb.gru.h", &Tensor::cat(&[&xt, &(r * &h)?], 1)?)?.tanh()?;
                h = ((z.affine(-1.0, 1.0)? * &h)? + (z * cand)?)?;
                acc = (acc + &h)?;
            }
            (h, acc.affine(1.0 / l as f64, 0.0)?)
        }
    };
    lin(w, "bb.out", &Tensor::cat(&[&last, &mean], 1)?)?.gelu()
}

/// Backbone + expert space + the family's self-supervised head.
#[derive(Debug, Clone)]
pub struct NeuralModel {
    pub cfg: NeuralConfig,
    pub ps: ParamStore,
    pub moe: MoeExpertSpace,
    pub norm: InputNorm,
}

/// One batch, already normalised.
#[derive(Debug, Clone)]
pub struct Batch {
    /// `[B, L, F]`
    pub x: Tensor,
    /// Next-step targets `[B, T]`.
    pub y: Tensor,
    /// Reconstruction target `[B, recon_steps · F]` (fsq_hybrid).
    pub recon: Option<Tensor>,
    /// Window `jepa_horizon` steps ahead `[B, L, F]` (jepa_vae).
    pub future: Option<Tensor>,
}

/// Forward results the losses and the encoder read.
pub struct Forward {
    pub h: Tensor,
    pub out: fa_expert_space::moe::MoeOutput,
}

impl NeuralModel {
    pub fn new(cfg: NeuralConfig, norm: InputNorm, seed: u64, dev: &Device) -> Result<Self> {
        let mut ps = ParamStore::new(seed, dev);
        init_backbone(&mut ps, &cfg)?;
        let moe = MoeExpertSpace::new(&mut ps, &cfg.moe, cfg.d_model, N_TARGETS)?;
        let ssl_out = match cfg.family {
            NeuralFamily::FsqHybrid => cfg.recon_steps * N_FEATURES,
            NeuralFamily::JepaVae => cfg.d_model,
        };
        for e in 0..cfg.moe.max_experts {
            ps.get(
                &format!("ssl.h{e}"),
                &[cfg.moe.expert_dim, ssl_out],
                Init::Uniform { fan_in: cfg.moe.expert_dim, gain: 1.0 },
            )?;
        }
        ps.get("ssl.b", &[ssl_out], Init::Zeros)?;
        Ok(Self { cfg, ps, moe, norm })
    }

    pub fn device(&self) -> &Device {
        self.ps.device()
    }

    pub fn forward(&self, x: &Tensor, noise: Option<&Tensor>) -> Result<Forward> {
        let h = backbone_forward(&self.ps, &self.cfg, x)?;
        let out = self.moe.forward(&self.ps, &h, noise)?;
        Ok(Forward { h, out })
    }

    /// Gated self-supervised head: `Σ_k p_k · mu_k H_k + b`.
    pub fn ssl_head(&self, f: &Forward) -> Result<Tensor> {
        let mut acc: Option<Tensor> = None;
        for (pos, &e) in self.moe.active.iter().enumerate() {
            let mu = f.out.mu.narrow(1, pos, 1)?.squeeze(1)?;
            let p = f.out.probs.narrow(1, pos, 1)?;
            let y = mu.matmul(&self.ps.w(&format!("ssl.h{e}"))?)?.broadcast_mul(&p)?;
            acc = Some(match acc {
                Some(a) => (a + y)?,
                None => y,
            });
        }
        acc.ok_or_else(|| candle_core::Error::Msg("no active experts".into()))?.broadcast_add(&self.ps.w("ssl.b")?)
    }

    /// Mixture next-step prediction `Σ_k p_k ŷ_k` `[B, T]`.
    pub fn mixture(f: &Forward) -> Result<Tensor> {
        f.out.pred.broadcast_mul(&f.out.probs.unsqueeze(2)?)?.sum(1)
    }

    /// Backbone parameter names (the JEPA EMA target copies these).
    pub fn backbone_names(&self) -> Vec<String> {
        self.ps.vars().map(|(k, _)| k.clone()).filter(|k| k.starts_with("bb.")).collect()
    }

    /// Normalised `[B, L, F]` input of windows `feats[..end]`.
    pub fn input(&self, windows: &[(&[[f32; N_FEATURES]], usize)]) -> Result<Tensor> {
        let l = self.cfg.context_steps;
        let mut flat = Vec::with_capacity(windows.len() * l * N_FEATURES);
        for (feats, end) in windows {
            flat.extend(self.norm.window(feats, *end, l));
        }
        Tensor::from_vec(flat, (windows.len(), l, N_FEATURES), self.device())
    }
}

/// `[B, L, F]` → the last `r` steps flattened `[B, r·F]`.
pub fn recon_target(x: &Tensor, r: usize) -> Result<Tensor> {
    let (b, l, f) = x.dims3()?;
    let r = r.min(l);
    x.narrow(1, l - r, r)?.reshape((b, r * f))
}

/// Soft-min squared distance of expert latents `mu [B, dim]` to centroids `c [M, dim]`:
/// `−τ · logsumexp(−d²/τ)`, mean over the batch (D-007 anchor).
pub fn anchor_loss(mu: &Tensor, c: &Tensor, tau: f64) -> Result<Tensor> {
    let d2 = mu.unsqueeze(1)?.broadcast_sub(&c.unsqueeze(0)?)?.sqr()?.sum(D::Minus1)?; // [B, M]
    let lse = d2.affine(-1.0 / tau, 0.0)?.log_sum_exp(D::Minus1)?; // [B]
    lse.affine(-tau, 0.0)?.mean_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backbones_produce_hidden_states() {
        let dev = Device::Cpu;
        for bb in [Backbone::Tcn, Backbone::Gru] {
            let cfg = NeuralConfig {
                backbone: bb,
                context_steps: 12,
                d_model: 8,
                layers: 2,
                ..NeuralConfig::new(NeuralFamily::JepaVae)
            };
            let m = NeuralModel::new(cfg.clone(), InputNorm::default(), 1, &dev).unwrap();
            let x = Tensor::ones((3, 12, N_FEATURES), candle_core::DType::F32, &dev).unwrap();
            let f = m.forward(&x, None).unwrap();
            assert_eq!(f.h.dims(), &[3, 8]);
            assert_eq!(f.out.mu.dims(), &[3, cfg.moe.max_experts, cfg.moe.expert_dim]);
            assert_eq!(m.ssl_head(&f).unwrap().dims(), &[3, 8]);
            assert_eq!(NeuralModel::mixture(&f).unwrap().dims(), &[3, N_TARGETS]);
        }
    }

    #[test]
    fn tcn_is_causal() {
        let dev = Device::Cpu;
        let cfg =
            NeuralConfig { context_steps: 10, d_model: 4, layers: 2, ..NeuralConfig::new(NeuralFamily::FsqHybrid) };
        let m = NeuralModel::new(cfg.clone(), InputNorm::default(), 2, &dev).unwrap();
        // the conv stack before pooling: changing the last step must not change earlier steps
        let a = Tensor::zeros((1, 10, N_FEATURES), candle_core::DType::F32, &dev).unwrap();
        let mut v = vec![0f32; 10 * N_FEATURES];
        v[9 * N_FEATURES] = 5.0;
        let b = Tensor::from_vec(v, (1, 10, N_FEATURES), &dev).unwrap();
        let conv = |x: &Tensor| {
            let mut u = lin(&m.ps, "bb.in", x).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
            for layer in 0..2 {
                let dil = 1usize << layer;
                let y = u
                    .pad_with_zeros(2, 2 * dil, 0)
                    .unwrap()
                    .conv1d(&m.ps.w(&format!("bb.conv{layer}.w")).unwrap(), 0, 1, dil, 1)
                    .unwrap();
                u = (u + y).unwrap();
            }
            u.narrow(2, 0, 9).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
        };
        assert_eq!(conv(&a), conv(&b));
    }

    #[test]
    fn anchor_pulls_toward_nearest_centroid() {
        let dev = Device::Cpu;
        let c = Tensor::from_vec(vec![0f32, 0.0, 1.0, 1.0], (2, 2), &dev).unwrap();
        let near = Tensor::from_vec(vec![0.9f32, 1.0], (1, 2), &dev).unwrap();
        let far = Tensor::from_vec(vec![0.5f32, 0.5], (1, 2), &dev).unwrap();
        let l = |m: &Tensor| fa_expert_space::nn::scalar(&anchor_loss(m, &c, 0.05).unwrap()).unwrap();
        assert!(l(&near) < l(&far));
        assert!(l(&near) < 0.02);
    }
}
