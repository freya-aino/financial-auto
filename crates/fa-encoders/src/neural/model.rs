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
//! * `hier_vae_flow` (family 2): a top latent `z_top ~ q(z_top | h)` conditions the expert
//!   latents top-down (`h + gelu(W z_top)` feeds the router and projectors: a two-level
//!   hierarchy), reconstruction of the last bars from both levels, KL of both levels to
//!   `N(0, I)`, and a flow-matching prior on the top latent ([`super::prior`]) that gives its
//!   exact density for novelty;
//! * `jepa_vae` (family 3): Gaussian (VAE) expert heads; self-supervised JEPA-style by predicting
//!   the **EMA target encoder's** embedding of the window `jepa_horizon` steps ahead from the
//!   gated expert latents (no reconstruction of noisy inputs);
//! * `latent_diffusion` (family 4, day/week only per D-010): the autoencoder (reconstruction)
//!   plus a conditional diffusion model over the full expert latent that samples the latent
//!   `jepa_horizon` steps ahead and scores latents under the unconditional prior.
//!
//! Both add the spec 32 terms (per-expert next-step loss, D-008 residual decorrelation, router
//! load balancing, KL) and the mixture next-step loss that trains the router.

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
    /// Family 2: hierarchical VAE (top latent conditions the expert latents top-down) with a
    /// flow-matching prior on the top latent.
    HierVaeFlow,
    JepaVae,
    /// Family 4: autoencoder + conditional latent diffusion over the expert latents (day/week
    /// timescales only, D-010).
    LatentDiffusion,
}

impl NeuralFamily {
    /// `manifest.encoder_family` / Helix `Snapshot.encoder_family`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FsqHybrid => crate::FamilyId::FsqHybrid.as_str(),
            Self::HierVaeFlow => crate::FamilyId::HierVaeFlow.as_str(),
            Self::JepaVae => crate::FamilyId::JepaVae.as_str(),
            Self::LatentDiffusion => crate::FamilyId::LatentDiffusion.as_str(),
        }
    }

    /// Short prefix of snapshot ids (`fsq-v1-<ts>-g<gen>-<secs>`).
    pub fn id_prefix(self) -> &'static str {
        match self {
            Self::FsqHybrid => "fsq-v1",
            Self::HierVaeFlow => "hvf-v1",
            Self::JepaVae => "jepa-v1",
            Self::LatentDiffusion => "ldm-v1",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fsq_hybrid" | "fsq" => Some(Self::FsqHybrid),
            "hier_vae_flow" | "hvf" => Some(Self::HierVaeFlow),
            "jepa_vae" | "jepa" => Some(Self::JepaVae),
            "latent_diffusion" | "ldm" => Some(Self::LatentDiffusion),
            _ => None,
        }
    }

    /// Timescales the family may be trained / served at (D-010: latent diffusion only at days
    /// and weeks).
    pub fn allows(self, ts: fa_core::Timescale) -> bool {
        match self {
            Self::LatentDiffusion => matches!(ts, fa_core::Timescale::Day | fa_core::Timescale::Week),
            _ => true,
        }
    }

    pub const ALL: [NeuralFamily; 4] = [Self::FsqHybrid, Self::HierVaeFlow, Self::JepaVae, Self::LatentDiffusion];
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
    /// Steps ahead the `jepa_vae` target window ends (also the `latent_diffusion` future latent).
    pub jepa_horizon: usize,
    /// `hier_vae_flow`: dimension of the top latent.
    #[serde(default = "default_top_dim")]
    pub top_dim: usize,
    /// Hidden width of the flow-matching velocity field / the diffusion denoiser.
    #[serde(default = "default_prior_hidden")]
    pub prior_hidden: usize,
    /// Weight of the family's prior loss (flow matching, diffusion denoising).
    #[serde(default = "default_one")]
    pub lambda_prior: f64,
    /// `latent_diffusion`: diffusion steps `T` (cosine schedule).
    #[serde(default = "default_diffusion_steps")]
    pub diffusion_steps: usize,
    /// `latent_diffusion`: share of rows trained without the condition (classifier-free style),
    /// so the denoiser is both the unconditional prior and the future sampler.
    #[serde(default = "default_cond_drop")]
    pub cond_drop: f64,
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
    /// Residuals the D-008 term decorrelates: `false` = per target (vector-valued), `true` = the
    /// scalarised residual (mean over the normalised targets) that the spec 32 §6.4 health and
    /// the promotion checklist measure (spec 32 §6.3 lists both forms).
    #[serde(default)]
    pub decor_scalar: bool,
    /// Spec 32 §10.1 option (e): decorrelation as a constraint `L_decor ≤ decor_eps`, `λ_dec`
    /// following projected dual ascent at this rate per step (0 = fixed `moe.lambda_dec`).
    #[serde(default)]
    pub decor_dual_lr: f64,
    #[serde(default)]
    pub decor_eps: f64,
    /// Collapse guard (spec 32 §6.3 `min_skill`): weight of the hinge
    /// `Σ_k relu(mse_k / naive_k − (1 − skill_floor))` on each expert's routed (gate-weighted) error.
    #[serde(default)]
    pub lambda_skill: f64,
    #[serde(default = "default_skill_floor")]
    pub skill_floor: f64,
    /// Spec 32 §10.1 option (c), negative-correlation learning on the scalarised predictions:
    /// `− λ_ncl · mean_b Σ_k (ŝ_k − s̄)²` (ŝ = mean over targets, s̄ = the experts' mean). Rewards
    /// experts that disagree symmetrically around the ensemble; bounded for `λ_ncl < 1`.
    #[serde(default)]
    pub lambda_ncl: f64,
    /// Full-batch refit of the per-expert next-step heads on the frozen latents after training
    /// (`moe.pred<e>` only; same objective with the scalarised decorrelation term and the skill
    /// hinge, but exact correlations instead of minibatch estimates). 0 = off.
    #[serde(default)]
    pub head_refit_steps: usize,
    /// Head refit solver: `""` / `adam` (the penalised full-batch refit above) or `constrained`
    /// ([`super::heads`]: gated least squares with the pairwise residual covariance as an
    /// equality constraint, solved exactly per expert; it ignores `head_refit_steps`). Both
    /// `trainer sweep` and a snapshot fit run the constrained solve whenever it is selected, even
    /// with `head_refit_steps` 0; the penalised refit runs only when `head_refit_steps` > 0
    /// ([`NeuralConfig::refits_heads`]).
    #[serde(default)]
    pub head_solver: String,
    /// `constrained` solver: target |residual correlation| per expert pair (0 = exact).
    #[serde(default)]
    pub head_solver_corr: f64,
    /// `constrained` solver: block-coordinate sweeps (0 = the solver default).
    #[serde(default)]
    pub head_solver_sweeps: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub batch: usize,
    pub steps: usize,
    pub min_window: usize,
    pub val_fraction: f64,
}

fn default_top_dim() -> usize {
    4
}

fn default_prior_hidden() -> usize {
    64
}

fn default_one() -> f64 {
    1.0
}

fn default_diffusion_steps() -> usize {
    50
}

fn default_cond_drop() -> f64 {
    0.2
}

fn default_skill_floor() -> f64 {
    0.05
}

fn default_gated_pred() -> f64 {
    0.5
}

impl NeuralConfig {
    /// `head_solver` selects the exact constrained solver ([`super::heads`]).
    pub fn constrained_heads(&self) -> bool {
        self.head_solver == "constrained"
    }

    /// Whether a fit refits the next-step heads after training: the constrained solver always
    /// when selected, the penalised refit when `head_refit_steps` > 0.
    pub fn refits_heads(&self) -> bool {
        self.constrained_heads() || self.head_refit_steps > 0
    }

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
            top_dim: default_top_dim(),
            prior_hidden: default_prior_hidden(),
            lambda_prior: 1.0,
            diffusion_steps: default_diffusion_steps(),
            cond_drop: default_cond_drop(),
            ema: 0.99,
            lambda_next: 1.0,
            gated_pred: default_gated_pred(),
            lambda_mix: 1.0,
            lambda_ssl: 1.0,
            lambda_anchor: 0.0,
            anchor_tau: 0.1,
            lambda_feedback: 0.0,
            decor_scalar: false,
            decor_dual_lr: 0.0,
            decor_eps: 0.0,
            lambda_skill: 0.0,
            skill_floor: default_skill_floor(),
            lambda_ncl: 0.0,
            head_refit_steps: 0,
            head_solver: String::new(),
            head_solver_corr: 0.0,
            head_solver_sweeps: 0,
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
            NeuralFamily::FsqHybrid | NeuralFamily::HierVaeFlow => 1,
            NeuralFamily::JepaVae | NeuralFamily::LatentDiffusion => self.jepa_horizon.max(1),
        }
    }

    /// Whether the family reads the window `horizon()` steps ahead (JEPA target, diffusion future).
    pub fn needs_future(&self) -> bool {
        matches!(self.family, NeuralFamily::JepaVae | NeuralFamily::LatentDiffusion)
    }

    /// Whether the family's SSL term reconstructs the last bars.
    pub fn reconstructs(&self) -> bool {
        !matches!(self.family, NeuralFamily::JepaVae)
    }
}

/// Where a forward pass reads its parameters (the trainable store, an EMA copy, a frozen parent,
/// a low-rank adapted view).
pub use fa_expert_space::nn::Weights;

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
    /// `(series, end)` of every row (the window is `bars[..end]`), for feedback decoders that
    /// need the row's own context.
    pub windows: Vec<(usize, usize)>,
}

/// The top latent of `hier_vae_flow`.
#[derive(Debug, Clone)]
pub struct Top {
    pub mu: Tensor,
    pub logvar: Tensor,
    /// Sample (training) or mean (inference) fed top-down.
    pub z: Tensor,
}

/// Forward results the losses and the encoder read.
pub struct Forward {
    pub h: Tensor,
    pub out: fa_expert_space::moe::MoeOutput,
    /// `hier_vae_flow` only.
    pub top: Option<Top>,
}

impl NeuralModel {
    pub fn new(cfg: NeuralConfig, norm: InputNorm, seed: u64, dev: &Device) -> Result<Self> {
        let mut ps = ParamStore::new(seed, dev);
        init_backbone(&mut ps, &cfg)?;
        if cfg.family == NeuralFamily::HierVaeFlow {
            Linear::new(&mut ps, "hvf.top", cfg.d_model, 2 * cfg.top_dim, true)?;
            Linear::new(&mut ps, "hvf.td", cfg.top_dim, cfg.d_model, true)?;
            super::prior::init_flow(&mut ps, cfg.top_dim, cfg.prior_hidden)?;
        }
        let moe = MoeExpertSpace::new(&mut ps, &cfg.moe, cfg.d_model, N_TARGETS)?;
        let ssl_out = if cfg.reconstructs() { cfg.recon_steps * N_FEATURES } else { cfg.d_model };
        for e in 0..cfg.moe.max_experts {
            ps.get(
                &format!("ssl.h{e}"),
                &[cfg.moe.expert_dim, ssl_out],
                Init::Uniform { fan_in: cfg.moe.expert_dim, gain: 1.0 },
            )?;
        }
        ps.get("ssl.b", &[ssl_out], Init::Zeros)?;
        if cfg.family == NeuralFamily::HierVaeFlow {
            ps.get("ssl.top", &[cfg.top_dim, ssl_out], Init::Uniform { fan_in: cfg.top_dim, gain: 1.0 })?;
        }
        if cfg.family == NeuralFamily::LatentDiffusion {
            super::prior::init_diffusion(&mut ps, cfg.moe.max_experts * cfg.moe.expert_dim, cfg.prior_hidden)?;
        }
        Ok(Self { cfg, ps, moe, norm })
    }

    pub fn device(&self) -> &Device {
        self.ps.device()
    }

    pub fn forward(&self, x: &Tensor, noise: Option<&Tensor>) -> Result<Forward> {
        self.forward_w(&self.ps, x, noise, None)
    }

    /// Forward pass reading parameters through `w` (adapted views); `top_eps` = the
    /// reparameterisation noise of the `hier_vae_flow` top latent (`None` = its mean).
    pub fn forward_w(
        &self,
        w: &dyn Weights,
        x: &Tensor,
        noise: Option<&Tensor>,
        top_eps: Option<&Tensor>,
    ) -> Result<Forward> {
        let h = backbone_forward(w, &self.cfg, x)?;
        let (h_moe, top) = if self.cfg.family == NeuralFamily::HierVaeFlow {
            let td = self.cfg.top_dim;
            let o = lin(w, "hvf.top", &h)?;
            let mu = o.narrow(1, 0, td)?;
            let logvar = o.narrow(1, td, td)?.clamp(-8.0f32, 4.0f32)?;
            let z = match top_eps {
                Some(e) => (&mu + (logvar.affine(0.5, 0.0)?.exp()? * e)?)?,
                None => mu.clone(),
            };
            // top-down: the expert latents are conditioned on the top latent
            let h2 = (&h + lin(w, "hvf.td", &z)?.gelu()?)?;
            (h2, Some(Top { mu, logvar, z }))
        } else {
            (h.clone(), None)
        };
        let out = self.moe.forward_w(w, &h_moe, noise)?;
        Ok(Forward { h, out, top })
    }

    /// Gated self-supervised head: `Σ_k p_k · mu_k H_k (+ z_top H_top) + b`.
    pub fn ssl_head(&self, f: &Forward) -> Result<Tensor> {
        self.ssl_head_w(&self.ps, f)
    }

    pub fn ssl_head_w(&self, w: &dyn Weights, f: &Forward) -> Result<Tensor> {
        let mut acc: Option<Tensor> = None;
        for (pos, &e) in self.moe.active.iter().enumerate() {
            let mu = f.out.mu.narrow(1, pos, 1)?.squeeze(1)?;
            let p = f.out.probs.narrow(1, pos, 1)?;
            let y = mu.matmul(&w.w(&format!("ssl.h{e}"))?)?.broadcast_mul(&p)?;
            acc = Some(match acc {
                Some(a) => (a + y)?,
                None => y,
            });
        }
        let mut acc = acc.ok_or_else(|| candle_core::Error::Msg("no active experts".into()))?;
        if let Some(t) = &f.top {
            acc = (acc + t.z.matmul(&w.w("ssl.top")?)?)?;
        }
        acc.broadcast_add(&w.w("ssl.b")?)
    }

    /// Full-width latent `[B, K_max · expert_dim]`: each active expert's means at its expert-id
    /// slot, zeros for pruned experts (the `latent_diffusion` space, stable under pruning).
    pub fn full_latent(&self, mu: &Tensor) -> Result<Tensor> {
        let (b, _, d) = mu.dims3()?;
        let mut parts = Vec::with_capacity(self.cfg.moe.max_experts);
        for e in 0..self.cfg.moe.max_experts {
            match self.moe.active.iter().position(|a| *a == e) {
                Some(pos) => parts.push(mu.narrow(1, pos, 1)?.squeeze(1)?),
                None => parts.push(Tensor::zeros((b, d), mu.dtype(), mu.device())?),
            }
        }
        Tensor::cat(&parts, 1)
    }

    /// Inverse of [`Self::full_latent`]: `[B, K_max · d]` → `[B, K_active, d]`.
    pub fn split_full(&self, x: &Tensor) -> Result<Tensor> {
        let d = self.cfg.moe.expert_dim;
        let parts: Vec<Tensor> = self.moe.active.iter().map(|&e| x.narrow(1, e * d, d)).collect::<Result<_>>()?;
        Tensor::stack(&parts, 1)
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
