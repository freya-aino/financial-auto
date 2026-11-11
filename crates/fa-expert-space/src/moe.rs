//! Learned expert space (spec 32 §4.1, §6.1–§6.3; D-008): a gated mixture of `K_max` experts on
//! top of an encoder's pooled hidden state, trained with the D-008 objective
//!
//! `L_32 = Σ_k L_pred,k + λ_dec · Σ_{i≠j} corr(r_i, r_j)² + λ_bal · L_balance + β · KL`
//!
//! * **Router**: a linear gate → softmax over `K_max`; during training a seeded Gumbel-style
//!   noise is added to the logits (noisy gating). Live inference keeps the top-k gates
//!   renormalised (the others 0) as `ExpertView.router_weights`.
//! * **Projectors**: per expert a linear map to `(mu, logvar)` in `R^expert_dim`. `mu` is bounded
//!   by `tanh` so expert spaces share the scale of the statistical family's latents (regions
//!   radii stay comparable). With `fsq_levels` (spec 31 family 1) the first `len(levels)` dims of
//!   `mu` are finite-scalar quantised with a straight-through estimator and the code index is
//!   exported; the remaining dims stay a continuous Gaussian residual.
//! * **Predictors**: per expert a linear next-step head from its own latent → the timescale's
//!   next-step targets; `r_k = y − ŷ_k` are the residuals the decorrelation term acts on.
//!   Each expert predicting alone is also the gen-0 "every expert carries information" guard.
//! * **Load balancing**: Switch-style `K · Σ_i f_i · p_i` (`f` = fraction of windows whose
//!   top gate is `i`, held constant; `p` = mean gate probability).
//!
//! Training uses the gated decorrelation term ([`decorrelation_loss_gated`]): every window counts
//! with the detached weight `p_i · p_j` of the pair, per target dimension (or on the scalarised
//! residual), averaged over targets; [`decorrelation_loss`] is the ungated form over every window. [`crate::DecorrelationLoss`] keeps the
//! host-side (f64) form used for health reporting.

use candle_core::{D, DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use crate::nn::{Init, Linear, ParamStore, Weights, linear_w, softmax_last};

/// Hyper-parameters of a learned expert space (spec 32 §7; small CPU defaults).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeConfig {
    /// `K_max`; experts below `prune_usage_below` are pruned after training (emergent count).
    pub max_experts: usize,
    pub expert_dim: usize,
    pub top_k: usize,
    /// Finite-scalar-quantisation levels of the leading dims (empty = fully continuous).
    #[serde(default)]
    pub fsq_levels: Vec<u32>,
    pub lambda_dec: f64,
    pub lambda_bal: f64,
    pub beta_kl: f64,
    /// Std of the router noise during training (0 = off).
    pub router_noise: f64,
    pub prune_usage_below: f64,
}

impl Default for MoeConfig {
    fn default() -> Self {
        Self {
            max_experts: 4,
            expert_dim: 8,
            top_k: 2,
            fsq_levels: Vec::new(),
            lambda_dec: 1.0,
            lambda_bal: 0.01,
            beta_kl: 1e-3,
            router_noise: 0.3,
            prune_usage_below: 0.05,
        }
    }
}

/// Forward output over a batch of `B` windows.
#[derive(Debug, Clone)]
pub struct MoeOutput {
    /// Router probabilities `[B, K]`.
    pub probs: Tensor,
    /// Expert means `[B, K, dim]` (bounded, quantised dims included).
    pub mu: Tensor,
    /// Log-variances `[B, K, dim]`.
    pub logvar: Tensor,
    /// Pre-quantisation means `[B, K, dim]` (for the KL term).
    pub mu_raw: Tensor,
    /// Next-step predictions per expert `[B, K, T]`.
    pub pred: Tensor,
    /// FSQ code index per window and expert `[B, K]` (`None` without `fsq_levels`).
    pub codes: Option<Tensor>,
}

/// Router + projectors + predictors.
#[derive(Debug, Clone)]
pub struct MoeExpertSpace {
    pub cfg: MoeConfig,
    pub d_in: usize,
    pub n_targets: usize,
    /// Expert ids in output order (after pruning / reordering a subset of `0..max_experts`).
    pub active: Vec<usize>,
}

impl MoeExpertSpace {
    pub fn new(ps: &mut ParamStore, cfg: &MoeConfig, d_in: usize, n_targets: usize) -> Result<Self> {
        let k = cfg.max_experts.max(1);
        Linear::new(ps, "moe.router", d_in, k, true)?;
        for e in 0..k {
            Linear::new(ps, &format!("moe.proj{e}"), d_in, 2 * cfg.expert_dim, true)?;
            Linear::new(ps, &format!("moe.pred{e}"), cfg.expert_dim, n_targets, true)?;
        }
        // logvar bias starts at −2 (tight-ish posteriors), not 0.
        for e in 0..k {
            ps.get(&format!("moe.lvbias{e}"), &[cfg.expert_dim], Init::Const(-2.0))?;
        }
        Ok(Self { cfg: cfg.clone(), d_in, n_targets, active: (0..k).collect() })
    }

    pub fn n_active(&self) -> usize {
        self.active.len()
    }

    /// Restricts the space to `experts` (in this order): pruning and identity reordering.
    pub fn select(&mut self, experts: Vec<usize>) {
        self.active = experts;
    }

    /// Forward pass. `noise` = router noise `[B, K_max]` (training) or `None` (inference).
    pub fn forward(&self, ps: &ParamStore, h: &Tensor, noise: Option<&Tensor>) -> Result<MoeOutput> {
        self.forward_w(ps, h, noise)
    }

    /// Weight names of the expert projectors of the active experts (`moe.proj<e>.w`): the
    /// targets of expert-level low-rank adapters (spec 36 §6.5).
    pub fn projector_weights(&self) -> Vec<String> {
        self.active.iter().map(|e| format!("moe.proj{e}.w")).collect()
    }

    /// [`Self::forward`] reading every parameter through `w` (EMA copies, adapted views).
    pub fn forward_w(&self, ps: &dyn Weights, h: &Tensor, noise: Option<&Tensor>) -> Result<MoeOutput> {
        let logits_all = linear_w(ps, "moe.router", h)?; // [B, K_max]
        let logits_all = match noise {
            Some(n) => (logits_all + n)?,
            None => logits_all,
        };
        let idx =
            Tensor::from_vec(self.active.iter().map(|&i| i as u32).collect::<Vec<_>>(), self.active.len(), h.device())?;
        let logits = logits_all.index_select(&idx, 1)?;
        let probs = softmax_last(&logits)?;
        let mut mus = Vec::with_capacity(self.active.len());
        let mut raws = Vec::with_capacity(self.active.len());
        let mut lvs = Vec::with_capacity(self.active.len());
        let mut preds = Vec::with_capacity(self.active.len());
        let mut codes = Vec::new();
        let d = self.cfg.expert_dim;
        for &e in &self.active {
            let out = linear_w(ps, &format!("moe.proj{e}"), h)?; // [B, 2d]
            let raw = out.narrow(1, 0, d)?;
            let lvb = ps.w(&format!("moe.lvbias{e}"))?;
            let lv = out.narrow(1, d, d)?.broadcast_add(&lvb)?.clamp(-8.0f32, 4.0f32)?;
            let bounded = raw.tanh()?;
            let mu = if self.cfg.fsq_levels.is_empty() {
                bounded
            } else {
                let q = self.cfg.fsq_levels.len().min(d);
                let (zq, code) = fsq(&raw.narrow(1, 0, q)?, &self.cfg.fsq_levels[..q])?;
                codes.push(code);
                if q < d { Tensor::cat(&[zq, bounded.narrow(1, q, d - q)?], 1)? } else { zq }
            };
            preds.push(linear_w(ps, &format!("moe.pred{e}"), &mu)?);
            mus.push(mu);
            raws.push(raw);
            lvs.push(lv);
        }
        Ok(MoeOutput {
            probs,
            mu: Tensor::stack(&mus, 1)?,
            logvar: Tensor::stack(&lvs, 1)?,
            mu_raw: Tensor::stack(&raws, 1)?,
            pred: Tensor::stack(&preds, 1)?,
            codes: if codes.is_empty() { None } else { Some(Tensor::stack(&codes, 1)?) },
        })
    }
}

/// Finite scalar quantisation (Mentzer et al. 2023) of `z [B, q]` with `levels[q]`:
/// `bounded = tanh(z + shift)·half − offset`, rounded with a straight-through estimator and
/// rescaled to `[−1, 1]`. Returns the quantised values and the mixed-radix code index `[B]`.
pub fn fsq(z: &Tensor, levels: &[u32]) -> Result<(Tensor, Tensor)> {
    let dev = z.device();
    let q = levels.len();
    let eps = 1e-3;
    let half: Vec<f32> = levels.iter().map(|l| ((f64::from(*l) - 1.0) * (1.0 - eps) / 2.0) as f32).collect();
    let offset: Vec<f32> = levels.iter().map(|l| if l % 2 == 0 { 0.5 } else { 0.0 }).collect();
    let shift: Vec<f32> = offset.iter().zip(&half).map(|(o, h)| (o / h).atanh()).collect();
    let width: Vec<f32> = levels.iter().map(|l| (l / 2).max(1) as f32).collect();
    let t = |v: Vec<f32>| Tensor::from_vec(v, q, dev);
    let (half_t, off_t, shift_t, width_t) = (t(half)?, t(offset)?, t(shift)?, t(width.clone())?);
    let bounded = z.broadcast_add(&shift_t)?.tanh()?.broadcast_mul(&half_t)?.broadcast_sub(&off_t)?;
    let rounded = bounded.round()?;
    let ste = (&bounded + (rounded.clone() - &bounded)?.detach())?;
    let zq = ste.broadcast_div(&width_t)?;
    // code = Σ_i (rounded_i + width_i) · Π_{j<i} L_j
    let mut radix = Vec::with_capacity(q);
    let mut acc = 1f32;
    for l in levels {
        radix.push(acc);
        acc *= *l as f32;
    }
    let idx = rounded.detach().broadcast_add(&width_t)?.broadcast_mul(&t(radix)?)?.sum(D::Minus1)?;
    Ok((zq, idx))
}

/// D-008 decorrelation term on residuals `[B, K, T]`: `Σ_{i≠j} corr(r_i, r_j)²` per target,
/// averaged over targets. Differentiable.
pub fn decorrelation_loss(resid: &Tensor) -> Result<Tensor> {
    let (b, k, t) = resid.dims3()?;
    if k < 2 || b < 3 {
        return Tensor::zeros((), DType::F32, resid.device());
    }
    let x = resid.permute((2, 0, 1))?.contiguous()?; // [T, B, K]
    let xc = x.broadcast_sub(&x.mean_keepdim(1)?)?;
    let std = (xc.sqr()?.mean_keepdim(1)? + 1e-8)?.sqrt()?;
    let z = xc.broadcast_div(&std)?; // [T, B, K]
    let c = z.transpose(1, 2)?.contiguous()?.matmul(&z)?.affine(1.0 / b as f64, 0.0)?; // [T, K, K]
    let total = c.sqr()?.sum_all()?;
    let diag = z.sqr()?.mean(1)?.sqr()?.sum_all()?; // Σ diag(C)²
    (total - diag)?.affine(1.0 / t as f64, 0.0)
}

/// Gated D-008 term (spec 32 §6.3 "computed over gated windows"): for every expert pair the
/// residual correlation is weighted by `p_i · p_j` (router probabilities, detached so the router
/// cannot lower the penalty by separating experts alone), then `2 Σ_{i<j} corr_w²` per target,
/// averaged over targets. Experts only have to disagree where they are both routed, so a
/// specialist is not pushed to be wrong on windows it never serves.
pub fn decorrelation_loss_gated(resid: &Tensor, probs: &Tensor) -> Result<Tensor> {
    let (b, k, t) = resid.dims3()?;
    if k < 2 || b < 3 {
        return Tensor::zeros((), DType::F32, resid.device());
    }
    let p = probs.detach();
    let mut acc: Option<Tensor> = None;
    for i in 0..k {
        for j in (i + 1)..k {
            let w = (p.narrow(1, i, 1)? * p.narrow(1, j, 1)?)?; // [B, 1]
            let w = w.broadcast_div(&(w.sum_all()? + 1e-8)?)?;
            let ri = resid.narrow(1, i, 1)?.squeeze(1)?; // [B, T]
            let rj = resid.narrow(1, j, 1)?.squeeze(1)?;
            let mi = ri.broadcast_mul(&w)?.sum_keepdim(0)?;
            let mj = rj.broadcast_mul(&w)?.sum_keepdim(0)?;
            let ci = ri.broadcast_sub(&mi)?;
            let cj = rj.broadcast_sub(&mj)?;
            let cov = (&ci * &cj)?.broadcast_mul(&w)?.sum(0)?; // [T]
            let vi = ci.sqr()?.broadcast_mul(&w)?.sum(0)?;
            let vj = cj.sqr()?.broadcast_mul(&w)?.sum(0)?;
            let c2 = cov.sqr()?.div(&((vi * vj)? + 1e-8)?)?.sum_all()?;
            acc = Some(match acc {
                Some(a) => (a + c2)?,
                None => c2,
            });
        }
    }
    acc.expect("k ≥ 2").affine(2.0 / t as f64, 0.0)
}

/// Switch-style load balancing `K · Σ_i f_i p_i` on router probabilities `[B, K]` (1 = balanced).
pub fn load_balance_loss(probs: &Tensor) -> Result<Tensor> {
    let (b, k) = probs.dims2()?;
    let rows = probs.detach().to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let mut f = vec![0f32; k];
    for r in &rows {
        let arg = r.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |x| x.0);
        f[arg] += 1.0 / b as f32;
    }
    let f = Tensor::from_vec(f, k, probs.device())?;
    probs.mean(0)?.mul(&f)?.sum_all()?.affine(k as f64, 0.0)
}

/// Mean over the batch of the Gaussian KL to `N(0, I)`, summed over experts and dims.
pub fn gaussian_kl(mu: &Tensor, logvar: &Tensor) -> Result<Tensor> {
    let b = mu.dim(0)? as f64;
    let kl = ((mu.sqr()? + logvar.exp()?)? - logvar)?.affine(0.5, -0.5)?;
    kl.sum_all()?.affine(1.0 / b, 0.0)
}

/// Top-k gates of router probabilities, renormalised (others 0), host side.
pub fn top_k_gates(probs: &[f32], k: usize) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_by(|a, b| probs[*b].total_cmp(&probs[*a]).then(a.cmp(b)));
    let keep = &idx[..k.clamp(1, probs.len().max(1)).min(probs.len())];
    let s: f32 = keep.iter().map(|i| probs[*i]).sum();
    let mut out = vec![0.0; probs.len()];
    for i in keep {
        out[*i] = if s > 0.0 { probs[*i] / s } else { 1.0 / keep.len() as f32 };
    }
    out
}

/// Seeded router noise `[b, k]` with std `sigma`.
pub fn router_noise(b: usize, k: usize, sigma: f64, seed: u64, dev: &Device) -> Result<Tensor> {
    let mut rng = fa_core::rng::Rng::seed(seed);
    Tensor::from_vec((0..b * k).map(|_| (rng.normal() * sigma) as f32).collect::<Vec<_>>(), (b, k), dev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DecorrelationLoss,
        nn::{Adam, mse, scalar},
    };

    #[test]
    fn decorrelation_matches_host_penalty() {
        let dev = Device::Cpu;
        let r0 = [1.0f32, -1.0, 1.0, -1.0, 0.5, -0.3];
        let r1 = [1.0f32, 1.0, -1.0, -1.0, 0.2, 0.1];
        let r2 = [0.9f32, -1.1, 1.2, -0.8, 0.4, -0.2];
        let mut flat = Vec::new();
        for t in 0..6 {
            flat.extend([r0[t], r1[t], r2[t]]);
        }
        let x = Tensor::from_vec(flat, (6, 3, 1), &dev).unwrap();
        let got = scalar(&decorrelation_loss(&x).unwrap()).unwrap();
        let host = crate::ResidualCorrelationPenalty { lambda: 1.0 }.loss(&[
            r0.iter().map(|v| f64::from(*v)).collect(),
            r1.iter().map(|v| f64::from(*v)).collect(),
            r2.iter().map(|v| f64::from(*v)).collect(),
        ]);
        assert!((got - host).abs() < 1e-3, "{got} vs {host}");
    }

    #[test]
    fn gated_decorrelation_equals_plain_with_uniform_gates() {
        let dev = Device::Cpu;
        let mut rng = fa_core::rng::Rng::seed(9);
        let x = Tensor::from_vec((0..40 * 3 * 2).map(|_| rng.normal() as f32).collect::<Vec<_>>(), (40, 3, 2), &dev)
            .unwrap();
        let p = Tensor::ones((40, 3), DType::F32, &dev).unwrap().affine(1.0 / 3.0, 0.0).unwrap();
        let a = scalar(&decorrelation_loss(&x).unwrap()).unwrap();
        let b = scalar(&decorrelation_loss_gated(&x, &p).unwrap()).unwrap();
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }

    #[test]
    fn fsq_quantises_onto_levels_with_codes() {
        let dev = Device::Cpu;
        let z = Tensor::from_vec(vec![-5f32, 0.0, 5.0, 0.3, -0.2, 0.0], (2, 3), &dev).unwrap();
        let (zq, code) = fsq(&z, &[5, 5, 4]).unwrap();
        let v = zq.to_vec2::<f32>().unwrap();
        assert!((v[0][0] + 1.0).abs() < 1e-5 && v[0][1].abs() < 1e-5 && (v[0][2] - 0.5).abs() < 1e-5, "{v:?}");
        let c = code.to_vec1::<f32>().unwrap();
        assert!(c.iter().all(|x| *x >= 0.0 && *x < 100.0), "{c:?}");
        assert_ne!(c[0], c[1]);
    }

    #[test]
    fn load_balance_and_top_k() {
        let dev = Device::Cpu;
        let p = Tensor::from_vec(vec![0.6f32, 0.4, 0.4, 0.6], (2, 2), &dev).unwrap();
        assert!((scalar(&load_balance_loss(&p).unwrap()).unwrap() - 1.0).abs() < 1e-6);
        assert_eq!(top_k_gates(&[0.1, 0.6, 0.3], 2), vec![0.0, 0.6 / 0.9, 0.3 / 0.9]);
        assert_eq!(top_k_gates(&[0.5, 0.5], 5), vec![0.5, 0.5]);
    }

    /// Spec 32 §9.1 in miniature: two independent factors, each predictable from its own input
    /// half; two experts trained with the D-008 objective end with decorrelated residuals and
    /// both beat the naive baseline.
    #[test]
    fn two_experts_learn_decorrelated_specialisations() {
        let dev = Device::Cpu;
        let n = 256;
        let mut rng = fa_core::rng::Rng::seed(3);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for _ in 0..n {
            let (a, b) = (rng.normal() as f32, rng.normal() as f32);
            xs.extend([a, b, rng.normal() as f32 * 0.1, rng.normal() as f32 * 0.1]);
            ys.extend([0.8 * a + 0.1 * rng.normal() as f32, 0.8 * b + 0.1 * rng.normal() as f32]);
        }
        let x = Tensor::from_vec(xs, (n, 4), &dev).unwrap();
        let y = Tensor::from_vec(ys, (n, 2), &dev).unwrap();
        let cfg = MoeConfig { max_experts: 2, expert_dim: 2, lambda_dec: 0.5, router_noise: 0.0, ..Default::default() };
        let mut ps = ParamStore::new(5, &dev);
        let moe = MoeExpertSpace::new(&mut ps, &cfg, 4, 2).unwrap();
        let mut opt = Adam::new(0.03, 0.0);
        for _ in 0..300 {
            let o = moe.forward(&ps, &x, None).unwrap();
            let resid = o.pred.broadcast_sub(&y.unsqueeze(1).unwrap()).unwrap();
            let pred_loss = resid.sqr().unwrap().mean_all().unwrap();
            let loss = (pred_loss
                + decorrelation_loss(&resid).unwrap().affine(cfg.lambda_dec, 0.0).unwrap()
                + load_balance_loss(&o.probs).unwrap().affine(cfg.lambda_bal, 0.0).unwrap())
            .unwrap();
            opt.step(&ps, &loss).unwrap();
        }
        let o = moe.forward(&ps, &x, None).unwrap();
        let resid = o.pred.broadcast_sub(&y.unsqueeze(1).unwrap()).unwrap();
        let naive = scalar(&mse(&y, &y.zeros_like().unwrap()).unwrap()).unwrap();
        let fit = scalar(&resid.sqr().unwrap().mean_all().unwrap()).unwrap();
        assert!(fit < naive, "experts beat the naive baseline: {fit} vs {naive}");
        let corr = scalar(&decorrelation_loss(&resid).unwrap()).unwrap();
        assert!(corr < 0.2, "residual correlation driven down: {corr}");
    }
}
