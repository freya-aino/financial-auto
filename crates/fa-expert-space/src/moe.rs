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
    /// Load balancing (spec 32 §6.1): `aux` = the Switch auxiliary loss weighted by
    /// `lambda_bal` (default); `bias` = auxiliary-loss-free balancing: a per-expert routing bias
    /// `moe.route_bias` (not trained by gradient) is added to the router logits and nudged after
    /// every step by `bias_rate · sign(mean load − load_k)` ([`MoeExpertSpace::update_route_bias`]),
    /// so balancing does not fight the decorrelation loss. With `bias` the auxiliary loss is off.
    #[serde(default)]
    pub balance: String,
    /// Step of the bias update (`balance = bias`).
    #[serde(default = "default_bias_rate")]
    pub bias_rate: f64,
    /// EMA decay of the residual moments the decorrelation term reads (spec 32 §6.3 / §7
    /// `cov_ema`): the penalty's covariances mix the detached running moments with the current
    /// minibatch's (`α·ema + (1 − α)·batch`), so it estimates a correlation over many batches
    /// rather than one ([`decorrelation_loss_gated_ema`]). 0 = minibatch only.
    #[serde(default)]
    pub cov_ema: f64,
}

fn default_bias_rate() -> f64 {
    1e-3
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
            balance: String::new(),
            bias_rate: default_bias_rate(),
            cov_ema: 0.0,
        }
    }
}

impl MoeConfig {
    /// Auxiliary-loss-free balancing by routing bias (`balance = "bias"`).
    pub fn bias_balancing(&self) -> bool {
        self.balance == "bias"
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
        if cfg.bias_balancing() {
            ps.get(ROUTE_BIAS, &[k], Init::Zeros)?;
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
        let mut logits_all = linear_w(ps, "moe.router", h)?; // [B, K_max]
        if self.cfg.bias_balancing() {
            // balancing bias: steers routing, never trained by gradient
            logits_all = logits_all.broadcast_add(&ps.w(ROUTE_BIAS)?.detach())?;
        }
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

/// Routing-bias parameter of auxiliary-loss-free balancing (`[K_max]`).
pub const ROUTE_BIAS: &str = "moe.route_bias";

/// Share of rows whose top-k contains each expert (`probs [B, K]`, host side): the load the
/// bias update balances.
pub fn top_k_load(probs: &Tensor, top_k: usize) -> Result<Vec<f64>> {
    let rows = probs.detach().to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let k = rows.first().map_or(0, Vec::len);
    let mut load = vec![0.0; k];
    for r in &rows {
        for (e, g) in top_k_gates(r, top_k).iter().enumerate() {
            if *g > 0.0 {
                load[e] += 1.0 / rows.len() as f64;
            }
        }
    }
    Ok(load)
}

impl MoeExpertSpace {
    /// Auxiliary-loss-free balancing step (spec 32 §6.1): every active expert's routing bias
    /// moves by `bias_rate` toward the mean top-k load (`+` when under-loaded, `−` when
    /// over-loaded, unchanged at the mean). No-op unless `balance = bias`.
    pub fn update_route_bias(&self, ps: &mut ParamStore, probs: &Tensor) -> Result<()> {
        if !self.cfg.bias_balancing() {
            return Ok(());
        }
        let load = top_k_load(probs, self.cfg.top_k)?;
        if load.is_empty() {
            return Ok(());
        }
        let mean = load.iter().sum::<f64>() / load.len() as f64;
        let mut b = ps.w(ROUTE_BIAS)?.to_vec1::<f32>()?;
        for (pos, &e) in self.active.iter().enumerate() {
            let d = mean - load[pos];
            if d.abs() > 1e-12 {
                b[e] += (self.cfg.bias_rate * d.signum()) as f32;
            }
        }
        let n = b.len();
        let t = Tensor::from_vec(b, n, ps.device())?;
        ps.load(&std::collections::HashMap::from([(ROUTE_BIAS.to_owned(), t)]))?;
        Ok(())
    }

    /// Expert **split** (spec 32 §6.1): expert `to` (an unused slot) becomes a perturbed clone of
    /// `from` — projector, log-variance bias and next-step head copied with seeded noise of
    /// relative size `scale`, router column copied and both router biases lowered by `ln 2`, so
    /// the two share `from`'s routing mass and diverge from there. `extra` names further
    /// per-expert parameters to clone (`<prefix>{e}` with the expert id appended, e.g. a family
    /// head). Activates `to` after `from`.
    pub fn split_expert(
        &mut self,
        ps: &mut ParamStore,
        from: usize,
        to: usize,
        extra: &[&str],
        scale: f64,
        seed: u64,
    ) -> Result<()> {
        if from == to || to >= self.cfg.max_experts || self.active.contains(&to) || !self.active.contains(&from) {
            candle_core::bail!("split {from} → {to}: `to` must be a free slot and `from` active");
        }
        let mut rng = fa_core::rng::Rng::seed(seed);
        let mut vals = std::collections::HashMap::new();
        let mut names: Vec<(String, String)> = Vec::new();
        for pre in ["moe.proj", "moe.pred"] {
            for suf in ["w", "b"] {
                names.push((format!("{pre}{from}.{suf}"), format!("{pre}{to}.{suf}")));
            }
        }
        names.push((format!("moe.lvbias{from}"), format!("moe.lvbias{to}")));
        for pre in extra {
            names.push((format!("{pre}{from}"), format!("{pre}{to}")));
        }
        for (src, dst) in names {
            let t = ps.w(&src)?;
            let v = t.flatten_all()?.to_vec1::<f32>()?;
            let rms = (v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt().max(1e-3);
            let noisy: Vec<f32> = v.iter().map(|x| x + (rng.normal() * scale * rms) as f32).collect();
            vals.insert(dst, Tensor::from_vec(noisy, t.dims(), ps.device())?);
        }
        let w = ps.w("moe.router.w")?.to_vec2::<f32>()?; // [d_in, K]
        let mut w2 = w.clone();
        for row in w2.iter_mut() {
            row[to] = row[from] + (rng.normal() * scale * 1e-2) as f32;
        }
        let mut b = ps.w("moe.router.b")?.to_vec1::<f32>()?;
        let ln2 = std::f32::consts::LN_2;
        b[to] = b[from] - ln2;
        b[from] -= ln2;
        let (din, k) = (w2.len(), b.len());
        vals.insert("moe.router.w".into(), Tensor::from_vec(w2.concat(), (din, k), ps.device())?);
        vals.insert("moe.router.b".into(), Tensor::from_vec(b, k, ps.device())?);
        if self.cfg.bias_balancing() {
            let mut rb = ps.w(ROUTE_BIAS)?.to_vec1::<f32>()?;
            rb[to] = rb[from];
            vals.insert(ROUTE_BIAS.into(), Tensor::from_vec(rb, k, ps.device())?);
        }
        ps.load(&vals)?;
        let at = self.active.iter().position(|e| *e == from).map_or(self.active.len(), |p| p + 1);
        self.active.insert(at, to);
        Ok(())
    }
}

/// Running (detached) residual moments of the gated decorrelation term (`cov_ema`):
/// `[pairs, 6, T]` = weight mass, `E_w[r_i]`, `E_w[r_j]`, `E_w[r_i r_j]`, `E_w[r_i²]`, `E_w[r_j²]`
/// per expert pair `i < j` (row-major) and target.
#[derive(Debug, Clone)]
pub struct DecorEma(pub Tensor);

/// [`decorrelation_loss_gated`] over moments mixed with running ones: for every pair the
/// gate-weighted first and second moments of the minibatch are blended as
/// `α·ema + (1 − α)·batch` with the detached running moments `ema` (none on the first step),
/// and the correlation is taken from the blended moments. The blend is straight-through: its
/// value is the running estimate, its gradient that of the minibatch moments at full weight, so
/// `cov_ema` lowers the variance of the penalty without shrinking its strength. Returns the loss and the updated
/// running moments (detached). `α = 0` equals [`decorrelation_loss_gated`] up to the
/// centring convention (moments instead of centred sums; identical in exact arithmetic).
pub fn decorrelation_loss_gated_ema(
    resid: &Tensor,
    probs: &Tensor,
    ema: Option<&DecorEma>,
    alpha: f64,
) -> Result<(Tensor, Option<DecorEma>)> {
    let (b, k, t) = resid.dims3()?;
    if k < 2 || b < 3 {
        return Ok((Tensor::zeros((), DType::F32, resid.device())?, ema.cloned()));
    }
    let p = probs.detach();
    let mut moments = Vec::new();
    let mut acc: Option<Tensor> = None;
    let mut pair = 0usize;
    for i in 0..k {
        for j in (i + 1)..k {
            let w = (p.narrow(1, i, 1)? * p.narrow(1, j, 1)?)?; // [B, 1]
            let w = w.broadcast_div(&(w.sum_all()? + 1e-8)?)?;
            let ri = resid.narrow(1, i, 1)?.squeeze(1)?; // [B, T]
            let rj = resid.narrow(1, j, 1)?.squeeze(1)?;
            let m = |x: &Tensor| x.broadcast_mul(&w)?.sum(0); // [T]
            let batch = [m(&ri)?, m(&rj)?, m(&(&ri * &rj)?)?, m(&ri.sqr()?)?, m(&rj.sqr()?)?];
            let mixed: Vec<Tensor> = match ema {
                Some(e) if alpha > 0.0 => batch
                    .iter()
                    .enumerate()
                    .map(|(q, bm)| {
                        // value α·ema + (1 − α)·batch, gradient of the batch moment at full
                        // weight (a straight-through running estimate: blending alone would scale
                        // the penalty's gradient down by 1 − α)
                        let old = e.0.narrow(0, pair, 1)?.squeeze(0)?.narrow(0, q + 1, 1)?.squeeze(0)?;
                        bm + (old - bm)?.affine(alpha, 0.0)?.detach()
                    })
                    .collect::<Result<_>>()?,
                _ => batch.to_vec(),
            };
            let cov = (&mixed[2] - (&mixed[0] * &mixed[1])?)?;
            let vi = (&mixed[3] - mixed[0].sqr()?)?.relu()?;
            let vj = (&mixed[4] - mixed[1].sqr()?)?.relu()?;
            let c2 = cov.sqr()?.div(&((vi * vj)? + 1e-8)?)?.sum_all()?;
            acc = Some(match acc {
                Some(a) => (a + c2)?,
                None => c2,
            });
            let mass = Tensor::ones(t, DType::F32, resid.device())?;
            let mut rows = vec![mass];
            rows.extend(mixed.iter().map(Tensor::detach));
            moments.push(Tensor::stack(&rows, 0)?); // [6, T]
            pair += 1;
        }
    }
    let state = (alpha > 0.0).then(|| Tensor::stack(&moments, 0).map(DecorEma)).transpose()?;
    Ok((acc.expect("k ≥ 2").affine(2.0 / t as f64, 0.0)?, state))
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
    fn ema_decorrelation_matches_gated_without_history_and_smooths_with_it() {
        let dev = Device::Cpu;
        let mut rng = fa_core::rng::Rng::seed(4);
        let mut draw = |n| {
            Tensor::from_vec((0..n * 3 * 2).map(|_| rng.normal() as f32).collect::<Vec<_>>(), (n, 3, 2), &dev).unwrap()
        };
        let (x, y) = (draw(50), draw(50));
        let p = Tensor::ones((50, 3), DType::F32, &dev).unwrap().affine(1.0 / 3.0, 0.0).unwrap();
        let plain = scalar(&decorrelation_loss_gated(&x, &p).unwrap()).unwrap();
        let (l0, st) = decorrelation_loss_gated_ema(&x, &p, None, 0.9).unwrap();
        assert!((scalar(&l0).unwrap() - plain).abs() < 1e-3, "first batch: plain moments");
        let st = st.expect("running moments");
        assert_eq!(st.0.dims(), &[3, 6, 2]);
        // with α → 1 the second batch reads (almost only) the running moments of the first
        let (l1, _) = decorrelation_loss_gated_ema(&y, &p, Some(&st), 0.999).unwrap();
        let fresh = scalar(&decorrelation_loss_gated(&y, &p).unwrap()).unwrap();
        let l1 = scalar(&l1).unwrap();
        assert!((l1 - plain).abs() < 2e-3 && (fresh - plain).abs() > 2e-3, "{l1} ≈ {plain}, not {fresh}");
        // straight-through: the gradient keeps the minibatch's full weight (not scaled by 1 − α)
        let yv = candle_core::Var::from_tensor(&y).unwrap();
        let grad_norm = |l: Tensor| {
            let g = l.backward().unwrap();
            scalar(&g.get(yv.as_tensor()).unwrap().sqr().unwrap().sum_all().unwrap().sqrt().unwrap()).unwrap()
        };
        let g_ema = grad_norm(decorrelation_loss_gated_ema(yv.as_tensor(), &p, Some(&st), 0.999).unwrap().0);
        let g_plain = grad_norm(decorrelation_loss_gated(yv.as_tensor(), &p).unwrap());
        assert!(g_ema > 0.1 * g_plain, "gradient {g_ema} vs plain {g_plain}");
        // α = 0 ignores history
        let (l2, none) = decorrelation_loss_gated_ema(&y, &p, Some(&st), 0.0).unwrap();
        assert!((scalar(&l2).unwrap() - fresh).abs() < 1e-3 && none.is_none());
    }

    #[test]
    fn bias_balancing_moves_load_toward_the_mean_and_split_clones_an_expert() {
        let dev = Device::Cpu;
        let cfg = MoeConfig {
            max_experts: 3,
            expert_dim: 2,
            top_k: 1,
            balance: "bias".into(),
            bias_rate: 0.05,
            router_noise: 0.0,
            ..Default::default()
        };
        let mut ps = ParamStore::new(1, &dev);
        let mut moe = MoeExpertSpace::new(&mut ps, &cfg, 4, 2).unwrap();
        // a router that sends everything to expert 0
        let mut b = vec![0f32; 3];
        b[0] = 2.0;
        ps.load(&std::collections::HashMap::from([("moe.router.b".to_owned(), Tensor::from_vec(b, 3, &dev).unwrap())]))
            .unwrap();
        let x = Tensor::from_vec((0..64 * 4).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect::<Vec<_>>(), (64, 4), &dev)
            .unwrap();
        let load0 = top_k_load(&moe.forward(&ps, &x, None).unwrap().probs, 1).unwrap();
        assert!(load0[0] > 0.9, "{load0:?}");
        for _ in 0..80 {
            let o = moe.forward(&ps, &x, None).unwrap();
            moe.update_route_bias(&mut ps, &o.probs).unwrap();
        }
        let load1 = top_k_load(&moe.forward(&ps, &x, None).unwrap().probs, 1).unwrap();
        assert!(load1[0] < 0.7, "bias spreads the load: {load1:?}");
        // the bias is not a gradient parameter: a loss on the probabilities leaves it alone
        let o = moe.forward(&ps, &x, None).unwrap();
        let grads = o.probs.sum_all().unwrap().backward().unwrap();
        assert!(grads.get(ps.vars().find(|(k, _)| k.as_str() == ROUTE_BIAS).unwrap().1.as_tensor()).is_none());
        // split: expert 2 pruned, then re-born as a perturbed clone of expert 0
        moe.select(vec![0, 1]);
        moe.split_expert(&mut ps, 0, 2, &[], 0.1, 7).unwrap();
        assert_eq!(moe.active, vec![0, 2, 1]);
        let o = moe.forward(&ps, &x, None).unwrap();
        let p = o.probs.to_vec2::<f32>().unwrap();
        assert!((p[0][0] - p[0][1]).abs() < 0.1, "clone shares the parent's routing mass: {:?}", p[0]);
        assert!(moe.split_expert(&mut ps, 0, 1, &[], 0.1, 7).is_err(), "occupied slot");
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
