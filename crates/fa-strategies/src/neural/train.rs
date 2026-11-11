//! SGD fine-tuning of a [`PolicyNet`] on candle (CPU), the differentiable half of the hybrid
//! ES + SGD training of spec 35 §4.3 (decision on §10 Q2).
//!
//! **Surrogate objective** (maximised): the cost-adjusted log utility of the policy's exposures
//! on recorded observations,
//!
//! `U = mean_i [ ln(1 + L·mᵢ·rᵢ) − c·|mᵢ − m_prev(i)| ] − w_s · mean_i (sᵢ − ŝᵢ)²`
//!
//! with `mᵢ = tanh(o₀)` the exposure for sample `i` (an asset at a step), `rᵢ` the asset's
//! realised log return over the strategy horizon after the step, `m_prev(i)` the exposure of the
//! same asset at its previous sample (0 for the first; the turnover cost `c` is the round-trip
//! cost per unit of sleeve), `L` a leverage scale so `L·r` is O(0.1), `sᵢ = softplus(o₁)` and
//! `ŝᵢ = min(1, |rᵢ| / (2σ_r))` the dispersion target of the std head. The inputs are those the
//! policy saw in a replay of the training window (its own book included), so the surrogate is
//! exact for the first step and an approximation thereafter (the book would move with the
//! weights). Samples are ordered by time; the last `val_frac` are held out and the weights with
//! the best held-out utility are kept, so a fine-tune never returns a net that is worse on the
//! held-out tail than the one it started from.

use std::{collections::BTreeMap, time::Instant};

use candle_core::{D, DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use fa_core::InstrumentId;
use serde::{Deserialize, Serialize};

use super::{Act, PolicyNet};

/// One training sample: the policy input of an asset at a step and its realised forward return.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainSample {
    pub t_ms: i64,
    pub instrument: InstrumentId,
    pub x: Vec<f32>,
    /// Log return over the strategy horizon after `t_ms`.
    pub fwd_ret: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainConfig {
    pub steps: usize,
    pub lr: f64,
    pub weight_decay: f64,
    /// `L` in the utility; `None` = `0.1 / mean |r|`.
    pub leverage: Option<f64>,
    /// Turnover cost per unit exposure change (fraction of the sleeve, e.g. 0.001 = 10 bp).
    pub turnover_cost: f64,
    /// Weight of the std head's dispersion loss.
    pub std_weight: f64,
    /// Share of the (time-ordered) samples held out for weight selection.
    pub val_frac: f64,
    /// Cap on samples (evenly strided, the newest kept).
    pub max_samples: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 150,
            lr: 5e-3,
            weight_decay: 1e-4,
            leverage: None,
            turnover_cost: 0.001,
            std_weight: 0.1,
            val_frac: 0.25,
            max_samples: 20_000,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TrainReport {
    pub train_rows: usize,
    pub val_rows: usize,
    pub steps: usize,
    /// Step whose weights were kept (0 = the initial weights).
    pub best_step: usize,
    pub leverage: f64,
    /// Utility (log-utility minus turnover, no std term) per sample.
    pub initial_train_utility: f64,
    pub final_train_utility: f64,
    pub initial_val_utility: f64,
    pub best_val_utility: f64,
    pub elapsed_ms: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("{0}")]
    Data(String),
}

/// Tensors of one split.
struct Split {
    x: Tensor,
    r: Tensor,
    /// Index of the same asset's previous sample within the split, or `n` (a zero pad).
    prev: Tensor,
    s_target: Tensor,
    n: usize,
}

fn split(rows: &[&TrainSample], dim: usize, leverage: f64, sigma: f64, dev: &Device) -> Result<Split, TrainError> {
    let n = rows.len();
    let mut xs = Vec::with_capacity(n * dim);
    let mut rs = Vec::with_capacity(n);
    let mut prev = Vec::with_capacity(n);
    let mut st = Vec::with_capacity(n);
    let mut last: BTreeMap<InstrumentId, u32> = BTreeMap::new();
    let cap = (0.5 / leverage) as f32;
    for (i, s) in rows.iter().enumerate() {
        xs.extend(s.x.iter().map(|v| if v.is_finite() { *v } else { 0.0 }));
        let r = if s.fwd_ret.is_finite() { s.fwd_ret.clamp(-cap, cap) } else { 0.0 };
        rs.push(r);
        st.push((f64::from(r.abs()) / (2.0 * sigma).max(1e-12)).min(1.0) as f32);
        prev.push(last.insert(s.instrument, i as u32).unwrap_or(n as u32));
    }
    Ok(Split {
        x: Tensor::from_vec(xs, (n, dim), dev)?,
        r: Tensor::from_vec(rs, n, dev)?,
        prev: Tensor::from_vec(prev, n, dev)?,
        s_target: Tensor::from_vec(st, n, dev)?,
        n,
    })
}

/// The network as candle variables.
struct CandleNet {
    layers: Vec<(Var, Var)>,
    act: Act,
}

impl CandleNet {
    fn new(net: &PolicyNet, dev: &Device) -> Result<Self, TrainError> {
        let mut layers = Vec::new();
        for (w, b) in net.unpack() {
            let (o, i) = (w.len(), w.first().map_or(0, Vec::len));
            let wt = Tensor::from_vec(w.into_iter().flatten().collect::<Vec<f32>>(), (o, i), dev)?;
            let bt = Tensor::from_vec(b, o, dev)?;
            layers.push((Var::from_tensor(&wt)?, Var::from_tensor(&bt)?));
        }
        Ok(Self { layers, act: net.act })
    }

    fn vars(&self) -> Vec<Var> {
        self.layers.iter().flat_map(|(w, b)| [w.clone(), b.clone()]).collect()
    }

    /// `[n, 2]` raw outputs.
    fn forward(&self, x: &Tensor) -> Result<Tensor, TrainError> {
        let mut h = x.clone();
        let last = self.layers.len() - 1;
        for (li, (w, b)) in self.layers.iter().enumerate() {
            h = h.matmul(&w.as_tensor().t()?)?.broadcast_add(b.as_tensor())?;
            if li < last {
                h = match self.act {
                    Act::Tanh => h.tanh()?,
                    Act::Relu => h.relu()?,
                };
            }
        }
        Ok(h)
    }

    fn weights(&self) -> Result<Vec<f32>, TrainError> {
        let mut out = Vec::new();
        for (w, b) in &self.layers {
            out.extend(w.as_tensor().flatten_all()?.to_vec1::<f32>()?);
            out.extend(b.as_tensor().to_vec1::<f32>()?);
        }
        Ok(out)
    }
}

/// `(utility without the std term, total loss)` of a split.
fn objective(net: &CandleNet, d: &Split, leverage: f64, cfg: &TrainConfig) -> Result<(Tensor, Tensor), TrainError> {
    let out = net.forward(&d.x)?;
    let m = out.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?.tanh()?;
    let o1 = out.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;
    // softplus(x) = relu(x) + ln(1 + e^{-|x|})
    let s = (o1.relu()? + (o1.abs()?.neg()?.exp()? + 1.0)?.log()?)?;
    let gain = ((m.mul(&d.r)? * leverage)? + 1.0)?.log()?;
    let m_ext = Tensor::cat(&[&m, &Tensor::zeros(1, DType::F32, m.device())?], 0)?;
    let m_prev = m_ext.index_select(&d.prev, 0)?;
    let turnover = (m.sub(&m_prev)?.abs()? * cfg.turnover_cost)?;
    let utility = gain.sub(&turnover)?.mean_all()?;
    let std_loss = s.sub(&d.s_target)?.sqr()?.mean_all()?;
    let loss = (utility.neg()? + (std_loss * cfg.std_weight)?)?;
    Ok((utility, loss))
}

/// Fine-tunes `net` on `samples` (module docs). Returns the best held-out weights (the input
/// weights when nothing improved) and the report.
pub fn fine_tune(net: &PolicyNet, samples: &[TrainSample], cfg: &TrainConfig) -> Result<(PolicyNet, TrainReport), TrainError> {
    let started = Instant::now();
    net.validate().map_err(|e| TrainError::Data(e.to_string()))?;
    let dim = net.input_dim();
    let mut rows: Vec<&TrainSample> = samples.iter().filter(|s| s.x.len() == dim && s.fwd_ret.is_finite()).collect();
    rows.sort_by_key(|s| s.t_ms);
    if rows.len() > cfg.max_samples.max(1) {
        let stride = rows.len().div_ceil(cfg.max_samples.max(1));
        let keep: Vec<&TrainSample> = rows.iter().rev().step_by(stride).rev().copied().collect();
        rows = keep;
    }
    if rows.len() < 8 {
        return Err(TrainError::Data(format!("{} usable samples (need ≥ 8)", rows.len())));
    }
    let n_val = ((rows.len() as f64) * cfg.val_frac.clamp(0.0, 0.9)).round() as usize;
    let n_val = n_val.clamp(1, rows.len() - 4);
    let (tr, va) = rows.split_at(rows.len() - n_val);
    let mean_abs = tr.iter().map(|s| f64::from(s.fwd_ret.abs())).sum::<f64>() / tr.len() as f64;
    let sigma = (tr.iter().map(|s| f64::from(s.fwd_ret).powi(2)).sum::<f64>() / tr.len() as f64).sqrt();
    let leverage = cfg.leverage.unwrap_or(if mean_abs > 0.0 { 0.1 / mean_abs } else { 1.0 }).clamp(1e-3, 1e6);
    let dev = Device::Cpu;
    let train = split(tr, dim, leverage, sigma, &dev)?;
    let val = split(va, dim, leverage, sigma, &dev)?;
    let cn = CandleNet::new(net, &dev)?;
    let mut opt = AdamW::new(
        cn.vars(),
        ParamsAdamW { lr: cfg.lr, weight_decay: cfg.weight_decay, ..ParamsAdamW::default() },
    )?;
    let scalar = |t: &Tensor| -> Result<f64, TrainError> { Ok(f64::from(t.to_scalar::<f32>()?)) };
    let (u0, _) = objective(&cn, &train, leverage, cfg)?;
    let (v0, _) = objective(&cn, &val, leverage, cfg)?;
    let mut report = TrainReport {
        train_rows: train.n,
        val_rows: val.n,
        leverage,
        initial_train_utility: scalar(&u0)?,
        initial_val_utility: scalar(&v0)?,
        ..TrainReport::default()
    };
    report.best_val_utility = report.initial_val_utility;
    let mut best = net.weights.clone();
    for step in 1..=cfg.steps {
        let (_, loss) = objective(&cn, &train, leverage, cfg)?;
        opt.backward_step(&loss)?;
        report.steps = step;
        if step % 5 == 0 || step == cfg.steps {
            let (v, _) = objective(&cn, &val, leverage, cfg)?;
            let v = scalar(&v)?;
            if v.is_finite() && v > report.best_val_utility {
                report.best_val_utility = v;
                report.best_step = step;
                best = cn.weights()?;
            }
        }
    }
    let mut out = net.clone();
    out.weights = best;
    if out.validate().is_err() {
        out = net.clone();
        report.best_step = 0;
        report.best_val_utility = report.initial_val_utility;
    }
    let tuned = CandleNet::new(&out, &dev)?;
    report.final_train_utility = scalar(&objective(&tuned, &train, leverage, cfg)?.0)?;
    report.elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
    Ok((out, report))
}

/// Raw outputs of the candle forward pass (test hook pinning it to [`PolicyNet::forward_raw`]).
pub fn candle_forward(net: &PolicyNet, x: &[Vec<f32>]) -> Result<Vec<(f32, f32)>, TrainError> {
    let dev = Device::Cpu;
    let cn = CandleNet::new(net, &dev)?;
    let flat: Vec<f32> = x.iter().flatten().copied().collect();
    let t = Tensor::from_vec(flat, (x.len(), net.input_dim()), &dev)?;
    let out = cn.forward(&t)?.to_vec2::<f32>()?;
    Ok(out.into_iter().map(|r| (r[0], r[1])).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::ALL_FEATURES;
    use fa_core::rng::Rng;

    #[test]
    fn candle_forward_matches_the_plain_forward() {
        for (seed, act, hidden) in [(1, Act::Tanh, vec![8]), (2, Act::Relu, vec![6, 5]), (3, Act::Tanh, vec![4, 4, 3])] {
            let net = PolicyNet::random(&mut Rng::seed(seed), ALL_FEATURES, 0b11, 3, hidden, act).perturb(&mut Rng::seed(9), 1.0);
            let mut rng = Rng::seed(seed + 10);
            let xs: Vec<Vec<f32>> =
                (0..16).map(|_| (0..net.input_dim()).map(|_| rng.normal() as f32).collect()).collect();
            let c = candle_forward(&net, &xs).unwrap();
            for (x, (a, b)) in xs.iter().zip(c) {
                let (p, q) = net.forward_raw(x);
                assert!((a - p).abs() < 1e-4 && (b - q).abs() < 1e-4, "{a} {b} vs {p} {q}");
            }
        }
    }

    #[test]
    fn fine_tune_learns_a_predictive_feature_and_keeps_held_out_gains() {
        // feature 0 predicts the forward return's sign; feature 1 is noise
        let mut rng = Rng::seed(5);
        let insts = [InstrumentId::new(), InstrumentId::new()];
        let mut samples = Vec::new();
        for t in 0..1500i64 {
            for inst in insts {
                let sig = rng.normal() as f32;
                let noise = rng.normal() as f32;
                let r = 0.002 * sig + 0.002 * rng.normal() as f32;
                samples.push(TrainSample { t_ms: t * 1000, instrument: inst, x: vec![sig, noise], fwd_ret: r });
            }
        }
        let net = PolicyNet::random(&mut Rng::seed(1), 0b11, 0, 0, vec![8], Act::Tanh);
        let cfg = TrainConfig { steps: 120, lr: 2e-2, turnover_cost: 0.0, ..TrainConfig::default() };
        let (tuned, rep) = fine_tune(&net, &samples, &cfg).unwrap();
        assert!(rep.best_step > 0, "{rep:?}");
        assert!(rep.best_val_utility > rep.initial_val_utility + 1e-4, "{rep:?}");
        assert!(rep.final_train_utility > rep.initial_train_utility, "{rep:?}");
        let up = tuned.forward(&[1.5, 0.0]).mean;
        let down = tuned.forward(&[-1.5, 0.0]).mean;
        assert!(up > 0.2 && down < -0.2, "learned the sign: {up} {down}");
        // deterministic
        let (again, _) = fine_tune(&net, &samples, &cfg).unwrap();
        assert_eq!(again, tuned);
        // too little data is an error, not a silent no-op
        assert!(fine_tune(&net, &samples[..4], &cfg).is_err());
        // a big turnover cost makes the policy trade less (smaller |Δm| between steps)
        let costly = TrainConfig { turnover_cost: 0.05, ..cfg.clone() };
        let (calm, _) = fine_tune(&net, &samples, &costly).unwrap();
        let churn = |n: &PolicyNet| -> f64 {
            samples.windows(4).map(|w| (n.forward(&w[2].x).mean - n.forward(&w[0].x).mean).abs()).sum::<f64>()
        };
        assert!(churn(&calm) < churn(&tuned), "turnover cost reduces churn");
    }
}
