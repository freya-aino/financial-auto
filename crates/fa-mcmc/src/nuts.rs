//! No-U-Turn Sampler (Hoffman & Gelman 2014, Algorithm 6: slice NUTS with dual-averaging step
//! size adaptation) and the convergence diagnostics spec 33 §8 asks for (split R-hat, effective
//! sample size, divergent transitions). Used by the batch refit of the covariate-conditioned
//! kernel ([`crate::refit`]); any differentiable log density plugs in.
//!
//! Identity mass matrix: callers standardise their parameters (the logistic-normal kernel uses
//! standardised covariates and a unit-scale prior), which is what the identity metric assumes.

use fa_core::rng::Rng;
use serde::{Deserialize, Serialize};

/// Log density with gradient: returns `log p(θ)` and writes `∇ log p(θ)` into `grad`.
pub trait LogDensity {
    fn dim(&self) -> usize;
    fn logp_grad(&self, theta: &[f64], grad: &mut [f64]) -> f64;
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NutsCfg {
    pub warmup: usize,
    pub samples: usize,
    /// Target mean acceptance statistic `δ`.
    pub target_accept: f64,
    pub max_depth: usize,
    /// Energy error that counts as a divergence (`Δmax`).
    pub max_energy_error: f64,
}

impl Default for NutsCfg {
    fn default() -> Self {
        Self { warmup: 200, samples: 200, target_accept: 0.8, max_depth: 6, max_energy_error: 1000.0 }
    }
}

/// One chain's post-warmup draws and diagnostics.
#[derive(Debug, Clone, Default)]
pub struct Chain {
    pub draws: Vec<Vec<f64>>,
    pub divergences: usize,
    pub step_size: f64,
    pub mean_accept: f64,
    pub mean_tree_depth: f64,
    /// Gradient evaluations (cost).
    pub grad_evals: u64,
}

#[derive(Clone)]
struct Point {
    theta: Vec<f64>,
    r: Vec<f64>,
    grad: Vec<f64>,
    logp: f64,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

struct Sampler<'a, D: LogDensity> {
    target: &'a D,
    grad_evals: u64,
    max_energy_error: f64,
    divergent: bool,
}

struct Tree {
    minus: Point,
    plus: Point,
    proposal: Point,
    n: f64,
    ok: bool,
    alpha: f64,
    n_alpha: f64,
}

impl<D: LogDensity> Sampler<'_, D> {
    fn leapfrog(&mut self, p: &Point, eps: f64) -> Point {
        let mut r: Vec<f64> = p.r.iter().zip(&p.grad).map(|(r, g)| r + 0.5 * eps * g).collect();
        let theta: Vec<f64> = p.theta.iter().zip(&r).map(|(t, r)| t + eps * r).collect();
        let mut grad = vec![0.0; theta.len()];
        let logp = self.target.logp_grad(&theta, &mut grad);
        self.grad_evals += 1;
        for (ri, g) in r.iter_mut().zip(&grad) {
            *ri += 0.5 * eps * g;
        }
        Point { theta, r, grad, logp }
    }

    fn joint(p: &Point) -> f64 {
        p.logp - 0.5 * dot(&p.r, &p.r)
    }

    fn no_u_turn(minus: &Point, plus: &Point) -> bool {
        let d: Vec<f64> = plus.theta.iter().zip(&minus.theta).map(|(a, b)| a - b).collect();
        dot(&d, &minus.r) >= 0.0 && dot(&d, &plus.r) >= 0.0
    }

    #[allow(clippy::too_many_arguments)]
    fn build(&mut self, p: &Point, log_u: f64, v: f64, depth: usize, eps: f64, joint0: f64, rng: &mut Rng) -> Tree {
        if depth == 0 {
            let q = self.leapfrog(p, v * eps);
            let joint = Self::joint(&q);
            let joint = if joint.is_finite() { joint } else { f64::NEG_INFINITY };
            let n = if log_u <= joint { 1.0 } else { 0.0 };
            let ok = log_u < joint + self.max_energy_error;
            if !ok {
                self.divergent = true;
            }
            let alpha = (joint - joint0).exp().min(1.0);
            let alpha = if alpha.is_finite() { alpha } else { 0.0 };
            return Tree { minus: q.clone(), plus: q.clone(), proposal: q, n, ok, alpha, n_alpha: 1.0 };
        }
        let mut t = self.build(p, log_u, v, depth - 1, eps, joint0, rng);
        if !t.ok {
            return t;
        }
        let t2 = if v < 0.0 {
            let start = t.minus.clone();
            let t2 = self.build(&start, log_u, v, depth - 1, eps, joint0, rng);
            t.minus = t2.minus.clone();
            t2
        } else {
            let start = t.plus.clone();
            let t2 = self.build(&start, log_u, v, depth - 1, eps, joint0, rng);
            t.plus = t2.plus.clone();
            t2
        };
        if t.n + t2.n > 0.0 && rng.uniform() < t2.n / (t.n + t2.n) {
            t.proposal = t2.proposal;
        }
        t.alpha += t2.alpha;
        t.n_alpha += t2.n_alpha;
        t.ok = t2.ok && Self::no_u_turn(&t.minus, &t.plus);
        t.n += t2.n;
        t
    }

    fn momentum(dim: usize, rng: &mut Rng) -> Vec<f64> {
        (0..dim).map(|_| rng.normal()).collect()
    }

    fn reasonable_eps(&mut self, start: &Point, rng: &mut Rng) -> f64 {
        let mut eps = 1.0;
        let mut p = start.clone();
        p.r = Self::momentum(start.theta.len(), rng);
        let j0 = Self::joint(&p);
        let mut q = self.leapfrog(&p, eps);
        let ratio = |q: &Point| (Self::joint(q) - j0).clamp(-700.0, 700.0);
        let a: f64 = if ratio(&q) > 0.5f64.ln() { 1.0 } else { -1.0 };
        for _ in 0..50 {
            if a * ratio(&q) <= -a * 2f64.ln() {
                break;
            }
            eps *= 2f64.powf(a);
            q = self.leapfrog(&p, eps);
        }
        eps.clamp(1e-6, 10.0)
    }
}

/// Runs one NUTS chain from `init`.
pub fn sample<D: LogDensity>(target: &D, init: &[f64], cfg: &NutsCfg, rng: &mut Rng) -> Chain {
    let dim = target.dim();
    let mut s = Sampler { target, grad_evals: 0, max_energy_error: cfg.max_energy_error, divergent: false };
    let mut grad = vec![0.0; dim];
    let logp = target.logp_grad(init, &mut grad);
    let mut cur = Point { theta: init.to_vec(), r: vec![0.0; dim], grad, logp };
    let mut eps = s.reasonable_eps(&cur, rng);
    let (mu, gamma, t0, kappa) = ((10.0 * eps).ln(), 0.05, 10.0, 0.75);
    let (mut h_bar, mut log_eps_bar) = (0.0, 0.0);
    let mut out = Chain::default();
    let (mut acc_sum, mut depth_sum) = (0.0, 0.0);
    for m in 1..=(cfg.warmup + cfg.samples) {
        cur.r = Sampler::<D>::momentum(dim, rng);
        let joint0 = Sampler::<D>::joint(&cur);
        let log_u = joint0 + rng.uniform().max(f64::MIN_POSITIVE).ln();
        let (mut minus, mut plus) = (cur.clone(), cur.clone());
        let mut next = cur.clone();
        let (mut n, mut ok, mut depth) = (1.0, true, 0usize);
        let (mut alpha, mut n_alpha) = (0.0, 1.0);
        s.divergent = false;
        while ok && depth < cfg.max_depth {
            let v = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
            let t = if v < 0.0 {
                let t = s.build(&minus, log_u, v, depth, eps, joint0, rng);
                minus = t.minus.clone();
                t
            } else {
                let t = s.build(&plus, log_u, v, depth, eps, joint0, rng);
                plus = t.plus.clone();
                t
            };
            if t.ok && rng.uniform() < (t.n / n).min(1.0) {
                next = t.proposal.clone();
            }
            n += t.n;
            alpha = t.alpha;
            n_alpha = t.n_alpha;
            ok = t.ok && Sampler::<D>::no_u_turn(&minus, &plus);
            depth += 1;
        }
        cur = next;
        let accept = (alpha / n_alpha).clamp(0.0, 1.0);
        if m <= cfg.warmup {
            let w = 1.0 / (m as f64 + t0);
            h_bar = (1.0 - w) * h_bar + w * (cfg.target_accept - accept);
            let log_eps = mu - (m as f64).sqrt() / gamma * h_bar;
            let mk = (m as f64).powf(-kappa);
            log_eps_bar = mk * log_eps + (1.0 - mk) * log_eps_bar;
            eps = log_eps.exp().clamp(1e-6, 10.0);
            if m == cfg.warmup {
                eps = log_eps_bar.exp().clamp(1e-6, 10.0);
            }
        } else {
            out.draws.push(cur.theta.clone());
            out.divergences += usize::from(s.divergent);
            acc_sum += accept;
            depth_sum += depth as f64;
        }
    }
    let n = cfg.samples.max(1) as f64;
    out.step_size = eps;
    out.mean_accept = acc_sum / n;
    out.mean_tree_depth = depth_sum / n;
    out.grad_evals = s.grad_evals;
    out
}

/// Split R-hat (Gelman et al. BDA3 §11.4) of one scalar over several chains.
pub fn split_rhat(chains: &[Vec<f64>]) -> f64 {
    let halves: Vec<&[f64]> = chains
        .iter()
        .filter(|c| c.len() >= 4)
        .flat_map(|c| {
            let h = c.len() / 2;
            [&c[..h], &c[c.len() - h..]]
        })
        .collect();
    if halves.len() < 2 {
        return f64::NAN;
    }
    let n = halves[0].len() as f64;
    let means: Vec<f64> = halves.iter().map(|h| h.iter().sum::<f64>() / h.len() as f64).collect();
    let grand = means.iter().sum::<f64>() / means.len() as f64;
    let b = n / (means.len() as f64 - 1.0) * means.iter().map(|m| (m - grand).powi(2)).sum::<f64>();
    let w = halves
        .iter()
        .zip(&means)
        .map(|(h, m)| h.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (h.len() as f64 - 1.0))
        .sum::<f64>()
        / halves.len() as f64;
    if w <= 0.0 {
        return if b <= 0.0 { 1.0 } else { f64::INFINITY };
    }
    let var_plus = (n - 1.0) / n * w + b / n;
    (var_plus / w).sqrt()
}

/// Effective sample size over chains (Geyer's initial positive sequence on the averaged
/// autocorrelation).
pub fn ess(chains: &[Vec<f64>]) -> f64 {
    let chains: Vec<&Vec<f64>> = chains.iter().filter(|c| c.len() >= 4).collect();
    if chains.is_empty() {
        return 0.0;
    }
    let n = chains.iter().map(|c| c.len()).min().unwrap_or(0);
    let m = chains.len() as f64;
    let mut acov = vec![0.0; n];
    let mut var = 0.0;
    for c in &chains {
        let c = &c[..n];
        let mean = c.iter().sum::<f64>() / n as f64;
        let v = c.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        var += v / m;
        for (lag, a) in acov.iter_mut().enumerate() {
            let s: f64 = (0..n - lag).map(|t| (c[t] - mean) * (c[t + lag] - mean)).sum::<f64>() / n as f64;
            *a += s / m;
        }
    }
    if var <= 0.0 {
        return m * n as f64;
    }
    let rho: Vec<f64> = acov.iter().map(|a| a / var).collect();
    let mut tau = -1.0;
    let mut k = 0;
    while k + 1 < n {
        let pair = rho[k] + rho[k + 1];
        if pair <= 0.0 {
            break;
        }
        tau += 2.0 * pair;
        k += 2;
    }
    (m * n as f64 / tau.max(1.0 / (m * n as f64))).min(m * n as f64)
}

/// R-hat reported for chains that do not mix at all (finite so it serialises).
pub const RHAT_DIVERGED: f64 = 1e9;

/// `x`, or [`RHAT_DIVERGED`] when it is not finite.
pub fn finite_rhat(x: f64) -> f64 {
    if x.is_finite() { x } else { RHAT_DIVERGED }
}

/// Worst diagnostics over every coordinate of several chains.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Diagnostics {
    pub max_rhat: f64,
    pub min_ess: f64,
    pub divergences: usize,
    pub draws: usize,
}

impl Diagnostics {
    pub fn of(chains: &[Chain]) -> Self {
        let dim = chains.iter().find_map(|c| c.draws.first().map(Vec::len)).unwrap_or(0);
        let (mut max_rhat, mut min_ess) = (1.0f64, f64::INFINITY);
        for j in 0..dim {
            let cols: Vec<Vec<f64>> = chains.iter().map(|c| c.draws.iter().map(|d| d[j]).collect()).collect();
            let r = split_rhat(&cols);
            if r.is_finite() {
                max_rhat = max_rhat.max(r);
            } else if r.is_infinite() {
                max_rhat = RHAT_DIVERGED;
            }
            min_ess = min_ess.min(ess(&cols));
        }
        Self {
            max_rhat,
            min_ess: if min_ess.is_finite() { min_ess } else { 0.0 },
            divergences: chains.iter().map(|c| c.divergences).sum(),
            draws: chains.iter().map(|c| c.draws.len()).sum(),
        }
    }

    /// Spec 33 §8: R-hat ≤ `max_rhat`, at most `max_div_share` divergent transitions, enough ESS.
    pub fn acceptable(&self, max_rhat: f64, max_div_share: f64, min_ess: f64) -> bool {
        self.max_rhat <= max_rhat
            && (self.divergences as f64) <= max_div_share * self.draws.max(1) as f64
            && self.min_ess >= min_ess
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Correlated 2-D Gaussian with known moments.
    struct Gauss {
        mean: [f64; 2],
        prec: [[f64; 2]; 2],
    }

    impl LogDensity for Gauss {
        fn dim(&self) -> usize {
            2
        }
        fn logp_grad(&self, t: &[f64], g: &mut [f64]) -> f64 {
            let d = [t[0] - self.mean[0], t[1] - self.mean[1]];
            g[0] = -(self.prec[0][0] * d[0] + self.prec[0][1] * d[1]);
            g[1] = -(self.prec[1][0] * d[0] + self.prec[1][1] * d[1]);
            0.5 * (d[0] * g[0] + d[1] * g[1])
        }
    }

    #[test]
    fn nuts_recovers_a_correlated_gaussian() {
        // covariance [[1, .8], [.8, 1]] → precision 1/(1-.64) [[1, -.8], [-.8, 1]]
        let k = 1.0 / 0.36;
        let g = Gauss { mean: [1.0, -2.0], prec: [[k, -0.8 * k], [-0.8 * k, k]] };
        let cfg = NutsCfg { warmup: 300, samples: 600, ..NutsCfg::default() };
        let chains: Vec<Chain> = (0..4).map(|c| sample(&g, &[5.0, 5.0], &cfg, &mut Rng::seed(10 + c))).collect();
        let all: Vec<&Vec<f64>> = chains.iter().flat_map(|c| &c.draws).collect();
        let m0 = all.iter().map(|d| d[0]).sum::<f64>() / all.len() as f64;
        let m1 = all.iter().map(|d| d[1]).sum::<f64>() / all.len() as f64;
        let v0 = all.iter().map(|d| (d[0] - m0).powi(2)).sum::<f64>() / all.len() as f64;
        let c01 = all.iter().map(|d| (d[0] - m0) * (d[1] - m1)).sum::<f64>() / all.len() as f64;
        assert!((m0 - 1.0).abs() < 0.15 && (m1 + 2.0).abs() < 0.15, "{m0} {m1}");
        assert!((v0 - 1.0).abs() < 0.25 && (c01 - 0.8).abs() < 0.25, "{v0} {c01}");
        let d = Diagnostics::of(&chains);
        assert!(d.max_rhat < 1.05, "{d:?}");
        assert!(d.min_ess > 200.0, "{d:?}");
        assert_eq!(d.divergences, 0);
        assert!(d.acceptable(1.05, 0.01, 100.0));
        assert!(chains[0].mean_accept > 0.6, "{}", chains[0].mean_accept);
    }

    #[test]
    fn rhat_flags_chains_stuck_in_different_places() {
        let mut rng = Rng::seed(1);
        let a: Vec<f64> = (0..200).map(|_| rng.normal()).collect();
        let b: Vec<f64> = (0..200).map(|_| 5.0 + rng.normal()).collect();
        assert!(split_rhat(&[a.clone(), b]) > 1.5);
        let c: Vec<f64> = (0..200).map(|_| rng.normal()).collect();
        assert!(split_rhat(&[a.clone(), c.clone()]) < 1.05);
        assert!(ess(&[a, c]) > 250.0);
    }
}
