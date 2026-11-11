//! Expert spaces: decoder-facing contract, router stage and the cross-expert decorrelation
//! objective (spec 32, D-008).
//!
//! The [`Router`] trait is the stage boundary between the encoder and the regions/MCMC
//! stages. [`EvidenceSoftmaxRouter`] is the pure-Rust baseline router of the statistical family.
//! The learned MoE gate lives in [`moe`] (candle): router, per-expert projectors and next-step
//! predictors trained with the D-008 objective (residual decorrelation + load balancing + KL);
//! a neural encoder fills `ExpertView.router_weights` with its learned top-k gates, which the
//! runtime keeps instead of re-routing. [`health`] holds the §6.4 measures (residual
//! correlation, `K_eff`, per-expert skill) and the §6.5 CKA expert matching; [`nn`] the small
//! deterministic parameter store / Adam shared with `fa-encoders::neural`.

pub mod health;
pub mod lora;
pub mod moe;
pub mod nn;
pub mod router;

use fa_core::Timescale;
pub use router::{Evidence, EvidenceSoftmaxRouter, Router, softmax};
use serde::{Deserialize, Serialize};

/// Capacity of one expert space (spec 32 §4.1 tunables).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertSpaceConfig {
    /// Capacity `K_max`; the active count emerges (or is fixed, for baselines).
    pub max_experts: usize,
    /// Manifold dimension per expert.
    pub expert_dim: usize,
    /// Experts active per window.
    pub top_k: usize,
}

impl Default for ExpertSpaceConfig {
    fn default() -> Self {
        Self { max_experts: 16, expert_dim: 16, top_k: 2 }
    }
}

/// Latent of one expert at one time step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertLatent {
    pub expert_index: u32,
    pub mean: Vec<f32>,
    pub log_var: Vec<f32>,
    pub region_id: Option<String>,
    pub region_lineage_id: Option<String>,
}

/// The contract every decoder and strategy consumes (spec 32 §4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertView {
    pub snapshot_id: String,
    pub timescale: Timescale,
    pub as_of_ms: i64,
    /// Which experts this consumer reads (evolved per genome).
    pub expert_mask: Vec<bool>,
    pub experts: Vec<ExpertLatent>,
    pub router_weights: Vec<f32>,
}

impl ExpertView {
    /// Experts selected by the mask.
    pub fn selected(&self) -> impl Iterator<Item = &ExpertLatent> {
        self.experts.iter().filter(|e| self.expert_mask.get(e.expert_index as usize).copied().unwrap_or(false))
    }
}

/// Decorrelation objective across experts; alternatives plug in behind this trait.
pub trait DecorrelationLoss {
    /// `residuals[i][t]`: next-step prediction residual of expert `i` at sample `t`.
    fn loss(&self, residuals: &[Vec<f64>]) -> f64;
}

/// D-008 default: λ · Σ_{i≠j} corr(r_i, r_j)².
#[derive(Debug, Clone, Copy)]
pub struct ResidualCorrelationPenalty {
    pub lambda: f64,
}

impl DecorrelationLoss for ResidualCorrelationPenalty {
    fn loss(&self, residuals: &[Vec<f64>]) -> f64 {
        let mut s = 0.0;
        for i in 0..residuals.len() {
            for j in 0..residuals.len() {
                if i != j {
                    s += pearson(&residuals[i], &residuals[j]).powi(2);
                }
            }
        }
        self.lambda * s
    }
}

/// Pearson correlation; 0 for degenerate (constant or empty) inputs.
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let ma = a[..n].iter().sum::<f64>() / n as f64;
    let mb = b[..n].iter().sum::<f64>() / n as f64;
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for k in 0..n {
        let (da, db) = (a[k] - ma, b[k] - mb);
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va == 0.0 || vb == 0.0 { 0.0 } else { cov / (va * vb).sqrt() }
}

/// Switch-transformer style load-balancing term: `n · Σ_i f_i · p_i`, where `f_i` is the
/// fraction of samples routed (argmax) to expert i and `p_i` its mean router probability.
/// Equals 1.0 for perfectly balanced routing.
pub fn router_load_balance(router_probs: &[Vec<f64>]) -> f64 {
    let Some(n) = router_probs.first().map(Vec::len) else {
        return 0.0;
    };
    let t = router_probs.len() as f64;
    let mut f = vec![0.0; n];
    let mut p = vec![0.0; n];
    for row in router_probs {
        let arg = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
        f[arg] += 1.0 / t;
        for (i, v) in row.iter().enumerate() {
            p[i] += v / t;
        }
    }
    n as f64 * f.iter().zip(&p).map(|(a, b)| a * b).sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_basics() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert!((pearson(&a, &[2.0, 4.0, 6.0, 8.0]) - 1.0).abs() < 1e-12);
        assert!((pearson(&a, &[4.0, 3.0, 2.0, 1.0]) + 1.0).abs() < 1e-12);
        assert_eq!(pearson(&a, &[1.0, 1.0, 1.0, 1.0]), 0.0);
    }

    #[test]
    fn penalty_zero_for_uncorrelated_and_max_for_identical() {
        let l = ResidualCorrelationPenalty { lambda: 1.0 };
        let r1 = vec![1.0, -1.0, 1.0, -1.0];
        let r2 = vec![1.0, 1.0, -1.0, -1.0];
        assert!(l.loss(&[r1.clone(), r2]).abs() < 1e-12);
        assert!((l.loss(&[r1.clone(), r1]) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn load_balance_is_one_when_uniform() {
        let probs = vec![vec![0.6, 0.4], vec![0.4, 0.6]];
        assert!((router_load_balance(&probs) - 1.0).abs() < 1e-12);
        let skewed = vec![vec![0.9, 0.1], vec![0.9, 0.1]];
        assert!(router_load_balance(&skewed) > 1.0);
    }
}
