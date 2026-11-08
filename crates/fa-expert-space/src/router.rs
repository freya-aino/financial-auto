//! Router stage (spec 32 §6.1).
//!
//! The spec's router is a learned MoE gate. Until the neural backends are enabled (D-004)
//! baseline routers derive the weights from per-expert evidence carried in the latents
//! themselves. The `Router` trait is the stage boundary, so a learned gate can replace a
//! baseline without touching the runtime.

use serde::{Deserialize, Serialize};

use crate::ExpertLatent;

/// Produces the gate weights over experts for one window. Weights sum to 1.
pub trait Router: Send + Sync {
    fn id(&self) -> &str;
    fn route(&self, experts: &[ExpertLatent]) -> Vec<f32>;
}

/// What a baseline router reads from each expert latent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Evidence {
    /// `−mean(log_var)`: experts with a tighter posterior get more weight.
    Confidence,
    /// `‖mean‖₂`: experts whose statistics deviate most from the neutral state get more weight
    /// (for z-scored baseline latents this is "which expert sees something happening").
    Energy,
}

/// `softmax(evidence_k / temperature)` over experts.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EvidenceSoftmaxRouter {
    pub evidence: Evidence,
    pub temperature: f32,
}

impl Default for EvidenceSoftmaxRouter {
    fn default() -> Self {
        Self { evidence: Evidence::Energy, temperature: 1.0 }
    }
}

impl EvidenceSoftmaxRouter {
    fn evidence_of(&self, e: &ExpertLatent) -> f32 {
        match self.evidence {
            Evidence::Confidence => {
                if e.log_var.is_empty() {
                    0.0
                } else {
                    -(e.log_var.iter().sum::<f32>() / e.log_var.len() as f32)
                }
            }
            Evidence::Energy => e.mean.iter().map(|x| x * x).sum::<f32>().sqrt(),
        }
    }
}

impl Router for EvidenceSoftmaxRouter {
    fn id(&self) -> &str {
        match self.evidence {
            Evidence::Confidence => "softmax.confidence",
            Evidence::Energy => "softmax.energy",
        }
    }

    fn route(&self, experts: &[ExpertLatent]) -> Vec<f32> {
        let t = if self.temperature > 0.0 { self.temperature } else { 1.0 };
        let logits: Vec<f32> = experts.iter().map(|e| self.evidence_of(e) / t).collect();
        softmax(&logits)
    }
}

/// Numerically stable softmax; uniform for an empty or non-finite input.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    if logits.iter().any(|x| !x.is_finite()) {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let s: f32 = exp.iter().sum();
    exp.into_iter().map(|x| x / s).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn latent(i: u32, mean: Vec<f32>, lv: f32) -> ExpertLatent {
        let n = mean.len();
        ExpertLatent { expert_index: i, mean, log_var: vec![lv; n], region_id: None, region_lineage_id: None }
    }

    #[test]
    fn softmax_sums_to_one_and_orders() {
        let w = softmax(&[1.0, 2.0, 3.0]);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(w[0] < w[1] && w[1] < w[2]);
        assert_eq!(softmax(&[f32::NAN, 1.0]), vec![0.5, 0.5]);
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn energy_router_prefers_the_active_expert() {
        let r = EvidenceSoftmaxRouter::default();
        let w = r.route(&[latent(0, vec![0.0, 0.0], 0.0), latent(1, vec![0.9, 0.9], 0.0)]);
        assert!(w[1] > w[0]);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn confidence_router_prefers_tight_posteriors() {
        let r = EvidenceSoftmaxRouter { evidence: Evidence::Confidence, temperature: 1.0 };
        let w = r.route(&[latent(0, vec![1.0], 0.0), latent(1, vec![1.0], -2.0)]);
        assert!(w[1] > w[0]);
        assert_eq!(r.id(), "softmax.confidence");
    }
}
