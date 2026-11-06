//! Pluggable meta-allocators and cross-strategy netting (spec 40).

use std::collections::BTreeMap;

use fa_core::InstrumentId;
use fa_strategies::ExposureDistribution;
use serde::{Deserialize, Serialize};

/// Posterior performance estimate per strategy used by allocators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyStats {
    pub strategy_id: String,
    /// Expected return per horizon (in sleeve units).
    pub mean_return: f64,
    /// Return volatility per horizon.
    pub volatility: f64,
}

/// Capital weight per strategy (sum ≤ 1).
pub type Allocation = BTreeMap<String, f64>;

pub trait Allocator: Send + Sync {
    fn id(&self) -> &str;
    fn allocate(&self, stats: &[StrategyStats]) -> Allocation;
}

/// Inverse-volatility (risk parity for uncorrelated sleeves), each weight capped,
/// excluding strategies with non-positive expected return.
#[derive(Debug, Clone, Copy)]
pub struct RiskParityCapped {
    pub max_weight: f64,
}

impl Allocator for RiskParityCapped {
    fn id(&self) -> &str {
        "risk_parity_capped"
    }

    fn allocate(&self, stats: &[StrategyStats]) -> Allocation {
        let eligible: Vec<_> = stats.iter().filter(|s| s.mean_return > 0.0 && s.volatility > 0.0).collect();
        let inv: f64 = eligible.iter().map(|s| 1.0 / s.volatility).sum();
        eligible
            .iter()
            .map(|s| (s.strategy_id.clone(), ((1.0 / s.volatility) / inv).min(self.max_weight)))
            .collect()
    }
}

/// Fractional Kelly: w = f · μ/σ², clipped to [0, max_weight], then scaled so Σw ≤ 1.
#[derive(Debug, Clone, Copy)]
pub struct FractionalKelly {
    pub fraction: f64,
    pub max_weight: f64,
}

impl Allocator for FractionalKelly {
    fn id(&self) -> &str {
        "fractional_kelly"
    }

    fn allocate(&self, stats: &[StrategyStats]) -> Allocation {
        let mut a: Allocation = stats
            .iter()
            .filter(|s| s.volatility > 0.0)
            .map(|s| (s.strategy_id.clone(), (self.fraction * s.mean_return / s.volatility.powi(2)).clamp(0.0, self.max_weight)))
            .filter(|(_, w)| *w > 0.0)
            .collect();
        let total: f64 = a.values().sum();
        if total > 1.0 {
            a.values_mut().for_each(|w| *w /= total);
        }
        a
    }
}

/// Nets strategy exposure means into one target per instrument, in EUR, given capital
/// weights and deployable capital. Exposure uncertainty shrinks the contribution:
/// `mean / (1 + stddev)`.
pub fn net_targets(dists: &[ExposureDistribution], alloc: &Allocation, deployable_eur: f64) -> BTreeMap<InstrumentId, f64> {
    let mut out: BTreeMap<InstrumentId, f64> = BTreeMap::new();
    for d in dists {
        let w = alloc.get(&d.strategy_id).copied().unwrap_or(0.0);
        if w == 0.0 || d.exposures.is_empty() {
            continue;
        }
        let per_asset = w * deployable_eur / d.exposures.len() as f64;
        for e in &d.exposures {
            *out.entry(e.instrument).or_insert(0.0) += per_asset * e.mean.clamp(-1.0, 1.0) / (1.0 + e.stddev.max(0.0));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use fa_core::{Horizon, Timescale};
    use fa_strategies::AssetExposure;

    fn stats() -> Vec<StrategyStats> {
        vec![
            StrategyStats { strategy_id: "a".into(), mean_return: 0.01, volatility: 0.1 },
            StrategyStats { strategy_id: "b".into(), mean_return: 0.01, volatility: 0.2 },
            StrategyStats { strategy_id: "c".into(), mean_return: -0.01, volatility: 0.1 },
        ]
    }

    #[test]
    fn risk_parity_weights_inverse_vol() {
        let a = RiskParityCapped { max_weight: 1.0 }.allocate(&stats());
        assert!((a["a"] - 2.0 / 3.0).abs() < 1e-12);
        assert!((a["b"] - 1.0 / 3.0).abs() < 1e-12);
        assert!(!a.contains_key("c"));
        let capped = RiskParityCapped { max_weight: 0.5 }.allocate(&stats());
        assert_eq!(capped["a"], 0.5);
    }

    #[test]
    fn kelly_scales_and_normalises() {
        let a = FractionalKelly { fraction: 0.25, max_weight: 1.0 }.allocate(&stats());
        assert!(a.values().sum::<f64>() <= 1.0 + 1e-12);
        assert!(a["a"] > a["b"]);
        assert!(!a.contains_key("c"));
    }

    #[test]
    fn netting_offsets_opposing_strategies() {
        let i = InstrumentId::new();
        let h = Horizon { timescale: Timescale::Hour, steps: 2 };
        let d = |id: &str, m| ExposureDistribution {
            strategy_id: id.into(),
            genome_id: String::new(),
            horizon: h,
            as_of_ms: 0,
            exposures: vec![AssetExposure { instrument: i, mean: m, stddev: 0.0 }],
        };
        let alloc = Allocation::from([("a".to_string(), 0.5), ("b".to_string(), 0.5)]);
        let t = net_targets(&[d("a", 1.0), d("b", -1.0)], &alloc, 10_000.0);
        assert!(t[&i].abs() < 1e-9);
        let t = net_targets(&[d("a", 1.0), d("b", 0.5)], &alloc, 10_000.0);
        assert!((t[&i] - 7_500.0).abs() < 1e-9);
    }
}
