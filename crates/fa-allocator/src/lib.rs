//! Pluggable meta-allocators and cross-strategy netting (spec 40).
//!
//! Allocators turn per-strategy performance posteriors ([`StrategyStats`]) into capital
//! weights; [`net_targets`] nets exposure distributions into EUR targets per instrument.
//! [`PerformanceTracker`] estimates the posteriors online from signals and realised returns.

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
        eligible.iter().map(|s| (s.strategy_id.clone(), ((1.0 / s.volatility) / inv).min(self.max_weight))).collect()
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
            .map(|s| {
                (
                    s.strategy_id.clone(),
                    (self.fraction * s.mean_return / s.volatility.powi(2)).clamp(0.0, self.max_weight),
                )
            })
            .filter(|(_, w)| *w > 0.0)
            .collect();
        let total: f64 = a.values().sum();
        if total > 1.0 {
            a.values_mut().for_each(|w| *w /= total);
        }
        a
    }
}

/// Equal-weight control allocator (spec 40 §6.2 "equal-risk baseline"): every strategy that
/// has emitted a signal gets `min(1/n, max_weight)`. Used while no performance history exists.
#[derive(Debug, Clone, Copy)]
pub struct EqualWeight {
    pub max_weight: f64,
}

impl Allocator for EqualWeight {
    fn id(&self) -> &str {
        "equal_weight"
    }

    fn allocate(&self, stats: &[StrategyStats]) -> Allocation {
        if stats.is_empty() {
            return Allocation::new();
        }
        let w = (1.0 / stats.len() as f64).min(self.max_weight);
        stats.iter().map(|s| (s.strategy_id.clone(), w)).collect()
    }
}

/// Builds the allocator named in config: `equal_weight` | `risk_parity` | `fractional_kelly`.
pub fn by_name(name: &str, max_weight: f64, kelly_fraction: f64) -> Option<Box<dyn Allocator>> {
    Some(match name {
        "equal_weight" => Box::new(EqualWeight { max_weight }),
        "risk_parity" | "risk_parity_capped" => Box::new(RiskParityCapped { max_weight }),
        "fractional_kelly" | "kelly" => Box::new(FractionalKelly { fraction: kelly_fraction, max_weight }),
        _ => return None,
    })
}

/// Online estimate of each strategy's return per step from its emitted exposure means and the
/// realised instrument returns (exponentially weighted). A strategy's step return is the
/// mean over its assets of `exposure_mean × realised_return`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTracker {
    /// EWMA decay in (0, 1); higher = longer memory.
    pub decay: f64,
    /// Prior volatility used until a strategy has observations.
    pub prior_volatility: f64,
    stats: BTreeMap<String, Ewma>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Ewma {
    mean: f64,
    var: f64,
    n: u64,
}

impl PerformanceTracker {
    pub fn new(decay: f64, prior_volatility: f64) -> Self {
        Self {
            decay: decay.clamp(0.0, 0.999_999),
            prior_volatility: prior_volatility.max(1e-12),
            stats: BTreeMap::new(),
        }
    }

    /// Records one step for `strategy_id` given its last exposures and the realised
    /// instrument returns over the step (missing returns count as 0).
    pub fn observe(&mut self, dist: &ExposureDistribution, realised: &BTreeMap<InstrumentId, f64>) {
        if dist.exposures.is_empty() {
            return;
        }
        let r = dist
            .exposures
            .iter()
            .map(|e| e.mean.clamp(-1.0, 1.0) * realised.get(&e.instrument).copied().unwrap_or(0.0))
            .sum::<f64>()
            / dist.exposures.len() as f64;
        let e = self.stats.entry(dist.strategy_id.clone()).or_default();
        if e.n == 0 {
            e.mean = r;
            e.var = 0.0;
        } else {
            let d = r - e.mean;
            e.mean += (1.0 - self.decay) * d;
            e.var = self.decay * (e.var + (1.0 - self.decay) * d * d);
        }
        e.n += 1;
    }

    /// Stats for the given strategies; unknown ones get a zero mean and the prior volatility.
    pub fn stats_for<'a>(&self, ids: impl IntoIterator<Item = &'a str>) -> Vec<StrategyStats> {
        ids.into_iter()
            .map(|id| {
                let e = self.stats.get(id);
                let vol = e.map_or(0.0, |e| e.var.sqrt());
                StrategyStats {
                    strategy_id: id.to_owned(),
                    mean_return: e.map_or(0.0, |e| e.mean),
                    volatility: if vol > 0.0 { vol } else { self.prior_volatility },
                }
            })
            .collect()
    }

    pub fn observations(&self, id: &str) -> u64 {
        self.stats.get(id).map_or(0, |e| e.n)
    }
}

/// Nets strategy exposure means into one target per instrument, in EUR, given capital
/// weights and deployable capital. Exposure uncertainty shrinks the contribution:
/// `mean / (1 + stddev)`.
pub fn net_targets(
    dists: &[ExposureDistribution],
    alloc: &Allocation,
    deployable_eur: f64,
) -> BTreeMap<InstrumentId, f64> {
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
    fn equal_weight_caps_and_lookup_by_name() {
        let a = EqualWeight { max_weight: 0.4 }.allocate(&stats());
        assert_eq!(a.len(), 3);
        assert!((a["c"] - 1.0 / 3.0).abs() < 1e-12, "negative-return strategies still get the control weight");
        assert_eq!(EqualWeight { max_weight: 0.25 }.allocate(&stats())["a"], 0.25);
        assert!(EqualWeight { max_weight: 1.0 }.allocate(&[]).is_empty());
        assert_eq!(by_name("risk_parity", 1.0, 0.25).unwrap().id(), "risk_parity_capped");
        assert!(by_name("evolved", 1.0, 0.25).is_none());
    }

    #[test]
    fn performance_tracker_estimates_mean_and_vol() {
        let i = InstrumentId::new();
        let h = Horizon { timescale: Timescale::Second, steps: 1 };
        let d = ExposureDistribution {
            strategy_id: "s".into(),
            genome_id: String::new(),
            horizon: h,
            as_of_ms: 0,
            exposures: vec![AssetExposure { instrument: i, mean: 1.0, stddev: 0.0 }],
        };
        let mut t = PerformanceTracker::new(0.9, 0.01);
        assert_eq!(t.stats_for(["s"])[0].volatility, 0.01, "prior before observations");
        for k in 0..50 {
            t.observe(&d, &BTreeMap::from([(i, if k % 2 == 0 { 0.02 } else { -0.01 })]));
        }
        let s = &t.stats_for(["s"])[0];
        assert!(s.mean_return > 0.0 && s.mean_return < 0.02);
        assert!(s.volatility > 0.005 && s.volatility < 0.02);
        assert_eq!(t.observations("s"), 50);
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
