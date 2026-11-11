//! Pluggable meta-allocators and cross-strategy netting (spec 40).
//!
//! Allocators turn per-strategy performance posteriors ([`StrategyStats`]) into capital
//! weights; [`net_targets`] nets exposure distributions into EUR targets per instrument.
//! [`PerformanceTracker`] estimates the posteriors online from signals and realised returns.
//! [`genome`] carries the evolvable form of an allocator ([`genome::AllocatorGenome`]: name +
//! genes, text round trip, variation operators) and [`WeightedBlend`] is the parametric family
//! mixing the fixed allocators (spec 40 §6.2 "evolved allocator"). [`ThompsonRiskParity`] is
//! the spec's proposed paper default (Bayesian bandit + risk parity) and [`tournament`] runs
//! every allocator side by side on the same live strategy returns (spec 40 §6.2 "allocator
//! tournament").

pub mod genome;
pub mod tournament;

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
    /// Observations behind the estimate (posterior width of the mean for Thompson sampling).
    #[serde(default)]
    pub observations: u64,
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

/// Parametric blend of the three fixed allocators (spec 40 §6.2 "evolved allocator"): the
/// equal-weight, risk-parity and fractional-Kelly allocations (each uncapped) are mixed with
/// the normalised weights `w_equal : w_risk_parity : w_kelly`, then every strategy weight is
/// capped at `max_weight` and the sum scaled to ≤ 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedBlend {
    pub max_weight: f64,
    pub kelly_fraction: f64,
    pub w_equal: f64,
    pub w_risk_parity: f64,
    pub w_kelly: f64,
}

impl Allocator for WeightedBlend {
    fn id(&self) -> &str {
        "weighted_blend"
    }

    fn allocate(&self, stats: &[StrategyStats]) -> Allocation {
        let ws = [self.w_equal.max(0.0), self.w_risk_parity.max(0.0), self.w_kelly.max(0.0)];
        let total_w: f64 = ws.iter().sum();
        if total_w <= 0.0 || stats.is_empty() {
            return Allocation::new();
        }
        let legs: [Allocation; 3] = [
            EqualWeight { max_weight: 1.0 }.allocate(stats),
            RiskParityCapped { max_weight: 1.0 }.allocate(stats),
            FractionalKelly { fraction: self.kelly_fraction, max_weight: 1.0 }.allocate(stats),
        ];
        let mut out = Allocation::new();
        for (w, leg) in ws.iter().zip(&legs) {
            if *w <= 0.0 {
                continue;
            }
            for (id, v) in leg {
                *out.entry(id.clone()).or_insert(0.0) += w / total_w * v;
            }
        }
        out.retain(|_, v| *v > 0.0);
        out.values_mut().for_each(|v| *v = v.min(self.max_weight));
        let total: f64 = out.values().sum();
        if total > 1.0 {
            out.values_mut().for_each(|v| *v /= total);
        }
        out
    }
}

/// Bayesian bandit + risk parity (spec 40 §6.2, the proposed paper default): every call draws
/// one Thompson sample of each strategy's mean return from its posterior
/// `N(mean, volatility² / max(n, 1))`; strategies whose draw is positive share the capital by
/// inverse volatility (risk parity for uncorrelated sleeves), each capped at `max_weight`.
/// Strategies with little history have wide posteriors and are explored; a strategy whose mean
/// is reliably negative is almost never drawn positive. Deterministic: the k-th call uses
/// `seed` and k, so a replay that makes the same calls gets the same weights.
#[derive(Debug)]
pub struct ThompsonRiskParity {
    pub max_weight: f64,
    pub seed: u64,
    draws: std::sync::atomic::AtomicU64,
}

impl ThompsonRiskParity {
    pub fn new(max_weight: f64, seed: u64) -> Self {
        Self { max_weight, seed, draws: std::sync::atomic::AtomicU64::new(0) }
    }
}

impl Allocator for ThompsonRiskParity {
    fn id(&self) -> &str {
        "thompson_rp"
    }

    fn allocate(&self, stats: &[StrategyStats]) -> Allocation {
        let k = self.draws.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut rng = fa_core::rng::Rng::seed(self.seed ^ k.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let mut sorted: Vec<&StrategyStats> = stats.iter().filter(|s| s.volatility > 0.0).collect();
        sorted.sort_by(|a, b| a.strategy_id.cmp(&b.strategy_id));
        let drawn: Vec<&StrategyStats> = sorted
            .into_iter()
            .filter(|s| {
                let se = s.volatility / (s.observations.max(1) as f64).sqrt();
                let draw = s.mean_return + se * rng.normal();
                draw.is_finite() && draw > 0.0
            })
            .collect();
        let inv: f64 = drawn.iter().map(|s| 1.0 / s.volatility).sum();
        if inv <= 0.0 {
            return Allocation::new();
        }
        drawn.iter().map(|s| (s.strategy_id.clone(), ((1.0 / s.volatility) / inv).min(self.max_weight))).collect()
    }
}

/// Builds the allocator named in config: `equal_weight` | `risk_parity` | `fractional_kelly` | `thompson_rp`
/// (`weighted_blend` needs its blend weights: use [`genome::AllocatorGenome::build`]).
pub fn by_name(name: &str, max_weight: f64, kelly_fraction: f64) -> Option<Box<dyn Allocator>> {
    Some(match name {
        "equal_weight" => Box::new(EqualWeight { max_weight }),
        "risk_parity" | "risk_parity_capped" => Box::new(RiskParityCapped { max_weight }),
        "fractional_kelly" | "kelly" => Box::new(FractionalKelly { fraction: kelly_fraction, max_weight }),
        "thompson_rp" | "thompson" | "bandit" => Box::new(ThompsonRiskParity::new(max_weight, 0x7a11_0ca7)),
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

    /// A strategy's sleeve return over one step: the mean over its assets of
    /// `exposure_mean × realised_return` (missing returns count as 0); `None` without exposures.
    pub fn step_return(dist: &ExposureDistribution, realised: &BTreeMap<InstrumentId, f64>) -> Option<f64> {
        if dist.exposures.is_empty() {
            return None;
        }
        Some(
            dist.exposures
                .iter()
                .map(|e| e.mean.clamp(-1.0, 1.0) * realised.get(&e.instrument).copied().unwrap_or(0.0))
                .sum::<f64>()
                / dist.exposures.len() as f64,
        )
    }

    /// Records one step for `strategy_id` given its last exposures and the realised
    /// instrument returns over the step (missing returns count as 0).
    pub fn observe(&mut self, dist: &ExposureDistribution, realised: &BTreeMap<InstrumentId, f64>) {
        let Some(r) = Self::step_return(dist, realised) else {
            return;
        };
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
                    observations: e.map_or(0, |e| e.n),
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
    net_targets_shrunk(dists, alloc, deployable_eur, 1.0)
}

/// [`net_targets`] with a tunable shrink exponent: `mean / (1 + stddev)^shrink` (`0` ignores
/// the uncertainty, `1` is the live default). Also returns nothing for strategies without a
/// weight. Used by evolved allocators (`genome::AllocatorGenome::shrink`).
pub fn net_targets_shrunk(
    dists: &[ExposureDistribution],
    alloc: &Allocation,
    deployable_eur: f64,
    shrink: f64,
) -> BTreeMap<InstrumentId, f64> {
    let mut out: BTreeMap<InstrumentId, f64> = BTreeMap::new();
    for d in dists {
        let w = alloc.get(&d.strategy_id).copied().unwrap_or(0.0);
        if w == 0.0 || d.exposures.is_empty() {
            continue;
        }
        let per_asset = w * deployable_eur / d.exposures.len() as f64;
        for e in &d.exposures {
            let denom = (1.0 + e.stddev.max(0.0)).powf(shrink.max(0.0));
            *out.entry(e.instrument).or_insert(0.0) += per_asset * e.mean.clamp(-1.0, 1.0) / denom;
        }
    }
    out
}

/// Fill attribution weights of a netted order (spec 40 §6.3, §10 Q3 — the causal pro-rata
/// rule). `parts` holds per contributing strategy its signed EUR contribution to the
/// instrument's netted target and the signed EUR value its virtual sub-book already holds in
/// it; `delta` is the order (target − position). A strategy's weight is how far its own
/// sub-target is from its sub-book **in the order's direction** — `max(0, sign(Δ)·(c − h))` —
/// so the fill (and its fees and slippage) goes to the strategies whose targets caused the
/// trade, in proportion, and each sub-book converges to its own contribution; a strategy whose
/// target moved against the order gets none of it. When no strategy wants the order's
/// direction (e.g. sub-books not built yet) the weights are `|c|`, the contribution shares;
/// when those are zero too, empty (= even split, the D-047 rule).
pub fn attribution_weights(delta: f64, parts: &[(f64, f64)]) -> Vec<f64> {
    let dir = delta.signum();
    let ok = |x: f64| if x.is_finite() && x > 0.0 { x } else { 0.0 };
    let causal: Vec<f64> = parts.iter().map(|(c, h)| ok(dir * (c - h))).collect();
    if delta != 0.0 && causal.iter().sum::<f64>() > 0.0 {
        return causal;
    }
    let share: Vec<f64> = parts.iter().map(|(c, _)| ok(c.abs())).collect();
    if share.iter().sum::<f64>() > 0.0 { share } else { Vec::new() }
}

/// Normalised shares (summing to 1) for `n` contributors from [`attribution_weights`]; an even
/// split when the weights are missing, of another length or sum to nothing.
pub fn attribution_shares(n: usize, weights: &[f64]) -> Vec<f64> {
    let total: f64 = weights.iter().filter(|w| w.is_finite() && **w > 0.0).sum();
    if n == 0 {
        return Vec::new();
    }
    if weights.len() != n || total <= 0.0 || !total.is_finite() {
        return vec![1.0 / n as f64; n];
    }
    weights.iter().map(|w| if w.is_finite() && *w > 0.0 { w / total } else { 0.0 }).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn attribution_follows_the_strategies_that_caused_the_trade() {
        use super::{attribution_shares, attribution_weights};
        // A wants +1 000 (holds 0), B wants −800 (holds −800): the +200 order is A's alone
        assert_eq!(attribution_weights(200.0, &[(1_000.0, 0.0), (-800.0, -800.0)]), vec![1_000.0, 0.0]);
        // both moved up: pro rata to how far each is from its sub-book
        let w = attribution_weights(300.0, &[(500.0, 300.0), (400.0, 300.0)]);
        assert_eq!(attribution_shares(2, &w), vec![200.0 / 300.0, 100.0 / 300.0]);
        // no sub-books in the order's direction: contribution shares; nothing at all: even
        assert_eq!(attribution_weights(-50.0, &[(100.0, 0.0), (300.0, 0.0)]), vec![100.0, 300.0]);
        assert!(attribution_weights(10.0, &[(0.0, 0.0), (0.0, 0.0)]).is_empty());
        assert_eq!(attribution_shares(2, &[]), vec![0.5, 0.5]);
        assert_eq!(attribution_shares(3, &[1.0]), vec![1.0 / 3.0; 3]);
    }

    use super::*;
    use fa_core::{Horizon, Timescale};
    use fa_strategies::AssetExposure;

    fn stats() -> Vec<StrategyStats> {
        vec![
            StrategyStats { strategy_id: "a".into(), mean_return: 0.01, volatility: 0.1, observations: 0 },
            StrategyStats { strategy_id: "b".into(), mean_return: 0.01, volatility: 0.2, observations: 0 },
            StrategyStats { strategy_id: "c".into(), mean_return: -0.01, volatility: 0.1, observations: 0 },
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
        // uncertainty shrink: stddev 1 halves with shrink 1, leaves alone with shrink 0, quarters with 2
        let u = ExposureDistribution {
            exposures: vec![AssetExposure { instrument: i, mean: 1.0, stddev: 1.0 }],
            ..d("a", 1.0)
        };
        let one = Allocation::from([("a".to_string(), 1.0)]);
        assert!((net_targets(std::slice::from_ref(&u), &one, 1_000.0)[&i] - 500.0).abs() < 1e-9);
        assert!((net_targets_shrunk(std::slice::from_ref(&u), &one, 1_000.0, 0.0)[&i] - 1_000.0).abs() < 1e-9);
        assert!((net_targets_shrunk(std::slice::from_ref(&u), &one, 1_000.0, 2.0)[&i] - 250.0).abs() < 1e-9);
    }

    #[test]
    fn thompson_explores_uncertain_and_drops_reliably_bad_strategies() {
        let st = |id: &str, m: f64, n: u64| StrategyStats {
            strategy_id: id.into(),
            mean_return: m,
            volatility: 0.01,
            observations: n,
        };
        // "good": mean 1e-3 over 10 000 obs (se 1e-4) → ~always drawn; "bad" the mirror → ~never;
        // "new": no history (se = vol) → about half the draws
        let stats = vec![st("good", 1e-3, 10_000), st("bad", -1e-3, 10_000), st("new", 0.0, 0)];
        let t = ThompsonRiskParity::new(0.6, 42);
        let (mut good, mut bad, mut new) = (0, 0, 0);
        for _ in 0..400 {
            let a = t.allocate(&stats);
            assert!(a.values().all(|w| *w <= 0.6 + 1e-12) && a.values().sum::<f64>() <= 1.0 + 1e-12);
            good += a.contains_key("good") as u32;
            bad += a.contains_key("bad") as u32;
            new += a.contains_key("new") as u32;
        }
        assert!(good > 390 && bad < 10 && (120..280).contains(&new), "{good} {bad} {new}");
        // deterministic per call sequence
        let (x, y) = (ThompsonRiskParity::new(1.0, 7), ThompsonRiskParity::new(1.0, 7));
        for _ in 0..5 {
            assert_eq!(x.allocate(&stats), y.allocate(&stats));
        }
        assert!(t.allocate(&[]).is_empty());
        assert_eq!(by_name("thompson_rp", 0.5, 0.25).unwrap().id(), "thompson_rp");
    }

    #[test]
    fn weighted_blend_mixes_caps_and_normalises() {
        let s = stats();
        let pure_rp =
            WeightedBlend { max_weight: 1.0, kelly_fraction: 0.25, w_equal: 0.0, w_risk_parity: 1.0, w_kelly: 0.0 };
        let rp = RiskParityCapped { max_weight: 1.0 }.allocate(&s);
        let b = pure_rp.allocate(&s);
        for (k, v) in &rp {
            assert!((b[k] - v).abs() < 1e-12, "pure leg equals the leg itself");
        }
        let mix =
            WeightedBlend { max_weight: 0.3, kelly_fraction: 0.25, w_equal: 2.0, w_risk_parity: 2.0, w_kelly: 0.0 };
        let m = mix.allocate(&s);
        assert!(m.values().all(|w| *w <= 0.3 + 1e-12), "{m:?}");
        assert!(m.values().sum::<f64>() <= 1.0 + 1e-12);
        assert!(m.contains_key("c"), "the equal-weight leg keeps the negative-return strategy");
        assert!(WeightedBlend { w_equal: 0.0, w_risk_parity: 0.0, w_kelly: 0.0, ..mix }.allocate(&s).is_empty());
        assert!(mix.allocate(&[]).is_empty());
        assert_eq!(mix.id(), "weighted_blend");
    }
}
