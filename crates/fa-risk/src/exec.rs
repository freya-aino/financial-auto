//! Execution-algorithm choice (spec 40 §6.1 step 7, §6.4): every order intent the allocator
//! chain emits names the algorithm the broker-gateway runs, chosen from the order's urgency, the
//! contributing strategies' horizon and the order's size against the instrument's traded volume.
//!
//! The result is the `OrderIntent.exec_algo` text the paper venue parses:
//! `name` or `name:k=v,k=v` with `name ∈ market | limit | smart_limit | twap | vwap | pov`.
//!
//! | Condition (first match wins) | Algo |
//! |---|---|
//! | urgent (halt flatten, kill switch, strategy flatten, stale-feed flatten) | `market` |
//! | horizon < `market_below_horizon_secs` (the edge decays faster than a passive order waits) | `market` |
//! | participation ≥ `pov_participation` of the horizon volume | `pov:rate=…,max_dur_s=horizon` |
//! | participation ≥ `twap_participation` and horizon ≥ `sliced_min_horizon_secs` | `vwap` (volume profile known) else `twap`, `dur_s = min(horizon/2, max_slice_dur_secs)` |
//! | notional < `market_below_eur` | `market` (a passive wait costs more than the spread it saves) |
//! | otherwise | `smart_limit:timeout_s=clamp(horizon/4, 5, smart_limit_timeout_secs)` |
//!
//! Participation = order notional / (traded notional over the volume window scaled to the
//! horizon). Unknown volume counts as participation 0 (no slicing on missing data).

use serde::{Deserialize, Serialize};

/// `[exec]` in `risk.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExecPolicy {
    /// Off: every intent is `market` (the D-040 behaviour).
    pub enabled: bool,
    pub market_below_horizon_secs: f64,
    pub market_below_eur: f64,
    pub smart_limit_timeout_secs: f64,
    pub twap_participation: f64,
    pub pov_participation: f64,
    /// Participation rate a `pov` order targets.
    pub pov_rate: f64,
    pub sliced_min_horizon_secs: f64,
    pub max_slice_dur_secs: f64,
    /// Seconds per child slice of a TWAP / VWAP (slices = dur / this, clamped to [2, 20]).
    pub slice_secs: f64,
}

impl Default for ExecPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            market_below_horizon_secs: 60.0,
            market_below_eur: 250.0,
            smart_limit_timeout_secs: 30.0,
            twap_participation: 0.01,
            pov_participation: 0.05,
            pov_rate: 0.1,
            sliced_min_horizon_secs: 600.0,
            max_slice_dur_secs: 3_600.0,
            slice_secs: 30.0,
        }
    }
}

/// What the chooser knows about one order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExecInputs {
    pub notional_eur: f64,
    /// Shortest horizon among the contributing strategies (seconds; 0 = unknown).
    pub horizon_secs: f64,
    /// Traded notional over the volume window (EUR) and the window length (seconds).
    pub window_volume_eur: f64,
    pub window_secs: f64,
    /// Must execute now (halt / flatten / kill switch).
    pub urgent: bool,
    /// The instrument's intraday volume profile is known (VWAP possible).
    pub volume_profile: bool,
}

/// The chosen algorithm and the expected working time (seconds) the risk-engine keeps the
/// intent pending for.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecChoice {
    pub algo: String,
    pub working_secs: f64,
    pub reason: &'static str,
}

impl ExecChoice {
    fn market(reason: &'static str) -> Self {
        Self { algo: "market".into(), working_secs: 0.0, reason }
    }

    /// Algorithm name without parameters.
    pub fn name(&self) -> &str {
        self.algo.split(':').next().unwrap_or("market")
    }
}

/// Participation of the order in the volume expected over the horizon (0 when unknown).
pub fn participation(i: &ExecInputs) -> f64 {
    if i.window_volume_eur <= 0.0 || i.window_secs <= 0.0 || i.horizon_secs <= 0.0 {
        return 0.0;
    }
    let horizon_volume = i.window_volume_eur * i.horizon_secs / i.window_secs;
    if horizon_volume > 0.0 { i.notional_eur.abs() / horizon_volume } else { 0.0 }
}

fn fmt_secs(s: f64) -> String {
    format!("{}", s.round().max(1.0) as i64)
}

pub fn choose(p: &ExecPolicy, i: &ExecInputs) -> ExecChoice {
    if !p.enabled {
        return ExecChoice::market("exec policy disabled");
    }
    if i.urgent {
        return ExecChoice::market("urgent");
    }
    if i.horizon_secs > 0.0 && i.horizon_secs < p.market_below_horizon_secs {
        return ExecChoice::market("short horizon");
    }
    let part = participation(i);
    if part >= p.pov_participation && i.horizon_secs > 0.0 {
        let max_dur = i.horizon_secs.min(p.max_slice_dur_secs);
        return ExecChoice {
            algo: format!("pov:rate={},max_dur_s={}", p.pov_rate, fmt_secs(max_dur)),
            working_secs: max_dur,
            reason: "large vs horizon volume",
        };
    }
    if part >= p.twap_participation && i.horizon_secs >= p.sliced_min_horizon_secs {
        let dur = (i.horizon_secs / 2.0).min(p.max_slice_dur_secs);
        let slices = (dur / p.slice_secs.max(1.0)).round().clamp(2.0, 20.0) as u32;
        let name = if i.volume_profile { "vwap" } else { "twap" };
        return ExecChoice {
            algo: format!("{name}:dur_s={},slices={slices}", fmt_secs(dur)),
            working_secs: dur,
            reason: "sizeable vs per-interval volume",
        };
    }
    if i.notional_eur.abs() < p.market_below_eur {
        return ExecChoice::market("small order");
    }
    let timeout = if i.horizon_secs > 0.0 {
        (i.horizon_secs / 4.0).clamp(5.0, p.smart_limit_timeout_secs.max(5.0))
    } else {
        p.smart_limit_timeout_secs
    };
    ExecChoice {
        algo: format!("smart_limit:timeout_s={}", fmt_secs(timeout)),
        working_secs: timeout,
        reason: "default passive with timeout",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(notional: f64, horizon: f64, volume: f64) -> ExecInputs {
        ExecInputs {
            notional_eur: notional,
            horizon_secs: horizon,
            window_volume_eur: volume,
            window_secs: 3_600.0,
            urgent: false,
            volume_profile: false,
        }
    }

    #[test]
    fn urgency_and_short_horizons_cross_the_spread() {
        let p = ExecPolicy::default();
        assert_eq!(choose(&p, &ExecInputs { urgent: true, ..inputs(1e6, 86_400.0, 1e6) }).algo, "market");
        assert_eq!(choose(&p, &inputs(5_000.0, 30.0, 1e9)).algo, "market");
        assert_eq!(choose(&p, &inputs(100.0, 3_600.0, 1e9)).algo, "market", "small order");
        let off = ExecPolicy { enabled: false, ..p };
        assert_eq!(choose(&off, &inputs(1e6, 86_400.0, 1e6)).algo, "market");
    }

    #[test]
    fn size_against_volume_picks_the_algorithm() {
        let p = ExecPolicy::default();
        // 1 h horizon, 1 M EUR/h traded: 5 000 EUR = 0.5 % → smart limit with a 30 s timeout
        let c = choose(&p, &inputs(5_000.0, 3_600.0, 1e6));
        assert_eq!((c.algo.as_str(), c.working_secs), ("smart_limit:timeout_s=30", 30.0));
        assert_eq!(c.name(), "smart_limit");
        // 2 % → twap over half the horizon in 30 s slices (capped at 20)
        let c = choose(&p, &inputs(20_000.0, 3_600.0, 1e6));
        assert_eq!(c.algo, "twap:dur_s=1800,slices=20");
        let c = choose(&p, &ExecInputs { volume_profile: true, ..inputs(20_000.0, 1_200.0, 3e6) });
        assert_eq!(c.algo, "vwap:dur_s=600,slices=20");
        // ≥ 5 % → participate at 10 % for at most the horizon
        let c = choose(&p, &inputs(80_000.0, 3_600.0, 1e6));
        assert_eq!((c.algo.as_str(), c.working_secs), ("pov:rate=0.1,max_dur_s=3600", 3_600.0));
        // a short (< 10 min) horizon is never time-sliced; a large share of its volume still participates
        let c = choose(&p, &inputs(20_000.0, 120.0, 1e6));
        assert_eq!(c.algo, "pov:rate=0.1,max_dur_s=120");
        let c = choose(&p, &inputs(500.0, 120.0, 1e6));
        assert_eq!(c.algo, "smart_limit:timeout_s=30");
        let c = choose(&p, &inputs(2_000.0, 60.0, 1e9));
        assert_eq!(c.algo, "smart_limit:timeout_s=15");
        // unknown volume never slices
        assert_eq!(participation(&inputs(1e6, 3_600.0, 0.0)), 0.0);
        assert_eq!(choose(&p, &inputs(1e6, 3_600.0, 0.0)).name(), "smart_limit");
    }
}
