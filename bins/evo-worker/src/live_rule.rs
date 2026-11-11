//! The rebalance rule fitness replays trade with (spec 40 §6.1 step 6, D-079).
//!
//! `evo.eval.strategy`, `evo.eval.strategy.holdout` and `evo.eval.allocator` replay with the
//! live risk-engine's cost-aware rule, read from the same `risk.toml` the risk-engine loads
//! (`FA_RISK`, default `bins/risk-engine/risk.toml`; a missing or unreadable file falls back to
//! the copy compiled into this binary) and priced with the same `costs.yaml` (`FA_COSTS`), so
//! selection ranks candidates on the trades the live engine would send.
//!
//! Reproducing pre-D-079 trials: `FA_EVO_TRADE_RULE=fraction` switches the whole worker back
//! to the old rule (strategy replay: any move above 2 % of the sleeve; allocator replay: above
//! 0.1 % of capital), and an eval envelope may carry `"trade_rule": "fraction" | "live" | {…}`
//! (a serialised `fa_backtest::trade_rule::TradeRule`) for one job.

use fa_backtest::trade_rule::TradeRule;
use serde_json::Value;

/// `risk.toml` as shipped, the fallback when `FA_RISK` is not readable.
pub const BUILTIN_RISK_TOML: &str = include_str!("../../risk-engine/risk.toml");

/// The worker's rule for `mode` (`live` | `fraction`, `FA_EVO_TRADE_RULE`) and the risk
/// settings source at `path`; the second value names where it came from (logged, reported).
pub fn load(mode: &str, path: &str) -> (TradeRule, String) {
    if mode.eq_ignore_ascii_case("fraction") || mode.eq_ignore_ascii_case("legacy") {
        return (TradeRule::Fraction, "FA_EVO_TRADE_RULE=fraction".into());
    }
    if !mode.is_empty() && !mode.eq_ignore_ascii_case("live") {
        tracing::warn!(mode, "unknown FA_EVO_TRADE_RULE (live | fraction); using live");
    }
    match std::fs::read_to_string(path) {
        Ok(src) => match TradeRule::from_risk_toml(&src) {
            Ok(r) => return (r, path.to_owned()),
            Err(e) => tracing::warn!(path, error = %e, "risk settings unparsable; using the built-in risk.toml"),
        },
        Err(e) => tracing::warn!(path, error = %e, "risk settings not readable; using the built-in risk.toml"),
    }
    (
        TradeRule::from_risk_toml(BUILTIN_RISK_TOML).unwrap_or_else(|_| TradeRule::live_default()),
        "built-in risk.toml".into(),
    )
}

/// A per-job override from the eval envelope (`trade_rule`): `"fraction"`, `"live"` (= the
/// worker's rule, i.e. no override) or a serialised [`TradeRule`].
pub fn from_envelope(v: Option<&Value>) -> Option<TradeRule> {
    match v? {
        Value::String(s) if s.eq_ignore_ascii_case("fraction") || s.eq_ignore_ascii_case("legacy") => {
            Some(TradeRule::Fraction)
        }
        Value::String(_) => None,
        other => serde_json::from_value(other.clone()).ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_rule_follows_risk_toml_and_can_be_switched_back() {
        let (r, src) = load("live", "/nonexistent/risk.toml");
        assert_eq!((r, src.as_str()), (TradeRule::live_default(), "built-in risk.toml"));
        let (r, _) = load("", concat!(env!("CARGO_MANIFEST_DIR"), "/../risk-engine/risk.toml"));
        assert_eq!(r.name(), "cost_aware");
        assert_eq!(load("fraction", "x").0, TradeRule::Fraction);
        assert_eq!(from_envelope(Some(&serde_json::json!("fraction"))), Some(TradeRule::Fraction));
        assert_eq!(from_envelope(Some(&serde_json::json!("live"))), None);
        assert_eq!(from_envelope(None), None);
        let band = serde_json::json!({"rule": "band", "frac": 0.01, "min_eur": 50.0});
        assert_eq!(from_envelope(Some(&band)), Some(TradeRule::Band { frac: 0.01, min_eur: 50.0 }));
    }
}
