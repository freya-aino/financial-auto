//! The rebalance rule a replay trades with (spec 40 §6.1 step 6, spec 41 §6.3).
//!
//! The live risk-engine sends an order only when the cost-aware rule of `fa_risk::rebalance`
//! says so (D-079): the D-041 no-trade band `max(min_order_eur, no_trade_band_frac · |target|)`
//! first, then min notional from the fee minimum, expected benefit ≥ cost, a confirmation
//! period and a per-strategy turnover budget. Fitness replays (`replay::replay` for
//! `evo.eval.strategy`, `allocator_replay::replay_allocator` for `evo.eval.allocator`) use
//! [`TradeRule::CostAware`] built from the same `risk.toml` ([`TradeRule::from_risk_toml`]) so
//! selection is scored on the trades the live engine would actually send.
//! [`TradeRule::Fraction`] (the pre-D-079 fitness rule: trade any move above
//! `min_trade_fraction` of the capital) stays for reproducing older trials.

use fa_risk::rebalance::RebalanceConfig;
use serde::{Deserialize, Serialize};

/// The live risk-engine's `performance_decay` default (`risk.toml`).
pub const LIVE_PERFORMANCE_DECAY: f64 = 0.95;

fn live_decay() -> f64 {
    LIVE_PERFORMANCE_DECAY
}

/// The rebalance rule of a replay.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "rule", rename_all = "snake_case")]
pub enum TradeRule {
    /// Trade any target move above `min_trade_fraction` of the capital (strategy replay: of the
    /// sleeve) — the fitness rule before D-079, kept to reproduce old trials.
    #[default]
    Fraction,
    /// The risk-engine's D-041 no-trade band: `max(min_eur, frac · |target|)`.
    Band { frac: f64, min_eur: f64 },
    /// The live cost-aware rule (`fa_risk::rebalance`, the same `Rebalancer` the risk-engine
    /// runs) on top of the band. `decay` is the risk-engine's `performance_decay` (the EWMA
    /// behind a strategy's tracked edge and its effective observation count); the allocator
    /// replay uses the genome's own tracker instead.
    CostAware {
        frac: f64,
        min_eur: f64,
        cfg: RebalanceConfig,
        #[serde(default = "live_decay")]
        decay: f64,
    },
}

/// The subset of `risk.toml` behind the live rebalance rule.
#[derive(Debug, Clone, Deserialize)]
struct RiskRebalanceToml {
    #[serde(default = "default_band")]
    no_trade_band_frac: f64,
    #[serde(default = "default_min_order")]
    min_order_eur: f64,
    #[serde(default = "live_decay")]
    performance_decay: f64,
    #[serde(default)]
    rebalance: RebalanceConfig,
}

fn default_band() -> f64 {
    0.005
}

fn default_min_order() -> f64 {
    25.0
}

impl TradeRule {
    /// The live rule with the risk-engine's defaults (`risk.toml` as shipped).
    pub fn live_default() -> Self {
        Self::CostAware {
            frac: default_band(),
            min_eur: default_min_order(),
            cfg: RebalanceConfig::default(),
            decay: LIVE_PERFORMANCE_DECAY,
        }
    }

    /// The live rule from a `risk.toml` source: top-level `no_trade_band_frac`,
    /// `min_order_eur`, `performance_decay` and the `[rebalance]` table (missing keys take
    /// the risk-engine's defaults; every other key is ignored). `[rebalance] enabled = false`
    /// gives the band alone, as the live engine does.
    pub fn from_risk_toml(src: &str) -> Result<Self, String> {
        let t: RiskRebalanceToml = toml::from_str(src).map_err(|e| e.to_string())?;
        if !t.rebalance.enabled {
            return Ok(Self::Band { frac: t.no_trade_band_frac, min_eur: t.min_order_eur });
        }
        Ok(Self::CostAware {
            frac: t.no_trade_band_frac,
            min_eur: t.min_order_eur,
            cfg: t.rebalance,
            decay: t.performance_decay,
        })
    }

    /// Short name for reports (`fraction | band | cost_aware`).
    pub fn name(&self) -> &'static str {
        match self {
            Self::Fraction => "fraction",
            Self::Band { .. } => "band",
            Self::CostAware { .. } => "cost_aware",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn risk_toml_gives_the_live_rule() {
        let r = TradeRule::from_risk_toml(include_str!("../../../bins/risk-engine/risk.toml")).unwrap();
        let TradeRule::CostAware { frac, min_eur, cfg, decay } = &r else { panic!("{r:?}") };
        assert_eq!((*frac, *min_eur, *decay), (0.005, 25.0, 0.95));
        assert_eq!(cfg, &RebalanceConfig::default(), "risk.toml repeats the RebalanceConfig defaults");
        assert_eq!(r, TradeRule::live_default());
        // overrides and the disabled form
        let r = TradeRule::from_risk_toml("min_order_eur = 50\n[rebalance]\nconfirm_secs = 0\n").unwrap();
        let TradeRule::CostAware { min_eur, cfg, .. } = r else { panic!() };
        assert_eq!((min_eur, cfg.confirm_secs, cfg.max_fee_bps), (50.0, 0.0, 10.0));
        let r = TradeRule::from_risk_toml("[rebalance]\nenabled = false\n").unwrap();
        assert_eq!(r, TradeRule::Band { frac: 0.005, min_eur: 25.0 });
        assert!(TradeRule::from_risk_toml("min_order_eur = \"x\"").is_err());
        // serde round trip; an older JSON form without `decay` still decodes
        let j = serde_json::to_string(&TradeRule::live_default()).unwrap();
        assert_eq!(serde_json::from_str::<TradeRule>(&j).unwrap(), TradeRule::live_default());
        let old = r#"{"rule":"cost_aware","frac":0.005,"min_eur":25.0,"cfg":{}}"#;
        assert_eq!(serde_json::from_str::<TradeRule>(old).unwrap(), TradeRule::live_default());
    }
}
