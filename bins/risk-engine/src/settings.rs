//! Risk-engine settings (`risk.toml`). Hard limits live in `fa_risk::RiskConfig` and can only
//! come from this file (spec 40 §6.5); nothing on the bus can loosen them.
use fa_core::AssetClass;
use fa_risk::RiskConfig;
use serde::{Deserialize, Serialize};

/// Starting paper cash (D-028).
pub const PAPER_CASH_EUR: f64 = 100_000.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Must be `paper`; anything else is refused at startup (D-029).
    pub mode: String,
    /// Account token in `order.*.<account>` subjects.
    pub account: String,
    pub starting_cash_eur: f64,
    /// `equal_weight` | `risk_parity` | `fractional_kelly` (spec 40 §6.2).
    pub allocator: String,
    pub max_strategy_weight: f64,
    pub kelly_fraction: f64,
    /// EWMA decay of the per-strategy performance posterior.
    pub performance_decay: f64,
    /// Signals older than this are dropped from allocation.
    pub signal_ttl_secs: u64,
    /// No-trade band as a fraction of the target (spec 40 `exec.no_trade_band`).
    pub no_trade_band_frac: f64,
    /// Absolute minimum order notional.
    pub min_order_eur: f64,
    /// Asset class assumed for instruments without reference data.
    pub default_asset_class: AssetClass,
    /// Window for the traded-volume estimate behind the liquidity cap.
    pub volume_window_secs: u64,
    /// Pending (unfilled) intents older than this stop counting as position.
    pub pending_ttl_secs: u64,
    /// Non-evolvable hard limits.
    pub limits: RiskConfig,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            mode: "paper".into(),
            account: "paper".into(),
            starting_cash_eur: PAPER_CASH_EUR,
            allocator: "equal_weight".into(),
            max_strategy_weight: 0.5,
            kelly_fraction: 0.25,
            performance_decay: 0.95,
            signal_ttl_secs: 60,
            no_trade_band_frac: 0.005,
            min_order_eur: 25.0,
            default_asset_class: AssetClass::Equity,
            volume_window_secs: 3_600,
            pending_ttl_secs: 10,
            limits: RiskConfig::default(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    #[error("parsing risk settings: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("mode `{0}` is not allowed: only `paper` exists (D-029)")]
    Mode(String),
    #[error("unknown allocator `{0}`")]
    Allocator(String),
}

impl Settings {
    pub fn parse(src: &str) -> Result<Self, SettingsError> {
        let s: Settings = toml::from_str(src)?;
        s.validate()?;
        Ok(s)
    }

    pub fn validate(&self) -> Result<(), SettingsError> {
        if self.mode != "paper" {
            return Err(SettingsError::Mode(self.mode.clone()));
        }
        if fa_allocator::by_name(
            &self.allocator,
            self.max_strategy_weight,
            self.kelly_fraction,
        )
        .is_none()
        {
            return Err(SettingsError::Allocator(self.allocator.clone()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_risk_toml_parses_and_live_is_refused() {
        let s = Settings::parse(include_str!("../risk.toml")).unwrap();
        assert_eq!(s.mode, "paper");
        assert_eq!(s.starting_cash_eur, PAPER_CASH_EUR);
        assert!(matches!(
            Settings::parse("mode = 'live'"),
            Err(SettingsError::Mode(_))
        ));
        assert!(matches!(
            Settings::parse("allocator = 'evolved'"),
            Err(SettingsError::Allocator(_))
        ));
    }
}
