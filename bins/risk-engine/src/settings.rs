//! Risk-engine settings (`risk.toml`). Hard limits live in `fa_risk::RiskConfig` and can only
//! come from this file (spec 40 §6.5); nothing on the bus can loosen them.
use fa_core::AssetClass;
use fa_risk::{RiskConfig, strategy::PromotionGate};
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
    /// `ledger.position_snapshot` interval when Postgres is configured (also on every halt).
    pub snapshot_secs: u64,
    /// In-memory book vs Postgres-derived book comparison interval (`reconciliation_run`).
    pub reconcile_secs: u64,
    /// Reference-data reload interval (`ref.instrument` / `ref.listing` / `ref.venue` →
    /// asset class, venue, quote currency, status, source per instrument); also reloaded on
    /// `ref.changed.*` and `ingest.unresolved.*` notices.
    pub refdata_refresh_secs: u64,
    /// EUR accounting FX (spec 42 §6, spec 12 §6).
    pub fx: FxSettings,
    /// Status given to a strategy the first time one of its signals is seen: `active` (allocated
    /// paper capital at once, the integrated-baseline behaviour) or `shadow` (virtual book only,
    /// needs `promote` through the gate).
    pub new_strategy_status: String,
    /// Virtual capital a shadow strategy's signals are tracked on (spec 41 §6.6).
    pub shadow_sleeve_eur: f64,
    /// Promotion gate thresholds (spec 40 §6.6, §7 `promotion.*`).
    pub promotion: PromotionGate,
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
            snapshot_secs: 60,
            reconcile_secs: 300,
            refdata_refresh_secs: 300,
            fx: FxSettings::default(),
            new_strategy_status: "active".into(),
            shadow_sleeve_eur: 10_000.0,
            promotion: PromotionGate::default(),
            limits: RiskConfig::default(),
        }
    }
}

/// `[fx]`: the ECB daily reference rate is the accounting conversion (spec 12 §7
/// `fx.accounting_source = ecb_daily`); marks and fills convert as of the event time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FxSettings {
    /// How often the ECB `eurofxref-daily.xml` is fetched and upserted into `ref.fx_rate`.
    pub ecb_refresh_secs: u64,
    /// A rate older than this at the event time is `stale` (spec 42 §8: last known rate, flagged).
    pub max_age_hours: f64,
    /// Raise the reduce-only halt scope `fx` while any used currency is stale (spec 42 §8
    /// "block risk-increasing orders in that currency after threshold"; off = warning only).
    pub halt_on_stale: bool,
    /// `ref.fx_rate.source_id` the engine reads (and the ECB task writes).
    pub source: String,
    /// Days of history loaded into the in-memory as-of table at start.
    pub history_days: u32,
}

impl Default for FxSettings {
    fn default() -> Self {
        Self {
            ecb_refresh_secs: 86_400,
            max_age_hours: 72.0,
            halt_on_stale: false,
            source: fa_warehouse::fx::ECB_SOURCE.into(),
            history_days: 60,
        }
    }
}

impl FxSettings {
    pub fn max_age(&self) -> chrono::Duration {
        chrono::Duration::milliseconds((self.max_age_hours.max(0.0) * 3_600_000.0) as i64)
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
    #[error("new_strategy_status `{0}` must be active or shadow")]
    NewStatus(String),
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
        if fa_allocator::by_name(&self.allocator, self.max_strategy_weight, self.kelly_fraction).is_none() {
            return Err(SettingsError::Allocator(self.allocator.clone()));
        }
        if !matches!(self.new_strategy_status.as_str(), "active" | "shadow") {
            return Err(SettingsError::NewStatus(self.new_strategy_status.clone()));
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
        assert_eq!((s.snapshot_secs, s.reconcile_secs, s.refdata_refresh_secs), (60, 300, 300));
        assert_eq!((s.fx.ecb_refresh_secs, s.fx.max_age_hours, s.fx.halt_on_stale), (86_400, 72.0, false));
        assert_eq!(s.fx.source, "ecb_daily");
        assert_eq!(s.limits.provisional_position_frac, None);
        assert_eq!(s.limits.provisional_position_frac(), 0.025);
        assert!(matches!(Settings::parse("mode = 'live'"), Err(SettingsError::Mode(_))));
        assert!(matches!(Settings::parse("allocator = 'evolved'"), Err(SettingsError::Allocator(_))));
        assert!(matches!(Settings::parse("new_strategy_status = 'live'"), Err(SettingsError::NewStatus(_))));
        assert_eq!(s.new_strategy_status, "active");
        assert_eq!((s.promotion.min_shadow_days, s.promotion.min_horizons), (30.0, 20));
        assert_eq!(
            (s.limits.strategy_max_drawdown, s.limits.max_class_frac, s.limits.max_venue_frac),
            (0.05, 0.40, 0.50)
        );
        assert_eq!(s.limits.portfolio_gross_cap, 2.0);
    }
}
