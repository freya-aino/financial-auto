//! Risk-engine settings (`risk.toml`). Hard limits live in `fa_risk::RiskConfig` and can only
//! come from this file (spec 40 §6.5); nothing on the bus can loosen them.
use std::collections::BTreeMap;

use fa_backtest::holdout::HoldoutPolicy;
use fa_core::{AssetClass, Timescale};
use fa_risk::{
    RiskConfig, exec::ExecPolicy, health::FeedGuardConfig, pipeline::PipelinePolicy, rebalance::RebalanceConfig,
    strategy::PromotionGate,
};
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
    /// Automatic evolved-strategy lifecycle (spec 35 §6.4).
    pub pipeline: PipelinePolicy,
    /// Automatic holdout locking (spec 41 §6.4).
    pub holdout: HoldoutSettings,
    /// Per-instrument stale-feed freeze and bad-tick filter (spec 40 §6.5, §8).
    pub feed: FeedGuardConfig,
    /// Cost charged on weight turnover in the live allocator tournament (bps).
    pub tournament_cost_bps: f64,
    /// Execution-algorithm choice per intent (spec 40 §6.4).
    pub exec: ExecPolicy,
    /// Cost-aware rebalancing (spec 40 §6.1 step 6): benefit vs full expected cost, min order
    /// notional from the fee minimum, per-strategy daily turnover budget, confirmation batching.
    pub rebalance: RebalanceConfig,
    /// Snapshot-scoped model-drift alarm (spec 40 §6.5).
    pub model_drift: ModelDriftSettings,
    /// Daily payout quota rule (spec 42 §10 Q1, `fa_ledger::payout`); default reserves nothing.
    pub payout: fa_ledger::payout::PayoutRule,
    /// Risk-chain history (`ledger.allocation_snapshot` / `portfolio_target` / `promotion_review`,
    /// spec 40 §5).
    pub history: crate::engine::history::HistorySettings,
}

/// `[model_drift]`: the risk-engine raises `model:<snapshot>` when a served snapshot's region
/// occupancy drifts far beyond the adapter-refit trigger (spec 33: KL > 0.1 × 3 requests a
/// refit); only the strategies of that snapshot's timescale are paused.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelDriftSettings {
    pub enabled: bool,
    /// `drift_kl` at or above this for `sustain` consecutive kernel updates raises the halt.
    pub halt_kl: f64,
    pub sustain: u32,
    /// Every expert at or below this for `clear_sustain` updates clears a halt this path raised.
    pub clear_kl: f64,
    pub clear_sustain: u32,
}

impl Default for ModelDriftSettings {
    fn default() -> Self {
        Self { enabled: true, halt_kl: 1.0, sustain: 5, clear_kl: 0.25, clear_sustain: 10 }
    }
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
            pipeline: PipelinePolicy::default(),
            holdout: HoldoutSettings::default(),
            feed: FeedGuardConfig::default(),
            tournament_cost_bps: 5.0,
            exec: ExecPolicy::default(),
            rebalance: RebalanceConfig::default(),
            model_drift: ModelDriftSettings::default(),
            payout: fa_ledger::payout::PayoutRule::default(),
            history: crate::engine::history::HistorySettings::default(),
        }
    }
}

/// `[holdout]`: keeps a locked holdout per timescale in `validation.holdout` (spec 41 §6.4). At
/// start and every `check_secs` the risk-engine plans the lock per timescale
/// (`fa_backtest::holdout::plan_lock`), checks the segment holds `min_bars` bars in
/// VictoriaMetrics and inserts it idempotently; a timescale that cannot be locked yet is
/// reported (`holdout.missing.<ts>` gauge, `ctl.risk.pipeline` status, a warning), never
/// silently empty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HoldoutSettings {
    pub auto_lock: bool,
    pub check_secs: u64,
    /// Timescales that must have a holdout (`1s`, `1m`, `1h`, `1d`); candidates of other
    /// timescales are locked on demand.
    pub timescales: Vec<String>,
    /// `holdout.length.<ts>` in seconds (defaults: `fa_backtest::holdout::HoldoutPolicy::default_for`).
    pub length_secs: BTreeMap<String, u64>,
    /// `embargo.<ts>` in seconds: recorded with every auto-locked row
    /// (`validation.holdout.embargo_ms`); evo-worker keeps now-relative selection windows
    /// `max(this, 2 × strategy horizon, 60 s)` before the holdout
    /// (`fa_backtest::holdout::selection_embargo_ms`).
    pub embargo_secs: BTreeMap<String, u64>,
    pub min_bars: usize,
    /// Re-lock a new segment once the latest holdout ended this many days ago (0 = never).
    pub relock_after_days: f64,
}

impl Default for HoldoutSettings {
    fn default() -> Self {
        Self {
            auto_lock: true,
            check_secs: 300,
            timescales: vec!["1s".into(), "1m".into()],
            length_secs: BTreeMap::new(),
            embargo_secs: BTreeMap::new(),
            min_bars: 300,
            relock_after_days: 0.0,
        }
    }
}

impl HoldoutSettings {
    pub fn policy(&self, ts: Timescale) -> HoldoutPolicy {
        let d = HoldoutPolicy::default_for(ts);
        HoldoutPolicy {
            length_ms: self.length_secs.get(ts.as_str()).map_or(d.length_ms, |s| *s as i64 * 1000),
            embargo_ms: self.embargo_secs.get(ts.as_str()).map_or(d.embargo_ms, |s| *s as i64 * 1000),
            min_bars: self.min_bars,
            relock_after_ms: (self.relock_after_days > 0.0).then_some((self.relock_after_days * 86_400_000.0) as i64),
        }
    }

    pub fn timescales(&self) -> Vec<Timescale> {
        self.timescales.iter().filter_map(|t| t.parse().ok()).collect()
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
    /// Block risk-increasing orders in a currency whose rate is stale (spec 42 §8): targets and
    /// foreign orders on instruments quoted in it are reduce-only; other currencies trade on.
    pub block_stale_ccy: bool,
    /// Intraday EUR rates from the trades of EUR FX pairs in reference data (Yahoo `EURUSD=X`;
    /// spec 42 §6 `fx.intraday_source`), written to `ref.fx_rate` under the feed's source id.
    pub intraday: bool,
    /// An intraday rate deviating more than this from the latest daily rate is dropped (bad tick).
    pub intraday_max_dev: f64,
}

impl Default for FxSettings {
    fn default() -> Self {
        Self {
            ecb_refresh_secs: 86_400,
            max_age_hours: 72.0,
            halt_on_stale: false,
            source: fa_warehouse::fx::ECB_SOURCE.into(),
            history_days: 60,
            block_stale_ccy: true,
            intraday: true,
            intraday_max_dev: 0.05,
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
    #[error("{0}")]
    Payout(String),
    #[error("holdout timescale `{0}` is not one of 1s | 1m | 1h | 1d | 1w")]
    Timescale(String),
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
        if let Some(t) = self.holdout.timescales.iter().find(|t| t.parse::<Timescale>().is_err()) {
            return Err(SettingsError::Timescale(t.clone()));
        }
        self.payout.validate().map_err(SettingsError::Payout)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holdout_embargo_travels_with_the_lock_and_fitness_uses_the_live_rule() {
        use fa_backtest::{
            holdout::{LockPlan, plan_lock, selection_embargo_ms},
            replay::Holdout,
            trade_rule::TradeRule,
        };
        let s = Settings::parse(include_str!("../risk.toml")).unwrap();
        let now = 1_000 * 3_600_000;
        for (ts, want) in [(Timescale::Second, 300_000), (Timescale::Minute, 3_600_000)] {
            let LockPlan::Lock { start_ms, end_ms, embargo_ms, .. } = plan_lock(ts, &s.holdout.policy(ts), now, &[])
            else {
                panic!("first lock");
            };
            assert_eq!(embargo_ms, want, "[holdout.embargo_secs] {ts} reaches the lock record");
            // what evo-worker reads back from the row: the configured floor, or 2 × horizon
            let h = Holdout { timescale: ts, start_ms, end_ms, locked_at_ms: now, embargo_ms: Some(embargo_ms) };
            assert_eq!(selection_embargo_ms(&h, 1_000), want);
            assert_eq!(selection_embargo_ms(&h, want), 2 * want);
        }
        // the rule evo-worker's fitness replays parse from the same file is the engine's
        assert_eq!(
            TradeRule::from_risk_toml(include_str!("../risk.toml")).unwrap(),
            TradeRule::CostAware {
                frac: s.no_trade_band_frac,
                min_eur: s.min_order_eur,
                cfg: s.rebalance.clone(),
                decay: s.performance_decay,
            }
        );
    }

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
        assert_eq!(s.payout, fa_ledger::payout::PayoutRule::default(), "no payout reserved by default");
        assert!(matches!(Settings::parse("[payout]\npct = 1.5"), Err(SettingsError::Payout(_))));
        assert_eq!(s.new_strategy_status, "active");
        assert_eq!((s.promotion.min_shadow_days, s.promotion.min_horizons), (30.0, 20));
        assert_eq!(
            (s.limits.strategy_max_drawdown, s.limits.max_class_frac, s.limits.max_venue_frac),
            (0.05, 0.40, 0.50)
        );
        assert_eq!(s.limits.portfolio_gross_cap, 2.0);
        assert_eq!((s.limits.max_order_frac, s.limits.limit_price_band), (0.10, 0.10));
        assert_eq!((s.promotion.max_pbo, s.promotion.consistency_z), (0.2, 2.0));
        assert!(s.pipeline.enabled && s.pipeline.auto_promote);
        assert_eq!((s.pipeline.dsr_min_candidate, s.pipeline.pbo_max), (0.95, 0.2));
        assert_eq!(s.holdout.timescales(), vec![Timescale::Second, Timescale::Minute]);
        assert_eq!(s.holdout.policy(Timescale::Second).length_ms, 2 * 3_600_000);
        assert_eq!(s.holdout.policy(Timescale::Minute).embargo_ms, 3_600_000);
        assert_eq!(s.holdout.policy(Timescale::Second).relock_after_ms, Some(fa_backtest::holdout::RELOCK_AFTER_MS));
        assert_eq!((s.feed.stale_freeze_bars, s.feed.stale_flatten_secs, s.feed.jump_max), (3.0, 0.0, 0.15));
        assert!(matches!(Settings::parse("[holdout]\ntimescales = ['5s']"), Err(SettingsError::Timescale(_))));
        assert!(Settings::parse("allocator = 'thompson_rp'").is_ok());
        assert_eq!(s.limits.short_margin, fa_risk::default_short_margin(), "risk.toml repeats the default schedule");
        assert_eq!(s.limits.short_margin_budget_frac, 0.5);
        assert_eq!(s.exec, fa_risk::exec::ExecPolicy::default());
        assert_eq!(s.model_drift, ModelDriftSettings::default());
    }
}
