//! Pure risk/allocation engine (no I/O): signals, trades, order events and halts in;
//! intents, approvals, rejections, halts and risk state out. Deterministic given the inputs
//! (intent ids are the only non-deterministic field).
//!
//! Decision tick (spec 40 §6.1): latest signal per **active** strategy → allocator weights →
//! EUR targets per instrument (`fa_allocator::net_targets`) → hard caps (`fa_risk`) → halt gate
//! (flatten / reduce-only) → diff against the paper book plus pending intents → `OrderIntent`s.
//!
//! Strategy lifecycle (D-051): every strategy seen on `signal.exposure.*` has a
//! [`StrategyState`] (`active|paused|flattening|shadow`). Paused/flattening/shadow strategies
//! contribute no exposure. Active strategies carry a virtual book built from fill attributions
//! (even split across the intent's contributors, as persisted by D-047) whose mark-to-market
//! drawdown against the allocated sleeve triggers the per-strategy kill switch (spec 40 §6.5).
//! Shadow strategies carry a virtual book of their own signals on `shadow_sleeve_eur` and count
//! completed horizons toward the promotion gate (spec 40 §6.6).
//!
//! Reference data and FX (D-061): a [`RefTable`] (asset class, venue, quote currency, status,
//! source per instrument) drives the per-class / per-venue caps and the stricter cap for
//! `provisional` instruments; unknown instruments fall back to `default_asset_class` and are
//! counted. Marks and fills are converted from the instrument's quote currency to EUR as of the
//! event time through an [`FxTable`] (ECB daily reference rates); a rate older than
//! `fx.max_age_hours` is flagged stale and, with `fx.halt_on_stale`, raises the reduce-only
//! halt scope `fx`. Sub-books are serialisable ([`Engine::strategy_rows`]) and restored at
//! start, so the per-strategy drawdown tracker survives restarts.
//!
//! Since the strategy pipeline (spec 35 §6.4): evolved strategies carry a
//! [`fa_risk::pipeline::PipelineRecord`] and move through `candidate → holdout_passed → shadow
//! → active → retired` by [`Engine::apply_decision`] (decided by `fa_risk::pipeline::decide`,
//! driven by `crate::pipeline`); the gate's PBO and shadow-vs-backtest items are computed
//! ([`Engine::checklist`]). Hard-limit additions: per-instrument feed guard (stale freeze,
//! bad-tick filter, `fa_risk::health`), pre-trade checks for foreign/manual intents
//! (`fa_risk::pretrade`), allocator NaN fallback, a live paper allocator tournament
//! (`fa_allocator::tournament`), limit events ([`Out::RiskEvent`] → `fa.v1.RiskEvent`), hourly equity
//! and mark history for the limit calibration rules (`fa_risk::calibrate`), and a persistable
//! [`RiskRecord`] so the kill switch, halts and drawdown peak survive a restart.
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use fa_allocator::{Allocator, PerformanceTracker, tournament::Tournament};
use fa_core::{Horizon, InstrumentId, Timescale};
use fa_ledger::{Fill, Ledger, Position, pg::FillDetails};
use fa_proto::{
    Timestamp,
    v1::{
        AssetExposure, ExposureDistribution as ProtoDist, Halt, OrderEvent, OrderIntent, RiskState, Side, TradingMode,
        order_event,
    },
};
use fa_risk::{
    HaltReason, HaltScope, MarketInfo, RiskConfig, RiskEngine,
    calibrate::{Calibration, CalibrationRules},
    health::{FeedGuard, FeedState, TradeVerdict},
    pipeline::{Decision, FrontMember},
    pretrade::{OrderCheck, check_order},
    strategy::{Checklist, GateEvidence, StrategyAction, StrategyState, StrategyStatus},
};
use fa_strategies::ExposureDistribution;
use fa_warehouse::{
    fx::{FxLookup, FxTable},
    venues::RefTable,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::settings::Settings;

const MAX_INTENTS: usize = 5_000;
const MS_PER_DAY: i64 = 86_400_000;
const MS_PER_HOUR: i64 = 3_600_000;
/// Hourly P&L points kept per strategy sub-book (60 days).
const MAX_HISTORY: usize = 24 * 60;
/// Hourly equity / mark points kept for the calibration rules (30 days).
const MAX_CALIBRATION_POINTS: usize = 24 * 30;
/// Same risk event (kind, scope) at most once per this interval.
const EVENT_INTERVAL_MS: i64 = 60_000;

/// Something the engine wants published.
#[derive(Debug, Clone, PartialEq)]
pub enum Out {
    /// `order.intent.<account>`
    Intent(OrderIntent),
    /// `order.approved.<account>`
    Approved(OrderIntent),
    /// `order.event.<account>` — risk rejection (spec 40: `risk.event` has no proto yet)
    Rejected { account: String, event: OrderEvent },
    /// `risk.halt.<scope>`
    Halt(Halt),
    /// `risk.resume.<scope>` — a halt the engine raised itself has cleared (stale FX fresh again).
    Resume(Halt),
    /// An automatic, consequential state change for `audit.log` (per-strategy drawdown pause,
    /// flatten completion); the I/O layer persists it.
    Audit { action: String, subject: String, payload: Value },
    /// A limit event (spec 40 §4.2 `risk.event`): limit clips, pre-trade rejections, feed
    /// freezes / bad ticks, allocator fallbacks. Published as `fa.v1.RiskEvent{LIMIT_BREACH}` on
    /// `risk.event.<account>` and appended to `audit.log` (never deleted, spec 40 §5).
    RiskEvent { kind: String, scope: String, detail: Value },
}

#[derive(Debug, Clone)]
struct IntentInfo {
    instrument: InstrumentId,
    signed_quantity: f64,
    ours: bool,
    strategy_ids: Vec<String>,
}

/// A fill booked into the paper ledger, queued for persistence (drained by the I/O layer).
#[derive(Debug, Clone, PartialEq)]
pub struct BookedFill {
    pub fill: Fill,
    pub strategy_ids: Vec<String>,
    pub at_ms: i64,
    /// Currency, FX source, arrival price and slippage for `ledger.fill`.
    pub details: FillDetails,
}

#[derive(Debug, Clone)]
struct Pending {
    instrument: InstrumentId,
    signed_eur: f64,
    created_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ActiveHalt {
    reason: String,
    since_ms: i64,
    manual: bool,
}

/// Everything about the hard-limit state that must survive a restart (spec 40 §8: "kill switch
/// triggered → persistent until manual re-arm (survives restart)"): the halt state machine
/// (peak equity, day-start equity, every active halt) plus the halt details, the UTC day, the
/// hourly equity history and the tournament records. Persisted in `ledger.risk_state`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskRecord {
    pub v: u32,
    risk: RiskEngine,
    halts: BTreeMap<String, ActiveHalt>,
    source_halts: BTreeMap<String, ActiveHalt>,
    day: Option<i64>,
    day_start_equity: f64,
    #[serde(default)]
    equity_history: VecDeque<(i64, f64)>,
    #[serde(default)]
    tournament: BTreeMap<String, fa_allocator::tournament::Record>,
}

impl RiskRecord {
    /// Active halt scopes (for logs).
    pub fn halt_scopes(&self) -> Vec<String> {
        self.halts.keys().cloned().chain(self.source_halts.keys().map(|s| format!("source:{s}"))).collect()
    }
}

/// External gate evidence the I/O layer reads from the validation registry.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Evidence {
    /// Holdout deflated Sharpe (evolved) or the registry's DSR of the strategy (fallback).
    pub deflated_sharpe: Option<f64>,
    /// PBO of the strategy's generation.
    pub pbo: Option<f64>,
}

/// Attributed virtual sub-book of an active strategy (spec 40 §6.3): fills are split evenly
/// across the intent's contributors (D-047 attribution), marked to market each tick. Positions
/// carry average cost and realised P&L (same arithmetic as the paper book). Persisted as JSON
/// in `ledger.strategy_state.book` and restored at start (D-061).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct VirtualBook {
    cash_eur: f64,
    positions: BTreeMap<InstrumentId, Position>,
    /// Marked value at the previous tick, for the P&L increment.
    last_value_eur: f64,
    /// Fees attributed so far.
    fees_eur: f64,
    /// Realised slippage of attributed fills: Σ slippage_bps and count.
    slippage_sum_bps: f64,
    slippage_n: u64,
    /// Hourly `(ms, pnl_eur)` points of the strategy's tracked P&L, newest last, bounded.
    history: VecDeque<(i64, f64)>,
    /// `last_value_eur` is valid against the current marks. A book restored from Postgres is not
    /// primed until every held instrument has a mark again, so the restart never books a
    /// spurious P&L jump between the persisted value and a cost-marked one.
    #[serde(skip)]
    primed: bool,
}

impl VirtualBook {
    /// Marked value: cash plus positions at the marks (average cost where a mark is missing, so a
    /// restart without marks yet does not read as a loss).
    fn value(&self, marks: &BTreeMap<InstrumentId, f64>) -> f64 {
        self.cash_eur
            + self
                .positions
                .iter()
                .map(|(i, p)| p.quantity * marks.get(i).copied().unwrap_or(p.avg_cost_eur))
                .sum::<f64>()
    }

    fn realised_eur(&self) -> f64 {
        self.positions.values().map(|p| p.realised_eur).sum()
    }

    /// Applies an attributed share of a fill with the paper book's arithmetic.
    fn apply(&mut self, share: &Fill) {
        let mut l = Ledger::with_cash(self.cash_eur);
        l.positions = std::mem::take(&mut self.positions);
        l.apply_fill(share);
        self.cash_eur = l.cash_eur;
        self.fees_eur += share.fee_eur;
        self.positions = l.positions;
        self.positions.retain(|_, p| p.quantity != 0.0 || p.realised_eur != 0.0);
    }

    /// Closes every position at the marks (the strategy leaves the allocated set).
    fn close(&mut self, marks: &BTreeMap<InstrumentId, f64>) {
        self.cash_eur = self.value(marks);
        self.positions.clear();
        self.last_value_eur = self.cash_eur;
    }

    fn slippage_bps_mean(&self) -> Option<f64> {
        (self.slippage_n > 0).then(|| self.slippage_sum_bps / self.slippage_n as f64)
    }

    fn push_history(&mut self, now_ms: i64, pnl: f64) {
        let hour = now_ms.div_euclid(MS_PER_HOUR);
        match self.history.back_mut() {
            Some((t, v)) if t.div_euclid(MS_PER_HOUR) == hour => *v = pnl,
            _ => {
                self.history.push_back((now_ms, pnl));
                while self.history.len() > MAX_HISTORY {
                    self.history.pop_front();
                }
            }
        }
    }
}

/// Everything persisted next to a `ledger.strategy_state` row (`book` JSONB).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SubBookRecord {
    pub v: u32,
    /// `active` (attributed fills) or `shadow` (virtual exposure).
    pub kind: String,
    #[serde(flatten)]
    book: VirtualBook,
    /// Shadow strategies: signed EUR exposure per instrument held virtually since the last tick.
    pub shadow_exposure: BTreeMap<InstrumentId, f64>,
    pub horizon_anchor_ms: Option<i64>,
}

/// FX status for the state views.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FxStatus {
    pub rates: usize,
    pub currencies: Vec<String>,
    pub stale: Vec<String>,
    pub halted: bool,
    pub max_age_hours: f64,
}

/// Bare `source` halts (no qualifier) are keyed by this in the per-source table.
const ANY_SOURCE: &str = "*";

pub struct Engine {
    settings: Settings,
    allocator: Box<dyn Allocator>,
    /// Name + parameters of the active allocator (`risk.toml` or the last runtime switch).
    allocator_spec: Value,
    risk: RiskEngine,
    ledger: Ledger,
    perf: PerformanceTracker,
    marks: BTreeMap<InstrumentId, f64>,
    /// Marks at the previous tick, for realised strategy returns.
    tick_marks: BTreeMap<InstrumentId, f64>,
    /// (event ms, EUR notional) per instrument within the volume window.
    volume: BTreeMap<InstrumentId, VecDeque<(i64, f64)>>,
    first_seen_ms: BTreeMap<InstrumentId, i64>,
    signals: BTreeMap<String, (ExposureDistribution, i64)>,
    intents: HashMap<String, IntentInfo>,
    intent_order: VecDeque<String>,
    pending: BTreeMap<String, Pending>,
    halts: BTreeMap<HaltScope, ActiveHalt>,
    /// Health halts per source id (`source:<id>`); `HaltScope::Source` is active while non-empty.
    source_halts: BTreeMap<String, ActiveHalt>,
    booked: Vec<BookedFill>,
    day: Option<i64>,
    day_start_equity: f64,
    // strategy lifecycle (D-051)
    strategies: BTreeMap<String, StrategyState>,
    books: BTreeMap<String, VirtualBook>,
    /// Shadow strategies: signed EUR exposure per instrument held virtually since the last tick.
    shadow_exposure: BTreeMap<String, BTreeMap<InstrumentId, f64>>,
    /// Signal `as_of` the horizon counter last advanced from, per shadow strategy.
    horizon_anchor_ms: BTreeMap<String, i64>,
    last_alloc: BTreeMap<String, f64>,
    dirty: BTreeSet<String>,
    // reference data and FX (D-061)
    refs: RefTable,
    /// Instruments seen in signals without reference data (gauge `refdata.unknown`).
    unknown_refs: BTreeSet<InstrumentId>,
    fx: FxTable,
    /// Currencies whose latest usable rate was stale at the last conversion.
    fx_stale: BTreeSet<String>,
    /// Latest venue → gross EUR exposure computed by the tick (state views).
    exposure_by_venue: BTreeMap<String, f64>,
    exposure_by_class: BTreeMap<String, f64>,
    // hard-limit additions (feed guard, tournament, events, calibration history)
    feed: FeedGuard,
    tournament: Tournament,
    event_last: BTreeMap<(String, String), i64>,
    equity_history: VecDeque<(i64, f64)>,
    hourly_marks: BTreeMap<InstrumentId, VecDeque<f64>>,
    risk_dirty: bool,
    pub counters: BTreeMap<&'static str, f64>,
}

fn parse_horizon(s: &str) -> Horizon {
    let fallback = Horizon { timescale: Timescale::Second, steps: 1 };
    let Some((steps, ts)) = s.split_once('x') else {
        return fallback;
    };
    match (steps.parse::<u32>(), ts.parse::<Timescale>()) {
        (Ok(steps), Ok(timescale)) => Horizon { timescale, steps: steps.max(1) },
        _ => fallback,
    }
}

fn ms_of(t: Option<&Timestamp>) -> Option<i64> {
    t.map(|t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000)
}

fn ts_of(ms: i64) -> Timestamp {
    Timestamp { seconds: ms.div_euclid(1000), nanos: (ms.rem_euclid(1000) * 1_000_000) as i32 }
}

impl Engine {
    pub fn new(settings: Settings) -> Self {
        settings.validate().expect("settings validated by the caller");
        let allocator =
            fa_allocator::by_name(&settings.allocator, settings.max_strategy_weight, settings.kelly_fraction)
                .expect("validated");
        let cash = settings.starting_cash_eur;
        let (settings_feed, settings_cap, settings_kelly, settings_cost_bps) = (
            settings.feed.clone(),
            settings.max_strategy_weight,
            settings.kelly_fraction,
            settings.tournament_cost_bps,
        );
        let allocator_spec = json!({
            "name": settings.allocator,
            "params": {"cap": settings.max_strategy_weight, "kelly": settings.kelly_fraction},
            "source": "risk.toml",
        });
        Self {
            allocator_spec,
            risk: RiskEngine::new(settings.limits.clone(), cash),
            ledger: Ledger::with_cash(cash),
            perf: PerformanceTracker::new(settings.performance_decay, 0.01),
            allocator,
            settings,
            marks: BTreeMap::new(),
            tick_marks: BTreeMap::new(),
            volume: BTreeMap::new(),
            first_seen_ms: BTreeMap::new(),
            signals: BTreeMap::new(),
            intents: HashMap::new(),
            intent_order: VecDeque::new(),
            pending: BTreeMap::new(),
            halts: BTreeMap::new(),
            source_halts: BTreeMap::new(),
            booked: Vec::new(),
            day: None,
            day_start_equity: cash,
            strategies: BTreeMap::new(),
            books: BTreeMap::new(),
            shadow_exposure: BTreeMap::new(),
            horizon_anchor_ms: BTreeMap::new(),
            last_alloc: BTreeMap::new(),
            dirty: BTreeSet::new(),
            refs: RefTable::default(),
            unknown_refs: BTreeSet::new(),
            fx: FxTable::default(),
            fx_stale: BTreeSet::new(),
            exposure_by_venue: BTreeMap::new(),
            exposure_by_class: BTreeMap::new(),
            feed: FeedGuard::new(settings_feed),
            tournament: Tournament::standard(settings_cap, settings_kelly, settings_cost_bps),
            event_last: BTreeMap::new(),
            equity_history: VecDeque::new(),
            hourly_marks: BTreeMap::new(),
            risk_dirty: false,
            counters: BTreeMap::new(),
        }
    }

    // ------------------------------------------------------- reference data and FX (D-061)

    /// Replaces the instrument reference table (startup, periodic refresh, `ref.changed.*`).
    pub fn set_refdata(&mut self, refs: RefTable) {
        self.unknown_refs.retain(|i| refs.get(i).is_none());
        self.counters.insert("refdata.instruments", refs.len() as f64);
        self.counters.insert("refdata.unknown", self.unknown_refs.len() as f64);
        self.refs = refs;
        self.bump("refdata.loads");
    }

    #[cfg(test)]
    pub fn refdata(&self) -> &RefTable {
        &self.refs
    }

    /// Replaces the FX table (startup, after each ECB refresh). Returns a halt/resume to publish
    /// when the stale-FX halt state changed.
    pub fn set_fx(&mut self, fx: FxTable, now_ms: i64) -> Vec<Out> {
        self.counters.insert("fx.rates", fx.len() as f64);
        self.fx = fx;
        // re-evaluate every currency in use with the new table
        let ccys: BTreeSet<String> = self.fx_stale.clone();
        for c in ccys {
            let l = self.fx.convert(&c, at(now_ms), self.settings.fx.max_age());
            if !l.stale {
                self.fx_stale.remove(&c);
            }
        }
        self.update_fx_halt(now_ms)
    }

    #[cfg(test)]
    pub fn fx_table(&self) -> &FxTable {
        &self.fx
    }

    /// Quote currency of an instrument per reference data (`None` = unknown, treated as EUR).
    fn quote_ccy(&self, inst: &InstrumentId) -> Option<&str> {
        self.refs.get(inst).and_then(|r| r.quote_ccy.as_deref())
    }

    /// Conversion `quote ccy of inst → EUR` as of `at_ms`; records staleness.
    fn fx_for(&mut self, inst: &InstrumentId, at_ms: i64) -> FxLookup {
        let Some(ccy) = self.quote_ccy(inst).map(str::to_owned) else {
            return FxLookup {
                rate_to_eur: 1.0,
                source: fa_warehouse::fx::EUR_SOURCE.into(),
                as_of: None,
                stale: false,
            };
        };
        let l = self.fx.convert(&ccy, at(at_ms), self.settings.fx.max_age());
        if l.stale {
            self.bump("fx.stale");
            self.fx_stale.insert(ccy);
        } else {
            self.fx_stale.remove(&ccy);
        }
        l
    }

    /// Raises / clears the reduce-only `fx` halt (only with `fx.halt_on_stale`; spec 42 §8).
    fn update_fx_halt(&mut self, now_ms: i64) -> Vec<Out> {
        self.counters.insert("fx.stale_now", self.fx_stale.len() as f64);
        let active = self.halts.contains_key(&HaltScope::Fx);
        if self.settings.fx.halt_on_stale && !self.fx_stale.is_empty() && !active {
            let reason = format!(
                "fx stale: {} older than {} h",
                self.fx_stale.iter().cloned().collect::<Vec<_>>().join(","),
                self.settings.fx.max_age_hours
            );
            self.risk.halt(HaltScope::Fx, HaltReason::Health(reason.clone()));
            self.halts.insert(HaltScope::Fx, ActiveHalt { reason: reason.clone(), since_ms: now_ms, manual: false });
            self.bump("halts.raised");
            return vec![Out::Halt(Halt { scope: "fx".into(), reason, since: Some(ts_of(now_ms)), manual: false })];
        }
        if active && self.fx_stale.is_empty() && self.halts.get(&HaltScope::Fx).is_some_and(|h| !h.manual) {
            self.risk.clear(HaltScope::Fx);
            self.halts.remove(&HaltScope::Fx);
            self.bump("resumes");
            return vec![Out::Resume(Halt {
                scope: "fx".into(),
                reason: "fx rates fresh again".into(),
                since: Some(ts_of(now_ms)),
                manual: false,
            })];
        }
        Vec::new()
    }

    pub fn fx_status(&self) -> FxStatus {
        FxStatus {
            rates: self.fx.len(),
            currencies: self.fx.currencies().map(str::to_owned).collect(),
            stale: self.fx_stale.iter().cloned().collect(),
            halted: self.halts.contains_key(&HaltScope::Fx),
            max_age_hours: self.settings.fx.max_age_hours,
        }
    }

    /// Today's ledger-side payout reserve (`ledger.payout_quota`, spec 42 §6).
    pub fn set_reserved_payout(&mut self, eur: f64) {
        self.risk.set_reserved_payout(eur);
        self.counters.insert("payout_reserve_eur", self.risk.payout_reserve());
    }

    pub fn payout_reserve(&self) -> f64 {
        self.risk.payout_reserve()
    }

    /// Gross EUR exposure per venue / asset class of the current book, plus FX, reference-data
    /// and payout status (`ctl.risk.strategy list` → `portfolio`; heartbeat counters).
    pub fn exposure_view(&self) -> Value {
        let equity = self.equity();
        let (mut by_venue, mut by_class, mut provisional) = (BTreeMap::new(), BTreeMap::new(), 0.0);
        for (i, p) in &self.ledger.positions {
            if p.quantity == 0.0 {
                continue;
            }
            let eur = (p.quantity * self.marks.get(i).copied().unwrap_or(p.avg_cost_eur)).abs();
            let r = self.refs.get(i);
            let venue = r.and_then(|r| r.venue.clone()).unwrap_or_else(|| "unknown".into());
            let class = r.and_then(|r| r.asset_class).map_or_else(
                || format!("{:?}", self.settings.default_asset_class).to_lowercase(),
                |c| format!("{c:?}").to_lowercase(),
            );
            *by_venue.entry(venue).or_insert(0.0) += eur;
            *by_class.entry(class).or_insert(0.0) += eur;
            if r.is_some_and(|r| r.provisional()) {
                provisional += eur;
            }
        }
        json!({
            "equity_eur": equity,
            "deployable_eur": self.risk.deployable(equity),
            "payout_reserve_eur": self.risk.payout_reserve(),
            "gross_exposure_eur": self.gross_exposure_eur(),
            "by_venue": by_venue,
            "by_class": by_class,
            "target_by_venue": self.exposure_by_venue,
            "target_by_class": self.exposure_by_class,
            "provisional_eur": provisional,
            "fx": self.fx_status(),
            "refdata": {"instruments": self.refs.len(), "unknown": self.unknown_refs.len()},
            "allocator": self.allocator_spec,
            "halts": self.state(0).active_halts.iter().map(|h| h.scope.clone()).collect::<Vec<_>>(),
        })
    }

    /// Venue → gross EUR exposure of the current book (heartbeat counters).
    pub fn exposure_by_venue(&self) -> BTreeMap<String, f64> {
        self.exposure_view()["by_venue"]
            .as_object()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(0.0))).collect())
            .unwrap_or_default()
    }

    pub fn limits(&self) -> &RiskConfig {
        &self.risk.cfg
    }

    fn bump(&mut self, k: &'static str) {
        *self.counters.entry(k).or_insert(0.0) += 1.0;
    }

    pub fn equity(&self) -> f64 {
        self.ledger.equity(&self.marks)
    }

    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// Replaces the paper book with one rebuilt from Postgres (startup). The drawdown peak and
    /// the day-start equity restart from the seeded equity; intents still pending at the previous
    /// shutdown are not restored (their fills, if any, are booked as unknown-intent events).
    pub fn seed_ledger(&mut self, ledger: Ledger) {
        self.ledger = ledger;
        let eq = self.equity();
        self.risk.rearm(eq);
        self.risk.start_day(eq);
        self.day_start_equity = eq;
    }

    /// Fills booked since the last call, oldest first.
    pub fn take_booked_fills(&mut self) -> Vec<BookedFill> {
        std::mem::take(&mut self.booked)
    }

    /// Strategy states (with their sub-book record) changed since the last call.
    pub fn take_dirty_strategies(&mut self) -> Vec<(StrategyState, SubBookRecord)> {
        let ids = std::mem::take(&mut self.dirty);
        ids.iter().filter_map(|id| self.strategy_row(id)).collect()
    }

    /// Every strategy state with its sub-book record (periodic full flush).
    pub fn strategy_rows(&self) -> Vec<(StrategyState, SubBookRecord)> {
        self.strategies.keys().filter_map(|id| self.strategy_row(id)).collect()
    }

    fn strategy_row(&self, id: &str) -> Option<(StrategyState, SubBookRecord)> {
        let st = self.strategies.get(id)?.clone();
        let book = self.books.get(id).cloned().unwrap_or_default();
        let rec = SubBookRecord {
            v: 1,
            kind: if st.status == StrategyStatus::Shadow { "shadow" } else { "active" }.into(),
            book,
            shadow_exposure: self.shadow_exposure.get(id).cloned().unwrap_or_default(),
            horizon_anchor_ms: self.horizon_anchor_ms.get(id).copied(),
        };
        Some((st, rec))
    }

    pub fn strategy_ids(&self) -> Vec<String> {
        self.strategies.keys().cloned().collect()
    }

    /// Ids a strategy's validation statistics may be filed under: its id and its genome id.
    pub fn candidate_ids(&self, id: &str) -> Vec<String> {
        let mut ids = vec![id.to_owned()];
        if let Some((d, _)) = self.signals.get(id)
            && !d.genome_id.is_empty()
            && d.genome_id != id
        {
            ids.push(d.genome_id.clone());
        }
        ids
    }

    /// Restores persisted lifecycle rows and sub-books at startup (rows win over the in-memory
    /// defaults; a row without a book starts an empty one).
    pub fn seed_strategies(&mut self, rows: Vec<(StrategyState, Option<SubBookRecord>)>) {
        for (r, book) in rows {
            let id = r.strategy_id.clone();
            if let Some(b) = book {
                if b.kind == "shadow" || r.status == StrategyStatus::Shadow {
                    if !b.shadow_exposure.is_empty() {
                        self.shadow_exposure.insert(id.clone(), b.shadow_exposure);
                    }
                    if let Some(a) = b.horizon_anchor_ms {
                        self.horizon_anchor_ms.insert(id.clone(), a);
                    }
                }
                self.books.insert(id.clone(), b.book);
                self.bump("strategies.restored");
            }
            self.strategies.insert(id, r);
        }
    }

    pub fn halted(&self) -> bool {
        !self.risk.trading_allowed()
    }

    pub fn gross_exposure_eur(&self) -> f64 {
        self.ledger
            .positions
            .iter()
            .map(|(i, p)| (p.quantity * self.marks.get(i).copied().unwrap_or(p.avg_cost_eur)).abs())
            .sum()
    }

    // ------------------------------------------------------------------ inputs

    /// Latest signal per strategy; exposures with non-UUID instruments are dropped.
    pub fn on_signal(&mut self, m: &ProtoDist, now_ms: i64) {
        if m.strategy_id.is_empty() {
            return;
        }
        let exposures = m
            .exposures
            .iter()
            .filter_map(|e: &AssetExposure| {
                let inst = InstrumentId(Uuid::parse_str(&e.instrument_id).ok()?);
                let mean = if e.mean.is_finite() { e.mean.clamp(-1.0, 1.0) } else { 0.0 };
                let stddev = if e.stddev.is_finite() { e.stddev.max(0.0) } else { 1.0 };
                Some(fa_strategies::AssetExposure { instrument: inst, mean, stddev })
            })
            .collect();
        let d = ExposureDistribution {
            strategy_id: m.strategy_id.clone(),
            genome_id: m.genome_id.clone(),
            horizon: parse_horizon(&m.horizon),
            as_of_ms: ms_of(m.as_of.as_ref()).unwrap_or(now_ms),
            exposures,
        };
        self.register_strategy(&m.strategy_id, now_ms);
        if self.strategies[&m.strategy_id].status == StrategyStatus::Shadow {
            let horizon_ms = d.horizon.duration().as_millis() as i64;
            let anchor = *self.horizon_anchor_ms.entry(m.strategy_id.clone()).or_insert(d.as_of_ms);
            if horizon_ms > 0 && d.as_of_ms - anchor >= horizon_ms {
                let st = self.strategies.get_mut(&m.strategy_id).expect("registered");
                st.horizons_completed += (d.as_of_ms - anchor) as u64 / horizon_ms as u64;
                self.horizon_anchor_ms.insert(m.strategy_id.clone(), d.as_of_ms);
                self.dirty.insert(m.strategy_id.clone());
            }
        }
        self.signals.insert(m.strategy_id.clone(), (d, now_ms));
        self.bump("signals");
    }

    fn register_strategy(&mut self, id: &str, now_ms: i64) {
        if !self.strategies.contains_key(id) {
            let status = StrategyStatus::parse(&self.settings.new_strategy_status).unwrap_or(StrategyStatus::Active);
            self.strategies.insert(id.to_owned(), StrategyState::new(id, status, now_ms));
            self.dirty.insert(id.to_owned());
            self.bump("strategies.seen");
        }
    }

    /// Mark (converted to EUR as of the trade time) + traded volume. Returns halts newly
    /// triggered by the equity move or by stale FX.
    pub fn on_trade(&mut self, inst: InstrumentId, at_ms: i64, price: f64, size: f64) -> Vec<Out> {
        if price <= 0.0 || !price.is_finite() {
            return Vec::new();
        }
        let fx = self.fx_for(&inst, at_ms).rate_to_eur;
        let price_eur = price * fx;
        // bad-tick filter (spec 40 §8): a jump is held back until confirmed, the instrument is
        // frozen meanwhile (FeedState::Frozen in the tick)
        match self.feed.on_trade(inst, at_ms, price_eur, self.marks.get(&inst).copied()) {
            TradeVerdict::Held => {
                self.bump("feed.jump_held");
                let prev = self.marks.get(&inst).copied().unwrap_or(0.0);
                let jump_max = self.feed.cfg.jump_max;
                return self
                    .event(
                        at_ms,
                        "feed.jump",
                        &inst.to_string(),
                        json!({"price_eur": price_eur, "mark_eur": prev, "jump_max": jump_max,
                               "action": "held back; instrument frozen until confirmed or discarded"}),
                    )
                    .into_iter()
                    .collect();
            }
            TradeVerdict::Discarded => self.bump("feed.jump_discarded"),
            TradeVerdict::Confirmed => self.bump("feed.jump_confirmed"),
            TradeVerdict::Accept => {}
        }
        self.marks.insert(inst, price_eur);
        self.first_seen_ms.entry(inst).or_insert(at_ms);
        let window = self.settings.volume_window_secs as i64 * 1000;
        let v = self.volume.entry(inst).or_default();
        v.push_back((at_ms, price_eur * size.max(0.0)));
        while v.front().is_some_and(|(t, _)| at_ms - *t > window) {
            v.pop_front();
        }
        self.bump("trades");
        let mut out = self.update_fx_halt(at_ms);
        out.extend(self.check_equity(at_ms));
        out
    }

    /// Records an intent seen on `order.intent.*` / `order.approved.*` so fills can be booked.
    /// Foreign intents on `order.intent.*` go through the risk gate (spec 40 pipeline).
    pub fn on_intent_seen(&mut self, m: &OrderIntent, approved_stage: bool, now_ms: i64) -> Vec<Out> {
        let Ok(uuid) = Uuid::parse_str(&m.instrument_id) else {
            return Vec::new();
        };
        let inst = InstrumentId(uuid);
        let signed = match Side::try_from(m.side).unwrap_or_default() {
            Side::Sell => -m.quantity,
            _ => m.quantity,
        };
        let known = self.intents.get(&m.intent_id).cloned();
        if known.as_ref().is_some_and(|k| k.ours) {
            return Vec::new();
        }
        if known.is_none() {
            self.remember(
                m.intent_id.clone(),
                IntentInfo {
                    instrument: inst,
                    signed_quantity: signed,
                    ours: false,
                    strategy_ids: m.strategy_ids.clone(),
                },
            );
            self.bump("intents.foreign");
        }
        if approved_stage || known.is_some() {
            return Vec::new();
        }
        // gate a foreign intent
        let mode = TradingMode::try_from(m.mode).unwrap_or_default();
        let reject = |reason: &str| Out::Rejected {
            account: m.account_id.clone(),
            event: OrderEvent {
                intent_id: m.intent_id.clone(),
                venue_order_id: String::new(),
                kind: order_event::Kind::Rejected as i32,
                event_time: Some(ts_of(now_ms)),
                reason: reason.into(),
                ..Default::default()
            },
        };
        if mode != TradingMode::Paper {
            self.bump("rejected.mode");
            return vec![reject("only paper trading exists; live/shadow intents are refused (D-029)")];
        }
        if self.instrument_blocked(&inst) {
            let current = self.ledger.positions.get(&inst).map_or(0.0, |p| p.quantity);
            let reducing =
                current != 0.0 && signed.signum() != current.signum() && signed.abs() <= current.abs() + 1e-9;
            if !(self.risk.reduce_only() && reducing) {
                self.bump("rejected.halted");
                return vec![reject("trading halted")];
            }
        }
        // pre-trade checks: a foreign/manual order meets the same caps as the allocator's targets
        if let Err(reason) = self.pretrade(inst, signed, m.limit_price, now_ms) {
            self.bump("rejected.pretrade");
            let mut out = vec![reject(&reason)];
            out.extend(self.event(
                now_ms,
                "pretrade.reject",
                &inst.to_string(),
                json!({"intent_id": m.intent_id, "quantity": signed, "limit_price": m.limit_price, "reason": reason}),
            ));
            return out;
        }
        self.bump("approved.foreign");
        vec![Out::Approved(m.clone())]
    }

    /// Market facts behind the caps for one instrument (reference data + horizon volume).
    fn market_info(&self, i: InstrumentId, now_ms: i64) -> MarketInfo {
        let r = self.refs.get(&i);
        MarketInfo {
            asset_class: r.and_then(|r| r.asset_class).unwrap_or(self.settings.default_asset_class),
            venue: r.and_then(|r| r.venue.clone()).unwrap_or_default(),
            horizon_volume_eur: self.horizon_volume_eur(i, now_ms),
            provisional: r.is_some_and(|r| r.provisional()),
        }
    }

    /// [`fa_risk::pretrade::check_order`] against the current book, plus the feed guard: a
    /// frozen instrument (stale feed, unconfirmed jump) only accepts reductions.
    fn pretrade(
        &mut self,
        inst: InstrumentId,
        signed_qty: f64,
        limit_price: Option<f64>,
        now_ms: i64,
    ) -> Result<(), String> {
        let mark = self.marks.get(&inst).copied();
        let current = self.current_eur();
        let position_eur = current.get(&inst).copied().unwrap_or(0.0);
        let fx_to_eur = self.fx_for(&inst, now_ms).rate_to_eur;
        let market = self.marks.contains_key(&inst).then(|| self.market_info(inst, now_ms));
        let (mut class_gross, mut venue_gross, mut gross) = (0.0, 0.0, 0.0);
        for (i, v) in &current {
            let mi = self.market_info(*i, now_ms);
            gross += v.abs();
            if let Some(m) = &market {
                if mi.asset_class == m.asset_class {
                    class_gross += v.abs();
                }
                if mi.venue == m.venue {
                    venue_gross += v.abs();
                }
            }
        }
        let deployable = self.risk.deployable(self.equity());
        let o = OrderCheck {
            quantity: signed_qty,
            limit_price,
            mark_eur: mark,
            fx_to_eur,
            position_eur,
            market: market.as_ref(),
            deployable_eur: deployable,
            class_gross_eur: class_gross,
            venue_gross_eur: venue_gross,
            portfolio_gross_eur: gross,
        };
        check_order(&self.risk.cfg, &o)?;
        let post = position_eur + signed_qty * mark.unwrap_or(0.0);
        let reducing =
            position_eur != 0.0 && post.signum() * position_eur.signum() >= 0.0 && post.abs() <= position_eur.abs();
        // the instrument's fastest strategy bar, else one second (its own trade cadence still
        // stretches the threshold for slow feeds)
        let bar = self
            .signals
            .values()
            .filter(|(d, _)| d.exposures.iter().any(|x| x.instrument == inst))
            .map(|(d, _)| d.horizon.timescale.duration().as_millis() as i64)
            .min()
            .unwrap_or(1000);
        let state = self.feed.state(&inst, now_ms, Some(bar));
        if state != FeedState::Fresh && !reducing {
            return Err(format!("pre-trade: feed {state:?} for the instrument (stale or unconfirmed price jump)"));
        }
        Ok(())
    }

    /// A `risk.event.<kind>` for `scope`, at most once per [`EVENT_INTERVAL_MS`] per
    /// (kind, scope); counted either way.
    fn event(&mut self, now_ms: i64, kind: &str, scope: &str, detail: Value) -> Option<Out> {
        *self.counters.entry("risk_events").or_insert(0.0) += 1.0;
        let key = (kind.to_owned(), scope.to_owned());
        if self.event_last.get(&key).is_some_and(|t| now_ms - t < EVENT_INTERVAL_MS) {
            return None;
        }
        self.event_last.insert(key, now_ms);
        if self.event_last.len() > 10_000 {
            self.event_last.retain(|_, t| now_ms - *t < EVENT_INTERVAL_MS);
        }
        Some(Out::RiskEvent { kind: kind.to_owned(), scope: scope.to_owned(), detail })
    }

    fn remember(&mut self, id: String, info: IntentInfo) {
        if self.intents.insert(id.clone(), info).is_none() {
            self.intent_order.push_back(id);
            if self.intent_order.len() > MAX_INTENTS
                && let Some(old) = self.intent_order.pop_front()
            {
                self.intents.remove(&old);
            }
        }
    }

    /// Books fills into the paper ledger; clears pending state on any terminal event.
    pub fn on_order_event(&mut self, m: &OrderEvent, now_ms: i64) -> Vec<Out> {
        let kind = order_event::Kind::try_from(m.kind).unwrap_or_default();
        if !matches!(kind, order_event::Kind::Ack) {
            self.pending.remove(&m.intent_id);
        }
        if !matches!(kind, order_event::Kind::Fill | order_event::Kind::PartialFill) || m.fill_quantity <= 0.0 {
            if matches!(kind, order_event::Kind::Rejected | order_event::Kind::Canceled) {
                self.bump("events.not_filled");
            }
            return Vec::new();
        }
        let Some(info) = self.intents.get(&m.intent_id).cloned() else {
            self.bump("events.unknown_intent");
            return Vec::new();
        };
        let fill_id =
            format!("{}:{}:{}", m.intent_id, m.venue_order_id, ms_of(m.event_time.as_ref()).unwrap_or(now_ms));
        let at_ms = ms_of(m.event_time.as_ref()).unwrap_or(now_ms);
        // spec 42 §6: the fill's quote currency → EUR as of the fill time, source recorded
        let fx = self.fx_for(&info.instrument, at_ms);
        let ccy = self.quote_ccy(&info.instrument).unwrap_or(fa_core::BASE_CURRENCY).to_owned();
        let fee_ccy = if m.fee_currency.trim().is_empty() { ccy.as_str() } else { m.fee_currency.trim() };
        let fee_fx = if fee_ccy.eq_ignore_ascii_case(fa_core::BASE_CURRENCY) {
            1.0
        } else if fee_ccy.eq_ignore_ascii_case(&ccy) {
            fx.rate_to_eur
        } else {
            self.fx.convert(fee_ccy, at(at_ms), self.settings.fx.max_age()).rate_to_eur
        };
        let fill = Fill {
            fill_id,
            intent_id: m.intent_id.clone(),
            instrument: info.instrument,
            quantity: m.fill_quantity.copysign(info.signed_quantity),
            price: m.fill_price,
            fx_to_eur: fx.rate_to_eur,
            fee_eur: m.fee * fee_fx,
        };
        let applied = self.ledger.apply_fill(&fill);
        let mut out = Vec::new();
        if applied {
            self.bump("fills");
            self.marks.entry(info.instrument).or_insert(m.fill_price * fx.rate_to_eur);
            let slippage = (m.arrival_price > 0.0).then_some(m.slippage_bps);
            self.attribute(&fill, &info.strategy_ids, slippage);
            let details = FillDetails {
                ccy,
                fx_source: fx.source.clone(),
                arrival_price: (m.arrival_price > 0.0).then_some(m.arrival_price),
                slippage_bps: slippage,
            };
            self.booked.push(BookedFill { fill, strategy_ids: info.strategy_ids.clone(), at_ms, details });
            out.extend(self.update_fx_halt(now_ms));
        }
        out.extend(self.check_equity(now_ms));
        out
    }

    /// Splits a fill evenly across its contributing strategies' virtual books (spec 40 §6.3,
    /// D-047). Flatten intents have no contributors and touch no book.
    fn attribute(&mut self, fill: &Fill, strategy_ids: &[String], slippage_bps: Option<f64>) {
        let ids: Vec<&String> = strategy_ids.iter().filter(|id| self.strategies.contains_key(*id)).collect();
        if ids.is_empty() {
            return;
        }
        let n = ids.len() as f64;
        let share = Fill { quantity: fill.quantity / n, fee_eur: fill.fee_eur / n, ..fill.clone() };
        for id in ids {
            let b =
                self.books.entry(id.clone()).or_insert_with(|| VirtualBook { primed: true, ..VirtualBook::default() });
            b.apply(&share);
            if let Some(s) = slippage_bps.filter(|s| s.is_finite()) {
                b.slippage_sum_bps += s;
                b.slippage_n += 1;
            }
            self.dirty.insert(id.clone());
        }
    }

    /// A halt from the bus (operator kill switch, the ingestor's `risk.halt.source:<id>`, or
    /// another service). Source halts are tracked per source id so one feed recovering does not
    /// clear another's halt.
    pub fn on_halt(&mut self, h: &Halt, now_ms: i64) {
        let Some(scope) = HaltScope::parse(&h.scope) else {
            self.bump("halts.ignored_scope");
            return;
        };
        let reason = if h.manual { HaltReason::Manual(h.reason.clone()) } else { HaltReason::Health(h.reason.clone()) };
        let active = ActiveHalt {
            reason: h.reason.clone(),
            since_ms: ms_of(h.since.as_ref()).unwrap_or(now_ms),
            manual: h.manual,
        };
        self.risk.halt(scope, reason);
        self.risk_dirty = true;
        if scope == HaltScope::Source {
            let id = h.scope.split_once(':').map_or(ANY_SOURCE, |(_, id)| id).to_owned();
            self.source_halts.entry(id).or_insert(active);
        } else {
            self.halts.entry(scope).or_insert(active);
        }
        self.bump("halts.received");
    }

    /// `ingest.health.<source>` (spec 40 §6.5 health halts): `down` halts the source's scope and
    /// flattens; `ok` clears a halt this path (or `risk.halt.source:<id>`) raised, never a manual
    /// one. Returns true when the halt set changed.
    pub fn on_source_health(&mut self, source: &str, status: &str, reason: &str, now_ms: i64) -> bool {
        match status {
            "down" => {
                if self.source_halts.contains_key(source) {
                    return false;
                }
                self.on_halt(
                    &Halt {
                        scope: format!("source:{source}"),
                        reason: if reason.is_empty() { format!("ingest: {source} down") } else { reason.into() },
                        since: Some(ts_of(now_ms)),
                        manual: false,
                    },
                    now_ms,
                );
                true
            }
            "ok" => match self.source_halts.get(source) {
                Some(h) if !h.manual => {
                    self.clear_source(source);
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    fn clear_source(&mut self, id: &str) {
        if id == ANY_SOURCE {
            self.source_halts.clear();
        } else {
            self.source_halts.remove(id);
        }
        if self.source_halts.is_empty() {
            self.risk.clear(HaltScope::Source);
        }
        self.bump("resumes");
    }

    /// `risk.resume.<scope>` (D-030): manual re-arm. `source:<id>` clears one source, bare
    /// `source` clears them all.
    pub fn on_resume(&mut self, scope: &str) {
        let Some(parsed) = HaltScope::parse(scope) else {
            return;
        };
        if parsed == HaltScope::Source {
            let id = scope.split_once(':').map_or(ANY_SOURCE, |(_, id)| id).to_owned();
            self.clear_source(&id);
            return;
        }
        if parsed == HaltScope::Portfolio {
            self.risk.rearm(self.equity());
        } else {
            self.risk.clear(parsed);
        }
        self.halts.remove(&parsed);
        self.risk_dirty = true;
        self.bump("resumes");
    }

    /// Any health halt (source / model / broker) is active (promotion gate item 5).
    fn health_halted(&self) -> bool {
        !self.source_halts.is_empty()
            || self.halts.contains_key(&HaltScope::Model)
            || self.halts.contains_key(&HaltScope::Broker)
    }

    // ------------------------------------------------------------ lifecycle (D-051)

    pub fn strategy(&self, id: &str) -> Option<&StrategyState> {
        self.strategies.get(id)
    }

    /// Gate checklist for a strategy; `deflated_sharpe` comes from `validation.trial` when the
    /// I/O layer could read it (`None` = unknown → blocks promotion).
    pub fn checklist(&self, id: &str, ext: &Evidence, now_ms: i64) -> Option<Checklist> {
        let st = self.strategies.get(id)?;
        let (consistent, detail) = self.consistency_of(st);
        let ev = GateEvidence {
            deflated_sharpe: ext.deflated_sharpe,
            pbo: ext.pbo.or_else(|| st.pipeline.as_ref().and_then(|p| p.selection.pbo)),
            consistent_with_backtest: consistent,
            consistency_detail: detail,
            unresolved_health_halts: self.health_halted(),
            selection_free: !st.evolved(),
        };
        Some(self.settings.promotion.check(st, &ev, now_ms))
    }

    /// Shadow/live-vs-backtest consistency of a strategy's own book since its current status
    /// began (spec 40 §6.6 item 4): hourly P&L points against the selection trial's per-step
    /// Sharpe. `None` for strategies without a backtest (baselines) or too few hours.
    fn consistency_of(&self, st: &StrategyState) -> (Option<bool>, String) {
        let Some(p) = st.pipeline.as_ref() else {
            return (None, "unknown: no backtest (not an evolved strategy)".into());
        };
        let Some(bt) = p.selection.sharpe else {
            return (None, "unknown: no selection Sharpe".into());
        };
        let pts: Vec<f64> = self
            .books
            .get(&st.strategy_id)
            .map(|b| b.history.iter().filter(|(t, _)| *t >= st.since_ms).map(|(_, v)| *v).collect())
            .unwrap_or_default();
        let min = self.settings.pipeline.decay_min_hours;
        match fa_risk::strategy::consistency(bt, p.step_secs.max(1.0), &pts, self.settings.promotion.consistency_z, min)
        {
            Some((ok, sh, b, n)) => (
                Some(ok),
                format!(
                    "shadow Sharpe/h {sh:.3} vs backtest {b:.3} − {}·se over {n} h",
                    self.settings.promotion.consistency_z
                ),
            ),
            None => (None, format!("unknown: {} of {min} hourly points", pts.len().saturating_sub(1))),
        }
    }

    /// Operator lifecycle action (`ctl.risk.strategy`). Errors follow the control-api prefix
    /// convention (`not_found:` / `conflict:` / `invalid:`).
    pub fn strategy_action(
        &mut self,
        id: &str,
        action: StrategyAction,
        ext: &Evidence,
        now_ms: i64,
    ) -> Result<StrategyState, String> {
        if !self.strategies.contains_key(id) {
            return Err(format!("not_found: unknown strategy {id:?} (no signal seen, no persisted state)"));
        }
        let gate = if action == StrategyAction::Promote { self.checklist(id, ext, now_ms) } else { None };
        let st = self.strategies.get_mut(id).expect("checked");
        let was = st.status;
        st.apply(action, now_ms, gate.as_ref())?;
        let now_status = st.status;
        self.after_transition(id, was, now_status);
        self.bump("strategy.actions");
        Ok(self.strategies[id].clone())
    }

    /// Book housekeeping when a strategy leaves or enters the allocated set.
    fn after_transition(&mut self, id: &str, was: StrategyStatus, now: StrategyStatus) {
        if was.allocates() && !now.allocates() {
            if let Some(b) = self.books.get_mut(id) {
                b.close(&self.marks);
            }
            self.last_alloc.remove(id);
        }
        if now != StrategyStatus::Shadow {
            self.shadow_exposure.remove(id);
            self.horizon_anchor_ms.remove(id);
        }
        if was != now {
            self.dirty.insert(id.to_owned());
        }
    }

    /// Portfolio flatten without a halt (`ctl.risk.flatten`): every active strategy goes to
    /// `flattening` (→ `paused` after this tick's closing intents); returns the affected ids.
    /// Resume them one by one; nothing re-enters on its own.
    pub fn flatten_all(&mut self, now_ms: i64) -> Vec<String> {
        let ids: Vec<String> = self
            .strategies
            .iter()
            .filter(|(_, s)| s.status == StrategyStatus::Active)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &ids {
            let _ = self.strategy_action(id, StrategyAction::Flatten, &Evidence::default(), now_ms);
        }
        self.bump("flatten.portfolio");
        ids
    }

    /// Runtime limit change (`ctl.risk.limits` `set`): `patch` is a partial JSON object merged
    /// over the current limits (`leverage_cap` keys merge too). Only tightening is accepted
    /// (spec 40 §6.5); otherwise `loosen: <fields>` names every offending field.
    pub fn set_limits(&mut self, patch: &Value) -> Result<RiskConfig, String> {
        let Some(obj) = patch.as_object() else {
            return Err("invalid: `set` must be a JSON object of limit fields".into());
        };
        let mut merged = serde_json::to_value(&self.risk.cfg).map_err(|e| format!("invalid: {e}"))?;
        let cur = merged.as_object_mut().expect("struct serialises to an object");
        for (k, v) in obj {
            if !cur.contains_key(k) {
                return Err(format!("invalid: unknown limit field {k:?}"));
            }
            if k == "leverage_cap" {
                let Some(caps) = v.as_object() else {
                    return Err("invalid: leverage_cap must be an object of asset class → cap".into());
                };
                let target = cur.get_mut(k).and_then(Value::as_object_mut).expect("leverage_cap is an object");
                for (class, cap) in caps {
                    target.insert(class.clone(), cap.clone());
                }
            } else {
                cur.insert(k.clone(), v.clone());
            }
        }
        let new: RiskConfig = serde_json::from_value(merged).map_err(|e| format!("invalid: {e}"))?;
        let bad = self.risk.cfg.check_tightens(&new);
        if !bad.is_empty() {
            return Err(format!("loosen: {}", bad.join("; ")));
        }
        self.risk.cfg = new.clone();
        self.settings.limits = new.clone();
        self.bump("limits.tightened");
        Ok(new)
    }

    /// Runtime allocator switch (`ctl.risk.limits {set: {allocator: {name, params}}}`, for the
    /// evolved allocator genomes, spec 40 §6.2). Not a limit: allowed in either direction but
    /// audited and logged by the caller. `name` is any `fa_allocator::genome::AllocatorName`
    /// (`equal_weight|risk_parity|fractional_kelly|weighted_blend`); `params` are genome genes
    /// (`cap`, `kelly`, `w_eq`, `w_rp`, `w_k`, `hl`, `pv`), clamped into their bounds; a `genome`
    /// string (`alloc:name=…;cap=…`) is accepted instead. `hl`/`pv` reset the performance
    /// tracker; the per-strategy cap never exceeds `max_strategy_weight` from `risk.toml`.
    pub fn set_allocator(&mut self, spec: &Value) -> Result<Value, String> {
        use fa_allocator::genome::{AllocatorGenome, AllocatorName};
        let mut g = if let Some(text) = spec.get("genome").and_then(Value::as_str).or_else(|| spec.as_str()) {
            AllocatorGenome::parse(text).map_err(|e| format!("invalid: allocator genome: {e}"))?
        } else {
            let name =
                spec.get("name").and_then(Value::as_str).ok_or("invalid: allocator needs `name` (or `genome`)")?;
            let name: AllocatorName = name.parse().map_err(|e| format!("invalid: allocator: {e}"))?;
            let mut g = AllocatorGenome::named(name);
            g.cap = self.settings.max_strategy_weight;
            g.kelly = self.settings.kelly_fraction;
            if let Some(params) = spec.get("params") {
                let obj = params.as_object().ok_or("invalid: allocator params must be an object")?;
                for (k, v) in obj {
                    let x = v.as_f64().ok_or_else(|| format!("invalid: allocator param {k} must be a number"))?;
                    if g.get(k).is_none() {
                        return Err(format!("invalid: unknown allocator param {k:?}"));
                    }
                    g.set(k, x);
                }
            }
            g.clamped()
        };
        g.cap = g.cap.min(self.settings.max_strategy_weight);
        let tracker_changed = spec.get("params").is_some_and(|p| p.get("hl").is_some() || p.get("pv").is_some())
            || spec.get("genome").is_some()
            || spec.is_string();
        self.allocator = g.build();
        if tracker_changed {
            self.perf = g.tracker();
        }
        self.allocator_spec =
            json!({"name": g.name.as_str(), "params": g.params(), "genome": g.to_string(), "source": "runtime"});
        self.bump("allocator.switches");
        Ok(self.allocator_spec.clone())
    }

    pub fn allocator_spec(&self) -> &Value {
        &self.allocator_spec
    }

    /// Strategy list for `ctl.risk.strategy list` / `GET /api/trading/strategies`.
    pub fn strategies_view(&self, evidence: &BTreeMap<String, Evidence>, now_ms: i64) -> Value {
        let none = Evidence::default();
        let rows: Vec<Value> = self
            .strategies
            .values()
            .map(|s| {
                let gate = self.checklist(&s.strategy_id, evidence.get(&s.strategy_id).unwrap_or(&none), now_ms);
                self.strategy_json(s, gate.as_ref(), now_ms)
            })
            .collect();
        json!({
            "as_of": ts_rfc3339(now_ms),
            "strategies": rows,
            "portfolio": self.exposure_view(),
            "allocator_tournament": self.tournament_view(),
        })
    }

    pub fn strategy_json(&self, s: &StrategyState, gate: Option<&Checklist>, now_ms: i64) -> Value {
        let weight = self.last_alloc.get(&s.strategy_id).copied().unwrap_or(0.0);
        let shadow_days = s.shadow_since_ms.map(|t| (now_ms - t) as f64 / MS_PER_DAY as f64);
        let book = self.books.get(&s.strategy_id);
        let positions: BTreeMap<String, f64> = book
            .map(|b| {
                b.positions
                    .iter()
                    .filter(|(_, p)| p.quantity != 0.0)
                    .map(|(i, p)| (i.to_string(), p.quantity))
                    .collect()
            })
            .unwrap_or_default();
        json!({
            "strategy_id": s.strategy_id,
            "status": s.status.as_str(),
            "prev_status": s.prev_status.map(StrategyStatus::as_str),
            "since": ts_rfc3339(s.since_ms),
            "first_seen": ts_rfc3339(s.first_seen_ms),
            "sleeve_eur": s.sleeve_eur,
            "weight": weight,
            "pnl_eur": s.pnl_eur,
            "peak_pnl_eur": s.peak_pnl_eur,
            "drawdown": s.drawdown(),
            "max_drawdown": s.max_drawdown,
            "last_reason": s.last_reason,
            "slippage_bps_mean": book.and_then(VirtualBook::slippage_bps_mean),
            "fills_with_slippage": book.map_or(0, |b| b.slippage_n),
            "book": {
                "cash_eur": book.map_or(0.0, |b| b.cash_eur),
                "value_eur": book.map_or(0.0, |b| b.value(&self.marks)),
                "realised_eur": book.map_or(0.0, VirtualBook::realised_eur),
                "fees_eur": book.map_or(0.0, |b| b.fees_eur),
                "positions": positions,
                "history_points": book.map_or(0, |b| b.history.len()),
            },
            "shadow": {
                "since": s.shadow_since_ms.map(ts_rfc3339),
                "days": shadow_days,
                "horizons_completed": s.horizons_completed,
            },
            "gate": gate,
            "evolved": s.evolved(),
            "pipeline": s.pipeline,
        })
    }

    // ---------------------------------------------------------------- internals

    /// Instruments to flatten for the active per-source halts, or `None` when the halts must
    /// flatten everything (no source halt → `None` too; the plain gate then applies).
    /// Whether new orders on `inst` are blocked by an active halt. Per-source halts block only
    /// the instruments their source feeds when that mapping is unambiguous (same rule as the tick).
    fn instrument_blocked(&self, inst: &InstrumentId) -> bool {
        if self.risk.trading_allowed() {
            return false;
        }
        let others = self.risk.halts().keys().any(|s| *s != HaltScope::Source);
        match self.source_flatten_set(std::iter::once(inst)) {
            Some(set) => others || set.contains(inst),
            None => true,
        }
    }

    fn source_flatten_set<'a>(&self, insts: impl Iterator<Item = &'a InstrumentId>) -> Option<BTreeSet<InstrumentId>> {
        if self.source_halts.is_empty() || self.source_halts.contains_key(ANY_SOURCE) {
            return None;
        }
        let mut set = BTreeSet::new();
        for i in insts {
            let src = self.refs.get(i)?.source()?;
            if self.source_halts.contains_key(src) {
                set.insert(*i);
            }
        }
        Some(set)
    }

    fn check_equity(&mut self, now_ms: i64) -> Vec<Out> {
        let eq = self.equity();
        let mut out = Vec::new();
        for (scope, reason) in self.risk.on_equity(eq) {
            let text = match &reason {
                HaltReason::Drawdown(dd) => {
                    format!("drawdown {:.2}% ≥ limit {:.2}%", dd * 100.0, self.settings.limits.max_drawdown * 100.0)
                }
                HaltReason::DailyLoss(dl) => {
                    format!("daily loss {:.2}% ≥ limit {:.2}%", dl * 100.0, self.settings.limits.max_daily_loss * 100.0)
                }
                HaltReason::Health(s) | HaltReason::Manual(s) => s.clone(),
            };
            self.halts.insert(scope, ActiveHalt { reason: text.clone(), since_ms: now_ms, manual: false });
            self.risk_dirty = true;
            self.bump("halts.raised");
            out.push(Out::Halt(Halt {
                scope: scope.as_str().into(),
                reason: text,
                since: Some(ts_of(now_ms)),
                manual: false,
            }));
        }
        out
    }

    fn roll_day(&mut self, now_ms: i64) {
        let day = now_ms.div_euclid(MS_PER_DAY);
        if self.day != Some(day) {
            if self.day.is_some() {
                let eq = self.equity();
                self.risk.start_day(eq);
                self.day_start_equity = eq;
                self.halts.remove(&HaltScope::DailyLoss);
            }
            self.day = Some(day);
        }
    }

    fn horizon_volume_eur(&self, inst: InstrumentId, now_ms: i64) -> f64 {
        let Some(v) = self.volume.get(&inst) else {
            return 0.0;
        };
        let sum: f64 = v.iter().map(|(_, e)| e).sum();
        let window = self.settings.volume_window_secs as f64 * 1000.0;
        let elapsed = (now_ms - self.first_seen_ms.get(&inst).copied().unwrap_or(now_ms)).max(1) as f64;
        // extrapolate while the window is not yet full
        if elapsed < window { sum * window / elapsed } else { sum }
    }

    /// Positions in EUR (book + pending intents).
    fn current_eur(&self) -> BTreeMap<InstrumentId, f64> {
        let mut cur: BTreeMap<InstrumentId, f64> = self
            .ledger
            .positions
            .iter()
            .filter(|(_, p)| p.quantity != 0.0)
            .map(|(i, p)| (*i, p.quantity * self.marks.get(i).copied().unwrap_or(p.avg_cost_eur)))
            .collect();
        for p in self.pending.values() {
            *cur.entry(p.instrument).or_insert(0.0) += p.signed_eur;
        }
        cur
    }

    fn update_performance(&mut self) {
        let mut realised: BTreeMap<InstrumentId, f64> = BTreeMap::new();
        for (i, m) in &self.marks {
            if let Some(prev) = self.tick_marks.get(i)
                && *prev > 0.0
            {
                realised.insert(*i, (m / prev).ln());
            }
        }
        if !self.tick_marks.is_empty() {
            let mut rets: BTreeMap<String, f64> = BTreeMap::new();
            for (d, _) in self.signals.values() {
                self.perf.observe(d, &realised);
                if let Some(r) = PerformanceTracker::step_return(d, &realised) {
                    rets.insert(d.strategy_id.clone(), r);
                }
            }
            // the weights every tournament entry set last tick earn this tick's returns
            self.tournament.realise(&rets);
        }
        self.tick_marks = self.marks.clone();
    }

    // -------------------------------------------------------------------- tick

    /// One decision tick. Returns intents/approvals/rejections/halts to publish, in order.
    pub fn tick(&mut self, now_ms: i64) -> Vec<Out> {
        self.roll_day(now_ms);
        let mut out = self.check_equity(now_ms);
        let equity = self.equity();
        let deployable = self.risk.deployable(equity);
        // strategy books first: they need the previous tick's marks, which update_performance replaces
        out.extend(self.track_strategy_books(deployable, now_ms));
        self.update_performance();
        self.record_hour(now_ms, equity);
        let ttl = self.settings.signal_ttl_secs as i64 * 1000;
        self.signals.retain(|_, (_, seen)| now_ms - *seen <= ttl);
        self.pending.retain(|_, p| now_ms - p.created_ms <= self.settings.pending_ttl_secs as i64 * 1000);
        // only active strategies are allocated; paused / flattening / shadow contribute nothing
        let dists: Vec<ExposureDistribution> = self
            .signals
            .values()
            .filter(|(d, _)| self.strategies.get(&d.strategy_id).is_some_and(|s| s.status.allocates()))
            .map(|(d, _)| d.clone())
            .collect();
        let stats = self.perf.stats_for(dists.iter().map(|d| d.strategy_id.as_str()));
        let mut alloc = self.allocator.allocate(&stats);
        // spec 40 §8: NaN / negative / over-allocated weights → previous allocation + risk event
        let total: f64 = alloc.values().sum();
        if alloc.values().any(|w| !w.is_finite() || *w < 0.0) || !total.is_finite() || total > 1.0 + 1e-9 {
            let bad = alloc.clone();
            alloc = self
                .last_alloc
                .iter()
                .filter(|(id, _)| dists.iter().any(|d| &d.strategy_id == *id))
                .map(|(k, v)| (k.clone(), *v))
                .collect();
            self.bump("allocator.fallbacks");
            let name = self.allocator.id().to_owned();
            out.extend(self.event(
                now_ms,
                "allocator.fallback",
                &name,
                json!({"weights": bad.iter().map(|(k, v)| (k.clone(), if v.is_finite() { json!(v) } else { json!(v.to_string()) })).collect::<serde_json::Map<_, _>>(),
                       "fallback": alloc}),
            ));
        }
        // live paper tournament: every control allocator on the same posteriors, plus the active one
        self.tournament.allocate(&stats);
        self.tournament.set_weights("active", alloc.clone());
        for (id, w) in &alloc {
            if let Some(s) = self.strategies.get_mut(id) {
                s.sleeve_eur = w * deployable;
            }
        }
        self.last_alloc = alloc.clone();
        let raw = fa_allocator::net_targets(&dists, &alloc, deployable);

        let mut market: BTreeMap<InstrumentId, MarketInfo> = BTreeMap::new();
        let known: Vec<InstrumentId> = raw.keys().filter(|i| self.marks.contains_key(i)).copied().collect();
        for i in known {
            if self.refs.get(&i).is_none() && !self.refs.is_empty() {
                self.unknown_refs.insert(i);
            }
            market.insert(i, self.market_info(i, now_ms));
        }
        self.counters.insert("refdata.unknown", self.unknown_refs.len() as f64);
        let capped = self.risk.cap_targets(&raw, &market, equity);
        // limit clips worth reporting (spec 40 §4.2 risk.event: "limit breaches")
        let clips: Vec<(InstrumentId, f64, f64)> = raw
            .iter()
            .filter(|(i, _)| market.contains_key(i))
            .filter_map(|(i, r)| {
                let c = capped.get(i).copied().unwrap_or(0.0);
                (r.abs() - c.abs() > self.settings.min_order_eur.max(0.01 * r.abs())).then_some((*i, *r, c))
            })
            .collect();
        if !clips.is_empty() {
            self.bump("limits.clipped_targets");
        }
        // events go after the orders of this tick
        let mut events: Vec<Out> = Vec::new();
        for (i, r, c) in clips {
            events.extend(self.event(now_ms, "limit.clip", &i.to_string(), json!({"target_eur": r, "capped_eur": c})));
        }
        let current = self.current_eur();
        // instruments we hold but no longer target are flattened
        let mut targets = capped;
        for i in current.keys() {
            targets.entry(*i).or_insert(0.0);
        }
        // health halts per source (spec 40 §6.5): when every instrument in play maps to exactly one
        // source, only the halted sources' instruments are flattened and the rest trades on;
        // otherwise (unknown or multi-source instruments, bare `source` halt) everything flattens.
        let gated = match self.source_flatten_set(targets.keys().chain(current.keys())) {
            Some(set) => {
                let mut g = self.risk.gate_excluding(&[HaltScope::Source], &targets, &current);
                for i in set {
                    g.insert(i, 0.0);
                }
                g
            }
            None => self.risk.gate(&targets, &current),
        };
        // feed guard (spec 40 §6.5 health halts): frozen instruments are reduce-only (the position
        // is kept or reduced, never increased or flipped), past the flatten threshold closed
        let mut gated = gated;
        let mut bar_ms: BTreeMap<InstrumentId, i64> = BTreeMap::new();
        for d in &dists {
            let b = d.horizon.timescale.duration().as_millis() as i64;
            for e in &d.exposures {
                let slot = bar_ms.entry(e.instrument).or_insert(b);
                *slot = (*slot).min(b);
            }
        }
        let mut frozen = 0usize;
        let insts: Vec<InstrumentId> = gated.keys().copied().collect();
        for i in insts {
            match self.feed.state(&i, now_ms, bar_ms.get(&i).copied()) {
                FeedState::Fresh => {}
                FeedState::Frozen => {
                    let c = current.get(&i).copied().unwrap_or(0.0);
                    let t = gated.get(&i).copied().unwrap_or(0.0);
                    // same rule as the daily-loss gate: toward zero, never beyond the position
                    let held =
                        if c == 0.0 || t.signum() != c.signum() { 0.0 } else { t.abs().min(c.abs()) * c.signum() };
                    if (held - t).abs() > 1e-9 {
                        frozen += 1;
                        events.extend(self.event(now_ms, "feed.frozen", &i.to_string(), json!({"action": "reduce-only: stale feed or unconfirmed price jump", "target_eur": t, "held_eur": held})));
                    }
                    gated.insert(i, held);
                }
                FeedState::Flatten => {
                    gated.insert(i, 0.0);
                    events.extend(self.event(
                        now_ms,
                        "feed.stale_flatten",
                        &i.to_string(),
                        json!({"stale_flatten_secs": self.feed.cfg.stale_flatten_secs}),
                    ));
                }
            }
        }
        self.counters.insert("feed.frozen", frozen as f64);
        self.counters.insert("feed.jump_pending", self.feed.held() as f64);
        self.exposure_by_venue = market.iter().fold(BTreeMap::new(), |mut m, (i, info)| {
            let venue = if info.venue.is_empty() { "unknown".to_owned() } else { info.venue.clone() };
            *m.entry(venue).or_insert(0.0) += gated.get(i).copied().unwrap_or(0.0).abs();
            m
        });
        self.exposure_by_class = market.iter().fold(BTreeMap::new(), |mut m, (i, info)| {
            *m.entry(format!("{:?}", info.asset_class).to_lowercase()).or_insert(0.0) +=
                gated.get(i).copied().unwrap_or(0.0).abs();
            m
        });

        let contributors = |inst: InstrumentId| -> Vec<String> {
            dists
                .iter()
                .filter(|d| d.exposures.iter().any(|e| e.instrument == inst && e.mean != 0.0))
                .map(|d| d.strategy_id.clone())
                .collect()
        };
        for (inst, target) in gated {
            let Some(mark) = self.marks.get(&inst).copied() else {
                continue;
            };
            let cur = current.get(&inst).copied().unwrap_or(0.0);
            let delta = target - cur;
            let band = self.settings.min_order_eur.max(self.settings.no_trade_band_frac * target.abs());
            if delta.abs() < band {
                continue;
            }
            let intent_id = ulid::Ulid::generate().to_string();
            let intent = OrderIntent {
                intent_id: intent_id.clone(),
                account_id: self.settings.account.clone(),
                instrument_id: inst.to_string(),
                side: if delta > 0.0 { Side::Buy } else { Side::Sell } as i32,
                quantity: delta.abs() / mark,
                exec_algo: "market".into(),
                limit_price: None,
                mode: TradingMode::Paper as i32, // never anything else (D-029)
                strategy_ids: if target == 0.0 { Vec::new() } else { contributors(inst) },
                created_at: Some(ts_of(now_ms)),
            };
            let _ = self.ledger.register_intent(&intent_id);
            self.remember(
                intent_id.clone(),
                IntentInfo {
                    instrument: inst,
                    signed_quantity: delta.signum() * intent.quantity,
                    ours: true,
                    strategy_ids: intent.strategy_ids.clone(),
                },
            );
            self.pending.insert(intent_id, Pending { instrument: inst, signed_eur: delta, created_ms: now_ms });
            self.bump("intents.ours");
            out.push(Out::Intent(intent.clone()));
            out.push(Out::Approved(intent));
        }
        // flattening strategies have had their closing intents issued: they are paused now
        let flattening: Vec<String> = self
            .strategies
            .iter()
            .filter(|(_, s)| s.status == StrategyStatus::Flattening)
            .map(|(id, _)| id.clone())
            .collect();
        for id in flattening {
            if let Some(s) = self.strategies.get_mut(&id) {
                s.flattened(now_ms);
            }
            self.dirty.insert(id.clone());
            out.push(Out::Audit {
                action: "strategy.flattened".into(),
                subject: id,
                payload: json!({"status": "paused"}),
            });
        }
        out.extend(events);
        out
    }

    /// Marks every strategy's virtual book to market, feeds the drawdown tracker and applies
    /// the per-strategy kill switch (spec 40 §6.5: flatten + pause). Shadow books hold the
    /// strategy's own signal on `shadow_sleeve_eur`; active books hold attributed fills.
    fn track_strategy_books(&mut self, deployable: f64, now_ms: i64) -> Vec<Out> {
        let limit = self.risk.cfg.strategy_max_drawdown;
        let shadow_sleeve = self.settings.shadow_sleeve_eur;
        let mut breached = Vec::new();
        let ids: Vec<String> = self.strategies.keys().cloned().collect();
        for id in ids {
            let status = self.strategies[&id].status;
            let (delta, sleeve) = match status {
                StrategyStatus::Active => {
                    let Some(b) = self.books.get_mut(&id) else { continue };
                    let v = b.value(&self.marks);
                    let delta = if b.primed {
                        v - b.last_value_eur
                    } else if b.positions.iter().all(|(i, p)| p.quantity == 0.0 || self.marks.contains_key(i)) {
                        b.primed = true;
                        0.0
                    } else {
                        continue; // restored book waiting for marks
                    };
                    b.last_value_eur = v;
                    (delta, self.last_alloc.get(&id).copied().unwrap_or(0.0) * deployable)
                }
                StrategyStatus::Shadow => {
                    // P&L of the exposure held since the previous tick
                    let held = self.shadow_exposure.remove(&id).unwrap_or_default();
                    let delta: f64 = held
                        .iter()
                        .filter_map(|(i, eur)| {
                            let (prev, now) = (self.tick_marks.get(i)?, self.marks.get(i)?);
                            (*prev > 0.0).then(|| eur * (now / prev - 1.0))
                        })
                        .sum();
                    // and the exposure the latest signal asks for, carried into the next tick
                    if let Some((d, _)) = self.signals.get(&id)
                        && !d.exposures.is_empty()
                    {
                        let per_asset = shadow_sleeve / d.exposures.len() as f64;
                        let next: BTreeMap<InstrumentId, f64> = d
                            .exposures
                            .iter()
                            .filter(|e| self.marks.contains_key(&e.instrument))
                            .map(|e| (e.instrument, per_asset * e.mean / (1.0 + e.stddev.max(0.0))))
                            .collect();
                        self.shadow_exposure.insert(id.clone(), next);
                    }
                    (delta, shadow_sleeve)
                }
                _ => continue,
            };
            let st = self.strategies.get_mut(&id).expect("iterating own keys");
            if delta == 0.0 && sleeve <= 0.0 {
                continue;
            }
            let dd = st.observe_pnl(delta, sleeve);
            let pnl = st.pnl_eur;
            if dd >= limit && limit > 0.0 && sleeve > 0.0 {
                st.breach(dd, limit, now_ms);
                breached.push((id.clone(), status, dd));
            }
            // hourly P&L point (bounded), persisted with the sub-book
            let b = self.books.entry(id.clone()).or_default();
            let fresh_hour =
                b.history.back().is_none_or(|(t, _)| t.div_euclid(MS_PER_HOUR) != now_ms.div_euclid(MS_PER_HOUR));
            b.push_history(now_ms, pnl);
            if fresh_hour || delta != 0.0 {
                self.dirty.insert(id);
            }
        }
        let mut out = Vec::new();
        for (id, was, dd) in breached {
            self.after_transition(&id, was, StrategyStatus::Paused);
            self.bump("strategy.drawdown_pauses");
            out.push(Out::Audit {
                action: "strategy.paused".into(),
                subject: id.clone(),
                payload: json!({"reason": self.strategies[&id].last_reason, "drawdown": dd, "limit": limit, "was": was.as_str(), "automatic": true}),
            });
        }
        out
    }

    /// Snapshot for `risk.state`.
    pub fn state(&self, now_ms: i64) -> RiskState {
        let equity = self.equity();
        let peak = self.risk.peak_equity();
        RiskState {
            as_of: Some(ts_of(now_ms)),
            equity_eur: equity,
            deployable_eur: self.risk.deployable(equity),
            gross_exposure_eur: self.gross_exposure_eur(),
            drawdown: if peak > 0.0 { (1.0 - equity / peak).max(0.0) } else { 0.0 },
            daily_pnl_eur: equity - self.day_start_equity,
            active_halts: self
                .halts
                .iter()
                .map(|(s, h)| (s.as_str().to_owned(), h))
                .chain(
                    self.source_halts.iter().map(|(id, h)| {
                        (if id == ANY_SOURCE { "source".to_owned() } else { format!("source:{id}") }, h)
                    }),
                )
                .map(|(scope, h)| Halt {
                    scope,
                    reason: h.reason.clone(),
                    since: Some(ts_of(h.since_ms)),
                    manual: h.manual,
                })
                .collect(),
        }
    }
}

impl Engine {
    // ------------------------------------------------------ strategy pipeline (spec 35 §6.4)

    pub fn known_strategy(&self, id: &str) -> bool {
        self.strategies.contains_key(id)
    }

    /// A new `candidate` from a generation's front (no-op when the id is known).
    pub fn admit_candidate(&mut self, m: &FrontMember, now_ms: i64) -> bool {
        if self.strategies.contains_key(&m.strategy_id) {
            return false;
        }
        let mut st = StrategyState::new(&m.strategy_id, StrategyStatus::Candidate, now_ms);
        st.pipeline = Some(m.record());
        st.last_reason = format!("selected from generation {}", m.generation);
        self.strategies.insert(m.strategy_id.clone(), st);
        self.dirty.insert(m.strategy_id.clone());
        self.bump("pipeline.candidates");
        true
    }

    /// Evolved strategies the pipeline still manages (not retired), sorted.
    pub fn pipeline_ids(&self) -> Vec<String> {
        self.strategies
            .iter()
            .filter(|(_, s)| s.evolved() && s.status != StrategyStatus::Retired)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Evolved strategies in the live strategy-runtime set (shadow, active, paused, flattening).
    pub fn live_evolved(&self) -> Vec<&StrategyState> {
        self.strategies.values().filter(|s| s.evolved() && s.status.in_live_set()).collect()
    }

    /// Genome texts of the managed live set, ordered by id.
    pub fn live_set_genomes(&self) -> Vec<String> {
        self.live_evolved().iter().filter_map(|s| s.pipeline.as_ref().map(|p| p.genome.clone())).collect()
    }

    /// Records that a holdout evaluation was requested (one attempt).
    pub fn record_holdout_request(&mut self, id: &str, holdout_id: &str, job_id: &str, now_ms: i64) {
        if let Some(p) = self.strategies.get_mut(id).and_then(|s| s.pipeline.as_mut()) {
            p.holdout_id = Some(holdout_id.to_owned());
            p.holdout_job_id = Some(job_id.to_owned());
            p.holdout_requested_ms = Some(now_ms);
            p.holdout_attempts += 1;
            self.dirty.insert(id.to_owned());
        }
    }

    /// A holdout job ended without a trial: note the error and allow the next attempt now.
    pub fn holdout_job_failed(&mut self, id: &str, error: &str) {
        if let Some(p) = self.strategies.get_mut(id).and_then(|s| s.pipeline.as_mut()) {
            p.note = format!("holdout job failed: {error}");
            p.holdout_requested_ms = None;
            self.dirty.insert(id.to_owned());
        }
    }

    /// One strategy's row with its sub-book (immediate persistence of a pipeline transition).
    pub fn strategy_row_pub(&self, id: &str) -> Option<(StrategyState, SubBookRecord)> {
        self.strategy_row(id)
    }

    /// Sets the pipeline note (what it waits on / last job error); persisted only on change.
    pub fn pipeline_note(&mut self, id: &str, note: &str) {
        if let Some(p) = self.strategies.get_mut(id).and_then(|s| s.pipeline.as_mut())
            && p.note != note
        {
            p.note = note.to_owned();
            self.dirty.insert(id.to_owned());
        }
    }

    /// Live-book consistency of an **active** evolved strategy (gate decay, spec 40 §6.6).
    pub fn live_consistent(&self, id: &str) -> Option<bool> {
        let st = self.strategies.get(id)?;
        if st.status != StrategyStatus::Active {
            return None;
        }
        self.consistency_of(st).0
    }

    /// Applies one pipeline decision. Returns `(audit action, payload)` for a transition, `None`
    /// when nothing changed. `RequestHoldout` / `Wait` / `Nothing` are the caller's business.
    pub fn apply_decision(
        &mut self,
        id: &str,
        d: &Decision,
        ext: &Evidence,
        now_ms: i64,
    ) -> Result<Option<(String, Value)>, String> {
        let Some(was) = self.strategies.get(id).map(|s| s.status) else {
            return Err(format!("not_found: {id}"));
        };
        let (action, reason) = match d {
            Decision::HoldoutPassed(h) => {
                let st = self.strategies.get_mut(id).expect("checked");
                if let Some(p) = st.pipeline.as_mut() {
                    p.holdout = Some(h.clone());
                    p.note.clear();
                }
                st.transition(StrategyStatus::HoldoutPassed, now_ms, "holdout passed");
                ("strategy.holdout_passed", "holdout passed".to_owned())
            }
            Decision::Admit => {
                let st = self.strategies.get_mut(id).expect("checked");
                st.apply(StrategyAction::Admit, now_ms, None)?;
                if let Some(p) = st.pipeline.as_mut() {
                    p.admitted_ms = Some(now_ms);
                }
                ("strategy.admitted", "admitted to the live set (shadow)".to_owned())
            }
            Decision::Promote => {
                self.strategy_action(id, StrategyAction::Promote, ext, now_ms)?;
                ("strategy.promote", "gate green: promoted automatically (paper)".to_owned())
            }
            Decision::Demote(r) => {
                let st = self.strategies.get_mut(id).expect("checked");
                st.transition(StrategyStatus::Shadow, now_ms, &format!("demoted: {r}"));
                ("strategy.demoted", r.clone())
            }
            Decision::Retire(r) => {
                let st = self.strategies.get_mut(id).expect("checked");
                st.transition(StrategyStatus::Retired, now_ms, &format!("retired: {r}"));
                if let Some(p) = st.pipeline.as_mut() {
                    p.note = r.clone();
                }
                ("strategy.retired", r.clone())
            }
            Decision::Nothing | Decision::Wait(_) | Decision::RequestHoldout { .. } => return Ok(None),
        };
        let now = self.strategies[id].status;
        self.after_transition(id, was, now);
        if now == StrategyStatus::Retired {
            self.books.remove(id);
            self.signals.remove(id);
        }
        self.dirty.insert(id.to_owned());
        self.bump("pipeline.transitions");
        Ok(Some((
            action.to_owned(),
            json!({"from": was.as_str(), "to": now.as_str(), "reason": reason, "automatic": true, "pipeline": self.strategies[id].pipeline}),
        )))
    }

    // ---------------------------------------------- allocator tournament (spec 40 §6.2)

    pub fn tournament_view(&self) -> Value {
        json!({
            "active": self.allocator_spec.get("name").cloned().unwrap_or(Value::Null),
            "cost_bps": self.tournament.cost_bps,
            "basis": "identical per-tick strategy sleeve returns; turnover charged at cost_bps (spec 40 §6.2)",
            "entries": self.tournament.report(),
        })
    }

    // -------------------------------------------- calibration history (spec 40 §7, §11)

    fn record_hour(&mut self, now_ms: i64, equity: f64) {
        let hour = now_ms.div_euclid(MS_PER_HOUR);
        let fresh = self.equity_history.back().is_none_or(|(t, _)| t.div_euclid(MS_PER_HOUR) != hour);
        if !fresh {
            if let Some(last) = self.equity_history.back_mut() {
                last.1 = equity;
            }
            return;
        }
        self.equity_history.push_back((now_ms, equity));
        while self.equity_history.len() > MAX_CALIBRATION_POINTS {
            self.equity_history.pop_front();
        }
        for (i, m) in &self.marks {
            let q = self.hourly_marks.entry(*i).or_default();
            q.push_back(*m);
            while q.len() > MAX_CALIBRATION_POINTS {
                q.pop_front();
            }
        }
        self.risk_dirty = true;
    }

    /// Limit suggestions from the realised paper book (tighten-only set included).
    pub fn calibration(&self, rules: &CalibrationRules) -> Calibration {
        let eq: Vec<f64> = self.equity_history.iter().map(|(_, e)| *e).collect();
        let vols: Vec<f64> = self
            .hourly_marks
            .values()
            .filter(|q| q.len() > rules.min_hours)
            .filter_map(|q| {
                let r: Vec<f64> = q
                    .iter()
                    .zip(q.iter().skip(1))
                    .filter(|(a, b)| **a > 0.0 && **b > 0.0)
                    .map(|(a, b)| (b / a).ln())
                    .collect();
                if r.len() < 2 {
                    return None;
                }
                let m = r.iter().sum::<f64>() / r.len() as f64;
                let v = r.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (r.len() - 1) as f64;
                Some(v.sqrt() * 24f64.sqrt())
            })
            .collect();
        fa_risk::calibrate::calibrate(&self.risk.cfg, rules, &eq, &vols)
    }

    // ------------------------------------------------- restart-safe risk state (spec 40 §8)

    pub fn risk_record(&self) -> RiskRecord {
        RiskRecord {
            v: 1,
            risk: self.risk.clone(),
            halts: self.halts.iter().map(|(k, v)| (k.as_str().to_owned(), v.clone())).collect(),
            source_halts: self.source_halts.clone(),
            day: self.day,
            day_start_equity: self.day_start_equity,
            equity_history: self.equity_history.clone(),
            tournament: self.tournament.records(),
        }
    }

    /// Restores the persisted halt state machine (after [`Self::seed_ledger`]). Limits stay the
    /// ones from `risk.toml` (a persisted record never loosens or tightens the config); the
    /// kill switch, every halt and the drawdown peak come back as they were.
    pub fn restore_risk(&mut self, r: RiskRecord) {
        let cfg = self.risk.cfg.clone();
        self.risk = r.risk;
        self.risk.cfg = cfg; // the payout reserve is re-read from the ledger every 30 s
        self.halts = r.halts.into_iter().filter_map(|(k, v)| HaltScope::parse(&k).map(|s| (s, v))).collect();
        self.source_halts = r.source_halts;
        self.day = r.day;
        self.day_start_equity = r.day_start_equity;
        self.equity_history = r.equity_history;
        self.tournament.restore(r.tournament);
        self.bump("risk_state.restored");
    }

    /// Whether the risk record changed since the last call.
    pub fn take_risk_dirty(&mut self) -> bool {
        std::mem::take(&mut self.risk_dirty)
    }
}

fn at(ms: i64) -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms).unwrap_or_default()
}

fn ts_rfc3339(ms: i64) -> String {
    chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms).map(|t| t.to_rfc3339()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: i64 = 1_762_560_000_000; // 2025-11-08T00:00:00Z
    const PX: f64 = 100.0;

    /// `risk.toml` with the feed guard off: the scenario tests below jump the clock and the
    /// price on purpose (the guard has its own tests).
    fn settings() -> Settings {
        let mut s = Settings::parse(include_str!("../risk.toml")).unwrap();
        s.feed.stale_freeze_bars = 0.0;
        s.feed.jump_max = 0.0;
        s
    }

    fn signal(id: &str, inst: InstrumentId, mean: f64) -> ProtoDist {
        ProtoDist {
            strategy_id: id.into(),
            genome_id: "baseline".into(),
            horizon: "5x1s".into(),
            as_of: Some(ts_of(T0)),
            exposures: vec![AssetExposure { instrument_id: inst.to_string(), mean, stddev: 0.0 }],
        }
    }

    /// Engine with a liquid instrument (1 h of 1000 EUR/s volume → liquidity cap 36 000 EUR).
    fn warm(inst: InstrumentId) -> Engine {
        let mut e = Engine::new(settings());
        for s in 0..3_600 {
            assert!(e.on_trade(inst, T0 - 3_600_000 + s * 1000, PX, 10.0).is_empty());
        }
        e
    }

    fn fill(e: &mut Engine, intent: &OrderIntent, px: f64, now: i64) {
        let ev = OrderEvent {
            intent_id: intent.intent_id.clone(),
            venue_order_id: format!("paper-{}", intent.intent_id),
            kind: order_event::Kind::Fill as i32,
            fill_price: px,
            fill_quantity: intent.quantity,
            fee: 0.0,
            fee_currency: "EUR".into(),
            event_time: Some(ts_of(now)),
            ..Default::default()
        };
        e.on_order_event(&ev, now);
    }

    fn approved(out: &[Out]) -> Vec<OrderIntent> {
        out.iter().filter_map(|o| if let Out::Approved(i) = o { Some(i.clone()) } else { None }).collect()
    }

    #[test]
    fn signals_become_netted_paper_intents_within_caps() {
        let inst = InstrumentId::new();
        let mut e = warm(inst);
        assert!(e.tick(T0).is_empty(), "no signals → no orders");
        e.on_signal(&signal("a", inst, 1.0), T0);
        e.on_signal(&signal("b", inst, -0.5), T0);
        let out = e.tick(T0 + 1000);
        let ints = approved(&out);
        assert_eq!(ints.len(), 1);
        assert!(matches!(out[0], Out::Intent(_)), "intent precedes approval");
        let i = &ints[0];
        assert_eq!(i.account_id, "paper");
        assert_eq!(i.mode, TradingMode::Paper as i32);
        assert_eq!(i.side, Side::Buy as i32);
        // equal weights 0.5 each of 100k: a → +50k, b → −25k, netted +25k; position cap 5% = 5k
        assert!((i.quantity * PX - 5_000.0).abs() < 1e-6, "capped to 5% of deployable, got {}", i.quantity * PX);
        assert_eq!(i.strategy_ids, vec!["a", "b"]);
        assert!(e.tick(T0 + 1500).is_empty(), "pending intent counts as position");
        fill(&mut e, i, PX, T0 + 1600);
        assert!((e.ledger().positions[&inst].quantity - 50.0).abs() < 1e-9);
        assert!(e.tick(T0 + 2000).is_empty(), "at target: nothing to trade");
        // strategy a flips short: target −5k → sell 100 units
        e.on_signal(&signal("a", inst, -1.0), T0 + 2500);
        let sell = approved(&e.tick(T0 + 3000));
        assert_eq!(sell.len(), 1);
        assert_eq!(sell[0].side, Side::Sell as i32);
        assert!((sell[0].quantity - 100.0).abs() < 1e-6);
        let st = e.state(T0 + 3000);
        assert!((st.equity_eur - 100_000.0).abs() < 1e-6);
        assert!((st.gross_exposure_eur - 5_000.0).abs() < 1e-6);
        assert!(st.active_halts.is_empty());
    }

    #[test]
    fn manual_halt_flattens_then_blocks_until_resume() {
        let inst = InstrumentId::new();
        let mut e = warm(inst);
        e.on_signal(&signal("a", inst, 1.0), T0);
        let buy = approved(&e.tick(T0 + 1000));
        fill(&mut e, &buy[0], PX, T0 + 1100);
        e.on_halt(
            &Halt { scope: "portfolio".into(), reason: "test".into(), since: Some(ts_of(T0 + 2000)), manual: true },
            T0 + 2000,
        );
        assert!(e.halted());
        let out = e.tick(T0 + 3000);
        let flat = approved(&out);
        assert_eq!(flat.len(), 1, "kill switch flattens");
        assert_eq!(flat[0].side, Side::Sell as i32);
        assert!(flat[0].strategy_ids.is_empty());
        fill(&mut e, &flat[0], PX, T0 + 3100);
        for k in 4..10 {
            e.on_signal(&signal("a", inst, 1.0), T0 + k * 1000);
            assert!(e.tick(T0 + k * 1000).is_empty(), "no orders while halted");
        }
        assert_eq!(e.state(T0 + 9000).active_halts[0].scope, "portfolio");
        assert!(e.state(T0 + 9000).active_halts[0].manual);
        e.on_resume("portfolio");
        assert!(!e.halted());
        e.on_signal(&signal("a", inst, 1.0), T0 + 10_000);
        assert_eq!(approved(&e.tick(T0 + 10_000)).len(), 1, "orders resume");
        assert!(e.state(T0 + 10_000).active_halts.is_empty());
    }

    #[test]
    fn daily_loss_is_reduce_only_and_drawdown_raises_kill_switch() {
        let inst = InstrumentId::new();
        let mut e = warm(inst);
        e.on_signal(&signal("a", inst, 1.0), T0);
        let buy = approved(&e.tick(T0 + 1000));
        fill(&mut e, &buy[0], PX, T0 + 1100); // 50 units
        // −2.5% equity: 50 units × 100 → need a 2 500 EUR loss → price 50
        let halts = e.on_trade(inst, T0 + 2000, 50.0, 1.0);
        assert_eq!(halts.len(), 1);
        let Out::Halt(h) = &halts[0] else { panic!() };
        assert_eq!(h.scope, "daily_loss");
        assert!(!h.manual);
        assert!(e.halted());
        e.on_signal(&signal("a", inst, 1.0), T0 + 2500);
        let out = e.tick(T0 + 3000);
        let ints = approved(&out);
        assert!(ints.iter().all(|i| i.side == Side::Sell as i32), "reduce-only: cannot add to the long");
        // the strategy lost 2 500 EUR of its 50 000 EUR sleeve = 5 % → per-strategy kill switch,
        // and its position is closed (a reducing order, allowed under the daily-loss halt)
        assert!(
            out.iter().any(
                |o| matches!(o, Out::Audit { action, subject, .. } if action == "strategy.paused" && subject == "a")
            ),
            "{out:?}"
        );
        assert_eq!(e.strategy("a").unwrap().status, StrategyStatus::Paused);
        assert!(e.strategy("a").unwrap().last_reason.contains("≥ limit 5.00%"));
        assert_eq!(ints.len(), 1);
        assert!((ints[0].quantity - 50.0).abs() < 1e-6, "may only close, not flip");
        fill(&mut e, &ints[0], 50.0, T0 + 3100);
        e.on_signal(&signal("a", inst, -1.0), T0 + 3500);
        assert!(approved(&e.tick(T0 + 4000)).is_empty(), "paused strategy places no new orders");
        assert!((e.state(T0 + 4100).daily_pnl_eur + 2_500.0).abs() < 1e-6);
        // next UTC day clears the daily-loss halt
        let out = e.tick(T0 + MS_PER_DAY + 1000);
        assert!(!e.halted());
        assert!(approved(&out).is_empty(), "signal expired (ttl) and book is flat → nothing to do");
        assert!(e.state(T0 + MS_PER_DAY + 1000).active_halts.is_empty());
    }

    #[test]
    fn drawdown_kill_switch_and_rearm() {
        let inst = InstrumentId::new();
        let mut e = warm(inst);
        e.on_signal(&signal("a", inst, 1.0), T0);
        let buy = approved(&e.tick(T0 + 1000));
        fill(&mut e, &buy[0], PX, T0 + 1100);
        // 50 units; −10% equity needs a 10 000 EUR loss → impossible at 5k notional, so cheat
        // through a second instrument? Simpler: price → 0.01 gives −4 999.5 (5%). Use a
        // leveraged book: fill a second foreign intent to hold 200 units.
        let foreign = OrderIntent {
            intent_id: "manual-1".into(),
            account_id: "paper".into(),
            instrument_id: inst.to_string(),
            side: Side::Buy as i32,
            quantity: 150.0,
            exec_algo: "market".into(),
            mode: TradingMode::Paper as i32,
            ..Default::default()
        };
        // the pre-trade checks refuse it (15 000 EUR > 10 % fat-finger cap, post-trade 20 000 >
        // position cap) — so this test books it as if approved elsewhere
        let gated = e.on_intent_seen(&foreign, false, T0 + 1200);
        assert!(
            matches!(&gated[0], Out::Rejected { event, .. } if event.reason.contains("max_order_frac")),
            "{gated:?}"
        );
        let forced = OrderIntent { intent_id: "manual-1b".into(), ..foreign.clone() };
        assert!(e.on_intent_seen(&forced, true, T0 + 1200).is_empty());
        fill(&mut e, &forced, PX, T0 + 1300);
        let halts = e.on_trade(inst, T0 + 2000, 40.0, 1.0); // −12 000 EUR
        let scopes: Vec<&str> =
            halts.iter().map(|o| if let Out::Halt(h) = o { h.scope.as_str() } else { "" }).collect();
        assert!(scopes.contains(&"portfolio") && scopes.contains(&"daily_loss"));
        assert!(!e.risk.reduce_only());
        let flat = approved(&e.tick(T0 + 3000));
        assert_eq!(flat.len(), 1);
        assert!((flat[0].quantity - 200.0).abs() < 1e-6, "kill switch flattens everything");
        e.on_resume("portfolio");
        assert!(e.risk.reduce_only(), "daily-loss halt survives the portfolio re-arm");
        assert!(e.state(T0 + 3000).drawdown.abs() < 1e-9, "peak re-armed");
    }

    #[test]
    fn foreign_intents_are_gated_and_live_is_impossible() {
        let inst = InstrumentId::new();
        let mut e = warm(inst);
        let live = OrderIntent {
            intent_id: "x".into(),
            account_id: "paper".into(),
            instrument_id: inst.to_string(),
            side: Side::Buy as i32,
            quantity: 1.0,
            mode: TradingMode::Live as i32,
            ..Default::default()
        };
        let out = e.on_intent_seen(&live, false, T0);
        let Out::Rejected { account, event } = &out[0] else { panic!("{out:?}") };
        assert_eq!(account, "paper");
        assert_eq!(event.kind, order_event::Kind::Rejected as i32);
        assert!(event.reason.contains("D-029"));
        // a paper intent already approved elsewhere (control-api manual order) is only remembered
        let manual = OrderIntent { intent_id: "m".into(), mode: TradingMode::Paper as i32, ..live.clone() };
        assert!(e.on_intent_seen(&manual, true, T0).is_empty());
        fill(&mut e, &manual, PX, T0 + 10);
        assert_eq!(e.ledger().positions[&inst].quantity, 1.0);
        // halted → foreign paper intents rejected
        e.on_halt(&Halt { scope: "portfolio".into(), manual: true, ..Default::default() }, T0 + 20);
        let paper = OrderIntent { intent_id: "p".into(), mode: TradingMode::Paper as i32, ..live.clone() };
        assert!(
            matches!(&e.on_intent_seen(&paper, false, T0 + 30)[0], Out::Rejected { event, .. } if event.reason == "trading halted")
        );
        assert!(Settings::parse("mode = 'live'").is_err());
    }

    #[test]
    fn unknown_market_and_stale_signals_produce_no_orders() {
        let inst = InstrumentId::new();
        let mut e = Engine::new(settings());
        e.on_signal(&signal("a", inst, 1.0), T0);
        assert!(e.tick(T0 + 1000).is_empty(), "no mark → no order");
        e.on_trade(inst, T0 + 1500, PX, 10.0);
        assert_eq!(approved(&e.tick(T0 + 2000)).len(), 1);
        assert!(
            e.tick(T0 + 61_000 + 2000 + 20_000).is_empty(),
            "signal expired and pending dropped: flatten? no position yet"
        );
    }

    #[test]
    fn booked_fills_are_queued_once_and_seeding_resets_the_risk_baseline() {
        let inst = InstrumentId::new();
        let mut e = warm(inst);
        let intent = OrderIntent {
            intent_id: "q".into(),
            account_id: "paper".into(),
            instrument_id: inst.to_string(),
            side: Side::Buy as i32,
            quantity: 2.0,
            exec_algo: "market".into(),
            mode: TradingMode::Paper as i32,
            strategy_ids: vec!["s1".into()],
            created_at: Some(ts_of(T0)),
            ..Default::default()
        };
        e.on_intent_seen(&intent, true, T0);
        fill(&mut e, &intent, PX, T0 + 1000);
        fill(&mut e, &intent, PX, T0 + 1000); // same event replayed
        let booked = e.take_booked_fills();
        assert_eq!(booked.len(), 1);
        assert_eq!(booked[0].fill.quantity, 2.0);
        assert_eq!(booked[0].strategy_ids, vec!["s1".to_owned()]);
        assert_eq!(booked[0].at_ms, T0 + 1000);
        assert!(e.take_booked_fills().is_empty(), "drained");

        let mut seeded = Ledger::with_cash(50_000.0);
        seeded.apply_fill(&Fill {
            fill_id: "old".into(),
            intent_id: "old-i".into(),
            instrument: inst,
            quantity: 10.0,
            price: PX,
            fx_to_eur: 1.0,
            fee_eur: 0.0,
        });
        e.seed_ledger(seeded);
        assert_eq!(e.ledger().positions[&inst].quantity, 10.0);
        assert!((e.equity() - 50_000.0).abs() < 1e-9);
        let st = e.state(T0 + 2000);
        assert!(st.drawdown.abs() < 1e-9, "peak re-armed at the seeded equity");
    }

    fn audits<'a>(out: &'a [Out], action: &str) -> Vec<&'a str> {
        out.iter()
            .filter_map(|o| match o {
                Out::Audit { action: a, subject, .. } if a == action => Some(subject.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn paused_strategy_contributes_no_exposure_until_resumed() {
        let inst = InstrumentId::new();
        let mut e = warm(inst);
        e.on_signal(&signal("a", inst, 1.0), T0);
        e.on_signal(&signal("b", inst, 1.0), T0);
        let buy = approved(&e.tick(T0 + 1000));
        assert_eq!(buy.len(), 1);
        assert!((buy[0].quantity * PX - 5_000.0).abs() < 1e-6, "a + b → 100k, capped to 5k");
        fill(&mut e, &buy[0], PX, T0 + 1100);
        assert_eq!(e.strategy("a").unwrap().status, StrategyStatus::Active);
        assert!((e.strategy("a").unwrap().sleeve_eur - 50_000.0).abs() < 1e-6);
        // lifecycle errors follow the control-api prefix convention
        assert!(
            e.strategy_action("nope", StrategyAction::Pause, &Evidence::default(), T0)
                .unwrap_err()
                .starts_with("not_found:")
        );
        assert!(
            e.strategy_action("a", StrategyAction::Resume, &Evidence::default(), T0)
                .unwrap_err()
                .starts_with("conflict:")
        );
        // pause a: only b remains (50k target, still capped at 5k) → nothing to trade
        let st = e.strategy_action("a", StrategyAction::Pause, &Evidence::default(), T0 + 2000).unwrap();
        assert_eq!(st.status, StrategyStatus::Paused);
        assert!(e.take_dirty_strategies().iter().any(|(s, _)| s.strategy_id == "a"));
        e.on_signal(&signal("a", inst, 1.0), T0 + 2500);
        e.on_signal(&signal("b", inst, 1.0), T0 + 2500);
        assert!(approved(&e.tick(T0 + 3000)).is_empty());
        // pause b too: no active strategy → position flattened
        e.strategy_action("b", StrategyAction::Pause, &Evidence::default(), T0 + 3500).unwrap();
        e.on_signal(&signal("b", inst, 1.0), T0 + 3600);
        let flat = approved(&e.tick(T0 + 4000));
        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].side, Side::Sell as i32);
        assert!((flat[0].quantity - 50.0).abs() < 1e-6, "paused strategies hold no exposure");
        assert!(flat[0].strategy_ids.is_empty());
        fill(&mut e, &flat[0], PX, T0 + 4100);
        assert!(
            e.strategy_action("a", StrategyAction::Pause, &Evidence::default(), T0 + 4200)
                .unwrap_err()
                .contains("already paused")
        );
        // resume a: its signal is allocated again
        assert_eq!(
            e.strategy_action("a", StrategyAction::Resume, &Evidence::default(), T0 + 4500).unwrap().status,
            StrategyStatus::Active
        );
        e.on_signal(&signal("a", inst, 1.0), T0 + 4600);
        let again = approved(&e.tick(T0 + 5000));
        assert_eq!(again.len(), 1);
        assert_eq!(again[0].side, Side::Buy as i32);
        assert_eq!(again[0].strategy_ids, vec!["a"]);
        // flatten a single active strategy: flattening → paused after the tick, audited
        fill(&mut e, &again[0], PX, T0 + 5100);
        assert_eq!(
            e.strategy_action("a", StrategyAction::Flatten, &Evidence::default(), T0 + 5500).unwrap().status,
            StrategyStatus::Flattening
        );
        let out = e.tick(T0 + 6000);
        assert_eq!(approved(&out)[0].side, Side::Sell as i32);
        assert_eq!(audits(&out, "strategy.flattened"), vec!["a"]);
        assert_eq!(e.strategy("a").unwrap().status, StrategyStatus::Paused);
        let view = e.strategies_view(&BTreeMap::new(), T0 + 6000);
        assert_eq!(view["strategies"].as_array().unwrap().len(), 2);
        assert_eq!(view["strategies"][0]["status"], "paused");
        assert_eq!(view["strategies"][0]["gate"]["green"], false);
    }

    #[test]
    fn shadow_strategies_track_a_virtual_book_and_need_a_green_gate() {
        let inst = InstrumentId::new();
        let mut st = settings();
        st.new_strategy_status = "shadow".into();
        st.promotion.min_shadow_days = 1.0 / 24.0; // one hour
        st.promotion.min_horizons = 2;
        let mut e = Engine::new(st);
        for s in 0..3_600 {
            e.on_trade(inst, T0 - 3_600_000 + s * 1000, PX, 10.0);
        }
        e.on_signal(&signal("s", inst, 1.0), T0);
        assert!(e.tick(T0 + 1000).is_empty(), "shadow strategies place no orders");
        assert_eq!(e.strategy("s").unwrap().status, StrategyStatus::Shadow);
        // the virtual book is long 10 000 EUR (one asset, full sleeve): a 1 % move is +100 EUR
        e.on_trade(inst, T0 + 1500, PX * 1.01, 1.0);
        e.tick(T0 + 2000);
        assert!((e.strategy("s").unwrap().pnl_eur - 100.0).abs() < 1e-6, "{}", e.strategy("s").unwrap().pnl_eur);
        assert_eq!(e.strategy("s").unwrap().sleeve_eur, 10_000.0);
        // horizons: 5x1s signals every 5 s advance the counter
        let mut sig = signal("s", inst, 1.0);
        sig.as_of = Some(ts_of(T0 + 5_000));
        e.on_signal(&sig, T0 + 5_000);
        sig.as_of = Some(ts_of(T0 + 10_000));
        e.on_signal(&sig, T0 + 10_000);
        assert_eq!(e.strategy("s").unwrap().horizons_completed, 2);
        // gate: shadow time short and no deflated Sharpe → refused with the missing list
        let err = e.strategy_action("s", StrategyAction::Promote, &Evidence::default(), T0 + 10_000).unwrap_err();
        assert!(err.starts_with("conflict: promotion gate not green"), "{err}");
        assert!(err.contains("min_shadow_days") && err.contains("deflated_sharpe"), "{err}");
        assert_eq!(e.strategy("s").unwrap().status, StrategyStatus::Shadow);
        // an hour later, with a DSR row: still refused without the statistic …
        let later = T0 + 3_600_000 + 1;
        let cl = e.checklist("s", &Evidence::default(), later).unwrap();
        assert_eq!(cl.missing, vec!["deflated_sharpe"]);
        // … and promoted with it; the strategy is allocated from the next tick
        let ev = Evidence { deflated_sharpe: Some(0.4), pbo: None };
        let st = e.strategy_action("s", StrategyAction::Promote, &ev, later).unwrap();
        assert_eq!(st.status, StrategyStatus::Active);
        assert_eq!(st.last_reason, "promoted: gate green");
        e.on_trade(inst, later, PX, 10_000.0); // refill the (now empty) hourly volume window
        e.on_signal(&signal("s", inst, 1.0), later);
        assert_eq!(approved(&e.tick(later + 1000)).len(), 1, "promoted strategy trades");
        // health halts block the gate for other shadow strategies
        e.strategy_action("s", StrategyAction::Shadow, &Evidence::default(), later + 2000).unwrap();
        e.on_source_health("yahoo", "down", "stale", later + 2000);
        assert_eq!(
            e.checklist("s", &Evidence { deflated_sharpe: Some(1.0), pbo: None }, later + 2000).unwrap().missing,
            vec!["min_shadow_days", "min_horizons", "no_health_halts"]
        );
    }

    #[test]
    fn source_health_halts_are_tracked_per_source() {
        let inst = InstrumentId::new();
        let mut e = warm(inst);
        e.on_signal(&signal("a", inst, 1.0), T0);
        let buy = approved(&e.tick(T0 + 1000));
        fill(&mut e, &buy[0], PX, T0 + 1100);
        assert!(e.on_source_health("yahoo", "down", "no data for 90 s", T0 + 2000));
        assert!(!e.on_source_health("yahoo", "down", "", T0 + 2001), "already halted");
        assert!(e.halted());
        let flat = approved(&e.tick(T0 + 3000));
        assert_eq!(flat.len(), 1, "health halt flattens");
        assert_eq!(flat[0].side, Side::Sell as i32);
        // the ingestor's own risk.halt.source:<id> lands in the same table
        e.on_halt(
            &Halt { scope: "source:hn".into(), reason: "down".into(), manual: false, ..Default::default() },
            T0 + 3500,
        );
        let scopes: Vec<String> = e.state(T0 + 3500).active_halts.iter().map(|h| h.scope.clone()).collect();
        assert_eq!(scopes, vec!["source:hn", "source:yahoo"]);
        assert!(e.on_source_health("yahoo", "ok", "", T0 + 4000));
        assert!(e.halted(), "hn is still down");
        e.on_resume("source:hn");
        assert!(!e.halted());
        assert!(e.state(T0 + 4000).active_halts.is_empty());
        // a manual source halt is never cleared by a health frame, only by resume
        e.on_halt(
            &Halt { scope: "source:yahoo".into(), reason: "operator".into(), manual: true, ..Default::default() },
            T0 + 5000,
        );
        assert!(!e.on_source_health("yahoo", "ok", "", T0 + 5001));
        assert!(e.halted());
        e.on_resume("source");
        assert!(!e.halted());
        assert_eq!(e.counters["halts.received"], 3.0);
    }

    #[test]
    fn portfolio_flatten_pauses_all_and_limits_only_tighten() {
        let inst = InstrumentId::new();
        let mut e = warm(inst);
        e.on_signal(&signal("a", inst, 1.0), T0);
        e.on_signal(&signal("b", inst, 0.5), T0);
        let buy = approved(&e.tick(T0 + 1000));
        fill(&mut e, &buy[0], PX, T0 + 1100);
        let mut ids = e.flatten_all(T0 + 2000);
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
        let out = e.tick(T0 + 3000);
        let flat = approved(&out);
        assert_eq!(flat.len(), 1);
        assert!((flat[0].quantity - 50.0).abs() < 1e-6);
        assert_eq!(audits(&out, "strategy.flattened").len(), 2);
        assert!(e.strategy_ids().iter().all(|id| e.strategy(id).unwrap().status == StrategyStatus::Paused));
        assert!(e.flatten_all(T0 + 4000).is_empty(), "nothing active any more");
        fill(&mut e, &flat[0], PX, T0 + 3100);

        // limits: loosening is refused field by field, tightening applies immediately
        let err = e.set_limits(&json!({"max_drawdown": 0.5, "leverage_cap": {"fx": 9.0}})).unwrap_err();
        assert!(
            err.starts_with("loosen: max_drawdown: 0.1 → 0.5 would loosen; leverage_cap.fx: 3 → 9 would loosen"),
            "{err}"
        );
        assert_eq!(e.limits().max_drawdown, 0.1, "refused change leaves limits untouched");
        assert!(e.set_limits(&json!({"nope": 1})).unwrap_err().starts_with("invalid: unknown limit field"));
        assert!(e.set_limits(&json!(3)).unwrap_err().starts_with("invalid:"));
        assert!(e.set_limits(&json!({"leverage_cap": 2})).unwrap_err().starts_with("invalid:"));
        let new = e.set_limits(&json!({"max_position_frac": 0.01, "leverage_cap": {"fx": 1.0}})).unwrap();
        assert_eq!(new.max_position_frac, 0.01);
        assert_eq!(new.leverage_cap[&fa_core::AssetClass::Fx], 1.0);
        assert_eq!(new.leverage_cap[&fa_core::AssetClass::Equity], 1.5, "other classes keep their caps");
        assert_eq!(e.limits(), &new);
        e.strategy_action("a", StrategyAction::Resume, &Evidence::default(), T0 + 5000).unwrap();
        e.on_signal(&signal("a", inst, 1.0), T0 + 5000);
        let buy = approved(&e.tick(T0 + 6000));
        assert!((buy[0].quantity * PX - 1_000.0).abs() < 1e-6, "new 1 % position cap binds: {}", buy[0].quantity * PX);
    }

    // ------------------------------------------------------------ D-061: refdata, FX, sub-books

    use fa_warehouse::{
        fx::{FxRate, FxTable},
        venues::{InstrumentRef, RefTable},
    };

    fn iref(
        id: InstrumentId,
        class: fa_core::AssetClass,
        venue: &str,
        ccy: &str,
        status: &str,
        source: &str,
    ) -> InstrumentRef {
        InstrumentRef {
            id,
            asset_class: Some(class),
            asset_class_raw: format!("{class:?}").to_lowercase(),
            kind: "equity".into(),
            venue: Some(venue.into()),
            venue_id: None,
            quote_ccy: Some(ccy.into()),
            status: status.into(),
            sources: if source.is_empty() { vec![] } else { vec![source.into()] },
        }
    }

    fn usd(days_before_t0: i64, rate: f64) -> FxRate {
        FxRate {
            ccy: "USD".into(),
            as_of: at(T0 - days_before_t0 * MS_PER_DAY),
            rate_to_eur: rate,
            source: "ecb_daily".into(),
        }
    }

    fn warm_many(insts: &[InstrumentId], s: Settings) -> Engine {
        let mut e = Engine::new(s);
        for s in 0..3_600 {
            for i in insts {
                e.on_trade(*i, T0 - 3_600_000 + s * 1000, PX, 10.0);
            }
        }
        e
    }

    fn multi_signal(id: &str, insts: &[InstrumentId], mean: f64) -> ProtoDist {
        ProtoDist {
            exposures: insts
                .iter()
                .map(|i| AssetExposure { instrument_id: i.to_string(), mean, stddev: 0.0 })
                .collect(),
            ..signal(id, insts[0], mean)
        }
    }

    #[test]
    fn venue_caps_use_reference_data_and_unknowns_fall_back() {
        let (a, b, c, u) = (InstrumentId::new(), InstrumentId::new(), InstrumentId::new(), InstrumentId::new());
        let mut st = settings();
        st.limits.max_position_frac = 0.2; // 20k each
        st.limits.max_class_frac = 1.0;
        st.limits.leverage_cap.insert(fa_core::AssetClass::Equity, 5.0);
        st.limits.max_venue_frac = 0.2; // 20k per venue
        st.limits.liquidity_k = 1.0;
        let mut e = warm_many(&[a, b, c, u], st);
        e.set_refdata(RefTable::from_refs([
            iref(a, fa_core::AssetClass::Equity, "xetr", "EUR", "active", "lemon"),
            iref(b, fa_core::AssetClass::Equity, "xetr", "EUR", "active", "lemon"),
            iref(c, fa_core::AssetClass::Equity, "yahoo", "EUR", "provisional", "yahoo"),
        ]));
        e.on_signal(&multi_signal("s", &[a, b, c, u], 1.0), T0);
        let ints = approved(&e.tick(T0 + 1000));
        let eur =
            |i: InstrumentId| ints.iter().find(|x| x.instrument_id == i.to_string()).map_or(0.0, |x| x.quantity * PX);
        // one strategy, sleeve 50k over 4 assets → 12.5k each. a, b on xetr: 25k > 20k venue cap →
        // 10k each; c provisional: cap 0.2 × 0.5 = 10k; u: no reference data → default class,
        // empty venue bucket, uncapped at 12.5k
        assert!((eur(a) - 10_000.0).abs() < 1e-6 && (eur(b) - 10_000.0).abs() < 1e-6, "{} {}", eur(a), eur(b));
        assert!((eur(c) - 10_000.0).abs() < 1e-6, "provisional cap: {}", eur(c));
        assert!((eur(u) - 12_500.0).abs() < 1e-6, "unknown: {}", eur(u));
        assert_eq!(e.counters["refdata.unknown"], 1.0);
        let v = e.exposure_view();
        assert!((v["target_by_venue"]["xetr"].as_f64().unwrap() - 20_000.0).abs() < 1e-6, "{v}");
        assert!((v["target_by_venue"]["yahoo"].as_f64().unwrap() - 10_000.0).abs() < 1e-6);
        for i in &ints {
            fill(&mut e, i, PX, T0 + 1100);
        }
        let v = e.exposure_view();
        assert!((v["by_venue"]["xetr"].as_f64().unwrap() - 20_000.0).abs() < 1e-6, "{v}");
        assert!((v["by_venue"]["unknown"].as_f64().unwrap() - 12_500.0).abs() < 1e-6);
        assert!((v["provisional_eur"].as_f64().unwrap() - 10_000.0).abs() < 1e-6);
        // reloading reference data that now knows u clears the unknown gauge
        let mut refs: Vec<InstrumentRef> = e.refdata().iter().cloned().collect();
        refs.push(iref(u, fa_core::AssetClass::Crypto, "kraken", "EUR", "active", "kraken"));
        e.set_refdata(RefTable::from_refs(refs));
        assert_eq!(e.counters["refdata.unknown"], 0.0);
    }

    #[test]
    fn marks_and_fills_convert_quote_currency_to_eur_as_of_the_event() {
        let inst = InstrumentId::new();
        let mut e = Engine::new(settings());
        e.set_refdata(RefTable::from_refs([iref(
            inst,
            fa_core::AssetClass::Equity,
            "yahoo",
            "USD",
            "active",
            "yahoo",
        )]));
        assert!(e.set_fx(FxTable::from_rates([usd(3, 0.80), usd(1, 0.90)]), T0).is_empty());
        assert_eq!(e.fx_table().len(), 2);
        // a trade two days before T0 uses the rate as of then (0.80); at T0 the 0.90 one
        e.on_trade(inst, T0 - 2 * MS_PER_DAY, 100.0, 1.0);
        assert!((e.marks[&inst] - 80.0).abs() < 1e-9);
        for s in 0..3_600 {
            e.on_trade(inst, T0 - 3_600_000 + s * 1000, 100.0, 10.0);
        }
        assert!((e.marks[&inst] - 90.0).abs() < 1e-9, "USD 100 × 0.90");
        let intent = OrderIntent {
            intent_id: "usd-1".into(),
            account_id: "paper".into(),
            instrument_id: inst.to_string(),
            side: Side::Buy as i32,
            quantity: 10.0,
            mode: TradingMode::Paper as i32,
            strategy_ids: vec!["s".into()],
            ..Default::default()
        };
        e.on_signal(&signal("s", inst, 0.0), T0);
        e.on_intent_seen(&intent, true, T0);
        let ev = OrderEvent {
            intent_id: "usd-1".into(),
            venue_order_id: "v".into(),
            kind: order_event::Kind::Fill as i32,
            fill_price: 100.0,
            fill_quantity: 10.0,
            fee: 2.0,
            fee_currency: "EUR".into(),
            event_time: Some(ts_of(T0 + 10)),
            arrival_price: 99.9,
            slippage_bps: 10.0,
            ..Default::default()
        };
        e.on_order_event(&ev, T0 + 10);
        // cash −(10 × 100 × 0.90) − 2 EUR fee; equity marks at 90 EUR
        assert!((e.ledger().cash_eur - (100_000.0 - 902.0)).abs() < 1e-9, "{}", e.ledger().cash_eur);
        assert!((e.equity() - (100_000.0 - 2.0)).abs() < 1e-9);
        let b = e.take_booked_fills().pop().unwrap();
        assert_eq!(
            (b.fill.fx_to_eur, b.details.ccy.as_str(), b.details.fx_source.as_str()),
            (0.90, "USD", "ecb_daily")
        );
        assert_eq!((b.details.arrival_price, b.details.slippage_bps), (Some(99.9), Some(10.0)));
        let view = e.strategies_view(&BTreeMap::new(), T0 + 20);
        assert_eq!(view["strategies"][0]["slippage_bps_mean"], 10.0);
        assert_eq!(view["strategies"][0]["book"]["positions"][inst.to_string()], 10.0);
        // EUR instruments and instruments without reference data stay at 1.0
        let eur_inst = InstrumentId::new();
        e.on_trade(eur_inst, T0, 50.0, 1.0);
        assert_eq!(e.marks[&eur_inst], 50.0);
        assert!(!e.counters.contains_key("fx.stale"));
    }

    #[test]
    fn stale_fx_is_flagged_and_optionally_halts_reduce_only() {
        let inst = InstrumentId::new();
        let mut st = settings();
        st.fx.halt_on_stale = true;
        let mut e = Engine::new(st);
        e.set_refdata(RefTable::from_refs([iref(
            inst,
            fa_core::AssetClass::Equity,
            "yahoo",
            "USD",
            "active",
            "yahoo",
        )]));
        e.set_fx(FxTable::from_rates([usd(5, 0.9)]), T0); // 5 days old > 72 h
        let out = e.on_trade(inst, T0, 100.0, 1.0);
        assert!((e.marks[&inst] - 90.0).abs() < 1e-9, "last known rate is still used");
        assert_eq!(e.counters["fx.stale"], 1.0);
        let Out::Halt(h) = &out[0] else { panic!("{out:?}") };
        assert_eq!(h.scope, "fx");
        assert!(h.reason.contains("USD"), "{}", h.reason);
        assert!(e.halted() && e.risk.reduce_only(), "fx halt is reduce-only");
        assert_eq!(e.fx_status().stale, vec!["USD"]);
        assert!(e.state(T0).active_halts.iter().any(|h| h.scope == "fx"));
        assert!(e.on_trade(inst, T0 + 1, 100.0, 1.0).is_empty(), "raised once");
        // a fresh rate clears it on the next table load
        let out = e.set_fx(FxTable::from_rates([usd(5, 0.9), usd(0, 0.91)]), T0 + 2);
        assert!(matches!(&out[..], [Out::Resume(h)] if h.scope == "fx"), "{out:?}");
        assert!(!e.halted());
        // without halt_on_stale only the flag and counter
        let mut e2 = Engine::new(settings());
        e2.set_refdata(RefTable::from_refs([iref(
            inst,
            fa_core::AssetClass::Equity,
            "yahoo",
            "USD",
            "active",
            "yahoo",
        )]));
        e2.set_fx(FxTable::from_rates([usd(5, 0.9)]), T0);
        assert!(e2.on_trade(inst, T0, 100.0, 1.0).is_empty());
        assert!(!e2.halted());
        assert_eq!(e2.fx_status().stale, vec!["USD"]);
    }

    #[test]
    fn sub_books_survive_a_restart_through_their_records() {
        let inst = InstrumentId::new();
        let mut e = warm(inst);
        e.on_signal(&signal("a", inst, 1.0), T0);
        let buy = approved(&e.tick(T0 + 1000));
        fill(&mut e, &buy[0], PX, T0 + 1100);
        e.tick(T0 + 1500);
        e.on_trade(inst, T0 + 2000, 98.0, 1.0); // −100 EUR on 50 units
        e.on_signal(&signal("a", inst, 1.0), T0 + 2000);
        e.tick(T0 + 2500);
        let before = e.strategies_view(&BTreeMap::new(), T0 + 3000);
        let a = &before["strategies"][0];
        assert!((a["pnl_eur"].as_f64().unwrap() + 100.0).abs() < 1e-6, "{a}");
        // persist (JSON, as written to ledger.strategy_state.book) and restore into a fresh engine
        let rows: Vec<(StrategyState, Option<SubBookRecord>)> = e
            .strategy_rows()
            .into_iter()
            .map(|(s, b)| (s, Some(serde_json::from_value(serde_json::to_value(&b).unwrap()).unwrap())))
            .collect();
        let mut r = Engine::new(settings());
        r.seed_ledger(e.ledger().clone());
        r.seed_strategies(rows);
        assert_eq!(r.counters["strategies.restored"], 1.0);
        let after = r.strategies_view(&BTreeMap::new(), T0 + 3000);
        let b = &after["strategies"][0];
        for k in ["pnl_eur", "peak_pnl_eur", "drawdown", "max_drawdown"] {
            assert_eq!(a[k], b[k], "{k}");
        }
        assert_eq!(a["book"]["positions"], b["book"]["positions"]);
        assert_eq!(a["book"]["history_points"], b["book"]["history_points"]);
        // no spurious P&L while the restored book waits for marks, then it continues from the mark
        r.on_signal(&signal("a", inst, 1.0), T0 + 3000);
        r.tick(T0 + 3500);
        assert_eq!(r.strategy("a").unwrap().pnl_eur, a["pnl_eur"].as_f64().unwrap());
        r.on_trade(inst, T0 + 4000, 98.0, 1.0);
        r.tick(T0 + 4500);
        r.on_trade(inst, T0 + 5000, 96.0, 1.0); // another −100
        r.tick(T0 + 5500);
        assert!((r.strategy("a").unwrap().pnl_eur + 200.0).abs() < 1e-6, "{}", r.strategy("a").unwrap().pnl_eur);
        assert!(r.strategy("a").unwrap().drawdown() > a["drawdown"].as_f64().unwrap());
    }

    #[test]
    fn source_health_halt_flattens_only_that_sources_instruments() {
        let (y, h) = (InstrumentId::new(), InstrumentId::new());
        let mut e = warm_many(&[y, h], settings());
        e.set_refdata(RefTable::from_refs([
            iref(y, fa_core::AssetClass::Equity, "yahoo", "EUR", "active", "yahoo"),
            iref(h, fa_core::AssetClass::Equity, "lemon", "EUR", "active", "lemon"),
        ]));
        e.on_signal(&multi_signal("s", &[y, h], 1.0), T0);
        for i in approved(&e.tick(T0 + 1000)) {
            fill(&mut e, &i, PX, T0 + 1100);
        }
        assert!(e.on_source_health("yahoo", "down", "stale", T0 + 2000));
        e.on_signal(&multi_signal("s", &[y, h], 1.0), T0 + 2000);
        let out = approved(&e.tick(T0 + 3000));
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!((out[0].instrument_id.as_str(), out[0].side), (y.to_string().as_str(), Side::Sell as i32));
        // foreign orders on the unaffected instrument still pass, on the halted one they don't
        // (a sell: h already sits at its position cap, a buy would fail the pre-trade check)
        let foreign = |id: &str, inst: InstrumentId| OrderIntent {
            intent_id: id.into(),
            account_id: "paper".into(),
            instrument_id: inst.to_string(),
            side: Side::Sell as i32,
            quantity: 1.0,
            mode: TradingMode::Paper as i32,
            ..Default::default()
        };
        assert!(matches!(&e.on_intent_seen(&foreign("f1", h), false, T0 + 3100)[0], Out::Approved(_)));
        assert!(matches!(&e.on_intent_seen(&foreign("f2", y), false, T0 + 3100)[0], Out::Rejected { .. }));
        assert_eq!(e.state(T0 + 3100).active_halts[0].scope, "source:yahoo");
        // an instrument without a single known source makes the rule ambiguous → flatten all
        let u = InstrumentId::new();
        e.on_trade(u, T0 + 3200, PX, 1_000_000.0);
        e.on_signal(&multi_signal("s", &[y, h, u], 1.0), T0 + 3200);
        let out = approved(&e.tick(T0 + 4000));
        assert!(out.iter().any(|i| i.instrument_id == h.to_string() && i.side == Side::Sell as i32), "{out:?}");
        // a bare `source` halt flattens everything as well; bare resume clears every source halt
        e.on_resume("source");
        assert!(!e.halted());
        e.on_halt(
            &Halt { scope: "source".into(), reason: "all".into(), manual: true, ..Default::default() },
            T0 + 5000,
        );
        assert_eq!(e.state(T0 + 5000).active_halts[0].scope, "source");
        assert!(e.instrument_blocked(&h));
    }

    #[test]
    fn payout_reserve_and_allocator_switch() {
        let inst = InstrumentId::new();
        let mut e = warm(inst);
        e.set_reserved_payout(20_000.0);
        assert_eq!(e.state(T0).deployable_eur, 80_000.0);
        e.on_signal(&signal("a", inst, 1.0), T0);
        e.on_signal(&signal("b", inst, 1.0), T0);
        e.tick(T0 + 1000);
        assert!((e.strategy("a").unwrap().sleeve_eur - 40_000.0).abs() < 1e-6, "sleeves sized on equity − reserve");
        // the configured quota may only rise; the ledger row binds when larger
        assert!(e.set_limits(&json!({"daily_payout_quota_eur": 30_000.0})).is_ok());
        assert_eq!(e.payout_reserve(), 30_000.0);
        assert!(e.set_limits(&json!({"daily_payout_quota_eur": 10_000.0})).unwrap_err().starts_with("loosen:"));
        // allocator switch: allowed in any direction, cap bounded by risk.toml
        let spec = e
            .set_allocator(&json!({"name": "weighted_blend", "params": {"w_eq": 1, "w_rp": 1, "w_k": 0, "cap": 0.9}}))
            .unwrap();
        assert_eq!(spec["name"], "weighted_blend");
        assert_eq!(spec["params"]["cap"], 0.5, "never above max_strategy_weight");
        assert!(spec["genome"].as_str().unwrap().starts_with("alloc:"));
        let g = spec["genome"].as_str().unwrap().to_owned();
        assert_eq!(e.set_allocator(&json!({"genome": g})).unwrap()["name"], "weighted_blend");
        assert_eq!(e.set_allocator(&json!({"name": "equal_weight"})).unwrap()["name"], "equal_weight");
        assert!(e.set_allocator(&json!({"name": "oracle"})).unwrap_err().starts_with("invalid:"));
        assert!(
            e.set_allocator(&json!({"name": "equal_weight", "params": {"nope": 1}})).unwrap_err().contains("unknown")
        );
        assert_eq!(e.counters["allocator.switches"], 3.0);
        assert_eq!(e.exposure_view()["allocator"]["name"], "equal_weight");
    }

    fn production() -> Settings {
        Settings::parse(include_str!("../risk.toml")).unwrap()
    }

    fn events<'a>(out: &'a [Out], kind: &str) -> Vec<&'a Value> {
        out.iter()
            .filter_map(|o| match o {
                Out::RiskEvent { kind: k, detail, .. } if k == kind => Some(detail),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn feed_guard_freezes_stale_instruments_and_holds_bad_ticks() {
        let inst = InstrumentId::new();
        let mut e = warm_many(&[inst], production());
        e.on_signal(&signal("a", inst, 1.0), T0);
        let buy = approved(&e.tick(T0 + 1000));
        assert_eq!(buy.len(), 1, "fresh feed trades");
        fill(&mut e, &buy[0], PX, T0 + 1100);
        // a bad tick is held back: no new mark, instrument frozen (reduce-only), event raised
        let out = e.on_trade(inst, T0 + 1500, 300.0, 1.0);
        assert_eq!(events(&out, "feed.jump").len(), 1);
        assert_eq!(e.marks[&inst], PX, "mark unchanged");
        e.on_signal(&signal("a", inst, -1.0), T0 + 1600);
        let out = e.tick(T0 + 2000);
        let flip = approved(&out);
        assert_eq!(flip.len(), 1, "reduce-only: closes the long but does not flip short");
        assert!((flip[0].quantity * PX - 5_000.0).abs() < 1e-6);
        assert_eq!(events(&out, "feed.frozen").len(), 1);
        fill(&mut e, &flip[0], PX, T0 + 2100);
        // a trade back near the old mark discards the jump
        e.on_trade(inst, T0 + 2200, 100.5, 1.0);
        assert_eq!(e.counters["feed.jump_discarded"], 1.0);
        // stale: 1 s strategy, last trade 2.2 s → after 3 s + it is frozen (no new short)
        e.on_signal(&signal("a", inst, -1.0), T0 + 9_000);
        assert!(approved(&e.tick(T0 + 9_000)).is_empty(), "stale feed: no new risk");
        // a manual risk-increasing order on a stale feed is refused too
        let manual = OrderIntent {
            intent_id: "m".into(),
            account_id: "paper".into(),
            instrument_id: inst.to_string(),
            side: Side::Buy as i32,
            quantity: 10.0,
            mode: TradingMode::Paper as i32,
            ..Default::default()
        };
        let out = e.on_intent_seen(&manual, false, T0 + 9_000);
        assert!(matches!(&out[0], Out::Rejected { event, .. } if event.reason.contains("feed")), "{out:?}");
        assert_eq!(events(&out, "pretrade.reject").len(), 1);
        // fresh again → trades
        e.on_trade(inst, T0 + 9_500, 100.0, 1.0);
        e.on_signal(&signal("a", inst, -1.0), T0 + 9_500);
        assert_eq!(approved(&e.tick(T0 + 10_000)).len(), 1);
    }

    #[test]
    fn a_kill_switch_flattens_through_a_frozen_feed() {
        let inst = InstrumentId::new();
        let mut e = warm_many(&[inst], production());
        e.on_signal(&signal("a", inst, 1.0), T0);
        let buy = approved(&e.tick(T0 + 1000));
        fill(&mut e, &buy[0], PX, T0 + 1100);
        e.on_halt(&Halt { scope: "portfolio".into(), reason: "op".into(), since: None, manual: true }, T0 + 50_000);
        let flat = approved(&e.tick(T0 + 60_000)); // feed stale for a minute
        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].side, Side::Sell as i32);
    }

    #[test]
    fn pre_trade_checks_gate_manual_orders_and_clips_are_reported() {
        let inst = InstrumentId::new();
        let mut e = warm(inst);
        let order = |id: &str, q: f64, lp: Option<f64>| OrderIntent {
            intent_id: id.into(),
            account_id: "paper".into(),
            instrument_id: inst.to_string(),
            side: Side::Buy as i32,
            quantity: q,
            limit_price: lp,
            mode: TradingMode::Paper as i32,
            ..Default::default()
        };
        assert!(matches!(&e.on_intent_seen(&order("ok", 10.0, None), false, T0)[0], Out::Approved(_)));
        for (id, q, lp, needle) in [
            ("big", 200.0, None, "max_order_frac"),
            ("cap", 60.0, None, "position cap"),
            ("far", 1.0, Some(150.0), "away from the mark"),
        ] {
            let out = e.on_intent_seen(&order(id, q, lp), false, T0);
            assert!(matches!(&out[0], Out::Rejected { event, .. } if event.reason.contains(needle)), "{id}: {out:?}");
        }
        assert_eq!(e.counters["rejected.pretrade"], 3.0);
        // the allocator's target 50 000 EUR is clipped to 5 000: reported once per minute
        e.on_signal(&signal("a", inst, 1.0), T0);
        let out = e.tick(T0 + 1000);
        assert!(matches!(out[0], Out::Intent(_)), "orders first, events after");
        let clip = events(&out, "limit.clip");
        assert_eq!(clip.len(), 1);
        assert!((clip[0]["capped_eur"].as_f64().unwrap() - 5_000.0).abs() < 1e-6);
        e.on_signal(&signal("a", inst, 1.0), T0 + 2000);
        assert!(events(&e.tick(T0 + 2000), "limit.clip").is_empty(), "rate-limited");
    }

    struct NanAllocator;
    impl Allocator for NanAllocator {
        fn id(&self) -> &str {
            "nan"
        }
        fn allocate(&self, stats: &[fa_allocator::StrategyStats]) -> fa_allocator::Allocation {
            stats.iter().map(|s| (s.strategy_id.clone(), f64::NAN)).collect()
        }
    }

    #[test]
    fn allocator_nan_falls_back_to_the_previous_allocation_and_the_tournament_compares() {
        let inst = InstrumentId::new();
        let mut e = warm(inst);
        e.on_signal(&signal("a", inst, 1.0), T0);
        e.tick(T0 + 1000);
        assert_eq!(e.last_alloc["a"], 0.5);
        e.allocator = Box::new(NanAllocator);
        e.on_signal(&signal("a", inst, 1.0), T0 + 1500);
        let out = e.tick(T0 + 2000);
        assert_eq!(events(&out, "allocator.fallback").len(), 1);
        assert_eq!(e.last_alloc["a"], 0.5, "previous weights kept");
        assert_eq!(e.counters["allocator.fallbacks"], 1.0);
        // tournament: every entry saw the same ticks
        for k in 3..10 {
            e.on_trade(inst, T0 + k * 1000, PX + k as f64, 1.0);
            e.on_signal(&signal("a", inst, 1.0), T0 + k * 1000);
            e.tick(T0 + k * 1000 + 500);
        }
        let t = e.tournament_view();
        let entries = t["entries"].as_array().unwrap();
        let names: Vec<&str> = entries.iter().map(|r| r["name"].as_str().unwrap()).collect();
        assert_eq!(names, vec!["active", "equal_weight", "fractional_kelly", "risk_parity", "thompson_rp"]);
        let steps: Vec<u64> = entries.iter().map(|r| r["steps"].as_u64().unwrap()).collect();
        assert!(steps.windows(2).all(|w| w[0] == w[1]) && steps[0] > 0, "{steps:?}");
        let eq = entries.iter().find(|r| r["name"] == "equal_weight").unwrap();
        assert!(eq["cumulative_return"].as_f64().unwrap() > 0.0, "rising price, long strategy: {eq}");
    }

    #[test]
    fn the_kill_switch_and_peak_survive_a_restart_through_the_risk_record() {
        let inst = InstrumentId::new();
        let mut e = warm(inst);
        e.on_halt(&Halt { scope: "portfolio".into(), reason: "operator".into(), since: None, manual: true }, T0);
        e.on_source_health("yahoo", "down", "stale", T0);
        assert!(e.take_risk_dirty());
        let rec: RiskRecord = serde_json::from_value(serde_json::to_value(e.risk_record()).unwrap()).unwrap();
        let mut r = Engine::new(settings());
        r.seed_ledger(e.ledger().clone());
        assert!(!r.halted(), "a fresh engine is armed …");
        r.restore_risk(rec);
        assert!(r.halted(), "… the restored one keeps the kill switch");
        let scopes: Vec<String> = r.state(T0).active_halts.iter().map(|h| h.scope.clone()).collect();
        assert_eq!(scopes, vec!["portfolio", "source:yahoo"]);
        assert_eq!(r.limits(), e.limits(), "limits always come from risk.toml");
        r.on_resume("portfolio");
        r.on_resume("source");
        assert!(!r.halted());
    }

    fn member(id: &str, sharpe: f64) -> FrontMember {
        FrontMember {
            strategy_id: id.into(),
            genome: format!(
                "{{ id = \"{id}\", genome_id = \"{id}\", timescale = \"1s\", steps = 1, lookback_bars = 8, signal = \"1\", uncertainty = \"0\" }}"
            ),
            generation: "g1".into(),
            timescale: "1s".into(),
            step_secs: 1.0,
            score: None,
            selection: fa_risk::pipeline::Selection {
                deflated_sharpe: Some(0.99),
                sharpe: Some(sharpe),
                n_obs: Some(1000),
                pbo: Some(0.1),
                window_ms: None,
            },
        }
    }

    #[test]
    fn pipeline_transitions_build_the_live_set_and_retire() {
        let inst = InstrumentId::new();
        let mut e = warm(inst);
        assert!(e.admit_candidate(&member("gp-1", 0.01), T0));
        assert!(!e.admit_candidate(&member("gp-1", 0.01), T0), "idempotent");
        assert_eq!(e.strategy("gp-1").unwrap().status, StrategyStatus::Candidate);
        assert!(e.live_set_genomes().is_empty(), "candidates do not run");
        e.record_holdout_request("gp-1", "h1", "job-1", T0);
        e.holdout_job_failed("gp-1", "insufficient market data");
        let p = e.strategy("gp-1").unwrap().pipeline.clone().unwrap();
        assert_eq!((p.holdout_attempts, p.holdout_requested_ms), (1, None));
        assert!(p.note.contains("insufficient"));
        let h = fa_risk::pipeline::HoldoutOutcome {
            deflated_sharpe: Some(0.99),
            sharpe: Some(0.02),
            n_obs: 500,
            ..Default::default()
        };
        let none = Evidence::default();
        let (action, _) = e.apply_decision("gp-1", &Decision::HoldoutPassed(h), &none, T0).unwrap().unwrap();
        assert_eq!(action, "strategy.holdout_passed");
        e.apply_decision("gp-1", &Decision::Admit, &none, T0 + 1).unwrap().unwrap();
        assert_eq!(e.strategy("gp-1").unwrap().status, StrategyStatus::Shadow);
        assert_eq!(e.live_set_genomes().len(), 1);
        // the shadow strategy is tracked but not allocated; the gate needs every statistic
        e.on_signal(&signal("gp-1", inst, 1.0), T0 + 10);
        assert!(approved(&e.tick(T0 + 1000)).is_empty());
        let cl = e.checklist("gp-1", &Evidence { deflated_sharpe: Some(0.99), pbo: Some(0.1) }, T0 + 1000).unwrap();
        assert!(cl.missing.contains(&"shadow_vs_backtest".to_owned()), "{cl:?}");
        assert!(!cl.missing.contains(&"pbo".to_owned()));
        // promotion refused while the gate is red
        assert!(e.apply_decision("gp-1", &Decision::Promote, &none, T0 + 1000).unwrap_err().contains("not green"));
        // retirement drops it from the live set and its signals are ignored
        let (action, payload) =
            e.apply_decision("gp-1", &Decision::Retire("test".into()), &none, T0 + 2000).unwrap().unwrap();
        assert_eq!((action.as_str(), payload["to"].as_str()), ("strategy.retired", Some("retired")));
        assert!(e.live_set_genomes().is_empty());
        assert!(e.pipeline_ids().is_empty());
        e.on_signal(&signal("gp-1", inst, 1.0), T0 + 3000);
        assert!(approved(&e.tick(T0 + 3000)).is_empty());
        assert!(e.strategy_action("gp-1", StrategyAction::Resume, &none, T0 + 3000).unwrap_err().contains("retired"));
    }

    #[test]
    fn shadow_consistency_uses_hourly_book_points_since_admission() {
        let inst = InstrumentId::new();
        let mut e = warm(inst);
        e.admit_candidate(&member("gp-2", 1e-4), T0);
        e.apply_decision("gp-2", &Decision::HoldoutPassed(Default::default()), &Evidence::default(), T0).unwrap();
        e.apply_decision("gp-2", &Decision::Admit, &Evidence::default(), T0).unwrap();
        // 30 hours of a rising price with the shadow strategy long → hourly P&L climbs
        for h in 1..=30 {
            let t = T0 + h * MS_PER_HOUR;
            e.on_trade(inst, t, PX * (1.0 + 0.001 * h as f64), 1_000.0);
            e.on_signal(&signal("gp-2", inst, 1.0), t);
            e.tick(t + 1);
            e.on_trade(inst, t + 2, PX * (1.0 + 0.001 * h as f64 + 0.0005), 1_000.0);
            e.tick(t + 3);
        }
        let cl = e
            .checklist("gp-2", &Evidence { deflated_sharpe: Some(0.99), pbo: Some(0.1) }, T0 + 31 * MS_PER_HOUR)
            .unwrap();
        let item = cl.items.iter().find(|i| i.name == "shadow_vs_backtest").unwrap();
        assert_eq!(item.pass, Some(true), "{item:?}");
        assert!(item.detail.contains("over"), "{}", item.detail);
        // calibration has hourly equity points too
        let cal = e.calibration(&fa_risk::calibrate::CalibrationRules { min_hours: 10, ..Default::default() });
        assert!(cal.hours >= 10, "{cal:?}");
    }
}
