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
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use fa_allocator::{Allocator, PerformanceTracker};
use fa_core::{Horizon, InstrumentId, Timescale};
use fa_ledger::{Fill, Ledger};
use fa_proto::{
    Timestamp,
    v1::{
        AssetExposure, ExposureDistribution as ProtoDist, Halt, OrderEvent, OrderIntent, RiskState, Side, TradingMode,
        order_event,
    },
};
use fa_risk::{
    HaltReason, HaltScope, MarketInfo, RiskConfig, RiskEngine,
    strategy::{Checklist, GateEvidence, StrategyAction, StrategyState, StrategyStatus},
};
use fa_strategies::ExposureDistribution;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::settings::Settings;

const MAX_INTENTS: usize = 5_000;
const MS_PER_DAY: i64 = 86_400_000;

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
    /// An automatic, consequential state change for `audit.log` (per-strategy drawdown pause,
    /// flatten completion); the I/O layer persists it.
    Audit { action: String, subject: String, payload: Value },
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
}

#[derive(Debug, Clone)]
struct Pending {
    instrument: InstrumentId,
    signed_eur: f64,
    created_ms: i64,
}

#[derive(Debug, Clone)]
struct ActiveHalt {
    reason: String,
    since_ms: i64,
    manual: bool,
}

/// Attributed virtual sub-book of an active strategy (spec 40 §6.3): fills are split evenly
/// across the intent's contributors (D-047 attribution), marked to market each tick.
#[derive(Debug, Clone, Default)]
struct VirtualBook {
    cash_eur: f64,
    qty: BTreeMap<InstrumentId, f64>,
    /// Marked value at the previous tick, for the P&L increment.
    last_value_eur: f64,
}

impl VirtualBook {
    fn value(&self, marks: &BTreeMap<InstrumentId, f64>) -> f64 {
        self.cash_eur + self.qty.iter().map(|(i, q)| q * marks.get(i).copied().unwrap_or(0.0)).sum::<f64>()
    }

    /// Closes every position at the marks (the strategy leaves the allocated set).
    fn close(&mut self, marks: &BTreeMap<InstrumentId, f64>) {
        self.cash_eur = self.value(marks);
        self.qty.clear();
        self.last_value_eur = self.cash_eur;
    }
}

/// Bare `source` halts (no qualifier) are keyed by this in the per-source table.
const ANY_SOURCE: &str = "*";

pub struct Engine {
    settings: Settings,
    allocator: Box<dyn Allocator>,
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
        Self {
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
            counters: BTreeMap::new(),
        }
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

    /// Strategy states changed since the last call (for `ledger.strategy_state`).
    pub fn take_dirty_strategies(&mut self) -> Vec<StrategyState> {
        let ids = std::mem::take(&mut self.dirty);
        ids.iter().filter_map(|id| self.strategies.get(id).cloned()).collect()
    }

    /// Every strategy state (periodic full flush).
    pub fn strategy_states(&self) -> Vec<StrategyState> {
        self.strategies.values().cloned().collect()
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

    /// Restores persisted lifecycle rows at startup (rows win over the in-memory defaults).
    pub fn seed_strategies(&mut self, rows: Vec<StrategyState>) {
        for r in rows {
            self.strategies.insert(r.strategy_id.clone(), r);
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

    /// Mark + traded volume. Returns halts newly triggered by the equity move.
    pub fn on_trade(&mut self, inst: InstrumentId, at_ms: i64, price: f64, size: f64) -> Vec<Out> {
        if price <= 0.0 || !price.is_finite() {
            return Vec::new();
        }
        self.marks.insert(inst, price);
        self.first_seen_ms.entry(inst).or_insert(at_ms);
        let window = self.settings.volume_window_secs as i64 * 1000;
        let v = self.volume.entry(inst).or_default();
        v.push_back((at_ms, price * size.max(0.0)));
        while v.front().is_some_and(|(t, _)| at_ms - *t > window) {
            v.pop_front();
        }
        self.bump("trades");
        self.check_equity(at_ms)
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
        if !self.risk.trading_allowed() {
            let current = self.ledger.positions.get(&inst).map_or(0.0, |p| p.quantity);
            let reducing =
                current != 0.0 && signed.signum() != current.signum() && signed.abs() <= current.abs() + 1e-9;
            if !(self.risk.reduce_only() && reducing) {
                self.bump("rejected.halted");
                return vec![reject("trading halted")];
            }
        }
        self.bump("approved.foreign");
        vec![Out::Approved(m.clone())]
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
        let fill = Fill {
            fill_id,
            intent_id: m.intent_id.clone(),
            instrument: info.instrument,
            quantity: m.fill_quantity.copysign(info.signed_quantity),
            price: m.fill_price,
            fx_to_eur: 1.0, // EUR-priced until FX lands (D-028)
            fee_eur: m.fee,
        };
        let applied = self.ledger.apply_fill(&fill);
        if applied {
            self.bump("fills");
            self.marks.entry(info.instrument).or_insert(m.fill_price);
            self.attribute(&fill, &info.strategy_ids);
            self.booked.push(BookedFill { fill, strategy_ids: info.strategy_ids.clone(), at_ms });
        }
        self.check_equity(now_ms)
    }

    /// Splits a fill evenly across its contributing strategies' virtual books (spec 40 §6.3,
    /// D-047). Flatten intents have no contributors and touch no book.
    fn attribute(&mut self, fill: &Fill, strategy_ids: &[String]) {
        let ids: Vec<&String> = strategy_ids.iter().filter(|id| self.strategies.contains_key(*id)).collect();
        if ids.is_empty() {
            return;
        }
        let n = ids.len() as f64;
        let (q, cost) = (fill.quantity / n, (fill.quantity * fill.price * fill.fx_to_eur + fill.fee_eur) / n);
        for id in ids {
            let b = self.books.entry(id.clone()).or_default();
            *b.qty.entry(fill.instrument).or_insert(0.0) += q;
            b.cash_eur -= cost;
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
        self.bump("resumes");
    }

    /// Any health halt (source / model / broker) is active (promotion gate item 5).
    fn health_halted(&self) -> bool {
        !self.source_halts.is_empty()
            || self.halts.contains_key(&HaltScope::Model)
            || self.halts.contains_key(&HaltScope::Broker)
    }

    // ------------------------------------------------------------ lifecycle (D-051)

    #[cfg(test)]
    pub fn strategy(&self, id: &str) -> Option<&StrategyState> {
        self.strategies.get(id)
    }

    /// Gate checklist for a strategy; `deflated_sharpe` comes from `validation.trial` when the
    /// I/O layer could read it (`None` = unknown → blocks promotion).
    pub fn checklist(&self, id: &str, deflated_sharpe: Option<f64>, now_ms: i64) -> Option<Checklist> {
        let st = self.strategies.get(id)?;
        let ev = GateEvidence { deflated_sharpe, unresolved_health_halts: self.health_halted(), ..Default::default() };
        Some(self.settings.promotion.check(st, &ev, now_ms))
    }

    /// Operator lifecycle action (`ctl.risk.strategy`). Errors follow the control-api prefix
    /// convention (`not_found:` / `conflict:` / `invalid:`).
    pub fn strategy_action(
        &mut self,
        id: &str,
        action: StrategyAction,
        deflated_sharpe: Option<f64>,
        now_ms: i64,
    ) -> Result<StrategyState, String> {
        if !self.strategies.contains_key(id) {
            return Err(format!("not_found: unknown strategy {id:?} (no signal seen, no persisted state)"));
        }
        let gate = if action == StrategyAction::Promote { self.checklist(id, deflated_sharpe, now_ms) } else { None };
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
            let _ = self.strategy_action(id, StrategyAction::Flatten, None, now_ms);
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

    /// Strategy list for `ctl.risk.strategy list` / `GET /api/trading/strategies`.
    pub fn strategies_view(&self, deflated_sharpe: &BTreeMap<String, Option<f64>>, now_ms: i64) -> Value {
        let rows: Vec<Value> = self
            .strategies
            .values()
            .map(|s| {
                let dsr = deflated_sharpe.get(&s.strategy_id).copied().flatten();
                let gate = self.checklist(&s.strategy_id, dsr, now_ms);
                self.strategy_json(s, gate.as_ref(), now_ms)
            })
            .collect();
        json!({ "as_of": ts_rfc3339(now_ms), "strategies": rows })
    }

    pub fn strategy_json(&self, s: &StrategyState, gate: Option<&Checklist>, now_ms: i64) -> Value {
        let weight = self.last_alloc.get(&s.strategy_id).copied().unwrap_or(0.0);
        let shadow_days = s.shadow_since_ms.map(|t| (now_ms - t) as f64 / MS_PER_DAY as f64);
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
            "shadow": {
                "since": s.shadow_since_ms.map(ts_rfc3339),
                "days": shadow_days,
                "horizons_completed": s.horizons_completed,
            },
            "gate": gate,
        })
    }

    // ---------------------------------------------------------------- internals

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
            for (d, _) in self.signals.values() {
                self.perf.observe(d, &realised);
            }
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
        let alloc = self.allocator.allocate(&stats);
        for (id, w) in &alloc {
            if let Some(s) = self.strategies.get_mut(id) {
                s.sleeve_eur = w * deployable;
            }
        }
        self.last_alloc = alloc.clone();
        let raw = fa_allocator::net_targets(&dists, &alloc, deployable);

        let market: BTreeMap<InstrumentId, MarketInfo> = raw
            .keys()
            .filter(|i| self.marks.contains_key(i))
            .map(|i| {
                (
                    *i,
                    MarketInfo {
                        asset_class: self.settings.default_asset_class,
                        venue: String::new(), // one bucket until reference data reaches the engine
                        horizon_volume_eur: self.horizon_volume_eur(*i, now_ms),
                    },
                )
            })
            .collect();
        let capped = self.risk.cap_targets(&raw, &market, equity);
        let current = self.current_eur();
        // instruments we hold but no longer target are flattened
        let mut targets = capped;
        for i in current.keys() {
            targets.entry(*i).or_insert(0.0);
        }
        let gated = self.risk.gate(&targets, &current);

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
                    let delta = v - b.last_value_eur;
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
            if dd >= limit && limit > 0.0 && sleeve > 0.0 {
                st.breach(dd, limit, now_ms);
                breached.push((id, status, dd));
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

fn ts_rfc3339(ms: i64) -> String {
    chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms).map(|t| t.to_rfc3339()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: i64 = 1_762_560_000_000; // 2025-11-08T00:00:00Z
    const PX: f64 = 100.0;

    fn settings() -> Settings {
        Settings::parse(include_str!("../risk.toml")).unwrap()
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
        assert_eq!(e.on_intent_seen(&foreign, false, T0 + 1200), vec![Out::Approved(foreign.clone())]);
        fill(&mut e, &foreign, PX, T0 + 1300);
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
        assert!(e.strategy_action("nope", StrategyAction::Pause, None, T0).unwrap_err().starts_with("not_found:"));
        assert!(e.strategy_action("a", StrategyAction::Resume, None, T0).unwrap_err().starts_with("conflict:"));
        // pause a: only b remains (50k target, still capped at 5k) → nothing to trade
        let st = e.strategy_action("a", StrategyAction::Pause, None, T0 + 2000).unwrap();
        assert_eq!(st.status, StrategyStatus::Paused);
        assert!(e.take_dirty_strategies().iter().any(|s| s.strategy_id == "a"));
        e.on_signal(&signal("a", inst, 1.0), T0 + 2500);
        e.on_signal(&signal("b", inst, 1.0), T0 + 2500);
        assert!(approved(&e.tick(T0 + 3000)).is_empty());
        // pause b too: no active strategy → position flattened
        e.strategy_action("b", StrategyAction::Pause, None, T0 + 3500).unwrap();
        e.on_signal(&signal("b", inst, 1.0), T0 + 3600);
        let flat = approved(&e.tick(T0 + 4000));
        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].side, Side::Sell as i32);
        assert!((flat[0].quantity - 50.0).abs() < 1e-6, "paused strategies hold no exposure");
        assert!(flat[0].strategy_ids.is_empty());
        fill(&mut e, &flat[0], PX, T0 + 4100);
        assert!(e.strategy_action("a", StrategyAction::Pause, None, T0 + 4200).unwrap_err().contains("already paused"));
        // resume a: its signal is allocated again
        assert_eq!(
            e.strategy_action("a", StrategyAction::Resume, None, T0 + 4500).unwrap().status,
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
            e.strategy_action("a", StrategyAction::Flatten, None, T0 + 5500).unwrap().status,
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
        let err = e.strategy_action("s", StrategyAction::Promote, None, T0 + 10_000).unwrap_err();
        assert!(err.starts_with("conflict: promotion gate not green"), "{err}");
        assert!(err.contains("min_shadow_days") && err.contains("deflated_sharpe"), "{err}");
        assert_eq!(e.strategy("s").unwrap().status, StrategyStatus::Shadow);
        // an hour later, with a DSR row: still refused without the statistic …
        let later = T0 + 3_600_000 + 1;
        let cl = e.checklist("s", None, later).unwrap();
        assert_eq!(cl.missing, vec!["deflated_sharpe"]);
        // … and promoted with it; the strategy is allocated from the next tick
        let st = e.strategy_action("s", StrategyAction::Promote, Some(0.4), later).unwrap();
        assert_eq!(st.status, StrategyStatus::Active);
        assert_eq!(st.last_reason, "promoted: gate green");
        e.on_trade(inst, later, PX, 10_000.0); // refill the (now empty) hourly volume window
        e.on_signal(&signal("s", inst, 1.0), later);
        assert_eq!(approved(&e.tick(later + 1000)).len(), 1, "promoted strategy trades");
        // health halts block the gate for other shadow strategies
        e.strategy_action("s", StrategyAction::Shadow, None, later + 2000).unwrap();
        e.on_source_health("yahoo", "down", "stale", later + 2000);
        assert_eq!(
            e.checklist("s", Some(1.0), later + 2000).unwrap().missing,
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
        e.strategy_action("a", StrategyAction::Resume, None, T0 + 5000).unwrap();
        e.on_signal(&signal("a", inst, 1.0), T0 + 5000);
        let buy = approved(&e.tick(T0 + 6000));
        assert!((buy[0].quantity * PX - 1_000.0).abs() < 1e-6, "new 1 % position cap binds: {}", buy[0].quantity * PX);
    }
}
