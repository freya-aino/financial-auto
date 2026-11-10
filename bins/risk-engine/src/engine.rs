//! Pure risk/allocation engine (no I/O): signals, trades, order events and halts in;
//! intents, approvals, rejections, halts and risk state out. Deterministic given the inputs
//! (intent ids are the only non-deterministic field).
//!
//! Decision tick (spec 40 §6.1): latest signal per strategy → allocator weights → EUR targets
//! per instrument (`fa_allocator::net_targets`) → hard caps (`fa_risk`) → halt gate (flatten /
//! reduce-only) → diff against the paper book plus pending intents → `OrderIntent`s.
use std::collections::{BTreeMap, HashMap, VecDeque};

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
use fa_risk::{HaltReason, HaltScope, MarketInfo, RiskEngine};
use fa_strategies::ExposureDistribution;
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
    booked: Vec<BookedFill>,
    day: Option<i64>,
    day_start_equity: f64,
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
            booked: Vec::new(),
            day: None,
            day_start_equity: cash,
            counters: BTreeMap::new(),
        }
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
        self.signals.insert(m.strategy_id.clone(), (d, now_ms));
        self.bump("signals");
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
            self.booked.push(BookedFill { fill, strategy_ids: info.strategy_ids.clone(), at_ms });
        }
        self.check_equity(now_ms)
    }

    /// A halt from the bus (operator kill switch or another service).
    pub fn on_halt(&mut self, h: &Halt, now_ms: i64) {
        let Some(scope) = HaltScope::parse(&h.scope) else {
            self.bump("halts.ignored_scope");
            return;
        };
        let reason = if h.manual { HaltReason::Manual(h.reason.clone()) } else { HaltReason::Health(h.reason.clone()) };
        self.risk.halt(scope, reason);
        self.halts.entry(scope).or_insert(ActiveHalt {
            reason: h.reason.clone(),
            since_ms: ms_of(h.since.as_ref()).unwrap_or(now_ms),
            manual: h.manual,
        });
        self.bump("halts.received");
    }

    /// `risk.resume.<scope>` (D-030): manual re-arm.
    pub fn on_resume(&mut self, scope: &str) {
        let Some(scope) = HaltScope::parse(scope) else {
            return;
        };
        if scope == HaltScope::Portfolio {
            self.risk.rearm(self.equity());
        } else {
            self.risk.clear(scope);
        }
        self.halts.remove(&scope);
        self.bump("resumes");
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
        self.update_performance();
        let ttl = self.settings.signal_ttl_secs as i64 * 1000;
        self.signals.retain(|_, (_, seen)| now_ms - *seen <= ttl);
        self.pending.retain(|_, p| now_ms - p.created_ms <= self.settings.pending_ttl_secs as i64 * 1000);

        let equity = self.equity();
        let deployable = self.risk.deployable(equity);
        let dists: Vec<ExposureDistribution> = self.signals.values().map(|(d, _)| d.clone()).collect();
        let stats = self.perf.stats_for(dists.iter().map(|d| d.strategy_id.as_str()));
        let alloc = self.allocator.allocate(&stats);
        let raw = fa_allocator::net_targets(&dists, &alloc, deployable);

        let market: BTreeMap<InstrumentId, MarketInfo> = raw
            .keys()
            .filter(|i| self.marks.contains_key(i))
            .map(|i| {
                (
                    *i,
                    MarketInfo {
                        asset_class: self.settings.default_asset_class,
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
                .map(|(s, h)| Halt {
                    scope: s.as_str().into(),
                    reason: h.reason.clone(),
                    since: Some(ts_of(h.since_ms)),
                    manual: h.manual,
                })
                .collect(),
        }
    }
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
            reason: String::new(),
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
        assert!(e.tick(T0 + 3000).is_empty(), "reduce-only: cannot add to the long");
        e.on_signal(&signal("a", inst, -1.0), T0 + 3500);
        let sell = approved(&e.tick(T0 + 4000));
        assert_eq!(sell.len(), 1);
        assert!((sell[0].quantity - 50.0).abs() < 1e-6, "may only close, not flip");
        fill(&mut e, &sell[0], 50.0, T0 + 4100);
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
}
