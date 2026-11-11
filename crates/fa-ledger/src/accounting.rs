//! Derived accounting (spec 42 §6): tax lots, the P&L breakdown, as-of positions and equity.
//!
//! Everything here is a pure function of the event stream (fills + cash movements): events are
//! sorted by `(time, id)` and de-duplicated by id before they are applied, so the result does not
//! depend on arrival order (late / out-of-order events simply land where they belong on the next
//! replay) and replaying the same stream twice yields the same state.
//!
//! Lots are kept per `(venue, instrument)` with signed quantities (+ long, − short). A fill first
//! closes lots of the opposite sign (oldest first under [`LotMethod::Fifo`]; the single pooled
//! lot under [`LotMethod::Average`]) and opens a new lot with any remainder. Fills settle in EUR
//! at the fill's rate (the paper venue's model, and that of the in-memory [`crate::Ledger`]); cash
//! movements in another currency are held as native balances and revalued.
//!
//! EUR P&L of a lot is split into a price part and an FX part:
//! `q·(P₁X₁ − P₀X₀) = q·(P₁ − P₀)·X₁ + q·P₀·(X₁ − X₀)`, with `P` native prices and `X` EUR per
//! unit of the lot currency at open (0) and close or mark (1). With that split
//! `equity − external flows = PnlBreakdown::net_eur` holds exactly (up to float rounding).
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

use chrono::{DateTime, Utc};
use fa_core::InstrumentId;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Quantities below this are treated as zero (closed lot / flat position).
pub const QTY_EPS: f64 = 1e-12;

/// `ledger.lot_method` (spec 42 §7).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LotMethod {
    /// First in, first out per `(venue, instrument)`.
    #[default]
    Fifo,
    /// One pooled lot per `(venue, instrument)` at the weighted average cost.
    Average,
}

impl LotMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fifo => "fifo",
            Self::Average => "average",
        }
    }
}

impl std::str::FromStr for LotMethod {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fifo" => Ok(Self::Fifo),
            "average" | "avg" | "average_cost" | "average-cost" => Ok(Self::Average),
            other => Err(format!("unknown lot method {other:?}: use fifo or average")),
        }
    }
}

/// A strategy's share of a fill (`ledger.fill_attribution`): fractions of the fill quantity and fee.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attribution {
    pub strategy: Uuid,
    pub qty_frac: f64,
    pub fee_frac: f64,
}

/// One fill as an accounting event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TradeEvent {
    /// `ledger.fill.id` (idempotency and ordering tie-break).
    pub fill_uuid: Uuid,
    /// Venue fill id (`ledger.fill.venue_fill_id`).
    pub fill_id: String,
    pub intent_id: String,
    pub venue: Uuid,
    pub instrument: InstrumentId,
    /// Signed quantity (+ buy, − sell).
    pub qty: f64,
    /// Price in `ccy`.
    pub price: f64,
    pub ccy: String,
    /// EUR per unit of `ccy` at fill time.
    pub fx_to_eur: f64,
    pub fx_source: String,
    pub fee_eur: f64,
    pub filled_at: DateTime<Utc>,
    #[serde(default)]
    pub attribution: Vec<Attribution>,
}

impl TradeEvent {
    pub fn notional_native(&self) -> f64 {
        self.qty * self.price
    }
    pub fn notional_eur(&self) -> f64 {
        self.qty * self.price * self.fx_to_eur
    }
}

/// One `ledger.cash_movement` as an accounting event. `amount` is signed (+ in, − out) in `ccy`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CashEvent {
    pub kind: String,
    pub reference: String,
    pub amount: f64,
    pub ccy: String,
    pub fx_to_eur: f64,
    pub occurred_at: DateTime<Utc>,
}

impl CashEvent {
    pub fn amount_eur(&self) -> f64 {
        self.amount * self.fx_to_eur
    }
}

/// How a cash movement kind enters the P&L.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CashClass {
    /// Deposits, withdrawals, transfers and payouts: capital flows, not P&L.
    External,
    Funding,
    Borrow,
    /// Fees booked as cash movements (venue fees outside fills); amount negative when paid.
    Fee,
    /// Dividends, interest, rebates, reconciliation adjustments, …
    Other,
}

pub fn classify_cash(kind: &str) -> CashClass {
    match kind.trim().to_ascii_lowercase().as_str() {
        "deposit" | "withdrawal" | "withdraw" | "payout" | "transfer" | "transfer_in" | "transfer_out" => {
            CashClass::External
        }
        "funding" => CashClass::Funding,
        "borrow" | "borrow_fee" => CashClass::Borrow,
        "fee" | "commission" => CashClass::Fee,
        _ => CashClass::Other,
    }
}

fn is_eur(ccy: &str) -> bool {
    ccy.is_empty() || ccy.eq_ignore_ascii_case(fa_core::BASE_CURRENCY)
}

/// Marks for valuation: native prices per instrument and EUR per unit per currency. Missing
/// entries fall back to the last fill (see [`BookState::resolved_marks`]).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Marks {
    pub prices: BTreeMap<InstrumentId, f64>,
    pub fx: BTreeMap<String, f64>,
}

impl Marks {
    pub fn price(&self, i: &InstrumentId) -> Option<f64> {
        self.prices.get(i).copied().filter(|p| p.is_finite())
    }
    /// EUR per unit of `ccy` (1.0 for EUR).
    pub fn fx(&self, ccy: &str) -> Option<f64> {
        if is_eur(ccy) {
            return Some(1.0);
        }
        self.fx.get(&ccy.to_ascii_uppercase()).or_else(|| self.fx.get(ccy)).copied().filter(|x| x.is_finite())
    }
}

/// An open (or partially closed) tax lot. `qty` is the remaining signed quantity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lot {
    /// `ledger.fill.id` of the opening fill.
    pub lot_id: Uuid,
    pub venue: Uuid,
    pub instrument: InstrumentId,
    pub opened_at: DateTime<Utc>,
    pub open_qty: f64,
    pub qty: f64,
    /// Native cost per unit.
    pub price: f64,
    pub ccy: String,
    /// EUR per unit of `ccy` at open (cost-weighted for average-cost lots).
    pub fx_to_eur: f64,
}

impl Lot {
    pub fn cost_native(&self) -> f64 {
        self.qty * self.price
    }
    pub fn cost_eur(&self) -> f64 {
        self.qty * self.price * self.fx_to_eur
    }
}

/// A (partial) lot closing with realised P&L.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LotClosing {
    pub lot_id: Uuid,
    pub close_fill: Uuid,
    pub close_fill_id: String,
    pub venue: Uuid,
    pub instrument: InstrumentId,
    pub opened_at: DateTime<Utc>,
    pub closed_at: DateTime<Utc>,
    /// Closed quantity, signed like the lot (+ a long was sold, − a short was bought back).
    pub qty: f64,
    pub open_price: f64,
    pub close_price: f64,
    pub ccy: String,
    pub open_fx: f64,
    pub close_fx: f64,
    /// `qty · (close − open)` in `ccy`.
    pub realised_native: f64,
    /// Price part in EUR: `realised_native · close_fx`.
    pub realised_eur: f64,
    /// FX part in EUR: `qty · open_price · (close_fx − open_fx)`.
    pub fx_eur: f64,
}

/// P&L separated as spec 42 §6 asks. Fees are positive when paid; every other component is
/// signed P&L in EUR. `fx_eur` and `net_eur` are derived (kept current by every constructor).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PnlBreakdown {
    /// Closed lots, price part at the closing FX rate.
    pub realised_eur: f64,
    /// Open lots, price part at the mark FX rate.
    pub unrealised_eur: f64,
    /// FX part of closed lots (cost basis revalued from open to close rate).
    pub fx_realised_eur: f64,
    /// FX part of open lots (cost basis revalued from open to mark rate).
    pub fx_unrealised_eur: f64,
    /// Revaluation of non-EUR cash balances.
    pub fx_cash_eur: f64,
    pub fees_eur: f64,
    pub funding_eur: f64,
    pub borrow_eur: f64,
    /// Dividends, interest, adjustments and other non-external cash movements.
    pub other_eur: f64,
    /// `fx_realised + fx_unrealised + fx_cash`.
    pub fx_eur: f64,
    /// `realised + unrealised + fx − fees + funding + borrow + other`.
    pub net_eur: f64,
}

impl PnlBreakdown {
    fn settled(mut self) -> Self {
        self.fx_eur = self.fx_realised_eur + self.fx_unrealised_eur + self.fx_cash_eur;
        self.net_eur = self.realised_eur + self.unrealised_eur + self.fx_eur - self.fees_eur
            + self.funding_eur
            + self.borrow_eur
            + self.other_eur;
        self
    }

    fn zip(&self, o: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        Self {
            realised_eur: f(self.realised_eur, o.realised_eur),
            unrealised_eur: f(self.unrealised_eur, o.unrealised_eur),
            fx_realised_eur: f(self.fx_realised_eur, o.fx_realised_eur),
            fx_unrealised_eur: f(self.fx_unrealised_eur, o.fx_unrealised_eur),
            fx_cash_eur: f(self.fx_cash_eur, o.fx_cash_eur),
            fees_eur: f(self.fees_eur, o.fees_eur),
            funding_eur: f(self.funding_eur, o.funding_eur),
            borrow_eur: f(self.borrow_eur, o.borrow_eur),
            other_eur: f(self.other_eur, o.other_eur),
            ..Self::default()
        }
        .settled()
    }

    pub fn plus(&self, o: &Self) -> Self {
        self.zip(o, |a, b| a + b)
    }

    /// `self − o`: the P&L over a range is `to_date(to) − to_date(from)`.
    pub fn minus(&self, o: &Self) -> Self {
        self.zip(o, |a, b| a - b)
    }

    /// Largest absolute component difference (tests / consistency checks).
    pub fn max_abs_diff(&self, o: &Self) -> f64 {
        let d = self.minus(o);
        [
            d.realised_eur,
            d.unrealised_eur,
            d.fx_realised_eur,
            d.fx_unrealised_eur,
            d.fx_cash_eur,
            d.fees_eur,
            d.funding_eur,
            d.borrow_eur,
            d.other_eur,
            d.net_eur,
        ]
        .into_iter()
        .fold(0.0, |m, x| m.max(x.abs()))
    }
}

/// A non-EUR cash balance: native amount and its EUR book value at the booking rates.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CashBalance {
    pub ccy: String,
    pub balance: f64,
    pub booked_eur: f64,
}

/// A position per `(venue, instrument)` derived from its lots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BookPosition {
    pub venue: Uuid,
    pub instrument: InstrumentId,
    /// Signed quantity (0 for a position that traded and is now flat).
    pub qty: f64,
    pub ccy: String,
    /// Native average cost per unit of the open lots (0 when flat).
    pub avg_price: f64,
    /// EUR cost of the open lots at their opening rates.
    pub cost_eur: f64,
    pub open_lots: usize,
    /// Cumulative realised (price part) and FX-realised P&L in EUR.
    pub realised_eur: f64,
    pub fx_realised_eur: f64,
    pub last_price: f64,
    pub last_fx: f64,
    pub last_fill_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq)]
struct Last {
    price: f64,
    fx: f64,
    ccy: String,
    at: DateTime<Utc>,
}

type LotKey = (Uuid, InstrumentId);

/// The book replayed from events up to a cutoff.
#[derive(Debug, Clone, PartialEq)]
pub struct BookState {
    pub method: LotMethod,
    /// Events strictly before this instant were applied (`None` = all).
    pub before: Option<DateTime<Utc>>,
    /// EUR cash (fills settle in EUR).
    pub cash_eur: f64,
    pub foreign_cash: BTreeMap<String, CashBalance>,
    lots: BTreeMap<LotKey, VecDeque<Lot>>,
    /// Fully closed lots (`qty = 0`), in closing order.
    pub closed_lots: Vec<Lot>,
    pub closings: Vec<LotClosing>,
    pub fees_eur: f64,
    pub funding_eur: f64,
    pub borrow_eur: f64,
    pub other_eur: f64,
    /// Net capital flows in EUR at booking rates (deposits − withdrawals − payouts …).
    pub external_eur: f64,
    /// Payouts to date in EUR (positive).
    pub payouts_eur: f64,
    pub fills: usize,
    pub cash_movements: usize,
    last_by_key: BTreeMap<LotKey, Last>,
    last_by_instrument: BTreeMap<InstrumentId, Last>,
    last_fx: BTreeMap<String, f64>,
    seen_fills: HashSet<Uuid>,
    seen_cash: HashSet<(String, String)>,
}

impl BookState {
    pub fn new(method: LotMethod) -> Self {
        Self {
            method,
            before: None,
            cash_eur: 0.0,
            foreign_cash: BTreeMap::new(),
            lots: BTreeMap::new(),
            closed_lots: Vec::new(),
            closings: Vec::new(),
            fees_eur: 0.0,
            funding_eur: 0.0,
            borrow_eur: 0.0,
            other_eur: 0.0,
            external_eur: 0.0,
            payouts_eur: 0.0,
            fills: 0,
            cash_movements: 0,
            last_by_key: BTreeMap::new(),
            last_by_instrument: BTreeMap::new(),
            last_fx: BTreeMap::new(),
            seen_fills: HashSet::new(),
            seen_cash: HashSet::new(),
        }
    }

    /// Applies one fill (no-op on a replayed `fill_uuid`). Callers replaying a stream should use
    /// [`replay`], which orders events first.
    pub fn apply_trade(&mut self, t: &TradeEvent) -> bool {
        if !self.seen_fills.insert(t.fill_uuid) || !t.qty.is_finite() || t.qty.abs() < QTY_EPS {
            return false;
        }
        self.fills += 1;
        let fx = if is_eur(&t.ccy) { 1.0 } else { t.fx_to_eur };
        self.cash_eur -= t.qty * t.price * fx + t.fee_eur;
        self.fees_eur += t.fee_eur;
        let ccy = if t.ccy.is_empty() { fa_core::BASE_CURRENCY.to_owned() } else { t.ccy.to_ascii_uppercase() };
        let last = Last { price: t.price, fx, ccy: ccy.clone(), at: t.filled_at };
        let key = (t.venue, t.instrument);
        self.last_by_key.insert(key, last.clone());
        if self.last_by_instrument.get(&t.instrument).is_none_or(|l| l.at <= t.filled_at) {
            self.last_by_instrument.insert(t.instrument, last);
        }
        if !is_eur(&ccy) {
            self.last_fx.insert(ccy.clone(), fx);
        }

        let lots = self.lots.entry(key).or_default();
        let mut rem = t.qty;
        while rem.abs() > QTY_EPS {
            let Some(lot) = lots.front_mut() else { break };
            if lot.qty.signum() == rem.signum() {
                break;
            }
            let take = rem.abs().min(lot.qty.abs());
            let c = take * lot.qty.signum();
            let realised_native = c * (t.price - lot.price);
            self.closings.push(LotClosing {
                lot_id: lot.lot_id,
                close_fill: t.fill_uuid,
                close_fill_id: t.fill_id.clone(),
                venue: t.venue,
                instrument: t.instrument,
                opened_at: lot.opened_at,
                closed_at: t.filled_at,
                qty: c,
                open_price: lot.price,
                close_price: t.price,
                ccy: lot.ccy.clone(),
                open_fx: lot.fx_to_eur,
                close_fx: fx,
                realised_native,
                realised_eur: realised_native * fx,
                fx_eur: c * lot.price * (fx - lot.fx_to_eur),
            });
            lot.qty -= c;
            rem += c;
            if lot.qty.abs() <= QTY_EPS {
                lot.qty = 0.0;
                self.closed_lots.extend(lots.pop_front());
            }
        }
        if rem.abs() > QTY_EPS {
            match (self.method, lots.back_mut()) {
                (LotMethod::Average, Some(lot)) => {
                    let q = lot.qty + rem;
                    let cost_native = lot.qty * lot.price + rem * t.price;
                    let cost_eur = lot.qty * lot.price * lot.fx_to_eur + rem * t.price * fx;
                    lot.price = cost_native / q;
                    lot.fx_to_eur = if cost_native.abs() > QTY_EPS { cost_eur / cost_native } else { fx };
                    lot.qty = q;
                    lot.open_qty += rem;
                }
                _ => lots.push_back(Lot {
                    lot_id: t.fill_uuid,
                    venue: t.venue,
                    instrument: t.instrument,
                    opened_at: t.filled_at,
                    open_qty: rem,
                    qty: rem,
                    price: t.price,
                    ccy,
                    fx_to_eur: fx,
                }),
            }
        }
        true
    }

    /// Applies one cash movement (no-op on a replayed `(kind, reference)`).
    pub fn apply_cash(&mut self, c: &CashEvent) -> bool {
        if !self.seen_cash.insert((c.kind.clone(), c.reference.clone())) || !c.amount.is_finite() {
            return false;
        }
        self.cash_movements += 1;
        let eur = c.amount_eur();
        if is_eur(&c.ccy) {
            self.cash_eur += eur;
        } else {
            let ccy = c.ccy.to_ascii_uppercase();
            let b = self.foreign_cash.entry(ccy.clone()).or_insert_with(|| CashBalance { ccy, ..Default::default() });
            b.balance += c.amount;
            b.booked_eur += eur;
        }
        match classify_cash(&c.kind) {
            CashClass::External => {
                self.external_eur += eur;
                if c.kind.eq_ignore_ascii_case("payout") {
                    self.payouts_eur -= eur;
                }
            }
            CashClass::Funding => self.funding_eur += eur,
            CashClass::Borrow => self.borrow_eur += eur,
            CashClass::Fee => self.fees_eur -= eur,
            CashClass::Other => self.other_eur += eur,
        }
        true
    }

    /// Open lots, ordered by `(venue, instrument)` then age.
    pub fn open_lots(&self) -> Vec<Lot> {
        self.lots.values().flatten().cloned().collect()
    }

    /// `base` completed from the book: missing prices from the latest fill of the instrument,
    /// missing rates from the latest fill in that currency (then booking rates for cash).
    pub fn resolved_marks(&self, base: &Marks) -> Marks {
        let mut m = base.clone();
        for (i, l) in &self.last_by_instrument {
            m.prices.entry(*i).or_insert(l.price);
        }
        for (ccy, x) in &self.last_fx {
            if base.fx(ccy).is_none() {
                m.fx.insert(ccy.clone(), *x);
            }
        }
        for (ccy, b) in &self.foreign_cash {
            if m.fx(ccy).is_none() && b.balance.abs() > QTY_EPS {
                m.fx.insert(ccy.clone(), b.booked_eur / b.balance);
            }
        }
        m
    }

    fn lot_mark(&self, lot: &Lot, marks: &Marks) -> (f64, f64) {
        let last = self.last_by_key.get(&(lot.venue, lot.instrument));
        let px = marks.price(&lot.instrument).or(last.map(|l| l.price)).unwrap_or(lot.price);
        let fx = marks.fx(&lot.ccy).or(last.map(|l| l.fx)).unwrap_or(lot.fx_to_eur);
        (px, fx)
    }

    fn cash_fx(&self, b: &CashBalance, marks: &Marks) -> f64 {
        marks.fx(&b.ccy).or_else(|| self.last_fx.get(&b.ccy).copied()).unwrap_or(if b.balance.abs() > QTY_EPS {
            b.booked_eur / b.balance
        } else {
            0.0
        })
    }

    /// Market value of open lots plus all cash, in EUR.
    pub fn equity_eur(&self, marks: &Marks) -> f64 {
        let lots: f64 = self
            .lots
            .values()
            .flatten()
            .map(|l| {
                let (px, fx) = self.lot_mark(l, marks);
                l.qty * px * fx
            })
            .sum();
        let foreign: f64 = self.foreign_cash.values().map(|b| b.balance * self.cash_fx(b, marks)).sum();
        self.cash_eur + foreign + lots
    }

    /// Cumulative P&L to the cutoff at `marks`.
    pub fn pnl(&self, marks: &Marks) -> PnlBreakdown {
        let mut p = PnlBreakdown {
            fees_eur: self.fees_eur,
            funding_eur: self.funding_eur,
            borrow_eur: self.borrow_eur,
            other_eur: self.other_eur,
            ..PnlBreakdown::default()
        };
        for c in &self.closings {
            p.realised_eur += c.realised_eur;
            p.fx_realised_eur += c.fx_eur;
        }
        for l in self.lots.values().flatten() {
            let (px, fx) = self.lot_mark(l, marks);
            p.unrealised_eur += l.qty * (px - l.price) * fx;
            p.fx_unrealised_eur += l.qty * l.price * (fx - l.fx_to_eur);
        }
        for b in self.foreign_cash.values() {
            p.fx_cash_eur += b.balance * self.cash_fx(b, marks) - b.booked_eur;
        }
        p.settled()
    }

    /// Positions per `(venue, instrument)` that ever traded (flat ones with `qty = 0`).
    pub fn positions(&self) -> Vec<BookPosition> {
        let mut realised: BTreeMap<LotKey, (f64, f64)> = BTreeMap::new();
        for c in &self.closings {
            let r = realised.entry((c.venue, c.instrument)).or_default();
            r.0 += c.realised_eur;
            r.1 += c.fx_eur;
        }
        self.last_by_key
            .iter()
            .map(|(key, last)| {
                let lots = self.lots.get(key);
                let qty: f64 = lots.map_or(0.0, |v| v.iter().map(|l| l.qty).sum());
                let cost_native: f64 = lots.map_or(0.0, |v| v.iter().map(Lot::cost_native).sum());
                let cost_eur: f64 = lots.map_or(0.0, |v| v.iter().map(Lot::cost_eur).sum());
                let (r, fxr) = realised.get(key).copied().unwrap_or_default();
                let flat = qty.abs() <= QTY_EPS;
                BookPosition {
                    venue: key.0,
                    instrument: key.1,
                    qty: if flat { 0.0 } else { qty },
                    ccy: lots.and_then(|v| v.front()).map_or_else(|| last.ccy.clone(), |l| l.ccy.clone()),
                    avg_price: if flat { 0.0 } else { cost_native / qty },
                    cost_eur: if flat { 0.0 } else { cost_eur },
                    open_lots: lots.map_or(0, VecDeque::len),
                    realised_eur: r,
                    fx_realised_eur: fxr,
                    last_price: last.price,
                    last_fx: last.fx,
                    last_fill_at: last.at,
                }
            })
            .collect()
    }

    /// Net quantity per instrument across venues (the reconciliation view).
    pub fn quantities(&self) -> BTreeMap<InstrumentId, f64> {
        let mut q: BTreeMap<InstrumentId, f64> = BTreeMap::new();
        for p in self.positions() {
            *q.entry(p.instrument).or_default() += p.qty;
        }
        q
    }

    /// Currencies the book is exposed to (lots and cash), EUR excluded.
    pub fn currencies(&self) -> BTreeSet<String> {
        self.last_by_key
            .values()
            .map(|l| l.ccy.clone())
            .chain(self.foreign_cash.keys().cloned())
            .filter(|c| !is_eur(c))
            .collect()
    }
}

/// Replays fills and cash movements with timestamp `< before` (all when `None`), in
/// `(time, id)` order, de-duplicated by id. Input order does not matter.
pub fn replay(
    method: LotMethod,
    trades: &[TradeEvent],
    cash: &[CashEvent],
    before: Option<DateTime<Utc>>,
) -> BookState {
    let keep = |t: &DateTime<Utc>| before.is_none_or(|b| *t < b);
    let mut ts: Vec<&TradeEvent> = trades.iter().filter(|t| keep(&t.filled_at)).collect();
    ts.sort_by(|a, b| a.filled_at.cmp(&b.filled_at).then(a.fill_uuid.cmp(&b.fill_uuid)));
    let mut cs: Vec<&CashEvent> = cash.iter().filter(|c| keep(&c.occurred_at)).collect();
    cs.sort_by(|a, b| {
        a.occurred_at.cmp(&b.occurred_at).then_with(|| a.kind.cmp(&b.kind)).then_with(|| a.reference.cmp(&b.reference))
    });
    let mut s = BookState::new(method);
    s.before = before;
    for c in cs {
        s.apply_cash(c);
    }
    for t in ts {
        s.apply_trade(t);
    }
    s
}

/// Key of the unattributed remainder in [`strategy_trades`] / per-strategy maps.
pub const UNATTRIBUTED: &str = "unattributed";

/// A strategy's virtual fills: each attributed fill scaled by the strategy's quantity and fee
/// fractions. `None` selects the unattributed remainder (`1 − Σ fractions`, e.g. manual or
/// flatten trades).
pub fn strategy_trades(trades: &[TradeEvent], strategy: Option<Uuid>) -> Vec<TradeEvent> {
    trades
        .iter()
        .filter_map(|t| {
            let (qf, ff) = match strategy {
                Some(s) => {
                    t.attribution.iter().filter(|a| a.strategy == s).fold(None, |acc: Option<(f64, f64)>, a| {
                        let (q, f) = acc.unwrap_or_default();
                        Some((q + a.qty_frac, f + a.fee_frac))
                    })?
                }
                None => {
                    let q: f64 = t.attribution.iter().map(|a| a.qty_frac).sum();
                    let f: f64 = t.attribution.iter().map(|a| a.fee_frac).sum();
                    (1.0 - q, 1.0 - f)
                }
            };
            ((t.qty * qf).abs() > QTY_EPS || (t.fee_eur * ff).abs() > QTY_EPS).then(|| TradeEvent {
                qty: t.qty * qf,
                fee_eur: t.fee_eur * ff,
                attribution: Vec::new(),
                ..t.clone()
            })
        })
        .collect()
}

/// Every strategy id appearing in the attribution of `trades`.
pub fn strategies(trades: &[TradeEvent]) -> BTreeSet<Uuid> {
    trades.iter().flat_map(|t| t.attribution.iter().map(|a| a.strategy)).collect()
}

/// P&L over `[from, to)` of the given trades and cash: `to_date(to) − to_date(from)`, each side
/// valued at its own marks (completed from the book via [`BookState::resolved_marks`]).
pub fn range_pnl(
    method: LotMethod,
    trades: &[TradeEvent],
    cash: &[CashEvent],
    from: Option<DateTime<Utc>>,
    to: Option<DateTime<Utc>>,
    marks_from: &Marks,
    marks_to: &Marks,
) -> PnlBreakdown {
    let end = replay(method, trades, cash, to).pnl(marks_to);
    match from {
        Some(f) => end.minus(&replay(method, trades, cash, Some(f)).pnl(marks_from)),
        None => end,
    }
}

/// Per-strategy P&L over `[from, to)` keyed by strategy UUID string, plus [`UNATTRIBUTED`]
/// carrying unattributed trades and all book-level cash items (funding, borrow, other, cash
/// FX). The values sum to the portfolio's [`range_pnl`] when the same (already resolved) marks
/// are used, which is why the marks must be complete: pass the portfolio's resolved marks.
pub fn strategy_pnls(
    method: LotMethod,
    trades: &[TradeEvent],
    cash: &[CashEvent],
    from: Option<DateTime<Utc>>,
    to: Option<DateTime<Utc>>,
    marks_from: &Marks,
    marks_to: &Marks,
) -> BTreeMap<String, PnlBreakdown> {
    let mut out = BTreeMap::new();
    for s in strategies(trades) {
        let st = strategy_trades(trades, Some(s));
        out.insert(s.to_string(), range_pnl(method, &st, &[], from, to, marks_from, marks_to));
    }
    let un = strategy_trades(trades, None);
    out.insert(UNATTRIBUTED.to_owned(), range_pnl(method, &un, cash, from, to, marks_from, marks_to));
    out
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    fn ts(s: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(1_800_000_000 + s, 0).unwrap()
    }

    fn tr(n: u128, inst: InstrumentId, q: f64, px: f64, fx: f64, ccy: &str, at: i64) -> TradeEvent {
        TradeEvent {
            fill_uuid: Uuid::from_u128(n),
            fill_id: format!("f{n}"),
            intent_id: format!("i{n}"),
            venue: Uuid::from_u128(7),
            instrument: inst,
            qty: q,
            price: px,
            ccy: ccy.into(),
            fx_to_eur: fx,
            fx_source: "test".into(),
            fee_eur: 0.5,
            filled_at: ts(at),
            attribution: Vec::new(),
        }
    }

    fn cash(kind: &str, r: &str, amount: f64, ccy: &str, fx: f64, at: i64) -> CashEvent {
        CashEvent {
            kind: kind.into(),
            reference: r.into(),
            amount,
            ccy: ccy.into(),
            fx_to_eur: fx,
            occurred_at: ts(at),
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn fifo_closes_oldest_lots_first() {
        let i = InstrumentId::new();
        let t = vec![
            tr(1, i, 10.0, 100.0, 1.0, "EUR", 0),
            tr(2, i, 10.0, 110.0, 1.0, "EUR", 1),
            tr(3, i, -15.0, 120.0, 1.0, "EUR", 2),
        ];
        let s = replay(LotMethod::Fifo, &t, &[], None);
        assert_eq!(s.closings.len(), 2);
        assert!(close(s.closings[0].realised_native, 200.0) && close(s.closings[1].realised_native, 50.0));
        let lots = s.open_lots();
        assert_eq!(lots.len(), 1);
        assert_eq!((lots[0].lot_id, lots[0].qty, lots[0].price), (Uuid::from_u128(2), 5.0, 110.0));
        // average cost: 105 basis → 15 × 15 realised, 5 left at 105
        let a = replay(LotMethod::Average, &t, &[], None);
        let r: f64 = a.closings.iter().map(|c| c.realised_native).sum();
        assert!(close(r, 225.0));
        assert!(close(a.open_lots()[0].price, 105.0) && close(a.open_lots()[0].qty, 5.0));
    }

    #[test]
    fn short_lots_and_flips() {
        let i = InstrumentId::new();
        let t = vec![tr(1, i, -5.0, 100.0, 1.0, "EUR", 0), tr(2, i, 8.0, 90.0, 1.0, "EUR", 1)];
        let s = replay(LotMethod::Fifo, &t, &[], None);
        assert_eq!(s.closings.len(), 1);
        assert_eq!(s.closings[0].qty, -5.0);
        assert!(close(s.closings[0].realised_eur, 50.0), "short covered lower is a gain");
        let lots = s.open_lots();
        assert_eq!((lots[0].qty, lots[0].price, lots[0].lot_id), (3.0, 90.0, Uuid::from_u128(2)));
        let p = &s.positions()[0];
        assert_eq!((p.qty, p.avg_price, p.open_lots), (3.0, 90.0, 1));
    }

    #[test]
    fn fx_pnl_is_split_from_price_pnl() {
        let i = InstrumentId::new();
        // buy 10 @ 100 USD at 0.9, sell 4 @ 110 USD at 0.95, mark 6 @ 120 USD at 0.8
        let t = vec![tr(1, i, 10.0, 100.0, 0.9, "USD", 0), tr(2, i, -4.0, 110.0, 0.95, "USD", 1)];
        let c = vec![cash("deposit", "initial", 10_000.0, "EUR", 1.0, -1)];
        let s = replay(LotMethod::Fifo, &t, &c, None);
        let marks = Marks { prices: BTreeMap::from([(i, 120.0)]), fx: BTreeMap::from([("USD".into(), 0.8)]) };
        let p = s.pnl(&marks);
        assert!(close(p.realised_eur, 4.0 * 10.0 * 0.95));
        assert!(close(p.fx_realised_eur, 4.0 * 100.0 * 0.05));
        assert!(close(p.unrealised_eur, 6.0 * 20.0 * 0.8));
        assert!(close(p.fx_unrealised_eur, 6.0 * 100.0 * -0.1));
        assert!(close(p.fees_eur, 1.0));
        assert!(close(s.equity_eur(&marks) - s.external_eur, p.net_eur));
    }

    #[test]
    fn foreign_cash_is_revalued_and_cash_kinds_are_classified() {
        let c = vec![
            cash("deposit", "initial", 1_000.0, "EUR", 1.0, 0),
            cash("deposit", "usd", 100.0, "USD", 0.9, 1),
            cash("funding", "f1", -2.0, "EUR", 1.0, 2),
            cash("borrow", "b1", -1.0, "EUR", 1.0, 2),
            cash("dividend", "d1", 3.0, "EUR", 1.0, 2),
            cash("payout", "2026-11-11", -10.0, "EUR", 1.0, 3),
            cash("payout", "2026-11-11", -10.0, "EUR", 1.0, 3),
        ];
        let s = replay(LotMethod::Fifo, &[], &c, None);
        assert_eq!(s.cash_movements, 6, "replayed (kind, ref) ignored");
        assert!(close(s.payouts_eur, 10.0));
        let marks = Marks { fx: BTreeMap::from([("USD".into(), 0.95)]), ..Marks::default() };
        let p = s.pnl(&marks);
        assert!(close(p.fx_cash_eur, 5.0));
        assert!(close(p.funding_eur, -2.0) && close(p.borrow_eur, -1.0) && close(p.other_eur, 3.0));
        assert!(close(s.equity_eur(&marks) - s.external_eur, p.net_eur));
    }

    fn scenario() -> (Vec<TradeEvent>, Vec<CashEvent>, InstrumentId, InstrumentId) {
        let (a, b) = (InstrumentId::new(), InstrumentId::new());
        let (s1, s2) = (Uuid::from_u128(101), Uuid::from_u128(102));
        let at = |sa: f64, sb: f64| {
            let mut v = Vec::new();
            if sa > 0.0 {
                v.push(Attribution { strategy: s1, qty_frac: sa, fee_frac: sa });
            }
            if sb > 0.0 {
                v.push(Attribution { strategy: s2, qty_frac: sb, fee_frac: sb });
            }
            v
        };
        let mut t = vec![
            tr(1, a, 10.0, 100.0, 1.0, "EUR", 0),
            tr(2, b, -3.0, 50.0, 0.9, "USD", 5),
            tr(3, a, -4.0, 105.0, 1.0, "EUR", 10),
            tr(4, a, 6.0, 101.0, 1.0, "EUR", 20),
            tr(5, b, 5.0, 48.0, 0.92, "USD", 25),
            tr(6, a, -12.0, 99.0, 1.0, "EUR", 30),
            tr(7, b, -1.0, 49.0, 0.91, "USD", 40),
        ];
        let w = [(0.5, 0.5), (1.0, 0.0), (0.5, 0.5), (0.0, 1.0), (0.25, 0.25), (0.5, 0.5), (0.0, 0.0)];
        for (x, (sa, sb)) in t.iter_mut().zip(w) {
            x.attribution = at(sa, sb);
        }
        let c = vec![
            cash("deposit", "initial", 10_000.0, "EUR", 1.0, -10),
            cash("funding", "f", -1.5, "USD", 0.9, 15),
            cash("payout", "p", -20.0, "EUR", 1.0, 35),
        ];
        (t, c, a, b)
    }

    #[test]
    fn replay_is_order_independent_and_idempotent() {
        let (t, c, _, _) = scenario();
        let base = replay(LotMethod::Fifo, &t, &c, None);
        let mut shuffled = t.clone();
        shuffled.reverse();
        shuffled.extend(t.iter().cloned()); // every fill delivered twice, late and out of order
        let mut cr = c.clone();
        cr.reverse();
        let again = replay(LotMethod::Fifo, &shuffled, &cr, None);
        assert_eq!(base.open_lots(), again.open_lots());
        assert_eq!(base.closings, again.closings);
        assert_eq!(base.positions(), again.positions());
        assert_eq!(base.cash_eur, again.cash_eur);
        assert_eq!(base.fills, 7);
    }

    #[test]
    fn as_of_cutoff_and_range_pnl_compose() {
        let (t, c, a, _) = scenario();
        let m = Marks::default();
        let s = replay(LotMethod::Fifo, &t, &c, Some(ts(10)));
        assert_eq!(s.fills, 2, "fills strictly before the cutoff");
        assert_eq!(s.quantities()[&a], 10.0);
        // to_date(t2) == to_date(t1) + range(t1, t2), with each side's resolved marks
        let full = replay(LotMethod::Fifo, &t, &c, None);
        let m_end = full.resolved_marks(&m);
        let m_mid = replay(LotMethod::Fifo, &t, &c, Some(ts(22))).resolved_marks(&m);
        let first = range_pnl(LotMethod::Fifo, &t, &c, None, Some(ts(22)), &m, &m_mid);
        let second = range_pnl(LotMethod::Fifo, &t, &c, Some(ts(22)), None, &m_mid, &m_end);
        assert!(first.plus(&second).max_abs_diff(&full.pnl(&m_end)) < 1e-9);
    }

    #[test]
    fn strategy_pnl_sums_to_portfolio_pnl() {
        let (t, c, _, _) = scenario();
        for method in [LotMethod::Fifo, LotMethod::Average] {
            for (from, to) in [(None, None), (Some(ts(8)), Some(ts(33))), (Some(ts(21)), None)] {
                let m_to = replay(method, &t, &c, to).resolved_marks(&Marks::default());
                let m_from = from
                    .map_or_else(Marks::default, |f| replay(method, &t, &c, Some(f)).resolved_marks(&Marks::default()));
                let total = range_pnl(method, &t, &c, from, to, &m_from, &m_to);
                let per = strategy_pnls(method, &t, &c, from, to, &m_from, &m_to);
                assert_eq!(per.len(), 3);
                let sum = per.values().fold(PnlBreakdown::default(), |acc, p| acc.plus(p));
                assert!((sum.net_eur - total.net_eur).abs() < 1e-9, "{method:?} {from:?} {sum:?} vs {total:?}");
                assert!((sum.fees_eur - total.fees_eur).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn equity_identity_holds_at_every_cutoff() {
        let (t, c, a, b) = scenario();
        let marks =
            Marks { prices: BTreeMap::from([(a, 97.0), (b, 51.0)]), fx: BTreeMap::from([("USD".into(), 0.93)]) };
        for method in [LotMethod::Fifo, LotMethod::Average] {
            for cut in [0, 6, 11, 21, 26, 31, 41] {
                let s = replay(method, &t, &c, Some(ts(cut)));
                let p = s.pnl(&marks);
                assert!(close(s.equity_eur(&marks) - s.external_eur, p.net_eur), "{method:?} @ {cut}");
            }
        }
    }

    #[test]
    fn average_method_matches_the_in_memory_ledger() {
        let (t, c, _, _) = scenario();
        let s = replay(LotMethod::Average, &t, &c, None);
        let mut l = crate::Ledger::with_cash(0.0);
        for x in c.iter().filter(|x| x.ccy == "EUR") {
            l.cash_eur += x.amount_eur();
        }
        let mut sorted = t.clone();
        sorted.sort_by_key(|x| x.filled_at);
        for x in &sorted {
            l.apply_fill(&crate::Fill {
                fill_id: x.fill_id.clone(),
                intent_id: x.intent_id.clone(),
                instrument: x.instrument,
                quantity: x.qty,
                price: x.price,
                fx_to_eur: if x.ccy == "EUR" { 1.0 } else { x.fx_to_eur },
                fee_eur: x.fee_eur,
            });
        }
        assert!(close(l.cash_eur, s.cash_eur), "fills settle in EUR in both");
        assert!(close(l.fees_eur, s.fees_eur));
        for (i, q) in s.quantities() {
            assert!(close(l.positions[&i].quantity, q));
        }
        // the in-memory ledger realises in EUR terms: price + FX parts of the closings
        let realised: f64 = l.positions.values().map(|p| p.realised_eur).sum();
        let ours: f64 = s.closings.iter().map(|c| c.realised_eur + c.fx_eur).sum();
        assert!(close(realised, ours), "{realised} vs {ours}");
    }

    #[test]
    fn lot_method_parses() {
        assert_eq!("FIFO".parse::<LotMethod>().unwrap(), LotMethod::Fifo);
        assert_eq!("average".parse::<LotMethod>().unwrap(), LotMethod::Average);
        assert!("lifo".parse::<LotMethod>().is_err());
        assert_eq!(serde_json::to_string(&LotMethod::Average).unwrap(), "\"average\"");
    }
}
