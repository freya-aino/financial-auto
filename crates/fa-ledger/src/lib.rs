//! Ledger: positions, cash and P&L in EUR, fees, idempotent intents, reconciliation (spec 42).
//! [`Ledger`] is the in-memory book; [`pg::PgLedger`] persists intents, fills, cash movements,
//! snapshots and reconciliation runs to the `ledger.*` tables and rebuilds a `Ledger` from them.
//! [`accounting`] derives tax lots (FIFO or average cost), as-of positions, equity and the P&L
//! breakdown (realised / unrealised / FX / fees / funding / borrow) per book and per strategy
//! from the event stream; [`statement`] builds book statements; `pg_accounting` adds the
//! Postgres reads, the lot cache, orphan fills and payout booking to [`pg::PgLedger`].

pub mod accounting;
pub mod money;
pub mod payout;
pub mod pg;
pub mod pg_accounting;
pub mod restate;
pub mod settlement;
pub mod statement;

pub use accounting::{BookPosition, BookState, LotMethod, Marks, PnlBreakdown};
pub use money::{Decimal, dec};
pub use statement::Statement;

use std::collections::{BTreeMap, BTreeSet, HashSet};

use fa_core::InstrumentId;
use serde::{Deserialize, Serialize};

/// A fill as it arrives on the wire (`f64`, `fa.v1.OrderEvent`). The ledger books its exact
/// decimal form ([`ExactFill`], via [`money::dec`]); Postgres stores the same decimals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fill {
    /// Unique per venue fill; used for idempotency.
    pub fill_id: String,
    pub intent_id: String,
    pub instrument: InstrumentId,
    /// Signed quantity (+ buy, − sell).
    pub quantity: f64,
    /// Price in instrument currency.
    pub price: f64,
    /// FX rate instrument currency → EUR at fill time.
    pub fx_to_eur: f64,
    pub fee_eur: f64,
}

/// A fill in exact decimals at the ledger scale ([`money::SCALE`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactFill {
    pub fill_id: String,
    pub instrument: InstrumentId,
    /// Signed quantity (+ buy, − sell).
    pub qty: Decimal,
    pub price: Decimal,
    pub fx_to_eur: Decimal,
    pub fee_eur: Decimal,
}

impl ExactFill {
    /// The canonical decimal form of a wire fill (what [`pg::PgLedger::record_fill_with`] stores).
    pub fn of(f: &Fill) -> Self {
        Self {
            fill_id: f.fill_id.clone(),
            instrument: f.instrument,
            qty: dec(f.quantity),
            price: dec(f.price),
            fx_to_eur: dec(f.fx_to_eur),
            fee_eur: dec(f.fee_eur),
        }
    }

    /// EUR price per unit (`price × fx`, at the ledger scale).
    pub fn price_eur(&self) -> Decimal {
        money::mul(self.price, self.fx_to_eur)
    }

    /// Cash effect in EUR: `−(qty × price_eur + fee)`. Shared by booking and reconciliation, so
    /// both use the same rounding.
    pub fn cash_flow(&self) -> Decimal {
        -(money::mul(self.qty, self.price_eur()) + self.fee_eur)
    }
}

/// `f64` view of a position (risk arithmetic, JSON). The exact state is [`ExactPosition`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub quantity: f64,
    /// Average cost in EUR per unit.
    pub avg_cost_eur: f64,
    pub realised_eur: f64,
}

impl Position {
    /// Average-cost arithmetic in `f64` for virtual (attributed) books, which are estimates and
    /// not reconciled; the paper book itself books through [`Ledger::apply_fill`] in decimals.
    pub fn apply_f64(&mut self, quantity: f64, price_eur: f64) {
        let q0 = self.quantity;
        let q1 = q0 + quantity;
        if q0 == 0.0 || q0.signum() == quantity.signum() {
            self.avg_cost_eur = (self.avg_cost_eur * q0.abs() + price_eur * quantity.abs()) / q1.abs();
        } else {
            let closed = quantity.abs().min(q0.abs());
            self.realised_eur += closed * (price_eur - self.avg_cost_eur) * q0.signum();
            if q1 != 0.0 && q1.signum() != q0.signum() {
                self.avg_cost_eur = price_eur;
            }
        }
        self.quantity = q1;
        if self.quantity.abs() < 1e-12 {
            self.quantity = 0.0;
            self.avg_cost_eur = 0.0;
        }
    }
}

/// Exact position state (average cost in EUR).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactPosition {
    pub qty: Decimal,
    pub avg_cost_eur: Decimal,
    pub realised_eur: Decimal,
}

impl ExactPosition {
    fn apply(&mut self, qty: Decimal, px: Decimal) {
        let q0 = self.qty;
        let q1 = q0 + qty;
        if q0.is_zero() || q0.is_sign_negative() == qty.is_sign_negative() {
            // opening / adding
            let cost = money::mul(self.avg_cost_eur, q0.abs()) + money::mul(px, qty.abs());
            self.avg_cost_eur = money::div(cost, q1.abs());
        } else {
            // reducing / closing / flipping
            let closed = qty.abs().min(q0.abs());
            let pnl = money::mul(closed, px - self.avg_cost_eur);
            self.realised_eur += if q0.is_sign_negative() { -pnl } else { pnl };
            if !q1.is_zero() && q1.is_sign_negative() != q0.is_sign_negative() {
                self.avg_cost_eur = px; // flipped: remainder opened at fill price
            }
        }
        self.qty = q1;
        if self.qty.is_zero() {
            self.qty = Decimal::ZERO;
            self.avg_cost_eur = Decimal::ZERO;
        }
    }

    fn view(&self) -> Position {
        Position {
            quantity: money::f(self.qty),
            avg_cost_eur: money::f(self.avg_cost_eur),
            realised_eur: money::f(self.realised_eur),
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum LedgerError {
    #[error("duplicate intent id {0}")]
    DuplicateIntent(String),
}

/// The exact state of a [`Ledger`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactBook {
    pub cash_eur: Decimal,
    pub fees_eur: Decimal,
    pub paid_out_eur: Decimal,
    pub positions: BTreeMap<InstrumentId, ExactPosition>,
}

/// The in-memory paper book. Its state is exact ([`ExactBook`], decimals at the ledger scale);
/// the public `f64` fields are read-only views refreshed after every change, kept for the risk
/// arithmetic and JSON. Mutate only through the methods.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Ledger {
    /// View of [`ExactBook::cash_eur`].
    pub cash_eur: f64,
    /// View of [`ExactBook::fees_eur`].
    pub fees_eur: f64,
    /// Views of [`ExactBook::positions`].
    pub positions: BTreeMap<InstrumentId, Position>,
    intents: HashSet<String>,
    fills: HashSet<String>,
    /// Cash movements applied (`venue:kind:ref`), so syncing them is idempotent.
    #[serde(default)]
    cash_refs: BTreeSet<String>,
    /// View of [`ExactBook::paid_out_eur`] (payout quota tracking).
    pub paid_out_eur: f64,
    #[serde(default)]
    exact: ExactBook,
}

/// Additive totals of a book (cash, fees, quantity per instrument): what reconciliation
/// compares. Unlike average cost and realised P&L they do not depend on the order fills are
/// applied in, so a fill can be added or taken out exactly.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Totals {
    pub cash_eur: Decimal,
    pub fees_eur: Decimal,
    pub qty: BTreeMap<InstrumentId, Decimal>,
}

impl Totals {
    pub fn of(l: &Ledger) -> Self {
        Self {
            cash_eur: l.exact.cash_eur,
            fees_eur: l.exact.fees_eur,
            qty: l.exact.positions.iter().map(|(i, p)| (*i, p.qty)).collect(),
        }
    }

    pub fn add_fill(&mut self, f: &ExactFill) {
        self.cash_eur += f.cash_flow();
        self.fees_eur += f.fee_eur;
        *self.qty.entry(f.instrument).or_default() += f.qty;
    }

    pub fn remove_fill(&mut self, f: &ExactFill) {
        self.cash_eur -= f.cash_flow();
        self.fees_eur -= f.fee_eur;
        *self.qty.entry(f.instrument).or_default() -= f.qty;
    }

    pub fn add_cash(&mut self, amount_eur: Decimal) {
        self.cash_eur += amount_eur;
    }
}

/// Cash and position differences between two books (reconciliation, spec 42 §6). Amounts are
/// reported as `f64`; the comparison itself is exact.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BookDiff {
    pub cash_ours: f64,
    pub cash_theirs: f64,
    pub positions: Vec<ReconDiff>,
    /// Fills booked in ours that theirs does not have (e.g. a fill that never reached Postgres).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fills_only_ours: Vec<String>,
    /// Fills in theirs that ours never booked.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fills_only_theirs: Vec<String>,
}

impl BookDiff {
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
            && self.cash_ours == self.cash_theirs
            && self.fills_only_ours.is_empty()
            && self.fills_only_theirs.is_empty()
    }
}

/// Compares two sets of totals: cash beyond `cash_tol` and quantities beyond `qty_tol` (both
/// in EUR / units; `0.0` = exact).
pub fn diff_totals(ours: &Totals, theirs: &Totals, cash_tol: f64, qty_tol: f64) -> BookDiff {
    let (ctol, qtol) = (dec(cash_tol).abs(), dec(qty_tol).abs());
    let mut keys: Vec<InstrumentId> = ours.qty.keys().chain(theirs.qty.keys()).copied().collect();
    keys.sort();
    keys.dedup();
    let positions = keys
        .into_iter()
        .filter_map(|i| {
            let o = ours.qty.get(&i).copied().unwrap_or_default();
            let b = theirs.qty.get(&i).copied().unwrap_or_default();
            ((o - b).abs() > qtol).then(|| ReconDiff { instrument: i, ours: money::f(o), broker: money::f(b) })
        })
        .collect();
    let cash_ours = money::f(ours.cash_eur);
    let cash_theirs =
        if (ours.cash_eur - theirs.cash_eur).abs() > ctol { money::f(theirs.cash_eur) } else { cash_ours };
    BookDiff { cash_ours, cash_theirs, positions, ..BookDiff::default() }
}

/// Compares two books exactly: cash beyond `cash_tol`, position quantities beyond `qty_tol`
/// and the sets of booked fill ids.
pub fn diff_books(ours: &Ledger, theirs: &Ledger, cash_tol: f64, qty_tol: f64) -> BookDiff {
    let mut d = diff_totals(&Totals::of(ours), &Totals::of(theirs), cash_tol, qty_tol);
    let mut only_ours: Vec<String> = ours.fills.difference(&theirs.fills).cloned().collect();
    let mut only_theirs: Vec<String> = theirs.fills.difference(&ours.fills).cloned().collect();
    only_ours.sort();
    only_theirs.sort();
    d.fills_only_ours = only_ours;
    d.fills_only_theirs = only_theirs;
    d
}

/// Key of a cash movement in [`Ledger`]'s applied set (`ledger.cash_movement` is unique on
/// `(book, venue_id, kind, ref)`).
pub fn cash_key(venue: &uuid::Uuid, kind: &str, reference: &str) -> String {
    format!("{venue}:{kind}:{reference}")
}

impl Ledger {
    pub fn with_cash(cash_eur: f64) -> Self {
        Self::with_cash_exact(dec(cash_eur))
    }

    pub fn with_cash_exact(cash_eur: Decimal) -> Self {
        let mut l = Self::default();
        l.exact.cash_eur = cash_eur;
        l.refresh_totals();
        l
    }

    /// The exact state.
    pub fn exact(&self) -> &ExactBook {
        &self.exact
    }

    /// Ids of every fill booked.
    pub fn fill_ids(&self) -> &HashSet<String> {
        &self.fills
    }

    pub fn has_fill(&self, fill_id: &str) -> bool {
        self.fills.contains(fill_id)
    }

    /// Cash movements applied through [`Self::apply_cash`].
    pub fn cash_refs(&self) -> &BTreeSet<String> {
        &self.cash_refs
    }

    /// Registers an order intent; intent ids are idempotency keys.
    pub fn register_intent(&mut self, intent_id: &str) -> Result<(), LedgerError> {
        if !self.intents.insert(intent_id.to_owned()) {
            return Err(LedgerError::DuplicateIntent(intent_id.to_owned()));
        }
        Ok(())
    }

    /// Applies a wire fill in its canonical decimal form. Replays of the same `fill_id` are
    /// ignored (returns false).
    pub fn apply_fill(&mut self, f: &Fill) -> bool {
        self.apply_exact(&ExactFill::of(f))
    }

    /// Applies an exact fill (average-cost positions, realised P&L in EUR). Idempotent on
    /// `fill_id`.
    pub fn apply_exact(&mut self, f: &ExactFill) -> bool {
        if !self.fills.insert(f.fill_id.clone()) {
            return false;
        }
        self.exact.cash_eur += f.cash_flow();
        self.exact.fees_eur += f.fee_eur;
        let p = self.exact.positions.entry(f.instrument).or_default();
        p.apply(f.qty, f.price_eur());
        let view = p.view();
        self.positions.insert(f.instrument, view);
        self.refresh_totals();
        true
    }

    /// Applies a cash movement (EUR amount; negative = outflow) once per `key` ([`cash_key`]).
    /// Payouts also count towards `paid_out_eur`.
    pub fn apply_cash(&mut self, key: &str, kind: &str, amount_eur: Decimal) -> bool {
        if !self.cash_refs.insert(key.to_owned()) {
            return false;
        }
        self.exact.cash_eur += amount_eur;
        if kind == "payout" {
            self.exact.paid_out_eur -= amount_eur;
        }
        self.refresh_totals();
        true
    }

    /// An untracked payout (in-memory books without Postgres; see [`Self::apply_cash`]).
    pub fn record_payout(&mut self, eur: f64) {
        let d = dec(eur);
        self.exact.cash_eur -= d;
        self.exact.paid_out_eur += d;
        self.refresh_totals();
    }

    fn refresh_totals(&mut self) {
        self.cash_eur = money::f(self.exact.cash_eur);
        self.fees_eur = money::f(self.exact.fees_eur);
        self.paid_out_eur = money::f(self.exact.paid_out_eur);
    }

    /// Equity in EUR given marks (EUR per unit).
    pub fn equity(&self, marks_eur: &BTreeMap<InstrumentId, f64>) -> f64 {
        self.cash_eur
            + self
                .positions
                .iter()
                .map(|(i, p)| p.quantity * marks_eur.get(i).copied().unwrap_or(p.avg_cost_eur))
                .sum::<f64>()
    }
}

/// One mismatch between our ledger and the broker's view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReconDiff {
    pub instrument: InstrumentId,
    pub ours: f64,
    pub broker: f64,
}

/// Compares positions with the broker's statement; differences above `tol` are reported.
pub fn reconcile(ours: &Ledger, broker: &BTreeMap<InstrumentId, f64>, tol: f64) -> Vec<ReconDiff> {
    let mut keys: Vec<InstrumentId> = ours.positions.keys().chain(broker.keys()).copied().collect();
    keys.sort();
    keys.dedup();
    keys.into_iter()
        .filter_map(|i| {
            let o = ours.positions.get(&i).map_or(0.0, |p| p.quantity);
            let b = broker.get(&i).copied().unwrap_or(0.0);
            ((o - b).abs() > tol).then_some(ReconDiff { instrument: i, ours: o, broker: b })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(id: &str, inst: InstrumentId, q: f64, px: f64) -> Fill {
        Fill {
            fill_id: id.into(),
            intent_id: "i".into(),
            instrument: inst,
            quantity: q,
            price: px,
            fx_to_eur: 1.0,
            fee_eur: 1.0,
        }
    }

    #[test]
    fn round_trip_realises_pnl_net_of_fees() {
        let i = InstrumentId::new();
        let mut l = Ledger::with_cash(10_000.0);
        assert!(l.apply_fill(&fill("f1", i, 10.0, 100.0)));
        assert!(!l.apply_fill(&fill("f1", i, 10.0, 100.0)), "replayed fill ignored");
        l.apply_fill(&fill("f2", i, -10.0, 110.0));
        assert_eq!(l.positions[&i].quantity, 0.0);
        assert!((l.positions[&i].realised_eur - 100.0).abs() < 1e-9);
        assert!((l.cash_eur - 10_098.0).abs() < 1e-9);
        assert_eq!(l.fees_eur, 2.0);
    }

    #[test]
    fn short_and_flip() {
        let i = InstrumentId::new();
        let mut l = Ledger::with_cash(0.0);
        l.apply_fill(&fill("a", i, -5.0, 100.0));
        l.apply_fill(&fill("b", i, 8.0, 90.0));
        let p = &l.positions[&i];
        assert_eq!(p.quantity, 3.0);
        assert!((p.realised_eur - 50.0).abs() < 1e-9);
        assert_eq!(p.avg_cost_eur, 90.0);
    }

    #[test]
    fn fx_conversion_and_equity() {
        let i = InstrumentId::new();
        let mut l = Ledger::with_cash(1_000.0);
        let mut f = fill("a", i, 1.0, 100.0);
        f.fx_to_eur = 0.9;
        f.fee_eur = 0.0;
        l.apply_fill(&f);
        assert!((l.cash_eur - 910.0).abs() < 1e-9);
        assert!((l.equity(&BTreeMap::from([(i, 95.0)])) - 1_005.0).abs() < 1e-9);
    }

    #[test]
    fn diff_books_reports_cash_and_positions() {
        let i = InstrumentId::new();
        let mut a = Ledger::with_cash(100.0);
        let mut b = Ledger::with_cash(100.0);
        a.apply_fill(&fill("f", i, 1.0, 10.0));
        b.apply_fill(&fill("f", i, 1.0, 10.0));
        assert!(diff_books(&a, &b, 1e-9, 1e-9).is_empty());
        b.record_payout(5.0);
        let d = diff_books(&a, &b, 1e-9, 1e-9);
        assert!(!d.is_empty());
        assert_eq!((d.cash_ours, d.cash_theirs), (89.0, 84.0));
        assert!(d.positions.is_empty());
    }

    /// Regression for the 2026-11-10/11 reconciliation diffs: the same fills applied in a
    /// different order (Postgres replays by `filled_at`, memory by arrival) give identical
    /// cash and quantities to the last digit, and a fill can be taken out of the totals exactly.
    #[test]
    fn totals_are_order_independent_and_exact() {
        let i = InstrumentId::new();
        let raw = [
            (2.85021203500556, 101.80316189408),
            (-1.15531427142454, 101.804900229592),
            (0.930829918855023, 102.007725167811),
            (-13.9885223944754, 101.553184926104),
            (11.8284533302793, 101.707185048985),
            (1.11897472440106, 100.77850057833),
        ];
        let fills: Vec<Fill> = raw
            .iter()
            .enumerate()
            .map(|(n, (q, px)): (usize, &(f64, f64))| Fill {
                fee_eur: 0.0123 * q.abs(),
                ..fill(&format!("f{n}"), i, *q, *px)
            })
            .collect();
        let mut a = Ledger::with_cash(100_000.0);
        let mut b = Ledger::with_cash(100_000.0);
        for f in &fills {
            a.apply_fill(f);
        }
        for f in fills.iter().rev() {
            b.apply_fill(f);
        }
        assert_eq!(Totals::of(&a), Totals::of(&b));
        assert!(diff_books(&a, &b, 0.0, 0.0).is_empty());
        // a fill still in flight is excluded exactly
        let mut c = Ledger::with_cash(100_000.0);
        for f in &fills[..5] {
            c.apply_fill(f);
        }
        let mut t = Totals::of(&a);
        t.remove_fill(&ExactFill::of(&fills[5]));
        assert_eq!(t, Totals::of(&c));
        let d = diff_books(&a, &c, 0.0, 0.0);
        assert_eq!(d.fills_only_ours, vec!["f5".to_owned()]);
        assert_eq!(d.positions.len(), 1);
    }

    #[test]
    fn cash_movements_apply_once() {
        let mut l = Ledger::with_cash(0.0);
        let k = cash_key(&uuid::Uuid::nil(), "deposit", "initial");
        assert!(l.apply_cash(&k, "deposit", dec(100_000.0)));
        assert!(!l.apply_cash(&k, "deposit", dec(100_000.0)));
        assert!(l.apply_cash("x:payout:2026-11-11", "payout", dec(-25.5)));
        assert_eq!(l.exact().cash_eur, dec(99_974.5));
        assert_eq!(l.paid_out_eur, 25.5);
        assert!(l.cash_refs().contains(&k));
    }

    #[test]
    fn f64_position_arithmetic_matches_the_exact_book() {
        let i = InstrumentId::new();
        let mut l = Ledger::with_cash(0.0);
        let mut p = Position::default();
        for (n, (q, px)) in [(5.0, 100.0), (3.0, 110.0), (-10.0, 90.0), (4.0, 95.0)].iter().enumerate() {
            l.apply_fill(&Fill { fee_eur: 0.0, ..fill(&format!("f{n}"), i, *q, *px) });
            p.apply_f64(*q, *px);
        }
        let v = &l.positions[&i];
        assert!((v.quantity - p.quantity).abs() < 1e-12);
        assert!((v.avg_cost_eur - p.avg_cost_eur).abs() < 1e-9);
        assert!((v.realised_eur - p.realised_eur).abs() < 1e-9);
    }

    #[test]
    fn idempotent_intents_and_reconciliation() {
        let mut l = Ledger::default();
        l.register_intent("x").unwrap();
        assert_eq!(l.register_intent("x"), Err(LedgerError::DuplicateIntent("x".into())));
        let (a, b) = (InstrumentId::new(), InstrumentId::new());
        l.apply_fill(&fill("f", a, 2.0, 1.0));
        let d = reconcile(&l, &BTreeMap::from([(a, 2.0), (b, 1.0)]), 1e-9);
        assert_eq!(d, vec![ReconDiff { instrument: b, ours: 0.0, broker: 1.0 }]);
    }
}
