//! Book statements (spec 42 §6): opening and closing balances, fills, cash movements, lot
//! closings and the P&L breakdown over `[from, to)`, as a typed JSON-serialisable struct with a
//! flat CSV export. Built purely from events (see [`crate::accounting`]), so a statement is
//! reproducible and unaffected by the order in which late events arrived.
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{
    accounting::{
        BookPosition, BookState, CashBalance, CashEvent, LotClosing, LotMethod, Marks, PnlBreakdown, TradeEvent, replay,
    },
    money::{self, Decimal},
};

/// Balances at one end of a statement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatementBalances {
    pub at: Option<DateTime<Utc>>,
    #[serde(with = "crate::money::json")]
    pub cash_eur: Decimal,
    pub foreign_cash: Vec<CashBalance>,
    /// Open positions only.
    pub positions: Vec<BookPosition>,
    #[serde(with = "crate::money::json")]
    pub equity_eur: Decimal,
    /// Marks used for valuation (after completion from the latest fills).
    pub marks: Marks,
}

impl StatementBalances {
    fn of(s: &BookState, at: Option<DateTime<Utc>>, marks: &Marks) -> Self {
        Self {
            at,
            cash_eur: s.cash_eur,
            foreign_cash: s.foreign_cash.values().cloned().collect(),
            positions: s.positions().into_iter().filter(|p| !p.qty.is_zero()).collect(),
            equity_eur: s.equity_eur(marks),
            marks: marks.clone(),
        }
    }
}

/// A fill line with native and EUR values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatementFill {
    pub filled_at: DateTime<Utc>,
    pub fill_id: String,
    pub intent_id: String,
    pub venue: uuid::Uuid,
    pub instrument: fa_core::InstrumentId,
    #[serde(with = "crate::money::json")]
    pub qty: Decimal,
    #[serde(with = "crate::money::json")]
    pub price: Decimal,
    pub ccy: String,
    #[serde(with = "crate::money::json")]
    pub fx_to_eur: Decimal,
    pub fx_source: String,
    #[serde(with = "crate::money::json")]
    pub notional_native: Decimal,
    #[serde(with = "crate::money::json")]
    pub notional_eur: Decimal,
    #[serde(with = "crate::money::json")]
    pub fee_eur: Decimal,
}

impl From<&TradeEvent> for StatementFill {
    fn from(t: &TradeEvent) -> Self {
        Self {
            filled_at: t.filled_at,
            fill_id: t.fill_id.clone(),
            intent_id: t.intent_id.clone(),
            venue: t.venue,
            instrument: t.instrument,
            qty: t.qty,
            price: t.price,
            ccy: t.ccy.clone(),
            fx_to_eur: t.fx_to_eur,
            fx_source: t.fx_source.clone(),
            notional_native: t.notional_native(),
            notional_eur: t.notional_eur(),
            fee_eur: t.fee_eur,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Statement {
    pub book: String,
    pub from: Option<DateTime<Utc>>,
    pub to: Option<DateTime<Utc>>,
    pub method: LotMethod,
    pub opening: StatementBalances,
    pub closing: StatementBalances,
    pub fills: Vec<StatementFill>,
    pub cash_movements: Vec<CashEvent>,
    pub lot_closings: Vec<LotClosing>,
    pub pnl: PnlBreakdown,
    #[serde(with = "crate::money::json")]
    pub fees_eur: Decimal,
    #[serde(with = "crate::money::json")]
    pub turnover_eur: Decimal,
    /// Net capital flows (deposits − withdrawals − payouts) in the range, EUR.
    #[serde(with = "crate::money::json")]
    pub external_flows_eur: Decimal,
    #[serde(with = "crate::money::json")]
    pub payouts_eur: Decimal,
    /// `closing.equity − opening.equity − external_flows − pnl.net`; 0 (to the 18th place) for a
    /// consistent book.
    #[serde(with = "crate::money::json")]
    pub unexplained_eur: Decimal,
}

/// Builds the statement of `[from, to)`. `marks_from` / `marks_to` override prices and rates at
/// each end; whatever they lack is completed from the latest fills before that end.
#[allow(clippy::too_many_arguments)]
pub fn build_statement(
    book: &str,
    method: LotMethod,
    trades: &[TradeEvent],
    cash: &[CashEvent],
    from: Option<DateTime<Utc>>,
    to: Option<DateTime<Utc>>,
    marks_from: &Marks,
    marks_to: &Marks,
) -> Statement {
    let open = match from {
        Some(f) => replay(method, trades, cash, Some(f)),
        None => BookState::new(method),
    };
    let close = replay(method, trades, cash, to);
    let m_open = open.resolved_marks(marks_from);
    let m_close = close.resolved_marks(marks_to);
    let in_range = |t: &DateTime<Utc>| from.is_none_or(|f| *t >= f) && to.is_none_or(|e| *t < e);

    let mut fills: Vec<&TradeEvent> = trades.iter().filter(|t| in_range(&t.filled_at)).collect();
    fills.sort_by(|a, b| a.filled_at.cmp(&b.filled_at).then(a.fill_uuid.cmp(&b.fill_uuid)));
    fills.dedup_by_key(|t| t.fill_uuid);
    let mut cash_movements: Vec<CashEvent> = cash.iter().filter(|c| in_range(&c.occurred_at)).cloned().collect();
    cash_movements.sort_by(|a, b| {
        a.occurred_at.cmp(&b.occurred_at).then_with(|| a.kind.cmp(&b.kind)).then_with(|| a.reference.cmp(&b.reference))
    });
    cash_movements.dedup_by(|a, b| a.kind == b.kind && a.reference == b.reference);
    let lot_closings: Vec<LotClosing> = close.closings.iter().filter(|c| in_range(&c.closed_at)).cloned().collect();

    let pnl = close.pnl(&m_close).minus(&open.pnl(&m_open));
    let opening = StatementBalances::of(&open, from, &m_open);
    let closing = StatementBalances::of(&close, to, &m_close);
    let external_flows_eur = close.external_eur - open.external_eur;
    let unexplained_eur = closing.equity_eur - opening.equity_eur - external_flows_eur - pnl.net_eur;
    Statement {
        book: book.to_owned(),
        from,
        to,
        method,
        fees_eur: pnl.fees_eur,
        turnover_eur: money::sum(fills.iter().map(|t| t.notional_eur().abs())),
        fills: fills.into_iter().map(StatementFill::from).collect(),
        cash_movements,
        lot_closings,
        external_flows_eur,
        payouts_eur: close.payouts_eur - open.payouts_eur,
        unexplained_eur,
        pnl,
        opening,
        closing,
    }
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_owned() }
}

fn ts(t: Option<DateTime<Utc>>) -> String {
    t.map(|t| t.to_rfc3339()).unwrap_or_default()
}

impl Statement {
    /// Flat CSV: one row per line item, `section` says which (`opening_position`, `opening_cash`,
    /// `fill`, `cash_movement`, `lot_closing`, `closing_position`, `closing_cash`, `pnl`, `summary`).
    pub fn to_csv(&self) -> String {
        const HEADER: &str = "section,at,ref,venue,instrument,ccy,qty,price,fx_to_eur,amount_native,amount_eur,note";
        let mut out = vec![HEADER.to_owned()];
        let mut row = |section: &str,
                       at: String,
                       r: &str,
                       venue: String,
                       inst: String,
                       ccy: &str,
                       nums: [Option<Decimal>; 5],
                       note: &str| {
            let n: Vec<String> =
                nums.iter().map(|x| x.map(|v| v.normalize().to_string()).unwrap_or_default()).collect();
            out.push(
                [section, &at, r, &venue, &inst, ccy, &n[0], &n[1], &n[2], &n[3], &n[4], note]
                    .iter()
                    .map(|s| csv_field(s))
                    .collect::<Vec<_>>()
                    .join(","),
            );
        };
        balance_rows(&mut row, "opening_position", "opening_cash", &self.opening);
        for f in &self.fills {
            row(
                "fill",
                f.filled_at.to_rfc3339(),
                &f.fill_id,
                f.venue.to_string(),
                f.instrument.0.to_string(),
                &f.ccy,
                [Some(f.qty), Some(f.price), Some(f.fx_to_eur), Some(f.notional_native), Some(f.notional_eur)],
                &format!("fee_eur={} fx_source={} intent={}", f.fee_eur, f.fx_source, f.intent_id),
            );
        }
        for c in &self.cash_movements {
            row(
                "cash_movement",
                c.occurred_at.to_rfc3339(),
                &c.reference,
                String::new(),
                String::new(),
                &c.ccy,
                [None, None, Some(c.fx_to_eur), Some(c.amount), Some(c.amount_eur())],
                &c.kind,
            );
        }
        for c in &self.lot_closings {
            row(
                "lot_closing",
                c.closed_at.to_rfc3339(),
                &c.close_fill_id,
                c.venue.to_string(),
                c.instrument.0.to_string(),
                &c.ccy,
                [Some(c.qty), Some(c.close_price), Some(c.close_fx), Some(c.realised_native), Some(c.realised_eur)],
                &format!("lot={} open_price={} open_fx={} fx_eur={}", c.lot_id, c.open_price, c.open_fx, c.fx_eur),
            );
        }
        balance_rows(&mut row, "closing_position", "closing_cash", &self.closing);
        let p = &self.pnl;
        for (k, v) in [
            ("realised_eur", p.realised_eur),
            ("unrealised_eur", p.unrealised_eur),
            ("fx_realised_eur", p.fx_realised_eur),
            ("fx_unrealised_eur", p.fx_unrealised_eur),
            ("fx_cash_eur", p.fx_cash_eur),
            ("fx_eur", p.fx_eur),
            ("fees_eur", p.fees_eur),
            ("funding_eur", p.funding_eur),
            ("borrow_eur", p.borrow_eur),
            ("other_eur", p.other_eur),
            ("net_eur", p.net_eur),
        ] {
            row("pnl", ts(self.to), k, String::new(), String::new(), "EUR", [None, None, None, None, Some(v)], "");
        }
        for (k, v) in [
            ("opening_equity_eur", self.opening.equity_eur),
            ("closing_equity_eur", self.closing.equity_eur),
            ("external_flows_eur", self.external_flows_eur),
            ("payouts_eur", self.payouts_eur),
            ("turnover_eur", self.turnover_eur),
            ("unexplained_eur", self.unexplained_eur),
        ] {
            row(
                "summary",
                ts(self.to),
                k,
                String::new(),
                String::new(),
                "EUR",
                [None, None, None, None, Some(v)],
                &self.book,
            );
        }
        let mut s = out.join("\n");
        s.push('\n');
        s
    }
}

type RowFn<'a> = dyn FnMut(&str, String, &str, String, String, &str, [Option<Decimal>; 5], &str) + 'a;

fn balance_rows(row: &mut RowFn<'_>, pos_section: &str, cash_section: &str, b: &StatementBalances) {
    for p in &b.positions {
        row(
            pos_section,
            ts(b.at),
            "",
            p.venue.to_string(),
            p.instrument.0.to_string(),
            &p.ccy,
            [Some(p.qty), Some(p.avg_price), None, Some(money::mul(p.qty, p.avg_price)), Some(p.cost_eur)],
            &format!("open_lots={}", p.open_lots),
        );
    }
    row(
        cash_section,
        ts(b.at),
        "",
        String::new(),
        String::new(),
        "EUR",
        [None, None, Some(Decimal::ONE), Some(b.cash_eur), Some(b.cash_eur)],
        "",
    );
    for c in &b.foreign_cash {
        row(
            cash_section,
            ts(b.at),
            "",
            String::new(),
            String::new(),
            &c.ccy,
            [None, None, None, Some(c.balance), Some(c.booked_eur)],
            "booked_eur at booking rates",
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use chrono::TimeZone;
    use fa_core::InstrumentId;
    use uuid::Uuid;

    use super::*;
    use crate::money::dec;

    fn t(n: u128, i: InstrumentId, q: f64, px: f64, at: i64) -> TradeEvent {
        TradeEvent {
            fill_uuid: Uuid::from_u128(n),
            fill_id: format!("f,{n}"),
            intent_id: format!("i{n}"),
            venue: Uuid::from_u128(1),
            instrument: i,
            qty: dec(q),
            price: dec(px),
            ccy: "EUR".into(),
            fx_to_eur: Decimal::ONE,
            fx_source: "eur".into(),
            fee_eur: Decimal::ONE,
            filled_at: Utc.timestamp_opt(1_800_000_000 + at, 0).unwrap(),
            attribution: vec![],
            settle_native: false,
        }
    }

    #[test]
    fn statement_balances_and_explains_equity() {
        let i = InstrumentId::new();
        let at = |s: i64| Utc.timestamp_opt(1_800_000_000 + s, 0).unwrap();
        let trades =
            vec![t(1, i, 10.0, 100.0, 0), t(2, i, -4.0, 110.0, 100), t(3, i, 2.0, 90.0, 200), t(4, i, -1.0, 95.0, 400)];
        let cash = vec![
            CashEvent {
                kind: "deposit".into(),
                reference: "initial".into(),
                amount: dec(5_000.0),
                ccy: "EUR".into(),
                fx_to_eur: Decimal::ONE,
                occurred_at: at(-1),
            },
            CashEvent {
                kind: "payout".into(),
                reference: "d1".into(),
                amount: dec(-50.0),
                ccy: "EUR".into(),
                fx_to_eur: Decimal::ONE,
                occurred_at: at(150),
            },
        ];
        let marks_to = Marks { prices: BTreeMap::from([(i, dec(92.0))]), ..Marks::default() };
        let s = build_statement(
            "b",
            LotMethod::Fifo,
            &trades,
            &cash,
            Some(at(50)),
            Some(at(300)),
            &Marks::default(),
            &marks_to,
        );
        assert_eq!(s.fills.len(), 2);
        assert_eq!(s.cash_movements.len(), 1);
        assert_eq!(s.lot_closings.len(), 1);
        assert_eq!(s.opening.positions[0].qty, dec(10.0));
        assert_eq!(s.closing.positions[0].qty, dec(8.0));
        assert_eq!((s.payouts_eur, s.external_flows_eur), (dec(50.0), dec(-50.0)));
        assert_eq!(s.pnl.realised_eur, dec(40.0));
        assert_eq!(s.fees_eur, dec(2.0));
        assert_eq!(s.unexplained_eur, Decimal::ZERO, "exact: {s:?}");
        // reproducible regardless of input order, JSON round trip, CSV escapes
        let mut rev = trades.clone();
        rev.reverse();
        let s2 = build_statement(
            "b",
            LotMethod::Fifo,
            &rev,
            &cash,
            Some(at(50)),
            Some(at(300)),
            &Marks::default(),
            &marks_to,
        );
        assert_eq!(s, s2);
        let back: Statement = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back, s);
        let csv = s.to_csv();
        assert!(csv.starts_with("section,at,ref"));
        assert!(csv.contains("\"f,2\""), "{csv}");
        assert!(csv.lines().any(|l| l.starts_with("pnl,") && l.contains("net_eur")));
        assert_eq!(csv.lines().filter(|l| l.starts_with("fill,")).count(), 2);
        assert_eq!(csv.lines().filter(|l| l.starts_with("opening_position,")).count(), 1);
    }
}
