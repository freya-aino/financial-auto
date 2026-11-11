//! Postgres reads and writes for derived accounting (spec 42 §6, §8, §9): events as
//! [`TradeEvent`]/[`CashEvent`], as-of positions and equity, the P&L breakdown per book and per
//! strategy, statements, the tax-lot cache (`ledger.lot`, `ledger.lot_closing`), orphan fills
//! (`ledger.orphan_fill`) and payout booking. Tables come from migration
//! `20261111000030_ledger_lots_orphans`.
//!
//! Everything derived is recomputed from `ledger.fill` + `ledger.cash_movement` on each call
//! (pure replay, [`crate::accounting::replay`]); `rebuild_lots` only caches the result.
//! Valuation: prices default to the latest fill of each instrument before the cutoff (the ledger
//! has no market-data access; pass [`Marks`] for mark-to-market), FX to the latest
//! `ref.fx_rate` at or before the cutoff (ECB daily, D-061), else the latest fill's rate.
use std::collections::BTreeMap;

use chrono::{DateTime, Duration, NaiveDate, NaiveTime, Utc};
use fa_core::InstrumentId;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqlx::{Row, postgres::PgRow};
use uuid::Uuid;

use crate::{
    Fill, ReconDiff,
    accounting::{
        Attribution, BookPosition, BookState, CashEvent, Lot, LotClosing, LotMethod, Marks, PnlBreakdown, TradeEvent,
        UNATTRIBUTED, replay, strategy_pnls, strategy_trades,
    },
    money::dec,
    pg::{CashMovement, FillDetails, IntentRecord, PAPER_VENUE, PgError, PgLedger, uuid_of},
    statement::{Statement, build_statement},
};

/// Actor written to `audit.log` by ledger-side actions without a caller-supplied actor.
pub const LEDGER_ACTOR: &str = "system:fa-ledger";

/// Exclusive cutoff equivalent to "as of `at` inclusive" at Postgres' microsecond resolution.
fn inclusive(at: Option<DateTime<Utc>>) -> Option<DateTime<Utc>> {
    at.map(|t| t + Duration::microseconds(1))
}

/// A fill as delivered by a venue, with everything needed to book it or park it as an orphan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillEvent {
    /// `fill.intent_id` is the client order id.
    pub fill: Fill,
    pub filled_at: DateTime<Utc>,
    #[serde(default)]
    pub strategy_ids: Vec<String>,
    #[serde(default)]
    pub details: FillDetails,
    /// Book the venue account belongs to, when known (recorded on an orphan).
    #[serde(default)]
    pub book: Option<String>,
    /// Venue (defaults to [`PAPER_VENUE`]).
    #[serde(default)]
    pub venue: Option<Uuid>,
}

/// Anything the ledger records (spec 42 §4 `Ledger::record`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LedgerEvent {
    Intent(IntentRecord),
    Fill(FillEvent),
    Cash(CashMovement),
}

/// Outcome of [`PgLedger::record`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "outcome")]
pub enum Recorded {
    New,
    /// Already recorded (idempotent replay).
    Duplicate,
    /// Fill for an unknown client order id, parked in `ledger.orphan_fill` (`new` = first time).
    Orphaned {
        id: Uuid,
        new: bool,
    },
}

/// A row of `ledger.orphan_fill`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrphanFill {
    pub id: Uuid,
    pub book: Option<String>,
    pub venue_id: Uuid,
    pub client_order_id: String,
    pub venue_fill_id: String,
    pub instrument: InstrumentId,
    /// Signed (+ buy).
    pub qty: f64,
    pub price: f64,
    pub ccy: String,
    pub fee_eur: f64,
    pub fx_rate_eur: f64,
    pub fx_source: String,
    pub arrival_price: Option<f64>,
    pub slippage_bps: Option<f64>,
    pub strategy_ids: Vec<String>,
    pub filled_at: DateTime<Utc>,
    pub received_at: DateTime<Utc>,
    /// `open` | `booked` | `dismissed`
    pub status: String,
    pub resolved_at: Option<DateTime<Utc>>,
    pub resolved_by: Option<String>,
    pub resolution: Option<String>,
}

impl OrphanFill {
    pub fn to_fill(&self) -> Fill {
        Fill {
            fill_id: self.venue_fill_id.clone(),
            intent_id: self.client_order_id.clone(),
            instrument: self.instrument,
            quantity: self.qty,
            price: self.price,
            fx_to_eur: self.fx_rate_eur,
            fee_eur: self.fee_eur,
        }
    }
}

/// How an operator resolves an orphan fill.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "action")]
pub enum OrphanAction {
    /// Book it as a normal fill now that its order is known (fails while it is still unknown).
    Book,
    /// Leave it out of the ledger, with a reason.
    Dismiss { reason: String },
}

/// Result of [`PgLedger::rebuild_lots`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LotRebuild {
    pub book: String,
    pub method: LotMethod,
    pub fills: usize,
    pub open_lots: usize,
    pub closed_lots: usize,
    pub closings: usize,
}

/// Derived positions vs the latest `position_snapshot` batch (spec 42 §9).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotCheck {
    /// `as_of` of the latest snapshot batch (`None`: the book has no snapshot).
    pub as_of: Option<DateTime<Utc>>,
    pub instruments: usize,
    pub diffs: Vec<ReconDiff>,
}

fn trade_row(r: &PgRow) -> TradeEvent {
    let qty: f64 = r.get("qty");
    let fee: f64 = r.get("fee");
    let side: String = r.get("side");
    let attr: Value = r.get("attr");
    let attribution = attr
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|a| {
            let s = Uuid::parse_str(a.get(0)?.as_str()?).ok()?;
            let q = a.get(1)?.as_f64()?;
            let f = a.get(2)?.as_f64().unwrap_or(0.0);
            let qty_frac = if qty.abs() > 0.0 { q / qty.abs() } else { 0.0 };
            let fee_frac = if fee.abs() > 0.0 { f / fee } else { qty_frac };
            Some(Attribution { strategy: s, qty_frac, fee_frac })
        })
        .collect();
    TradeEvent {
        fill_uuid: r.get("id"),
        fill_id: r.get("venue_fill_id"),
        intent_id: r.get("intent_id"),
        venue: r.get("venue_id"),
        instrument: InstrumentId(r.get::<Uuid, _>("instrument_id")),
        qty: if side == "sell" { -qty } else { qty },
        price: r.get("price"),
        ccy: r.get("ccy"),
        fx_to_eur: r.get("fx"),
        fx_source: r.get("fx_source"),
        fee_eur: fee,
        filled_at: r.get("filled_at"),
        attribution,
        settle_native: false,
    }
}

fn orphan_row(r: &PgRow) -> OrphanFill {
    let ids: Value = r.get("strategy_ids");
    OrphanFill {
        id: r.get("id"),
        book: r.get("book"),
        venue_id: r.get("venue_id"),
        client_order_id: r.get("client_order_id"),
        venue_fill_id: r.get("venue_fill_id"),
        instrument: InstrumentId(r.get::<Uuid, _>("instrument_id")),
        qty: r.get("qty"),
        price: r.get("price"),
        ccy: r.get("ccy"),
        fee_eur: r.get("fee"),
        fx_rate_eur: r.get("fx"),
        fx_source: r.get("fx_source"),
        arrival_price: r.get("arrival_price"),
        slippage_bps: r.get("slippage_bps"),
        strategy_ids: serde_json::from_value(ids).unwrap_or_default(),
        filled_at: r.get("filled_at"),
        received_at: r.get("received_at"),
        status: r.get("status"),
        resolved_at: r.get("resolved_at"),
        resolved_by: r.get("resolved_by"),
        resolution: r.get("resolution"),
    }
}

const ORPHAN_COLS: &str = "id, book, venue_id, client_order_id, venue_fill_id, instrument_id, qty::float8 AS qty, \
     price::float8 AS price, ccy, fee::float8 AS fee, fx_rate_eur::float8 AS fx, fx_source, \
     arrival_price::float8 AS arrival_price, slippage_bps::float8 AS slippage_bps, strategy_ids, filled_at, \
     received_at, status, resolved_at, resolved_by, resolution";

async fn audit<'e, E: sqlx::PgExecutor<'e>>(
    ex: E,
    actor: &str,
    action: &str,
    subject: &str,
    payload: Value,
) -> Result<(), PgError> {
    sqlx::query("INSERT INTO audit.log (actor, action, subject, payload) VALUES ($1, $2, $3, $4)")
        .bind(actor)
        .bind(action)
        .bind(subject)
        .bind(payload)
        .execute(ex)
        .await?;
    Ok(())
}

impl PgLedger {
    // ---------------------------------------------------------------- events

    /// Fills of a book with timestamp `< before` (all when `None`) as accounting events, with
    /// venue and strategy attribution, oldest first.
    pub async fn trade_events(&self, book: &str, before: Option<DateTime<Utc>>) -> Result<Vec<TradeEvent>, PgError> {
        let rows = sqlx::query(
            "SELECT f.id, f.venue_fill_id, oi.idempotency_key AS intent_id, vo.venue_id, oi.instrument_id, oi.side, \
             f.qty::float8 AS qty, f.price::float8 AS price, f.ccy, f.fx_rate_eur::float8 AS fx, f.fx_source, \
             f.fee::float8 AS fee, f.filled_at, \
             COALESCE((SELECT jsonb_agg(jsonb_build_array(fa.strategy_id::text, fa.qty::float8, fa.fee_share::float8) \
                                        ORDER BY fa.strategy_id) \
                       FROM ledger.fill_attribution fa WHERE fa.fill_id = f.id), '[]'::jsonb) AS attr \
             FROM ledger.fill f \
             JOIN ledger.venue_order vo ON vo.client_order_id = f.client_order_id \
             JOIN ledger.order_intent oi ON oi.id = vo.intent_id \
             WHERE oi.book = $1 AND ($2::timestamptz IS NULL OR f.filled_at < $2) \
             ORDER BY f.filled_at, f.id",
        )
        .bind(book)
        .bind(before)
        .fetch_all(self.pool())
        .await?;
        Ok(rows
            .iter()
            .map(|r| {
                let mut t = trade_row(r);
                t.settle_native = self.settlement().of(&t.venue) == crate::settlement::Settlement::Native;
                t
            })
            .collect())
    }

    /// Cash movements of a book with `occurred_at < before` (all when `None`), oldest first.
    pub async fn cash_events(&self, book: &str, before: Option<DateTime<Utc>>) -> Result<Vec<CashEvent>, PgError> {
        let rows = sqlx::query(
            "SELECT kind, ref, amount::float8 AS amount, ccy, fx_rate_eur::float8 AS fx, occurred_at \
             FROM ledger.cash_movement WHERE book = $1 AND ($2::timestamptz IS NULL OR occurred_at < $2) \
             ORDER BY occurred_at, kind, ref",
        )
        .bind(book)
        .bind(before)
        .fetch_all(self.pool())
        .await?;
        Ok(rows
            .iter()
            .map(|r| CashEvent {
                kind: r.get("kind"),
                reference: r.get("ref"),
                amount: r.get("amount"),
                ccy: r.get("ccy"),
                fx_to_eur: r.get("fx"),
                occurred_at: r.get("occurred_at"),
            })
            .collect())
    }

    /// The book replayed from events strictly before `before` (all when `None`).
    pub async fn book_state(&self, book: &str, before: Option<DateTime<Utc>>) -> Result<BookState, PgError> {
        let trades = self.trade_events(book, before).await?;
        let cash = self.cash_events(book, before).await?;
        Ok(replay(self.lot_method(), &trades, &cash, before))
    }

    // ---------------------------------------------------------------- valuation

    /// Latest `ref.fx_rate.rate_to_eur` per currency with `ts ≤ at` (any source). EUR is omitted.
    pub async fn fx_rates_at(&self, ccys: &[String], at: DateTime<Utc>) -> Result<BTreeMap<String, f64>, PgError> {
        let ccys: Vec<String> =
            ccys.iter().map(|c| c.to_ascii_uppercase()).filter(|c| c != fa_core::BASE_CURRENCY).collect();
        if ccys.is_empty() {
            return Ok(BTreeMap::new());
        }
        let rows = sqlx::query(
            "SELECT DISTINCT ON (ccy) ccy, rate_to_eur::float8 AS rate FROM ref.fx_rate \
             WHERE ccy = ANY($1) AND ts <= $2 ORDER BY ccy, ts DESC",
        )
        .bind(&ccys)
        .bind(at)
        .fetch_all(self.pool())
        .await?;
        Ok(rows.iter().map(|r| (r.get::<String, _>("ccy"), r.get::<f64, _>("rate"))).collect())
    }

    /// Marks for `state` at `at` (now when `None`): `base`, then `ref.fx_rate` as of `at` for
    /// the book's currencies, then the latest fills.
    pub async fn marks_for(
        &self,
        state: &BookState,
        at: Option<DateTime<Utc>>,
        base: &Marks,
    ) -> Result<Marks, PgError> {
        let mut m = base.clone();
        let ccys: Vec<String> = state.currencies().into_iter().filter(|c| base.fx(c).is_none()).collect();
        for (c, x) in self.fx_rates_at(&ccys, at.unwrap_or_else(Utc::now)).await? {
            m.fx.insert(c, x);
        }
        Ok(state.resolved_marks(&m))
    }

    /// Positions per `(venue, instrument)` as of `at` inclusive (now when `None`), from events.
    pub async fn positions(&self, book: &str, at: Option<DateTime<Utc>>) -> Result<Vec<BookPosition>, PgError> {
        Ok(self.book_state(book, inclusive(at)).await?.positions())
    }

    /// Open tax lots as of `at` inclusive.
    pub async fn open_lots(&self, book: &str, at: Option<DateTime<Utc>>) -> Result<Vec<Lot>, PgError> {
        Ok(self.book_state(book, inclusive(at)).await?.open_lots())
    }

    /// Equity in EUR as of `at` inclusive: cash + revalued foreign cash + open lots at `marks`
    /// (completed from `ref.fx_rate` and the latest fills).
    pub async fn equity_eur(
        &self,
        book: &str,
        at: Option<DateTime<Utc>>,
        marks: Option<&Marks>,
    ) -> Result<f64, PgError> {
        let s = self.book_state(book, inclusive(at)).await?;
        let m = self.marks_for(&s, at, marks.unwrap_or(&Marks::default())).await?;
        Ok(s.equity_eur(&m))
    }

    /// Events and resolved marks at both ends of `[from, to)`.
    async fn range_inputs(
        &self,
        book: &str,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
        marks_to: Option<&Marks>,
    ) -> Result<(Vec<TradeEvent>, Vec<CashEvent>, Marks, Marks), PgError> {
        let trades = self.trade_events(book, to).await?;
        let cash = self.cash_events(book, to).await?;
        let method = self.lot_method();
        let end = replay(method, &trades, &cash, to);
        let m_to = self.marks_for(&end, to, marks_to.unwrap_or(&Marks::default())).await?;
        let m_from = match from {
            Some(f) => self.marks_for(&replay(method, &trades, &cash, Some(f)), Some(f), &Marks::default()).await?,
            None => Marks::default(),
        };
        Ok((trades, cash, m_from, m_to))
    }

    /// Book P&L over `[from, to)` (`to = None`: now), realised / unrealised / FX / fees /
    /// funding / borrow / other separated.
    pub async fn book_pnl(
        &self,
        book: &str,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
        marks_to: Option<&Marks>,
    ) -> Result<PnlBreakdown, PgError> {
        let (trades, cash, m_from, m_to) = self.range_inputs(book, from, to, marks_to).await?;
        Ok(crate::accounting::range_pnl(self.lot_method(), &trades, &cash, from, to, &m_from, &m_to))
    }

    /// P&L of one strategy (by its id string, hashed like `fill_attribution.strategy_id`) over
    /// `[from, to)`: its attributed share of every fill, valued at the book's marks. Book-level
    /// cash items (funding, borrow, …) are not attributed; see [`Self::strategy_pnls`].
    pub async fn strategy_pnl(
        &self,
        book: &str,
        strategy: &str,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
    ) -> Result<PnlBreakdown, PgError> {
        let (trades, _, m_from, m_to) = self.range_inputs(book, from, to, None).await?;
        let st = strategy_trades(&trades, Some(uuid_of(strategy)));
        Ok(crate::accounting::range_pnl(self.lot_method(), &st, &[], from, to, &m_from, &m_to))
    }

    /// P&L per strategy UUID over `[from, to)` plus [`UNATTRIBUTED`] (unattributed trades and
    /// book-level cash items); the values sum to [`Self::book_pnl`].
    pub async fn strategy_pnls(
        &self,
        book: &str,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
    ) -> Result<BTreeMap<String, PnlBreakdown>, PgError> {
        let (trades, cash, m_from, m_to) = self.range_inputs(book, from, to, None).await?;
        let mut out = strategy_pnls(self.lot_method(), &trades, &cash, from, to, &m_from, &m_to);
        out.entry(UNATTRIBUTED.to_owned()).or_default();
        Ok(out)
    }

    /// Statement of `[from, to)` (`None` = since inception / up to now); `marks_to` overrides
    /// closing prices (opening ones come from fills and `ref.fx_rate`).
    pub async fn statement(
        &self,
        book: &str,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
        marks_to: Option<&Marks>,
    ) -> Result<Statement, PgError> {
        let (trades, cash, m_from, m_to) = self.range_inputs(book, from, to, marks_to).await?;
        Ok(build_statement(book, self.lot_method(), &trades, &cash, from, to, &m_from, &m_to))
    }

    /// Daily statement for `day` (UTC midnight to midnight).
    pub async fn daily_statement(&self, book: &str, day: NaiveDate) -> Result<Statement, PgError> {
        let from = day.and_time(NaiveTime::MIN).and_utc();
        self.statement(book, Some(from), Some(from + Duration::days(1)), None).await
    }

    // ---------------------------------------------------------------- lots cache

    /// Recomputes the book's lots from all fills under the configured method and replaces the
    /// cached `ledger.lot` / `ledger.lot_closing` rows in one transaction. Deterministic, so
    /// running it twice yields identical rows; run it after late fills to refresh the cache.
    pub async fn rebuild_lots(&self, book: &str) -> Result<LotRebuild, PgError> {
        let method = self.lot_method();
        let s = self.book_state(book, None).await?;
        let lots: Vec<Lot> = s.open_lots().into_iter().chain(s.closed_lots.iter().cloned()).collect();
        let mut tx = self.pool().begin().await?;
        for sql in [
            "DELETE FROM ledger.lot WHERE book = $1 AND method = $2",
            "DELETE FROM ledger.lot_closing WHERE book = $1 AND method = $2",
        ] {
            sqlx::query(sql).bind(book).bind(method.as_str()).execute(&mut *tx).await?;
        }
        for chunk in lots.chunks(5_000) {
            sqlx::query(
                "INSERT INTO ledger.lot (book, method, lot_id, venue_id, instrument_id, opened_at, open_qty, \
                 remaining_qty, price, ccy, fx_rate_eur) \
                 SELECT $1, $2, * FROM UNNEST($3::uuid[], $4::uuid[], $5::uuid[], $6::timestamptz[], $7::float8[], \
                                              $8::float8[], $9::float8[], $10::text[], $11::float8[])",
            )
            .bind(book)
            .bind(method.as_str())
            .bind(chunk.iter().map(|l| l.lot_id).collect::<Vec<_>>())
            .bind(chunk.iter().map(|l| l.venue).collect::<Vec<_>>())
            .bind(chunk.iter().map(|l| l.instrument.0).collect::<Vec<_>>())
            .bind(chunk.iter().map(|l| l.opened_at).collect::<Vec<_>>())
            .bind(chunk.iter().map(|l| l.open_qty).collect::<Vec<_>>())
            .bind(chunk.iter().map(|l| l.qty).collect::<Vec<_>>())
            .bind(chunk.iter().map(|l| l.price).collect::<Vec<_>>())
            .bind(chunk.iter().map(|l| l.ccy.clone()).collect::<Vec<_>>())
            .bind(chunk.iter().map(|l| l.fx_to_eur).collect::<Vec<_>>())
            .execute(&mut *tx)
            .await?;
        }
        for chunk in s.closings.chunks(4_000) {
            sqlx::query(
                "INSERT INTO ledger.lot_closing (book, method, lot_id, close_fill_id, venue_id, instrument_id, \
                 opened_at, closed_at, qty, open_price, close_price, ccy, open_fx, close_fx, realised_native, \
                 realised_eur, fx_eur) \
                 SELECT $1, $2, * FROM UNNEST($3::uuid[], $4::uuid[], $5::uuid[], $6::uuid[], $7::timestamptz[], \
                   $8::timestamptz[], $9::float8[], $10::float8[], $11::float8[], $12::text[], $13::float8[], \
                   $14::float8[], $15::float8[], $16::float8[], $17::float8[])",
            )
            .bind(book)
            .bind(method.as_str())
            .bind(chunk.iter().map(|c| c.lot_id).collect::<Vec<_>>())
            .bind(chunk.iter().map(|c| c.close_fill).collect::<Vec<_>>())
            .bind(chunk.iter().map(|c| c.venue).collect::<Vec<_>>())
            .bind(chunk.iter().map(|c| c.instrument.0).collect::<Vec<_>>())
            .bind(chunk.iter().map(|c| c.opened_at).collect::<Vec<_>>())
            .bind(chunk.iter().map(|c| c.closed_at).collect::<Vec<_>>())
            .bind(chunk.iter().map(|c| c.qty).collect::<Vec<_>>())
            .bind(chunk.iter().map(|c| c.open_price).collect::<Vec<_>>())
            .bind(chunk.iter().map(|c| c.close_price).collect::<Vec<_>>())
            .bind(chunk.iter().map(|c| c.ccy.clone()).collect::<Vec<_>>())
            .bind(chunk.iter().map(|c| c.open_fx).collect::<Vec<_>>())
            .bind(chunk.iter().map(|c| c.close_fx).collect::<Vec<_>>())
            .bind(chunk.iter().map(|c| c.realised_native).collect::<Vec<_>>())
            .bind(chunk.iter().map(|c| c.realised_eur).collect::<Vec<_>>())
            .bind(chunk.iter().map(|c| c.fx_eur).collect::<Vec<_>>())
            .execute(&mut *tx)
            .await?;
        }
        let open = s.open_lots().len();
        sqlx::query(
            "INSERT INTO ledger.lot_rebuild (book, method, rebuilt_at, fills, lots, closings) \
             VALUES ($1, $2, now(), $3, $4, $5) ON CONFLICT (book, method) DO UPDATE SET \
             rebuilt_at = excluded.rebuilt_at, fills = excluded.fills, lots = excluded.lots, closings = excluded.closings",
        )
        .bind(book)
        .bind(method.as_str())
        .bind(s.fills as i64)
        .bind(lots.len() as i64)
        .bind(s.closings.len() as i64)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(LotRebuild {
            book: book.to_owned(),
            method,
            fills: s.fills,
            open_lots: open,
            closed_lots: s.closed_lots.len(),
            closings: s.closings.len(),
        })
    }

    /// Cached lots of the book under the configured method (open and closed), as written by
    /// [`Self::rebuild_lots`], ordered by `(venue, instrument, opened_at, lot_id)`.
    pub async fn cached_lots(&self, book: &str) -> Result<Vec<Lot>, PgError> {
        let rows = sqlx::query(
            "SELECT lot_id, venue_id, instrument_id, opened_at, open_qty::float8 AS open_qty, \
             remaining_qty::float8 AS qty, price::float8 AS price, ccy, fx_rate_eur::float8 AS fx \
             FROM ledger.lot WHERE book = $1 AND method = $2 ORDER BY venue_id, instrument_id, opened_at, lot_id",
        )
        .bind(book)
        .bind(self.lot_method().as_str())
        .fetch_all(self.pool())
        .await?;
        Ok(rows
            .iter()
            .map(|r| Lot {
                lot_id: r.get("lot_id"),
                venue: r.get("venue_id"),
                instrument: InstrumentId(r.get::<Uuid, _>("instrument_id")),
                opened_at: r.get("opened_at"),
                open_qty: r.get("open_qty"),
                qty: r.get("qty"),
                price: r.get("price"),
                ccy: r.get("ccy"),
                fx_to_eur: r.get("fx"),
            })
            .collect())
    }

    /// Cached lot closings in `[from, to)` ordered by `(closed_at, close_fill, lot_id)`.
    pub async fn cached_lot_closings(
        &self,
        book: &str,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
    ) -> Result<Vec<LotClosing>, PgError> {
        let rows = sqlx::query(
            "SELECT c.lot_id, c.close_fill_id, f.venue_fill_id, c.venue_id, c.instrument_id, c.opened_at, c.closed_at, \
             c.qty::float8 AS qty, c.open_price::float8 AS op, c.close_price::float8 AS cp, c.ccy, \
             c.open_fx::float8 AS ofx, c.close_fx::float8 AS cfx, c.realised_native::float8 AS rn, \
             c.realised_eur::float8 AS re, c.fx_eur::float8 AS fe \
             FROM ledger.lot_closing c LEFT JOIN ledger.fill f ON f.id = c.close_fill_id \
             WHERE c.book = $1 AND c.method = $2 \
               AND ($3::timestamptz IS NULL OR c.closed_at >= $3) AND ($4::timestamptz IS NULL OR c.closed_at < $4) \
             ORDER BY c.closed_at, c.close_fill_id, c.lot_id",
        )
        .bind(book)
        .bind(self.lot_method().as_str())
        .bind(from)
        .bind(to)
        .fetch_all(self.pool())
        .await?;
        Ok(rows
            .iter()
            .map(|r| LotClosing {
                lot_id: r.get("lot_id"),
                close_fill: r.get("close_fill_id"),
                close_fill_id: r.get::<Option<String>, _>("venue_fill_id").unwrap_or_default(),
                venue: r.get("venue_id"),
                instrument: InstrumentId(r.get::<Uuid, _>("instrument_id")),
                opened_at: r.get("opened_at"),
                closed_at: r.get("closed_at"),
                qty: r.get("qty"),
                open_price: r.get("op"),
                close_price: r.get("cp"),
                ccy: r.get("ccy"),
                open_fx: r.get("ofx"),
                close_fx: r.get("cfx"),
                realised_native: r.get("rn"),
                realised_eur: r.get("re"),
                fx_eur: r.get("fe"),
            })
            .collect())
    }

    // ---------------------------------------------------------------- checks

    /// Derived quantities as of the latest `position_snapshot` batch vs that batch (spec 42 §9
    /// "positions derived from events equal the latest position snapshot").
    pub async fn snapshot_check(&self, book: &str, qty_tol: f64) -> Result<SnapshotCheck, PgError> {
        let (as_of,): (Option<DateTime<Utc>>,) =
            sqlx::query_as("SELECT max(as_of) FROM ledger.position_snapshot WHERE book = $1")
                .bind(book)
                .fetch_one(self.pool())
                .await?;
        let Some(as_of) = as_of else {
            return Ok(SnapshotCheck { as_of: None, instruments: 0, diffs: Vec::new() });
        };
        let rows = sqlx::query(
            "SELECT instrument_id, sum(qty)::float8 AS qty FROM ledger.position_snapshot \
             WHERE book = $1 AND as_of = $2 GROUP BY instrument_id",
        )
        .bind(book)
        .bind(as_of)
        .fetch_all(self.pool())
        .await?;
        let snap: BTreeMap<InstrumentId, f64> =
            rows.iter().map(|r| (InstrumentId(r.get::<Uuid, _>("instrument_id")), r.get::<f64, _>("qty"))).collect();
        let derived = self.book_state(book, inclusive(Some(as_of))).await?.quantities();
        let mut keys: Vec<InstrumentId> = snap.keys().chain(derived.keys()).copied().collect();
        keys.sort();
        keys.dedup();
        let diffs = keys
            .iter()
            .filter_map(|i| {
                let o = derived.get(i).copied().unwrap_or(0.0);
                let b = snap.get(i).copied().unwrap_or(0.0);
                ((o - b).abs() > qty_tol).then_some(ReconDiff { instrument: *i, ours: o, broker: b })
            })
            .collect();
        Ok(SnapshotCheck { as_of: Some(as_of), instruments: keys.len(), diffs })
    }

    /// Fills of the book lacking a native or EUR value, an FX source or an FX timestamp
    /// (`ledger.fill_valued`); 0 is the spec 42 §9 acceptance.
    pub async fn fill_valuation_gaps(&self, book: &str) -> Result<i64, PgError> {
        let (n,): (i64,) = sqlx::query_as(
            "SELECT count(*) FROM ledger.fill_valued WHERE book = $1 AND (notional_native IS NULL \
             OR notional_eur IS NULL OR fx_rate_eur IS NULL OR fx_rate_eur <= 0 OR coalesce(fx_source, '') = '' \
             OR fx_as_of IS NULL)",
        )
        .bind(book)
        .fetch_one(self.pool())
        .await?;
        Ok(n)
    }

    // ---------------------------------------------------------------- recording

    /// Records any ledger event idempotently (spec 42 §4): intents and cash movements as
    /// [`Self::record_intent`] / [`Self::record_cash_movement`], fills as [`Self::record_fill_event`].
    pub async fn record(&self, ev: &LedgerEvent) -> Result<Recorded, PgError> {
        let new = |b: bool| if b { Recorded::New } else { Recorded::Duplicate };
        match ev {
            LedgerEvent::Intent(i) => Ok(new(self.record_intent(i).await?)),
            LedgerEvent::Cash(c) => Ok(new(self.record_cash_movement(c).await?)),
            LedgerEvent::Fill(f) => self.record_fill_event(f).await,
        }
    }

    /// Books a fill; a fill whose client order id is unknown is parked in `ledger.orphan_fill`
    /// (and audited) instead of failing (spec 42 §8). Booking a fill that was parked earlier
    /// marks the orphan `booked`.
    pub async fn record_fill_event(&self, e: &FillEvent) -> Result<Recorded, PgError> {
        match self.record_fill_with(&e.fill, e.filled_at, &e.strategy_ids, &e.details).await {
            Ok(true) => {
                sqlx::query(
                    "UPDATE ledger.orphan_fill SET status = 'booked', resolved_at = now(), resolved_by = $3, \
                     resolution = 'order became known; booked on redelivery' \
                     WHERE client_order_id = $1 AND venue_fill_id = $2 AND status = 'open'",
                )
                .bind(&e.fill.intent_id)
                .bind(&e.fill.fill_id)
                .bind(LEDGER_ACTOR)
                .execute(self.pool())
                .await?;
                Ok(Recorded::New)
            }
            Ok(false) => Ok(Recorded::Duplicate),
            Err(PgError::UnknownIntent { .. }) => self.record_orphan(e).await,
            Err(err) => Err(err),
        }
    }

    async fn record_orphan(&self, e: &FillEvent) -> Result<Recorded, PgError> {
        let f = &e.fill;
        let x = crate::ExactFill::of(f);
        let id = uuid_of(&format!("orphan:{}:{}", f.intent_id, f.fill_id));
        let venue = e.venue.unwrap_or(PAPER_VENUE);
        let ccy = if e.details.ccy.is_empty() { fa_core::BASE_CURRENCY } else { e.details.ccy.as_str() };
        let src = if e.details.fx_source.is_empty() { "fixed" } else { e.details.fx_source.as_str() };
        let mut tx = self.pool().begin().await?;
        let n = sqlx::query(
            "INSERT INTO ledger.orphan_fill (id, book, venue_id, client_order_id, venue_fill_id, instrument_id, qty, \
             price, ccy, fee, fee_ccy, fx_rate_eur, fx_source, arrival_price, slippage_bps, strategy_ids, filled_at) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'EUR', $11, $12, $13, $14, $15, $16) \
             ON CONFLICT DO NOTHING",
        )
        .bind(id)
        .bind(&e.book)
        .bind(venue)
        .bind(&f.intent_id)
        .bind(&f.fill_id)
        .bind(f.instrument.0)
        .bind(x.qty)
        .bind(x.price)
        .bind(ccy)
        .bind(x.fee_eur)
        .bind(x.fx_to_eur)
        .bind(src)
        .bind(e.details.arrival_price.filter(|v| v.is_finite()).map(dec))
        .bind(e.details.slippage_bps.filter(|v| v.is_finite()).map(dec))
        .bind(json!(e.strategy_ids))
        .bind(e.filled_at)
        .execute(&mut *tx)
        .await?
        .rows_affected();
        if n > 0 {
            audit(
                &mut *tx,
                LEDGER_ACTOR,
                "ledger.orphan_fill",
                &f.intent_id,
                json!({"orphan_id": id, "venue_id": venue, "book": e.book, "venue_fill_id": f.fill_id,
                       "instrument_id": f.instrument, "qty": f.quantity, "price": f.price}),
            )
            .await?;
        }
        tx.commit().await?;
        Ok(Recorded::Orphaned { id, new: n > 0 })
    }

    /// Orphan fills, newest first; `status` filters (`open`, `booked`, `dismissed`).
    pub async fn orphan_fills(&self, status: Option<&str>, limit: i64) -> Result<Vec<OrphanFill>, PgError> {
        let sql = format!(
            "SELECT {ORPHAN_COLS} FROM ledger.orphan_fill WHERE ($1::text IS NULL OR status = $1) \
             ORDER BY received_at DESC, id LIMIT $2"
        );
        let rows = sqlx::query(sqlx::AssertSqlSafe(sql)).bind(status).bind(limit).fetch_all(self.pool()).await?;
        Ok(rows.iter().map(orphan_row).collect())
    }

    /// Open orphans (optionally of one venue): non-zero means the venue should stay halted.
    pub async fn open_orphan_count(&self, venue: Option<Uuid>) -> Result<i64, PgError> {
        let (n,): (i64,) = sqlx::query_as(
            "SELECT count(*) FROM ledger.orphan_fill WHERE status = 'open' AND ($1::uuid IS NULL OR venue_id = $1)",
        )
        .bind(venue)
        .fetch_one(self.pool())
        .await?;
        Ok(n)
    }

    /// Resolves an open orphan: book it as a normal fill (its order must now be known and on the
    /// same side) or dismiss it. Audited with `actor`. Returns the updated row.
    pub async fn resolve_orphan(&self, id: Uuid, action: &OrphanAction, actor: &str) -> Result<OrphanFill, PgError> {
        let sql = format!("SELECT {ORPHAN_COLS} FROM ledger.orphan_fill WHERE id = $1");
        let o = sqlx::query(sqlx::AssertSqlSafe(sql))
            .bind(id)
            .fetch_optional(self.pool())
            .await?
            .map(|r| orphan_row(&r))
            .ok_or_else(|| PgError::NotFound(format!("orphan fill {id}")))?;
        if o.status != "open" {
            return Err(PgError::Invalid(format!("orphan fill {id} is already {}", o.status)));
        }
        let (status, resolution) = match action {
            OrphanAction::Book => {
                let side: Option<(String,)> = sqlx::query_as(
                    "SELECT oi.side FROM ledger.venue_order vo JOIN ledger.order_intent oi ON oi.id = vo.intent_id \
                     WHERE vo.client_order_id = $1",
                )
                .bind(&o.client_order_id)
                .fetch_optional(self.pool())
                .await?;
                let Some((side,)) = side else {
                    return Err(PgError::UnknownIntent {
                        fill_id: o.venue_fill_id.clone(),
                        intent_id: o.client_order_id.clone(),
                    });
                };
                if (side == "sell") != (o.qty < 0.0) {
                    return Err(PgError::Invalid(format!(
                        "orphan fill {id} has qty {} but order {} is a {side}",
                        o.qty, o.client_order_id
                    )));
                }
                let details = FillDetails {
                    ccy: o.ccy.clone(),
                    fx_source: o.fx_source.clone(),
                    arrival_price: o.arrival_price,
                    slippage_bps: o.slippage_bps,
                };
                let new = self.record_fill_with(&o.to_fill(), o.filled_at, &o.strategy_ids, &details).await?;
                ("booked", if new { "booked by operator".to_owned() } else { "already booked (duplicate)".to_owned() })
            }
            OrphanAction::Dismiss { reason } => ("dismissed", reason.clone()),
        };
        let mut tx = self.pool().begin().await?;
        sqlx::query(
            "UPDATE ledger.orphan_fill SET status = $2, resolved_at = now(), resolved_by = $3, resolution = $4 \
             WHERE id = $1 AND status = 'open'",
        )
        .bind(id)
        .bind(status)
        .bind(actor)
        .bind(&resolution)
        .execute(&mut *tx)
        .await?;
        audit(
            &mut *tx,
            actor,
            "ledger.orphan_fill.resolve",
            &o.client_order_id,
            json!({"orphan_id": id, "status": status, "resolution": resolution}),
        )
        .await?;
        tx.commit().await?;
        let sql = format!("SELECT {ORPHAN_COLS} FROM ledger.orphan_fill WHERE id = $1");
        let r = sqlx::query(sqlx::AssertSqlSafe(sql)).bind(id).fetch_one(self.pool()).await?;
        Ok(orphan_row(&r))
    }

    // ---------------------------------------------------------------- payouts

    /// Computes `day`'s payout quota for `book` with `rule` ([`crate::payout`]) from the book
    /// as of the start of `day` (previous close equity, previous day's net P&L, previous quota
    /// and payouts) and stores it in `ledger.payout_quota` — only ever raising the stored quota
    /// (tighten-only, spec 40 §6.5), with the rule and its inputs in `rule`. Returns the quota
    /// now in force for the day.
    pub async fn apply_payout_rule(
        &self,
        book: &str,
        day: NaiveDate,
        rule: &crate::payout::PayoutRule,
    ) -> Result<f64, PgError> {
        rule.validate().map_err(PgError::Invalid)?;
        let start = day.and_time(NaiveTime::MIN).and_utc();
        let prev = day.pred_opt().unwrap_or(day);
        let prev_start = prev.and_time(NaiveTime::MIN).and_utc();
        let equity = self.equity_eur(book, Some(start - Duration::microseconds(1)), None).await?;
        let pnl = self.book_pnl(book, Some(prev_start), Some(start), None).await?;
        let prev_q = self.payout_quota(prev).await?;
        let inputs = crate::payout::PayoutInputs {
            prev_equity_eur: dec(equity),
            prev_net_pnl_eur: dec(pnl.net_eur),
            prev_quota_eur: dec(prev_q.as_ref().map_or(0.0, |q| q.quota_eur)),
            prev_paid_eur: dec(prev_q.as_ref().map_or(0.0, |q| q.paid_eur)),
        };
        let quota = rule.quota(&inputs);
        let (q,): (rust_decimal::Decimal,) = sqlx::query_as(
            "INSERT INTO ledger.payout_quota (day, quota_eur, reserved_eur, paid_eur, rule) VALUES ($1, $2, 0, 0, $3) \
             ON CONFLICT (day) DO UPDATE SET \
               rule = CASE WHEN excluded.quota_eur > ledger.payout_quota.quota_eur THEN excluded.rule \
                           ELSE coalesce(ledger.payout_quota.rule, excluded.rule) END, \
               quota_eur = greatest(ledger.payout_quota.quota_eur, excluded.quota_eur) \
             RETURNING quota_eur",
        )
        .bind(day)
        .bind(quota)
        .bind(json!({"rule": rule, "inputs": inputs, "book": book, "quota_eur": quota}))
        .fetch_one(self.pool())
        .await?;
        Ok(crate::money::f(q))
    }

    /// Books a payout of `amount_eur` (> 0) for `day` as `cash_movement(kind = 'payout', ref = day)`
    /// and sets `ledger.payout_quota.paid_eur` of that day to the sum of all books' payouts with
    /// that reference (row created with quota 0 when missing). One payout per book and day:
    /// a repeat is a no-op returning `false`. Audited. The quota rule itself is spec 42 Q1.
    pub async fn record_payout(&self, book: &str, day: NaiveDate, amount_eur: f64) -> Result<bool, PgError> {
        if !(amount_eur.is_finite() && amount_eur > 0.0) {
            return Err(PgError::Invalid(format!("payout amount must be positive, got {amount_eur}")));
        }
        let start = day.and_time(NaiveTime::MIN).and_utc();
        let end = start + Duration::days(1) - Duration::microseconds(1);
        let occurred_at = Utc::now().clamp(start, end);
        let reference = day.to_string();
        let id = uuid_of(&format!("cash:{book}:payout:{reference}"));
        let mut tx = self.pool().begin().await?;
        let n = sqlx::query(
            "INSERT INTO ledger.cash_movement (id, book, venue_id, kind, amount, ccy, fx_rate_eur, ref, occurred_at) \
             VALUES ($1, $2, $3, 'payout', $4, 'EUR', 1, $5, $6) ON CONFLICT DO NOTHING",
        )
        .bind(id)
        .bind(book)
        .bind(PAPER_VENUE)
        .bind(-dec(amount_eur))
        .bind(&reference)
        .bind(occurred_at)
        .execute(&mut *tx)
        .await?
        .rows_affected();
        if n > 0 {
            sqlx::query(
                "INSERT INTO ledger.payout_quota (day, quota_eur, reserved_eur, paid_eur) \
                 SELECT $1, 0, 0, coalesce(-sum(amount * fx_rate_eur), 0) FROM ledger.cash_movement \
                 WHERE kind = 'payout' AND ref = $2 \
                 ON CONFLICT (day) DO UPDATE SET paid_eur = excluded.paid_eur",
            )
            .bind(day)
            .bind(&reference)
            .execute(&mut *tx)
            .await?;
            audit(&mut *tx, LEDGER_ACTOR, "ledger.payout", book, json!({"day": reference, "amount_eur": amount_eur}))
                .await?;
        }
        tx.commit().await?;
        Ok(n > 0)
    }
}

/// Live tests run only with `DATABASE_URL` set (Postgres with all migrations applied, including
/// `20261111000030`). They use a throwaway `test-<uuid>` book and remove it afterwards.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff_books;

    async fn ledger() -> Option<PgLedger> {
        let url = std::env::var("DATABASE_URL").ok()?;
        Some(PgLedger::connect(&url).await.expect("connect"))
    }

    fn intent(book: &str, id: &str, inst: InstrumentId, side: &str, qty: f64, at: DateTime<Utc>) -> IntentRecord {
        IntentRecord {
            intent_id: id.into(),
            book: book.into(),
            instrument: inst,
            side: side.into(),
            qty,
            exec_algo: "market".into(),
            limit_price: None,
            strategy_ids: vec![],
            created_at: at,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fe(
        intent: &str,
        inst: InstrumentId,
        q: f64,
        px: f64,
        fx: f64,
        ccy: &str,
        at: DateTime<Utc>,
        s: &[&str],
    ) -> FillEvent {
        FillEvent {
            fill: Fill {
                fill_id: format!("{intent}:f"),
                intent_id: intent.into(),
                instrument: inst,
                quantity: q,
                price: px,
                fx_to_eur: fx,
                fee_eur: 0.5,
            },
            filled_at: at,
            strategy_ids: s.iter().map(|x| x.to_string()).collect(),
            details: FillDetails {
                ccy: ccy.into(),
                fx_source: if ccy == "EUR" { "eur".into() } else { "ecb_daily".into() },
                arrival_price: None,
                slippage_bps: None,
            },
            book: None,
            venue: None,
        }
    }

    fn sorted(mut v: Vec<Lot>) -> Vec<Lot> {
        v.sort_by_key(|l| (l.venue, l.instrument, l.opened_at, l.lot_id));
        v
    }

    #[tokio::test]
    async fn derived_accounting_round_trip() {
        let Some(pg) = ledger().await else { return };
        let book = format!("test-{}", Uuid::now_v7());
        let (a, b) = (InstrumentId::new(), InstrumentId::new());
        // whole seconds: Postgres keeps microseconds
        let t0 = DateTime::from_timestamp(Utc::now().timestamp() - 7200, 0).unwrap();
        let s = |n: i64| t0 + Duration::seconds(n);
        let id = |n: u32| format!("{book}-i{n}");
        let (s1, s2) = (format!("{book}-s1"), format!("{book}-s2"));

        pg.record_cash_movement(&CashMovement {
            book: book.clone(),
            kind: "deposit".into(),
            amount: 10_000.0,
            ccy: "EUR".into(),
            fx_rate_eur: 1.0,
            reference: "initial".into(),
            occurred_at: s(-1),
        })
        .await
        .unwrap();
        for (n, inst, side, q, at) in
            [(1, a, "buy", 10.0, 0), (2, b, "buy", 5.0, 10), (3, a, "sell", 4.0, 20), (4, b, "sell", 5.0, 30)]
        {
            let ev = LedgerEvent::Intent(intent(&book, &id(n), inst, side, q, s(at)));
            assert_eq!(pg.record(&ev).await.unwrap(), Recorded::New);
            assert_eq!(pg.record(&ev).await.unwrap(), Recorded::Duplicate);
        }
        // fill 4 arrives before fill 3 (late / out of order)
        let fills = [
            fe(&id(1), a, 10.0, 100.0, 1.0, "EUR", s(0), &[&s1, &s2]),
            fe(&id(2), b, 5.0, 50.0, 0.9, "USD", s(10), &[&s1]),
            fe(&id(4), b, -5.0, 52.0, 0.91, "USD", s(30), &[]),
            fe(&id(3), a, -4.0, 105.0, 1.0, "EUR", s(20), &[&s2]),
        ];
        for f in &fills {
            assert_eq!(pg.record_fill_event(f).await.unwrap(), Recorded::New);
        }
        assert_eq!(pg.record_fill_event(&fills[0]).await.unwrap(), Recorded::Duplicate);
        pg.record_cash_movement(&CashMovement {
            book: book.clone(),
            kind: "funding".into(),
            amount: -1.0,
            ccy: "EUR".into(),
            fx_rate_eur: 1.0,
            reference: format!("{book}-fund"),
            occurred_at: s(35),
        })
        .await
        .unwrap();

        // orphans: one booked once its order is known, one dismissed
        let mut orphan = fe(&id(5), a, 2.0, 99.0, 1.0, "EUR", s(40), &[&s1]);
        orphan.book = Some(book.clone());
        let Recorded::Orphaned { id: oid, new: true } = pg.record_fill_event(&orphan).await.unwrap() else {
            panic!("expected a new orphan")
        };
        assert_eq!(pg.record_fill_event(&orphan).await.unwrap(), Recorded::Orphaned { id: oid, new: false });
        let mut stray = fe(&format!("{book}-nosuch"), a, 1.0, 1.0, 1.0, "EUR", s(41), &[]);
        stray.book = Some(book.clone());
        let Recorded::Orphaned { id: sid, .. } = pg.record_fill_event(&stray).await.unwrap() else { panic!() };
        assert!(pg.open_orphan_count(Some(PAPER_VENUE)).await.unwrap() >= 2);
        assert!(pg.orphan_fills(Some("open"), 1000).await.unwrap().iter().any(|o| o.id == oid && o.qty == 2.0));
        assert!(matches!(pg.resolve_orphan(oid, &OrphanAction::Book, "ui").await, Err(PgError::UnknownIntent { .. })));
        pg.record_intent(&intent(&book, &id(5), a, "buy", 2.0, s(40))).await.unwrap();
        let booked = pg.resolve_orphan(oid, &OrphanAction::Book, "ui").await.unwrap();
        assert_eq!((booked.status.as_str(), booked.resolved_by.as_deref()), ("booked", Some("ui")));
        assert!(matches!(pg.resolve_orphan(oid, &OrphanAction::Book, "ui").await, Err(PgError::Invalid(_))));
        let d = pg.resolve_orphan(sid, &OrphanAction::Dismiss { reason: "test".into() }, "ui").await.unwrap();
        assert_eq!(d.status, "dismissed");

        // payout booking
        let day = chrono::NaiveDate::from_ymd_opt(2099, 2, 3).unwrap();
        assert!(pg.record_payout(&book, day, 25.0).await.unwrap());
        assert!(!pg.record_payout(&book, day, 25.0).await.unwrap(), "one payout per book and day");
        assert!(pg.record_payout(&book, day, -1.0).await.is_err());
        let (paid,): (f64,) =
            sqlx::query_as("SELECT -sum(amount)::float8 FROM ledger.cash_movement WHERE kind = 'payout' AND ref = $1")
                .bind(day.to_string())
                .fetch_one(pg.pool())
                .await
                .unwrap();
        assert!(paid >= 25.0);
        assert!((pg.payout_quota(day).await.unwrap().unwrap().paid_eur - paid).abs() < 1e-9);

        // replaying twice yields the same state; it equals the in-memory ledger
        let st = pg.book_state(&book, None).await.unwrap();
        let again = pg.book_state(&book, None).await.unwrap();
        assert_eq!(st.positions(), again.positions());
        assert_eq!(st.closings, again.closings);
        assert_eq!(st.fills, 5);
        assert!((st.payouts_eur - 25.0).abs() < 1e-9);
        let mem = pg.load_book(&book).await.unwrap();
        assert!((mem.cash_eur - st.cash_eur).abs() < 1e-9, "{} vs {}", mem.cash_eur, st.cash_eur);
        assert_eq!(
            crate::pg::quantities(&mem).into_iter().filter(|(_, q)| *q != 0.0).collect::<BTreeMap<_, _>>(),
            st.quantities().into_iter().filter(|(_, q)| *q != 0.0).collect::<BTreeMap<_, _>>()
        );

        // as-of positions
        let early = pg.positions(&book, Some(s(15))).await.unwrap();
        assert_eq!(early.iter().map(|p| p.qty).collect::<Vec<_>>().iter().sum::<f64>(), 15.0);
        let now = pg.positions(&book, None).await.unwrap();
        let qa = now.iter().find(|p| p.instrument == a).unwrap();
        assert_eq!((qa.qty, qa.open_lots), (8.0, 2));
        assert_eq!(now.iter().find(|p| p.instrument == b).unwrap().qty, 0.0);
        let marks = Marks { prices: BTreeMap::from([(a, 101.0)]), ..Marks::default() };
        let eq = pg.equity_eur(&book, None, Some(&marks)).await.unwrap();
        assert!((eq - (st.cash_eur + 8.0 * 101.0)).abs() < 1e-9);

        // P&L breakdown, FX separated; strategies sum to the book
        let pnl = pg.book_pnl(&book, None, None, None).await.unwrap();
        assert!((pnl.fx_realised_eur - 5.0 * 50.0 * 0.01).abs() < 1e-9, "{pnl:?}");
        assert!((pnl.realised_eur - (4.0 * 5.0 + 5.0 * 2.0 * 0.91)).abs() < 1e-9, "{pnl:?}");
        assert!((pnl.fees_eur - 2.5).abs() < 1e-9 && (pnl.funding_eur + 1.0).abs() < 1e-9);
        for from in [None, Some(s(5)), Some(s(25))] {
            let total = pg.book_pnl(&book, from, None, None).await.unwrap();
            let per = pg.strategy_pnls(&book, from, None).await.unwrap();
            let sum = per.values().fold(PnlBreakdown::default(), |acc, p| acc.plus(p));
            assert!((sum.net_eur - total.net_eur).abs() < 1e-9, "{from:?}: {sum:?} vs {total:?}");
            let one = pg.strategy_pnl(&book, &s1, from, None).await.unwrap();
            assert!(one.max_abs_diff(&per[&uuid_of(&s1).to_string()]) < 1e-12);
        }

        // statement over a window explains the equity change
        let stmt = pg.statement(&book, Some(s(5)), Some(s(3600)), None).await.unwrap();
        assert_eq!(stmt.fills.len(), 4);
        assert_eq!(stmt.cash_movements.len(), 1);
        assert!(stmt.unexplained_eur.abs() < 1e-9, "{}", stmt.unexplained_eur);
        assert!(stmt.to_csv().lines().count() > 10);
        pg.daily_statement(&book, t0.date_naive()).await.unwrap();

        // lot cache: deterministic and equal to the derived lots
        let r1 = pg.rebuild_lots(&book).await.unwrap();
        let c1 = pg.cached_lots(&book).await.unwrap();
        let r2 = pg.rebuild_lots(&book).await.unwrap();
        assert_eq!(r1, r2);
        assert_eq!(c1, pg.cached_lots(&book).await.unwrap());
        assert_eq!((r1.open_lots, r1.closed_lots, r1.closings), (2, 1, 2), "{r1:?}");
        let derived = sorted(st.open_lots().into_iter().chain(st.closed_lots.iter().cloned()).collect());
        assert_eq!(c1.len(), derived.len());
        for (x, y) in c1.iter().zip(&derived) {
            assert_eq!((x.lot_id, x.instrument, x.ccy.as_str()), (y.lot_id, y.instrument, y.ccy.as_str()));
            assert!((x.qty - y.qty).abs() < 1e-12 && (x.price - y.price).abs() < 1e-12);
        }
        let closings = pg.cached_lot_closings(&book, None, None).await.unwrap();
        assert_eq!(closings.len(), 2);
        assert!((closings.iter().map(|c| c.fx_eur).sum::<f64>() - pnl.fx_realised_eur).abs() < 1e-9);

        // every fill valued natively and in EUR with an FX source and time
        let usd_rate: (i64,) = sqlx::query_as(
            "SELECT count(*) FROM ref.fx_rate WHERE ccy = 'USD' AND source_id = 'ecb_daily' AND ts <= $1",
        )
        .bind(s(10))
        .fetch_one(pg.pool())
        .await
        .unwrap();
        assert_eq!(pg.fill_valuation_gaps(&book).await.unwrap(), if usd_rate.0 > 0 { 0 } else { 2 });

        // derived positions equal the latest snapshot; an injected fill is detected
        let as_of = s(3600);
        pg.snapshot_positions(&book, &mem, as_of).await.unwrap();
        let chk = pg.snapshot_check(&book, 1e-9).await.unwrap();
        assert_eq!((chk.as_of, chk.diffs.len()), (Some(as_of), 0), "{chk:?}");
        pg.record_intent(&intent(&book, &id(9), a, "buy", 1.0, s(100))).await.unwrap();
        pg.record_fill_event(&fe(&id(9), a, 1.0, 100.0, 1.0, "EUR", s(100), &[])).await.unwrap();
        let chk = pg.snapshot_check(&book, 1e-9).await.unwrap();
        assert_eq!(chk.diffs, vec![ReconDiff { instrument: a, ours: 9.0, broker: 8.0 }]);
        let theirs = pg.load_book(&book).await.unwrap();
        assert!(!diff_books(&mem, &theirs, 0.01, 1e-9).is_empty(), "reconciliation sees the injected fill");

        // cleanup
        pg.delete_book(&book).await.unwrap();
        assert!(pg.orphan_fills(None, 10_000).await.unwrap().iter().all(|o| o.id != oid && o.id != sid));
        assert!(pg.cached_lots(&book).await.unwrap().is_empty());
        sqlx::query(
            "DELETE FROM ledger.payout_quota WHERE day = $1 AND NOT EXISTS (SELECT 1 FROM ledger.cash_movement \
                     WHERE kind = 'payout' AND ref = $2)",
        )
        .bind(day)
        .bind(day.to_string())
        .execute(pg.pool())
        .await
        .unwrap();
        sqlx::query(
            "DELETE FROM audit.log WHERE actor IN ('ui', $2) AND (subject LIKE $1 || '%' OR payload->>'book' = $1)",
        )
        .bind(&book)
        .bind(LEDGER_ACTOR)
        .execute(pg.pool())
        .await
        .unwrap();
    }

    /// Read-only acceptance pass over an existing book (`LEDGER_VERIFY_BOOK=paper`): derived
    /// positions vs the latest snapshot, replay determinism, fill valuation, strategy P&L sums
    /// and the statement identity. Writes nothing.
    #[tokio::test]
    async fn verify_existing_book_read_only() {
        let (Some(pg), Ok(book)) = (ledger().await, std::env::var("LEDGER_VERIFY_BOOK")) else { return };
        let t = std::time::Instant::now();
        let now = Utc::now();
        let st = pg.book_state(&book, Some(now)).await.unwrap();
        let again = pg.book_state(&book, Some(now)).await.unwrap();
        assert_eq!(st.positions(), again.positions());
        assert_eq!(st.closings, again.closings);
        let mem = pg.load_book(&book).await.unwrap();
        for (i, q) in st.quantities() {
            let m = mem.positions.get(&i).map_or(0.0, |p| p.quantity);
            assert!((m - q).abs() < 1e-6, "{i:?}: in-memory {m} vs derived {q}");
        }
        let chk = pg.snapshot_check(&book, 1e-6).await.unwrap();
        let gaps = pg.fill_valuation_gaps(&book).await.unwrap();
        let from = Some(now - Duration::days(1));
        let total = pg.book_pnl(&book, from, Some(now), None).await.unwrap();
        let per = pg.strategy_pnls(&book, from, Some(now)).await.unwrap();
        let sum = per.values().fold(PnlBreakdown::default(), |acc, p| acc.plus(p));
        let stmt = pg.statement(&book, from, Some(now), None).await.unwrap();
        eprintln!(
            "book {book}: fills {} closings {} open lots {} | snapshot {:?} diffs {} | valuation gaps {gaps} | \
             24h net {:.4} (strategies {} sum {:.4}) | statement unexplained {:.6} | {:?}",
            st.fills,
            st.closings.len(),
            st.open_lots().len(),
            chk.as_of,
            chk.diffs.len(),
            total.net_eur,
            per.len(),
            sum.net_eur,
            stmt.unexplained_eur,
            t.elapsed()
        );
        assert!((sum.net_eur - total.net_eur).abs() < 1e-6 * (1.0 + total.net_eur.abs()));
        assert!(stmt.unexplained_eur.abs() < 1e-6 * (1.0 + stmt.closing.equity_eur.abs()));
        assert_eq!(gaps, 0);
    }
}
