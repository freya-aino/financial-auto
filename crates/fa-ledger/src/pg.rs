//! Postgres persistence for the ledger (spec 42; `migrations/20261106000005_ledger.sql`).
//!
//! Every write is idempotent through the table constraints (`ON CONFLICT DO NOTHING` on the
//! intent id / idempotency key, the fill id and `(client_order_id, venue_fill_id)`, the cash
//! movement `(venue_id, kind, ref)`), so replaying the same event stream yields the same book.
//! [`PgLedger::load_book`] rebuilds a [`Ledger`] from cash movements plus fills in time order.
//!
//! Money is `NUMERIC(38,18)` in Postgres and exact [`Decimal`] at the boundary: wire `f64`s
//! are converted once with [`money::dec`] and bound as `NUMERIC` (no `float8` cast), and the
//! book is rebuilt from the stored decimals, so [`PgLedger::load_book`] and the in-memory
//! [`Ledger`] agree digit for digit ([`PgLedger::reconcile_cut`]). Rows written before the
//! decimal change hold Postgres' `float8 → numeric` rendering and are read as stored. The
//! reporting reads ([`FillRow`], snapshots, the derived accounting) stay `f64`. Paper fills
//! have no venue, so the fixed [`PAPER_VENUE`] id stands in for `venue_id` (it must be non-null for the uniqueness
//! constraints to bite).
use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use fa_core::InstrumentId;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::{PgPool, Row, postgres::PgPoolOptions};
use uuid::Uuid;

use crate::{
    BookDiff, ExactFill, Fill, Ledger, Totals,
    accounting::LotMethod,
    cash_key,
    money::{self, Decimal, dec},
    settlement::{Settlement, SettlementPolicy},
};

/// Venue id used for paper books (no real venue exists yet).
pub const PAPER_VENUE: Uuid = Uuid::from_u128(0x7061_7065_7200_0000_0000_0000_0000_0001);
/// Namespace for deriving UUIDs from string ids (ULID intent ids, composite fill ids).
const NS: Uuid = Uuid::from_u128(0x6661_2d6c_6564_6765_7200_0000_0000_0001);
/// Cash movement kind/ref of the initial paper deposit.
pub const INITIAL_DEPOSIT_KIND: &str = "deposit";
pub const INITIAL_DEPOSIT_REF: &str = "initial";

/// Deterministic UUID for a string id: parsed when it already is one, UUIDv5 otherwise.
pub fn uuid_of(id: &str) -> Uuid {
    Uuid::parse_str(id).unwrap_or_else(|_| Uuid::new_v5(&NS, id.as_bytes()))
}

#[derive(Debug, thiserror::Error)]
pub enum PgError {
    #[error("sql: {0}")]
    Sql(#[from] sqlx::Error),
    #[error("fill {fill_id} references unknown intent {intent_id}")]
    UnknownIntent { fill_id: String, intent_id: String },
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid: {0}")]
    Invalid(String),
}

/// An order intent as persisted to `ledger.order_intent` (+ its paper `venue_order`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentRecord {
    pub intent_id: String,
    pub book: String,
    pub instrument: InstrumentId,
    /// `buy` | `sell`
    pub side: String,
    pub qty: f64,
    pub exec_algo: String,
    pub limit_price: Option<f64>,
    pub strategy_ids: Vec<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CashMovement {
    pub book: String,
    pub kind: String,
    pub amount: f64,
    pub ccy: String,
    pub fx_rate_eur: f64,
    /// Idempotency reference (unique per venue + kind).
    pub reference: String,
    pub occurred_at: DateTime<Utc>,
}

/// Accounting details of a fill beyond [`Fill`] (spec 42 §5 `fill.ccy/fx_source/arrival_price`,
/// D-051 `slippage_bps`); [`PgLedger::record_fill`] uses the defaults (EUR, source `fixed`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FillDetails {
    /// Currency of `Fill.price`.
    pub ccy: String,
    /// Where `Fill.fx_to_eur` came from (`eur`, `ecb_daily`, `fixed`, `missing`, …).
    pub fx_source: String,
    /// Last trade when the intent was accepted (execution quality, spec 40 §6.4).
    pub arrival_price: Option<f64>,
    /// Signed slippage vs the arrival price in basis points (positive = paid).
    pub slippage_bps: Option<f64>,
}

impl Default for FillDetails {
    fn default() -> Self {
        Self { ccy: fa_core::BASE_CURRENCY.into(), fx_source: "fixed".into(), arrival_price: None, slippage_bps: None }
    }
}

/// A fill row joined with its intent, as read back for replay / P&L.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillRow {
    pub fill_id: String,
    pub intent_id: String,
    pub book: String,
    pub instrument: InstrumentId,
    pub side: String,
    /// Unsigned (`side` gives the sign), exact as stored.
    #[serde(with = "money::json")]
    pub qty: Decimal,
    #[serde(with = "money::json")]
    pub price: Decimal,
    #[serde(with = "money::json")]
    pub fee_eur: Decimal,
    #[serde(with = "money::json")]
    pub fx_rate_eur: Decimal,
    pub filled_at: DateTime<Utc>,
    #[serde(default)]
    pub ccy: String,
    #[serde(default)]
    pub fx_source: String,
    #[serde(default, with = "money::json_opt")]
    pub arrival_price: Option<Decimal>,
    #[serde(default, with = "money::json_opt")]
    pub slippage_bps: Option<Decimal>,
}

impl FillRow {
    pub fn to_fill(&self) -> Fill {
        Fill {
            fill_id: self.fill_id.clone(),
            intent_id: self.intent_id.clone(),
            instrument: self.instrument,
            quantity: money::f(if self.side == "sell" { -self.qty } else { self.qty }),
            price: money::f(self.price),
            fx_to_eur: money::f(self.fx_rate_eur),
            fee_eur: money::f(self.fee_eur),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotRow {
    pub book: String,
    pub instrument: InstrumentId,
    #[serde(with = "money::json")]
    pub qty: Decimal,
    #[serde(with = "money::json_opt")]
    pub avg_price: Option<Decimal>,
    pub as_of: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconciliationRun {
    pub id: Uuid,
    pub venue_id: Uuid,
    pub started_at: DateTime<Utc>,
    pub status: String,
    pub diffs: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoutQuota {
    pub day: chrono::NaiveDate,
    pub quota_eur: f64,
    pub reserved_eur: f64,
    pub paid_eur: f64,
}

/// A cash movement with its exact EUR amount ([`PgLedger::cash_rows`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CashRow {
    /// [`cash_key`] of the row.
    pub key: String,
    pub kind: String,
    pub amount_eur: Decimal,
    pub occurred_at: DateTime<Utc>,
}

/// Result of [`PgLedger::reconcile_cut`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Reconciliation {
    /// Differences at the common cut; empty = the books agree.
    pub diff: BookDiff,
    /// Fills compared (booked on both sides).
    pub compared_fills: usize,
    /// In-memory fills excluded because their write is still in flight.
    pub in_flight: usize,
    /// Fills in Postgres that the in-memory book has not booked (not a diff: apply them).
    pub fills_only_theirs: Vec<(ExactFill, DateTime<Utc>)>,
    /// Cash movements in Postgres that the in-memory book has not applied (apply them).
    pub cash_only_theirs: Vec<CashRow>,
}

#[derive(Clone)]
pub struct PgLedger {
    pool: PgPool,
    /// `ledger.lot_method` (spec 42 §7) used by the derived reads in `pg_accounting`.
    lot_method: LotMethod,
    /// Settlement per venue (spec 42 §10 Q5) for the derived reads; default EUR everywhere.
    settlement: SettlementPolicy,
}

impl PgLedger {
    pub async fn connect(url: &str) -> Result<Self, PgError> {
        let pool = PgPoolOptions::new()
            .max_connections(4)
            .acquire_timeout(std::time::Duration::from_secs(3))
            .connect(url)
            .await?;
        Ok(Self::from_pool(pool))
    }

    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool, lot_method: LotMethod::default(), settlement: SettlementPolicy::default() }
    }

    /// Same ledger with another lot method for lots, positions and P&L (default FIFO).
    pub fn with_lot_method(mut self, method: LotMethod) -> Self {
        self.lot_method = method;
        self
    }

    pub fn lot_method(&self) -> LotMethod {
        self.lot_method
    }

    /// Same ledger with another settlement policy for the derived reads.
    pub fn with_settlement(mut self, policy: SettlementPolicy) -> Self {
        self.settlement = policy;
        self
    }

    pub fn settlement(&self) -> &SettlementPolicy {
        &self.settlement
    }

    /// Reads `ledger.venue_settlement` (migration `20261112000010`) into a policy (default EUR;
    /// an unreadable or missing table yields the default).
    pub async fn load_settlement(&self) -> SettlementPolicy {
        let rows: Vec<(Uuid, String)> = sqlx::query_as("SELECT venue_id, settlement FROM ledger.venue_settlement")
            .fetch_all(&self.pool)
            .await
            .unwrap_or_default();
        let mut p = SettlementPolicy::default();
        for (v, s) in rows {
            if let Some(s) = Settlement::parse(&s) {
                p.venues.insert(v, s);
            }
        }
        p
    }

    /// [`Self::with_settlement`] with the stored policy.
    pub async fn with_stored_settlement(self) -> Self {
        let p = self.load_settlement().await;
        self.with_settlement(p)
    }

    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Persists an intent and its paper venue order. Returns `false` when it already existed.
    pub async fn record_intent(&self, i: &IntentRecord) -> Result<bool, PgError> {
        let params = serde_json::json!({
            "strategy_ids": i.strategy_ids,
            "limit_price": i.limit_price,
        });
        let mut tx = self.pool.begin().await?;
        let inserted = sqlx::query(
            "INSERT INTO ledger.order_intent \
             (id, book, instrument_id, side, qty, exec_algo, params, idempotency_key, created_at) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) \
             ON CONFLICT DO NOTHING",
        )
        .bind(uuid_of(&i.intent_id))
        .bind(&i.book)
        .bind(i.instrument.0)
        .bind(&i.side)
        .bind(dec(i.qty))
        .bind(&i.exec_algo)
        .bind(&params)
        .bind(&i.intent_id)
        .bind(i.created_at)
        .execute(&mut *tx)
        .await?
        .rows_affected();
        sqlx::query(
            "INSERT INTO ledger.venue_order \
             (client_order_id, intent_id, venue_id, venue_order_id, status, qty, limit_price, created_at, updated_at) \
             VALUES ($1, $2, $3, $4, 'approved', $5, $6, $7, $7) \
             ON CONFLICT DO NOTHING",
        )
        .bind(&i.intent_id)
        .bind(uuid_of(&i.intent_id))
        .bind(PAPER_VENUE)
        .bind(format!("paper-{}", i.intent_id))
        .bind(dec(i.qty))
        .bind(i.limit_price.filter(|v| v.is_finite()).map(dec))
        .bind(i.created_at)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(inserted > 0)
    }

    /// Persists a fill (plus its strategy attribution, split evenly). Returns `false` on replay.
    /// The intent must have been recorded first ([`PgError::UnknownIntent`] otherwise).
    /// EUR-priced with FX source `fixed`; see [`Self::record_fill_with`] for the full details.
    pub async fn record_fill(
        &self,
        f: &Fill,
        filled_at: DateTime<Utc>,
        strategy_ids: &[String],
    ) -> Result<bool, PgError> {
        self.record_fill_with(f, filled_at, strategy_ids, &FillDetails::default()).await
    }

    /// [`Self::record_fill`] with the fill's currency, FX source, arrival price and slippage
    /// (`ledger.fill.arrival_price` / `slippage_bps`, migration `20261110000060`).
    pub async fn record_fill_with(
        &self,
        f: &Fill,
        filled_at: DateTime<Utc>,
        strategy_ids: &[String],
        d: &FillDetails,
    ) -> Result<bool, PgError> {
        self.record_fill_weighted(f, filled_at, strategy_ids, &[], d).await
    }

    /// [`Self::record_fill_with`] with attribution weights aligned with `strategy_ids` (spec 40
    /// §10 Q3, causal pro-rata): each strategy's quantity and fee share is its weight's fraction
    /// of the fill, the last one takes the remainder so the shares sum to the fill exactly;
    /// missing, mismatched or zero-sum weights split evenly (D-047).
    pub async fn record_fill_weighted(
        &self,
        f: &Fill,
        filled_at: DateTime<Utc>,
        strategy_ids: &[String],
        weights: &[f64],
        d: &FillDetails,
    ) -> Result<bool, PgError> {
        let mut tx = self.pool.begin().await?;
        let known: Option<(String,)> =
            sqlx::query_as("SELECT client_order_id FROM ledger.venue_order WHERE client_order_id = $1")
                .bind(&f.intent_id)
                .fetch_optional(&mut *tx)
                .await?;
        if known.is_none() {
            return Err(PgError::UnknownIntent { fill_id: f.fill_id.clone(), intent_id: f.intent_id.clone() });
        }
        let fill_uuid = uuid_of(&f.fill_id);
        let x = ExactFill::of(f);
        let inserted = sqlx::query(
            "INSERT INTO ledger.fill \
             (id, client_order_id, venue_fill_id, qty, price, ccy, fee, fee_ccy, fx_rate_eur, fx_source, filled_at, \
              arrival_price, slippage_bps) \
             VALUES ($1, $2, $3, $4, $5, $9, $6, 'EUR', $7, $10, $8, $11, $12) \
             ON CONFLICT DO NOTHING",
        )
        .bind(fill_uuid)
        .bind(&f.intent_id)
        .bind(&f.fill_id)
        .bind(x.qty.abs())
        .bind(x.price)
        .bind(x.fee_eur)
        .bind(x.fx_to_eur)
        .bind(filled_at)
        .bind(if d.ccy.is_empty() { fa_core::BASE_CURRENCY } else { d.ccy.as_str() })
        .bind(if d.fx_source.is_empty() { "fixed" } else { d.fx_source.as_str() })
        .bind(d.arrival_price.filter(|v| v.is_finite()).map(dec))
        .bind(d.slippage_bps.filter(|v| v.is_finite()).map(dec))
        .execute(&mut *tx)
        .await?
        .rows_affected();
        if inserted > 0 && !strategy_ids.is_empty() {
            for (s, (qty, fee)) in strategy_ids.iter().zip(attribution_split(x.qty.abs(), x.fee_eur, strategy_ids.len(), weights)) {
                sqlx::query(
                    "INSERT INTO ledger.fill_attribution (fill_id, strategy_id, qty, fee_share) \
                     VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING",
                )
                .bind(fill_uuid)
                .bind(uuid_of(s))
                .bind(qty)
                .bind(fee)
                .execute(&mut *tx)
                .await?;
            }
        }
        tx.commit().await?;
        Ok(inserted > 0)
    }

    /// Books a cash movement; `(venue, kind, reference)` is the idempotency key.
    pub async fn record_cash_movement(&self, m: &CashMovement) -> Result<bool, PgError> {
        let id = uuid_of(&format!("cash:{}:{}:{}", m.book, m.kind, m.reference));
        let n = sqlx::query(
            "INSERT INTO ledger.cash_movement \
             (id, book, venue_id, kind, amount, ccy, fx_rate_eur, ref, occurred_at) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) \
             ON CONFLICT DO NOTHING",
        )
        .bind(id)
        .bind(&m.book)
        .bind(PAPER_VENUE)
        .bind(&m.kind)
        .bind(dec(m.amount))
        .bind(&m.ccy)
        .bind(dec(m.fx_rate_eur))
        .bind(&m.reference)
        .bind(m.occurred_at)
        .execute(&self.pool)
        .await?
        .rows_affected();
        Ok(n > 0)
    }

    /// Records the opening deposit of a book once; later calls are no-ops (returns `false`).
    pub async fn ensure_initial_deposit(&self, book: &str, eur: f64) -> Result<bool, PgError> {
        self.record_cash_movement(&CashMovement {
            book: book.into(),
            kind: INITIAL_DEPOSIT_KIND.into(),
            amount: eur,
            ccy: "EUR".into(),
            fx_rate_eur: 1.0,
            reference: INITIAL_DEPOSIT_REF.into(),
            occurred_at: Utc::now(),
        })
        .await
    }

    /// Has the book any cash movement at all (i.e. was it ever opened)?
    pub async fn book_exists(&self, book: &str) -> Result<bool, PgError> {
        let n: (i64,) = sqlx::query_as("SELECT count(*) FROM ledger.cash_movement WHERE book = $1")
            .bind(book)
            .fetch_one(&self.pool)
            .await?;
        Ok(n.0 > 0)
    }

    /// Writes one `position_snapshot` row per position in the book (zero positions included so
    /// a closed position is visible as such). Returns rows written.
    pub async fn snapshot_positions(
        &self,
        book: &str,
        ledger: &Ledger,
        as_of: DateTime<Utc>,
    ) -> Result<usize, PgError> {
        let mut tx = self.pool.begin().await?;
        let mut n = 0;
        for (inst, p) in &ledger.exact().positions {
            n += sqlx::query(
                "INSERT INTO ledger.position_snapshot (book, venue_id, instrument_id, qty, avg_price, as_of) \
                 VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT DO NOTHING",
            )
            .bind(book)
            .bind(PAPER_VENUE)
            .bind(inst.0)
            .bind(p.qty)
            .bind(p.avg_cost_eur)
            .bind(as_of)
            .execute(&mut *tx)
            .await?
            .rows_affected() as usize;
        }
        tx.commit().await?;
        Ok(n)
    }

    /// Latest snapshot row per instrument of a book.
    pub async fn latest_snapshot(&self, book: &str) -> Result<Vec<SnapshotRow>, PgError> {
        let rows = sqlx::query(
            "SELECT DISTINCT ON (instrument_id) book, instrument_id, qty, avg_price, as_of \
             FROM ledger.position_snapshot WHERE book = $1 ORDER BY instrument_id, as_of DESC",
        )
        .bind(book)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| SnapshotRow {
                book: r.get("book"),
                instrument: InstrumentId(r.get::<Uuid, _>("instrument_id")),
                qty: r.get("qty"),
                avg_price: r.get("avg_price"),
                as_of: r.get("as_of"),
            })
            .collect())
    }

    /// Fills of a book in `[from, to)`, oldest first; optionally only those attributed to
    /// `strategy` (attribution is by strategy id string, hashed like intents).
    pub async fn fills(
        &self,
        book: &str,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
        strategy: Option<&str>,
    ) -> Result<Vec<FillRow>, PgError> {
        let rows = sqlx::query(
            "SELECT f.venue_fill_id, oi.idempotency_key AS intent_id, oi.book, oi.instrument_id, oi.side, \
             f.qty, f.price, f.fee, f.fx_rate_eur AS fx, f.filled_at, f.ccy, f.fx_source, f.arrival_price, \
             f.slippage_bps \
             FROM ledger.fill f \
             JOIN ledger.venue_order vo ON vo.client_order_id = f.client_order_id \
             JOIN ledger.order_intent oi ON oi.id = vo.intent_id \
             WHERE oi.book = $1 \
               AND ($2::timestamptz IS NULL OR f.filled_at >= $2) \
               AND ($3::timestamptz IS NULL OR f.filled_at < $3) \
               AND ($4::uuid IS NULL OR EXISTS (SELECT 1 FROM ledger.fill_attribution fa \
                                                WHERE fa.fill_id = f.id AND fa.strategy_id = $4)) \
             ORDER BY f.filled_at, f.id",
        )
        .bind(book)
        .bind(from)
        .bind(to)
        .bind(strategy.map(uuid_of))
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| FillRow {
                fill_id: r.get("venue_fill_id"),
                intent_id: r.get("intent_id"),
                book: r.get("book"),
                instrument: InstrumentId(r.get::<Uuid, _>("instrument_id")),
                side: r.get("side"),
                qty: r.get("qty"),
                price: r.get("price"),
                fee_eur: r.get("fee"),
                fx_rate_eur: r.get("fx"),
                filled_at: r.get("filled_at"),
                ccy: r.get("ccy"),
                fx_source: r.get("fx_source"),
                arrival_price: r.get("arrival_price"),
                slippage_bps: r.get("slippage_bps"),
            })
            .collect())
    }

    /// Realised slippage per strategy over attributed fills in `[from, to)`: mean and count of
    /// `slippage_bps` (fills without a recorded slippage are skipped).
    pub async fn slippage_by_strategy(
        &self,
        book: &str,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
    ) -> Result<BTreeMap<Uuid, (f64, i64)>, PgError> {
        let rows = sqlx::query(
            "SELECT fa.strategy_id, avg(f.slippage_bps)::float8 AS mean, count(*) AS n \
             FROM ledger.fill f \
             JOIN ledger.fill_attribution fa ON fa.fill_id = f.id \
             JOIN ledger.venue_order vo ON vo.client_order_id = f.client_order_id \
             JOIN ledger.order_intent oi ON oi.id = vo.intent_id \
             WHERE oi.book = $1 AND f.slippage_bps IS NOT NULL \
               AND ($2::timestamptz IS NULL OR f.filled_at >= $2) \
               AND ($3::timestamptz IS NULL OR f.filled_at < $3) \
             GROUP BY fa.strategy_id",
        )
        .bind(book)
        .bind(from)
        .bind(to)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| (r.get::<Uuid, _>("strategy_id"), (r.get::<f64, _>("mean"), r.get::<i64, _>("n"))))
            .collect())
    }

    /// Cash of a book in EUR: Σ amount × fx over its cash movements.
    pub async fn cash_eur(&self, book: &str) -> Result<f64, PgError> {
        let (cash,): (Option<f64>,) =
            sqlx::query_as("SELECT sum(amount * fx_rate_eur)::float8 FROM ledger.cash_movement WHERE book = $1")
                .bind(book)
                .fetch_one(&self.pool)
                .await?;
        Ok(cash.unwrap_or(0.0))
    }

    /// Every cash movement of a book with its exact EUR amount (`amount × fx_rate_eur` at the
    /// ledger scale), oldest first.
    pub async fn cash_rows(&self, book: &str) -> Result<Vec<CashRow>, PgError> {
        let rows = sqlx::query(
            "SELECT venue_id, kind, ref, amount, fx_rate_eur, occurred_at FROM ledger.cash_movement \
             WHERE book = $1 ORDER BY occurred_at, kind, ref",
        )
        .bind(book)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| {
                let venue: Uuid = r.get("venue_id");
                let kind: String = r.get("kind");
                let reference: String = r.get("ref");
                let amount: Decimal = r.get("amount");
                let fx: Decimal = r.get("fx_rate_eur");
                CashRow {
                    key: cash_key(&venue, &kind, &reference),
                    kind,
                    amount_eur: money::mul(money::round(amount), money::round(fx)),
                    occurred_at: r.get("occurred_at"),
                }
            })
            .collect())
    }

    /// Every fill of a book in exact decimals, in `(filled_at, id)` order.
    pub async fn exact_fills(&self, book: &str) -> Result<Vec<(ExactFill, DateTime<Utc>)>, PgError> {
        let rows = sqlx::query(
            "SELECT f.venue_fill_id, oi.instrument_id, oi.side, f.qty, f.price, f.fee, f.fx_rate_eur, f.filled_at \
             FROM ledger.fill f \
             JOIN ledger.venue_order vo ON vo.client_order_id = f.client_order_id \
             JOIN ledger.order_intent oi ON oi.id = vo.intent_id \
             WHERE oi.book = $1 ORDER BY f.filled_at, f.id",
        )
        .bind(book)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| {
                let qty = money::round(r.get::<Decimal, _>("qty"));
                let side: String = r.get("side");
                (
                    ExactFill {
                        fill_id: r.get("venue_fill_id"),
                        instrument: InstrumentId(r.get::<Uuid, _>("instrument_id")),
                        qty: if side == "sell" { -qty } else { qty },
                        price: money::round(r.get("price")),
                        fx_to_eur: money::round(r.get("fx_rate_eur")),
                        fee_eur: money::round(r.get("fee")),
                    },
                    r.get("filled_at"),
                )
            })
            .collect())
    }

    /// Rebuilds the book exactly: every cash movement applied once, intents registered, fills
    /// replayed in `filled_at` order. Payouts (`kind = 'payout'`) are counted in `paid_out_eur`.
    pub async fn load_book(&self, book: &str) -> Result<Ledger, PgError> {
        let cash = self.cash_rows(book).await?;
        let intents: Vec<(String,)> = sqlx::query_as("SELECT idempotency_key FROM ledger.order_intent WHERE book = $1")
            .bind(book)
            .fetch_all(&self.pool)
            .await?;
        let mut ledger = Ledger::with_cash_exact(Decimal::ZERO);
        for c in &cash {
            ledger.apply_cash(&c.key, &c.kind, c.amount_eur);
        }
        for (id,) in intents {
            let _ = ledger.register_intent(&id);
        }
        for (f, _) in self.exact_fills(book).await? {
            ledger.apply_exact(&f);
        }
        Ok(ledger)
    }

    /// Reconciles an in-memory book against Postgres at the same cut (spec 42 §6).
    ///
    /// The comparison is on additive totals (cash, fees, quantity per instrument), which do not
    /// depend on fill order, in exact decimals:
    /// - ours = `Totals::of(ours)` minus `in_flight` (fills booked in memory whose write to
    ///   Postgres has not been confirmed yet — they are not expected there);
    /// - theirs = the cash movements ours has applied plus the fills ours has booked (minus
    ///   `in_flight`), as stored in Postgres.
    ///
    /// So rows written to Postgres after the in-memory snapshot was taken cannot show as a diff
    /// (the 2026-11-10/11 false halts: the snapshot was taken, a fill arrived and was persisted,
    /// then Postgres was read). What remains is real: fills ours booked that Postgres lacks
    /// (`fills_only_ours`), stored values that differ, and — returned separately for the
    /// caller to apply — rows Postgres has that ours has not seen (fills booked by another
    /// writer, e.g. an operator-booked orphan, and cash movements such as payouts).
    pub async fn reconcile_cut(
        &self,
        book: &str,
        ours: &Ledger,
        in_flight: &[ExactFill],
        cash_tol: f64,
        qty_tol: f64,
    ) -> Result<Reconciliation, PgError> {
        let cash = self.cash_rows(book).await?;
        let fills = self.exact_fills(book).await?;
        let pending: std::collections::HashSet<&str> =
            in_flight.iter().filter(|f| ours.has_fill(&f.fill_id)).map(|f| f.fill_id.as_str()).collect();
        let mut mine = Totals::of(ours);
        for f in in_flight.iter().filter(|f| pending.contains(f.fill_id.as_str())) {
            mine.remove_fill(f);
        }
        let mut theirs = Totals::default();
        let mut cash_only_theirs = Vec::new();
        for c in cash {
            if ours.cash_refs().contains(&c.key) {
                theirs.add_cash(c.amount_eur);
            } else {
                cash_only_theirs.push(c);
            }
        }
        let mut seen = std::collections::HashSet::new();
        let mut fills_only_theirs = Vec::new();
        let mut compared = 0usize;
        for (f, at) in fills {
            seen.insert(f.fill_id.clone());
            if pending.contains(f.fill_id.as_str()) {
                continue;
            }
            if ours.has_fill(&f.fill_id) {
                theirs.add_fill(&f);
                compared += 1;
            } else {
                fills_only_theirs.push((f, at));
            }
        }
        // Postgres has no cash movement for a book that was opened in memory only (with_cash);
        // that is a cash diff like any other.
        let mut diff: BookDiff = crate::diff_totals(&mine, &theirs, cash_tol, qty_tol);
        let mut only_ours: Vec<String> = ours
            .fill_ids()
            .iter()
            .filter(|id| !seen.contains(*id) && !pending.contains(id.as_str()))
            .cloned()
            .collect();
        only_ours.sort();
        diff.fills_only_ours = only_ours;
        Ok(Reconciliation {
            diff,
            compared_fills: compared,
            in_flight: pending.len(),
            fills_only_theirs,
            cash_only_theirs,
        })
    }

    pub async fn record_reconciliation(
        &self,
        venue_id: Uuid,
        status: &str,
        diffs: Option<&Value>,
    ) -> Result<Uuid, PgError> {
        let id = Uuid::now_v7();
        sqlx::query(
            "INSERT INTO ledger.reconciliation_run (id, venue_id, started_at, status, diffs) \
             VALUES ($1, $2, now(), $3, $4)",
        )
        .bind(id)
        .bind(venue_id)
        .bind(status)
        .bind(diffs)
        .execute(&self.pool)
        .await?;
        Ok(id)
    }

    /// Most recent reconciliation runs, newest first.
    pub async fn reconciliation_runs(&self, limit: i64) -> Result<Vec<ReconciliationRun>, PgError> {
        let rows = sqlx::query(
            "SELECT id, venue_id, started_at, status, diffs FROM ledger.reconciliation_run \
             ORDER BY started_at DESC LIMIT $1",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| ReconciliationRun {
                id: r.get("id"),
                venue_id: r.get("venue_id"),
                started_at: r.get("started_at"),
                status: r.get("status"),
                diffs: r.get("diffs"),
            })
            .collect())
    }

    pub async fn payout_quota(&self, day: chrono::NaiveDate) -> Result<Option<PayoutQuota>, PgError> {
        let row = sqlx::query(
            "SELECT day, quota_eur::float8 AS q, reserved_eur::float8 AS r, paid_eur::float8 AS p \
             FROM ledger.payout_quota WHERE day = $1",
        )
        .bind(day)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|r| PayoutQuota {
            day: r.get("day"),
            quota_eur: r.get("q"),
            reserved_eur: r.get("r"),
            paid_eur: r.get("p"),
        }))
    }

    pub async fn upsert_payout_quota(&self, q: &PayoutQuota) -> Result<(), PgError> {
        sqlx::query(
            "INSERT INTO ledger.payout_quota (day, quota_eur, reserved_eur, paid_eur) \
             VALUES ($1, $2, $3, $4) \
             ON CONFLICT (day) DO UPDATE SET quota_eur = excluded.quota_eur, \
             reserved_eur = excluded.reserved_eur, paid_eur = excluded.paid_eur",
        )
        .bind(q.day)
        .bind(dec(q.quota_eur))
        .bind(dec(q.reserved_eur))
        .bind(dec(q.paid_eur))
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Deletes everything belonging to a book (tests / paper resets), including its orphan fills
    /// and cached lots (tables of migration `20261111000030`, which must be applied). Reconciliation
    /// runs and the payout quota are not per book and are left alone.
    pub async fn delete_book(&self, book: &str) -> Result<(), PgError> {
        let mut tx = self.pool.begin().await?;
        sqlx::query(
            "DELETE FROM ledger.fill_attribution WHERE fill_id IN (SELECT f.id FROM ledger.fill f \
             JOIN ledger.venue_order vo ON vo.client_order_id = f.client_order_id \
             JOIN ledger.order_intent oi ON oi.id = vo.intent_id WHERE oi.book = $1)",
        )
        .bind(book)
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "DELETE FROM ledger.fill WHERE client_order_id IN (SELECT vo.client_order_id FROM ledger.venue_order vo \
             JOIN ledger.order_intent oi ON oi.id = vo.intent_id WHERE oi.book = $1)",
        )
        .bind(book)
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "DELETE FROM ledger.venue_order WHERE intent_id IN (SELECT id FROM ledger.order_intent WHERE book = $1)",
        )
        .bind(book)
        .execute(&mut *tx)
        .await?;
        for sql in [
            "DELETE FROM ledger.order_intent WHERE book = $1",
            "DELETE FROM ledger.orphan_fill WHERE book = $1",
            "DELETE FROM ledger.lot WHERE book = $1",
            "DELETE FROM ledger.lot_closing WHERE book = $1",
            "DELETE FROM ledger.lot_rebuild WHERE book = $1",
            "DELETE FROM ledger.cash_movement WHERE book = $1",
            "DELETE FROM ledger.position_snapshot WHERE book = $1",
        ] {
            sqlx::query(sql).bind(book).execute(&mut *tx).await?;
        }
        tx.commit().await?;
        Ok(())
    }
}

/// Positions of a book as `instrument → quantity` (helper for reconciliation views).
pub fn quantities(ledger: &Ledger) -> BTreeMap<InstrumentId, f64> {
    ledger.positions.iter().map(|(i, p)| (*i, p.quantity)).collect()
}

/// Live tests run only with `DATABASE_URL` set (the compose Postgres with migrations applied).
/// Quantity and fee shares of `n` contributors: even (`qty / n`, D-047) without usable
/// weights, else by weight with the last contributor taking the remainder, so the weighted
/// shares sum to the fill exactly (spec 40 §10 Q3).
pub fn attribution_split(qty: Decimal, fee: Decimal, n: usize, weights: &[f64]) -> Vec<(Decimal, Decimal)> {
    let total: f64 = weights.iter().filter(|w| w.is_finite() && **w > 0.0).sum();
    if n == 0 {
        return Vec::new();
    }
    if weights.len() != n || !(total > 0.0 && total.is_finite()) {
        let d = Decimal::from(n);
        return vec![(money::div(qty, d), money::div(fee, d)); n];
    }
    let mut out = Vec::with_capacity(n);
    let (mut q_left, mut f_left) = (qty, fee);
    for (k, w) in weights.iter().enumerate() {
        if k + 1 == n {
            out.push((q_left, f_left));
            break;
        }
        let frac = money::dec(if w.is_finite() && *w > 0.0 { w / total } else { 0.0 });
        let (q, fe) = (money::mul(qty, frac), money::mul(fee, frac));
        q_left -= q;
        f_left -= fe;
        out.push((q, fe));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_attribution_sums_to_the_fill_exactly() {
        let (q, fee) = (money::dec(0.3), money::dec(1.0));
        let w = attribution_split(q, fee, 3, &[1.0, 1.0, 1.0]);
        assert_eq!(w.iter().map(|x| x.0).sum::<Decimal>(), q);
        assert_eq!(w.iter().map(|x| x.1).sum::<Decimal>(), fee);
        let w = attribution_split(q, fee, 2, &[3.0, 1.0]);
        assert_eq!(w[0].0, money::dec(0.225));
        assert_eq!(w[0].0 + w[1].0, q);
        // no usable weights: the even split of D-047
        assert_eq!(attribution_split(q, fee, 2, &[]), vec![(money::dec(0.15), money::dec(0.5)); 2]);
    }

    async fn ledger() -> Option<PgLedger> {
        let url = std::env::var("DATABASE_URL").ok()?;
        Some(PgLedger::connect(&url).await.expect("connect"))
    }

    fn intent(book: &str, id: &str, inst: InstrumentId, side: &str, qty: f64) -> IntentRecord {
        IntentRecord {
            intent_id: id.into(),
            book: book.into(),
            instrument: inst,
            side: side.into(),
            qty,
            exec_algo: "market".into(),
            limit_price: None,
            strategy_ids: vec!["strat-a".into(), "strat-b".into()],
            created_at: Utc::now(),
        }
    }

    fn fill(id: &str, intent: &str, inst: InstrumentId, q: f64, px: f64) -> Fill {
        Fill {
            fill_id: id.into(),
            intent_id: intent.into(),
            instrument: inst,
            quantity: q,
            price: px,
            fx_to_eur: 1.0,
            fee_eur: 1.0,
        }
    }

    #[tokio::test]
    async fn round_trip_is_idempotent_and_rebuilds_the_book() {
        let Some(pg) = ledger().await else { return };
        let book = format!("test-{}", Uuid::now_v7());
        let inst = InstrumentId::new();
        let i1 = format!("{book}-i1");
        let i2 = format!("{book}-i2");

        assert!(pg.ensure_initial_deposit(&book, 10_000.0).await.unwrap());
        assert!(!pg.ensure_initial_deposit(&book, 10_000.0).await.unwrap(), "deposit once");
        assert!(pg.book_exists(&book).await.unwrap());

        assert!(pg.record_intent(&intent(&book, &i1, inst, "buy", 10.0)).await.unwrap());
        assert!(!pg.record_intent(&intent(&book, &i1, inst, "buy", 10.0)).await.unwrap());
        pg.record_intent(&intent(&book, &i2, inst, "sell", 10.0)).await.unwrap();

        let t0 = Utc::now();
        let f1 = fill(&format!("{i1}:f"), &i1, inst, 10.0, 100.0);
        assert!(pg.record_fill(&f1, t0, &["strat-a".into(), "strat-b".into()]).await.unwrap());
        assert!(!pg.record_fill(&f1, t0, &[]).await.unwrap(), "replayed fill ignored");
        let mut f2 = fill(&format!("{i2}:f"), &i2, inst, -10.0, 110.0);
        f2.fx_to_eur = 1.0;
        let details = FillDetails {
            ccy: "USD".into(),
            fx_source: "ecb_daily".into(),
            arrival_price: Some(109.5),
            slippage_bps: Some(-4.2),
        };
        assert!(
            pg.record_fill_with(&f2, t0 + chrono::Duration::seconds(1), &["strat-a".into()], &details).await.unwrap()
        );
        let orphan = fill("orphan", "no-such-intent", inst, 1.0, 1.0);
        assert!(matches!(pg.record_fill(&orphan, t0, &[]).await, Err(PgError::UnknownIntent { .. })));

        // same arithmetic as the in-memory ledger
        let mut expect = Ledger::with_cash(10_000.0);
        expect.apply_fill(&f1);
        expect.apply_fill(&f2);
        let mut loaded = pg.load_book(&book).await.unwrap();
        assert!((loaded.cash_eur - expect.cash_eur).abs() < 1e-9, "{loaded:?}");
        assert_eq!(loaded.positions[&inst].quantity, 0.0);
        assert!((loaded.positions[&inst].realised_eur - 100.0).abs() < 1e-9);
        assert_eq!(loaded.fees_eur, 2.0);
        assert_eq!(loaded.register_intent(&i1), Err(crate::LedgerError::DuplicateIntent(i1.clone())));
        assert!(crate::diff_books(&loaded, &expect, 1e-9, 1e-9).is_empty());

        // strategy filter goes through fill_attribution
        let a = pg.fills(&book, None, None, Some("strat-a")).await.unwrap();
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].fill_id, f1.fill_id);
        assert_eq!((a[0].ccy.as_str(), a[0].fx_source.as_str(), a[0].slippage_bps), ("EUR", "fixed", None));
        assert_eq!((a[1].ccy.as_str(), a[1].fx_source.as_str()), ("USD", "ecb_daily"));
        assert_eq!((a[1].arrival_price, a[1].slippage_bps), (Some(dec(109.5)), Some(dec(-4.2))));
        let slip = pg.slippage_by_strategy(&book, None, None).await.unwrap();
        assert_eq!(slip[&uuid_of("strat-a")], (-4.2, 1));
        assert!(!slip.contains_key(&uuid_of("strat-b")), "no slippage recorded on strat-b's only fill");
        assert!(pg.fills(&book, None, None, Some("nobody")).await.unwrap().is_empty());
        assert_eq!(pg.fills(&book, Some(t0 + chrono::Duration::seconds(1)), None, None).await.unwrap().len(), 1);

        // snapshots: same instant twice is one row
        let as_of = Utc::now();
        assert_eq!(pg.snapshot_positions(&book, &loaded, as_of).await.unwrap(), 1);
        assert_eq!(pg.snapshot_positions(&book, &loaded, as_of).await.unwrap(), 0);
        let snap = pg.latest_snapshot(&book).await.unwrap();
        assert_eq!(snap.len(), 1);
        assert!(snap[0].qty.is_zero());

        // reconciliation runs
        let id = pg.record_reconciliation(PAPER_VENUE, "ok", Some(&serde_json::json!({"book": book}))).await.unwrap();
        let runs = pg.reconciliation_runs(5).await.unwrap();
        assert!(runs.iter().any(|r| r.id == id && r.status == "ok"));
        sqlx::query("DELETE FROM ledger.reconciliation_run WHERE id = $1").bind(id).execute(pg.pool()).await.unwrap();

        pg.delete_book(&book).await.unwrap();
        assert!(!pg.book_exists(&book).await.unwrap());
        assert!(pg.fills(&book, None, None, None).await.unwrap().is_empty());
    }

    /// Regression for the recurring `broker` halts of 2026-11-10/11: the in-memory book was
    /// cloned, a fill arrived and was persisted, then Postgres was read — a diff of exactly
    /// that fill. `reconcile_cut` compares at the memory's cut, excludes writes in flight, and
    /// still reports a fill that never reached Postgres; the decimals agree digit for digit.
    #[tokio::test]
    async fn reconcile_cut_has_no_snapshot_race_and_finds_real_gaps() {
        let Some(pg) = ledger().await else { return };
        let book = format!("test-{}", Uuid::now_v7());
        let inst = InstrumentId::new();
        pg.ensure_initial_deposit(&book, 100_000.0).await.unwrap();
        let mut mem = pg.load_book(&book).await.unwrap();
        let raw: [(f64, f64); 3] = [
            (2.85021203500556, 101.80316189408),
            (-1.15531427142454, 101.804900229592),
            (1.11897472440106, 100.77850057833),
        ];
        let mut fills = Vec::new();
        for (n, (q, px)) in raw.iter().enumerate() {
            let id = format!("{book}-i{n}");
            pg.record_intent(&intent(&book, &id, inst, if *q < 0.0 { "sell" } else { "buy" }, q.abs())).await.unwrap();
            let mut f = fill(&format!("{id}:f"), &id, inst, *q, *px);
            f.fee_eur = 0.0123 * q.abs();
            fills.push(f);
        }
        let t0 = Utc::now();
        for f in &fills[..2] {
            mem.apply_fill(f);
            pg.record_fill(f, t0, &[]).await.unwrap();
        }
        // the reconciliation snapshot is taken here; the third fill lands in both afterwards
        let snap = mem.clone();
        mem.apply_fill(&fills[2]);
        pg.record_fill(&fills[2], t0, &[]).await.unwrap();
        let old_way = crate::diff_books(&snap, &pg.load_book(&book).await.unwrap(), 0.01, 1e-6);
        assert!(!old_way.is_empty(), "the pre-fix comparison reports the race as a diff");
        let r = pg.reconcile_cut(&book, &snap, &[], 0.0, 0.0).await.unwrap();
        assert!(r.diff.is_empty(), "{:?}", r.diff);
        assert_eq!((r.compared_fills, r.fills_only_theirs.len()), (2, 1));
        assert_eq!(r.fills_only_theirs[0].0, ExactFill::of(&fills[2]));
        // exact agreement of the full book, decimals included
        let full = pg.load_book(&book).await.unwrap();
        assert_eq!(Totals::of(&full), Totals::of(&mem));
        assert!(pg.reconcile_cut(&book, &mem, &[], 0.0, 0.0).await.unwrap().diff.is_empty());

        // a fill booked in memory whose write is still in flight is not a diff ...
        let id = format!("{book}-i9");
        pg.record_intent(&intent(&book, &id, inst, "buy", 3.0)).await.unwrap();
        let late = fill(&format!("{id}:f"), &id, inst, 3.0, 99.5);
        mem.apply_fill(&late);
        let r = pg.reconcile_cut(&book, &mem, &[ExactFill::of(&late)], 0.0, 0.0).await.unwrap();
        assert!(r.diff.is_empty() && r.in_flight == 1, "{r:?}");
        // ... but a fill that never reached Postgres is
        let r = pg.reconcile_cut(&book, &mem, &[], 0.0, 0.0).await.unwrap();
        assert_eq!(r.diff.fills_only_ours, vec![late.fill_id.clone()]);
        assert_eq!(r.diff.positions.len(), 1);
        assert!(r.diff.cash_ours != r.diff.cash_theirs);

        // cash written by another writer (a payout from control-api) is returned to apply
        crate::pg::PgLedger::record_payout(&pg, &book, chrono::NaiveDate::from_ymd_opt(2099, 1, 2).unwrap(), 12.5)
            .await
            .unwrap();
        pg.record_fill(&late, t0, &[]).await.unwrap();
        let r = pg.reconcile_cut(&book, &mem, &[], 0.0, 0.0).await.unwrap();
        assert!(r.diff.is_empty(), "{:?}", r.diff);
        assert_eq!(r.cash_only_theirs.len(), 1);
        let c = &r.cash_only_theirs[0];
        assert!(mem.apply_cash(&c.key, &c.kind, c.amount_eur));
        assert_eq!(Totals::of(&pg.load_book(&book).await.unwrap()), Totals::of(&mem));
        sqlx::query("DELETE FROM ledger.payout_quota WHERE day = '2099-01-02'").execute(pg.pool()).await.unwrap();
        pg.delete_book(&book).await.unwrap();
    }

    #[tokio::test]
    async fn payout_quota_upserts() {
        let Some(pg) = ledger().await else { return };
        let day = chrono::NaiveDate::from_ymd_opt(2099, 1, 1).unwrap();
        let q = PayoutQuota { day, quota_eur: 10.0, reserved_eur: 1.0, paid_eur: 0.5 };
        pg.upsert_payout_quota(&q).await.unwrap();
        pg.upsert_payout_quota(&PayoutQuota { paid_eur: 2.0, ..q.clone() }).await.unwrap();
        let got = pg.payout_quota(day).await.unwrap().unwrap();
        assert_eq!((got.quota_eur, got.reserved_eur, got.paid_eur), (10.0, 1.0, 2.0));
        sqlx::query("DELETE FROM ledger.payout_quota WHERE day = $1").bind(day).execute(pg.pool()).await.unwrap();
    }

    #[test]
    fn uuid_of_is_stable_and_passes_real_uuids() {
        let u = Uuid::now_v7();
        assert_eq!(uuid_of(&u.to_string()), u);
        assert_eq!(uuid_of("01J0ABCDEF"), uuid_of("01J0ABCDEF"));
        assert_ne!(uuid_of("a"), uuid_of("b"));
    }
}
