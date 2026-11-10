//! Postgres persistence for the paper book (spec 42): intents, fills, cash movements, position
//! snapshots, reconciliation runs and the audit log. Everything here is best effort and
//! idempotent: a failed write is logged and the in-memory engine carries on (the next
//! reconciliation pass will surface the gap as a `diff`).
use chrono::{DateTime, TimeZone, Utc};
use fa_ledger::{
    BookDiff, Ledger,
    pg::{IntentRecord, PAPER_VENUE, PgLedger},
};
use fa_proto::v1::{OrderIntent, Side};
use serde_json::{Value, json};

use crate::engine::BookedFill;

pub const ACTOR: &str = "system:risk-engine";
/// Cash differences below this are not reported by the reconciliation pass.
const CASH_TOL_EUR: f64 = 0.01;
const QTY_TOL: f64 = 1e-9;

pub struct Persist {
    pg: PgLedger,
    book: String,
}

fn at(ms: i64) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(ms).single().unwrap_or_else(Utc::now)
}

impl Persist {
    /// Connects and opens the book: on the first run the starting cash is booked as the initial
    /// `cash_movement` (idempotent, so restarts never deposit twice). Returns the rebuilt book.
    pub async fn open(url: &str, book: &str, starting_cash_eur: f64) -> anyhow::Result<(Self, Ledger)> {
        let pg = PgLedger::connect(url).await?;
        let fresh = pg.ensure_initial_deposit(book, starting_cash_eur).await?;
        let ledger = pg.load_book(book).await?;
        let me = Self { pg, book: book.into() };
        if fresh {
            me.audit("book.opened", Some(book), json!({"starting_cash_eur": starting_cash_eur})).await;
        }
        Ok((me, ledger))
    }

    pub fn book(&self) -> &str {
        &self.book
    }

    /// Records an intent seen on the bus (ours or foreign); replays are no-ops.
    pub async fn intent(&self, i: &OrderIntent) {
        let Ok(uuid) = uuid::Uuid::parse_str(&i.instrument_id) else {
            return;
        };
        let rec = IntentRecord {
            intent_id: i.intent_id.clone(),
            book: i.account_id.clone(),
            instrument: fa_core::InstrumentId(uuid),
            side: if Side::try_from(i.side).unwrap_or_default() == Side::Sell { "sell".into() } else { "buy".into() },
            qty: i.quantity,
            exec_algo: i.exec_algo.clone(),
            limit_price: i.limit_price,
            strategy_ids: i.strategy_ids.clone(),
            created_at: i
                .created_at
                .as_ref()
                .map_or_else(Utc::now, |t| at(t.seconds * 1000 + i64::from(t.nanos) / 1_000_000)),
        };
        if let Err(e) = self.pg.record_intent(&rec).await {
            tracing::warn!(intent = %i.intent_id, error = %e, "intent not persisted");
        }
    }

    pub async fn fills(&self, fills: Vec<BookedFill>) {
        for b in fills {
            if let Err(e) = self.pg.record_fill(&b.fill, at(b.at_ms), &b.strategy_ids).await {
                tracing::warn!(fill = %b.fill.fill_id, error = %e, "fill not persisted");
            }
        }
    }

    pub async fn snapshot(&self, ledger: &Ledger, reason: &str) {
        match self.pg.snapshot_positions(&self.book, ledger, Utc::now()).await {
            Ok(n) => tracing::debug!(rows = n, reason, "position snapshot written"),
            Err(e) => tracing::warn!(error = %e, reason, "position snapshot failed"),
        }
    }

    pub async fn audit(&self, action: &str, subject: Option<&str>, payload: Value) {
        if let Err(e) = fa_warehouse::audit::append(self.pg.pool(), ACTOR, action, subject, payload).await {
            tracing::warn!(action, error = %e, "audit entry not written");
        }
    }

    /// Compares the in-memory book with the one derived from Postgres and records a
    /// `reconciliation_run` (`ok` | `diff`). Returns the diff when there is one. Spec 42 says a
    /// diff halts the venue; for the paper book we only log it (follow-up).
    pub async fn reconcile(&self, ours: &Ledger) -> Option<BookDiff> {
        let theirs = match self.pg.load_book(&self.book).await {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!(error = %e, "reconciliation skipped: cannot load book");
                return None;
            }
        };
        let diff = fa_ledger::diff_books(ours, &theirs, CASH_TOL_EUR, QTY_TOL);
        let (status, diffs) = if diff.is_empty() {
            ("ok", None)
        } else {
            (
                "diff",
                Some(json!({"book": self.book, "cash": {"memory": diff.cash_ours, "postgres": diff.cash_theirs},
                "positions": diff.positions.iter().map(|d| json!({"instrument_id": d.instrument, "memory": d.ours, "postgres": d.broker})).collect::<Vec<_>>()})),
            )
        };
        if let Err(e) = self.pg.record_reconciliation(PAPER_VENUE, status, diffs.as_ref()).await {
            tracing::warn!(error = %e, "reconciliation run not recorded");
        }
        if diff.is_empty() {
            tracing::debug!("reconciliation ok");
            None
        } else {
            tracing::error!(
                ?diff,
                "paper book differs from Postgres-derived book (not halting yet; spec 42 follow-up)"
            );
            Some(diff)
        }
    }
}
