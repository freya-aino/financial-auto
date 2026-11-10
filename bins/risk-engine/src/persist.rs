//! Postgres persistence for the paper book (spec 42): intents, fills (with currency, FX source,
//! arrival price and slippage), cash movements, position snapshots, reconciliation runs,
//! strategy lifecycle rows with their sub-books (`ledger.strategy_state`, D-051, D-061), the
//! payout quota and the audit log; plus the reference-data, FX and ECB reads/writes (D-061). Everything here is best effort and idempotent: a failed write is logged and
//! the in-memory engine carries on (the next reconciliation pass will surface the gap as a
//! `diff`, which halts scope `broker`).
use chrono::{DateTime, TimeZone, Utc};
use fa_ledger::{
    BookDiff, Ledger,
    pg::{IntentRecord, PAPER_VENUE, PgLedger},
};
use fa_proto::v1::{OrderIntent, Side};
use fa_risk::strategy::{Checklist, StrategyState, StrategyStatus};
use serde_json::{Value, json};
use sqlx::Row;

use crate::engine::{BookedFill, SubBookRecord};

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
            if let Err(e) = self.pg.record_fill_with(&b.fill, at(b.at_ms), &b.strategy_ids, &b.details).await {
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
        self.audit_as(ACTOR, action, subject, payload).await;
    }

    /// Audit entry on behalf of an operator (`ui`) or another service.
    pub async fn audit_as(&self, actor: &str, action: &str, subject: Option<&str>, payload: Value) {
        if let Err(e) = fa_warehouse::audit::append(self.pg.pool(), actor, action, subject, payload).await {
            tracing::warn!(action, error = %e, "audit entry not written");
        }
    }

    /// Upserts one lifecycle row (`ledger.strategy_state`); `gate` is the last checklist, `book`
    /// the strategy's sub-book (`book` JSONB, migration `20261110000060`; `None` keeps the stored one).
    pub async fn strategy_state(&self, s: &StrategyState, book: Option<&SubBookRecord>, gate: Option<&Checklist>) {
        let r = sqlx::query(
            "INSERT INTO ledger.strategy_state (strategy_id, status, prev_status, since, first_seen, shadow_since, \
             horizons_completed, sleeve_eur, pnl_eur, peak_pnl_eur, max_drawdown, gate, book, updated_at) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, now()) \
             ON CONFLICT (strategy_id) DO UPDATE SET status = EXCLUDED.status, prev_status = EXCLUDED.prev_status, \
             since = EXCLUDED.since, shadow_since = EXCLUDED.shadow_since, horizons_completed = EXCLUDED.horizons_completed, \
             sleeve_eur = EXCLUDED.sleeve_eur, pnl_eur = EXCLUDED.pnl_eur, peak_pnl_eur = EXCLUDED.peak_pnl_eur, \
             max_drawdown = EXCLUDED.max_drawdown, gate = COALESCE(EXCLUDED.gate, ledger.strategy_state.gate), \
             book = COALESCE(EXCLUDED.book, ledger.strategy_state.book), updated_at = now()",
        )
        .bind(&s.strategy_id)
        .bind(s.status.as_str())
        .bind(s.prev_status.map(StrategyStatus::as_str))
        .bind(at(s.since_ms))
        .bind(at(s.first_seen_ms))
        .bind(s.shadow_since_ms.map(at))
        .bind(s.horizons_completed as i64)
        .bind(s.sleeve_eur)
        .bind(s.pnl_eur)
        .bind(s.peak_pnl_eur)
        .bind(s.max_drawdown)
        .bind(gate.and_then(|g| serde_json::to_value(g).ok()))
        .bind(book.and_then(|b| serde_json::to_value(b).ok()))
        .execute(self.pg.pool())
        .await;
        if let Err(e) = r {
            tracing::warn!(strategy = %s.strategy_id, error = %e, "strategy state not persisted");
        }
    }

    /// Loads every persisted lifecycle row with its sub-book, if any (startup seed).
    pub async fn load_strategy_states(&self) -> Vec<(StrategyState, Option<SubBookRecord>)> {
        let rows = match sqlx::query(
            "SELECT strategy_id, status, prev_status, since, first_seen, shadow_since, horizons_completed, \
             sleeve_eur::float8 AS sleeve_eur, pnl_eur::float8 AS pnl_eur, peak_pnl_eur::float8 AS peak_pnl_eur, \
             max_drawdown::float8 AS max_drawdown, to_jsonb(s) -> 'book' AS book FROM ledger.strategy_state s",
        )
        .fetch_all(self.pg.pool())
        .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "strategy states not loaded");
                return Vec::new();
            }
        };
        rows.iter()
            .filter_map(|r| {
                let status = StrategyStatus::parse(&r.get::<String, _>("status"))?;
                let book = r
                    .get::<Option<Value>, _>("book")
                    .filter(|v| !v.is_null())
                    .and_then(|v| match serde_json::from_value::<SubBookRecord>(v) {
                        Ok(b) => Some(b),
                        Err(e) => {
                            tracing::warn!(strategy = %r.get::<String, _>("strategy_id"), error = %e, "sub-book not restored");
                            None
                        }
                    });
                let st = StrategyState {
                    strategy_id: r.get("strategy_id"),
                    status,
                    prev_status: r.get::<Option<String>, _>("prev_status").and_then(|s| StrategyStatus::parse(&s)),
                    since_ms: r.get::<DateTime<Utc>, _>("since").timestamp_millis(),
                    first_seen_ms: r.get::<DateTime<Utc>, _>("first_seen").timestamp_millis(),
                    shadow_since_ms: r.get::<Option<DateTime<Utc>>, _>("shadow_since").map(|t| t.timestamp_millis()),
                    horizons_completed: r.get::<i64, _>("horizons_completed").max(0) as u64,
                    sleeve_eur: r.get("sleeve_eur"),
                    pnl_eur: r.get("pnl_eur"),
                    peak_pnl_eur: r.get("peak_pnl_eur"),
                    max_drawdown: r.get("max_drawdown"),
                    last_reason: String::new(),
                };
                Some((st, book))
            })
            .collect()
    }

    /// Instrument reference table (`fa_warehouse::venues`).
    pub async fn load_refdata(&self) -> Option<fa_warehouse::venues::RefTable> {
        match fa_warehouse::venues::load_instruments(self.pg.pool()).await {
            Ok(v) => Some(fa_warehouse::venues::RefTable::from_refs(v)),
            Err(e) => {
                tracing::warn!(error = %e, "reference data not loaded");
                None
            }
        }
    }

    /// FX as-of table of `source` covering the last `days` days.
    pub async fn load_fx(&self, source: &str, days: u32) -> Option<fa_warehouse::fx::FxTable> {
        let since = Utc::now() - chrono::Duration::days(i64::from(days));
        match fa_warehouse::fx::load_recent(self.pg.pool(), source, since).await {
            Ok(r) => Some(fa_warehouse::fx::FxTable::from_rates(r)),
            Err(e) => {
                tracing::warn!(error = %e, "fx rates not loaded");
                None
            }
        }
    }

    /// Fetches the ECB daily reference rates and upserts them into `ref.fx_rate`
    /// (`source = ecb_daily`, idempotent). Returns `(date, currencies, new rows)`.
    pub async fn refresh_ecb(&self, http: &reqwest::Client) -> anyhow::Result<(chrono::NaiveDate, usize, usize)> {
        let resp = http.get(fa_warehouse::fx::ECB_DAILY_URL).send().await?.error_for_status()?;
        let body = resp.text().await?;
        let daily = fa_warehouse::fx::parse_ecb_daily(&body)?;
        let rates = daily.to_rates();
        let n = fa_warehouse::fx::upsert_rates(self.pg.pool(), &rates).await?;
        Ok((daily.date, rates.len(), n))
    }

    /// Today's reserved payout from `ledger.payout_quota` (spec 42 §6): the larger of
    /// `quota − paid` and `reserved`, 0 without a row.
    pub async fn payout_reserve_today(&self) -> f64 {
        match self.pg.payout_quota(Utc::now().date_naive()).await {
            Ok(Some(q)) => (q.quota_eur - q.paid_eur).max(q.reserved_eur).max(0.0),
            Ok(None) => 0.0,
            Err(e) => {
                tracing::debug!(error = %e, "payout quota not read");
                0.0
            }
        }
    }

    /// Raises today's quota to `quota_eur` (never lowers it: tighten-only, spec 40 §6.5).
    pub async fn raise_payout_quota(&self, quota_eur: f64) -> anyhow::Result<f64> {
        let day = Utc::now().date_naive();
        let cur = self.pg.payout_quota(day).await?;
        let q = fa_ledger::pg::PayoutQuota {
            day,
            quota_eur: cur.as_ref().map_or(quota_eur, |c| c.quota_eur.max(quota_eur)),
            reserved_eur: cur.as_ref().map_or(0.0, |c| c.reserved_eur),
            paid_eur: cur.as_ref().map_or(0.0, |c| c.paid_eur),
        };
        self.pg.upsert_payout_quota(&q).await?;
        Ok(q.quota_eur)
    }

    /// Latest deflated Sharpe for a strategy (promotion gate item 2). Two sources, in order:
    /// `validation.trial(strategy_id, deflated_sharpe)` when the validation workstream adds
    /// per-strategy rows, else `validation.generation_stats(best_candidate, deflated_sharpe)` —
    /// the DSR corrects the *selected* candidate of a generation, so it applies to a strategy
    /// only when it is that generation's best candidate (matched by strategy id or genome id).
    /// `None` when neither table/column exists or no row matches: the gate reports "unknown"
    /// and refuses to promote.
    pub async fn deflated_sharpe(&self, ids: &[&str]) -> Option<f64> {
        let cols = |table: &'static str| async move {
            sqlx::query_scalar::<_, String>(
                "SELECT column_name::text FROM information_schema.columns WHERE table_schema = 'validation' AND table_name = $1",
            )
            .bind(table)
            .fetch_all(self.pg.pool())
            .await
            .unwrap_or_default()
        };
        let has = |cols: &[String], c: &str| cols.iter().any(|x| x == c);
        let ids: Vec<String> = ids.iter().filter(|s| !s.is_empty()).map(|s| (*s).to_owned()).collect();
        if ids.is_empty() {
            return None;
        }
        let trial = cols("trial").await;
        if has(&trial, "strategy_id") && has(&trial, "deflated_sharpe") {
            let order = ["evaluated_at", "created_at", "at"].iter().find(|c| has(&trial, c));
            let sql = match order {
                Some(c) => format!(
                    "SELECT deflated_sharpe::float8 FROM validation.trial WHERE strategy_id = ANY($1) AND deflated_sharpe IS NOT NULL ORDER BY {c} DESC LIMIT 1"
                ),
                None => "SELECT deflated_sharpe::float8 FROM validation.trial WHERE strategy_id = ANY($1) AND deflated_sharpe IS NOT NULL LIMIT 1".into(),
            };
            // the only dynamic part is a column name from the whitelist above
            if let Ok(Some(v)) =
                sqlx::query_scalar::<_, f64>(sqlx::AssertSqlSafe(sql)).bind(&ids).fetch_optional(self.pg.pool()).await
            {
                return Some(v);
            }
        }
        let stats = cols("generation_stats").await;
        if has(&stats, "best_candidate") && has(&stats, "deflated_sharpe") {
            let order = if has(&stats, "computed_at") { "ORDER BY computed_at DESC " } else { "" };
            let sql = format!(
                "SELECT deflated_sharpe::float8 FROM validation.generation_stats WHERE best_candidate = ANY($1) AND deflated_sharpe IS NOT NULL {order}LIMIT 1"
            );
            return sqlx::query_scalar::<_, f64>(sqlx::AssertSqlSafe(sql))
                .bind(&ids)
                .fetch_optional(self.pg.pool())
                .await
                .ok()
                .flatten();
        }
        None
    }

    /// Compares the in-memory book with the one derived from Postgres and records a
    /// `reconciliation_run` (`ok` | `diff`). Returns the diff when there is one; the caller
    /// halts scope `broker` on it (spec 42 §6, D-051).
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
            tracing::error!(?diff, "paper book differs from Postgres-derived book: halting scope broker");
            Some(diff)
        }
    }
}

#[cfg(test)]
mod tests {
    use fa_risk::strategy::{GateEvidence, PromotionGate, StrategyState, StrategyStatus};

    use super::*;

    /// Needs `DATABASE_URL` (compose Postgres with migration 20261110000020 applied).
    #[tokio::test]
    async fn strategy_state_round_trips_and_gate_is_kept() {
        let Ok(url) = std::env::var("DATABASE_URL") else { return };
        let pg = PgLedger::connect(&url).await.unwrap();
        let p = Persist { pg, book: "paper".into() };
        let id = format!("test-{}", uuid::Uuid::now_v7());
        let mut s = StrategyState::new(&id, StrategyStatus::Shadow, 1_762_560_000_000);
        s.horizons_completed = 3;
        s.observe_pnl(-120.0, 10_000.0);
        let gate = PromotionGate::default().check(&s, &GateEvidence::default(), 1_762_560_000_000 + 86_400_000);
        p.strategy_state(&s, None, Some(&gate)).await;
        s.apply(fa_risk::strategy::StrategyAction::Pause, 1_762_560_100_000, None).unwrap();
        p.strategy_state(&s, None, None).await; // gate NULL keeps the stored checklist
        let rows = p.load_strategy_states().await;
        let (got, book) = rows.iter().find(|(r, _)| r.strategy_id == id).expect("row loaded");
        assert!(book.is_none(), "no book written yet");
        assert_eq!((got.status, got.prev_status), (StrategyStatus::Paused, Some(StrategyStatus::Shadow)));
        assert_eq!(got.horizons_completed, 3);
        assert_eq!(got.since_ms, 1_762_560_100_000);
        assert!((got.max_drawdown - 0.012).abs() < 1e-9);
        let (stored_gate,): (Option<Value>,) =
            sqlx::query_as("SELECT gate FROM ledger.strategy_state WHERE strategy_id = $1")
                .bind(&id)
                .fetch_one(p.pg.pool())
                .await
                .unwrap();
        assert_eq!(stored_gate.unwrap()["green"], false);
        // sub-book JSONB round trip (migration 20261110000060)
        let inst = fa_core::InstrumentId::new();
        let rec: SubBookRecord = serde_json::from_value(json!({
            "v": 1, "kind": "active", "cash_eur": -4900.5, "last_value_eur": 99.5, "fees_eur": 0.5,
            "positions": {inst.to_string(): {"quantity": 50.0, "avg_cost_eur": 100.0, "realised_eur": 1.25}},
            "slippage_sum_bps": 12.0, "slippage_n": 3, "history": [[1_762_560_000_000i64, 0.0], [1_762_563_600_000i64, -120.0]],
        }))
        .unwrap();
        p.strategy_state(&s, Some(&rec), None).await;
        p.strategy_state(&s, None, None).await; // NULL book keeps the stored one
        let rows = p.load_strategy_states().await;
        let (_, book) = rows.iter().find(|(r, _)| r.strategy_id == id).expect("row loaded");
        assert_eq!(book.as_ref(), Some(&rec), "sub-book restored verbatim");
        assert_eq!(p.deflated_sharpe(&[&id]).await, None, "no validation row → unknown");
        assert_eq!(p.deflated_sharpe(&[]).await, None);
        // generation_stats path (migration 20261110000030, validation workstream): the DSR of the
        // generation whose best candidate is this strategy (or its genome)
        let have_stats: bool = sqlx::query_scalar("SELECT to_regclass('validation.generation_stats') IS NOT NULL")
            .fetch_one(p.pg.pool())
            .await
            .unwrap();
        if have_stats {
            sqlx::query(
                "INSERT INTO validation.generation_stats (generation, gen_index, kind, n_trials, best_candidate, deflated_sharpe) VALUES ($1, 0, 'evo.eval.strategy', 3, $1, 0.42)",
            )
            .bind(&id)
            .execute(p.pg.pool())
            .await
            .unwrap();
            assert_eq!(p.deflated_sharpe(&["other", &id]).await, Some(0.42));
            sqlx::query("DELETE FROM validation.generation_stats WHERE generation = $1")
                .bind(&id)
                .execute(p.pg.pool())
                .await
                .unwrap();
        }
        sqlx::query("DELETE FROM ledger.strategy_state WHERE strategy_id = $1")
            .bind(&id)
            .execute(p.pg.pool())
            .await
            .unwrap();
    }
}
