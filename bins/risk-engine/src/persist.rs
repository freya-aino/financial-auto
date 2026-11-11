//! Postgres persistence for the paper book (spec 42): intents, fills (with currency, FX source,
//! arrival price and slippage), cash movements, position snapshots, reconciliation runs,
//! strategy lifecycle rows with their sub-books (`ledger.strategy_state`, D-051, D-061), the
//! payout quota and the audit log; plus the reference-data, FX and ECB reads/writes (D-061).
//! Everything here is best effort and idempotent: a failed write is logged and the in-memory
//! engine carries on. Fill persistence, orphan handling and reconciliation live in
//! [`crate::recon`] (unconfirmed fill writes are retried; one that never lands surfaces as a
//! reconciliation `diff`, which halts scope `broker`).
use chrono::{DateTime, TimeZone, Utc};
use fa_ledger::{
    Ledger,
    pg::{IntentRecord, PgLedger},
};
use fa_proto::v1::{OrderIntent, Side};
use fa_risk::strategy::{Checklist, StrategyState, StrategyStatus};
use serde_json::{Value, json};
use sqlx::Row;

use crate::engine::{Evidence, RiskRecord, SubBookRecord};

pub const ACTOR: &str = "system:risk-engine";

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

    /// The ledger (fill events, orphans, reconciliation: [`crate::recon`]).
    pub fn pg(&self) -> &PgLedger {
        &self.pg
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
             horizons_completed, sleeve_eur, pnl_eur, peak_pnl_eur, max_drawdown, gate, book, pipeline, last_reason, updated_at) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, now()) \
             ON CONFLICT (strategy_id) DO UPDATE SET status = EXCLUDED.status, prev_status = EXCLUDED.prev_status, \
             since = EXCLUDED.since, shadow_since = EXCLUDED.shadow_since, horizons_completed = EXCLUDED.horizons_completed, \
             sleeve_eur = EXCLUDED.sleeve_eur, pnl_eur = EXCLUDED.pnl_eur, peak_pnl_eur = EXCLUDED.peak_pnl_eur, \
             max_drawdown = EXCLUDED.max_drawdown, gate = COALESCE(EXCLUDED.gate, ledger.strategy_state.gate), \
             book = COALESCE(EXCLUDED.book, ledger.strategy_state.book), \
             pipeline = COALESCE(EXCLUDED.pipeline, ledger.strategy_state.pipeline), last_reason = EXCLUDED.last_reason, updated_at = now()",
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
        .bind(s.pipeline.as_ref().and_then(|p| serde_json::to_value(p).ok()))
        .bind(&s.last_reason)
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
             max_drawdown::float8 AS max_drawdown, to_jsonb(s) -> 'book' AS book, to_jsonb(s) -> 'pipeline' AS pipeline, \
             to_jsonb(s) ->> 'last_reason' AS last_reason FROM ledger.strategy_state s",
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
                    last_reason: r.get::<Option<String>, _>("last_reason").unwrap_or_default(),
                    pipeline: r
                        .get::<Option<Value>, _>("pipeline")
                        .filter(|v| !v.is_null())
                        .and_then(|v| serde_json::from_value(v).ok()),
                };
                Some((st, book))
            })
            .collect()
    }

    /// Borrow availability per instrument from the current `ref.contract_spec` rows (spec 40
    /// §10 Q5): false when any current venue row says not shortable, true when all known rows
    /// allow it; instruments without a row are absent (the class rule applies).
    pub async fn load_shortable(&self) -> Option<std::collections::BTreeMap<fa_core::InstrumentId, bool>> {
        let rows: Result<Vec<(uuid::Uuid, Option<bool>)>, _> = sqlx::query_as(
            "select instrument_id, bool_and(shortable) from ref.contract_spec \
             where shortable is not null and now() >= valid_from and now() < valid_to group by instrument_id",
        )
        .fetch_all(self.pg.pool())
        .await;
        match rows {
            Ok(r) => Some(r.into_iter().filter_map(|(i, b)| Some((fa_core::InstrumentId(i), b?))).collect()),
            Err(e) => {
                tracing::warn!(error = %e, "contract specs (shortable) not loaded");
                None
            }
        }
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

    /// Gate evidence for a strategy (spec 40 §6.6 items 2 and 3):
    /// - **deflated Sharpe**: the strategy's latest holdout trial (`validation.trial`, `kind =
    ///   holdout`, matched by genome text or `{ id = "<id>"` prefix), else the DSR of a
    ///   generation whose best candidate it is (`validation.generation_stats`);
    /// - **PBO**: the latest loop of the strategy's generation.
    ///
    /// `None` fields are unknown and block promotion.
    pub async fn evidence(&self, ids: &[&str], genome: Option<&str>, generation: Option<&str>) -> Evidence {
        let ids: Vec<String> = ids.iter().filter(|s| !s.is_empty()).map(|s| (*s).to_owned()).collect();
        let mut exact: Vec<String> = ids.clone();
        if let Some(g) = genome.filter(|g| !g.is_empty()) {
            exact.push(g.to_owned());
        }
        let prefixes: Vec<String> = ids.iter().map(|id| format!("{{ id = \"{id}\"%")).collect();
        let mut ev = Evidence::default();
        if exact.is_empty() {
            return ev;
        }
        ev.deflated_sharpe = sqlx::query_scalar::<_, Option<f64>>(
            "SELECT (objectives->>'deflated_sharpe')::float8 FROM validation.trial \
             WHERE kind = 'holdout' AND (candidate = ANY($1) OR candidate LIKE ANY($2)) ORDER BY created_at DESC LIMIT 1",
        )
        .bind(&exact)
        .bind(&prefixes)
        .fetch_optional(self.pg.pool())
        .await
        .ok()
        .flatten()
        .flatten();
        if ev.deflated_sharpe.is_none() {
            ev.deflated_sharpe = sqlx::query_scalar::<_, Option<f64>>(
                "SELECT deflated_sharpe::float8 FROM validation.generation_stats \
                 WHERE best_candidate = ANY($1) OR best_candidate LIKE ANY($2) ORDER BY computed_at DESC LIMIT 1",
            )
            .bind(&exact)
            .bind(&prefixes)
            .fetch_optional(self.pg.pool())
            .await
            .ok()
            .flatten()
            .flatten();
        }
        if let Some(g) = generation.filter(|g| !g.is_empty()) {
            ev.pbo = self.generation_pbo(g).await;
        }
        ev
    }

    /// PBO of the latest loop of a generation (`None` when unknown or NaN).
    pub async fn generation_pbo(&self, generation: &str) -> Option<f64> {
        sqlx::query_scalar::<_, Option<f64>>(
            "SELECT pbo FROM validation.generation_stats WHERE generation = $1 ORDER BY gen_index DESC, computed_at DESC LIMIT 1",
        )
        .bind(generation)
        .fetch_optional(self.pg.pool())
        .await
        .ok()
        .flatten()
        .flatten()
        .filter(|p| p.is_finite())
    }

    /// Selection evidence per strategy id of one strategy generation: the highest-rung trial of
    /// each candidate (`kind = evo.eval.strategy`, id read from the genome text) and the
    /// generation's PBO.
    pub async fn front_selection(
        &self,
        generation: &str,
    ) -> std::collections::BTreeMap<String, fa_risk::pipeline::Selection> {
        let pbo = self.generation_pbo(generation).await;
        let rows = sqlx::query(
            "SELECT DISTINCT ON (candidate) candidate, sharpe, n_obs, (objectives->>'deflated_sharpe')::float8 AS dsr \
             FROM validation.trial WHERE generation = $1 AND kind = 'evo.eval.strategy' \
             ORDER BY candidate, rung DESC, created_at DESC",
        )
        .bind(generation)
        .fetch_all(self.pg.pool())
        .await
        .unwrap_or_default();
        let windows: std::collections::BTreeMap<String, (i64, i64)> = sqlx::query(
            "SELECT candidate, min(window_start) AS ws, max(window_end) AS we FROM validation.trial \
             WHERE generation = $1 AND kind = 'evo.eval.strategy' GROUP BY candidate",
        )
        .bind(generation)
        .fetch_all(self.pg.pool())
        .await
        .unwrap_or_default()
        .iter()
        .filter_map(|r| {
            let ws = r.get::<Option<DateTime<Utc>>, _>("ws")?.timestamp_millis();
            let we = r.get::<Option<DateTime<Utc>>, _>("we")?.timestamp_millis();
            Some((r.get::<String, _>("candidate"), (ws, we)))
        })
        .collect();
        let mut out = std::collections::BTreeMap::new();
        for r in &rows {
            let cand: String = r.get("candidate");
            let window_ms = windows.get(&cand).copied();
            let Some(id) = crate::pipeline::genome_id_of(&cand) else { continue };
            out.insert(
                id.to_owned(),
                fa_risk::pipeline::Selection {
                    deflated_sharpe: r.get::<Option<f64>, _>("dsr").filter(|v| v.is_finite()),
                    sharpe: r.get::<Option<f64>, _>("sharpe").filter(|v| v.is_finite()),
                    n_obs: Some(r.get::<i32, _>("n_obs").max(0) as u64),
                    pbo,
                    window_ms,
                },
            );
        }
        out
    }

    /// The holdout trial of a strategy on one locked holdout, if any.
    pub async fn holdout_outcome(
        &self,
        id: &str,
        genome: &str,
        holdout_id: &str,
    ) -> Option<fa_risk::pipeline::HoldoutOutcome> {
        let r = sqlx::query(
            "SELECT id::text AS id, sharpe, n_obs, (objectives->>'deflated_sharpe')::float8 AS dsr, created_at \
             FROM validation.trial WHERE kind = 'holdout' AND holdout_id::text = $1 \
             AND (candidate = $2 OR candidate LIKE $3) ORDER BY created_at DESC LIMIT 1",
        )
        .bind(holdout_id)
        .bind(genome)
        .bind(format!("{{ id = \"{id}\"%"))
        .fetch_optional(self.pg.pool())
        .await
        .ok()
        .flatten()?;
        Some(fa_risk::pipeline::HoldoutOutcome {
            holdout_id: holdout_id.to_owned(),
            trial_id: r.get("id"),
            deflated_sharpe: r.get::<Option<f64>, _>("dsr").filter(|v| v.is_finite()),
            sharpe: r.get::<Option<f64>, _>("sharpe").filter(|v| v.is_finite()),
            n_obs: r.get::<i32, _>("n_obs").max(0) as u64,
            at_ms: r.get::<DateTime<Utc>, _>("created_at").timestamp_millis(),
        })
    }

    /// Every locked holdout with its id.
    pub async fn holdouts(&self) -> anyhow::Result<Vec<(String, fa_backtest::replay::Holdout)>> {
        let rows = sqlx::query(
            // `embargo_ms` through to_jsonb: readable before migration 20261113000040 too (NULL)
            "SELECT id::text AS id, timescale, start_at, end_at, locked_at, (to_jsonb(h) ->> 'embargo_ms')::bigint AS embargo_ms \
             FROM validation.holdout h ORDER BY timescale, start_at",
        )
        .fetch_all(self.pg.pool())
        .await?;
        Ok(rows
            .iter()
            .filter_map(|r| {
                Some((
                    r.get::<String, _>("id"),
                    fa_backtest::replay::Holdout {
                        timescale: r.get::<String, _>("timescale").parse().ok()?,
                        start_ms: r.get::<DateTime<Utc>, _>("start_at").timestamp_millis(),
                        end_ms: r.get::<DateTime<Utc>, _>("end_at").timestamp_millis(),
                        locked_at_ms: r.get::<DateTime<Utc>, _>("locked_at").timestamp_millis(),
                        embargo_ms: r.get::<Option<i64>, _>("embargo_ms"),
                    },
                ))
            })
            .collect())
    }

    /// Locks `[start, end)` for `timescale` unless a row of the timescale overlaps it (then
    /// `Ok(None)`): serialised by a transaction-scoped advisory lock so concurrent lockers
    /// (another risk-engine, the control-api route) cannot insert overlapping rows.
    pub async fn lock_holdout(
        &self,
        timescale: &str,
        start_ms: i64,
        end_ms: i64,
        embargo_ms: i64,
        note: &str,
    ) -> anyhow::Result<Option<String>> {
        let mut tx = self.pg.pool().begin().await?;
        sqlx::query("SELECT pg_advisory_xact_lock(hashtext('validation.holdout'))").execute(&mut *tx).await?;
        let overlap: bool = sqlx::query_scalar(
            "SELECT EXISTS (SELECT 1 FROM validation.holdout WHERE timescale = $1 AND start_at < $3 AND end_at > $2)",
        )
        .bind(timescale)
        .bind(at(start_ms))
        .bind(at(end_ms))
        .fetch_one(&mut *tx)
        .await?;
        if overlap {
            tx.rollback().await?;
            return Ok(None);
        }
        // the embargo travels with the lock (spec 41 §6.4); before migration 20261113000040 the
        // column is missing and the row is locked without it (readers use the default)
        let has_embargo: bool = sqlx::query_scalar(
            "SELECT EXISTS (SELECT 1 FROM information_schema.columns \
             WHERE table_schema = 'validation' AND table_name = 'holdout' AND column_name = 'embargo_ms')",
        )
        .fetch_one(&mut *tx)
        .await?;
        let id: String = if has_embargo {
            sqlx::query_scalar(
                "INSERT INTO validation.holdout (timescale, start_at, end_at, note, embargo_ms) VALUES ($1, $2, $3, $4, $5) RETURNING id::text",
            )
            .bind(timescale)
            .bind(at(start_ms))
            .bind(at(end_ms))
            .bind(note)
            .bind(embargo_ms.max(0))
            .fetch_one(&mut *tx)
            .await?
        } else {
            tracing::warn!(
                "validation.holdout.embargo_ms missing (migration 20261113000040 not applied); locking without it"
            );
            sqlx::query_scalar(
                "INSERT INTO validation.holdout (timescale, start_at, end_at, note) VALUES ($1, $2, $3, $4) RETURNING id::text",
            )
            .bind(timescale)
            .bind(at(start_ms))
            .bind(at(end_ms))
            .bind(note)
            .fetch_one(&mut *tx)
            .await?
        };
        tx.commit().await?;
        Ok(Some(id))
    }

    /// Persists the restart-safe risk state (`ledger.risk_state`, migration 20261111000051).
    pub async fn save_risk_state(&self, r: &RiskRecord) {
        let res = sqlx::query(
            "INSERT INTO ledger.risk_state (book, state, updated_at) VALUES ($1, $2, now()) \
             ON CONFLICT (book) DO UPDATE SET state = EXCLUDED.state, updated_at = now()",
        )
        .bind(&self.book)
        .bind(serde_json::to_value(r).unwrap_or_default())
        .execute(self.pg.pool())
        .await;
        if let Err(e) = res {
            tracing::warn!(error = %e, "risk state not persisted");
        }
    }

    pub async fn load_risk_state(&self) -> Option<RiskRecord> {
        let v: Value = sqlx::query_scalar("SELECT state FROM ledger.risk_state WHERE book = $1")
            .bind(&self.book)
            .fetch_optional(self.pg.pool())
            .await
            .map_err(|e| tracing::warn!(error = %e, "risk state not loaded"))
            .ok()
            .flatten()?;
        serde_json::from_value(v).map_err(|e| tracing::warn!(error = %e, "risk state unreadable; starting fresh")).ok()
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
        // pipeline record + last_reason round trip (migration 20261111000050)
        let mut ev_s = s.clone();
        ev_s.pipeline = Some(fa_risk::pipeline::PipelineRecord {
            genome: "{ id = \"x\" }".into(),
            demotions: 1,
            ..Default::default()
        });
        ev_s.last_reason = "strategy drawdown 6% ≥ limit 5%".into();
        p.strategy_state(&ev_s, None, None).await;
        let rows = p.load_strategy_states().await;
        let (got, _) = rows.iter().find(|(r, _)| r.strategy_id == id).expect("row loaded");
        assert_eq!(got.pipeline, ev_s.pipeline);
        assert_eq!(got.last_reason, ev_s.last_reason);
        assert_eq!(p.evidence(&[&id], None, None).await, Evidence::default(), "no validation row → unknown");
        assert_eq!(p.evidence(&[], None, None).await, Evidence::default());
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
            assert_eq!(p.evidence(&["other", &id], None, None).await.deflated_sharpe, Some(0.42));
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
