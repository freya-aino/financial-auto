//! FX restatement of booked fills (spec 42 §6, D-072's open item).
//!
//! Fills booked before their instrument's quote currency was known carry `ccy = 'EUR'` at rate
//! 1 (`fx_source` `eur` / `fixed`) although their prices are in the quote currency. A
//! restatement re-labels them with the instrument's quote currency and the EUR rate valid at
//! the fill time (latest `ref.fx_rate.ts ≤ filled_at` of the accounting source `ecb_daily`), leaving
//! quantity, price and the (EUR) fee untouched.
//!
//! It is a dry run unless explicitly applied: [`PgLedger::plan_fx_restatement`] reads the
//! book, builds the per-fill diff and replays the book with and without the restatement (cash,
//! equity, P&L, per-strategy sum vs book), writing nothing. [`PgLedger::apply_fx_restatement`]
//! applies exactly a plan's fills in one transaction (optimistic: every row must still carry its
//! planned old values), records the old values in `ledger.fill_restatement` (migration
//! `20261113000011`) and audits `ledger.restatement`. Applying changes the book of record's EUR
//! cash, so the in-memory paper book of the risk-engine must be reloaded right after (restart
//! it), otherwise its next reconciliation reports the difference and halts `broker`; the lot
//! cache is rebuilt, position snapshots and strategy sub-books keep their old EUR averages
//! until the next snapshot / re-attribution.
use std::collections::BTreeMap;

use chrono::{DateTime, Duration, Utc};
use fa_core::InstrumentId;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::Row;
use uuid::Uuid;

use crate::{
    accounting::{PnlBreakdown, TradeEvent, replay, strategy_pnls},
    money::{self, Decimal},
    pg::{PgError, PgLedger},
};

/// What to restate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FxRestatementRequest {
    pub book: String,
    pub instrument: InstrumentId,
    /// Target currency; default the instrument's `ref.instrument.quote_ccy`.
    #[serde(default)]
    pub ccy: Option<String>,
    /// Only fills booked in this currency are restated (default `EUR`).
    #[serde(default)]
    pub from_ccy: Option<String>,
    /// Window over which the per-strategy vs book check is reported (default 24 h).
    #[serde(default)]
    pub check_hours: Option<i64>,
}

/// One fill's change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FillRestatement {
    pub fill_uuid: Uuid,
    pub venue_fill_id: String,
    pub filled_at: DateTime<Utc>,
    #[serde(with = "money::json")]
    pub qty: Decimal,
    #[serde(with = "money::json")]
    pub price: Decimal,
    pub old_ccy: String,
    #[serde(with = "money::json")]
    pub old_fx: Decimal,
    pub old_source: String,
    pub new_ccy: String,
    #[serde(with = "money::json")]
    pub new_fx: Decimal,
    pub new_source: String,
    pub rate_ts: DateTime<Utc>,
    /// EUR cash effect of the restatement (`+` = more cash): `−qty·price·(new − old)`.
    #[serde(with = "money::json")]
    pub cash_delta_eur: Decimal,
}

/// Book figures before / after.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BookFigures {
    #[serde(with = "money::json")]
    pub cash_eur: Decimal,
    #[serde(with = "money::json")]
    pub equity_eur: Decimal,
    /// Since inception, at the latest marks.
    pub pnl: PnlBreakdown,
    /// Over the check window.
    #[serde(with = "money::json")]
    pub window_book_net_eur: Decimal,
    #[serde(with = "money::json")]
    pub window_strategy_sum_net_eur: Decimal,
    /// `strategy sum − book` over the window (0 for a consistent book).
    #[serde(with = "money::json")]
    pub window_strategy_gap_eur: Decimal,
}

/// A dry-run restatement: the fills it would change and what changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FxRestatementPlan {
    pub book: String,
    pub instrument: InstrumentId,
    pub from_ccy: String,
    pub ccy: String,
    pub planned_at: DateTime<Utc>,
    pub fills: Vec<FillRestatement>,
    /// Fills in `from_ccy` without any rate at or before their time (not restated).
    pub missing_rate: Vec<Uuid>,
    /// Fills per old `fx_source`.
    pub by_old_source: BTreeMap<String, usize>,
    #[serde(with = "money::json")]
    pub cash_delta_eur: Decimal,
    pub before: BookFigures,
    pub after: BookFigures,
    pub check_from: DateTime<Utc>,
    pub notes: Vec<String>,
}

/// Result of [`PgLedger::apply_fx_restatement`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FxRestatementApplied {
    pub batch_id: Uuid,
    pub fills: usize,
    #[serde(with = "money::json")]
    pub cash_delta_eur: Decimal,
}

impl PgLedger {
    /// Builds the restatement plan (writes nothing).
    pub async fn plan_fx_restatement(&self, req: &FxRestatementRequest) -> Result<FxRestatementPlan, PgError> {
        let from_ccy = req.from_ccy.as_deref().unwrap_or(fa_core::BASE_CURRENCY).to_ascii_uppercase();
        let ccy = match &req.ccy {
            Some(c) => c.trim().to_ascii_uppercase(),
            None => self
                .quote_ccy(req.instrument)
                .await?
                .ok_or_else(|| PgError::Invalid(format!("instrument {} has no quote currency", req.instrument)))?,
        };
        if ccy == from_ccy {
            return Err(PgError::Invalid(format!("nothing to restate: target currency is {ccy}")));
        }
        let rows = sqlx::query(
            "SELECT f.id, f.venue_fill_id, f.filled_at, oi.side, f.qty, f.price, f.ccy, f.fx_rate_eur, f.fx_source, \
                    r.rate_to_eur, r.source_id, r.ts \
             FROM ledger.fill f \
             JOIN ledger.venue_order vo ON vo.client_order_id = f.client_order_id \
             JOIN ledger.order_intent oi ON oi.id = vo.intent_id \
             LEFT JOIN LATERAL (SELECT x.rate_to_eur, x.source_id, x.ts FROM ref.fx_rate x \
                                WHERE x.ccy = $4 AND x.ts <= f.filled_at AND x.source_id = $5 \
                                ORDER BY x.ts DESC LIMIT 1) r ON true \
             WHERE oi.book = $1 AND oi.instrument_id = $2 AND upper(f.ccy) = $3 \
             ORDER BY f.filled_at, f.id",
        )
        .bind(&req.book)
        .bind(req.instrument.0)
        .bind(&from_ccy)
        .bind(&ccy)
        .bind(crate::pg_accounting::ACCOUNTING_FX_SOURCE)
        .fetch_all(self.pool())
        .await?;
        let mut fills = Vec::new();
        let mut missing_rate = Vec::new();
        let mut by_old_source: BTreeMap<String, usize> = BTreeMap::new();
        for r in &rows {
            let id: Uuid = r.get("id");
            let Some(rate) = r.get::<Option<Decimal>, _>("rate_to_eur") else {
                missing_rate.push(id);
                continue;
            };
            let qty: Decimal = r.get("qty");
            let qty = if r.get::<String, _>("side") == "sell" { -qty } else { qty };
            let price: Decimal = r.get("price");
            let old_fx: Decimal = r.get("fx_rate_eur");
            let new_fx = money::round(rate);
            let old_source: String = r.get("fx_source");
            *by_old_source.entry(old_source.clone()).or_default() += 1;
            fills.push(FillRestatement {
                fill_uuid: id,
                venue_fill_id: r.get("venue_fill_id"),
                filled_at: r.get("filled_at"),
                qty,
                price,
                old_ccy: r.get("ccy"),
                old_fx,
                old_source,
                new_ccy: ccy.clone(),
                new_fx,
                new_source: r.get("source_id"),
                rate_ts: r.get("ts"),
                cash_delta_eur: money::mul(qty, money::mul(price, old_fx)) - money::mul(qty, money::mul(price, new_fx)),
            });
        }
        let cash_delta_eur = money::sum(fills.iter().map(|f| f.cash_delta_eur));

        // replay the book with and without the restatement
        let now = Utc::now();
        let hours = req.check_hours.unwrap_or(24).max(1);
        let check_from = now - Duration::hours(hours);
        let trades = self.trade_events(&req.book, Some(now)).await?;
        let cash = self.cash_events(&req.book, Some(now)).await?;
        let by_id: BTreeMap<Uuid, &FillRestatement> = fills.iter().map(|f| (f.fill_uuid, f)).collect();
        let restated: Vec<TradeEvent> = trades
            .iter()
            .map(|t| match by_id.get(&t.fill_uuid) {
                Some(f) => TradeEvent {
                    ccy: f.new_ccy.clone(),
                    fx_to_eur: f.new_fx,
                    fx_source: f.new_source.clone(),
                    ..t.clone()
                },
                None => t.clone(),
            })
            .collect();
        let before = self.figures(&trades, &cash, check_from, now).await?;
        let after = self.figures(&restated, &cash, check_from, now).await?;
        let notes = vec![
            "dry run: nothing was written".to_owned(),
            format!(
                "applying changes the book's EUR cash by {} EUR; restart the risk-engine right after so its \
                 in-memory book reloads from Postgres (else its next reconciliation reports the difference and \
                 halts broker)",
                cash_delta_eur.round_dp(6)
            ),
            "quantities, prices and fees are unchanged; the lot cache is rebuilt on apply; position snapshots and \
             strategy sub-books keep their old EUR averages until the next snapshot"
                .to_owned(),
            "rates are the valid-time rate (ref.fx_rate.ts ≤ filled_at), regardless of when the rate was recorded"
                .to_owned(),
        ];
        Ok(FxRestatementPlan {
            book: req.book.clone(),
            instrument: req.instrument,
            from_ccy,
            ccy,
            planned_at: now,
            fills,
            missing_rate,
            by_old_source,
            cash_delta_eur,
            before,
            after,
            check_from,
            notes,
        })
    }

    async fn figures(
        &self,
        trades: &[TradeEvent],
        cash: &[crate::accounting::CashEvent],
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<BookFigures, PgError> {
        let method = self.lot_method();
        let end = replay(method, trades, cash, Some(to));
        let m_to = self.marks_for(&end, Some(to), &crate::Marks::default()).await?;
        let start = replay(method, trades, cash, Some(from));
        let m_from = self.marks_for(&start, Some(from), &crate::Marks::default()).await?;
        let book = crate::accounting::range_pnl(method, trades, cash, Some(from), Some(to), &m_from, &m_to);
        let per = strategy_pnls(method, trades, cash, Some(from), Some(to), &m_from, &m_to);
        let sum = money::sum(per.values().map(|p| p.net_eur));
        Ok(BookFigures {
            cash_eur: end.cash_eur,
            equity_eur: end.equity_eur(&m_to),
            pnl: end.pnl(&m_to),
            window_book_net_eur: book.net_eur,
            window_strategy_sum_net_eur: sum,
            window_strategy_gap_eur: sum - book.net_eur,
        })
    }

    /// Applies a plan's fills in one transaction. Fails (and writes nothing) when any fill no
    /// longer carries the plan's old currency / rate / source. Audited with `actor`.
    pub async fn apply_fx_restatement(
        &self,
        plan: &FxRestatementPlan,
        actor: &str,
        reason: &str,
    ) -> Result<FxRestatementApplied, PgError> {
        if plan.fills.is_empty() {
            return Err(PgError::Invalid("the plan restates no fill".into()));
        }
        let batch_id = Uuid::now_v7();
        let mut tx = self.pool().begin().await?;
        for chunk in plan.fills.chunks(2_000) {
            let ids: Vec<Uuid> = chunk.iter().map(|f| f.fill_uuid).collect();
            let n = sqlx::query(
                "WITH p AS (SELECT * FROM UNNEST($1::uuid[], $2::text[], $3::numeric[], $4::text[], $5::text[], \
                                                 $6::numeric[], $7::text[], $8::timestamptz[]) \
                            AS p(id, old_ccy, old_fx, old_src, new_ccy, new_fx, new_src, rate_ts)), \
                 ins AS (INSERT INTO ledger.fill_restatement (batch_id, fill_id, old_ccy, old_fx_rate_eur, \
                           old_fx_source, new_ccy, new_fx_rate_eur, new_fx_source, rate_ts, reason, actor) \
                         SELECT $9, p.id, p.old_ccy, p.old_fx, p.old_src, p.new_ccy, p.new_fx, p.new_src, p.rate_ts, \
                                $10, $11 FROM p) \
                 UPDATE ledger.fill f SET ccy = p.new_ccy, fx_rate_eur = p.new_fx, fx_source = p.new_src \
                 FROM p WHERE f.id = p.id AND f.ccy = p.old_ccy AND f.fx_rate_eur = p.old_fx \
                   AND f.fx_source = p.old_src",
            )
            .bind(&ids)
            .bind(chunk.iter().map(|f| f.old_ccy.clone()).collect::<Vec<_>>())
            .bind(chunk.iter().map(|f| f.old_fx).collect::<Vec<_>>())
            .bind(chunk.iter().map(|f| f.old_source.clone()).collect::<Vec<_>>())
            .bind(chunk.iter().map(|f| f.new_ccy.clone()).collect::<Vec<_>>())
            .bind(chunk.iter().map(|f| f.new_fx).collect::<Vec<_>>())
            .bind(chunk.iter().map(|f| f.new_source.clone()).collect::<Vec<_>>())
            .bind(chunk.iter().map(|f| f.rate_ts).collect::<Vec<_>>())
            .bind(batch_id)
            .bind(reason)
            .bind(actor)
            .execute(&mut *tx)
            .await?
            .rows_affected();
            if n as usize != chunk.len() {
                return Err(PgError::Invalid(format!(
                    "{} of {} fills changed since the plan was made; re-plan",
                    chunk.len() - n as usize,
                    chunk.len()
                )));
            }
        }
        sqlx::query(
            "INSERT INTO audit.log (actor, action, subject, payload) VALUES ($1, 'ledger.restatement', $2, $3)",
        )
        .bind(actor)
        .bind(plan.instrument.0.to_string())
        .bind(json!({"batch_id": batch_id, "book": plan.book, "fills": plan.fills.len(), "from_ccy": plan.from_ccy,
                         "ccy": plan.ccy, "cash_delta_eur": money::f(plan.cash_delta_eur), "reason": reason}))
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        let cached: Option<(i64,)> =
            sqlx::query_as("SELECT fills FROM ledger.lot_rebuild WHERE book = $1 AND method = $2")
                .bind(&plan.book)
                .bind(self.lot_method().as_str())
                .fetch_optional(self.pool())
                .await?;
        if cached.is_some() {
            self.rebuild_lots(&plan.book).await?;
        }
        Ok(FxRestatementApplied { batch_id, fills: plan.fills.len(), cash_delta_eur: plan.cash_delta_eur })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Fill,
        pg::{CashMovement, FillDetails, IntentRecord},
    };

    async fn ledger() -> Option<PgLedger> {
        let url = std::env::var("DATABASE_URL").ok()?;
        Some(PgLedger::connect(&url).await.expect("connect"))
    }

    /// A throwaway book with a USD instrument whose first fills were booked as EUR at 1.0:
    /// the dry run changes nothing, the plan closes the per-strategy gap, apply rewrites exactly
    /// the planned fills, a stale plan is refused.
    #[tokio::test]
    async fn restatement_dry_run_then_apply() {
        let Some(pg) = ledger().await else { return };
        let at = DateTime::from_timestamp(Utc::now().timestamp() - 3_600, 0).unwrap();
        let Some((usd, _)) = pg.rate_to_eur("USD", at).await.unwrap() else { return };
        let book = format!("test-{}", Uuid::now_v7());
        let inst = InstrumentId::new();
        pg.record_cash_movement(&CashMovement {
            book: book.clone(),
            kind: "deposit".into(),
            amount: 10_000.0,
            ccy: "EUR".into(),
            fx_rate_eur: 1.0,
            reference: "initial".into(),
            occurred_at: at - Duration::seconds(1),
        })
        .await
        .unwrap();
        // s1 buys 1 booked as EUR at 1.0 (wrong), s2 buys 1 in USD and sells it: FIFO leaves the book a
        // USD lot and s1 an EUR lot, so the strategies do not sum to the book until restated
        let legs = [
            (1, "buy", 1.0, 100.0, &["s1"][..], "EUR"),
            (2, "buy", 1.0, 100.0, &["s2"][..], "USD"),
            (3, "sell", 1.0, 105.0, &["s2"][..], "USD"),
        ];
        for (n, side, q, px, strats, ccy) in legs {
            let id = format!("{book}-i{n}");
            let t = at + Duration::seconds(n * 10);
            pg.record_intent(&IntentRecord {
                intent_id: id.clone(),
                book: book.clone(),
                instrument: inst,
                side: side.into(),
                qty: q,
                exec_algo: "market".into(),
                limit_price: None,
                strategy_ids: strats.iter().map(|s| format!("{book}-{s}")).collect(),
                created_at: t,
            })
            .await
            .unwrap();
            let usd_f = money::f(usd);
            let fill = Fill {
                fill_id: format!("{id}:f"),
                intent_id: id,
                instrument: inst,
                quantity: if side == "sell" { -q } else { q },
                price: px,
                fx_to_eur: if ccy == "EUR" { 1.0 } else { usd_f },
                fee_eur: 0.1,
            };
            let d = FillDetails {
                ccy: ccy.into(),
                fx_source: if ccy == "EUR" { "fixed".into() } else { "ecb_daily".into() },
                arrival_price: None,
                slippage_bps: None,
            };
            let strategy_ids: Vec<String> = strats.iter().map(|s| format!("{book}-{s}")).collect();
            assert!(pg.record_fill_with(&fill, t, &strategy_ids, &d).await.unwrap());
        }
        let req = FxRestatementRequest {
            book: book.clone(),
            instrument: inst,
            ccy: Some("USD".into()),
            from_ccy: None,
            check_hours: Some(2),
        };
        let cash_before = pg.book_state(&book, None).await.unwrap().cash_eur;
        let plan = pg.plan_fx_restatement(&req).await.unwrap();
        assert_eq!(pg.book_state(&book, None).await.unwrap().cash_eur, cash_before, "dry run writes nothing");
        assert_eq!(plan.fills.len(), 1);
        assert_eq!(plan.by_old_source, BTreeMap::from([("fixed".to_owned(), 1)]));
        assert!(plan.fills.iter().all(|f| f.new_fx == money::round(usd) && f.new_ccy == "USD"));
        // the buy cost 100 EUR as booked, 100·usd restated
        let expect = dec(100.0) - money::mul(dec(100.0), money::round(usd));
        assert_eq!(plan.cash_delta_eur, expect);
        assert_eq!(plan.after.cash_eur - plan.before.cash_eur, expect);
        assert!(plan.before.window_strategy_gap_eur.abs() > Decimal::new(1, 6), "{:?}", plan.before);
        assert!(plan.after.window_strategy_gap_eur.abs() < Decimal::new(1, 12), "{:?}", plan.after);

        let applied = pg.apply_fx_restatement(&plan, "test", "unit test").await.unwrap();
        assert_eq!(applied.fills, 1);
        let st = pg.book_state(&book, None).await.unwrap();
        assert_eq!(st.cash_eur, plan.after.cash_eur);
        assert_eq!(pg.load_book(&book).await.unwrap().exact().cash_eur, plan.after.cash_eur);
        let (n,): (i64,) = sqlx::query_as("SELECT count(*) FROM ledger.fill_restatement WHERE batch_id = $1")
            .bind(applied.batch_id)
            .fetch_one(pg.pool())
            .await
            .unwrap();
        assert_eq!(n, 1);
        assert!(pg.apply_fx_restatement(&plan, "test", "again").await.is_err(), "a stale plan is refused");
        assert!(pg.plan_fx_restatement(&req).await.unwrap().fills.is_empty(), "nothing left to restate");
        assert_eq!(pg.fill_valuation_gaps(&book).await.unwrap(), 0);

        pg.delete_book(&book).await.unwrap();
        sqlx::query("DELETE FROM audit.log WHERE action = 'ledger.restatement' AND payload->>'book' = $1")
            .bind(&book)
            .execute(pg.pool())
            .await
            .unwrap();
    }

    fn dec(x: f64) -> Decimal {
        money::dec(x)
    }

    /// FX source policy: the ledger's reads use the ECB daily rate even when a newer intraday
    /// row (risk-engine `[fx] intraday`, source `yahoo`) exists. Test-scoped currency, removed after.
    #[tokio::test]
    async fn ledger_fx_reads_use_the_accounting_source_only() {
        let Some(pg) = ledger().await else { return };
        let ccy = format!("Q{}", &uuid::Uuid::now_v7().simple().to_string()[24..26]).to_uppercase();
        let d1 = chrono::DateTime::parse_from_rfc3339("2099-03-01T00:00:00Z").unwrap().with_timezone(&chrono::Utc);
        for (ts, rate, src) in [(d1, "0.5", "ecb_daily"), (d1 + chrono::Duration::hours(1), "0.6", "yahoo")] {
            sqlx::query("INSERT INTO ref.fx_rate (ccy, ts, rate_to_eur, source_id) VALUES ($1, $2, $3::numeric, $4)")
                .bind(&ccy)
                .bind(ts)
                .bind(rate)
                .bind(src)
                .execute(pg.pool())
                .await
                .unwrap();
        }
        let at = d1 + chrono::Duration::hours(2);
        let r = pg.rate_to_eur(&ccy, at).await;
        let m = pg.fx_rates_at(std::slice::from_ref(&ccy), at).await;
        sqlx::query("DELETE FROM ref.fx_rate WHERE ccy = $1").bind(&ccy).execute(pg.pool()).await.unwrap();
        assert_eq!(r.unwrap(), Some((dec(0.5), "ecb_daily".to_owned())), "intraday row ignored");
        assert_eq!(m.unwrap().get(&ccy).copied(), Some(dec(0.5)));
    }
}
