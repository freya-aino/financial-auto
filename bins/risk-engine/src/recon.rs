//! The ledger ↔ risk-engine interface (spec 42 §6, §8):
//!
//! - **Fill persistence** through `fa_ledger` [`PgLedger::record_fill_event`]: a booked fill is
//!   confirmed once Postgres has it (new or duplicate); unconfirmed writes are retried every
//!   [`RETRY_AFTER_MS`]. A fill whose own intent is missing in Postgres (its write failed) is
//!   parked as an orphan by the ledger; the intent is then re-recorded from memory and the fill
//!   retried, which books it and closes the orphan.
//! - **Unknown intents**: a fill event for an intent the engine does not know (pending at a
//!   restart, evicted from the intent cache) is recovered from `ledger.order_intent` and booked
//!   in both books; without an intent anywhere it is parked in `ledger.orphan_fill`.
//! - **Orphan halt**: any open orphan fill on the paper venue halts scope `broker` (spec 42 §8)
//!   until it is resolved (`POST /api/trading/orphans/{id}/resolve`) and the operator resumes.
//! - **Reconciliation** at one cut ([`PgLedger::reconcile_cut`]): exact decimals, in-flight
//!   writes excluded, rows only Postgres has (operator-booked orphans, payouts) applied to the
//!   paper book. A remaining diff halts scope `broker`.
use std::sync::Arc;

use chrono::{DateTime, TimeZone, Utc};
use fa_ledger::{
    Fill,
    pg::{FillDetails, IntentRecord, PAPER_VENUE, PgLedger},
    pg_accounting::{FillEvent, Recorded},
};
use fa_proto::v1::{Halt, OrderEvent};
use fa_risk::HaltScope;
use serde_json::json;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    engine::{BookedFill, Engine, Out, fill_id_of},
    persist::Persist,
};

/// Unconfirmed fills younger than this are in flight (not expected in Postgres yet); older
/// ones count as missing.
pub const IN_FLIGHT_GRACE_MS: i64 = 60_000;
/// Retry an unconfirmed fill write after this long.
pub const RETRY_AFTER_MS: i64 = 5_000;
/// The comparison is exact (decimals at the ledger scale on both sides).
const CASH_TOL_EUR: f64 = 0.0;
const QTY_TOL: f64 = 0.0;

type Shared = Arc<Mutex<Engine>>;

fn at(ms: i64) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(ms).single().unwrap_or_else(Utc::now)
}

fn now_ms() -> i64 {
    Utc::now().timestamp_millis()
}

fn event(p: &Persist, b: &BookedFill) -> FillEvent {
    FillEvent {
        fill: b.fill.clone(),
        filled_at: at(b.at_ms),
        strategy_ids: b.strategy_ids.clone(),
        strategy_weights: b.strategy_weights.clone(),
        details: b.details.clone(),
        book: Some(p.book().to_owned()),
        venue: Some(PAPER_VENUE),
        venue_symbol: None,
        fee_native: None,
    }
}

fn intent_record(p: &Persist, id: &str, info: (fa_core::InstrumentId, f64, Vec<String>), at_ms: i64) -> IntentRecord {
    let (instrument, signed, strategy_ids) = info;
    IntentRecord {
        intent_id: id.to_owned(),
        book: p.book().to_owned(),
        instrument,
        side: if signed < 0.0 { "sell".into() } else { "buy".into() },
        qty: signed.abs(),
        exec_algo: "recovered".into(),
        limit_price: None,
        strategy_ids,
        created_at: at(at_ms),
    }
}

/// Whether an `order.<stage>.<account>` subject belongs to this engine's account. Fills of other
/// accounts are filed under their own book in Postgres, so booking them into the paper book was
/// a real (persistent) reconciliation diff on 2026-11-11 (a `paper-test` harness).
pub fn own_account(subject: &str, account: &str) -> bool {
    let mut parts = subject.splitn(3, '.');
    matches!((parts.next(), parts.next(), parts.next()), (Some("order"), Some(_), Some(a)) if a == account)
}

/// Writes booked fills through `record_fill_event` and confirms them in the engine. Returns
/// the halt raised when an orphan is left open.
pub async fn persist_fills(p: &Persist, engine: &Shared, fills: Vec<BookedFill>) -> Vec<Out> {
    if fills.is_empty() {
        return Vec::new();
    }
    let mut confirmed = Vec::new();
    let mut orphaned = false;
    for b in fills {
        let ev = event(p, &b);
        match p.pg().record_fill_event(&ev).await {
            Ok(Recorded::New | Recorded::Duplicate) => confirmed.push(b.fill.fill_id.clone()),
            Ok(Recorded::Orphaned { id, .. }) => {
                // our own intent did not reach Postgres: re-record it and book the fill
                let info = engine.lock().await.intent_info(&b.fill.intent_id);
                let mut booked = false;
                if let Some(info) = info {
                    let rec = intent_record(p, &b.fill.intent_id, info, b.at_ms);
                    if p.pg().record_intent(&rec).await.is_ok()
                        && matches!(p.pg().record_fill_event(&ev).await, Ok(Recorded::New | Recorded::Duplicate))
                    {
                        tracing::warn!(fill = %b.fill.fill_id, orphan = %id, "intent re-recorded; orphan fill booked");
                        confirmed.push(b.fill.fill_id.clone());
                        booked = true;
                    }
                }
                orphaned |= !booked;
            }
            Err(e) => tracing::warn!(fill = %b.fill.fill_id, error = %e, "fill not persisted (will retry)"),
        }
    }
    engine.lock().await.confirm_fills(&confirmed);
    if orphaned { check_orphans(p, engine).await } else { Vec::new() }
}

/// Retries unconfirmed fill writes (called every second; each fill at most every
/// [`RETRY_AFTER_MS`]).
pub async fn retry_unconfirmed(p: &Persist, engine: &Shared) -> Vec<Out> {
    let due = engine.lock().await.retry_fills(now_ms(), RETRY_AFTER_MS);
    if due.is_empty() {
        return Vec::new();
    }
    tracing::info!(fills = due.len(), "retrying unconfirmed fill writes");
    persist_fills(p, engine, due).await
}

/// Fill events for intents the engine did not know: recover the intent from Postgres and book
/// the fill in both books, else park it as an orphan (and halt).
pub async fn resolve_unknown(p: &Persist, engine: &Shared, events: Vec<OrderEvent>) -> Vec<Out> {
    let mut outs = Vec::new();
    let mut orphaned = false;
    for ev in events {
        let row: Option<(Uuid, String, fa_ledger::Decimal, serde_json::Value)> = sqlx::query_as(
            "SELECT instrument_id, side, qty, params FROM ledger.order_intent WHERE idempotency_key = $1 AND book = $2",
        )
        .bind(&ev.intent_id)
        .bind(p.book())
        .fetch_optional(p.pg().pool())
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(intent = %ev.intent_id, error = %e, "intent lookup failed");
            None
        });
        if let Some((inst, side, qty, params)) = row {
            let qty = fa_ledger::money::f(qty).max(ev.fill_quantity);
            let signed = if side == "sell" { -qty } else { qty };
            let ids: Vec<String> = serde_json::from_value(params["strategy_ids"].clone()).unwrap_or_default();
            let (o, booked) = {
                let mut e = engine.lock().await;
                e.recover_intent(&ev.intent_id, fa_core::InstrumentId(inst), signed, ids);
                let o = e.on_order_event(&ev, now_ms());
                (o, e.take_booked_fills())
            };
            tracing::warn!(intent = %ev.intent_id, fills = booked.len(), "fill for an unknown intent recovered from postgres");
            outs.extend(o);
            outs.extend(persist_fills(p, engine, booked).await);
            continue;
        }
        // no intent anywhere: an orphan (spec 42 §8). Nothing is assumed: the instrument comes
        // from the event (id, else the venue symbol resolved by the ledger), the fill currency is
        // left to the instrument's quote currency, the fee stays in the venue's currency.
        let now = now_ms();
        let fill = Fill {
            fill_id: fill_id_of(&ev, now),
            intent_id: ev.intent_id.clone(),
            instrument: fa_core::InstrumentId(Uuid::parse_str(ev.instrument_id.trim()).unwrap_or(Uuid::nil())),
            quantity: ev.fill_quantity,
            price: ev.fill_price,
            fx_to_eur: 1.0,
            fee_eur: 0.0,
        };
        let fe = FillEvent {
            filled_at: at(ev.event_time.as_ref().map_or(now, |t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000)),
            fill,
            strategy_ids: Vec::new(),
            strategy_weights: Vec::new(),
            details: FillDetails {
                ccy: String::new(),
                fx_source: "unknown".into(),
                arrival_price: (ev.arrival_price > 0.0).then_some(ev.arrival_price),
                slippage_bps: (ev.arrival_price > 0.0).then_some(ev.slippage_bps),
            },
            book: Some(p.book().to_owned()),
            venue: Some(PAPER_VENUE),
            venue_symbol: Some(ev.venue_symbol.trim().to_owned()).filter(|s| !s.is_empty()),
            fee_native: Some(fa_ledger::pg_accounting::NativeFee {
                amount: ev.fee,
                ccy: ev.fee_currency.trim().to_owned(),
            }),
        };
        match p.pg().record_fill_event(&fe).await {
            Ok(r) => {
                tracing::error!(intent = %ev.intent_id, outcome = ?r, "fill for an intent unknown to memory and postgres");
                orphaned = true;
            }
            Err(e) => tracing::warn!(intent = %ev.intent_id, error = %e, "orphan fill not recorded"),
        }
    }
    if orphaned {
        outs.extend(check_orphans(p, engine).await);
    }
    outs
}

/// Halts scope `broker` while the paper venue has open orphan fills (spec 42 §8). Only raises
/// (never clears): resolving the orphans and `risk.resume.broker` is the operator's call.
pub async fn check_orphans(p: &Persist, engine: &Shared) -> Vec<Out> {
    let n = match p.pg().open_orphan_count(Some(PAPER_VENUE)).await {
        Ok(n) => n,
        Err(e) => {
            tracing::debug!(error = %e, "open orphan count not read");
            return Vec::new();
        }
    };
    engine.lock().await.counters.insert("ledger.open_orphans", n as f64);
    if n == 0 {
        return Vec::new();
    }
    let mut e = engine.lock().await;
    if e.has_halt(HaltScope::Broker) {
        return Vec::new();
    }
    let h = Halt {
        scope: "broker".into(),
        reason: format!("{n} open orphan fill(s) on the paper venue: resolve them, then resume broker"),
        since: Some(fa_proto::now()),
        manual: false,
    };
    e.on_halt(&h, now_ms());
    tracing::error!(open_orphans = n, "halting scope broker on open orphan fills");
    vec![Out::Halt(h)]
}

/// One reconciliation pass (spec 42 §6): the paper book against Postgres at the same cut.
/// Rows only Postgres has are applied to the paper book (audited `ledger.synced`); a remaining
/// diff is recorded, audited and halts scope `broker`. Returns the halt, if any.
pub async fn reconcile(p: &Persist, engine: &Shared) -> Vec<Out> {
    let (ours, in_flight) = engine.lock().await.recon_cut(now_ms(), IN_FLIGHT_GRACE_MS);
    let pg: &PgLedger = p.pg();
    let r = match pg.reconcile_cut(p.book(), &ours, &in_flight, CASH_TOL_EUR, QTY_TOL).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(error = %e, "reconciliation skipped: cannot read the ledger");
            return Vec::new();
        }
    };
    if !r.fills_only_theirs.is_empty() || !r.cash_only_theirs.is_empty() {
        let fills: Vec<fa_ledger::ExactFill> = r.fills_only_theirs.iter().map(|(f, _)| f.clone()).collect();
        let (f_ids, c_keys) = engine.lock().await.sync_from_ledger(&fills, &r.cash_only_theirs);
        if !f_ids.is_empty() || !c_keys.is_empty() {
            tracing::warn!(
                fills = f_ids.len(),
                cash = c_keys.len(),
                "ledger rows from other writers applied to the paper book"
            );
            p.audit("ledger.synced", Some(p.book()), json!({"fills": f_ids, "cash": c_keys})).await;
        }
    }
    let d = &r.diff;
    let summary = json!({"compared_fills": r.compared_fills, "in_flight": r.in_flight});
    let (status, diffs) = if d.is_empty() {
        ("ok", Some(json!({"book": p.book(), "cut": summary})))
    } else {
        (
            "diff",
            Some(json!({"book": p.book(), "cut": summary,
                "cash": {"memory": d.cash_ours, "postgres": d.cash_theirs},
                "positions": d.positions.iter().map(|x| json!({"instrument_id": x.instrument, "memory": x.ours, "postgres": x.broker})).collect::<Vec<_>>(),
                "fills_missing_in_postgres": d.fills_only_ours})),
        )
    };
    if let Err(e) = pg.record_reconciliation(PAPER_VENUE, status, diffs.as_ref()).await {
        tracing::warn!(error = %e, "reconciliation run not recorded");
    }
    if d.is_empty() {
        tracing::debug!(compared = r.compared_fills, in_flight = r.in_flight, "reconciliation ok");
        return Vec::new();
    }
    tracing::error!(diff = ?d, "paper book differs from Postgres at the same cut: halting scope broker");
    p.audit("reconciliation.diff", Some(p.book()), serde_json::to_value(d).unwrap_or_default()).await;
    let h = Halt {
        scope: "broker".into(),
        reason: format!(
            "reconciliation diff: cash memory {:.2} vs postgres {:.2}, {} position diffs, {} fills missing in postgres",
            d.cash_ours,
            d.cash_theirs,
            d.positions.len(),
            d.fills_only_ours.len()
        ),
        since: Some(fa_proto::now()),
        manual: false,
    };
    engine.lock().await.on_halt(&h, now_ms());
    vec![Out::Halt(h)]
}

/// Today's payout quota from the configured rule (spec 42 §10 Q1); raise-only. Returns the
/// quota in force.
pub async fn apply_payout_rule(p: &Persist, rule: &fa_ledger::payout::PayoutRule) -> Option<f64> {
    let day = Utc::now().date_naive();
    match p.pg().apply_payout_rule(p.book(), day, rule).await {
        Ok(q) => Some(q),
        Err(e) => {
            tracing::warn!(error = %e, "payout rule not applied");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_this_accounts_order_subjects_are_booked() {
        assert!(own_account("order.event.paper", "paper"));
        assert!(own_account("order.intent.paper", "paper"));
        assert!(!own_account("order.event.paper-test", "paper"));
        assert!(!own_account("order.approved.live", "paper"));
        assert!(!own_account("order.event", "paper"));
        assert!(!own_account("risk.event.paper", "paper"));
    }
}
