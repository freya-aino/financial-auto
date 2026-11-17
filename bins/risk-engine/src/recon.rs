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
//! - **Venue reconciliation** ([`reconcile_venues`], D-121): in real-money mode the live book is
//!   also compared against what the venues themselves say they hold (`ctl.exec.positions`). The
//!   two reconciliations answer different questions — one asks whether this process and Postgres
//!   agree about the fills they saw, the other whether either of them describes the position that
//!   actually exists at the venue — and only the second can catch a fill that never reached the
//!   bus at all.
use std::sync::Arc;

use std::time::Duration;

use chrono::{DateTime, TimeZone, Utc};
use fa_bus::Bus;
use fa_ledger::{
    Fill,
    pg::{self, FillDetails, IntentRecord, PAPER_VENUE, PgLedger},
    pg_accounting::{FillEvent, Recorded},
};
use fa_proto::v1::{CtlReply, CtlRequest, Halt, OrderEvent};
use fa_risk::HaltScope;
use fa_proto::prost::Message as _;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    engine::{BookedFill, Engine, Out, fill_id_of},
    persist::Persist,
};

/// How long the gateway gets to ask every venue and answer `ctl.exec.positions`. Longer than its
/// own per-pass budget, so a slow venue comes back as a reported error rather than as silence.
const VENUE_BOOKS_TIMEOUT: Duration = Duration::from_secs(12);
/// Quantity difference that counts as a difference. The book's quantities came from the venue's
/// own fills, so this is float noise, not a tolerance.
const VENUE_QTY_TOL: f64 = 1e-9;

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
        book: Some(p.book().as_str().to_owned()),
        // Which venue filled it, so a real fill is not filed under the paper venue id: the
        // ledger's `venue_fill`, `lot`, `position_snapshot` and `venue_settlement` are all keyed
        // by it, and the paper venue's settlement rule is not every venue's (D-121).
        venue: Some(pg::venue_uuid(&b.venue)),
        venue_symbol: Some(b.venue_symbol.clone()).filter(|s| !s.is_empty()),
        fee_native: None,
    }
}

fn intent_record(p: &Persist, id: &str, info: (fa_core::InstrumentId, f64, Vec<String>), at_ms: i64) -> IntentRecord {
    let (instrument, signed, strategy_ids) = info;
    IntentRecord {
        intent_id: id.to_owned(),
        book: p.book().as_str().to_owned(),
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
        .bind(p.book().as_str())
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
            book: Some(p.book().as_str().to_owned()),
            venue: Some(pg::venue_uuid(&ev.venue)),
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
    let r = match pg.reconcile_cut(p.book().as_str(), &ours, &in_flight, CASH_TOL_EUR, QTY_TOL).await {
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
            p.audit("ledger.synced", Some(p.book().as_str()), json!({"fills": f_ids, "cash": c_keys})).await;
        }
    }
    let d = &r.diff;
    let summary = json!({"compared_fills": r.compared_fills, "in_flight": r.in_flight});
    let (status, diffs) = if d.is_empty() {
        ("ok", Some(json!({"book": p.book().as_str(), "cut": summary})))
    } else {
        (
            "diff",
            Some(json!({"book": p.book().as_str(), "cut": summary,
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
    p.audit("reconciliation.diff", Some(p.book().as_str()), serde_json::to_value(d).unwrap_or_default()).await;
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

// ---------------------------------------------------------------- the venue's own book (D-121)

/// One venue's own view of what it holds, as the broker-gateway reports it.
#[derive(Debug, Clone, Deserialize)]
struct VenueBooks {
    venue: String,
    #[serde(default)]
    configured: bool,
    /// Why it is not configured (no credentials, a mis-shaped secret).
    #[serde(default)]
    detail: String,
    /// Why it could not be asked. A venue that did not answer is not a venue that holds nothing.
    #[serde(default)]
    error: String,
    #[serde(default)]
    positions: Vec<VenuePosition>,
    #[serde(default)]
    balances: Vec<VenueBalance>,
}

#[derive(Debug, Clone, Deserialize)]
struct VenuePosition {
    #[serde(default)]
    symbol: String,
    #[serde(default)]
    quantity: f64,
    /// Our instrument id, when the operator's symbol mapping names this symbol. Empty otherwise,
    /// which the gateway deliberately does not guess at.
    #[serde(default)]
    instrument_id: String,
}

#[derive(Debug, Clone, Deserialize)]
struct VenueBalance {
    #[serde(default)]
    currency: String,
    #[serde(default)]
    amount: f64,
}

/// Asks the broker-gateway what each venue holds (`ctl.exec.positions`).
async fn ask_venues(bus: &Bus) -> Result<Vec<VenueBooks>, String> {
    let req = CtlRequest { verb: "positions".into(), args: Default::default() };
    let msg = tokio::time::timeout(
        VENUE_BOOKS_TIMEOUT,
        bus.client().request(fa_bus::subjects::ctl("exec", "positions"), req.encode_to_vec().into()),
    )
    .await
    .map_err(|_| "the broker-gateway did not answer ctl.exec.positions in time".to_owned())?
    .map_err(|e| format!("ctl.exec.positions: {e}"))?;
    let reply = CtlReply::decode(msg.payload.as_ref()).map_err(|e| format!("decoding the reply: {e}"))?;
    let body: serde_json::Value =
        serde_json::from_str(&reply.message).map_err(|e| format!("the reply is not JSON: {e}"))?;
    serde_json::from_value(body["venues"].clone()).map_err(|e| format!("the reply has no venues: {e}"))
}

/// What comparing the book against the venues found, before any of it is recorded.
struct VenueComparison {
    status: &'static str,
    /// Positions the book and the venues disagree about. Non-empty means a halt.
    diffs: Vec<serde_json::Value>,
    /// Holdings at a venue in instruments the operator's mapping does not name.
    foreign: Vec<serde_json::Value>,
    /// Venues that could not be compared, and why. Non-empty means nothing was compared.
    unavailable: Vec<serde_json::Value>,
    balances: Vec<serde_json::Value>,
}

/// The comparison itself, with no I/O: the book's positions against the venues' own.
fn compare_books(ours: &[(fa_core::InstrumentId, f64)], books: &[VenueBooks]) -> VenueComparison {
    // Their side, summed per instrument: one instrument may be held at more than one venue, while
    // the book holds one position in it.
    let mut theirs: std::collections::BTreeMap<String, (f64, Vec<String>)> = Default::default();
    let (mut foreign, mut unavailable, mut balances) = (Vec::new(), Vec::new(), Vec::new());
    for b in books {
        if !b.configured {
            unavailable.push(json!({"venue": b.venue, "why": "no credentials", "detail": b.detail}));
            continue;
        }
        if !b.error.is_empty() {
            unavailable.push(json!({"venue": b.venue, "why": "did not answer", "detail": b.error}));
            continue;
        }
        for pos in &b.positions {
            if pos.quantity.abs() <= VENUE_QTY_TOL {
                continue;
            }
            if pos.instrument_id.trim().is_empty() {
                foreign.push(json!({"venue": b.venue, "symbol": pos.symbol, "quantity": pos.quantity}));
                continue;
            }
            let e = theirs.entry(pos.instrument_id.trim().to_owned()).or_insert((0.0, Vec::new()));
            e.0 += pos.quantity;
            e.1.push(b.venue.clone());
        }
        for bal in &b.balances {
            balances.push(json!({"venue": b.venue, "currency": bal.currency, "amount": bal.amount}));
        }
    }
    // A pass that could not ask every venue compares nothing: see `reconcile_venues`.
    if !unavailable.is_empty() {
        return VenueComparison { status: "incomplete", diffs: Vec::new(), foreign, unavailable, balances };
    }
    let mut diffs = Vec::new();
    for (instrument, qty) in ours {
        let id = instrument.0.to_string();
        match theirs.remove(&id) {
            Some((at_venue, venues)) if (at_venue - qty).abs() <= VENUE_QTY_TOL => {
                tracing::debug!(instrument = %id, qty, venues = ?venues, "position agrees with the venue")
            }
            Some((at_venue, venues)) => diffs.push(json!({
                "instrument_id": id, "book": qty, "venue": at_venue, "venues": venues,
                "why": "the quantities differ"})),
            None => diffs.push(json!({
                "instrument_id": id, "book": qty, "venue": 0.0,
                "why": "no venue reports this position"})),
        }
    }
    // Whatever is left is a position at a venue, in an instrument the mapping names, that the book
    // does not have: the platform could have opened it and does not know it exists.
    for (id, (qty, venues)) in theirs {
        diffs.push(json!({
            "instrument_id": id, "book": 0.0, "venue": qty, "venues": venues,
            "why": "a position only the venue has"}));
    }
    let status = if diffs.is_empty() { "ok" } else { "diff" };
    VenueComparison { status, diffs, foreign, unavailable, balances }
}

/// One venue reconciliation pass: the live book against the venues that hold it (D-121).
///
/// Three rules decide what this does, and each exists because the alternative is worse:
///
/// - **Only in real-money mode.** A paper position lives nowhere, so comparing it to a venue would
///   report the whole book as missing.
/// - **Partial information never halts.** A venue that did not answer, or whose credentials were
///   removed, could be holding exactly the position that looks missing here. Such a pass is
///   recorded as `incomplete` and compares nothing: a halt must mean "the books disagree", not
///   "a request failed", or it will be re-armed by habit.
/// - **A holding the operator's mapping does not name is foreign.** An instrument absent from the
///   mapping cannot be traded live at all, so a holding in it cannot be one this platform opened —
///   it is somebody's own asset in their own account. It is reported and left alone. A holding the
///   mapping *does* name and the book does not is the dangerous case, and it halts.
pub async fn reconcile_venues(p: &Persist, engine: &Shared, bus: &Bus) -> Vec<Out> {
    if !engine.lock().await.is_live() {
        return Vec::new();
    }
    let books = match ask_venues(bus).await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(error = %e, "venue reconciliation skipped: the gateway did not answer");
            let mut g = engine.lock().await;
            g.set_venue_recon(json!({"status": "unreachable", "at": Utc::now(), "detail": e}));
            g.counters.insert("venue_recon.unreachable", 1.0);
            return Vec::new();
        }
    };
    // Our side of the comparison: every non-zero position of the active (live) book.
    let ours: Vec<(fa_core::InstrumentId, f64)> = engine.lock().await.book_positions();
    let c = compare_books(&ours, &books);
    // Balances are reported, not compared: the book is EUR at the fill's rate (spec 42 §6) and a
    // venue reports settled native cash, so an equality between them is not a thing that can be
    // true. The positions are what protect against trading around a position that is not there.
    let view = json!({
        "status": c.status, "at": Utc::now(), "compared": ours.len(),
        "diffs": c.diffs, "foreign": c.foreign, "unavailable": c.unavailable, "balances": c.balances,
    });
    let (status, diffs, complete) = (c.status, c.diffs, c.unavailable.is_empty());
    {
        let mut g = engine.lock().await;
        g.set_venue_recon(view.clone());
        g.counters.insert("venue_recon.diffs", diffs.len() as f64);
        g.counters.insert("venue_recon.foreign", c.foreign.len() as f64);
        g.counters.insert("venue_recon.unavailable", c.unavailable.len() as f64);
        g.counters.insert("venue_recon.unreachable", 0.0);
    }
    // One row per venue that was asked, each carrying the whole pass: the comparison is
    // cross-venue (an instrument may be held at two venues, and a book position missing everywhere
    // belongs to no single one of them), so attributing parts of it to one venue would be a guess.
    for b in &books {
        let per_venue = if b.configured {
            if b.error.is_empty() { status } else { "error" }
        } else {
            "not_configured"
        };
        if let Err(e) = p.pg().record_reconciliation(pg::venue_uuid(&b.venue), per_venue, Some(&view)).await {
            tracing::warn!(venue = %b.venue, error = %e, "venue reconciliation run not recorded");
        }
    }
    if status != "diff" {
        if !complete {
            tracing::warn!(view = %view, "venue reconciliation incomplete: nothing compared");
        }
        return Vec::new();
    }
    tracing::error!(diffs = ?diffs, "the live book differs from the venues: halting scope broker");
    p.audit("reconciliation.venue_diff", Some(p.book().as_str()), view).await;
    let mut e = engine.lock().await;
    if e.has_halt(HaltScope::Broker) {
        return Vec::new();
    }
    let h = Halt {
        scope: "broker".into(),
        reason: format!(
            "venue reconciliation: {} position(s) differ from what the venues report (first: {})",
            diffs.len(),
            diffs[0]
        ),
        since: Some(fa_proto::now()),
        manual: false,
    };
    e.on_halt(&h, now_ms());
    vec![Out::Halt(h)]
}

/// Today's payout quota from the configured rule (spec 42 §10 Q1); raise-only. Returns the
/// quota in force.
pub async fn apply_payout_rule(p: &Persist, rule: &fa_ledger::payout::PayoutRule) -> Option<f64> {
    let day = Utc::now().date_naive();
    match p.pg().apply_payout_rule(p.book().as_str(), day, rule).await {
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

    fn book(venue: &str, configured: bool, error: &str, pos: Vec<(&str, f64, &str)>) -> VenueBooks {
        VenueBooks {
            venue: venue.into(),
            configured,
            detail: if configured { String::new() } else { "no credentials".into() },
            error: error.into(),
            positions: pos
                .into_iter()
                .map(|(symbol, quantity, instrument_id)| VenuePosition {
                    symbol: symbol.into(),
                    quantity,
                    instrument_id: instrument_id.into(),
                })
                .collect(),
            balances: vec![VenueBalance { currency: "EUR".into(), amount: 500.0 }],
        }
    }

    fn inst(n: u128) -> fa_core::InstrumentId {
        fa_core::InstrumentId(Uuid::from_u128(n))
    }

    #[test]
    fn a_book_that_matches_the_venue_reconciles() {
        let ours = vec![(inst(1), 0.025)];
        let id = inst(1).0.to_string();
        let c = compare_books(&ours, &[book("kraken", true, "", vec![("XBTEUR", 0.025, &id)])]);
        assert_eq!(c.status, "ok");
        assert!(c.diffs.is_empty(), "{:?}", c.diffs);
        assert_eq!(c.balances.len(), 1, "the balances are reported even when nothing differs");
    }

    #[test]
    fn every_way_the_books_can_disagree_is_a_diff() {
        let (a, b) = (inst(1), inst(2));
        let (id_a, id_b) = (a.0.to_string(), b.0.to_string());
        // the quantity differs
        let c = compare_books(&[(a, 0.025)], &[book("kraken", true, "", vec![("XBTEUR", 0.02, &id_a)])]);
        assert_eq!(c.status, "diff");
        assert_eq!(c.diffs[0]["why"], "the quantities differ");
        // the book holds a position no venue reports: a fill that was undone, or never happened
        let c = compare_books(&[(a, 0.025)], &[book("kraken", true, "", vec![])]);
        assert_eq!(c.status, "diff");
        assert_eq!(c.diffs[0]["why"], "no venue reports this position");
        // a venue holds a position the book does not know about, in a mapped instrument
        let c = compare_books(&[], &[book("kraken", true, "", vec![("XBTEUR", 0.025, &id_a)])]);
        assert_eq!(c.status, "diff");
        assert_eq!(c.diffs[0]["why"], "a position only the venue has");
        // one instrument held at two venues sums to the book's one position
        let c = compare_books(
            &[(a, 3.0)],
            &[
                book("kraken", true, "", vec![("XBTEUR", 1.0, &id_a)]),
                book("lemon", true, "", vec![("US0378331005", 2.0, &id_a)]),
            ],
        );
        assert_eq!(c.status, "ok", "{:?}", c.diffs);
        // and an instrument the book holds at neither is still a diff
        let c = compare_books(&[(a, 1.0), (b, 1.0)], &[book("kraken", true, "", vec![("XBTEUR", 1.0, &id_a)])]);
        assert_eq!(c.diffs.len(), 1);
        assert_eq!(c.diffs[0]["instrument_id"], id_b);
    }

    /// A holding the operator's symbol mapping does not name cannot have been opened by this
    /// platform (an unmapped instrument is rejected before it reaches a venue), so it is reported
    /// and left alone rather than halting trading on somebody's own asset.
    #[test]
    fn an_unmapped_holding_is_foreign_and_does_not_halt() {
        let c = compare_books(&[], &[book("kraken", true, "", vec![("DOGEEUR", 10_000.0, "")])]);
        assert_eq!(c.status, "ok");
        assert!(c.diffs.is_empty());
        assert_eq!(c.foreign.len(), 1);
        assert_eq!(c.foreign[0]["symbol"], "DOGEEUR");
    }

    /// The rule that keeps a halt meaningful: a venue that could not be asked might be holding
    /// exactly the position that looks missing, so an incomplete pass compares nothing.
    #[test]
    fn a_venue_that_cannot_be_asked_compares_nothing() {
        let a = inst(1);
        let id = a.0.to_string();
        for missing in [
            book("lemon", false, "", vec![]),
            book("lemon", true, "connection refused", vec![]),
        ] {
            let c = compare_books(&[(a, 0.025)], &[book("kraken", true, "", vec![("XBTEUR", 0.03, &id)]), missing]);
            assert_eq!(c.status, "incomplete");
            assert!(c.diffs.is_empty(), "an incomplete pass must not report a diff: {:?}", c.diffs);
            assert_eq!(c.unavailable.len(), 1);
        }
    }

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
