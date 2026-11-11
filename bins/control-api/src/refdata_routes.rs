//! Reference-data overrides and ledger reports (spec 12 §4 "CRUD for manual overrides",
//! spec 42 §4 statements / lots / orphans / payouts; spec 50 §4.2, §4.7).
//!
//! Reads (point-in-time: `at` = real-world instant, `known_at` = knowledge instant, both RFC 3339,
//! default now):
//! * `GET /api/refdata/instruments/{id}?at=&known_at=`: instrument, listings, identifiers, contract
//!   specs and corporate actions as of then (`fa_warehouse::asof`, `corpactions`).
//! * `GET /api/refdata/resolve?venue=|source=&symbol=&at=&known_at=`: `none | one | ambiguous`.
//! * `GET /api/refdata/session?venue=&at=`: session state with basis (table / rule / missing).
//! * `GET /api/refdata/fx?ccy=&at=&known_at=`: EUR per unit with rate time and source.
//!
//! Mutations (each audited as actor `ui` and announced as `fa.v1.RefChanged` on
//! `ref.changed.<entity>` so ingestor and risk-engine caches reload):
//! * `POST /api/refdata/corporate-actions` (`NewCorporateAction`), `.../{id}/cancel`
//! * `POST /api/refdata/symbol-change` `{venue, old_symbol, new_symbol, effective?}`
//! * `POST /api/refdata/identifiers` `{instrument_id, scheme, value, valid_from?}`
//! * `POST /api/refdata/contract-specs` (`NewContractSpec`)
//! * `POST /api/refdata/calendar/{venue}/materialize` `{from, to}` (built-in rule of the venue)
//!
//! Ledger reports (`fa_ledger::pg::PgLedger`, spec 42):
//! * `GET /api/trading/statement?book=paper&from=&to=&format=json|csv`
//! * `GET /api/trading/lots?book=&at=`, `GET /api/trading/pnl/breakdown?book=&from=&to=`
//! * `GET /api/trading/orphans?status=`, `POST /api/trading/orphans/{id}/resolve` (`OrphanAction`)
//! * `POST /api/trading/payouts` `{book?, day, amount_eur}`
//!
//! Everything answers 503 without Postgres.
use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use chrono::{DateTime, NaiveDate, Utc};
use fa_ledger::{pg::PgLedger, pg_accounting::OrphanAction};
use fa_proto::Message;
use fa_warehouse::{
    asof, calendar,
    corpactions::{self, NewCorporateAction, SymbolChangeError},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::PgPool;
use uuid::Uuid;

use crate::state::AppState;

const ACTOR: &str = "ui";

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/refdata/instruments/{id}", get(instrument))
        .route("/api/refdata/resolve", get(resolve))
        .route("/api/refdata/session", get(session))
        .route("/api/refdata/fx", get(fx))
        .route("/api/refdata/corporate-actions", post(add_action))
        .route("/api/refdata/corporate-actions/{id}/cancel", post(cancel_action))
        .route("/api/refdata/symbol-change", post(symbol_change))
        .route("/api/refdata/identifiers", post(add_identifier))
        .route("/api/refdata/contract-specs", post(add_contract_spec))
        .route("/api/refdata/calendar/{venue}/materialize", post(materialize))
        .route("/api/trading/statement", get(statement))
        .route("/api/trading/lots", get(lots))
        .route("/api/trading/pnl/breakdown", get(pnl_breakdown))
        .route("/api/trading/orphans", get(orphans))
        .route("/api/trading/orphans/{id}/resolve", post(resolve_orphan))
        .route("/api/trading/payouts", post(payout))
        .route("/api/trading/restatements/fx", post(restate_fx))
}

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({ "error": msg.into() }))).into_response()
}

fn db_err(e: impl std::fmt::Display) -> Response {
    err(StatusCode::BAD_GATEWAY, format!("postgres: {e}"))
}

fn pool(s: &AppState) -> Result<PgPool, Box<Response>> {
    s.pg.get().cloned().ok_or_else(|| Box::new(err(StatusCode::SERVICE_UNAVAILABLE, "postgres unavailable")))
}

macro_rules! pool_or_return {
    ($s:expr) => {
        match pool(&$s) {
            Ok(p) => p,
            Err(r) => return *r,
        }
    };
}

fn parse_ts(v: &Option<String>, name: &str) -> Result<Option<DateTime<Utc>>, Box<Response>> {
    match v.as_deref().map(str::trim).filter(|x| !x.is_empty()) {
        None => Ok(None),
        Some(x) => DateTime::parse_from_rfc3339(x)
            .map(|t| Some(t.with_timezone(&Utc)))
            .or_else(|_| NaiveDate::parse_from_str(x, "%Y-%m-%d").map(|d| Some(d.and_hms_opt(0, 0, 0).expect("midnight").and_utc())))
            .map_err(|_| Box::new(err(StatusCode::BAD_REQUEST, format!("{name} must be RFC 3339 or YYYY-MM-DD")))),
    }
}

macro_rules! ts_or_return {
    ($v:expr, $name:expr) => {
        match parse_ts(&$v, $name) {
            Ok(t) => t,
            Err(r) => return *r,
        }
    };
}

/// Audit + `ref.changed.<entity>` notice; failures are logged, never fatal.
async fn announce(s: &AppState, pool: &PgPool, entity: &str, ids: Vec<String>, change: &str, payload: Value) {
    if let Err(e) = fa_warehouse::audit::append(pool, ACTOR, &format!("ref.{entity}.{change}"), ids.first().map(String::as_str), payload).await {
        tracing::warn!(error = %e, "audit append failed");
    }
    if let Some(nats) = &s.nats {
        let n = fa_bus::ingest::ref_changed_notice(entity, ids, change, ACTOR);
        if let Err(e) = nats.publish(fa_bus::subjects::ref_changed(entity), n.encode_to_vec().into()).await {
            tracing::warn!(error = %e, "ref.changed publish failed");
        }
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct AsOfQuery {
    at: Option<String>,
    known_at: Option<String>,
    venue: Option<String>,
    source: Option<String>,
    symbol: Option<String>,
    ccy: Option<String>,
}

async fn instrument(State(s): State<AppState>, Path(id): Path<String>, Query(q): Query<AsOfQuery>) -> Response {
    let pool = pool_or_return!(s);
    let Ok(id) = Uuid::parse_str(id.trim()) else { return err(StatusCode::BAD_REQUEST, "id must be a UUID") };
    let at = ts_or_return!(q.at, "at").unwrap_or_else(Utc::now);
    let known_at = ts_or_return!(q.known_at, "known_at");
    let inst = match asof::instrument(&pool, id, at, known_at).await {
        Ok(Some(i)) => i,
        Ok(None) => return err(StatusCode::NOT_FOUND, format!("no instrument {id} valid at {at}")),
        Err(e) => return db_err(e),
    };
    let specs = asof::contract_specs(&pool, id, None, at, known_at).await.unwrap_or_default();
    let actions = corpactions::list_as_of(&pool, id, known_at).await.unwrap_or_default();
    Json(json!({ "at": at, "known_at": known_at, "instrument": inst, "contract_specs": specs, "corporate_actions": actions }))
        .into_response()
}

async fn resolve(State(s): State<AppState>, Query(q): Query<AsOfQuery>) -> Response {
    let pool = pool_or_return!(s);
    let at = ts_or_return!(q.at, "at").unwrap_or_else(Utc::now);
    let known_at = ts_or_return!(q.known_at, "known_at");
    let Some(symbol) = q.symbol.as_deref().filter(|x| !x.is_empty()) else {
        return err(StatusCode::BAD_REQUEST, "symbol is required");
    };
    let r = match (q.venue.as_deref(), q.source.as_deref()) {
        (Some(v), _) => asof::resolve(&pool, v, symbol, at, known_at).await,
        (None, Some(src)) => asof::resolve_source_symbol(&pool, src, symbol, at, known_at).await,
        _ => return err(StatusCode::BAD_REQUEST, "venue or source is required"),
    };
    match r {
        Ok(r) => Json(json!({ "symbol": symbol, "at": at, "resolved": r })).into_response(),
        Err(e) => db_err(e),
    }
}

async fn session(State(s): State<AppState>, Query(q): Query<AsOfQuery>) -> Response {
    let pool = pool_or_return!(s);
    let at = ts_or_return!(q.at, "at").unwrap_or_else(Utc::now);
    let Some(venue) = q.venue.as_deref().filter(|x| !x.is_empty()) else {
        return err(StatusCode::BAD_REQUEST, "venue is required");
    };
    match calendar::session_state_at(&pool, venue, at).await {
        Ok(a) => Json(json!({ "tradable": a.state.tradable(), "session": a })).into_response(),
        Err(e) => db_err(e),
    }
}

async fn fx(State(s): State<AppState>, Query(q): Query<AsOfQuery>) -> Response {
    let pool = pool_or_return!(s);
    let at = ts_or_return!(q.at, "at").unwrap_or_else(Utc::now);
    let known_at = ts_or_return!(q.known_at, "known_at");
    let Some(ccy) = q.ccy.as_deref().filter(|x| !x.is_empty()) else { return err(StatusCode::BAD_REQUEST, "ccy is required") };
    match asof::fx_to_eur(&pool, ccy, at, known_at).await {
        Ok(Some((rate, ts, source))) => {
            Json(json!({ "ccy": ccy.to_uppercase(), "at": at, "rate_to_eur": rate, "rate_time": ts, "source": source }))
                .into_response()
        }
        Ok(None) => err(StatusCode::NOT_FOUND, format!("no {ccy} rate at or before {at}")),
        Err(e) => db_err(e),
    }
}

async fn add_action(State(s): State<AppState>, Json(a): Json<NewCorporateAction>) -> Response {
    let pool = pool_or_return!(s);
    if let Err(m) = a.validate() {
        return err(StatusCode::BAD_REQUEST, m);
    }
    match corpactions::record(&pool, &a).await {
        Ok((id, created)) => {
            if created {
                announce(&s, &pool, "corporate_action", vec![a.instrument_id.to_string(), id.to_string()], "created", json!(a))
                    .await;
            }
            Json(json!({ "ok": true, "id": id, "created": created })).into_response()
        }
        Err(e) => db_err(e),
    }
}

async fn cancel_action(State(s): State<AppState>, Path(id): Path<String>) -> Response {
    let pool = pool_or_return!(s);
    let Ok(id) = Uuid::parse_str(id.trim()) else { return err(StatusCode::BAD_REQUEST, "id must be a UUID") };
    match corpactions::cancel(&pool, id).await {
        Ok(true) => {
            announce(&s, &pool, "corporate_action", vec![id.to_string()], "cancelled", json!({ "id": id })).await;
            Json(json!({ "ok": true, "id": id })).into_response()
        }
        Ok(false) => err(StatusCode::NOT_FOUND, format!("no active corporate action {id}")),
        Err(e) => db_err(e),
    }
}

#[derive(Debug, Deserialize, serde::Serialize)]
pub struct SymbolChangeBody {
    venue: String,
    old_symbol: String,
    new_symbol: String,
    effective: Option<String>,
}

async fn symbol_change(State(s): State<AppState>, Json(b): Json<SymbolChangeBody>) -> Response {
    let pool = pool_or_return!(s);
    let effective = ts_or_return!(b.effective, "effective").unwrap_or_else(Utc::now);
    match corpactions::symbol_change(&pool, &b.venue, b.old_symbol.trim(), b.new_symbol.trim(), effective).await {
        Ok(out) => {
            announce(&s, &pool, "listing", vec![out.instrument_id.to_string()], "symbol_change", json!(out)).await;
            Json(json!({ "ok": true, "change": out })).into_response()
        }
        Err(SymbolChangeError::NotFound(m)) => err(StatusCode::NOT_FOUND, m),
        Err(SymbolChangeError::Invalid(m)) => err(StatusCode::CONFLICT, m),
        Err(SymbolChangeError::Warehouse(e)) => db_err(e),
    }
}

#[derive(Debug, Deserialize, serde::Serialize)]
pub struct IdentifierBody {
    instrument_id: Uuid,
    scheme: String,
    value: String,
    valid_from: Option<String>,
}

async fn add_identifier(State(s): State<AppState>, Json(b): Json<IdentifierBody>) -> Response {
    let pool = pool_or_return!(s);
    let from = ts_or_return!(b.valid_from, "valid_from");
    if b.scheme.trim().is_empty() || b.value.trim().is_empty() {
        return err(StatusCode::BAD_REQUEST, "scheme and value are required");
    }
    match asof::add_identifier(&pool, b.instrument_id, b.scheme.trim(), b.value.trim(), from).await {
        Ok(true) => {
            announce(&s, &pool, "identifier", vec![b.instrument_id.to_string()], "created", json!(b)).await;
            Json(json!({ "ok": true })).into_response()
        }
        Ok(false) => err(StatusCode::CONFLICT, format!("{}:{} belongs to another instrument", b.scheme, b.value)),
        Err(e) => db_err(e),
    }
}

async fn add_contract_spec(State(s): State<AppState>, Json(b): Json<asof::NewContractSpec>) -> Response {
    let pool = pool_or_return!(s);
    match asof::upsert_contract_spec(&pool, &b).await {
        Ok(true) => {
            announce(&s, &pool, "contract_spec", vec![b.instrument_id.to_string()], "updated", json!(b)).await;
            Json(json!({ "ok": true })).into_response()
        }
        Ok(false) => err(StatusCode::NOT_FOUND, format!("unknown venue {}", b.venue)),
        Err(e) => db_err(e),
    }
}

#[derive(Debug, Deserialize)]
pub struct RangeBody {
    from: NaiveDate,
    to: NaiveDate,
}

async fn materialize(State(s): State<AppState>, Path(venue): Path<String>, Json(b): Json<RangeBody>) -> Response {
    let pool = pool_or_return!(s);
    if b.to < b.from || (b.to - b.from).num_days() > 3 * 366 {
        return err(StatusCode::BAD_REQUEST, "need from <= to and at most 3 years");
    }
    let row: Option<(Option<String>, bool)> = sqlx::query_as("SELECT mic, always_open FROM ref.venue WHERE code = $1")
        .bind(&venue)
        .fetch_optional(&pool)
        .await
        .unwrap_or(None);
    let Some((mic, always)) = row else { return err(StatusCode::NOT_FOUND, format!("unknown venue {venue}")) };
    let Some(rule) = calendar::rule_for_venue(&venue, mic.as_deref(), always) else {
        return err(StatusCode::CONFLICT, format!("no built-in calendar rule for venue {venue}"));
    };
    match calendar::materialize(&pool, &venue, rule, b.from, b.to).await {
        Ok(n) => {
            announce(&s, &pool, "calendar", vec![venue.clone()], "materialized", json!({ "from": b.from, "to": b.to, "rows": n }))
                .await;
            Json(json!({ "ok": true, "venue": venue, "rule": rule, "rows": n })).into_response()
        }
        Err(e) => db_err(e),
    }
}

// ------------------------------------------------------------------ ledger reports

#[derive(Debug, Default, Deserialize)]
pub struct BookQuery {
    book: Option<String>,
    from: Option<String>,
    to: Option<String>,
    at: Option<String>,
    format: Option<String>,
    status: Option<String>,
    limit: Option<i64>,
}

fn book(q: &BookQuery) -> String {
    q.book.clone().filter(|b| !b.trim().is_empty()).unwrap_or_else(|| "paper".into())
}

async fn statement(State(s): State<AppState>, Query(q): Query<BookQuery>) -> Response {
    let pool = pool_or_return!(s);
    let from = ts_or_return!(q.from, "from");
    let to = ts_or_return!(q.to, "to");
    match PgLedger::from_pool(pool).statement(&book(&q), from, to, None).await {
        Ok(st) if q.format.as_deref() == Some("csv") => {
            ([(header::CONTENT_TYPE, "text/csv; charset=utf-8")], st.to_csv()).into_response()
        }
        Ok(st) => Json(st).into_response(),
        Err(e) => db_err(e),
    }
}

async fn lots(State(s): State<AppState>, Query(q): Query<BookQuery>) -> Response {
    let pool = pool_or_return!(s);
    let at = ts_or_return!(q.at, "at");
    match PgLedger::from_pool(pool).open_lots(&book(&q), at).await {
        Ok(l) => Json(json!({ "book": book(&q), "at": at, "count": l.len(), "lots": l })).into_response(),
        Err(e) => db_err(e),
    }
}

async fn pnl_breakdown(State(s): State<AppState>, Query(q): Query<BookQuery>) -> Response {
    let pool = pool_or_return!(s);
    let from = ts_or_return!(q.from, "from");
    let to = ts_or_return!(q.to, "to");
    let l = PgLedger::from_pool(pool);
    let b = book(&q);
    let total = match l.book_pnl(&b, from, to, None).await {
        Ok(p) => p,
        Err(e) => return db_err(e),
    };
    let strategies = l.strategy_pnls(&b, from, to).await.unwrap_or_default();
    Json(json!({ "book": b, "from": from, "to": to, "book_pnl": total, "strategies": strategies })).into_response()
}

async fn orphans(State(s): State<AppState>, Query(q): Query<BookQuery>) -> Response {
    let pool = pool_or_return!(s);
    let status = q.status.clone().unwrap_or_else(|| "open".into());
    let filter = (status != "all").then_some(status.as_str());
    match PgLedger::from_pool(pool).orphan_fills(filter, q.limit.unwrap_or(200)).await {
        Ok(rows) => Json(json!({ "status": status, "count": rows.len(), "orphans": rows })).into_response(),
        Err(e) => db_err(e),
    }
}

async fn resolve_orphan(State(s): State<AppState>, Path(id): Path<String>, Json(a): Json<OrphanAction>) -> Response {
    let pool = pool_or_return!(s);
    let Ok(id) = Uuid::parse_str(id.trim()) else { return err(StatusCode::BAD_REQUEST, "id must be a UUID") };
    match PgLedger::from_pool(pool).resolve_orphan(id, &a, ACTOR).await {
        Ok(o) => Json(json!({ "ok": true, "orphan": o })).into_response(),
        Err(fa_ledger::pg::PgError::NotFound(m)) => err(StatusCode::NOT_FOUND, m),
        Err(fa_ledger::pg::PgError::Invalid(m)) => err(StatusCode::CONFLICT, m),
        Err(e) => db_err(e),
    }
}

#[derive(Debug, Deserialize, serde::Serialize)]
pub struct PayoutBody {
    book: Option<String>,
    day: NaiveDate,
    amount_eur: f64,
}

async fn payout(State(s): State<AppState>, Json(b): Json<PayoutBody>) -> Response {
    let pool = pool_or_return!(s);
    let bk = b.book.clone().unwrap_or_else(|| "paper".into());
    match PgLedger::from_pool(pool.clone()).record_payout(&bk, b.day, b.amount_eur).await {
        Ok(created) => {
            if let Err(e) = fa_warehouse::audit::append(&pool, ACTOR, "ledger.payout", Some(&bk), json!(b)).await {
                tracing::warn!(error = %e, "audit append failed");
            }
            Json(json!({ "ok": true, "created": created })).into_response()
        }
        Err(fa_ledger::pg::PgError::Invalid(m)) => err(StatusCode::BAD_REQUEST, m),
        Err(e) => db_err(e),
    }
}

#[derive(Debug, Deserialize)]
pub struct RestateBody {
    #[serde(flatten)]
    req: fa_ledger::restate::FxRestatementRequest,
    /// Write the restatement (default false: dry run).
    #[serde(default)]
    apply: bool,
    /// Required with `apply` (428 otherwise).
    #[serde(default)]
    confirm: bool,
    #[serde(default)]
    reason: Option<String>,
    /// Per-fill lines returned (default 20; the totals always cover every fill).
    #[serde(default)]
    fills: Option<usize>,
}

/// `POST /api/trading/restatements/fx` (spec 42 §6): dry run by default — the plan with the
/// before/after book figures; `{"apply": true, "confirm": true, "reason": …}` writes it
/// (audited `ledger.restatement`), after which the risk-engine must be restarted.
async fn restate_fx(State(s): State<AppState>, Json(b): Json<RestateBody>) -> Response {
    let pool = pool_or_return!(s);
    if b.apply && !b.confirm {
        return err(StatusCode::PRECONDITION_REQUIRED, "apply rewrites booked fills: send {\"confirm\": true} too");
    }
    let l = PgLedger::from_pool(pool).with_stored_settlement().await;
    let mut plan = match l.plan_fx_restatement(&b.req).await {
        Ok(p) => p,
        Err(fa_ledger::pg::PgError::Invalid(m)) => return err(StatusCode::BAD_REQUEST, m),
        Err(e) => return db_err(e),
    };
    let applied = if b.apply {
        let Some(reason) = b.reason.as_deref().filter(|r| !r.trim().is_empty()) else {
            return err(StatusCode::BAD_REQUEST, "apply needs a reason");
        };
        match l.apply_fx_restatement(&plan, ACTOR, reason).await {
            Ok(a) => Some(a),
            Err(fa_ledger::pg::PgError::Invalid(m)) => return err(StatusCode::CONFLICT, m),
            Err(e) => return db_err(e),
        }
    } else {
        None
    };
    if let Some(a) = &applied
        && let Some(n) = plan.notes.first_mut()
    {
        *n = format!("applied as batch {}: restart the risk-engine now", a.batch_id);
    }
    let total = plan.fills.len();
    plan.fills.truncate(b.fills.unwrap_or(20));
    Json(json!({ "dry_run": applied.is_none(), "fills_total": total, "plan": plan, "applied": applied }))
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_accept_rfc3339_and_dates() {
        assert!(parse_ts(&Some("2026-03-01".into()), "at").unwrap().is_some());
        assert!(parse_ts(&Some("2026-03-01T10:00:00Z".into()), "at").unwrap().is_some());
        assert!(parse_ts(&None, "at").unwrap().is_none());
        assert!(parse_ts(&Some("yesterday".into()), "at").is_err());
    }

    #[test]
    fn book_defaults_to_paper() {
        assert_eq!(book(&BookQuery::default()), "paper");
    }
}
