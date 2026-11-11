//! Operator-console routes for the D-066..D-071 surface (spec 50 §4.5–§4.7, spec 51):
//!
//! | Route | Answer |
//! |---|---|
//! | `GET /api/decoders/serving` | `ctl.decoders.status`: neural slots per (family, timescale), serving vs shadow, online scores, pins, disabled slots |
//! | `GET /api/decoders/cards?limit=` | decoder cards (`fa-models/decoders/**/*.card.json`) newest first, card JSON inline |
//! | `POST /api/decoders/{reload\|promote\|pin\|unpin\|off\|on}` | `ctl.decoders.<verb>` with the JSON body as args (`family`, `timescale`, `key`); audited `decoders.<verb>` |
//! | `GET /api/trading/pipeline` | `ctl.risk.pipeline {action: status}` (spec 35 §6.4 lifecycle round, holdouts) |
//! | `POST /api/trading/pipeline/run` | `ctl.risk.pipeline {action: run}` (risk-engine audits `pipeline.run`, actor `ui`) |
//! | `GET /api/trading/allocators` | `ctl.risk.allocators` (live paper allocator tournament) |
//! | `GET /api/trading/halts` | persisted halts (`ledger.risk_state`, survive restarts) merged with the halts control-api saw on the bus |
//! | `POST /api/trading/halts/{scope}/rearm` | `{"confirm": true, "reason"?}` → `fa.v1.Resume` on `risk.resume.<scope>`; 428 without confirm; audited `halt.rearm` |
//! | `GET /api/trading/risk-events?since=&limit=&kind=` | `RiskEvent`s, halts, resumes and reconciliation diffs from `audit.log`, newest first |
//! | `GET /api/refdata/venues` | venues with their calendar rule and materialised session range |
//! | `GET /api/refdata/calendar/{venue}?from=&to=` | sessions per day: `table` rows, else the built-in rule |
//! | `GET /api/refdata/corporate-actions?instrument_id=&known_at=&limit=` | corporate actions (point-in-time when `instrument_id` is given) |
//!
//! [`decode_extra`] is the bus-tap decoder for the subjects added by D-066/D-069
//! (`ops.alert` JSON, `latent.membership.*`, `latent.lifecycle.*`, `latent.mcmc.*`,
//! `latent.hier.*`, `latent.crossprior.*`) plus a JSON fallback for any `{`-prefixed payload.
// Handlers return ready `Response`s as their error type (axum style), like mcmc_routes.
#![allow(clippy::result_large_err)]

use std::time::Duration;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use bytes::Bytes;
use chrono::{DateTime, Days, NaiveDate, Utc};
use fa_blob::BlobStore;
use fa_proto::{
    Message,
    v1::{
        CrossAssetPriors, CtlReply, CtlRequest, KernelUpdate, RegionHierarchy, RegionLifecycleEvent, RegionMembership,
        Resume,
    },
};
use fa_warehouse::{calendar, corpactions};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use sqlx::Row;
use uuid::Uuid;

use crate::state::AppState;

const CTL_TIMEOUT: Duration = Duration::from_secs(5);
const ACTOR: &str = "ui";
/// `ctl.decoders.<verb>` an operator may send (status is the GET).
pub const DECODER_VERBS: [&str; 6] = ["reload", "promote", "pin", "unpin", "off", "on"];
/// Halt scope families the risk-engine re-arms (`fa_risk::HaltScope::parse`).
pub const REARM_FAMILIES: [&str; 9] =
    ["portfolio", "daily_loss", "source", "feed", "model", "snapshot", "broker", "venue", "fx"];

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/decoders/serving", get(decoders_serving))
        .route("/api/decoders/cards", get(decoder_cards))
        .route("/api/decoders/{verb}", post(decoder_action))
        .route("/api/trading/pipeline", get(pipeline_status))
        .route("/api/trading/pipeline/run", post(pipeline_run))
        .route("/api/trading/allocators", get(allocators))
        .route("/api/trading/halts", get(halts))
        .route("/api/trading/halts/{scope}/rearm", post(rearm))
        .route("/api/trading/risk-events", get(risk_events))
        .route("/api/refdata/venues", get(venues))
        .route("/api/refdata/calendar/{venue}", get(calendar_range))
        .route("/api/refdata/corporate-actions", get(corporate_actions))
}

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({ "error": msg.into() }))).into_response()
}

// ------------------------------------------------------------------ ctl helper

async fn ctl(s: &AppState, service: &str, verb: &str, args: Vec<(String, String)>) -> Result<CtlReply, Response> {
    let Some(client) = s.nats.as_ref().filter(|_| s.nats_connected()) else {
        return Err(err(StatusCode::SERVICE_UNAVAILABLE, "NATS not connected"));
    };
    let subject = fa_bus::subjects::ctl(service, verb);
    let req = CtlRequest {
        verb: verb.into(),
        args: args.into_iter().chain([("actor".to_owned(), ACTOR.to_owned())]).collect(),
    };
    match tokio::time::timeout(CTL_TIMEOUT, client.request(subject.clone(), Bytes::from(req.encode_to_vec()))).await {
        Ok(Ok(msg)) => CtlReply::decode(msg.payload.as_ref())
            .map_err(|e| err(StatusCode::BAD_GATEWAY, format!("bad reply on {subject}: {e}"))),
        Ok(Err(e)) => Err(err(StatusCode::BAD_GATEWAY, format!("{subject}: {e} (is the {service} service running?)"))),
        Err(_) => Err(err(StatusCode::GATEWAY_TIMEOUT, format!("{subject}: no reply within {CTL_TIMEOUT:?}"))),
    }
}

fn message_json(m: &str) -> Value {
    serde_json::from_str(m).unwrap_or_else(|_| Value::String(m.to_owned()))
}

/// `CtlReply{ok:false}` → HTTP status: agreed prefixes first, then model-runtime's plain messages.
pub fn reply_status(message: &str) -> StatusCode {
    match message.split(':').next().unwrap_or_default() {
        "not_found" => return StatusCode::NOT_FOUND,
        "conflict" => return StatusCode::CONFLICT,
        "invalid" => return StatusCode::BAD_REQUEST,
        _ => {}
    }
    if message.starts_with("missing") || message.starts_with("unknown verb") || message.contains("invalid") {
        StatusCode::BAD_REQUEST
    } else if message.contains("not a loaded candidate") {
        StatusCode::NOT_FOUND
    } else {
        StatusCode::CONFLICT
    }
}

async fn audit(s: &AppState, action: &str, subject: Option<&str>, payload: Value) -> bool {
    crate::audit::record(s, action, subject, payload).await
}

// ------------------------------------------------------------------ decoders (D-068)

async fn decoders_serving(State(s): State<AppState>) -> Response {
    match ctl(&s, "decoders", "status", vec![]).await {
        Ok(r) if r.ok => Json(message_json(&r.message)).into_response(),
        Ok(r) => err(reply_status(&r.message), r.message),
        Err(resp) => resp,
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct LimitQuery {
    limit: Option<usize>,
}

async fn decoder_cards(State(s): State<AppState>, Query(q): Query<LimitQuery>) -> Response {
    let limit = q.limit.unwrap_or(60).clamp(1, 300);
    let secrets = match fa_core::Secrets::load() {
        Ok(x) => x,
        Err(e) => return err(StatusCode::SERVICE_UNAVAILABLE, format!("garage secrets: {e}")),
    };
    let store = match fa_blob::S3BlobStore::from_config(&s.cfg, &secrets) {
        Ok(x) => x,
        Err(e) => return err(StatusCode::SERVICE_UNAVAILABLE, format!("garage: {e}")),
    };
    let mut listed = match store.list(fa_blob::buckets::MODELS, "decoders/").await {
        Ok(l) => l,
        Err(e) => return err(StatusCode::BAD_GATEWAY, format!("garage list: {e}")),
    };
    listed.retain(|o| o.key.ends_with(".card.json"));
    let total = listed.len();
    listed.sort_by(|a, b| b.last_modified.cmp(&a.last_modified).then_with(|| b.key.cmp(&a.key)));
    listed.truncate(limit);
    let mut cards = Vec::with_capacity(listed.len());
    for o in listed {
        let key = fa_blob::ObjectKey { bucket: fa_blob::buckets::MODELS, key: o.key.clone() };
        let card = match store.get(&key).await {
            Ok(b) => serde_json::from_slice::<Value>(&b).unwrap_or(Value::Null),
            Err(e) => json!({ "error": e.to_string() }),
        };
        cards.push(json!({ "key": o.key, "size": o.size, "last_modified": o.last_modified, "card": card }));
    }
    Json(json!({ "total": total, "count": cards.len(), "cards": cards })).into_response()
}

/// Turns a JSON object body into `ctl` args (strings as-is, other scalars stringified).
pub fn body_args(body: &Value) -> Vec<(String, String)> {
    body.as_object()
        .map(|o| {
            o.iter()
                .filter(|(k, _)| k.as_str() != "actor" && k.as_str() != "confirm")
                .filter_map(|(k, v)| match v {
                    Value::String(x) => Some((k.clone(), x.clone())),
                    Value::Number(n) => Some((k.clone(), n.to_string())),
                    Value::Bool(b) => Some((k.clone(), b.to_string())),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Lenient JSON body: empty → `{}`; invalid JSON → `Err(400)`.
fn json_body(b: &Bytes) -> Result<Value, Response> {
    if b.iter().all(u8::is_ascii_whitespace) {
        return Ok(json!({}));
    }
    serde_json::from_slice(b).map_err(|e| err(StatusCode::BAD_REQUEST, format!("body is not JSON: {e}")))
}

async fn decoder_action(State(s): State<AppState>, Path(verb): Path<String>, body: Bytes) -> Response {
    if !DECODER_VERBS.contains(&verb.as_str()) {
        return err(StatusCode::NOT_FOUND, format!("unknown decoder action {verb:?}; use {}", DECODER_VERBS.join("|")));
    }
    let body = match json_body(&body) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let args = body_args(&body);
    let subject = args.iter().find(|(k, _)| k == "key").map(|(_, v)| v.clone()).or_else(|| {
        let f = args.iter().find(|(k, _)| k == "family").map(|(_, v)| v.as_str())?;
        let t = args.iter().find(|(k, _)| k == "timescale").map(|(_, v)| v.as_str()).unwrap_or("");
        Some(format!("{f}@{t}"))
    });
    let reply = ctl(&s, "decoders", &verb, args.clone()).await;
    let (ok, message) = match &reply {
        Ok(r) => (r.ok, r.message.clone()),
        Err(_) => (false, "no reply".into()),
    };
    let args_json: Map<String, Value> = args.into_iter().map(|(k, v)| (k, Value::String(v))).collect();
    let audited = audit(
        &s,
        &format!("decoders.{verb}"),
        subject.as_deref(),
        json!({ "args": args_json, "ok": ok, "reply": message }),
    )
    .await;
    match reply {
        Ok(r) if r.ok => {
            Json(json!({ "ok": true, "verb": verb, "message": message_json(&r.message), "audited": audited }))
                .into_response()
        }
        Ok(r) => err(reply_status(&r.message), r.message),
        Err(resp) => resp,
    }
}

// ------------------------------------------------------------------ strategy pipeline + allocators (D-071)

async fn risk_json(s: &AppState, verb: &str, args: Vec<(String, String)>) -> Response {
    match ctl(s, "risk", verb, args).await {
        Ok(r) if r.ok => Json(message_json(&r.message)).into_response(),
        Ok(r) => err(reply_status(&r.message), r.message),
        Err(resp) => resp,
    }
}

async fn pipeline_status(State(s): State<AppState>) -> Response {
    risk_json(&s, "pipeline", vec![("action".into(), "status".into())]).await
}

async fn pipeline_run(State(s): State<AppState>) -> Response {
    risk_json(&s, "pipeline", vec![("action".into(), "run".into())]).await
}

async fn allocators(State(s): State<AppState>) -> Response {
    risk_json(&s, "allocators", vec![]).await
}

// ------------------------------------------------------------------ halts + re-arm (D-071)

/// Halts of a persisted `RiskRecord` (`ledger.risk_state.state`) as UI rows.
pub fn persisted_halts(state: &Value) -> Vec<Value> {
    let since = |v: &Value| v["since_ms"].as_i64().and_then(DateTime::from_timestamp_millis).map(|t| t.to_rfc3339());
    let mut out = Vec::new();
    if let Some(m) = state["halts"].as_object() {
        for (scope, h) in m {
            out.push(json!({ "scope": scope, "reason": h["reason"], "manual": h["manual"], "since": since(h) }));
        }
    }
    if let Some(m) = state["source_halts"].as_object() {
        for (id, h) in m {
            out.push(json!({ "scope": format!("source:{id}"), "reason": h["reason"], "manual": h["manual"], "since": since(h) }));
        }
    }
    out
}

/// `Some(family)` when `scope` names a halt the risk-engine re-arms; `None` otherwise.
pub fn rearm_family(scope: &str) -> Option<&'static str> {
    let scope = scope.trim();
    if scope.is_empty() || scope.len() > 128 || scope.contains(['.', '*', '>', ' ']) {
        return None;
    }
    let family = scope.split(':').next().unwrap_or(scope);
    REARM_FAMILIES.iter().copied().find(|f| *f == family)
}

async fn load_persisted(s: &AppState) -> Option<(String, DateTime<Utc>, Value)> {
    let pool = s.pg.get()?;
    let row = sqlx::query("SELECT book, updated_at, state FROM ledger.risk_state ORDER BY updated_at DESC LIMIT 1")
        .fetch_optional(pool)
        .await
        .map_err(|e| tracing::warn!(error = %e, "ledger.risk_state not read"))
        .ok()??;
    Some((row.get("book"), row.get("updated_at"), row.get("state")))
}

async fn halts(State(s): State<AppState>) -> Response {
    let persisted = load_persisted(&s).await;
    let live: Vec<Value> = s.live().risk_view()["halts"].as_array().cloned().unwrap_or_default();
    let p_rows = persisted.as_ref().map(|(_, _, st)| persisted_halts(st)).unwrap_or_default();
    let mut merged: std::collections::BTreeMap<String, Value> = std::collections::BTreeMap::new();
    for h in &p_rows {
        let scope = h["scope"].as_str().unwrap_or_default().to_owned();
        let mut row = h.clone();
        row["persisted"] = json!(true);
        row["live"] = json!(false);
        merged.insert(scope, row);
    }
    for h in &live {
        let scope = h["scope"].as_str().unwrap_or_default().to_owned();
        let e = merged.entry(scope.clone()).or_insert_with(|| {
            let mut r = h.clone();
            r["persisted"] = json!(false);
            r
        });
        e["live"] = json!(true);
    }
    let halts: Vec<Value> = merged
        .into_iter()
        .map(|(scope, mut v)| {
            v["rearmable"] = json!(rearm_family(&scope).is_some());
            v["family"] = json!(scope.split(':').next().unwrap_or(&scope));
            v
        })
        .collect();
    Json(json!({
        "halts": halts,
        "persisted": persisted.as_ref().map(|(book, at, st)| json!({
            "book": book,
            "updated_at": at,
            "peak_equity": st["risk"]["peak_equity"],
            "day_start_equity": st["day_start_equity"],
            "risk_halts": st["risk"]["halts"],
        })),
        "postgres": s.pg.get().is_some(),
        "families": REARM_FAMILIES,
    }))
    .into_response()
}

#[derive(Debug, Default, Deserialize)]
pub struct RearmBody {
    #[serde(default)]
    confirm: bool,
    reason: Option<String>,
}

async fn rearm(State(s): State<AppState>, Path(scope): Path<String>, body: Bytes) -> Response {
    let body: RearmBody = match json_body(&body).map(serde_json::from_value) {
        Ok(Ok(b)) => b,
        Ok(Err(e)) => return err(StatusCode::BAD_REQUEST, format!("bad body: {e}")),
        Err(r) => return r,
    };
    let Some(family) = rearm_family(&scope) else {
        return err(
            StatusCode::BAD_REQUEST,
            format!("scope {scope:?} cannot be re-armed; families: {}", REARM_FAMILIES.join(", ")),
        );
    };
    if !body.confirm {
        audit(&s, "halt.rearm.refused", Some(&scope), json!({ "why": "confirm required" })).await;
        return (
            StatusCode::PRECONDITION_REQUIRED,
            Json(json!({ "error": "confirm_required", "detail": format!("re-arming {scope} needs {{\"confirm\": true}}") })),
        )
            .into_response();
    }
    let persisted = load_persisted(&s).await.map(|(_, _, st)| persisted_halts(&st));
    let known_live = s.live().is_halted(&scope);
    let known_persisted = persisted.as_ref().is_some_and(|p| p.iter().any(|h| h["scope"] == scope.as_str()));
    if persisted.is_some() && !known_live && !known_persisted {
        return err(StatusCode::NOT_FOUND, format!("no active halt with scope {scope}"));
    }
    let Some(client) = s.nats.as_ref().filter(|_| s.nats_connected()) else {
        return err(StatusCode::SERVICE_UNAVAILABLE, "NATS not connected");
    };
    let reason = body.reason.filter(|r| !r.trim().is_empty()).unwrap_or_else(|| "manual re-arm (operator UI)".into());
    let msg = Resume {
        scope: scope.clone(),
        reason: reason.clone(),
        at: Some(fa_proto::now()),
        manual: true,
        actor: ACTOR.into(),
    };
    let subject = fa_bus::subjects::risk_resume(&scope);
    if let Err(e) = client.publish(subject.clone(), Bytes::from(msg.encode_to_vec())).await {
        return err(StatusCode::BAD_GATEWAY, e.to_string());
    }
    if let Err(e) = client.flush().await {
        return err(StatusCode::BAD_GATEWAY, e.to_string());
    }
    s.live_mut().on_resume(&scope);
    let audited = audit(
        &s,
        "halt.rearm",
        Some(&scope),
        json!({ "family": family, "reason": reason, "subject": subject, "was_live": known_live, "was_persisted": known_persisted }),
    )
    .await;
    Json(json!({ "ok": true, "resumed": scope, "subject": subject, "audited": audited })).into_response()
}

#[derive(Debug, Default, Deserialize)]
pub struct EventsQuery {
    since: Option<String>,
    limit: Option<i64>,
    kind: Option<String>,
}

async fn risk_events(State(s): State<AppState>, Query(q): Query<EventsQuery>) -> Response {
    let Some(pool) = s.pg.get() else { return err(StatusCode::SERVICE_UNAVAILABLE, "postgres unavailable") };
    let since = match crate::ledger_routes::parse_since(q.since.as_deref().unwrap_or("24h"), Utc::now()) {
        Ok(t) => t,
        Err(e) => return err(StatusCode::BAD_REQUEST, e),
    };
    let limit = q.limit.unwrap_or(200).clamp(1, 2000);
    let kind = q.kind.as_deref().map(str::trim).filter(|k| !k.is_empty()).map(|k| format!("risk.event.{k}%"));
    let rows = sqlx::query(
        "SELECT id, at, actor, action, subject, payload FROM audit.log \
         WHERE ($1::timestamptz IS NULL OR at >= $1) \
           AND (action LIKE 'risk.event.%' OR action IN ('halt', 'resume', 'halt.rearm', 'halt.request', 'resume.request', 'reconciliation.diff')) \
           AND ($3::text IS NULL OR action LIKE $3) \
         ORDER BY at DESC, id DESC LIMIT $2",
    )
    .bind(since)
    .bind(limit)
    .bind(kind)
    .fetch_all(pool)
    .await;
    let counts = sqlx::query(
        "SELECT action, count(*) AS n FROM audit.log WHERE ($1::timestamptz IS NULL OR at >= $1) \
           AND (action LIKE 'risk.event.%' OR action IN ('halt', 'resume', 'halt.rearm', 'halt.request', 'resume.request', 'reconciliation.diff')) \
         GROUP BY action ORDER BY n DESC",
    )
    .bind(since)
    .fetch_all(pool)
    .await;
    match (rows, counts) {
        (Ok(rows), Ok(counts)) => {
            let events: Vec<Value> = rows
                .iter()
                .map(|r| {
                    let action: String = r.get("action");
                    json!({
                        "id": r.get::<i64, _>("id"),
                        "at": r.get::<DateTime<Utc>, _>("at"),
                        "actor": r.get::<String, _>("actor"),
                        "action": action,
                        "kind": action.strip_prefix("risk.event.").unwrap_or(&action),
                        "scope": r.get::<Option<String>, _>("subject"),
                        "detail": r.get::<Option<Value>, _>("payload"),
                    })
                })
                .collect();
            let counts: Map<String, Value> =
                counts.iter().map(|r| (r.get::<String, _>("action"), json!(r.get::<i64, _>("n")))).collect();
            Json(json!({ "since": since, "count": events.len(), "counts": counts, "events": events })).into_response()
        }
        (Err(e), _) | (_, Err(e)) => err(StatusCode::BAD_GATEWAY, format!("postgres: {e}")),
    }
}

// ------------------------------------------------------------------ refdata reads (D-067)

async fn venues(State(s): State<AppState>) -> Response {
    let Some(pool) = s.pg.get() else { return err(StatusCode::SERVICE_UNAVAILABLE, "postgres unavailable") };
    let rows = sqlx::query(
        "SELECT v.code, v.mic, v.name, v.timezone, v.kind, v.always_open, \
                min(c.date) AS first_day, max(c.date) AS last_day, count(c.date) AS days \
         FROM ref.venue v LEFT JOIN ref.calendar_session c ON c.venue_id = v.id \
         GROUP BY v.id ORDER BY v.code",
    )
    .fetch_all(pool)
    .await;
    match rows {
        Ok(rows) => {
            let venues: Vec<Value> = rows
                .iter()
                .map(|r| {
                    let code: String = r.get("code");
                    let mic: Option<String> = r.get("mic");
                    let always: bool = r.get("always_open");
                    json!({
                        "code": code,
                        "mic": mic,
                        "name": r.get::<String, _>("name"),
                        "timezone": r.get::<String, _>("timezone"),
                        "kind": r.get::<String, _>("kind"),
                        "always_open": always,
                        "rule": calendar::rule_for_venue(&code, mic.as_deref(), always),
                        "materialized": { "first": r.get::<Option<NaiveDate>, _>("first_day"), "last": r.get::<Option<NaiveDate>, _>("last_day"), "days": r.get::<i64, _>("days") },
                    })
                })
                .collect();
            Json(json!({ "venues": venues })).into_response()
        }
        Err(e) => err(StatusCode::BAD_GATEWAY, format!("postgres: {e}")),
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct DayRange {
    from: Option<NaiveDate>,
    to: Option<NaiveDate>,
}

async fn calendar_range(State(s): State<AppState>, Path(venue): Path<String>, Query(q): Query<DayRange>) -> Response {
    let Some(pool) = s.pg.get() else { return err(StatusCode::SERVICE_UNAVAILABLE, "postgres unavailable") };
    let today = Utc::now().date_naive();
    let from = q.from.unwrap_or_else(|| today.checked_sub_days(Days::new(3)).unwrap_or(today));
    let to = q.to.unwrap_or_else(|| from.checked_add_days(Days::new(13)).unwrap_or(from));
    if to < from || (to - from).num_days() > 92 {
        return err(StatusCode::BAD_REQUEST, "need from <= to and at most 93 days");
    }
    let v: Option<(Uuid, Option<String>, bool, String)> =
        sqlx::query_as("SELECT id, mic, always_open, timezone FROM ref.venue WHERE code = $1")
            .bind(&venue)
            .fetch_optional(pool)
            .await
            .unwrap_or(None);
    let Some((venue_id, mic, always, tz)) = v else {
        return err(StatusCode::NOT_FOUND, format!("unknown venue {venue}"));
    };
    let rule = calendar::rule_for_venue(&venue, mic.as_deref(), always);
    let stored = sqlx::query(
        "SELECT date, open, close, segments, is_holiday, recorded_at FROM ref.calendar_session \
         WHERE venue_id = $1 AND date BETWEEN $2 AND $3",
    )
    .bind(venue_id)
    .bind(from)
    .bind(to)
    .fetch_all(pool)
    .await
    .unwrap_or_default();
    let mut by_day: std::collections::BTreeMap<NaiveDate, Value> = stored
        .iter()
        .map(|r| {
            let d: NaiveDate = r.get("date");
            (
                d,
                json!({
                    "date": d,
                    "open": r.get::<Option<DateTime<Utc>>, _>("open"),
                    "close": r.get::<Option<DateTime<Utc>>, _>("close"),
                    "segments": r.get::<Value, _>("segments"),
                    "is_holiday": r.get::<bool, _>("is_holiday"),
                    "recorded_at": r.get::<DateTime<Utc>, _>("recorded_at"),
                    "basis": "table",
                }),
            )
        })
        .collect();
    let mut d = from;
    while d <= to {
        by_day.entry(d).or_insert_with(|| match rule {
            Some(r) => {
                let mut v = serde_json::to_value(calendar::session_for(r, d)).unwrap_or_default();
                v["basis"] = json!("rule");
                v
            }
            None => json!({ "date": d, "basis": "missing_calendar" }),
        });
        let Some(n) = d.succ_opt() else { break };
        d = n;
    }
    Json(json!({ "venue": venue, "timezone": tz, "rule": rule, "from": from, "to": to, "sessions": by_day.into_values().collect::<Vec<_>>() }))
        .into_response()
}

#[derive(Debug, Default, Deserialize)]
pub struct ActionsQuery {
    instrument_id: Option<String>,
    known_at: Option<String>,
    limit: Option<i64>,
}

async fn corporate_actions(State(s): State<AppState>, Query(q): Query<ActionsQuery>) -> Response {
    let Some(pool) = s.pg.get() else { return err(StatusCode::SERVICE_UNAVAILABLE, "postgres unavailable") };
    let known_at = match q.known_at.as_deref().map(str::trim).filter(|x| !x.is_empty()) {
        None => None,
        Some(x) => match DateTime::parse_from_rfc3339(x) {
            Ok(t) => Some(t.with_timezone(&Utc)),
            Err(_) => return err(StatusCode::BAD_REQUEST, "known_at must be RFC 3339"),
        },
    };
    if let Some(id) = q.instrument_id.as_deref().map(str::trim).filter(|x| !x.is_empty()) {
        let Ok(id) = Uuid::parse_str(id) else { return err(StatusCode::BAD_REQUEST, "instrument_id must be a UUID") };
        return match corpactions::list_as_of(pool, id, known_at).await {
            Ok(a) => Json(json!({ "instrument_id": id, "known_at": known_at, "point_in_time": true, "actions": a }))
                .into_response(),
            Err(e) => err(StatusCode::BAD_GATEWAY, format!("postgres: {e}")),
        };
    }
    let limit = q.limit.unwrap_or(100).clamp(1, 1000);
    let rows = sqlx::query(
        "SELECT a.id, a.instrument_id, i.name AS instrument_name, a.kind, a.ex_date, a.record_date, a.pay_date, \
                a.ratio::float8 AS ratio, a.amount::float8 AS amount, a.ccy, a.source_id, a.recorded_at, a.status, \
                a.cancelled_at, a.old_symbol, a.new_symbol \
         FROM ref.corporate_action a LEFT JOIN ref.instrument i ON i.id = a.instrument_id \
         ORDER BY a.recorded_at DESC LIMIT $1",
    )
    .bind(limit)
    .fetch_all(pool)
    .await;
    match rows {
        Ok(rows) => {
            let actions: Vec<Value> = rows
                .iter()
                .map(|r| {
                    json!({
                        "id": r.get::<Uuid, _>("id"),
                        "instrument_id": r.get::<Uuid, _>("instrument_id"),
                        "instrument_name": r.get::<Option<String>, _>("instrument_name"),
                        "kind": r.get::<String, _>("kind"),
                        "ex_date": r.get::<NaiveDate, _>("ex_date"),
                        "record_date": r.get::<Option<NaiveDate>, _>("record_date"),
                        "pay_date": r.get::<Option<NaiveDate>, _>("pay_date"),
                        "ratio": r.get::<Option<f64>, _>("ratio"),
                        "amount": r.get::<Option<f64>, _>("amount"),
                        "ccy": r.get::<Option<String>, _>("ccy"),
                        "source_id": r.get::<Option<String>, _>("source_id"),
                        "recorded_at": r.get::<DateTime<Utc>, _>("recorded_at"),
                        "status": r.get::<String, _>("status"),
                        "cancelled_at": r.get::<Option<DateTime<Utc>>, _>("cancelled_at"),
                        "old_symbol": r.get::<Option<String>, _>("old_symbol"),
                        "new_symbol": r.get::<Option<String>, _>("new_symbol"),
                    })
                })
                .collect();
            Json(json!({ "point_in_time": false, "actions": actions })).into_response()
        }
        Err(e) => err(StatusCode::BAD_GATEWAY, format!("postgres: {e}")),
    }
}

// ------------------------------------------------------------------ bus-tap decoders (D-066, D-069)

fn ts(t: &Option<fa_proto::Timestamp>) -> Value {
    t.as_ref().map_or(Value::Null, |t| {
        DateTime::from_timestamp(t.seconds, t.nanos.max(0) as u32)
            .map_or(Value::Null, |d| Value::String(d.to_rfc3339()))
    })
}

fn space(s: &Option<fa_proto::v1::ExpertSpaceRef>) -> Value {
    s.as_ref().map_or(
        Value::Null,
        |s| json!({ "snapshot_id": s.snapshot_id, "timescale": s.timescale, "expert_index": s.expert_index }),
    )
}

/// Decodes the D-066/D-069 subjects for `/ws/bus`; `None` leaves the subject to the caller.
pub fn decode_extra(subject: &str, payload: &[u8]) -> Option<Value> {
    if subject.starts_with("latent.membership.") {
        let m = RegionMembership::decode(payload).ok()?;
        return Some(json!({
            "type": "membership",
            "snapshot_id": m.snapshot_id,
            "timescale": m.timescale,
            "instrument_id": m.instrument_id,
            "as_of": ts(&m.as_of),
            "experts": m.experts.iter().map(|e| json!({
                "expert_index": e.expert_index,
                "novelty": e.novelty,
                "out_of_region": e.out_of_region,
                "weights": e.weights.iter().map(|w| json!({ "region_id": w.region_id, "weight": w.weight })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        }));
    }
    if subject.starts_with("latent.lifecycle.") {
        let e = RegionLifecycleEvent::decode(payload).ok()?;
        let kind = e.kind().as_str_name().trim_start_matches("KIND_").to_ascii_lowercase();
        return Some(json!({
            "type": "lifecycle",
            "kind": kind,
            "space": space(&e.space),
            "region_id": e.region_id,
            "lineage_id": e.lineage_id,
            "as_of": ts(&e.as_of),
            "ancestors": e.ancestors.iter().map(|a| json!({ "region_id": a.region_id, "lineage_id": a.lineage_id, "overlap": a.overlap, "kind": a.kind })).collect::<Vec<_>>(),
            "reason": e.reason,
            "mass": e.mass,
            "occupancy": e.occupancy,
            "related_region_id": e.related_region_id,
            "regime_break": e.regime_break,
        }));
    }
    if subject.starts_with("latent.mcmc.") {
        let k = KernelUpdate::decode(payload).ok()?;
        return Some(json!({
            "type": "kernel",
            "space": space(&k.space),
            "as_of": ts(&k.as_of),
            "serving_sampler_id": k.serving_sampler_id,
            "relaxation_steps": k.relaxation_steps,
            "observation_mass": k.observation_mass,
            "drift_kl": k.drift_kl,
            "refit_due": k.refit_due,
        }));
    }
    if subject.starts_with("latent.hier.") {
        let h = RegionHierarchy::decode(payload).ok()?;
        return Some(json!({
            "type": "hierarchy",
            "fine": space(&h.fine),
            "coarse": space(&h.coarse),
            "as_of": ts(&h.as_of),
            "mutual_information_bits": h.mutual_information_bits,
            "n": h.n,
        }));
    }
    if subject.starts_with("latent.crossprior.") {
        let c = CrossAssetPriors::decode(payload).ok()?;
        return Some(
            json!({ "type": "cross_priors", "timescale": c.timescale, "as_of": ts(&c.as_of), "proposals": c.proposals.len(), "instruments": c.instruments, "pairs_evaluated": c.pairs_evaluated }),
        );
    }
    // `ops.alert` and any other JSON payload (D-066 alert transitions are JSON)
    if payload.first() == Some(&b'{') {
        let mut v: Value = serde_json::from_slice(payload).ok()?;
        if subject == "ops.alert" && v.is_object() && v.get("type").is_none() {
            v["type"] = json!("alert");
        }
        return Some(v);
    }
    None
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    fn app() -> Router {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let cfg = fa_core::config::PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        let s = AppState::new(cfg, format!("{root}/sources.yaml").into(), None);
        // merged like main.rs: GET /api/refdata/corporate-actions next to refdata_routes' POST
        router().merge(crate::refdata_routes::router()).with_state(s)
    }

    async fn call(method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
        let req = Request::builder().method(method).uri(uri).header("content-type", "application/json");
        let req = req.body(body.map_or_else(Body::empty, |b| Body::from(b.to_string()))).unwrap();
        let res = app().oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    #[tokio::test]
    async fn offline_answers_are_explicit() {
        // re-arm: bad scope 400, missing confirm 428, confirmed without NATS 503 (nothing published)
        assert_eq!(
            call("POST", "/api/trading/halts/strategy:x/rearm", Some(json!({"confirm": true}))).await.0,
            StatusCode::BAD_REQUEST
        );
        let (st, v) = call("POST", "/api/trading/halts/broker/rearm", Some(json!({}))).await;
        assert_eq!(st, StatusCode::PRECONDITION_REQUIRED, "{v}");
        assert_eq!(v["error"], "confirm_required");
        assert_eq!(
            call("POST", "/api/trading/halts/source:yahoo/rearm", Some(json!({"confirm": true}))).await.0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        // ctl-backed routes without NATS
        for (m, uri) in [
            ("GET", "/api/decoders/serving"),
            ("POST", "/api/decoders/reload"),
            ("GET", "/api/trading/pipeline"),
            ("POST", "/api/trading/pipeline/run"),
            ("GET", "/api/trading/allocators"),
        ] {
            assert_eq!(call(m, uri, None).await.0, StatusCode::SERVICE_UNAVAILABLE, "{m} {uri}");
        }
        assert_eq!(call("POST", "/api/decoders/explode", None).await.0, StatusCode::NOT_FOUND);
        // Postgres-backed reads without a pool
        for uri in [
            "/api/trading/risk-events",
            "/api/refdata/venues",
            "/api/refdata/calendar/xetr",
            "/api/refdata/corporate-actions",
        ] {
            assert_eq!(call("GET", uri, None).await.0, StatusCode::SERVICE_UNAVAILABLE, "{uri}");
        }
        // the POST on the shared path still reaches refdata_routes (503 without Postgres, not 405)
        assert_eq!(call("POST", "/api/refdata/corporate-actions", Some(json!({}))).await.0.as_u16() / 100, 4);
        let (st, v) = call("GET", "/api/trading/halts", None).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(v["postgres"], false);
        assert!(v["halts"].as_array().unwrap().is_empty());
    }

    #[test]
    fn rearm_family_accepts_known_scopes_only() {
        for ok in
            ["portfolio", "daily_loss", "broker", "source:yahoo", "source", "model:stat-v1-1m", "fx", "venue:xetr"]
        {
            assert!(rearm_family(ok).is_some(), "{ok}");
        }
        for bad in ["", "strategy:x", "risk.halt.portfolio", "source.>", "port folio", "*"] {
            assert!(rearm_family(bad).is_none(), "{bad}");
        }
        assert_eq!(rearm_family("source:kraken"), Some("source"));
    }

    #[test]
    fn persisted_halts_reads_both_maps() {
        let st = json!({
            "halts": {"broker": {"manual": false, "reason": "diff", "since_ms": 1_794_365_673_685_i64}},
            "source_halts": {"yahoo": {"manual": false, "reason": "stale", "since_ms": 0}}
        });
        let rows = persisted_halts(&st);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["scope"], "broker");
        assert!(rows[0]["since"].as_str().unwrap().starts_with("2026-11-11"));
        assert_eq!(rows[1]["scope"], "source:yahoo");
        assert!(persisted_halts(&json!({})).is_empty());
    }

    #[test]
    fn reply_status_maps_prefixes_and_plain_messages() {
        assert_eq!(reply_status("not_found: x"), StatusCode::NOT_FOUND);
        assert_eq!(reply_status("invalid: pipeline action"), StatusCode::BAD_REQUEST);
        assert_eq!(reply_status("missing `family`"), StatusCode::BAD_REQUEST);
        assert_eq!(reply_status("`k` is not a loaded candidate"), StatusCode::NOT_FOUND);
        assert_eq!(reply_status("nn_forecast@1m: no shadow decoder to promote"), StatusCode::CONFLICT);
    }

    #[test]
    fn body_args_stringifies_scalars_and_drops_control_keys() {
        let a = body_args(
            &json!({"family": "nn_forecast", "timescale": "1m", "n": 3, "actor": "x", "confirm": true, "o": {}}),
        );
        assert!(a.contains(&("family".into(), "nn_forecast".into())));
        assert!(a.contains(&("n".into(), "3".into())));
        assert!(!a.iter().any(|(k, _)| k == "actor" || k == "confirm" || k == "o"));
    }

    #[test]
    fn decode_extra_handles_membership_lifecycle_and_json() {
        let m = RegionMembership {
            snapshot_id: "s".into(),
            timescale: "1m".into(),
            instrument_id: "i".into(),
            as_of: None,
            experts: vec![fa_proto::v1::ExpertMembership {
                expert_index: 1,
                weights: vec![fa_proto::v1::RegionWeight { region_id: "r1".into(), weight: 0.75 }],
                novelty: 0.5,
                out_of_region: false,
            }],
        };
        let v = decode_extra("latent.membership.1m.i", &m.encode_to_vec()).unwrap();
        assert_eq!(v["type"], "membership");
        assert_eq!(v["experts"][0]["weights"][0]["region_id"], "r1");
        let e =
            RegionLifecycleEvent { kind: 5, region_id: "r2".into(), reason: "support".into(), ..Default::default() };
        let v = decode_extra("latent.lifecycle.1m", &e.encode_to_vec()).unwrap();
        assert_eq!(v["kind"], "died");
        assert_eq!(v["reason"], "support");
        let v = decode_extra("ops.alert", br#"{"alertname":"Watchdog","status":"firing"}"#).unwrap();
        assert_eq!(v["type"], "alert");
        assert!(decode_extra("md.trade.x", b"\x0a\x01").is_none());
    }
}
