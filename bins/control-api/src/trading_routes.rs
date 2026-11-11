//! Strategy lifecycle and limit routes (spec 50 §4.7, spec 40 §4.4; D-051):
//! `GET /api/trading/strategies`, `POST /api/trading/strategies/{id}/{pause|resume|flatten|promote|shadow}`,
//! `GET|PUT /api/trading/limits`, `POST /api/trading/flatten`. Commands are request/reply to the
//! risk-engine (`ctl.risk.strategy|limits|flatten`, `CtlRequest` → `CtlReply` whose message is
//! JSON); `CtlReply{ok:false}` is mapped by prefix (`not_found:` 404, `conflict:` 409,
//! `invalid:` 400, anything else — e.g. `loosen:` from a PUT that would loosen a limit — 422).
//!
//! The strategy list prefers the risk-engine's live view (fresh shadow progress + gate) and
//! falls back to `ledger.strategy_state` in Postgres when the engine does not answer; either
//! way it is joined with the latest `signal.exposure.*` seen in `Live`. Layered in front of the
//! main router with `fallback_service` (like `ledger_routes`) so the `501` placeholders in
//! `routes.rs` stay untouched.
use std::{collections::BTreeMap, time::Duration};

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use bytes::Bytes;
use fa_bus::subjects;
use fa_proto::{
    Message,
    v1::{CtlReply, CtlRequest},
};
use serde_json::{Value, json};
use sqlx::Row;

use crate::state::AppState;

const CTL_TIMEOUT: Duration = Duration::from_secs(5);
const ACTIONS: [&str; 7] = ["pause", "resume", "flatten", "promote", "shadow", "retire", "admit"];

/// Returns `app` with the trading routes layered in front.
pub fn install(state: AppState, app: Router) -> Router {
    router(state).fallback_service(app)
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/api/trading/strategies", get(strategies))
        .route("/api/trading/strategies/{id}/{action}", post(strategy_action))
        .route("/api/trading/limits", get(limits_get).put(limits_put))
        .route("/api/trading/flatten", post(flatten))
        .with_state(state)
}

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({ "error": msg.into() }))).into_response()
}

enum CtlError {
    Unavailable(String),
    Gateway(String),
    Timeout(String),
}

impl CtlError {
    fn into_response(self) -> Response {
        match self {
            Self::Unavailable(m) => err(StatusCode::SERVICE_UNAVAILABLE, m),
            Self::Gateway(m) => err(StatusCode::BAD_GATEWAY, m),
            Self::Timeout(m) => err(StatusCode::GATEWAY_TIMEOUT, m),
        }
    }
}

async fn ctl(s: &AppState, verb: &str, args: Vec<(&str, String)>) -> Result<CtlReply, CtlError> {
    let Some(client) = s.nats.as_ref().filter(|_| s.nats_connected()) else {
        return Err(CtlError::Unavailable("NATS not connected".into()));
    };
    let subject = subjects::ctl("risk", verb);
    let req = CtlRequest {
        verb: verb.into(),
        args: args.into_iter().map(|(k, v)| (k.to_owned(), v)).chain([("actor".to_owned(), "ui".to_owned())]).collect(),
    };
    let fut = client.request(subject.clone(), Bytes::from(req.encode_to_vec()));
    match tokio::time::timeout(CTL_TIMEOUT, fut).await {
        Ok(Ok(msg)) => CtlReply::decode(msg.payload.as_ref())
            .map_err(|e| CtlError::Gateway(format!("bad reply on {subject}: {e}"))),
        Ok(Err(e)) => Err(CtlError::Gateway(format!("{subject}: {e} (is risk-engine running?)"))),
        Err(_) => Err(CtlError::Timeout(format!("{subject}: no reply within {CTL_TIMEOUT:?}"))),
    }
}

/// Maps `CtlReply{ok:false}` by message prefix (same convention as the Go services).
fn reply_status(message: &str) -> StatusCode {
    match message.split(':').next().unwrap_or_default() {
        "not_found" => StatusCode::NOT_FOUND,
        "conflict" => StatusCode::CONFLICT,
        "invalid" => StatusCode::BAD_REQUEST,
        _ => StatusCode::UNPROCESSABLE_ENTITY,
    }
}

/// The reply message as JSON (the risk-engine answers with JSON), or the raw string.
fn message_json(m: &str) -> Value {
    serde_json::from_str(m).unwrap_or_else(|_| Value::String(m.to_owned()))
}

fn reply_response(r: CtlReply, wrap: impl FnOnce(Value) -> Value) -> Response {
    if r.ok { Json(wrap(message_json(&r.message))).into_response() } else { err(reply_status(&r.message), r.message) }
}

// ----------------------------------------------------------------- strategies

/// Latest `signal.exposure.*` per strategy from the live aggregator.
fn live_signals(s: &AppState) -> BTreeMap<String, Value> {
    let view = s.live().signals_view();
    view["signals"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v["strategy_id"].as_str().map(|id| (id.to_owned(), v.clone())))
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default()
}

async fn strategies_from_postgres(s: &AppState) -> Result<Vec<Value>, String> {
    let pool = s.pg.get().ok_or_else(|| "postgres unavailable".to_owned())?;
    let rows = sqlx::query(
        "SELECT strategy_id, status, prev_status, since, first_seen, shadow_since, horizons_completed, \
         sleeve_eur::float8 AS sleeve_eur, pnl_eur::float8 AS pnl_eur, peak_pnl_eur::float8 AS peak_pnl_eur, \
         max_drawdown::float8 AS max_drawdown, gate, updated_at FROM ledger.strategy_state ORDER BY strategy_id",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| format!("postgres: {e}"))?;
    Ok(rows
        .iter()
        .map(|r| {
            let shadow_since: Option<chrono::DateTime<chrono::Utc>> = r.get("shadow_since");
            let (sleeve, pnl, peak): (f64, f64, f64) = (r.get("sleeve_eur"), r.get("pnl_eur"), r.get("max_drawdown"));
            let drawdown = if sleeve > 0.0 { ((r.get::<f64, _>("peak_pnl_eur") - pnl) / sleeve).max(0.0) } else { 0.0 };
            json!({
                "strategy_id": r.get::<String, _>("strategy_id"),
                "status": r.get::<String, _>("status"),
                "prev_status": r.get::<Option<String>, _>("prev_status"),
                "since": r.get::<chrono::DateTime<chrono::Utc>, _>("since"),
                "first_seen": r.get::<chrono::DateTime<chrono::Utc>, _>("first_seen"),
                "sleeve_eur": sleeve,
                "pnl_eur": pnl,
                "peak_pnl_eur": r.get::<f64, _>("peak_pnl_eur"),
                "drawdown": drawdown,
                "max_drawdown": peak,
                "shadow": {
                    "since": shadow_since,
                    "days": shadow_since.map(|t| (chrono::Utc::now() - t).num_milliseconds() as f64 / 86_400_000.0),
                    "horizons_completed": r.get::<i64, _>("horizons_completed"),
                },
                "gate": r.get::<Option<Value>, _>("gate"),
                "updated_at": r.get::<chrono::DateTime<chrono::Utc>, _>("updated_at"),
            })
        })
        .collect())
}

/// Status, sleeve, shadow progress and gate checklist per strategy, joined with its latest signal.
pub async fn strategies(State(s): State<AppState>) -> Response {
    let (source, rows): (&str, Vec<Value>) = match ctl(&s, "strategy", vec![("action", "list".into())]).await {
        Ok(r) if r.ok => {
            ("risk-engine", message_json(&r.message)["strategies"].as_array().cloned().unwrap_or_default())
        }
        Ok(r) => return err(reply_status(&r.message), r.message),
        Err(_) => match strategies_from_postgres(&s).await {
            Ok(rows) => ("postgres", rows),
            Err(e) => return err(StatusCode::SERVICE_UNAVAILABLE, format!("risk-engine not answering and {e}")),
        },
    };
    let mut signals = live_signals(&s);
    let strategies: Vec<Value> = rows
        .into_iter()
        .map(|mut row| {
            let id = row["strategy_id"].as_str().unwrap_or_default().to_owned();
            row["signal"] = signals.remove(&id).unwrap_or(Value::Null);
            row
        })
        .collect();
    let counts = strategies.iter().fold(BTreeMap::<String, usize>::new(), |mut m, r| {
        *m.entry(r["status"].as_str().unwrap_or("unknown").to_owned()).or_default() += 1;
        m
    });
    Json(json!({
        "source": source,
        "count": strategies.len(),
        "status_counts": counts,
        "strategies": strategies,
        // signals seen on the bus for strategies the risk-engine has no state for (yet)
        "signals_without_state": signals.into_values().collect::<Vec<_>>(),
    }))
    .into_response()
}

/// `pause|resume|flatten|promote|shadow` via `ctl.risk.strategy`.
pub async fn strategy_action(State(s): State<AppState>, Path((id, action)): Path<(String, String)>) -> Response {
    if !ACTIONS.contains(&action.as_str()) {
        return err(StatusCode::NOT_FOUND, format!("unknown action {action:?}; use {}", ACTIONS.join("|")));
    }
    match ctl(&s, "strategy", vec![("strategy_id", id.clone()), ("action", action.clone())]).await {
        Ok(r) => {
            reply_response(r, |state| json!({ "ok": true, "strategy_id": id, "action": action, "strategy": state }))
        }
        Err(e) => e.into_response(),
    }
}

// --------------------------------------------------------------------- limits

/// Current hard limits as the risk-engine holds them.
pub async fn limits_get(State(s): State<AppState>) -> Response {
    match ctl(&s, "limits", Vec::new()).await {
        Ok(r) => reply_response(r, |limits| json!({ "limits": limits, "note": "PUT may only tighten (spec 40 §6.5)" })),
        Err(e) => e.into_response(),
    }
}

/// Tighten limits: the body is a partial limits object (`leverage_cap` keys merge). A change
/// that would loosen any field is refused with 422 and the offending fields.
pub async fn limits_put(State(s): State<AppState>, Json(body): Json<Value>) -> Response {
    if !body.is_object() || body.as_object().is_some_and(|o| o.is_empty()) {
        return err(StatusCode::BAD_REQUEST, "body must be a non-empty JSON object of limit fields");
    }
    match ctl(&s, "limits", vec![("set", body.to_string())]).await {
        Ok(r) => reply_response(r, |limits| json!({ "ok": true, "limits": limits })),
        Err(e) => e.into_response(),
    }
}

// -------------------------------------------------------------------- flatten

/// Portfolio flatten without a halt: every active strategy is flattened and paused.
pub async fn flatten(State(s): State<AppState>) -> Response {
    match ctl(&s, "flatten", Vec::new()).await {
        Ok(r) => reply_response(r, |v| v),
        Err(e) => e.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    fn state() -> AppState {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let cfg = fa_core::config::PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        AppState::new(cfg, format!("{root}/sources.yaml").into(), None)
    }

    async fn call(app: Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
        let req = Request::builder().method(method).uri(uri).header("content-type", "application/json");
        let req = req.body(body.map_or_else(Body::empty, |b| Body::from(b.to_string()))).unwrap();
        let res = app.oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    #[test]
    fn reply_prefixes_map_to_statuses() {
        assert_eq!(reply_status("not_found: x"), StatusCode::NOT_FOUND);
        assert_eq!(reply_status("conflict: gate"), StatusCode::CONFLICT);
        assert_eq!(reply_status("invalid: bad"), StatusCode::BAD_REQUEST);
        assert_eq!(reply_status("loosen: max_drawdown 0.1 → 0.5"), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(message_json("{\"a\":1}")["a"], 1);
        assert_eq!(message_json("plain"), Value::String("plain".into()));
    }

    #[tokio::test]
    async fn without_nats_commands_are_503_and_bad_input_is_rejected_locally() {
        let s = state();
        let app = install(s.clone(), crate::routes::router(s, None));
        for (m, uri, body) in [
            ("POST", "/api/trading/strategies/baseline-momentum-1s/pause", None),
            ("POST", "/api/trading/flatten", None),
            ("GET", "/api/trading/limits", None),
            ("PUT", "/api/trading/limits", Some(json!({"max_drawdown": 0.05}))),
            ("GET", "/api/trading/strategies", None),
        ] {
            let (status, v) = call(app.clone(), m, uri, body).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{m} {uri}: {v}");
        }
        let (status, v) = call(app.clone(), "POST", "/api/trading/strategies/x/kill", None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(v["error"].as_str().unwrap().contains("pause|resume|flatten|promote|shadow"));
        let (status, _) = call(app.clone(), "PUT", "/api/trading/limits", Some(json!({}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _) = call(app.clone(), "PUT", "/api/trading/limits", Some(json!([1]))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        // everything else still reaches the main router
        let (status, _) = call(app, "GET", "/api/trading/positions", None).await;
        assert_eq!(status, StatusCode::OK);
    }

    /// Needs `DATABASE_URL` (compose Postgres with migration 20261110000020 applied): the
    /// strategy list falls back to `ledger.strategy_state` when the risk-engine does not answer.
    #[tokio::test]
    async fn strategies_fall_back_to_postgres_rows() {
        let Ok(url) = std::env::var("DATABASE_URL") else { return };
        let s = state();
        let pool = sqlx::postgres::PgPoolOptions::new().max_connections(2).connect(&url).await.unwrap();
        s.pg.set(pool.clone()).unwrap();
        let id = format!("test-{}", uuid::Uuid::now_v7());
        sqlx::query(
            "INSERT INTO ledger.strategy_state (strategy_id, status, shadow_since, horizons_completed, sleeve_eur, pnl_eur, peak_pnl_eur, max_drawdown, gate) \
             VALUES ($1, 'shadow', now() - interval '2 days', 7, 10000, -50, 25, 0.0075, '{\"green\": false, \"missing\": [\"min_shadow_days\"]}')",
        )
        .bind(&id)
        .execute(&pool)
        .await
        .unwrap();
        let (status, v) = call(router(s), "GET", "/api/trading/strategies", None).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["source"], "postgres");
        let row = v["strategies"].as_array().unwrap().iter().find(|r| r["strategy_id"] == id).expect("row listed");
        assert_eq!(row["status"], "shadow");
        assert_eq!(row["shadow"]["horizons_completed"], 7);
        assert!(row["shadow"]["days"].as_f64().unwrap() >= 2.0);
        assert!((row["drawdown"].as_f64().unwrap() - 0.0075).abs() < 1e-12);
        assert_eq!(row["gate"]["green"], false);
        assert!(row["signal"].is_null(), "no live signal for a test strategy");
        assert!(v["status_counts"]["shadow"].as_u64().unwrap() >= 1);
        sqlx::query("DELETE FROM ledger.strategy_state WHERE strategy_id = $1").bind(&id).execute(&pool).await.unwrap();
    }
}
