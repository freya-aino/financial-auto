//! Organisational-books API (D-104): balances and segment funds, the journal, and a manual
//! close. Reads come straight from the `acct` schema; the close shares the scheduled job's
//! code path, so a manual close of an already-closed day posts nothing twice.
use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use fa_accounting::{money, pg::PgAccounting};
use serde_json::json;

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/accounting/balances", get(balances))
        .route("/api/accounting/entries", get(entries))
        .route("/api/accounting/close", post(close))
}

fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(json!({ "error": msg.into() }))).into_response()
}

macro_rules! pool_or_return {
    ($s:expr) => {
        match $s.pg.get() {
            Some(p) => p.clone(),
            None => return err(StatusCode::SERVICE_UNAVAILABLE, "postgres not ready"),
        }
    };
}

async fn balances(State(s): State<AppState>) -> Response {
    let acct = PgAccounting::from_pool(pool_or_return!(s));
    match acct.load().await {
        Ok(j) => {
            let accounts: serde_json::Map<String, serde_json::Value> = j
                .balances()
                .iter()
                .map(|(id, b)| {
                    let a = j.chart().get(id);
                    (
                        id.clone(),
                        json!({
                            "normal_eur": money::f(b.normal),
                            "kind": a.map(|a| a.kind.as_str()),
                            "segment": a.map(|a| a.segment.clone()),
                            "external_ref": a.and_then(|a| a.external_ref.clone()),
                        }),
                    )
                })
                .collect();
            let segments: serde_json::Map<String, serde_json::Value> =
                j.segment_funds().iter().map(|(seg, eur)| (seg.clone(), json!(money::f(*eur)))).collect();
            Json(json!({
                "accounts": accounts,
                "segment_funds_eur": segments,
                "trial_balance": money::f(j.trial_balance()),
                "entries": j.entries().len(),
            }))
            .into_response()
        }
        Err(e) => err(StatusCode::BAD_GATEWAY, e.to_string()),
    }
}

#[derive(serde::Deserialize)]
struct EntriesQuery {
    limit: Option<i64>,
}

async fn entries(State(s): State<AppState>, Query(q): Query<EntriesQuery>) -> Response {
    let acct = PgAccounting::from_pool(pool_or_return!(s));
    match acct.recent_entries(q.limit.unwrap_or(100)).await {
        Ok(es) => Json(json!({ "count": es.len(), "entries": es })).into_response(),
        Err(e) => err(StatusCode::BAD_GATEWAY, e.to_string()),
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
struct CloseBody {
    /// UTC day to close; default yesterday.
    day: Option<chrono::NaiveDate>,
}

async fn close(State(s): State<AppState>, Json(b): Json<CloseBody>) -> Response {
    let pool = pool_or_return!(s);
    let day = b.day.unwrap_or_else(|| (chrono::Utc::now() - chrono::Duration::days(1)).date_naive());
    if day >= chrono::Utc::now().date_naive() {
        return err(StatusCode::BAD_REQUEST, "only a finished UTC day can be closed");
    }
    match crate::accounting::close_day(&s, day).await {
        Ok(report) => {
            if let Err(e) = crate::audit::append(
                &pool,
                "operator",
                "ops.accounting_close",
                None,
                json!({ "ok": true, "report": report }),
            )
            .await
            {
                tracing::warn!(error = %e, "audit append failed");
            }
            Json(json!({ "ok": true, "report": report })).into_response()
        }
        Err(e) => err(StatusCode::BAD_GATEWAY, e),
    }
}
