//! Ledger-backed routes (spec 50 §4.7, spec 42): `GET /api/trading/pnl`,
//! `GET /api/trading/reconciliation`, `GET /api/audit`, read from Postgres through `fa-ledger`.
//! Also seeds the in-memory paper book (`live.rs`, D-028) from the persisted `paper` book at
//! startup so a restart no longer resets it.
//!
//! The Postgres pool lives in `AppState::pg` (a `OnceLock` filled by [`install`]); the routes
//! answer 503 while it is unset. The router is layered in front of the main one with
//! `fallback_service`, so these three paths win over their `501 not_implemented` placeholders
//! in `routes.rs` without touching that file.
use std::time::Duration;

use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use fa_ledger::{Ledger, pg::PgLedger};
use serde::Deserialize;
use serde_json::json;
use sqlx::{PgPool, Row};

use crate::state::AppState;

/// Book the UI's paper views are seeded from (D-028, same account token as risk-engine).
pub const PAPER_BOOK: &str = "paper";

/// Connects to Postgres (if configured and reachable), publishes the pool in `state.pg`, seeds
/// the in-memory paper book, and returns `app` with the ledger routes layered in front.
pub async fn install(state: AppState, app: Router) -> Router {
    attach(&state).await;
    router(state).fallback_service(app)
}

/// The three ledger routes as a standalone router (used by [`install`] and tests).
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/api/trading/pnl", get(pnl))
        .route("/api/trading/reconciliation", get(reconciliation))
        .route("/api/audit", get(audit))
        .with_state(state)
}

/// Connects and seeds; logs and leaves `state.pg` unset when Postgres is unavailable.
pub async fn attach(state: &AppState) {
    let url = state.cfg.postgres.url.clone();
    if url.is_empty() {
        tracing::warn!("no [postgres] url: ledger routes disabled, paper book resets on restart (D-028)");
        return;
    }
    let connect =
        sqlx::postgres::PgPoolOptions::new().max_connections(4).acquire_timeout(Duration::from_secs(3)).connect(&url);
    let pool = match tokio::time::timeout(Duration::from_secs(5), connect).await {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "postgres unavailable: ledger routes disabled, paper book resets on restart");
            return;
        }
        Err(_) => {
            tracing::warn!("postgres connect timed out: ledger routes disabled, paper book resets on restart");
            return;
        }
    };
    match PgLedger::from_pool(pool.clone()).load_book(PAPER_BOOK).await {
        Ok(ledger) => {
            tracing::info!(
                cash_eur = ledger.cash_eur,
                positions = ledger.positions.values().filter(|p| p.quantity != 0.0).count(),
                "paper book seeded from postgres"
            );
            state.live_mut().seed_ledger(ledger);
        }
        Err(e) => tracing::warn!(error = %e, "paper book not seeded from postgres"),
    }
    let _ = state.pg.set(pool);
}

/// The pool, or the 503 the handler should return.
fn pool(s: &AppState) -> Result<PgPool, Box<Response>> {
    s.pg.get().cloned().ok_or_else(|| {
        Box::new((StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "postgres unavailable"}))).into_response())
    })
}

fn db_err(e: impl std::fmt::Display) -> Response {
    (StatusCode::BAD_GATEWAY, Json(json!({"error": format!("postgres: {e}")}))).into_response()
}

/// `since` / `range` grammar: RFC 3339 timestamp, or a relative window `<n>(m|h|d)` (`all` =
/// unbounded). Returns the lower bound.
pub fn parse_since(s: &str, now: DateTime<Utc>) -> Result<Option<DateTime<Utc>>, String> {
    let s = s.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("all") {
        return Ok(None);
    }
    if let Ok(t) = DateTime::parse_from_rfc3339(s) {
        return Ok(Some(t.with_timezone(&Utc)));
    }
    let (num, unit) = s.split_at(s.len() - 1);
    let n: i64 = num.parse().map_err(|_| format!("bad range {s:?}: use RFC 3339 or <n>m|h|d"))?;
    let d = match unit {
        "m" => ChronoDuration::minutes(n),
        "h" => ChronoDuration::hours(n),
        "d" => ChronoDuration::days(n),
        _ => return Err(format!("bad range unit in {s:?}: use m, h or d")),
    };
    Ok(Some(now - d))
}

#[derive(Debug, Deserialize)]
pub struct PnlQuery {
    pub range: Option<String>,
    pub strategy: Option<String>,
    pub book: Option<String>,
}

/// P&L from persisted fills (realised, FIFO-free average-cost as in `fa_ledger::Ledger`) plus
/// the latest position snapshot. `range` defaults to `1d`; `strategy` filters by attribution.
pub async fn pnl(State(s): State<AppState>, Query(q): Query<PnlQuery>) -> Response {
    let pool = match pool(&s) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let now = Utc::now();
    let from = match parse_since(q.range.as_deref().unwrap_or("1d"), now) {
        Ok(t) => t,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": e}))).into_response(),
    };
    let book = q.book.as_deref().unwrap_or(PAPER_BOOK);
    let pg = PgLedger::from_pool(pool);
    let fills = match pg.fills(book, from, None, q.strategy.as_deref()).await {
        Ok(f) => f,
        Err(e) => return db_err(e),
    };
    let snapshot = match pg.latest_snapshot(book).await {
        Ok(v) => v,
        Err(e) => return db_err(e),
    };
    let mut replay = Ledger::with_cash(0.0);
    let mut turnover = fa_ledger::Decimal::ZERO;
    for f in &fills {
        turnover += fa_ledger::money::mul(f.qty, fa_ledger::money::mul(f.price, f.fx_rate_eur));
        replay.apply_fill(&f.to_fill());
    }
    let turnover = fa_ledger::money::f(turnover);
    let realised: f64 =
        fa_ledger::money::f(fa_ledger::money::sum(replay.exact().positions.values().map(|p| p.realised_eur)));
    let by_instrument: Vec<_> = replay
        .positions
        .iter()
        .map(|(i, p)| {
            json!({
                "instrument_id": i, "net_quantity_in_range": p.quantity,
                "realised_eur": p.realised_eur, "avg_cost_eur": p.avg_cost_eur,
            })
        })
        .collect();
    Json(json!({
        "book": book,
        "strategy": q.strategy,
        "range": {"from": from, "to": now},
        "fills": fills.len(),
        "turnover_eur": turnover,
        "realised_eur": realised,
        "fees_eur": replay.fees_eur,
        "net_eur": fa_ledger::money::f(fa_ledger::money::sum(replay.exact().positions.values().map(|p| p.realised_eur)) - replay.exact().fees_eur),
        "by_instrument": by_instrument,
        "snapshot": snapshot,
        "note": "realised P&L over fills in range with average-cost lots; unrealised needs marks (not persisted yet)",
    }))
    .into_response()
}

#[derive(Debug, Deserialize)]
pub struct LimitQuery {
    pub limit: Option<i64>,
    pub since: Option<String>,
}

/// Latest `reconciliation_run` rows (newest first) with their diffs.
pub async fn reconciliation(State(s): State<AppState>, Query(q): Query<LimitQuery>) -> Response {
    let pool = match pool(&s) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let limit = q.limit.unwrap_or(20).clamp(1, 500);
    match PgLedger::from_pool(pool).reconciliation_runs(limit).await {
        Ok(runs) => {
            let latest = runs.first().map(|r| r.status.clone());
            Json(json!({"latest_status": latest, "runs": runs})).into_response()
        }
        Err(e) => db_err(e),
    }
}

/// `audit.log` entries since `since` (default `24h`), oldest first.
pub async fn audit(State(s): State<AppState>, Query(q): Query<LimitQuery>) -> Response {
    let pool = match pool(&s) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let since = match parse_since(q.since.as_deref().unwrap_or("24h"), Utc::now()) {
        Ok(t) => t,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": e}))).into_response(),
    };
    let limit = q.limit.unwrap_or(200).clamp(1, 2000);
    let rows = sqlx::query(
        "SELECT id, at, actor, action, subject, payload FROM audit.log \
         WHERE ($1::timestamptz IS NULL OR at >= $1) ORDER BY at, id LIMIT $2",
    )
    .bind(since)
    .bind(limit)
    .fetch_all(&pool)
    .await;
    match rows {
        Ok(rows) => {
            let entries: Vec<_> = rows
                .iter()
                .map(|r| {
                    json!({
                        "id": r.get::<i64, _>("id"),
                        "at": r.get::<DateTime<Utc>, _>("at"),
                        "actor": r.get::<String, _>("actor"),
                        "action": r.get::<String, _>("action"),
                        "subject": r.get::<Option<String>, _>("subject"),
                        "payload": r.get::<Option<serde_json::Value>, _>("payload"),
                    })
                })
                .collect();
            Json(json!({"since": since, "count": entries.len(), "entries": entries})).into_response()
        }
        Err(e) => db_err(e),
    }
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use http_body_util::BodyExt;
    use serde_json::Value;
    use tower::ServiceExt;

    use super::*;

    fn state() -> AppState {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let cfg = fa_core::config::PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        AppState::new(cfg, format!("{root}/sources.yaml").into(), None)
    }

    async fn get_json(app: Router, uri: &str) -> (StatusCode, Value) {
        let res = app.oneshot(axum::http::Request::builder().uri(uri).body(Body::empty()).unwrap()).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    #[test]
    fn since_grammar() {
        let now = Utc::now();
        assert_eq!(parse_since("all", now).unwrap(), None);
        assert_eq!(parse_since("2h", now).unwrap(), Some(now - ChronoDuration::hours(2)));
        assert_eq!(parse_since("7d", now).unwrap(), Some(now - ChronoDuration::days(7)));
        assert_eq!(
            parse_since("2026-01-01T00:00:00Z", now).unwrap().map(|t| t.to_rfc3339()),
            Some("2026-01-01T00:00:00+00:00".into())
        );
        assert!(parse_since("yesterday", now).is_err());
        assert!(parse_since("3w", now).is_err());
    }

    #[tokio::test]
    async fn routes_answer_503_without_postgres_and_shadow_the_placeholders() {
        let s = state();
        let app = router(s.clone()).fallback_service(crate::routes::router(s, None));
        for uri in ["/api/trading/pnl", "/api/trading/reconciliation", "/api/audit?since=1h"] {
            let (status, v) = get_json(app.clone(), uri).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{uri}");
            assert_eq!(v["error"], "postgres unavailable");
        }
        // everything else still reaches the main router
        let (status, _) = get_json(app.clone(), "/api/trading/positions").await;
        assert_eq!(status, StatusCode::OK);
        let (status, v) = get_json(app, "/api/trading/strategies").await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(v["spec"], "bins/control-api/README.md (ex spec 50 §4.7)");
    }

    /// Needs `DATABASE_URL` (compose Postgres with migrations applied).
    #[tokio::test]
    async fn routes_read_the_live_ledger() {
        let Ok(url) = std::env::var("DATABASE_URL") else { return };
        let s = state();
        let pool = sqlx::postgres::PgPoolOptions::new().max_connections(2).connect(&url).await.unwrap();
        s.pg.set(pool.clone()).unwrap();
        let app = router(s);
        let marker = format!("test-{}", uuid::Uuid::now_v7());
        let id = fa_warehouse_free_append(&pool, &marker).await;

        let (status, v) = get_json(app.clone(), "/api/audit?since=1h&limit=2000").await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert!(v["entries"].as_array().unwrap().iter().any(|e| e["subject"] == marker), "{v}");

        let (status, v) = get_json(app.clone(), "/api/trading/pnl?range=7d&book=no-such-book").await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["fills"], 0);
        assert_eq!(v["realised_eur"], 0.0);
        let (status, _) = get_json(app.clone(), "/api/trading/pnl?range=bogus").await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let (status, v) = get_json(app, "/api/trading/reconciliation?limit=5").await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert!(v["runs"].is_array());

        sqlx::query("DELETE FROM audit.log WHERE id = $1").bind(id).execute(&pool).await.unwrap();
    }

    /// control-api has no `fa-warehouse` dependency (its Cargo.toml is shared), so the test
    /// writes the audit row directly.
    async fn fa_warehouse_free_append(pool: &PgPool, subject: &str) -> i64 {
        let (id,): (i64,) = sqlx::query_as(
            "INSERT INTO audit.log (actor, action, subject, payload) VALUES ('ui', 'test', $1, '{}') RETURNING id",
        )
        .bind(subject)
        .fetch_one(pool)
        .await
        .unwrap();
        id
    }
}
