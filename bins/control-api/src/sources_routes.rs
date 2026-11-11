//! Sources, ingestion and reference-data routes (spec 50 §4.2; D-054).
//!
//! * `GET /api/sources`, `GET /api/sources/{id}`: the YAML catalog with the Postgres status
//!   overlaid (`catalog.source` wins once synced, spec 11 §8), the ingestor's latest
//!   `ingest.health.<source>` frame, cursors, recent runs/errors, gaps and unresolved symbols.
//! * `POST /api/sources/{id}/enable|disable|backfill`: `ctl.ingest.*` requests to the ingestor;
//!   `backfill` enqueues a `jobs.ingest.backfill` job and returns its id (watch it on
//!   `GET /api/jobs?kind=ingest.backfill`). Each mutation is audited when Postgres is up.
//! * `GET /api/instruments?q=`: `fa_warehouse::refdata::search_instruments`.
//! * `GET /api/instruments/quarantine`, `POST /api/instruments/quarantine/{id}/resolve`: the
//!   review loop for provisional instruments (`{id}` = provisional instrument UUID).
//!
//! Postgres-backed routes answer 503 while `AppState::pg` is unset; bus commands answer 503
//! without NATS and 504 when the ingestor does not reply within 5 s.
use std::time::Duration;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use fa_bus::subjects;
use fa_proto::{
    Message,
    v1::{CtlReply, CtlRequest},
};
use fa_warehouse::refdata::{self, Resolution, ResolveError};
use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::{sources, state::AppState};

const CTL_TIMEOUT: Duration = Duration::from_secs(5);
/// Audit actor for operator actions through this API (spec 42 §6).
const ACTOR: &str = "ui";

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/sources", get(list_sources))
        .route("/api/sources/{id}", get(get_source))
        .route("/api/sources/{id}/{action}", post(source_action))
        .route("/api/instruments", get(instruments))
        .route("/api/instruments/quarantine", get(quarantine))
        .route("/api/instruments/quarantine/{id}/resolve", post(quarantine_resolve))
}

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({ "error": msg.into() }))).into_response()
}

fn db_err(e: impl std::fmt::Display) -> Response {
    err(StatusCode::BAD_GATEWAY, format!("postgres: {e}"))
}

/// The pool, or the boxed 503 the handler should return (boxed: `Response` is large).
fn pool(s: &AppState) -> Result<PgPool, Box<Response>> {
    s.pg.get().cloned().ok_or_else(|| Box::new(err(StatusCode::SERVICE_UNAVAILABLE, "postgres unavailable")))
}

// ------------------------------------------------------------------ catalog

/// `catalog.source` status per id (empty without Postgres).
async fn db_statuses(s: &AppState) -> (bool, std::collections::BTreeMap<String, (String, DateTime<Utc>)>) {
    let Some(pool) = s.pg.get() else { return (false, Default::default()) };
    match sqlx::query("SELECT id, status, updated_at FROM catalog.source").fetch_all(pool).await {
        Ok(rows) => (
            true,
            rows.iter()
                .map(|r| {
                    (r.get::<String, _>("id"), (r.get::<String, _>("status"), r.get::<DateTime<Utc>, _>("updated_at")))
                })
                .collect(),
        ),
        Err(e) => {
            tracing::warn!(error = %e, "catalog.source read failed; serving yaml statuses");
            (false, Default::default())
        }
    }
}

/// Catalog entry as JSON with the DB status overlaid and the live health frame attached.
fn source_json(src: &sources::Source, db: Option<&(String, DateTime<Utc>)>, health: &sources::HealthStore) -> Value {
    let mut v = serde_json::to_value(src).unwrap_or_default();
    v["yaml_status"] = json!(src.status);
    if let Some((status, at)) = db {
        v["status"] = json!(status);
        v["status_updated_at"] = json!(at);
    }
    v["running"] = json!(health.running(&src.id).unwrap_or(false));
    v["health"] = health.health(&src.id).cloned().unwrap_or(Value::Null);
    v
}

async fn list_sources(State(s): State<AppState>) -> Response {
    let cat = match sources::load(&s.sources_path) {
        Ok(c) => c,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let (from_db, statuses) = db_statuses(&s).await;
    let health = sources::health().read().expect("health lock");
    let list: Vec<Value> = cat.sources.iter().map(|src| source_json(src, statuses.get(&src.id), &health)).collect();
    // Sources only known to the ingestor (the synthetic heartbeat) still get a row.
    let extra: Vec<Value> = health
        .all_health()
        .iter()
        .filter(|(id, _)| !cat.sources.iter().any(|s| &s.id == *id))
        .map(|(id, h)| json!({ "id": id, "name": id, "category": "synthetic", "status": "live", "yaml_status": null,
            "capabilities": ["data"], "asset_classes": [], "transport": [], "auth": "none", "tos_notes": "", "notes": "not in config/sources.yaml",
            "running": h.get("running").and_then(Value::as_bool).unwrap_or(false), "health": h }))
        .collect();
    Json(json!({
        "version": cat.version,
        "sources": list.into_iter().chain(extra).collect::<Vec<_>>(),
        "status_from_postgres": from_db,
        "health_frames_seen": health.frames_seen,
    }))
    .into_response()
}

#[derive(Debug, Deserialize)]
pub struct DetailQuery {
    pub limit: Option<i64>,
}

async fn get_source(State(s): State<AppState>, Path(id): Path<String>, Query(q): Query<DetailQuery>) -> Response {
    let cat = match sources::load(&s.sources_path) {
        Ok(c) => c,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let (from_db, statuses) = db_statuses(&s).await;
    let (mut v, gaps, unresolved) = {
        let health = sources::health().read().expect("health lock");
        let v = match cat.sources.iter().find(|x| x.id == id) {
            Some(src) => source_json(src, statuses.get(&id), &health),
            None if health.health(&id).is_some() => {
                json!({ "id": id, "name": id, "category": "synthetic", "status": "live",
                "running": health.running(&id).unwrap_or(false), "health": health.health(&id), "notes": "not in config/sources.yaml" })
            }
            None => return err(StatusCode::NOT_FOUND, format!("unknown source {id}")),
        };
        (v, health.gaps(&id), health.unresolved(&id))
    };
    v["status_from_postgres"] = json!(from_db);
    v["gaps"] = json!(gaps);
    v["unresolved"] = json!(unresolved);
    let limit = q.limit.unwrap_or(20).clamp(1, 200);
    if let Some(pool) = s.pg.get() {
        match detail_from_db(pool, &id, limit).await {
            Ok(d) => {
                for (k, val) in d {
                    v[k] = val;
                }
            }
            Err(e) => return db_err(e),
        }
    }
    Json(v).into_response()
}

async fn detail_from_db(pool: &PgPool, id: &str, limit: i64) -> Result<Vec<(&'static str, Value)>, sqlx::Error> {
    let cursors = sqlx::query(
        "SELECT instrument_id, timescale, range_start, range_end, cursor, status, updated_at, \
                range_start = to_timestamp(0) AS live \
           FROM ingest.backfill_cursor WHERE source_id = $1 ORDER BY updated_at DESC LIMIT $2",
    )
    .bind(id)
    .bind(limit)
    .fetch_all(pool)
    .await?
    .iter()
    .map(|r| {
        json!({
            "instrument_id": r.get::<Option<Uuid>, _>("instrument_id"), "timescale": r.get::<String, _>("timescale"),
            "range_start": r.get::<DateTime<Utc>, _>("range_start"), "range_end": r.get::<DateTime<Utc>, _>("range_end"),
            "cursor": r.get::<Option<DateTime<Utc>>, _>("cursor"), "status": r.get::<String, _>("status"),
            "updated_at": r.get::<DateTime<Utc>, _>("updated_at"), "live": r.get::<bool, _>("live"),
        })
    })
    .collect::<Vec<_>>();
    let runs = sqlx::query(
        "SELECT id, started_at, ended_at, status, events, errors, last_error FROM ingest.connector_run \
          WHERE source_id = $1 ORDER BY started_at DESC LIMIT $2",
    )
    .bind(id)
    .bind(limit)
    .fetch_all(pool)
    .await?
    .iter()
    .map(|r| {
        json!({
            "id": r.get::<Uuid, _>("id"), "started_at": r.get::<DateTime<Utc>, _>("started_at"),
            "ended_at": r.get::<Option<DateTime<Utc>>, _>("ended_at"), "status": r.get::<String, _>("status"),
            "events": r.get::<i64, _>("events"), "errors": r.get::<i64, _>("errors"),
            "last_error": r.get::<Option<String>, _>("last_error"),
        })
    })
    .collect::<Vec<_>>();
    let recent_errors: Vec<Value> = runs
        .iter()
        .filter(|r| r["last_error"].is_string())
        .map(|r| json!({ "at": r["ended_at"], "run_id": r["id"], "error": r["last_error"] }))
        .collect();
    let (quarantine_open,): (i64,) =
        sqlx::query_as("SELECT count(*) FROM ingest.quarantine WHERE source_id = $1 AND status = 'open'")
            .bind(id)
            .fetch_one(pool)
            .await?;
    Ok(vec![
        ("cursors", json!(cursors)),
        ("runs", json!(runs)),
        ("recent_errors", json!(recent_errors)),
        ("quarantine_open", json!(quarantine_open)),
    ])
}

// ------------------------------------------------------------------ actions

#[derive(Debug, Default, Deserialize)]
pub struct BackfillBody {
    #[serde(default)]
    pub instruments: Vec<String>,
    pub timescale: Option<String>,
    /// RFC 3339; the ingestor defaults `end` to now and `start` to `end − depth(timescale)`
    /// (spec 10 §3: seconds 4 w, minutes 1 y, hours 5 y, days max).
    pub start: Option<String>,
    pub end: Option<String>,
    /// Alternative to start/end: `range` as `<n>(m|h|d)` back from now.
    pub range: Option<String>,
}

async fn source_action(
    State(s): State<AppState>,
    Path((id, action)): Path<(String, String)>,
    body: Option<Json<BackfillBody>>,
) -> Response {
    let mut args = vec![("source".to_owned(), id.clone())];
    let payload = match action.as_str() {
        "enable" | "disable" => json!({}),
        "backfill" => {
            let b = body.map(|Json(b)| b).unwrap_or_default();
            let mut start = b.start.clone();
            if let Some(range) = b.range.as_deref().filter(|r| !r.trim().is_empty()) {
                match crate::ledger_routes::parse_since(range, Utc::now()) {
                    Ok(Some(t)) => start = Some(t.to_rfc3339()),
                    Ok(None) => {}
                    Err(e) => return err(StatusCode::BAD_REQUEST, e),
                }
            }
            let instruments: Vec<&str> = b.instruments.iter().map(|x| x.trim()).filter(|x| !x.is_empty()).collect();
            if !instruments.is_empty() {
                args.push(("instruments".into(), instruments.join(",")));
            }
            for (k, v) in [("timescale", &b.timescale), ("start", &start), ("end", &b.end)] {
                if let Some(v) = v.as_deref().filter(|v| !v.trim().is_empty()) {
                    args.push((k.into(), v.trim().to_owned()));
                }
            }
            json!({ "instruments": instruments, "timescale": b.timescale, "start": start, "end": b.end })
        }
        _ => return err(StatusCode::NOT_FOUND, format!("unknown action {action:?}; use enable|disable|backfill")),
    };
    let req = CtlRequest { verb: action.clone(), args: args.into_iter().collect() };
    let reply = match ctl_request(&s, subjects::ctl("ingest", &action), &req).await {
        Ok(r) => r,
        Err(e) => return *e,
    };
    if !reply.ok {
        return ctl_error(&reply.message);
    }
    if let Some(pool) = s.pg.get() {
        let mut p = payload;
        p["reply"] = json!(reply.message);
        if let Err(e) = fa_warehouse::audit::append(pool, ACTOR, &format!("source.{action}"), Some(&id), p).await {
            tracing::warn!(error = %e, "audit append failed");
        }
    }
    let mut out = json!({ "ok": true, "source": id, "action": action, "message": reply.message });
    if action == "backfill" {
        out["job_id"] = json!(reply.message);
        out["kind"] = json!("ingest.backfill");
    }
    Json(out).into_response()
}

/// `CtlReply{ok:false}` prefixes agreed with the Go services (D-036).
fn ctl_error(message: &str) -> Response {
    let status = match message.split(':').next().unwrap_or_default() {
        "not_found" => StatusCode::NOT_FOUND,
        "conflict" => StatusCode::CONFLICT,
        "invalid" => StatusCode::BAD_REQUEST,
        _ => StatusCode::UNPROCESSABLE_ENTITY,
    };
    err(status, message)
}

async fn ctl_request(s: &AppState, subject: String, req: &CtlRequest) -> Result<CtlReply, Box<Response>> {
    let Some(client) = s.nats.as_ref().filter(|_| s.nats_connected()) else {
        return Err(Box::new(err(StatusCode::SERVICE_UNAVAILABLE, "NATS not connected")));
    };
    let fut = client.request(subject.clone(), Bytes::from(req.encode_to_vec()));
    match tokio::time::timeout(CTL_TIMEOUT, fut).await {
        Ok(Ok(msg)) => CtlReply::decode(msg.payload.as_ref())
            .map_err(|e| Box::new(err(StatusCode::BAD_GATEWAY, format!("bad reply on {subject}: {e}")))),
        Ok(Err(e)) => Err(Box::new(err(StatusCode::BAD_GATEWAY, format!("{subject}: {e} (is the ingestor running?)")))),
        Err(_) => {
            Err(Box::new(err(StatusCode::GATEWAY_TIMEOUT, format!("{subject}: no reply within {CTL_TIMEOUT:?}"))))
        }
    }
}

// -------------------------------------------------------------- instruments

#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub q: Option<String>,
    pub status: Option<String>,
    pub limit: Option<i64>,
}

async fn instruments(State(s): State<AppState>, Query(q): Query<SearchQuery>) -> Response {
    let pool = match pool(&s) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let term = q.q.unwrap_or_default();
    let status = q.status.as_deref().map(str::trim).filter(|x| !x.is_empty());
    match refdata::search_instruments(&pool, &term, status, q.limit.unwrap_or(50)).await {
        Ok(hits) => {
            Json(json!({ "q": term, "status": status, "count": hits.len(), "instruments": hits })).into_response()
        }
        Err(e) => db_err(e),
    }
}

async fn quarantine(State(s): State<AppState>, Query(q): Query<SearchQuery>) -> Response {
    let pool = match pool(&s) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let status = q.status.clone().unwrap_or_else(|| "open".into());
    let filter = (status != "all").then_some(status.as_str());
    match refdata::quarantine_list(&pool, filter, q.limit.unwrap_or(200)).await {
        Ok(rows) => Json(json!({ "status": status, "count": rows.len(), "quarantine": rows })).into_response(),
        Err(e) => db_err(e),
    }
}

async fn quarantine_resolve(
    State(s): State<AppState>,
    Path(id): Path<String>,
    body: Option<Json<Resolution>>,
) -> Response {
    let pool = match pool(&s) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let Ok(prov) = Uuid::parse_str(id.trim()) else {
        return err(StatusCode::BAD_REQUEST, "quarantine id must be the provisional instrument UUID");
    };
    let res = body.map(|Json(b)| b).unwrap_or_default();
    match refdata::resolve_quarantine(&pool, prov, &res).await {
        Ok(out) => {
            let payload = json!({ "resolution": res, "outcome": out });
            if let Err(e) = fa_warehouse::audit::append(
                &pool,
                ACTOR,
                "instrument.quarantine.resolve",
                Some(&prov.to_string()),
                payload,
            )
            .await
            {
                tracing::warn!(error = %e, "audit append failed");
            }
            // spec 12 §6: caches (ingestor resolver, risk-engine RefTable) reload on ref.changed.*
            if let Some(nats) = &s.nats {
                let n = fa_bus::ingest::ref_changed_notice("instrument", vec![prov.to_string()], "quarantine_resolved", ACTOR);
                let _ = nats.publish(subjects::ref_changed("instrument"), n.encode_to_vec().into()).await;
            }
            Json(json!({ "ok": true, "instrument_id": out.instrument_id, "merged_from": out.merged_from,
                "action": out.action, "quarantine_rows": out.quarantine_rows }))
            .into_response()
        }
        Err(ResolveError::NotFound(_)) => {
            err(StatusCode::NOT_FOUND, format!("no open quarantine row for instrument {prov}"))
        }
        Err(ResolveError::Invalid(m)) => err(StatusCode::BAD_REQUEST, m),
        Err(ResolveError::Warehouse(e)) => db_err(e),
    }
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use fa_core::config::PlatformConfig;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    fn app() -> Router {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let cfg = PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        router().with_state(AppState::new(cfg, format!("{root}/sources.yaml").into(), None))
    }

    async fn call(req: Request<Body>) -> (StatusCode, Value) {
        let res = app().oneshot(req).await.unwrap();
        let status = res.status();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
    }

    async fn get_json(uri: &str) -> (StatusCode, Value) {
        call(Request::get(uri).body(Body::empty()).unwrap()).await
    }

    async fn post_json(uri: &str, body: Option<Value>) -> (StatusCode, Value) {
        let req = match body {
            Some(b) => {
                Request::post(uri).header("content-type", "application/json").body(Body::from(b.to_string())).unwrap()
            }
            None => Request::post(uri).body(Body::empty()).unwrap(),
        };
        call(req).await
    }

    #[tokio::test]
    async fn sources_list_catalog_with_health_overlay() {
        sources::health().write().unwrap().ingest(
            "ingest.health.yahoo",
            br#"{"source":"yahoo","status":"ok","running":true,"events_total":7}"#,
            Utc::now(),
        );
        let (s, v) = get_json("/api/sources").await;
        assert_eq!(s, StatusCode::OK);
        assert!(v["sources"].as_array().unwrap().len() >= 70);
        assert_eq!(v["status_from_postgres"], false);
        let yahoo = v["sources"].as_array().unwrap().iter().find(|s| s["id"] == "yahoo").unwrap();
        assert_eq!(yahoo["running"], true);
        assert_eq!(yahoo["health"]["events_total"], 7);
        assert_eq!(yahoo["yaml_status"], "live");
        assert_eq!(yahoo["config"]["timeframe"], "1m");
        let (s, v) = get_json("/api/sources/oanda").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["id"], "oanda");
        assert_eq!(v["running"], false);
        assert!(v["health"].is_null());
        assert_eq!(v["gaps"], json!([]));
        assert!(v.get("cursors").is_none(), "no postgres: no cursor section");
        let (s, _) = get_json("/api/sources/nope").await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn actions_and_reference_routes_need_their_backends() {
        let (s, _) = post_json("/api/sources/yahoo/enable", None).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
        let (s, _) =
            post_json("/api/sources/yahoo/backfill", Some(json!({"instruments": ["AAPL"], "range": "1d"}))).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
        let (s, v) = post_json("/api/sources/yahoo/backfill", Some(json!({"range": "soon"}))).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert!(v["error"].as_str().unwrap().contains("bad range"));
        let (s, _) = post_json("/api/sources/yahoo/pause", None).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        for uri in ["/api/instruments?q=aapl", "/api/instruments/quarantine"] {
            let (s, _) = get_json(uri).await;
            assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{uri}");
        }
        let (s, _) = post_json("/api/instruments/quarantine/not-a-uuid/resolve", Some(json!({}))).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "503 before validation while postgres is down");
    }

    #[test]
    fn ctl_error_prefixes_map_to_http() {
        assert_eq!(ctl_error("not_found: x").status(), StatusCode::NOT_FOUND);
        assert_eq!(ctl_error("conflict: x").status(), StatusCode::CONFLICT);
        assert_eq!(ctl_error("invalid: x").status(), StatusCode::BAD_REQUEST);
        assert_eq!(ctl_error("boom").status(), StatusCode::UNPROCESSABLE_ENTITY);
    }
}
