//! HTTP routes (spec 50 §4). Implemented: health, services, sources, timescales, bus tap,
//! latent views, decoder outputs, jobs/GPU/generations, paper trading (see `live_routes`).
//! Everything else in the catalog answers 501 with a pointer to its spec section.
use std::{path::PathBuf, time::Duration};

use axum::{
    Json, Router,
    extract::State,
    http::{StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::{any, get, post},
};
use serde_json::json;
use tower_http::{
    cors::CorsLayer,
    services::{ServeDir, ServeFile},
    trace::TraceLayer,
};

use crate::{bus_tap, live_routes as lr, state::AppState};

pub fn router(state: AppState, frontend: Option<PathBuf>) -> Router {
    let mut app = Router::new()
        // 4.1 Ops
        .route("/api/health", get(health))
        .route("/api/services", get(services))
        .route("/api/config/timescales", get(timescales))
        .route("/api/jobs", get(lr::jobs))
        .route("/api/jobs/{id}/cancel", post(lr::cancel_job))
        .route("/api/gpu", get(lr::gpu))
        // 4.2 Sources & ingestion
        // (sources, instruments and quarantine live in `sources_routes`, merged in main)
        .route("/api/ingest/{source}/{action}", post(lr::ingest_control))
        // 4.3 Bus tap
        .route("/ws/bus", get(bus_tap::ws_bus))
        // 4.4 Hub / graph: `graph_routes` (Helix-backed, merged in main)
        // 4.5 Latent / MCMC (in-memory views; the persisted `/api/latent/snapshots`, `/api/latent/{snapshot}/regions`
        // and `/api/mcmc/{snapshot}/graph` live in `graph_routes`)
        .route("/api/latent/timescales", get(lr::latent_timescales))
        .route("/api/latent/timescales/{ts}/{view}", get(lr::latent_view))
        .route("/api/decoders/outputs", get(lr::decoder_outputs))
        // 4.6 Evolution
        .route("/api/evolution/jobs", get(lr::jobs))
        .route("/api/evolution/gpu", get(lr::gpu))
        .route("/api/evolution/generations", get(crate::evolution_routes::generations).post(lr::start_generation))
        // 4.7 Trading
        .route("/api/trading/positions", get(lr::positions))
        .route("/api/trading/orders", get(lr::orders).post(lr::paper_order))
        .route("/api/trading/fills", get(lr::fills))
        .route("/api/trading/risk", get(lr::risk))
        .route("/api/trading/signals", get(lr::signals))
        .route("/api/trading/halt", post(lr::halt))
        .route("/api/trading/resume", post(lr::resume))
        .route("/api/trading/paper-order", post(lr::paper_order))
        .route("/api/trading/pnl", get(not_implemented))
        .route("/api/trading/strategies", get(not_implemented))
        .route("/api/trading/strategies/{id}/{action}", post(not_implemented))
        .route("/api/trading/flatten", post(not_implemented))
        .route("/api/trading/kill-switch", post(lr::halt))
        .route("/api/trading/kill-switch/rearm", post(lr::resume))
        .route("/api/trading/limits", any(not_implemented))
        .route("/api/trading/reconciliation", get(not_implemented))
        .route("/api/audit", get(not_implemented))
        .route("/api/{*rest}", any(api_not_found));

    if let Some(dir) = frontend.filter(|d| d.join("index.html").is_file()) {
        let index = dir.join("index.html");
        app = app.fallback_service(ServeDir::new(dir).fallback(ServeFile::new(index)));
    }

    app.layer(CorsLayer::permissive()).layer(TraceLayer::new_for_http()).with_state(state)
}

/// 501 response naming the spec section that defines the route.
struct NotImpl(Uri);

impl IntoResponse for NotImpl {
    fn into_response(self) -> Response {
        let path = self.0.path().to_owned();
        let section = spec_section(&path);
        (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({"error": "not_implemented", "path": path, "spec": format!("specs/50-control-api.md §{section}")})),
        )
            .into_response()
    }
}

async fn not_implemented(uri: Uri) -> NotImpl {
    NotImpl(uri)
}

fn spec_section(path: &str) -> &'static str {
    const AREAS: [(&str, &str); 10] = [
        ("/api/sources", "4.2"),
        ("/api/instruments", "4.2"),
        ("/api/bus", "4.3"),
        ("/api/graph", "4.4"),
        ("/api/latent", "4.5"),
        ("/api/mcmc", "4.5"),
        ("/ws/latent", "4.5"),
        ("/api/evolution", "4.6"),
        ("/ws/evolution", "4.6"),
        ("/api/trading", "4.7"),
    ];
    if path.starts_with("/api/audit") || path.starts_with("/ws/trading") {
        return "4.7";
    }
    AREAS.iter().find(|(p, _)| path.starts_with(p)).map_or("4.1", |(_, s)| s)
}

async fn api_not_found(uri: Uri) -> impl IntoResponse {
    (StatusCode::NOT_FOUND, Json(json!({"error": "not_found", "path": uri.path()})))
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let cfg = &state.cfg;
    let (helix, vm, pg, garage) = tokio::join!(
        probe(&cfg.helix.url),
        probe(&cfg.victoriametrics.url),
        probe(&cfg.postgres.url),
        probe(&cfg.garage.endpoint),
    );
    Json(json!({
        "service": "control-api",
        "version": env!("CARGO_PKG_VERSION"),
        "nats": { "connected": state.nats_connected(), "url": cfg.nats.url },
        // TCP reachability only; real health checks arrive with each client crate.
        "dependencies": {
            "helix": helix,
            "victoriametrics": vm,
            "postgres": pg,
            "garage": garage,
        }
    }))
}

/// TCP reachability of `scheme://[user@]host[:port]/...`.
async fn probe(url: &str) -> bool {
    let Some(addr) = host_port(url) else {
        return false;
    };
    tokio::time::timeout(Duration::from_millis(300), tokio::net::TcpStream::connect(addr))
        .await
        .is_ok_and(|r| r.is_ok())
}

fn host_port(url: &str) -> Option<String> {
    let (scheme, rest) = url.split_once("://")?;
    let authority = rest.split('/').next()?;
    let authority = authority.rsplit_once('@').map_or(authority, |(_, a)| a);
    if authority.contains(':') {
        return Some(authority.to_owned());
    }
    let port = match scheme {
        "https" => 443,
        "postgres" | "postgresql" => 5432,
        "nats" => 4222,
        _ => 80,
    };
    Some(format!("{authority}:{port}"))
}

async fn services(State(state): State<AppState>) -> impl IntoResponse {
    let list: Vec<_> = state.services.read().expect("services lock").values().cloned().collect();
    Json(json!({ "services": list, "nats_connected": state.nats_connected() }))
}

async fn timescales(State(state): State<AppState>) -> impl IntoResponse {
    let ladder: Vec<_> =
        state.cfg.timescales.iter().map(|t| json!({ "id": t.as_str(), "seconds": t.duration().as_secs() })).collect();
    Json(json!({ "timescales": ladder }))
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use fa_core::config::PlatformConfig;
    use http_body_util::BodyExt;
    use serde_json::Value;
    use tower::ServiceExt;

    use super::*;

    fn app() -> Router {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let cfg = PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        router(AppState::new(cfg, format!("{root}/sources.yaml").into(), None), None)
    }

    async fn get_json(uri: &str) -> (StatusCode, Value) {
        let res = app().oneshot(Request::get(uri).body(Body::empty()).unwrap()).await.unwrap();
        let status = res.status();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&body).unwrap())
    }

    #[tokio::test]
    async fn health_reports_nats_disconnected_without_client() {
        let (s, v) = get_json("/api/health").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["nats"]["connected"], false);
    }

    #[tokio::test]
    async fn timescales_follow_config() {
        let (_, v) = get_json("/api/config/timescales").await;
        assert_eq!(v["timescales"][0]["id"], "1s");
        assert_eq!(v["timescales"][4]["seconds"], 604_800);
    }

    #[tokio::test]
    async fn unimplemented_routes_return_501_with_spec_pointer() {
        let (s, v) = get_json("/api/audit").await;
        assert_eq!(s, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(v["spec"], "specs/50-control-api.md §4.7");
        let (s, v) = get_json("/api/trading/pnl").await;
        assert_eq!(s, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(v["spec"], "specs/50-control-api.md §4.7");
    }

    async fn post_json(uri: &str, body: Value) -> (StatusCode, Value) {
        let req =
            Request::post(uri).header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
        let res = app().oneshot(req).await.unwrap();
        let status = res.status();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&body).unwrap())
    }

    #[tokio::test]
    async fn live_views_answer_with_empty_data() {
        for uri in [
            "/api/latent/timescales/1h/states",
            "/api/latent/timescales/1h/regions",
            "/api/latent/timescales/1h/transitions",
            "/api/decoders/outputs?family=text",
            "/api/evolution/jobs",
            "/api/evolution/gpu",
            "/api/trading/positions",
            "/api/trading/orders",
            "/api/trading/fills",
            "/api/trading/risk",
            "/api/trading/signals",
        ] {
            let (s, _) = get_json(uri).await;
            assert_eq!(s, StatusCode::OK, "{uri}");
        }
        let (_, v) = get_json("/api/trading/risk").await;
        assert_eq!(v["live_available"], false);
        let (s, _) = get_json("/api/latent/timescales/1h/nope").await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn live_orders_are_rejected_and_commands_need_nats() {
        let order = json!({"instrument_id": "01900000-0000-7000-8000-000000000001", "side": "buy", "quantity": 1});
        let mut live = order.clone();
        live["mode"] = json!("live");
        let (s, v) = post_json("/api/trading/paper-order", live).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert!(v["error"].as_str().unwrap().contains("only paper"));
        let (s, _) = post_json("/api/trading/paper-order", order).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
        let (s, _) = post_json("/api/trading/halt", json!({})).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
        let (s, _) = post_json("/api/evolution/generations", json!({"kind": "", "genomes": []})).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, _) = post_json("/api/ingest/oanda/start", json!({})).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
        let (s, _) = post_json("/api/ingest/oanda/pause", json!({})).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        let (s, _) = post_json("/api/jobs/01ARZ3NDEKTSV4RRFFQ69G5FAV/cancel", json!({})).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn unknown_api_path_is_404() {
        let (s, _) = get_json("/api/does/not/exist/at/all").await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }

    #[test]
    fn host_port_parsing() {
        assert_eq!(host_port("postgres://fa:fa@127.0.0.1:5432/fa").as_deref(), Some("127.0.0.1:5432"));
        assert_eq!(host_port("http://127.0.0.1:6969").as_deref(), Some("127.0.0.1:6969"));
        assert_eq!(host_port("http://localhost/x").as_deref(), Some("localhost:80"));
    }
}
