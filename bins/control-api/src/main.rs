//! control-api: axum REST + WebSocket control plane for the frontend (spec 50).
//! Local only, no auth (ADR 0006).

mod alerts_routes;
mod audit;
mod bus_tap;
mod config_routes;
mod console_routes;
mod docker;
mod evolution_routes;
mod graph_routes;
mod latent_routes;
mod ledger_routes;
mod live;
mod live_routes;
mod mcmc_routes;
mod metrics_routes;
mod openapi;
mod ops_routes;
mod ref_mirror;
mod refdata_maint;
mod refdata_routes;
mod routes;
mod services_routes;
mod sources;
mod sources_routes;
mod state;
mod streams_routes;
mod trading_routes;
mod validation_routes;
mod ws_routes;

use std::{net::SocketAddr, path::PathBuf};

use anyhow::Context;
use clap::Parser;
use fa_core::config::PlatformConfig;

#[derive(Debug, Parser)]
#[command(about = "financial-auto control plane API")]
struct Args {
    /// Platform config (spec 60).
    #[arg(long, env = "FA_CONFIG", default_value = "config/platform.toml")]
    config: PathBuf,
    /// Source catalog (spec 11).
    #[arg(long, env = "FA_SOURCES", default_value = "config/sources.yaml")]
    sources: PathBuf,
    /// Built frontend served at `/` when present (D-015).
    #[arg(long, env = "FA_FRONTEND", default_value = "frontend/build")]
    frontend: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    fa_core::logging::init("control-api"); // FA_LOG_FORMAT=json|text (spec 60 §4)

    let args = Args::parse();
    fa_bus::metrics::serve_from_env("control-api", env!("CARGO_PKG_VERSION")); // GET /metrics on FA_METRICS_BIND (spec 60 §4, D-056)
    let cfg = PlatformConfig::load(&args.config).context("loading platform config")?;
    let bind: SocketAddr = cfg.control_api.bind.parse().context("parsing control_api.bind")?;

    // D-013: apply Postgres migrations on startup; keep serving if Postgres is unreachable.
    if let Err(e) = migrate(&cfg.postgres.url).await {
        tracing::warn!(error = %e, "postgres migrations not applied");
    }

    // Starts even if NATS is down: the client retries in the background.
    let nats = fa_bus::async_nats::ConnectOptions::new()
        .retry_on_initial_connect()
        .name("control-api")
        .connect(&cfg.nats.url)
        .await
        .context("configuring NATS client")?;

    let state = state::AppState::new(cfg, args.sources, Some(nats));
    state.live_mut().set_runtime(state.cfg.runtime.clone()); // [runtime] + config/runtime.toml overlay
    config_routes::set_runtime_path(&args.config); // `PUT /api/config/runtime` persists next to it
    state.spawn_heartbeat_collector();
    state.spawn_live_collector();
    streams_routes::spawn_ensure_streams(&state);
    sources::spawn_health_collector(&state); // ingest.health|gap|unresolved.* mirror (D-054)

    ops_routes::spawn(&state); // JOB_RESULTS → ops.job_run, config → ops.config_snapshot (wait for Postgres)
    ref_mirror::spawn(&state); // Postgres ref → Helix Instrument/Entity/Venue mirror: full at start + periodic, incremental on ref.changed.* (spec 12 §5)
    refdata_maint::spawn(&state); // daily calendar refresh + adjusted daily bars (spec 12 §7)
    let app = app(state, Some(args.frontend)).await;
    let listener = tokio::net::TcpListener::bind(bind).await.with_context(|| format!("binding {bind}"))?;
    tracing::info!(%bind, "control-api listening");
    axum::serve(listener, app).await?;
    Ok(())
}

/// The complete HTTP surface (also built by the request-audit test).
async fn app(state: state::AppState, frontend: Option<PathBuf>) -> axum::Router {
    let app = routes::router(state.clone(), frontend)
        .merge(graph_routes::router().with_state(state.clone()))
        .merge(mcmc_routes::router().with_state(state.clone())) // spec 33 §4.2 queries, lineage, hierarchy, cross priors
        .merge(openapi::router().with_state(state.clone())) // GET /api/openapi.json, generated from the route declarations
        .merge(streams_routes::router().with_state(state.clone()))
        .merge(sources_routes::router().with_state(state.clone()))
        .merge(refdata_routes::router().with_state(state.clone()))
        .merge(latent_routes::router().with_state(state.clone()))
        .merge(evolution_routes::router().with_state(state.clone()))
        .merge(config_routes::router().with_state(state.clone()))
        .merge(ws_routes::router().with_state(state.clone()))
        .merge(validation_routes::router().with_state(state.clone())) // holdout, trial reports, populations (D-052)
        .merge(alerts_routes::router(state.clone())) // vmalert receiver → ops.alert + audit (spec 60 §6.5)
        .merge(console_routes::router().with_state(state.clone())) // decoders, pipeline, halts/re-arm, risk events, refdata reads
        .merge(ops_routes::router().with_state(state.clone())) // ops schema: job history, config snapshots (spec 20 §5.4)
        .merge(metrics_routes::router().with_state(state.clone())); // VictoriaMetrics query passthrough with limits (spec 50 §5)
    let app = ledger_routes::install(state.clone(), app).await; // Postgres pool + paper-book seed + ledger routes (D-047)
    let app = services_routes::install(state.clone(), app); // start/stop/restart via the Docker socket (D-060)
    let app = trading_routes::install(state.clone(), app); // strategy lifecycle + limits via ctl.risk.* (D-051)
    let app = app.layer(fa_bus::metrics::http::layer()); // fa_http_requests_total / fa_http_request_duration_seconds (D-056)
    audit::wrap(state, app) // outermost: /api/v1 alias, trace id, request audit (spec 50 §3, Q2)
}

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("../../migrations");

async fn migrate(url: &str) -> anyhow::Result<()> {
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .acquire_timeout(std::time::Duration::from_secs(3))
        .connect(url)
        .await
        .context("connecting to postgres")?;
    MIGRATOR.run(&pool).await.context("running migrations")?;
    tracing::info!(count = MIGRATOR.iter().count(), "postgres migrations applied");
    Ok(())
}

#[cfg(test)]
mod tests {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use fa_core::config::PlatformConfig;
    use tower::ServiceExt;

    /// The full router with no NATS, no Postgres and unreachable stores, so no request can
    /// change anything outside the test.
    async fn test_app() -> axum::Router {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let mut cfg = PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        cfg.postgres.url = String::new();
        cfg.helix.url = "http://127.0.0.1:9".into();
        cfg.victoriametrics.url = "http://127.0.0.1:9".into();
        cfg.garage.endpoint = "http://127.0.0.1:9".into();
        let state = crate::state::AppState::new(cfg, format!("{root}/sources.yaml").into(), None);
        super::app(state, None).await
    }

    async fn send(app: &axum::Router, method: &str, uri: &str, body: &str) -> (StatusCode, Option<String>, Option<String>) {
        let req = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .header("traceparent", "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
            .body(Body::from(body.to_owned()))
            .unwrap();
        let res = app.clone().oneshot(req).await.unwrap();
        let h = |n: &str| res.headers().get(n).and_then(|v| v.to_str().ok()).map(str::to_owned);
        (res.status(), h(crate::audit::AUDIT_HEADER), h("traceparent"))
    }

    /// Spec 50 §3 / §9: every mutating route of the OpenAPI catalog writes an audit row (the
    /// handler's domain row or the generic `api.request` row), whatever the outcome.
    #[tokio::test]
    async fn every_mutating_route_audits() {
        let app = test_app().await;
        let mutating: Vec<_> = crate::openapi::all_routes()
            .into_iter()
            .filter(|r| matches!(r.method.as_str(), "post" | "put" | "patch" | "delete"))
            .collect();
        assert!(mutating.len() >= 35, "{} mutating routes", mutating.len());
        for r in mutating {
            let uri: String = r
                .path
                .split('/')
                .map(|seg| if seg.starts_with('{') { "zz-test" } else { seg })
                .collect::<Vec<_>>()
                .join("/");
            let (status, audit, trace) = send(&app, &r.method.to_uppercase(), &uri, "{}").await;
            if crate::audit::NOT_AUDITED.contains(&r.path.as_str()) {
                assert_eq!(audit, None, "{} {} is not audited per request", r.method, r.path);
            } else {
                assert!(audit.is_some(), "{} {} ({status}) wrote no audit row", r.method, r.path);
            }
            let trace = trace.unwrap_or_default();
            assert!(trace.starts_with("00-0af7651916cd43dd8448eb211c80319c-"), "trace continued: {trace}");
        }
    }

    /// The command routes named in spec 50 §9 write their own domain rows, also when the owning
    /// service (here: NATS itself) is unreachable; `/api/v1` is an alias.
    #[tokio::test]
    async fn commands_write_domain_rows_and_v1_is_an_alias() {
        let app = test_app().await;
        let order = r#"{"instrument_id":"01900000-0000-7000-8000-000000000001","side":"buy","quantity":1}"#;
        let generation = r#"{"kind":"evo.eval.decoder","genomes":["forecast"]}"#;
        for (method, uri, body, action) in [
            ("POST", "/api/evolution/generations", generation, "evolution.generation.start"),
            ("POST", "/api/evolution/runs", generation, "evolution.generation.start"),
            ("POST", "/api/jobs/j1/cancel", "", "evolution.job.cancel"),
            ("POST", "/api/ingest/kraken/start", "", "ingest.start"),
            ("POST", "/api/ingest/kraken/stop", "", "ingest.stop"),
            ("POST", "/api/trading/kill-switch", "{}", "halt.request"),
            ("POST", "/api/trading/halt", "{}", "halt.request"),
            ("POST", "/api/trading/kill-switch/rearm", "", "resume.request"),
            ("POST", "/api/trading/resume", "", "resume.request"),
            ("POST", "/api/trading/orders", order, "order.manual"),
            ("POST", "/api/trading/paper-order", order, "order.manual"),
            ("POST", "/api/v1/jobs/j1/cancel", "", "evolution.job.cancel"),
        ] {
            let (status, audit, _) = send(&app, method, uri, body).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{uri}: no NATS");
            assert_eq!(audit.as_deref(), Some(action), "{uri}");
        }
        // validation failures are still on record, as the generic row
        let (status, audit, _) = send(&app, "POST", "/api/evolution/generations", r#"{"kind":"x","genomes":[]}"#).await;
        assert_eq!((status, audit.as_deref()), (StatusCode::BAD_REQUEST, Some(crate::audit::GENERIC_ACTION)));
        // pre-trade check: request/reply on ctl.risk.pretrade, a read (not audited)
        let (status, audit, _) = send(&app, "POST", "/api/trading/pretrade", order).await;
        assert_eq!((status, audit), (StatusCode::SERVICE_UNAVAILABLE, None));
        let (status, _, _) = send(&app, "POST", "/api/trading/pretrade", r#"{"instrument_id":"x","side":"hold","quantity":1}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        // reads are not audited; the v1 alias serves every route
        let (status, audit, _) = send(&app, "GET", "/api/v1/health", "").await;
        assert_eq!((status, audit), (StatusCode::OK, None));
        let (status, _, _) = send(&app, "GET", "/api/v1/openapi.json", "").await;
        assert_eq!(status, StatusCode::OK);
        let (status, _, _) = send(&app, "GET", "/api/v10/health", "").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}
