//! control-api: axum REST + WebSocket control plane for the frontend (spec 50).
//! Local only, no auth (ADR 0006).

mod alerts_routes;
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
mod routes;
mod services_routes;
mod sources;
mod refdata_routes;
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
use tracing_subscriber::EnvFilter;

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
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

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

    let app = routes::router(state.clone(), Some(args.frontend))
        .merge(graph_routes::router().with_state(state.clone()))
        .merge(mcmc_routes::router().with_state(state.clone())) // spec 33 §4.2 queries, lineage, hierarchy, cross priors
        .merge(streams_routes::router().with_state(state.clone()))
        .merge(sources_routes::router().with_state(state.clone()))
        .merge(refdata_routes::router().with_state(state.clone()))
        .merge(latent_routes::router().with_state(state.clone()))
        .merge(evolution_routes::router().with_state(state.clone()))
        .merge(config_routes::router().with_state(state.clone()))
        .merge(ws_routes::router().with_state(state.clone()))
        .merge(validation_routes::router().with_state(state.clone())) // holdout, trial reports, populations (D-052)
        .merge(alerts_routes::router(state.clone())) // vmalert receiver → ops.alert + audit (spec 60 §6.5)
        .merge(console_routes::router().with_state(state.clone())); // decoders, pipeline, halts/re-arm, risk events, refdata reads
    let app = ledger_routes::install(state.clone(), app).await; // Postgres pool + paper-book seed + ledger routes (D-047)
    let app = services_routes::install(state.clone(), app); // start/stop/restart via the Docker socket (D-060)
    let app = trading_routes::install(state, app); // strategy lifecycle + limits via ctl.risk.* (D-051)
    let app = app.layer(fa_bus::metrics::http::layer()); // fa_http_requests_total / fa_http_request_duration_seconds (D-056)
    let listener = tokio::net::TcpListener::bind(bind).await.with_context(|| format!("binding {bind}"))?;
    tracing::info!(%bind, "control-api listening");
    axum::serve(listener, app).await?;
    Ok(())
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
