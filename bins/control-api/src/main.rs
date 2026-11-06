//! control-api: axum REST + WebSocket control plane for the frontend (spec 50).
//! Local only, no auth (ADR 0006).

mod bus_tap;
mod routes;
mod sources;
mod state;

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
    state.spawn_heartbeat_collector();

    let app = routes::router(state, Some(args.frontend));
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
