//! strategy-runtime: runs a population of GP strategies live and publishes
//! `ExposureDistribution`s on `signal.exposure.<strategy_id>` (spec 35, spec 21 §4.1).
//!
//! Inputs: `md.trade.>` (prices, always) and `latent.state.>` (regions, when model-runtime
//! publishes them). While graph/decoder feeds are missing, programs are evaluated on the
//! price-only fallback features of `fa_strategies::features`, so signals flow as soon as
//! trades do. The default population is the hand-written **baseline** set in
//! `crates/fa-strategies/baselines.toml` (not evolved); pass `--strategies <file>` to run
//! another population.
mod runtime;

use std::{collections::BTreeMap, path::PathBuf, sync::Arc, time::Duration};

use anyhow::Context;
use clap::Parser;
use fa_bus::{Bus, subjects};
use fa_core::{InstrumentId, config::PlatformConfig};
use fa_proto::v1::{Envelope, Heartbeat, LatentState, envelope::Payload};
use fa_strategies::population::StrategyFile;
use futures::StreamExt;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::runtime::Runtime;

const SERVICE: &str = "strategy-runtime";

#[derive(Debug, Parser)]
#[command(about = "live GP strategy population → signal.exposure.*")]
struct Args {
    /// Platform config (spec 60).
    #[arg(long, env = "FA_CONFIG", default_value = "config/platform.toml")]
    config: PathBuf,
    /// Strategy population file (TOML). Defaults to the built-in baselines.
    #[arg(long, env = "FA_STRATEGIES")]
    strategies: Option<PathBuf>,
    /// Scheduler resolution in milliseconds.
    #[arg(long, default_value_t = 250)]
    tick_ms: u64,
    #[arg(long, default_value_t = 2)]
    heartbeat_secs: u64,
}

fn ms(t: Option<&fa_proto::Timestamp>) -> Option<i64> {
    t.map(|t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000)
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();
    let cfg = PlatformConfig::load(&args.config).context("loading platform config")?;
    let file = match &args.strategies {
        Some(p) => StrategyFile::parse(
            &std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?,
        )
        .with_context(|| format!("parsing {}", p.display()))?,
        None => StrategyFile::baselines(),
    };
    let baselines = file.strategy.iter().filter(|s| s.is_baseline()).count();
    let rt = Arc::new(Mutex::new(Runtime::new(&file)?));
    tracing::info!(
        strategies = ?rt.lock().await.strategy_ids(),
        baselines,
        instruments = if file.instruments.is_empty() { "all seen".to_owned() } else { file.instruments.join(",") },
        "strategy-runtime starting (baseline programs are hand-written, not evolved)"
    );

    let bus = Bus::connect(&cfg.nats.url)
        .await
        .context("connecting to NATS")?;
    let instance_id = ulid::Ulid::generate().to_string();
    let published = Arc::new(Mutex::new(0u64));

    // heartbeat
    {
        let (bus, rt, published, instance_id) = (
            bus.clone(),
            rt.clone(),
            published.clone(),
            instance_id.clone(),
        );
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(args.heartbeat_secs));
            loop {
                tick.tick().await;
                let counters: BTreeMap<String, f64> = {
                    let r = rt.lock().await;
                    BTreeMap::from([
                        ("strategies".to_owned(), r.strategy_ids().len() as f64),
                        ("instruments".to_owned(), r.instruments() as f64),
                        ("trades_seen".to_owned(), r.trades_seen as f64),
                        ("latent_seen".to_owned(), r.latent_seen as f64),
                        ("signals_emitted".to_owned(), r.signals_emitted as f64),
                        (
                            "signals_published".to_owned(),
                            *published.lock().await as f64,
                        ),
                    ])
                };
                let hb = Heartbeat {
                    service: SERVICE.into(),
                    instance_id: instance_id.clone(),
                    at: Some(fa_proto::now()),
                    version: env!("CARGO_PKG_VERSION").into(),
                    counters: counters.into_iter().collect(),
                };
                if let Err(e) = bus.publish(subjects::heartbeat(SERVICE), &hb).await {
                    tracing::warn!(error = %e, "heartbeat publish failed");
                }
            }
        });
    }

    // trades
    {
        let (bus, rt) = (bus.clone(), rt.clone());
        let mut trades = bus
            .subscribe::<Envelope>(subjects::ALL_MD_TRADES.into())
            .await?;
        tokio::spawn(async move {
            while let Some((_, env)) = trades.next().await {
                if let Some(Payload::Trade(t)) = &env.payload
                    && let Ok(uuid) = Uuid::parse_str(&env.instrument_id)
                {
                    let at = ms(env.event_time.as_ref()).unwrap_or_else(now_ms);
                    rt.lock().await.on_trade(InstrumentId(uuid), at, t.price);
                }
            }
        });
    }

    // latent states (optional feed)
    {
        let (bus, rt) = (bus.clone(), rt.clone());
        let mut states = bus
            .subscribe::<LatentState>("latent.state.>".into())
            .await?;
        tokio::spawn(async move {
            while let Some((_, st)) = states.next().await {
                rt.lock().await.on_latent(&st);
            }
        });
    }

    // scheduler
    let mut tick = tokio::time::interval(Duration::from_millis(args.tick_ms.max(50)));
    loop {
        tick.tick().await;
        let dists = rt.lock().await.tick(now_ms());
        for d in dists {
            let proto = d.to_proto();
            match bus
                .publish(subjects::signal_exposure(&d.strategy_id), &proto)
                .await
            {
                Ok(()) => *published.lock().await += 1,
                Err(e) => {
                    tracing::warn!(strategy = %d.strategy_id, error = %e, "signal publish failed")
                }
            }
        }
    }
}
