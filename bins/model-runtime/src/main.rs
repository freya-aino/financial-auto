//! model-runtime: live pipeline per timescale (spec 01 §6, D-005, D-018).
//!
//! Scaffold behaviour: consumes `md.trade.>`, counts messages per subject and publishes a
//! heartbeat with counters. Pipeline stages are wired as trait objects with no-op
//! implementations until real models exist.

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use clap::Parser;
use fa_bus::{Bus, subjects};
use fa_core::{Timescale, config::PlatformConfig};
use fa_proto::v1::{Envelope, Heartbeat};
use futures::StreamExt;
use tokio::sync::Mutex;

#[derive(Parser)]
struct Args {
    #[arg(long, env = "FA_CONFIG", default_value = "config/platform.toml")]
    config: String,
    #[arg(long, default_value_t = 2)]
    heartbeat_secs: u64,
}

/// One stage of the live pipeline (encode → regions/MCMC → decode → strategies).
trait Stage: Send + Sync {
    fn name(&self) -> &'static str;
    fn on_event(&self, timescale: Timescale, env: &Envelope);
}

struct NoopStage(&'static str);

impl Stage for NoopStage {
    fn name(&self) -> &'static str {
        self.0
    }
    fn on_event(&self, _: Timescale, _: &Envelope) {}
}

fn pipeline() -> Vec<Box<dyn Stage>> {
    ["encoders", "expert_spaces", "regions_mcmc", "decoders", "strategies"].into_iter().map(|n| Box::new(NoopStage(n)) as Box<dyn Stage>).collect()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?)).init();
    let args = Args::parse();
    let cfg = PlatformConfig::load(&args.config)?;
    let bus = Bus::connect(&cfg.nats.url).await?;
    let stages = Arc::new(pipeline());
    tracing::info!(timescales = ?cfg.timescales, stages = ?stages.iter().map(|s| s.name()).collect::<Vec<_>>(), "model-runtime starting");

    let counters: Arc<Mutex<BTreeMap<String, f64>>> = Arc::default();
    let instance_id = ulid::Ulid::generate().to_string();

    // heartbeat
    {
        let bus = bus.clone();
        let counters = counters.clone();
        let instance_id = instance_id.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(args.heartbeat_secs));
            loop {
                tick.tick().await;
                let hb = Heartbeat {
                    service: "model-runtime".into(),
                    instance_id: instance_id.clone(),
                    at: Some(fa_proto::now()),
                    version: env!("CARGO_PKG_VERSION").into(),
                    counters: counters.lock().await.iter().map(|(k, v)| (k.clone(), *v)).collect(),
                };
                if let Err(e) = bus.publish(subjects::heartbeat("model-runtime"), &hb).await {
                    tracing::warn!(error = %e, "heartbeat publish failed");
                }
            }
        });
    }

    let mut trades = bus.subscribe::<Envelope>(subjects::ALL_MD_TRADES.into()).await?;
    while let Some((subject, env)) = trades.next().await {
        for ts in &cfg.timescales {
            for s in stages.iter() {
                s.on_event(*ts, &env);
            }
        }
        let mut c = counters.lock().await;
        *c.entry("trades_total".into()).or_default() += 1.0;
        *c.entry(format!("trades.{subject}")).or_default() += 1.0;
    }
    Ok(())
}
