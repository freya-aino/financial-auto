//! model-runtime: live latent pipeline per timescale (spec 01 §6, D-005, D-018).
//!
//! Consumes `md.trade.>`, builds rolling bars per timescale and instrument ([`bars`]) and
//! runs the baseline pipeline ([`pipeline`]): statistical encoder → evidence-softmax router
//! → online leader-clustering regions → Dirichlet transition posteriors. It publishes
//! `LatentState` on `latent.state.<ts>.<instrument>`, `RegionEvent` on `latent.region.<ts>`
//! and `TransitionUpdate` on `latent.transition.<ts>`, plus a heartbeat with counters.
//!
//! Neural encoders are off (D-004): the "snapshot" is the versioned statistical baseline
//! (`stat-v1-<ts>`), not a trained model. Every stage sits behind a trait so trained
//! models can replace the baselines without touching the bus contract.

mod bars;
mod pipeline;

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
    time::{Duration, Instant},
};

use bars::{BarSeries, BarUpdate};
use clap::Parser;
use fa_bus::{Bus, subjects};
use fa_core::{Timescale, config::PlatformConfig};
use fa_patterns::OnlineRegionCfg;
use fa_proto::v1::{Envelope, Heartbeat, envelope::Payload};
use futures::StreamExt;
use pipeline::{PipelineConfig, StepInput, TimescalePipeline};
use tokio::sync::Mutex;

#[derive(Parser)]
struct Args {
    #[arg(long, env = "FA_CONFIG", default_value = "config/platform.toml")]
    config: String,
    #[arg(long, default_value_t = 2)]
    heartbeat_secs: u64,
    /// Between bar closes, re-encode the forming bar and publish a preview state at most this
    /// often per (timescale, instrument). Keeps slow timescales (1h, 1d) visible in the UI.
    #[arg(long, default_value_t = 5)]
    preview_secs: u64,
    /// Leader-clustering radius in latent units (features are bounded to (-1, 1) per dim).
    #[arg(long, default_value_t = 1.0)]
    region_radius: f32,
    /// Hard cap on live regions per (timescale, expert).
    #[arg(long, default_value_t = 64)]
    max_regions: usize,
}

type Counters = Arc<Mutex<BTreeMap<String, f64>>>;

struct TimescaleState {
    pipeline: TimescalePipeline,
    series: HashMap<String, BarSeries>,
    last_preview: HashMap<String, Instant>,
}

fn trade_of(env: &Envelope) -> Option<(i64, f64, f64)> {
    let Some(Payload::Trade(t)) = &env.payload else {
        return None;
    };
    if env.instrument_id.is_empty() || t.price <= 0.0 || !t.price.is_finite() {
        return None;
    }
    let t_ms = env
        .event_time
        .as_ref()
        .or(env.ingest_time.as_ref())
        .map_or_else(|| fa_proto::now().seconds * 1000, |ts| ts.seconds * 1000 + i64::from(ts.nanos) / 1_000_000);
    Some((t_ms, t.price, t.size.max(0.0)))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();
    let args = Args::parse();
    let cfg = PlatformConfig::load(&args.config)?;
    let bus = Bus::connect(&cfg.nats.url).await?;

    let pcfg = PipelineConfig {
        regions: OnlineRegionCfg {
            radius: args.region_radius,
            merge_radius: args.region_radius * 0.35,
            max_regions: args.max_regions,
            ..OnlineRegionCfg::default()
        },
        ..PipelineConfig::default()
    };
    let mut states: Vec<TimescaleState> = cfg
        .timescales
        .iter()
        .map(|&ts| TimescaleState {
            pipeline: TimescalePipeline::baseline(ts, &pcfg),
            series: HashMap::new(),
            last_preview: HashMap::new(),
        })
        .collect();
    tracing::info!(
        timescales = ?cfg.timescales,
        snapshots = ?states.iter().map(|s| s.pipeline.snapshot_id().to_owned()).collect::<Vec<_>>(),
        region_radius = args.region_radius,
        "model-runtime starting (baseline pipeline: statistical encoder, no neural models)"
    );

    let counters: Counters = Arc::default();
    let instance_id = ulid::Ulid::generate().to_string();
    spawn_heartbeat(bus.clone(), counters.clone(), instance_id, args.heartbeat_secs);

    let preview = Duration::from_secs(args.preview_secs);
    let mut trades = bus.subscribe::<Envelope>(subjects::ALL_MD_TRADES.into()).await?;
    while let Some((subject, env)) = trades.next().await {
        {
            let mut c = counters.lock().await;
            *c.entry("trades_total".into()).or_default() += 1.0;
            *c.entry(format!("trades.{subject}")).or_default() += 1.0;
        }
        let Some((t_ms, price, size)) = trade_of(&env) else {
            continue;
        };
        let inst = env.instrument_id.as_str();
        let now = Instant::now();

        for st in &mut states {
            let ts = st.pipeline.timescale();
            let context = st.pipeline.context_steps();
            let series = st.series.entry(inst.to_owned()).or_insert_with(|| BarSeries::new(ts, context));
            let (bars, as_of_ms, closed) = match series.on_trade(t_ms, price, size) {
                BarUpdate::Closed { bar_end_ms } => (series.closed(), bar_end_ms, true),
                BarUpdate::Extended => {
                    let due = st.last_preview.get(inst).is_none_or(|t| now.duration_since(*t) >= preview);
                    if !due {
                        continue;
                    }
                    (series.with_forming(), t_ms, false)
                }
            };
            st.last_preview.insert(inst.to_owned(), now);
            let out =
                match st.pipeline.step(&StepInput { instrument_id: inst, as_of_ms, bars: &bars, bar_closed: closed }) {
                    Ok(o) => o,
                    Err(e) => {
                        tracing::warn!(%ts, instrument = inst, error = %e, "encode failed");
                        continue;
                    }
                };
            publish_step(&bus, &counters, ts, inst, closed, out, &st.pipeline).await;
        }
    }
    Ok(())
}

async fn publish_step(
    bus: &Bus,
    counters: &Counters,
    ts: Timescale,
    inst: &str,
    closed: bool,
    out: pipeline::StepOutput,
    pl: &TimescalePipeline,
) {
    let mut published = BTreeMap::<&str, f64>::new();
    for ev in &out.region_events {
        if let Err(e) = bus.publish(subjects::latent_region(ts), ev).await {
            tracing::warn!(error = %e, "region event publish failed");
        }
        *published.entry("region_events").or_default() += 1.0;
    }
    for tr in &out.transitions {
        if let Err(e) = bus.publish(subjects::latent_transition(ts), tr).await {
            tracing::warn!(error = %e, "transition publish failed");
        }
        *published.entry("transitions_published").or_default() += 1.0;
    }
    if let Some(state) = &out.state {
        // Subject per instrument: `latent.state.<ts>.<instrument token>` (D-017 token).
        let token = inst.replace('-', "");
        if let Err(e) = bus.publish(subjects::latent_state(ts, &token), state).await {
            tracing::warn!(error = %e, "latent state publish failed");
        }
        *published.entry("states_published").or_default() += 1.0;
    }
    let mut c = counters.lock().await;
    for (k, v) in published {
        *c.entry(k.to_owned()).or_default() += v;
        *c.entry(format!("{k}.{ts}")).or_default() += v;
    }
    if closed {
        *c.entry(format!("bars_closed.{ts}")).or_default() += 1.0;
    }
    c.insert(format!("regions_live.{ts}"), pl.live_regions() as f64);
    c.insert(format!("transition_rows.{ts}"), pl.transition_rows() as f64);
}

fn spawn_heartbeat(bus: Bus, counters: Counters, instance_id: String, every_secs: u64) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(every_secs));
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
