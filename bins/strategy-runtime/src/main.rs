//! strategy-runtime: runs a population of GP strategies live and publishes
//! `ExposureDistribution`s on `signal.exposure.<strategy_id>` (spec 35, spec 21 §4.1).
//!
//! Inputs: `md.trade.>` (prices, always), `latent.state.>` / `latent.region.>` /
//! `latent.transition.>` (the latent graph from model-runtime) and `decoder.output.>`
//! (evo-worker). Programs are evaluated through `fa_strategies::features::LiveContext`: graph
//! and decoder primitives use the real data when the bus has it for the strategy's timescale
//! and instrument, price-only fallbacks otherwise; the heartbeat reports how much was real
//! (`primitives.graph|decoder|fallback`). The default population is the hand-written
//! **baseline** set in `crates/fa-strategies/baselines.toml` (not evolved); pass
//! `--strategies <file>` to run another one, or hot-reload with `ctl.strategy.reload`
//! (`artifact_key` of a `strategies.<generation>.toml` in Garage, or a local `path`).
//! `ctl.strategy.pause` / `resume` (`strategy_id`) flatten a strategy without unloading it.
mod runtime;

use std::{collections::BTreeMap, path::PathBuf, sync::Arc, time::Duration};

use anyhow::Context;
use bytes::Bytes;
use clap::Parser;
use fa_blob::{BlobStore, ObjectKey, S3BlobStore, buckets};
use fa_bus::{Bus, subjects};
use fa_core::{InstrumentId, config::PlatformConfig, secrets::Secrets};
use fa_proto::{
    Message,
    v1::{
        CtlReply, CtlRequest, DecoderOutput, Envelope, Heartbeat, LatentState, RegionEvent, TransitionUpdate,
        envelope::Payload,
    },
};
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
    /// Decoder outputs older than this are ignored by the decoder primitives.
    #[arg(long, default_value_t = 6 * 3600)]
    decoder_max_age_secs: u64,
}

/// Loads a population from Garage (`artifact_key`) or a local file (`path`) and validates it.
async fn load_population(
    cfg: &PlatformConfig,
    args: &std::collections::HashMap<String, String>,
) -> anyhow::Result<(StrategyFile, String)> {
    if let Some(path) = args.get("path").filter(|p| !p.is_empty()) {
        let src = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
        let file = StrategyFile::parse(&src).with_context(|| format!("parsing {path}"))?;
        return Ok((file, path.clone()));
    }
    let Some(key) = args.get("artifact_key").filter(|k| !k.is_empty()) else {
        anyhow::bail!("invalid: reload needs `artifact_key` (Garage key under fa-artifacts/) or `path`");
    };
    let key = key.strip_prefix(&format!("{}/", buckets::ARTIFACTS)).unwrap_or(key).to_owned();
    let secrets = Secrets::load().context("loading secrets")?;
    let blob = S3BlobStore::from_config(cfg, &secrets).context("Garage credentials (config/secrets.toml [garage])")?;
    let bytes = blob
        .get(&ObjectKey { bucket: buckets::ARTIFACTS, key: key.clone() })
        .await
        .with_context(|| format!("not_found: fetching {}/{key}", buckets::ARTIFACTS))?;
    let src = String::from_utf8(bytes).context("invalid: population is not UTF-8")?;
    let file = StrategyFile::parse(&src).with_context(|| format!("invalid: parsing {key}"))?;
    Ok((file, key))
}

/// Handles one `ctl.strategy.<verb>` request.
async fn handle_ctl(cfg: &PlatformConfig, rt: &Mutex<Runtime>, verb: &str, req: &CtlRequest) -> CtlReply {
    let reply = |ok: bool, message: String| CtlReply { ok, message };
    match verb {
        "reload" => match load_population(cfg, &req.args).await {
            Ok((file, label)) => {
                let mut r = rt.lock().await;
                match r.replace_population(&file, &label) {
                    Ok(ids) => {
                        tracing::info!(population = %label, strategies = ?ids, "population reloaded");
                        reply(true, format!("loaded {} strategies from {label}: {}", ids.len(), ids.join(",")))
                    }
                    Err(e) => reply(false, format!("invalid: {e}")),
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "reload failed");
                reply(false, format!("{e:#}"))
            }
        },
        "pause" | "resume" => {
            let Some(id) = req.args.get("strategy_id").filter(|s| !s.is_empty()) else {
                return reply(false, "invalid: `strategy_id` required".into());
            };
            let mut r = rt.lock().await;
            let ok = if verb == "pause" { r.pause(id) } else { r.resume(id) };
            if ok {
                tracing::info!(strategy = %id, verb, "strategy state changed");
                reply(true, format!("{verb}d {id}"))
            } else {
                reply(false, format!("not_found: strategy `{id}`"))
            }
        }
        other => reply(false, format!("invalid: unknown verb `{other}` (reload | pause | resume)")),
    }
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
    fa_bus::metrics::serve_from_env("strategy-runtime", env!("CARGO_PKG_VERSION")); // GET /metrics on FA_METRICS_BIND (spec 60 §4, D-056)
    let cfg = PlatformConfig::load(&args.config).context("loading platform config")?;
    let file = match &args.strategies {
        Some(p) => {
            StrategyFile::parse(&std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?)
                .with_context(|| format!("parsing {}", p.display()))?
        }
        None => StrategyFile::baselines(),
    };
    let baselines = file.strategy.iter().filter(|s| s.is_baseline()).count();
    let mut runtime = Runtime::new(&file)?;
    runtime.decoder_max_age_ms = (args.decoder_max_age_secs * 1000) as i64;
    if let Some(p) = &args.strategies {
        runtime.population = p.display().to_string();
    }
    let rt = Arc::new(Mutex::new(runtime));
    let cfg = Arc::new(cfg);
    tracing::info!(
        strategies = ?rt.lock().await.strategy_ids(),
        baselines,
        instruments = if file.instruments.is_empty() { "all seen".to_owned() } else { file.instruments.join(",") },
        "strategy-runtime starting (baseline programs are hand-written, not evolved)"
    );

    let bus = Bus::connect(&cfg.nats.url).await.context("connecting to NATS")?;
    let instance_id = ulid::Ulid::generate().to_string();
    let published = Arc::new(Mutex::new(0u64));

    // heartbeat
    {
        let (bus, rt, published, instance_id) = (bus.clone(), rt.clone(), published.clone(), instance_id.clone());
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(args.heartbeat_secs));
            loop {
                tick.tick().await;
                let counters: BTreeMap<String, f64> = {
                    let r = rt.lock().await;
                    let src = r.sources();
                    let mut c = BTreeMap::from([
                        ("strategies".to_owned(), r.strategy_ids().len() as f64),
                        ("paused".to_owned(), r.paused().count() as f64),
                        ("reloads".to_owned(), r.reloads as f64),
                        ("instruments".to_owned(), r.instruments() as f64),
                        ("trades_seen".to_owned(), r.trades_seen as f64),
                        ("latent_seen".to_owned(), r.latent_seen as f64),
                        ("regions_seen".to_owned(), r.graph.seen.regions as f64),
                        ("transitions_seen".to_owned(), r.graph.seen.transitions as f64),
                        ("closes_seen".to_owned(), r.graph.seen.closes as f64),
                        ("regions".to_owned(), r.graph.region_count() as f64),
                        ("transition_rows".to_owned(), r.graph.row_count() as f64),
                        ("decoder_outputs_seen".to_owned(), r.decoders.seen as f64),
                        ("decoder_outputs".to_owned(), r.decoders.len() as f64),
                        ("primitives.graph".to_owned(), src.graph as f64),
                        ("primitives.decoder".to_owned(), src.decoder as f64),
                        ("primitives.fallback".to_owned(), src.fallback as f64),
                        ("signals_emitted".to_owned(), r.signals_emitted as f64),
                        ("signals_published".to_owned(), *published.lock().await as f64),
                    ]);
                    for (id, st) in r.member_stats() {
                        c.insert(format!("strategy.{id}.mean"), st.last_mean);
                        c.insert(format!("strategy.{id}.std"), st.last_std);
                        c.insert(format!("strategy.{id}.steps"), st.steps as f64);
                        c.insert(format!("strategy.{id}.paused"), f64::from(u8::from(r.is_paused(id))));
                        c.insert(format!("strategy.{id}.primitives.graph"), st.sources.graph as f64);
                        c.insert(format!("strategy.{id}.primitives.decoder"), st.sources.decoder as f64);
                        c.insert(format!("strategy.{id}.primitives.fallback"), st.sources.fallback as f64);
                    }
                    c
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
        let mut trades = bus.subscribe::<Envelope>(subjects::ALL_MD_TRADES.into()).await?;
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

    // latent graph feeds (model-runtime)
    {
        let (bus, rt) = (bus.clone(), rt.clone());
        let mut states = bus.subscribe::<LatentState>(subjects::ALL_LATENT_STATES.into()).await?;
        tokio::spawn(async move {
            while let Some((_, st)) = states.next().await {
                rt.lock().await.on_latent(&st);
            }
        });
    }
    {
        let (bus, rt) = (bus.clone(), rt.clone());
        let mut regions = bus.subscribe::<RegionEvent>("latent.region.>".into()).await?;
        tokio::spawn(async move {
            while let Some((_, ev)) = regions.next().await {
                rt.lock().await.on_region(&ev, now_ms());
            }
        });
    }
    {
        let (bus, rt) = (bus.clone(), rt.clone());
        let mut transitions = bus.subscribe::<TransitionUpdate>("latent.transition.>".into()).await?;
        tokio::spawn(async move {
            while let Some((_, tu)) = transitions.next().await {
                rt.lock().await.on_transition(&tu);
            }
        });
    }
    // decoder outputs (evo-worker)
    {
        let (bus, rt) = (bus.clone(), rt.clone());
        let mut outputs = bus.subscribe::<DecoderOutput>("decoder.output.>".into()).await?;
        tokio::spawn(async move {
            while let Some((_, out)) = outputs.next().await {
                rt.lock().await.on_decoder_output(&out, now_ms());
            }
        });
    }
    // control: ctl.strategy.{reload,pause,resume} (request/reply, CtlRequest → CtlReply)
    {
        let (bus, rt, cfg) = (bus.clone(), rt.clone(), cfg.clone());
        let mut ctl = bus.subscribe_raw(subjects::ctl("strategy", "*")).await?;
        tokio::spawn(async move {
            while let Some(msg) = ctl.next().await {
                let verb = msg.subject.rsplit('.').next().unwrap_or("").to_owned();
                let req = match CtlRequest::decode(msg.payload.as_ref()) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!(subject = %msg.subject, error = %e, "undecodable ctl request");
                        continue;
                    }
                };
                let verb = if req.verb.is_empty() { verb } else { req.verb.clone() };
                let reply = handle_ctl(&cfg, &rt, &verb, &req).await;
                if let Some(to) = msg.reply
                    && let Err(e) = bus.client().publish(to, Bytes::from(reply.encode_to_vec())).await
                {
                    tracing::warn!(error = %e, "ctl reply failed");
                }
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
            match bus.publish(subjects::signal_exposure(&d.strategy_id), &proto).await {
                Ok(()) => *published.lock().await += 1,
                Err(e) => {
                    tracing::warn!(strategy = %d.strategy_id, error = %e, "signal publish failed")
                }
            }
        }
    }
}
