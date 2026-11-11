//! strategy-runtime: runs a population of GP strategies live and publishes
//! `ExposureDistribution`s on `signal.exposure.<strategy_id>` (spec 35, spec 21 §4.1).
//!
//! Inputs: `md.trade.>` (prices, always), `latent.state.>` / `latent.region.>` /
//! `latent.transition.>` (the latent graph from model-runtime) and `decoder.output.>`
//! (evo-worker's `rnn_prior` / `retrieval`, model-runtime's `text.*` / `visual.*`). Programs are
//! evaluated through `fa_strategies::features::LiveContext`: graph and decoder primitives use
//! the real data when the bus has it for the strategy's timescale and instrument, price-only
//! fallbacks otherwise; the heartbeat reports how much was real
//! (`primitives.graph|decoder|fallback`). Retrieval analogs older than the in-memory history
//! are priced from VictoriaMetrics between steps (`prefetch`). Spec 35 §8 failure modes
//! (step budget, NaN outputs, starved primitives) are handled in `runtime`.
//!
//! Population: `--strategies` = a local file, `garage:fa-artifacts/strategies/<gen>.toml`,
//! `latest` (newest object under `fa-artifacts/strategies/`), `baselines` (the hand-written
//! programs, not evolved) or nothing / `managed` (default: the baselines plus the managed live
//! set `fa-artifacts/live/strategies.toml` the risk-engine pipeline writes when an evolved
//! strategy passes the locked holdout, spec 35 §6.4); any failure at start falls back to the
//! baselines with a warning. Control: `ctl.strategy.reload|pause|resume|status`; the pipeline
//! reloads with `{artifact_key: "managed"}`.
mod prefetch;
mod runtime;
mod source;

use std::{path::PathBuf, sync::Arc, time::Duration};

use anyhow::Context;
use bytes::Bytes;
use clap::Parser;
use fa_blob::S3BlobStore;
use fa_bus::{Bus, subjects};
use fa_core::{InstrumentId, config::PlatformConfig, secrets::Secrets};
use fa_proto::{
    Message,
    v1::{
        CtlReply, CtlRequest, DecoderOutput, Envelope, Heartbeat, LatentState, RegionEvent, TransitionUpdate,
        envelope::Payload,
    },
};
use futures::StreamExt;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    runtime::{Runtime, Settings},
    source::PopulationSpec,
};

const SERVICE: &str = "strategy-runtime";

#[derive(Debug, Parser)]
#[command(about = "live GP strategy population → signal.exposure.*")]
struct Args {
    /// Platform config (spec 60).
    #[arg(long, env = "FA_CONFIG", default_value = "config/platform.toml")]
    config: PathBuf,
    /// Population: a TOML file, `garage:fa-artifacts/strategies/<gen>.toml`, `latest`,
    /// `baselines` or `managed` (default: baselines + the pipeline's live set). Falls back to the
    /// baselines with a warning.
    #[arg(long, env = "FA_STRATEGIES")]
    strategies: Option<String>,
    /// Scheduler resolution in milliseconds.
    #[arg(long, default_value_t = 250)]
    tick_ms: u64,
    #[arg(long, default_value_t = 2)]
    heartbeat_secs: u64,
    /// Decoder outputs older than this are ignored by the decoder primitives.
    #[arg(long, env = "FA_STRATEGY_DECODER_MAX_AGE_SECS", default_value_t = 6 * 3600)]
    decoder_max_age_secs: u64,
    /// Trades kept in memory per instrument (bars, analog outcomes).
    #[arg(long, env = "FA_STRATEGY_PRICE_HISTORY", default_value_t = 100_000)]
    price_history: usize,
    /// Per-strategy evaluation budget per step (spec 35 §8); slower steps are skipped.
    #[arg(long, env = "FA_STRATEGY_STEP_BUDGET_MS", default_value_t = 50)]
    step_budget_ms: u64,
    /// Consecutive budget breaches before a strategy is auto-paused (N).
    #[arg(long, env = "FA_STRATEGY_BREACH_LIMIT", default_value_t = 5)]
    breach_limit: u32,
    /// Consecutive steps with NaN/inf output before a strategy is auto-paused (N).
    #[arg(long, env = "FA_STRATEGY_NAN_LIMIT", default_value_t = 5)]
    nan_limit: u32,
    /// Minutes a referenced primitive may answer only from the fallback before `starved=1` (M).
    #[arg(long, env = "FA_STRATEGY_STARVED_AFTER_MINS", default_value_t = 10.0)]
    starved_after_mins: f64,
    /// Analog price cache entries.
    #[arg(long, env = "FA_STRATEGY_ANALOG_CACHE", default_value_t = 4_096)]
    analog_cache: usize,
    /// Analogs fetched from VictoriaMetrics per prefetch round.
    #[arg(long, default_value_t = 32)]
    analog_prefetch_batch: usize,
    /// Prefetch round interval.
    #[arg(long, default_value_t = 500)]
    analog_prefetch_ms: u64,
    /// VictoriaMetrics base URL (defaults to `[victoriametrics] url` of the platform config).
    #[arg(long, env = "FA_VM_URL")]
    vm_url: Option<String>,
}

impl Args {
    fn settings(&self) -> Settings {
        Settings {
            price_history: self.price_history.max(2),
            decoder_max_age_ms: (self.decoder_max_age_secs * 1000) as i64,
            step_budget: Duration::from_millis(self.step_budget_ms.max(1)),
            budget_breach_limit: self.breach_limit.max(1),
            nan_breach_limit: self.nan_limit.max(1),
            starved_after_ms: (self.starved_after_mins.max(0.0) * 60_000.0) as i64,
            analog_cache_max: self.analog_cache.max(16),
            ..Settings::default()
        }
    }
}

/// Garage client, when credentials are configured.
fn blob_store(cfg: &PlatformConfig) -> Option<S3BlobStore> {
    let secrets = Secrets::load().map_err(|e| tracing::warn!(error = %e, "no secrets; Garage unavailable")).ok()?;
    S3BlobStore::from_config(cfg, &secrets)
        .map_err(|e| tracing::warn!(error = %e, "Garage credentials missing; Garage populations unavailable"))
        .ok()
}

/// Population spec of a reload request: `path`, or `artifact_key` (a key or `latest`).
fn reload_spec(args: &std::collections::HashMap<String, String>) -> anyhow::Result<PopulationSpec> {
    if let Some(path) = args.get("path").filter(|p| !p.is_empty()) {
        return Ok(PopulationSpec::Path(path.into()));
    }
    match args.get("artifact_key").map(|k| k.trim()).filter(|k| !k.is_empty()) {
        Some("latest") => Ok(PopulationSpec::Latest),
        Some("managed") => Ok(PopulationSpec::Managed),
        Some("baselines") => Ok(PopulationSpec::Baselines),
        Some(k) => Ok(PopulationSpec::parse(&format!("garage:{k}"))),
        None => anyhow::bail!(
            "invalid: reload needs `artifact_key` (Garage key under fa-artifacts/, `latest`, `managed` or `baselines`) or `path`"
        ),
    }
}

/// Handles one `ctl.strategy.<verb>` request.
async fn handle_ctl(blob: Option<&S3BlobStore>, rt: &Mutex<Runtime>, verb: &str, req: &CtlRequest) -> CtlReply {
    let reply = |ok: bool, message: String| CtlReply { ok, message };
    match verb {
        "reload" => {
            let loaded = match reload_spec(&req.args) {
                Ok(spec) => source::load(&spec, blob).await,
                Err(e) => Err(e),
            };
            match loaded {
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
                    tracing::warn!(error = %format!("{e:#}"), "reload failed");
                    reply(false, format!("{e:#}"))
                }
            }
        }
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
        "status" => reply(true, rt.lock().await.status(now_ms()).to_string()),
        other => reply(false, format!("invalid: unknown verb `{other}` (reload | pause | resume | status)")),
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
    let blob = Arc::new(blob_store(&cfg));
    let spec = PopulationSpec::parse(args.strategies.as_deref().unwrap_or("managed"));
    let (file, label, warning) = source::load_or_baselines(&spec, blob.as_ref().as_ref()).await;
    if let Some(w) = &warning {
        tracing::warn!("{w}");
    }
    let baselines = file.strategy.iter().filter(|s| s.is_baseline()).count();
    let settings = args.settings();
    let mut runtime = Runtime::with_settings(&file, settings)?;
    runtime.population = label.clone();
    let rt = Arc::new(Mutex::new(runtime));
    let vm = fa_tsdb::Tsdb::new(args.vm_url.clone().unwrap_or_else(|| cfg.victoriametrics.url.clone()));
    let cfg = Arc::new(cfg);
    tracing::info!(
        population = %label,
        strategies = ?rt.lock().await.strategy_ids(),
        baselines,
        instruments = if file.instruments.is_empty() { "all seen".to_owned() } else { file.instruments.join(",") },
        ?settings,
        vm = vm.base_url(),
        "strategy-runtime starting (baseline programs are hand-written, not evolved)"
    );

    let bus = Bus::connect(&cfg.nats.url).await.context("connecting to NATS")?;
    let instance_id = ulid::Ulid::generate().to_string();
    let published = Arc::new(Mutex::new(0u64));

    let _ = &cfg;
    // heartbeat
    {
        let (bus, rt, published, instance_id) = (bus.clone(), rt.clone(), published.clone(), instance_id.clone());
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(args.heartbeat_secs));
            loop {
                tick.tick().await;
                let mut counters = rt.lock().await.heartbeat_counters(now_ms());
                counters.insert("signals_published".to_owned(), *published.lock().await as f64);
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
    // analog price prefetch from VictoriaMetrics (between steps; D-063)
    {
        let rt = rt.clone();
        let (batch, every) =
            (args.analog_prefetch_batch.max(1), Duration::from_millis(args.analog_prefetch_ms.max(50)));
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(every);
            loop {
                tick.tick().await;
                prefetch::prefetch_once(&rt, &vm, batch, now_ms()).await;
            }
        });
    }
    // control: ctl.strategy.{reload,pause,resume,status} (request/reply, CtlRequest → CtlReply)
    {
        let (bus, rt, blob) = (bus.clone(), rt.clone(), blob.clone());
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
                let reply = handle_ctl(blob.as_ref().as_ref(), &rt, &verb, &req).await;
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
