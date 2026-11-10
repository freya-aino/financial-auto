//! evo-worker: stateless worker pulling decoder evaluation jobs from the `JOBS_EVO`
//! JetStream work queue (specs 21, 34, 41, 60).
//!
//! For `evo.eval.decoder` jobs it parses the genome (`fa-decoders::genome`), builds a
//! baseline decoder, scores it point-in-time on the live trade window it buffers from
//! `md.trade.>` (`fa-backtest::eval`), and replies with a `JobResult` carrying
//! `{"score", "objectives", ...}` (D-025 compatible) plus `DecoderOutput`s on
//! `decoder.output.<family>`. It publishes `JobProgress`, holds a GPU lease from the arbiter
//! while evaluating (preemption → `STATUS_PREEMPTED`), extends the JetStream ack window
//! with in-progress acks, and acks every message after a result was published. Other job
//! kinds get `STATUS_FAILED` ("not implemented"). Strategy (`evo.eval|vary.strategy`,
//! `evo.eval.strategy.holdout`) and allocator (`evo.eval|vary.allocator`) jobs are handled too;
//! every kind runs under `run::JobRun` (deadline, budget, preempt / cancel-while-running). When `config/secrets.toml` (or `FA_GARAGE_*`)
//! holds Garage credentials, every OK result JSON is also written to
//! `fa-artifacts/evals/<generation>/<job>.json` (`report_ref.persisted`).

mod allocator;
mod data;
mod gpu;
mod holdout;
mod job;
mod latent;
mod market;
mod recorded;
mod run;
mod strategy;
mod trials;
mod vary;
mod vary_strategy;

use std::time::Duration;

use clap::Parser;
use fa_blob::S3BlobStore;
use fa_bus::{
    Bus,
    async_nats::jetstream::{AckKind, consumer::pull, stream},
    subjects,
};
use fa_core::{Secrets, config::PlatformConfig};
use fa_proto::{
    Message,
    v1::{Heartbeat, Job, JobResult, job_result::Status},
};
use futures::StreamExt;
use tokio::sync::broadcast;

const STREAM: &str = "JOBS_EVO";
const CONSUMER: &str = "evo-worker";
const ACK_WAIT: Duration = Duration::from_secs(60);
const ACK_PROGRESS_EVERY: Duration = Duration::from_secs(20);

#[derive(Parser)]
struct Args {
    #[arg(long, env = "FA_CONFIG", default_value = "config/platform.toml")]
    config: String,
    /// GPU memory requested from the arbiter per job (0 disables leasing).
    #[arg(long, env = "FA_EVO_GPU_LEASE_MB", default_value_t = 512)]
    gpu_lease_mb: u32,
    /// Default evaluation window (hours) read from VictoriaMetrics when the job has none.
    #[arg(long, env = "FA_EVO_DATA_WINDOW_HOURS", default_value_t = 6.0)]
    data_window_hours: f64,
    /// Disable VictoriaMetrics reads (buffer only).
    #[arg(long, env = "FA_EVO_NO_TSDB", default_value_t = false)]
    no_tsdb: bool,
    /// Execution cost model shared with the paper venue (YAML); missing file → built-in defaults.
    #[arg(long, env = "FA_COSTS", default_value = fa_backtest::cost::COSTS_YAML_PATH)]
    costs: String,
    /// Disable the Postgres trial registry.
    #[arg(long, env = "FA_EVO_NO_REGISTRY", default_value_t = false)]
    no_registry: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();
    let args = Args::parse();
    fa_bus::metrics::serve_from_env("evo-worker", env!("CARGO_PKG_VERSION")); // GET /metrics on FA_METRICS_BIND (spec 60 §4, D-056)
    let cfg = PlatformConfig::load(&args.config)?;
    // Secrets: `$FA_SECRETS` / config/secrets.toml + FA_GARAGE_* env (D-020). Missing = none.
    let secrets = Secrets::load()?;
    let blob = match S3BlobStore::from_config(&cfg, &secrets) {
        Ok(s) => {
            tracing::info!(endpoint = %cfg.garage.endpoint, "eval reports persist to Garage");
            Some(s)
        }
        Err(fa_blob::BlobError::NotConfigured) => {
            tracing::warn!(
                secrets = %fa_core::secrets::default_path().display(),
                "no Garage credentials; eval reports are not persisted"
            );
            None
        }
        Err(e) => {
            tracing::warn!(error = %e, "Garage client unavailable; eval reports are not persisted");
            None
        }
    };
    let bus = Bus::connect(&cfg.nats.url).await?;
    let worker_id = format!("evo-worker-{}", ulid::Ulid::generate());

    let js = bus.jetstream();
    let stream = js
        .get_or_create_stream(stream::Config {
            name: STREAM.into(),
            subjects: vec!["jobs.evo.>".into()],
            retention: stream::RetentionPolicy::WorkQueue,
            ..Default::default()
        })
        .await?;
    let consumer_cfg =
        pull::Config { durable_name: Some(CONSUMER.into()), ack_wait: ACK_WAIT, max_deliver: 5, ..Default::default() };
    // `create_consumer` updates an existing durable's config; fall back for older servers.
    let consumer = match stream.create_consumer(consumer_cfg.clone()).await {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "consumer create/update failed; using existing config");
            stream.get_or_create_consumer(CONSUMER, consumer_cfg).await?
        }
    };
    fa_bus::streams::spawn_consumer_lag(consumer.clone()); // fa_bus_consumer_num_pending{stream,consumer} (spec 60 §6.5)

    let market = market::MarketBuffer::default();
    tokio::spawn(market.clone().run(bus.clone()));
    let latents = latent::LatentBuffer::default();
    tokio::spawn(latents.clone().run(bus.clone()));
    let tsdb = if args.no_tsdb {
        None
    } else {
        let vm_url = std::env::var("FA_VM_URL")
            .ok()
            .filter(|u| !u.is_empty())
            .unwrap_or_else(|| cfg.victoriametrics.url.clone());
        let t = fa_tsdb::Tsdb::new(vm_url);
        match t.health().await {
            Ok(()) => tracing::info!(
                url = t.base_url(),
                "VictoriaMetrics reachable; evaluation data comes from fa_trade_price"
            ),
            Err(e) => {
                tracing::warn!(url = t.base_url(), error = %e, "VictoriaMetrics unreachable at startup; jobs fall back to the md.trade.> buffer until it answers")
            }
        }
        Some(t)
    };
    let hub = Some(fa_hub::HubClient::new(fa_hub::HubClient::url_from_env_or(&cfg.helix.url)));
    // Trial registry (spec 41 §6.5): validation.trial / generation_stats / holdout.
    let trials = if args.no_registry {
        None
    } else {
        match trials::TrialStore::connect(&cfg.postgres.url).await {
            Some(t) if t.ready().await => {
                tracing::info!("trial registry: postgres validation.* ready");
                Some(t)
            }
            Some(_) => {
                tracing::warn!(
                    "postgres reachable but validation.* tables missing (start control-api to migrate); registry disabled"
                );
                None
            }
            None => None,
        }
    };
    // Cost model (spec 41 §6.3): same file as the paper venue.
    let cost = std::sync::Arc::new(match std::fs::read_to_string(&args.costs) {
        Ok(src) => match fa_backtest::cost::CostModel::from_yaml(&src) {
            Ok(m) => {
                tracing::info!(path = %args.costs, "cost model loaded");
                m
            }
            Err(e) => {
                tracing::warn!(path = %args.costs, error = %e, "cost model unreadable; using built-in defaults");
                fa_backtest::cost::CostModel::default()
            }
        },
        Err(e) => {
            tracing::warn!(path = %args.costs, error = %e, "no cost model file; using built-in defaults");
            fa_backtest::cost::CostModel::default()
        }
    });
    let (notices, _) = broadcast::channel(256);
    tokio::spawn(gpu::run_notices(bus.clone(), notices.clone()));

    let counters = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::<String, f64>::new()));
    let recorded = recorded::RecordedStore::from_env();
    {
        let bus = bus.clone();
        let worker_id = worker_id.clone();
        let market = market.clone();
        let latents = latents.clone();
        let counters = counters.clone();
        let recorded = recorded.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(5));
            loop {
                tick.tick().await;
                let (inst, trades) = market.stats();
                let (linst, lpoints) = latents.stats();
                let mut c = counters.lock().map(|c| c.clone()).unwrap_or_default();
                c.insert("market.instruments".into(), inst as f64);
                c.insert("market.trades".into(), trades as f64);
                c.insert("latents.instruments".into(), linst as f64);
                c.insert("latents.points".into(), lpoints as f64);
                recorded.export(&mut c); // latents.cache_hit|extend|miss, latents.messages_read, signals.*
                let hb = Heartbeat {
                    service: "evo-worker".into(),
                    instance_id: worker_id.clone(),
                    at: Some(fa_proto::now()),
                    version: env!("CARGO_PKG_VERSION").into(),
                    counters: c,
                };
                let _ = bus.publish(subjects::heartbeat("evo-worker"), &hb).await;
            }
        });
    }
    let add = |k: &str, v: f64| {
        if let Ok(mut c) = counters.lock() {
            *c.entry(k.to_owned()).or_insert(0.0) += v;
        }
    };
    let bump = |k: &str| add(k, 1.0);

    let worker = job::Worker {
        bus: bus.clone(),
        worker_id: worker_id.clone(),
        market,
        notices,
        gpu_lease_mb: args.gpu_lease_mb,
        tsdb,
        data_window_hours: args.data_window_hours,
        latents,
        hub,
        blob,
        trials,
        cost,
        recorded,
    };
    tracing::info!(%worker_id, gpu_lease_mb = args.gpu_lease_mb, "pulling jobs from {STREAM}");
    // One message at a time so idle workers do not hoard jobs other replicas could take.
    let mut messages = consumer.stream().max_messages_per_batch(1).messages().await?;
    while let Some(msg) = messages.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(error = %e, "pull error");
                continue;
            }
        };
        let job = match Job::decode(msg.payload.as_ref()) {
            Ok(j) => j,
            Err(e) => {
                tracing::warn!(error = %e, "undecodable job; acking to drop");
                bump("jobs.undecodable");
                let _ = msg.ack().await;
                continue;
            }
        };
        tracing::info!(job = %job.job_id, kind = %job.kind, genome = %job.genome_id, budget_ms = job.budget_gpu_ms, "job received");

        // Keep the ack window open while the job runs.
        let mut keepalive = tokio::time::interval(ACK_PROGRESS_EVERY);
        keepalive.tick().await;
        let handle = worker.handle(&job);
        tokio::pin!(handle);
        let outcome = loop {
            tokio::select! {
                o = &mut handle => break o,
                _ = keepalive.tick() => {
                    if let Err(e) = msg.ack_with(AckKind::Progress).await {
                        tracing::debug!(error = %e, "in-progress ack failed");
                    }
                }
            }
        };

        let result = outcome.result;
        let status = Status::try_from(result.status).unwrap_or(Status::Failed);
        let status_name = status_name(status);
        bump(&format!("jobs.{status_name}"));
        // per kind (spec 60 §6.3): jobs.<kind>.ok|failed|preempted|cancelled|budget_exceeded, jobs.<kind>.ms (total)
        let kind = if worker::KINDS.contains(&job.kind.as_str()) { job.kind.as_str() } else { "other" };
        bump(&format!("jobs.{kind}.{status_name}"));
        add(&format!("jobs.{kind}.ms"), result.used_gpu_ms as f64);
        for out in &outcome.outputs {
            if let Err(e) = bus.publish(subjects::decoder_output(&out.family), out).await {
                tracing::warn!(error = %e, "decoder output publish failed");
            }
        }
        if let Err(e) = publish_result(&bus, &job.kind, &result).await {
            // Never lose the job: leave it unacked so JetStream redelivers it.
            tracing::error!(error = %e, job = %job.job_id, "result publish failed; not acking");
            continue;
        }
        if let Err(e) = msg.ack().await {
            tracing::warn!(error = %e, job = %job.job_id, "ack failed");
        }
        tracing::info!(job = %job.job_id, ?status, used_ms = result.used_gpu_ms, error = %result.error, outputs = outcome.outputs.len(), "job done");
    }
    Ok(())
}

/// Counter suffix of a status.
fn status_name(s: Status) -> &'static str {
    match s {
        Status::Ok => "ok",
        Status::Failed => "failed",
        Status::Preempted => "preempted",
        Status::BudgetExceeded => "budget_exceeded",
        Status::Cancelled => "cancelled",
    }
}

mod worker {
    pub const KINDS: [&str; 7] = crate::job::Worker::KINDS;
}

async fn publish_result(bus: &Bus, kind: &str, result: &JobResult) -> anyhow::Result<()> {
    bus.publish(subjects::job_result(kind), result).await?;
    Ok(())
}
