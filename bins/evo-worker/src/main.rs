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
//! kinds get `STATUS_FAILED` ("not implemented").

mod gpu;
mod job;
mod market;

use std::time::Duration;

use clap::Parser;
use fa_bus::{
    Bus,
    async_nats::jetstream::{AckKind, consumer::pull, stream},
    subjects,
};
use fa_core::config::PlatformConfig;
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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();
    let args = Args::parse();
    let cfg = PlatformConfig::load(&args.config)?;
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

    let market = market::MarketBuffer::default();
    tokio::spawn(market.clone().run(bus.clone()));
    let (notices, _) = broadcast::channel(256);
    tokio::spawn(gpu::run_notices(bus.clone(), notices.clone()));

    let counters = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::<String, f64>::new()));
    {
        let bus = bus.clone();
        let worker_id = worker_id.clone();
        let market = market.clone();
        let counters = counters.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(5));
            loop {
                tick.tick().await;
                let (inst, trades) = market.stats();
                let mut c = counters.lock().map(|c| c.clone()).unwrap_or_default();
                c.insert("market.instruments".into(), inst as f64);
                c.insert("market.trades".into(), trades as f64);
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
    let bump = |k: &str| {
        if let Ok(mut c) = counters.lock() {
            *c.entry(k.to_owned()).or_insert(0.0) += 1.0;
        }
    };

    let worker = job::Worker {
        bus: bus.clone(),
        worker_id: worker_id.clone(),
        market,
        notices,
        gpu_lease_mb: args.gpu_lease_mb,
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
        bump(&format!("jobs.{}", format!("{status:?}").to_ascii_lowercase()));
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

async fn publish_result(bus: &Bus, kind: &str, result: &JobResult) -> anyhow::Result<()> {
    bus.publish(subjects::job_result(kind), result).await?;
    Ok(())
}
