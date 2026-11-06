//! evo-worker: stateless worker pulling decoder/strategy evaluation jobs from the
//! `JOBS_EVO` JetStream work queue (specs 21, 34, 60).
//!
//! Scaffold behaviour: replies `STATUS_FAILED` ("not implemented") for each job and acks it.

use std::time::Duration;

use clap::Parser;
use fa_bus::{
    Bus,
    async_nats::jetstream::{consumer::pull, stream},
    subjects,
};
use fa_core::config::PlatformConfig;
use fa_proto::{
    Message,
    v1::{Heartbeat, Job, JobResult, job_result::Status},
};
use futures::StreamExt;

const STREAM: &str = "JOBS_EVO";
const CONSUMER: &str = "evo-worker";

#[derive(Parser)]
struct Args {
    #[arg(long, env = "FA_CONFIG", default_value = "config/platform.toml")]
    config: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?)).init();
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
    let consumer = stream
        .get_or_create_consumer(CONSUMER, pull::Config { durable_name: Some(CONSUMER.into()), ..Default::default() })
        .await?;

    {
        let bus = bus.clone();
        let worker_id = worker_id.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(5));
            loop {
                tick.tick().await;
                let hb = Heartbeat {
                    service: "evo-worker".into(),
                    instance_id: worker_id.clone(),
                    at: Some(fa_proto::now()),
                    version: env!("CARGO_PKG_VERSION").into(),
                    counters: Default::default(),
                };
                let _ = bus.publish(subjects::heartbeat("evo-worker"), &hb).await;
            }
        });
    }

    tracing::info!(%worker_id, "pulling jobs from {STREAM}");
    let mut messages = consumer.messages().await?;
    while let Some(msg) = messages.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(error = %e, "pull error");
                continue;
            }
        };
        let (job_id, kind) = match Job::decode(msg.payload.as_ref()) {
            Ok(j) => (j.job_id, j.kind),
            Err(e) => {
                tracing::warn!(error = %e, "undecodable job; acking to drop");
                (String::new(), "unknown".into())
            }
        };
        let result = JobResult {
            job_id: job_id.clone(),
            status: Status::Failed as i32,
            error: format!("job kind `{kind}` not implemented"),
            worker_id: worker_id.clone(),
            ..Default::default()
        };
        bus.publish(subjects::job_result(&kind), &result).await?;
        if let Err(e) = msg.ack().await {
            tracing::warn!(error = %e, %job_id, "ack failed");
        }
        tracing::info!(%job_id, %kind, "replied not-implemented");
    }
    Ok(())
}
