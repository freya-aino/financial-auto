//! `trainer serve`: the `JOBS_TRAIN` work-queue worker (spec 21 §4.3, spec 36 §6.7).
//!
//! Pulls `jobs.train.pretrain | generation | adapter` one at a time (`fa_bus::streams::
//! ensure_work_queue("train", "trainer")`), acquires a GPU lease in the kind's class, runs the
//! job ([`crate::run`]) while publishing `JobProgress` and in-progress acks, checkpoints the fit
//! every `--checkpoint-secs` to `fa-ckpt/<job_id>/state.json`, yields on `ctl.gpu.preempt`
//! (`STATUS_PREEMPTED`, the message is NAKed so JetStream redelivers it and the run resumes from
//! the checkpoint), stops at `budget_gpu_ms` (`STATUS_BUDGET_EXCEEDED`) and publishes the
//! `JobResult` on `jobs.result.train.<kind>`.

use std::{
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
    time::{Duration, Instant},
};

use fa_bus::{Bus, async_nats::jetstream::AckKind, streams::ensure_work_queue, subjects};
use fa_proto::{
    Message,
    v1::{Heartbeat, Job, JobProgress, JobResult, job_result::Status},
};
use futures::StreamExt;
use tokio::sync::broadcast;

use crate::{
    fit::FitSpec,
    gpu::{self, Lease, Notice},
    run::{self, AdapterSpec, Control, Kind, Outcome, PREEMPT, RUN, Services},
};

pub const KIND: &str = "train";
pub const CONSUMER: &str = "trainer";
const ACK_PROGRESS_EVERY: Duration = Duration::from_secs(20);
/// Delay before a preempted job is redelivered.
const REDELIVER_AFTER: Duration = Duration::from_secs(5);

pub struct WorkerConfig {
    pub gpu_lease_mb: u32,
    pub checkpoint_secs: u64,
}

pub async fn serve(bus: Bus, svc: Services<'_>, cfg: WorkerConfig) -> anyhow::Result<()> {
    let worker_id = format!("trainer-{}", ulid::Ulid::generate());
    let (_, consumer) = ensure_work_queue(&bus.jetstream(), KIND, CONSUMER).await?;
    let (notices, _) = broadcast::channel(256);
    tokio::spawn(gpu::run_notices(bus.clone(), notices.clone()));
    let counters = Arc::new(std::sync::Mutex::new(std::collections::HashMap::<String, f64>::new()));
    spawn_heartbeat(bus.clone(), worker_id.clone(), counters.clone());
    let bump = |k: &str| {
        if let Ok(mut c) = counters.lock() {
            *c.entry(k.to_owned()).or_insert(0.0) += 1.0;
        }
    };
    tracing::info!(%worker_id, gpu_lease_mb = cfg.gpu_lease_mb, checkpoint_secs = cfg.checkpoint_secs, "pulling jobs from JOBS_TRAIN (jobs.train.>)");
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
        tracing::info!(job = %job.job_id, kind = %job.kind, spec = %String::from_utf8_lossy(&job.spec), budget_ms = job.budget_gpu_ms, checkpoint_ref = %job.checkpoint_ref, "job received");

        let mut keepalive = tokio::time::interval(ACK_PROGRESS_EVERY);
        keepalive.tick().await;
        let handle = handle(&bus, &svc, &cfg, &notices, &job);
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
        bump(&format!("jobs.{}", format!("{:?}", outcome.status).to_ascii_lowercase()));
        let result = JobResult {
            job_id: job.job_id.clone(),
            status: outcome.status as i32,
            result: serde_json::to_vec(&outcome.result).unwrap_or_default(),
            output_refs: outcome.output_refs.clone(),
            checkpoint_ref: outcome.checkpoint_ref.clone(),
            used_gpu_ms: outcome.result.get("elapsed_ms").and_then(serde_json::Value::as_u64).unwrap_or(0),
            worker_id: worker_id.clone(),
            error: outcome.error.clone(),
            objectives: Default::default(),
            directions: Default::default(),
        };
        if let Err(e) = bus.publish(subjects::job_result(&job.kind), &result).await {
            tracing::error!(error = %e, job = %job.job_id, "result publish failed; not acking");
            continue;
        }
        let ack = if outcome.status == Status::Preempted {
            // Leave the job in the queue: the next delivery resumes from the checkpoint.
            msg.ack_with(AckKind::Nak(Some(REDELIVER_AFTER))).await
        } else {
            msg.ack().await
        };
        if let Err(e) = ack {
            tracing::warn!(error = %e, job = %job.job_id, "ack failed");
        }
        tracing::info!(job = %job.job_id, status = ?outcome.status, snapshot = ?outcome.snapshot_id, error = %outcome.error, "job done");
    }
    Ok(())
}

async fn handle(
    bus: &Bus,
    svc: &Services<'_>,
    cfg: &WorkerConfig,
    notices: &broadcast::Sender<Notice>,
    job: &Job,
) -> Outcome {
    let Some(kind) = Kind::parse_job_kind(&job.kind) else {
        return failed(&job.kind, format!("unknown job kind `{}` (train.pretrain|generation|adapter)", job.kind));
    };
    let budget = (job.budget_gpu_ms > 0).then(|| Duration::from_millis(job.budget_gpu_ms));
    let deadline = Instant::now() + budget.unwrap_or(Duration::from_secs(600)) + Duration::from_secs(30);

    // GPU lease in the kind's class; preemption flips the stop flag.
    let mut rx = notices.subscribe();
    let lease = match gpu::acquire(bus, &mut rx, &job.job_id, kind.gpu_class(), cfg.gpu_lease_mb, deadline).await {
        Ok(l) => l,
        Err(e) => return failed(&job.kind, e),
    };
    let stop = Arc::new(AtomicU8::new(RUN));
    let watcher = {
        let stop = stop.clone();
        let id = job.job_id.clone();
        let mut rx = notices.subscribe();
        tokio::spawn(async move {
            while let Ok(n) = rx.recv().await {
                if n == Notice::Preempt(id.clone()) {
                    stop.store(PREEMPT, Ordering::Relaxed);
                    break;
                }
            }
        })
    };
    let progress = {
        let bus = bus.clone();
        let id = job.job_id.clone();
        Box::new(move |fraction: f32, message: String| {
            let bus = bus.clone();
            let p = JobProgress { job_id: id.clone(), fraction, message };
            tokio::spawn(async move {
                let _ = bus.publish(subjects::job_progress(&p.job_id), &p).await;
            });
        })
    };
    let ctl = Control {
        job_id: job.job_id.clone(),
        stop,
        budget,
        checkpoint_every: (cfg.checkpoint_secs > 0).then(|| Duration::from_secs(cfg.checkpoint_secs)),
        checkpoint_ref: (!job.checkpoint_ref.is_empty()).then(|| job.checkpoint_ref.clone()),
        progress,
    };
    let mut outcome = match kind {
        Kind::Adapter => {
            let mut spec: AdapterSpec = parse_spec(&job.spec).unwrap_or_default();
            if spec.snapshot_id.is_none() && !job.snapshot_id.is_empty() {
                spec.snapshot_id = Some(job.snapshot_id.clone());
            }
            run::run_adapter(spec, svc, &ctl).await
        }
        _ => match parse_spec::<FitSpec>(&job.spec) {
            Ok(mut spec) => {
                if spec.parent_snapshot_id.is_none() && !job.snapshot_id.is_empty() {
                    spec.parent_snapshot_id = Some(job.snapshot_id.clone());
                }
                if job.seed != 0 {
                    spec.seed = job.seed;
                }
                run::run_fit(kind, spec, svc, &ctl).await
            }
            Err(e) => failed(&job.kind, format!("spec: {e} (need at least {{\"timescale\":\"1m\"}})")),
        },
    };
    watcher.abort();
    if lease == Lease::Granted {
        gpu::release(bus, &job.job_id).await;
    }
    outcome.result["gpu_lease"] = serde_json::json!({ "class": kind.gpu_class(), "granted": lease == Lease::Granted, "mem_mb": cfg.gpu_lease_mb });
    outcome
}

fn parse_spec<T: serde::de::DeserializeOwned>(spec: &[u8]) -> Result<T, serde_json::Error> {
    let s = if spec.is_empty() { b"{}".as_slice() } else { spec };
    serde_json::from_slice(s)
}

fn failed(kind: &str, e: impl std::fmt::Display) -> Outcome {
    Outcome {
        status: Status::Failed,
        result: serde_json::json!({ "protocol": run::PROTOCOL, "kind": kind, "error": e.to_string() }),
        checkpoint_ref: String::new(),
        output_refs: vec![],
        error: e.to_string(),
        snapshot_id: None,
    }
}

fn spawn_heartbeat(
    bus: Bus,
    worker_id: String,
    counters: Arc<std::sync::Mutex<std::collections::HashMap<String, f64>>>,
) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(5));
        loop {
            tick.tick().await;
            let hb = Heartbeat {
                service: "trainer".into(),
                instance_id: worker_id.clone(),
                at: Some(fa_proto::now()),
                version: env!("CARGO_PKG_VERSION").into(),
                counters: counters.lock().map(|c| c.clone()).unwrap_or_default(),
            };
            let _ = bus.publish(subjects::heartbeat("trainer"), &hb).await;
        }
    });
}
