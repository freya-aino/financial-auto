//! Batch MCMC refits on `jobs.mcmc.refit` (spec 33 §6.2–§6.3) and Garage persistence of the
//! cross-asset tracker (spec 33 §6.5).
//!
//! **Who enqueues, who runs:** the pipeline reports due refits ([`crate::pipeline::RefitRequest`],
//! first once a window holds `refit_min_obs` transitions, then every `batch_refit_interval`);
//! [`McmcJobs::enqueue`] uploads the observation window to Garage
//! (`fa-artifacts/mcmc/<kernel_id>/obs/<job_id>.json`) and publishes `Job{kind: "mcmc.refit"}`
//! on the `JOBS_MCMC` work queue (`Nats-Msg-Id = job_id`). model-runtime's own worker task
//! (durable pull consumer `model-runtime-mcmc`, one job at a time, CPU work on the blocking pool
//! with 2 threads, in-progress acks every 20 s, so live inference keeps its async workers) runs
//! `fa_mcmc::refit::run` (NUTS + Gibbs), writes the posterior samples to
//! `fa-artifacts/mcmc/<kernel_id>/<ts>.bin` and publishes `JobResult` on `jobs.result.mcmc.refit`.
//! The result listener loads the samples and the main loop hands them to the arena. Because the
//! hand-off is Garage + JetStream, the worker can move to any pull worker (evo-worker) unchanged.
//! Undecodable jobs are terminated (→ `DLQ`, spec 21 §8); Garage failures are retried
//! (`max_deliver` 5, then `DLQ`).

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use fa_blob::{BlobStore, ObjectKey, S3BlobStore, buckets};
use fa_bus::{
    Bus,
    async_nats::{HeaderMap, jetstream::AckKind},
    subjects,
};
use fa_mcmc::refit::{self, JOB_KIND, ObsDump, RefitCfg, RefitOutcome, RefitSpec};
use fa_proto::{
    Message,
    v1::{Job, JobResult, job_result::Status},
};
use futures::StreamExt;
use tokio::sync::mpsc;

use crate::pipeline::RefitRequest;

/// Garage key of the persisted cross-asset tracker.
pub const CROSS_STATE_KEY: &str = "mcmc/crossprior/state.json";
const DURABLE: &str = "model-runtime-mcmc";
/// A refit not answered within this long no longer blocks the next one of its space.
const IN_FLIGHT_TIMEOUT: Duration = Duration::from_secs(30 * 60);

pub struct McmcJobs {
    bus: Bus,
    blob: Option<S3BlobStore>,
    in_flight: HashMap<String, Instant>,
    tx: mpsc::Sender<RefitOutcome>,
    pub results: mpsc::Receiver<RefitOutcome>,
    cfg: RefitCfg,
    seed: u64,
}

impl McmcJobs {
    /// Starts the worker and the result listener (both need Garage; without it refits are off).
    pub fn spawn(bus: &Bus, blob: Option<S3BlobStore>) -> Self {
        let (tx, results) = mpsc::channel(16);
        if let Some(b) = &blob {
            tokio::spawn(run_worker(bus.clone(), b.clone()));
            tokio::spawn(run_results(bus.clone(), b.clone(), tx.clone()));
        } else {
            tracing::warn!("no Garage: jobs.mcmc.refit disabled (richer kernels stay on their online fallbacks)");
        }
        Self { bus: bus.clone(), blob, in_flight: HashMap::new(), tx, results, cfg: RefitCfg::default(), seed: 0x3C3C }
    }

    /// Uploads the window and publishes the job (in the background). Skipped while a refit of the
    /// same space is in flight.
    pub fn enqueue(&mut self, req: RefitRequest) {
        let Some(blob) = self.blob.clone() else { return };
        let kernel_id = req.dump.kernel_id.clone();
        if self.in_flight.get(&kernel_id).is_some_and(|t| t.elapsed() < IN_FLIGHT_TIMEOUT) {
            return;
        }
        self.in_flight.insert(kernel_id.clone(), Instant::now());
        self.seed = self.seed.wrapping_add(0x9E37_79B9);
        let job_id = ulid::Ulid::generate().to_string();
        let spec = RefitSpec {
            kernel_id: kernel_id.clone(),
            snapshot_id: req.space.snapshot_id.clone(),
            timescale: req.space.timescale.clone(),
            expert_index: req.space.expert_index,
            obs_ref: refit::obs_key(&kernel_id, &job_id),
            kernels: vec!["logit_cov".into(), "hier_asset".into()],
            seed: self.seed,
            cfg: self.cfg,
        };
        let bus = self.bus.clone();
        tokio::spawn(async move {
            let records = req.dump.records.len();
            let body = match serde_json::to_vec(&req.dump) {
                Ok(b) => b,
                Err(e) => return tracing::warn!(error = %e, "refit dump not serialisable"),
            };
            let key = ObjectKey { bucket: buckets::ARTIFACTS, key: spec.obs_ref.clone() };
            if let Err(e) = blob.put(&key, body, "application/json").await {
                return tracing::warn!(error = %e, kernel = %spec.kernel_id, "refit window upload failed");
            }
            let job = Job {
                job_id: job_id.clone(),
                kind: JOB_KIND.into(),
                priority: 3,
                snapshot_id: spec.snapshot_id.clone(),
                spec: serde_json::to_vec(&spec).unwrap_or_default(),
                input_refs: vec![format!("{}/{}", buckets::ARTIFACTS, spec.obs_ref)],
                seed: spec.seed,
                ..Default::default()
            };
            let mut h = HeaderMap::new();
            h.insert("Nats-Msg-Id", job_id.as_str());
            let sent =
                bus.jetstream().publish_with_headers(subjects::jobs(JOB_KIND), h, job.encode_to_vec().into()).await;
            match sent {
                Ok(ack) => match ack.await {
                    Ok(_) => tracing::info!(job = %job_id, kernel = %spec.kernel_id, records, "mcmc refit enqueued"),
                    Err(e) => tracing::warn!(error = %e, "mcmc refit not acknowledged by JetStream"),
                },
                Err(e) => tracing::warn!(error = %e, "mcmc refit publish failed"),
            }
        });
    }

    /// A result arrived (or failed): the space may refit again.
    pub fn done(&mut self, kernel_id: &str) {
        self.in_flight.remove(kernel_id);
    }

    /// Reloads persisted refits (after a restore) through the result channel.
    pub fn reload(&self, refs: Vec<String>) {
        let Some(blob) = self.blob.clone() else { return };
        let tx = self.tx.clone();
        tokio::spawn(async move {
            for r in refs {
                match load_outcome(&blob, &r).await {
                    Ok(o) => {
                        let _ = tx.send(o).await;
                    }
                    Err(e) => tracing::warn!(key = %r, error = %e, "persisted refit not reloadable"),
                }
            }
        });
    }

    /// Persists the cross-asset tracker (fire and forget).
    pub fn save_cross(&self, state: serde_json::Value) {
        let Some(blob) = self.blob.clone() else { return };
        tokio::spawn(async move {
            let key = ObjectKey { bucket: buckets::ARTIFACTS, key: CROSS_STATE_KEY.into() };
            if let Err(e) = blob.put(&key, serde_json::to_vec(&state).unwrap_or_default(), "application/json").await {
                tracing::warn!(error = %e, "cross-asset tracker not persisted");
            }
        });
    }

    /// The persisted cross-asset tracker, if any.
    pub async fn load_cross(&self) -> Option<serde_json::Value> {
        let blob = self.blob.as_ref()?;
        let key = ObjectKey { bucket: buckets::ARTIFACTS, key: CROSS_STATE_KEY.into() };
        let bytes = tokio::time::timeout(Duration::from_secs(10), blob.get(&key)).await.ok()?.ok()?;
        serde_json::from_slice(&bytes).ok()
    }
}

async fn load_outcome(blob: &S3BlobStore, key: &str) -> Result<RefitOutcome, String> {
    let k = ObjectKey { bucket: buckets::ARTIFACTS, key: key.to_owned() };
    let bytes = blob.get(&k).await.map_err(|e| e.to_string())?;
    let mut o = refit::decode(&bytes)?;
    o.samples_ref = key.to_owned();
    Ok(o)
}

/// `jobs.result.mcmc.refit` → decoded posterior samples for the main loop.
async fn run_results(bus: Bus, blob: S3BlobStore, tx: mpsc::Sender<RefitOutcome>) {
    let Ok(mut sub) = bus.subscribe::<JobResult>(subjects::job_result(JOB_KIND)).await else {
        return tracing::error!("cannot subscribe to jobs.result.mcmc.refit");
    };
    while let Some((_, r)) = sub.next().await {
        if r.status != Status::Ok as i32 {
            tracing::warn!(job = %r.job_id, error = %r.error, "mcmc refit failed");
            continue;
        }
        for key in &r.output_refs {
            match load_outcome(&blob, key).await {
                Ok(o) => {
                    if tx.send(o).await.is_err() {
                        return;
                    }
                }
                Err(e) => tracing::warn!(job = %r.job_id, key, error = %e, "refit samples not loadable"),
            }
        }
    }
}

/// The `JOBS_MCMC` pull worker.
async fn run_worker(bus: Bus, blob: S3BlobStore) {
    let js = bus.jetstream();
    let consumer = loop {
        match fa_bus::streams::ensure_work_queue(&js, "mcmc", DURABLE).await {
            Ok((_, c)) => break c,
            Err(e) => {
                tracing::warn!(error = %e, "JOBS_MCMC not available yet; retrying");
                tokio::time::sleep(Duration::from_secs(15)).await;
            }
        }
    };
    tracing::info!("mcmc refit worker pulling jobs.mcmc.refit (JOBS_MCMC / {DURABLE})");
    let jobs_done = fa_bus::metrics::counter("fa_mcmc_refit_jobs_total", "mcmc.refit jobs by outcome");
    loop {
        let batch = consumer.fetch().max_messages(1).expires(Duration::from_secs(30)).messages().await;
        let mut batch = match batch {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(error = %e, "JOBS_MCMC fetch failed");
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };
        while let Some(Ok(msg)) = batch.next().await {
            let outcome = handle(&bus, &blob, &msg).await;
            jobs_done.inc_with(&[("outcome", outcome)]);
        }
    }
}

async fn handle(bus: &Bus, blob: &S3BlobStore, msg: &fa_bus::async_nats::jetstream::Message) -> &'static str {
    let parsed = Job::decode(msg.payload.as_ref())
        .map_err(|e| e.to_string())
        .and_then(|j| serde_json::from_slice::<RefitSpec>(&j.spec).map(|s| (j, s)).map_err(|e| e.to_string()));
    let (job, spec) = match parsed {
        Ok(x) => x,
        Err(e) => {
            tracing::warn!(error = %e, "undecodable mcmc.refit job: terminated (→ DLQ)");
            let _ = msg.ack_with(AckKind::Term).await;
            return "poison";
        }
    };
    let key = ObjectKey { bucket: buckets::ARTIFACTS, key: spec.obs_ref.clone() };
    let dump: ObsDump = match blob
        .get(&key)
        .await
        .map_err(|e| e.to_string())
        .and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string()))
    {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!(job = %job.job_id, error = %e, "refit window unavailable; retrying");
            let _ = msg.ack_with(AckKind::Nak(Some(Duration::from_secs(30)))).await;
            return "retry";
        }
    };
    let t0 = Instant::now();
    let run_spec = spec.clone();
    let mut task = tokio::task::spawn_blocking(move || refit::run(&dump, &run_spec));
    let mut ticker = tokio::time::interval(Duration::from_secs(20));
    ticker.tick().await;
    let res = loop {
        tokio::select! {
            r = &mut task => break r,
            _ = ticker.tick() => { let _ = msg.ack_with(AckKind::Progress).await; }
        }
    };
    let mut outcome = match res {
        Ok(o) => o,
        Err(e) => {
            publish_result(bus, &job, Status::Failed, serde_json::json!({}), vec![], &e.to_string()).await;
            let _ = msg.ack().await;
            return "failed";
        }
    };
    let samples = refit::samples_key(&spec.kernel_id, outcome.created_ms);
    outcome.samples_ref = samples.clone();
    let k = ObjectKey { bucket: buckets::ARTIFACTS, key: samples.clone() };
    if let Err(e) = blob.put(&k, refit::encode(&outcome), "application/octet-stream").await {
        tracing::warn!(job = %job.job_id, error = %e, "posterior samples upload failed; retrying");
        let _ = msg.ack_with(AckKind::Nak(Some(Duration::from_secs(30)))).await;
        return "retry";
    }
    let summary = outcome.summary();
    tracing::info!(job = %job.job_id, kernel = %spec.kernel_id, secs = t0.elapsed().as_secs_f64(), %summary, "mcmc refit done");
    publish_result(bus, &job, Status::Ok, summary, vec![samples], "").await;
    let _ = msg.ack().await;
    "ok"
}

async fn publish_result(
    bus: &Bus,
    job: &Job,
    status: Status,
    summary: serde_json::Value,
    refs: Vec<String>,
    error: &str,
) {
    let r = JobResult {
        job_id: job.job_id.clone(),
        status: status as i32,
        result: serde_json::to_vec(&summary).unwrap_or_default(),
        output_refs: refs,
        worker_id: DURABLE.into(),
        error: error.into(),
        ..Default::default()
    };
    if let Err(e) = bus.publish(subjects::job_result(JOB_KIND), &r).await {
        tracing::warn!(error = %e, "mcmc refit result publish failed");
    }
}
