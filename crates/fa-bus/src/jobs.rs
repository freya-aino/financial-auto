//! Work-queue job loop (spec 21 §4.3, §6 "Job lifecycle", criteria 2–3).
//!
//! The horizontal-scaling contract: any number of stateless workers share one durable pull
//! consumer of a `JOBS_<kind>` work-queue stream and pull **one job at a time**, so an idle worker
//! never hoards jobs another replica could run. Per job: decode (undecodable → `Term`, dead
//! letter), run the handler while sending `in-progress` acks, publish the `JobResult` through
//! JetStream with `Nats-Msg-Id = `[`result_msg_id`] and wait for the stream's ack, then ack the
//! job (double ack: the server confirms). A worker that dies before the result is stored leaves
//! the job unacked → redelivered after `ack_wait`; one that dies after storing the result but
//! before the ack makes the next worker publish an identical result, which the results stream
//! drops as a duplicate within its dedup window (2 min by default) — beyond that, result consumers
//! are idempotent by `job_id` (control-api's `ops.job_run` keeps the newest stream position).
//! Hence "exactly once in effect" (`live_exactly_once_across_worker_crashes`).
use std::{future::Future, sync::Arc, time::Duration};

use async_nats::{
    HeaderMap,
    jetstream::{self, AckKind, consumer::pull},
};
use bytes::Bytes;
use fa_proto::{
    Message,
    v1::{Job, JobResult},
};
use futures::StreamExt;

#[derive(Debug, thiserror::Error)]
pub enum JobsError {
    #[error("publish result: {0}")]
    Publish(#[from] jetstream::context::PublishError),
    #[error("pull: {0}")]
    Pull(String),
}

/// Dedup id of a result: one per job, status and checkpoint, so a redelivered job that ends the
/// same way is a duplicate, while `PREEMPTED` (with its checkpoint) and the final `OK` of the
/// resumed job are distinct messages.
pub fn result_msg_id(r: &JobResult) -> String {
    format!("result:{}:{}:{}", r.job_id, r.status, r.checkpoint_ref)
}

/// Publishes `result` on `subject` through JetStream (waits for the stream's ack) with
/// `Nats-Msg-Id = `[`result_msg_id`]. Returns `true` when the stream already had it (duplicate).
pub async fn publish_result(js: &jetstream::Context, subject: String, result: &JobResult) -> Result<bool, JobsError> {
    let mut h = HeaderMap::new();
    h.insert("Nats-Msg-Id", result_msg_id(result).as_str());
    let ack = js.publish_with_headers(subject, h, Bytes::from(result.encode_to_vec())).await?.await?;
    Ok(ack.duplicate)
}

/// Worker settings.
#[derive(Clone)]
pub struct WorkerConfig {
    pub worker_id: String,
    /// In-progress ack cadence while a job runs (well under the consumer's `ack_wait`).
    pub progress_every: Duration,
    /// Result subject for a job kind; services use [`crate::subjects::job_result`].
    pub result_subject: Arc<dyn Fn(&str) -> String + Send + Sync>,
}

impl WorkerConfig {
    pub fn new(worker_id: impl Into<String>) -> Self {
        Self {
            worker_id: worker_id.into(),
            progress_every: Duration::from_secs(20),
            result_subject: Arc::new(|kind: &str| crate::subjects::job_result(kind)),
        }
    }
}

/// What happened to one pulled message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Processed {
    /// Result stored (or found already stored) and the job acked.
    Done { duplicate: bool },
    /// Undecodable or of an unknown contract version: terminated (→ dead letter).
    Rejected,
    /// The result could not be stored: left unacked for redelivery.
    Retry,
}

/// Runs one pulled job message through `handler` (module docs).
pub async fn process<H, Fut>(js: &jetstream::Context, cfg: &WorkerConfig, msg: jetstream::Message, handler: &H) -> Processed
where
    H: Fn(Job) -> Fut,
    Fut: Future<Output = JobResult>,
{
    if !crate::contract::accept(&msg.subject, msg.headers.as_ref()) {
        let _ = msg.ack_with(AckKind::Term).await;
        return Processed::Rejected;
    }
    let job = match Job::decode(msg.payload.as_ref()) {
        Ok(j) => j,
        Err(e) => {
            tracing::warn!(subject = %msg.subject, error = %e, "undecodable job; terminating");
            let _ = msg.ack_with(AckKind::Term).await;
            return Processed::Rejected;
        }
    };
    let kind = job.kind.clone();
    let job_id = job.job_id.clone();
    let mut keepalive = tokio::time::interval(cfg.progress_every);
    keepalive.tick().await;
    let run = handler(job);
    tokio::pin!(run);
    let mut result = loop {
        tokio::select! {
            r = &mut run => break r,
            _ = keepalive.tick() => { let _ = msg.ack_with(AckKind::Progress).await; }
        }
    };
    if result.job_id.is_empty() {
        result.job_id = job_id;
    }
    if result.worker_id.is_empty() {
        result.worker_id = cfg.worker_id.clone();
    }
    match publish_result(js, (cfg.result_subject)(&kind), &result).await {
        Ok(duplicate) => {
            if let Err(e) = msg.double_ack().await {
                tracing::warn!(job = %result.job_id, error = %e, "job ack failed (result stored; a redelivery is a duplicate)");
            }
            Processed::Done { duplicate }
        }
        Err(e) => {
            tracing::error!(job = %result.job_id, error = %e, "result not stored; leaving the job for redelivery");
            Processed::Retry
        }
    }
}

/// Pulls jobs one at a time from `consumer` until the stream of messages ends.
pub async fn run_worker<H, Fut>(
    consumer: jetstream::consumer::Consumer<pull::Config>,
    js: jetstream::Context,
    cfg: WorkerConfig,
    handler: H,
) -> Result<(), JobsError>
where
    H: Fn(Job) -> Fut,
    Fut: Future<Output = JobResult>,
{
    let mut messages = consumer
        .stream()
        .max_messages_per_batch(1)
        .messages()
        .await
        .map_err(|e| JobsError::Pull(e.to_string()))?;
    while let Some(msg) = messages.next().await {
        match msg {
            Ok(m) => {
                process(&js, &cfg, m, &handler).await;
            }
            Err(e) => tracing::warn!(error = %e, "pull error"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        sync::{
            Mutex,
            atomic::{AtomicUsize, Ordering},
        },
        time::Instant,
    };

    use async_nats::jetstream::stream;

    use super::*;

    #[test]
    fn result_ids_separate_preemption_from_completion() {
        let ok = JobResult { job_id: "j1".into(), status: 0, ..Default::default() };
        let pre = JobResult { job_id: "j1".into(), status: 2, checkpoint_ref: "ck/1".into(), ..Default::default() };
        assert_eq!(result_msg_id(&ok), "result:j1:0:");
        assert_ne!(result_msg_id(&ok), result_msg_id(&pre));
        assert_eq!(result_msg_id(&ok), result_msg_id(&JobResult { worker_id: "other".into(), ..ok.clone() }));
    }

    async fn scratch(tag: &str, ack_wait: Duration) -> Option<(async_nats::Client, jetstream::Context, String, String)> {
        if std::env::var("FA_NATS_TEST").is_err() {
            eprintln!("skipping live JetStream test (set FA_NATS_TEST=1)");
            return None;
        }
        let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
        let client = async_nats::connect(&url).await.expect("connect");
        let js = jetstream::new(client.clone());
        let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() % 1_000_000_000;
        let (jobs, results) = (format!("FABUS{tag}{n}"), format!("FABUS{tag}{n}_RES"));
        // test-scoped subjects outside every spec stream (`jobs.result.>` would reach JOB_RESULTS)
        let root = format!("fabustest.{}.{n}", tag.to_ascii_lowercase());
        js.create_stream(stream::Config {
            name: jobs.clone(),
            subjects: vec![format!("{root}.jobs.>")],
            retention: stream::RetentionPolicy::WorkQueue,
            ..Default::default()
        })
        .await
        .unwrap();
        js.create_stream(stream::Config {
            name: results.clone(),
            subjects: vec![format!("{root}.result.>")],
            duplicate_window: Duration::from_secs(120),
            ..Default::default()
        })
        .await
        .unwrap();
        let mut cfg = crate::streams::work_queue_consumer("workers");
        cfg.ack_wait = ack_wait;
        js.get_stream(&jobs).await.unwrap().create_consumer(cfg).await.unwrap();
        Some((client, js, root, jobs))
    }

    async fn publish_jobs(js: &jetstream::Context, root: &str, n: usize) -> Vec<String> {
        let mut ids = Vec::new();
        for i in 0..n {
            let id = format!("job-{i:03}");
            let job = Job { job_id: id.clone(), kind: "fake".into(), seed: i as u64, ..Default::default() };
            let mut h = HeaderMap::new();
            h.insert("Nats-Msg-Id", id.as_str());
            js.publish_with_headers(format!("{root}.jobs.fake"), h, job.encode_to_vec().into()).await.unwrap().await.unwrap();
            ids.push(id);
        }
        ids
    }

    async fn worker(
        url: &str,
        jobs: &str,
        root: &str,
        id: &str,
    ) -> (async_nats::Client, jetstream::Context, jetstream::consumer::Consumer<pull::Config>, WorkerConfig) {
        let client = async_nats::connect(url).await.unwrap(); // own connection: a crash drops it
        let js = jetstream::new(client.clone());
        let consumer = js.get_stream(jobs).await.unwrap().get_consumer::<pull::Config>("workers").await.unwrap();
        let root = root.to_owned();
        let cfg = WorkerConfig {
            worker_id: id.into(),
            progress_every: Duration::from_millis(500),
            result_subject: Arc::new(move |kind: &str| format!("{root}.result.{kind}")),
        };
        (client, js, consumer, cfg)
    }

    /// CPU-bound fake job: a deterministic hash chain of `iters` rounds.
    fn burn(seed: u64, iters: u64) -> u64 {
        let mut x = seed ^ 0x9E37_79B9_7F4A_7C15;
        for _ in 0..iters {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
        }
        x
    }

    async fn results(js: &jetstream::Context, results_stream: &str) -> Vec<JobResult> {
        let recs = crate::replay::read_range(
            js,
            results_stream,
            ">",
            0,
            i64::MAX / 2,
            &crate::replay::ReadOptions { idle: Duration::from_millis(500), ..Default::default() },
        )
        .await
        .unwrap();
        recs.iter().map(|r| JobResult::decode(r.payload.as_ref()).unwrap()).collect()
    }

    /// Spec 21 criterion 2 against the compose NATS (test-scoped streams, deleted afterwards):
    /// `FA_NATS_TEST=1 cargo test -p fa-bus live_exactly_once -- --nocapture`. One worker dies
    /// after storing a result but before acking; another dies mid-job; two healthy workers then
    /// drain the queue. Every job ends with exactly one stored result, though handlers ran more
    /// often than there are jobs.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn live_exactly_once_across_worker_crashes() {
        let Some((_c, js, root, jobs)) = scratch("XO", Duration::from_secs(2)).await else { return };
        let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
        let ids = publish_jobs(&js, &root, 20).await;
        let runs = Arc::new(AtomicUsize::new(0));
        let effect = |runs: Arc<AtomicUsize>| {
            move |job: Job| {
                let runs = runs.clone();
                async move {
                    runs.fetch_add(1, Ordering::SeqCst);
                    let v = burn(job.seed, 200_000);
                    JobResult { job_id: job.job_id, result: v.to_le_bytes().to_vec(), ..Default::default() }
                }
            }
        };

        // 1. crash after the result is stored, before the ack
        {
            let (client, wjs, consumer, cfg) = worker(&url, &jobs, &root, "crash-after-publish").await;
            let mut batch = consumer.fetch().max_messages(1).messages().await.unwrap();
            let msg = batch.next().await.unwrap().unwrap();
            let job = Job::decode(msg.payload.as_ref()).unwrap();
            let r = effect(runs.clone())(job.clone()).await;
            assert!(!publish_result(&wjs, (cfg.result_subject)("fake"), &r).await.unwrap());
            drop(msg); // no ack
            drop(client);
            eprintln!("crashed after storing the result of {}", job.job_id);
        }
        // 2. crash mid-job
        {
            let (client, wjs, consumer, cfg) = worker(&url, &jobs, &root, "crash-mid-job").await;
            let (tx, rx) = tokio::sync::oneshot::channel::<String>();
            let tx = Mutex::new(Some(tx));
            let runs2 = runs.clone();
            let task = tokio::spawn(run_worker(consumer, wjs, cfg, move |job: Job| {
                runs2.fetch_add(1, Ordering::SeqCst);
                if let Some(tx) = tx.lock().unwrap().take() {
                    let _ = tx.send(job.job_id.clone());
                }
                async move { futures::future::pending::<JobResult>().await }
            }));
            let id = rx.await.unwrap();
            task.abort();
            drop(client);
            eprintln!("crashed while running {id}");
        }
        // 3. two healthy workers drain the queue (the crashed jobs come back after ack_wait)
        let mut tasks = Vec::new();
        let mut clients = Vec::new();
        for w in ["w1", "w2"] {
            let (client, wjs, consumer, cfg) = worker(&url, &jobs, &root, w).await;
            clients.push(client);
            tasks.push(tokio::spawn(run_worker(consumer, wjs, cfg, effect(runs.clone()))));
        }
        let res_stream = format!("{jobs}_RES");
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let left = js.get_stream(&jobs).await.unwrap().info().await.unwrap().state.messages;
            if left == 0 || Instant::now() > deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
        for t in &tasks {
            t.abort();
        }
        let stored = results(&js, &res_stream).await;
        let mut per_job: BTreeMap<String, usize> = BTreeMap::new();
        for r in &stored {
            *per_job.entry(r.job_id.clone()).or_default() += 1;
            assert_eq!(r.result, burn(ids.iter().position(|i| *i == r.job_id).unwrap() as u64, 200_000).to_le_bytes());
        }
        let runs = runs.load(Ordering::SeqCst);
        eprintln!("{} jobs, {runs} handler runs, {} stored results", ids.len(), stored.len());
        assert_eq!(js.get_stream(&jobs).await.unwrap().info().await.unwrap().state.messages, 0, "queue drained");
        assert_eq!(per_job.keys().cloned().collect::<Vec<_>>(), ids, "every job has a result");
        assert!(per_job.values().all(|n| *n == 1), "exactly one stored result per job: {per_job:?}");
        assert!(runs >= ids.len() + 2, "both crashed jobs ran again ({runs} runs)");
        js.delete_stream(&jobs).await.unwrap();
        js.delete_stream(&res_stream).await.unwrap();
    }

    /// Spec 21 criterion 3: `FA_NATS_TEST=1 cargo test -p fa-bus live_two_workers -- --nocapture`.
    /// The same 40 CPU-bound jobs drained by one worker, then by two workers on the same durable
    /// consumer (fresh streams each time); two workers must reach ≥ 1.7× the throughput.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn live_two_workers_double_throughput() {
        if std::env::var("FA_NATS_TEST").is_err() {
            eprintln!("skipping live JetStream test (set FA_NATS_TEST=1)");
            return;
        }
        let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
        // calibrate ~60 ms per job on this machine
        let t0 = Instant::now();
        std::hint::black_box(burn(1, 10_000_000));
        let iters = (10_000_000.0 * 0.060 / t0.elapsed().as_secs_f64()) as u64;
        let mut secs = Vec::new();
        for n_workers in [1usize, 2] {
            let (_c, js, root, jobs) = scratch(&format!("TP{n_workers}"), Duration::from_secs(30)).await.unwrap();
            let ids = publish_jobs(&js, &root, 40).await;
            let start = Instant::now();
            let mut tasks = Vec::new();
            let mut clients = Vec::new();
            for w in 0..n_workers {
                let (client, wjs, consumer, cfg) = worker(&url, &jobs, &root, &format!("w{w}")).await;
                clients.push(client);
                tasks.push(tokio::spawn(run_worker(consumer, wjs, cfg, move |job: Job| async move {
                    let v = tokio::task::spawn_blocking(move || burn(job.seed, iters)).await.unwrap();
                    JobResult { job_id: job.job_id, result: v.to_le_bytes().to_vec(), ..Default::default() }
                })));
            }
            let res_stream = format!("{jobs}_RES");
            loop {
                let n = js.get_stream(&res_stream).await.unwrap().info().await.unwrap().state.messages;
                if n as usize >= ids.len() || start.elapsed() > Duration::from_secs(120) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            let took = start.elapsed().as_secs_f64();
            for t in &tasks {
                t.abort();
            }
            assert_eq!(results(&js, &res_stream).await.len(), ids.len());
            eprintln!("{n_workers} worker(s): {} jobs in {took:.2} s ({:.1} jobs/s)", ids.len(), ids.len() as f64 / took);
            secs.push(took);
            js.delete_stream(&jobs).await.unwrap();
            js.delete_stream(&res_stream).await.unwrap();
        }
        let speedup = secs[0] / secs[1];
        eprintln!("speed-up with two workers: {speedup:.2}x");
        assert!(speedup >= 1.7, "two workers gave only {speedup:.2}x");
    }
}
