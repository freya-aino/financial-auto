//! JetStream stream table (spec 21 §5) and idempotent provisioning.
//!
//! The table lives in `crates/fa-bus/streams.json`, embedded here and in `go/pkg/bus/streams.go`
//! so Rust and Go cannot drift. `JOBS_<kind>` work queues are owned by the worker/orchestrator
//! pair for that kind; [`ensure_work_queue`] is the shared helper they can call.
//!
//! Core NATS publishes on a subject captured by a stream are persisted by JetStream without any
//! publisher change, so producers keep using plain `client.publish`.

use std::time::Duration;

use async_nats::jetstream::{
    self,
    consumer::{self, pull},
    stream::{self, RetentionPolicy, StorageType},
};
use futures::TryStreamExt;
use serde::{Deserialize, Serialize};

/// The embedded fixture shared with Go.
pub const STREAMS_JSON: &str = include_str!("../streams.json");

/// One limits-retention stream from spec 21 §5.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamSpec {
    pub name: String,
    pub subjects: Vec<String>,
    pub max_age_hours: u64,
    #[serde(default)]
    pub storage: Storage,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Storage {
    #[default]
    File,
    Memory,
}

impl From<Storage> for StorageType {
    fn from(s: Storage) -> Self {
        match s {
            Storage::File => StorageType::File,
            Storage::Memory => StorageType::Memory,
        }
    }
}

/// Work-queue consumer tunables (spec 21 §7 `jobs.ack_wait`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkQueueSpec {
    pub ack_wait_secs: u64,
    pub max_deliver: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamTable {
    pub streams: Vec<StreamSpec>,
    pub work_queue: WorkQueueSpec,
}

impl StreamSpec {
    /// The JetStream config this spec provisions.
    pub fn config(&self) -> stream::Config {
        stream::Config {
            name: self.name.clone(),
            subjects: self.subjects.clone(),
            retention: RetentionPolicy::Limits,
            max_age: Duration::from_secs(self.max_age_hours * 3600),
            storage: self.storage.into(),
            description: self.description.clone(),
            ..Default::default()
        }
    }
}

/// Parses the embedded table. Panics only if the fixture itself is malformed (covered by tests).
pub fn table() -> StreamTable {
    serde_json::from_str(STREAMS_JSON).expect("crates/fa-bus/streams.json is valid")
}

/// Stream name for a work-queue kind: `evo` → `JOBS_EVO`, `evo.eval` → `JOBS_EVO_EVAL`.
pub fn work_queue_name(kind: &str) -> String {
    format!("JOBS_{}", kind.to_ascii_uppercase().replace(['.', '-'], "_"))
}

/// Subject filter captured by a work queue: `jobs.<kind>.>`.
pub fn work_queue_subject(kind: &str) -> String {
    format!("jobs.{kind}.>")
}

/// Durable pull-consumer config with the spec's `ack_wait` / `max_deliver`.
pub fn work_queue_consumer(durable: &str) -> pull::Config {
    let wq = table().work_queue;
    pull::Config {
        durable_name: Some(durable.to_owned()),
        ack_wait: Duration::from_secs(wq.ack_wait_secs),
        max_deliver: wq.max_deliver,
        ..Default::default()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StreamsError {
    #[error("stream {name}: get/create: {source}")]
    Create { name: String, source: jetstream::context::CreateStreamError },
    #[error("stream {name}: update: {source}")]
    Update { name: String, source: jetstream::context::UpdateStreamError },
    #[error("stream {name}: info: {source}")]
    Info { name: String, source: jetstream::stream::InfoError },
    #[error("stream {name}: consumer: {source}")]
    Consumer { name: String, source: jetstream::stream::ConsumerError },
    #[error("list streams/consumers: {0}")]
    List(#[from] jetstream::context::RequestError),
    #[error("stream {name}: lookup: {source}")]
    Lookup { name: String, source: jetstream::context::GetStreamError },
}

/// Creates every spec 21 §5 stream, updating the ones whose live config drifted
/// (subjects, max age, storage, description). Idempotent; safe to call at every service start.
/// Returns the names that were created or updated.
pub async fn ensure_streams(js: &jetstream::Context) -> Result<Vec<String>, StreamsError> {
    let mut changed = Vec::new();
    for spec in table().streams {
        if ensure_stream(js, &spec.config()).await? {
            changed.push(spec.name);
        }
    }
    Ok(changed)
}

/// Get-or-create then update if the config differs. Returns `true` when something changed.
async fn ensure_stream(js: &jetstream::Context, want: &stream::Config) -> Result<bool, StreamsError> {
    let name = want.name.clone();
    let mut existing = match js.get_stream(&name).await {
        Ok(s) => s,
        Err(_) => {
            js.get_or_create_stream(want.clone())
                .await
                .map_err(|source| StreamsError::Create { name: name.clone(), source })?;
            tracing::info!(stream = %name, subjects = ?want.subjects, "created JetStream stream");
            return Ok(true);
        }
    };
    let have = &existing.info().await.map_err(|source| StreamsError::Info { name: name.clone(), source })?.config;
    if config_matches(have, want) {
        return Ok(false);
    }
    // Keep server-side fields we do not manage (limits, replicas, dedup window) and overlay ours.
    let mut cfg = have.clone();
    cfg.subjects = want.subjects.clone();
    cfg.max_age = want.max_age;
    cfg.description = want.description.clone();
    cfg.retention = want.retention;
    cfg.storage = want.storage;
    js.update_stream(cfg).await.map_err(|source| StreamsError::Update { name: name.clone(), source })?;
    tracing::info!(stream = %name, subjects = ?want.subjects, "updated JetStream stream");
    Ok(true)
}

fn config_matches(have: &stream::Config, want: &stream::Config) -> bool {
    let mut a = have.subjects.clone();
    let mut b = want.subjects.clone();
    a.sort();
    b.sort();
    a == b
        && have.max_age == want.max_age
        && have.retention == want.retention
        && have.storage == want.storage
        && have.description.as_deref().unwrap_or("") == want.description.as_deref().unwrap_or("")
}

/// Ensures the `JOBS_<KIND>` work-queue stream on `jobs.<kind>.>` and a durable pull consumer
/// with the spec's `ack_wait` 60 s / `max_deliver` 5. Owners of a kind (evo-worker,
/// evo-orchestrator) can call this instead of hand-rolling the config.
pub async fn ensure_work_queue(
    js: &jetstream::Context,
    kind: &str,
    durable: &str,
) -> Result<(stream::Stream, consumer::Consumer<pull::Config>), StreamsError> {
    let name = work_queue_name(kind);
    let stream = js
        .get_or_create_stream(stream::Config {
            name: name.clone(),
            subjects: vec![work_queue_subject(kind)],
            retention: RetentionPolicy::WorkQueue,
            ..Default::default()
        })
        .await
        .map_err(|source| StreamsError::Create { name: name.clone(), source })?;
    let cfg = work_queue_consumer(durable);
    // `create_consumer` updates an existing durable's config; fall back for older servers.
    let consumer = match stream.create_consumer(cfg.clone()).await {
        Ok(c) => c,
        Err(_) => stream
            .get_or_create_consumer(durable, cfg)
            .await
            .map_err(|source| StreamsError::Consumer { name: name.clone(), source })?,
    };
    spawn_consumer_lag(consumer.clone());
    Ok((stream, consumer))
}

/// How often [`spawn_consumer_lag`] refreshes the consumer gauges.
pub const CONSUMER_LAG_POLL: Duration = Duration::from_secs(10);

/// Polls `consumer.info()` every [`CONSUMER_LAG_POLL`] and exports the lag gauges spec 60 §6.5
/// asks for: `fa_bus_consumer_num_pending{stream,consumer}` (messages not yet delivered),
/// `fa_bus_consumer_ack_pending`, `fa_bus_consumer_redelivered` and `fa_bus_consumer_waiting`.
/// Idempotent per `(stream, consumer)`; the task ends when the consumer cannot be described
/// ten times in a row (deleted) and removes its series.
pub fn spawn_consumer_lag(mut consumer: consumer::Consumer<pull::Config>) -> Option<tokio::task::JoinHandle<()>> {
    static ACTIVE: std::sync::Mutex<Vec<(String, String)>> = std::sync::Mutex::new(Vec::new());
    let info = consumer.cached_info();
    let key = (info.stream_name.clone(), info.name.clone());
    {
        let mut active = ACTIVE.lock().unwrap_or_else(|p| p.into_inner());
        if active.contains(&key) {
            return None;
        }
        active.push(key.clone());
    }
    Some(tokio::spawn(async move {
        let (stream, name) = key;
        let labels = [("stream", stream.as_str()), ("consumer", name.as_str())];
        let pending =
            fa_metrics::gauge("fa_bus_consumer_num_pending", "JetStream messages not yet delivered to the consumer");
        let acks = fa_metrics::gauge("fa_bus_consumer_ack_pending", "JetStream messages delivered and awaiting ack");
        let redelivered =
            fa_metrics::gauge("fa_bus_consumer_redelivered", "JetStream messages redelivered to the consumer");
        let waiting = fa_metrics::gauge("fa_bus_consumer_waiting", "Pull requests waiting on the consumer");
        let mut failures = 0u32;
        let mut tick = tokio::time::interval(CONSUMER_LAG_POLL);
        loop {
            tick.tick().await;
            match consumer.info().await {
                Ok(i) => {
                    failures = 0;
                    pending.set_with(&labels, i.num_pending as f64);
                    acks.set_with(&labels, i.num_ack_pending as f64);
                    redelivered.set_with(&labels, i.num_redelivered as f64);
                    waiting.set_with(&labels, i.num_waiting as f64);
                }
                Err(e) => {
                    failures += 1;
                    tracing::debug!(%stream, consumer = %name, error = %e, "consumer info failed");
                    if failures >= 10 {
                        for g in [&pending, &acks, &redelivered, &waiting] {
                            g.remove_with(&labels);
                        }
                        ACTIVE.lock().unwrap_or_else(|p| p.into_inner()).retain(|k| k.0 != stream || k.1 != name);
                        return;
                    }
                }
            }
        }
    }))
}

/// Status of one stream for `GET /api/bus/streams` (spec 50 §4.3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamStatus {
    pub name: String,
    pub subjects: Vec<String>,
    pub retention: String,
    pub storage: String,
    pub max_age_secs: u64,
    pub messages: u64,
    pub bytes: u64,
    pub first_seq: u64,
    pub last_seq: u64,
    pub consumer_count: usize,
    pub consumers: Vec<ConsumerStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumerStatus {
    pub name: String,
    pub durable: bool,
    pub filter_subjects: Vec<String>,
    pub ack_pending: usize,
    pub num_pending: u64,
    pub redelivered: usize,
    pub waiting: usize,
    pub delivered_seq: u64,
    pub ack_floor_seq: u64,
}

/// Lists every stream on the server (spec streams, work queues and anything else) with
/// per-consumer lag, sorted by name.
pub async fn list_streams(js: &jetstream::Context) -> Result<Vec<StreamStatus>, StreamsError> {
    let infos: Vec<stream::Info> = js.streams().try_collect().await?;
    let mut out = Vec::with_capacity(infos.len());
    for info in infos {
        let name = info.config.name.clone();
        let stream = js.get_stream_no_info(&name).await.map_err(|source| StreamsError::Lookup { name, source })?;
        let consumers: Vec<consumer::Info> = stream.consumers().try_collect().await?;
        out.push(StreamStatus {
            name: info.config.name.clone(),
            subjects: info.config.subjects.clone(),
            retention: format!("{:?}", info.config.retention).to_ascii_lowercase(),
            storage: format!("{:?}", info.config.storage).to_ascii_lowercase(),
            max_age_secs: info.config.max_age.as_secs(),
            messages: info.state.messages,
            bytes: info.state.bytes,
            first_seq: info.state.first_sequence,
            last_seq: info.state.last_sequence,
            consumer_count: info.state.consumer_count,
            consumers: consumers
                .into_iter()
                .map(|c| ConsumerStatus {
                    name: c.name,
                    durable: c.config.durable_name.is_some(),
                    filter_subjects: if c.config.filter_subjects.is_empty() {
                        if c.config.filter_subject.is_empty() { vec![] } else { vec![c.config.filter_subject] }
                    } else {
                        c.config.filter_subjects
                    },
                    ack_pending: c.num_ack_pending,
                    num_pending: c.num_pending,
                    redelivered: c.num_redelivered,
                    waiting: c.num_waiting,
                    delivered_seq: c.delivered.stream_sequence,
                    ack_floor_seq: c.ack_floor.stream_sequence,
                })
                .collect(),
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// True when some concrete subject matches both filters (NATS `*` / `>` wildcards).
/// JetStream refuses to create a stream whose subjects overlap another stream's.
pub fn subjects_overlap(a: &str, b: &str) -> bool {
    let (a, b): (Vec<&str>, Vec<&str>) = (a.split('.').collect(), b.split('.').collect());
    for i in 0..a.len().max(b.len()) {
        match (a.get(i), b.get(i)) {
            (Some(&">"), Some(_)) | (Some(_), Some(&">")) => return true,
            (Some(x), Some(y)) if *x == "*" || *y == "*" || x == y => {}
            _ => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_matches_spec_21() {
        let t = table();
        let names: Vec<&str> = t.streams.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["MD", "TEXT", "MEDIA", "MACRO", "LATENT", "SIGNAL", "ORDERS", "JOB_RESULTS"]);
        let by = |n: &str| t.streams.iter().find(|s| s.name == n).unwrap();
        assert_eq!(by("MD").max_age_hours, 24);
        assert_eq!(by("TEXT").subjects, ["news.>", "filing.>"]);
        assert_eq!(by("MACRO").max_age_hours, 30 * 24);
        assert_eq!(by("ORDERS").max_age_hours, 365 * 24);
        assert_eq!(by("ORDERS").storage, Storage::File);
        assert_eq!(by("JOB_RESULTS").subjects, ["jobs.result.>"]);
        assert_eq!(t.work_queue, WorkQueueSpec { ack_wait_secs: 60, max_deliver: 5 });
        let cfg = by("MD").config();
        assert_eq!(cfg.max_age, Duration::from_secs(86_400));
        assert_eq!(cfg.retention, RetentionPolicy::Limits);
    }

    /// `go/pkg/bus/streams.json` is an embedded copy (Go cannot `embed` across the module root).
    #[test]
    fn go_copy_of_fixture_is_identical() {
        assert_eq!(
            include_str!("../../../go/pkg/bus/streams.json"),
            STREAMS_JSON,
            "run: cp crates/fa-bus/streams.json go/pkg/bus/streams.json"
        );
    }

    #[test]
    fn overlap_predicate() {
        assert!(subjects_overlap("md.>", "md.trade.sim.x"));
        assert!(subjects_overlap("md.*.sim.>", "md.>"));
        assert!(subjects_overlap("a.*", "a.b"));
        assert!(!subjects_overlap("md.>", "md"));
        assert!(!subjects_overlap("jobs.evo.>", "jobs.result.>"));
        assert!(!subjects_overlap("news.>", "filing.>"));
        assert!(!subjects_overlap("a.b", "a.c"));
    }

    #[test]
    fn no_stream_subjects_overlap_including_work_queues() {
        let t = table();
        let mut filters: Vec<(String, String)> =
            t.streams.iter().flat_map(|s| s.subjects.iter().map(move |f| (s.name.clone(), f.clone()))).collect();
        // Work queues owned elsewhere today (bins/evo-worker, go/services/evo-orchestrator).
        filters.push((work_queue_name("evo"), work_queue_subject("evo")));
        for (i, (sa, fa)) in filters.iter().enumerate() {
            for (sb, fb) in &filters[i + 1..] {
                if sa == sb {
                    continue;
                }
                assert!(!subjects_overlap(fa, fb), "{sa} `{fa}` overlaps {sb} `{fb}`");
            }
        }
        // Every spec 21 §4.1 subject root lands in exactly one stream (or none, for ctl/svc/ingest/jobs.progress).
        for (subject, want) in [
            ("md.trade.sim.0190", Some("MD")),
            ("news.item.hn", Some("TEXT")),
            ("filing.event.sec", Some("TEXT")),
            ("macro.obs.fred.CPI", Some("MACRO")),
            ("onchain.event.eth", Some("MACRO")),
            ("latent.state.1m.x", Some("LATENT")),
            ("decoder.output.rnn_prior", Some("LATENT")),
            ("signal.exposure.s1", Some("SIGNAL")),
            ("order.event.paper", Some("ORDERS")),
            ("risk.state", Some("ORDERS")),
            ("risk.halt.portfolio", Some("ORDERS")),
            ("jobs.result.evo.eval.decoder", Some("JOB_RESULTS")),
            ("jobs.evo.eval.decoder", Some("JOBS_EVO")),
            ("jobs.progress.01J", None),
            ("ctl.ingest.start", None),
            ("svc.heartbeat.ingestor", None),
        ] {
            let hits: Vec<&str> =
                filters.iter().filter(|(_, f)| subjects_overlap(f, subject)).map(|(s, _)| s.as_str()).collect();
            assert_eq!(hits, want.into_iter().collect::<Vec<_>>(), "{subject}");
        }
    }

    #[test]
    fn work_queue_naming() {
        assert_eq!(work_queue_name("evo"), "JOBS_EVO");
        assert_eq!(work_queue_subject("evo"), "jobs.evo.>");
        assert_eq!(work_queue_name("media.asr"), "JOBS_MEDIA_ASR");
        let c = work_queue_consumer("evo-worker");
        assert_eq!(c.ack_wait, Duration::from_secs(60));
        assert_eq!(c.max_deliver, 5);
    }

    /// Against the compose NATS: `FA_NATS_TEST=1 cargo test -p fa-bus -- --nocapture live`.
    #[tokio::test]
    async fn live_ensure_and_list_streams() {
        if std::env::var("FA_NATS_TEST").is_err() {
            eprintln!("skipping live JetStream test (set FA_NATS_TEST=1)");
            return;
        }
        let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
        let client = async_nats::connect(&url).await.expect("connect");
        let js = jetstream::new(client.clone());

        let first = ensure_streams(&js).await.expect("ensure");
        let second = ensure_streams(&js).await.expect("ensure again");
        assert!(second.is_empty(), "second ensure must be a no-op, changed {second:?}");
        eprintln!("ensure_streams: first run changed {first:?}, second run changed nothing");

        // Drift repair: shrink MD's max_age, then ensure must restore it.
        let md = table().streams.into_iter().find(|s| s.name == "MD").unwrap();
        let mut drifted = md.config();
        drifted.max_age = Duration::from_secs(60);
        js.update_stream(drifted).await.expect("drift");
        let fixed = ensure_streams(&js).await.expect("repair");
        assert_eq!(fixed, ["MD"]);

        // A core NATS publish on a captured subject lands in the stream with no publisher change.
        let before = js.get_stream("MD").await.unwrap().info().await.unwrap().state.messages;
        let subject = format!("md.trade.test.{}", uuid_like());
        client.publish(subject.clone(), "hello".into()).await.unwrap();
        client.flush().await.unwrap();
        let mut md_stream = js.get_stream("MD").await.unwrap();
        let mut after = md_stream.info().await.unwrap().state.messages;
        for _ in 0..20 {
            if after > before {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
            after = md_stream.info().await.unwrap().state.messages;
        }
        assert!(after > before, "MD messages {before} -> {after}");
        eprintln!("core publish on {subject}: MD messages {before} -> {after}");

        // Work-queue helper on a throwaway kind; evo's queue stays untouched.
        let (mut wq, consumer) = ensure_work_queue(&js, "fabustest", "fa-bus-test").await.expect("work queue");
        assert_eq!(wq.info().await.unwrap().config.retention, RetentionPolicy::WorkQueue);
        assert_eq!(consumer.cached_info().config.ack_wait, Duration::from_secs(60));
        assert_eq!(consumer.cached_info().config.max_deliver, 5);

        let list = list_streams(&js).await.expect("list");
        let names: Vec<&str> = list.iter().map(|s| s.name.as_str()).collect();
        for want in table().streams {
            assert!(names.contains(&want.name.as_str()), "{} missing from {names:?}", want.name);
        }
        let wq_status = list.iter().find(|s| s.name == "JOBS_FABUSTEST").unwrap();
        assert_eq!(wq_status.consumers.len(), 1);
        assert_eq!(wq_status.consumers[0].name, "fa-bus-test");
        eprintln!("list_streams: {}", serde_json::to_string_pretty(&list).unwrap());
        js.delete_stream("JOBS_FABUSTEST").await.unwrap();
    }

    fn uuid_like() -> String {
        format!("{:032x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos())
    }
}
