//! Dead-letter stream for poison messages (spec 21 §8).
//!
//! JetStream announces a message that exhausted a consumer's `max_deliver` on
//! `$JS.EVENT.ADVISORY.CONSUMER.MAX_DELIVERIES.<stream>.<consumer>` and one a consumer
//! terminated (`AckKind::Term`, e.g. an undecodable payload) on
//! `$JS.EVENT.ADVISORY.CONSUMER.MSG_TERMINATED.<stream>.<consumer>`. [`spawn_forwarder`] listens
//! to both, fetches the original message by stream sequence and republishes it on
//! `dlq.<STREAM>.<original subject>` (captured by the `DLQ` stream of `streams.json`, 30 d,
//! 1 GiB) with the original headers plus:
//!
//! | header | value |
//! |---|---|
//! | `Fa-Dlq-Stream` / `Fa-Dlq-Consumer` / `Fa-Dlq-Seq` | where it came from |
//! | `Fa-Dlq-Subject` | original subject |
//! | `Fa-Dlq-Reason` | `max_deliveries` or `terminated` (+ the consumer's term reason) |
//! | `Fa-Dlq-Deliveries` | delivery count at the advisory |
//! | `Nats-Msg-Id` | `dlq:<stream>:<seq>` (several forwarders deduplicate) |
//!
//! A message of a **work-queue** stream is then deleted from its source (it would otherwise
//! sit there forever, invisible to the consumer that gave up on it); limits streams keep it.
//! Every dead letter increments `fa_bus_dead_letters_total{stream,consumer,reason}` and is
//! announced once per (stream, consumer) and minute as a JSON alert on `ops.alert` (the D-066
//! shape: `status = firing`, `alertname = DeadLetter`, labels `stream`, `consumer`, `reason`).
//! The `DLQ` stream's own advisories are ignored, so dead letters never loop.

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use async_nats::{HeaderMap, jetstream};
use futures::StreamExt;
use serde::Deserialize;

/// The dead-letter stream (`streams.json`).
pub const DLQ_STREAM: &str = "DLQ";
/// D-066 alert subject.
pub const OPS_ALERT: &str = crate::subjects::OPS_ALERT;
pub const ADVISORY_MAX_DELIVERIES: &str = "$JS.EVENT.ADVISORY.CONSUMER.MAX_DELIVERIES.>";
pub const ADVISORY_TERMINATED: &str = "$JS.EVENT.ADVISORY.CONSUMER.MSG_TERMINATED.>";

/// `dlq.<stream>.<subject>`.
pub fn dlq_subject(stream: &str, subject: &str) -> String {
    crate::subjects::dead_letter(stream, subject)
}

/// Max-deliveries / terminated advisory (only the fields used).
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Advisory {
    #[serde(rename = "type", default)]
    pub kind: String,
    pub stream: String,
    pub consumer: String,
    pub stream_seq: u64,
    #[serde(default)]
    pub deliveries: u64,
    #[serde(default)]
    pub reason: String,
}

impl Advisory {
    pub fn reason_label(&self) -> &'static str {
        if self.kind.ends_with("terminated") { "terminated" } else { "max_deliveries" }
    }
}

/// Parses an advisory payload; `None` for other advisories and the DLQ's own.
pub fn parse_advisory(payload: &[u8]) -> Option<Advisory> {
    let a: Advisory = serde_json::from_slice(payload).ok()?;
    (a.stream != DLQ_STREAM && a.stream_seq > 0).then_some(a)
}

/// Headers of the dead-letter copy.
pub fn dlq_headers(original: &HeaderMap, a: &Advisory, subject: &str) -> HeaderMap {
    let mut h = HeaderMap::new();
    for (k, vs) in original.iter() {
        if AsRef::<str>::as_ref(k) == "Nats-Msg-Id" {
            continue;
        }
        for v in vs {
            h.append(k.clone(), v.clone());
        }
    }
    let reason =
        if a.reason.is_empty() { a.reason_label().to_owned() } else { format!("{}: {}", a.reason_label(), a.reason) };
    h.insert("Fa-Dlq-Stream", a.stream.as_str());
    h.insert("Fa-Dlq-Consumer", a.consumer.as_str());
    h.insert("Fa-Dlq-Seq", a.stream_seq.to_string().as_str());
    h.insert("Fa-Dlq-Subject", subject);
    h.insert("Fa-Dlq-Reason", reason.as_str());
    h.insert("Fa-Dlq-Deliveries", a.deliveries.to_string().as_str());
    h.insert("Nats-Msg-Id", format!("dlq:{}:{}", a.stream, a.stream_seq).as_str());
    h
}

/// The `ops.alert` JSON of one dead letter (D-066 `AlertEvent` fields).
pub fn alert_json(a: &Advisory, subject: &str, count: u64) -> serde_json::Value {
    let now = chrono::Utc::now();
    let labels = serde_json::json!({
        "alertname": "DeadLetter", "severity": "warning", "stream": a.stream, "consumer": a.consumer,
        "reason": a.reason_label(),
    });
    serde_json::json!({
        "status": "firing",
        "alertname": "DeadLetter",
        "severity": "warning",
        "summary": format!("{count} dead letter(s) from {}/{} (last: seq {} on {subject}, {})", a.stream, a.consumer, a.stream_seq, a.reason_label()),
        "fingerprint": format!("dlq:{}:{}", a.stream, a.consumer),
        "labels": labels,
        "annotations": { "dlq_subject": dlq_subject(&a.stream, subject), "deliveries": a.deliveries.to_string() },
        "starts_at": now,
        "ends_at": null,
        "generator_url": null,
        "at": now,
    })
}

/// Copies one advised message into the DLQ. Returns the DLQ subject, or `None` when the message
/// is already gone (aged out, or another forwarder moved it).
pub async fn forward(js: &jetstream::Context, a: &Advisory) -> Result<Option<String>, String> {
    let stream = js.get_stream(&a.stream).await.map_err(|e| e.to_string())?;
    let msg = match stream.get_raw_message(a.stream_seq).await {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    let subject = msg.subject.to_string();
    let target = dlq_subject(&a.stream, &subject);
    let headers = dlq_headers(&msg.headers, a, &subject);
    js.publish_with_headers(target.clone(), headers, msg.payload.clone())
        .await
        .map_err(|e| e.to_string())?
        .await
        .map_err(|e| e.to_string())?;
    if stream.cached_info().config.retention == jetstream::stream::RetentionPolicy::WorkQueue {
        let _ = stream.delete_message(a.stream_seq).await;
    }
    Ok(Some(target))
}

/// Runs the forwarder until the client closes. Safe to run in several services (the copies
/// deduplicate on `Nats-Msg-Id`); control-api runs it next to `ensure_streams`.
pub fn spawn_forwarder(client: async_nats::Client) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let js = jetstream::new(client.clone());
        let subs =
            futures::future::try_join(client.subscribe(ADVISORY_MAX_DELIVERIES), client.subscribe(ADVISORY_TERMINATED))
                .await;
        let (a, b) = match subs {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "dead-letter forwarder: subscribe failed");
                return;
            }
        };
        let counter = fa_metrics::counter("fa_bus_dead_letters_total", "Messages moved to the DLQ stream");
        let mut merged = futures::stream::select(a, b);
        let mut last_alert: HashMap<(String, String), (Instant, u64)> = HashMap::new();
        tracing::info!("dead-letter forwarder running (spec 21 §8)");
        while let Some(m) = merged.next().await {
            let Some(adv) = parse_advisory(&m.payload) else { continue };
            match forward(&js, &adv).await {
                Ok(Some(target)) => {
                    counter.inc_with(&[
                        ("stream", adv.stream.as_str()),
                        ("consumer", adv.consumer.as_str()),
                        ("reason", adv.reason_label()),
                    ]);
                    tracing::warn!(stream = %adv.stream, consumer = %adv.consumer, seq = adv.stream_seq, %target, "dead letter");
                    let key = (adv.stream.clone(), adv.consumer.clone());
                    let entry = last_alert.entry(key).or_insert((Instant::now() - Duration::from_secs(120), 0));
                    entry.1 += 1;
                    if entry.0.elapsed() >= Duration::from_secs(60) {
                        let subject = target.splitn(3, '.').nth(2).unwrap_or_default().to_owned();
                        let body = alert_json(&adv, &subject, entry.1);
                        if let Ok(bytes) = serde_json::to_vec(&body) {
                            let _ = client.publish(OPS_ALERT, bytes.into()).await;
                        }
                        *entry = (Instant::now(), 0);
                    }
                }
                Ok(None) => tracing::debug!(stream = %adv.stream, seq = adv.stream_seq, "advised message already gone"),
                Err(e) => {
                    tracing::warn!(stream = %adv.stream, seq = adv.stream_seq, error = %e, "dead-letter copy failed")
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advisories_parse_and_the_dlq_never_loops() {
        let max = br#"{"type":"io.nats.jetstream.advisory.v1.max_deliver","id":"x","timestamp":"2026-11-11T00:00:00Z","stream":"JOBS_MCMC","consumer":"model-runtime-mcmc","stream_seq":7,"deliveries":5}"#;
        let a = parse_advisory(max).unwrap();
        assert_eq!(
            (a.stream.as_str(), a.stream_seq, a.deliveries, a.reason_label()),
            ("JOBS_MCMC", 7, 5, "max_deliveries")
        );
        let term = br#"{"type":"io.nats.jetstream.advisory.v1.terminated","stream":"LATENT","consumer":"c","stream_seq":3,"deliveries":1,"reason":"undecodable"}"#;
        let t = parse_advisory(term).unwrap();
        assert_eq!(t.reason_label(), "terminated");
        let own =
            br#"{"type":"io.nats.jetstream.advisory.v1.max_deliver","stream":"DLQ","consumer":"c","stream_seq":3}"#;
        assert!(parse_advisory(own).is_none());
        assert!(parse_advisory(b"not json").is_none());

        let mut orig = HeaderMap::new();
        orig.insert("Nats-Msg-Id", "job-1");
        orig.insert("Fa-Trace", "t");
        let h = dlq_headers(&orig, &t, "latent.state.1m.x");
        assert_eq!(h.get("Nats-Msg-Id").unwrap().as_str(), "dlq:LATENT:3");
        assert_eq!(h.get("Fa-Trace").unwrap().as_str(), "t");
        assert_eq!(h.get("Fa-Dlq-Reason").unwrap().as_str(), "terminated: undecodable");
        assert_eq!(dlq_subject("LATENT", "latent.state.1m.x"), "dlq.LATENT.latent.state.1m.x");
        let j = alert_json(&t, "latent.state.1m.x", 2);
        assert_eq!(j["alertname"], "DeadLetter");
        assert_eq!(j["labels"]["stream"], "LATENT");
    }

    /// `FA_NATS_TEST=1 cargo test -p fa-bus -- --nocapture live_dead_letter`: a job that is
    /// nacked until `max_deliver` lands in the DLQ with its headers and leaves the work queue.
    #[tokio::test]
    async fn live_dead_letter() {
        if std::env::var("FA_NATS_TEST").is_err() {
            eprintln!("skipping live DLQ test (set FA_NATS_TEST=1)");
            return;
        }
        let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
        let client = async_nats::connect(&url).await.expect("connect");
        let js = jetstream::new(client.clone());
        crate::streams::ensure_streams(&js).await.expect("ensure");
        let fwd = spawn_forwarder(client.clone());
        tokio::time::sleep(Duration::from_millis(200)).await;
        let kind = "fadlqtest";
        let _ = crate::streams::ensure_work_queue(&js, kind, "dlq-test").await.unwrap();
        let mut cfg = crate::streams::work_queue_consumer("dlq-test");
        cfg.max_deliver = 2;
        cfg.ack_wait = Duration::from_millis(500);
        let stream = js.get_stream(crate::streams::work_queue_name(kind)).await.unwrap();
        let consumer: jetstream::consumer::PullConsumer = stream.create_consumer(cfg).await.unwrap();
        let subject = format!("jobs.{kind}.poison");
        js.publish(subject.clone(), "poison".into()).await.unwrap().await.unwrap();
        // two deliveries, then a third pull: the server advises when it would redeliver past max_deliver
        for _ in 0..3 {
            let mut batch = consumer.fetch().max_messages(1).expires(Duration::from_secs(2)).messages().await.unwrap();
            if let Some(Ok(m)) = batch.next().await {
                m.ack_with(jetstream::AckKind::Nak(None)).await.unwrap();
            }
        }
        let dlq = js.get_stream(DLQ_STREAM).await.unwrap();
        let want = dlq_subject(&crate::streams::work_queue_name(kind), &subject);
        let mut got = None;
        for _ in 0..50 {
            if let Ok(m) = dlq.get_last_raw_message_by_subject(&want).await {
                got = Some(m);
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let m = got.expect("dead letter in DLQ");
        assert_eq!(&m.payload[..], b"poison");
        assert_eq!(m.headers.get("Fa-Dlq-Reason").unwrap().as_str(), "max_deliveries");
        eprintln!("dead letter on {want}: {:?}", m.headers);
        js.delete_stream(crate::streams::work_queue_name(kind)).await.unwrap();
        dlq.purge().filter(format!("dlq.{}.>", crate::streams::work_queue_name(kind))).await.unwrap();
        fwd.abort();
    }
}
