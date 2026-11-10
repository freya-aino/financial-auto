//! Time-range reader over a JetStream stream (spec 41 §3 "replay through the same code path as
//! live", §6.1): the recorded `latent.*` / `decoder.*` traffic of the `LATENT` stream (spec 21
//! §5, 7 d retention) is read back **by stream time** so a replay can rebuild the exact latent
//! graph and decoder cache the live strategies saw.
//!
//! [`read_range`] creates an **ordered** ephemeral pull consumer with
//! `deliver_policy = by_start_time(start)` and one filter subject, drains it until the first
//! message published after `end`, the server reports nothing pending, or the consumer stays idle
//! for [`ReadOptions::idle`], and returns the raw [`RecordedMessage`]s with their stream
//! sequence and the server's publish timestamp (the **knowledge time** of the message: when
//! the bus had it, which is what point-in-time replays filter on). [`read_latents`] runs it for
//! the four [`LATENT_FILTERS`] and merges the result in `(published_ms, stream_seq)` order.
//! [`LatentMessage::decode`] turns a record into the typed protobuf message by subject.
//!
//! Records serialise to one JSON object per line (`payload` hex-encoded) so a slice of the
//! live stream can be kept as a test fixture ([`to_jsonl`] / [`from_jsonl`]).

use std::{fmt, time::Duration};

use async_nats::jetstream::{
    self,
    consumer::{DeliverPolicy, ReplayPolicy, pull::OrderedConfig},
};
use fa_core::Timescale;
use fa_proto::{
    Message,
    v1::{DecoderOutput, LatentState, RegionEvent, TransitionUpdate},
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

/// The stream holding `latent.>` and `decoder.>` (spec 21 §5).
pub const LATENT_STREAM: &str = "LATENT";
/// Subjects a strategy replay needs, in the order they are read.
pub const LATENT_FILTERS: [&str; 4] = ["latent.state.>", "latent.region.>", "latent.transition.>", "decoder.output.>"];

/// One message as stored in the stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordedMessage {
    pub subject: String,
    pub stream_seq: u64,
    /// Server receive time of the message (ms since the epoch): its knowledge time.
    pub published_ms: i64,
    #[serde(with = "hex_bytes")]
    pub payload: Vec<u8>,
}

/// Reader tunables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadOptions {
    /// Stop when no message arrives for this long (the consumer has reached the live edge or the
    /// range is empty).
    pub idle: Duration,
    /// Hard cap per filter subject.
    pub max_messages: usize,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self { idle: Duration::from_secs(2), max_messages: 500_000 }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ReplayReadError {
    #[error("stream {stream}: {source}")]
    Stream { stream: String, source: jetstream::context::GetStreamError },
    #[error("stream {stream}: info: {source}")]
    Info { stream: String, source: jetstream::stream::InfoError },
    #[error("stream {stream}: ordered consumer: {source}")]
    Consumer { stream: String, source: jetstream::stream::ConsumerError },
    #[error("stream {stream}: messages: {source}")]
    Messages { stream: String, source: jetstream::consumer::StreamError },
    #[error("stream {stream}: read: {source}")]
    Read { stream: String, source: jetstream::consumer::pull::OrderedError },
    #[error("stream {stream}: message without JetStream metadata")]
    NoInfo { stream: String },
    #[error("invalid range: start {start_ms} > end {end_ms}")]
    Range { start_ms: i64, end_ms: i64 },
}

fn ms_of(t: async_nats::datetime::DateTime) -> i64 {
    i64::try_from(t.unix_timestamp_nanos() / 1_000_000).unwrap_or(i64::MAX)
}

fn datetime_of(ms: i64) -> async_nats::datetime::DateTime {
    async_nats::datetime::DateTime::from_unix_timestamp_nanos(i128::from(ms) * 1_000_000)
        .unwrap_or(async_nats::datetime::DateTime::UNIX_EPOCH)
}

/// Reads every message of `stream` matching `filter_subject` published in `[start_ms, end_ms]`
/// (inclusive, by server receive time), oldest first.
pub async fn read_range(
    js: &jetstream::Context,
    stream: &str,
    filter_subject: &str,
    start_ms: i64,
    end_ms: i64,
    opts: &ReadOptions,
) -> Result<Vec<RecordedMessage>, ReplayReadError> {
    if start_ms > end_ms {
        return Err(ReplayReadError::Range { start_ms, end_ms });
    }
    let name = stream.to_owned();
    let mut st =
        js.get_stream(stream).await.map_err(|source| ReplayReadError::Stream { stream: name.clone(), source })?;
    let info = st.info().await.map_err(|source| ReplayReadError::Info { stream: name.clone(), source })?;
    if info.state.messages == 0 || ms_of(info.state.last_timestamp) < start_ms {
        return Ok(Vec::new());
    }
    let consumer = st
        .create_consumer(OrderedConfig {
            filter_subject: filter_subject.to_owned(),
            deliver_policy: DeliverPolicy::ByStartTime { start_time: datetime_of(start_ms) },
            replay_policy: ReplayPolicy::Instant,
            ..Default::default()
        })
        .await
        .map_err(|source| ReplayReadError::Consumer { stream: name.clone(), source })?;
    let mut messages =
        consumer.messages().await.map_err(|source| ReplayReadError::Messages { stream: name.clone(), source })?;
    let mut out = Vec::new();
    loop {
        if out.len() >= opts.max_messages {
            break;
        }
        let Ok(next) = tokio::time::timeout(opts.idle, messages.next()).await else {
            break; // idle: live edge reached or nothing in range
        };
        let Some(next) = next else { break };
        let msg = next.map_err(|source| ReplayReadError::Read { stream: name.clone(), source })?;
        let info = msg.info().map_err(|_| ReplayReadError::NoInfo { stream: name.clone() })?;
        let published_ms = ms_of(info.published);
        if published_ms > end_ms {
            break;
        }
        let pending = info.pending;
        out.push(RecordedMessage {
            subject: msg.subject.to_string(),
            stream_seq: info.stream_sequence,
            published_ms,
            payload: msg.payload.to_vec(),
        });
        if pending == 0 {
            break;
        }
    }
    Ok(out)
}

/// Reads the four [`LATENT_FILTERS`] of the `LATENT` stream over `[start_ms, end_ms]` and merges
/// them in `(published_ms, stream_seq)` order.
pub async fn read_latents(
    js: &jetstream::Context,
    start_ms: i64,
    end_ms: i64,
    opts: &ReadOptions,
) -> Result<Vec<RecordedMessage>, ReplayReadError> {
    let mut all = Vec::new();
    for f in LATENT_FILTERS {
        all.extend(read_range(js, LATENT_STREAM, f, start_ms, end_ms, opts).await?);
    }
    sort_records(&mut all);
    Ok(all)
}

/// Knowledge-time order: `(published_ms, stream_seq)`; duplicates (same sequence) removed.
pub fn sort_records(records: &mut Vec<RecordedMessage>) {
    records.sort_by(|a, b| a.published_ms.cmp(&b.published_ms).then(a.stream_seq.cmp(&b.stream_seq)));
    records.dedup_by(|a, b| a.stream_seq == b.stream_seq && a.subject == b.subject);
}

/// Typed view of a recorded `LATENT` message.
#[derive(Debug, Clone, PartialEq)]
pub enum LatentMessage {
    State(LatentState),
    Region(RegionEvent),
    Transition(TransitionUpdate),
    DecoderOutput(DecoderOutput),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DecodeError {
    #[error("subject `{0}` is not a latent/decoder subject")]
    Subject(String),
    #[error("protobuf: {0}")]
    Proto(String),
}

fn ts_ms(t: Option<&fa_proto::Timestamp>) -> Option<i64> {
    t.map(|t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000)
}

impl LatentMessage {
    /// Decodes by subject prefix (`latent.state.`, `latent.region.`, `latent.transition.`,
    /// `decoder.output.`).
    pub fn decode(subject: &str, payload: &[u8]) -> Result<Self, DecodeError> {
        let proto = |e: fa_proto::prost::DecodeError| DecodeError::Proto(e.to_string());
        if subject.starts_with("latent.state.") {
            LatentState::decode(payload).map(Self::State).map_err(proto)
        } else if subject.starts_with("latent.region.") {
            RegionEvent::decode(payload).map(Self::Region).map_err(proto)
        } else if subject.starts_with("latent.transition.") {
            TransitionUpdate::decode(payload).map(Self::Transition).map_err(proto)
        } else if subject.starts_with("decoder.output.") {
            DecoderOutput::decode(payload).map(Self::DecoderOutput).map_err(proto)
        } else {
            Err(DecodeError::Subject(subject.to_owned()))
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::State(_) => "state",
            Self::Region(_) => "region",
            Self::Transition(_) => "transition",
            Self::DecoderOutput(_) => "decoder_output",
        }
    }

    /// Encoder snapshot the message was produced under (empty when unset).
    pub fn snapshot_id(&self) -> &str {
        match self {
            Self::State(s) => &s.snapshot_id,
            Self::Region(r) => r.space.as_ref().map_or("", |s| s.snapshot_id.as_str()),
            Self::Transition(t) => t.space.as_ref().map_or("", |s| s.snapshot_id.as_str()),
            Self::DecoderOutput(d) => &d.snapshot_id,
        }
    }

    pub fn timescale(&self) -> Option<Timescale> {
        let s = match self {
            Self::State(s) => s.timescale.as_str(),
            Self::Region(r) => r.space.as_ref().map_or("", |s| s.timescale.as_str()),
            Self::Transition(t) => t.space.as_ref().map_or("", |s| s.timescale.as_str()),
            Self::DecoderOutput(d) => d.timescale.as_str(),
        };
        s.parse().ok()
    }

    /// Event time carried by the message (`as_of`); region events carry none.
    pub fn as_of_ms(&self) -> Option<i64> {
        match self {
            Self::State(s) => ts_ms(s.as_of.as_ref()),
            Self::Region(_) => None,
            Self::Transition(t) => ts_ms(t.as_of.as_ref()),
            Self::DecoderOutput(d) => ts_ms(d.as_of.as_ref()),
        }
    }

    /// Whether this is a `latent.*` message (as opposed to a decoder output).
    pub fn is_latent(&self) -> bool {
        !matches!(self, Self::DecoderOutput(_))
    }
}

/// A decoded record with its stream metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Recorded {
    pub subject: String,
    pub stream_seq: u64,
    pub published_ms: i64,
    pub message: LatentMessage,
}

impl Recorded {
    pub fn decode(raw: &RecordedMessage) -> Result<Self, DecodeError> {
        Ok(Self {
            subject: raw.subject.clone(),
            stream_seq: raw.stream_seq,
            published_ms: raw.published_ms,
            message: LatentMessage::decode(&raw.subject, &raw.payload)?,
        })
    }
}

/// Decodes every record, dropping (and counting) the undecodable ones.
pub fn decode_all(raw: &[RecordedMessage]) -> (Vec<Recorded>, usize) {
    let mut out = Vec::with_capacity(raw.len());
    let mut bad = 0;
    for r in raw {
        match Recorded::decode(r) {
            Ok(d) => out.push(d),
            Err(_) => bad += 1,
        }
    }
    (out, bad)
}

/// One JSON object per line.
pub fn to_jsonl(records: &[RecordedMessage]) -> String {
    let mut s = String::new();
    for r in records {
        s.push_str(&serde_json::to_string(r).expect("record serialises"));
        s.push('\n');
    }
    s
}

/// Parses [`to_jsonl`] output; blank lines are skipped.
pub fn from_jsonl(text: &str) -> Result<Vec<RecordedMessage>, serde_json::Error> {
    text.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect()
}

impl fmt::Display for RecordedMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} seq={} at={} ({} bytes)", self.subject, self.stream_seq, self.published_ms, self.payload.len())
    }
}

/// `Vec<u8>` as a lowercase hex string (no extra dependency).
pub mod hex_bytes {
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    pub fn encode(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    pub fn decode(s: &str) -> Result<Vec<u8>, String> {
        if !s.len().is_multiple_of(2) {
            return Err("odd hex length".into());
        }
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| e.to_string())).collect()
    }

    pub fn serialize<S: Serializer>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        decode(&s).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use fa_proto::v1::{ExpertLatent, ExpertSpaceRef};

    use super::*;

    fn state(snapshot: &str, ms: i64) -> LatentState {
        LatentState {
            snapshot_id: snapshot.into(),
            timescale: "1s".into(),
            instrument_id: "01900000-0000-7000-8000-000000000001".into(),
            as_of: Some(fa_proto::Timestamp { seconds: ms / 1000, nanos: ((ms % 1000) * 1_000_000) as i32 }),
            experts: vec![ExpertLatent { expert_index: 0, region_id: "r1".into(), ..Default::default() }],
            router_weights: vec![],
        }
    }

    #[test]
    fn records_round_trip_through_jsonl_and_decode_by_subject() {
        let st = state("snap-a", 1_700_000_000_123);
        let tu = TransitionUpdate {
            space: Some(ExpertSpaceRef { snapshot_id: "snap-a".into(), timescale: "1m".into(), expert_index: 2 }),
            from_region_id: "a".into(),
            ..Default::default()
        };
        let raw = vec![
            RecordedMessage {
                subject: "latent.state.1s.0190".into(),
                stream_seq: 7,
                published_ms: 1_700_000_000_200,
                payload: st.encode_to_vec(),
            },
            RecordedMessage {
                subject: "latent.transition.1m".into(),
                stream_seq: 3,
                published_ms: 1_700_000_000_100,
                payload: tu.encode_to_vec(),
            },
            RecordedMessage { subject: "md.trade.x".into(), stream_seq: 9, published_ms: 1, payload: vec![0xff] },
        ];
        let text = to_jsonl(&raw);
        assert_eq!(text.lines().count(), 3);
        assert!(text.contains("\"payload\":\"ff\""));
        let mut back = from_jsonl(&text).unwrap();
        assert_eq!(back, raw);
        sort_records(&mut back);
        assert_eq!(back.iter().map(|r| r.stream_seq).collect::<Vec<_>>(), [9, 3, 7]);
        let (typed, bad) = decode_all(&back);
        assert_eq!((typed.len(), bad), (2, 1));
        assert_eq!(typed[0].message.kind(), "transition");
        assert_eq!(typed[0].message.timescale(), Some(Timescale::Minute));
        assert_eq!(typed[0].message.snapshot_id(), "snap-a");
        assert_eq!(typed[0].message.as_of_ms(), None);
        assert_eq!(typed[1].message.as_of_ms(), Some(1_700_000_000_123));
        assert!(typed[1].message.is_latent());
        assert_eq!(LatentMessage::decode("md.trade.x", &[]), Err(DecodeError::Subject("md.trade.x".into())));
        assert!(matches!(LatentMessage::decode("latent.state.1s.x", &[0xff, 0xff]), Err(DecodeError::Proto(_))));
        assert_eq!(hex_bytes::decode("0aff").unwrap(), vec![0x0a, 0xff]);
        assert!(hex_bytes::decode("abc").is_err());
        assert_eq!(ms_of(datetime_of(1_700_000_000_123)), 1_700_000_000_123);
    }

    /// Against the compose NATS: `FA_NATS_TEST=1 cargo test -p fa-bus -- --nocapture live_read`.
    /// Publishes three batches into a scratch stream with gaps between them and reads the
    /// middle batch back by time.
    #[tokio::test]
    async fn live_read_range_by_time() {
        if std::env::var("FA_NATS_TEST").is_err() {
            eprintln!("skipping live JetStream test (set FA_NATS_TEST=1)");
            return;
        }
        let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
        let client = async_nats::connect(&url).await.expect("connect");
        let js = jetstream::new(client.clone());
        let name = "FABUSREPLAY";
        let _ = js.delete_stream(name).await;
        js.get_or_create_stream(jetstream::stream::Config {
            name: name.into(),
            subjects: vec!["fabusreplay.>".into()],
            ..Default::default()
        })
        .await
        .expect("scratch stream");
        let now = || i64::try_from(std::time::UNIX_EPOCH.elapsed().unwrap().as_millis()).unwrap();
        let publish = |tag: &'static str, n: usize| {
            let js = js.clone();
            async move {
                for i in 0..n {
                    let st = state(tag, i as i64);
                    js.publish(format!("fabusreplay.{tag}.{i}"), st.encode_to_vec().into())
                        .await
                        .unwrap()
                        .await
                        .unwrap();
                }
            }
        };
        publish("a", 3).await;
        tokio::time::sleep(Duration::from_millis(300)).await;
        let t1 = now();
        tokio::time::sleep(Duration::from_millis(300)).await;
        publish("b", 4).await;
        tokio::time::sleep(Duration::from_millis(300)).await;
        let t2 = now();
        tokio::time::sleep(Duration::from_millis(300)).await;
        publish("c", 2).await;
        let opts = ReadOptions { idle: Duration::from_millis(800), max_messages: 100 };
        let mid = read_range(&js, name, "fabusreplay.>", t1, t2, &opts).await.expect("read");
        let subjects: Vec<&str> = mid.iter().map(|m| m.subject.as_str()).collect();
        eprintln!("[{t1}, {t2}] → {subjects:?}");
        assert_eq!(subjects, ["fabusreplay.b.0", "fabusreplay.b.1", "fabusreplay.b.2", "fabusreplay.b.3"]);
        assert!(mid.iter().all(|m| m.published_ms >= t1 && m.published_ms <= t2));
        assert!(mid.windows(2).all(|w| w[0].stream_seq < w[1].stream_seq));
        assert_eq!(LatentState::decode(mid[0].payload.as_slice()).unwrap().snapshot_id, "b");
        assert!(matches!(Recorded::decode(&mid[0]), Err(DecodeError::Subject(_))), "scratch subjects are not latent");
        let all = read_range(&js, name, "fabusreplay.>", 0, i64::MAX, &opts).await.expect("read all");
        assert_eq!(all.len(), 9);
        let only_b = read_range(&js, name, "fabusreplay.b.>", 0, i64::MAX, &opts).await.unwrap();
        assert_eq!(only_b.len(), 4, "filter subject applies");
        let started = std::time::Instant::now();
        let none = read_range(&js, name, "fabusreplay.>", now() + 60_000, now() + 120_000, &opts).await.unwrap();
        assert!(none.is_empty());
        assert!(started.elapsed() < Duration::from_millis(500), "a range after the last message returns at once");
        let capped = read_range(&js, name, "fabusreplay.>", 0, i64::MAX, &ReadOptions { max_messages: 2, ..opts })
            .await
            .unwrap();
        assert_eq!(capped.len(), 2);
        assert!(read_range(&js, name, "fabusreplay.>", 5, 4, &opts).await.is_err());
        js.delete_stream(name).await.unwrap();
        eprintln!("live_read_range_by_time: 9 published, middle batch {} read, scratch stream deleted", mid.len());
    }
}
