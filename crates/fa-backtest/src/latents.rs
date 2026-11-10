//! Recorded latents for strategy replay (spec 41 §3 "same code path as live", §6.1–6.2; spec 35
//! §6.1 "point-in-time inputs (expert views, graph state, decoder outputs) of the same
//! snapshot").
//!
//! The live `strategy-runtime` evaluates its programs through `fa_strategies::features::LiveContext`
//! fed by a [`LatentGraph`] (`latent.state.>` / `latent.region.>` / `latent.transition.>`) and a
//! [`DecoderOutputs`] cache (`decoder.output.>`). A replay rebuilds exactly that from the
//! `LATENT` JetStream stream (`fa_bus::replay::read_latents`, 7 d retention):
//!
//! * [`RecordedLatents`] holds the decoded records of one window in **knowledge-time order**
//!   (server publish time, then stream sequence) with a [`LatentSummary`];
//! * [`LatentFeed`] replays them into a fresh graph + decoder cache **incrementally**: before the
//!   decision at step time `t` only records published at or before `t` have been ingested, and
//!   the graph view (`LatentGraph::at(ts, t)`) additionally filters on the messages' own
//!   `as_of ≤ t`, so a latent published or dated after the step is never visible (test);
//! * [`check_snapshot`] enforces spec 41 §4 `ReplaySpec.snapshot`: latents of a different encoder
//!   snapshot than the declared one are refused unless the purpose is `Diagnostic`;
//! * [`RecordingCache`] / [`LatentCache`] cache loaded windows keyed by [`CacheKey`] (window,
//!   timescale, instrument set) so one load serves every candidate of a generation, and **extend**
//!   a cached recording when a new window overlaps it ([`CachePlan::Extend`]: only the missing
//!   ranges are read and merged in knowledge-time order; eviction by total messages), so the
//!   now-relative windows of consecutive jobs hit the cache instead of re-reading the stream.
//!
//! Windows that predate the recordings simply yield no graph/decoder data; every primitive then
//! falls back to the price statistics and the report's `sources.fallback` counts it (D-053).

use std::{
    collections::{BTreeSet, VecDeque},
    sync::Arc,
};

use fa_bus::replay::{LatentMessage, Recorded, RecordedMessage, decode_all, sort_records};
use fa_core::{InstrumentId, Timescale};
use fa_strategies::{decoders::DecoderOutputs, features::Sources, graph::LatentGraph};
use serde::{Deserialize, Serialize};

use crate::{
    ReplayError,
    replay::{ReplayPurpose, ReplaySpec},
};

/// Default recency window for decoder outputs (the live runtime's default).
pub const DEFAULT_DECODER_MAX_AGE_MS: i64 = fa_strategies::decoders::DEFAULT_MAX_AGE_MS;

/// What a recording contains.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatentSummary {
    pub messages: usize,
    pub states: usize,
    pub regions: usize,
    pub transitions: usize,
    pub decoder_outputs: usize,
    pub undecodable: usize,
    pub first_published_ms: Option<i64>,
    pub last_published_ms: Option<i64>,
    /// Distinct snapshot ids of the `latent.*` messages (all timescales).
    pub snapshots: Vec<String>,
    pub timescales: Vec<String>,
}

/// Decoded `LATENT` records of one window, ordered by knowledge time.
#[derive(Debug, Clone, Default)]
pub struct RecordedLatents {
    records: Vec<Recorded>,
    pub summary: LatentSummary,
}

impl RecordedLatents {
    /// Sorts, decodes and summarises raw records (undecodable ones are counted and dropped).
    pub fn new(mut raw: Vec<RecordedMessage>) -> Self {
        sort_records(&mut raw);
        let (records, undecodable) = decode_all(&raw);
        let mut summary = LatentSummary { messages: records.len(), undecodable, ..Default::default() };
        let mut snapshots = BTreeSet::new();
        let mut timescales = BTreeSet::new();
        for r in &records {
            match &r.message {
                LatentMessage::State(_) => summary.states += 1,
                LatentMessage::Region(_) => summary.regions += 1,
                LatentMessage::Transition(_) => summary.transitions += 1,
                LatentMessage::DecoderOutput(_) => summary.decoder_outputs += 1,
            }
            if r.message.is_latent() && !r.message.snapshot_id().is_empty() {
                snapshots.insert(r.message.snapshot_id().to_owned());
            }
            if let Some(ts) = r.message.timescale() {
                timescales.insert(ts.to_string());
            }
        }
        summary.first_published_ms = records.first().map(|r| r.published_ms);
        summary.last_published_ms = records.last().map(|r| r.published_ms);
        summary.snapshots = snapshots.into_iter().collect();
        summary.timescales = timescales.into_iter().collect();
        Self { records, summary }
    }

    /// From a `fa_bus::replay::to_jsonl` fixture.
    pub fn from_jsonl(text: &str) -> Result<Self, serde_json::Error> {
        Ok(Self::new(fa_bus::replay::from_jsonl(text)?))
    }

    pub fn records(&self) -> &[Recorded] {
        &self.records
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Distinct snapshot ids of the `latent.*` messages of `ts`.
    pub fn snapshots_of(&self, ts: Timescale) -> BTreeSet<String> {
        self.records
            .iter()
            .filter(|r| r.message.is_latent() && r.message.timescale() == Some(ts))
            .map(|r| r.message.snapshot_id().to_owned())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Instruments with a `LatentState` at `ts`.
    pub fn instruments_of(&self, ts: Timescale) -> BTreeSet<InstrumentId> {
        self.records
            .iter()
            .filter_map(|r| match &r.message {
                LatentMessage::State(s) if s.timescale.parse::<Timescale>().ok() == Some(ts) => {
                    uuid::Uuid::parse_str(&s.instrument_id).ok().map(InstrumentId)
                }
                _ => None,
            })
            .collect()
    }
}

/// Per-source primitive call counts (serialisable twin of `features::Sources`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceCounts {
    pub graph: u64,
    pub decoder: u64,
    pub fallback: u64,
}

impl From<Sources> for SourceCounts {
    fn from(s: Sources) -> Self {
        Self { graph: s.graph, decoder: s.decoder, fallback: s.fallback }
    }
}

impl SourceCounts {
    pub fn add(&mut self, s: Sources) {
        self.graph += s.graph;
        self.decoder += s.decoder;
        self.fallback += s.fallback;
    }

    pub fn total(&self) -> u64 {
        self.graph + self.decoder + self.fallback
    }
}

/// Records ingested so far by a [`LatentFeed`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedCounters {
    pub states: usize,
    pub regions: usize,
    pub transitions: usize,
    pub decoder_outputs: usize,
}

/// Incremental point-in-time feed of recorded latents into a graph + decoder cache.
pub struct LatentFeed<'a> {
    records: &'a [Recorded],
    cursor: usize,
    pub graph: LatentGraph,
    pub decoders: DecoderOutputs,
    pub decoder_max_age_ms: i64,
    pub fed: FeedCounters,
}

impl<'a> LatentFeed<'a> {
    pub fn new(rec: &'a RecordedLatents) -> Self {
        Self {
            records: &rec.records,
            cursor: 0,
            graph: LatentGraph::new(),
            decoders: DecoderOutputs::new(),
            decoder_max_age_ms: DEFAULT_DECODER_MAX_AGE_MS,
            fed: FeedCounters::default(),
        }
    }

    /// Ingests every record published at or before `now_ms` (knowledge time); returns how many.
    pub fn advance(&mut self, now_ms: i64) -> usize {
        let mut n = 0;
        while let Some(r) = self.records.get(self.cursor) {
            if r.published_ms > now_ms {
                break;
            }
            match &r.message {
                LatentMessage::State(s) => {
                    self.graph.on_state(s);
                    self.fed.states += 1;
                }
                LatentMessage::Region(ev) => {
                    self.graph.on_region(ev, r.published_ms);
                    self.fed.regions += 1;
                }
                LatentMessage::Transition(tu) => {
                    self.graph.on_transition(tu);
                    self.fed.transitions += 1;
                }
                LatentMessage::DecoderOutput(out) => {
                    self.decoders.on_output(out, r.published_ms);
                    self.fed.decoder_outputs += 1;
                }
            }
            self.cursor += 1;
            n += 1;
        }
        n
    }

    /// Records not yet visible.
    pub fn remaining(&self) -> usize {
        self.records.len() - self.cursor
    }
}

/// Outcome of [`check_snapshot`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotCheck {
    /// The snapshot the replay runs under (`spec.snapshot`, or the recording's when unset).
    pub declared: String,
    /// Snapshot ids seen in the latents of the replay's timescale.
    pub seen: Vec<String>,
    /// Latents of another snapshot are present (only allowed for `Diagnostic` replays).
    pub mixed: bool,
}

/// Spec 41 §4: a replay declares its encoder snapshot and must not mix latents of another one.
/// An empty `spec.snapshot` adopts the first snapshot of the recording.
pub fn check_snapshot(spec: &ReplaySpec, rec: &RecordedLatents) -> Result<SnapshotCheck, ReplayError> {
    let seen: Vec<String> = rec.snapshots_of(spec.timescale).into_iter().collect();
    let declared =
        if spec.snapshot.is_empty() { seen.first().cloned().unwrap_or_default() } else { spec.snapshot.clone() };
    let foreign: Vec<&String> = seen.iter().filter(|s| **s != declared).collect();
    let mixed = !foreign.is_empty();
    if mixed && spec.purpose != ReplayPurpose::Diagnostic {
        return Err(ReplayError::SnapshotMismatch {
            declared,
            found: foreign.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(","),
        });
    }
    Ok(SnapshotCheck { declared, seen, mixed })
}

/// Cache identity of a loaded window.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CacheKey {
    pub start_ms: i64,
    pub end_ms: i64,
    pub timescale: Timescale,
    /// Sorted, de-duplicated instrument tokens (empty = every instrument).
    pub instruments: Vec<String>,
}

impl CacheKey {
    pub fn of(spec: &ReplaySpec) -> Self {
        Self::new(spec.window_ms.0, spec.window_ms.1, spec.timescale, &spec.instruments)
    }

    pub fn new(start_ms: i64, end_ms: i64, timescale: Timescale, instruments: &[String]) -> Self {
        let mut instruments: Vec<String> = instruments.iter().map(|i| fa_decoders::instrument_token(i)).collect();
        instruments.sort();
        instruments.dedup();
        Self { start_ms, end_ms, timescale, instruments }
    }

    /// The window-free part: which recording this key addresses.
    pub fn identity(&self) -> RecordingId {
        RecordingId { timescale: self.timescale, instruments: self.instruments.clone() }
    }
}

/// Identity of one cached recording: everything of a [`CacheKey`] but the time range.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RecordingId {
    pub timescale: Timescale,
    pub instruments: Vec<String>,
}

/// What a cache lookup needs the caller to read (D-059 open item: incremental extension).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CachePlan {
    /// The cached recording covers the window: slice it.
    Hit,
    /// A cached recording overlaps (or touches) the window: read only `read` (at most a range
    /// before and a range after the cached span, each `[start, end]` inclusive) and
    /// [`RecordingCache::extend`].
    Extend { read: Vec<(i64, i64)> },
    /// Nothing usable: read the whole window and [`RecordingCache::insert`].
    Miss,
}

/// Forward extensions re-read this much of the cached tail so messages that arrived at the live
/// edge after the previous read are picked up (duplicates are dropped by stream sequence).
pub const EXTEND_OVERLAP_MS: i64 = 5_000;
/// Default [`RecordingCache`] budget in messages across every recording.
pub const DEFAULT_CACHE_MESSAGES: usize = 2_000_000;

/// One cached recording: raw messages in knowledge-time order covering `covered` (inclusive).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CachedRecording {
    pub covered: (i64, i64),
    pub records: Vec<RecordedMessage>,
}

impl CachedRecording {
    /// Records published within `[start, end]` (the records are sorted by publish time).
    pub fn slice(&self, start_ms: i64, end_ms: i64) -> Vec<RecordedMessage> {
        let lo = self.records.partition_point(|r| r.published_ms < start_ms);
        let hi = self.records.partition_point(|r| r.published_ms <= end_ms);
        if lo >= hi { Vec::new() } else { self.records[lo..hi].to_vec() }
    }
}

/// Bounded cache of **raw** stream recordings that grows a cached time range instead of
/// re-reading it: a window overlapping a cached span is served by reading only the missing
/// ranges and merging them in knowledge-time order ([`sort_records`]); the budget is counted in
/// messages over every recording (least recently used recordings evicted first, the oldest
/// records of a single oversized recording dropped). Pure: the caller does the stream reads.
#[derive(Debug)]
pub struct RecordingCache {
    max_messages: usize,
    /// Least recently used first.
    entries: VecDeque<(RecordingId, CachedRecording)>,
    pub hits: u64,
    pub extends: u64,
    pub misses: u64,
}

impl Default for RecordingCache {
    fn default() -> Self {
        Self::new(DEFAULT_CACHE_MESSAGES)
    }
}

impl RecordingCache {
    pub fn new(max_messages: usize) -> Self {
        Self { max_messages: max_messages.max(1), entries: VecDeque::new(), hits: 0, extends: 0, misses: 0 }
    }

    fn position(&self, id: &RecordingId) -> Option<usize> {
        self.entries.iter().position(|(k, _)| k == id)
    }

    fn touch(&mut self, i: usize) -> &mut CachedRecording {
        let e = self.entries.remove(i).expect("position exists");
        self.entries.push_back(e);
        &mut self.entries.back_mut().expect("just pushed").1
    }

    /// Decides how to serve `[start, end]` for `id` (counts the outcome, marks the entry used).
    pub fn plan(&mut self, id: &RecordingId, start_ms: i64, end_ms: i64) -> CachePlan {
        let Some(i) = self.position(id) else {
            self.misses += 1;
            return CachePlan::Miss;
        };
        let (c0, c1) = self.entries[i].1.covered;
        if c0 <= start_ms && c1 >= end_ms {
            self.hits += 1;
            self.touch(i);
            return CachePlan::Hit;
        }
        // overlapping or touching (the next millisecond counts as touching)
        if start_ms > c1 + 1 || end_ms < c0 - 1 {
            self.misses += 1;
            return CachePlan::Miss;
        }
        let mut read = Vec::new();
        if start_ms < c0 {
            read.push((start_ms, c0 - 1));
        }
        if end_ms > c1 {
            read.push(((c1 - EXTEND_OVERLAP_MS).max(c0), end_ms));
        }
        self.extends += 1;
        self.touch(i);
        CachePlan::Extend { read }
    }

    /// Records of `id` published within `[start, end]`, when the span is covered.
    pub fn slice(&self, id: &RecordingId, start_ms: i64, end_ms: i64) -> Option<Vec<RecordedMessage>> {
        let (_, rec) = self.entries.iter().find(|(k, _)| k == id)?;
        (rec.covered.0 <= start_ms && rec.covered.1 >= end_ms).then(|| rec.slice(start_ms, end_ms))
    }

    /// Replaces the recording of `id` with `records` covering `[start, end]`.
    pub fn insert(&mut self, id: &RecordingId, start_ms: i64, end_ms: i64, mut records: Vec<RecordedMessage>) {
        sort_records(&mut records);
        self.entries.retain(|(k, _)| k != id);
        self.entries.push_back((id.clone(), CachedRecording { covered: (start_ms, end_ms), records }));
        self.evict();
    }

    /// Merges `extra` (read for the ranges of a [`CachePlan::Extend`]) into the recording of
    /// `id`; the covered span becomes the union with `[start, end]`. Falls back to an insert when
    /// the recording was evicted meanwhile.
    pub fn extend(&mut self, id: &RecordingId, start_ms: i64, end_ms: i64, extra: Vec<RecordedMessage>) {
        let Some(i) = self.position(id) else {
            return self.insert(id, start_ms, end_ms, extra);
        };
        let rec = self.touch(i);
        rec.records.extend(extra);
        sort_records(&mut rec.records);
        rec.covered = (rec.covered.0.min(start_ms), rec.covered.1.max(end_ms));
        self.evict();
    }

    /// Enforces the message budget: least recently used recordings go first; a single recording
    /// above the budget loses its oldest records (its covered start moves to the first kept one).
    fn evict(&mut self) {
        while self.entries.len() > 1 && self.total_messages() > self.max_messages {
            self.entries.pop_front();
        }
        if let Some((_, rec)) = self.entries.back_mut()
            && rec.records.len() > self.max_messages
        {
            let drop = rec.records.len() - self.max_messages;
            rec.records.drain(..drop);
            rec.covered.0 = rec.records.first().map_or(rec.covered.1, |r| r.published_ms);
        }
    }

    pub fn total_messages(&self) -> usize {
        self.entries.iter().map(|(_, r)| r.records.len()).sum()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Covered span of a recording.
    pub fn covered(&self, id: &RecordingId) -> Option<(i64, i64)> {
        self.entries.iter().find(|(k, _)| k == id).map(|(_, r)| r.covered)
    }
}

/// [`RecordingCache`] for `LATENT` recordings plus a small memo of decoded windows, so repeated
/// identical windows share one [`RecordedLatents`] and overlapping windows share the raw records.
#[derive(Debug)]
pub struct LatentCache {
    pub raw: RecordingCache,
    decoded: VecDeque<(CacheKey, Arc<RecordedLatents>)>,
}

/// Decoded windows kept by [`LatentCache`].
pub const DECODED_MEMO: usize = 4;

impl Default for LatentCache {
    fn default() -> Self {
        Self::new(DEFAULT_CACHE_MESSAGES)
    }
}

impl LatentCache {
    pub fn new(max_messages: usize) -> Self {
        Self { raw: RecordingCache::new(max_messages), decoded: VecDeque::new() }
    }

    /// How to serve `key` (see [`RecordingCache::plan`]).
    pub fn plan(&mut self, key: &CacheKey) -> CachePlan {
        self.raw.plan(&key.identity(), key.start_ms, key.end_ms)
    }

    /// The decoded window after a [`CachePlan::Hit`] (`None` when not covered).
    pub fn get(&mut self, key: &CacheKey) -> Option<Arc<RecordedLatents>> {
        if let Some(i) = self.decoded.iter().position(|(k, _)| k == key) {
            let e = self.decoded.remove(i).expect("position exists");
            let v = e.1.clone();
            self.decoded.push_back(e);
            return Some(v);
        }
        let raw = self.raw.slice(&key.identity(), key.start_ms, key.end_ms)?;
        Some(self.memo(key, raw))
    }

    fn memo(&mut self, key: &CacheKey, raw: Vec<RecordedMessage>) -> Arc<RecordedLatents> {
        let v = Arc::new(RecordedLatents::new(raw));
        self.decoded.retain(|(k, _)| k != key);
        self.decoded.push_back((key.clone(), v.clone()));
        while self.decoded.len() > DECODED_MEMO {
            self.decoded.pop_front();
        }
        v
    }

    /// Stores a freshly read window and returns it decoded.
    pub fn insert(&mut self, key: &CacheKey, records: Vec<RecordedMessage>) -> Arc<RecordedLatents> {
        self.raw.insert(&key.identity(), key.start_ms, key.end_ms, records);
        self.decoded.clear();
        let raw = self.raw.slice(&key.identity(), key.start_ms, key.end_ms).unwrap_or_default();
        self.memo(key, raw)
    }

    /// Merges the records read for a [`CachePlan::Extend`] and returns the window decoded.
    pub fn extend(&mut self, key: &CacheKey, extra: Vec<RecordedMessage>) -> Arc<RecordedLatents> {
        self.raw.extend(&key.identity(), key.start_ms, key.end_ms, extra);
        self.decoded.clear();
        let raw = self.raw.slice(&key.identity(), key.start_ms, key.end_ms).unwrap_or_default();
        self.memo(key, raw)
    }

    pub fn hits(&self) -> u64 {
        self.raw.hits
    }
    pub fn extends(&self) -> u64 {
        self.raw.extends
    }
    pub fn misses(&self) -> u64 {
        self.raw.misses
    }
    /// Recordings held.
    pub fn len(&self) -> usize {
        self.raw.len()
    }
    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }
    pub fn total_messages(&self) -> usize {
        self.raw.total_messages()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use fa_proto::{
        Message,
        v1::{DecoderOutput, ExpertLatent, ExpertSpaceRef, LatentState, RegionEvent, TransitionProb, TransitionUpdate},
    };

    use super::*;

    pub const FIXTURE: &str = include_str!("../testdata/latent_fixture.jsonl");

    pub fn pts(ms: i64) -> fa_proto::Timestamp {
        fa_proto::Timestamp { seconds: ms.div_euclid(1000), nanos: (ms.rem_euclid(1000) * 1_000_000) as i32 }
    }

    pub fn raw(subject: &str, seq: u64, published_ms: i64, payload: Vec<u8>) -> RecordedMessage {
        RecordedMessage { subject: subject.into(), stream_seq: seq, published_ms, payload }
    }

    pub fn state_msg(snapshot: &str, inst: InstrumentId, as_of_ms: i64, region: &str) -> Vec<u8> {
        LatentState {
            snapshot_id: snapshot.into(),
            timescale: "1s".into(),
            instrument_id: inst.to_string(),
            as_of: Some(pts(as_of_ms)),
            experts: vec![ExpertLatent {
                expert_index: 0,
                mean: vec![0.0],
                log_var: vec![(0.1f32).ln() * 2.0],
                region_id: region.into(),
                region_lineage_id: region.into(),
            }],
            router_weights: vec![],
        }
        .encode_to_vec()
    }

    pub fn row_msg(snapshot: &str, from: &str, as_of_ms: i64, to: &[(&str, f64)]) -> Vec<u8> {
        TransitionUpdate {
            space: Some(ExpertSpaceRef { snapshot_id: snapshot.into(), timescale: "1s".into(), expert_index: 0 }),
            from_region_id: from.into(),
            to: to
                .iter()
                .map(|(r, m)| TransitionProb {
                    region_id: (*r).into(),
                    mean: *m,
                    lower: (m - 0.05).max(0.0),
                    upper: (m + 0.05).min(1.0),
                })
                .collect(),
            sampler_id: "conjugate".into(),
            as_of: Some(pts(as_of_ms)),
        }
        .encode_to_vec()
    }

    pub fn region_msg(snapshot: &str, id: &str) -> Vec<u8> {
        RegionEvent {
            kind: 0,
            space: Some(ExpertSpaceRef { snapshot_id: snapshot.into(), timescale: "1s".into(), expert_index: 0 }),
            region_id: id.into(),
            lineage_id: id.into(),
            related_region_ids: vec![],
        }
        .encode_to_vec()
    }

    pub fn forecast_msg(inst: InstrumentId, as_of_ms: i64, q: &[f64]) -> Vec<u8> {
        DecoderOutput {
            decoder_id: "d".into(),
            genome_id: "g".into(),
            family: "rnn_prior".into(),
            snapshot_id: "snap".into(),
            timescale: "1s".into(),
            as_of: Some(pts(as_of_ms)),
            output_refs: vec![format!("inst:{inst}")],
            quantiles: q.to_vec(),
            analogs: vec![],
        }
        .encode_to_vec()
    }

    /// SYNTHETIC toy recording: regions, one row, states every second, a forecast.
    pub fn toy(inst: InstrumentId, snapshot: &str) -> Vec<RecordedMessage> {
        let mut v = vec![
            raw("latent.region.1s", 1, 500, region_msg(snapshot, "a")),
            raw("latent.region.1s", 2, 500, region_msg(snapshot, "b")),
            raw(
                "latent.transition.1s",
                3,
                900,
                row_msg(snapshot, "a", 900, &[("b", 0.7), ("a", 0.2), ("unknown", 0.1)]),
            ),
            raw(
                "latent.transition.1s",
                4,
                900,
                row_msg(snapshot, "b", 900, &[("a", 0.6), ("b", 0.3), ("unknown", 0.1)]),
            ),
        ];
        for t in 1..=20 {
            let ms = t * 1000;
            let region = if t < 10 { "a" } else { "b" };
            v.push(raw(
                &format!("latent.state.1s.{}", inst.subject_token()),
                10 + t as u64,
                ms + 50,
                state_msg(snapshot, inst, ms, region),
            ));
        }
        v.push(raw(
            "decoder.output.rnn_prior",
            100,
            5_100,
            forecast_msg(inst, 5_000, &[-0.02, -0.01, 0.0, 0.01, 0.03]),
        ));
        v
    }

    #[test]
    fn summary_feed_and_point_in_time() {
        let inst = InstrumentId::new();
        let rec = RecordedLatents::new(toy(inst, "snap-1"));
        let s = &rec.summary;
        assert_eq!(
            (s.messages, s.states, s.regions, s.transitions, s.decoder_outputs, s.undecodable),
            (25, 20, 2, 2, 1, 0)
        );
        assert_eq!(s.snapshots, vec!["snap-1".to_owned()]);
        assert_eq!(s.timescales, vec!["1s".to_owned()]);
        assert_eq!((s.first_published_ms, s.last_published_ms), (Some(500), Some(20_050)));
        assert_eq!(rec.instruments_of(Timescale::Second).into_iter().collect::<Vec<_>>(), vec![inst]);
        assert!(rec.instruments_of(Timescale::Minute).is_empty());
        // knowledge-time order even though sequences interleave
        assert!(rec.records().windows(2).all(|w| w[0].published_ms <= w[1].published_ms));

        let mut feed = LatentFeed::new(&rec);
        assert_eq!(feed.advance(999), 4, "regions and rows only");
        assert_eq!(feed.graph.at(Timescale::Second, 999).cur_region(inst, 0), None);
        // the state dated 1 000 is published at 1 050: invisible at 1 000, visible at 1 050
        assert_eq!(feed.advance(1_000), 0);
        assert_eq!(feed.graph.at(Timescale::Second, 1_000).cur_region(inst, 0), None, "published after the step");
        assert_eq!(feed.advance(1_050), 1);
        assert_eq!(feed.graph.at(Timescale::Second, 1_050).cur_region(inst, 0).as_deref(), Some("a"));
        // the forecast dated 5 000 is published at 5 100
        feed.advance(5_050);
        assert_eq!(feed.decoders.at(inst, 5_050, DEFAULT_DECODER_MAX_AGE_MS).fc_dist(1), None, "published at 5 100");
        feed.advance(5_100);
        assert!(feed.decoders.at(inst, 5_100, DEFAULT_DECODER_MAX_AGE_MS).fc_dist(1).is_some());
        // jump: everything up to 9 s; the switch to b (as_of 10 s) is not visible
        feed.advance(9_999);
        let v = feed.graph.at(Timescale::Second, 9_999);
        assert_eq!(v.cur_region(inst, 0).as_deref(), Some("a"));
        assert!((v.p_trans(0, "a", "b", 1).unwrap() - 0.7).abs() < 0.02);
        feed.advance(i64::MAX);
        assert_eq!(feed.remaining(), 0);
        assert_eq!(feed.fed, FeedCounters { states: 20, regions: 2, transitions: 2, decoder_outputs: 1 });
        // a view fixed at an earlier time still hides later states even after they were fed
        assert_eq!(feed.graph.at(Timescale::Second, 9_999).cur_region(inst, 0).as_deref(), Some("a"));
        assert_eq!(feed.graph.at(Timescale::Second, 20_000).cur_region(inst, 0).as_deref(), Some("b"));
    }

    #[test]
    fn snapshot_check_and_cache() {
        let inst = InstrumentId::new();
        let mut raw_msgs = toy(inst, "snap-1");
        raw_msgs.extend(toy(inst, "snap-2").into_iter().filter(|r| r.subject.starts_with("latent.state")).map(
            |mut r| {
                r.stream_seq += 1000;
                r
            },
        ));
        let mixed = RecordedLatents::new(raw_msgs);
        assert_eq!(mixed.summary.snapshots, vec!["snap-1".to_owned(), "snap-2".to_owned()]);
        let mut spec = ReplaySpec::selection(Timescale::Second, 0, 30_000);
        spec.snapshot = "snap-1".into();
        assert!(matches!(
            check_snapshot(&spec, &mixed),
            Err(ReplayError::SnapshotMismatch { declared, found }) if declared == "snap-1" && found == "snap-2"
        ));
        spec.purpose = ReplayPurpose::Diagnostic;
        let c = check_snapshot(&spec, &mixed).unwrap();
        assert!(c.mixed && c.seen.len() == 2 && c.declared == "snap-1");
        // an undeclared snapshot adopts the recording's; a clean recording passes
        let clean = RecordedLatents::new(toy(inst, "snap-9"));
        let spec = ReplaySpec::selection(Timescale::Second, 0, 30_000);
        assert_eq!(
            check_snapshot(&spec, &clean).unwrap(),
            SnapshotCheck { declared: "snap-9".into(), seen: vec!["snap-9".into()], mixed: false }
        );
        // another timescale's latents do not count
        let spec_m = ReplaySpec { snapshot: "other".into(), ..ReplaySpec::selection(Timescale::Minute, 0, 30_000) };
        assert!(!check_snapshot(&spec_m, &mixed).unwrap().mixed);
        // empty recording: nothing to check
        assert_eq!(check_snapshot(&spec, &RecordedLatents::default()).unwrap().declared, "");

        let k = |s: i64, insts: &[&str]| {
            CacheKey::of(&ReplaySpec {
                instruments: insts.iter().map(|i| (*i).to_owned()).collect(),
                ..ReplaySpec::selection(Timescale::Second, s, s + 10)
            })
        };
        assert_eq!(
            k(0, &["B-1", "a", "a"]).instruments,
            vec!["a".to_owned(), "b1".to_owned()],
            "tokens sorted, deduped"
        );
        assert_eq!(k(0, &["a"]).identity(), k(5, &["A"]).identity(), "identity ignores the window");
        assert_ne!(k(0, &["a"]).identity(), k(0, &[]).identity());
    }

    /// Pure cache extension logic (D-059 open item): overlapping windows read only the missing
    /// ranges, hits are exact slices, misses replace, the message budget evicts.
    #[test]
    fn recording_cache_extends_slices_and_evicts() {
        let inst = InstrumentId::new();
        let msgs = |from: i64, to: i64| -> Vec<RecordedMessage> {
            (from..=to)
                .map(|t| raw("latent.state.1s.x", t as u64, t * 1000, state_msg("s", inst, t * 1000, "a")))
                .collect()
        };
        let id = RecordingId { timescale: Timescale::Second, instruments: vec![] };
        let other = RecordingId { timescale: Timescale::Minute, instruments: vec![] };
        let mut c = RecordingCache::new(100);
        assert_eq!(c.plan(&id, 10_000, 20_000), CachePlan::Miss);
        c.insert(&id, 10_000, 20_000, msgs(10, 20));
        assert_eq!(c.covered(&id), Some((10_000, 20_000)));
        // exact and inner windows hit; the slice is exactly the requested publish range
        assert_eq!(c.plan(&id, 10_000, 20_000), CachePlan::Hit);
        assert_eq!(c.plan(&id, 12_000, 15_000), CachePlan::Hit);
        let sl = c.slice(&id, 12_000, 15_000).unwrap();
        assert_eq!(sl.iter().map(|r| r.published_ms).collect::<Vec<_>>(), [12_000, 13_000, 14_000, 15_000]);
        assert!(c.slice(&id, 9_000, 15_000).is_none(), "not covered → no slice");
        // a now-relative window moving forward: only the tail (with the overlap) is read
        let plan = c.plan(&id, 13_000, 25_000);
        assert_eq!(plan, CachePlan::Extend { read: vec![(20_000 - EXTEND_OVERLAP_MS, 25_000)] });
        c.extend(&id, 13_000, 25_000, msgs(15, 25)); // overlap duplicates 15..20
        assert_eq!(c.covered(&id), Some((10_000, 25_000)));
        assert_eq!(c.total_messages(), 16, "duplicates dropped by stream sequence");
        assert_eq!(c.plan(&id, 13_000, 25_000), CachePlan::Hit);
        assert_eq!(c.slice(&id, 13_000, 25_000).unwrap().len(), 13);
        // backward + forward at once; touching ranges count as overlapping
        assert_eq!(
            c.plan(&id, 5_000, 30_000),
            CachePlan::Extend { read: vec![(5_000, 9_999), (25_000 - EXTEND_OVERLAP_MS, 30_000)] }
        );
        assert_eq!(c.plan(&id, 25_001, 30_000), CachePlan::Extend { read: vec![(20_000, 30_000)] });
        // a disjoint window is a miss and the insert replaces the recording
        assert_eq!(c.plan(&id, 40_000, 50_000), CachePlan::Miss);
        assert_eq!(c.plan(&id, 0, 8_000), CachePlan::Miss);
        assert_eq!((c.hits, c.extends, c.misses), (3, 3, 3));
        // another identity is independent; the budget evicts least recently used recordings first
        let mut c = RecordingCache::new(100);
        c.insert(&id, 10_000, 20_000, msgs(10, 20));
        c.insert(&other, 0, 1_000_000, msgs(0, 90));
        assert_eq!(c.len(), 1, "11 + 91 > 100: least recently used recording evicted");
        assert_eq!(c.plan(&id, 10_000, 20_000), CachePlan::Miss);
        assert_eq!(c.plan(&other, 0, 1_000_000), CachePlan::Hit);
        // a single oversized recording loses its oldest records and its covered start moves
        c.extend(&other, 0, 1_000_000, msgs(91, 130));
        assert_eq!(c.total_messages(), 100);
        assert_eq!(c.covered(&other), Some((31_000, 1_000_000)));
        assert_eq!(c.plan(&other, 0, 1_000_000), CachePlan::Extend { read: vec![(0, 30_999)] });
        // extending an evicted identity falls back to an insert
        c.extend(&id, 1, 2, msgs(1, 2));
        assert_eq!(c.covered(&id), Some((1, 2)));
    }

    #[test]
    fn latent_cache_decodes_slices_and_memoises() {
        let inst = InstrumentId::new();
        let mut cache = LatentCache::new(1_000);
        let key = |s: i64, e: i64| CacheKey::new(s, e, Timescale::Second, &[]);
        assert_eq!(cache.plan(&key(0, 30_000)), CachePlan::Miss);
        let full = cache.insert(&key(0, 30_000), toy(inst, "snap-1"));
        assert_eq!(full.len(), 25);
        assert_eq!(cache.plan(&key(0, 30_000)), CachePlan::Hit);
        assert!(Arc::ptr_eq(&cache.get(&key(0, 30_000)).unwrap(), &full), "memoised decode");
        // an inner window is a different (smaller) recording: knowledge-time slice
        assert_eq!(cache.plan(&key(1_000, 5_000)), CachePlan::Hit);
        let inner = cache.get(&key(1_000, 5_000)).unwrap();
        assert_eq!(inner.summary.states, 4, "states published 1 050 … 4 050");
        assert_eq!(inner.summary.regions, 0, "regions published at 500 are outside the window");
        assert!(cache.get(&key(-1, 5_000)).is_none(), "not covered");
        // extension merges and re-slices
        let more: Vec<RecordedMessage> = (21..=30i64)
            .map(|t| {
                raw("latent.state.1s.x", (100 + t) as u64, t * 1000 + 50, state_msg("snap-1", inst, t * 1000, "b"))
            })
            .collect();
        assert!(matches!(cache.plan(&key(10_000, 40_000)), CachePlan::Extend { .. }));
        let ext = cache.extend(&key(10_000, 40_000), more);
        assert_eq!(ext.summary.states, 21, "states published 10 050 … 30 050");
        assert_eq!((cache.hits(), cache.extends(), cache.misses(), cache.len()), (2, 1, 1, 1));
        assert_eq!(cache.total_messages(), 35);
    }

    #[test]
    fn fixture_is_a_real_latent_recording() {
        let rec = RecordedLatents::from_jsonl(FIXTURE).expect("fixture parses");
        let s = &rec.summary;
        assert!(s.messages >= 200, "{s:?}");
        assert!(s.states > 0 && s.transitions > 0, "{s:?}");
        assert_eq!(s.undecodable, 0);
        assert!(!s.snapshots.is_empty());
        eprintln!("fixture: {s:?}");
    }

    /// Writes `testdata/latent_fixture.jsonl` from the live stream:
    /// `FA_LATENT_FIXTURE_DUMP=1 cargo test -p fa-backtest -- --nocapture dump_fixture`
    /// (`FA_LATENT_FIXTURE_MINUTES` back from now, default 3; `FA_NATS_URL`).
    #[tokio::test]
    async fn dump_fixture_from_live_stream() {
        if std::env::var("FA_LATENT_FIXTURE_DUMP").is_err() {
            return;
        }
        let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
        let minutes: i64 = std::env::var("FA_LATENT_FIXTURE_MINUTES").ok().and_then(|m| m.parse().ok()).unwrap_or(3);
        let client = fa_bus::async_nats::connect(&url).await.expect("connect");
        let js = fa_bus::async_nats::jetstream::new(client);
        let now = i64::try_from(std::time::UNIX_EPOCH.elapsed().unwrap().as_millis()).unwrap();
        let start = now - minutes * 60_000;
        let opts = fa_bus::replay::ReadOptions { max_messages: 400, ..Default::default() };
        let mut all = Vec::new();
        // states and rows from the window; the sparse region events and decoder outputs from
        // wherever the stream still has them
        for (f, cap, from) in [
            ("latent.state.>", 300usize, start),
            ("latent.region.>", 60, 0),
            ("latent.transition.>", 120, start),
            ("decoder.output.>", 60, 0),
        ] {
            let mut got = fa_bus::replay::read_range(&js, fa_bus::replay::LATENT_STREAM, f, from, now, &opts)
                .await
                .expect("read");
            got.truncate(cap);
            eprintln!("{f}: {} kept", got.len());
            all.extend(got);
        }
        let rec = RecordedLatents::new(all.clone());
        eprintln!("summary: {:?}", rec.summary);
        sort_records(&mut all);
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/latent_fixture.jsonl");
        std::fs::write(path, fa_bus::replay::to_jsonl(&all)).expect("write fixture");
        eprintln!("wrote {} records to {path}", all.len());
    }
}
