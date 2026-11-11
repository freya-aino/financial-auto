//! Signal-for-signal replay of the **deployed** strategy runtime (spec 41 §9: replaying a
//! recorded live day reproduces the live decisions).
//!
//! The deployed service subscribes to `md.trade.>`,
//! `latent.state|shadow|region|transition|lifecycle.>` and `decoder.output.>`, feeds every message into [`Runtime`] on arrival and calls
//! [`Runtime::tick`] every 250 ms; each step's distributions go out on
//! `signal.exposure.<id>` stamped with the tick time (`as_of`). All of it is retained in
//! JetStream (`MD` 24 h, `LATENT` 7 d, `SIGNAL` 30 d). The replay rebuilds the service from
//! those streams alone:
//!
//! * a **cold** [`Runtime`] with the deployed population and settings at the process start
//!   (the service keeps no state across restarts; the region lineage it seeds from the last 7 d
//!   of `latent.lifecycle.>` is applied first, as `main.rs` does);
//! * every input in **knowledge-time order** (stream publish time, then stream sequence),
//!   dispatched by [`apply`] exactly as `main.rs` does (a test pins the subject set to
//!   `main.rs`'s subscriptions);
//! * a tick at every `as_of` the live service emitted at, after the inputs published at or
//!   before it. While the 1 s strategies run, every live tick that opens a new 1 s bucket emits,
//!   so the ticks that emitted nothing change nothing the replay misses (they only re-check due
//!   times);
//! * the replayed distributions compared with the live ones per `(as_of, strategy)`:
//!   byte-identical protobuf, equal within 1e-9, different, live-only or replay-only.
//!
//! Two inputs do not come from the bus. **Analog outcome prices** the in-memory history cannot
//! price are fetched by the service from VictoriaMetrics between steps (`prefetch.rs`, every
//! 500 ms, 32 per round): the replay drains the same queue after every tick through an
//! [`AnalogResolver`] — VictoriaMetrics again for a stream replay (its fills are recorded as
//! `analog.fill` records), the recorded fills for a fixture — so the next tick resolves them as
//! the live one did. The **evaluation-time budget** of spec 35 §8 (wall clock) is not
//! replayable, so its outcome is taken from the record: a known strategy the live service did
//! not emit at a tick it emitted others at was skipped for its budget (the only reason a due
//! strategy stays silent), and the replay's clock reports it over budget there — the replayed
//! runtime then skips, counts the streak and auto-pauses exactly as the live one did. All other
//! strategies measure zero.
//!
//! **Publisher restarts.** JetStream drops a re-published message whose `Nats-Msg-Id` it stored
//! within the stream's duplicate window (`MD`: 60 s), but core subscribers — the service —
//! still receive it. The Kraken connector re-emits each symbol's last trade before a restart
//! with its first poll after it (its `since` cursor is persisted, the trade-id filter is not),
//! so after an ingestor restart the service may have seen trades that the stream holds only at
//! their first publication. Which symbols were re-sent depends on the connector's persisted
//! cursors at the restart, which no stream records: [`redelivery_candidates`] lists, per Kraken
//! symbol, the last trade before the restart re-delivered with the symbol's first batch after
//! it (within the window), and [`infer_redelivery`] keeps a candidate only where it makes the
//! replay of the first ticks agree better with the live record — an initial condition pinned
//! from the live output, like the subscription time. Re-sent trades the inference cannot place
//! (they leave no trace in the stream) only matter until each symbol's next batch replaces the
//! bars they started: the comparison counts that **cold-start transient** (until the second
//! post-restart publication of every Kraken symbol, [`transient_end`]) separately, and the
//! acceptance is on the steady state after it.
//!
//! Tests: `signal_fixture_reproduces_the_deployed_runtime` (always run) replays
//! `testdata/signal_fixture.jsonl`, the first two minutes after a deployed restart cut from the
//! streams; `deployed_signals_from_retained_streams` (ignored, `FA_REPLAY_FROM_MS=<process
//! start>`) replays any retained span in chunks and can rewrite the fixture.
#![cfg(test)]

use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    time::Duration,
};

use fa_bus::replay::RecordedMessage;
use fa_core::InstrumentId;
use fa_proto::{
    Message,
    v1::{
        DecoderOutput, Envelope, ExposureDistribution, LatentState, RegionEvent, RegionLifecycleEvent,
        TransitionUpdate, envelope::Payload,
    },
};
use fa_strategies::population::StrategyFile;
use serde::Serialize;
use uuid::Uuid;

use fa_strategies::decoders::AnalogRequest;

use crate::runtime::{AnalogKey, Runtime, Settings};

/// Input subjects of the deployed service (`main.rs` subscriptions) and their streams.
pub const INPUTS: [(&str, &str); 7] = [
    ("MD", "md.trade.>"),
    ("LATENT", "latent.state.>"),
    ("LATENT", "latent.shadow.>"),
    ("LATENT", "latent.region.>"),
    ("LATENT", "latent.transition.>"),
    ("LATENT", "latent.lifecycle.>"),
    ("LATENT", "decoder.output.>"),
];
pub const SIGNALS: (&str, &str) = ("SIGNAL", "signal.exposure.>");

fn ms(t: Option<&fa_proto::Timestamp>) -> Option<i64> {
    t.map(|t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000)
}

/// Applies one recorded input as `main.rs` does (receive time = stream publish time). Returns
/// `false` for subjects the service does not consume.
pub fn apply(rt: &mut Runtime, r: &RecordedMessage) -> bool {
    let p = r.payload.as_slice();
    let now = r.published_ms;
    let s = r.subject.as_str();
    if s.starts_with("md.trade.") {
        if let Ok(env) = Envelope::decode(p)
            && let Some(Payload::Trade(t)) = &env.payload
            && let Ok(uuid) = Uuid::parse_str(&env.instrument_id)
        {
            rt.on_trade(InstrumentId(uuid), ms(env.event_time.as_ref()).unwrap_or(now), t.price);
        }
    } else if s.starts_with("latent.state.") {
        if let Ok(m) = LatentState::decode(p) {
            rt.on_latent(&m);
        }
    } else if s.starts_with("latent.shadow.") {
        if let Ok(m) = LatentState::decode(p) {
            rt.on_parallel_latent(&m);
        }
    } else if s.starts_with("latent.region.") {
        if let Ok(m) = RegionEvent::decode(p) {
            rt.on_region(&m, now);
        }
    } else if s.starts_with("latent.transition.") {
        if let Ok(m) = TransitionUpdate::decode(p) {
            rt.on_transition(&m);
        }
    } else if s.starts_with("latent.lifecycle.") {
        if let Ok(m) = RegionLifecycleEvent::decode(p) {
            rt.on_lifecycle(&m);
        }
    } else if s.starts_with("decoder.output.") {
        if let Ok(m) = DecoderOutput::decode(p) {
            rt.on_decoder_output(&m, now);
        }
    } else {
        return false;
    }
    true
}

/// How one live distribution compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct Counts {
    /// Byte-identical protobuf.
    pub identical: u64,
    /// Same strategy, genome, horizon and instruments; every mean / stddev within 1e-9.
    pub equal: u64,
    pub different: u64,
    /// Emitted live, not by the replay.
    pub live_only: u64,
    /// Emitted by the replay, not live (e.g. a live step skipped for its time budget).
    pub replay_only: u64,
}

impl Counts {
    pub fn add(&mut self, o: &Counts) {
        self.identical += o.identical;
        self.equal += o.equal;
        self.different += o.different;
        self.live_only += o.live_only;
        self.replay_only += o.replay_only;
    }

    pub fn live(&self) -> u64 {
        self.identical + self.equal + self.different + self.live_only
    }

    /// Share of live distributions reproduced (identical or equal).
    pub fn reproduced(&self) -> f64 {
        (self.identical + self.equal) as f64 / self.live().max(1) as f64
    }
}

/// Comparison of a replayed span.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Comparison {
    pub ticks: u64,
    pub inputs: u64,
    pub total: Counts,
    /// Ticks before [`Replayer::transient_until`] (included in `total`).
    pub transient: Counts,
    /// `total − transient`.
    pub steady: Counts,
    pub by_strategy: BTreeMap<String, Counts>,
    /// Live strategies the replayed population does not contain (not compared).
    pub unknown: BTreeMap<String, u64>,
    /// First differences, `as_of strategy: detail`.
    pub examples: Vec<String>,
    /// Largest |Δ| (mean or stddev) among different distributions.
    pub max_abs_diff: f64,
    /// Largest |Δ| among distributions counted `equal` (floating-point noise).
    pub max_equal_diff: f64,
}

fn equal_within(a: &ExposureDistribution, b: &ExposureDistribution, eps: f64) -> Result<(), f64> {
    if a.strategy_id != b.strategy_id
        || a.genome_id != b.genome_id
        || a.horizon != b.horizon
        || a.exposures.len() != b.exposures.len()
    {
        return Err(f64::INFINITY);
    }
    let mut worst = 0.0f64;
    for (x, y) in a.exposures.iter().zip(&b.exposures) {
        if x.instrument_id != y.instrument_id {
            return Err(f64::INFINITY);
        }
        worst = worst.max((x.mean - y.mean).abs()).max((x.stddev - y.stddev).abs());
    }
    if worst <= eps { Ok(()) } else { Err(worst) }
}

/// Fetch result of one queued analog (`AnalogCache::fill` input).
pub type AnalogFill = (AnalogKey, Result<Option<(f64, f64)>, ()>);

/// Resolves the analogs queued by the tick at `t_ms` (the service's prefetch).
pub trait AnalogResolver {
    fn resolve(&mut self, t_ms: i64, keys: Vec<AnalogKey>) -> Vec<AnalogFill>;
}

/// Subject of a recorded analog fill (payload: [`FillRecord`] JSON, publish time = the tick).
pub const ANALOG_FILL: &str = "analog.fill";

/// Serialised [`AnalogFill`].
#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
pub struct FillRecord {
    pub instrument: String,
    pub window_end_ms: i64,
    pub k: u32,
    pub bar: String,
    /// `Some(Some((p0, p1)))` resolved, `Some(None)` nothing found, `None` failed fetch.
    pub result: Option<Option<(f64, f64)>>,
}

impl FillRecord {
    pub fn of(f: &AnalogFill) -> Self {
        let ((inst, req), r) = f;
        Self {
            instrument: inst.to_string(),
            window_end_ms: req.window_end_ms,
            k: req.k,
            bar: req.bar.to_string(),
            result: r.as_ref().ok().copied(),
        }
    }

    pub fn fill(&self) -> Option<AnalogFill> {
        let inst = InstrumentId(Uuid::parse_str(&self.instrument).ok()?);
        let req = AnalogRequest { window_end_ms: self.window_end_ms, k: self.k, bar: self.bar.parse().ok()? };
        Some(((inst, req), self.result.ok_or(())))
    }
}

/// Replays recorded `analog.fill` records: the fills recorded at a tick answer that tick's keys.
#[derive(Default)]
pub struct RecordedFills(pub BTreeMap<i64, Vec<AnalogFill>>);

impl RecordedFills {
    pub fn from_records(records: &[RecordedMessage]) -> Self {
        let mut m: BTreeMap<i64, Vec<AnalogFill>> = BTreeMap::new();
        for r in records.iter().filter(|r| r.subject == ANALOG_FILL) {
            if let Some(f) = serde_json::from_slice::<FillRecord>(&r.payload).ok().and_then(|f| f.fill()) {
                m.entry(r.published_ms).or_default().push(f);
            }
        }
        Self(m)
    }
}

impl AnalogResolver for RecordedFills {
    fn resolve(&mut self, t_ms: i64, keys: Vec<AnalogKey>) -> Vec<AnalogFill> {
        let recorded = self.0.remove(&t_ms).unwrap_or_default();
        keys.into_iter()
            .map(|k| recorded.iter().find(|(rk, _)| *rk == k).cloned().unwrap_or((k, Err(()))))
            .collect()
    }
}

/// [`RecordedFills`] that answer repeatedly (cut inference replays the same ticks many times).
pub struct CachedFills(pub RecordedFills);

impl AnalogResolver for CachedFills {
    fn resolve(&mut self, t_ms: i64, keys: Vec<AnalogKey>) -> Vec<AnalogFill> {
        let recorded = self.0.0.get(&t_ms).cloned().unwrap_or_default();
        keys.into_iter()
            .map(|k| recorded.iter().find(|(rk, _)| *rk == k).cloned().unwrap_or((k, Err(()))))
            .collect()
    }
}

/// Analogs the service fetches per 500 ms prefetch round.
const PREFETCH_BATCH: usize = 32;
const PREFETCH_MS: i64 = 500;

/// `MD` duplicate window (`crates/fa-bus/streams.json` default of the server: 60 s).
pub const DUPLICATE_WINDOW_MS: i64 = 60_000;

/// Possible re-deliveries of a publisher restarted at `restart_ms` (module docs), per subject
/// under one of `prefixes`: the suffixes of the subject's last batch before the restart (the
/// messages of `before` sharing its last publish time; 1, 2, … trades, as the connector re-reads
/// everything from its persisted cursor on), stamped with the subject's first publication in
/// `inputs` after the restart when that comes within [`DUPLICATE_WINDOW_MS`].
pub fn redelivery_candidates(
    inputs: &[RecordedMessage],
    before: &[RecordedMessage],
    restart_ms: i64,
    prefixes: &[&str],
) -> Vec<Vec<Vec<RecordedMessage>>> {
    let mut by_subject: BTreeMap<&str, Vec<&RecordedMessage>> = BTreeMap::new();
    for r in before.iter().filter(|r| r.published_ms < restart_ms && prefixes.iter().any(|p| r.subject.starts_with(p))) {
        by_subject.entry(r.subject.as_str()).or_default().push(r);
    }
    let mut out = Vec::new();
    for (subject, mut msgs) in by_subject {
        msgs.sort_by_key(|r| (r.published_ms, r.stream_seq));
        let last_pub = msgs.last().map_or(0, |r| r.published_ms);
        // a batch is published within a few ms
        let batch: Vec<&RecordedMessage> = msgs.into_iter().filter(|r| last_pub - r.published_ms <= 50).collect();
        let first = inputs
            .iter()
            .filter(|r| r.subject == subject && r.published_ms >= restart_ms)
            .min_by_key(|r| (r.published_ms, r.stream_seq));
        let Some(f) = first.filter(|f| f.published_ms - last_pub <= DUPLICATE_WINDOW_MS) else { continue };
        // the originals' sequences sort them before the batch they came with
        let options: Vec<Vec<RecordedMessage>> = (1..=batch.len())
            .map(|k| batch[batch.len() - k..].iter().map(|r| RecordedMessage { published_ms: f.published_ms, ..(*r).clone() }).collect())
            .collect();
        out.push(options);
    }
    out
}

/// End of the cold-start transient after a publisher restart (module docs): the second distinct
/// publication time after `restart_ms` of every subject under `prefixes`, the latest of them.
pub fn transient_end(inputs: &[RecordedMessage], restart_ms: i64, prefixes: &[&str]) -> i64 {
    let mut pubs: BTreeMap<&str, BTreeSet<i64>> = BTreeMap::new();
    for r in inputs.iter().filter(|r| r.published_ms >= restart_ms && prefixes.iter().any(|p| r.subject.starts_with(p))) {
        // a batch is published within a few ms: bucket by 1 s
        pubs.entry(r.subject.as_str()).or_default().insert(r.published_ms / 1000);
    }
    pubs.values().filter_map(|v| v.iter().nth(1)).map(|s| s * 1000).max().unwrap_or(i64::MIN)
}

/// Picks per subject the [`redelivery_candidates`] option (or none) under which the replay of
/// the first `probe_ticks` live ticks agrees best with the record (greedy, one subject at a
/// time; inputs are cut at `cut`).
#[allow(clippy::too_many_arguments)]
pub fn infer_redelivery(
    file: &StrategyFile,
    seed: &[RecordedMessage],
    inputs: &[RecordedMessage],
    live: &[RecordedMessage],
    cut: i64,
    candidates: Vec<Vec<Vec<RecordedMessage>>>,
    probe_ticks: usize,
    analogs: &mut dyn AnalogResolver,
) -> Vec<RecordedMessage> {
    let as_of = |r: &RecordedMessage| ExposureDistribution::decode(r.payload.as_slice()).ok().and_then(|d| ms(d.as_of.as_ref()));
    let mut ticks: Vec<i64> = live.iter().filter_map(as_of).collect();
    ticks.sort_unstable();
    ticks.dedup();
    let Some(&last_probe) = ticks.get(probe_ticks.min(ticks.len()).saturating_sub(1)) else { return Vec::new() };
    let probe_live: Vec<RecordedMessage> = live.iter().filter(|r| as_of(r).is_some_and(|t| t <= last_probe)).cloned().collect();
    let base: Vec<RecordedMessage> =
        inputs.iter().filter(|x| x.published_ms >= cut && x.published_ms <= last_probe).cloned().collect();
    let score = |extra: &[RecordedMessage], analogs: &mut dyn AnalogResolver| {
        let mut r = Replayer::new(file, Settings::default());
        for s in seed {
            apply(&mut r.rt, s);
        }
        let mut v = base.clone();
        v.extend(extra.iter().cloned());
        r.run(v, &probe_live, analogs);
        r.cmp.total.identical + r.cmp.total.equal
    };
    let mut kept: Vec<RecordedMessage> = Vec::new();
    let mut best = score(&kept, analogs);
    for options in candidates {
        let mut choice: Option<Vec<RecordedMessage>> = None;
        for o in options {
            let mut trial = kept.clone();
            trial.extend(o);
            let s = score(&trial, analogs);
            if s > best {
                best = s;
                choice = Some(trial);
            }
        }
        if let Some(c) = choice {
            kept = c;
        }
    }
    kept
}

/// The deployed service rebuilt from recorded inputs.
pub struct Replayer {
    pub rt: Runtime,
    known: BTreeSet<String>,
    pub cmp: Comparison,
    /// End of the cold-start transient ([`transient_end`]); `i64::MIN` = none.
    pub transient_until: i64,
    last_tick: Option<i64>,
    /// Strategies the live service skipped for their budget at the current tick.
    over_budget: std::sync::Arc<std::sync::Mutex<BTreeSet<String>>>,
}

impl Replayer {
    /// A cold runtime with `file` and `settings` (the deployed defaults: [`Settings::default`]);
    /// the evaluation time is zero, or over the budget where the live record says so.
    pub fn new(file: &StrategyFile, settings: Settings) -> Self {
        let mut rt = Runtime::with_settings(file, settings).expect("population builds");
        let over_budget: std::sync::Arc<std::sync::Mutex<BTreeSet<String>>> = Default::default();
        let (flag, over) = (over_budget.clone(), settings.step_budget + Duration::from_millis(1));
        rt.set_eval_clock(Box::new(move |id, _| {
            if flag.lock().expect("budget flags").contains(id) { over } else { Duration::ZERO }
        }));
        let known = rt.strategy_ids().into_iter().map(str::to_owned).collect();
        Self { rt, known, cmp: Comparison::default(), transient_until: i64::MIN, last_tick: None, over_budget }
    }

    /// Replays `inputs` (any order; sorted here by knowledge time) with ticks at the `as_of` of
    /// every `live` signal, comparing each tick's output with the live distributions of that
    /// `as_of`. Inputs published after the last live tick stay unapplied and are returned, so a
    /// chunked run carries them into the next chunk.
    pub fn run(
        &mut self,
        mut inputs: Vec<RecordedMessage>,
        live: &[RecordedMessage],
        analogs: &mut dyn AnalogResolver,
    ) -> Vec<RecordedMessage> {
        inputs.sort_by(|a, b| a.published_ms.cmp(&b.published_ms).then(a.stream_seq.cmp(&b.stream_seq)));
        let mut by_tick: BTreeMap<i64, Vec<(ExposureDistribution, Vec<u8>)>> = BTreeMap::new();
        for r in live {
            if let Ok(d) = ExposureDistribution::decode(r.payload.as_slice())
                && let Some(t) = ms(d.as_of.as_ref())
            {
                by_tick.entry(t).or_default().push((d, r.payload.clone()));
            }
        }
        let mut k = 0usize;
        for (t, live_dists) in by_tick {
            while k < inputs.len() && inputs[k].published_ms <= t {
                if apply(&mut self.rt, &inputs[k]) {
                    self.cmp.inputs += 1;
                }
                k += 1;
            }
            self.cmp.ticks += 1;
            // live silence of a known strategy at an emitting tick = a budget skip (module docs)
            *self.over_budget.lock().expect("budget flags") = self
                .known
                .iter()
                .filter(|id| !live_dists.iter().any(|(d, _)| &d.strategy_id == *id))
                .cloned()
                .collect();
            let replayed: BTreeMap<String, ExposureDistribution> =
                self.rt.tick(t).into_iter().map(|d| (d.strategy_id.clone(), d.to_proto())).collect();
            // parallel old-snapshot output goes to `signal.shadow.*`, not compared here
            drop(self.rt.take_shadow_signals());
            // the prefetch rounds between this tick and the next
            let rounds = self.last_tick.map_or(2, |p| ((t - p) / PREFETCH_MS).clamp(1, 8)) as usize;
            self.last_tick = Some(t);
            let keys = self.rt.analogs.take_pending(PREFETCH_BATCH * rounds);
            if !keys.is_empty() {
                for (key, r) in analogs.resolve(t, keys) {
                    self.rt.fill_analog(key, r, t);
                }
            }
            let mut seen: BTreeSet<&str> = BTreeSet::new();
            for (d, raw) in &live_dists {
                if !self.known.contains(&d.strategy_id) {
                    *self.cmp.unknown.entry(d.strategy_id.clone()).or_default() += 1;
                    continue;
                }
                seen.insert(d.strategy_id.as_str());
                let c = self.cmp.by_strategy.entry(d.strategy_id.clone()).or_default();
                let mut one = Counts::default();
                match replayed.get(&d.strategy_id) {
                    None => one.live_only += 1,
                    Some(r) if r.encode_to_vec() == *raw => one.identical += 1,
                    Some(r) => match equal_within(d, r, 1e-9) {
                        Ok(()) => {
                            one.equal += 1;
                            if let Err(e) = equal_within(d, r, 0.0) {
                                self.cmp.max_equal_diff = self.cmp.max_equal_diff.max(e);
                            }
                        }
                        Err(diff) => {
                            one.different += 1;
                            if diff.is_finite() {
                                self.cmp.max_abs_diff = self.cmp.max_abs_diff.max(diff);
                            }
                            if c.different < 3 || c.different.is_multiple_of(200) {
                                let detail: Vec<String> = d
                                    .exposures
                                    .iter()
                                    .zip(&r.exposures)
                                    .filter(|(x, y)| (x.mean - y.mean).abs() > 1e-9 || (x.stddev - y.stddev).abs() > 1e-9)
                                    .map(|(x, y)| {
                                        format!(
                                            "{}: live {:.6}/{:.6} replay {:.6}/{:.6}",
                                            &x.instrument_id[..8],
                                            x.mean,
                                            x.stddev,
                                            y.mean,
                                            y.stddev
                                        )
                                    })
                                    .collect();
                                let n = (d.exposures.len(), r.exposures.len());
                                self.cmp.examples.push(format!(
                                    "{t} {}: max |Δ| {diff:.3e} n={n:?} {}",
                                    d.strategy_id,
                                    detail.join("; ")
                                ));
                            }
                        }
                    },
                }
                c.add(&one);
                self.cmp.total.add(&one);
                if t < self.transient_until { self.cmp.transient.add(&one) } else { self.cmp.steady.add(&one) }
            }
            for id in replayed.keys().filter(|id| !seen.contains(id.as_str())) {
                let one = Counts { replay_only: 1, ..Default::default() };
                self.cmp.by_strategy.entry(id.clone()).or_default().add(&one);
                self.cmp.total.add(&one);
                if t < self.transient_until { self.cmp.transient.add(&one) } else { self.cmp.steady.add(&one) }
            }
        }
        inputs.split_off(k)
    }
}

/// The deployed service subscribes some time after its process starts (config, population load
/// from Garage, connect): inputs published before that were never seen by it. Returns the cut
/// — the publish time from which inputs count — as the candidate in `[first input, first live
/// tick]` whose replay of the first `probe_ticks` live ticks reproduces the most live
/// distributions (the earliest among equals: fewest inputs dropped).
pub fn infer_subscription(
    file: &StrategyFile,
    seed: &[RecordedMessage],
    inputs: &[RecordedMessage],
    live: &[RecordedMessage],
    probe_ticks: usize,
    analogs: &mut dyn AnalogResolver,
) -> i64 {
    let as_of = |r: &RecordedMessage| ExposureDistribution::decode(r.payload.as_slice()).ok().and_then(|d| ms(d.as_of.as_ref()));
    let mut ticks: Vec<i64> = live.iter().filter_map(as_of).collect();
    ticks.sort_unstable();
    ticks.dedup();
    let Some(&first_tick) = ticks.first() else { return i64::MIN };
    let last_probe = ticks[probe_ticks.min(ticks.len()) - 1];
    let probe_live: Vec<RecordedMessage> =
        live.iter().filter(|r| as_of(r).is_some_and(|t| t <= last_probe)).cloned().collect();
    let mut cands: Vec<i64> = inputs.iter().map(|r| r.published_ms).filter(|t| *t <= first_tick).collect();
    cands.sort_unstable();
    cands.dedup();
    let mut best = (f64::NEG_INFINITY, i64::MIN);
    for c in cands {
        let mut r = Replayer::new(file, Settings::default());
        for s in seed {
            apply(&mut r.rt, s);
        }
        r.run(
            inputs.iter().filter(|x| x.published_ms >= c && x.published_ms <= last_probe).cloned().collect(),
            &probe_live,
            analogs,
        );
        let score = r.cmp.total.reproduced();
        if score > best.0 {
            best = (score, c);
        }
    }
    best.1
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/signal_fixture.jsonl")
}

/// Fixture marker: end of the cold-start transient (publish time).
const TRANSIENT: &str = "replay.transient_until";

/// Lineage seed: lifecycle events before the process start (subject `seed.` prefix in fixtures).
const SEED_PREFIX: &str = "seed.";

mod tests {
    use super::*;

    fn load_fixture() -> Vec<RecordedMessage> {
        let text = std::fs::read_to_string(fixture_path()).expect("testdata/signal_fixture.jsonl");
        fa_bus::replay::from_jsonl(&text).expect("fixture parses")
    }

    /// Splits a fixture into lineage seed, inputs and live signals.
    fn split(records: Vec<RecordedMessage>) -> (Vec<RecordedMessage>, Vec<RecordedMessage>, Vec<RecordedMessage>) {
        let (mut seed, mut inputs, mut live) = (Vec::new(), Vec::new(), Vec::new());
        for mut r in records {
            if let Some(rest) = r.subject.strip_prefix(SEED_PREFIX) {
                r.subject = rest.to_owned();
                seed.push(r);
            } else if r.subject.starts_with("signal.") {
                live.push(r);
            } else {
                inputs.push(r);
            }
        }
        (seed, inputs, live)
    }

    #[test]
    fn replay_consumes_exactly_the_subjects_main_subscribes_to() {
        let main = include_str!("main.rs");
        let mut subscribed: BTreeSet<&str> = BTreeSet::new();
        for (needle, subject) in [
            ("subjects::ALL_MD_TRADES", fa_bus::subjects::ALL_MD_TRADES),
            ("subjects::ALL_LATENT_STATES", fa_bus::subjects::ALL_LATENT_STATES),
            ("\"latent.shadow.>\"", "latent.shadow.>"),
            ("\"latent.region.>\"", "latent.region.>"),
            ("\"latent.transition.>\"", "latent.transition.>"),
            ("\"latent.lifecycle.>\"", "latent.lifecycle.>"),
            ("\"decoder.output.>\"", "decoder.output.>"),
        ] {
            if main.contains(needle) {
                subscribed.insert(subject);
            }
        }
        let replayed: BTreeSet<&str> = INPUTS.iter().map(|(_, s)| *s).collect();
        assert_eq!(subscribed, replayed, "main.rs subscriptions and replay inputs must agree");
        assert_eq!(main.matches(".subscribe::<").count(), 7, "a new input subscription in main.rs needs a replay arm");
    }

    #[test]
    fn signal_fixture_reproduces_the_deployed_runtime() {
        let (seed, inputs, live) = split(load_fixture());
        assert!(live.len() >= 500 && inputs.len() >= 500, "fixture too small: {} signals, {} inputs", live.len(), inputs.len());
        let file = StrategyFile::baselines();
        let cut = infer_subscription(&file, &seed, &inputs, &live, 20, &mut CachedFills(RecordedFills::from_records(&inputs)));
        let mut r = Replayer::new(&file, Settings::default());
        r.transient_until = inputs.iter().find(|x| x.subject == TRANSIENT).map_or(i64::MIN, |x| x.published_ms);
        for s in &seed {
            apply(&mut r.rt, s);
        }
        let first = inputs.iter().map(|x| x.published_ms).min().unwrap_or(cut);
        eprintln!("subscription inferred {} ms after the first input", cut - first);
        let mut fills = RecordedFills::from_records(&inputs);
        r.run(inputs.into_iter().filter(|x| x.published_ms >= cut).collect(), &live, &mut fills);
        let c = &r.cmp;
        eprintln!(
            "signal replay: {} ticks, {} inputs, total {:?}, cold-start transient {:?}, steady {:?}, unknown {:?}, max |Δ| {:.3e} (equal ones {:.1e}), examples {:?}",
            c.ticks, c.inputs, c.total, c.transient, c.steady, c.unknown, c.max_abs_diff, c.max_equal_diff, c.examples
        );
        for (id, k) in &c.by_strategy {
            eprintln!("  {id}: {k:?}");
        }
        assert!(c.unknown.is_empty(), "every live strategy is in the replayed population");
        assert!(c.steady.live() >= 300, "steady state too short: {:?}", c.steady);
        assert_eq!(c.steady.reproduced(), 1.0, "steady state reproduced: {:?}", c.steady);
        assert!(c.total.reproduced() >= 0.85, "including the cold-start transient: {:.4}", c.total.reproduced());
        assert!(c.max_equal_diff < 1e-12, "equal = floating-point noise");
    }

    /// The service's prefetch against VictoriaMetrics (blocking inside the multi-thread test
    /// runtime); every fill is also recorded as an `analog.fill` record for fixtures.
    struct VmFills {
        tsdb: fa_tsdb::Tsdb,
        recorded: Vec<RecordedMessage>,
        /// Completed-window answers (they do not depend on the tick).
        cache: BTreeMap<AnalogKey, Result<Option<(f64, f64)>, ()>>,
    }

    impl AnalogResolver for VmFills {
        fn resolve(&mut self, t_ms: i64, keys: Vec<AnalogKey>) -> Vec<AnalogFill> {
            use crate::prefetch::{LOOKBACK_BARS, PriceSource, resolve};
            let handle = tokio::runtime::Handle::current();
            let out: Vec<AnalogFill> = keys
                .into_iter()
                .map(|(inst, req)| {
                    if req.outcome_end_ms() > t_ms {
                        return ((inst, req), Ok(None));
                    }
                    if let Some(r) = self.cache.get(&(inst, req)) {
                        return ((inst, req), *r);
                    }
                    let bar_ms = req.bar.duration().as_millis() as i64;
                    let fut = self.tsdb.prices(inst, req.window_end_ms - LOOKBACK_BARS * bar_ms, req.outcome_end_ms(), req.bar);
                    let r = tokio::task::block_in_place(|| handle.block_on(fut)).map(|pts| resolve(&pts, &req)).map_err(|_| ());
                    self.cache.insert((inst, req), r);
                    ((inst, req), r)
                })
                .collect();
            for f in &out {
                let payload = serde_json::to_vec(&FillRecord::of(f)).expect("json");
                // unique sequence: `sort_records` drops equal (sequence, subject) neighbours
                let stream_seq = self.recorded.len() as u64 + 1;
                self.recorded.push(RecordedMessage { subject: ANALOG_FILL.into(), stream_seq, published_ms: t_ms, payload });
            }
            out
        }
    }

    fn env_ms(name: &str) -> Option<i64> {
        std::env::var(name).ok().and_then(|s| s.trim().parse().ok())
    }

    /// Replays a retained span of the deployed service (`FA_REPLAY_FROM_MS` = its process start,
    /// `FA_REPLAY_TO_MS` default now − 10 s) chunk by chunk (`FA_REPLAY_CHUNK_SECS`, 600) and
    /// prints the comparison as JSON; asserts ≥ `FA_REPLAY_MIN` (0.99) reproduced.
    /// `FA_REPLAY_INGESTOR_START_MS` (the ingestor's start when it restarted with the service)
    /// re-delivers the Kraken trades its connector re-sent; `FA_REPLAY_FIXTURE_SECS=<n>` also
    /// rewrites the fixture with the first `n` seconds. Both process starts:
    /// `date -d "$(docker inspect -f '{{.State.StartedAt}}' financial-auto-<svc>-1)" +%s%3N`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "reads the live JetStream streams; FA_REPLAY_FROM_MS=<deployed process start, ms>"]
    async fn deployed_signals_from_retained_streams() {
        let Some(from) = env_ms("FA_REPLAY_FROM_MS") else { return };
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64;
        let to = env_ms("FA_REPLAY_TO_MS").unwrap_or(now - 10_000);
        let chunk = env_ms("FA_REPLAY_CHUNK_SECS").unwrap_or(600) * 1000;
        let fixture_ms = env_ms("FA_REPLAY_FIXTURE_SECS").map(|s| s * 1000);
        let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
        let js = fa_bus::async_nats::jetstream::new(fa_bus::async_nats::connect(&url).await.expect("nats"));
        let opts = fa_bus::replay::ReadOptions { max_messages: 5_000_000, ..Default::default() };
        let read = |stream: &'static str, filter: &'static str, a: i64, b: i64| {
            let js = js.clone();
            async move { fa_bus::replay::read_range(&js, stream, filter, a, b, &opts).await.expect("read stream") }
        };
        let file = StrategyFile::baselines();
        let vm_url = std::env::var("FA_VM_URL").unwrap_or_else(|_| "http://127.0.0.1:8428".into());
        let mut fills = VmFills { tsdb: fa_tsdb::Tsdb::new(vm_url), recorded: Vec::new(), cache: BTreeMap::new() };
        let mut r = Replayer::new(&file, Settings::default());
        // the service seeds the lineage from the last 7 d of lifecycle events at start
        let seed = read("LATENT", "latent.lifecycle.>", from - 7 * 86_400_000, from - 1).await;
        for s in &seed {
            apply(&mut r.rt, s);
        }
        let mut fixture: Vec<RecordedMessage> = seed
            .iter()
            .cloned()
            .map(|mut s| {
                s.subject = format!("{SEED_PREFIX}{}", s.subject);
                s
            })
            .collect();
        let mut carry: Vec<RecordedMessage> = Vec::new();
        let mut transient_until = i64::MIN;
        let mut a = from;
        let started = std::time::Instant::now();
        while a < to {
            let b = (a + chunk).min(to);
            let mut inputs = std::mem::take(&mut carry);
            for (stream, filter) in INPUTS {
                inputs.extend(read(stream, filter, a, b - 1).await);
            }
            // signals stamped in [a, b) (published a few ms after their as_of)
            let live: Vec<RecordedMessage> = read(SIGNALS.0, SIGNALS.1, a, b + 5_000)
                .await
                .into_iter()
                .filter(|s| {
                    ExposureDistribution::decode(s.payload.as_slice())
                        .ok()
                        .and_then(|d| ms(d.as_of.as_ref()))
                        .is_some_and(|t| t >= a && t < b)
                })
                .collect();
            if a == from {
                // the service subscribes after its start (config, population load): inputs
                // published before that never reached it
                let cut = infer_subscription(&file, &seed, &inputs, &live, 20, &mut fills);
                eprintln!("subscription inferred {} ms after the process start", cut - from);
                inputs.retain(|x| x.published_ms >= cut);
                if let Some(restart) = env_ms("FA_REPLAY_INGESTOR_START_MS")
                    && restart >= a
                    && restart < b
                {
                    let before = read("MD", "md.trade.kraken.>", restart - DUPLICATE_WINDOW_MS, restart - 1).await;
                    let cands = redelivery_candidates(&inputs, &before, restart, &["md.trade.kraken."]);
                    let n: usize = cands.iter().map(Vec::len).sum();
                    let kept = infer_redelivery(&file, &seed, &inputs, &live, cut, cands, 120, &mut fills);
                    eprintln!(
                        "ingestor restart at {restart}: {} re-delivered Kraken trades ({n} options) agree with the live record: {:?}",
                        kept.len(),
                        kept.iter().map(|k| k.subject.as_str()).collect::<Vec<_>>()
                    );
                    inputs.extend(kept);
                    r.transient_until = transient_end(&inputs, restart, &["md.trade.kraken."]);
                    transient_until = r.transient_until;
                    eprintln!("cold-start transient until {} ({} ms after the start)", r.transient_until, r.transient_until - from);
                }
                fills.recorded.clear();
            }
            if let Some(f) = fixture_ms
                && a < from + f
            {
                let end = from + f;
                fixture.extend(inputs.iter().filter(|x| x.published_ms < end).cloned());
                fixture.extend(live.iter().filter(|s| s.published_ms < end).cloned());
            }
            let n_in = inputs.len();
            carry = r.run(inputs, &live, &mut fills);
            eprintln!(
                "chunk {a}..{b}: {n_in} inputs, {} signals; cumulative {:?} ({:.4} reproduced, steady {:.4}) after {:.0} s",
                live.len(),
                r.cmp.total,
                r.cmp.total.reproduced(),
                r.cmp.steady.reproduced(),
                started.elapsed().as_secs_f64()
            );
            a = b;
        }
        if let Some(f) = fixture_ms {
            // the fixture ends at the last live tick before `from + f`: drop later inputs
            let end = from + f;
            let last_tick = fixture
                .iter()
                .filter(|s| s.subject.starts_with("signal."))
                .filter_map(|s| ExposureDistribution::decode(s.payload.as_slice()).ok())
                .filter_map(|d| ms(d.as_of.as_ref()))
                .filter(|t| *t < end)
                .max()
                .unwrap_or(end);
            fixture.extend(fills.recorded.iter().cloned());
            if transient_until > i64::MIN {
                fixture.push(RecordedMessage {
                    subject: TRANSIENT.into(),
                    stream_seq: 0,
                    published_ms: transient_until,
                    payload: Vec::new(),
                });
            }
            fixture.retain(|x| {
                x.subject.starts_with(SEED_PREFIX)
                    || x.subject == TRANSIENT
                    || if x.subject.starts_with("signal.") {
                        ExposureDistribution::decode(x.payload.as_slice())
                            .ok()
                            .and_then(|d| ms(d.as_of.as_ref()))
                            .is_some_and(|t| t <= last_tick)
                    } else {
                        x.published_ms <= last_tick
                    }
            });
            fa_bus::replay::sort_records(&mut fixture);
            std::fs::write(fixture_path(), fa_bus::replay::to_jsonl(&fixture)).expect("write fixture");
            eprintln!("fixture: {} records", fixture.len());
        }
        let c = &r.cmp;
        println!("{}", serde_json::to_string_pretty(c).expect("json"));
        let min: f64 = std::env::var("FA_REPLAY_MIN").ok().and_then(|s| s.parse().ok()).unwrap_or(0.99);
        assert!(c.steady.reproduced() >= min, "steady state reproduced {:.4} < {min}", c.steady.reproduced());
    }
}
