//! Pure strategy stepping logic (no I/O): prices, latent graph and decoder outputs in,
//! exposure distributions out. Population hot reload, pause/resume, the spec 35 §8 failure
//! responses (step budget, NaN outputs, starved primitives) and the analog price cache live here.
//!
//! **Analog outcomes.** `analog_outcome(k)` needs `p(window_end)` and `p(window_end + k·bar)`
//! for every retrieval analog. The in-memory history (`--price-history` trades per instrument)
//! covers recent analogs; older ones are skipped at the step they are first seen
//! (`analog.unresolved`) and queued in [`AnalogCache`]. The I/O side (`main.rs`,
//! `prefetch.rs`) drains the queue between steps, reads the prices from VictoriaMetrics and
//! fills the cache, so the next step resolves them (`analog.resolved_tsdb`). Point-in-time:
//! only analogs whose outcome window ended at or before the step time are counted or queued.
//!
//! **Snapshot changes (spec 35 §6.5).** A strategy bound to an encoder snapshot
//! (`StrategyDef::snapshot`) whose timescale starts serving another snapshot is remapped through
//! the region lineage of that swap ([`fa_strategies::remap`], fed from `latent.lifecycle.>`):
//! when every region literal maps, the remapped programs replace the old ones in place (books
//! kept) and the remap is reported with its time (`ctl.strategy.status` `snapshot.remapped_ms`,
//! the risk-engine pipeline sends an active strategy back to shadow); when a literal has no
//! descendant (or the lineage was never seen) the strategy is suspended (`PauseReason::Snapshot`,
//! code 4: flat, not evaluated) until a later lineage maps it or it is reloaded.
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    time::{Duration, Instant},
};

use fa_core::{InstrumentId, Timescale};
use fa_proto::v1::{DecoderOutput, LatentState, RegionEvent, RegionLifecycleEvent, TransitionUpdate};
use fa_strategies::{
    AssetExposure, ExposureDistribution, GpStrategy, Strategy, StrategyContext,
    decoders::{AnalogRequest, DEFAULT_MAX_AGE_MS, DecoderOutputs, PriceLookup},
    features::{FallbackContext, LatentInfo, LiveContext, PriceSeries, Sources, VirtualBook},
    graph::LatentGraph,
    population::{PopulationError, StrategyDef, StrategyFile},
    remap::{LineageBook, RemapReport, remap_def},
};
use serde_json::{Value, json};
use uuid::Uuid;

/// Runtime tunables (CLI flags / env in `main.rs`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// Trades kept per instrument (`--price-history`).
    pub price_history: usize,
    /// Decoder outputs older than this are ignored.
    pub decoder_max_age_ms: i64,
    /// Per-strategy evaluation budget per step (`--step-budget-ms`, spec 35 §8).
    pub step_budget: Duration,
    /// Consecutive budget breaches before the strategy is auto-paused (`N`).
    pub budget_breach_limit: u32,
    /// Consecutive steps with a NaN / inf output before the strategy is auto-paused (`N`).
    pub nan_breach_limit: u32,
    /// A primitive answering only from the fallback for longer than this marks the strategy starved (`M`).
    pub starved_after_ms: i64,
    /// Analog price cache entries kept.
    pub analog_cache_max: usize,
    /// Analogs waiting to be fetched at most.
    pub analog_pending_max: usize,
    /// A fetch that found no prices is retried after this long.
    pub analog_retry_ms: i64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            price_history: 100_000,
            decoder_max_age_ms: DEFAULT_MAX_AGE_MS,
            step_budget: Duration::from_millis(50),
            budget_breach_limit: 5,
            nan_breach_limit: 5,
            starved_after_ms: 10 * 60_000,
            analog_cache_max: 4_096,
            analog_pending_max: 512,
            analog_retry_ms: 120_000,
        }
    }
}

/// Why a strategy is paused. Heartbeat code in `strategy.<id>.paused_reason`.
#[derive(Debug, Clone, PartialEq)]
pub enum PauseReason {
    /// `ctl.strategy.pause` (code 1).
    Operator,
    /// `N` consecutive steps over the evaluation budget (code 2).
    StepBudget { breaches: u32, last_ms: f64, budget_ms: f64 },
    /// `N` consecutive steps with NaN / inf outputs (code 3).
    NonFinite { breaches: u32 },
    /// The live encoder snapshot changed and a region literal could not be remapped through the
    /// lineage (code 4, spec 35 §8 "Snapshot or region lineage breaks").
    Snapshot { from: String, to: String, why: String },
}

impl PauseReason {
    pub fn code(&self) -> f64 {
        match self {
            Self::Operator => 1.0,
            Self::StepBudget { .. } => 2.0,
            Self::NonFinite { .. } => 3.0,
            Self::Snapshot { .. } => 4.0,
        }
    }

    pub fn is_auto(&self) -> bool {
        !matches!(self, Self::Operator)
    }
}

impl std::fmt::Display for PauseReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Operator => f.write_str("operator: ctl.strategy.pause"),
            Self::StepBudget { breaches, last_ms, budget_ms } => write!(
                f,
                "step_budget: {breaches} consecutive steps over the {budget_ms} ms evaluation budget (last {last_ms:.1} ms)"
            ),
            Self::NonFinite { breaches } => write!(f, "non_finite: {breaches} consecutive steps with NaN/inf output"),
            Self::Snapshot { from, to, why } => {
                write!(f, "snapshot: bound to {from}, live snapshot is {to}; region literals unmappable ({why})")
            }
        }
    }
}

/// Per-strategy gauges for the heartbeat.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MemberStats {
    /// Mean of the last emitted exposure means across instruments.
    pub last_mean: f64,
    /// Mean of the last emitted stddevs across instruments.
    pub last_std: f64,
    pub steps: u64,
    pub sources: Sources,
    /// Steps skipped for exceeding the evaluation budget (total / consecutive).
    pub budget_exceeded: u64,
    pub budget_streak: u32,
    /// Steps with a NaN / inf output (total / consecutive) and instrument exposures zeroed for it.
    pub nan_steps: u64,
    pub nan_streak: u32,
    pub nan_outputs: u64,
    /// Wall time of the last evaluation in microseconds.
    pub last_eval_us: u64,
    /// Analogs skipped for missing prices / priced from the TSDB cache by this strategy.
    pub analog_unresolved: u64,
    pub analog_resolved_tsdb: u64,
}

struct Member {
    def: StrategyDef,
    strategy: GpStrategy,
    next_step_ms: i64,
    books: BTreeMap<InstrumentId, VirtualBook>,
    last_price: BTreeMap<InstrumentId, f64>,
    stats: MemberStats,
    /// Primitive → step time since it has answered only from the fallback.
    fallback_since: BTreeMap<&'static str, i64>,
    /// Primitives that answered from real data at least once.
    real_ever: BTreeSet<&'static str>,
}

impl Member {
    fn new(def: &StrategyDef) -> Result<Self, PopulationError> {
        Ok(Self {
            def: def.clone(),
            strategy: def.build()?,
            next_step_ms: 0,
            books: BTreeMap::new(),
            last_price: BTreeMap::new(),
            stats: MemberStats::default(),
            fallback_since: BTreeMap::new(),
            real_ever: BTreeSet::new(),
        })
    }

    /// Primitives starved for longer than `after_ms` at `now_ms`.
    fn starved(&self, now_ms: i64, after_ms: i64) -> Vec<&'static str> {
        self.fallback_since.iter().filter(|(_, since)| now_ms - **since > after_ms).map(|(p, _)| *p).collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Latent {
    info: LatentInfo,
}

/// One analog outcome to price: instrument + window end + horizon.
pub type AnalogKey = (InstrumentId, AnalogRequest);

#[derive(Debug, Clone, Copy, PartialEq)]
enum AnalogEntry {
    Resolved(f64, f64),
    Missing { at_ms: i64 },
}

/// Bounded cache of analog outcome prices fetched from the TSDB, plus the queue of analogs
/// waiting to be fetched. Keys are `(instrument, window_end, k, bar)`; the two prices of a
/// resolved key are exposed to [`PriceLookup`] at exactly `window_end` and `window_end + k·bar`.
#[derive(Debug, Default)]
pub struct AnalogCache {
    entries: BTreeMap<AnalogKey, AnalogEntry>,
    order: VecDeque<AnalogKey>,
    /// Exact-time prices per instrument with a reference count (keys may share a time).
    points: BTreeMap<InstrumentId, BTreeMap<i64, (f64, u32)>>,
    pending: BTreeSet<AnalogKey>,
    inflight: BTreeSet<AnalogKey>,
    pub fetches: u64,
    pub fetch_errors: u64,
    pub resolved_fetches: u64,
    pub missing_fetches: u64,
}

impl AnalogCache {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn pending(&self) -> usize {
        self.pending.len() + self.inflight.len()
    }

    pub fn is_resolved(&self, key: &AnalogKey) -> bool {
        matches!(self.entries.get(key), Some(AnalogEntry::Resolved(..)))
    }

    /// Queues `key` unless it is cached, in flight, recently missing, not realised at
    /// `now_ms` (point-in-time) or the queue is full.
    fn want(&mut self, key: AnalogKey, now_ms: i64, s: &Settings) {
        if key.1.outcome_end_ms() > now_ms || self.inflight.contains(&key) || self.pending.len() >= s.analog_pending_max
        {
            return;
        }
        match self.entries.get(&key) {
            Some(AnalogEntry::Resolved(..)) => {}
            Some(AnalogEntry::Missing { at_ms }) if now_ms - at_ms < s.analog_retry_ms => {}
            _ => {
                self.pending.insert(key);
            }
        }
    }

    /// Takes up to `n` queued analogs and marks them in flight.
    pub fn take_pending(&mut self, n: usize) -> Vec<AnalogKey> {
        let keys: Vec<AnalogKey> = self.pending.iter().take(n).copied().collect();
        for k in &keys {
            self.pending.remove(k);
            self.inflight.insert(*k);
        }
        keys
    }

    /// Records the outcome of a fetch: `Some((p0, p1))` resolved, `None` nothing found, and
    /// `Err` a failed fetch (the key is queued again on the next step that needs it).
    pub fn fill(&mut self, key: AnalogKey, result: Result<Option<(f64, f64)>, ()>, now_ms: i64, max: usize) {
        self.inflight.remove(&key);
        self.fetches += 1;
        let entry = match result {
            Err(()) => {
                self.fetch_errors += 1;
                return;
            }
            Ok(Some((p0, p1))) if p0 > 0.0 && p1 > 0.0 && p0.is_finite() && p1.is_finite() => {
                self.resolved_fetches += 1;
                AnalogEntry::Resolved(p0, p1)
            }
            Ok(_) => {
                self.missing_fetches += 1;
                AnalogEntry::Missing { at_ms: now_ms }
            }
        };
        self.remove(&key);
        if let AnalogEntry::Resolved(p0, p1) = entry {
            let pts = self.points.entry(key.0).or_default();
            for (t, p) in [(key.1.window_end_ms, p0), (key.1.outcome_end_ms(), p1)] {
                let e = pts.entry(t).or_insert((p, 0));
                e.0 = p;
                e.1 += 1;
            }
        }
        self.entries.insert(key, entry);
        self.order.push_back(key);
        while self.entries.len() > max.max(1) {
            let Some(old) = self.order.pop_front() else { break };
            self.remove(&old);
        }
    }

    fn remove(&mut self, key: &AnalogKey) {
        let Some(e) = self.entries.remove(key) else { return };
        self.order.retain(|k| k != key);
        if matches!(e, AnalogEntry::Resolved(..))
            && let Some(pts) = self.points.get_mut(&key.0)
        {
            for t in [key.1.window_end_ms, key.1.outcome_end_ms()] {
                if let Some(x) = pts.get_mut(&t) {
                    x.1 -= 1;
                    if x.1 == 0 {
                        pts.remove(&t);
                    }
                }
            }
            if pts.is_empty() {
                self.points.remove(&key.0);
            }
        }
    }
}

/// In-memory trade history first, TSDB-fetched analog prices (exact times) second.
struct LayeredPrices<'a> {
    series: &'a PriceSeries,
    fetched: Option<&'a BTreeMap<i64, (f64, u32)>>,
}

impl PriceLookup for LayeredPrices<'_> {
    fn price_at(&self, at_ms: i64) -> Option<f64> {
        self.series.price_at(at_ms).or_else(|| self.fetched?.get(&at_ms).map(|p| p.0))
    }
}

/// Measures one strategy's evaluation; tests replace it with a fixed cost per strategy id.
pub type EvalClock = Box<dyn Fn(&str, Duration) -> Duration + Send>;

/// Deterministic strategy population runtime.
pub struct Runtime {
    members: Vec<Member>,
    /// Instruments restricted by the strategy file; empty = all seen.
    allowed: BTreeSet<InstrumentId>,
    prices: BTreeMap<InstrumentId, PriceSeries>,
    /// Legacy expert-0 summary per (timescale, instrument), the fallback's `LatentInfo`.
    latent: BTreeMap<(Timescale, InstrumentId), Latent>,
    /// Last closed bar bucket fed to the graph per (timescale, instrument).
    last_bucket: BTreeMap<(Timescale, InstrumentId), i64>,
    pub graph: LatentGraph,
    pub decoders: DecoderOutputs,
    pub analogs: AnalogCache,
    paused: BTreeMap<String, PauseReason>,
    pub settings: Settings,
    /// Label of the loaded population (`baselines`, a path or an artifact key).
    pub population: String,
    pub reloads: u64,
    pub signals_emitted: u64,
    pub trades_seen: u64,
    pub latent_seen: u64,
    /// Analog evaluations skipped for missing prices / priced from the TSDB cache / from memory.
    pub analog_unresolved: u64,
    pub analog_resolved_tsdb: u64,
    pub analog_resolved_memory: u64,
    /// Strategies auto-paused so far (budget + NaN).
    pub auto_pauses: u64,
    /// Snapshot-swap lineage from `latent.lifecycle.>` (spec 35 §6.5).
    pub lineage: LineageBook,
    /// Latest snapshot remap per strategy id (kept across reloads so a reload of the old genome
    /// re-applies the same remap without reporting a new one).
    remaps: BTreeMap<String, SnapshotRemap>,
    clock: Option<EvalClock>,
}

/// One snapshot remap attempt of a strategy.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotRemap {
    pub from: String,
    pub to: String,
    pub ok: bool,
    /// When this outcome was first reached.
    pub at_ms: i64,
    pub report: Option<RemapReport>,
    /// Lineage events seen when it was attempted (an unmappable one is retried on new lineage).
    lineage_events: u64,
}

impl Runtime {
    #[cfg(test)]
    pub fn new(file: &StrategyFile) -> Result<Self, PopulationError> {
        Self::with_settings(file, Settings::default())
    }

    pub fn with_settings(file: &StrategyFile, settings: Settings) -> Result<Self, PopulationError> {
        let members = file.strategy.iter().map(Member::new).collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            members,
            allowed: Self::allowed_of(file),
            prices: BTreeMap::new(),
            latent: BTreeMap::new(),
            last_bucket: BTreeMap::new(),
            graph: LatentGraph::new(),
            decoders: DecoderOutputs::new(),
            analogs: AnalogCache::default(),
            paused: BTreeMap::new(),
            settings,
            population: "baselines".into(),
            reloads: 0,
            signals_emitted: 0,
            trades_seen: 0,
            latent_seen: 0,
            analog_unresolved: 0,
            analog_resolved_tsdb: 0,
            analog_resolved_memory: 0,
            auto_pauses: 0,
            lineage: LineageBook::new(),
            remaps: BTreeMap::new(),
            clock: None,
        })
    }

    #[cfg(test)]
    /// Replaces the wall-clock measurement of a strategy's evaluation (tests).
    pub fn set_eval_clock(&mut self, clock: EvalClock) {
        self.clock = Some(clock);
    }

    fn allowed_of(file: &StrategyFile) -> BTreeSet<InstrumentId> {
        file.instruments.iter().filter_map(|s| Uuid::parse_str(s).ok()).map(InstrumentId).collect()
    }

    pub fn strategy_ids(&self) -> Vec<&str> {
        self.members.iter().map(|m| m.def.id.as_str()).collect()
    }

    pub fn instruments(&self) -> usize {
        self.prices.len()
    }

    #[cfg(test)]
    pub fn paused(&self) -> impl Iterator<Item = &str> {
        self.paused.keys().map(String::as_str)
    }

    pub fn is_paused(&self, id: &str) -> bool {
        self.paused.contains_key(id)
    }

    #[cfg(test)]
    pub fn pause_reason(&self, id: &str) -> Option<&PauseReason> {
        self.paused.get(id)
    }

    #[cfg(test)]
    /// Per-strategy gauges (id, stats).
    pub fn member_stats(&self) -> Vec<(&str, &MemberStats)> {
        self.members.iter().map(|m| (m.def.id.as_str(), &m.stats)).collect()
    }

    /// Starved primitives per strategy at `now_ms` (empty list = not starved).
    pub fn starved(&self, now_ms: i64) -> Vec<(&str, Vec<&'static str>)> {
        self.members.iter().map(|m| (m.def.id.as_str(), m.starved(now_ms, self.settings.starved_after_ms))).collect()
    }

    /// Primitive source counters summed over the population.
    pub fn sources(&self) -> Sources {
        let mut s = Sources::default();
        for m in &self.members {
            s.add(m.stats.sources);
        }
        s
    }

    /// Replaces the population atomically (hot reload). Every program is validated first; on
    /// any error the running population is untouched. Strategies whose id survives keep their
    /// virtual books, cadence, stats and pause state.
    pub fn replace_population(&mut self, file: &StrategyFile, label: &str) -> Result<Vec<String>, PopulationError> {
        let mut fresh = file.strategy.iter().map(Member::new).collect::<Result<Vec<_>, _>>()?;
        let mut old: BTreeMap<String, Member> =
            std::mem::take(&mut self.members).into_iter().map(|m| (m.def.id.clone(), m)).collect();
        for m in &mut fresh {
            if let Some(prev) = old.remove(&m.def.id) {
                m.books = prev.books;
                m.last_price = prev.last_price;
                m.next_step_ms = prev.next_step_ms;
                m.stats = prev.stats;
                if prev.def.signal == m.def.signal && prev.def.uncertainty == m.def.uncertainty {
                    m.fallback_since = prev.fallback_since;
                    m.real_ever = prev.real_ever;
                }
            }
        }
        let ids: Vec<String> = fresh.iter().map(|m| m.def.id.clone()).collect();
        self.paused.retain(|p, _| ids.contains(p));
        self.members = fresh;
        self.allowed = Self::allowed_of(file);
        self.population = label.to_owned();
        self.reloads += 1;
        Ok(ids)
    }

    /// Pauses a strategy (operator): it keeps stepping but emits `mean = 0, stddev = 1` so
    /// downstream flattens.
    pub fn pause(&mut self, id: &str) -> bool {
        if self.members.iter().any(|m| m.def.id == id) {
            self.paused.insert(id.to_owned(), PauseReason::Operator);
            true
        } else {
            false
        }
    }

    /// Resumes a strategy, clearing an automatic pause and its breach streaks.
    pub fn resume(&mut self, id: &str) -> bool {
        let Some(m) = self.members.iter_mut().find(|m| m.def.id == id) else { return false };
        m.stats.budget_streak = 0;
        m.stats.nan_streak = 0;
        self.paused.remove(id);
        true
    }

    fn tracks(&self, inst: InstrumentId) -> bool {
        self.allowed.is_empty() || self.allowed.contains(&inst)
    }

    pub fn on_trade(&mut self, inst: InstrumentId, at_ms: i64, price: f64) {
        if !self.tracks(inst) {
            return;
        }
        self.trades_seen += 1;
        let cap = self.settings.price_history;
        self.prices.entry(inst).or_insert_with(|| PriceSeries::new(cap)).push(at_ms, price);
    }
    pub fn on_latent(&mut self, state: &LatentState) {
        let (Ok(ts), Ok(uuid)) = (state.timescale.parse::<Timescale>(), Uuid::parse_str(&state.instrument_id)) else {
            return;
        };
        let inst = InstrumentId(uuid);
        if !self.tracks(inst) {
            return;
        }
        self.graph.on_state(state);
        let Some(e) = state.experts.first() else {
            return;
        };
        self.latent_seen += 1;
        let uncertainty = if e.log_var.is_empty() {
            0.0
        } else {
            e.log_var.iter().map(|lv| (f64::from(*lv) / 2.0).exp()).sum::<f64>() / e.log_var.len() as f64
        };
        let dwell = match self.latent.get(&(ts, inst)) {
            Some(prev) if prev.info.region_id == e.region_id => prev.info.dwell.saturating_add(1),
            _ => 0,
        };
        self.latent
            .insert((ts, inst), Latent { info: LatentInfo { region_id: e.region_id.clone(), uncertainty, dwell } });
    }

    pub fn on_region(&mut self, ev: &RegionEvent, now_ms: i64) {
        self.graph.on_region(ev, now_ms);
    }

    /// `latent.lifecycle.<ts>`: snapshot-alignment lineage for remapping (spec 35 §6.5).
    pub fn on_lifecycle(&mut self, ev: &RegionLifecycleEvent) -> bool {
        self.lineage.on_lifecycle(ev)
    }

    /// Remaps / suspends strategies bound to a snapshot their timescale no longer serves.
    /// Returns the ids whose programs were replaced.
    pub fn check_snapshots(&mut self, now_ms: i64) -> Vec<String> {
        let mut replaced = Vec::new();
        for m in &mut self.members {
            let Some(bound) = m.def.snapshot.clone() else { continue };
            let Some(live) = self.graph.snapshot(m.def.timescale).map(str::to_owned) else { continue };
            if live == bound {
                // a snapshot pause from an earlier mismatch no longer applies
                if matches!(self.paused.get(&m.def.id), Some(PauseReason::Snapshot { .. })) {
                    self.paused.remove(&m.def.id);
                }
                continue;
            }
            let prev = self.remaps.get(&m.def.id);
            if prev.is_some_and(|p| {
                !p.ok && p.from == bound && p.to == live && p.lineage_events == self.lineage.events
            }) {
                continue; // nothing new to try
            }
            let (def, report) = remap_def(&m.def, &live, &self.lineage);
            let ok = def.is_some();
            let at_ms = match prev {
                Some(p) if p.from == bound && p.to == live && p.ok == ok => p.at_ms,
                _ => now_ms,
            };
            match def.map(|d| d.build().map(|s| (d, s))) {
                Some(Ok((d, strategy))) => {
                    tracing::warn!(strategy = %m.def.id, from = %bound, to = %live, signal = %d.signal, "strategy remapped onto the new snapshot");
                    m.def = d;
                    m.strategy = strategy;
                    m.fallback_since.clear();
                    if matches!(self.paused.get(&m.def.id), Some(PauseReason::Snapshot { .. })) {
                        self.paused.remove(&m.def.id);
                    }
                    replaced.push(m.def.id.clone());
                }
                _ => {
                    let why = report
                        .as_ref()
                        .map(|r| {
                            r.literals
                                .iter()
                                .filter(|l| l.to.is_none())
                                .map(|l| format!("{}: {}", l.from_region, l.kind))
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_else(|| "programs did not parse".into());
                    if !matches!(self.paused.get(&m.def.id), Some(PauseReason::Snapshot { .. })) {
                        tracing::warn!(strategy = %m.def.id, from = %bound, to = %live, %why, "strategy suspended: region lineage breaks");
                        self.auto_pauses += 1;
                    }
                    self.paused.insert(m.def.id.clone(), PauseReason::Snapshot { from: bound.clone(), to: live.clone(), why });
                }
            }
            self.remaps.insert(
                m.def.id.clone(),
                SnapshotRemap { from: bound, to: live, ok, at_ms, report, lineage_events: self.lineage.events },
            );
        }
        replaced
    }

    #[cfg(test)]
    pub fn remap_of(&self, id: &str) -> Option<&SnapshotRemap> {
        self.remaps.get(id)
    }

    pub fn on_transition(&mut self, tu: &TransitionUpdate) {
        self.graph.on_transition(tu);
    }

    pub fn on_decoder_output(&mut self, out: &DecoderOutput, now_ms: i64) {
        if let Some(inst) = DecoderOutputs::instrument_of(out)
            && !self.tracks(inst)
        {
            return;
        }
        self.decoders.on_output(out, now_ms);
    }

    /// Feeds closed bars to the graph's region statistics: for every timescale in the population
    /// and every instrument, the bar before the current bucket closes once.
    fn feed_closes(&mut self, now_ms: i64) {
        let timescales: BTreeSet<Timescale> = self.members.iter().map(|m| m.def.timescale).collect();
        let instruments: Vec<InstrumentId> = self.prices.keys().copied().collect();
        for ts in timescales {
            let bar_ms = ts.duration().as_millis() as i64;
            let bucket = now_ms.div_euclid(bar_ms);
            for inst in &instruments {
                let key = (ts, *inst);
                let last = self.last_bucket.get(&key).copied();
                if last.is_some_and(|b| b >= bucket) {
                    continue;
                }
                let close_at = bucket * bar_ms;
                let Some(price) = self.prices.get(inst).and_then(|s| s.price_at(close_at - 1)) else {
                    continue;
                };
                self.last_bucket.insert(key, bucket);
                self.graph.record_close(ts, *inst, close_at, price);
            }
        }
    }

    /// Steps every strategy whose cadence is due at `now_ms`; returns the distributions to publish.
    ///
    /// Spec 35 §8 per strategy: an evaluation slower than `step_budget` is discarded (nothing
    /// published, books unchanged) and counted; `budget_breach_limit` consecutive breaches
    /// auto-pause it. An instrument whose evaluation produced NaN / inf gets `mean = 0,
    /// stddev = 1`; `nan_breach_limit` consecutive such steps auto-pause it. Auto-paused
    /// strategies are not evaluated and emit `mean = 0, stddev = 1` until `resume`.
    pub fn tick(&mut self, now_ms: i64) -> Vec<ExposureDistribution> {
        let mut out = Vec::new();
        let instruments: Vec<InstrumentId> = self.prices.keys().copied().collect();
        if instruments.is_empty() {
            return out;
        }
        self.feed_closes(now_ms);
        self.check_snapshots(now_ms);
        let settings = self.settings;
        let (graph, decoders, prices, latent, analogs) =
            (&self.graph, &self.decoders, &self.prices, &self.latent, &self.analogs);
        let mut wanted: Vec<AnalogKey> = Vec::new();
        let (mut unresolved, mut resolved_tsdb, mut resolved_memory) = (0u64, 0u64, 0u64);
        let mut new_pauses: Vec<(String, PauseReason)> = Vec::new();
        for m in &mut self.members {
            if now_ms < m.next_step_ms {
                continue;
            }
            let step_ms = (m.def.step_secs() * 1000) as i64;
            m.next_step_ms = now_ms.div_euclid(step_ms) * step_ms + step_ms;
            let pause = self.paused.get(&m.def.id);

            // auto-paused: not evaluated, flat with full uncertainty
            if pause.is_some_and(PauseReason::is_auto) {
                let exposures: Vec<AssetExposure> =
                    instruments.iter().map(|i| AssetExposure { instrument: *i, mean: 0.0, stddev: 1.0 }).collect();
                for e in &exposures {
                    m.books.insert(e.instrument, VirtualBook::default());
                }
                m.stats.last_mean = 0.0;
                m.stats.last_std = 1.0;
                self.signals_emitted += 1;
                out.push(ExposureDistribution {
                    strategy_id: m.def.id.clone(),
                    genome_id: m.def.genome_id.clone(),
                    horizon: m.def.horizon(),
                    as_of_ms: now_ms,
                    exposures,
                });
                continue;
            }

            let ts = m.def.timescale;
            let view = graph.at(ts, now_ms);
            let lookups: Vec<(InstrumentId, LayeredPrices<'_>)> = instruments
                .iter()
                .filter_map(|i| Some((*i, LayeredPrices { series: prices.get(i)?, fetched: analogs.points.get(i) })))
                .collect();
            let mut new_prices: Vec<(InstrumentId, f64)> = Vec::new();
            let contexts: Vec<(InstrumentId, LiveContext<'_>)> = lookups
                .iter()
                .filter_map(|(inst, lookup)| {
                    let closes = lookup.series.closes(ts, m.def.lookback_bars + 1);
                    let price = *closes.last()?;
                    let mut book = m.books.get(inst).copied().unwrap_or_default();
                    if let Some(prev) = m.last_price.get(inst)
                        && *prev > 0.0
                    {
                        book.unrealised += book.exposure * (price / prev).ln();
                    }
                    new_prices.push((*inst, price));
                    let info = latent.get(&(ts, *inst)).map(|l| l.info.clone());
                    let ctx = LiveContext::new(FallbackContext::new(&closes, info, book), *inst, m.def.steps)
                        .with_graph(&view)
                        .with_snapshot(m.def.snapshot.as_deref())
                        .with_bar(ts)
                        .with_decoders(decoders.at(*inst, now_ms, settings.decoder_max_age_ms))
                        .with_prices(lookup as &dyn PriceLookup);
                    Some((*inst, ctx))
                })
                .collect();

            // evaluate instrument by instrument so a NaN only zeroes its own exposure
            let started = Instant::now();
            let mut exposures = Vec::with_capacity(contexts.len());
            let mut nan_insts = 0u64;
            fa_strategies::take_non_finite();
            for (inst, ctx) in &contexts {
                let d = m.strategy.step(now_ms, &[(*inst, ctx as &dyn StrategyContext)]);
                let mut e = d.exposures[0];
                let nan = fa_strategies::take_non_finite() > 0 || !e.mean.is_finite() || !e.stddev.is_finite();
                if nan {
                    nan_insts += 1;
                    e.mean = 0.0;
                    e.stddev = 1.0;
                }
                exposures.push(e);
            }
            let wall = started.elapsed();
            let elapsed = self.clock.as_ref().map_or(wall, |c| c(&m.def.id, wall));
            m.stats.last_eval_us = elapsed.as_micros() as u64;
            m.stats.steps += 1;

            // sources and starvation are tracked whether or not the result is used
            let mut real_now: BTreeSet<&'static str> = BTreeSet::new();
            let mut fb_now: BTreeSet<&'static str> = BTreeSet::new();
            for (inst, c) in &contexts {
                m.stats.sources.add(c.sources());
                let ps = c.primitive_sources();
                real_now.extend(ps.real);
                fb_now.extend(ps.fallback);
                let (_, skipped) = c.analog_counts();
                unresolved += skipped;
                m.stats.analog_unresolved += skipped;
                for req in c.take_analog_requests() {
                    wanted.push((*inst, req));
                }
                for req in c.take_analog_resolved() {
                    if analogs.is_resolved(&(*inst, req)) {
                        resolved_tsdb += 1;
                        m.stats.analog_resolved_tsdb += 1;
                    } else {
                        resolved_memory += 1;
                    }
                }
            }
            for p in &real_now {
                m.fallback_since.remove(p);
            }
            m.real_ever.extend(real_now.iter().copied());
            for p in fb_now.difference(&real_now) {
                m.fallback_since.entry(p).or_insert(now_ms);
            }

            // spec 35 §8: over budget → skipped this step (exposure unchanged)
            if elapsed > settings.step_budget {
                m.stats.budget_exceeded += 1;
                m.stats.budget_streak += 1;
                if m.stats.budget_streak >= settings.budget_breach_limit.max(1) {
                    new_pauses.push((
                        m.def.id.clone(),
                        PauseReason::StepBudget {
                            breaches: m.stats.budget_streak,
                            last_ms: elapsed.as_secs_f64() * 1000.0,
                            budget_ms: settings.step_budget.as_secs_f64() * 1000.0,
                        },
                    ));
                }
                continue;
            }
            m.stats.budget_streak = 0;
            for (inst, p) in new_prices {
                m.last_price.insert(inst, p);
            }
            if nan_insts > 0 {
                m.stats.nan_steps += 1;
                m.stats.nan_outputs += nan_insts;
                m.stats.nan_streak += 1;
                if m.stats.nan_streak >= settings.nan_breach_limit.max(1) {
                    new_pauses.push((m.def.id.clone(), PauseReason::NonFinite { breaches: m.stats.nan_streak }));
                }
            } else {
                m.stats.nan_streak = 0;
            }

            let is_paused = pause.is_some();
            for e in &mut exposures {
                sanitize(e);
                if is_paused {
                    e.mean = 0.0;
                    e.stddev = 1.0;
                }
                let (_, ctx) = contexts.iter().find(|(i, _)| *i == e.instrument).expect("context exists");
                let prev = m.books.get(&e.instrument).copied().unwrap_or_default();
                let flipped = (e.mean == 0.0) != (prev.exposure == 0.0) || e.mean.signum() != prev.exposure.signum();
                let book = if flipped {
                    VirtualBook { exposure: e.mean, unrealised: 0.0, steps_in_trade: 0 }
                } else {
                    VirtualBook {
                        exposure: e.mean,
                        unrealised: ctx.unrealised_pnl(),
                        steps_in_trade: prev.steps_in_trade.saturating_add(1),
                    }
                };
                m.books.insert(e.instrument, book);
            }
            let n = exposures.len().max(1) as f64;
            m.stats.last_mean = exposures.iter().map(|e| e.mean).sum::<f64>() / n;
            m.stats.last_std = exposures.iter().map(|e| e.stddev).sum::<f64>() / n;
            self.signals_emitted += 1;
            out.push(ExposureDistribution {
                strategy_id: m.def.id.clone(),
                genome_id: m.def.genome_id.clone(),
                horizon: m.def.horizon(),
                as_of_ms: now_ms,
                exposures,
            });
        }
        self.analog_unresolved += unresolved;
        self.analog_resolved_tsdb += resolved_tsdb;
        self.analog_resolved_memory += resolved_memory;
        for key in wanted {
            self.analogs.want(key, now_ms, &settings);
        }
        for (id, reason) in new_pauses {
            tracing::warn!(strategy = %id, reason = %reason, "strategy auto-paused (spec 35 §8)");
            self.paused.insert(id, reason);
            self.auto_pauses += 1;
        }
        out
    }

    /// Records fetched analog prices (see [`AnalogCache::fill`]).
    pub fn fill_analog(&mut self, key: AnalogKey, result: Result<Option<(f64, f64)>, ()>, now_ms: i64) {
        let max = self.settings.analog_cache_max;
        self.analogs.fill(key, result, now_ms, max);
    }

    /// Heartbeat counters (numbers only; reason texts are in [`Self::status`]).
    pub fn heartbeat_counters(&self, now_ms: i64) -> BTreeMap<String, f64> {
        let src = self.sources();
        let starved = self.starved(now_ms);
        let mut c = BTreeMap::from([
            ("strategies".to_owned(), self.members.len() as f64),
            ("paused".to_owned(), self.paused.len() as f64),
            ("auto_paused".to_owned(), self.paused.values().filter(|r| r.is_auto()).count() as f64),
            ("auto_pauses".to_owned(), self.auto_pauses as f64),
            ("lineage.events".to_owned(), self.lineage.events as f64),
            ("lineage.snapshots".to_owned(), self.lineage.snapshots() as f64),
            ("snapshot.remapped".to_owned(), self.remaps.values().filter(|r| r.ok).count() as f64),
            (
                "snapshot.suspended".to_owned(),
                self.paused.values().filter(|r| matches!(r, PauseReason::Snapshot { .. })).count() as f64,
            ),
            ("starved".to_owned(), starved.iter().filter(|(_, p)| !p.is_empty()).count() as f64),
            ("reloads".to_owned(), self.reloads as f64),
            ("instruments".to_owned(), self.instruments() as f64),
            ("trades_seen".to_owned(), self.trades_seen as f64),
            ("latent_seen".to_owned(), self.latent_seen as f64),
            ("regions_seen".to_owned(), self.graph.seen.regions as f64),
            ("transitions_seen".to_owned(), self.graph.seen.transitions as f64),
            ("closes_seen".to_owned(), self.graph.seen.closes as f64),
            ("regions".to_owned(), self.graph.region_count() as f64),
            ("transition_rows".to_owned(), self.graph.row_count() as f64),
            ("decoder_outputs_seen".to_owned(), self.decoders.seen as f64),
            ("decoder_outputs".to_owned(), self.decoders.len() as f64),
            ("primitives.graph".to_owned(), src.graph as f64),
            ("primitives.decoder".to_owned(), src.decoder as f64),
            ("primitives.fallback".to_owned(), src.fallback as f64),
            ("signals_emitted".to_owned(), self.signals_emitted as f64),
            ("analog.unresolved".to_owned(), self.analog_unresolved as f64),
            ("analog.resolved_tsdb".to_owned(), self.analog_resolved_tsdb as f64),
            ("analog.resolved_memory".to_owned(), self.analog_resolved_memory as f64),
            ("analog.cache".to_owned(), self.analogs.len() as f64),
            ("analog.pending".to_owned(), self.analogs.pending() as f64),
            ("analog.fetches".to_owned(), self.analogs.fetches as f64),
            ("analog.fetch_resolved".to_owned(), self.analogs.resolved_fetches as f64),
            ("analog.fetch_missing".to_owned(), self.analogs.missing_fetches as f64),
            ("analog.fetch_errors".to_owned(), self.analogs.fetch_errors as f64),
            ("settings.step_budget_ms".to_owned(), self.settings.step_budget.as_secs_f64() * 1000.0),
            ("settings.breach_limit".to_owned(), f64::from(self.settings.budget_breach_limit)),
            ("settings.nan_limit".to_owned(), f64::from(self.settings.nan_breach_limit)),
            ("settings.starved_after_min".to_owned(), self.settings.starved_after_ms as f64 / 60_000.0),
            ("settings.price_history".to_owned(), self.settings.price_history as f64),
        ]);
        let mut budget_total = 0u64;
        let mut nan_total = 0u64;
        for (m, (_, starved)) in self.members.iter().zip(&starved) {
            let (id, st) = (&m.def.id, &m.stats);
            budget_total += st.budget_exceeded;
            nan_total += st.nan_outputs;
            c.insert(format!("budget_exceeded.{id}"), st.budget_exceeded as f64);
            c.insert(format!("strategy.{id}.mean"), st.last_mean);
            c.insert(format!("strategy.{id}.std"), st.last_std);
            c.insert(format!("strategy.{id}.steps"), st.steps as f64);
            c.insert(format!("strategy.{id}.paused"), f64::from(u8::from(self.is_paused(id))));
            c.insert(format!("strategy.{id}.paused_reason"), self.paused.get(id).map_or(0.0, PauseReason::code));
            c.insert(format!("strategy.{id}.starved"), f64::from(u8::from(!starved.is_empty())));
            c.insert(format!("strategy.{id}.budget_exceeded"), st.budget_exceeded as f64);
            c.insert(format!("strategy.{id}.nan_outputs"), st.nan_outputs as f64);
            c.insert(format!("strategy.{id}.eval_us"), st.last_eval_us as f64);
            c.insert(format!("strategy.{id}.primitives.graph"), st.sources.graph as f64);
            c.insert(format!("strategy.{id}.primitives.decoder"), st.sources.decoder as f64);
            c.insert(format!("strategy.{id}.primitives.fallback"), st.sources.fallback as f64);
            c.insert(format!("strategy.{id}.analog.unresolved"), st.analog_unresolved as f64);
            c.insert(format!("strategy.{id}.analog.resolved_tsdb"), st.analog_resolved_tsdb as f64);
        }
        c.insert("budget_exceeded".into(), budget_total as f64);
        c.insert("nan_outputs".into(), nan_total as f64);
        c
    }

    /// `ctl.strategy.status` payload: population, settings, per-strategy stats and primitive sources.
    pub fn status(&self, now_ms: i64) -> Value {
        let s = &self.settings;
        let strategies: Vec<Value> = self
            .members
            .iter()
            .map(|m| {
                let st = &m.stats;
                let starved = m.starved(now_ms, s.starved_after_ms);
                let fallback_only: BTreeMap<&str, i64> =
                    m.fallback_since.iter().map(|(p, since)| (*p, (now_ms - since).max(0) / 1000)).collect();
                json!({
                    "id": m.def.id,
                    "genome_id": m.def.genome_id,
                    "baseline": m.def.is_baseline(),
                    "timescale": m.def.timescale.as_str(),
                    "steps": m.def.steps,
                    "step_secs": m.def.step_secs(),
                    "lookback_bars": m.def.lookback_bars,
                    "signal": m.def.signal,
                    "uncertainty": m.def.uncertainty,
                    "paused": self.is_paused(&m.def.id),
                    "paused_reason": self.paused.get(&m.def.id).map(ToString::to_string),
                    "paused_reason_code": self.paused.get(&m.def.id).map_or(0.0, PauseReason::code),
                    "snapshot": {
                        "bound": m.def.snapshot,
                        "live": self.graph.snapshot(m.def.timescale),
                        "remap": self.remaps.get(&m.def.id).map(|r| json!({
                            "from": r.from, "to": r.to, "ok": r.ok, "at_ms": r.at_ms, "report": r.report,
                        })),
                        "remapped_ms": self.remaps.get(&m.def.id).filter(|r| r.ok).map(|r| r.at_ms),
                    },
                    "starved": !starved.is_empty(),
                    "starved_primitives": starved,
                    "stats": {
                        "steps": st.steps,
                        "last_mean": st.last_mean,
                        "last_std": st.last_std,
                        "last_eval_us": st.last_eval_us,
                        "budget_exceeded": st.budget_exceeded,
                        "budget_streak": st.budget_streak,
                        "nan_steps": st.nan_steps,
                        "nan_outputs": st.nan_outputs,
                        "nan_streak": st.nan_streak,
                        "analog_unresolved": st.analog_unresolved,
                        "analog_resolved_tsdb": st.analog_resolved_tsdb,
                        "instruments": m.books.len(),
                    },
                    "sources": {"graph": st.sources.graph, "decoder": st.sources.decoder, "fallback": st.sources.fallback},
                    "primitives": {
                        "real": m.real_ever.iter().collect::<Vec<_>>(),
                        "fallback_only_secs": fallback_only,
                    },
                })
            })
            .collect();
        let src = self.sources();
        json!({
            "population": self.population,
            "reloads": self.reloads,
            "instruments": self.instruments(),
            "settings": {
                "step_budget_ms": s.step_budget.as_secs_f64() * 1000.0,
                "budget_breach_limit": s.budget_breach_limit,
                "nan_breach_limit": s.nan_breach_limit,
                "starved_after_secs": s.starved_after_ms / 1000,
                "price_history": s.price_history,
                "decoder_max_age_secs": s.decoder_max_age_ms / 1000,
                "analog_cache_max": s.analog_cache_max,
            },
            "sources": {"graph": src.graph, "decoder": src.decoder, "fallback": src.fallback},
            "analog": {
                "unresolved": self.analog_unresolved,
                "resolved_tsdb": self.analog_resolved_tsdb,
                "resolved_memory": self.analog_resolved_memory,
                "cache": self.analogs.len(),
                "pending": self.analogs.pending(),
                "fetches": self.analogs.fetches,
                "fetch_errors": self.analogs.fetch_errors,
            },
            "decoder_families": self.decoders.families(),
            "strategies": strategies,
        })
    }
}

/// spec 35 §8: NaN / out-of-range → 0 exposure.
fn sanitize(e: &mut AssetExposure) {
    if !e.mean.is_finite() {
        e.mean = 0.0;
    }
    e.mean = e.mean.clamp(-1.0, 1.0);
    if !e.stddev.is_finite() {
        e.stddev = 1.0;
    }
    e.stddev = e.stddev.clamp(0.0, 1.0);
}

#[cfg(test)]
mod tests {
    use fa_proto::v1::{ExpertLatent, ExpertSpaceRef, TransitionProb};

    use super::*;

    fn runtime() -> Runtime {
        Runtime::new(&StrategyFile::baselines()).unwrap()
    }

    fn ts(ms: i64) -> fa_proto::Timestamp {
        fa_proto::Timestamp { seconds: ms.div_euclid(1000), nanos: (ms.rem_euclid(1000) * 1_000_000) as i32 }
    }

    #[test]
    fn no_signals_without_trades_then_signals_at_cadence() {
        let mut rt = runtime();
        assert!(rt.tick(1_000).is_empty());
        let inst = InstrumentId::new();
        for i in 0..40 {
            rt.on_trade(inst, i * 1_000, 100.0 + i as f64 * 0.1);
        }
        let first = rt.tick(40_000);
        assert_eq!(first.len(), rt.strategy_ids().len(), "every strategy steps on first due tick");
        let s = first.iter().find(|d| d.strategy_id == "baseline-momentum-1s").unwrap();
        assert_eq!(s.exposures.len(), 1);
        assert!(s.exposures[0].mean > 0.0, "rising prices → positive momentum exposure");
        let rev = first.iter().find(|d| d.strategy_id == "baseline-reversion-1s").unwrap();
        assert!(rev.exposures[0].mean < 0.0);
        assert!(rt.tick(40_500).is_empty(), "nothing due half a second later");
        let second = rt.tick(41_000);
        assert_eq!(
            second.iter().filter(|d| d.horizon.timescale == Timescale::Second).count(),
            6,
            "1s strategies step every second"
        );
        assert!(!second.iter().any(|d| d.strategy_id == "baseline-regime-1m"), "10 s cadence");
        assert!(rt.tick(50_000).iter().any(|d| d.strategy_id == "baseline-regime-1m"));
        // price only → every primitive came from the fallback
        let s = rt.sources();
        assert!(s.fallback > 0 && s.graph == 0 && s.decoder == 0, "{s:?}");
        let (_, st) = rt.member_stats().into_iter().find(|(id, _)| *id == "baseline-momentum-1s").unwrap();
        assert_eq!(st.steps, 3);
        assert!(st.last_mean > 0.0);
    }

    #[test]
    fn instrument_filter_and_latent_dwell() {
        let allowed = InstrumentId::new();
        let other = InstrumentId::new();
        let mut file = StrategyFile::baselines();
        file.instruments = vec![allowed.to_string()];
        let mut rt = Runtime::new(&file).unwrap();
        rt.on_trade(other, 0, 1.0);
        rt.on_trade(allowed, 0, 1.0);
        assert_eq!(rt.instruments(), 1);
        let st = |region: &str| LatentState {
            timescale: "1s".into(),
            instrument_id: allowed.to_string(),
            experts: vec![ExpertLatent { region_id: region.into(), log_var: vec![0.0], ..Default::default() }],
            ..Default::default()
        };
        rt.on_latent(&st("r1"));
        rt.on_latent(&st("r1"));
        rt.on_latent(&st("r2"));
        assert_eq!(rt.latent_seen, 3);
        assert_eq!(rt.latent[&(Timescale::Second, allowed)].info.dwell, 0);
        rt.on_latent(&st("r2"));
        assert_eq!(rt.latent[&(Timescale::Second, allowed)].info.dwell, 1);
        assert_eq!(rt.latent[&(Timescale::Second, allowed)].info.uncertainty, 1.0);
        assert_eq!(rt.graph.seen.states, 4);
    }

    #[test]
    fn virtual_book_tracks_own_exposure() {
        let mut rt = runtime();
        let inst = InstrumentId::new();
        for i in 0..40 {
            rt.on_trade(inst, i * 1_000, 100.0 + i as f64 * 0.1);
        }
        rt.tick(40_000);
        rt.on_trade(inst, 41_000, 105.0);
        rt.tick(41_000);
        let m = rt.members.iter().find(|m| m.def.id == "baseline-momentum-1s").unwrap();
        let b = m.books[&inst];
        assert!(b.exposure > 0.0);
        assert_eq!(b.steps_in_trade, 1);
        assert!(b.unrealised > 0.0, "long exposure into an up move");
    }

    /// Feeds a 1s graph: regions a/b, rows a→b 0.8, b→a 0.9, the instrument in `a`.
    fn feed_graph(rt: &mut Runtime, inst: InstrumentId, at_ms: i64) {
        let space = Some(ExpertSpaceRef { snapshot_id: "stat-v1-1s".into(), timescale: "1s".into(), expert_index: 0 });
        for r in ["a", "b"] {
            rt.on_region(
                &RegionEvent {
                    kind: 1,
                    space: space.clone(),
                    region_id: r.into(),
                    lineage_id: r.into(),
                    ..Default::default()
                },
                at_ms,
            );
        }
        let row = |from: &str, to: &[(&str, f64)]| TransitionUpdate {
            space: space.clone(),
            from_region_id: from.into(),
            to: to
                .iter()
                .map(|(r, m)| TransitionProb { region_id: (*r).into(), mean: *m, lower: m - 0.05, upper: m + 0.05 })
                .collect(),
            sampler_id: "conjugate".into(),
            as_of: Some(ts(at_ms)),
        };
        rt.on_transition(&row("a", &[("b", 0.8), ("a", 0.1), ("unknown", 0.1)]));
        rt.on_transition(&row("b", &[("a", 0.9), ("unknown", 0.1)]));
        rt.on_latent(&LatentState {
            timescale: "1s".into(),
            instrument_id: inst.to_string(),
            as_of: Some(ts(at_ms)),
            experts: vec![ExpertLatent { region_id: "a".into(), log_var: vec![-4.0], ..Default::default() }],
            ..Default::default()
        });
    }

    #[test]
    fn graph_feed_makes_graph_primitives_real_and_feeds_region_returns() {
        let mut rt = runtime();
        let inst = InstrumentId::new();
        for i in 0..40 {
            rt.on_trade(inst, i * 1_000, 100.0 + i as f64 * 0.1);
        }
        feed_graph(&mut rt, inst, 30_000);
        let dists = rt.tick(40_000);
        let nb = dists.iter().find(|d| d.strategy_id == "baseline-graph-neighbors-1s").unwrap();
        // p_to_set(a, neighbors(a, 3), 1) = mass to b = 0.8 → signal 4·(0.8 − 0.5) = 1.2 → tanh
        assert!((nb.exposures[0].mean - 1.2f64.tanh()).abs() < 0.02, "{}", nb.exposures[0].mean);
        let s = rt.sources();
        assert!(s.graph > 0, "{s:?}");
        let (_, st) = rt.member_stats().into_iter().find(|(id, _)| *id == "baseline-graph-neighbors-1s").unwrap();
        assert_eq!(st.sources.fallback, 0, "{:?}", st.sources);
        assert_eq!(st.sources.graph, 2);
        // the bar before bucket 40 closed → one close recorded per population timescale
        assert_eq!(rt.graph.seen.closes, 0, "first close only seeds the previous price");
        rt.on_trade(inst, 41_000, 104.5);
        rt.tick(41_000);
        assert_eq!(rt.graph.seen.closes, 1);
        rt.tick(41_000);
        assert_eq!(rt.graph.seen.closes, 1, "a bucket closes once");
        // region a now has a realised return → region_feature answers from the graph
        let view = rt.graph.at(Timescale::Second, 42_000);
        assert!(view.region_feature(0, "a", fa_strategies::RegionFeature::MeanFwdReturn).is_some());
    }

    fn state_of(snap: &str, inst: InstrumentId, at_ms: i64, region: &str) -> LatentState {
        LatentState {
            snapshot_id: snap.into(),
            timescale: "1s".into(),
            instrument_id: inst.to_string(),
            as_of: Some(ts(at_ms)),
            experts: vec![ExpertLatent {
                expert_index: 0,
                mean: vec![0.0],
                log_var: vec![0.0],
                region_id: region.into(),
                region_lineage_id: region.into(),
            }],
            router_weights: vec![],
        }
    }

    fn created(snap: &str, id: &str) -> RegionEvent {
        RegionEvent {
            kind: fa_proto::v1::region_event::Kind::Created as i32,
            space: Some(ExpertSpaceRef { snapshot_id: snap.into(), timescale: "1s".into(), expert_index: 0 }),
            region_id: id.into(),
            lineage_id: id.into(),
            related_region_ids: vec![],
        }
    }

    fn carried(snap: &str, id: &str, parent: &str) -> RegionLifecycleEvent {
        RegionLifecycleEvent {
            kind: fa_proto::v1::region_lifecycle_event::Kind::Carried as i32,
            space: Some(ExpertSpaceRef { snapshot_id: snap.into(), timescale: "1s".into(), expert_index: 0 }),
            region_id: id.into(),
            reason: "snapshot".into(),
            ancestors: vec![fa_proto::v1::RegionAncestor {
                region_id: parent.into(),
                lineage_id: String::new(),
                overlap: 0.9,
                kind: "carry".into(),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn snapshot_change_remaps_literals_or_suspends() {
        let file = StrategyFile::parse(
            "[[strategy]]\nid='gp-lit'\ngenome_id='gp-lit'\ntimescale='1s'\nsnapshot='s1'\n\
             signal='(mul 3 (sub 0.5 (dwell (region 0 2))))'\n\
             [[strategy]]\nid='gp-dead'\ngenome_id='gp-dead'\ntimescale='1s'\nsnapshot='s1'\nsignal='(dwell (region 0 4))'\n",
        )
        .unwrap();
        let mut rt = Runtime::new(&file).unwrap();
        let inst = InstrumentId::new();
        for i in 0..20 {
            rt.on_trade(inst, i * 1_000, 100.0 + i as f64 * 0.1);
        }
        for r in ["s1-e0-r002", "s1-e0-r004"] {
            rt.on_region(&created("s1", r), 0);
        }
        rt.on_latent(&state_of("s1", inst, 19_000, "s1-e0-r002"));
        rt.tick(20_000);
        assert!(rt.remap_of("gp-lit").is_none(), "bound snapshot is live: nothing to do");
        assert!(!rt.is_paused("gp-lit"));

        // the encoder swaps to s2: r002 is carried to r007, r004 has no descendant
        rt.on_lifecycle(&carried("s2", "s2-e0-r007", "s1-e0-r002"));
        rt.on_region(&created("s2", "s2-e0-r007"), 20_500);
        rt.on_latent(&state_of("s2", inst, 21_000, "s2-e0-r007"));
        rt.tick(22_000);
        let m = rt.members.iter().find(|m| m.def.id == "gp-lit").unwrap();
        assert_eq!(m.def.signal, "(mul 3 (sub 0.5 (dwell (region 0 7))))");
        assert_eq!(m.def.snapshot.as_deref(), Some("s2"));
        let r = rt.remap_of("gp-lit").unwrap();
        assert!(r.ok && r.at_ms == 22_000 && r.from == "s1" && r.to == "s2");
        assert!(!rt.is_paused("gp-lit"));
        assert!(matches!(rt.paused.get("gp-dead"), Some(PauseReason::Snapshot { .. })));
        let d = rt.tick(23_000);
        let dead = d.iter().find(|d| d.strategy_id == "gp-dead").unwrap().exposures[0];
        assert_eq!((dead.mean, dead.stddev), (0.0, 1.0), "suspended: flat");
        let hb = rt.heartbeat_counters(23_000);
        assert_eq!((hb["snapshot.remapped"], hb["snapshot.suspended"]), (1.0, 1.0));
        assert_eq!(hb["strategy.gp-dead.paused_reason"], 4.0);
        let st = rt.status(23_000);
        let lit = st["strategies"].as_array().unwrap().iter().find(|s| s["id"] == "gp-lit").unwrap();
        assert_eq!(lit["snapshot"]["remapped_ms"], 22_000);

        // a reload brings the old genome back: same remap, same time (no new demotion signal)
        rt.replace_population(&file, "reload").unwrap();
        rt.tick(30_000);
        assert_eq!(rt.remap_of("gp-lit").unwrap().at_ms, 22_000);
        assert!(rt.members.iter().any(|m| m.def.signal.contains("(region 0 7)")));
    }

    #[test]
    fn reload_swaps_population_atomically_and_keeps_books() {
        let mut rt = runtime();
        let inst = InstrumentId::new();
        for i in 0..40 {
            rt.on_trade(inst, i * 1_000, 100.0 + i as f64 * 0.1);
        }
        rt.tick(40_000);
        let before = rt.members.iter().find(|m| m.def.id == "baseline-momentum-1s").unwrap().books[&inst];
        assert!(rt.pause("baseline-reversion-1s"));
        assert!(!rt.pause("nope"));

        // invalid file → untouched
        let bad = StrategyFile::parse("[[strategy]]\nid='x'\ntimescale='1s'\nsignal='1'\n").unwrap();
        let mut bad = bad;
        bad.strategy[0].signal = "(rsi 14)".into();
        assert!(rt.replace_population(&bad, "bad").is_err());
        assert_eq!(rt.strategy_ids().len(), 9);
        assert_eq!(rt.reloads, 0);

        let next = StrategyFile::parse(
            "[[strategy]]\nid='baseline-momentum-1s'\ngenome_id='gen-3-abc'\ntimescale='1s'\nsignal='(mul 400 analog_forecast)'\n\
             [[strategy]]\nid='evolved-1'\ngenome_id='gen-3-def'\ntimescale='1s'\nsignal='(p_to_set (cur_region 0) (neighbors (cur_region 0) 2) 1)'\n",
        )
        .unwrap();
        let ids = rt.replace_population(&next, "strategies/strategies.3.toml").unwrap();
        assert_eq!(ids, vec!["baseline-momentum-1s", "evolved-1"]);
        assert_eq!(rt.reloads, 1);
        assert_eq!(rt.population, "strategies/strategies.3.toml");
        assert_eq!(rt.paused().count(), 0, "paused id that left the population is forgotten");
        let kept = rt.members.iter().find(|m| m.def.id == "baseline-momentum-1s").unwrap();
        assert_eq!(kept.books[&inst], before, "surviving id keeps its virtual book");
        assert_eq!(kept.strategy.genome_id, "gen-3-abc");
        let dists = rt.tick(41_000);
        assert_eq!(dists.len(), 2);
        assert_eq!(dists[0].genome_id, "gen-3-abc");

        // pause → flat with full uncertainty; resume → signal again
        assert!(rt.pause("evolved-1"));
        let d = rt.tick(42_000);
        let e = d.iter().find(|d| d.strategy_id == "evolved-1").unwrap().exposures[0];
        assert_eq!((e.mean, e.stddev), (0.0, 1.0));
        assert!(rt.resume("evolved-1"));
        assert!(!rt.is_paused("evolved-1"));
    }

    fn one(signal: &str) -> StrategyFile {
        StrategyFile::parse(&format!("[[strategy]]\nid='s'\ntimescale='1s'\nsignal='{signal}'\n")).unwrap()
    }

    fn trades(rt: &mut Runtime, inst: InstrumentId) {
        for i in 0..40 {
            rt.on_trade(inst, i * 1_000, 100.0 + i as f64 * 0.1);
        }
    }

    #[test]
    fn step_budget_skips_then_auto_pauses_after_n_breaches() {
        let settings = Settings { budget_breach_limit: 3, ..Settings::default() };
        let mut rt = Runtime::with_settings(&one("(mul 400 analog_forecast)"), settings).unwrap();
        let slow = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let flag = slow.clone();
        rt.set_eval_clock(Box::new(move |_, wall| {
            if flag.load(std::sync::atomic::Ordering::SeqCst) { Duration::from_millis(80) } else { wall }
        }));
        let inst = InstrumentId::new();
        trades(&mut rt, inst);
        // two breaches: skipped (nothing published), not paused
        assert!(rt.tick(40_000).is_empty());
        assert!(rt.tick(41_000).is_empty());
        assert!(!rt.is_paused("s"));
        // a fast step resets the streak
        slow.store(false, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(rt.tick(42_000).len(), 1);
        slow.store(true, std::sync::atomic::Ordering::SeqCst);
        for t in [43_000, 44_000] {
            assert!(rt.tick(t).is_empty());
        }
        assert!(!rt.is_paused("s"), "streak restarted after the fast step");
        assert!(rt.tick(45_000).is_empty());
        let reason = rt.pause_reason("s").unwrap().clone();
        assert!(matches!(reason, PauseReason::StepBudget { breaches: 3, .. }), "{reason:?}");
        assert!(reason.to_string().starts_with("step_budget: 3 consecutive steps over the 50 ms"), "{reason}");
        let hb = rt.heartbeat_counters(45_000);
        assert_eq!(hb["budget_exceeded.s"], 5.0);
        assert_eq!(hb["strategy.s.paused_reason"], 2.0);
        assert_eq!(hb["strategy.s.paused"], 1.0);
        assert_eq!(hb["auto_paused"], 1.0);
        // auto-paused → flat, not evaluated (the slow clock is not consulted)
        let d = rt.tick(46_000);
        assert_eq!((d[0].exposures[0].mean, d[0].exposures[0].stddev), (0.0, 1.0));
        assert_eq!(rt.heartbeat_counters(46_000)["budget_exceeded.s"], 5.0);
        // operator resume clears the pause and the streak
        slow.store(false, std::sync::atomic::Ordering::SeqCst);
        assert!(rt.resume("s"));
        let d = rt.tick(47_000);
        assert!(d[0].exposures[0].mean > 0.0);
        assert_eq!(rt.heartbeat_counters(47_000)["strategy.s.paused_reason"], 0.0);
    }

    #[test]
    fn nan_outputs_zero_exposure_then_auto_pause_after_n_steps() {
        let settings = Settings { nan_breach_limit: 2, ..Settings::default() };
        // finite while position ≤ 0.5, then inf·inf → NaN path (eval replaces it with 0 and counts it)
        let mut rt = Runtime::with_settings(&one("(mul 1e308 1e308)"), settings).unwrap();
        let inst = InstrumentId::new();
        trades(&mut rt, inst);
        let d = rt.tick(40_000);
        assert_eq!((d[0].exposures[0].mean, d[0].exposures[0].stddev), (0.0, 1.0), "NaN → 0 exposure");
        assert!(!rt.is_paused("s"));
        rt.tick(41_000);
        assert!(matches!(rt.pause_reason("s"), Some(PauseReason::NonFinite { breaches: 2 })));
        let hb = rt.heartbeat_counters(41_000);
        assert_eq!(hb["strategy.s.nan_outputs"], 2.0);
        assert_eq!(hb["strategy.s.paused_reason"], 3.0);
        assert_eq!(hb["nan_outputs"], 2.0);
        assert_eq!(rt.auto_pauses, 1);
        // a finite program is never flagged
        let mut ok = Runtime::with_settings(&one("(mul 400 analog_forecast)"), settings).unwrap();
        trades(&mut ok, inst);
        for t in 0..5 {
            ok.tick(40_000 + t * 1_000);
        }
        assert!(!ok.is_paused("s"));
        assert_eq!(ok.heartbeat_counters(45_000)["strategy.s.nan_outputs"], 0.0);
    }

    #[test]
    fn starved_when_a_primitive_only_falls_back_for_longer_than_m() {
        let settings = Settings { starved_after_ms: 5_000, ..Settings::default() };
        let mut rt =
            Runtime::with_settings(&one("(add (text_score news) (mul 400 analog_forecast))"), settings).unwrap();
        let inst = InstrumentId::new();
        trades(&mut rt, inst);
        rt.tick(40_000);
        assert_eq!(rt.heartbeat_counters(40_000)["strategy.s.starved"], 0.0);
        rt.tick(46_000);
        let hb = rt.heartbeat_counters(46_000);
        assert_eq!(hb["strategy.s.starved"], 1.0, "text_score fell back for 6 s > 5 s");
        assert_eq!(hb["starved"], 1.0);
        let st = rt.status(46_000);
        let s0 = &st["strategies"][0];
        assert_eq!(s0["starved_primitives"], serde_json::json!(["analog_forecast", "region_entropy", "text_score"]));
        // a producer appears → text_score is real again; analog_forecast (no retrieval) and the
        // default uncertainty's region_entropy (no graph) stay starved
        rt.on_decoder_output(&fa_strategies_scored("text.news", Some(inst), 46_500, 0.4), 46_500);
        rt.tick(47_000);
        let st = rt.status(47_000);
        assert_eq!(st["strategies"][0]["starved_primitives"], serde_json::json!(["analog_forecast", "region_entropy"]));
        assert!(st["strategies"][0]["primitives"]["real"].as_array().unwrap().iter().any(|p| p == "text_score"));
    }

    fn fa_strategies_scored(family: &str, inst: Option<InstrumentId>, at_ms: i64, score: f64) -> DecoderOutput {
        DecoderOutput {
            decoder_id: format!("baseline-{family}"),
            genome_id: format!("{family}-baseline"),
            family: family.into(),
            snapshot_id: "snap".into(),
            timescale: "1s".into(),
            as_of: Some(ts(at_ms)),
            output_refs: vec![
                inst.map_or_else(|| "inst:*".to_owned(), |i| format!("inst:{i}")),
                "topic:news".into(),
                "src:gnews".into(),
            ],
            quantiles: vec![score],
            analogs: vec![],
        }
    }

    #[test]
    fn text_visual_baselines_read_synthetic_decoder_outputs() {
        let mut rt = runtime();
        let inst = InstrumentId::new();
        trades(&mut rt, inst);
        for (family, i, score) in [
            ("text.news", Some(inst), 0.6),
            ("visual.future", None, 0.5),
            ("text.narrative", Some(inst), 0.3),
            ("text.summary", None, 0.2),
            ("visual.chart", Some(inst), 0.7),
        ] {
            rt.on_decoder_output(&fa_strategies_scored(family, i, 39_000, score), 39_000);
        }
        let d = rt.tick(40_000);
        assert_eq!(d.len(), 9, "every baseline steps");
        let tv = d.iter().find(|x| x.strategy_id == "baseline-text-visual-1s").unwrap();
        assert!((tv.exposures[0].mean - (3.0f64 * 0.6 * 0.5).tanh()).abs() < 1e-12, "{}", tv.exposures[0].mean);
        let nc = d.iter().find(|x| x.strategy_id == "baseline-narrative-chart-1m").unwrap();
        assert!((nc.exposures[0].mean - (2.0f64 * 0.5).tanh()).abs() < 1e-12, "{}", nc.exposures[0].mean);
        for x in &d {
            let e = x.exposures[0];
            assert!((-1.0..=1.0).contains(&e.mean) && (0.0..=1.0).contains(&e.stddev), "{}", x.strategy_id);
        }
        let hb = rt.heartbeat_counters(40_000);
        assert!(hb["strategy.baseline-text-visual-1s.primitives.decoder"] >= 3.0, "{hb:?}");
        assert_eq!(hb["strategy.baseline-text-visual-1s.primitives.fallback"], 0.0);
        assert_eq!(hb["strategy.baseline-narrative-chart-1m.primitives.fallback"], 0.0);
        // status JSON lists the population, settings and per-strategy sources
        let st = rt.status(40_000);
        assert_eq!(st["population"], "baselines");
        assert_eq!(st["strategies"].as_array().unwrap().len(), 9);
        assert_eq!(st["settings"]["step_budget_ms"], 50.0);
        let fams = st["decoder_families"].as_array().unwrap();
        assert!(fams.iter().any(|f| f == "text.news") && fams.iter().any(|f| f == "visual.chart"));
        let tvs = st["strategies"].as_array().unwrap().iter().find(|s| s["id"] == "baseline-text-visual-1s").unwrap();
        assert!(tvs["sources"]["decoder"].as_u64().unwrap() >= 3);
        assert_eq!(tvs["paused"], false);
        assert!(tvs["primitives"]["real"].as_array().unwrap().iter().any(|p| p == "visual_score"));
    }

    #[test]
    fn analog_cache_is_bounded_and_point_in_time() {
        use fa_strategies::decoders::AnalogRequest;
        let s = Settings::default();
        let inst = InstrumentId::new();
        let key = |end: i64| (inst, AnalogRequest { window_end_ms: end, k: 5, bar: Timescale::Second });
        let mut c = AnalogCache::default();
        c.want(key(10_000), 14_000, &s);
        assert_eq!(c.pending(), 0, "outcome ends at 15 s > now: never queued");
        c.want(key(10_000), 15_000, &s);
        c.want(key(10_000), 15_000, &s);
        assert_eq!(c.pending(), 1, "deduplicated");
        let taken = c.take_pending(10);
        assert_eq!(taken, vec![key(10_000)]);
        c.want(key(10_000), 15_000, &s);
        assert_eq!(c.pending(), 1, "in flight, not queued twice");
        c.fill(key(10_000), Ok(Some((100.0, 101.0))), 15_000, 2);
        assert!(c.is_resolved(&key(10_000)));
        assert_eq!(c.points[&inst][&15_000].0, 101.0);
        c.want(key(10_000), 16_000, &s);
        assert_eq!(c.pending(), 0, "cached");
        // missing → retried only after analog_retry_ms
        c.fill(key(20_000), Ok(None), 30_000, 2);
        c.want(key(20_000), 31_000, &s);
        assert_eq!(c.pending(), 0);
        c.want(key(20_000), 30_000 + s.analog_retry_ms, &s);
        assert_eq!(c.pending(), 1);
        // bound: a third entry evicts the oldest and its price points
        c.fill(key(40_000), Ok(Some((1.0, 2.0))), 50_000, 2);
        assert_eq!(c.len(), 2);
        assert!(!c.is_resolved(&key(10_000)));
        assert!(!c.points[&inst].contains_key(&15_000));
    }
}
