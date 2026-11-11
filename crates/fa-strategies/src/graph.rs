//! Point-in-time latent region graph built from the bus (spec 33 §4.2 query surface for the
//! spec 35 §4.2 graph primitives).
//!
//! [`LatentGraph`] is fed by the three `model-runtime` feeds and by realised bar closes:
//!
//! | input | what is kept |
//! |---|---|
//! | `latent.state.<ts>.<inst>` (`LatentState`) | per `(timescale, instrument)` a bounded history of `(as_of, region per expert, posterior std per expert)` → `cur_region`, `dwell`, `novelty`, occupancy |
//! | `latent.region.<ts>` (`RegionEvent`) | per `(timescale, expert)` a region registry: id, lineage, status (`live`/`merged`/`split`/`retired`), related ids, event time |
//! | `latent.transition.<ts>` (`TransitionUpdate`) | per `(timescale, expert)` and source region a bounded list of **versions** `(as_of, successors with mean/lower/upper)` |
//! | bar closes (`record_close`) | per `(timescale, expert, region)` the realised one-bar log returns observed while an instrument sat in the region → `region_feature` |
//!
//! Every query goes through a [`GraphAt`] view fixed at an evaluation time `at_ms`: only
//! messages with event time ≤ `at_ms` are visible (rows take the latest version as of
//! `at_ms`), which keeps live evaluation and replays point-in-time. A view rebuilds the
//! kernel of each expert space it touches as an [`fa_mcmc::MarkovGraph`] (rows from the
//! published posterior summaries, see `MarkovGraph::set_row_from_summary`), so k-step
//! probabilities are matrix powers of the posterior-mean kernel and sampled walks carry the
//! publisher's parameter uncertainty. Sampling seeds are derived from
//! `(timescale, expert, region, k, n, at_ms)`, so a step is reproducible.
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
};

use fa_core::{InstrumentId, Timescale, rng::Rng};
use fa_mcmc::{ConjugateSampler, MarkovGraph, ProbSummary, RegionKey, Sampler, TransitionMatrix, UNKNOWN};
use fa_proto::v1::{LatentState, RegionEvent, TransitionUpdate, region_event};

use crate::{HITTING_HORIZON, RegionFeature};

/// Credible-interval mass the publisher uses (model-runtime: 90 %).
pub const INTERVAL_LEVEL: f64 = 0.9;
/// Default bounded history sizes.
pub const MAX_STATES_PER_INSTRUMENT: usize = 512;
pub const MAX_ROW_VERSIONS: usize = 16;
pub const MAX_RETURNS_PER_REGION: usize = 2048;
/// Walks per `hitting_time` / `path_up` query.
pub const WALKS: u32 = 256;

fn ms(t: Option<&fa_proto::Timestamp>) -> Option<i64> {
    t.map(|t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000)
}

/// `-e<expert>-r<nnn>`: the id suffix a region literal matches.
pub fn literal_suffix(expert: u32, ordinal: u32) -> String {
    format!("-e{expert}-r{ordinal:03}")
}

/// `(expert, ordinal)` of a region id `<prefix>-e<k>-r<nnn>`, if it has that shape.
pub fn parse_region_id(id: &str) -> Option<(u32, u32)> {
    let (head, ord) = id.rsplit_once("-r")?;
    let (_, expert) = head.rsplit_once("-e")?;
    Some((expert.parse().ok()?, ord.parse().ok()?))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionStatus {
    Live,
    Merged,
    Split,
    Retired,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RegionInfo {
    pub region_id: String,
    pub lineage_id: String,
    pub status: RegionStatus,
    pub related: Vec<String>,
    /// Event time of the creation event (receive time: `RegionEvent` carries no timestamp).
    pub created_ms: i64,
    pub updated_ms: i64,
}

/// One successor row version as published.
#[derive(Debug, Clone, PartialEq)]
pub struct RowVersion {
    pub as_of_ms: i64,
    pub sampler_id: String,
    pub to: Vec<(RegionKey, ProbSummary)>,
}

#[derive(Debug, Clone, PartialEq)]
struct ExpertObs {
    expert: u32,
    region_id: String,
    /// Mean posterior std of the latent (`exp(log_var / 2)`).
    uncertainty: f64,
    /// First [`LATENT_DIMS_KEPT`] dims of the latent mean (neural policy input, spec 35 §4.3).
    mean: Vec<f32>,
}

/// Latent mean dimensions kept per state (the neural policy reads at most this many).
pub const LATENT_DIMS_KEPT: usize = crate::neural::MAX_LATENT_DIMS;

#[derive(Debug, Clone, PartialEq)]
struct StateObs {
    as_of_ms: i64,
    /// Encoder snapshot that produced the state (`""` = unknown / in-process baseline).
    snapshot: String,
    experts: Vec<ExpertObs>,
}

#[derive(Debug, Default)]
struct Space {
    regions: BTreeMap<String, RegionInfo>,
    /// Versions ascending by `as_of_ms`.
    rows: BTreeMap<String, VecDeque<RowVersion>>,
    /// `(at_ms, one-bar log return)` per region, ascending.
    returns: BTreeMap<String, VecDeque<(i64, f64)>>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GraphCounters {
    pub states: u64,
    pub regions: u64,
    pub transitions: u64,
    pub closes: u64,
}

/// Bus-fed latent graph state (all timescales, all experts).
#[derive(Debug, Default)]
pub struct LatentGraph {
    spaces: BTreeMap<(Timescale, u32), Space>,
    states: BTreeMap<(Timescale, InstrumentId), VecDeque<StateObs>>,
    /// Last bar close per `(timescale, instrument)`: `(at_ms, price)`.
    last_close: BTreeMap<(Timescale, InstrumentId), (i64, f64)>,
    /// Encoder snapshot of the newest `LatentState` per timescale (spec 35 §6.5).
    snapshots: BTreeMap<Timescale, String>,
    pub seen: GraphCounters,
}

impl LatentGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingests a `LatentState` (all experts). States are kept per `(timescale, instrument)` in
    /// event-time order; out-of-order states are dropped.
    pub fn on_state(&mut self, st: &LatentState) -> bool {
        self.ingest_state(st, true)
    }

    /// Ingests a state of a snapshot served **in parallel** (model-runtime's
    /// `latent.shadow.<ts>.<inst>`, spec 35 §6.5): kept next to the live states and readable
    /// through [`Self::at_snapshot`], but it never changes which snapshot the timescale serves.
    pub fn on_parallel_state(&mut self, st: &LatentState) -> bool {
        self.ingest_state(st, false)
    }

    /// Event time of the newest state of `snapshot` on `ts` (any instrument).
    pub fn last_state_ms(&self, ts: Timescale, snapshot: &str) -> Option<i64> {
        self.states
            .iter()
            .filter(|((t, _), _)| *t == ts)
            .filter_map(|(_, h)| h.iter().rev().find(|o| o.snapshot == snapshot).map(|o| o.as_of_ms))
            .max()
    }

    fn ingest_state(&mut self, st: &LatentState, live: bool) -> bool {
        let (Ok(ts), Some(inst)) = (st.timescale.parse::<Timescale>(), parse_inst(&st.instrument_id)) else {
            return false;
        };
        let as_of_ms = ms(st.as_of.as_ref()).unwrap_or(0);
        let hist = self.states.entry((ts, inst)).or_default();
        if hist.back().is_some_and(|b| as_of_ms < b.as_of_ms) {
            return false;
        }
        let experts = st
            .experts
            .iter()
            .filter(|e| !e.region_id.is_empty())
            .map(|e| ExpertObs {
                expert: e.expert_index,
                region_id: e.region_id.clone(),
                uncertainty: if e.log_var.is_empty() {
                    0.0
                } else {
                    e.log_var.iter().map(|lv| (f64::from(*lv) / 2.0).exp()).sum::<f64>() / e.log_var.len() as f64
                },
                mean: e.mean.iter().take(LATENT_DIMS_KEPT).copied().collect(),
            })
            .collect();
        hist.push_back(StateObs { as_of_ms, snapshot: st.snapshot_id.clone(), experts });
        if live && !st.snapshot_id.is_empty() && self.snapshots.get(&ts) != Some(&st.snapshot_id) {
            self.snapshots.insert(ts, st.snapshot_id.clone());
        }
        while hist.len() > MAX_STATES_PER_INSTRUMENT {
            hist.pop_front();
        }
        self.seen.states += 1;
        true
    }

    /// Ingests a `RegionEvent`; `at_ms` is the receive time (the message has no timestamp).
    pub fn on_region(&mut self, ev: &RegionEvent, at_ms: i64) -> bool {
        let Some(space) = ev.space.as_ref() else {
            return false;
        };
        let Ok(ts) = space.timescale.parse::<Timescale>() else {
            return false;
        };
        let status = match region_event::Kind::try_from(ev.kind).unwrap_or_default() {
            region_event::Kind::Merged => RegionStatus::Merged,
            region_event::Kind::Split => RegionStatus::Split,
            region_event::Kind::Retired => RegionStatus::Retired,
            _ => RegionStatus::Live,
        };
        let sp = self.spaces.entry((ts, space.expert_index)).or_default();
        let entry = sp.regions.entry(ev.region_id.clone()).or_insert_with(|| RegionInfo {
            region_id: ev.region_id.clone(),
            lineage_id: ev.lineage_id.clone(),
            status,
            related: Vec::new(),
            created_ms: at_ms,
            updated_ms: at_ms,
        });
        entry.status = status;
        if !ev.lineage_id.is_empty() {
            entry.lineage_id = ev.lineage_id.clone();
        }
        entry.related = ev.related_region_ids.clone();
        entry.updated_ms = at_ms;
        self.seen.regions += 1;
        true
    }

    /// Ingests a `TransitionUpdate` as a new version of its row.
    pub fn on_transition(&mut self, tu: &TransitionUpdate) -> bool {
        let Some(space) = tu.space.as_ref() else {
            return false;
        };
        let Ok(ts) = space.timescale.parse::<Timescale>() else {
            return false;
        };
        if tu.from_region_id.is_empty() || tu.to.is_empty() {
            return false;
        }
        let as_of_ms = ms(tu.as_of.as_ref()).unwrap_or(0);
        let sp = self.spaces.entry((ts, space.expert_index)).or_default();
        let versions = sp.rows.entry(tu.from_region_id.clone()).or_default();
        let v = RowVersion {
            as_of_ms,
            sampler_id: tu.sampler_id.clone(),
            to: tu
                .to
                .iter()
                .map(|p| (p.region_id.clone(), ProbSummary { mean: p.mean, lower: p.lower, upper: p.upper }))
                .collect(),
        };
        // keep ascending; a same-time republish replaces the last version
        match versions.back() {
            Some(b) if as_of_ms < b.as_of_ms => return false,
            Some(b) if as_of_ms == b.as_of_ms => {
                versions.pop_back();
            }
            _ => {}
        }
        versions.push_back(v);
        while versions.len() > MAX_ROW_VERSIONS {
            versions.pop_front();
        }
        self.seen.transitions += 1;
        true
    }

    /// Records a closed bar of `inst` at `ts` (`at_ms` = bar end). The log return against the
    /// previous close is attributed to the region each expert placed the instrument in at the
    /// previous close (realised forward return of that region).
    pub fn record_close(&mut self, ts: Timescale, inst: InstrumentId, at_ms: i64, close: f64) {
        if close <= 0.0 || !close.is_finite() {
            return;
        }
        let prev = self.last_close.insert((ts, inst), (at_ms, close));
        let Some((prev_ms, prev_close)) = prev else {
            return;
        };
        if at_ms <= prev_ms || prev_close <= 0.0 {
            return;
        }
        let ret = (close / prev_close).ln();
        // the newest state of every snapshot served at the time (region ids are per snapshot)
        let Some(hist) = self.states.get(&(ts, inst)) else { return };
        let mut snaps: Vec<&str> = Vec::new();
        let mut regions: Vec<(u32, String)> = Vec::new();
        for obs in hist.iter().rev().filter(|o| o.as_of_ms <= prev_ms).take(8) {
            if snaps.contains(&obs.snapshot.as_str()) {
                continue;
            }
            snaps.push(&obs.snapshot);
            regions.extend(obs.experts.iter().map(|e| (e.expert, e.region_id.clone())));
        }
        if regions.is_empty() {
            return;
        }
        for (expert, region) in regions {
            let sp = self.spaces.entry((ts, expert)).or_default();
            let q = sp.returns.entry(region).or_default();
            q.push_back((at_ms, ret));
            while q.len() > MAX_RETURNS_PER_REGION {
                q.pop_front();
            }
        }
        self.seen.closes += 1;
    }

    fn state_at(&self, ts: Timescale, inst: InstrumentId, at_ms: i64, snap: Option<&str>) -> Option<&StateObs> {
        self.states.get(&(ts, inst))?.iter().rev().find(|s| s.as_of_ms <= at_ms && snap_ok(s, snap))
    }

    /// Timescales with any graph data.
    pub fn timescales(&self) -> BTreeSet<Timescale> {
        self.spaces.keys().map(|k| k.0).chain(self.states.keys().map(|k| k.0)).collect()
    }

    /// Number of regions registered (any status) per space.
    pub fn region_count(&self) -> usize {
        self.spaces.values().map(|s| s.regions.len()).sum()
    }

    pub fn row_count(&self) -> usize {
        self.spaces.values().map(|s| s.rows.len()).sum()
    }

    /// Whether the timescale has at least one transition row (the graph primitives have
    /// something real to answer with).
    pub fn has_graph(&self, ts: Timescale) -> bool {
        self.spaces.iter().any(|((t, _), s)| *t == ts && !s.rows.is_empty())
    }

    /// Point-in-time view at `at_ms` for one timescale.
    pub fn at(&self, ts: Timescale, at_ms: i64) -> GraphAt<'_> {
        GraphAt { g: self, ts, at_ms, spaces: RefCell::new(BTreeMap::new()), snap: self.snapshots.get(&ts).cloned() }
    }

    /// Point-in-time view reading the states of `snapshot` (spec 35 §6.5 parallel serving: a
    /// strategy bound to the previous snapshot keeps reading its latents while model-runtime
    /// still serves it). Kernel rows and regions are shared (ids are per snapshot).
    pub fn at_snapshot(&self, ts: Timescale, snapshot: &str, at_ms: i64) -> GraphAt<'_> {
        GraphAt { g: self, ts, at_ms, spaces: RefCell::new(BTreeMap::new()), snap: Some(snapshot.to_owned()) }
    }

    /// Encoder snapshot of the newest `LatentState` of `ts`.
    pub fn snapshot(&self, ts: Timescale) -> Option<&str> {
        self.snapshots.get(&ts).map(String::as_str)
    }
}

/// Whether a state belongs to the view's snapshot (states without a snapshot id always do).
fn snap_ok(s: &StateObs, snap: Option<&str>) -> bool {
    match snap {
        Some(want) => s.snapshot.is_empty() || s.snapshot == want,
        None => true,
    }
}

pub(crate) fn parse_inst(s: &str) -> Option<InstrumentId> {
    uuid::Uuid::parse_str(s).ok().map(InstrumentId)
}

/// Kernel of one expert space as of a time.
struct SpaceView {
    graph: MarkovGraph,
    matrix: Option<TransitionMatrix>,
    /// `(n, mean, std)` per region from returns ≤ `at_ms`.
    stats: BTreeMap<String, (usize, f64, f64)>,
    total_n: usize,
}

impl SpaceView {
    fn matrix(&mut self) -> &TransitionMatrix {
        if self.matrix.is_none() {
            self.matrix = Some(self.graph.transition_matrix());
        }
        self.matrix.as_ref().expect("set above")
    }

    fn feature(&self, region: &str, f: RegionFeature) -> Option<f64> {
        let (n, mean, std) = *self.stats.get(region)?;
        Some(match f {
            RegionFeature::Mass => {
                if self.total_n == 0 {
                    0.0
                } else {
                    n as f64 / self.total_n as f64
                }
            }
            RegionFeature::MeanFwdReturn => mean,
            RegionFeature::Vol => std,
        })
    }
}

/// Point-in-time query surface over one timescale (spec 33 §4.2). Cheap to create; kernels
/// are built lazily per expert on first use and cached for the view's lifetime.
pub struct GraphAt<'a> {
    g: &'a LatentGraph,
    ts: Timescale,
    at_ms: i64,
    spaces: RefCell<BTreeMap<u32, SpaceView>>,
    /// Snapshot whose states the view reads (the timescale's live one for [`LatentGraph::at`]).
    snap: Option<String>,
}

impl GraphAt<'_> {
    pub fn at_ms(&self) -> i64 {
        self.at_ms
    }

    pub fn timescale(&self) -> Timescale {
        self.ts
    }

    fn with_space<T>(&self, expert: u32, f: impl FnOnce(&mut SpaceView) -> T) -> T {
        let mut spaces = self.spaces.borrow_mut();
        let view = spaces.entry(expert).or_insert_with(|| self.build_space(expert));
        f(view)
    }

    fn build_space(&self, expert: u32) -> SpaceView {
        let mut graph = MarkovGraph::new(1e-3, 1.0);
        let mut stats = BTreeMap::new();
        let mut total_n = 0;
        if let Some(sp) = self.g.spaces.get(&(self.ts, expert)) {
            for (from, versions) in &sp.rows {
                if let Some(v) = versions.iter().rev().find(|v| v.as_of_ms <= self.at_ms) {
                    graph.set_row_from_summary(from, &v.to, INTERVAL_LEVEL);
                }
            }
            for (r, info) in &sp.regions {
                if info.created_ms <= self.at_ms && info.status == RegionStatus::Live {
                    graph.register_region(r);
                }
            }
            for (r, q) in &sp.returns {
                let xs: Vec<f64> = q.iter().filter(|(t, _)| *t <= self.at_ms).map(|x| x.1).collect();
                if xs.is_empty() {
                    continue;
                }
                let n = xs.len();
                let mean = xs.iter().sum::<f64>() / n as f64;
                let std = if n > 1 {
                    (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
                } else {
                    0.0
                };
                total_n += n;
                stats.insert(r.clone(), (n, mean, std));
            }
        }
        SpaceView { graph, matrix: None, stats, total_n }
    }

    fn seed(&self, expert: u32, from: &str, k: u32, n: u32) -> u64 {
        // FNV-1a over the query identity: deterministic per step.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |b: &[u8]| {
            for x in b {
                h ^= u64::from(*x);
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        };
        eat(self.ts.to_string().as_bytes());
        eat(&expert.to_le_bytes());
        eat(from.as_bytes());
        eat(&k.to_le_bytes());
        eat(&n.to_le_bytes());
        eat(&self.at_ms.to_le_bytes());
        h
    }

    /// Whether this expert space has any transition row as of the view time.
    pub fn has_rows(&self, expert: u32) -> bool {
        self.with_space(expert, |v| v.graph.regions().next().is_some())
    }

    /// Whether any region of this expert space has realised-return statistics.
    pub fn has_features(&self, expert: u32) -> bool {
        self.with_space(expert, |v| v.total_n > 0)
    }

    // ---- instrument state

    /// Latest state of `inst` as of the view time: `(as_of_ms, region, uncertainty)` for `expert`.
    pub fn current(&self, inst: InstrumentId, expert: u32) -> Option<(i64, String, f64)> {
        let obs = self.g.state_at(self.ts, inst, self.at_ms, self.snap.as_deref())?;
        let e = obs.experts.iter().find(|e| e.expert == expert)?;
        Some((obs.as_of_ms, e.region_id.clone(), e.uncertainty))
    }

    /// Latent mean (first [`LATENT_DIMS_KEPT`] dims) of `inst` in `expert` as of the view time.
    pub fn latent(&self, inst: InstrumentId, expert: u32) -> Option<Vec<f32>> {
        let obs = self.g.state_at(self.ts, inst, self.at_ms, self.snap.as_deref())?;
        obs.experts.iter().find(|e| e.expert == expert).map(|e| e.mean.clone())
    }

    /// `cur_region(expert)`.
    pub fn cur_region(&self, inst: InstrumentId, expert: u32) -> Option<String> {
        self.current(inst, expert).map(|c| c.1)
    }

    /// `dwell(r)`: bars `inst` has spent in its current region of `expert` (0 if it just entered;
    /// `None` without a state). Measured in event time, so preview states do not inflate it.
    pub fn dwell(&self, inst: InstrumentId, expert: u32) -> Option<f64> {
        let hist = self.g.states.get(&(self.ts, inst))?;
        let snap = self.snap.as_deref();
        let mut it = hist.iter().rev().filter(|s| s.as_of_ms <= self.at_ms && snap_ok(s, snap));
        let latest = it.next()?;
        let region = &latest.experts.iter().find(|e| e.expert == expert)?.region_id;
        let mut entered = latest.as_of_ms;
        for s in it {
            match s.experts.iter().find(|e| e.expert == expert) {
                Some(e) if &e.region_id == region => entered = s.as_of_ms,
                _ => break,
            }
        }
        let bar_ms = self.ts.duration().as_millis() as f64;
        Some(((latest.as_of_ms - entered) as f64 / bar_ms).max(0.0))
    }

    /// Instruments whose latest state (as of the view) sits in `region` for `expert`.
    pub fn occupancy(&self, expert: u32, region: &str) -> usize {
        self.g
            .states
            .iter()
            .filter(|((t, _), _)| *t == self.ts)
            .filter_map(|(_, h)| h.iter().rev().find(|s| s.as_of_ms <= self.at_ms && snap_ok(s, self.snap.as_deref())))
            .filter(|s| s.experts.iter().any(|e| e.expert == expert && e.region_id == region))
            .count()
    }

    /// `novelty(expert)`: `max(posterior std of the latent, unknown-successor mass of the current
    /// row)`, clamped to [0, 1]. `None` without a state.
    pub fn novelty(&self, inst: InstrumentId, expert: u32) -> Option<f64> {
        let (_, region, unc) = self.current(inst, expert)?;
        let unknown = self.with_space(expert, |v| {
            v.graph.posterior_mean(&region).into_iter().find(|(k, _)| k == UNKNOWN).map_or(0.0, |x| x.1)
        });
        Some(unc.clamp(0.0, 1.0).max(unknown))
    }

    // ---- kernel queries

    /// Encoder snapshot currently served on this timescale (newest `LatentState`).
    pub fn snapshot(&self) -> Option<&str> {
        self.snap.as_deref().or_else(|| self.g.snapshots.get(&self.ts).map(String::as_str))
    }

    /// Resolves the literal `(region expert ordinal)` to a region id of this timescale: the live
    /// region whose id ends in `-e<expert>-r<nnn>` (ids are `<prefix>-e<k>-r<nnn>`, the prefix
    /// being the snapshot or, for the in-process baseline, the timescale). `None` when no such
    /// region is known as of the view time.
    pub fn region_literal(&self, expert: u32, ordinal: u32) -> Option<String> {
        let suffix = literal_suffix(expert, ordinal);
        let sp = self.g.spaces.get(&(self.ts, expert))?;
        sp.regions
            .values()
            .filter(|r| r.created_ms <= self.at_ms && r.region_id.ends_with(&suffix))
            .max_by_key(|r| (r.status == RegionStatus::Live, r.created_ms))
            .map(|r| r.region_id.clone())
    }

    /// Registry entry of a region (any status) if known.
    pub fn region_info(&self, expert: u32, region: &str) -> Option<&RegionInfo> {
        self.g.spaces.get(&(self.ts, expert))?.regions.get(region).filter(|r| r.created_ms <= self.at_ms)
    }

    /// Posterior-mean one-step row of `from` (empty when the row is unknown).
    pub fn row(&self, expert: u32, from: &str) -> Vec<(RegionKey, f64)> {
        self.with_space(expert, |v| v.graph.posterior_mean(from))
    }

    /// `p_trans(a, b, k)`: posterior-mean probability of `a → b` in exactly `k` steps
    /// (matrix power). `None` when `a` has no row.
    pub fn p_trans(&self, expert: u32, a: &str, b: &str, k: u32) -> Option<f64> {
        self.with_space(expert, |v| {
            v.graph.row(a)?;
            Some(if k <= 1 {
                v.graph.posterior_mean(a).into_iter().find(|(r, _)| r == b).map_or(0.0, |x| x.1)
            } else {
                v.matrix().k_step(a, k).into_iter().find(|(r, _)| r == b).map_or(0.0, |x| x.1)
            })
        })
    }

    /// `p_to_set(r, S, k)`: k-step mass from `r` into `S`.
    pub fn p_to_set(&self, expert: u32, from: &str, set: &[String], k: u32) -> Option<f64> {
        self.with_space(expert, |v| {
            v.graph.row(from)?;
            let dist = if k <= 1 { v.graph.posterior_mean(from) } else { v.matrix().k_step(from, k) };
            Some(dist.into_iter().filter(|(r, _)| set.iter().any(|s| s == r)).map(|x| x.1).sum())
        })
    }

    /// `neighbors(r, n)`: successors by transition mass (excluding `r` and `unknown`).
    pub fn neighbors(&self, expert: u32, from: &str, n: u32) -> Vec<String> {
        self.with_space(expert, |v| v.graph.neighbors(from, n as usize).into_iter().map(|x| x.0).collect())
    }

    /// `region_entropy`: normalised Shannon entropy of the current row. `None` without a row.
    pub fn row_entropy(&self, expert: u32, from: &str) -> Option<f64> {
        self.with_space(expert, |v| v.graph.row(from).map(|_| v.graph.row_entropy(from)))
    }

    /// `sample_walk(r, k, n)`: `n` posterior-predictive walks of `k` steps, deterministic for the view.
    pub fn sample_walk(&self, expert: u32, from: &str, k: u32, n: u32) -> Vec<Vec<RegionKey>> {
        let mut rng = Rng::seed(self.seed(expert, from, k, n));
        self.with_space(expert, |v| ConjugateSampler.sample_paths(&v.graph, from, k as usize, n as usize, &mut rng))
    }

    /// `path_value(sample_walk(r, k, n), fn)`: mean of `fn` over the walks' end regions, where
    /// `fn` is `up` (end region has positive mean forward return) or a [`RegionFeature`]. Walks
    /// whose end region has no statistics are skipped; `None` when none has.
    pub fn path_value(&self, expert: u32, from: &str, k: u32, n: u32, f: Option<RegionFeature>) -> Option<f64> {
        let paths = self.sample_walk(expert, from, k, n);
        self.with_space(expert, |v| {
            let mut sum = 0.0;
            let mut cnt = 0usize;
            for p in &paths {
                let end = p.last()?;
                let val = match f {
                    None => v.feature(end, RegionFeature::MeanFwdReturn).map(|m| f64::from(m > 0.0)),
                    Some(f) => v.feature(end, f),
                };
                if let Some(x) = val {
                    sum += x;
                    cnt += 1;
                }
            }
            (cnt > 0).then(|| sum / cnt as f64)
        })
    }

    /// `hitting_time(r, S)`: mean first-hitting step of `S` over [`WALKS`] walks of
    /// [`HITTING_HORIZON`] steps (non-hitting walks count as the horizon). `None` without a row.
    pub fn hitting_time(&self, expert: u32, from: &str, set: &[String]) -> Option<f64> {
        if !self.with_space(expert, |v| v.graph.row(from).is_some()) {
            return None;
        }
        let paths = self.sample_walk(expert, from, HITTING_HORIZON, WALKS);
        Some(fa_mcmc::hitting_time(&paths, set, HITTING_HORIZON as usize).0)
    }

    /// `region_feature(r, f)` from realised returns as of the view; `None` without observations.
    pub fn region_feature(&self, expert: u32, region: &str, f: RegionFeature) -> Option<f64> {
        self.with_space(expert, |v| v.feature(region, f))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use fa_proto::v1::{ExpertLatent, ExpertSpaceRef, TransitionProb};

    use super::*;

    pub fn ts(ms: i64) -> fa_proto::Timestamp {
        fa_proto::Timestamp { seconds: ms.div_euclid(1000), nanos: (ms.rem_euclid(1000) * 1_000_000) as i32 }
    }

    pub fn space(expert: u32) -> Option<ExpertSpaceRef> {
        Some(ExpertSpaceRef { snapshot_id: "stat-v1-1s".into(), timescale: "1s".into(), expert_index: expert })
    }

    pub fn state(inst: InstrumentId, at_ms: i64, regions: &[&str]) -> LatentState {
        LatentState {
            snapshot_id: "stat-v1-1s".into(),
            timescale: "1s".into(),
            instrument_id: inst.to_string(),
            as_of: Some(ts(at_ms)),
            experts: regions
                .iter()
                .enumerate()
                .map(|(i, r)| ExpertLatent {
                    expert_index: i as u32,
                    mean: vec![0.0; 2],
                    log_var: vec![(0.25f32).ln() * 2.0], // std 0.25
                    region_id: (*r).into(),
                    region_lineage_id: (*r).into(),
                })
                .collect(),
            router_weights: vec![],
        }
    }

    pub fn row(from: &str, at_ms: i64, to: &[(&str, f64)]) -> TransitionUpdate {
        TransitionUpdate {
            space: space(0),
            from_region_id: from.into(),
            to: to
                .iter()
                .map(|(r, m)| TransitionProb {
                    region_id: (*r).into(),
                    mean: *m,
                    lower: (m - 0.1).max(0.0),
                    upper: (m + 0.1).min(1.0),
                })
                .collect(),
            sampler_id: "conjugate".into(),
            as_of: Some(ts(at_ms)),
        }
    }

    pub fn region(id: &str, kind: region_event::Kind) -> RegionEvent {
        RegionEvent {
            kind: kind as i32,
            space: space(0),
            region_id: id.into(),
            lineage_id: id.into(),
            related_region_ids: vec![],
        }
    }

    /// Toy chain: a → b 0.8 | c 0.1 | unknown 0.1; b → a 0.9 | unknown 0.1; c → c 0.9 | unknown 0.1.
    pub fn toy(inst: InstrumentId) -> LatentGraph {
        let mut g = LatentGraph::new();
        for r in ["a", "b", "c"] {
            g.on_region(&region(r, region_event::Kind::Created), 0);
        }
        g.on_transition(&row("a", 1_000, &[("b", 0.8), ("c", 0.1), (UNKNOWN, 0.1)]));
        g.on_transition(&row("b", 1_000, &[("a", 0.9), (UNKNOWN, 0.1)]));
        g.on_transition(&row("c", 1_000, &[("c", 0.9), (UNKNOWN, 0.1)]));
        for t in 0..5 {
            g.on_state(&state(inst, 1_000 + t * 1_000, &["a", "x"]));
        }
        g
    }

    #[test]
    fn a_snapshot_served_in_parallel_is_readable_without_becoming_live() {
        let inst = InstrumentId::new();
        let mut g = LatentGraph::new();
        let mut old = state(inst, 1_000, &["old-e0-r001"]);
        old.snapshot_id = "old".into();
        g.on_state(&old);
        let mut new = state(inst, 2_000, &["new-e0-r004"]);
        new.snapshot_id = "new".into();
        new.experts[0].mean = vec![0.5, -0.5];
        g.on_state(&new);
        // the old snapshot keeps being served in parallel (model-runtime shadow)
        for t in [2_000, 3_000] {
            let mut st = state(inst, t, &["old-e0-r001"]);
            st.snapshot_id = "old".into();
            assert!(g.on_parallel_state(&st));
        }
        assert_eq!(g.snapshot(Timescale::Second), Some("new"), "parallel states never change the live snapshot");
        let live = g.at(Timescale::Second, 3_500);
        assert_eq!(live.cur_region(inst, 0).as_deref(), Some("new-e0-r004"));
        assert_eq!(live.latent(inst, 0), Some(vec![0.5, -0.5]));
        let prev = g.at_snapshot(Timescale::Second, "old", 3_500);
        assert_eq!(prev.cur_region(inst, 0).as_deref(), Some("old-e0-r001"));
        assert_eq!(prev.snapshot(), Some("old"));
        assert_eq!(prev.dwell(inst, 0), Some(2.0), "dwell counts the old snapshot's own states");
        assert_eq!(g.last_state_ms(Timescale::Second, "old"), Some(3_000));
        assert_eq!(g.last_state_ms(Timescale::Second, "new"), Some(2_000));
    }

    #[test]
    fn k_step_probabilities_match_matrix_power() {
        let inst = InstrumentId::new();
        let g = toy(inst);
        let v = g.at(Timescale::Second, 10_000);
        assert_eq!(v.cur_region(inst, 0).as_deref(), Some("a"));
        assert_eq!(v.cur_region(inst, 1).as_deref(), Some("x"));
        assert_eq!(v.cur_region(inst, 2), None);
        let p1 = v.p_trans(0, "a", "b", 1).unwrap();
        assert!((p1 - 0.8).abs() < 1e-3, "{p1}");
        // two steps: a→b→a = 0.8·0.9 = 0.72; a→c→c = 0.1·0.9 = 0.09
        let p2 = v.p_trans(0, "a", "a", 2).unwrap();
        assert!((p2 - 0.72).abs() < 5e-3, "{p2}");
        let pc = v.p_trans(0, "a", "c", 2).unwrap();
        assert!((pc - 0.09).abs() < 5e-3, "{pc}");
        // matches fa_mcmc's dense power
        let dense = v.with_space(0, |s| s.graph.k_step_prob("a", "a", 2));
        assert!((p2 - dense).abs() < 1e-12);
        assert_eq!(v.p_trans(0, "zzz", "a", 1), None);
        assert_eq!(v.neighbors(0, "a", 5), vec!["b", "c"]);
        assert_eq!(v.neighbors(0, "a", 1), vec!["b"]);
        let to_set = v.p_to_set(0, "a", &["b".into(), "c".into()], 1).unwrap();
        assert!((to_set - 0.9).abs() < 1e-3);
        assert!(v.row_entropy(0, "a").unwrap() > 0.0);
        assert!(v.row_entropy(0, "c").unwrap() < v.row_entropy(0, "a").unwrap());
    }

    #[test]
    fn sampled_walks_are_deterministic_per_view() {
        let inst = InstrumentId::new();
        let g = toy(inst);
        let v = g.at(Timescale::Second, 10_000);
        let w1 = v.sample_walk(0, "a", 5, 50);
        let w2 = v.sample_walk(0, "a", 5, 50);
        assert_eq!(w1, w2);
        assert_eq!(w1.len(), 50);
        assert!(w1.iter().all(|p| p.len() <= 6 && p[0] == "a"));
        let v2 = g.at(Timescale::Second, 11_000);
        assert_ne!(v2.sample_walk(0, "a", 5, 50), w1, "a different step reseeds");
        let h = v.hitting_time(0, "a", &["c".into()]).unwrap();
        assert!(h > 1.0 && h < f64::from(HITTING_HORIZON), "{h}");
        let hb = v.hitting_time(0, "a", &["b".into()]).unwrap();
        assert!(hb < h, "b is hit sooner than c: {hb} vs {h}");
        assert_eq!(v.hitting_time(0, "nope", &["b".into()]), None);
    }

    #[test]
    fn point_in_time_filtering() {
        let inst = InstrumentId::new();
        let mut g = toy(inst);
        // a later version of row a flips the mass to c
        g.on_transition(&row("a", 20_000, &[("b", 0.1), ("c", 0.8), (UNKNOWN, 0.1)]));
        g.on_state(&state(inst, 20_000, &["b", "x"]));
        let early = g.at(Timescale::Second, 10_000);
        assert!((early.p_trans(0, "a", "b", 1).unwrap() - 0.8).abs() < 1e-3);
        assert_eq!(early.cur_region(inst, 0).as_deref(), Some("a"));
        let late = g.at(Timescale::Second, 20_000);
        assert!((late.p_trans(0, "a", "c", 1).unwrap() - 0.8).abs() < 1e-3);
        assert_eq!(late.cur_region(inst, 0).as_deref(), Some("b"));
        let before = g.at(Timescale::Second, 500);
        assert_eq!(before.cur_region(inst, 0), None, "no state yet");
        assert_eq!(before.p_trans(0, "a", "b", 1), None, "no row yet");
        // out-of-order messages are dropped
        assert!(!g.on_transition(&row("a", 15_000, &[("b", 1.0)])));
        assert!(!g.on_state(&state(inst, 19_000, &["c"])));
    }

    #[test]
    fn dwell_novelty_and_region_features() {
        let inst = InstrumentId::new();
        let mut g = toy(inst);
        let v = g.at(Timescale::Second, 10_000);
        assert_eq!(v.dwell(inst, 0), Some(4.0), "entered at 1 s, latest 5 s");
        assert_eq!(v.dwell(inst, 1), Some(4.0));
        assert_eq!(v.dwell(inst, 7), None);
        // uncertainty 0.25 vs unknown mass 0.1 → 0.25
        assert!((v.novelty(inst, 0).unwrap() - 0.25).abs() < 1e-9);
        assert_eq!(v.occupancy(0, "a"), 1);
        assert_eq!(v.occupancy(0, "b"), 0);
        assert_eq!(v.region_info(0, "a").unwrap().status, RegionStatus::Live);
        assert!(!v.has_features(0));
        drop(v);
        g.on_region(&region("c", region_event::Kind::Retired), 6_000);
        // closes: in region a from 1 s..; three up bars then one down bar
        g.record_close(Timescale::Second, inst, 2_000, 100.0);
        g.record_close(Timescale::Second, inst, 3_000, 101.0);
        g.record_close(Timescale::Second, inst, 4_000, 102.0);
        g.record_close(Timescale::Second, inst, 5_000, 101.5);
        assert_eq!(g.seen.closes, 3);
        let v = g.at(Timescale::Second, 10_000);
        assert!(v.has_features(0));
        assert_eq!(v.region_feature(0, "a", RegionFeature::Mass), Some(1.0));
        assert!(v.region_feature(0, "a", RegionFeature::MeanFwdReturn).unwrap() > 0.0);
        assert!(v.region_feature(0, "a", RegionFeature::Vol).unwrap() > 0.0);
        assert_eq!(v.region_feature(0, "b", RegionFeature::Mass), None);
        // only region a has stats → walks ending elsewhere are skipped, those in a count as up
        assert_eq!(v.path_value(0, "b", 1, 40, None), Some(1.0));
        assert_eq!(v.region_info(0, "c").unwrap().status, RegionStatus::Retired);
        // point in time: only the first two closes are visible at 3 s
        let v3 = g.at(Timescale::Second, 3_000);
        let (n, _, _) = v3.with_space(0, |s| s.stats["a"]);
        assert_eq!(n, 1);
    }

    #[test]
    fn region_literals_resolve_by_suffix_and_bind_to_the_snapshot() {
        use crate::{
            StrategyContext,
            features::{FallbackContext, LiveContext},
            sexpr,
        };
        let inst = InstrumentId::new();
        let mut g = LatentGraph::new();
        for r in ["1s-e0-r000", "1s-e0-r001", "1s-e0-r002"] {
            g.on_region(&region(r, region_event::Kind::Created), 0);
        }
        g.on_transition(&row("1s-e0-r000", 1_000, &[("1s-e0-r001", 0.7), ("1s-e0-r002", 0.2), (UNKNOWN, 0.1)]));
        for t in 0..4 {
            g.on_state(&state(inst, 1_000 + t * 1_000, &["1s-e0-r000"]));
        }
        assert_eq!(parse_region_id("stat-v1-1m-e3-r012"), Some((3, 12)));
        assert_eq!(parse_region_id("nope"), None);
        assert_eq!(g.snapshot(Timescale::Second), Some("stat-v1-1s"));
        let v = g.at(Timescale::Second, 10_000);
        assert_eq!(v.region_literal(0, 1).as_deref(), Some("1s-e0-r001"));
        assert_eq!(v.region_literal(0, 9), None);
        let e = sexpr::parse("(p_trans (cur_region 0) (region 0 1) 1)").unwrap();
        let ctx = |snap: Option<&'static str>| {
            LiveContext::new(FallbackContext::new(&[1.0, 1.0], None, Default::default()), inst, 1)
                .with_graph(&v)
                .with_snapshot(snap)
        };
        for snap in [None, Some("stat-v1-1s")] {
            let p = e.eval(&ctx(snap));
            assert!((p - 0.7).abs() < 1e-3, "{snap:?}: {p}");
        }
        // bound to another snapshot: the literal does not resolve (fallback 0, counted)
        let other = ctx(Some("stat-v2-1s"));
        assert_eq!(e.eval(&other), 0.0);
        assert_eq!(other.sources().fallback, 1);
        // dwell in a literal region: only when it is the current one
        let dwell_cur = sexpr::parse("(dwell (region 0 0))").unwrap().eval(&ctx(None));
        let dwell_other = sexpr::parse("(dwell (region 0 2))").unwrap().eval(&ctx(None));
        assert_eq!((dwell_cur, dwell_other), (3.0, 0.0));
        let _ = ctx(None).current_region();
    }
}
