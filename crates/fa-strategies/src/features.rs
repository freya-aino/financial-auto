//! Live observation for the strategy runtime: real graph / decoder data when available,
//! price-derived fallbacks otherwise.
//!
//! [`LiveContext`] wraps a price-only [`FallbackContext`] and, when given, a point-in-time
//! [`GraphAt`] view (spec 33 query surface over `latent.*`) and a [`DecodersAt`] view over
//! `decoder.output.*`. Every primitive answers from the first source that has data for the
//! strategy's timescale and instrument and records where the value came from
//! ([`Source::Graph`] / [`Source::Decoder`] / [`Source::Fallback`]) so the runtime heartbeat
//! can report how much of a strategy's input is real (`primitives.graph|decoder|fallback`).
//! Views are fixed at the step time, so primitives only see messages with event time ≤ now.
//!
//! | primitive | graph / decoder source | fallback (placeholder, not the spec'd semantics) |
//! |---|---|---|
//! | `cur_region(e)`, `current_region` | latest `LatentState` of the asset, expert `e` | expert-0 region of the latest state (`LatentInfo`) |
//! | `stay_prob`, `(p_trans a b k)` | matrix power of the posterior-mean kernel | dwell/(dwell+1) or sign persistence of bar returns |
//! | `(neighbors r n)`, `(p_to_set r S k)` | successors by transition mass, k-step mass into the set | empty / 0 |
//! | `(hitting_time r S)` | mean first-hitting step over seeded posterior walks | `HITTING_HORIZON` |
//! | `(dwell r)` | bars in the current region (event time) | `LatentInfo::dwell` |
//! | `(novelty e)` | max(posterior std, unknown-successor mass) | `region_entropy` fallback |
//! | `(region_feature r f)` | realised returns per region (`record_close`) | mass 0, mean/vol of lookback returns |
//! | `(path_up k)`, `(path_value (sample_walk r k n) f)` | seeded walks, end-region statistics | share of positive k-bar windows |
//! | `region_entropy` | normalised entropy of the current row | latent posterior std, else `std / (|mean| + std)` |
//! | `(quantile q)`, `(fc_quantile q k)`, `(fc_dist k)` | `rnn_prior` quantiles of the output with horizon closest to `k` (`quantile` uses the strategy's `steps`) | normal-approx quantile / mean of lookback returns |
//! | `analog_forecast`, `(analog_outcome k)` | realised outcomes after the `retrieval` analogs (unpriced analogs are skipped and reported through [`LiveContext::take_analog_requests`]) | mean log return per bar |
//! | `analog_age`, `(score family)` | analog time distance, output freshness | 0 |
//! | `(text_score t)`, `(visual_score k)` | score of the latest fresh `text.<t>` / `visual.<k>` output | 0 |
//!
//! Besides the per-source counters, the context remembers **which primitives** answered from
//! real data and which fell back ([`LiveContext::primitive_sources`]), so the runtime can flag a
//! program that keeps referencing a primitive nobody produces (spec 35 §8, `starved`).
//!
//! Portfolio primitives read the strategy's **virtual book**: its own last emitted exposure.
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeSet, VecDeque},
};

use fa_core::{InstrumentId, Timescale};

use crate::{
    HITTING_HORIZON, PathFn, RegionFeature, RegionRef, SetRef, StrategyContext,
    decoders::{AnalogRequest, DecodersAt, PriceLookup},
    graph::{GraphAt, WALKS},
};

/// Bounded trade-price history for one instrument.
#[derive(Debug, Clone, Default)]
pub struct PriceSeries {
    /// (event time ms, price), ascending.
    points: VecDeque<(i64, f64)>,
    max_points: usize,
}

impl PriceSeries {
    pub fn new(max_points: usize) -> Self {
        Self { points: VecDeque::new(), max_points: max_points.max(2) }
    }

    pub fn push(&mut self, at_ms: i64, price: f64) {
        if price <= 0.0 || !price.is_finite() {
            return;
        }
        // keep ascending order; late ticks are dropped
        if self.points.back().is_some_and(|(t, _)| at_ms < *t) {
            return;
        }
        self.points.push_back((at_ms, price));
        while self.points.len() > self.max_points {
            self.points.pop_front();
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn last(&self) -> Option<f64> {
        self.points.back().map(|p| p.1)
    }

    /// Last price at or before `at_ms`; `None` before the first point.
    pub fn price_at(&self, at_ms: i64) -> Option<f64> {
        let n = self.points.partition_point(|(t, _)| *t <= at_ms);
        (n > 0).then(|| self.points[n - 1].1)
    }

    /// Event time of the first retained point.
    pub fn first_ms(&self) -> Option<i64> {
        self.points.front().map(|p| p.0)
    }

    /// Last `n` bar closes at `timescale`, resampled from trades (last price per bucket,
    /// gaps carried forward). Fewer than `n` when history is short.
    pub fn closes(&self, timescale: Timescale, n: usize) -> Vec<f64> {
        let bucket_ms = timescale.duration().as_millis() as i64;
        let mut out: Vec<f64> = Vec::new();
        let mut cur_bucket: Option<i64> = None;
        for (t, p) in &self.points {
            let b = t.div_euclid(bucket_ms);
            match cur_bucket {
                None => out.push(*p),
                Some(cb) if b == cb => *out.last_mut().expect("non-empty") = *p,
                Some(cb) => {
                    let last = *out.last().expect("non-empty");
                    for _ in cb + 1..b {
                        out.push(last);
                    }
                    out.push(*p);
                }
            }
            cur_bucket = Some(b);
        }
        if out.len() > n {
            out.drain(..out.len() - n);
        }
        out
    }
}

impl PriceLookup for PriceSeries {
    fn price_at(&self, at_ms: i64) -> Option<f64> {
        PriceSeries::price_at(self, at_ms)
    }
}

/// Summary of the latest `LatentState` for an instrument (expert 0 of the strategy's timescale).
#[derive(Debug, Clone, PartialEq)]
pub struct LatentInfo {
    pub region_id: String,
    /// Mean posterior std of the latent (exp(log_var / 2)).
    pub uncertainty: f64,
    /// Consecutive states observed in the same region.
    pub dwell: u32,
}

/// A strategy's own virtual position for portfolio primitives.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VirtualBook {
    /// Last emitted exposure mean in [-1, 1].
    pub exposure: f64,
    /// Return of the instrument since the exposure's sign was last set, times exposure.
    pub unrealised: f64,
    /// Steps since the exposure's sign was last set.
    pub steps_in_trade: u32,
}

/// Price-only observation for one instrument at one step (see module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct FallbackContext {
    /// Log returns between consecutive bar closes, oldest first.
    returns: Vec<f64>,
    mean: f64,
    std: f64,
    latent: Option<LatentInfo>,
    book: VirtualBook,
}

impl FallbackContext {
    pub fn new(closes: &[f64], latent: Option<LatentInfo>, book: VirtualBook) -> Self {
        let returns: Vec<f64> =
            closes.windows(2).filter(|w| w[0] > 0.0 && w[1] > 0.0).map(|w| (w[1] / w[0]).ln()).collect();
        let n = returns.len() as f64;
        let mean = if n > 0.0 { returns.iter().sum::<f64>() / n } else { 0.0 };
        let std =
            if n > 1.0 { (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt() } else { 0.0 };
        Self { returns, mean, std, latent, book }
    }

    pub fn n_returns(&self) -> usize {
        self.returns.len()
    }

    pub fn mean_return(&self) -> f64 {
        self.mean
    }

    pub fn std_return(&self) -> f64 {
        self.std
    }
}

/// Acklam's rational approximation of the standard normal quantile (|err| < 1.2e-9).
pub fn norm_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.38357751867269e2,
        -3.066479806614716e1,
        2.506628277459239,
    ];
    const B: [f64; 5] =
        [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const C: [f64; 6] = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ];
    const D: [f64; 4] = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    let (lo, hi) = (0.02425, 1.0 - 0.02425);
    if p < lo {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= hi {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

impl StrategyContext for FallbackContext {
    fn current_region(&self) -> Option<&str> {
        self.latent.as_ref().map(|l| l.region_id.as_str())
    }

    fn transition_prob(&self, _to_region: &str) -> f64 {
        0.0
    }

    fn stay_prob(&self) -> f64 {
        if let Some(l) = &self.latent {
            return f64::from(l.dwell) / (f64::from(l.dwell) + 1.0);
        }
        if self.returns.len() < 2 {
            return 0.5;
        }
        let same = self.returns.windows(2).filter(|w| w[0].signum() == w[1].signum()).count();
        same as f64 / (self.returns.len() - 1) as f64
    }

    fn path_up_prob(&self, k: u32) -> f64 {
        let k = k.max(1) as usize;
        if self.returns.len() < k {
            return 0.5;
        }
        let windows = self.returns.windows(k);
        let n = windows.len();
        let up = windows.filter(|w| w.iter().sum::<f64>() > 0.0).count();
        up as f64 / n as f64
    }

    fn region_entropy(&self) -> f64 {
        if let Some(l) = &self.latent {
            return l.uncertainty.clamp(0.0, 1.0);
        }
        let d = self.mean.abs() + self.std;
        if d > 0.0 { self.std / d } else { 1.0 }
    }

    fn forecast_quantile(&self, level: f64) -> f64 {
        self.mean + norm_quantile(level) * self.std
    }

    fn analog_forecast(&self) -> f64 {
        self.mean
    }

    fn decoder_score(&self, _family: &str) -> f64 {
        0.0
    }

    fn position(&self) -> f64 {
        self.book.exposure
    }

    fn unrealised_pnl(&self) -> f64 {
        self.book.unrealised
    }

    fn time_in_trade(&self) -> f64 {
        f64::from(self.book.steps_in_trade)
    }
}

/// Where a primitive's value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Graph,
    Decoder,
    Fallback,
}

/// Per-source primitive call counts (snapshot of a context's counters).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Sources {
    pub graph: u64,
    pub decoder: u64,
    pub fallback: u64,
}

impl Sources {
    pub fn add(&mut self, o: Sources) {
        self.graph += o.graph;
        self.decoder += o.decoder;
        self.fallback += o.fallback;
    }

    pub fn total(&self) -> u64 {
        self.graph + self.decoder + self.fallback
    }
}

/// Observation for one instrument at one step: graph + decoder primitives on real data when
/// available, [`FallbackContext`] otherwise (see module docs).
pub struct LiveContext<'a> {
    fallback: FallbackContext,
    inst: InstrumentId,
    /// Strategy horizon in bars (the `k` of `analog_forecast`).
    steps: u32,
    /// Bar timescale of the strategy (the unit of `k`); set by `with_graph` / `with_bar`.
    bar: Timescale,
    graph: Option<&'a GraphAt<'a>>,
    decoders: Option<DecodersAt<'a>>,
    prices: Option<&'a dyn PriceLookup>,
    /// Current region per expert 0 as of the view (cached for `current_region`).
    cur0: Option<String>,
    graph_n: Cell<u64>,
    decoder_n: Cell<u64>,
    fallback_n: Cell<u64>,
    /// Primitive names that answered from real data / fell back in this context.
    real_prims: RefCell<BTreeSet<&'static str>>,
    fallback_prims: RefCell<BTreeSet<&'static str>>,
    /// Analogs priced / skipped by `analog_outcome` calls so far.
    analog_resolved: Cell<u64>,
    analog_unresolved: Cell<u64>,
    analog_requests: RefCell<Vec<AnalogRequest>>,
    analog_done: RefCell<Vec<AnalogRequest>>,
}

/// Primitive names split by where they answered from (one context, one step).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrimitiveSources {
    pub real: BTreeSet<&'static str>,
    pub fallback: BTreeSet<&'static str>,
}

impl PrimitiveSources {
    /// Primitives that only ever fell back in this context.
    pub fn fallback_only(&self) -> BTreeSet<&'static str> {
        self.fallback.difference(&self.real).copied().collect()
    }
}

impl<'a> LiveContext<'a> {
    pub fn new(fallback: FallbackContext, inst: InstrumentId, steps: u32) -> Self {
        let cur0 = fallback.current_region().map(str::to_owned);
        Self {
            fallback,
            inst,
            steps: steps.max(1),
            bar: Timescale::Second,
            graph: None,
            decoders: None,
            prices: None,
            cur0,
            graph_n: Cell::new(0),
            decoder_n: Cell::new(0),
            fallback_n: Cell::new(0),
            real_prims: RefCell::new(BTreeSet::new()),
            fallback_prims: RefCell::new(BTreeSet::new()),
            analog_resolved: Cell::new(0),
            analog_unresolved: Cell::new(0),
            analog_requests: RefCell::new(Vec::new()),
            analog_done: RefCell::new(Vec::new()),
        }
    }

    /// Price-only context (no graph, no decoders): every primitive reports `Fallback`.
    pub fn price_only(closes: &[f64], latent: Option<LatentInfo>, book: VirtualBook) -> Self {
        Self::new(FallbackContext::new(closes, latent, book), InstrumentId::new(), 1)
    }

    pub fn with_graph(mut self, graph: &'a GraphAt<'a>) -> Self {
        if let Some(r) = graph.cur_region(self.inst, 0) {
            self.cur0 = Some(r);
        }
        self.bar = graph.timescale();
        self.graph = Some(graph);
        self
    }

    /// Bar timescale the decoder `k` arguments are measured in (defaults to the graph view's
    /// timescale, else 1 s).
    pub fn with_bar(mut self, bar: Timescale) -> Self {
        self.bar = bar;
        self
    }

    pub fn with_decoders(mut self, decoders: DecodersAt<'a>) -> Self {
        self.decoders = Some(decoders);
        self
    }

    pub fn with_prices(mut self, prices: &'a dyn PriceLookup) -> Self {
        self.prices = Some(prices);
        self
    }

    pub fn fallback(&self) -> &FallbackContext {
        &self.fallback
    }

    /// Primitive calls so far by source.
    pub fn sources(&self) -> Sources {
        Sources { graph: self.graph_n.get(), decoder: self.decoder_n.get(), fallback: self.fallback_n.get() }
    }

    /// Primitive names by source so far.
    pub fn primitive_sources(&self) -> PrimitiveSources {
        PrimitiveSources { real: self.real_prims.borrow().clone(), fallback: self.fallback_prims.borrow().clone() }
    }

    /// `(analogs priced, analogs skipped for missing prices)` so far.
    pub fn analog_counts(&self) -> (u64, u64) {
        (self.analog_resolved.get(), self.analog_unresolved.get())
    }

    /// Drains the analogs that could not be priced in this context (deduplicated), so the
    /// caller can fetch their prices before the next step.
    pub fn take_analog_requests(&self) -> Vec<AnalogRequest> {
        let mut v = std::mem::take(&mut *self.analog_requests.borrow_mut());
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Drains the analogs that were priced in this context (deduplicated).
    pub fn take_analog_resolved(&self) -> Vec<AnalogRequest> {
        let mut v = std::mem::take(&mut *self.analog_done.borrow_mut());
        v.sort_unstable();
        v.dedup();
        v
    }

    fn count(&self, name: &'static str, s: Source) {
        match s {
            Source::Graph => self.graph_n.set(self.graph_n.get() + 1),
            Source::Decoder => self.decoder_n.set(self.decoder_n.get() + 1),
            Source::Fallback => self.fallback_n.set(self.fallback_n.get() + 1),
        }
        if s == Source::Fallback {
            self.fallback_prims.borrow_mut().insert(name);
        } else {
            self.real_prims.borrow_mut().insert(name);
        }
    }

    fn pick<T>(&self, name: &'static str, real: Option<T>, src: Source, fallback: impl FnOnce() -> T) -> T {
        match real {
            Some(v) => {
                self.count(name, src);
                v
            }
            None => {
                self.count(name, Source::Fallback);
                fallback()
            }
        }
    }

    fn graph_q<T>(&self, f: impl FnOnce(&GraphAt<'a>) -> Option<T>) -> Option<T> {
        self.graph.and_then(f)
    }

    fn dec_q<T>(&self, f: impl FnOnce(&DecodersAt<'a>) -> Option<T>) -> Option<T> {
        self.decoders.as_ref().and_then(f)
    }

    /// Resolves a region reference to `(expert, region id)` through the graph (falls back to
    /// the legacy expert-0 region).
    fn region(&self, r: RegionRef) -> Option<(u32, String)> {
        match r {
            RegionRef::Cur(e) => {
                let id = match self.graph {
                    Some(g) => g.cur_region(self.inst, e),
                    None if e == 0 => self.cur0.clone(),
                    None => None,
                }?;
                Some((e, id))
            }
        }
    }

    fn set(&self, s: SetRef) -> Option<(u32, Vec<String>)> {
        match s {
            SetRef::Neighbors(r, n) => {
                let (e, id) = self.region(r)?;
                Some((e, self.graph?.neighbors(e, &id, n)))
            }
        }
    }

    fn analog_outcome_at(&self, k: u32) -> Option<f64> {
        let (d, p) = (self.decoders.as_ref()?, self.prices?);
        let det = d.analog_outcome_detail(k, self.bar, p);
        self.analog_resolved.set(self.analog_resolved.get() + det.resolved.len() as u64);
        self.analog_unresolved.set(self.analog_unresolved.get() + det.unresolved.len() as u64);
        self.analog_requests.borrow_mut().extend(det.unresolved);
        self.analog_done.borrow_mut().extend(det.resolved);
        det.value
    }
}

impl StrategyContext for LiveContext<'_> {
    fn current_region(&self) -> Option<&str> {
        self.cur0.as_deref()
    }

    fn transition_prob(&self, to_region: &str) -> f64 {
        let real = self.graph_q(|g| {
            let cur = g.cur_region(self.inst, 0)?;
            g.p_trans(0, &cur, to_region, 1)
        });
        self.pick("transition_prob", real, Source::Graph, || self.fallback.transition_prob(to_region))
    }

    fn stay_prob(&self) -> f64 {
        let real = self.graph_q(|g| {
            let cur = g.cur_region(self.inst, 0)?;
            g.p_trans(0, &cur, &cur, 1)
        });
        self.pick("stay_prob", real, Source::Graph, || self.fallback.stay_prob())
    }

    fn path_up_prob(&self, k: u32) -> f64 {
        let real = self.graph_q(|g| {
            let cur = g.cur_region(self.inst, 0)?;
            g.path_value(0, &cur, k.max(1), WALKS, None)
        });
        self.pick("path_up_prob", real, Source::Graph, || self.fallback.path_up_prob(k))
    }

    fn region_entropy(&self) -> f64 {
        let real = self.graph_q(|g| {
            let cur = g.cur_region(self.inst, 0)?;
            g.row_entropy(0, &cur)
        });
        self.pick("region_entropy", real, Source::Graph, || self.fallback.region_entropy())
    }

    fn cur_region(&self, expert: u32) -> Option<String> {
        let real = self.graph_q(|g| g.cur_region(self.inst, expert));
        self.pick("cur_region", real.map(Some), Source::Graph, || if expert == 0 { self.cur0.clone() } else { None })
    }

    fn neighbors(&self, r: RegionRef, n: u32) -> Vec<String> {
        let real = self.set(SetRef::Neighbors(r, n)).map(|x| x.1);
        self.pick("neighbors", real, Source::Graph, Vec::new)
    }

    fn p_trans(&self, a: RegionRef, b: RegionRef, k: u32) -> f64 {
        let real = (|| {
            let g = self.graph?;
            let (e, from) = self.region(a)?;
            let (_, to) = self.region(b)?;
            g.p_trans(e, &from, &to, k.max(1))
        })();
        self.pick("p_trans", real, Source::Graph, || {
            if a == b { self.fallback.stay_prob().powi(k.max(1) as i32) } else { 0.0 }
        })
    }

    fn p_to_set(&self, r: RegionRef, s: SetRef, k: u32) -> f64 {
        let real = (|| {
            let g = self.graph?;
            let (e, from) = self.region(r)?;
            let (_, set) = self.set(s)?;
            g.p_to_set(e, &from, &set, k.max(1))
        })();
        self.pick("p_to_set", real, Source::Graph, || 0.0)
    }

    fn hitting_time(&self, r: RegionRef, s: SetRef) -> f64 {
        let real = (|| {
            let g = self.graph?;
            let (e, from) = self.region(r)?;
            let (_, set) = self.set(s)?;
            g.hitting_time(e, &from, &set)
        })();
        self.pick("hitting_time", real, Source::Graph, || f64::from(HITTING_HORIZON))
    }

    fn dwell(&self, r: RegionRef) -> f64 {
        let RegionRef::Cur(e) = r;
        let real = self.graph_q(|g| g.dwell(self.inst, e));
        self.pick("dwell", real, Source::Graph, || self.fallback.latent.as_ref().map_or(0.0, |l| f64::from(l.dwell)))
    }

    fn novelty(&self, expert: u32) -> f64 {
        let real = self.graph_q(|g| g.novelty(self.inst, expert));
        self.pick("novelty", real, Source::Graph, || self.fallback.region_entropy())
    }

    fn region_feature(&self, r: RegionRef, f: RegionFeature) -> f64 {
        let real = (|| {
            let g = self.graph?;
            let (e, id) = self.region(r)?;
            g.region_feature(e, &id, f)
        })();
        self.pick("region_feature", real, Source::Graph, || match f {
            RegionFeature::Mass => 0.0,
            RegionFeature::MeanFwdReturn => self.fallback.mean_return(),
            RegionFeature::Vol => self.fallback.std_return(),
        })
    }

    fn path_value(&self, r: RegionRef, k: u32, n: u32, f: PathFn) -> f64 {
        let real = (|| {
            let g = self.graph?;
            let (e, from) = self.region(r)?;
            let feat = match f {
                PathFn::Up => None,
                PathFn::Feature(x) => Some(x),
            };
            g.path_value(e, &from, k.max(1), n.max(1), feat)
        })();
        self.pick("path_value", real, Source::Graph, || match f {
            PathFn::Up => self.fallback.path_up_prob(k),
            PathFn::Feature(RegionFeature::Mass) => 0.0,
            PathFn::Feature(RegionFeature::MeanFwdReturn) => self.fallback.mean_return(),
            PathFn::Feature(RegionFeature::Vol) => self.fallback.std_return(),
        })
    }

    fn forecast_quantile(&self, level: f64) -> f64 {
        let real = self.dec_q(|d| d.fc_quantile(level, self.steps));
        self.pick("forecast_quantile", real, Source::Decoder, || self.fallback.forecast_quantile(level))
    }

    fn analog_forecast(&self) -> f64 {
        let real = self.analog_outcome_at(self.steps).or_else(|| self.dec_q(|d| d.fc_dist(self.steps)));
        self.pick("analog_forecast", real, Source::Decoder, || self.fallback.analog_forecast())
    }

    fn decoder_score(&self, family: &str) -> f64 {
        let real = self.dec_q(|d| d.score(family));
        self.pick("decoder_score", real, Source::Decoder, || 0.0)
    }

    fn fc_quantile(&self, q: f64, k: u32) -> f64 {
        let real = self.dec_q(|d| d.fc_quantile(q, k));
        self.pick("fc_quantile", real, Source::Decoder, || self.fallback.forecast_quantile(q))
    }

    fn fc_dist(&self, k: u32) -> f64 {
        let real = self.dec_q(|d| d.fc_dist(k));
        self.pick("fc_dist", real, Source::Decoder, || self.fallback.analog_forecast())
    }

    fn analog_outcome(&self, k: u32) -> f64 {
        let real = self.analog_outcome_at(k);
        self.pick("analog_outcome", real, Source::Decoder, || self.fallback.analog_forecast())
    }

    fn analog_age(&self) -> f64 {
        let real = self.dec_q(|d| d.analog_age());
        self.pick("analog_age", real, Source::Decoder, || 0.0)
    }

    fn text_score(&self, topic: &str) -> f64 {
        let real = self.dec_q(|d| d.text_score(topic));
        self.pick("text_score", real, Source::Decoder, || 0.0)
    }

    fn visual_score(&self, kind: &str) -> f64 {
        let real = self.dec_q(|d| d.visual_score(kind));
        self.pick("visual_score", real, Source::Decoder, || 0.0)
    }

    fn position(&self) -> f64 {
        self.fallback.position()
    }

    fn unrealised_pnl(&self) -> f64 {
        self.fallback.unrealised_pnl()
    }

    fn time_in_trade(&self) -> f64 {
        self.fallback.time_in_trade()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resamples_into_bars_with_carry_forward() {
        let mut s = PriceSeries::new(100);
        s.push(0, 100.0);
        s.push(500, 101.0); // same 1s bucket → close 101
        s.push(3_000, 103.0); // buckets 1 and 2 empty → carried forward
        assert_eq!(s.closes(Timescale::Second, 10), vec![101.0, 101.0, 101.0, 103.0]);
        assert_eq!(s.closes(Timescale::Second, 2), vec![101.0, 103.0]);
        assert_eq!(s.closes(Timescale::Minute, 10), vec![103.0]);
        s.push(2_000, 50.0); // late tick dropped
        assert_eq!(s.last(), Some(103.0));
    }

    #[test]
    fn fallback_statistics() {
        let closes = [100.0, 101.0, 102.0, 101.0, 103.0];
        let c = FallbackContext::new(&closes, None, VirtualBook::default());
        assert_eq!(c.n_returns(), 4);
        assert!(c.analog_forecast() > 0.0);
        assert_eq!(c.path_up_prob(1), 0.75);
        assert!((c.forecast_quantile(0.5) - c.mean_return()).abs() < 1e-9);
        assert!(c.forecast_quantile(0.9) > c.forecast_quantile(0.1));
        assert!(c.region_entropy() > 0.0 && c.region_entropy() <= 1.0);
        assert_eq!(c.stay_prob(), 1.0 / 3.0);
        assert_eq!(c.current_region(), None);

        let empty = FallbackContext::new(&[100.0], None, VirtualBook::default());
        assert_eq!(empty.analog_forecast(), 0.0);
        assert_eq!(empty.path_up_prob(3), 0.5);
        assert_eq!(empty.region_entropy(), 1.0);

        let l = LatentInfo { region_id: "r7".into(), uncertainty: 0.3, dwell: 3 };
        let with =
            FallbackContext::new(&closes, Some(l), VirtualBook { exposure: -0.4, unrealised: 0.01, steps_in_trade: 2 });
        assert_eq!(with.current_region(), Some("r7"));
        assert_eq!(with.stay_prob(), 0.75);
        assert_eq!(with.region_entropy(), 0.3);
        assert_eq!(with.position(), -0.4);
        assert_eq!(with.time_in_trade(), 2.0);
    }

    #[test]
    fn live_context_prefers_graph_and_decoders_and_counts_sources() {
        use crate::{Strategy, decoders::DecoderOutputs, population::StrategyFile, sexpr};
        let inst = InstrumentId::new();
        let closes: Vec<f64> = (0..40).map(|i| 100.0 + (i as f64 * 0.7).sin() + i as f64 * 0.05).collect();
        let book = VirtualBook::default();

        // price only → everything is fallback
        let fb = LiveContext::price_only(&closes, None, book);
        let e = sexpr::parse("(p_to_set (cur_region 0) (neighbors (cur_region 0) 3) 1)").unwrap();
        assert_eq!(e.eval(&fb), 0.0);
        assert_eq!(fb.sources(), Sources { graph: 0, decoder: 0, fallback: 1 });

        // synthetic graph (toy chain, asset in region a) + a forecast output
        let g = crate::graph::tests::toy(inst);
        let view = g.at(Timescale::Second, 10_000);
        let mut d = DecoderOutputs::new();
        d.on_output(&crate::decoders::tests::forecast(Some(inst), 9_000, &[-0.02, -0.01, 0.0, 0.01, 0.02]), 9_000);
        // text / visual decoders on the model-runtime contract: instrument-specific news, market-wide chart
        d.on_output(&crate::decoders::tests::scored("text.news", Some(inst), 9_500, 0.5), 9_500);
        d.on_output(&crate::decoders::tests::scored("visual.future", None, 9_000, -0.25), 9_000);
        let prices = PriceSeries::new(100);
        let ctx = LiveContext::new(FallbackContext::new(&closes, None, book), inst, 5)
            .with_graph(&view)
            .with_decoders(d.at(inst, 10_000, 3_600_000))
            .with_prices(&prices);
        assert_eq!(ctx.current_region(), Some("a"));
        let p = e.eval(&ctx);
        assert!((p - 0.9).abs() < 1e-3, "mass to b+c: {p}");
        let stay = sexpr::parse("stay_prob").unwrap().eval(&ctx);
        assert!(stay < 0.01, "a never stays: {stay}");
        assert!((sexpr::parse("(p_trans (cur_region 0) (cur_region 0) 2)").unwrap().eval(&ctx) - 0.72).abs() < 5e-3);
        assert_eq!(sexpr::parse("(dwell (cur_region 0))").unwrap().eval(&ctx), 4.0);
        assert!((sexpr::parse("(novelty 0)").unwrap().eval(&ctx) - 0.25).abs() < 1e-9);
        assert_eq!(sexpr::parse("(quantile 0.5)").unwrap().eval(&ctx), 0.0);
        assert_eq!(sexpr::parse("(fc_quantile 0.75 5)").unwrap().eval(&ctx), 0.01);
        assert!(sexpr::parse("(score rnn_prior)").unwrap().eval(&ctx) > 0.99);
        assert_eq!(sexpr::parse("(text_score news)").unwrap().eval(&ctx), 0.5);
        assert_eq!(sexpr::parse("(mul (text_score news) (visual_score future))").unwrap().eval(&ctx), -0.125);
        assert!(sexpr::parse("(score text.news)").unwrap().eval(&ctx) > 0.99, "dotted family in `score`");
        assert_eq!(sexpr::parse("(text_score narrative)").unwrap().eval(&ctx), 0.0, "no producer → fallback 0");
        // no region has return statistics yet → path_up falls back to prices
        let _ = sexpr::parse("(path_up 3)").unwrap().eval(&ctx);
        let s = ctx.sources();
        assert_eq!(s.graph, 5, "{s:?}");
        assert_eq!(s.decoder, 7, "{s:?}");
        assert_eq!(s.fallback, 2, "{s:?}");
        let ps = ctx.primitive_sources();
        assert!(ps.real.contains("text_score") && ps.real.contains("visual_score") && ps.real.contains("p_to_set"));
        assert_eq!(ps.fallback_only(), ["path_up_prob"].into_iter().collect::<BTreeSet<_>>(), "{ps:?}");
        assert!(ps.fallback.contains("text_score"), "`narrative` fell back although `news` was real");
        assert_eq!(ctx.analog_counts(), (0, 0));
        assert!(ctx.take_analog_requests().is_empty());

        // every baseline evaluates on the synthetic graph with bounded output
        for def in StrategyFile::baselines().strategy {
            let mut st = def.build().unwrap();
            let dist = st.step(10_000, &[(inst, &ctx)]);
            let x = dist.exposures[0];
            assert!((-1.0..=1.0).contains(&x.mean), "{}: {}", def.id, x.mean);
            assert!((0.0..=1.0).contains(&x.stddev), "{}: {}", def.id, x.stddev);
        }
        assert!(ctx.sources().graph > s.graph, "graph baselines hit the graph");
        assert!(ctx.sources().decoder > s.decoder, "text/visual baselines hit the decoder cache");
        // the text/visual baselines answer from the decoders alone once every family they read is published
        let mut d2 = DecoderOutputs::new();
        for (family, inst_ref, score) in [
            ("text.news", Some(inst), 0.5),
            ("visual.future", None, -0.25),
            ("text.narrative", Some(inst), 0.3),
            ("text.summary", None, 0.2),
            ("visual.chart", Some(inst), 0.6),
        ] {
            d2.on_output(&crate::decoders::tests::scored(family, inst_ref, 9_800, score), 9_800);
        }
        for id in ["baseline-text-visual-1s", "baseline-narrative-chart-1m"] {
            let def = StrategyFile::baselines().strategy.into_iter().find(|d| d.id == id).unwrap();
            let probe = LiveContext::new(FallbackContext::new(&closes, None, book), inst, 5)
                .with_decoders(d2.at(inst, 10_000, 3_600_000));
            let mut st = def.build().unwrap();
            let before = probe.sources();
            let x = st.step(10_000, &[(inst, &probe)]).exposures[0];
            assert!(x.mean.abs() > 0.0 && x.mean.abs() <= 1.0, "{id}: {}", x.mean);
            let after = probe.sources();
            assert!(after.decoder > before.decoder && after.fallback == 0, "{id}: {after:?}");
        }
    }

    #[test]
    fn analog_outcome_reports_unpriced_analogs_for_prefetch() {
        use crate::{decoders::DecoderOutputs, sexpr};
        let inst = InstrumentId::new();
        let mut d = DecoderOutputs::new();
        d.on_output(
            &crate::decoders::tests::retrieval(inst, 30_000, &[(10_000, 1.0, 2.0), (20_000, 3.0, 4.0)]),
            30_000,
        );
        // in-memory history starts at 15 s: the 10 s analog cannot be priced
        let mut prices = PriceSeries::new(100);
        for t in 15..=30 {
            prices.push(t * 1_000, 100.0 + t as f64);
        }
        let closes = prices.closes(Timescale::Second, 20);
        let ctx = LiveContext::new(FallbackContext::new(&closes, None, VirtualBook::default()), inst, 5)
            .with_decoders(d.at(inst, 30_000, 3_600_000))
            .with_prices(&prices);
        let v = sexpr::parse("(analog_outcome 5)").unwrap().eval(&ctx);
        assert!((v - (125.0f64 / 120.0).ln()).abs() < 1e-12, "{v}");
        assert_eq!(ctx.analog_counts(), (1, 1));
        assert_eq!(ctx.sources(), Sources { graph: 0, decoder: 1, fallback: 0 });
        let reqs = ctx.take_analog_requests();
        assert_eq!(reqs, vec![AnalogRequest { window_end_ms: 10_000, k: 5, bar: Timescale::Second }]);
        assert!(ctx.take_analog_requests().is_empty(), "drained");
        // a layered lookup that also knows the fetched prices resolves everything
        let fetched =
            |at: i64| -> Option<f64> { prices.price_at(at).or_else(|| (at >= 0).then(|| 100.0 + at as f64 / 1000.0)) };
        let ctx2 = LiveContext::new(FallbackContext::new(&closes, None, VirtualBook::default()), inst, 5)
            .with_decoders(d.at(inst, 30_000, 3_600_000))
            .with_prices(&fetched);
        sexpr::parse("(analog_outcome 5)").unwrap().eval(&ctx2);
        assert_eq!(ctx2.analog_counts(), (2, 0));
        assert_eq!(ctx2.take_analog_resolved().len(), 2);
    }

    #[test]
    fn price_at_is_last_price_at_or_before() {
        let mut s = PriceSeries::new(10);
        s.push(1_000, 1.0);
        s.push(3_000, 3.0);
        assert_eq!(s.price_at(500), None);
        assert_eq!(s.price_at(1_000), Some(1.0));
        assert_eq!(s.price_at(2_999), Some(1.0));
        assert_eq!(s.price_at(9_000), Some(3.0));
        assert_eq!(s.first_ms(), Some(1_000));
    }

    #[test]
    fn normal_quantile_reference_values() {
        assert!((norm_quantile(0.5)).abs() < 1e-9);
        assert!((norm_quantile(0.975) - 1.959964).abs() < 1e-5);
        assert!((norm_quantile(0.01) + 2.326348).abs() < 1e-5);
    }
}
