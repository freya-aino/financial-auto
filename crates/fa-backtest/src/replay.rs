//! Pipeline replay for strategies (spec 41 §4, §6.1–6.4; spec 35 §6.1–6.2).
//!
//! A [`ReplaySpec`] names the data window, purpose, instruments, timescale, cost model and
//! seed; [`Evaluator::evaluate`] drives one `fa_strategies` program step by step over
//! [`BarSeries`] in **knowledge-time order** (a bar is visible at its close), with a virtual
//! book per instrument and execution simulated through the [`CostModel`](crate::cost):
//!
//! 1. every event (bar close) passes through [`crate::run_replay`], which rejects any
//!    out-of-order source (look-ahead guard);
//! 2. at every step of the strategy's timescale the open positions are marked to the new
//!    closes (equity point, per-step return net of carry), then the program decides on the
//!    instruments that have `lookback_bars + 1` closes through the same
//!    `features::LiveContext` the live runtime uses (spec 35 §4.2.1): a `graph::LatentGraph` and
//!    `decoders::DecoderOutputs` rebuilt point-in-time from recorded `LATENT` messages
//!    ([`crate::latents`]: only records published `≤ t` are ingested, views filter `as_of ≤ t`),
//!    the price-only fallback where the recording has nothing, counted per source in
//!    `sources.graph|decoder|fallback`; plus its own virtual book;
//! 3. the target exposure `mean ∈ [-1, 1]` of each sleeve (`capital / n_instruments`) is
//!    filled at the close through [`CostModel::price`] (half spread + square-root impact on the
//!    fill price, commission with a floor; σ_day from the lookback's per-bar return std), and
//!    held positions pay funding / borrow carry per step.
//!
//! The report carries the objective set of spec 35 §6.2 as implemented today:
//!
//! | key | definition | direction |
//! |---|---|---|
//! | `deflated_sharpe` | PSR of the per-step net returns against the expected max Sharpe of `n_trials` trials (`trial_sharpe_var`); with 1 trial = PSR vs 0 | max |
//! | `sharpe` | per-step Sharpe of net returns | max |
//! | `max_drawdown` | max peak-to-trough of equity / capital | min |
//! | `turnover` | Σ traded notional / capital / steps | min |
//! | `cost_sensitivity` | `sharpe − sharpe(returns under 2× costs)` (spec 35: Sharpe degradation under 2× costs) | min |
//! | `compute_cost` | log10 of program nodes evaluated per step | min |
//! | `calibration` | corr(emitted `std`, next |return|) (informational) | — |
//! | `cost_share` | Σ costs / Σ |gross P&L| (informational) | — |
//! | `n_trades`, `n_obs` | counts (informational) | — |
//!
//! Deterministic: the programs and the cost model are pure, events are totally ordered by
//! `(close time, instrument index)`, recorded latents by `(publish time, sequence)` and the
//! graph's sampled walks are seeded per step, so identical spec + seed + data + recording give
//! an identical report (CI test). The seed is recorded and reserved for stochastic components
//! (none today). A replay declares its encoder snapshot (`ReplaySpec::snapshot`); latents of
//! another snapshot are refused unless the purpose is `Diagnostic` ([`crate::latents::check_snapshot`]).

use std::{collections::BTreeMap, time::Instant};

use fa_core::{InstrumentId, Timescale};
use fa_strategies::{
    GpStrategy, Strategy, StrategyContext,
    decoders::PriceLookup,
    features::{FallbackContext, LiveContext, VirtualBook},
    population::StrategyDef,
};
use serde::{Deserialize, Serialize};

use crate::{
    ReplayError, ReplaySink, ReplaySource,
    cost::{ClassCosts, CostInputs, CostModel},
    eval::Stop,
    latents::{FeedCounters, LatentFeed, RecordedLatents, SnapshotCheck, SourceCounts, check_snapshot},
    run_replay, stats,
};

/// Why a replay runs (spec 41 §6.4): only `Promotion` may touch the locked holdout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReplayPurpose {
    #[default]
    Selection,
    Promotion,
    Diagnostic,
}

/// Replay request (spec 41 §4 `ReplaySpec`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplaySpec {
    /// `[start_ms, end_ms]` of bar close times.
    pub window_ms: (i64, i64),
    pub purpose: ReplayPurpose,
    /// Instrument tokens to trade; empty = every series in the data set.
    pub instruments: Vec<String>,
    pub timescale: Timescale,
    /// Encoder snapshot the candidate was evolved on. Recorded latents of another snapshot are
    /// refused unless `purpose == Diagnostic`; empty adopts the recording's snapshot.
    pub snapshot: String,
    pub cost_model: String,
    pub seed: u64,
    /// Strategy capital in quote currency (returns are relative to it).
    pub capital: f64,
    /// Trials in the generation so far (deflated Sharpe); `1` = plain PSR.
    pub n_trials: usize,
    /// Variance of the trials' Sharpe ratios (deflated Sharpe).
    pub trial_sharpe_var: f64,
    /// Smallest rebalance executed, as a fraction of the sleeve's capital (smaller target
    /// changes are carried to the next step); avoids micro-churn on constant-fraction targets.
    pub min_trade_fraction: f64,
}

impl ReplaySpec {
    /// Selection replay over `[start_ms, end_ms]` at `timescale`, default cost model, capital 1.
    pub fn selection(timescale: Timescale, start_ms: i64, end_ms: i64) -> Self {
        Self {
            window_ms: (start_ms, end_ms),
            purpose: ReplayPurpose::Selection,
            instruments: Vec::new(),
            timescale,
            snapshot: String::new(),
            cost_model: "default".into(),
            seed: 0,
            capital: 1.0,
            n_trials: 1,
            trial_sharpe_var: 0.0,
            min_trade_fraction: 0.02,
        }
    }
}

/// A locked holdout period (`validation.holdout`, spec 41 §6.4).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Holdout {
    pub timescale: Timescale,
    pub start_ms: i64,
    pub end_ms: i64,
    pub locked_at_ms: i64,
}

/// Rejects a `Selection` replay whose window intersects a locked holdout of the same timescale.
pub fn check_holdout(spec: &ReplaySpec, holdouts: &[Holdout]) -> Result<(), ReplayError> {
    if spec.purpose != ReplayPurpose::Selection {
        return Ok(());
    }
    let (s, e) = spec.window_ms;
    if let Some(h) = holdouts.iter().find(|h| h.timescale == spec.timescale && s <= h.end_ms && e >= h.start_ms) {
        return Err(ReplayError::HoldoutIntersect {
            timescale: h.timescale.to_string(),
            holdout_start_ms: h.start_ms,
            holdout_end_ms: h.end_ms,
        });
    }
    Ok(())
}

/// Bars of one instrument, visible at close (knowledge time = `ts_ms`, ascending).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct BarSeries {
    pub instrument_id: String,
    pub venue: String,
    /// Close times (ms), strictly ascending.
    pub ts_ms: Vec<i64>,
    pub close: Vec<f64>,
}

impl BarSeries {
    /// Resamples `(ts_ms, price)` points into bars of `step_ms` (last price per bucket) stamped
    /// at the bucket close; non-positive prices and out-of-order points are dropped.
    pub fn from_points(instrument_id: &str, venue: &str, points: &[(i64, f64)], step_ms: i64) -> Self {
        let step_ms = step_ms.max(1);
        let mut s = Self { instrument_id: instrument_id.into(), venue: venue.into(), ts_ms: vec![], close: vec![] };
        let mut cur: Option<i64> = None;
        for &(t, p) in points {
            if p.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || !p.is_finite() {
                continue;
            }
            let b = t.div_euclid(step_ms);
            match cur {
                Some(c) if b == c => *s.close.last_mut().expect("non-empty") = p,
                Some(c) if b < c => {}
                _ => {
                    cur = Some(b);
                    s.ts_ms.push((b + 1) * step_ms);
                    s.close.push(p);
                }
            }
        }
        s
    }

    pub fn len(&self) -> usize {
        self.close.len()
    }

    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
}

/// One simulated fill.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trade {
    pub ts_ms: i64,
    pub instrument_id: String,
    /// Signed change in units (+ buy, − sell).
    pub delta_units: f64,
    pub price: f64,
    /// Absolute traded notional.
    pub notional: f64,
    pub fill_price: f64,
    pub fee: f64,
    pub slippage_bps: f64,
    /// Total cost of the fill: `|Δunits| · (half_spread + impact) + fee`.
    pub cost: f64,
    /// Exposure the strategy asked for (mean of its distribution).
    pub target_exposure: f64,
}

/// Ranked objective directions of the strategy report (spec 35 §6.2).
pub const DIRECTIONS: [(&str, &str); 6] = [
    ("deflated_sharpe", "max"),
    ("sharpe", "max"),
    ("max_drawdown", "min"),
    ("turnover", "min"),
    ("cost_sensitivity", "min"),
    ("compute_cost", "min"),
];

/// Result of replaying one strategy (spec 41 §4 `EvalReport`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    pub strategy_id: String,
    pub genome_id: String,
    pub timescale: Timescale,
    pub purpose: ReplayPurpose,
    pub cost_model: String,
    pub seed: u64,
    pub capital: f64,
    pub instruments: Vec<String>,
    /// Decision steps run.
    pub n_steps: usize,
    /// Return observations (steps with a mark-to-market).
    pub n_obs: usize,
    pub first_ts_ms: i64,
    pub last_ts_ms: i64,
    pub stop: Stop,
    pub sharpe: f64,
    pub psr: f64,
    pub deflated_sharpe: f64,
    pub n_trials: usize,
    pub max_drawdown: f64,
    pub turnover: f64,
    pub cost_share: f64,
    pub cost_sensitivity: f64,
    pub calibration: f64,
    pub compute_cost: f64,
    pub n_trades: usize,
    pub gross_pnl: f64,
    pub total_cost: f64,
    pub net_pnl: f64,
    pub final_equity: f64,
    /// Wall-clock ms (reported, never ranked).
    pub elapsed_ms: f64,
    /// `(ts_ms, equity)` after every step.
    pub equity: Vec<(i64, f64)>,
    /// Per-step net returns relative to `capital` (trial registry / PBO input).
    pub returns: Vec<f64>,
    /// Per-trade log (capped at [`MAX_TRADE_LOG`] entries; `n_trades` counts all).
    pub trades: Vec<Trade>,
    /// Primitive calls answered by the graph, the decoder cache or the price fallback (D-053).
    #[serde(default)]
    pub sources: SourceCounts,
    /// What the recorded latents contributed.
    #[serde(default)]
    pub latents: ReplayLatents,
}

/// Recorded-latent bookkeeping of a report.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayLatents {
    /// A recording was supplied (even if it covered none of the window).
    pub available: bool,
    /// Records in the recording.
    pub recorded: usize,
    /// Records ingested by the last step (published at or before it).
    pub fed: FeedCounters,
    pub snapshot: SnapshotCheck,
    /// Transition rows of the graph at the end of the replay.
    pub graph_rows: usize,
    /// Decoder outputs cached at the end of the replay.
    pub decoder_outputs: usize,
}

/// Trade-log cap per report.
pub const MAX_TRADE_LOG: usize = 20_000;

impl EvalReport {
    /// Objectives as `name → value`, all finite (NaN → neutral 0).
    pub fn objectives(&self) -> Vec<(&'static str, f64)> {
        let f = |x: f64| if x.is_finite() { x } else { 0.0 };
        vec![
            ("deflated_sharpe", f(self.deflated_sharpe)),
            ("sharpe", f(self.sharpe)),
            ("max_drawdown", f(self.max_drawdown)),
            ("turnover", f(self.turnover)),
            ("cost_sensitivity", f(self.cost_sensitivity)),
            ("compute_cost", f(self.compute_cost)),
            ("calibration", f(self.calibration)),
            ("cost_share", f(self.cost_share)),
            ("psr", f(self.psr)),
            ("n_trades", self.n_trades as f64),
            ("n_obs", self.n_obs as f64),
            ("sources.graph", self.sources.graph as f64),
            ("sources.decoder", self.sources.decoder as f64),
            ("sources.fallback", self.sources.fallback as f64),
        ]
    }

    /// Provisional scalar for the orchestrator's fallback ranking: the deflated Sharpe.
    pub fn score(&self) -> f64 {
        if self.deflated_sharpe.is_finite() { self.deflated_sharpe } else { f64::NEG_INFINITY }
    }
}

/// Replay controls (budget / preemption hooks), as `eval::EvalSpec`.
#[derive(Default)]
pub struct ReplayControls {
    pub max_steps: Option<usize>,
    pub should_stop: Option<Box<dyn Fn(usize) -> bool + Send + Sync>>,
    pub on_progress: Option<Box<dyn Fn(usize, usize) + Send + Sync>>,
    pub progress_every: usize,
}

/// Something that can score a candidate on a replay (spec 41 §4 `Evaluator`).
pub trait Evaluator {
    fn evaluate(&self, spec: &ReplaySpec, candidate: &StrategyDef) -> Result<EvalReport, ReplayError>;
}

/// The in-process pipeline replay over loaded bars.
pub struct PipelineReplay<'a> {
    pub data: &'a [BarSeries],
    pub cost: &'a CostModel,
    pub holdouts: &'a [Holdout],
    pub controls: ReplayControls,
    /// Recorded `LATENT` traffic covering the window; `None` = price-only fallback throughout.
    pub latents: Option<&'a RecordedLatents>,
}

impl Evaluator for PipelineReplay<'_> {
    fn evaluate(&self, spec: &ReplaySpec, candidate: &StrategyDef) -> Result<EvalReport, ReplayError> {
        check_holdout(spec, self.holdouts)?;
        let strategy = candidate.build().map_err(|e| ReplayError::Candidate(e.to_string()))?;
        replay(&strategy, candidate.lookback_bars, self.data, spec, self.cost, &self.controls, self.latents)
    }
}

// ───────────────────────────────────────────── engine ─────────────────────────────────────────────

struct Event {
    inst: usize,
    close: f64,
}

struct VecSource {
    events: std::vec::IntoIter<(i64, Event)>,
}

impl ReplaySource for VecSource {
    type Event = Event;
    fn next_event(&mut self) -> Option<(i64, Event)> {
        self.events.next()
    }
}

/// One instrument's sleeve: history, position and the virtual book the program reads.
struct Sleeve {
    id: InstrumentId,
    token: String,
    venue: String,
    class: String,
    schedule: ClassCosts,
    schedule_2x: ClassCosts,
    closes: Vec<f64>,
    /// Close times of `closes` (for the analog price lookup).
    close_ts: Vec<i64>,
    units: f64,
    /// Volume-weighted entry price of the open position (for `unrealised_pnl`).
    entry_price: f64,
    last_price: f64,
    book: VirtualBook,
    pending_decision: bool,
}

/// Cash + mark-to-market accounting: `equity = cash + Σ units · last_price`; every cost is
/// taken from cash when paid, so `Δequity` per step is the net return and `Δequity + costs`
/// the gross one.
/// Last close at or before a time (the `PriceLookup` behind `analog_outcome`).
struct SleevePrices<'a> {
    ts: &'a [i64],
    px: &'a [f64],
}

impl PriceLookup for SleevePrices<'_> {
    fn price_at(&self, at_ms: i64) -> Option<f64> {
        let n = self.ts.partition_point(|t| *t <= at_ms);
        (n > 0).then(|| self.px[n - 1])
    }
}

struct Engine<'a> {
    strategy: GpStrategy,
    lookback: usize,
    /// Strategy horizon in bars (the `k` of `analog_forecast`).
    steps: u32,
    /// Recorded latents replayed point-in-time; `None` = fallback only.
    feed: Option<LatentFeed<'a>>,
    sources: SourceCounts,
    spec: &'a ReplaySpec,
    controls: &'a ReplayControls,
    cost: &'a CostModel,
    cost_2x: CostModel,
    /// `sqrt(bars per day)`: per-bar return std → daily σ for the impact model.
    sigma_scale: f64,
    sleeves: Vec<Sleeve>,
    sleeve_capital: f64,
    step_days: f64,
    cur_ts: Option<i64>,
    cash: f64,
    equity: f64,
    equity_curve: Vec<(i64, f64)>,
    returns: Vec<f64>,
    returns_2x: Vec<f64>,
    /// Costs paid since the last mark (actual and under the 2× model).
    step_cost: f64,
    step_cost_2x: f64,
    /// Mean `std` emitted at the last decision, consumed by the next mark (calibration).
    pending_std: Option<f64>,
    /// Interleaved `(std, |next return|)` pairs.
    calib_pairs: Vec<f64>,
    gross_pnl: f64,
    total_cost: f64,
    n_trades: usize,
    n_steps: usize,
    trades: Vec<Trade>,
    stop: Stop,
    first_ts: Option<i64>,
    last_ts: i64,
    halted: bool,
    peak: f64,
    max_dd: f64,
    turnover_notional: f64,
    total_steps_hint: usize,
}

impl<'a> Engine<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        strategy: GpStrategy,
        lookback: usize,
        spec: &'a ReplaySpec,
        controls: &'a ReplayControls,
        cost: &'a CostModel,
        sleeves: Vec<Sleeve>,
        total_steps_hint: usize,
        feed: Option<LatentFeed<'a>>,
    ) -> Self {
        let n = sleeves.len().max(1) as f64;
        Self {
            steps: strategy.horizon.steps.max(1),
            strategy,
            lookback: lookback.max(2),
            feed,
            sources: SourceCounts::default(),
            spec,
            controls,
            cost,
            cost_2x: cost.scaled(2.0),
            sigma_scale: (86_400.0 / spec.timescale.duration().as_secs_f64()).sqrt(),
            sleeves,
            sleeve_capital: spec.capital / n,
            step_days: spec.timescale.duration().as_secs_f64() / 86_400.0,
            cur_ts: None,
            cash: spec.capital,
            equity: spec.capital,
            equity_curve: Vec::new(),
            returns: Vec::new(),
            returns_2x: Vec::new(),
            step_cost: 0.0,
            step_cost_2x: 0.0,
            pending_std: None,
            calib_pairs: Vec::new(),
            gross_pnl: 0.0,
            total_cost: 0.0,
            n_trades: 0,
            n_steps: 0,
            trades: Vec::new(),
            stop: Stop::Complete,
            first_ts: None,
            last_ts: 0,
            halted: false,
            peak: spec.capital,
            max_dd: 0.0,
            turnover_notional: 0.0,
            total_steps_hint,
        }
    }

    fn pay(&mut self, cost: f64, cost_2x: f64) {
        self.cash -= cost;
        self.total_cost += cost;
        self.step_cost += cost;
        self.step_cost_2x += cost_2x;
    }

    /// Carry, mark to the latest closes, record the equity point and the step returns.
    fn mark(&mut self, ts: i64) {
        let before = self.equity;
        let mut carry = 0.0;
        let mut carry_2x = 0.0;
        for s in &self.sleeves {
            if s.units != 0.0 {
                let notional = s.units * s.last_price;
                carry += s.schedule.carry(notional, self.step_days);
                carry_2x += s.schedule_2x.carry(notional, self.step_days);
            }
        }
        self.pay(carry, carry_2x);
        let mtm: f64 = self.sleeves.iter().map(|s| s.units * s.last_price).sum();
        self.equity = self.cash + mtm;
        let net = self.equity - before;
        self.gross_pnl += net + self.step_cost;
        let r = net / self.spec.capital;
        self.returns.push(r);
        self.returns_2x.push(r - (self.step_cost_2x - self.step_cost) / self.spec.capital);
        self.step_cost = 0.0;
        self.step_cost_2x = 0.0;
        self.equity_curve.push((ts, self.equity));
        self.peak = self.peak.max(self.equity);
        self.max_dd = self.max_dd.max((self.peak - self.equity) / self.spec.capital);
        if let Some(sd) = self.pending_std.take() {
            self.calib_pairs.push(sd);
            self.calib_pairs.push(r.abs());
        }
        for s in &mut self.sleeves {
            s.book.unrealised =
                if s.units != 0.0 { s.units * (s.last_price - s.entry_price) / self.sleeve_capital } else { 0.0 };
        }
    }

    /// Decision + fills for every sleeve with enough history at `ts`.
    fn decide(&mut self, ts: i64) {
        let ready: Vec<usize> = (0..self.sleeves.len())
            .filter(|&i| self.sleeves[i].pending_decision && self.sleeves[i].closes.len() > self.lookback)
            .collect();
        for s in &mut self.sleeves {
            s.pending_decision = false;
        }
        if ready.is_empty() {
            return;
        }
        if self.controls.max_steps.is_some_and(|m| self.n_steps >= m) {
            self.stop = Stop::MaxSteps;
            self.halted = true;
            return;
        }
        if self.controls.should_stop.as_ref().is_some_and(|f| f(self.n_steps)) {
            self.stop = Stop::Interrupted;
            self.halted = true;
            return;
        }
        // Knowledge time: only recordings published at or before this step are visible.
        if let Some(f) = &mut self.feed {
            f.advance(ts);
        }
        let (lookback, steps, timescale) = (self.lookback, self.steps, self.spec.timescale);
        let (dist, stds, sources) = {
            let view = self.feed.as_ref().map(|f| f.graph.at(timescale, ts));
            let prices: Vec<SleevePrices<'_>> = ready
                .iter()
                .map(|&i| SleevePrices { ts: &self.sleeves[i].close_ts, px: &self.sleeves[i].closes })
                .collect();
            let ctxs: Vec<LiveContext<'_>> = ready
                .iter()
                .zip(&prices)
                .map(|(&i, p)| {
                    let s = &self.sleeves[i];
                    let from = s.closes.len() - (lookback + 1);
                    let mut c = LiveContext::new(FallbackContext::new(&s.closes[from..], None, s.book), s.id, steps);
                    if let Some(v) = &view {
                        c = c.with_graph(v);
                    }
                    if let Some(f) = &self.feed {
                        c = c.with_decoders(f.decoders.at(s.id, ts, f.decoder_max_age_ms));
                    }
                    c.with_prices(p)
                })
                .collect();
            let assets: Vec<(InstrumentId, &dyn StrategyContext)> =
                ready.iter().zip(&ctxs).map(|(&i, c)| (self.sleeves[i].id, c as &dyn StrategyContext)).collect();
            let dist = self.strategy.step(ts, &assets);
            let stds: Vec<f64> = ctxs.iter().map(|c| c.fallback().std_return()).collect();
            let mut sources = SourceCounts::default();
            for c in &ctxs {
                sources.add(c.sources());
            }
            (dist, stds, sources)
        };
        self.sources.graph += sources.graph;
        self.sources.decoder += sources.decoder;
        self.sources.fallback += sources.fallback;
        self.n_steps += 1;
        let mut std_sum = 0.0;
        for (k, &i) in ready.iter().enumerate() {
            let e = dist.exposures[k];
            let mean = if e.mean.is_finite() { e.mean.clamp(-1.0, 1.0) } else { 0.0 };
            let sd = if e.stddev.is_finite() { e.stddev.clamp(0.0, 1.0) } else { 1.0 };
            std_sum += sd;
            self.fill(i, ts, mean, stds[k]);
        }
        self.pending_std = Some(std_sum / ready.len() as f64);
        if self.n_steps.is_multiple_of(self.controls.progress_every.max(1))
            && let Some(cb) = &self.controls.on_progress
        {
            cb(self.n_steps, self.total_steps_hint);
        }
    }

    /// Moves sleeve `i` to `target_exposure` of its capital at the current close.
    fn fill(&mut self, i: usize, ts: i64, target_exposure: f64, sigma: f64) {
        let sleeve_capital = self.sleeve_capital;
        let s = &mut self.sleeves[i];
        let price = s.last_price;
        let target_units = target_exposure * sleeve_capital / price;
        let delta = target_units - s.units;
        let notional = (delta * price).abs();
        s.book.exposure = target_exposure;
        if notional <= self.spec.min_trade_fraction.max(0.0) * sleeve_capital || notional <= 1e-12 * sleeve_capital {
            s.book.steps_in_trade = s.book.steps_in_trade.saturating_add(1);
            return;
        }
        let inputs = CostInputs {
            venue: s.venue.clone(),
            class: s.class.clone(),
            side: if delta > 0.0 { "buy".into() } else { "sell".into() },
            price,
            qty: delta.abs(),
            sigma: sigma * self.sigma_scale,
            ..Default::default()
        };
        let priced = self.cost.price(&inputs);
        let priced_2x = self.cost_2x.price(&inputs);
        let same_side = s.units == 0.0 || delta.signum() == s.units.signum();
        if s.units == 0.0 || (!same_side && delta.abs() > s.units.abs()) {
            s.entry_price = price; // opened or flipped at this close
        } else if same_side {
            let total = s.units + delta;
            s.entry_price = (s.units * s.entry_price + delta * price) / total;
        }
        let sign_changed = s.units == 0.0 || (s.units > 0.0) != (target_units > 0.0);
        s.units = target_units;
        s.book.steps_in_trade = if sign_changed { 0 } else { s.book.steps_in_trade.saturating_add(1) };
        let token = s.token.clone();
        // cash moves at the fill price; the fee is explicit. Marking at `price` makes the
        // implicit cost |Δ|·(half_spread + impact) show up as negative P&L this step.
        self.cash -= delta * priced.fill_price + priced.fee;
        let cost = priced.total_cost(delta);
        self.step_cost += cost;
        self.total_cost += cost;
        self.step_cost_2x += priced_2x.total_cost(delta);
        self.turnover_notional += notional;
        self.n_trades += 1;
        if self.trades.len() < MAX_TRADE_LOG {
            self.trades.push(Trade {
                ts_ms: ts,
                instrument_id: token,
                delta_units: delta,
                price,
                notional,
                fill_price: priced.fill_price,
                fee: priced.fee,
                slippage_bps: priced.slippage_bps,
                cost,
                target_exposure,
            });
        }
    }

    /// End of a knowledge-time step: mark, then decide and fill at the close.
    fn flush_step(&mut self) {
        let Some(ts) = self.cur_ts else { return };
        self.mark(ts);
        self.last_ts = ts;
        if !self.halted {
            self.decide(ts);
        }
    }
}

impl ReplaySink<Event> for Engine<'_> {
    fn on_event(&mut self, publish_time_ms: i64, event: Event) {
        if self.cur_ts.is_some_and(|t| publish_time_ms != t) {
            self.flush_step();
        }
        if self.cur_ts != Some(publish_time_ms) {
            self.cur_ts = Some(publish_time_ms);
            self.first_ts.get_or_insert(publish_time_ms);
        }
        if self.halted {
            return;
        }
        let timescale = self.spec.timescale;
        let s = &mut self.sleeves[event.inst];
        s.closes.push(event.close);
        s.close_ts.push(publish_time_ms);
        s.last_price = event.close;
        s.pending_decision = true;
        if let Some(f) = &mut self.feed {
            // realised one-bar returns per region (`region_feature`), as the live runtime does
            f.graph.record_close(timescale, s.id, publish_time_ms, event.close);
        }
    }
}

/// Replays `strategy` over `data` (one series per instrument) under `spec`. Instruments listed
/// in `spec.instruments` select a subset of the data; an empty list uses every series.
/// `latents` is the recorded `LATENT` traffic the programs see point-in-time (`None` = the
/// price-only fallback throughout).
#[allow(clippy::too_many_arguments)]
pub fn replay(
    strategy: &GpStrategy,
    lookback_bars: usize,
    data: &[BarSeries],
    spec: &ReplaySpec,
    cost: &CostModel,
    controls: &ReplayControls,
    latents: Option<&RecordedLatents>,
) -> Result<EvalReport, ReplayError> {
    let started = Instant::now();
    if spec.capital.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || !spec.capital.is_finite() {
        return Err(ReplayError::Candidate("capital must be positive".into()));
    }
    let snapshot = match latents {
        Some(rec) => check_snapshot(spec, rec)?,
        None => SnapshotCheck { declared: spec.snapshot.clone(), ..Default::default() },
    };
    let selected: Vec<&BarSeries> = data
        .iter()
        .filter(|s| !s.is_empty())
        .filter(|s| {
            spec.instruments.is_empty()
                || spec.instruments.iter().any(|i| fa_decoders::instrument_token(i) == s.instrument_id)
        })
        .collect();
    if selected.is_empty() {
        return Err(ReplayError::NoData);
    }
    let sleeves: Vec<Sleeve> = selected
        .iter()
        .map(|s| Sleeve {
            id: instrument_id(&s.instrument_id),
            token: s.instrument_id.clone(),
            venue: s.venue.clone(),
            class: cost.class_of(&s.instrument_id),
            schedule: cost.schedule(&s.venue, &cost.class_of(&s.instrument_id)),
            schedule_2x: cost.scaled(2.0).schedule(&s.venue, &cost.class_of(&s.instrument_id)),
            closes: Vec::with_capacity(s.len()),
            close_ts: Vec::with_capacity(s.len()),
            units: 0.0,
            entry_price: 0.0,
            last_price: 0.0,
            book: VirtualBook::default(),
            pending_decision: false,
        })
        .collect();
    // Totally ordered event stream: (close time, instrument index).
    let (w0, w1) = spec.window_ms;
    let mut events: Vec<(i64, Event)> = Vec::new();
    for (i, s) in selected.iter().enumerate() {
        for (t, c) in s.ts_ms.iter().zip(&s.close) {
            if *t >= w0 && *t <= w1 {
                events.push((*t, Event { inst: i, close: *c }));
            }
        }
    }
    events.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.inst.cmp(&b.1.inst)));
    if events.is_empty() {
        return Err(ReplayError::NoData);
    }
    let distinct_steps = events.windows(2).filter(|w| w[0].0 != w[1].0).count() + 1;
    let feed = latents.map(LatentFeed::new);
    let mut engine = Engine::new(strategy.clone(), lookback_bars, spec, controls, cost, sleeves, distinct_steps, feed);
    let mut src = VecSource { events: events.into_iter() };
    run_replay(&mut src, &mut engine)?;
    engine.flush_step();
    let latents_report = ReplayLatents {
        available: latents.is_some(),
        recorded: latents.map_or(0, RecordedLatents::len),
        fed: engine.feed.as_ref().map(|f| f.fed).unwrap_or_default(),
        snapshot,
        graph_rows: engine.feed.as_ref().map_or(0, |f| f.graph.row_count()),
        decoder_outputs: engine.feed.as_ref().map_or(0, |f| f.decoders.len()),
    };

    let n_obs = engine.returns.len();
    let (sharpe, psr, dsr, sharpe_2x) = if n_obs > 2 {
        (
            stats::sharpe(&engine.returns),
            stats::probabilistic_sharpe(&engine.returns, 0.0),
            stats::deflated_sharpe(&engine.returns, spec.n_trials.max(1), spec.trial_sharpe_var.max(0.0)),
            stats::sharpe(&engine.returns_2x),
        )
    } else {
        (0.0, 0.5, 0.5, 0.0)
    };
    let gross_abs: f64 = engine.gross_pnl.abs();
    let cost_share =
        if gross_abs + engine.total_cost > 0.0 { engine.total_cost / (gross_abs + engine.total_cost) } else { 0.0 };
    let calibration = correlation(&engine.calib_pairs);
    let nodes = (strategy.signal.size() + strategy.uncertainty.size()) as f64 * selected.len() as f64;
    let stop = if engine.n_steps == 0 { Stop::NoData } else { engine.stop };
    Ok(EvalReport {
        strategy_id: strategy.id.clone(),
        genome_id: strategy.genome_id.clone(),
        timescale: spec.timescale,
        purpose: spec.purpose,
        cost_model: cost.id.clone(),
        seed: spec.seed,
        capital: spec.capital,
        instruments: selected.iter().map(|s| s.instrument_id.clone()).collect(),
        n_steps: engine.n_steps,
        n_obs,
        first_ts_ms: engine.first_ts.unwrap_or(0),
        last_ts_ms: engine.last_ts,
        stop,
        sharpe,
        psr,
        deflated_sharpe: dsr,
        n_trials: spec.n_trials.max(1),
        max_drawdown: engine.max_dd,
        turnover: if engine.n_steps > 0 {
            engine.turnover_notional / spec.capital / engine.n_steps as f64
        } else {
            0.0
        },
        cost_share,
        cost_sensitivity: sharpe - sharpe_2x,
        calibration,
        compute_cost: nodes.max(1.0).log10(),
        n_trades: engine.n_trades,
        gross_pnl: engine.gross_pnl,
        total_cost: engine.total_cost,
        net_pnl: engine.equity - spec.capital,
        final_equity: engine.equity,
        elapsed_ms: started.elapsed().as_secs_f64() * 1e3,
        equity: engine.equity_curve,
        returns: engine.returns,
        trades: engine.trades,
        sources: engine.sources,
        latents: latents_report,
    })
}

/// `InstrumentId` from a dashed UUID or a 32-hex token; anything else hashes into a stable id.
pub fn instrument_id(token: &str) -> InstrumentId {
    let t = fa_decoders::instrument_token(token);
    if t.len() == 32
        && t.chars().all(|c| c.is_ascii_hexdigit())
        && let Ok(u) = uuid::Uuid::parse_str(&t)
    {
        return InstrumentId(u);
    }
    let h = fa_decoders::genome::stable_hash(token);
    InstrumentId(uuid::Uuid::from_u64_pair(h, h.rotate_left(32)))
}

/// Pearson correlation of interleaved `[x0, y0, x1, y1, …]` pairs; 0 when undefined.
fn correlation(pairs: &[f64]) -> f64 {
    let n = pairs.len() / 2;
    if n < 3 {
        return 0.0;
    }
    let xs: Vec<f64> = pairs.iter().step_by(2).copied().collect();
    let ys: Vec<f64> = pairs.iter().skip(1).step_by(2).copied().collect();
    let (mx, my) = (stats::mean(&xs), stats::mean(&ys));
    let cov: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let vx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    let vy: f64 = ys.iter().map(|y| (y - my).powi(2)).sum();
    if vx > 0.0 && vy > 0.0 { cov / (vx * vy).sqrt() } else { 0.0 }
}

/// `(venue, (ts_ms, price) points)` of one instrument.
pub type VenuePoints = (String, Vec<(i64, f64)>);

/// Bars of several instruments from a map `token → (venue, points)`.
pub fn bars_from_points(series: &BTreeMap<String, VenuePoints>, step_ms: i64) -> Vec<BarSeries> {
    series.iter().map(|(inst, (venue, pts))| BarSeries::from_points(inst, venue, pts, step_ms)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fa_core::rng::Rng;

    /// SYNTHETIC: random walk; not market data.
    fn walk(n: usize, seed: u64, drift: f64, step_ms: i64) -> Vec<(i64, f64)> {
        let mut rng = Rng::seed(seed);
        let mut p = 100.0;
        (0..n)
            .map(|i| {
                p *= 1.0 + drift + 0.002 * rng.normal();
                (i as i64 * step_ms, p)
            })
            .collect()
    }

    fn def(signal: &str) -> StrategyDef {
        StrategyDef {
            id: "t".into(),
            genome_id: "g".into(),
            timescale: Timescale::Second,
            steps: 3,
            lookback_bars: 16,
            step_secs: None,
            signal: signal.into(),
            uncertainty: "region_entropy".into(),
        }
    }

    fn data(seed: u64) -> Vec<BarSeries> {
        vec![
            BarSeries::from_points("a", "sim", &walk(400, seed, 0.0, 1000), 1000),
            BarSeries::from_points("b", "yahoo", &walk(380, seed + 1, 0.0003, 1000), 1000),
        ]
    }

    #[test]
    fn bars_are_stamped_at_close_and_ordered() {
        let s =
            BarSeries::from_points("x", "sim", &[(0, 1.0), (500, 2.0), (3000, 3.0), (2000, 9.0), (3500, -1.0)], 1000);
        assert_eq!(s.ts_ms, vec![1000, 4000]);
        assert_eq!(s.close, vec![2.0, 3.0]);
    }

    #[test]
    fn replay_is_deterministic_and_pays_costs() {
        let d = data(3);
        let cost = CostModel::default();
        let spec = ReplaySpec { capital: 10_000.0, ..ReplaySpec::selection(Timescale::Second, 0, i64::MAX) };
        let ev =
            PipelineReplay { data: &d, cost: &cost, holdouts: &[], controls: ReplayControls::default(), latents: None };
        let a = ev.evaluate(&spec, &def("(mul 400 analog_forecast)")).unwrap();
        let b = ev.evaluate(&spec, &def("(mul 400 analog_forecast)")).unwrap();
        let strip = |mut r: EvalReport| {
            r.elapsed_ms = 0.0;
            r
        };
        assert_eq!(strip(a.clone()), strip(b), "identical spec + data must give an identical report");
        assert_eq!(a.instruments, vec!["a".to_string(), "b".to_string()]);
        assert!(a.n_steps > 300 && a.n_obs >= a.n_steps, "{a:?}");
        assert!(a.n_trades > 0 && a.total_cost > 0.0);
        assert!(a.trades.iter().all(|t| t.cost > 0.0 && t.notional > 0.0 && t.fee >= 1.0));
        assert!((0.0..=1.0).contains(&a.max_drawdown) && a.turnover > 0.0);
        assert!(a.objectives().iter().all(|(_, v)| v.is_finite()));
        assert!((a.final_equity - (spec.capital + a.net_pnl)).abs() < 1e-6);
        // equity curve is consistent with the per-step returns
        let total: f64 = a.returns.iter().sum::<f64>() * spec.capital;
        assert!((total - a.net_pnl).abs() < 1e-6, "{total} vs {}", a.net_pnl);
        assert_eq!(a.equity.len(), a.n_obs);
        assert!(a.stop == Stop::Complete);
        // a flat strategy trades nothing and has zero return
        let flat = ev.evaluate(&spec, &def("0")).unwrap();
        assert_eq!((flat.n_trades, flat.total_cost), (0, 0.0));
        assert!(flat.returns.iter().all(|r| *r == 0.0));
        // costs matter: a 2× cost model gives lower net P&L for the same decisions
        let twice = cost.scaled(2.0);
        let ev2 = PipelineReplay {
            data: &d,
            cost: &twice,
            holdouts: &[],
            controls: ReplayControls::default(),
            latents: None,
        };
        let c = ev2.evaluate(&spec, &def("(mul 400 analog_forecast)")).unwrap();
        assert!(c.total_cost > a.total_cost && c.net_pnl < a.net_pnl);
        assert!((a.cost_sensitivity - (a.sharpe - c.sharpe)).abs() < 1e-9, "2× sensitivity matches a 2× replay");
    }

    #[test]
    fn long_only_on_a_drifting_series_makes_money_before_costs() {
        let d = vec![BarSeries::from_points("up", "paper", &walk(600, 11, 0.001, 1000), 1000)];
        let cost = CostModel::default();
        let spec = ReplaySpec { capital: 100_000.0, ..ReplaySpec::selection(Timescale::Second, 0, i64::MAX) };
        let ev =
            PipelineReplay { data: &d, cost: &cost, holdouts: &[], controls: ReplayControls::default(), latents: None };
        let r = ev.evaluate(&spec, &def("1000")).unwrap();
        assert!(r.gross_pnl > 0.0, "{r:?}");
        assert!(r.n_trades < 60, "a constant target rebalances only past the threshold: {}", r.n_trades);
        let churn =
            PipelineReplay { data: &d, cost: &cost, holdouts: &[], controls: ReplayControls::default(), latents: None }
                .evaluate(&ReplaySpec { min_trade_fraction: 0.0, ..spec.clone() }, &def("1000"))
                .unwrap();
        assert!(churn.n_trades > r.n_trades && churn.total_cost > r.total_cost);
        assert!(r.sharpe > 0.0 && r.psr > 0.5);
    }

    #[test]
    fn holdout_window_and_controls() {
        let d = data(5);
        let cost = CostModel::default();
        let h = [Holdout { timescale: Timescale::Second, start_ms: 300_000, end_ms: 400_000, locked_at_ms: 1 }];
        let spec = ReplaySpec::selection(Timescale::Second, 0, 350_000);
        assert!(matches!(check_holdout(&spec, &h), Err(ReplayError::HoldoutIntersect { .. })));
        let ev =
            PipelineReplay { data: &d, cost: &cost, holdouts: &h, controls: ReplayControls::default(), latents: None };
        assert!(ev.evaluate(&spec, &def("1")).is_err());
        let promo = ReplaySpec { purpose: ReplayPurpose::Promotion, ..spec.clone() };
        assert!(check_holdout(&promo, &h).is_ok());
        let other_ts = ReplaySpec { timescale: Timescale::Minute, ..spec.clone() };
        assert!(check_holdout(&other_ts, &h).is_ok());
        let before = ReplaySpec::selection(Timescale::Second, 0, 299_999);
        let r =
            PipelineReplay { data: &d, cost: &cost, holdouts: &h, controls: ReplayControls::default(), latents: None }
                .evaluate(&before, &def("1"))
                .unwrap();
        assert!(r.last_ts_ms <= 299_999);
        // stop hooks
        let ctl = ReplayControls { max_steps: Some(5), ..Default::default() };
        let r = PipelineReplay { data: &d, cost: &cost, holdouts: &[], controls: ctl, latents: None }
            .evaluate(&ReplaySpec::selection(Timescale::Second, 0, i64::MAX), &def("1"))
            .unwrap();
        assert_eq!((r.n_steps, r.stop), (5, Stop::MaxSteps));
        let ctl = ReplayControls { should_stop: Some(Box::new(|n| n >= 2)), ..Default::default() };
        let r = PipelineReplay { data: &d, cost: &cost, holdouts: &[], controls: ctl, latents: None }
            .evaluate(&ReplaySpec::selection(Timescale::Second, 0, i64::MAX), &def("1"))
            .unwrap();
        assert_eq!((r.n_steps, r.stop), (2, Stop::Interrupted));
        // no data in window
        assert!(matches!(
            PipelineReplay { data: &d, cost: &cost, holdouts: &[], controls: ReplayControls::default(), latents: None }
                .evaluate(&ReplaySpec::selection(Timescale::Second, 10_000_000, 20_000_000), &def("1")),
            Err(ReplayError::NoData)
        ));
        let bad = StrategyDef { signal: "(rsi 14)".into(), ..def("1") };
        assert!(matches!(
            PipelineReplay { data: &d, cost: &cost, holdouts: &[], controls: ReplayControls::default(), latents: None }
                .evaluate(&ReplaySpec::selection(Timescale::Second, 0, i64::MAX), &bad),
            Err(ReplayError::Candidate(_))
        ));
    }

    #[test]
    fn look_ahead_guard_catches_future_event() {
        // A source that emits a later close before an earlier one is rejected (spec 41 §9).
        struct Bad(Vec<(i64, Event)>);
        impl ReplaySource for Bad {
            type Event = Event;
            fn next_event(&mut self) -> Option<(i64, Event)> {
                if self.0.is_empty() { None } else { Some(self.0.remove(0)) }
            }
        }
        struct Count(usize);
        impl ReplaySink<Event> for Count {
            fn on_event(&mut self, _: i64, _: Event) {
                self.0 += 1;
            }
        }
        let mut sink = Count(0);
        let r = run_replay(
            &mut Bad(vec![(2000, Event { inst: 0, close: 1.0 }), (1000, Event { inst: 0, close: 1.0 })]),
            &mut sink,
        );
        assert_eq!(r, Err(ReplayError::OutOfOrder { prev: 2000, got: 1000 }));
        assert_eq!(
            instrument_id("01900000-0000-7000-8000-000000000001").subject_token(),
            "01900000000070008000000000000001"
        );
        assert_eq!(instrument_id("a"), instrument_id("a"));
    }

    /// The real recording (`testdata/latent_fixture.jsonl`, trimmed from the live `LATENT`
    /// stream) over SYNTHETIC bars placed in its time range.
    fn fixture_setup() -> (crate::latents::RecordedLatents, Vec<BarSeries>, ReplaySpec) {
        let rec = crate::latents::RecordedLatents::from_jsonl(crate::latents::tests::FIXTURE).unwrap();
        let insts = rec.instruments_of(Timescale::Second);
        assert!(!insts.is_empty(), "fixture has 1s states");
        let states: Vec<i64> = rec
            .records()
            .iter()
            .filter(|r| matches!(&r.message, fa_bus::replay::LatentMessage::State(s) if s.timescale == "1s"))
            .map(|r| r.published_ms)
            .collect();
        let (t0, t1) = (*states.iter().min().unwrap(), *states.iter().max().unwrap());
        let start = (t0 / 1000) * 1000 - 60_000;
        let n = ((t1 - start) / 1000) as usize + 2;
        let data: Vec<BarSeries> = insts
            .iter()
            .enumerate()
            .map(|(k, i)| {
                let pts: Vec<(i64, f64)> =
                    walk(n, 40 + k as u64, 0.0, 1000).into_iter().map(|(t, p)| (start + t, p)).collect();
                BarSeries::from_points(&i.subject_token(), "sim", &pts, 1000)
            })
            .collect();
        let spec = ReplaySpec { capital: 10_000.0, ..ReplaySpec::selection(Timescale::Second, start, t1 + 1000) };
        (rec, data, spec)
    }

    #[test]
    fn recorded_latents_feed_the_graph_deterministically() {
        let (rec, data, spec) = fixture_setup();
        let cost = CostModel::default();
        let graph_program = def("(add (mul 4 (sub stay_prob 0.5)) (mul 50 (fc_dist 5)))");
        let ev = PipelineReplay {
            data: &data,
            cost: &cost,
            holdouts: &[],
            controls: ReplayControls::default(),
            latents: Some(&rec),
        };
        let a = ev.evaluate(&spec, &graph_program).unwrap();
        let b = ev.evaluate(&spec, &graph_program).unwrap();
        let strip = |mut r: EvalReport| {
            r.elapsed_ms = 0.0;
            r
        };
        assert_eq!(strip(a.clone()), strip(b), "identical spec + data + recording → identical report");
        eprintln!("sources {:?} latents {:?}", a.sources, a.latents);
        assert!(a.latents.available && a.latents.recorded == rec.len());
        assert!(
            a.latents.fed.states > 0 && a.latents.fed.transitions > 0 && a.latents.graph_rows > 0,
            "{:?}",
            a.latents
        );
        assert!(a.sources.graph > 0, "graph primitives answered from the recording: {:?}", a.sources);
        assert!(a.sources.fallback > 0, "steps before the first state fall back");
        assert_eq!(a.latents.snapshot.declared, "stat-v1-1s");
        assert!(!a.latents.snapshot.mixed);
        let obj = a.objectives();
        assert!(obj.iter().any(|(k, v)| *k == "sources.graph" && *v > 0.0));
        // the same program without the recording is fallback-only and differs
        let plain = PipelineReplay {
            data: &data,
            cost: &cost,
            holdouts: &[],
            controls: ReplayControls::default(),
            latents: None,
        }
        .evaluate(&spec, &graph_program)
        .unwrap();
        assert_eq!((plain.sources.graph, plain.sources.decoder), (0, 0));
        assert!(!plain.latents.available);
        assert_ne!(plain.returns, a.returns, "graph-fed decisions differ from the price fallback");
        // snapshot consistency (spec 41 §4): a replay declared on another snapshot is refused…
        let other = ReplaySpec { snapshot: "stat-v2-1s".into(), ..spec.clone() };
        assert!(matches!(ev.evaluate(&other, &graph_program), Err(ReplayError::SnapshotMismatch { .. })));
        // …unless it is diagnostic
        let diag = ReplaySpec { purpose: ReplayPurpose::Diagnostic, ..other };
        let d = ev.evaluate(&diag, &graph_program).unwrap();
        assert!(d.latents.snapshot.mixed && d.latents.snapshot.declared == "stat-v2-1s");
    }

    #[test]
    fn a_latent_after_the_step_is_never_visible() {
        // SYNTHETIC recording: the instrument sits in region `a`, then switches to `b` at 10 s.
        // Up to and including the step at 10 s nothing of `b` is visible (published at 10.05 s).
        use crate::latents::tests::{raw, region_msg, row_msg, state_msg};
        let inst = instrument_id("01900000-0000-7000-8000-000000000001");
        let mut msgs = vec![
            raw("latent.region.1s", 1, 0, region_msg("s", "a")),
            raw("latent.region.1s", 2, 0, region_msg("s", "b")),
            raw("latent.transition.1s", 3, 0, row_msg("s", "a", 0, &[("a", 0.9), ("unknown", 0.1)])),
            raw("latent.transition.1s", 4, 0, row_msg("s", "b", 0, &[("b", 0.1), ("a", 0.8), ("unknown", 0.1)])),
        ];
        for t in 1..=20i64 {
            let region = if t < 10 { "a" } else { "b" };
            msgs.push(raw("latent.state.1s.x", 10 + t as u64, t * 1000 + 50, state_msg("s", inst, t * 1000, region)));
        }
        let rec = crate::latents::RecordedLatents::new(msgs);
        let bars = vec![BarSeries::from_points(&inst.subject_token(), "sim", &walk(40, 1, 0.0, 1000), 1000)];
        let cost = CostModel::default();
        // stay_prob is 0.9 in `a` and 0.1 in `b`: the exposure flips sign at the switch
        let prog = StrategyDef { lookback_bars: 2, ..def("(mul 100 (sub stay_prob 0.5))") };
        let step = |end_ms: i64| {
            PipelineReplay {
                data: &bars,
                cost: &cost,
                holdouts: &[],
                controls: ReplayControls::default(),
                latents: Some(&rec),
            }
            .evaluate(&ReplaySpec::selection(Timescale::Second, 0, end_ms), &prog)
            .unwrap()
        };
        let upto_10 = step(10_000);
        assert!(!upto_10.trades.is_empty());
        assert!(
            upto_10.trades.iter().all(|t| t.target_exposure > 0.9),
            "through 10 s the switch (published 10.05 s) is invisible: {:?}",
            upto_10.trades
        );
        let upto_11 = step(11_000);
        let last = upto_11.trades.last().unwrap();
        assert_eq!(last.ts_ms, 11_000);
        assert!(last.target_exposure < -0.9, "at 11 s the state dated 10 s is visible: {last:?}");
        assert_eq!(upto_11.latents.fed.states, 10, "states published ≤ 11 s");
        assert!(upto_11.sources.graph > 0 && upto_11.sources.graph >= upto_10.sources.graph);
        // before any state exists everything falls back
        let early = step(2_000);
        assert_eq!(early.sources.graph, 0);
    }
}
