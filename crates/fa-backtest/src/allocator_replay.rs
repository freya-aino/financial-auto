//! Allocator replay (spec 40 §6.2 "allocator tournament", spec 41 §6.1): recorded strategy
//! signals (`signal.exposure.<strategy>` from the `SIGNAL` JetStream stream, read by time with
//! `fa_bus::replay::read_range`) and marks (bar closes from VictoriaMetrics) are replayed
//! through one allocator ([`AllocatorGenome`]) → cross-strategy netting
//! ([`fa_allocator::net_targets_shrunk`]) → execution priced by the same [`CostModel`] as the
//! paper venue and the strategy replay, producing a **portfolio** [`AllocatorReport`].
//!
//! Per step `t` of the grid `[start, end]` (`step_ms`):
//! 1. signals published `≤ t` become visible (knowledge time = stream publish time); each
//!    strategy's latest one counts while younger than `signal_ttl_ms` (the risk-engine's
//!    `signal_ttl_secs`);
//! 2. marks: the last close `≤ t` per instrument; the book is marked to market (equity point,
//!    per-step net return) and every strategy that held a signal over the step feeds its
//!    realised step return to the allocator's [`PerformanceTracker`] (half-life and prior from
//!    the genome);
//! 3. the allocator weighs the visible strategies from the tracker's posteriors, the weighted
//!    exposures are netted into EUR targets on the current equity, and each instrument whose
//!    target moved by more than `min_trade_fraction · capital` is traded at the close through
//!    [`CostModel::price`] (σ_day from the instrument's recent step returns).
//!
//! Deterministic: signals ordered by `(publish time, stream sequence)`, instruments by token,
//! the allocators and the cost model pure — identical spec + genome + data give an identical
//! report (test). Every allocator of a tournament ([`tournament`]) sees identical signals and
//! marks, so their portfolio P&L is comparable (spec 40 §9).
//!
//! Objectives ([`DIRECTIONS`]): `deflated_sharpe` (max), `sharpe` (max, per-step net),
//! `max_drawdown` (min), `turnover` (min, traded notional / capital / step), `concentration`
//! (min, mean Herfindahl index of the absolute instrument targets), `cost_share` (min,
//! costs / (|gross P&L| + costs)); informational `psr`, `n_trades`, `n_obs`, `n_strategies`,
//! `mean_gross`, `max_weight`.

use std::collections::{BTreeMap, BTreeSet};

use fa_allocator::{Allocation, PerformanceTracker, genome::AllocatorGenome, net_targets_shrunk};
use fa_bus::replay::RecordedMessage;
use fa_core::{Horizon, InstrumentId, Timescale};
use fa_proto::Message;
use fa_risk::rebalance::{
    CostQuote, RebalanceConfig, RebalanceStats, Rebalancer, TradeCandidate, effective_obs, strategy_edge,
};
use fa_strategies::{AssetExposure, ExposureDistribution};
use serde::{Deserialize, Serialize};

use crate::{
    ReplayError,
    cost::{CostInputs, CostModel},
    eval::Stop,
    replay::{BarSeries, ReplayControls, instrument_id},
    stats,
};

/// The stream holding `signal.>` (spec 21 §5, 30 d retention).
pub const SIGNAL_STREAM: &str = "SIGNAL";
/// Subject filter of strategy exposure signals.
pub const SIGNAL_FILTER: &str = "signal.exposure.>";
/// Ranked objectives of the portfolio report.
pub const DIRECTIONS: [(&str, &str); 6] = [
    ("deflated_sharpe", "max"),
    ("sharpe", "max"),
    ("max_drawdown", "min"),
    ("turnover", "min"),
    ("concentration", "min"),
    ("cost_share", "min"),
];
/// Recent step returns used for σ_day of the impact model.
const SIGMA_WINDOW: usize = 64;

/// One recorded signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalRecord {
    /// Knowledge time (stream publish time).
    pub published_ms: i64,
    pub stream_seq: u64,
    pub dist: ExposureDistribution,
}

/// `"5x1s"` → 5 steps of 1 s (the `to_proto` format); unknown → 1 × 1 s.
pub fn parse_horizon(s: &str) -> Horizon {
    let fallback = Horizon { timescale: Timescale::Second, steps: 1 };
    let Some((steps, ts)) = s.split_once('x') else { return fallback };
    match (steps.parse::<u32>(), ts.parse::<Timescale>()) {
        (Ok(steps), Ok(timescale)) => Horizon { timescale, steps: steps.max(1) },
        _ => fallback,
    }
}

impl SignalRecord {
    /// Decodes a recorded `signal.exposure.*` message (`fa.v1.ExposureDistribution`); `None` for
    /// other subjects or undecodable payloads. Instruments that are not UUIDs map through
    /// [`instrument_id`] (stable hash), non-finite values are neutralised as the risk-engine does.
    pub fn decode(raw: &RecordedMessage) -> Option<Self> {
        if !raw.subject.starts_with("signal.exposure.") {
            return None;
        }
        let m = fa_proto::v1::ExposureDistribution::decode(raw.payload.as_slice()).ok()?;
        let strategy_id =
            if m.strategy_id.is_empty() { raw.subject["signal.exposure.".len()..].to_owned() } else { m.strategy_id };
        let as_of_ms = m.as_of.as_ref().map_or(raw.published_ms, |t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000);
        let exposures = m
            .exposures
            .iter()
            .map(|e| AssetExposure {
                instrument: instrument_id(&e.instrument_id),
                mean: if e.mean.is_finite() { e.mean.clamp(-1.0, 1.0) } else { 0.0 },
                stddev: if e.stddev.is_finite() { e.stddev.max(0.0) } else { 1.0 },
            })
            .collect();
        Some(Self {
            published_ms: raw.published_ms,
            stream_seq: raw.stream_seq,
            dist: ExposureDistribution {
                strategy_id,
                genome_id: m.genome_id,
                horizon: parse_horizon(&m.horizon),
                as_of_ms,
                exposures,
            },
        })
    }
}

/// Decodes and orders a recording by `(publish time, sequence)`; returns the signals and the
/// number of undecodable records.
pub fn decode_signals(raw: &[RecordedMessage]) -> (Vec<SignalRecord>, usize) {
    let mut out: Vec<SignalRecord> = raw.iter().filter_map(SignalRecord::decode).collect();
    let bad = raw.len() - out.len();
    out.sort_by(|a, b| a.published_ms.cmp(&b.published_ms).then(a.stream_seq.cmp(&b.stream_seq)));
    out.dedup_by(|a, b| a.stream_seq == b.stream_seq && a.published_ms == b.published_ms);
    (out, bad)
}

/// Replay request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AllocatorReplaySpec {
    /// `[start_ms, end_ms]` of the decision grid.
    pub window_ms: (i64, i64),
    pub step_ms: i64,
    /// Portfolio capital in EUR (returns are relative to it).
    pub capital: f64,
    /// A strategy's latest signal counts while younger than this.
    pub signal_ttl_ms: i64,
    /// Smallest rebalance executed, as a fraction of capital.
    pub min_trade_fraction: f64,
    /// Trials in the generation so far (deflated Sharpe); `1` = plain PSR.
    pub n_trials: usize,
    pub trial_sharpe_var: f64,
    /// When a target change is traded (default: `min_trade_fraction` of capital).
    #[serde(default)]
    pub trade_rule: TradeRule,
    /// Per-source knowledge delays applied to the marks (spec 41 §6.2).
    #[serde(default = "crate::knowledge::catalog_delays")]
    pub knowledge_delays: crate::knowledge::KnowledgeDelays,
}

/// The rebalance rule of a replay.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "rule", rename_all = "snake_case")]
pub enum TradeRule {
    /// Trade any target move above `min_trade_fraction · capital` (the tournament's rule).
    #[default]
    Fraction,
    /// The risk-engine's D-041 no-trade band: `max(min_eur, frac · |target|)`.
    Band { frac: f64, min_eur: f64 },
    /// The live cost-aware rule (`fa_risk::rebalance`, the same [`Rebalancer`] the risk-engine
    /// runs) on top of the band.
    CostAware { frac: f64, min_eur: f64, cfg: RebalanceConfig },
}

impl AllocatorReplaySpec {
    /// 1 s grid, 100 000 EUR, 30 s signal TTL, 0.1 % no-trade band, one trial.
    pub fn new(start_ms: i64, end_ms: i64) -> Self {
        Self {
            window_ms: (start_ms, end_ms),
            step_ms: 1_000,
            capital: 100_000.0,
            signal_ttl_ms: 30_000,
            min_trade_fraction: 0.001,
            n_trials: 1,
            trial_sharpe_var: 0.0,
            trade_rule: TradeRule::Fraction,
            knowledge_delays: crate::knowledge::catalog_delays(),
        }
    }
}

/// Portfolio result of one allocator (spec 41 §4 `EvalReport`, portfolio form).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AllocatorReport {
    pub genome: String,
    pub allocator: String,
    pub capital: f64,
    pub cost_model: String,
    pub n_steps: usize,
    pub n_obs: usize,
    /// Steps where at least one strategy had a live signal.
    pub n_active_steps: usize,
    pub first_ts_ms: i64,
    pub last_ts_ms: i64,
    pub stop: Stop,
    pub n_signals: usize,
    pub strategies: Vec<String>,
    pub instruments: Vec<String>,
    pub sharpe: f64,
    pub psr: f64,
    pub deflated_sharpe: f64,
    pub n_trials: usize,
    pub max_drawdown: f64,
    pub turnover: f64,
    /// Mean Herfindahl index of |target| over instruments (steps with exposure).
    pub concentration: f64,
    pub cost_share: f64,
    /// Mean gross exposure / equity.
    pub mean_gross: f64,
    /// Largest capital weight the allocator gave any strategy.
    pub max_weight: f64,
    pub n_trades: usize,
    pub gross_pnl: f64,
    pub total_cost: f64,
    /// Commissions alone (part of `total_cost`).
    #[serde(default)]
    pub total_fees: f64,
    /// Decisions of the cost-aware rule (`TradeRule::CostAware`).
    #[serde(default)]
    pub rebalance: Option<RebalanceStats>,
    pub net_pnl: f64,
    pub final_equity: f64,
    /// Mean capital weight per strategy over active steps.
    pub mean_weights: BTreeMap<String, f64>,
    pub equity: Vec<(i64, f64)>,
    pub returns: Vec<f64>,
}

impl AllocatorReport {
    /// Objectives as `name → value`, all finite.
    pub fn objectives(&self) -> Vec<(&'static str, f64)> {
        let f = |x: f64| if x.is_finite() { x } else { 0.0 };
        vec![
            ("deflated_sharpe", f(self.deflated_sharpe)),
            ("sharpe", f(self.sharpe)),
            ("max_drawdown", f(self.max_drawdown)),
            ("turnover", f(self.turnover)),
            ("concentration", f(self.concentration)),
            ("cost_share", f(self.cost_share)),
            ("psr", f(self.psr)),
            ("n_trades", self.n_trades as f64),
            ("n_obs", self.n_obs as f64),
            ("n_strategies", self.strategies.len() as f64),
            ("mean_gross", f(self.mean_gross)),
            ("max_weight", f(self.max_weight)),
        ]
    }

    /// Scalar fallback for the orchestrator: the deflated Sharpe.
    pub fn score(&self) -> f64 {
        if self.deflated_sharpe.is_finite() { self.deflated_sharpe } else { f64::NEG_INFINITY }
    }
}

struct Mark {
    token: String,
    venue: String,
    class: String,
    ts: Vec<i64>,
    px: Vec<f64>,
    cursor: usize,
    last: Option<f64>,
    units: f64,
    rets: Vec<f64>,
}

impl Mark {
    /// Advances to the last close `≤ t`; returns the price.
    fn advance(&mut self, t: i64) -> Option<f64> {
        while self.cursor < self.ts.len() && self.ts[self.cursor] <= t {
            self.last = Some(self.px[self.cursor]);
            self.cursor += 1;
        }
        self.last
    }
}

/// Replays `signals` (decoded, any order) and `marks` through the allocator of `genome`.
pub fn replay_allocator(
    genome: &AllocatorGenome,
    signals: &[SignalRecord],
    marks: &[BarSeries],
    spec: &AllocatorReplaySpec,
    cost: &CostModel,
    controls: &ReplayControls,
) -> Result<AllocatorReport, ReplayError> {
    if spec.capital.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || !spec.capital.is_finite() {
        return Err(ReplayError::Candidate("capital must be positive".into()));
    }
    if spec.step_ms <= 0 || spec.window_ms.1 <= spec.window_ms.0 {
        return Err(ReplayError::Candidate("empty window or non-positive step".into()));
    }
    let mut signals: Vec<&SignalRecord> = signals.iter().filter(|s| s.published_ms <= spec.window_ms.1).collect();
    signals.sort_by(|a, b| a.published_ms.cmp(&b.published_ms).then(a.stream_seq.cmp(&b.stream_seq)));
    if signals.is_empty() {
        return Err(ReplayError::NoData);
    }
    // knowledge time (spec 41 §6.2): late-publishing sources become visible after their delay
    let marks = spec.knowledge_delays.apply(marks);
    // instrument token → mark (sorted for determinism)
    let mut by_token: BTreeMap<String, Mark> = BTreeMap::new();
    for b in marks.iter().filter(|b| !b.is_empty()) {
        let token = fa_decoders::instrument_token(&b.instrument_id);
        by_token.entry(token.clone()).or_insert_with(|| Mark {
            class: cost.class_of(&token),
            token,
            venue: b.venue.clone(),
            ts: b.ts_ms.clone(),
            px: b.close.clone(),
            cursor: 0,
            last: None,
            units: 0.0,
            rets: Vec::new(),
        });
    }
    if by_token.is_empty() {
        return Err(ReplayError::NoData);
    }
    let allocator = genome.build();
    let mut tracker: PerformanceTracker = genome.tracker();
    let sigma_scale = (86_400_000.0 / spec.step_ms as f64).sqrt();

    let mut latest: BTreeMap<String, &SignalRecord> = BTreeMap::new();
    let mut held: Vec<ExposureDistribution> = Vec::new(); // signals active over the last step
    let mut cursor = 0usize;
    let (mut cash, mut equity) = (spec.capital, spec.capital);
    let mut equity_curve = Vec::new();
    let mut returns = Vec::new();
    let (mut peak, mut max_dd) = (spec.capital, 0.0f64);
    let (mut gross_pnl, mut total_cost, mut step_cost) = (0.0, 0.0, 0.0);
    let (mut turnover_notional, mut n_trades, mut total_fees) = (0.0f64, 0usize, 0.0f64);
    let mut rebalancer = match &spec.trade_rule {
        TradeRule::CostAware { cfg, .. } => Some(Rebalancer::new(cfg.clone())),
        _ => None,
    };
    let step_secs = spec.step_ms as f64 / 1000.0;
    let mut last_dists: BTreeMap<String, ExposureDistribution> = BTreeMap::new();
    let mut hhi_sum = 0.0;
    let mut hhi_n = 0usize;
    let mut gross_sum = 0.0;
    let mut max_weight = 0.0f64;
    let mut weight_sums: BTreeMap<String, f64> = BTreeMap::new();
    let mut strategies: BTreeSet<String> = BTreeSet::new();
    let mut n_signals = 0usize;
    let (mut n_steps, mut n_active) = (0usize, 0usize);
    let mut stop = Stop::Complete;
    let total_steps = ((spec.window_ms.1 - spec.window_ms.0) / spec.step_ms + 1) as usize;
    let mut prev_px: BTreeMap<String, f64> = BTreeMap::new();

    let mut t = spec.window_ms.0;
    while t <= spec.window_ms.1 {
        if controls.max_steps.is_some_and(|m| n_steps >= m) {
            stop = Stop::MaxSteps;
            break;
        }
        if controls.should_stop.as_ref().is_some_and(|f| f(n_steps)) {
            stop = Stop::Interrupted;
            break;
        }
        // 1. signals visible at t
        while cursor < signals.len() && signals[cursor].published_ms <= t {
            let s = signals[cursor];
            if s.published_ms >= spec.window_ms.0 {
                n_signals += 1;
            }
            latest.insert(s.dist.strategy_id.clone(), s);
            cursor += 1;
        }
        // 2. marks, mark-to-market, realised step returns
        let mut realised: BTreeMap<InstrumentId, f64> = BTreeMap::new();
        let mut mtm = 0.0;
        for m in by_token.values_mut() {
            let Some(p) = m.advance(t) else { continue };
            if let Some(&q) = prev_px.get(&m.token)
                && q > 0.0
            {
                let r = p / q - 1.0;
                realised.insert(instrument_id(&m.token), r);
                m.rets.push(r);
                if m.rets.len() > SIGMA_WINDOW {
                    m.rets.remove(0);
                }
            }
            prev_px.insert(m.token.clone(), p);
            mtm += m.units * p;
        }
        let before = equity;
        equity = cash + mtm;
        if n_steps > 0 {
            let net = equity - before;
            gross_pnl += net + step_cost;
            returns.push(net / spec.capital);
            equity_curve.push((t, equity));
            peak = peak.max(equity);
            max_dd = max_dd.max((peak - equity) / spec.capital);
            for d in &held {
                tracker.observe(d, &realised);
            }
        }
        step_cost = 0.0;
        n_steps += 1;
        if controls.progress_every > 0
            && n_steps.is_multiple_of(controls.progress_every)
            && let Some(cb) = &controls.on_progress
        {
            cb(n_steps, total_steps);
        }

        // 3. allocate the strategies with a live signal, net, trade
        let live: Vec<&SignalRecord> =
            latest.values().copied().filter(|s| t - s.published_ms <= spec.signal_ttl_ms).collect();
        held = live.iter().map(|s| s.dist.clone()).collect();
        let alloc: Allocation = if live.is_empty() {
            Allocation::new()
        } else {
            n_active += 1;
            for s in &live {
                strategies.insert(s.dist.strategy_id.clone());
            }
            let stats = tracker.stats_for(live.iter().map(|s| s.dist.strategy_id.as_str()));
            let a = allocator.allocate(&stats);
            let a: Allocation = a.into_iter().filter(|(_, w)| w.is_finite() && *w > 0.0).collect();
            for (k, w) in &a {
                *weight_sums.entry(k.clone()).or_insert(0.0) += w;
                max_weight = max_weight.max(*w);
            }
            a
        };
        // strategies over their turnover budget are netted with their last distribution (frozen)
        let netted: Vec<ExposureDistribution> = match rebalancer.as_mut() {
            Some(rb) => {
                let frozen = rb.over_budget(t).clone();
                held.iter()
                    .map(|d| {
                        if frozen.contains(&d.strategy_id) {
                            last_dists.get(&d.strategy_id).cloned().unwrap_or_else(|| d.clone())
                        } else {
                            last_dists.insert(d.strategy_id.clone(), d.clone());
                            d.clone()
                        }
                    })
                    .collect()
            }
            None => held.clone(),
        };
        let targets = net_targets_shrunk(&netted, &alloc, equity.max(0.0), genome.shrink);
        // per-instrument contributions (EUR, edge, holding) for the cost-aware rule — the
        // risk-engine tick's computation
        let mut contributions: BTreeMap<InstrumentId, Vec<(String, f64, f64, f64)>> = BTreeMap::new();
        if let Some(rb) = &rebalancer {
            let stats_now = tracker.stats_for(netted.iter().map(|d| d.strategy_id.as_str()));
            for (d, st) in netted.iter().zip(&stats_now) {
                let w = alloc.get(&d.strategy_id).copied().unwrap_or(0.0);
                if w <= 0.0 || d.exposures.is_empty() {
                    continue;
                }
                let mean_abs = d.exposures.iter().map(|e| e.mean.abs()).sum::<f64>() / d.exposures.len() as f64;
                let hold = d.horizon.duration().as_secs_f64() * rb.cfg.holding_horizons.max(0.0);
                let n_eff = effective_obs(st.observations, tracker.decay);
                let edge = strategy_edge(&rb.cfg, st.mean_return, mean_abs, n_eff, step_secs, hold);
                let per_asset = w * equity.max(0.0) / d.exposures.len() as f64;
                for e in &d.exposures {
                    let c =
                        per_asset * e.mean.clamp(-1.0, 1.0) / (1.0 + e.stddev.max(0.0)).powf(genome.shrink.max(0.0));
                    contributions.entry(e.instrument).or_default().push((d.strategy_id.clone(), c, edge, hold));
                }
            }
        }
        let sleeves: BTreeMap<String, f64> = alloc.iter().map(|(k, w)| (k.clone(), w * equity.max(0.0))).collect();
        let mut abs_targets = Vec::new();
        for m in by_token.values_mut() {
            let Some(p) = m.last else { continue };
            let inst = instrument_id(&m.token);
            let target_eur = targets.get(&inst).copied().unwrap_or(0.0);
            abs_targets.push(target_eur.abs());
            let delta = target_eur / p - m.units;
            let notional = (delta * p).abs();
            if notional <= 1e-9 {
                continue;
            }
            let sigma = if m.rets.len() >= 2 {
                let mean = stats::mean(&m.rets);
                (m.rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (m.rets.len() - 1) as f64).sqrt()
                    * sigma_scale
            } else {
                0.0
            };
            let skip = match &spec.trade_rule {
                TradeRule::Fraction => notional <= spec.min_trade_fraction.max(0.0) * spec.capital,
                TradeRule::Band { frac, min_eur } => notional < min_eur.max(frac * target_eur.abs()),
                TradeRule::CostAware { frac, min_eur, .. } => {
                    let rb = rebalancer.as_mut().expect("cost-aware rule has a rebalancer");
                    let parts = contributions.get(&inst).map(Vec::as_slice).unwrap_or(&[]);
                    let weight: f64 = parts.iter().map(|(_, c, _, _)| c.abs()).sum();
                    let dir = if target_eur != 0.0 { target_eur.signum() } else { 0.0 };
                    let (edge, hold) = if weight > 0.0 {
                        (
                            parts.iter().map(|(_, c, e, _)| c.signum() * dir * c.abs() * e).sum::<f64>() / weight,
                            parts.iter().map(|(_, c, _, h)| c.abs() * h).sum::<f64>() / weight,
                        )
                    } else {
                        (0.0, 0.0)
                    };
                    let s = cost.schedule(&m.venue, &m.class);
                    let cand = TradeCandidate {
                        instrument: inst,
                        current_eur: m.units * p,
                        target_eur,
                        edge,
                        holding_secs: hold,
                        urgent: false,
                        limit_bound: false,
                        contributors: parts.iter().filter(|(_, c, _, _)| *c != 0.0).map(|(s, ..)| s.clone()).collect(),
                        cost: CostQuote {
                            commission_bps: s.commission_bps,
                            commission_min_eur: s.commission_min_eur,
                            half_spread_bps: s.spread_bps / 2.0,
                            impact_k: s.impact_k.unwrap_or(cost.default_k),
                            sigma_day: if sigma > 0.0 { sigma } else { cost.default_sigma },
                            adv_eur: cost.default_adv_eur,
                        },
                    };
                    let band = min_eur.max(frac * target_eur.abs());
                    !rb.decide(&cand, band, *min_eur, &sleeves, t).trades()
                }
            };
            if skip {
                continue;
            }
            let priced = cost.price(&CostInputs {
                venue: m.venue.clone(),
                class: m.class.clone(),
                side: if delta > 0.0 { "buy".into() } else { "sell".into() },
                price: p,
                qty: delta.abs(),
                sigma,
                ..Default::default()
            });
            cash -= delta * priced.fill_price + priced.fee;
            total_fees += priced.fee;
            let c = priced.total_cost(delta);
            step_cost += c;
            total_cost += c;
            m.units += delta;
            turnover_notional += notional;
            n_trades += 1;
        }
        let gross: f64 = abs_targets.iter().sum();
        if gross > 0.0 {
            hhi_sum += abs_targets.iter().map(|x| (x / gross).powi(2)).sum::<f64>();
            hhi_n += 1;
            if equity > 0.0 {
                gross_sum += gross / equity;
            }
        }
        t += spec.step_ms;
    }
    // final mark at the last grid point already happened; costs paid after it hit the equity now
    let mtm: f64 = by_token.values().map(|m| m.units * m.last.unwrap_or(0.0)).sum();
    let final_equity = cash + mtm;

    let n_obs = returns.len();
    let (sharpe, psr, dsr) = if n_obs > 2 {
        (
            stats::sharpe(&returns),
            stats::probabilistic_sharpe(&returns, 0.0),
            stats::deflated_sharpe(&returns, spec.n_trials.max(1), spec.trial_sharpe_var.max(0.0)),
        )
    } else {
        (0.0, 0.5, 0.5)
    };
    let gross_abs = gross_pnl.abs();
    let mean_weights = weight_sums.into_iter().map(|(k, v)| (k, v / n_active.max(1) as f64)).collect();
    Ok(AllocatorReport {
        genome: genome.to_string(),
        allocator: genome.name.to_string(),
        capital: spec.capital,
        cost_model: cost.id.clone(),
        n_steps,
        n_obs,
        n_active_steps: n_active,
        first_ts_ms: spec.window_ms.0,
        last_ts_ms: spec.window_ms.0 + (n_steps.saturating_sub(1) as i64) * spec.step_ms,
        stop: if n_active == 0 && stop == Stop::Complete { Stop::NoData } else { stop },
        n_signals,
        strategies: strategies.into_iter().collect(),
        instruments: by_token.keys().cloned().collect(),
        sharpe,
        psr,
        deflated_sharpe: dsr,
        n_trials: spec.n_trials.max(1),
        max_drawdown: max_dd,
        turnover: if n_steps > 0 { turnover_notional / spec.capital / n_steps as f64 } else { 0.0 },
        concentration: if hhi_n > 0 { hhi_sum / hhi_n as f64 } else { 0.0 },
        cost_share: if gross_abs + total_cost > 0.0 { total_cost / (gross_abs + total_cost) } else { 0.0 },
        mean_gross: if hhi_n > 0 { gross_sum / hhi_n as f64 } else { 0.0 },
        max_weight,
        n_trades,
        gross_pnl,
        total_cost,
        total_fees,
        rebalance: rebalancer.map(|r| r.stats),
        net_pnl: final_equity - spec.capital,
        final_equity,
        mean_weights,
        equity: equity_curve,
        returns,
    })
}

/// Allocator tournament (spec 40 §6.2): every genome replayed on identical signals and marks.
pub fn tournament(
    genomes: &[AllocatorGenome],
    signals: &[SignalRecord],
    marks: &[BarSeries],
    spec: &AllocatorReplaySpec,
    cost: &CostModel,
) -> Vec<(AllocatorGenome, Result<AllocatorReport, ReplayError>)> {
    genomes
        .iter()
        .map(|g| (g.clone(), replay_allocator(g, signals, marks, spec, cost, &ReplayControls::default())))
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use fa_allocator::genome::AllocatorName;
    use fa_core::rng::Rng;

    use super::*;

    pub const FIXTURE: &str = include_str!("../testdata/signal_fixture.jsonl");

    /// The real recording (`testdata/signal_fixture.jsonl`, trimmed from the live `SIGNAL`
    /// stream) with SYNTHETIC random-walk marks for its instruments over its time range.
    pub fn fixture() -> (Vec<SignalRecord>, Vec<BarSeries>, AllocatorReplaySpec) {
        let raw = fa_bus::replay::from_jsonl(FIXTURE).expect("fixture parses");
        let (signals, bad) = decode_signals(&raw);
        assert_eq!(bad, 0, "every fixture record is a signal");
        let t0 = signals.first().unwrap().published_ms / 1000 * 1000;
        let t1 = signals.last().unwrap().published_ms / 1000 * 1000 + 1000;
        let insts: BTreeSet<String> =
            signals.iter().flat_map(|s| s.dist.exposures.iter().map(|e| e.instrument.subject_token())).collect();
        let marks = insts
            .iter()
            .enumerate()
            .map(|(k, tok)| {
                let mut rng = Rng::seed(100 + k as u64);
                let mut p = 100.0;
                let pts: Vec<(i64, f64)> = (0..=((t1 - t0) / 1000 + 1))
                    .map(|i| {
                        p *= 1.0 + 0.001 * rng.normal();
                        (t0 - 1000 + i * 1000, p)
                    })
                    .collect();
                BarSeries::from_points(tok, "sim", &pts, 1000)
            })
            .collect();
        (signals, marks, AllocatorReplaySpec::new(t0, t1))
    }

    /// Writes `testdata/signal_fixture.jsonl` from the live `SIGNAL` stream:
    /// `FA_SIGNAL_FIXTURE_DUMP=1 cargo test -p fa-backtest -- --nocapture dump_signal_fixture`
    /// (`FA_SIGNAL_FIXTURE_MINUTES` back from now, default 3; at most 400 records; `FA_NATS_URL`).
    #[tokio::test]
    async fn dump_signal_fixture_from_live_stream() {
        if std::env::var("FA_SIGNAL_FIXTURE_DUMP").is_err() {
            return;
        }
        let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
        let js = fa_bus::async_nats::jetstream::new(fa_bus::async_nats::connect(&url).await.expect("connect"));
        let now = i64::try_from(std::time::UNIX_EPOCH.elapsed().unwrap().as_millis()).unwrap();
        let minutes: i64 = std::env::var("FA_SIGNAL_FIXTURE_MINUTES").ok().and_then(|m| m.parse().ok()).unwrap_or(3);
        let opts = fa_bus::replay::ReadOptions { max_messages: 400, ..Default::default() };
        let mut got = fa_bus::replay::read_range(&js, SIGNAL_STREAM, SIGNAL_FILTER, now - minutes * 60_000, now, &opts)
            .await
            .expect("read");
        fa_bus::replay::sort_records(&mut got);
        let (sig, bad) = decode_signals(&got);
        let strategies: BTreeSet<&str> = sig.iter().map(|s| s.dist.strategy_id.as_str()).collect();
        eprintln!("{} records, {bad} undecodable, strategies {strategies:?}", got.len());
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/signal_fixture.jsonl");
        std::fs::write(path, fa_bus::replay::to_jsonl(&got)).expect("write fixture");
    }

    #[test]
    fn horizon_and_decode() {
        assert_eq!(parse_horizon("5x1m"), Horizon { timescale: Timescale::Minute, steps: 5 });
        assert_eq!(parse_horizon("junk"), Horizon { timescale: Timescale::Second, steps: 1 });
        let d = ExposureDistribution {
            strategy_id: "s1".into(),
            genome_id: "g".into(),
            horizon: Horizon { timescale: Timescale::Second, steps: 3 },
            as_of_ms: 1_234,
            exposures: vec![AssetExposure { instrument: InstrumentId::new(), mean: 0.5, stddev: 0.1 }],
        };
        let raw = RecordedMessage {
            subject: "signal.exposure.s1".into(),
            stream_seq: 7,
            published_ms: 2_000,
            payload: d.to_proto().encode_to_vec(),
        };
        let r = SignalRecord::decode(&raw).unwrap();
        assert_eq!((r.published_ms, r.dist.as_of_ms, r.dist.horizon.steps), (2_000, 1_234, 3));
        assert_eq!(r.dist.exposures[0].instrument, d.exposures[0].instrument);
        assert!(SignalRecord::decode(&RecordedMessage { subject: "order.x".into(), ..raw.clone() }).is_none());
        let (v, bad) = decode_signals(&[raw.clone(), RecordedMessage { payload: vec![0xff, 0xff], ..raw }]);
        assert_eq!((v.len(), bad), (1, 1));
    }

    #[test]
    fn replay_on_the_fixture_is_deterministic_and_comparable() {
        let (signals, marks, spec) = fixture();
        assert!(signals.len() >= 50, "{}", signals.len());
        let cost = CostModel::default();
        let g = AllocatorGenome::parse("alloc:name=risk_parity;cap=0.4;hl=30").unwrap();
        let a = replay_allocator(&g, &signals, &marks, &spec, &cost, &ReplayControls::default()).unwrap();
        let b = replay_allocator(&g, &signals, &marks, &spec, &cost, &ReplayControls::default()).unwrap();
        assert_eq!(a, b, "identical spec + genome + data → identical report");
        eprintln!(
            "fixture replay: steps {} active {} signals {} strategies {:?} trades {} sharpe {:.4} dsr {:.4} dd {:.5} hhi {:.3} cost_share {:.3}",
            a.n_steps,
            a.n_active_steps,
            a.n_signals,
            a.strategies,
            a.n_trades,
            a.sharpe,
            a.deflated_sharpe,
            a.max_drawdown,
            a.concentration,
            a.cost_share
        );
        assert!(a.n_active_steps > 0 && a.strategies.len() >= 2, "{a:?}");
        assert!(a.n_trades > 0 && a.total_cost > 0.0);
        assert!(a.max_weight <= 0.4 + 1e-12);
        assert!((0.0..=1.0).contains(&a.concentration) && (0.0..=1.0).contains(&a.cost_share));
        assert!(a.objectives().iter().all(|(_, v)| v.is_finite()));
        let total: f64 = a.returns.iter().sum::<f64>() * spec.capital;
        assert!((total - (a.equity.last().unwrap().1 - spec.capital)).abs() < 1e-6, "returns sum to the equity curve");
        // tournament: every allocator on identical inputs, results differ where the policy differs
        let genomes: Vec<AllocatorGenome> = AllocatorName::ALL.iter().map(|n| AllocatorGenome::named(*n)).collect();
        let t = tournament(&genomes, &signals, &marks, &spec, &cost);
        assert_eq!(t.len(), AllocatorName::ALL.len());
        let reports: Vec<&AllocatorReport> = t.iter().map(|(_, r)| r.as_ref().unwrap()).collect();
        assert!(reports.iter().all(|r| r.n_steps == a.n_steps && r.n_signals == a.n_signals));
        assert_ne!(reports[0].returns, reports[2].returns, "equal weight vs Kelly decide differently");
        // stop hooks
        let ctl = ReplayControls { max_steps: Some(10), ..Default::default() };
        let r = replay_allocator(&g, &signals, &marks, &spec, &cost, &ctl).unwrap();
        assert_eq!((r.n_steps, r.stop), (10, Stop::MaxSteps));
        let ctl = ReplayControls { should_stop: Some(Box::new(|n| n >= 3)), ..Default::default() };
        assert_eq!(replay_allocator(&g, &signals, &marks, &spec, &cost, &ctl).unwrap().stop, Stop::Interrupted);
        // no signals / no marks
        assert!(matches!(
            replay_allocator(&g, &[], &marks, &spec, &cost, &ReplayControls::default()),
            Err(ReplayError::NoData)
        ));
        assert!(matches!(
            replay_allocator(&g, &signals, &[], &spec, &cost, &ReplayControls::default()),
            Err(ReplayError::NoData)
        ));
    }

    #[test]
    fn signals_after_the_step_are_invisible_and_ttl_expires() {
        // SYNTHETIC: one strategy long 1.0 from t = 5 s; marks rise 1 % per step.
        let inst = InstrumentId::new();
        let tok = inst.subject_token();
        let sig = |pub_ms: i64, mean: f64| SignalRecord {
            published_ms: pub_ms,
            stream_seq: pub_ms as u64,
            dist: ExposureDistribution {
                strategy_id: "s".into(),
                genome_id: "g".into(),
                horizon: Horizon { timescale: Timescale::Second, steps: 1 },
                as_of_ms: pub_ms,
                exposures: vec![AssetExposure { instrument: inst, mean, stddev: 0.0 }],
            },
        };
        let pts: Vec<(i64, f64)> = (0..30).map(|i| (i * 1000 - 1, 100.0 * 1.01f64.powi(i as i32))).collect();
        let marks = vec![BarSeries::from_points(&tok, "sim", &pts, 1000)];
        let g = AllocatorGenome::parse("alloc:name=equal_weight;cap=1").unwrap();
        let spec = AllocatorReplaySpec { signal_ttl_ms: 4_000, ..AllocatorReplaySpec::new(0, 20_000) };
        let r =
            replay_allocator(&g, &[sig(5_050, 1.0)], &marks, &spec, &CostModel::default(), &ReplayControls::default())
                .unwrap();
        // no exposure before the step at 6 s (signal published 5.05 s), none after the TTL (9.05 s)
        let first_move = r.returns.iter().position(|x| x.abs() > 0.0).unwrap();
        // returns[k] is the return earned from step k s to k+1 s: entered at 6 s → index 6
        assert_eq!(first_move, 6, "{:?}", r.returns);
        assert!(r.returns[..6].iter().all(|x| *x == 0.0));
        assert_eq!(r.n_active_steps, 4, "6, 7, 8, 9 s");
        assert!(r.returns[12..].iter().all(|x| *x == 0.0), "flat after the TTL");
        assert!(r.gross_pnl > 0.0 && r.net_pnl > 0.0, "{r:?}");
        assert_eq!(r.concentration, 1.0, "single instrument");
    }
}
