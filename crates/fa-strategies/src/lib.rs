//! Strategies: GP programs and neural policies competing in one population (spec 35).
//!
//! Every strategy emits a distribution over target exposure per asset. GP primitives
//! read only graph ops, decoder outputs and portfolio state (no classic TA).
//!
//! - [`sexpr`]: text syntax for GP programs (`(mul 4 (sub (path_up 3) 0.5))`).
//! - [`features`]: the live observation ([`features::LiveContext`]): graph and decoder
//!   primitives on real bus data, price-only fallbacks otherwise, with per-source counters.
//! - [`graph`]: [`graph::LatentGraph`], the point-in-time latent region graph built from
//!   `latent.state.>` / `latent.region.>` / `latent.transition.>` over `fa_mcmc::MarkovGraph`.
//! - [`decoders`]: [`decoders::DecoderOutputs`], the recency cache of `decoder.output.<family>`.
//! - [`population`]: `strategies.toml` definitions and the hand-written baseline set.
//! - [`neural`]: neural policies ([`neural::PolicyNet`], [`neural::NeuralStrategy`]), their
//!   text form and variation; SGD fine-tuning on candle behind the `train` feature.
//! - [`genome`], [`random`], [`ops`]: strategy genomes (GP or neural, asset set, snapshot
//!   binding), random programs and the typed GP operators; [`assets`]: asset-set mutation over
//!   the MCMC pair proposals; [`remap`]: region literals across snapshot changes.

pub mod assets;
pub mod decoders;
pub mod features;
pub mod genome;
pub mod graph;
pub mod neural;
pub mod ops;
pub mod population;
pub mod random;
pub mod remap;
pub mod sexpr;

use fa_core::{Horizon, InstrumentId};
use serde::{Deserialize, Serialize};

/// Target exposure for one asset: mean in [-1, 1] of the strategy's sleeve, plus uncertainty.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AssetExposure {
    pub instrument: InstrumentId,
    pub mean: f64,
    pub stddev: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExposureDistribution {
    pub strategy_id: String,
    pub genome_id: String,
    pub horizon: Horizon,
    pub as_of_ms: i64,
    pub exposures: Vec<AssetExposure>,
}

impl ExposureDistribution {
    pub fn to_proto(&self) -> fa_proto::v1::ExposureDistribution {
        fa_proto::v1::ExposureDistribution {
            strategy_id: self.strategy_id.clone(),
            genome_id: self.genome_id.clone(),
            horizon: format!("{}x{}", self.horizon.steps, self.horizon.timescale),
            as_of: Some(fa_proto::Timestamp {
                seconds: self.as_of_ms.div_euclid(1000),
                nanos: (self.as_of_ms.rem_euclid(1000) * 1_000_000) as i32,
            }),
            exposures: self
                .exposures
                .iter()
                .map(|e| fa_proto::v1::AssetExposure {
                    instrument_id: e.instrument.to_string(),
                    mean: e.mean,
                    stddev: e.stddev,
                })
                .collect(),
        }
    }
}

/// A region-typed sub-expression (spec 35 §4.2 `Region`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionRef {
    /// `(cur_region expert)`: the asset's current region in the given expert space.
    Cur(u32),
    /// `(region expert ordinal)`: a fixed region literal — region `<prefix>-e<expert>-r<nnn>` of
    /// the encoder snapshot the strategy is bound to (spec 35 §6.5). It resolves only while that
    /// snapshot is live; on a snapshot change [`remap`] rewrites it through the region lineage.
    Lit(u32, u32),
}

impl RegionRef {
    /// Expert space the reference lives in.
    pub fn expert(self) -> u32 {
        match self {
            Self::Cur(e) | Self::Lit(e, _) => e,
        }
    }
}

/// A region-set-typed sub-expression (`RegionSet`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SetRef {
    /// `(neighbors r n)`: the `n` most likely successors of `r` by transition mass.
    Neighbors(RegionRef, u32),
}

/// Per-region statistics accumulated online from realised returns (`region_feature(r, f)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionFeature {
    /// Share of observed bars spent in the region.
    Mass,
    /// Mean realised one-bar log return while in the region.
    MeanFwdReturn,
    /// Std of that return.
    Vol,
}

/// Function applied to the end region of sampled paths (`path_value(paths, fn)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PathFn {
    /// 1 when the end region's mean forward return is positive.
    Up,
    Feature(RegionFeature),
}

/// Comparison of a `Bool` node (`(gt a b)`, `(lt a b)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CmpOp {
    Gt,
    Lt,
}

/// A `Prob`-typed distribution `prob_gt` reads (spec 35 §4.2 `prob_gt(Prob, Num)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProbSrc {
    /// `(fc_dist k)`: the elite forecast distribution `k` bars ahead (its quantile vector).
    Fc(u32),
    /// `(analog_outcome k)`: the retrieval analogs' realised `k`-bar outcome, as a normal around
    /// its similarity-weighted mean with the forecast's inter-quartile spread.
    Analog(u32),
}

/// Forecast quantile levels the default `prob_gt` CDF is interpolated over.
pub const PROB_LEVELS: [f64; 5] = [0.05, 0.25, 0.5, 0.75, 0.95];

/// Standard normal CDF (Abramowitz–Stegun 7.1.26 erf, |err| < 1.5e-7).
pub fn norm_cdf(x: f64) -> f64 {
    let z = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let poly = t * (0.254_829_592 + t * (-0.284_496_736 + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-z * z).exp();
    if x >= 0.0 { 0.5 * (1.0 + erf) } else { 0.5 * (1.0 - erf) }
}

/// `P(X > x)` of a distribution given by quantile values at [`PROB_LEVELS`]: piecewise-linear
/// CDF between the quantiles, linear tails to 0 / 1 one inter-quantile step beyond the outer
/// ones. Unsorted inputs are sorted (crossing quantiles); a degenerate distribution is a step.
pub fn prob_gt_from_quantiles(q: &[f64; 5], x: f64) -> f64 {
    let mut v = *q;
    if v.iter().any(|a| !a.is_finite()) || !x.is_finite() {
        return 0.5;
    }
    v.sort_by(f64::total_cmp);
    let l = PROB_LEVELS;
    let cdf = if x <= v[0] {
        let w = (v[1] - v[0]).max(1e-12) * (l[0] / (l[1] - l[0]));
        (l[0] * (1.0 - (v[0] - x) / w)).max(0.0)
    } else if x >= v[4] {
        let w = (v[4] - v[3]).max(1e-12) * ((1.0 - l[4]) / (l[4] - l[3]));
        (l[4] + (1.0 - l[4]) * ((x - v[4]) / w)).min(1.0)
    } else {
        let i = (0..4).find(|&i| x <= v[i + 1]).unwrap_or(3);
        let span = v[i + 1] - v[i];
        if span <= 1e-15 { l[i + 1] } else { l[i] + (l[i + 1] - l[i]) * (x - v[i]) / span }
    };
    (1.0 - cdf).clamp(0.0, 1.0)
}

/// Read-only view a strategy sees at a decision step for one asset.
///
/// The spec 35 §4.2 primitives (`cur_region`, `neighbors`, `p_trans`, …) have default
/// implementations in terms of the original primitives so simple contexts keep working;
/// [`features::LiveContext`] overrides them with real graph / decoder data.
pub trait StrategyContext {
    // graph ops (spec 33)
    fn current_region(&self) -> Option<&str>;
    fn transition_prob(&self, to_region: &str) -> f64;
    fn stay_prob(&self) -> f64;
    /// Probability mass of sampled k-step paths ending in a region whose analog outcome is positive.
    fn path_up_prob(&self, k: u32) -> f64;
    fn region_entropy(&self) -> f64;
    /// `cur_region(expert)` → region id of this asset in expert space `expert`.
    fn cur_region(&self, expert: u32) -> Option<String> {
        if expert == 0 { self.current_region().map(str::to_owned) } else { None }
    }
    /// `neighbors(r, n)` → successor ids ordered by transition mass.
    fn neighbors(&self, _r: RegionRef, _n: u32) -> Vec<String> {
        Vec::new()
    }
    /// `p_trans(a, b, k)` → posterior-mean probability of `a → b` in exactly `k` steps.
    fn p_trans(&self, a: RegionRef, b: RegionRef, k: u32) -> f64 {
        if a == b { self.stay_prob().powi(k.max(1) as i32) } else { 0.0 }
    }
    /// `p_to_set(r, S, k)` → k-step mass from `r` into `S`.
    fn p_to_set(&self, _r: RegionRef, _s: SetRef, _k: u32) -> f64 {
        0.0
    }
    /// `hitting_time(r, S)` → expected first-hitting step of `S` from `r` over sampled walks
    /// (capped at the walk horizon).
    fn hitting_time(&self, _r: RegionRef, _s: SetRef) -> f64 {
        HITTING_HORIZON as f64
    }
    /// `dwell(r)` → bars this asset has spent in `r` if `r` is its current region, else 0.
    fn dwell(&self, _r: RegionRef) -> f64 {
        0.0
    }
    /// `novelty(expert)` → how unfamiliar the current latent is (posterior std / unknown mass).
    fn novelty(&self, _expert: u32) -> f64 {
        self.region_entropy()
    }
    fn region_feature(&self, _r: RegionRef, f: RegionFeature) -> f64 {
        match f {
            RegionFeature::Mass => 0.0,
            RegionFeature::MeanFwdReturn => self.analog_forecast(),
            RegionFeature::Vol => (self.forecast_quantile(0.75) - self.forecast_quantile(0.25)).abs(),
        }
    }
    /// `path_value(sample_walk(r, k, n), fn)` → mean of `fn` over the end regions of `n` sampled k-step walks.
    fn path_value(&self, _r: RegionRef, k: u32, _n: u32, f: PathFn) -> f64 {
        match f {
            PathFn::Up => self.path_up_prob(k),
            PathFn::Feature(f) => self.region_feature(RegionRef::Cur(0), f),
        }
    }
    // decoder outputs (spec 34)
    fn forecast_quantile(&self, level: f64) -> f64;
    fn analog_forecast(&self) -> f64;
    fn decoder_score(&self, family: &str) -> f64;
    /// `fc_quantile(asset, q, k)`: forecast quantile `q` of the elite forecast decoder.
    fn fc_quantile(&self, q: f64, _k: u32) -> f64 {
        self.forecast_quantile(q)
    }
    /// `fc_dist(asset, k)`: expected value of the forecast distribution.
    fn fc_dist(&self, _k: u32) -> f64 {
        self.analog_forecast()
    }
    /// `analog_outcome(asset, k)`: similarity-weighted realised k-bar outcome of the retrieval analogs.
    fn analog_outcome(&self, _k: u32) -> f64 {
        self.analog_forecast()
    }
    /// `analog_age(asset)`: mean time distance of the analogs in days.
    fn analog_age(&self) -> f64 {
        0.0
    }
    /// `text_score(topic)`: score in [-1, 1] of the latest `text.<topic>` decoder output
    /// (news reconstruction / narrative similarity); contexts without decoders answer 0.
    fn text_score(&self, _topic: &str) -> f64 {
        0.0
    }
    /// `visual_score(kind)`: score in [-1, 1] of the latest `visual.<kind>` decoder output
    /// (chart reconstruction, future-chart direction); contexts without decoders answer 0.
    fn visual_score(&self, _kind: &str) -> f64 {
        0.0
    }
    /// `prob_gt(P, x)`: probability that the distribution `P` exceeds `x` (see [`ProbSrc`]).
    fn prob_gt(&self, src: ProbSrc, x: f64) -> f64 {
        match src {
            ProbSrc::Fc(k) => {
                let q = PROB_LEVELS.map(|l| self.fc_quantile(l, k));
                prob_gt_from_quantiles(&q, x)
            }
            ProbSrc::Analog(k) => {
                let mean = self.analog_outcome(k);
                let iqr = (self.fc_quantile(0.75, k) - self.fc_quantile(0.25, k)).abs();
                let sd = iqr / 1.349;
                if sd > 1e-15 && sd.is_finite() && mean.is_finite() {
                    1.0 - norm_cdf((x - mean) / sd)
                } else if mean > x {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
    /// Latent mean of this asset in expert space `expert` (spec 35 §4.3 policy input), as of
    /// the step; `None` without a latent state.
    fn latent(&self, _expert: u32) -> Option<Vec<f32>> {
        None
    }
    // portfolio state
    fn position(&self) -> f64;
    fn unrealised_pnl(&self) -> f64;
    fn time_in_trade(&self) -> f64;
    /// `exposure_gross()`: Σ |exposure| over the strategy's whole asset set (fraction of its
    /// sleeve); contexts without the strategy-level book answer this asset's `|position|`.
    fn exposure_gross(&self) -> f64 {
        self.position().abs()
    }
    /// `drawdown()`: the strategy's virtual-book drawdown from its peak, in sleeve-return units
    /// (≥ 0); 0 where the strategy-level book is unknown.
    fn drawdown(&self) -> f64 {
        0.0
    }
}

/// Walk horizon (steps) used by `hitting_time`; walks that never hit count as this.
pub const HITTING_HORIZON: u32 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GraphOp {
    StayProb,
    PathUpProb(u32),
    RegionEntropy,
    /// `(p_trans a b k)`
    PTrans(RegionRef, RegionRef, u32),
    /// `(p_to_set r S k)`
    PToSet(RegionRef, SetRef, u32),
    /// `(hitting_time r S)`
    HittingTime(RegionRef, SetRef),
    /// `(dwell r)`
    Dwell(RegionRef),
    /// `(novelty expert)`
    Novelty(u32),
    /// `(region_feature r f)`
    RegionFeature(RegionRef, RegionFeature),
    /// `(path_value (sample_walk r k n) f)`
    PathValue(RegionRef, u32, u32, PathFn),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecoderOp {
    Quantile(f64),
    AnalogForecast,
    Score(String),
    /// `(fc_quantile q k)` (asset = the evaluated instrument)
    FcQuantile(f64, u32),
    /// `(fc_dist k)`
    FcDist(u32),
    /// `(analog_outcome k)`
    AnalogOutcome(u32),
    /// `analog_age`
    AnalogAge,
    /// `(text_score topic)`
    TextScore(String),
    /// `(visual_score kind)`
    VisualScore(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PortfolioOp {
    Position,
    UnrealisedPnl,
    TimeInTrade,
    /// `exposure_gross`
    ExposureGross,
    /// `drawdown`
    Drawdown,
}

/// GP DSL expression tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Const(f64),
    Graph(GraphOp),
    Decoder(DecoderOp),
    Portfolio(PortfolioOp),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    /// Protected division: returns 0 when the denominator is ~0.
    Div(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Tanh(Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
    /// `if a > b then c else d`
    IfGt(Box<Expr>, Box<Expr>, Box<Expr>, Box<Expr>),
    /// `(clip x lo hi)`: `x` limited to `[min(lo, hi), max(lo, hi)]`.
    Clip(Box<Expr>, Box<Expr>, Box<Expr>),
    /// `(if c a b)` with a `Bool`-typed condition `c`.
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    /// `(prob_gt P x)` → probability (a `Num` in [0, 1]).
    ProbGt(ProbSrc, Box<Expr>),
    // ── Bool-typed nodes (only valid as conditions / operands of Bool nodes) ──
    /// `(gt a b)` / `(lt a b)`
    Cmp(CmpOp, Box<Expr>, Box<Expr>),
    /// `(and p q)`
    And(Box<Expr>, Box<Expr>),
    /// `(or p q)`
    Or(Box<Expr>, Box<Expr>),
    /// `(not p)`
    Not(Box<Expr>),
}

thread_local! {
    static NON_FINITE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Non-finite intermediate values (NaN / ±inf) [`Expr::eval`] replaced by 0 on this thread
/// since the last [`take_non_finite`]. Evaluation stays total (a NaN node yields 0), but the
/// runtime reads this counter around a step to treat the step as a NaN output (spec 35 §8).
pub fn take_non_finite() -> u64 {
    NON_FINITE.with(|c| c.replace(0))
}

impl Expr {
    pub fn eval(&self, ctx: &dyn StrategyContext) -> f64 {
        use Expr::*;
        let v = match self {
            Const(c) => *c,
            Graph(GraphOp::StayProb) => ctx.stay_prob(),
            Graph(GraphOp::PathUpProb(k)) => ctx.path_up_prob(*k),
            Graph(GraphOp::RegionEntropy) => ctx.region_entropy(),
            Graph(GraphOp::PTrans(a, b, k)) => ctx.p_trans(*a, *b, *k),
            Graph(GraphOp::PToSet(r, s, k)) => ctx.p_to_set(*r, *s, *k),
            Graph(GraphOp::HittingTime(r, s)) => ctx.hitting_time(*r, *s),
            Graph(GraphOp::Dwell(r)) => ctx.dwell(*r),
            Graph(GraphOp::Novelty(e)) => ctx.novelty(*e),
            Graph(GraphOp::RegionFeature(r, f)) => ctx.region_feature(*r, *f),
            Graph(GraphOp::PathValue(r, k, n, f)) => ctx.path_value(*r, *k, *n, *f),
            Decoder(DecoderOp::Quantile(l)) => ctx.forecast_quantile(*l),
            Decoder(DecoderOp::AnalogForecast) => ctx.analog_forecast(),
            Decoder(DecoderOp::Score(f)) => ctx.decoder_score(f),
            Decoder(DecoderOp::FcQuantile(q, k)) => ctx.fc_quantile(*q, *k),
            Decoder(DecoderOp::FcDist(k)) => ctx.fc_dist(*k),
            Decoder(DecoderOp::AnalogOutcome(k)) => ctx.analog_outcome(*k),
            Decoder(DecoderOp::AnalogAge) => ctx.analog_age(),
            Decoder(DecoderOp::TextScore(t)) => ctx.text_score(t),
            Decoder(DecoderOp::VisualScore(k)) => ctx.visual_score(k),
            Portfolio(PortfolioOp::Position) => ctx.position(),
            Portfolio(PortfolioOp::UnrealisedPnl) => ctx.unrealised_pnl(),
            Portfolio(PortfolioOp::TimeInTrade) => ctx.time_in_trade(),
            Portfolio(PortfolioOp::ExposureGross) => ctx.exposure_gross(),
            Portfolio(PortfolioOp::Drawdown) => ctx.drawdown(),
            Add(a, b) => a.eval(ctx) + b.eval(ctx),
            Sub(a, b) => a.eval(ctx) - b.eval(ctx),
            Mul(a, b) => a.eval(ctx) * b.eval(ctx),
            Div(a, b) => {
                let d = b.eval(ctx);
                if d.abs() < 1e-12 { 0.0 } else { a.eval(ctx) / d }
            }
            Neg(a) => -a.eval(ctx),
            Tanh(a) => a.eval(ctx).tanh(),
            Min(a, b) => a.eval(ctx).min(b.eval(ctx)),
            Max(a, b) => a.eval(ctx).max(b.eval(ctx)),
            IfGt(a, b, c, d) => {
                if a.eval(ctx) > b.eval(ctx) {
                    c.eval(ctx)
                } else {
                    d.eval(ctx)
                }
            }
            Clip(x, lo, hi) => {
                let (lo, hi) = (lo.eval(ctx), hi.eval(ctx));
                x.eval(ctx).clamp(lo.min(hi), lo.max(hi))
            }
            If(c, a, b) => {
                if c.truth(ctx) {
                    a.eval(ctx)
                } else {
                    b.eval(ctx)
                }
            }
            ProbGt(src, x) => ctx.prob_gt(*src, x.eval(ctx)),
            Cmp(..) | And(..) | Or(..) | Not(_) => f64::from(u8::from(self.truth(ctx))),
        };
        if v.is_finite() {
            v
        } else {
            NON_FINITE.with(|c| c.set(c.get() + 1));
            0.0
        }
    }

    /// Truth value of a `Bool` node (a `Num` node is true when > 0). Comparisons with a
    /// non-finite operand are false.
    pub fn truth(&self, ctx: &dyn StrategyContext) -> bool {
        use Expr::*;
        match self {
            Cmp(op, a, b) => {
                let (x, y) = (a.eval(ctx), b.eval(ctx));
                match op {
                    CmpOp::Gt => x > y,
                    CmpOp::Lt => x < y,
                }
            }
            And(p, q) => p.truth(ctx) && q.truth(ctx),
            Or(p, q) => p.truth(ctx) || q.truth(ctx),
            Not(p) => !p.truth(ctx),
            other => other.eval(ctx) > 0.0,
        }
    }

    /// Whether this node is `Bool`-typed (spec 35 §4.2 STGP types; everything else is `Num`).
    pub fn is_bool(&self) -> bool {
        matches!(self, Expr::Cmp(..) | Expr::And(..) | Expr::Or(..) | Expr::Not(_))
    }

    pub fn size(&self) -> usize {
        use Expr::*;
        match self {
            Const(_) | Graph(_) | Decoder(_) | Portfolio(_) => 1,
            Neg(a) | Tanh(a) | Not(a) | ProbGt(_, a) => 1 + a.size(),
            Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Min(a, b) | Max(a, b) | Cmp(_, a, b) | And(a, b)
            | Or(a, b) => 1 + a.size() + b.size(),
            Clip(a, b, c) | If(a, b, c) => 1 + a.size() + b.size() + c.size(),
            IfGt(a, b, c, d) => 1 + a.size() + b.size() + c.size() + d.size(),
        }
    }
}

/// Estimated floating-point operations of one evaluation of `e` for one asset: the
/// representation-independent compute unit of the `compute_cost` objective (spec 35 §6.2, §10
/// Q3). Arithmetic and combinators 1, constants 0.5, a primitive read 10, a `k`-step kernel query
/// `10·k`, sampled walks `10·k·n`, the forecast-CDF `prob_gt` 50.
pub fn op_cost(e: &Expr) -> f64 {
    use Expr::*;
    let own = match e {
        Const(_) => 0.5,
        Graph(GraphOp::PTrans(_, _, k)) | Graph(GraphOp::PToSet(_, _, k)) => 10.0 * f64::from((*k).max(1)),
        Graph(GraphOp::PathUpProb(k)) => 10.0 * f64::from((*k).max(1)),
        Graph(GraphOp::PathValue(_, k, n, _)) => 10.0 * f64::from((*k).max(1)) * f64::from((*n).max(1)),
        Graph(GraphOp::HittingTime(..)) => 10.0 * f64::from(HITTING_HORIZON) * 8.0,
        Decoder(DecoderOp::AnalogOutcome(k)) => 10.0 * f64::from((*k).max(1)),
        Graph(_) | Decoder(_) | Portfolio(_) => 10.0,
        ProbGt(..) => 50.0,
        _ => 1.0,
    };
    own + match e {
        Const(_) | Graph(_) | Decoder(_) | Portfolio(_) => 0.0,
        Neg(a) | Tanh(a) | Not(a) | ProbGt(_, a) => op_cost(a),
        Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Min(a, b) | Max(a, b) | Cmp(_, a, b) | And(a, b)
        | Or(a, b) => op_cost(a) + op_cost(b),
        Clip(a, b, c) | If(a, b, c) => op_cost(a) + op_cost(b) + op_cost(c),
        IfGt(a, b, c, d) => op_cost(a) + op_cost(b) + op_cost(c) + op_cost(d),
    }
}

/// A strategy step for one asset set.
pub trait Strategy: Send {
    fn id(&self) -> &str;
    fn horizon(&self) -> Horizon;
    fn step(&mut self, as_of_ms: i64, assets: &[(InstrumentId, &dyn StrategyContext)]) -> ExposureDistribution;
    /// Estimated operations per asset and step ([`op_cost`] units), for the `compute_cost`
    /// objective shared by both representations.
    fn ops_per_asset(&self) -> f64 {
        1.0
    }
    /// Genome id carried by the emitted distributions.
    fn genome_id(&self) -> &str {
        self.id()
    }
}

/// GP strategy: `mean = tanh(signal)`, `stddev = sigmoid(uncertainty)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpStrategy {
    pub id: String,
    pub genome_id: String,
    pub horizon: Horizon,
    pub signal: Expr,
    pub uncertainty: Expr,
}

impl Strategy for GpStrategy {
    fn id(&self) -> &str {
        &self.id
    }

    fn horizon(&self) -> Horizon {
        self.horizon
    }

    fn ops_per_asset(&self) -> f64 {
        op_cost(&self.signal) + op_cost(&self.uncertainty)
    }

    fn genome_id(&self) -> &str {
        &self.genome_id
    }

    fn step(&mut self, as_of_ms: i64, assets: &[(InstrumentId, &dyn StrategyContext)]) -> ExposureDistribution {
        let exposures = assets
            .iter()
            .map(|(inst, ctx)| AssetExposure {
                instrument: *inst,
                mean: self.signal.eval(*ctx).tanh(),
                stddev: 1.0 / (1.0 + (-self.uncertainty.eval(*ctx)).exp()),
            })
            .collect();
        ExposureDistribution {
            strategy_id: self.id.clone(),
            genome_id: self.genome_id.clone(),
            horizon: self.horizon,
            as_of_ms,
            exposures,
        }
    }
}

/// Neural policy: maps a feature vector built from the same primitives to (mean, stddev).
/// Implemented by [`neural::PolicyNet`] (a small MLP, run as [`neural::NeuralStrategy`]).
pub trait NeuralPolicy: Send {
    fn forward(&self, features: &[f32]) -> (f64, f64);
}

impl NeuralPolicy for neural::PolicyNet {
    fn forward(&self, features: &[f32]) -> (f64, f64) {
        let o = neural::PolicyNet::forward(self, features);
        (o.mean, o.std)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fa_core::Timescale;

    struct Ctx {
        up: f64,
        pos: f64,
    }

    impl StrategyContext for Ctx {
        fn current_region(&self) -> Option<&str> {
            Some("r1")
        }
        fn transition_prob(&self, _: &str) -> f64 {
            0.0
        }
        fn stay_prob(&self) -> f64 {
            0.5
        }
        fn path_up_prob(&self, _: u32) -> f64 {
            self.up
        }
        fn region_entropy(&self) -> f64 {
            1.0
        }
        fn forecast_quantile(&self, l: f64) -> f64 {
            l - 0.5
        }
        fn analog_forecast(&self) -> f64 {
            0.01
        }
        fn decoder_score(&self, _: &str) -> f64 {
            0.0
        }
        fn position(&self) -> f64 {
            self.pos
        }
        fn unrealised_pnl(&self) -> f64 {
            0.0
        }
        fn time_in_trade(&self) -> f64 {
            0.0
        }
    }

    fn b(e: Expr) -> Box<Expr> {
        Box::new(e)
    }

    #[test]
    fn interpreter_evaluates_primitives_and_ops() {
        let ctx = Ctx { up: 0.8, pos: 0.0 };
        // 4 * (path_up(3) - 0.5)
        let e =
            Expr::Mul(b(Expr::Const(4.0)), b(Expr::Sub(b(Expr::Graph(GraphOp::PathUpProb(3))), b(Expr::Const(0.5)))));
        assert!((e.eval(&ctx) - 1.2).abs() < 1e-12);
        assert_eq!(Expr::Div(b(Expr::Const(1.0)), b(Expr::Const(0.0))).eval(&ctx), 0.0);
        let cond = Expr::IfGt(
            b(Expr::Portfolio(PortfolioOp::Position)),
            b(Expr::Const(0.0)),
            b(Expr::Const(1.0)),
            b(Expr::Const(-1.0)),
        );
        assert_eq!(cond.eval(&ctx), -1.0);
        assert_eq!(e.size(), 5);
        take_non_finite();
        let inf = Expr::Mul(b(Expr::Const(1e308)), b(Expr::Const(1e308)));
        assert_eq!(inf.eval(&ctx), 0.0);
        assert_eq!(take_non_finite(), 1);
        assert_eq!(take_non_finite(), 0);
    }

    #[test]
    fn spec_primitives_default_to_legacy_fallbacks() {
        let ctx = Ctx { up: 0.8, pos: 0.0 };
        let r = RegionRef::Cur(0);
        assert_eq!(ctx.cur_region(0).as_deref(), Some("r1"));
        assert_eq!(ctx.cur_region(1), None);
        assert_eq!(Expr::Graph(GraphOp::PTrans(r, r, 2)).eval(&ctx), 0.25);
        assert_eq!(Expr::Graph(GraphOp::PToSet(r, SetRef::Neighbors(r, 3), 1)).eval(&ctx), 0.0);
        assert_eq!(
            Expr::Graph(GraphOp::HittingTime(r, SetRef::Neighbors(r, 1))).eval(&ctx),
            f64::from(HITTING_HORIZON)
        );
        assert_eq!(Expr::Graph(GraphOp::PathValue(r, 3, 10, PathFn::Up)).eval(&ctx), 0.8);
        assert_eq!(Expr::Graph(GraphOp::RegionFeature(r, RegionFeature::MeanFwdReturn)).eval(&ctx), 0.01);
        assert_eq!(Expr::Decoder(DecoderOp::FcQuantile(0.75, 1)).eval(&ctx), 0.25);
        assert_eq!(Expr::Decoder(DecoderOp::TextScore("news".into())).eval(&ctx), 0.0);
    }

    #[test]
    fn gp_strategy_emits_bounded_exposure() {
        let mut s = GpStrategy {
            id: "s1".into(),
            genome_id: "g1".into(),
            horizon: Horizon { timescale: Timescale::Hour, steps: 2 },
            signal: Expr::Mul(b(Expr::Const(100.0)), b(Expr::Decoder(DecoderOp::AnalogForecast))),
            uncertainty: Expr::Graph(GraphOp::RegionEntropy),
        };
        let ctx = Ctx { up: 0.5, pos: 0.0 };
        let i = InstrumentId::new();
        let d = s.step(1_000, &[(i, &ctx)]);
        let e = d.exposures[0];
        assert!(e.mean > 0.7 && e.mean <= 1.0);
        assert!(e.stddev > 0.5 && e.stddev < 1.0);
        let p = d.to_proto();
        assert_eq!(p.horizon, "2x1h");
        assert_eq!(p.exposures[0].instrument_id, i.to_string());
    }
}
