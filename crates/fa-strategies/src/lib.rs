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

pub mod decoders;
pub mod features;
pub mod genome;
pub mod graph;
pub mod ops;
pub mod population;
pub mod random;
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
    // portfolio state
    fn position(&self) -> f64;
    fn unrealised_pnl(&self) -> f64;
    fn time_in_trade(&self) -> f64;
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
        };
        if v.is_finite() {
            v
        } else {
            NON_FINITE.with(|c| c.set(c.get() + 1));
            0.0
        }
    }

    pub fn size(&self) -> usize {
        use Expr::*;
        match self {
            Const(_) | Graph(_) | Decoder(_) | Portfolio(_) => 1,
            Neg(a) | Tanh(a) => 1 + a.size(),
            Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Min(a, b) | Max(a, b) => 1 + a.size() + b.size(),
            IfGt(a, b, c, d) => 1 + a.size() + b.size() + c.size() + d.size(),
        }
    }
}

/// A strategy step for one asset set.
pub trait Strategy: Send {
    fn id(&self) -> &str;
    fn horizon(&self) -> Horizon;
    fn step(&mut self, as_of_ms: i64, assets: &[(InstrumentId, &dyn StrategyContext)]) -> ExposureDistribution;
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
/// Implementations live behind ML backend features (D-004).
pub trait NeuralPolicy: Send {
    fn forward(&self, features: &[f32]) -> (f64, f64);
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
