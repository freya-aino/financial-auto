//! Strategies: GP programs and neural policies competing in one population (spec 35).
//!
//! Every strategy emits a distribution over target exposure per asset. GP primitives
//! read only graph ops, decoder outputs and portfolio state (no classic TA).

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
            as_of: Some(fa_proto::Timestamp { seconds: self.as_of_ms.div_euclid(1000), nanos: (self.as_of_ms.rem_euclid(1000) * 1_000_000) as i32 }),
            exposures: self
                .exposures
                .iter()
                .map(|e| fa_proto::v1::AssetExposure { instrument_id: e.instrument.to_string(), mean: e.mean, stddev: e.stddev })
                .collect(),
        }
    }
}

/// Read-only view a strategy sees at a decision step for one asset.
pub trait StrategyContext {
    // graph ops (spec 33)
    fn current_region(&self) -> Option<&str>;
    fn transition_prob(&self, to_region: &str) -> f64;
    fn stay_prob(&self) -> f64;
    /// Probability mass of sampled k-step paths ending in a region whose analog outcome is positive.
    fn path_up_prob(&self, k: u32) -> f64;
    fn region_entropy(&self) -> f64;
    // decoder outputs (spec 34)
    fn forecast_quantile(&self, level: f64) -> f64;
    fn analog_forecast(&self) -> f64;
    fn decoder_score(&self, family: &str) -> f64;
    // portfolio state
    fn position(&self) -> f64;
    fn unrealised_pnl(&self) -> f64;
    fn time_in_trade(&self) -> f64;
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GraphOp {
    StayProb,
    PathUpProb(u32),
    RegionEntropy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecoderOp {
    Quantile(f64),
    AnalogForecast,
    Score(String),
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

impl Expr {
    pub fn eval(&self, ctx: &dyn StrategyContext) -> f64 {
        use Expr::*;
        let v = match self {
            Const(c) => *c,
            Graph(GraphOp::StayProb) => ctx.stay_prob(),
            Graph(GraphOp::PathUpProb(k)) => ctx.path_up_prob(*k),
            Graph(GraphOp::RegionEntropy) => ctx.region_entropy(),
            Decoder(DecoderOp::Quantile(l)) => ctx.forecast_quantile(*l),
            Decoder(DecoderOp::AnalogForecast) => ctx.analog_forecast(),
            Decoder(DecoderOp::Score(f)) => ctx.decoder_score(f),
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
                if a.eval(ctx) > b.eval(ctx) { c.eval(ctx) } else { d.eval(ctx) }
            }
        };
        if v.is_finite() { v } else { 0.0 }
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
        ExposureDistribution { strategy_id: self.id.clone(), genome_id: self.genome_id.clone(), horizon: self.horizon, as_of_ms, exposures }
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
        let e = Expr::Mul(b(Expr::Const(4.0)), b(Expr::Sub(b(Expr::Graph(GraphOp::PathUpProb(3))), b(Expr::Const(0.5)))));
        assert!((e.eval(&ctx) - 1.2).abs() < 1e-12);
        assert_eq!(Expr::Div(b(Expr::Const(1.0)), b(Expr::Const(0.0))).eval(&ctx), 0.0);
        let cond = Expr::IfGt(b(Expr::Portfolio(PortfolioOp::Position)), b(Expr::Const(0.0)), b(Expr::Const(1.0)), b(Expr::Const(-1.0)));
        assert_eq!(cond.eval(&ctx), -1.0);
        assert_eq!(e.size(), 5);
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
