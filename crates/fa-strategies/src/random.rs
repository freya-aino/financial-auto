//! Typed random program generation (spec 35 §4.2, §6.3): grow-method trees over the GP DSL
//! with depth and node limits and ephemeral random constants (ERCs).
//!
//! Every expression node is `Num`-typed; the typing lives in the literal arguments of the
//! primitives — `Dur` steps (`k`), `Prob` levels (`q`), expert indices, region features and
//! family / topic words — which the generator draws from their own domains so every program it
//! produces parses back through `sexpr` unchanged. Generation is deterministic in the `Rng`.

use fa_core::{Timescale, rng::Rng};
use serde::{Deserialize, Serialize};

use crate::{DecoderOp, Expr, GraphOp, PathFn, PortfolioOp, RegionFeature, RegionRef, SetRef, genome::StrategyGenome};

/// Limits and literal domains for generation and mutation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomSpec {
    pub max_depth: usize,
    pub max_nodes: usize,
    /// Depth of the `uncertainty` program (usually shallower than `signal`).
    pub max_depth_uncertainty: usize,
    /// Probability of stopping at a terminal before `max_depth` (grow method).
    pub terminal_prob: f64,
    /// ERC range `[lo, hi]` (log-uniform magnitude, random sign) — constants like `400` scale
    /// return-sized primitives into the tanh range.
    pub erc_range: (f64, f64),
    /// `k` / `n` step literals are drawn from `1..=max_k`.
    pub max_k: u32,
    /// Expert spaces addressable by `cur_region` / `novelty`.
    pub experts: u32,
    pub score_families: Vec<String>,
    pub text_topics: Vec<String>,
    pub visual_kinds: Vec<String>,
    /// Include the spec 35 §4.2 graph/decoder primitives (`p_trans`, `fc_quantile`, …) in
    /// addition to the D-040 subset.
    pub extended: bool,
    /// Horizon (bars) range for random genomes.
    pub steps_range: (u32, u32),
    pub lookback_range: (usize, usize),
}

impl Default for RandomSpec {
    fn default() -> Self {
        Self {
            max_depth: 5,
            max_nodes: 48,
            max_depth_uncertainty: 3,
            terminal_prob: 0.3,
            erc_range: (0.05, 500.0),
            max_k: 8,
            experts: 1,
            score_families: vec!["rnn_prior".into(), "retrieval".into(), "text".into()],
            text_topics: vec!["news".into()],
            visual_kinds: vec!["future".into()],
            extended: true,
            steps_range: (1, 10),
            lookback_range: (8, 64),
        }
    }
}

/// Quantile levels an ERC `q` is drawn from.
pub const LEVELS: [f64; 9] = [0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95];

fn pick<'a, T>(rng: &mut Rng, xs: &'a [T]) -> Option<&'a T> {
    if xs.is_empty() { None } else { Some(&xs[rng.below(xs.len())]) }
}

/// An ephemeral random constant: log-uniform magnitude in `erc_range`, random sign, rounded to
/// 4 significant digits so the text form stays short.
pub fn erc(rng: &mut Rng, spec: &RandomSpec) -> f64 {
    let (lo, hi) = (spec.erc_range.0.max(1e-9), spec.erc_range.1.max(spec.erc_range.0.max(1e-9)));
    let m = (lo.ln() + rng.uniform() * (hi.ln() - lo.ln())).exp();
    let v = if rng.uniform() < 0.5 { -m } else { m };
    round_sig(v, 4)
}

/// Rounds to `digits` significant digits.
pub fn round_sig(v: f64, digits: i32) -> f64 {
    if v == 0.0 || !v.is_finite() {
        return v;
    }
    let scale = 10f64.powi(digits - 1 - v.abs().log10().floor() as i32);
    (v * scale).round() / scale
}

fn k(rng: &mut Rng, spec: &RandomSpec) -> u32 {
    1 + rng.below(spec.max_k.max(1) as usize) as u32
}

fn level(rng: &mut Rng) -> f64 {
    LEVELS[rng.below(LEVELS.len())]
}

fn feature(rng: &mut Rng) -> RegionFeature {
    [RegionFeature::Mass, RegionFeature::MeanFwdReturn, RegionFeature::Vol][rng.below(3)]
}

/// A random terminal (leaf) node.
pub fn random_terminal(rng: &mut Rng, spec: &RandomSpec) -> Expr {
    let base = 10;
    let ext = if spec.extended { 12 } else { 0 };
    let cur = RegionRef::Cur(0);
    match rng.below(base + ext) {
        0 | 1 => Expr::Const(erc(rng, spec)),
        2 => Expr::Graph(GraphOp::StayProb),
        3 => Expr::Graph(GraphOp::PathUpProb(k(rng, spec))),
        4 => Expr::Graph(GraphOp::RegionEntropy),
        5 => Expr::Decoder(DecoderOp::Quantile(level(rng))),
        6 => Expr::Decoder(DecoderOp::AnalogForecast),
        7 => match pick(rng, &spec.score_families) {
            Some(f) => Expr::Decoder(DecoderOp::Score(f.clone())),
            None => Expr::Decoder(DecoderOp::AnalogForecast),
        },
        8 => Expr::Portfolio(PortfolioOp::Position),
        9 => {
            if rng.uniform() < 0.5 {
                Expr::Portfolio(PortfolioOp::UnrealisedPnl)
            } else {
                Expr::Portfolio(PortfolioOp::TimeInTrade)
            }
        }
        // spec 35 §4.2 graph primitives
        10 => {
            let e = rng.below(spec.experts.max(1) as usize) as u32;
            Expr::Graph(GraphOp::PTrans(cur, RegionRef::Cur(e), k(rng, spec)))
        }
        11 => Expr::Graph(GraphOp::PToSet(cur, SetRef::Neighbors(cur, k(rng, spec)), k(rng, spec))),
        12 => Expr::Graph(GraphOp::HittingTime(cur, SetRef::Neighbors(cur, k(rng, spec)))),
        13 => Expr::Graph(GraphOp::Dwell(cur)),
        14 => Expr::Graph(GraphOp::Novelty(rng.below(spec.experts.max(1) as usize) as u32)),
        15 => Expr::Graph(GraphOp::RegionFeature(cur, feature(rng))),
        16 => {
            let f = if rng.uniform() < 0.5 { PathFn::Up } else { PathFn::Feature(feature(rng)) };
            let steps = k(rng, spec);
            Expr::Graph(GraphOp::PathValue(cur, steps, 8 * k(rng, spec), f))
        }
        // decoder primitives
        17 => Expr::Decoder(DecoderOp::FcQuantile(level(rng), k(rng, spec))),
        18 => Expr::Decoder(DecoderOp::FcDist(k(rng, spec))),
        19 => Expr::Decoder(DecoderOp::AnalogOutcome(k(rng, spec))),
        20 => Expr::Decoder(DecoderOp::AnalogAge),
        _ => {
            if rng.uniform() < 0.5 {
                match pick(rng, &spec.text_topics) {
                    Some(t) => Expr::Decoder(DecoderOp::TextScore(t.clone())),
                    None => Expr::Decoder(DecoderOp::AnalogAge),
                }
            } else {
                match pick(rng, &spec.visual_kinds) {
                    Some(v) => Expr::Decoder(DecoderOp::VisualScore(v.clone())),
                    None => Expr::Decoder(DecoderOp::AnalogAge),
                }
            }
        }
    }
}

/// Arity of a random function node: 1, 2 or 4 (`if_gt`).
pub fn random_arity(rng: &mut Rng) -> usize {
    match rng.below(10) {
        0 | 1 => 1,
        2 => 4,
        _ => 2,
    }
}

/// A function node over `kids` (arity 1, 2 or 4; other arities fall back to `add` of the first two).
pub fn random_function(rng: &mut Rng, kids: Vec<Expr>) -> Expr {
    let n = kids.len();
    let mut it = kids.into_iter();
    let mut next = || Box::new(it.next().unwrap_or(Expr::Const(0.0)));
    match n {
        1 => {
            if rng.uniform() < 0.5 {
                Expr::Neg(next())
            } else {
                Expr::Tanh(next())
            }
        }
        4 => Expr::IfGt(next(), next(), next(), next()),
        _ => match rng.below(6) {
            0 => Expr::Add(next(), next()),
            1 => Expr::Sub(next(), next()),
            2 => Expr::Mul(next(), next()),
            3 => Expr::Div(next(), next()),
            4 => Expr::Min(next(), next()),
            _ => Expr::Max(next(), next()),
        },
    }
}

/// Grow-method tree of depth `≤ depth`.
pub fn grow(rng: &mut Rng, depth: usize, spec: &RandomSpec) -> Expr {
    if depth == 0 || rng.uniform() < spec.terminal_prob {
        return random_terminal(rng, spec);
    }
    let arity = random_arity(rng);
    let kids: Vec<Expr> = (0..arity).map(|_| grow(rng, depth - 1, spec)).collect();
    random_function(rng, kids)
}

/// A random program within `max_depth` and `max_nodes` (retries, then falls back to a terminal).
pub fn random_expr(rng: &mut Rng, max_depth: usize, spec: &RandomSpec) -> Expr {
    for _ in 0..16 {
        let e = grow(rng, max_depth, spec);
        if e.size() <= spec.max_nodes.max(1) {
            return e;
        }
    }
    random_terminal(rng, spec)
}

/// A random strategy genome at `timescale` (immigrant, spec 34 §6.2 step 1).
pub fn random_genome(rng: &mut Rng, timescale: Timescale, spec: &RandomSpec) -> StrategyGenome {
    let (s0, s1) = (spec.steps_range.0.max(1), spec.steps_range.1.max(spec.steps_range.0.max(1)));
    let (l0, l1) = (spec.lookback_range.0.max(2), spec.lookback_range.1.max(spec.lookback_range.0.max(2)));
    // signal: at least one function node so immigrants are not bare constants most of the time
    let signal = random_expr(rng, spec.max_depth.max(1), spec);
    StrategyGenome {
        timescale,
        steps: s0 + rng.below((s1 - s0 + 1) as usize) as u32,
        lookback_bars: l0 + rng.below(l1 - l0 + 1),
        step_secs: None,
        signal,
        uncertainty: random_expr(rng, spec.max_depth_uncertainty, spec),
        parents: Vec::new(),
        generation: 0,
        snapshot: None,
    }
}

/// `n` random genomes (deterministic in `seed`).
pub fn random_population(seed: u64, n: usize, timescale: Timescale, spec: &RandomSpec) -> Vec<StrategyGenome> {
    let mut rng = Rng::seed(seed);
    (0..n).map(|_| random_genome(&mut rng, timescale, spec)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sexpr;

    #[test]
    fn programs_respect_limits_and_round_trip_through_sexpr() {
        let spec = RandomSpec::default();
        let mut rng = Rng::seed(1);
        let mut sizes = Vec::new();
        for _ in 0..300 {
            let e = random_expr(&mut rng, spec.max_depth, &spec);
            assert!(e.size() <= spec.max_nodes, "{e}");
            let text = e.to_string();
            let back = sexpr::parse(&text).unwrap_or_else(|err| panic!("{text}: {err}"));
            assert_eq!(back, e, "{text}");
            sizes.push(e.size());
        }
        assert!(sizes.iter().any(|s| *s > 10), "trees should sometimes be deep");
        assert!(sizes.contains(&1), "and sometimes a bare terminal");
        // every extended primitive family appears over many draws
        let mut seen = [false; 4];
        for _ in 0..2000 {
            let i = match random_terminal(&mut rng, &spec) {
                Expr::Const(_) => 0,
                Expr::Graph(_) => 1,
                Expr::Decoder(_) => 2,
                _ => 3,
            };
            seen[i] = true;
        }
        assert!(seen.iter().all(|s| *s), "Const, Graph, Decoder, Portfolio all appear");
        let texts: std::collections::BTreeSet<String> =
            (0..2000).map(|_| random_terminal(&mut rng, &spec).to_string()).collect();
        for needle in [
            "(p_trans",
            "(p_to_set",
            "(hitting_time",
            "(dwell",
            "(novelty",
            "(region_feature",
            "(path_value",
            "(fc_quantile",
            "(fc_dist",
            "(analog_outcome",
            "analog_age",
            "(text_score",
            "(visual_score",
            "(score",
            "(quantile",
            "(path_up",
        ] {
            assert!(texts.iter().any(|t| t.contains(needle)), "{needle} never generated");
        }
        let legacy = RandomSpec { extended: false, ..RandomSpec::default() };
        for _ in 0..500 {
            let t = random_terminal(&mut rng, &legacy).to_string();
            assert!(!t.contains("p_trans") && !t.contains("fc_"), "{t}");
        }
    }

    #[test]
    fn genomes_are_deterministic_and_valid() {
        let spec = RandomSpec::default();
        let a = random_population(7, 12, Timescale::Second, &spec);
        let b = random_population(7, 12, Timescale::Second, &spec);
        assert_eq!(a, b);
        assert_ne!(a, random_population(8, 12, Timescale::Second, &spec));
        for g in &a {
            assert!((1..=10).contains(&g.steps) && (8..=64).contains(&g.lookback_bars));
            let text = g.to_string();
            assert_eq!(StrategyGenome::parse(&text).unwrap().id(), g.id(), "{text}");
            assert!(g.size() <= 2 * spec.max_nodes);
        }
        let ids: std::collections::BTreeSet<String> = a.iter().map(StrategyGenome::id).collect();
        assert!(ids.len() >= 11, "random genomes should be distinct");
        assert_eq!(round_sig(123.456, 4), 123.5);
        assert_eq!(round_sig(-0.0012345, 4), -0.001235);
        let mut rng = Rng::seed(3);
        for _ in 0..100 {
            let c = erc(&mut rng, &spec).abs();
            assert!((0.05..=500.0).contains(&c), "{c}");
        }
    }
}
