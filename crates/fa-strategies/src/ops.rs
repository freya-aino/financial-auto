//! GP variation operators (spec 35 §6.3): subtree crossover, point / subtree / hoist mutation
//! and constant jitter over the typed DSL, plus the spec-gene mutations of a
//! [`StrategyGenome`] (horizon, lookback, asset set). All operators are type-safe by
//! construction (strongly-typed GP): a subtree is only exchanged or regrown with one of the same
//! type (`Num` or `Bool`), a point mutation keeps the node's signature, a hoist keeps the root
//! `Num`, and literal arguments are jittered inside their own domains — so the result always
//! parses back through `sexpr`. Deterministic in the `Rng`.

use fa_core::rng::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    CmpOp, DecoderOp, Expr, GraphOp, ProbSrc,
    genome::StrategyGenome,
    random::{self, LEVELS, RandomSpec, round_sig},
};

// ─────────────────────────────────────────── tree helpers ───────────────────────────────────────────

/// Children of a node, in order.
pub fn children(e: &Expr) -> Vec<&Expr> {
    use Expr::*;
    match e {
        Const(_) | Graph(_) | Decoder(_) | Portfolio(_) => vec![],
        Neg(a) | Tanh(a) | Not(a) | ProbGt(_, a) => vec![a],
        Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Min(a, b) | Max(a, b) | Cmp(_, a, b) | And(a, b)
        | Or(a, b) => vec![a, b],
        Clip(a, b, c) | If(a, b, c) => vec![a, b, c],
        IfGt(a, b, c, d) => vec![a, b, c, d],
    }
}

/// The same operator over new children (`kids.len()` must equal the node's arity).
pub fn rebuild(e: &Expr, kids: Vec<Expr>) -> Expr {
    use Expr::*;
    let mut it = kids.into_iter();
    let mut n = || Box::new(it.next().expect("arity"));
    match e {
        Const(_) | Graph(_) | Decoder(_) | Portfolio(_) => e.clone(),
        Neg(_) => Neg(n()),
        Tanh(_) => Tanh(n()),
        Add(..) => Add(n(), n()),
        Sub(..) => Sub(n(), n()),
        Mul(..) => Mul(n(), n()),
        Div(..) => Div(n(), n()),
        Min(..) => Min(n(), n()),
        Max(..) => Max(n(), n()),
        IfGt(..) => IfGt(n(), n(), n(), n()),
        Clip(..) => Clip(n(), n(), n()),
        If(..) => If(n(), n(), n()),
        ProbGt(src, _) => ProbGt(*src, n()),
        Cmp(op, ..) => Cmp(*op, n(), n()),
        And(..) => And(n(), n()),
        Or(..) => Or(n(), n()),
        Not(_) => Not(n()),
    }
}

/// Tree depth (a terminal has depth 1).
pub fn depth(e: &Expr) -> usize {
    1 + children(e).iter().map(|c| depth(c)).max().unwrap_or(0)
}

/// Subtree at pre-order index `idx` (0 = the root).
pub fn subtree(e: &Expr, idx: usize) -> Option<&Expr> {
    if idx == 0 {
        return Some(e);
    }
    let mut rest = idx - 1;
    for c in children(e) {
        let n = c.size();
        if rest < n {
            return subtree(c, rest);
        }
        rest -= n;
    }
    None
}

/// Copy of `e` with the subtree at pre-order index `idx` replaced by `new`.
pub fn replace(e: &Expr, idx: usize, new: Expr) -> Expr {
    if idx == 0 {
        return new;
    }
    let mut rest = idx - 1;
    let mut kids = Vec::new();
    let mut done = false;
    for c in children(e) {
        let n = c.size();
        if !done && rest < n {
            kids.push(replace(c, rest, new.clone()));
            done = true;
        } else {
            if !done {
                rest -= n;
            }
            kids.push(c.clone());
        }
    }
    rebuild(e, kids)
}

/// Pre-order indices of the function (non-terminal) nodes.
pub fn function_indices(e: &Expr) -> Vec<usize> {
    (0..e.size()).filter(|&i| subtree(e, i).is_some_and(|n| !children(n).is_empty())).collect()
}

// ───────────────────────────────────────────── operators ─────────────────────────────────────────────

/// Operator probabilities and strengths.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OpSpec {
    /// Relative Gaussian scale of constant jitter.
    pub sigma: f64,
    /// Probability of each tree mutation kind given a mutation happens (normalised).
    pub p_point: f64,
    pub p_subtree: f64,
    pub p_hoist: f64,
    pub p_jitter: f64,
    /// Probability that the spec genes (`steps`, `lookback_bars`) mutate.
    pub p_spec: f64,
    /// Probability that the `uncertainty` program (instead of `signal`) is the target.
    pub p_uncertainty: f64,
}

impl Default for OpSpec {
    fn default() -> Self {
        Self {
            sigma: 0.2,
            p_point: 0.3,
            p_subtree: 0.25,
            p_hoist: 0.15,
            p_jitter: 0.3,
            p_spec: 0.2,
            p_uncertainty: 0.25,
        }
    }
}

/// Which operator produced a child (lineage `op`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpKind {
    Crossover,
    Point,
    Subtree,
    Hoist,
    Jitter,
    SpecGene,
}

impl OpKind {
    pub fn as_str(self) -> &'static str {
        match self {
            OpKind::Crossover => "crossover",
            OpKind::Point => "point_mutation",
            OpKind::Subtree => "subtree_mutation",
            OpKind::Hoist => "hoist_mutation",
            OpKind::Jitter => "constant_jitter",
            OpKind::SpecGene => "spec_gene",
        }
    }
}

/// Pre-order indices of the subtrees of `e` with the given type (`Bool` when `bool`).
pub fn typed_indices(e: &Expr, bool: bool) -> Vec<usize> {
    (0..e.size()).filter(|&i| subtree(e, i).is_some_and(|n| n.is_bool() == bool)).collect()
}

/// Subtree crossover: a random subtree of `a` is replaced by a random subtree of `b` of the
/// same type, retrying a few times when the child would exceed `max_nodes` (or `b` has no
/// subtree of that type); falls back to `a`.
pub fn subtree_crossover(a: &Expr, b: &Expr, rng: &mut Rng, max_nodes: usize) -> Expr {
    for _ in 0..8 {
        let i = rng.below(a.size());
        let want_bool = subtree(a, i).expect("index in range").is_bool();
        let donors = typed_indices(b, want_bool);
        if donors.is_empty() {
            continue;
        }
        let j = donors[rng.below(donors.len())];
        let donor = subtree(b, j).expect("index in range").clone();
        let child = replace(a, i, donor);
        if child.size() <= max_nodes.max(1) {
            return child;
        }
    }
    a.clone()
}

/// A node with the same signature (child types and result type) as `node` over `kids`.
fn swap_function(rng: &mut Rng, node: &Expr, kids: Vec<Expr>) -> Expr {
    use Expr::*;
    let mut it = kids.into_iter();
    let mut n = || Box::new(it.next().expect("arity"));
    match node {
        Add(..) | Sub(..) | Mul(..) | Div(..) | Min(..) | Max(..) => {
            let (a, b) = (n(), n());
            random::binary_num(rng, a, b)
        }
        Neg(_) | Tanh(_) => {
            if rng.uniform() < 0.5 {
                Neg(n())
            } else {
                Tanh(n())
            }
        }
        Cmp(..) => Cmp(if rng.uniform() < 0.5 { CmpOp::Gt } else { CmpOp::Lt }, n(), n()),
        And(..) | Or(..) => {
            if rng.uniform() < 0.5 {
                And(n(), n())
            } else {
                Or(n(), n())
            }
        }
        ProbGt(src, _) => {
            let k = match src {
                ProbSrc::Fc(k) | ProbSrc::Analog(k) => *k,
            };
            ProbGt(if rng.uniform() < 0.5 { ProbSrc::Fc(k) } else { ProbSrc::Analog(k) }, n())
        }
        // unique signatures: IfGt (4 Num), Clip (3 Num), If (Bool, Num, Num), Not (Bool)
        other => rebuild(other, it.by_ref().collect()),
    }
}

/// Point mutation: one node is swapped for a random node of the same signature (terminal →
/// terminal, function → function with the same child and result types).
pub fn point_mutation(e: &Expr, rng: &mut Rng, spec: &RandomSpec) -> Expr {
    let i = rng.below(e.size());
    let node = subtree(e, i).expect("index in range");
    let kids: Vec<Expr> = children(node).into_iter().cloned().collect();
    let new = if kids.is_empty() { random::random_terminal(rng, spec) } else { swap_function(rng, node, kids) };
    replace(e, i, new)
}

/// Subtree mutation: a random subtree is replaced by a fresh random tree of the same type
/// (depth ≤ 3), within `max_nodes`.
pub fn subtree_mutation(e: &Expr, rng: &mut Rng, spec: &RandomSpec) -> Expr {
    let i = rng.below(e.size());
    let depth = 3.min(spec.max_depth.max(1));
    let fresh = if subtree(e, i).expect("index in range").is_bool() {
        random::random_bool(rng, depth, spec)
    } else {
        random::random_expr(rng, depth, spec)
    };
    let child = replace(e, i, fresh);
    if child.size() <= spec.max_nodes.max(1) { child } else { e.clone() }
}

/// Hoist (shrink) mutation: the tree becomes one of its own `Num` subtrees.
pub fn hoist_mutation(e: &Expr, rng: &mut Rng) -> Expr {
    let nums: Vec<usize> = typed_indices(e, false).into_iter().filter(|&i| i > 0).collect();
    if nums.is_empty() {
        return e.clone();
    }
    let i = nums[rng.below(nums.len())];
    subtree(e, i).expect("index in range").clone()
}

/// Constant jitter: every ERC is scaled by `1 + sigma · N(0,1)`, quantile levels move to a
/// neighbouring level with probability `sigma`, step literals `k` move by ±1 likewise.
pub fn constant_jitter(e: &Expr, rng: &mut Rng, sigma: f64) -> Expr {
    let kids: Vec<Expr> = children(e).into_iter().map(|c| constant_jitter(c, rng, sigma)).collect();
    let step = |rng: &mut Rng, k: u32| -> u32 {
        if rng.uniform() < sigma {
            if rng.uniform() < 0.5 { k.saturating_sub(1).max(1) } else { k.saturating_add(1) }
        } else {
            k
        }
    };
    let lvl = |rng: &mut Rng, q: f64| -> f64 {
        if rng.uniform() < sigma {
            let pos = LEVELS.iter().position(|l| (l - q).abs() < 1e-9).unwrap_or(LEVELS.len() / 2);
            let next = if rng.uniform() < 0.5 { pos.saturating_sub(1) } else { (pos + 1).min(LEVELS.len() - 1) };
            LEVELS[next]
        } else {
            q
        }
    };
    match e {
        Expr::Const(c) => {
            let v = c * (1.0 + sigma * rng.normal());
            Expr::Const(round_sig(if v.is_finite() { v } else { *c }, 4))
        }
        Expr::Graph(GraphOp::PathUpProb(k)) => Expr::Graph(GraphOp::PathUpProb(step(rng, *k))),
        Expr::Graph(GraphOp::PTrans(a, b, k)) => Expr::Graph(GraphOp::PTrans(*a, *b, step(rng, *k))),
        Expr::Graph(GraphOp::PToSet(r, s, k)) => Expr::Graph(GraphOp::PToSet(*r, *s, step(rng, *k))),
        Expr::Graph(GraphOp::PathValue(r, k, n, f)) => Expr::Graph(GraphOp::PathValue(*r, step(rng, *k), *n, *f)),
        Expr::Decoder(DecoderOp::Quantile(q)) => Expr::Decoder(DecoderOp::Quantile(lvl(rng, *q))),
        Expr::Decoder(DecoderOp::FcQuantile(q, k)) => {
            let q = lvl(rng, *q);
            Expr::Decoder(DecoderOp::FcQuantile(q, step(rng, *k)))
        }
        Expr::Decoder(DecoderOp::FcDist(k)) => Expr::Decoder(DecoderOp::FcDist(step(rng, *k))),
        Expr::Decoder(DecoderOp::AnalogOutcome(k)) => Expr::Decoder(DecoderOp::AnalogOutcome(step(rng, *k))),
        Expr::ProbGt(src, _) => {
            let src = match src {
                ProbSrc::Fc(k) => ProbSrc::Fc(step(rng, *k)),
                ProbSrc::Analog(k) => ProbSrc::Analog(step(rng, *k)),
            };
            Expr::ProbGt(src, Box::new(kids.into_iter().next().expect("arity")))
        }
        other if kids.is_empty() => other.clone(),
        other => rebuild(other, kids),
    }
}

/// One tree mutation drawn by the `OpSpec` weights.
pub fn mutate_expr(e: &Expr, rng: &mut Rng, spec: &RandomSpec, ops: &OpSpec) -> (Expr, OpKind) {
    let total = ops.p_point + ops.p_subtree + ops.p_hoist + ops.p_jitter;
    let u = rng.uniform() * if total > 0.0 { total } else { 1.0 };
    if u < ops.p_point {
        (point_mutation(e, rng, spec), OpKind::Point)
    } else if u < ops.p_point + ops.p_subtree {
        (subtree_mutation(e, rng, spec), OpKind::Subtree)
    } else if u < ops.p_point + ops.p_subtree + ops.p_hoist {
        (hoist_mutation(e, rng), OpKind::Hoist)
    } else {
        (constant_jitter(e, rng, ops.sigma), OpKind::Jitter)
    }
}

/// Mutates one genome: with `p_spec` a spec gene (horizon ×/÷ up to 2, lookback × (1 ± σ)),
/// else one program (`uncertainty` with `p_uncertainty`, otherwise `signal`). Lineage is set.
pub fn mutate_genome(g: &StrategyGenome, rng: &mut Rng, spec: &RandomSpec, ops: &OpSpec) -> (StrategyGenome, OpKind) {
    let mut c = g.clone();
    c.parents = vec![g.id()];
    c.generation = g.generation + 1;
    if rng.uniform() < ops.p_spec {
        if rng.uniform() < 0.5 {
            let f = 1.0 + ops.sigma * rng.normal();
            let steps =
                ((f64::from(g.steps) * f).round() as u32).clamp(spec.steps_range.0.max(1), spec.steps_range.1.max(1));
            c.steps = if steps == g.steps { g.steps.saturating_add(1).max(1) } else { steps };
        } else {
            let f = 1.0 + ops.sigma * rng.normal();
            let lb = ((g.lookback_bars as f64 * f).round() as usize)
                .clamp(spec.lookback_range.0.max(2), spec.lookback_range.1.max(2));
            c.lookback_bars = if lb == g.lookback_bars { g.lookback_bars + 1 } else { lb };
        }
        return (c, OpKind::SpecGene);
    }
    if rng.uniform() < ops.p_uncertainty {
        let (e, k) = mutate_expr(&g.uncertainty, rng, spec, ops);
        c.uncertainty = e;
        (c, k)
    } else {
        let (e, k) = mutate_expr(&g.signal, rng, spec, ops);
        c.signal = e;
        (c, k)
    }
}

/// Crossover of two genomes: subtree crossover on `signal` (and on `uncertainty` with
/// `p_uncertainty`); spec genes come from `a`. Lineage points at both parents.
pub fn crossover_genome(
    a: &StrategyGenome,
    b: &StrategyGenome,
    rng: &mut Rng,
    spec: &RandomSpec,
    ops: &OpSpec,
) -> StrategyGenome {
    let mut c = a.clone();
    c.signal = subtree_crossover(&a.signal, &b.signal, rng, spec.max_nodes);
    if rng.uniform() < ops.p_uncertainty {
        c.uncertainty = subtree_crossover(&a.uncertainty, &b.uncertainty, rng, spec.max_nodes);
    }
    c.parents = vec![a.id(), b.id()];
    c.generation = a.generation.max(b.generation) + 1;
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sexpr::parse;

    #[test]
    fn tree_helpers_index_in_preorder() {
        let e = parse("(mul 4 (sub (path_up 3) 0.5))").unwrap();
        assert_eq!(e.size(), 5);
        assert_eq!(depth(&e), 3);
        assert_eq!(subtree(&e, 0), Some(&e));
        assert_eq!(subtree(&e, 1), Some(&Expr::Const(4.0)));
        assert_eq!(subtree(&e, 2).unwrap().to_string(), "(sub (path_up 3) 0.5)");
        assert_eq!(subtree(&e, 3).unwrap().to_string(), "(path_up 3)");
        assert_eq!(subtree(&e, 4), Some(&Expr::Const(0.5)));
        assert_eq!(subtree(&e, 5), None);
        assert_eq!(replace(&e, 4, Expr::Const(1.0)).to_string(), "(mul 4 (sub (path_up 3) 1))");
        assert_eq!(replace(&e, 2, Expr::Const(1.0)).to_string(), "(mul 4 1)");
        assert_eq!(replace(&e, 0, Expr::Const(1.0)), Expr::Const(1.0));
        assert_eq!(function_indices(&e), vec![0, 2]);
        let four = parse("(if_gt position 0 1 -1)").unwrap();
        assert_eq!(rebuild(&four, children(&four).into_iter().cloned().collect()), four);
    }

    #[test]
    fn operators_are_type_safe_bounded_and_deterministic() {
        let spec = RandomSpec::default();
        let ops = OpSpec::default();
        let a = parse("(mul 400 (sub (path_up 3) (quantile 0.25)))").unwrap();
        let b = parse("(if_gt stay_prob 0.55 (tanh (fc_quantile 0.9 2)) (neg position))").unwrap();
        let mut r1 = Rng::seed(9);
        let mut r2 = Rng::seed(9);
        let mut kinds = std::collections::BTreeSet::new();
        for _ in 0..400 {
            let x = subtree_crossover(&a, &b, &mut r1, spec.max_nodes);
            assert_eq!(x, subtree_crossover(&a, &b, &mut r2, spec.max_nodes));
            assert!(x.size() <= spec.max_nodes);
            assert_eq!(parse(&x.to_string()).unwrap(), x, "{x}");
            let (m, k) = mutate_expr(&x, &mut r1, &spec, &ops);
            let (m2, _) = mutate_expr(&x, &mut r2, &spec, &ops);
            assert_eq!(m, m2);
            assert!(m.size() <= spec.max_nodes, "{m}");
            assert_eq!(parse(&m.to_string()).unwrap(), m, "{m}");
            kinds.insert(k);
        }
        assert_eq!(kinds.len(), 4, "all four mutation kinds fire: {kinds:?}");
        assert!((0..50).any(|i| subtree_crossover(&a, &b, &mut Rng::seed(i), 64) != a), "crossover changes something");
        // jitter keeps literals in their domains and only touches constants
        let mut rng = Rng::seed(2);
        for _ in 0..200 {
            let j = constant_jitter(&b, &mut rng, 0.9);
            assert_eq!(j.size(), b.size());
            let t = j.to_string();
            assert!(t.starts_with("(if_gt stay_prob ") && t.contains("(fc_quantile 0."), "{t}");
            if let Expr::Decoder(DecoderOp::FcQuantile(q, k)) = subtree(&j, 4).unwrap() {
                assert!(LEVELS.contains(q) && *k >= 1);
            } else {
                panic!("structure changed: {t}");
            }
        }
        assert_eq!(hoist_mutation(&Expr::Const(1.0), &mut rng), Expr::Const(1.0));
        let h = hoist_mutation(&a, &mut rng);
        assert!(h.size() < a.size());
    }

    #[test]
    fn genome_operators_record_lineage_and_clamp_spec_genes() {
        let spec = RandomSpec::default();
        let ops = OpSpec { p_spec: 1.0, ..OpSpec::default() };
        let g = random::random_population(1, 2, fa_core::Timescale::Minute, &spec);
        let mut rng = Rng::seed(4);
        for _ in 0..100 {
            let (c, k) = mutate_genome(&g[0], &mut rng, &spec, &ops);
            assert_eq!(k, OpKind::SpecGene);
            assert_eq!(c.parents, vec![g[0].id()]);
            assert_eq!(c.generation, 1);
            assert!((1..=10).contains(&c.steps) && (8..=64).contains(&c.lookback_bars));
            assert!(c.steps != g[0].steps || c.lookback_bars != g[0].lookback_bars, "a spec mutation changes a gene");
            assert_eq!(c.timescale, g[0].timescale);
        }
        let ops = OpSpec { p_spec: 0.0, ..OpSpec::default() };
        let (c, k) = mutate_genome(&g[0], &mut rng, &spec, &ops);
        assert_ne!(k, OpKind::SpecGene);
        assert_eq!((c.steps, c.lookback_bars), (g[0].steps, g[0].lookback_bars));
        let x = crossover_genome(&g[0], &g[1], &mut rng, &spec, &ops);
        assert_eq!(x.parents, vec![g[0].id(), g[1].id()]);
        assert_eq!((x.steps, x.lookback_bars, x.timescale), (g[0].steps, g[0].lookback_bars, g[0].timescale));
        assert_eq!(StrategyGenome::parse(&x.to_string()).unwrap().id(), x.id());
        assert_eq!(OpKind::Jitter.as_str(), "constant_jitter");
    }
}
