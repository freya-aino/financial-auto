//! `evo.vary.strategy` (spec 35 §6.3): children of a strategy generation's elites through the
//! shared `fa_evolution::vary::vary_with` loop, in **two breeding pools** that share the
//! orchestrator's one selection front:
//!
//! - **GP** (`fa_strategies::ops`): typed subtree crossover, point / subtree / hoist mutation,
//!   constant jitter, spec-gene mutation;
//! - **neural** (`fa_strategies::neural`): arithmetic crossover between nets of the same
//!   topology, Gaussian weight perturbation (ES), topology mutation, input-mask flips,
//!   spec-gene mutation, and an **SGD fine-tune** (candle, [`crate::neural_policy`]) of a share
//!   (`sgd_prob`, 0.3) of the neural children on a training window before the selection window.
//!
//! No crossover between the pools. Each pool gets `n_children × share` children, the share
//! being its parents' share clamped to `[min_share_per_repr, 1 − min_share_per_repr]`
//! (default 0.3), so selection can never drive a representation extinct: a pool without
//! parents is refilled with random immigrants of its representation. Each pool also gets its
//! `immigrant_frac` of random immigrants.
//!
//! Every child may also get an **asset-set mutation** (`asset_mutation_prob`, 0.15; add /
//! remove / swap with the MCMC cross-asset pair proposals of its timescale as the distribution,
//! `fa_strategies::assets`), and children are **stamped with the generation's encoder
//! snapshot** (`Job.snapshot_id`) unless already bound — so region literals and latent inputs of
//! evolved strategies refer to a known snapshot (spec 35 §6.5) and bound GP immigrants may
//! carry region literals.
//!
//! `Job.spec` has the same shape as `evo.vary.decoder` (`parents` best first with their
//! objectives, `n_children`, `seed`, `generation`, optional `budget`, operator overrides
//! `sigma`, `crossover_prob`, `immigrant_frac`, `max_depth`, `max_nodes`,
//! `min_share_per_repr`, `asset_mutation_prob`, `sgd_prob`). Parents are genome text forms;
//! children are returned as canonical text forms, de-duplicated by id against the parents and
//! each other. Lineage (`upsert_genome`, `link_parent`, `add_fitness_record` for the parents'
//! objectives) goes to Helix best effort.

use std::{
    collections::BTreeSet,
    sync::{Mutex, OnceLock},
};

use fa_core::{Timescale, rng::Rng};
use fa_evolution::{
    plan_halving,
    vary::{Child, Evolvable, Operators, vary_with},
};
use fa_hub::{
    HubClient,
    queries::{NewFitness, NewGenome},
};
use fa_strategies::{
    assets::{AssetProposals, mutate_assets},
    genome::{Representation, StrategyGenome},
    ops::{OpSpec, crossover_genome, mutate_genome},
    random::{RandomSpec, random_genome, random_neural_genome},
};
use serde_json::{Value, json};

use crate::vary::{Lineage, MAX_CHILDREN, VaryRequest};

pub const KIND: &str = "evo.vary.strategy";
pub const PROTOCOL: &str = "evo.vary.strategy/v1";
/// Share of children that are random immigrants (spec 34 §6.2 `immigrant_frac`).
pub const DEFAULT_IMMIGRANT_FRAC: f64 = 0.1;
/// Population share floor per representation (spec 35 §7 `min_share_per_repr`).
pub const DEFAULT_MIN_SHARE: f64 = 0.3;
pub const DEFAULT_ASSET_MUTATION_PROB: f64 = 0.15;
pub const DEFAULT_SGD_PROB: f64 = 0.3;
/// Region-literal ordinals random GP programs of snapshot-bound genomes may use.
pub const LITERAL_ORDINALS: u32 = 8;

/// `StrategyGenome` in the shared loop: family = timescale (no crossover across timescales).
#[derive(Clone, Debug, PartialEq)]
pub struct Evo(pub StrategyGenome);

impl Evolvable for Evo {
    fn genome_id(&self) -> String {
        self.0.id()
    }
    fn family(&self) -> &str {
        self.0.timescale.as_str()
    }
}

/// `fa_strategies::ops` as `fa_evolution` operators.
pub struct GpOps {
    pub random: RandomSpec,
    pub ops: OpSpec,
}

impl Operators<Evo> for GpOps {
    fn crossover(&self, a: &Evo, b: &Evo, rng: &mut Rng) -> Evo {
        Evo(crossover_genome(&a.0, &b.0, rng, &self.random, &self.ops))
    }
    fn mutate(&self, g: &Evo, rng: &mut Rng) -> (Evo, &'static str) {
        let (c, k) = mutate_genome(&g.0, rng, &self.random, &self.ops);
        (Evo(c), k.as_str())
    }
}

/// Neural-pool operators (`fa_strategies::neural`).
pub struct NnOps {
    pub sigma: f64,
    pub random: RandomSpec,
    /// Expert spaces addressable by input-mask flips.
    pub experts: u32,
}

impl Operators<Evo> for NnOps {
    fn crossover(&self, a: &Evo, b: &Evo, rng: &mut Rng) -> Evo {
        let mut c = a.0.clone();
        if let (Some(x), Some(y)) = (&a.0.policy, &b.0.policy) {
            c.policy = Some(x.crossover(y, rng));
        }
        c.parents = vec![a.0.id(), b.0.id()];
        c.generation = a.0.generation.max(b.0.generation) + 1;
        Evo(c)
    }
    fn mutate(&self, g: &Evo, rng: &mut Rng) -> (Evo, &'static str) {
        let mut c = g.0.clone();
        c.parents = vec![g.0.id()];
        c.generation = g.0.generation + 1;
        let Some(net) = &g.0.policy else { return (Evo(c), "mutation") };
        let u = rng.uniform();
        let op = if u < 0.5 {
            c.policy = Some(net.perturb(rng, self.sigma));
            "es_perturb"
        } else if u < 0.7 {
            c.policy = Some(net.mutate_topology(rng));
            "topology"
        } else if u < 0.85 {
            c.policy = Some(net.mutate_inputs(rng, self.experts));
            "input_mask"
        } else {
            let (lo, hi) = self.random.steps_range;
            if rng.uniform() < 0.5 {
                let f = 1.0 + self.sigma * rng.normal();
                let steps = ((f64::from(g.0.steps) * f).round() as u32).clamp(lo.max(1), hi.max(1));
                c.steps = if steps == g.0.steps { g.0.steps.saturating_add(1).min(hi.max(1)) } else { steps };
            } else {
                let (l0, l1) = self.random.lookback_range;
                let f = 1.0 + self.sigma * rng.normal();
                let lb = ((g.0.lookback_bars as f64 * f).round() as usize).clamp(l0.max(2), l1.max(2));
                c.lookback_bars = if lb == g.0.lookback_bars { g.0.lookback_bars + 1 } else { lb };
            }
            "spec_gene"
        };
        (Evo(c), op)
    }
}

/// Inputs of a variation step that come from the bus / the job rather than the spec.
#[derive(Debug, Clone, Default)]
pub struct VaryContext {
    /// Encoder snapshot of the generation (`Job.snapshot_id`); children are bound to it.
    pub snapshot: Option<String>,
    /// Cross-asset pair proposals per timescale (`latent.crossprior.<ts>`).
    pub proposals: std::collections::BTreeMap<Timescale, AssetProposals>,
}

#[derive(Debug)]
pub struct Outcome {
    pub parents: Vec<StrategyGenome>,
    pub children: Vec<StrategyGenome>,
    pub lineage: Vec<Lineage>,
    pub immigrants: usize,
    pub plan: Option<Value>,
    /// Children per representation (`gp`, `neural`).
    pub pools: Value,
    /// Indices into `children` of neural children to fine-tune by SGD before they are returned.
    pub fine_tune: Vec<usize>,
    pub asset_mutations: usize,
}

/// Interns operator names (a small bounded set of combinations) for `Lineage.op`.
pub fn intern(op: &str) -> &'static str {
    static NAMES: OnceLock<Mutex<BTreeSet<&'static str>>> = OnceLock::new();
    let mut set = NAMES.get_or_init(|| Mutex::new(BTreeSet::new())).lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = set.get(op) {
        return s;
    }
    let s: &'static str = Box::leak(op.to_owned().into_boxed_str());
    set.insert(s);
    s
}

/// Child counts per pool: `(n_gp, n_neural)` (module docs).
pub fn pool_sizes(n_children: usize, n_gp_parents: usize, n_nn_parents: usize, min_share: f64) -> (usize, usize) {
    let total = n_gp_parents + n_nn_parents;
    let min_share = min_share.clamp(0.0, 0.5);
    let share_nn = if total == 0 { 0.5 } else { n_nn_parents as f64 / total as f64 };
    let share_nn = share_nn.clamp(min_share, 1.0 - min_share);
    let n_nn = ((n_children as f64) * share_nn).round() as usize;
    (n_children - n_nn.min(n_children), n_nn.min(n_children))
}

/// Pure variation step without bus context (no snapshot binding, no asset proposals).
#[cfg(test)]
pub fn produce(req: &VaryRequest) -> Result<Outcome, String> {
    produce_with(req, &VaryContext::default())
}

/// Pure variation step. Errors on unparsable parents or an empty parent set.
pub fn produce_with(req: &VaryRequest, ctx: &VaryContext) -> Result<Outcome, String> {
    if req.parents.is_empty() {
        return Err("no parents".into());
    }
    let mut parents = Vec::with_capacity(req.parents.len());
    for p in &req.parents {
        let g = StrategyGenome::parse(p.genome()).map_err(|e| format!("parent `{}`: {e}", p.genome()))?;
        parents.push(g);
    }
    let (mut n_children, plan) = match (req.n_children, req.budget) {
        (0, Some(b)) => {
            let plan = plan_halving(b.total_gpu_ms, b.min_budget_gpu_ms, b.eta, b.rungs);
            (
                plan.initial_candidates.saturating_sub(parents.len()),
                Some(json!({"initial_candidates": plan.initial_candidates, "rungs": plan.rungs, "eta": plan.eta,
                            "min_budget_gpu_ms": plan.min_budget_gpu_ms, "total_gpu_ms": b.total_gpu_ms})),
            )
        }
        (n, _) => (n, None),
    };
    n_children = n_children.min(MAX_CHILDREN);
    let mut random = RandomSpec::default();
    if let Some(d) = req.max_depth {
        random.max_depth = d.max(1);
    }
    if let Some(n) = req.max_nodes {
        random.max_nodes = n.max(1);
    }
    if ctx.snapshot.is_some() {
        random.region_literals = LITERAL_ORDINALS;
    }
    let sigma = req.sigma.unwrap_or(OpSpec::default().sigma);
    let ops = OpSpec { sigma, ..OpSpec::default() };
    let crossover_prob = req.crossover_prob.unwrap_or(0.5);
    let immigrant_frac = req.immigrant_frac.unwrap_or(DEFAULT_IMMIGRANT_FRAC).clamp(0.0, 1.0);
    let min_share = req.min_share_per_repr.unwrap_or(DEFAULT_MIN_SHARE);
    let p_assets = req.asset_mutation_prob.unwrap_or(DEFAULT_ASSET_MUTATION_PROB).clamp(0.0, 1.0);
    let p_sgd = req.sgd_prob.unwrap_or(DEFAULT_SGD_PROB).clamp(0.0, 1.0);
    // immigrants at the parents' (most common) timescale
    let timescale = {
        let mut counts: std::collections::BTreeMap<Timescale, usize> = std::collections::BTreeMap::new();
        for p in &parents {
            *counts.entry(p.timescale).or_default() += 1;
        }
        counts.into_iter().max_by_key(|(t, n)| (*n, std::cmp::Reverse(*t))).map_or(parents[0].timescale, |x| x.0)
    };

    let mut seen: BTreeSet<String> = parents.iter().map(StrategyGenome::id).collect();
    let mut children: Vec<StrategyGenome> = Vec::with_capacity(n_children);
    let mut lineage = Vec::with_capacity(n_children);
    let mut immigrants = 0usize;
    let mut asset_mutations = 0usize;
    let mut fine_tune = Vec::new();
    let mut rng = Rng::seed(req.seed ^ 0x5EED_1A11);
    let gp_parents: Vec<&StrategyGenome> =
        parents.iter().filter(|p| p.representation() == Representation::Gp).collect();
    let nn_parents: Vec<&StrategyGenome> =
        parents.iter().filter(|p| p.representation() == Representation::Neural).collect();
    let (n_gp, n_nn) = pool_sizes(n_children, gp_parents.len(), nn_parents.len(), min_share);
    let mut pools = json!({});

    // Finishes a child: snapshot binding, optional asset-set mutation, dedup; returns whether kept.
    let mut admit = |mut g: StrategyGenome,
                     parents_of: Vec<String>,
                     op: String,
                     rng: &mut Rng,
                     children: &mut Vec<StrategyGenome>,
                     lineage: &mut Vec<Lineage>|
     -> bool {
        g.generation = req.generation;
        if g.snapshot.is_none() {
            g.snapshot.clone_from(&ctx.snapshot);
        }
        let mut op = op;
        let mut mutated = false;
        if let Some(props) = ctx.proposals.get(&g.timescale)
            && !props.is_empty()
            && rng.uniform() < p_assets
        {
            let k = mutate_assets(&mut g, rng, props);
            if k != fa_strategies::assets::AssetOp::None {
                op = format!("{op}+{}", k.as_str());
                mutated = true;
            }
        }
        if !seen.insert(g.id()) {
            return false;
        }
        asset_mutations += usize::from(mutated);
        lineage.push(Lineage { child: g.to_string(), parents: parents_of, op: intern(&op), parents_as_sent: Vec::new() });
        children.push(g);
        true
    };

    for (repr, n_pool, pool_parents) in
        [(Representation::Gp, n_gp, &gp_parents), (Representation::Neural, n_nn, &nn_parents)]
    {
        let first = children.len();
        let n_imm = if pool_parents.is_empty() { n_pool } else { ((n_pool as f64) * immigrant_frac).round() as usize };
        let n_bred = n_pool.saturating_sub(n_imm);
        let evo_parents: Vec<Evo> = pool_parents
            .iter()
            .map(|g| {
                let mut g = (*g).clone();
                g.generation = req.generation.saturating_sub(1);
                Evo(g)
            })
            .collect();
        let gp = GpOps { random: random.clone(), ops };
        let nn = NnOps { sigma, random: random.clone(), experts: 2 };
        // Over-produce in rounds so duplicates (a mutation collapsing onto a parent) are replaced.
        let mut round = 0u64;
        let salt = if repr == Representation::Neural { 0xA5A5_0000 } else { 0 };
        while !evo_parents.is_empty() && children.len() - first < n_bred && round < 8 {
            let want = (n_bred - (children.len() - first)) * 2;
            let seed = req.seed.wrapping_add(salt).wrapping_add(round.wrapping_mul(0x9E37_79B9));
            let batch: Vec<Child<Evo>> = match repr {
                Representation::Gp => vary_with(&evo_parents, want, seed, crossover_prob, &gp),
                Representation::Neural => vary_with(&evo_parents, want, seed, crossover_prob, &nn),
            };
            for c in batch {
                if children.len() - first >= n_bred {
                    break;
                }
                admit(c.genome.0, c.parents.clone(), c.op.clone(), &mut rng, &mut children, &mut lineage);
            }
            round += 1;
        }
        let mut made = 0usize;
        let mut tries = 0;
        let want_imm = n_pool - (children.len() - first);
        while made < want_imm && tries < want_imm * 8 + 8 {
            tries += 1;
            let g = match repr {
                Representation::Gp => random_genome(&mut rng, timescale, &random),
                Representation::Neural => random_neural_genome(&mut rng, timescale, &random),
            };
            if admit(g, Vec::new(), "immigrant".into(), &mut rng, &mut children, &mut lineage) {
                made += 1;
            }
        }
        immigrants += made;
        if repr == Representation::Neural {
            for i in first..children.len() {
                if rng.uniform() < p_sgd {
                    fine_tune.push(i);
                }
            }
        }
        pools[repr.as_str()] = json!({"parents": pool_parents.len(), "children": children.len() - first,
                                      "quota": n_pool, "immigrants": made});
    }
    Ok(Outcome { parents, children, lineage, immigrants, plan, pools, fine_tune, asset_mutations })
}

/// Replaces child `i` by its fine-tuned version (new id, `+sgd` appended to the operator);
/// `false` when the tuned genome collides with another genome (the untuned child is kept).
pub fn replace_tuned(out: &mut Outcome, i: usize, tuned: StrategyGenome) -> bool {
    let id = tuned.id();
    if out.children.iter().chain(&out.parents).any(|g| g.id() == id) {
        return false;
    }
    out.lineage[i].child = tuned.to_string();
    out.lineage[i].op = intern(&format!("{}+sgd", out.lineage[i].op));
    out.children[i] = tuned;
    true
}

/// Best-effort Helix lineage write (kind `strategy`, family `gp` | `neural`).
pub async fn persist(hub: Option<&HubClient>, req: &VaryRequest, out: &Outcome) -> Value {
    let Some(hub) = hub else { return json!({"persisted": false, "reason": "no helix client"}) };
    let (mut genomes, mut edges, mut fitness) = (0usize, 0usize, 0usize);
    let mut errors: Vec<String> = Vec::new();
    let mut err = |e: fa_hub::HubError| {
        if errors.len() < 5 {
            errors.push(e.to_string());
        }
    };
    let new_genome = |g: &StrategyGenome| NewGenome {
        genome_id: g.id(),
        kind: "strategy".into(),
        family: g.representation().as_str().into(),
        params: g.to_string(),
    };
    for (p, spec) in out.parents.iter().zip(&req.parents) {
        match hub.upsert_genome(&new_genome(p)).await {
            Ok(_) => genomes += 1,
            Err(e) => err(e),
        }
        if let Some(obj) = spec.objectives() {
            let f = NewFitness {
                genome_id: p.id(),
                generation: req.generation.saturating_sub(1),
                objectives: serde_json::to_string(obj).unwrap_or_default(),
                pareto_rank: 0,
            };
            match hub.add_fitness_record(&f).await {
                Ok(_) => fitness += 1,
                Err(e) => err(e),
            }
        }
    }
    for (c, l) in out.children.iter().zip(&out.lineage) {
        match hub.upsert_genome(&new_genome(c)).await {
            Ok(_) => genomes += 1,
            Err(e) => {
                err(e);
                continue;
            }
        }
        for p in &l.parents {
            match hub.link_parent(p, &c.id(), l.op).await {
                Ok(_) => edges += 1,
                Err(e) => err(e),
            }
        }
    }
    json!({"persisted": errors.is_empty(), "genomes": genomes, "lineage_edges": edges, "fitness_records": fitness, "errors": errors})
}

/// Result JSON for the job.
pub fn payload(req: &VaryRequest, out: &Outcome, helix: Value, worker_id: &str, elapsed_ms: u64) -> Value {
    json!({
        "protocol": PROTOCOL,
        "children": out.children.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "child_ids": out.children.iter().map(StrategyGenome::id).collect::<Vec<_>>(),
        "n_parents": out.parents.len(),
        "n_children": out.children.len(),
        "immigrants": out.immigrants,
        "pools": out.pools,
        "asset_mutations": out.asset_mutations,
        "requested_children": req.n_children,
        "generation": req.generation,
        "seed": req.seed,
        "plan": out.plan,
        "lineage": out.lineage,
        "helix": helix,
        "worker_id": worker_id,
        "elapsed_ms": elapsed_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fa_strategies::random::random_population;

    fn req(parents: &[StrategyGenome], n: usize, seed: u64, extra: &str) -> VaryRequest {
        let ps: Vec<Value> = parents
            .iter()
            .enumerate()
            .map(|(i, p)| json!({"genome": p.to_string(), "objectives": {"sharpe": 0.1 * i as f64}}))
            .collect();
        let mut v = json!({"parents": ps, "n_children": n, "seed": seed, "generation": 2});
        if !extra.is_empty() {
            let e: Value = serde_json::from_str(extra).unwrap();
            for (k, val) in e.as_object().unwrap() {
                v[k] = val.clone();
            }
        }
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn children_are_valid_distinct_deterministic_and_include_immigrants() {
        let parents = random_population(3, 4, fa_core::Timescale::Second, &RandomSpec::default());
        let r = req(&parents, 10, 11, r#"{"min_share_per_repr": 0}"#);
        let a = produce(&r).unwrap();
        let b = produce(&r).unwrap();
        assert_eq!(a.children, b.children, "deterministic in seed");
        assert_eq!(a.children.len(), 10);
        assert_eq!(a.immigrants, 1, "10% immigrants");
        let parent_ids: BTreeSet<String> = parents.iter().map(StrategyGenome::id).collect();
        let mut seen = BTreeSet::new();
        for (c, l) in a.children.iter().zip(&a.lineage) {
            let id = c.id();
            assert!(!parent_ids.contains(&id), "child equals a parent");
            assert!(seen.insert(id.clone()), "duplicate child {id}");
            assert_eq!(StrategyGenome::parse(&c.to_string()).unwrap().id(), id, "text round trip");
            assert_eq!(c.timescale, fa_core::Timescale::Second);
            assert_eq!(l.child, c.to_string());
            if l.op == "immigrant" {
                assert!(l.parents.is_empty() && c.generation == 2);
            } else {
                assert!(!l.parents.is_empty() && l.parents.iter().all(|p| parent_ids.contains(p)), "{l:?}");
                assert_eq!(c.generation, 2);
            }
        }
        assert!(a.lineage.iter().any(|l| l.op.starts_with("crossover+")), "crossover never happened");
        assert_ne!(
            produce(&req(&parents, 3, 1, "")).unwrap().children,
            produce(&req(&parents, 3, 2, "")).unwrap().children
        );
        let no_imm = produce(&req(&parents, 10, 11, r#"{"immigrant_frac": 0, "min_share_per_repr": 0}"#)).unwrap();
        assert_eq!(no_imm.immigrants, 0);
        let p = payload(&r, &a, json!({"persisted": false}), "w", 3);
        assert_eq!(p["children"].as_array().unwrap().len(), 10);
        assert_eq!(p["protocol"], PROTOCOL);
    }

    #[test]
    fn budget_sizes_population_and_errors_are_clear() {
        let parents = random_population(5, 2, fa_core::Timescale::Minute, &RandomSpec::default());
        // plan_halving(3_600_000, 10_000, 3, 4) → 90 initial candidates; 2 parents → 88 children
        let r = req(
            &parents,
            0,
            5,
            r#"{"budget":{"total_gpu_ms":3600000,"min_budget_gpu_ms":10000,"eta":3,"rungs":4},"min_share_per_repr":0}"#,
        );
        let out = produce(&r).unwrap();
        assert_eq!(out.children.len(), 88);
        assert_eq!(out.plan.as_ref().unwrap()["initial_candidates"], 90);
        assert!(out.children.iter().all(|c| c.timescale == fa_core::Timescale::Minute));
        let empty: VaryRequest = serde_json::from_str(r#"{"parents":[],"n_children":1}"#).unwrap();
        assert!(produce(&empty).is_err());
        let bad: VaryRequest = serde_json::from_str(r#"{"parents":["forecast:h=3"],"n_children":1}"#).unwrap();
        assert!(produce(&bad).unwrap_err().contains("parent"));
    }

    #[test]
    fn breeding_pools_keep_both_representations_alive() {
        use fa_strategies::random::random_neural_genome;
        let gp = random_population(3, 6, fa_core::Timescale::Second, &RandomSpec::default());
        // GP-only parents: the neural pool is refilled with immigrants at the share floor
        let out = produce(&req(&gp, 20, 4, "")).unwrap();
        let nn: Vec<&StrategyGenome> =
            out.children.iter().filter(|c| c.representation() == Representation::Neural).collect();
        assert_eq!(nn.len(), 6, "0.3 × 20: {}", out.pools);
        assert!(out.lineage.iter().filter(|l| l.child.contains("policy")).all(|l| l.op == "immigrant"));
        assert_eq!(pool_sizes(20, 6, 0, 0.3), (14, 6));
        assert_eq!(pool_sizes(20, 1, 9, 0.3), (6, 14), "the share follows the parents within the floor");
        assert_eq!(pool_sizes(20, 5, 5, 0.3), (10, 10));
        // mixed parents: neural children are bred from neural parents only (no cross-pool crossover)
        let mut rng = Rng::seed(8);
        let mut mixed = gp.clone();
        for _ in 0..4 {
            mixed.push(random_neural_genome(&mut rng, fa_core::Timescale::Second, &RandomSpec::default()));
        }
        let nn_ids: BTreeSet<String> =
            mixed.iter().filter(|g| g.representation() == Representation::Neural).map(StrategyGenome::id).collect();
        let gp_ids: BTreeSet<String> =
            mixed.iter().filter(|g| g.representation() == Representation::Gp).map(StrategyGenome::id).collect();
        let out = produce(&req(&mixed, 30, 9, "")).unwrap();
        let mut neural_ops = BTreeSet::new();
        for (c, l) in out.children.iter().zip(&out.lineage) {
            let pool = if c.representation() == Representation::Neural { &nn_ids } else { &gp_ids };
            assert!(l.parents.iter().all(|p| pool.contains(p)), "{} from {:?}", l.op, l.parents);
            if c.representation() == Representation::Neural {
                neural_ops.insert(l.op.split('+').next_back().unwrap_or("").to_owned());
                c.policy.as_ref().unwrap().validate().unwrap();
            }
        }
        assert!(neural_ops.len() >= 3, "{neural_ops:?}");
        assert!(!out.fine_tune.is_empty() && out.fine_tune.iter().all(|i| out.children[*i].policy.is_some()));
        // a tuned replacement updates the child and its lineage
        let mut out2 = produce(&req(&mixed, 30, 9, "")).unwrap();
        let i = out2.fine_tune[0];
        let mut tuned = out2.children[i].clone();
        tuned.policy = Some(tuned.policy.unwrap().perturb(&mut rng, 0.5));
        assert!(replace_tuned(&mut out2, i, tuned.clone()));
        assert!(out2.lineage[i].op.ends_with("+sgd") && out2.lineage[i].child == tuned.to_string());
    }

    #[test]
    fn children_are_stamped_with_the_snapshot_and_asset_sets_mutate() {
        let gp = random_population(5, 6, fa_core::Timescale::Minute, &RandomSpec::default());
        let mut props = std::collections::BTreeMap::new();
        props.insert(
            fa_core::Timescale::Minute,
            AssetProposals { pairs: vec![("a".into(), "b".into(), 3.0), ("b".into(), "c".into(), 1.0)], universe: vec![] },
        );
        let ctx = VaryContext { snapshot: Some("stat-v2-1m".into()), proposals: props };
        let r = req(&gp, 40, 3, r#"{"asset_mutation_prob": 0.5}"#);
        let out = produce_with(&r, &ctx).unwrap();
        assert!(out.children.iter().all(|c| c.snapshot.as_deref() == Some("stat-v2-1m")), "every child bound");
        assert!(out.asset_mutations > 5, "{}", out.asset_mutations);
        let with_assets: Vec<&StrategyGenome> = out.children.iter().filter(|c| !c.assets.is_empty()).collect();
        assert_eq!(with_assets.len(), out.asset_mutations);
        assert!(out.lineage.iter().any(|l| l.op.contains("+asset_pair")));
        for c in &out.children {
            assert_eq!(StrategyGenome::parse(&c.to_string()).unwrap().id(), c.id());
        }
        // bound genomes keep their binding (a remap, not a vary, rebinds)
        let mut bound = gp.clone();
        bound[0].snapshot = Some("old".into());
        let out = produce_with(&req(&bound[..1], 3, 1, r#"{"immigrant_frac": 0, "min_share_per_repr": 0}"#), &ctx).unwrap();
        assert!(out.children.iter().all(|c| c.snapshot.as_deref() == Some("old")));
    }
}
