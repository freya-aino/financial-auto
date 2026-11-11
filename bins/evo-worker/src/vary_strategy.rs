//! `evo.vary.strategy` (spec 35 §6.3): children of a strategy generation's elites through the
//! shared `fa_evolution::vary::vary_with` loop with `fa_strategies::ops` operators (subtree
//! crossover, point / subtree / hoist mutation, constant jitter, spec-gene mutation), plus
//! random immigrants (`immigrant_frac`, spec 34 §6.2 step 1).
//!
//! `Job.spec` has the same shape as `evo.vary.decoder` (`parents` best first with their
//! objectives, `n_children`, `seed`, `generation`, optional `budget`, operator overrides
//! `sigma`, `crossover_prob`, `immigrant_frac`, `max_depth`, `max_nodes`). Parents are genome
//! text forms; children are returned as canonical text forms, de-duplicated by id against the
//! parents and each other. Lineage (`upsert_genome`, `link_parent`, `add_fitness_record` for
//! the parents' objectives) goes to Helix best effort.

use std::collections::BTreeSet;

use fa_core::rng::Rng;
use fa_evolution::{
    plan_halving,
    vary::{Child, Evolvable, Operators, vary_with},
};
use fa_hub::{
    HubClient,
    queries::{NewFitness, NewGenome},
};
use fa_strategies::{
    genome::StrategyGenome,
    ops::{OpSpec, crossover_genome, mutate_genome},
    random::{RandomSpec, random_genome},
};
use serde_json::{Value, json};

use crate::vary::{Lineage, MAX_CHILDREN, VaryRequest};

pub const KIND: &str = "evo.vary.strategy";
pub const PROTOCOL: &str = "evo.vary.strategy/v0";
/// Share of children that are random immigrants (spec 34 §6.2 `immigrant_frac`).
pub const DEFAULT_IMMIGRANT_FRAC: f64 = 0.1;

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

#[derive(Debug)]
pub struct Outcome {
    pub parents: Vec<StrategyGenome>,
    pub children: Vec<StrategyGenome>,
    pub lineage: Vec<Lineage>,
    pub immigrants: usize,
    pub plan: Option<Value>,
}

/// Pure variation step. Errors on unparsable parents or an empty parent set.
pub fn produce(req: &VaryRequest) -> Result<Outcome, String> {
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
    let ops = OpSpec { sigma: req.sigma.unwrap_or(OpSpec::default().sigma), ..OpSpec::default() };
    let crossover_prob = req.crossover_prob.unwrap_or(0.5);
    let immigrant_frac = req.immigrant_frac.unwrap_or(DEFAULT_IMMIGRANT_FRAC).clamp(0.0, 1.0);
    let n_immigrants = ((n_children as f64) * immigrant_frac).round() as usize;
    let n_bred = n_children.saturating_sub(n_immigrants);

    let gp = GpOps { random: random.clone(), ops };
    let evo_parents: Vec<Evo> = parents
        .iter()
        .cloned()
        .map(|mut g| {
            g.generation = req.generation.saturating_sub(1);
            Evo(g)
        })
        .collect();
    let mut seen: BTreeSet<String> = parents.iter().map(StrategyGenome::id).collect();
    let mut children = Vec::with_capacity(n_children);
    let mut lineage = Vec::with_capacity(n_children);
    // Over-produce in rounds so duplicates (a mutation collapsing onto a parent) are replaced.
    let mut round = 0u64;
    while children.len() < n_bred && round < 8 {
        let want = (n_bred - children.len()) * 2;
        let batch: Vec<Child<Evo>> =
            vary_with(&evo_parents, want, req.seed.wrapping_add(round.wrapping_mul(0x9E37_79B9)), crossover_prob, &gp);
        for c in batch {
            if children.len() >= n_bred {
                break;
            }
            let key = c.genome.0.id();
            if !seen.insert(key.clone()) {
                continue;
            }
            let mut g = c.genome.0;
            g.generation = req.generation; // crossover + mutation each bumped it; children belong to this loop
            lineage.push(Lineage {
                child: g.to_string(),
                parents: c.parents.clone(),
                op: leak(&c.op),
                parents_as_sent: Vec::new(),
            });
            children.push(g);
        }
        round += 1;
    }
    // Immigrants: random programs at the parents' (most common) timescale.
    let timescale = parents[0].timescale;
    let mut rng = Rng::seed(req.seed ^ 0x5EED_1A11);
    let mut immigrants = 0usize;
    let mut tries = 0;
    while immigrants < n_immigrants && tries < n_immigrants * 8 + 8 {
        tries += 1;
        let mut g = random_genome(&mut rng, timescale, &random);
        g.generation = req.generation;
        let key = g.id();
        if !seen.insert(key) {
            continue;
        }
        lineage.push(Lineage {
            child: g.to_string(),
            parents: Vec::new(),
            op: "immigrant",
            parents_as_sent: Vec::new(),
        });
        children.push(g);
        immigrants += 1;
    }
    Ok(Outcome { parents, children, lineage, immigrants, plan })
}

/// Operator names are a small fixed set; intern them as `&'static str` for `Lineage`.
fn leak(op: &str) -> &'static str {
    const KNOWN: [&str; 12] = [
        "point_mutation",
        "subtree_mutation",
        "hoist_mutation",
        "constant_jitter",
        "spec_gene",
        "crossover+point_mutation",
        "crossover+subtree_mutation",
        "crossover+hoist_mutation",
        "crossover+constant_jitter",
        "crossover+spec_gene",
        "immigrant",
        "mutation",
    ];
    KNOWN.iter().find(|k| **k == op).copied().unwrap_or("variation")
}

/// Best-effort Helix lineage write (kind `strategy`, family `gp`).
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
        family: "gp".into(),
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
        let r = req(&parents, 10, 11, "");
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
        let no_imm = produce(&req(&parents, 10, 11, r#"{"immigrant_frac": 0}"#)).unwrap();
        assert_eq!(no_imm.immigrants, 0);
        let p = payload(&r, &a, json!({"persisted": false}), "w", 3);
        assert_eq!(p["children"].as_array().unwrap().len(), 10);
        assert_eq!(p["protocol"], PROTOCOL);
    }

    #[test]
    fn budget_sizes_population_and_errors_are_clear() {
        let parents = random_population(5, 2, fa_core::Timescale::Minute, &RandomSpec::default());
        // plan_halving(3_600_000, 10_000, 3, 4) → 90 initial candidates; 2 parents → 88 children
        let r =
            req(&parents, 0, 5, r#"{"budget":{"total_gpu_ms":3600000,"min_budget_gpu_ms":10000,"eta":3,"rungs":4}}"#);
        let out = produce(&r).unwrap();
        assert_eq!(out.children.len(), 88);
        assert_eq!(out.plan.as_ref().unwrap()["initial_candidates"], 90);
        assert!(out.children.iter().all(|c| c.timescale == fa_core::Timescale::Minute));
        let empty: VaryRequest = serde_json::from_str(r#"{"parents":[],"n_children":1}"#).unwrap();
        assert!(produce(&empty).is_err());
        let bad: VaryRequest = serde_json::from_str(r#"{"parents":["forecast:h=3"],"n_children":1}"#).unwrap();
        assert!(produce(&bad).unwrap_err().contains("parent"));
    }
}
