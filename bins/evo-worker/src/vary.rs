//! `evo.vary.decoder` jobs (spec 34 §6.2 step 2, §6.5): the orchestrator sends the elites of
//! a finished halving loop and asks for the next population's children.
//!
//! `Job.spec` (JSON, protocol `evo.vary.decoder/v0`):
//! ```json
//! {"parents": [{"genome": "forecast:h=5;…", "objectives": {"crps": 0.1, …}, "directions": {…}} | "forecast:…", …],
//!  "n_children": 3, "seed": 7, "generation": 1,
//!  "budget": {"total_gpu_ms": 3600000, "min_budget_gpu_ms": 10000, "eta": 2, "rungs": 4}}   // optional
//! ```
//! Parents are best first (NSGA-II order). When `n_children` is 0 and `budget` is present the
//! population is sized with `fa_evolution::plan_halving` (spec 34 §6.4):
//! `children = initial_candidates − parents`. Children come from `fa_evolution::vary::vary`
//! (same-family uniform crossover + Gaussian mutation on the numeric genes of
//! `DecoderGenome::params`, clamped back by `with_params`) and are returned as canonical kv
//! strings (`DecoderGenome::to_string`), de-duplicated against the parents and each other.
//! The expert mask (`mask` gene) travels as the evolution genome's `expert_mask` bits
//! ([`MASK_GENES`] bits, `0` = every bit set) and is bred by bit flips and uniform crossover;
//! `inst` / `src` (an MCMC-proposed cross-asset pair) are kept from the first parent.
//!
//! Result: `{"protocol": "evo.vary.decoder/v0", "children": [..], "n_parents", "n_children",
//! "plan": {..}?, "lineage": [{"child", "parents"}], "helix": {..}}`.
//! Lineage is also written to Helix (`upsert_genome`, `link_parent`, `add_fitness_record` for
//! the parents' objectives, and `TrainedOn` parent → the generation's encoder snapshot
//! `Job.snapshot_id` when set: the parents were trained/evaluated on its latents) when a hub
//! client is configured; failures only warn.

use std::collections::{BTreeMap, BTreeSet};

use fa_decoders::genome::DecoderGenome;
use fa_evolution::{
    Genome, GenomeKind, plan_halving,
    vary::{VarySpec, vary},
};
use fa_hub::{
    HubClient,
    queries::{NewFitness, NewGenome, TrainedOnLink},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const KIND: &str = "evo.vary.decoder";
pub const PROTOCOL: &str = "evo.vary.decoder/v0";
/// Hard cap on children per job.
pub const MAX_CHILDREN: usize = 10_000;
/// Expert-mask bits bred by evolution (expert indices `0..MASK_GENES`; higher experts are read
/// only by unmasked genomes).
pub const MASK_GENES: u32 = 8;

/// `mask` gene → evolution bits (`0` = all experts = every bit set).
pub fn mask_bits(mask: u32) -> Vec<bool> {
    (0..MASK_GENES).map(|i| mask == 0 || mask & (1 << i) != 0).collect()
}

/// Evolution bits → `mask` gene (every bit set = `0`, all experts; no bit set = `None`, invalid).
pub fn mask_of(bits: &[bool]) -> Option<u32> {
    if bits.is_empty() || bits.iter().all(|b| *b) {
        return Some(0);
    }
    let m = bits.iter().enumerate().filter(|(_, b)| **b).fold(0u32, |m, (i, _)| m | (1 << i));
    (m != 0).then_some(m)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ParentSpec {
    Plain(String),
    /// `directions` / `score` in the spec are accepted and ignored (serde drops unknown keys).
    Scored {
        genome: String,
        #[serde(default)]
        objectives: BTreeMap<String, f64>,
    },
}

impl ParentSpec {
    pub fn genome(&self) -> &str {
        match self {
            Self::Plain(g) => g,
            Self::Scored { genome, .. } => genome,
        }
    }
    pub fn objectives(&self) -> Option<&BTreeMap<String, f64>> {
        match self {
            Self::Scored { objectives, .. } if !objectives.is_empty() => Some(objectives),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct BudgetSpec {
    pub total_gpu_ms: u64,
    pub min_budget_gpu_ms: u64,
    pub eta: u64,
    pub rungs: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VaryRequest {
    pub parents: Vec<ParentSpec>,
    #[serde(default)]
    pub n_children: usize,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub generation: u32,
    #[serde(default)]
    pub budget: Option<BudgetSpec>,
    /// Operator overrides (defaults: sigma 0.2, crossover 0.5).
    #[serde(default)]
    pub sigma: Option<f64>,
    #[serde(default)]
    pub crossover_prob: Option<f64>,
    /// Strategy-only knobs (`evo.vary.strategy`): random-immigrant share, program limits.
    #[serde(default)]
    pub immigrant_frac: Option<f64>,
    #[serde(default)]
    pub max_depth: Option<usize>,
    #[serde(default)]
    pub max_nodes: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Lineage {
    pub child: String,
    /// Parents in canonical kv form (Helix genome ids).
    pub parents: Vec<String>,
    pub op: &'static str,
    /// The same parents exactly as the request named them (the orchestrator's genome ids, so it
    /// can hand a child its parent's weights); omitted for strategy / allocator lineage.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub parents_as_sent: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VaryOutcome {
    pub parents: Vec<DecoderGenome>,
    pub children: Vec<DecoderGenome>,
    pub lineage: Vec<Lineage>,
    pub plan: Option<Value>,
}

/// `DecoderGenome` → evolution genome (numeric genes in `params`, family + timescale explicit).
pub fn to_evo(g: &DecoderGenome, generation: u32) -> Genome {
    Genome {
        id: g.to_string(),
        kind: GenomeKind::Decoder,
        family: g.family.as_str().to_owned(),
        timescale: g.timescale,
        expert_mask: mask_bits(g.expert_mask),
        assets: Vec::new(),
        params: g.params(),
        parents: Vec::new(),
        generation,
        seed: 0,
    }
}

/// Pure variation step (no I/O). Errors only on unparsable parents / empty parent set.
pub fn produce(req: &VaryRequest) -> Result<VaryOutcome, String> {
    if req.parents.is_empty() {
        return Err("no parents".into());
    }
    let mut parents = Vec::with_capacity(req.parents.len());
    for p in &req.parents {
        let g = DecoderGenome::parse(p.genome()).map_err(|e| format!("parent `{}`: {e}", p.genome()))?;
        parents.push(g);
    }
    let (mut n_children, plan) = match (req.n_children, req.budget) {
        (0, Some(b)) => {
            let plan = plan_halving(b.total_gpu_ms, b.min_budget_gpu_ms, b.eta, b.rungs);
            let n = plan.initial_candidates.saturating_sub(parents.len());
            (
                n,
                Some(json!({"initial_candidates": plan.initial_candidates, "rungs": plan.rungs, "eta": plan.eta,
                            "min_budget_gpu_ms": plan.min_budget_gpu_ms, "total_gpu_ms": b.total_gpu_ms})),
            )
        }
        (n, _) => (n, None),
    };
    n_children = n_children.min(MAX_CHILDREN);
    let spec = VarySpec {
        sigma: req.sigma.unwrap_or(VarySpec::default().sigma),
        crossover_prob: req.crossover_prob.unwrap_or(VarySpec::default().crossover_prob),
        ..VarySpec::default()
    };
    let evo_parents: Vec<Genome> = parents.iter().map(|g| to_evo(g, req.generation.saturating_sub(1))).collect();
    let by_id: BTreeMap<&str, &DecoderGenome> =
        evo_parents.iter().zip(&parents).map(|(e, g)| (e.id.as_str(), g)).collect();
    let as_sent: BTreeMap<&str, &str> =
        evo_parents.iter().zip(&req.parents).map(|(e, p)| (e.id.as_str(), p.genome())).collect();

    let mut seen: BTreeSet<String> = evo_parents.iter().map(|p| p.id.clone()).collect();
    let mut children = Vec::with_capacity(n_children);
    let mut lineage = Vec::with_capacity(n_children);
    // Over-produce in rounds so duplicates (clamped genes collapsing onto a parent) can be replaced.
    let mut round = 0u64;
    while children.len() < n_children && round < 8 {
        let want = (n_children - children.len()) * 2;
        for c in vary(&evo_parents, want, req.seed.wrapping_add(round.wrapping_mul(0x9E37_79B9)), &spec) {
            if children.len() >= n_children {
                break;
            }
            let Some(base) = c.parents.first().and_then(|p| by_id.get(p.as_str())) else { continue };
            let Ok(mut child) = base.with_params(&c.params) else { continue };
            // an empty mask (every expert flipped off) keeps the parent's
            child.expert_mask = mask_of(&c.expert_mask).unwrap_or(base.expert_mask);
            child.id = child.to_string();
            let key = child.to_string();
            if !seen.insert(key.clone()) {
                continue;
            }
            lineage.push(Lineage {
                child: key,
                parents_as_sent: c
                    .parents
                    .iter()
                    .map(|p| as_sent.get(p.as_str()).map_or(p.clone(), |s| (*s).to_owned()))
                    .collect(),
                parents: c.parents.clone(),
                op: if c.parents.len() == 2 { "crossover+mutation" } else { "mutation" },
            });
            children.push(child);
        }
        round += 1;
    }
    Ok(VaryOutcome { parents, children, lineage, plan })
}

/// `TrainedOn` rows: every parent (the evaluated elites) → `snapshot_id` (empty: none).
pub fn trained_on_links(req: &VaryRequest, out: &VaryOutcome, snapshot_id: &str) -> Vec<TrainedOnLink> {
    if snapshot_id.trim().is_empty() {
        return Vec::new();
    }
    out.parents
        .iter()
        .map(|p| TrainedOnLink {
            genome_id: p.to_string(),
            snapshot_id: snapshot_id.trim().to_owned(),
            generation: req.generation.saturating_sub(1),
        })
        .collect()
}

/// Best-effort Helix lineage write. Returns a JSON summary for the result payload.
/// `snapshot_id` is the job's encoder snapshot (`Job.snapshot_id`; empty = no `TrainedOn`).
pub async fn persist(hub: Option<&HubClient>, req: &VaryRequest, out: &VaryOutcome, snapshot_id: &str) -> Value {
    let Some(hub) = hub else { return json!({"persisted": false, "reason": "no helix client"}) };
    let mut genomes = 0usize;
    let mut edges = 0usize;
    let mut fitness = 0usize;
    let mut errors: Vec<String> = Vec::new();
    let mut err = |e: fa_hub::HubError| {
        if errors.len() < 5 {
            errors.push(e.to_string());
        }
    };
    let new_genome = |g: &DecoderGenome| NewGenome {
        genome_id: g.to_string(),
        kind: "decoder".into(),
        family: g.family.as_str().into(),
        params: serde_json::to_string(&g.params()).unwrap_or_default(),
    };
    for (p, spec) in out.parents.iter().zip(&req.parents) {
        match hub.upsert_genome(&new_genome(p)).await {
            Ok(_) => genomes += 1,
            Err(e) => err(e),
        }
        if let Some(obj) = spec.objectives() {
            let f = NewFitness {
                genome_id: p.to_string(),
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
    // parents exist now (upserted above); a missing snapshot row is skipped by the probe
    let trained = trained_on_links(req, out, snapshot_id);
    let mut trained_on = 0usize;
    if !trained.is_empty() {
        match hub.link_trained_on(&trained).await {
            Ok(()) => trained_on = trained.len(),
            Err(e) => err(e),
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
            match hub.link_parent(p, &l.child, l.op).await {
                Ok(_) => edges += 1,
                Err(e) => err(e),
            }
        }
    }
    json!({"persisted": errors.is_empty(), "genomes": genomes, "lineage_edges": edges, "fitness_records": fitness,
           "trained_on_edges": trained_on, "errors": errors})
}

/// Result JSON for the job.
pub fn payload(req: &VaryRequest, out: &VaryOutcome, helix: Value, worker_id: &str, elapsed_ms: u64) -> Value {
    json!({
        "protocol": PROTOCOL,
        "children": out.children.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "n_parents": out.parents.len(),
        "n_children": out.children.len(),
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

    fn req(json: &str) -> VaryRequest {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn parents_are_trained_on_the_job_snapshot() {
        let r = req(
            r#"{"parents":["forecast:h=3;lb=16;kernel=8","retrieval:h=3;lb=8;k=10"],"n_children":2,"seed":1,"generation":3}"#,
        );
        let out = produce(&r).unwrap();
        let links = trained_on_links(&r, &out, "stat-v1-1m");
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].genome_id, out.parents[0].to_string());
        assert_eq!((links[0].snapshot_id.as_str(), links[0].generation), ("stat-v1-1m", 2));
        assert!(trained_on_links(&r, &out, "").is_empty(), "no snapshot on the job: no edge");
    }

    #[test]
    fn children_are_valid_distinct_same_family_and_deterministic() {
        let r = req(
            r#"{"parents":[{"genome":"forecast:h=3;lb=16;kernel=8","objectives":{"crps":0.1},"directions":{"crps":"min"}},
                            "retrieval:h=3;lb=8;k=10", {"genome":"forecast:h=4;lb=20"}],
                "n_children":6,"seed":11,"generation":1}"#,
        );
        let a = produce(&r).unwrap();
        let b = produce(&r).unwrap();
        assert_eq!(a.children, b.children, "deterministic in seed");
        assert_eq!(a.children.len(), 6);
        let parents: BTreeSet<String> = a.parents.iter().map(ToString::to_string).collect();
        let mut seen = BTreeSet::new();
        for (c, l) in a.children.iter().zip(&a.lineage) {
            let s = c.to_string();
            assert!(!parents.contains(&s), "child equals a parent: {s}");
            assert!(seen.insert(s.clone()), "duplicate child {s}");
            assert_eq!(DecoderGenome::parse(&s).unwrap().family, c.family);
            assert!(c.min_gap >= c.horizon && c.lookback >= 2 && c.horizon >= 1);
            for p in &l.parents {
                assert_eq!(DecoderGenome::parse(p).unwrap().family, c.family, "crossover across families");
            }
            assert_eq!(l.child, s);
        }
        assert!(a.lineage.iter().any(|l| l.op == "crossover+mutation"), "forecast pair never crossed");
        let sent = ["forecast:h=3;lb=16;kernel=8", "forecast:h=4;lb=20", "retrieval:h=3;lb=8;k=10"];
        for l in &a.lineage {
            assert_eq!(l.parents_as_sent.len(), l.parents.len());
            assert!(l.parents_as_sent.iter().all(|p| sent.contains(&p.as_str())), "{:?}", l.parents_as_sent);
        }
        assert_ne!(
            produce(&req(r#"{"parents":["forecast:h=3;lb=16"],"n_children":2,"seed":1}"#)).unwrap().children,
            produce(&req(r#"{"parents":["forecast:h=3;lb=16"],"n_children":2,"seed":2}"#)).unwrap().children
        );
    }

    #[test]
    fn expert_masks_and_pairs_are_bred() {
        assert_eq!(mask_of(&mask_bits(0)), Some(0));
        assert_eq!(mask_of(&mask_bits(0b101)), Some(0b101));
        assert_eq!(mask_of(&[false; 8]), None);
        let r = req(
            r#"{"parents":["nn_forecast:h=3;lb=16;mask=3;inst=t1;src=s1","nn_forecast:h=4;lb=16;mask=5;inst=t1;src=s1"],
                "n_children":40,"seed":3,"sigma":0.1}"#,
        );
        let mut r = r;
        r.crossover_prob = Some(1.0);
        let out = produce(&r).unwrap();
        assert_eq!(out.children.len(), 40);
        let masks: BTreeSet<u32> = out.children.iter().map(|c| c.expert_mask).collect();
        assert!(masks.len() > 2, "masks are bred: {masks:?}");
        for c in &out.children {
            assert_eq!((c.instrument.as_deref(), c.source.as_deref()), (Some("t1"), Some("s1")), "the pair is kept");
            assert!(c.to_string().contains("src=s1"));
            assert_eq!(DecoderGenome::parse(&c.to_string()).unwrap().expert_mask, c.expert_mask);
        }
    }

    #[test]
    fn budget_sizes_population_and_errors_are_clear() {
        // plan_halving(3_600_000, 10_000, 3, 4) → 90 initial candidates; 2 parents → 88 children
        let r = req(r#"{"parents":["forecast:h=3;lb=16","retrieval:h=2;lb=8;k=5"],"n_children":0,"seed":5,
                "budget":{"total_gpu_ms":3600000,"min_budget_gpu_ms":10000,"eta":3,"rungs":4}}"#);
        let out = produce(&r).unwrap();
        assert_eq!(out.children.len(), 88);
        assert_eq!(out.plan.as_ref().unwrap()["initial_candidates"], 90);
        assert!(produce(&req(r#"{"parents":[],"n_children":1}"#)).is_err());
        assert!(
            produce(&req(r#"{"parents":["visual:h=1"],"n_children":1}"#))
                .unwrap_err()
                .contains("unknown decoder family")
        );
        let p = payload(&r, &out, json!({"persisted": false}), "w", 3);
        assert_eq!(p["children"].as_array().unwrap().len(), 88);
        assert_eq!(p["protocol"], PROTOCOL);
    }
}
