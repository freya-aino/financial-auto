//! Genomes, variation operators, NSGA-II selection and successive halving (specs 34, 35).

use std::collections::BTreeMap;

use fa_core::rng::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenomeKind {
    Encoder,
    Decoder,
    Strategy,
    Sampler,
    Allocator,
}

/// Evolvable description (spec 34 §5). Numeric genes live in `params`; structural
/// choices (family, expert mask, assets) are explicit fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub id: String,
    pub kind: GenomeKind,
    pub family: String,
    pub timescale: fa_core::Timescale,
    pub expert_mask: Vec<bool>,
    pub assets: Vec<fa_core::InstrumentId>,
    pub params: BTreeMap<String, f64>,
    pub parents: Vec<String>,
    pub generation: u32,
    pub seed: u64,
}

/// Objectives are all **minimised** (negate returns/skill before storing).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fitness {
    pub genome_id: String,
    pub objectives: Vec<f64>,
}

pub trait Mutation {
    fn mutate(&self, g: &Genome, rng: &mut Rng) -> Genome;
}

pub trait Crossover {
    fn crossover(&self, a: &Genome, b: &Genome, rng: &mut Rng) -> Genome;
}

fn child_of(parents: &[&Genome], base: &Genome) -> Genome {
    let mut c = base.clone();
    c.id = fa_core::JobId::new().to_string();
    c.parents = parents.iter().map(|p| p.id.clone()).collect();
    c.generation = parents.iter().map(|p| p.generation).max().unwrap_or(0) + 1;
    c
}

/// Gaussian perturbation of numeric params (relative scale) + expert-mask bit flips.
#[derive(Debug, Clone, Copy)]
pub struct GaussianMutation {
    pub sigma: f64,
    pub mask_flip_prob: f64,
}

impl Mutation for GaussianMutation {
    fn mutate(&self, g: &Genome, rng: &mut Rng) -> Genome {
        let mut c = child_of(&[g], g);
        for v in c.params.values_mut() {
            *v += self.sigma * rng.normal() * v.abs().max(1e-3);
        }
        for b in &mut c.expert_mask {
            if rng.uniform() < self.mask_flip_prob {
                *b = !*b;
            }
        }
        c.seed = rng.next_u64();
        c
    }
}

/// Uniform crossover over params and expert mask (structure taken from parent `a`).
#[derive(Debug, Clone, Copy, Default)]
pub struct UniformCrossover;

impl Crossover for UniformCrossover {
    fn crossover(&self, a: &Genome, b: &Genome, rng: &mut Rng) -> Genome {
        let mut c = child_of(&[a, b], a);
        for (k, v) in &mut c.params {
            if let Some(bv) = b.params.get(k)
                && rng.uniform() < 0.5
            {
                *v = *bv;
            }
        }
        for (i, bit) in c.expert_mask.iter_mut().enumerate() {
            if let Some(bb) = b.expert_mask.get(i)
                && rng.uniform() < 0.5
            {
                *bit = *bb;
            }
        }
        c.seed = rng.next_u64();
        c
    }
}

/// `a` dominates `b` (all objectives minimised).
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
}

/// NSGA-II fast non-dominated sort; returns fronts of indices (front 0 = Pareto front).
pub fn non_dominated_sort(objs: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let n = objs.len();
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut count = vec![0usize; n];
    let mut fronts = vec![Vec::new()];
    for p in 0..n {
        for q in 0..n {
            if dominates(&objs[p], &objs[q]) {
                dominated_by[p].push(q);
            } else if dominates(&objs[q], &objs[p]) {
                count[p] += 1;
            }
        }
        if count[p] == 0 {
            fronts[0].push(p);
        }
    }
    let mut i = 0;
    while !fronts[i].is_empty() {
        let mut next = Vec::new();
        for &p in &fronts[i] {
            for &q in &dominated_by[p] {
                count[q] -= 1;
                if count[q] == 0 {
                    next.push(q);
                }
            }
        }
        i += 1;
        fronts.push(next);
    }
    fronts.pop();
    fronts
}

/// Crowding distance for the members of one front (same order as `front`).
#[allow(clippy::needless_range_loop)] // `k` indexes the objective dimension of each member
pub fn crowding_distance(objs: &[Vec<f64>], front: &[usize]) -> Vec<f64> {
    let n = front.len();
    let mut d = vec![0.0; n];
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }
    let m = objs[front[0]].len();
    for k in 0..m {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| objs[front[a]][k].total_cmp(&objs[front[b]][k]));
        let lo = objs[front[order[0]]][k];
        let hi = objs[front[order[n - 1]]][k];
        d[order[0]] = f64::INFINITY;
        d[order[n - 1]] = f64::INFINITY;
        if hi > lo {
            for w in 1..n - 1 {
                d[order[w]] += (objs[front[order[w + 1]]][k] - objs[front[order[w - 1]]][k]) / (hi - lo);
            }
        }
    }
    d
}

/// NSGA-II survivor selection: fill by fronts, break the last front by crowding distance.
pub fn nsga2_select(objs: &[Vec<f64>], k: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(k);
    for front in non_dominated_sort(objs) {
        if out.len() + front.len() <= k {
            out.extend(&front);
        } else {
            let cd = crowding_distance(objs, &front);
            let mut idx: Vec<usize> = (0..front.len()).collect();
            idx.sort_by(|&a, &b| cd[b].total_cmp(&cd[a]));
            out.extend(idx.into_iter().take(k - out.len()).map(|i| front[i]));
        }
        if out.len() >= k {
            break;
        }
    }
    out
}

/// Successive halving: keep the best `ceil(n / eta)` candidates by scalar score (lower is
/// better). Used at each budget rung (spec 34, budget-driven population).
pub fn successive_halving_keep(scores: &[(String, f64)], eta: usize) -> Vec<String> {
    let keep = scores.len().div_ceil(eta.max(2));
    let mut s = scores.to_vec();
    s.sort_by(|a, b| a.1.total_cmp(&b.1));
    s.into_iter().take(keep).map(|x| x.0).collect()
}

/// Budget plan for one generation (spec 34 §6, budget-driven population).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HalvingPlan {
    pub initial_candidates: usize,
    pub rungs: u32,
    pub eta: u64,
    pub min_budget_gpu_ms: u64,
}

/// Equal total budget per rung: rung `r` runs `n / eta^r` candidates at
/// `min_budget · eta^r` each, so `n = (total / rungs) / min_budget`.
pub fn plan_halving(total_gpu_ms: u64, min_budget_gpu_ms: u64, eta: u64, rungs: u32) -> HalvingPlan {
    let per_rung = total_gpu_ms / u64::from(rungs.max(1));
    HalvingPlan {
        initial_candidates: usize::try_from(per_rung / min_budget_gpu_ms.max(1)).unwrap_or(usize::MAX),
        rungs,
        eta,
        min_budget_gpu_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn genome() -> Genome {
        Genome {
            id: "g0".into(),
            kind: GenomeKind::Decoder,
            family: "rnn_prior".into(),
            timescale: fa_core::Timescale::Hour,
            expert_mask: vec![true, false, true, false],
            assets: vec![],
            params: BTreeMap::from([("lr".into(), 1e-3), ("hidden".into(), 64.0)]),
            parents: vec![],
            generation: 0,
            seed: 1,
        }
    }

    #[test]
    fn mutation_records_lineage() {
        let g = genome();
        let mut rng = Rng::seed(5);
        let c = GaussianMutation { sigma: 0.1, mask_flip_prob: 0.5 }.mutate(&g, &mut rng);
        assert_eq!(c.parents, vec!["g0".to_string()]);
        assert_eq!(c.generation, 1);
        assert_ne!(c.params["lr"], g.params["lr"]);
        let x = UniformCrossover.crossover(&g, &c, &mut rng);
        assert_eq!(x.parents.len(), 2);
    }

    #[test]
    fn fronts_are_correct() {
        let objs = vec![vec![1.0, 4.0], vec![2.0, 2.0], vec![4.0, 1.0], vec![3.0, 3.0], vec![5.0, 5.0]];
        let f = non_dominated_sort(&objs);
        assert_eq!(f[0], vec![0, 1, 2]);
        assert_eq!(f[1], vec![3]);
        assert_eq!(f[2], vec![4]);
    }

    #[test]
    fn crowding_prefers_extremes() {
        let objs = vec![vec![0.0, 3.0], vec![1.0, 2.0], vec![1.1, 1.9], vec![3.0, 0.0]];
        let cd = crowding_distance(&objs, &[0, 1, 2, 3]);
        assert!(cd[0].is_infinite() && cd[3].is_infinite());
        let sel = nsga2_select(&objs, 3);
        assert!(sel.contains(&0) && sel.contains(&3));
    }

    #[test]
    fn halving_keeps_best() {
        let s = vec![("a".into(), 3.0), ("b".into(), 1.0), ("c".into(), 2.0), ("d".into(), 4.0)];
        assert_eq!(successive_halving_keep(&s, 2), vec!["b".to_string(), "c".to_string()]);
        let p = plan_halving(3_600_000, 10_000, 3, 4);
        assert_eq!(p.initial_candidates, 90);
    }
}
