//! Population variation (spec 34 §6.2 step 2): children from a ranked parent set by
//! same-family uniform crossover followed by Gaussian mutation. Deterministic in `seed`.

use fa_core::rng::Rng;

use crate::{Crossover, GaussianMutation, Genome, Mutation, UniformCrossover};

/// Operator settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarySpec {
    /// Relative Gaussian scale on numeric genes.
    pub sigma: f64,
    /// Expert-mask bit-flip probability.
    pub mask_flip_prob: f64,
    /// Probability that a child has two (same-family) parents.
    pub crossover_prob: f64,
}

impl Default for VarySpec {
    fn default() -> Self {
        Self { sigma: 0.2, mask_flip_prob: 0.05, crossover_prob: 0.5 }
    }
}

/// Binary tournament on position: `parents` are expected in Pareto-rank order (best first),
/// so the lower index wins.
fn pick(rng: &mut Rng, n: usize) -> usize {
    let a = rng.below(n);
    let b = rng.below(n);
    a.min(b)
}

/// Produces `n_children` genomes from `parents` (best first). Crossover partners are drawn
/// from parents of the same `family`; a parent without a same-family partner is mutated only.
/// Returns an empty vector when `parents` is empty.
pub fn vary(parents: &[Genome], n_children: usize, seed: u64, spec: &VarySpec) -> Vec<Genome> {
    if parents.is_empty() {
        return Vec::new();
    }
    let mut rng = Rng::seed(seed);
    let mutation = GaussianMutation { sigma: spec.sigma, mask_flip_prob: spec.mask_flip_prob };
    let mut out = Vec::with_capacity(n_children);
    for _ in 0..n_children {
        let a = &parents[pick(&mut rng, parents.len())];
        let partners: Vec<&Genome> = parents.iter().filter(|p| p.family == a.family && !std::ptr::eq(*p, a)).collect();
        let crossed = !partners.is_empty() && rng.uniform() < spec.crossover_prob;
        let base = if crossed {
            let b = partners[pick(&mut rng, partners.len())];
            UniformCrossover.crossover(a, b, &mut rng)
        } else {
            a.clone()
        };
        let mut child = mutation.mutate(&base, &mut rng);
        if crossed {
            // lineage points at the two real parents, not at the intermediate crossover child
            child.parents = base.parents.clone();
            child.generation = base.generation;
        }
        out.push(child);
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::GenomeKind;

    fn g(id: &str, family: &str, lr: f64) -> Genome {
        Genome {
            id: id.into(),
            kind: GenomeKind::Decoder,
            family: family.into(),
            timescale: fa_core::Timescale::Second,
            expert_mask: vec![],
            assets: vec![],
            params: BTreeMap::from([("lr".into(), lr)]),
            parents: vec![],
            generation: 3,
            seed: 0,
        }
    }

    #[test]
    fn deterministic_and_family_preserving() {
        let parents = vec![g("a", "forecast", 1.0), g("b", "forecast", 2.0), g("c", "retrieval", 3.0)];
        let spec = VarySpec { crossover_prob: 1.0, ..VarySpec::default() };
        let c1 = vary(&parents, 8, 42, &spec);
        let c2 = vary(&parents, 8, 42, &spec);
        assert_eq!(c1.len(), 8);
        for (x, y) in c1.iter().zip(&c2) {
            assert_eq!(x.params, y.params);
            assert_eq!(x.parents, y.parents);
        }
        for c in &c1 {
            assert_eq!(c.generation, 4);
            assert!(!c.parents.is_empty());
            for p in &c.parents {
                let parent = parents.iter().find(|x| &x.id == p).unwrap();
                assert_eq!(parent.family, c.family, "crossover crossed families");
            }
            if c.family == "retrieval" {
                assert_eq!(c.parents, vec!["c".to_string()], "no same-family partner → mutation only");
            }
        }
        assert!(c1.iter().any(|c| c.parents.len() == 2), "crossover never happened");
        assert_ne!(vary(&parents, 2, 1, &spec)[0].params, vary(&parents, 2, 2, &spec)[0].params);
        assert!(vary(&[], 3, 1, &spec).is_empty());
    }
}
