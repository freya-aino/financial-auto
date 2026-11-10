//! Population variation (spec 34 §6.2 step 2, spec 35 §6.3): children from a ranked parent
//! set by binary tournament, same-family crossover and mutation. Deterministic in `seed`.
//!
//! The loop is generic over [`Evolvable`] genomes and an [`Operators`] implementation, so
//! decoder genomes ([`Genome`], Gaussian mutation + uniform crossover, [`vary`]) and strategy
//! genomes (`fa_strategies::ops`, adapted in `evo-worker`) share tournament and selection.

use fa_core::rng::Rng;

use crate::{Crossover, GaussianMutation, Genome, Mutation, UniformCrossover};

/// A genome kind the shared variation loop can breed.
pub trait Evolvable: Clone {
    /// Identity recorded in children's lineage.
    fn genome_id(&self) -> String;
    /// Crossover partners are drawn from the same family only.
    fn family(&self) -> &str;
}

/// Variation operators for one genome kind.
pub trait Operators<G: Evolvable> {
    fn crossover(&self, a: &G, b: &G, rng: &mut Rng) -> G;
    /// Returns the child and the operator name (lineage `op`).
    fn mutate(&self, g: &G, rng: &mut Rng) -> (G, &'static str);
}

/// A child with its lineage.
#[derive(Debug, Clone, PartialEq)]
pub struct Child<G> {
    pub genome: G,
    /// Genome ids of the real parents (one or two).
    pub parents: Vec<String>,
    /// `crossover+<mutation>` or `<mutation>`.
    pub op: String,
}

/// Operator settings for decoder genomes.
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
pub fn tournament(rng: &mut Rng, n: usize) -> usize {
    let a = rng.below(n);
    let b = rng.below(n);
    a.min(b)
}

/// Produces `n_children` from `parents` (best first): tournament picks a parent; with
/// `crossover_prob` it is crossed with a same-family tournament partner; the result is mutated.
/// Returns an empty vector when `parents` is empty.
pub fn vary_with<G: Evolvable, O: Operators<G>>(
    parents: &[G],
    n_children: usize,
    seed: u64,
    crossover_prob: f64,
    ops: &O,
) -> Vec<Child<G>> {
    if parents.is_empty() {
        return Vec::new();
    }
    let mut rng = Rng::seed(seed);
    let mut out = Vec::with_capacity(n_children);
    for _ in 0..n_children {
        let ai = tournament(&mut rng, parents.len());
        let a = &parents[ai];
        let partners: Vec<&G> =
            parents.iter().enumerate().filter(|(i, p)| *i != ai && p.family() == a.family()).map(|(_, p)| p).collect();
        let crossed = !partners.is_empty() && rng.uniform() < crossover_prob;
        let (base, mut lineage) = if crossed {
            let b = partners[tournament(&mut rng, partners.len())];
            (ops.crossover(a, b, &mut rng), vec![a.genome_id(), b.genome_id()])
        } else {
            (a.clone(), vec![a.genome_id()])
        };
        let (genome, op) = ops.mutate(&base, &mut rng);
        lineage.dedup();
        out.push(Child {
            genome,
            parents: lineage,
            op: if crossed { format!("crossover+{op}") } else { op.to_owned() },
        });
    }
    out
}

impl Evolvable for Genome {
    fn genome_id(&self) -> String {
        self.id.clone()
    }
    fn family(&self) -> &str {
        &self.family
    }
}

/// Decoder-genome operators: uniform crossover + Gaussian mutation (spec 34 §6.2).
#[derive(Debug, Clone, Copy)]
pub struct DecoderOps(pub VarySpec);

impl Operators<Genome> for DecoderOps {
    fn crossover(&self, a: &Genome, b: &Genome, rng: &mut Rng) -> Genome {
        UniformCrossover.crossover(a, b, rng)
    }
    fn mutate(&self, g: &Genome, rng: &mut Rng) -> (Genome, &'static str) {
        (GaussianMutation { sigma: self.0.sigma, mask_flip_prob: self.0.mask_flip_prob }.mutate(g, rng), "mutation")
    }
}

/// Decoder genomes from `parents` (best first) through [`vary_with`]; lineage points at the
/// real parents (not at the intermediate crossover child), generation = max(parents) + 1.
pub fn vary(parents: &[Genome], n_children: usize, seed: u64, spec: &VarySpec) -> Vec<Genome> {
    let next_gen = parents.iter().map(|p| p.generation).max().unwrap_or(0) + 1;
    vary_with(parents, n_children, seed, spec.crossover_prob, &DecoderOps(*spec))
        .into_iter()
        .map(|c| {
            let mut g = c.genome;
            g.parents = c.parents;
            g.generation = next_gen;
            g
        })
        .collect()
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

    /// A toy genome kind to show the loop is shared: strings with a family prefix.
    #[derive(Clone, Debug, PartialEq)]
    struct Word(String);
    impl Evolvable for Word {
        fn genome_id(&self) -> String {
            self.0.clone()
        }
        fn family(&self) -> &str {
            self.0.split(':').next().unwrap_or("")
        }
    }
    struct WordOps;
    impl Operators<Word> for WordOps {
        fn crossover(&self, a: &Word, b: &Word, _: &mut Rng) -> Word {
            Word(format!("{}+{}", a.0, b.0.split(':').nth(1).unwrap_or("")))
        }
        fn mutate(&self, g: &Word, rng: &mut Rng) -> (Word, &'static str) {
            (Word(format!("{}~{}", g.0, rng.below(10))), "tweak")
        }
    }

    #[test]
    fn generic_loop_records_lineage_and_ops() {
        let parents = vec![Word("x:1".into()), Word("x:2".into()), Word("y:1".into())];
        let kids = vary_with(&parents, 20, 5, 1.0, &WordOps);
        assert_eq!(kids.len(), 20);
        assert!(kids.iter().any(|c| c.op == "crossover+tweak" && c.parents.len() == 2));
        for c in &kids {
            if c.parents[0] == "y:1" {
                assert_eq!((c.op.as_str(), c.parents.len()), ("tweak", 1));
            }
            assert!(c.genome.0.contains('~'));
        }
        assert_eq!(vary_with(&parents, 3, 5, 1.0, &WordOps), vary_with(&parents, 3, 5, 1.0, &WordOps));
        assert!(vary_with::<Word, WordOps>(&[], 3, 5, 1.0, &WordOps).is_empty());
        // tournament favours lower (better) indices
        let mut rng = Rng::seed(1);
        let mean = (0..10_000).map(|_| tournament(&mut rng, 10) as f64).sum::<f64>() / 10_000.0;
        assert!(mean < 4.0, "{mean}");
    }
}
