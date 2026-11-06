//! Markov state graph over latent regions with Bayesian transition kernels (spec 33).
//!
//! Each outgoing row of the graph has a Dirichlet posterior over next-region
//! probabilities (Dirichlet–multinomial conjugacy). Samplers are pluggable and compete
//! on calibration and cost.

use std::collections::{BTreeMap, HashMap};

use fa_core::rng::Rng;
use serde::{Deserialize, Serialize};

pub type RegionKey = String;

/// Posterior of one row: Dirichlet(prior + counts) over successor regions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransitionRow {
    pub counts: BTreeMap<RegionKey, f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProbSummary {
    pub mean: f64,
    pub lower: f64,
    pub upper: f64,
}

/// Markov state graph with exponential forgetting (online adaptation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkovGraph {
    /// Symmetric Dirichlet prior concentration per known successor.
    pub prior: f64,
    /// Multiplicative decay applied to a row's counts on each update (1.0 = no forgetting).
    pub decay: f64,
    rows: HashMap<RegionKey, TransitionRow>,
}

impl MarkovGraph {
    pub fn new(prior: f64, decay: f64) -> Self {
        Self { prior, decay, rows: HashMap::new() }
    }

    /// Records an observed transition `from → to`.
    pub fn observe(&mut self, from: &str, to: &str) {
        let row = self.rows.entry(from.to_owned()).or_default();
        for c in row.counts.values_mut() {
            *c *= self.decay;
        }
        *row.counts.entry(to.to_owned()).or_insert(0.0) += 1.0;
    }

    pub fn row(&self, from: &str) -> Option<&TransitionRow> {
        self.rows.get(from)
    }

    pub fn regions(&self) -> impl Iterator<Item = &RegionKey> {
        self.rows.keys()
    }

    /// Dirichlet posterior parameters for a row.
    pub fn alpha(&self, from: &str) -> Vec<(RegionKey, f64)> {
        self.rows
            .get(from)
            .map(|r| r.counts.iter().map(|(k, c)| (k.clone(), c + self.prior)).collect())
            .unwrap_or_default()
    }

    /// Posterior mean transition probabilities.
    pub fn posterior_mean(&self, from: &str) -> Vec<(RegionKey, f64)> {
        let a = self.alpha(from);
        let s: f64 = a.iter().map(|x| x.1).sum();
        a.into_iter().map(|(k, v)| (k, v / s)).collect()
    }

    /// Posterior mean plus a central credible interval of mass `level`, estimated with `draws` samples.
    pub fn summarize(&self, from: &str, level: f64, draws: usize, rng: &mut Rng) -> Vec<(RegionKey, ProbSummary)> {
        let a = self.alpha(from);
        if a.is_empty() {
            return Vec::new();
        }
        let alphas: Vec<f64> = a.iter().map(|x| x.1).collect();
        let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(draws); alphas.len()];
        for _ in 0..draws {
            for (i, p) in rng.dirichlet(&alphas).into_iter().enumerate() {
                samples[i].push(p);
            }
        }
        let s: f64 = alphas.iter().sum();
        let lo_q = (1.0 - level) / 2.0;
        a.into_iter()
            .zip(samples)
            .map(|((k, al), mut xs)| {
                xs.sort_by(f64::total_cmp);
                let q = |p: f64| xs[((p * (xs.len() - 1) as f64).round() as usize).min(xs.len() - 1)];
                (k, ProbSummary { mean: al / s, lower: q(lo_q), upper: q(1.0 - lo_q) })
            })
            .collect()
    }
}

/// Pluggable sampler over the graph (spec 33): produces sampled future paths.
pub trait Sampler: Send + Sync {
    fn id(&self) -> &str;
    /// Samples `n_paths` region paths of length `steps` starting from `start`.
    fn sample_paths(&self, g: &MarkovGraph, start: &str, steps: usize, n_paths: usize, rng: &mut Rng) -> Vec<Vec<RegionKey>>;
}

/// Conjugate posterior-predictive sampler: for each path, draw one transition matrix
/// row per visited state from its Dirichlet posterior (parameter uncertainty), then walk.
#[derive(Debug, Default, Clone)]
pub struct ConjugateSampler;

impl Sampler for ConjugateSampler {
    fn id(&self) -> &str {
        "conjugate"
    }

    fn sample_paths(&self, g: &MarkovGraph, start: &str, steps: usize, n_paths: usize, rng: &mut Rng) -> Vec<Vec<RegionKey>> {
        (0..n_paths)
            .map(|_| {
                let mut drawn: HashMap<RegionKey, Vec<(RegionKey, f64)>> = HashMap::new();
                let mut cur = start.to_owned();
                let mut path = vec![cur.clone()];
                for _ in 0..steps {
                    let row = drawn.entry(cur.clone()).or_insert_with(|| {
                        let a = g.alpha(&cur);
                        let alphas: Vec<f64> = a.iter().map(|x| x.1).collect();
                        if alphas.is_empty() {
                            return Vec::new();
                        }
                        a.into_iter().map(|x| x.0).zip(rng.dirichlet(&alphas)).collect()
                    });
                    if row.is_empty() {
                        break; // absorbing / unseen state
                    }
                    let u = rng.uniform();
                    let mut acc = 0.0;
                    let mut next = row[row.len() - 1].0.clone();
                    for (k, p) in row.iter() {
                        acc += p;
                        if u < acc {
                            next = k.clone();
                            break;
                        }
                    }
                    path.push(next.clone());
                    cur = next;
                }
                path
            })
            .collect()
    }
}

/// Calibration score for competing samplers: mean log predictive probability of
/// realised transitions (higher is better).
pub fn log_score(g: &MarkovGraph, realised: &[(RegionKey, RegionKey)]) -> f64 {
    if realised.is_empty() {
        return 0.0;
    }
    let total: f64 = realised
        .iter()
        .map(|(f, t)| {
            let p = g.posterior_mean(f).into_iter().find(|x| &x.0 == t).map_or(1e-9, |x| x.1);
            p.max(1e-9).ln()
        })
        .sum();
    total / realised.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> MarkovGraph {
        let mut g = MarkovGraph::new(1.0, 1.0);
        for _ in 0..8 {
            g.observe("a", "b");
        }
        for _ in 0..2 {
            g.observe("a", "c");
        }
        g.observe("b", "a");
        g
    }

    #[test]
    fn posterior_mean_is_conjugate() {
        let g = graph();
        let m: BTreeMap<_, _> = g.posterior_mean("a").into_iter().collect();
        assert!((m["b"] - 9.0 / 12.0).abs() < 1e-12);
        assert!((m["c"] - 3.0 / 12.0).abs() < 1e-12);
    }

    #[test]
    fn credible_interval_contains_mean() {
        let g = graph();
        let mut rng = Rng::seed(3);
        for (_, s) in g.summarize("a", 0.9, 4000, &mut rng) {
            assert!(s.lower <= s.mean && s.mean <= s.upper, "{s:?}");
            assert!(s.upper - s.lower < 0.6);
        }
    }

    #[test]
    fn decay_forgets_old_transitions() {
        let mut g = MarkovGraph::new(0.01, 0.5);
        for _ in 0..10 {
            g.observe("x", "old");
        }
        for _ in 0..10 {
            g.observe("x", "new");
        }
        let m: BTreeMap<_, _> = g.posterior_mean("x").into_iter().collect();
        assert!(m["new"] > 0.99);
    }

    #[test]
    fn sampler_paths_follow_graph() {
        let g = graph();
        let mut rng = Rng::seed(11);
        let paths = ConjugateSampler.sample_paths(&g, "a", 3, 200, &mut rng);
        assert_eq!(paths.len(), 200);
        let to_b = paths.iter().filter(|p| p[1] == "b").count();
        assert!(to_b > 100, "expected mostly a→b, got {to_b}");
        // c is unseen as a source → paths stop there.
        assert!(paths.iter().all(|p| p.len() <= 4));
    }

    #[test]
    fn log_score_prefers_likely_transitions() {
        let g = graph();
        assert!(log_score(&g, &[("a".into(), "b".into())]) > log_score(&g, &[("a".into(), "c".into())]));
    }
}
