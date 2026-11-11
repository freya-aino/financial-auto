//! Competing online samplers over one region set (spec 33 §4.1 `Sampler`, §6.4).
//!
//! Both samplers see the same observations and keep the same Dirichlet evidence
//! ([`MarkovGraph`], decayed counts with the geometric prior), so kernel transfer and the
//! graph-level queries ([`crate::inference`]) work for either. They differ in how the posterior
//! predictive and its intervals are produced:
//!
//! - [`ConjugateExact`]: closed form (Dirichlet mean, Monte Carlo intervals). The baseline.
//! - [`SmcParticle`]: per source row a population of `n` particles `θ⁽ᵖ⁾` (categorical kernels)
//!   with log weights. Each observation reweights by `θ⁽ᵖ⁾[to]` (an unseen successor counts as
//!   `unknown`); weights are tempered by the forgetting factor so old evidence fades like the
//!   counts do; when the effective sample size falls below `ess_threshold · n` (or the row's
//!   successor set grew) the row is resampled systematically and **moved** by an exact Gibbs
//!   draw from the row's Dirichlet posterior. Predictions are the weighted particle mean,
//!   intervals weighted quantiles.
//!
//! Which one serves is decided by [`crate::calibration::Arena`].

use std::{collections::HashMap, time::Instant};

use fa_core::rng::Rng;

pub use crate::rich::KernelCtx;
use crate::{MarkovGraph, ProbSummary, PruneReport, RegionKey, UNKNOWN, refit::RefitOutcome};

pub trait KernelSampler: Send + Sync {
    /// Stable id (`conjugate`, `smc`), the `sampler_id` of published rows.
    fn name(&self) -> &str;
    /// The evidence this sampler works on.
    fn graph(&self) -> &MarkovGraph;
    fn graph_mut(&mut self) -> &mut MarkovGraph;
    /// Replaces the evidence (kernel transfer on a snapshot swap); particles restart.
    fn replace_graph(&mut self, g: MarkovGraph);
    fn register_region(&mut self, id: &str) {
        self.graph_mut().register_region(id);
    }
    fn retire_region(&mut self, id: &str) {
        let g = self.graph_mut();
        g.retire_region(id);
        g.drop_centroid(id);
    }
    fn observe(&mut self, from: &str, to: &str);
    /// One-step posterior predictive from `from` (never empty).
    fn predict(&mut self, from: &str) -> Vec<(RegionKey, f64)>;
    /// Posterior mean and central interval of mass `level` per successor.
    fn summarize(&mut self, from: &str, level: f64) -> Vec<(RegionKey, ProbSummary)>;

    /// Observation with its context (asset, previous region, dwell, covariates; spec 33 §4.1
    /// `KernelCtx`). Context-free kernels ignore it.
    fn observe_ctx(&mut self, from: &str, to: &str, _ctx: &KernelCtx) {
        self.observe(from, to);
    }
    /// Posterior predictive given the context (the richer kernels of spec 33 §6.2 condition on it).
    fn predict_ctx(&mut self, from: &str, _ctx: &KernelCtx) -> Vec<(RegionKey, f64)> {
        self.predict(from)
    }
    fn summarize_ctx(&mut self, from: &str, level: f64, _ctx: &KernelCtx) -> Vec<(RegionKey, ProbSummary)> {
        self.summarize(from, level)
    }
    /// Takes the part of a batch refit (`jobs.mcmc.refit`) meant for this kernel. Returns
    /// whether anything changed.
    fn apply_refit(&mut self, _o: &RefitOutcome) -> bool {
        false
    }
    /// Sampler-specific state beyond the shared Dirichlet evidence (persisted with the kernel).
    fn export_state(&self) -> Option<serde_json::Value> {
        None
    }
    fn import_state(&mut self, _v: &serde_json::Value) {}
    /// Support-based pruning of the evidence (spec 33 §8, [`MarkovGraph::prune`]); samplers with
    /// per-row state drop it for removed rows and rebuild it for changed ones.
    fn prune(&mut self, min_edge: f64) -> PruneReport {
        self.graph_mut().prune(min_edge)
    }
}

/// Closed-form Dirichlet–multinomial sampler (spec 33 `ConjugateExact`).
pub struct ConjugateExact {
    graph: MarkovGraph,
    rng: Rng,
    draws: usize,
}

impl ConjugateExact {
    pub fn new(graph: MarkovGraph, draws: usize, seed: u64) -> Self {
        Self { graph, rng: Rng::seed(seed), draws: draws.max(16) }
    }
}

impl KernelSampler for ConjugateExact {
    fn name(&self) -> &str {
        "conjugate"
    }
    fn graph(&self) -> &MarkovGraph {
        &self.graph
    }
    fn graph_mut(&mut self) -> &mut MarkovGraph {
        &mut self.graph
    }
    fn replace_graph(&mut self, g: MarkovGraph) {
        self.graph = g;
    }
    fn observe(&mut self, from: &str, to: &str) {
        self.graph.observe(from, to);
    }
    fn predict(&mut self, from: &str) -> Vec<(RegionKey, f64)> {
        self.graph.predictive(from)
    }
    fn summarize(&mut self, from: &str, level: f64) -> Vec<(RegionKey, ProbSummary)> {
        self.graph.summarize(from, level, self.draws, &mut self.rng)
    }
}

#[derive(Debug, Clone)]
struct RowParticles {
    keys: Vec<RegionKey>,
    /// `theta[p][j]`: probability of `keys[j]` under particle `p`.
    theta: Vec<Vec<f32>>,
    logw: Vec<f64>,
}

impl RowParticles {
    fn weights(&self) -> Vec<f64> {
        let m = self.logw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = self.logw.iter().map(|l| (l - m).exp()).collect();
        let s: f64 = w.iter().sum();
        w.into_iter().map(|x| x / s).collect()
    }

    fn ess(&self) -> f64 {
        let w = self.weights();
        1.0 / w.iter().map(|x| x * x).sum::<f64>()
    }

    fn col(&self, key: &str) -> Option<usize> {
        self.keys.iter().position(|k| k == key).or_else(|| self.keys.iter().position(|k| k == UNKNOWN))
    }
}

/// Sequential Monte Carlo sampler (spec 33 `SmcParticle`).
pub struct SmcParticle {
    graph: MarkovGraph,
    rng: Rng,
    n: usize,
    ess_threshold: f64,
    rows: HashMap<RegionKey, RowParticles>,
    /// Rejuvenations so far (diagnostic).
    pub moves: u64,
}

impl SmcParticle {
    pub fn new(graph: MarkovGraph, particles: usize, seed: u64) -> Self {
        Self { graph, rng: Rng::seed(seed), n: particles.max(8), ess_threshold: 0.5, rows: HashMap::new(), moves: 0 }
    }

    /// Fresh particles for `from`: exact draws from the row posterior (or its prior).
    fn rejuvenate(&mut self, from: &str) {
        let mut a = self.graph.alpha(from);
        if a.is_empty() {
            // prior row: same keys as `predictive`, concentrations = prior mass
            a = self.graph.predictive(from).into_iter().map(|(k, p)| (k, p * self.graph.prior.max(1e-6))).collect();
            let s: f64 = a.iter().map(|x| x.1).sum();
            let keys = a.len() as f64;
            for x in &mut a {
                x.1 = x.1 / s * self.graph.prior * keys;
            }
        }
        let alphas: Vec<f64> = a.iter().map(|x| x.1.max(1e-9)).collect();
        let theta = (0..self.n).map(|_| self.rng.dirichlet(&alphas).into_iter().map(|x| x as f32).collect()).collect();
        let keys = a.into_iter().map(|x| x.0).collect();
        self.rows.insert(from.to_owned(), RowParticles { keys, theta, logw: vec![0.0; self.n] });
        self.moves += 1;
    }

    fn row(&mut self, from: &str) -> &RowParticles {
        if !self.rows.contains_key(from) {
            self.rejuvenate(from);
        }
        &self.rows[from]
    }

    pub fn particles(&self) -> usize {
        self.n
    }
}

impl KernelSampler for SmcParticle {
    fn name(&self) -> &str {
        "smc"
    }
    fn graph(&self) -> &MarkovGraph {
        &self.graph
    }
    fn graph_mut(&mut self) -> &mut MarkovGraph {
        &mut self.graph
    }
    fn replace_graph(&mut self, g: MarkovGraph) {
        self.graph = g;
        self.rows.clear();
    }
    fn retire_region(&mut self, id: &str) {
        self.graph.retire_region(id);
        self.graph.drop_centroid(id);
        self.rows.remove(id);
    }
    fn prune(&mut self, min_edge: f64) -> PruneReport {
        let rep = self.graph.prune(min_edge);
        // particles of removed / shrunk rows are regenerated from the posterior on next use
        for k in rep.rows.iter().chain(&rep.changed) {
            self.rows.remove(k);
        }
        rep
    }

    fn observe(&mut self, from: &str, to: &str) {
        let decay = self.graph.decay;
        let grew = {
            let row = self.rows.get_mut(from);
            match row {
                Some(r) => {
                    let exact = r.keys.iter().position(|k| k == to);
                    let col = exact.or_else(|| r.col(to));
                    if let Some(c) = col {
                        for (lw, th) in r.logw.iter_mut().zip(&r.theta) {
                            *lw = *lw * decay + f64::from(th[c]).max(1e-12).ln();
                        }
                    }
                    exact.is_none()
                }
                None => true,
            }
        };
        self.graph.observe(from, to);
        let degenerate = self.rows.get(from).is_some_and(|r| r.ess() < self.ess_threshold * self.n as f64);
        if grew || degenerate {
            // resample + move: the move is an exact Gibbs draw from the row posterior, which
            // makes the resampled population a fresh posterior sample with uniform weights.
            self.rejuvenate(from);
        }
    }

    fn predict(&mut self, from: &str) -> Vec<(RegionKey, f64)> {
        let r = self.row(from).clone();
        let w = r.weights();
        let mut out: Vec<(RegionKey, f64)> = r
            .keys
            .iter()
            .enumerate()
            .map(|(j, k)| (k.clone(), r.theta.iter().zip(&w).map(|(t, w)| f64::from(t[j]) * w).sum()))
            .collect();
        let s: f64 = out.iter().map(|x| x.1).sum();
        if s > 0.0 {
            for x in &mut out {
                x.1 /= s;
            }
        }
        out
    }

    fn summarize(&mut self, from: &str, level: f64) -> Vec<(RegionKey, ProbSummary)> {
        let mean = self.predict(from);
        let r = &self.rows[from];
        let w = r.weights();
        let lo = (1.0 - level) / 2.0;
        mean.into_iter()
            .enumerate()
            .map(|(j, (k, m))| {
                let mut xs: Vec<(f64, f64)> = r.theta.iter().zip(&w).map(|(t, w)| (f64::from(t[j]), *w)).collect();
                xs.sort_by(|a, b| a.0.total_cmp(&b.0));
                let q = |p: f64| {
                    let mut acc = 0.0;
                    for (x, w) in &xs {
                        acc += w;
                        if acc >= p {
                            return *x;
                        }
                    }
                    xs.last().map_or(0.0, |x| x.0)
                };
                (k, ProbSummary { mean: m, lower: q(lo).min(m), upper: q(1.0 - lo).max(m) })
            })
            .collect()
    }
}

/// Wall-clock cost of a closure in nanoseconds (for [`crate::calibration`]).
pub fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let out = f();
    (out, t.elapsed().as_nanos() as f64)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn smc_tracks_the_conjugate_posterior() {
        let mut conj = ConjugateExact::new(MarkovGraph::new(1.0, 1.0), 256, 1);
        let mut smc = SmcParticle::new(MarkovGraph::new(1.0, 1.0), 256, 2);
        let mut rng = Rng::seed(3);
        for _ in 0..600 {
            let to = if rng.uniform() < 0.7 { "b" } else { "c" };
            for s in [&mut conj as &mut dyn KernelSampler, &mut smc] {
                s.predict("a");
                s.observe("a", to);
            }
        }
        let c: BTreeMap<_, _> = conj.predict("a").into_iter().collect();
        let s: BTreeMap<_, _> = smc.predict("a").into_iter().collect();
        assert!((c["b"] - s["b"]).abs() < 0.03, "{c:?} vs {s:?}");
        assert!((s.values().sum::<f64>() - 1.0).abs() < 1e-9);
        let sum: BTreeMap<_, _> = smc.summarize("a", 0.9).into_iter().collect();
        assert!(sum["b"].lower <= sum["b"].mean && sum["b"].mean <= sum["b"].upper);
        assert!(sum["b"].upper - sum["b"].lower < 0.15);
        assert!(smc.moves > 1, "rows were rejuvenated");
        assert_eq!(smc.name(), "smc");
    }

    #[test]
    fn unseen_successors_score_as_unknown_and_retire_drops_rows() {
        let mut smc = SmcParticle::new(MarkovGraph::new(1.0, 1.0), 64, 9);
        smc.register_region("a");
        let p: BTreeMap<_, _> = smc.predict("a").into_iter().collect();
        assert!(p.contains_key(UNKNOWN));
        smc.observe("a", "z");
        assert!(smc.predict("a").iter().any(|(k, _)| k == "z"), "the row grew and was rejuvenated");
        smc.retire_region("a");
        assert!(!smc.graph().support().any(|k| k == "a"));
    }
}
