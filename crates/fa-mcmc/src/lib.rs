//! Markov state graph over latent regions with Bayesian transition kernels (spec 33).
//!
//! Each outgoing row of the graph has a Dirichlet posterior over next-region
//! probabilities (Dirichlet–multinomial conjugacy). Samplers are pluggable and compete
//! on calibration and cost.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use fa_core::rng::Rng;
use serde::{Deserialize, Serialize};

pub type RegionKey = String;

/// Successor that stands for "somewhere not yet seen" (spec 33 §5: every row keeps prior mass on it).
pub const UNKNOWN: &str = "unknown";

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
    /// Live regions that receive prior mass in every row even before being observed as successors.
    #[serde(default)]
    support: BTreeSet<RegionKey>,
}

impl MarkovGraph {
    pub fn new(prior: f64, decay: f64) -> Self {
        Self { prior, decay, rows: HashMap::new(), support: BTreeSet::new() }
    }

    /// Records an observed transition `from → to`.
    pub fn observe(&mut self, from: &str, to: &str) {
        let row = self.rows.entry(from.to_owned()).or_default();
        for c in row.counts.values_mut() {
            *c *= self.decay;
        }
        *row.counts.entry(to.to_owned()).or_insert(0.0) += 1.0;
    }

    /// Declares a live region: it gets prior mass in every row (spec 33 §5), so a row with a
    /// single observed successor no longer collapses to a degenerate interval.
    pub fn register_region(&mut self, id: &str) {
        self.support.insert(id.to_owned());
    }

    /// Removes a retired region from the prior support (its observed counts are kept).
    pub fn retire_region(&mut self, id: &str) {
        self.support.remove(id);
    }

    pub fn support(&self) -> impl Iterator<Item = &RegionKey> {
        self.support.iter()
    }

    pub fn row(&self, from: &str) -> Option<&TransitionRow> {
        self.rows.get(from)
    }

    pub fn regions(&self) -> impl Iterator<Item = &RegionKey> {
        self.rows.keys()
    }

    /// Dirichlet posterior parameters for a row: `prior` on every observed successor, every
    /// registered live region and [`UNKNOWN`], plus the (decayed) counts. Empty for unknown rows.
    pub fn alpha(&self, from: &str) -> Vec<(RegionKey, f64)> {
        let Some(row) = self.rows.get(from) else {
            return Vec::new();
        };
        let mut keys: BTreeSet<&str> = row.counts.keys().map(String::as_str).collect();
        keys.extend(self.support.iter().map(String::as_str));
        keys.insert(UNKNOWN);
        keys.into_iter().map(|k| (k.to_owned(), row.counts.get(k).copied().unwrap_or(0.0) + self.prior)).collect()
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

/// Dense view of the posterior-mean transition matrix: `index[i]` is the region of row/column
/// `i`, `matrix[i][j]` the mean probability `index[i] → index[j]`. Regions that are only
/// successors (never observed as sources, e.g. [`UNKNOWN`]) get an absorbing row.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionMatrix {
    pub index: Vec<RegionKey>,
    pub matrix: Vec<Vec<f64>>,
}

impl TransitionMatrix {
    pub fn position(&self, region: &str) -> Option<usize> {
        self.index.iter().position(|r| r == region)
    }

    /// `k`-step distribution from `from` by repeated vector–matrix multiplication
    /// (equal to the `from` row of the matrix power `P^k`). `k = 0` is the point mass on `from`.
    pub fn k_step(&self, from: &str, k: u32) -> Vec<(RegionKey, f64)> {
        let n = self.index.len();
        let Some(i) = self.position(from) else {
            return Vec::new();
        };
        let mut v = vec![0.0; n];
        v[i] = 1.0;
        for _ in 0..k {
            let mut next = vec![0.0; n];
            for (a, pa) in v.iter().enumerate() {
                if *pa == 0.0 {
                    continue;
                }
                for (b, pab) in self.matrix[a].iter().enumerate() {
                    next[b] += pa * pab;
                }
            }
            v = next;
        }
        self.index.iter().cloned().zip(v).collect()
    }
}

impl MarkovGraph {
    /// Overwrites a row from a published posterior summary (mean + central credible interval
    /// per successor), so a consumer of `TransitionUpdate`s can rebuild the kernel without the
    /// raw counts. The Dirichlet concentration `S = Σα` is recovered from the interval of the
    /// most likely successor (`Var p = m(1−m)/(S+1)`, width ≈ `2·z·√Var`), so sampled walks keep
    /// the publisher's parameter uncertainty. Means are renormalised; `level` is the interval mass.
    pub fn set_row_from_summary(&mut self, from: &str, to: &[(RegionKey, ProbSummary)], level: f64) {
        let total: f64 = to.iter().map(|(_, s)| s.mean.max(0.0)).sum();
        if total <= 0.0 {
            return;
        }
        let z = normal_quantile(0.5 + level.clamp(0.5, 0.999) / 2.0);
        let concentration = to
            .iter()
            .max_by(|a, b| a.1.mean.total_cmp(&b.1.mean))
            .map(|(_, s)| {
                let m = (s.mean / total).clamp(1e-6, 1.0 - 1e-6);
                let width = (s.upper - s.lower).max(1e-6);
                (m * (1.0 - m) * (2.0 * z / width).powi(2) - 1.0).clamp(1.0, 1e6)
            })
            .unwrap_or(1.0);
        let prior = self.prior;
        let row = self.rows.entry(from.to_owned()).or_default();
        row.counts.clear();
        for (k, s) in to {
            let a = s.mean.max(0.0) / total * concentration;
            // `alpha` adds `prior` back for every key, so store `α − prior` (never negative).
            row.counts.insert(k.clone(), (a - prior).max(0.0));
        }
    }

    /// Dense posterior-mean matrix over every known region (sources, successors, support).
    pub fn transition_matrix(&self) -> TransitionMatrix {
        let mut keys: BTreeSet<RegionKey> = self.rows.keys().cloned().collect();
        keys.extend(self.support.iter().cloned());
        for row in self.rows.values() {
            keys.extend(row.counts.keys().cloned());
        }
        if !self.rows.is_empty() {
            keys.insert(UNKNOWN.to_owned());
        }
        let index: Vec<RegionKey> = keys.into_iter().collect();
        let pos: HashMap<&str, usize> = index.iter().enumerate().map(|(i, k)| (k.as_str(), i)).collect();
        let matrix = index
            .iter()
            .enumerate()
            .map(|(i, from)| {
                let mut row = vec![0.0; index.len()];
                let mean = self.posterior_mean(from);
                if mean.is_empty() {
                    row[i] = 1.0; // absorbing
                } else {
                    for (k, p) in mean {
                        row[pos[k.as_str()]] += p;
                    }
                }
                row
            })
            .collect();
        TransitionMatrix { index, matrix }
    }

    /// Posterior-mean probability of being in `to` exactly `k` steps after `from`
    /// (matrix power of the mean kernel). 0 for unknown regions.
    pub fn k_step_prob(&self, from: &str, to: &str, k: u32) -> f64 {
        self.transition_matrix().k_step(from, k).into_iter().find(|(r, _)| r == to).map_or(0.0, |x| x.1)
    }

    /// Successors of `from` ordered by posterior mean transition mass (descending), excluding
    /// `from` itself and [`UNKNOWN`], at most `n`.
    pub fn neighbors(&self, from: &str, n: usize) -> Vec<(RegionKey, f64)> {
        let mut v: Vec<(RegionKey, f64)> =
            self.posterior_mean(from).into_iter().filter(|(k, _)| k != from && k != UNKNOWN).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v.truncate(n);
        v
    }

    /// Shannon entropy of the posterior-mean row of `from`, normalised to [0, 1] by `ln(n)`.
    pub fn row_entropy(&self, from: &str) -> f64 {
        let m = self.posterior_mean(from);
        if m.len() < 2 {
            return 0.0;
        }
        let h: f64 = m.iter().filter(|(_, p)| *p > 0.0).map(|(_, p)| -p * p.ln()).sum();
        (h / (m.len() as f64).ln()).clamp(0.0, 1.0)
    }
}

/// Standard normal quantile (Acklam), used to recover Dirichlet concentrations from intervals.
fn normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.38357751867269e2,
        -3.066479806614716e1,
        2.506628277459239,
    ];
    const B: [f64; 5] =
        [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const C: [f64; 6] = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ];
    const D: [f64; 4] = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    if p < 0.02425 {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - 0.02425 {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Walk statistics from sampled paths (spec 33 §4.2 `hitting_time`): mean first-hitting step
/// of `targets` over `paths` (a path that never hits counts as `horizon`), and the fraction that hit.
pub fn hitting_time(paths: &[Vec<RegionKey>], targets: &[RegionKey], horizon: usize) -> (f64, f64) {
    if paths.is_empty() {
        return (horizon as f64, 0.0);
    }
    let mut total = 0.0;
    let mut hits = 0usize;
    for p in paths {
        match p.iter().skip(1).position(|r| targets.contains(r)) {
            Some(i) => {
                total += (i + 1) as f64;
                hits += 1;
            }
            None => total += horizon as f64,
        }
    }
    (total / paths.len() as f64, hits as f64 / paths.len() as f64)
}

/// Pluggable sampler over the graph (spec 33): produces sampled future paths.
pub trait Sampler: Send + Sync {
    fn id(&self) -> &str;
    /// Samples `n_paths` region paths of length `steps` starting from `start`.
    fn sample_paths(
        &self,
        g: &MarkovGraph,
        start: &str,
        steps: usize,
        n_paths: usize,
        rng: &mut Rng,
    ) -> Vec<Vec<RegionKey>>;
}

/// Conjugate posterior-predictive sampler: for each path, draw one transition matrix
/// row per visited state from its Dirichlet posterior (parameter uncertainty), then walk.
#[derive(Debug, Default, Clone)]
pub struct ConjugateSampler;

impl Sampler for ConjugateSampler {
    fn id(&self) -> &str {
        "conjugate"
    }

    fn sample_paths(
        &self,
        g: &MarkovGraph,
        start: &str,
        steps: usize,
        n_paths: usize,
        rng: &mut Rng,
    ) -> Vec<Vec<RegionKey>> {
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
        // successors b, c plus the implicit `unknown` share prior mass 1 each: total 10 + 3 = 13
        assert!((m["b"] - 9.0 / 13.0).abs() < 1e-12);
        assert!((m["c"] - 3.0 / 13.0).abs() < 1e-12);
        assert!((m[UNKNOWN] - 1.0 / 13.0).abs() < 1e-12);
    }

    #[test]
    fn registered_regions_keep_prior_mass_and_widen_intervals() {
        let mut g = MarkovGraph::new(1.0, 1.0);
        g.register_region("a");
        g.register_region("b");
        g.observe("a", "b");
        let m: BTreeMap<_, _> = g.posterior_mean("a").into_iter().collect();
        // row a: b=2, a=1, unknown=1 → b = 0.5, not 1.0
        assert!((m["b"] - 0.5).abs() < 1e-12);
        assert!((m["a"] - 0.25).abs() < 1e-12);
        let mut rng = Rng::seed(5);
        let s: BTreeMap<_, _> = g.summarize("a", 0.9, 2000, &mut rng).into_iter().collect();
        assert!(s["b"].upper - s["b"].lower > 0.3, "interval should be wide: {:?}", s["b"]);
        g.retire_region("a");
        assert!(
            !g.alpha("a").iter().any(|(k, c)| k == "a" && *c > 0.0 && !g.row("a").unwrap().counts.contains_key("a"))
        );
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
        // `unknown` keeps prior mass 0.01, so the ceiling is just under 0.99
        assert!(m["new"] > 0.98, "{m:?}");
        assert!(m["old"] < 0.01, "{m:?}");
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
    fn k_step_equals_matrix_power_on_toy_chain() {
        // a → b (0.9) | c (0.1); b → a (1.0); c absorbing (unseen as a source)
        let mut g = MarkovGraph::new(1e-9, 1.0);
        for _ in 0..9 {
            g.observe("a", "b");
        }
        g.observe("a", "c");
        g.observe("b", "a");
        let m = g.transition_matrix();
        assert_eq!(m.index, vec!["a", "b", "c", UNKNOWN]);
        // P^2[a→a] = P[a→b]·P[b→a] = 0.9
        assert!((g.k_step_prob("a", "a", 2) - 0.9).abs() < 1e-6, "{}", g.k_step_prob("a", "a", 2));
        // P^3[a→b] = P[a→b]P[b→a]P[a→b] = 0.81; P^3[a→c] = 0.1 (step 1, absorbing) + 0.9·0.1 = 0.19
        assert!((g.k_step_prob("a", "b", 3) - 0.81).abs() < 1e-6);
        assert!((g.k_step_prob("a", "c", 3) - 0.19).abs() < 1e-6);
        // explicit matrix power check for k = 4 against hand multiplication
        let p = &m.matrix;
        let n = p.len();
        let mut pk = p.clone();
        for _ in 1..4 {
            let mut next = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    next[i][j] = (0..n).map(|l| pk[i][l] * p[l][j]).sum();
                }
            }
            pk = next;
        }
        for (j, (_, v)) in m.k_step("a", 4).into_iter().enumerate() {
            assert!((v - pk[0][j]).abs() < 1e-12);
        }
        assert_eq!(m.k_step("zzz", 2), Vec::<(RegionKey, f64)>::new());
        assert_eq!(g.k_step_prob("a", "a", 0), 1.0);
        let nb = g.neighbors("a", 5);
        assert_eq!(nb.iter().map(|x| x.0.as_str()).collect::<Vec<_>>(), vec!["b", "c"]);
        assert!(g.row_entropy("a") > 0.0 && g.row_entropy("a") < 1.0);
    }

    #[test]
    fn row_from_summary_recovers_means_and_concentration() {
        let mut src = MarkovGraph::new(1.0, 1.0);
        for _ in 0..20 {
            src.observe("a", "b");
        }
        for _ in 0..5 {
            src.observe("a", "c");
        }
        let mut rng = Rng::seed(1);
        let summary = src.summarize("a", 0.9, 4000, &mut rng);
        let mut dst = MarkovGraph::new(1.0, 1.0);
        dst.set_row_from_summary("a", &summary, 0.9);
        let want: BTreeMap<_, _> = src.posterior_mean("a").into_iter().collect();
        for (k, p) in dst.posterior_mean("a") {
            assert!((p - want[&k]).abs() < 1e-9, "{k}: {p} vs {}", want[&k]);
        }
        // Σα of the source row is 25 + 3 = 28; the recovered concentration should be in range.
        let s: f64 = dst.alpha("a").iter().map(|x| x.1).sum();
        assert!((15.0..=45.0).contains(&s), "recovered concentration {s}");
    }

    #[test]
    fn hitting_time_from_paths() {
        let paths: Vec<Vec<RegionKey>> =
            vec![vec!["a".into(), "b".into(), "c".into()], vec!["a".into(), "a".into(), "a".into()]];
        let (t, frac) = hitting_time(&paths, &["c".into()], 2);
        assert_eq!((t, frac), (2.0, 0.5));
        let (t, frac) = hitting_time(&paths, &["b".into()], 5);
        assert_eq!((t, frac), (3.0, 0.5));
        assert_eq!(hitting_time(&[], &["b".into()], 7), (7.0, 0.0));
    }

    #[test]
    fn log_score_prefers_likely_transitions() {
        let g = graph();
        assert!(log_score(&g, &[("a".into(), "b".into())]) > log_score(&g, &[("a".into(), "c".into())]));
    }
}
