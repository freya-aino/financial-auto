//! Graph-level posterior inference (spec 33 §4.2, §6.1 "k-step prediction: posterior-averaged
//! matrix powers via Monte Carlo over Dirichlet draws, not the power of the mean").
//!
//! [`Posterior::draw`] samples `n` full transition kernels from the row posteriors
//! (`θ_i ~ Dirichlet(α_i)` per source row, successor-only states absorbing). Every query then
//! evaluates the same function on each draw and summarises the results as a [`Dist`]
//! (mean, sd, central interval, median), so parameter uncertainty reaches every answer:
//!
//! | spec 33 §4.2 | here |
//! |---|---|
//! | `transition_prob(a, b, horizon)` | [`Posterior::transition_prob`] |
//! | k-step distribution | [`Posterior::k_step`] |
//! | `sample_paths(from, k, n)` | [`Posterior::sample_paths`] (one kernel draw per path) |
//! | `path_prob(path)` | [`Posterior::path_prob`] |
//! | `hitting_time(from, target_set)` | [`Posterior::hitting`] (exact first-passage per draw) |
//! | `stationary(region_set)` | [`Posterior::stationary`] (+ [`Posterior::relaxation`]) |
//! | `neighbors(region)` with uncertainty | [`Posterior::neighbors`] |
//!
//! The stationary distribution is taken over the **recurrent part** of the graph: states that
//! were observed as sources, with each row renormalised over them (the `unknown` successor and
//! never-left states would otherwise absorb all mass). Relaxation time is `1 / (1 − |λ₂|)` of
//! that chain, `|λ₂|` estimated by power iteration on `P − 1π`.

use std::collections::HashMap;

use fa_core::rng::Rng;
use serde::{Deserialize, Serialize};

use crate::{MarkovGraph, RegionKey, TransitionMatrix};

/// Summary of a posterior distribution of a scalar.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Dist {
    pub mean: f64,
    pub sd: f64,
    /// Central interval of mass `level`.
    pub lower: f64,
    pub upper: f64,
    pub median: f64,
    pub level: f64,
    pub n: usize,
}

impl Dist {
    pub fn from_samples(mut xs: Vec<f64>, level: f64) -> Dist {
        xs.retain(|x| x.is_finite());
        let n = xs.len();
        if n == 0 {
            return Dist { mean: 0.0, sd: 0.0, lower: 0.0, upper: 0.0, median: 0.0, level, n: 0 };
        }
        xs.sort_by(f64::total_cmp);
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n.max(2) - 1) as f64;
        let q = |p: f64| xs[((p * (n - 1) as f64).round() as usize).min(n - 1)];
        let lo = (1.0 - level.clamp(0.0, 1.0)) / 2.0;
        Dist { mean, sd: var.sqrt(), lower: q(lo), upper: q(1.0 - lo), median: q(0.5), level, n }
    }

    pub fn point(x: f64) -> Dist {
        Dist { mean: x, sd: 0.0, lower: x, upper: x, median: x, level: 1.0, n: 1 }
    }
}

/// First-passage summary of [`Posterior::hitting`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hitting {
    /// `P(τ ≤ horizon)`.
    pub p_hit: Dist,
    /// `E[min(τ, horizon)]` (a path that never hits counts as `horizon`, like
    /// [`crate::hitting_time`]).
    pub expected_steps: Dist,
    /// `E[τ | τ ≤ horizon]` (draws that never hit are left out).
    pub conditional_steps: Dist,
    /// Posterior-mean first-passage probabilities for steps `1..=horizon`.
    pub pmf: Vec<f64>,
    pub horizon: u32,
}

/// `n` kernel draws over a fixed state index.
#[derive(Debug, Clone)]
pub struct Posterior {
    pub index: Vec<RegionKey>,
    pos: HashMap<RegionKey, usize>,
    /// Whether state `i` was observed as a source (recurrent part).
    source: Vec<bool>,
    draws: Vec<Vec<Vec<f64>>>,
    pub level: f64,
}

impl Posterior {
    /// Draws `n` kernels from `g`'s posterior. `level` is the interval mass of every [`Dist`].
    pub fn draw(g: &MarkovGraph, n: usize, level: f64, rng: &mut Rng) -> Posterior {
        let mean = g.transition_matrix();
        let index = mean.index.clone();
        let pos: HashMap<RegionKey, usize> = index.iter().enumerate().map(|(i, k)| (k.clone(), i)).collect();
        let source: Vec<bool> = index.iter().map(|k| g.row(k).is_some()).collect();
        let alphas: Vec<Option<(Vec<usize>, Vec<f64>)>> = index
            .iter()
            .map(|k| {
                let a = g.alpha(k);
                (!a.is_empty()).then(|| {
                    let cols = a.iter().map(|(k, _)| pos[k.as_str()]).collect();
                    (cols, a.into_iter().map(|x| x.1).collect())
                })
            })
            .collect();
        let draws = (0..n.max(1))
            .map(|_| {
                alphas
                    .iter()
                    .enumerate()
                    .map(|(i, a)| {
                        let mut row = vec![0.0; index.len()];
                        match a {
                            Some((cols, al)) => {
                                for (c, p) in cols.iter().zip(rng.dirichlet(al)) {
                                    row[*c] += p;
                                }
                            }
                            None => row[i] = 1.0,
                        }
                        row
                    })
                    .collect()
            })
            .collect();
        Posterior { index, pos, source, draws, level }
    }

    /// A posterior from explicit kernels (tests, or a sampler's own particles).
    pub fn from_matrices(m: Vec<TransitionMatrix>, sources: &[RegionKey], level: f64) -> Option<Posterior> {
        let index = m.first()?.index.clone();
        if m.iter().any(|x| x.index != index) {
            return None;
        }
        let pos: HashMap<RegionKey, usize> = index.iter().enumerate().map(|(i, k)| (k.clone(), i)).collect();
        let source = index.iter().map(|k| sources.contains(k)).collect();
        Some(Posterior { index, pos, source, draws: m.into_iter().map(|x| x.matrix).collect(), level })
    }

    pub fn n_draws(&self) -> usize {
        self.draws.len()
    }

    pub fn position(&self, region: &str) -> Option<usize> {
        self.pos.get(region).copied()
    }

    fn step(row: &[f64], p: &[Vec<f64>]) -> Vec<f64> {
        let mut next = vec![0.0; row.len()];
        for (a, pa) in row.iter().enumerate() {
            if *pa == 0.0 {
                continue;
            }
            for (b, pab) in p[a].iter().enumerate() {
                next[b] += pa * pab;
            }
        }
        next
    }

    fn k_step_vec(&self, d: usize, from: usize, k: u32) -> Vec<f64> {
        let mut v = vec![0.0; self.index.len()];
        v[from] = 1.0;
        for _ in 0..k {
            v = Self::step(&v, &self.draws[d]);
        }
        v
    }

    /// `P(X_{t+k} = to | X_t = from)` as a posterior distribution. `None` for unknown regions.
    pub fn transition_prob(&self, from: &str, to: &str, k: u32) -> Option<Dist> {
        let (i, j) = (self.position(from)?, self.position(to)?);
        let xs = (0..self.draws.len()).map(|d| self.k_step_vec(d, i, k)[j]).collect();
        Some(Dist::from_samples(xs, self.level))
    }

    /// The k-step distribution from `from` over every state, each probability a [`Dist`],
    /// sorted by posterior mean (descending).
    pub fn k_step(&self, from: &str, k: u32) -> Vec<(RegionKey, Dist)> {
        let Some(i) = self.position(from) else { return Vec::new() };
        let per_draw: Vec<Vec<f64>> = (0..self.draws.len()).map(|d| self.k_step_vec(d, i, k)).collect();
        let mut out: Vec<(RegionKey, Dist)> = self
            .index
            .iter()
            .enumerate()
            .map(|(j, key)| (key.clone(), Dist::from_samples(per_draw.iter().map(|v| v[j]).collect(), self.level)))
            .collect();
        out.sort_by(|a, b| b.1.mean.total_cmp(&a.1.mean).then_with(|| a.0.cmp(&b.0)));
        out
    }

    /// One-step successors of `region` other than itself, ranked by posterior mean, at most `n`.
    pub fn neighbors(&self, region: &str, n: usize) -> Vec<(RegionKey, Dist)> {
        self.k_step(region, 1).into_iter().filter(|(k, d)| k != region && d.mean > 0.0).take(n).collect()
    }

    /// Probability of following `path` exactly (`path[0]` given), as a distribution.
    pub fn path_prob(&self, path: &[RegionKey]) -> Option<Dist> {
        let idx: Vec<usize> = path.iter().map(|k| self.position(k)).collect::<Option<_>>()?;
        let xs = self.draws.iter().map(|p| idx.windows(2).map(|w| p[w[0]][w[1]]).product::<f64>()).collect();
        Some(Dist::from_samples(xs, self.level))
    }

    /// Monte Carlo walks: path `i` uses kernel draw `i mod n_draws`, so the spread of the
    /// paths includes parameter uncertainty (posterior predictive).
    pub fn sample_paths(&self, from: &str, k: usize, n: usize, rng: &mut Rng) -> Vec<Vec<RegionKey>> {
        let Some(start) = self.position(from) else { return Vec::new() };
        (0..n)
            .map(|i| {
                let p = &self.draws[i % self.draws.len()];
                let mut cur = start;
                let mut path = vec![self.index[cur].clone()];
                for _ in 0..k {
                    let u = rng.uniform();
                    let mut acc = 0.0;
                    let mut next = cur;
                    for (j, pj) in p[cur].iter().enumerate() {
                        acc += pj;
                        if u < acc {
                            next = j;
                            break;
                        }
                    }
                    cur = next;
                    path.push(self.index[cur].clone());
                }
                path
            })
            .collect()
    }

    /// Exact first passage from `from` into `targets` within `horizon` steps, per draw.
    pub fn hitting(&self, from: &str, targets: &[RegionKey], horizon: u32) -> Option<Hitting> {
        let i = self.position(from)?;
        let t: Vec<usize> = targets.iter().filter_map(|k| self.position(k)).collect();
        if t.is_empty() {
            return None;
        }
        let h = horizon.max(1) as usize;
        let mut p_hit = Vec::with_capacity(self.draws.len());
        let mut expected = Vec::with_capacity(self.draws.len());
        let mut conditional = Vec::new();
        let mut pmf_sum = vec![0.0; h];
        for p in &self.draws {
            if t.contains(&i) {
                p_hit.push(1.0);
                expected.push(0.0);
                conditional.push(0.0);
                continue;
            }
            let mut v = vec![0.0; self.index.len()];
            v[i] = 1.0;
            let mut hit = 0.0;
            let mut steps = 0.0;
            for (s, slot) in pmf_sum.iter_mut().enumerate() {
                v = Self::step(&v, p);
                let f: f64 = t.iter().map(|j| v[*j]).sum();
                for j in &t {
                    v[*j] = 0.0;
                }
                hit += f;
                steps += (s + 1) as f64 * f;
                *slot += f;
            }
            p_hit.push(hit);
            expected.push(steps + h as f64 * (1.0 - hit));
            if hit > 1e-12 {
                conditional.push(steps / hit);
            }
        }
        let n = self.draws.len() as f64;
        Some(Hitting {
            p_hit: Dist::from_samples(p_hit, self.level),
            expected_steps: Dist::from_samples(expected, self.level),
            conditional_steps: Dist::from_samples(conditional, self.level),
            pmf: pmf_sum.into_iter().map(|x| x / n).collect(),
            horizon: h as u32,
        })
    }

    /// Recurrent sub-kernel of draw `d`: source states only, rows renormalised over them.
    fn recurrent(&self, d: usize) -> (Vec<usize>, Vec<Vec<f64>>) {
        let states: Vec<usize> = (0..self.index.len()).filter(|&i| self.source[i]).collect();
        let m = states
            .iter()
            .map(|&a| {
                let row: Vec<f64> = states.iter().map(|&b| self.draws[d][a][b]).collect();
                let s: f64 = row.iter().sum();
                if s > 0.0 {
                    row.into_iter().map(|x| x / s).collect()
                } else {
                    states.iter().map(|&b| if a == b { 1.0 } else { 0.0 }).collect()
                }
            })
            .collect();
        (states, m)
    }

    /// Stationary distribution: a direct solve of `π(P − I) = 0, Σπ = 1` (Gaussian elimination
    /// with partial pivoting, O(n³)); reducible chains (singular system) fall back to the lazy
    /// power iteration from the uniform start.
    fn stationary_of(m: &[Vec<f64>]) -> Vec<f64> {
        Self::stationary_solve(m).unwrap_or_else(|| Self::stationary_power(m))
    }

    fn stationary_solve(m: &[Vec<f64>]) -> Option<Vec<f64>> {
        let n = m.len();
        if n == 0 {
            return None;
        }
        // rows = equations: (Pᵀ − I) π = 0 for j < n−1, last equation Σ π = 1
        let mut a: Vec<Vec<f64>> = (0..n)
            .map(|j| {
                if j == n - 1 {
                    vec![1.0; n + 1]
                } else {
                    let mut row: Vec<f64> = (0..n).map(|i| m[i][j] - if i == j { 1.0 } else { 0.0 }).collect();
                    row.push(0.0);
                    row
                }
            })
            .collect();
        for c in 0..n {
            let p = (c..n).max_by(|&x, &y| a[x][c].abs().total_cmp(&a[y][c].abs()))?;
            if a[p][c].abs() < 1e-10 {
                return None;
            }
            a.swap(c, p);
            let pivot = a[c].clone();
            for (r, row) in a.iter_mut().enumerate() {
                if r != c && row[c] != 0.0 {
                    let f = row[c] / pivot[c];
                    for (x, p) in row[c..].iter_mut().zip(&pivot[c..]) {
                        *x -= f * p;
                    }
                }
            }
        }
        let pi: Vec<f64> = (0..n).map(|i| (a[i][n] / a[i][i]).max(0.0)).collect();
        let s: f64 = pi.iter().sum();
        (s > 0.0 && pi.iter().all(|x| x.is_finite())).then(|| pi.into_iter().map(|x| x / s).collect())
    }

    fn stationary_power(m: &[Vec<f64>]) -> Vec<f64> {
        let n = m.len();
        let mut pi = vec![1.0 / n as f64; n];
        for _ in 0..2000 {
            // lazy chain (P + I) / 2: same stationary distribution, no periodicity
            let next: Vec<f64> = Self::step(&pi, m).iter().zip(&pi).map(|(a, b)| 0.5 * (a + b)).collect();
            let diff: f64 = next.iter().zip(&pi).map(|(a, b)| (a - b).abs()).sum();
            pi = next;
            if diff < 1e-12 {
                break;
            }
        }
        let s: f64 = pi.iter().sum();
        pi.into_iter().map(|x| x / s).collect()
    }

    /// Stationary distribution over the recurrent states, each probability a [`Dist`], sorted
    /// by posterior mean (descending). Empty when no state was observed as a source.
    pub fn stationary(&self) -> Vec<(RegionKey, Dist)> {
        let mut samples: Vec<Vec<f64>> = Vec::new();
        let mut states = Vec::new();
        for d in 0..self.draws.len() {
            let (s, m) = self.recurrent(d);
            if s.is_empty() {
                return Vec::new();
            }
            samples.push(Self::stationary_of(&m));
            states = s;
        }
        let mut out: Vec<(RegionKey, Dist)> = states
            .iter()
            .enumerate()
            .map(|(j, &i)| {
                (self.index[i].clone(), Dist::from_samples(samples.iter().map(|v| v[j]).collect(), self.level))
            })
            .collect();
        out.sort_by(|a, b| b.1.mean.total_cmp(&a.1.mean).then_with(|| a.0.cmp(&b.0)));
        out
    }

    /// Relaxation time `1 / (1 − |λ₂|)` of the recurrent chain (steps), as a distribution;
    /// capped at `cap`. A larger value means slower mixing / more persistent regimes.
    pub fn relaxation(&self, cap: f64) -> Dist {
        let xs = (0..self.draws.len())
            .filter_map(|d| {
                let (s, m) = self.recurrent(d);
                (s.len() >= 2).then(|| {
                    let pi = Self::stationary_of(&m);
                    let lam = second_eigen_modulus(&m, &pi);
                    if lam >= 1.0 - 1.0 / cap { cap } else { 1.0 / (1.0 - lam) }
                })
            })
            .collect();
        Dist::from_samples(xs, self.level)
    }
}

/// `|λ₂|` of a stochastic matrix by power iteration on `v ↦ vP − (Σv)π` (removes the unit
/// eigenvalue), averaging the norm ratio over the last iterations (complex pairs oscillate).
fn second_eigen_modulus(m: &[Vec<f64>], pi: &[f64]) -> f64 {
    let n = m.len();
    // deterministic, not orthogonal to any eigenvector in practice
    let mut v: Vec<f64> = (0..n).map(|i| ((i * 7919 + 13) % 101) as f64 / 101.0 - 0.5).collect();
    let mut log_ratio: Vec<f64> = Vec::new();
    for it in 0..300 {
        // converged: the mean log ratio of the last 20 iterations equals that of the 20 before
        if it >= 100 && it % 10 == 0 && log_ratio.len() >= 40 {
            let k = log_ratio.len();
            let a = log_ratio[k - 20..].iter().sum::<f64>() / 20.0;
            let b = log_ratio[k - 40..k - 20].iter().sum::<f64>() / 20.0;
            if (a - b).abs() < 1e-5 {
                log_ratio.drain(..k - 20);
                break;
            }
        }
        let s: f64 = v.iter().sum();
        let mut next = Posterior::step(&v, m);
        for (x, p) in next.iter_mut().zip(pi) {
            *x -= s * p;
        }
        let a: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let b: f64 = next.iter().map(|x| x * x).sum::<f64>().sqrt();
        if a < 1e-300 || b < 1e-300 {
            return 0.0;
        }
        if it >= 60 {
            log_ratio.push((b / a).ln());
        }
        v = next.into_iter().map(|x| x / b).collect();
    }
    let tail = &log_ratio[log_ratio.len().saturating_sub(60)..];
    let mean = tail.iter().sum::<f64>() / tail.len().max(1) as f64;
    mean.exp().clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::UNKNOWN;

    /// Synthetic chain with known kernel (test data, not market data).
    fn chain() -> MarkovGraph {
        let mut g = MarkovGraph::new(0.01, 1.0);
        for (from, to, n) in [("a", "a", 70), ("a", "b", 30), ("b", "a", 40), ("b", "b", 60)] {
            for _ in 0..n {
                g.observe(from, to);
            }
        }
        g
    }

    #[test]
    fn direct_stationary_solve_matches_the_power_iteration() {
        let m = vec![vec![0.9, 0.1, 0.0], vec![0.05, 0.9, 0.05], vec![0.0, 0.2, 0.8]];
        let a = Posterior::stationary_solve(&m).unwrap();
        let b = Posterior::stationary_power(&m);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-9, "{a:?} {b:?}");
        }
        // two closed classes: singular → power-iteration fallback
        let red = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!(Posterior::stationary_solve(&red).is_none());
        assert_eq!(Posterior::stationary_of(&red), vec![0.5, 0.5]);
    }

    #[test]
    fn transition_probabilities_are_distributions_around_the_matrix_power() {
        let g = chain();
        let mut rng = Rng::seed(1);
        let post = Posterior::draw(&g, 400, 0.9, &mut rng);
        let d = post.transition_prob("a", "b", 1).unwrap();
        assert!((d.mean - 0.3).abs() < 0.02, "{d:?}");
        assert!(d.lower < 0.3 && 0.3 < d.upper && d.sd > 0.01);
        // two steps: P²[a→b] = .7·.3 + .3·.6 = .39
        let d2 = post.transition_prob("a", "b", 2).unwrap();
        assert!((d2.mean - 0.39).abs() < 0.02, "{d2:?}");
        assert!((d2.mean - g.k_step_prob("a", "b", 2)).abs() < 0.02);
        let ks: BTreeMap<_, _> = post.k_step("a", 3).into_iter().map(|(k, d)| (k, d.mean)).collect();
        assert!((ks.values().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(post.transition_prob("zz", "a", 1).is_none());
        let nb = post.neighbors("a", 3);
        assert_eq!(nb[0].0, "b");
    }

    #[test]
    fn stationary_and_relaxation_match_the_two_state_chain() {
        // π = (b/(a+b), a/(a+b)) with a = P[a→b] = .3, b = P[b→a] = .4 → (4/7, 3/7); λ₂ = 1 − .3 − .4 = .3
        let g = chain();
        let post = Posterior::draw(&g, 200, 0.9, &mut Rng::seed(2));
        let st: BTreeMap<_, _> = post.stationary().into_iter().collect();
        assert!(!st.contains_key(UNKNOWN), "only recurrent states");
        assert!((st["a"].mean - 4.0 / 7.0).abs() < 0.03, "{st:?}");
        assert!(st["a"].lower < 4.0 / 7.0 && 4.0 / 7.0 < st["a"].upper);
        let r = post.relaxation(1e4);
        assert!((r.median - 1.0 / 0.7).abs() < 0.2, "{r:?}");
    }

    #[test]
    fn hitting_time_is_exact_first_passage() {
        // From a, hit b: geometric with p = .3 → P(τ ≤ 5) = 1 − .7⁵, E[τ | hit] < 5
        let g = chain();
        let post = Posterior::draw(&g, 300, 0.9, &mut Rng::seed(3));
        let h = post.hitting("a", &["b".into()], 5).unwrap();
        let want = 1.0 - 0.7f64.powi(5);
        assert!((h.p_hit.mean - want).abs() < 0.03, "{h:?}");
        assert!((h.pmf[0] - 0.3).abs() < 0.03);
        assert!((h.pmf.iter().sum::<f64>() - h.p_hit.mean).abs() < 1e-9);
        assert!(h.conditional_steps.mean < h.expected_steps.mean);
        let at = post.hitting("b", &["b".into()], 5).unwrap();
        assert_eq!(at.p_hit.mean, 1.0);
        assert!(post.hitting("a", &["nope".into()], 5).is_none());
    }

    #[test]
    fn path_probabilities_and_sampled_paths() {
        let g = chain();
        let post = Posterior::draw(&g, 200, 0.9, &mut Rng::seed(4));
        let p = post.path_prob(&["a".into(), "a".into(), "b".into()]).unwrap();
        assert!((p.mean - 0.21).abs() < 0.02, "{p:?}");
        let mut rng = Rng::seed(5);
        let paths = post.sample_paths("a", 4, 500, &mut rng);
        assert_eq!(paths.len(), 500);
        assert!(paths.iter().all(|p| p.len() == 5 && p[0] == "a"));
        let to_b = paths.iter().filter(|p| p[1] == "b").count() as f64 / 500.0;
        assert!((to_b - 0.3).abs() < 0.07, "{to_b}");
        let (t, frac) = crate::hitting_time(&paths, &["b".into()], 4);
        let h = post.hitting("a", &["b".into()], 4).unwrap();
        assert!((frac - h.p_hit.mean).abs() < 0.06 && (t - h.expected_steps.mean).abs() < 0.3);
    }

    #[test]
    fn credible_intervals_cover_the_true_kernel_at_the_nominal_rate() {
        // Spec 33 §9.1: on a known synthetic chain, 90 % intervals cover the truth on ≥ 85 % of edges.
        let truth = [[0.6, 0.25, 0.1, 0.05], [0.1, 0.5, 0.3, 0.1], [0.2, 0.1, 0.6, 0.1], [0.25, 0.25, 0.25, 0.25]];
        let names = ["r0", "r1", "r2", "r3"];
        let mut covered = 0;
        let mut total = 0;
        for seed in 0..20u64 {
            let mut rng = Rng::seed(100 + seed);
            let mut g = MarkovGraph::new(0.5, 1.0);
            for n in names {
                g.register_region(n);
            }
            let mut cur = 0usize;
            for _ in 0..400 {
                let u = rng.uniform();
                let mut acc = 0.0;
                let mut next = 3;
                for (j, p) in truth[cur].iter().enumerate() {
                    acc += p;
                    if u < acc {
                        next = j;
                        break;
                    }
                }
                g.observe(names[cur], names[next]);
                cur = next;
            }
            let post = Posterior::draw(&g, 300, 0.9, &mut rng);
            for (i, from) in names.iter().enumerate() {
                for (j, to) in names.iter().enumerate() {
                    let d = post.transition_prob(from, to, 1).unwrap();
                    total += 1;
                    if d.lower <= truth[i][j] && truth[i][j] <= d.upper {
                        covered += 1;
                    }
                }
            }
        }
        let rate = covered as f64 / total as f64;
        assert!(rate >= 0.85, "coverage {rate}");
    }
}
