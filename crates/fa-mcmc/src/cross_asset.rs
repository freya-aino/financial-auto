//! Cross-asset priors (spec 33 §6.5): covariance, lead-lag and predictive quality between
//! assets at one timescale, and the ranked pair proposals for asset selection (spec 31),
//! RNN-prior decoder pairs (spec 34) and strategy asset sets (spec 35).
//!
//! Per instrument the tracker keeps the last `window` closed-bar log returns keyed by bar close
//! time. For every ordered pair `(a, b)` on the union of their bar closes inside the overlap of
//! both series (a close missing on one side is a bar without trades: return 0):
//!
//! - **shrunk correlation**: Ledoit–Wolf shrinkage of the standardised 2×2 covariance toward
//!   the identity (`ρ̂ = (1 − δ)·ρ`, `δ` from the LW formula);
//! - **lead-lag**: the lag `ℓ ∈ 1..=max_lag` maximising `|corr(a_{t−ℓ}, b_t)|`;
//! - **predictive gain**: `Δ CRPS` of a Gaussian next-step model of `b` when `a_{t−ℓ}` joins its
//!   own lag (`b_t ~ 1 + b_{t−1}` vs `b_t ~ 1 + b_{t−1} + a_{t−ℓ}`, OLS): for a calibrated
//!   Gaussian `E[CRPS] = σ/√π`, so `gain = (σ₀ − σ₁)/√π` in return units;
//! - **uncertainty**: the posterior sd of the gain, `gain / |t|` with `t` the coefficient's
//!   t-statistic (Bayesian linear regression with a flat prior).
//!
//! Proposals are the pairs with positive gain ranked by `gain / uncertainty` (= `|t|`), top `N`.
//!
//! **Expert residuals** (spec 33 §6.5): when the caller also pushes the residual of each return
//! against the instrument's own expert view ([`CrossAssetPriors::push_with_residual`]; model-runtime
//! uses `r_t − Σ_k w_k · μ_k(region_k at t−1)`, the router-weighted region-conditional mean return),
//! the shrunk residual correlation is reported next to the return correlation and the lead-lag,
//! predictive gain and uncertainty are computed on the residuals: the gain then measures what
//! `a`'s view adds to `b`'s **expert** next-step model, not to a bare AR(1). Pairs without
//! residuals fall back to returns (`basis = "return"`).
//!
//! **Evolution evidence** (spec 33 §6.5 "fitness results flow back as evidence"): every evaluated
//! `src → target` decoder (spec 34 `nn_forecast` with a `src` gene) is a Bernoulli trial of the
//! pair — a win when its CRPS skill over the naive forecast is positive. With a Beta(1, 1) prior
//! the pair's evidence weight is `2·(1 + wins)/(2 + wins + losses)` (1 without evidence, → 2 for
//! a pair that keeps winning, → 0 for one that keeps losing), and
//! [`CrossAssetPriors::proposals_with_evidence`] ranks by `score = |t| · weight` (`raw_score`
//! keeps `|t|`), so evolution's results raise or lower the pair's prior.
//!
//! The tracker is `serde` (model-runtime persists it to Garage and restores it on start).

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CrossAssetCfg {
    /// Returns kept per instrument.
    pub window: usize,
    pub max_lag: usize,
    /// Common observations needed before a pair is evaluated.
    pub min_obs: usize,
    /// `cross_prior_top_n`.
    pub top_n: usize,
    /// Up to this many instruments every ordered pair is evaluated; above it pairs go through
    /// the candidate screen ([`CrossAssetPriors::candidates`], spec 33 §10 q5).
    pub exhaustive_max: usize,
    /// Screened sources kept per target (top-k by lagged correlation).
    pub per_target: usize,
    /// Extra sources per target drawn pseudo-randomly per round (exploration).
    pub explore: usize,
}

impl Default for CrossAssetCfg {
    fn default() -> Self {
        Self { window: 500, max_lag: 5, min_obs: 60, top_n: 50, exhaustive_max: 48, per_target: 8, explore: 2 }
    }
}

/// Result of [`CrossAssetPriors::rank`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Ranked {
    /// Top-N proposals (positive gain, ranked by evidence-weighted score).
    pub proposals: Vec<CrossPrior>,
    /// Candidate pairs considered.
    pub candidates: usize,
    /// Candidate pairs that were evaluable (enough joint observations).
    pub evaluated: usize,
    /// The candidate screen ran (more than `exhaustive_max` instruments).
    pub screened: bool,
}

/// One directed pair: `source` (a) → `target` (b).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossPrior {
    pub source: String,
    pub target: String,
    pub n: usize,
    /// Ledoit–Wolf shrunk contemporaneous correlation.
    pub correlation: f64,
    pub shrinkage: f64,
    /// Bars by which `source` leads `target`.
    pub lead_lag: usize,
    pub lead_corr: f64,
    /// `Δ CRPS` (return units) from adding `source` to `target`'s own model.
    pub predictive_gain: f64,
    /// Posterior sd of the gain.
    pub uncertainty: f64,
    /// `predictive_gain / uncertainty` (ranking key).
    pub score: f64,
    /// Ledoit–Wolf shrunk correlation of the expert residuals (= `correlation` on basis `return`).
    #[serde(default)]
    pub residual_correlation: f64,
    /// `residual` (lead-lag and gain on expert residuals) or `return`.
    #[serde(default)]
    pub basis: String,
    /// `gain / uncertainty` before the evolution evidence (= `score` without evidence).
    #[serde(default)]
    pub raw_score: f64,
    /// Evolution evidence of the pair (wins, losses of evaluated src→target decoders).
    #[serde(default)]
    pub evidence: PairEvidence,
}

/// Evolution evidence of one directed pair (spec 33 §6.5 feedback).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct PairEvidence {
    /// Evaluated src→target decoders with positive CRPS skill.
    pub wins: f64,
    /// … with non-positive skill.
    pub losses: f64,
}

impl PairEvidence {
    /// Folds one evaluated decoder's skill in.
    pub fn record(&mut self, skill: f64) {
        if !skill.is_finite() {
            return;
        }
        if skill > 0.0 {
            self.wins += 1.0;
        } else {
            self.losses += 1.0;
        }
    }

    /// `2·(1 + wins)/(2 + wins + losses)`: Beta(1, 1) posterior mean of a win, scaled so no
    /// evidence is neutral (1).
    pub fn weight(&self) -> f64 {
        2.0 * (1.0 + self.wins) / (2.0 + self.wins + self.losses)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CrossAssetPriors {
    cfg: CrossAssetCfg,
    series: BTreeMap<String, VecDeque<(i64, f64)>>,
    /// Expert residuals at the same closes (optional per instrument).
    #[serde(default)]
    resid: BTreeMap<String, VecDeque<(i64, f64)>>,
}

/// Real observations each side needs inside the overlap (the rest may be filled with 0).
const MIN_OWN_OBS: usize = 20;

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len().max(1) as f64
}

fn corr(x: &[f64], y: &[f64]) -> f64 {
    let (mx, my) = (mean(x), mean(y));
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx).powi(2);
        syy += (b - my).powi(2);
    }
    if sxx <= 0.0 || syy <= 0.0 { 0.0 } else { sxy / (sxx * syy).sqrt() }
}

fn standardise(x: &[f64]) -> Vec<f64> {
    let m = mean(x);
    let sd = (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len().max(1) as f64).sqrt();
    if sd <= 0.0 { vec![0.0; x.len()] } else { x.iter().map(|v| (v - m) / sd).collect() }
}

/// Ledoit–Wolf intensity for the standardised pair (target = identity).
fn ledoit_wolf(x: &[f64], y: &[f64]) -> (f64, f64) {
    let (x, y) = (standardise(x), standardise(y));
    let n = x.len() as f64;
    let s = [[1.0, mean(&x.iter().zip(&y).map(|(a, b)| a * b).collect::<Vec<_>>())], [0.0, 1.0]];
    let s01 = s[0][1];
    // π̂ = (1/n²) Σ_t ‖x_t x_tᵀ − S‖²_F ; d̂² = ‖S − I‖²_F
    let pi: f64 = x
        .iter()
        .zip(&y)
        .map(|(a, b)| (a * a - 1.0).powi(2) + (b * b - 1.0).powi(2) + 2.0 * (a * b - s01).powi(2))
        .sum::<f64>()
        / (n * n);
    let d2 = 2.0 * s01 * s01;
    let delta = if d2 <= 0.0 { 1.0 } else { (pi / d2).clamp(0.0, 1.0) };
    ((1.0 - delta) * s01, delta)
}

/// OLS of `y` on the columns of `x` (with intercept). Returns (residual sd, coef, se) of the
/// last column, or `None` when singular.
fn ols_last(y: &[f64], cols: &[&[f64]]) -> Option<(f64, f64, f64)> {
    let n = y.len();
    let k = cols.len() + 1;
    if n <= k + 1 {
        return None;
    }
    let row = |t: usize| -> Vec<f64> { std::iter::once(1.0).chain(cols.iter().map(|c| c[t])).collect() };
    let mut xtx = vec![vec![0.0; k]; k];
    let mut xty = vec![0.0; k];
    for (t, yt) in y.iter().enumerate() {
        let r = row(t);
        for i in 0..k {
            xty[i] += r[i] * yt;
            for j in 0..k {
                xtx[i][j] += r[i] * r[j];
            }
        }
    }
    let inv = invert(xtx)?;
    let beta: Vec<f64> = (0..k).map(|i| (0..k).map(|j| inv[i][j] * xty[j]).sum()).collect();
    let rss: f64 = y
        .iter()
        .enumerate()
        .map(|(t, yt)| {
            let r = row(t);
            (yt - r.iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>()).powi(2)
        })
        .sum();
    let s2 = rss / (n - k) as f64;
    let se = (s2 * inv[k - 1][k - 1]).max(0.0).sqrt();
    Some((s2.sqrt(), beta[k - 1], se))
}

fn invert(mut a: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut inv: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, p);
        inv.swap(c, p);
        let d = a[c][c];
        for j in 0..n {
            a[c][j] /= d;
            inv[c][j] /= d;
        }
        for i in 0..n {
            if i != c {
                let f = a[i][c];
                for j in 0..n {
                    a[i][j] -= f * a[c][j];
                    inv[i][j] -= f * inv[c][j];
                }
            }
        }
    }
    Some(inv)
}

impl CrossAssetPriors {
    pub fn new(cfg: CrossAssetCfg) -> Self {
        Self { cfg, series: BTreeMap::new(), resid: BTreeMap::new() }
    }

    /// Keeps the series but adopts `cfg` (a restored tracker follows the current tunables).
    pub fn set_cfg(&mut self, cfg: CrossAssetCfg) {
        self.cfg = cfg;
    }

    /// A closed-bar log return and its residual against the instrument's expert view.
    pub fn push_with_residual(&mut self, instrument: &str, t_ms: i64, ret: f64, resid: f64) {
        if !ret.is_finite() || !resid.is_finite() {
            return;
        }
        let before = self.series.get(instrument).map_or(0, VecDeque::len);
        let last = self.series.get(instrument).and_then(|s| s.back().map(|x| x.0));
        self.push(instrument, t_ms, ret);
        if last.is_some_and(|t| t >= t_ms) && before > 0 {
            return; // stale: ignored by `push` too
        }
        let r = self.resid.entry(instrument.to_owned()).or_default();
        r.push_back((t_ms, resid));
        while r.len() > self.cfg.window {
            r.pop_front();
        }
    }

    /// A closed-bar log return of `instrument` at bar close `t_ms`.
    pub fn push(&mut self, instrument: &str, t_ms: i64, ret: f64) {
        if !ret.is_finite() {
            return;
        }
        let s = self.series.entry(instrument.to_owned()).or_default();
        if s.back().is_some_and(|(t, _)| *t >= t_ms) {
            return;
        }
        s.push_back((t_ms, ret));
        while s.len() > self.cfg.window {
            s.pop_front();
        }
    }

    pub fn instruments(&self) -> usize {
        self.series.len()
    }

    /// Evaluates the directed pair `source → target`.
    pub fn pair(&self, source: &str, target: &str) -> Option<CrossPrior> {
        let a = self.series.get(source)?;
        let b = self.series.get(target)?;
        let bm: BTreeMap<i64, f64> = b.iter().copied().collect();
        let am: BTreeMap<i64, f64> = a.iter().copied().collect();
        // Union timeline over the overlap of both series. A bar close missing on one side means
        // no trade there, i.e. an unchanged price: its return is 0 (the next close's return
        // spans the gap), which is exactly a forward-filled price series.
        let lo = a.front()?.0.max(b.front()?.0);
        let hi = a.back()?.0.min(b.back()?.0);
        if lo > hi {
            return None; // no overlap (BTreeMap::range panics on an inverted range)
        }
        let mut ts: Vec<i64> = am.range(lo..=hi).chain(bm.range(lo..=hi)).map(|(t, _)| *t).collect();
        ts.sort_unstable();
        ts.dedup();
        let own_a = am.range(lo..=hi).count();
        let own_b = bm.range(lo..=hi).count();
        if ts.len() < self.cfg.min_obs || own_a.min(own_b) < MIN_OWN_OBS {
            return None;
        }
        let x: Vec<f64> = ts.iter().map(|t| am.get(t).copied().unwrap_or(0.0)).collect();
        let y: Vec<f64> = ts.iter().map(|t| bm.get(t).copied().unwrap_or(0.0)).collect();
        let (correlation, shrinkage) = ledoit_wolf(&x, &y);
        // expert residuals on the same timeline when both sides have them
        let (basis, x, y) = match (self.resid.get(source), self.resid.get(target)) {
            (Some(ra), Some(rb)) if ra.len() >= MIN_OWN_OBS && rb.len() >= MIN_OWN_OBS => {
                let ra: BTreeMap<i64, f64> = ra.iter().copied().collect();
                let rb: BTreeMap<i64, f64> = rb.iter().copied().collect();
                let xr: Vec<f64> = ts.iter().map(|t| ra.get(t).copied().unwrap_or(0.0)).collect();
                let yr: Vec<f64> = ts.iter().map(|t| rb.get(t).copied().unwrap_or(0.0)).collect();
                ("residual", xr, yr)
            }
            _ => ("return", x, y),
        };
        let residual_correlation = if basis == "residual" { ledoit_wolf(&x, &y).0 } else { correlation };
        let max_lag = self.cfg.max_lag.max(1).min(ts.len() / 4);
        let (lead_lag, lead_corr) = (1..=max_lag)
            .map(|l| (l, corr(&x[..x.len() - l], &y[l..])))
            .max_by(|p, q| p.1.abs().total_cmp(&q.1.abs()))?;
        let l = lead_lag.max(1);
        let start = l.max(1);
        let yt = &y[start..];
        let own = &y[start - 1..y.len() - 1];
        let lead = &x[start - l..x.len() - l];
        let (s0, _, _) = ols_last(yt, &[own])?;
        let (s1, beta, se) = ols_last(yt, &[own, lead])?;
        let gain = (s0 - s1) / std::f64::consts::PI.sqrt();
        let t = if se > 0.0 { beta / se } else { 0.0 };
        let uncertainty = if t.abs() > 1e-12 { gain.abs() / t.abs() } else { f64::INFINITY };
        let score = if uncertainty.is_finite() && uncertainty > 0.0 { gain / uncertainty } else { 0.0 };
        Some(CrossPrior {
            source: source.to_owned(),
            target: target.to_owned(),
            n: ts.len(),
            correlation,
            shrinkage,
            lead_lag: l,
            lead_corr,
            predictive_gain: gain,
            uncertainty,
            score,
            residual_correlation,
            basis: basis.into(),
            raw_score: score,
            evidence: PairEvidence::default(),
        })
    }

    /// Every evaluable directed pair, ranked by score (descending).
    pub fn all_pairs(&self) -> Vec<CrossPrior> {
        let names: Vec<&String> = self.series.keys().collect();
        let mut out: Vec<CrossPrior> = names
            .iter()
            .flat_map(|a| names.iter().filter(move |b| a != *b).map(move |b| (a, b)))
            .filter_map(|(a, b)| self.pair(a, b))
            .collect();
        out.sort_by(|p, q| {
            q.score.total_cmp(&p.score).then_with(|| (&p.source, &p.target).cmp(&(&q.source, &q.target)))
        });
        out
    }

    /// Top-N proposals: positive predictive gain, ranked by `gain / uncertainty`.
    pub fn proposals(&self) -> Vec<CrossPrior> {
        self.proposals_with_evidence(&BTreeMap::new())
    }

    /// Candidate pairs `(source, target)` to evaluate (spec 33 §10 q5, universe scale).
    ///
    /// With at most `exhaustive_max` instruments: every ordered pair. Above it, the union of
    /// - the **screen**: per target, the `per_target` sources with the largest
    ///   `max_ℓ |corr(a_{t−ℓ}, b_t)|`, `ℓ ∈ 1..=max_lag`, on one common grid (the last `window`
    ///   bar closes of any instrument; a missing close is a 0 return as in [`Self::pair`];
    ///   residuals where an instrument has them, else returns; standardised). This is the
    ///   statistic the exact evaluation maximises, so a pair that would rank high passes it;
    ///   cost `n²·max_lag·window` multiply-adds without allocation, versus `n²` full pair
    ///   evaluations (timeline merge, two OLS fits) before;
    /// - every pair with **evolution evidence** (a pair evolution keeps testing stays scored);
    /// - `explore` **random** sources per target, rotating with `round` (pairs the screen misses,
    ///   e.g. non-linear lead, still get evaluated now and then).
    ///
    /// Exact evaluation then costs `O(n · (per_target + explore) + |evidence|)` pairs.
    pub fn candidates(&self, evidence: &BTreeMap<(String, String), PairEvidence>, round: u64) -> Vec<(String, String)> {
        let names: Vec<&String> = self.series.keys().collect();
        let n = names.len();
        if n <= self.cfg.exhaustive_max.max(2) {
            return names
                .iter()
                .flat_map(|a| names.iter().filter(move |b| a != *b).map(move |b| ((*a).clone(), (*b).clone())))
                .collect();
        }
        let grid = self.screen_grid();
        let g = grid.first().map_or(0, Vec::len);
        let lags = self.cfg.max_lag.max(1).min(g / 4).max(1);
        let mut out: std::collections::BTreeSet<(String, String)> = std::collections::BTreeSet::new();
        for (j, b) in grid.iter().enumerate() {
            if b.is_empty() {
                continue;
            }
            let mut scores: Vec<(f64, usize)> = grid
                .iter()
                .enumerate()
                .filter(|(i, a)| *i != j && !a.is_empty())
                .map(|(i, a)| {
                    let best = (1..=lags)
                        .map(|l| {
                            let dot: f64 = a[..g - l].iter().zip(&b[l..]).map(|(x, y)| x * y).sum();
                            (dot / (g - l) as f64).abs()
                        })
                        .fold(0.0, f64::max);
                    (best, i)
                })
                .collect();
            let k = self.cfg.per_target.min(scores.len());
            if k > 0 {
                scores.select_nth_unstable_by(k - 1, |p, q| q.0.total_cmp(&p.0).then(p.1.cmp(&q.1)));
                for (_, i) in &scores[..k] {
                    out.insert((names[*i].clone(), names[j].clone()));
                }
            }
            for e in 0..self.cfg.explore {
                let h = splitmix(round.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((j as u64) << 20) ^ e as u64);
                let i = (h % n as u64) as usize;
                if i != j {
                    out.insert((names[i].clone(), names[j].clone()));
                }
            }
        }
        for (a, b) in evidence.keys() {
            if a != b && self.series.contains_key(a) && self.series.contains_key(b) {
                out.insert((a.clone(), b.clone()));
            }
        }
        out.into_iter().collect()
    }

    /// Standardised screening vectors on a common grid, in `series` key order (empty for an
    /// instrument without variance on the grid).
    fn screen_grid(&self) -> Vec<Vec<f64>> {
        let mut times: Vec<i64> = self.series.values().flat_map(|s| s.iter().map(|x| x.0)).collect();
        times.sort_unstable();
        times.dedup();
        let keep = self.cfg.window.max(self.cfg.min_obs).min(times.len());
        let times = &times[times.len() - keep..];
        self.series
            .iter()
            .map(|(name, s)| {
                let src = match self.resid.get(name) {
                    Some(r) if r.len() >= MIN_OWN_OBS => r,
                    _ => s,
                };
                let mut v = vec![0.0; times.len()];
                let mut own = 0;
                for (t, x) in src {
                    if let Ok(i) = times.binary_search(t) {
                        v[i] = *x;
                        own += 1;
                    }
                }
                if own < MIN_OWN_OBS {
                    return Vec::new();
                }
                let m = mean(&v);
                let sd = (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt();
                if sd <= 1e-15 || !sd.is_finite() {
                    return Vec::new();
                }
                v.iter_mut().for_each(|x| *x = (*x - m) / sd);
                v
            })
            .collect()
    }

    /// Ranked proposals over the [`Self::candidates`] of `round`, with the evolution evidence
    /// folded in (`score = raw_score · evidence.weight()`), top `top_n`.
    pub fn rank(&self, evidence: &BTreeMap<(String, String), PairEvidence>, round: u64) -> Ranked {
        let cands = self.candidates(evidence, round);
        let screened = self.series.len() > self.cfg.exhaustive_max.max(2);
        let evaluated: Vec<CrossPrior> = cands.iter().filter_map(|(a, b)| self.pair(a, b)).collect();
        let n_eval = evaluated.len();
        let mut v: Vec<CrossPrior> = evaluated
            .into_iter()
            .filter(|p| p.predictive_gain > 0.0)
            .map(|mut p| {
                if let Some(e) = evidence.get(&(p.source.clone(), p.target.clone())) {
                    p.evidence = *e;
                    p.score = p.raw_score * e.weight();
                }
                p
            })
            .collect();
        v.sort_by(|p, q| q.score.total_cmp(&p.score).then_with(|| (&p.source, &p.target).cmp(&(&q.source, &q.target))));
        v.truncate(self.cfg.top_n);
        Ranked { proposals: v, candidates: cands.len(), evaluated: n_eval, screened }
    }

    /// Top-N proposals with the evolution evidence of `(source, target)` pairs folded in:
    /// `score = raw_score · evidence.weight()`, ranked by it (spec 33 §6.5 feedback); the
    /// candidates of round 0 ([`Self::rank`]).
    pub fn proposals_with_evidence(&self, evidence: &BTreeMap<(String, String), PairEvidence>) -> Vec<CrossPrior> {
        self.rank(evidence, 0).proposals
    }
}

#[cfg(test)]
mod tests {
    use fa_core::rng::Rng;

    use super::*;

    #[test]
    fn the_candidate_screen_finds_the_leader_in_a_large_universe() {
        // 120 instruments (> exhaustive_max): one lead → follow pair, the rest noise.
        let mut rng = Rng::seed(11);
        let cfg = CrossAssetCfg { window: 300, ..CrossAssetCfg::default() };
        let mut cp = CrossAssetPriors::new(cfg);
        let mut lead_hist = vec![0.0; 2];
        for t in 0..300i64 {
            let l = 0.01 * rng.normal();
            lead_hist.push(l);
            cp.push("lead", t * 60_000, l);
            cp.push("follow", t * 60_000, 0.7 * lead_hist[lead_hist.len() - 2] + 0.005 * rng.normal());
            for i in 0..118 {
                cp.push(&format!("n{i:03}"), t * 60_000, 0.01 * rng.normal());
            }
        }
        let mut ev = BTreeMap::new();
        ev.insert(("n000".to_owned(), "n001".to_owned()), PairEvidence { wins: 3.0, losses: 0.0 });
        let r = cp.rank(&ev, 7);
        assert!(r.screened);
        // per target ≤ per_target + explore sources, plus the evidence pair; far below n(n−1)
        assert!(r.candidates <= 120 * 10 + 1 && r.candidates >= 120 * 8, "{}", r.candidates);
        assert!(r.evaluated <= r.candidates);
        let top = &r.proposals[0];
        assert_eq!((top.source.as_str(), top.target.as_str(), top.lead_lag), ("lead", "follow", 1));
        let c = cp.candidates(&ev, 7);
        assert!(c.contains(&("n000".to_owned(), "n001".to_owned())), "evidence pairs stay candidates");
        assert_eq!(c, cp.candidates(&ev, 7), "deterministic per round");
        assert_ne!(c, cp.candidates(&ev, 8), "exploration rotates");
        // small universes keep the exhaustive set
        let mut small = CrossAssetPriors::new(cfg);
        for t in 0..100i64 {
            for i in 0..5 {
                small.push(&format!("s{i}"), t, rng.normal());
            }
        }
        assert_eq!(small.candidates(&BTreeMap::new(), 0).len(), 20);
        assert!(!small.rank(&BTreeMap::new(), 0).screened);
    }

    #[test]
    fn a_leading_asset_is_proposed_and_independent_noise_is_not() {
        // Synthetic returns: `lead` drives `follow` two bars later; `noise` is independent.
        let mut rng = Rng::seed(7);
        let mut cp = CrossAssetPriors::new(CrossAssetCfg { window: 400, ..CrossAssetCfg::default() });
        let mut lead_hist = vec![0.0; 3];
        for t in 0..400i64 {
            let l = 0.01 * rng.normal();
            lead_hist.push(l);
            let f = 0.8 * lead_hist[lead_hist.len() - 3] + 0.003 * rng.normal();
            cp.push("lead", t * 60_000, l);
            cp.push("follow", t * 60_000, f);
            cp.push("noise", t * 60_000, 0.01 * rng.normal());
        }
        let p = cp.pair("lead", "follow").unwrap();
        assert_eq!(p.lead_lag, 2);
        assert!(p.lead_corr > 0.9, "{p:?}");
        assert!(p.predictive_gain > 0.0 && p.score > 10.0, "{p:?}");
        let props = cp.proposals();
        assert_eq!((props[0].source.as_str(), props[0].target.as_str()), ("lead", "follow"));
        let n = cp.pair("noise", "follow").unwrap();
        assert!(n.score < 4.0, "{n:?}");
        assert!(n.correlation.abs() < 0.15 && n.shrinkage > 0.0);
        // stale or duplicate timestamps are ignored
        cp.push("lead", 0, 1.0);
        assert_eq!(cp.instruments(), 3);
    }

    #[test]
    fn residuals_reveal_a_lead_hidden_by_a_shared_regime_mean() {
        // Both assets move with a large shared regime mean the experts explain; only the residual
        // of `a` leads the residual of `b` by one bar.
        let mut rng = Rng::seed(21);
        let mut cp = CrossAssetPriors::new(CrossAssetCfg::default());
        let mut prev_ea = 0.0;
        for t in 0..400i64 {
            let regime = if (t / 20) % 2 == 0 { 0.02 } else { -0.02 };
            let ea = 0.002 * rng.normal();
            let eb = 0.9 * prev_ea + 0.0005 * rng.normal();
            cp.push_with_residual("a", t, regime + 0.01 * rng.normal() + ea, ea);
            cp.push_with_residual("b", t, regime + 0.01 * rng.normal() + eb, eb);
            prev_ea = ea;
        }
        let p = cp.pair("a", "b").unwrap();
        assert_eq!((p.basis.as_str(), p.lead_lag), ("residual", 1));
        assert!(p.lead_corr > 0.9 && p.score > 10.0, "{p:?}");
        assert!(p.correlation > 0.5, "returns share the regime: {p:?}");
        assert!(p.residual_correlation.abs() < 0.2, "{p:?}");
        // serde round trip (persistence)
        let back: CrossAssetPriors = serde_json::from_str(&serde_json::to_string(&cp).unwrap()).unwrap();
        assert_eq!(back.pair("a", "b").unwrap().lead_lag, 1);
    }

    /// Spec 33 §6.5: evolution results re-rank the proposals. Two equally good leaders; the one
    /// whose src→target decoders keep losing drops below the one that keeps winning.
    #[test]
    fn evolution_evidence_reranks_pairs() {
        let mut rng = Rng::seed(3);
        let mut cp = CrossAssetPriors::new(CrossAssetCfg::default());
        let mut la = vec![0.0; 2];
        let mut lb = vec![0.0; 2];
        for t in 0..400i64 {
            let (a, b) = (0.01 * rng.normal(), 0.01 * rng.normal());
            la.push(a);
            lb.push(b);
            cp.push("a", t, a);
            cp.push("b", t, b);
            cp.push("ta", t, 0.8 * la[la.len() - 2] + 0.004 * rng.normal());
            cp.push("tb", t, 0.8 * lb[lb.len() - 2] + 0.004 * rng.normal());
        }
        let base = cp.proposals();
        let rank = |v: &[CrossPrior], s: &str, t: &str| v.iter().position(|p| p.source == s && p.target == t).unwrap();
        let (ra, rb) = (rank(&base, "a", "ta"), rank(&base, "b", "tb"));
        let (winner, loser) = if ra < rb { (("b", "tb"), ("a", "ta")) } else { (("a", "ta"), ("b", "tb")) };
        let mut ev = BTreeMap::new();
        let mut w = PairEvidence::default();
        let mut l = PairEvidence::default();
        for _ in 0..6 {
            w.record(0.05);
            l.record(-0.02);
        }
        assert!((PairEvidence::default().weight() - 1.0).abs() < 1e-12);
        assert!(w.weight() > 1.7 && l.weight() < 0.3);
        ev.insert((winner.0.to_owned(), winner.1.to_owned()), w);
        ev.insert((loser.0.to_owned(), loser.1.to_owned()), l);
        let re = cp.proposals_with_evidence(&ev);
        assert!(rank(&re, winner.0, winner.1) < rank(&re, loser.0, loser.1), "{re:?}");
        let top = &re[rank(&re, winner.0, winner.1)];
        assert!((top.score - top.raw_score * w.weight()).abs() < 1e-9);
        assert_eq!(top.evidence.wins, 6.0);
    }

    #[test]
    fn too_few_common_bars_give_nothing() {
        let mut cp = CrossAssetPriors::new(CrossAssetCfg::default());
        for t in 0..10 {
            cp.push("a", t, 0.1);
            cp.push("b", t, 0.2);
        }
        assert!(cp.pair("a", "b").is_none());
        assert!(cp.proposals().is_empty());
    }

    #[test]
    fn sparse_series_align_on_the_union_timeline() {
        // `a` trades every bar, `b` every other bar carrying a's return of the bar before (lag 1
        // on the forward-filled timeline). Exact-timestamp alignment would halve the sample.
        let mut rng = Rng::seed(11);
        let mut cp = CrossAssetPriors::new(CrossAssetCfg { min_obs: 60, ..CrossAssetCfg::default() });
        let mut prev = 0.0;
        let mut acc = 0.0;
        for t in 0..300i64 {
            let r = 0.01 * rng.normal();
            cp.push("a", t, r);
            acc += prev;
            if t % 2 == 1 {
                cp.push("b", t, acc + 0.0005 * rng.normal());
                acc = 0.0;
            }
            prev = r;
        }
        let p = cp.pair("a", "b").expect("union timeline has enough bars");
        assert!(p.n >= 290, "{p:?}");
        assert!(p.predictive_gain > 0.0);
        assert!(cp.pair("b", "a").is_some());
    }
}
