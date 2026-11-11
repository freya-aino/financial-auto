//! Expert-space health (spec 32 §6.4) and expert identity across generations (§6.5), host side.

use serde::{Deserialize, Serialize};

use crate::pearson;

/// Health of one expert space on a validation window.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HealthReport {
    /// Residual correlation matrix `[K][K]` (scalarised residual: mean over targets of the
    /// standardised residuals).
    pub resid_corr: Vec<Vec<f64>>,
    /// Largest off-diagonal |corr|.
    pub resid_corr_max: f64,
    /// `(Σλ)² / Σλ²` over the eigenvalues of `resid_corr`.
    pub k_eff: f64,
    /// Per-expert skill vs the naive (training-mean) predictor: `1 − mse_k / mse_naive`.
    pub skill: Vec<f64>,
    /// Mean router probability per expert.
    pub usage: Vec<f64>,
    pub n_windows: usize,
    /// Calibration of each expert's `next_step` (§6.4), gate-weighted over the windows routed
    /// to it, averaged over targets: mean CRPS of its Gaussian predictive distribution.
    #[serde(default)]
    pub crps: Vec<f64>,
    /// CRPS skill vs the naive predictive distribution (training mean and spread: `N(0, 1)` on
    /// normalised targets): `1 − crps_k / crps_naive`.
    #[serde(default)]
    pub crpss: Vec<f64>,
    /// Kolmogorov–Smirnov distance of the expert's PIT values from uniform (0 = calibrated).
    #[serde(default)]
    pub pit_ks: Vec<f64>,
    /// Per-expert decodability (§6.4): best score per decoder family reading this expert's
    /// latent alone (`family → R²`-style score, higher = more decodable).
    #[serde(default)]
    pub decodability: Vec<std::collections::BTreeMap<String, f64>>,
    /// Collapse monitor (spec 31 §8): active latent units per expert (dims whose posterior mean
    /// varies across windows, variance > [`ACTIVE_UNIT_VAR`]).
    #[serde(default)]
    pub active_units: Vec<usize>,
    /// Collapse monitor, FSQ families: perplexity of each expert's code usage (`exp(H)`).
    #[serde(default)]
    pub code_perplexity: Vec<f64>,
    /// Experts flagged collapsed (posterior collapse: no active unit; codebook collapse:
    /// perplexity < [`MIN_CODE_PERPLEXITY`]); non-empty = the candidate is unfit.
    #[serde(default)]
    pub collapsed: Vec<usize>,
}

/// `(metric, labels, value)` rows of [`HealthReport::vm_series`].
pub type VmSeries = Vec<(String, Vec<(String, String)>, f64)>;

/// A latent dimension counts as active when the variance of its posterior mean over the
/// windows exceeds this (Burda et al.'s active-unit criterion).
pub const ACTIVE_UNIT_VAR: f64 = 1e-2;
/// An FSQ expert whose code perplexity is below this uses (effectively) fewer than 2 codes.
pub const MIN_CODE_PERPLEXITY: f64 = 2.0;

impl HealthReport {
    /// `resid[k][t]`: residual of expert `k` at window `t` (scalarised); `sq_err[k]` and
    /// `naive_sq_err` are mean squared errors over the window set.
    pub fn compute(resid: &[Vec<f64>], sq_err: &[f64], naive_sq_err: f64, usage: Vec<f64>) -> Self {
        let k = resid.len();
        let mut c = vec![vec![0.0; k]; k];
        let mut mx = 0.0f64;
        for i in 0..k {
            for j in 0..k {
                c[i][j] = if i == j { 1.0 } else { pearson(&resid[i], &resid[j]) };
                if i != j {
                    mx = mx.max(c[i][j].abs());
                }
            }
        }
        let skill = sq_err.iter().map(|m| if naive_sq_err > 0.0 { 1.0 - m / naive_sq_err } else { 0.0 }).collect();
        Self {
            k_eff: k_eff(&c),
            resid_corr: c,
            resid_corr_max: mx,
            skill,
            usage,
            n_windows: resid.first().map_or(0, Vec::len),
            ..Default::default()
        }
    }

    /// Gated form (spec 32 §6.3/§6.4): residual correlation of experts `i, j` over the windows
    /// where **both** are active (`gate > 0`, falling back to every window when fewer than
    /// `min_shared` share them), skill of expert `k` over the windows it is active on, weighted by
    /// its gate. `sq[k][t]` = squared error of expert `k` at window `t`, `y2[t]` = naive squared
    /// error, `gates[t][k]` = top-k gates.
    pub fn compute_gated(
        resid: &[Vec<f64>],
        sq: &[Vec<f64>],
        y2: &[f64],
        gates: &[Vec<f32>],
        usage: Vec<f64>,
        min_shared: usize,
    ) -> Self {
        let k = resid.len();
        let n = y2.len();
        let active = |t: usize, e: usize| gates.get(t).and_then(|g| g.get(e)).is_some_and(|g| *g > 0.0);
        let mut c = vec![vec![0.0; k]; k];
        let mut mx = 0.0f64;
        for i in 0..k {
            c[i][i] = 1.0;
            for j in (i + 1)..k {
                let shared: Vec<usize> = (0..n).filter(|&t| active(t, i) && active(t, j)).collect();
                let v = if shared.len() >= min_shared {
                    let a: Vec<f64> = shared.iter().map(|&t| resid[i][t]).collect();
                    let b: Vec<f64> = shared.iter().map(|&t| resid[j][t]).collect();
                    pearson(&a, &b)
                } else {
                    pearson(&resid[i], &resid[j])
                };
                c[i][j] = v;
                c[j][i] = v;
                mx = mx.max(v.abs());
            }
        }
        let skill = (0..k)
            .map(|e| {
                let (mut num, mut den) = (0.0, 0.0);
                for t in 0..n {
                    let g = f64::from(gates.get(t).and_then(|g| g.get(e)).copied().unwrap_or(0.0));
                    num += g * sq[e][t];
                    den += g * y2[t];
                }
                if den > 0.0 { 1.0 - num / den } else { 0.0 }
            })
            .collect();
        Self { k_eff: k_eff(&c), resid_corr: c, resid_corr_max: mx, skill, usage, n_windows: n, ..Default::default() }
    }

    /// Adds the §6.4 calibration of each expert's Gaussian `next_step`: `y[t][j]` realised
    /// (normalised) targets, `mean[k][t][j]` / `sd[k][t][j]` the expert's predictive mean and
    /// standard deviation, `gates[t][k]` top-k gates (each expert is scored where routed; every
    /// window when it is never routed).
    pub fn with_calibration(
        mut self,
        y: &[Vec<f64>],
        mean: &[Vec<Vec<f64>>],
        sd: &[Vec<Vec<f64>>],
        gates: &[Vec<f32>],
    ) -> Self {
        let k = mean.len();
        let (mut crps, mut crpss, mut ks) = (Vec::with_capacity(k), Vec::with_capacity(k), Vec::with_capacity(k));
        for e in 0..k {
            let mut wsum = 0.0;
            for t in 0..y.len() {
                wsum += f64::from(gates.get(t).and_then(|g| g.get(e)).copied().unwrap_or(0.0));
            }
            let weight = |t: usize| {
                if wsum > 0.0 { f64::from(gates.get(t).and_then(|g| g.get(e)).copied().unwrap_or(0.0)) } else { 1.0 }
            };
            let (mut c, mut c0, mut w) = (0.0, 0.0, 0.0);
            let mut pits: Vec<(f64, f64)> = Vec::new();
            for (t, yt) in y.iter().enumerate() {
                let g = weight(t);
                if g <= 0.0 {
                    continue;
                }
                for (j, yv) in yt.iter().enumerate() {
                    let (m, s) = (mean[e][t][j], sd[e][t][j].max(1e-6));
                    c += g * gaussian_crps(*yv, m, s);
                    c0 += g * gaussian_crps(*yv, 0.0, 1.0);
                    w += g;
                    pits.push((normal_cdf((yv - m) / s), g));
                }
            }
            let cm = if w > 0.0 { c / w } else { 0.0 };
            crps.push(cm);
            crpss.push(if c0 > 0.0 { 1.0 - c / c0 } else { 0.0 });
            ks.push(weighted_ks_uniform(&mut pits));
        }
        self.crps = crps;
        self.crpss = crpss;
        self.pit_ks = ks;
        self
    }

    /// Adds the collapse monitor (spec 31 §8): `mu[k][t][d]` posterior means per expert and
    /// window, `codes[k][t]` FSQ code indices (empty for continuous families).
    pub fn with_collapse(mut self, mu: &[Vec<Vec<f64>>], codes: &[Vec<u32>]) -> Self {
        self.active_units = mu.iter().map(|m| active_units(m)).collect();
        self.code_perplexity = codes.iter().map(|c| code_perplexity(c)).collect();
        let mut collapsed: Vec<usize> = (0..mu.len())
            .filter(|&e| {
                self.active_units[e] == 0 || self.code_perplexity.get(e).is_some_and(|p| *p < MIN_CODE_PERPLEXITY)
            })
            .collect();
        collapsed.dedup();
        self.collapsed = collapsed;
        self
    }

    /// The spec 32 §5 VictoriaMetrics series of this report (`metric, labels, value`):
    /// `fa_expert_resid_corr{tf, expert="<i>_<j>", src}` for every pair `i < j` and
    /// `fa_expert_usage|fa_expert_skill{tf, expert, src}` per expert, plus
    /// `fa_expert_k_eff{tf, src}`. `expert` = the position in `expert_ids` (stable ids after
    /// alignment); `src` = `validation` (a fit) or `live` (model-runtime).
    pub fn vm_series(&self, tf: &str, src: &str, expert_ids: &[u32]) -> VmSeries {
        let id = |e: usize| expert_ids.get(e).map_or(e.to_string(), u32::to_string);
        let labels = |extra: Option<String>| {
            let mut l = vec![("tf".to_owned(), tf.to_owned()), ("src".to_owned(), src.to_owned())];
            if let Some(x) = extra {
                l.push(("expert".to_owned(), x));
            }
            l
        };
        let mut out = Vec::new();
        for i in 0..self.resid_corr.len() {
            for j in (i + 1)..self.resid_corr.len() {
                out.push((
                    "fa_expert_resid_corr".to_owned(),
                    labels(Some(format!("{}_{}", id(i), id(j)))),
                    self.resid_corr[i][j],
                ));
            }
        }
        for (e, u) in self.usage.iter().enumerate() {
            out.push(("fa_expert_usage".to_owned(), labels(Some(id(e))), *u));
        }
        for (e, sk) in self.skill.iter().enumerate() {
            out.push(("fa_expert_skill".to_owned(), labels(Some(id(e))), *sk));
        }
        out.push(("fa_expert_k_eff".to_owned(), labels(None), self.k_eff));
        out.retain(|(_, _, v)| v.is_finite());
        out
    }

    /// Smallest per-expert CRPS skill (0 when uncalibrated).
    pub fn crpss_min(&self) -> f64 {
        self.crpss.iter().copied().reduce(f64::min).unwrap_or(0.0)
    }

    /// Largest per-expert PIT distance from uniform.
    pub fn pit_ks_max(&self) -> f64 {
        self.pit_ks.iter().copied().fold(0.0, f64::max)
    }

    pub fn skill_min(&self) -> f64 {
        if self.skill.is_empty() { 0.0 } else { self.skill.iter().copied().fold(f64::INFINITY, f64::min) }
    }
}

/// CRPS of `N(m, s²)` at `y` (closed form): `s·[z(2Φ(z) − 1) + 2φ(z) − 1/√π]`, `z = (y − m)/s`.
pub fn gaussian_crps(y: f64, m: f64, s: f64) -> f64 {
    let z = (y - m) / s;
    let pdf = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    s * (z * (2.0 * normal_cdf(z) - 1.0) + 2.0 * pdf - 1.0 / std::f64::consts::PI.sqrt())
}

/// Standard normal CDF (Abramowitz–Stegun 7.1.26 erf, |error| < 1.5e-7).
pub fn normal_cdf(z: f64) -> f64 {
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly =
        t * (0.254_829_592 + t * (-0.284_496_736 + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-x * x).exp();
    if z >= 0.0 { 0.5 * (1.0 + erf) } else { 0.5 * (1.0 - erf) }
}

/// Weighted Kolmogorov–Smirnov distance of PIT values `(u, weight)` from `U(0, 1)`.
pub fn weighted_ks_uniform(pits: &mut [(f64, f64)]) -> f64 {
    let total: f64 = pits.iter().map(|p| p.1).sum();
    if total <= 0.0 {
        return 0.0;
    }
    pits.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut acc, mut d) = (0.0, 0.0f64);
    for (u, w) in pits.iter() {
        d = d.max((acc / total - u).abs());
        acc += w;
        d = d.max((acc / total - u).abs());
    }
    d
}

/// Dims of `mu[t][d]` whose variance across windows exceeds [`ACTIVE_UNIT_VAR`].
pub fn active_units(mu: &[Vec<f64>]) -> usize {
    let n = mu.len();
    let Some(d) = mu.first().map(Vec::len) else { return 0 };
    if n < 2 {
        return 0;
    }
    (0..d)
        .filter(|&j| {
            let m = mu.iter().map(|r| r[j]).sum::<f64>() / n as f64;
            mu.iter().map(|r| (r[j] - m).powi(2)).sum::<f64>() / n as f64 > ACTIVE_UNIT_VAR
        })
        .count()
}

/// `exp(entropy)` of the empirical code distribution (∞-free: 0 for no codes).
pub fn code_perplexity(codes: &[u32]) -> f64 {
    if codes.is_empty() {
        return 0.0;
    }
    let mut counts = std::collections::BTreeMap::new();
    for c in codes {
        *counts.entry(*c).or_insert(0usize) += 1;
    }
    let n = codes.len() as f64;
    let h: f64 = counts.values().map(|&c| c as f64 / n).map(|p| -p * p.ln()).sum();
    h.exp()
}

/// Effective number of independent experts from a correlation matrix: `(Σλ)² / Σλ²`.
/// For a symmetric matrix `Σλ = tr(C)` and `Σλ² = ‖C‖_F²`, so no eigen-decomposition is needed.
pub fn k_eff(c: &[Vec<f64>]) -> f64 {
    let tr: f64 = (0..c.len()).map(|i| c[i][i]).sum();
    let fro: f64 = c.iter().flatten().map(|x| x * x).sum();
    if fro > 0.0 { tr * tr / fro } else { 0.0 }
}

/// Linear CKA between two representations of the same `n` samples (`a[n][da]`, `b[n][db]`):
/// `‖AᵀB‖_F² / (‖AᵀA‖_F · ‖BᵀB‖_F)` on column-centred matrices. 1 = same up to rotation/scale.
pub fn linear_cka(a: &[Vec<f32>], b: &[Vec<f32>]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let centre = |x: &[Vec<f32>]| -> Vec<Vec<f64>> {
        let d = x[0].len();
        let mut m = vec![0.0; d];
        for r in &x[..n] {
            for (j, v) in r.iter().enumerate() {
                m[j] += f64::from(*v) / n as f64;
            }
        }
        x[..n].iter().map(|r| r.iter().enumerate().map(|(j, v)| f64::from(*v) - m[j]).collect()).collect()
    };
    let (a, b) = (centre(a), centre(b));
    let cross = |x: &[Vec<f64>], y: &[Vec<f64>]| -> f64 {
        let (dx, dy) = (x[0].len(), y[0].len());
        let mut s = 0.0;
        for i in 0..dx {
            for j in 0..dy {
                let v: f64 = x.iter().zip(y).map(|(r, q)| r[i] * q[j]).sum();
                s += v * v;
            }
        }
        s
    };
    let ab = cross(&a, &b);
    let den = (cross(&a, &a) * cross(&b, &b)).sqrt();
    if den > 0.0 { ab / den } else { 0.0 }
}

/// One expert of a new generation matched to the previous one (spec 32 §6.5, `ALIGNED_TO`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertMatch {
    /// Index in the new snapshot.
    pub expert: usize,
    /// Parent expert it continues (`None` = new expert).
    pub parent_expert: Option<usize>,
    pub cka: f64,
}

/// Greedy one-to-one matching of new experts to parent experts by CKA (`cka[new][parent]`),
/// best pairs first; pairs below `threshold` stay unmatched.
pub fn match_experts(cka: &[Vec<f64>], threshold: f64) -> Vec<ExpertMatch> {
    let mut pairs: Vec<(usize, usize, f64)> =
        cka.iter().enumerate().flat_map(|(i, r)| r.iter().enumerate().map(move |(j, v)| (i, j, *v))).collect();
    pairs.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
    let mut out: Vec<ExpertMatch> =
        (0..cka.len()).map(|i| ExpertMatch { expert: i, parent_expert: None, cka: 0.0 }).collect();
    let mut used_new = vec![false; cka.len()];
    let mut used_old = vec![false; cka.first().map_or(0, Vec::len)];
    for (i, j, v) in pairs {
        if v < threshold || used_new[i] || used_old[j] {
            continue;
        }
        used_new[i] = true;
        used_old[j] = true;
        out[i] = ExpertMatch { expert: i, parent_expert: Some(j), cka: v };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_eff_bounds() {
        let id = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        assert!((k_eff(&id) - 3.0).abs() < 1e-12);
        let ones = vec![vec![1.0; 3]; 3];
        assert!((k_eff(&ones) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cka_is_rotation_invariant_and_matching_is_one_to_one() {
        let mut rng = fa_core::rng::Rng::seed(1);
        let a: Vec<Vec<f32>> = (0..50).map(|_| vec![rng.normal() as f32, rng.normal() as f32]).collect();
        let rot: Vec<Vec<f32>> = a.iter().map(|r| vec![r[1] * 2.0, -r[0] * 2.0]).collect();
        let noise: Vec<Vec<f32>> = (0..50).map(|_| vec![rng.normal() as f32, rng.normal() as f32]).collect();
        assert!((linear_cka(&a, &rot) - 1.0).abs() < 1e-6);
        assert!(linear_cka(&a, &noise) < 0.3);
        let m = match_experts(&[vec![0.2, 0.9], vec![0.8, 0.85], vec![0.1, 0.1]], 0.5);
        assert_eq!(m[0].parent_expert, Some(1));
        assert_eq!(m[1].parent_expert, Some(0));
        assert_eq!(m[2].parent_expert, None);
    }

    #[test]
    fn gated_health_uses_active_windows_only() {
        // expert 0 is perfect where active (t < 4), expert 1 where active (t >= 4)
        let resid =
            vec![vec![0.0, 0.0, 0.0, 0.0, 5.0, -5.0, 5.0, -5.0], vec![5.0, -5.0, 5.0, -5.0, 0.0, 0.0, 0.0, 0.0]];
        let sq: Vec<Vec<f64>> = resid.iter().map(|r| r.iter().map(|x| x * x).collect()).collect();
        let y2 = vec![1.0; 8];
        let gates: Vec<Vec<f32>> = (0..8).map(|t| if t < 4 { vec![1.0, 0.0] } else { vec![0.0, 1.0] }).collect();
        let h = HealthReport::compute_gated(&resid, &sq, &y2, &gates, vec![0.5, 0.5], 3);
        assert_eq!(h.skill, vec![1.0, 1.0], "specialists are scored where they are routed");
        let ungated = HealthReport::compute(&resid, &[12.5, 12.5], 1.0, vec![0.5, 0.5]);
        assert!(ungated.skill_min() < 0.0);
    }

    #[test]
    fn calibration_and_collapse_monitor() {
        let mut rng = fa_core::rng::Rng::seed(2);
        let n = 2000;
        let y: Vec<Vec<f64>> = (0..n).map(|_| vec![rng.normal()]).collect();
        // expert 0: calibrated N(0, 1); expert 1: overconfident N(0, 0.2²)
        let mean = vec![vec![vec![0.0]; n], vec![vec![0.0]; n]];
        let sd = vec![vec![vec![1.0]; n], vec![vec![0.2]; n]];
        let gates = vec![vec![0.5f32, 0.5]; n];
        let h = HealthReport::default().with_calibration(&y, &mean, &sd, &gates);
        assert!(h.pit_ks[0] < 0.05 && h.pit_ks[1] > 0.2, "{:?}", h.pit_ks);
        assert!(h.crpss[0].abs() < 1e-9 && h.crpss[1] < -0.2, "{:?}", h.crpss);
        assert!((gaussian_crps(0.0, 0.0, 1.0) - 0.233_695).abs() < 1e-4);
        assert!((normal_cdf(1.0) - 0.841_344_7).abs() < 1e-6);
        // collapse: expert 1's latent never moves; expert 0's codes use a single symbol
        let mu = vec![
            (0..50).map(|i| vec![(i as f64 * 0.1).sin()]).collect::<Vec<_>>(),
            (0..50).map(|_| vec![0.3]).collect::<Vec<_>>(),
        ];
        let h = HealthReport::default().with_collapse(&mu, &[vec![4; 50], (0..50).map(|i| i % 8).collect()]);
        assert_eq!(h.active_units, vec![1, 0]);
        assert!(h.code_perplexity[0] < 1.01 && h.code_perplexity[1] > 7.9);
        assert_eq!(h.collapsed, vec![0, 1]);
    }

    #[test]
    fn vm_series_cover_pairs_usage_and_skill() {
        let h = HealthReport::compute(
            &[vec![1.0, -1.0, 1.0], vec![1.0, 1.0, -1.0], vec![0.5, 0.1, 0.2]],
            &[0.5, 0.6, 0.7],
            1.0,
            vec![0.3, 0.3, 0.4],
        );
        let s = h.vm_series("1m", "validation", &[4, 7, 9]);
        let pairs: Vec<&str> = s
            .iter()
            .filter(|(m, _, _)| m == "fa_expert_resid_corr")
            .map(|(_, l, _)| l.iter().find(|(k, _)| k == "expert").unwrap().1.as_str())
            .collect();
        assert_eq!(pairs, vec!["4_7", "4_9", "7_9"]);
        assert_eq!(s.iter().filter(|(m, _, _)| m == "fa_expert_usage").count(), 3);
        assert!(
            s.iter().all(|(m, l, _)| m.starts_with("fa_")
                && l.iter().all(|(k, _)| ["tf", "src", "expert"].contains(&k.as_str())))
        );
    }

    #[test]
    fn health_report() {
        let r = vec![vec![1.0, -1.0, 1.0, -1.0], vec![1.0, 1.0, -1.0, -1.0]];
        let h = HealthReport::compute(&r, &[0.5, 0.8], 1.0, vec![0.6, 0.4]);
        assert!(h.resid_corr_max < 1e-12);
        assert!((h.k_eff - 2.0).abs() < 1e-9);
        assert_eq!(h.skill, vec![0.5, 0.19999999999999996]);
        assert!((h.skill_min() - 0.2).abs() < 1e-9);
    }
}
