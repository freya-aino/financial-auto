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
}

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
        Self { k_eff: k_eff(&c), resid_corr: c, resid_corr_max: mx, skill, usage, n_windows: n }
    }

    pub fn skill_min(&self) -> f64 {
        if self.skill.is_empty() { 0.0 } else { self.skill.iter().copied().fold(f64::INFINITY, f64::min) }
    }
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
    fn health_report() {
        let r = vec![vec![1.0, -1.0, 1.0, -1.0], vec![1.0, 1.0, -1.0, -1.0]];
        let h = HealthReport::compute(&r, &[0.5, 0.8], 1.0, vec![0.6, 0.4]);
        assert!(h.resid_corr_max < 1e-12);
        assert!((h.k_eff - 2.0).abs() < 1e-9);
        assert_eq!(h.skill, vec![0.5, 0.19999999999999996]);
        assert!((h.skill_min() - 0.2).abs() < 1e-9);
    }
}
