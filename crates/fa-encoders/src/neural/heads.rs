//! Constrained least-squares solver for the per-expert next-step heads (spec 32 §6.3.1 "next
//! steps": the residual covariance is quadratic in the linear heads given the latents).
//!
//! Each expert `k` predicts the `T` normalised next-step targets with a linear head on its own
//! latent, `ŷ_kt = θ_kt · [μ_k, 1]`. Its scalarised residual (the one the health report
//! correlates) is `r_k = (1/T) Σ_t (ŷ_kt − y_t)`, linear in `θ_k`. With the other experts' heads
//! fixed, the covariance `cov_S(r_k, r_j)` over a window set `S` is **linear** in `θ_k`, so
//!
//! ```text
//!   min_θk  Σ_n g_nk ‖ŷ_k(n) − y(n)‖²  (+ ridge)      s.t.  cov_{S_kj}(r_k, r_j) = c_kj   ∀ j ≠ k
//! ```
//!
//! is an equality-constrained weighted least-squares problem solved exactly by its KKT system.
//! `g_nk` are the top-k router gates (the expert's skill is measured on the windows routed to it,
//! spec 32 §6.4), `S_kj` the windows where both experts are routed (all windows when fewer than
//! `min_shared` are shared — exactly the health report's rule), and `c_kj` the target covariance
//! `±ρ* · sd_k · sd_j` (`ρ*` = `target_corr`, sign of the current covariance; 0 = exact
//! decorrelation). Block-coordinate sweeps over the experts keep every pair feasible after the
//! first sweep (an update of `k` enforces all pairs of `k` and leaves the others untouched) and
//! never increase the objective (the previous `θ_k` stays feasible for its own block).

// dense linear algebra over small index ranges reads best with explicit indices
#![allow(clippy::needless_range_loop)]

/// Tunables of [`solve`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadSolverCfg {
    /// Target |correlation| of every expert pair's scalarised residuals (in sample).
    pub target_corr: f64,
    /// Block-coordinate sweeps over the experts.
    pub sweeps: usize,
    /// Ridge on the head weights, relative to the mean diagonal of the gated Gram matrix.
    pub ridge: f64,
    /// Shared routed windows a pair needs before its correlation is taken on them only.
    pub min_shared: usize,
    /// Initial disagreement (spec 32 §6.3.1 bound): the constraint is bilinear across experts,
    /// and from identical least-squares residuals no single expert can decorrelate alone (it
    /// would have to predict the noise). Before the constrained sweeps each expert's head gets
    /// a seeded disagreement `d_k` with `var(d_k) = spread · (K − 1) · V` (`V` the mean residual
    /// variance — the bound's minimum for exact decorrelation), built from the directions its
    /// latent shares with the others, so the sweeps can settle into the symmetric-disagreement
    /// solution. 0 = start from least squares.
    pub spread: f64,
    pub seed: u64,
}

impl Default for HeadSolverCfg {
    fn default() -> Self {
        Self { target_corr: 0.0, sweeps: 12, ridge: 1e-3, min_shared: 30, spread: 1.0, seed: 0x4EAD }
    }
}

/// What the solver reached on its own (training) windows.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HeadSolverReport {
    pub resid_corr_max: f64,
    /// Gate-weighted skill per expert (1 − MSE / mean y²).
    pub skill: Vec<f64>,
    /// Objective after each sweep (non-increasing).
    pub objective: Vec<f64>,
    /// Constraint rows dropped as degenerate (zero covariance direction).
    pub dropped: usize,
}

/// Heads `θ[k][t][p]` (`p < d` weights on the latent, `p = d` the bias).
pub type Heads = Vec<Vec<Vec<f64>>>;

fn features(mu: &[f32]) -> Vec<f64> {
    mu.iter().map(|v| f64::from(*v)).chain(std::iter::once(1.0)).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Dense solve with partial pivoting; `None` when singular.
fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for i in c + 1..n {
            let f = a[i][c] / a[c][c];
            if f != 0.0 {
                for j in c..n {
                    a[i][j] -= f * a[c][j];
                }
                b[i] -= f * b[c];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|j| a[i][j] * x[j]).sum();
        x[i] = (b[i] - s) / a[i][i];
    }
    Some(x)
}

struct Problem<'a> {
    /// `x[n][k]` features of window n for expert k.
    x: Vec<Vec<Vec<f64>>>,
    gates: &'a [Vec<f32>],
    y: Vec<Vec<f64>>,
    ybar: Vec<f64>,
    t: usize,
    p: usize,
    /// Window set per pair (i < j).
    pairs: Vec<Vec<Vec<usize>>>,
}

impl Problem<'_> {
    fn resid(&self, k: usize, th: &[Vec<f64>]) -> Vec<f64> {
        (0..self.y.len())
            .map(|n| (0..self.t).map(|t| dot(&th[t], &self.x[n][k]) - self.y[n][t]).sum::<f64>() / self.t as f64)
            .collect()
    }

    fn set(&self, i: usize, j: usize) -> &[usize] {
        if i < j { &self.pairs[i][j] } else { &self.pairs[j][i] }
    }

    fn objective(&self, heads: &Heads) -> f64 {
        let mut s = 0.0;
        for (n, xn) in self.x.iter().enumerate() {
            for (k, th) in heads.iter().enumerate() {
                let g = f64::from(self.gates[n][k]);
                if g > 0.0 {
                    s += g * (0..self.t).map(|t| (dot(&th[t], &xn[k]) - self.y[n][t]).powi(2)).sum::<f64>();
                }
            }
        }
        s
    }
}

/// Seeded initial disagreement of the heads (see [`HeadSolverCfg::spread`]): expert `k` adds
/// `s_k · z_k(x)` to every target, `z_k` a random unit-variance combination of its centred
/// latent and `s_k` scaled to the target variance. The signs alternate around the ensemble so
/// the disagreements of different experts start anti-correlated where their latents overlap.
fn spread(pr: &Problem<'_>, heads: &mut Heads, resid: &mut [Vec<f64>], cfg: &HeadSolverCfg) {
    let k = heads.len();
    let n = pr.y.len() as f64;
    let var = |r: &[f64]| {
        let m = r.iter().sum::<f64>() / n;
        r.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n
    };
    let v = resid.iter().map(|r| var(r)).sum::<f64>() / k as f64;
    let target = cfg.spread * (k as f64 - 1.0) * v;
    let mut rng = fa_core::rng::Rng::seed(cfg.seed);
    // one shared random direction in the common latent coordinates, sign-alternated per expert,
    // plus a private random part so no two experts' disagreements are collinear
    let d = pr.p - 1;
    let shared: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
    for e in 0..k {
        let sign = if e % 2 == 0 { 1.0 } else { -1.0 };
        let dir: Vec<f64> = shared.iter().map(|s| sign * s + 0.5 * rng.normal()).collect();
        let z: Vec<f64> = pr.x.iter().map(|xw| dot(&dir, &xw[e][..d])).collect();
        let vz = var(&z);
        if vz <= 1e-12 {
            continue;
        }
        let scale = (target / vz).sqrt();
        let mz = z.iter().sum::<f64>() / n;
        for row in heads[e].iter_mut() {
            for q in 0..d {
                row[q] += scale * dir[q];
            }
            row[d] -= scale * mz;
        }
        resid[e] = pr.resid(e, &heads[e]);
    }
}

fn centred_cov(a: &[f64], b: &[f64], set: &[usize]) -> (f64, f64, f64) {
    let m = set.len().max(1) as f64;
    let (ma, mb) = (set.iter().map(|&n| a[n]).sum::<f64>() / m, set.iter().map(|&n| b[n]).sum::<f64>() / m);
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for &n in set {
        c += (a[n] - ma) * (b[n] - mb);
        va += (a[n] - ma).powi(2);
        vb += (b[n] - mb).powi(2);
    }
    (c, va, vb)
}

/// Solves the heads (see the module docs). `mu[n][k]` latents, `gates[n][k]` top-k gates, `y[n]`
/// targets, `init` the starting heads. Returns the heads and the in-sample report.
pub fn solve(
    mu: &[Vec<Vec<f32>>],
    gates: &[Vec<f32>],
    y: &[Vec<f32>],
    init: Heads,
    cfg: &HeadSolverCfg,
) -> (Heads, HeadSolverReport) {
    let n = mu.len();
    let k = init.len();
    if n == 0 || k == 0 {
        return (init, HeadSolverReport::default());
    }
    let t = init[0].len();
    let p = init[0][0].len();
    let x: Vec<Vec<Vec<f64>>> = mu.iter().map(|w| w.iter().map(|m| features(m)).collect()).collect();
    let yv: Vec<Vec<f64>> = y.iter().map(|r| r.iter().map(|v| f64::from(*v)).collect()).collect();
    let ybar: Vec<f64> = yv.iter().map(|r| r.iter().sum::<f64>() / t as f64).collect();
    let active = |w: usize, e: usize| gates[w].get(e).is_some_and(|g| *g > 0.0);
    let mut pairs = vec![vec![Vec::new(); k]; k];
    for i in 0..k {
        for j in i + 1..k {
            let shared: Vec<usize> = (0..n).filter(|&w| active(w, i) && active(w, j)).collect();
            pairs[i][j] = if shared.len() >= cfg.min_shared { shared } else { (0..n).collect() };
        }
    }
    let pr = Problem { x, gates, y: yv, ybar, t, p, pairs };
    let mut heads = init;
    let mut resid: Vec<Vec<f64>> = (0..k).map(|e| pr.resid(e, &heads[e])).collect();
    let mut report = HeadSolverReport::default();
    // sweep 0: the unconstrained gated least-squares heads; then the experts are pushed apart
    // (see `HeadSolverCfg::spread`) and the constrained sweeps start from there
    for sweep in 0..=cfg.sweeps.max(1) {
        if sweep == 1 && cfg.spread > 0.0 && k > 1 {
            spread(&pr, &mut heads, &mut resid, cfg);
        }
        for e in 0..k {
            // gated Gram matrix and right-hand sides
            let mut h = vec![vec![0.0; pr.p]; pr.p];
            let mut b = vec![vec![0.0; pr.p]; pr.t];
            for (w, xw) in pr.x.iter().enumerate() {
                let g = f64::from(gates[w][e]);
                if g <= 0.0 {
                    continue;
                }
                let xe = &xw[e];
                for a in 0..pr.p {
                    for c in 0..pr.p {
                        h[a][c] += g * xe[a] * xe[c];
                    }
                    for tt in 0..pr.t {
                        b[tt][a] += g * pr.y[w][tt] * xe[a];
                    }
                }
            }
            let diag = (0..pr.p).map(|a| h[a][a]).sum::<f64>() / pr.p as f64;
            let ridge = cfg.ridge * diag.max(1e-6);
            for (a, row) in h.iter_mut().enumerate() {
                row[a] += ridge;
            }
            // one linear constraint per other expert
            let mut rows: Vec<(Vec<f64>, f64)> = Vec::new();
            for j in (0..k).filter(|j| *j != e && sweep > 0) {
                let set = pr.set(e, j);
                let m = set.len().max(1) as f64;
                let mj = set.iter().map(|&w| resid[j][w]).sum::<f64>() / m;
                let mut a = vec![0.0; pr.t * pr.p];
                let mut rhs_base = 0.0;
                let mut acc = vec![0.0; pr.p];
                for &w in set {
                    let c = resid[j][w] - mj;
                    for (q, v) in acc.iter_mut().enumerate() {
                        *v += c * pr.x[w][e][q];
                    }
                    rhs_base += pr.ybar[w] * c;
                }
                for tt in 0..pr.t {
                    for q in 0..pr.p {
                        a[tt * pr.p + q] = acc[q] / (pr.t as f64 * m);
                    }
                }
                let norm = a.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm < 1e-10 {
                    report.dropped += 1;
                    continue;
                }
                let target = if cfg.target_corr > 0.0 {
                    let (c, va, vb) = centred_cov(&resid[e], &resid[j], set);
                    cfg.target_corr * c.signum() * (va * vb).sqrt()
                } else {
                    0.0
                };
                rows.push((a, (target + rhs_base) / m));
            }
            // drop rows linearly dependent on earlier ones (e.g. two experts with equal
            // residuals): Gram–Schmidt on the rows, tracking the right-hand sides alike
            let mut basis: Vec<(Vec<f64>, f64)> = Vec::new();
            let mut kept = Vec::new();
            for (a, c) in rows {
                let (mut v, mut vc) = (a.clone(), c);
                for (u, uc) in &basis {
                    let f = dot(&v, u);
                    for (x, y) in v.iter_mut().zip(u) {
                        *x -= f * y;
                    }
                    vc -= f * uc;
                }
                let nv = dot(&v, &v).sqrt();
                if nv < 1e-8 * dot(&a, &a).sqrt() {
                    report.dropped += 1;
                    continue;
                }
                basis.push((v.iter().map(|x| x / nv).collect(), vc / nv));
                kept.push((a, c));
            }
            let rows = kept;
            // KKT: [2H 0..; A 0][θ; λ] = [2b; rhs]
            let nv = pr.t * pr.p;
            let mc = rows.len();
            let mut kkt = vec![vec![0.0; nv + mc]; nv + mc];
            let mut rhs = vec![0.0; nv + mc];
            for tt in 0..pr.t {
                for a in 0..pr.p {
                    for c in 0..pr.p {
                        kkt[tt * pr.p + a][tt * pr.p + c] = 2.0 * h[a][c];
                    }
                    rhs[tt * pr.p + a] = 2.0 * b[tt][a];
                }
            }
            for (r, (a, c)) in rows.iter().enumerate() {
                for (q, v) in a.iter().enumerate() {
                    kkt[nv + r][q] = *v;
                    kkt[q][nv + r] = *v;
                }
                rhs[nv + r] = *c;
            }
            if let Some(sol) = solve_dense(kkt, rhs) {
                heads[e] = (0..pr.t).map(|tt| sol[tt * pr.p..(tt + 1) * pr.p].to_vec()).collect();
                resid[e] = pr.resid(e, &heads[e]);
            }
        }
        if sweep > 0 {
            report.objective.push(pr.objective(&heads));
        }
    }
    let mut mx = 0.0f64;
    for i in 0..k {
        for j in i + 1..k {
            let (c, va, vb) = centred_cov(&resid[i], &resid[j], pr.set(i, j));
            if va > 0.0 && vb > 0.0 {
                mx = mx.max((c / (va * vb).sqrt()).abs());
            }
        }
    }
    report.resid_corr_max = mx;
    report.skill = (0..k)
        .map(|e| {
            let (mut num, mut den) = (0.0, 0.0);
            for w in 0..n {
                let g = f64::from(gates[w][e]);
                num += g * (0..pr.t).map(|tt| (dot(&heads[e][tt], &pr.x[w][e]) - pr.y[w][tt]).powi(2)).sum::<f64>();
                den += g * pr.y[w].iter().map(|v| v * v).sum::<f64>();
            }
            if den > 0.0 { 1.0 - num / den } else { 0.0 }
        })
        .collect();
    (heads, report)
}

#[cfg(test)]
mod tests {
    use fa_core::rng::Rng;

    use super::*;

    /// Experts reading one backbone: every latent holds the predictive factor `f`, two shared
    /// nuisance factors (independent of the targets) and a private dimension; targets are
    /// `0.8 f` plus noise. `gate(w)` gives the router gates of window `w`.
    #[allow(clippy::type_complexity)]
    fn data(
        n: usize,
        k: usize,
        seed: u64,
        gate: impl Fn(usize) -> Vec<f32>,
    ) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut rng = Rng::seed(seed);
        let (mut mu, mut y, mut g) = (Vec::new(), Vec::new(), Vec::new());
        for w in 0..n {
            let (f, z1, z2) = (rng.normal(), rng.normal(), rng.normal());
            mu.push(
                (0..k)
                    .map(|_| {
                        vec![
                            (f + 0.1 * rng.normal()) as f32,
                            (z1 + 0.1 * rng.normal()) as f32,
                            (z2 + 0.1 * rng.normal()) as f32,
                            rng.normal() as f32,
                        ]
                    })
                    .collect(),
            );
            y.push((0..3).map(|_| (0.8 * f + 0.6 * rng.normal()) as f32).collect());
            g.push(gate(w));
        }
        (mu, y, g)
    }

    /// Unconstrained least squares leaves the three experts' residuals almost perfectly
    /// correlated (they share the noise); the constrained solve reaches the target correlation
    /// in sample with every expert still skilled, never increases its objective and is
    /// deterministic.
    #[test]
    fn constrained_heads_decorrelate_residuals() {
        let k = 3;
        let (mu, y, gates) = data(900, k, 3, |_| vec![1.0 / 3.0; 3]);
        let init: Heads = vec![vec![vec![0.0; 5]; 3]; k];
        let (_, ls) =
            solve(&mu, &gates, &y, init.clone(), &HeadSolverCfg { sweeps: 1, spread: 0.0, ..Default::default() });
        let (h1, r1) = solve(&mu, &gates, &y, init.clone(), &HeadSolverCfg::default());
        assert!(r1.resid_corr_max < 1e-6, "exact decorrelation in sample: {r1:?}");
        assert!(r1.skill.iter().all(|s| *s > 0.1), "{r1:?} (least squares: {ls:?})");
        for w in r1.objective.windows(2) {
            assert!(w[1] <= w[0] * (1.0 + 1e-9), "{:?}", r1.objective);
        }
        let (h2, _) = solve(&mu, &gates, &y, init.clone(), &HeadSolverCfg::default());
        assert_eq!(h1, h2);
        let (_, r3) = solve(&mu, &gates, &y, init, &HeadSolverCfg { target_corr: 0.08, ..Default::default() });
        assert!(r3.resid_corr_max <= 0.1, "{r3:?}");
        assert!(
            r3.skill.iter().sum::<f64>() >= r1.skill.iter().sum::<f64>() - 1e-6,
            "a looser target costs less skill"
        );
    }

    /// Top-1 routing: skill counts on each expert's own windows only, correlation over all.
    #[test]
    fn top1_routing_decorrelates_with_skill() {
        let k = 3;
        let (mu, y, gates) = data(900, k, 9, |w| {
            let mut g = vec![0.0f32; 3];
            g[w % 3] = 1.0;
            g
        });
        let init: Heads = vec![vec![vec![0.0; 5]; 3]; k];
        let (_, r) = solve(&mu, &gates, &y, init, &HeadSolverCfg::default());
        assert!(r.resid_corr_max < 1e-6, "{r:?}");
        assert!(r.skill.iter().all(|s| *s > 0.1), "skilled on their own windows: {r:?}");
    }
}
