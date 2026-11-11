//! Hungarian (Kuhn–Munkres) assignment, used for region identity across refits (spec 30 §6.4
//! step 4, D-007). Pure Rust, `O(n³)` on the padded square matrix, deterministic.

/// Maximum-weight assignment on a rectangular `rows × cols` matrix of non-negative weights.
///
/// Returns, for every row, the column it is assigned to (`None` when the row is left over in a
/// rectangular problem or only assignable to a zero-weight cell). Every column is used at most
/// once. Ties are broken deterministically by the algorithm's row order.
pub fn max_weight_assignment(weights: &[Vec<f64>]) -> Vec<Option<usize>> {
    let rows = weights.len();
    let cols = weights.iter().map(Vec::len).max().unwrap_or(0);
    if rows == 0 || cols == 0 {
        return vec![None; rows];
    }
    let n = rows.max(cols);
    let max_w = weights.iter().flatten().copied().filter(|w| w.is_finite()).fold(0.0f64, f64::max);
    // Minimisation cost: max − w (padding cells cost `max_w`, i.e. weight 0).
    let cost = |i: usize, j: usize| -> f64 {
        let w = weights.get(i).and_then(|r| r.get(j)).copied().filter(|w| w.is_finite()).unwrap_or(0.0).max(0.0);
        max_w - w
    };
    // Jonker-style potentials formulation (1-indexed, row 0 / col 0 are sentinels).
    let inf = f64::INFINITY;
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; n + 1];
    let mut p = vec![0usize; n + 1]; // p[j] = row matched to column j
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=n {
                if used[j] {
                    continue;
                }
                let cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut out = vec![None; rows];
    for (j, &i) in p.iter().enumerate().skip(1) {
        if i == 0 || i > rows || j > cols {
            continue;
        }
        let w = weights[i - 1].get(j - 1).copied().unwrap_or(0.0);
        if w > 0.0 {
            out[i - 1] = Some(j - 1);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(w: &[Vec<f64>], a: &[Option<usize>]) -> f64 {
        a.iter().enumerate().filter_map(|(i, j)| j.map(|j| w[i][j])).sum()
    }

    #[test]
    fn beats_greedy_on_the_classic_counterexample() {
        // Greedy takes (0,0)=0.9 and then (1,1)=0.1 → 1.0; optimal is (0,1)+(1,0) = 0.8+0.8.
        let w = vec![vec![0.9, 0.8], vec![0.8, 0.1]];
        let a = max_weight_assignment(&w);
        assert_eq!(a, vec![Some(1), Some(0)]);
        assert!((total(&w, &a) - 1.6).abs() < 1e-12);
    }

    #[test]
    fn rectangular_and_zero_rows() {
        let w = vec![vec![0.0, 0.0, 0.0], vec![0.2, 0.7, 0.1]];
        assert_eq!(max_weight_assignment(&w), vec![None, Some(1)]);
        let tall = vec![vec![0.5], vec![0.9], vec![0.1]];
        assert_eq!(max_weight_assignment(&tall), vec![None, Some(0), None]);
        assert!(max_weight_assignment(&[]).is_empty());
    }

    #[test]
    fn matches_brute_force_on_random_matrices() {
        let mut rng = fa_core::rng::Rng::seed(9);
        for _ in 0..50 {
            let n = 1 + rng.below(5);
            let m = 1 + rng.below(5);
            let w: Vec<Vec<f64>> = (0..n).map(|_| (0..m).map(|_| rng.uniform()).collect()).collect();
            let a = max_weight_assignment(&w);
            // brute force over permutations of columns padded with "none"
            fn best(w: &[Vec<f64>], i: usize, used: &mut Vec<bool>) -> f64 {
                if i == w.len() {
                    return 0.0;
                }
                let mut b = best(w, i + 1, used);
                for j in 0..w[i].len() {
                    if !used[j] {
                        used[j] = true;
                        b = b.max(w[i][j] + best(w, i + 1, used));
                        used[j] = false;
                    }
                }
                b
            }
            let opt = best(&w, 0, &mut vec![false; m]);
            assert!((total(&w, &a) - opt).abs() < 1e-9, "{w:?} {a:?}");
        }
    }
}
