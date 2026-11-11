//! Deterministic low-resolution chart rasters (spec 34 §6.1 "chart rendering", `fa-decoders::render`).
//!
//! A raster is `res` columns (time bins, oldest first) × `res` rows (price bins, row 0 = lowest),
//! stored **column-major** (`[col * res + row]`). Every column is a probability distribution over
//! rows: the price of that time bin is drawn as a Gaussian bump (σ = [`BUMP_SIGMA`] rows) and the
//! column is normalised to sum 1. This keeps the target smooth for a per-column cross-entropy loss
//! and lets the last column of a *future* raster be read back as a predictive distribution of the
//! horizon return ([`column_quantiles`]).

/// Width of the drawn line in rows (Gaussian σ).
pub const BUMP_SIGMA: f64 = 0.6;

/// Soft one-hot column: Gaussian bump around the fractional row `pos` ∈ [0, res − 1].
pub fn bump(pos: f64, res: usize) -> Vec<f32> {
    let pos = if pos.is_finite() { pos.clamp(0.0, (res - 1) as f64) } else { (res - 1) as f64 / 2.0 };
    let mut col: Vec<f64> =
        (0..res).map(|r| (-0.5 * ((r as f64 - pos) / BUMP_SIGMA).powi(2)).exp()).collect();
    let s: f64 = col.iter().sum();
    col.iter_mut().for_each(|v| *v /= s);
    col.into_iter().map(|v| v as f32).collect()
}

/// Mean of `xs` over the `res` equal time bins (bins of an `n < res` path repeat values).
fn binned(xs: &[f64], res: usize) -> Vec<f64> {
    let n = xs.len();
    (0..res)
        .map(|j| {
            let a = j * n / res;
            let b = ((j + 1) * n / res).max(a + 1).min(n);
            xs[a.min(n - 1)..b].iter().sum::<f64>() / (b - a.min(n - 1)) as f64
        })
        .collect()
}

/// Raster of a (log-)price path, min–max scaled into the rows. `None` for an empty path.
pub fn chart_raster(path: &[f64], res: usize) -> Option<Vec<f32>> {
    if path.is_empty() || res < 2 || path.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let bins = binned(path, res);
    let lo = bins.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = bins.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = hi - lo;
    let mut out = Vec::with_capacity(res * res);
    for b in bins {
        let pos = if span > 0.0 { (b - lo) / span * (res - 1) as f64 } else { (res - 1) as f64 / 2.0 };
        out.extend(bump(pos, res));
    }
    Some(out)
}

/// Fractional row of a return `r` on a future raster of half-height `scale` (row 0 = −scale).
pub fn row_of(r: f64, scale: f64, res: usize) -> f64 {
    (r / scale + 1.0) / 2.0 * (res - 1) as f64
}

/// Return at the centre of row `row` on a future raster of half-height `scale`.
pub fn row_value(row: f64, scale: f64, res: usize) -> f64 {
    (row / (res - 1) as f64 * 2.0 - 1.0) * scale
}

/// Raster of the next `cum.len()` steps: `cum[i]` = log return from now to step `i + 1`. Column
/// `j` shows the return at step `ceil((j + 1)·h / res)`, rows span `[−scale, scale]` (clamped).
pub fn future_raster(cum: &[f64], scale: f64, res: usize) -> Option<Vec<f32>> {
    let h = cum.len();
    if h == 0 || res < 2 || scale.is_nan() || scale <= 0.0 {
        return None;
    }
    let mut out = Vec::with_capacity(res * res);
    for j in 0..res {
        let idx = ((j + 1) * h).div_ceil(res).clamp(1, h) - 1;
        out.extend(bump(row_of(cum[idx], scale, res), res));
    }
    Some(out)
}

/// Quantiles of the return distribution of one future-raster column (`probs` over rows, any
/// non-negative weights): each row is a uniform bin of width `2·scale/(res − 1)` around its
/// centre; levels are read off the piecewise-linear CDF.
pub fn column_quantiles(probs: &[f32], levels: &[f64], scale: f64) -> Vec<f64> {
    let res = probs.len();
    let total: f64 = probs.iter().map(|p| f64::from(p.max(0.0))).sum();
    if res < 2 || total <= 0.0 {
        return vec![0.0; levels.len()];
    }
    let w = 2.0 * scale / (res - 1) as f64;
    levels
        .iter()
        .map(|&l| {
            let target = l.clamp(0.0, 1.0) * total;
            let mut cum = 0.0;
            for (r, p) in probs.iter().enumerate() {
                let p = f64::from(p.max(0.0));
                if cum + p >= target && p > 0.0 {
                    let frac = (target - cum) / p;
                    return row_value(r as f64, scale, res) - w / 2.0 + frac * w;
                }
                cum += p;
            }
            row_value((res - 1) as f64, scale, res) + w / 2.0
        })
        .collect()
}

/// `P(return > 0)` of a future-raster column (rows above the centre, half of a centre row).
pub fn prob_up(probs: &[f32]) -> f64 {
    let res = probs.len();
    let total: f64 = probs.iter().map(|p| f64::from(p.max(0.0))).sum();
    if total <= 0.0 {
        return 0.5;
    }
    let mid = (res - 1) as f64 / 2.0;
    probs
        .iter()
        .enumerate()
        .map(|(r, p)| {
            let p = f64::from(p.max(0.0));
            if (r as f64) > mid + 1e-9 {
                p
            } else if (r as f64 - mid).abs() < 1e-9 {
                p / 2.0
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / total
}

/// Net direction of a raster in [-1, 1]: expected row of the last column minus that of the first,
/// over the row span.
pub fn direction(raster: &[f32], res: usize) -> f64 {
    if res < 2 || raster.len() != res * res {
        return 0.0;
    }
    let mean_row = |c: usize| {
        let col = &raster[c * res..(c + 1) * res];
        let s: f64 = col.iter().map(|p| f64::from(p.max(0.0))).sum();
        if s <= 0.0 {
            return (res - 1) as f64 / 2.0;
        }
        col.iter().enumerate().map(|(r, p)| r as f64 * f64::from(p.max(0.0))).sum::<f64>() / s
    };
    ((mean_row(res - 1) - mean_row(0)) / (res - 1) as f64).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columns_are_distributions_and_follow_the_path() {
        let path: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let r = chart_raster(&path, 8).unwrap();
        assert_eq!(r.len(), 64);
        for c in 0..8 {
            let s: f32 = r[c * 8..(c + 1) * 8].iter().sum();
            assert!((s - 1.0).abs() < 1e-5);
        }
        assert!(direction(&r, 8) > 0.9, "a ramp points up");
        let flat = chart_raster(&[1.0; 5], 6).unwrap();
        assert!(direction(&flat, 6).abs() < 1e-9);
        assert!(chart_raster(&[], 6).is_none());
    }

    #[test]
    fn future_column_quantiles_recover_the_return() {
        let scale = 0.01;
        let cum = [0.001, 0.002, 0.004];
        let r = future_raster(&cum, scale, 16).unwrap();
        let last = &r[15 * 16..];
        let q = column_quantiles(last, &[0.05, 0.5, 0.95], scale);
        assert!((q[1] - 0.004).abs() < 2.0 * scale / 15.0, "{q:?}");
        assert!(q[0] < q[1] && q[1] < q[2]);
        assert!(prob_up(last) > 0.9);
        assert!((row_value(row_of(0.003, scale, 16), scale, 16) - 0.003).abs() < 1e-12);
        let uniform = vec![1.0f32; 9];
        assert!((prob_up(&uniform) - 0.5).abs() < 1e-9);
        let q = column_quantiles(&uniform, &[0.5], 1.0);
        assert!(q[0].abs() < 1e-9);
    }
}
