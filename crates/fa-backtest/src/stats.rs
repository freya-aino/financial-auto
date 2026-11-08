//! Deflated Sharpe ratio (Bailey & López de Prado 2014) and Probability of Backtest
//! Overfitting via Combinatorially Symmetric Cross-Validation (Bailey et al. 2017).

const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

pub fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Per-period Sharpe ratio (no annualisation), sample std.
pub fn sharpe(r: &[f64]) -> f64 {
    let n = r.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(r);
    let v = r.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64;
    if v == 0.0 { 0.0 } else { m / v.sqrt() }
}

fn moments(r: &[f64]) -> (f64, f64) {
    let n = r.len() as f64;
    let m = mean(r);
    let m2 = r.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
    let m3 = r.iter().map(|x| (x - m).powi(3)).sum::<f64>() / n;
    let m4 = r.iter().map(|x| (x - m).powi(4)).sum::<f64>() / n;
    if m2 == 0.0 { (0.0, 3.0) } else { (m3 / m2.powf(1.5), m4 / (m2 * m2)) }
}

/// Standard normal CDF (Abramowitz–Stegun 7.1.26 via erf).
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t + 0.254_829_592)
            * t
            * (-x * x).exp();
    s * y
}

/// Inverse standard normal CDF (Acklam's rational approximation).
pub fn norm_ppf(p: f64) -> f64 {
    let a = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    let b = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    let c = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    let d = [7.784_695_709_041_462e-3, 3.224_671_290_700_398e-1, 2.445_134_137_142_996, 3.754_408_661_907_416];
    let pl = 0.02425;
    if p < pl {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= 1.0 - pl {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        -norm_ppf(1.0 - p)
    }
}

/// Probabilistic Sharpe ratio: P(true SR > `sr_benchmark`) given the observed returns.
pub fn probabilistic_sharpe(r: &[f64], sr_benchmark: f64) -> f64 {
    let n = r.len() as f64;
    let sr = sharpe(r);
    let (skew, kurt) = moments(r);
    let denom = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr).max(1e-12).sqrt();
    norm_cdf((sr - sr_benchmark) * (n - 1.0).sqrt() / denom)
}

/// Expected maximum Sharpe among `n_trials` independent trials with Sharpe variance `var_sr`
/// under the null (true SR = 0).
pub fn expected_max_sharpe(n_trials: usize, var_sr: f64) -> f64 {
    let n = n_trials.max(2) as f64;
    var_sr.sqrt()
        * ((1.0 - EULER_GAMMA) * norm_ppf(1.0 - 1.0 / n)
            + EULER_GAMMA * norm_ppf(1.0 - 1.0 / (n * std::f64::consts::E)))
}

/// Deflated Sharpe ratio: PSR against the expected max Sharpe of `n_trials` trials.
pub fn deflated_sharpe(r: &[f64], n_trials: usize, var_trial_sharpes: f64) -> f64 {
    probabilistic_sharpe(r, expected_max_sharpe(n_trials, var_trial_sharpes))
}

/// PBO via CSCV. `perf[t][k]`: return of trial (strategy) `k` in period `t`.
/// Rows are split into `s` (even) blocks; for every half/half combination the
/// in-sample best trial's out-of-sample relative rank gives a logit; PBO is the share
/// of combinations where that logit ≤ 0.
pub fn pbo_cscv(perf: &[Vec<f64>], s: usize) -> f64 {
    assert!(s >= 2 && s.is_multiple_of(2), "s must be even");
    let t = perf.len();
    let k = perf.first().map_or(0, Vec::len);
    if t < s || k < 2 {
        return f64::NAN;
    }
    let block = t / s;
    let blocks: Vec<std::ops::Range<usize>> =
        (0..s).map(|i| i * block..if i == s - 1 { t } else { (i + 1) * block }).collect();
    let score = |rows: &[usize], trial: usize| -> f64 {
        let r: Vec<f64> = rows.iter().map(|&i| perf[i][trial]).collect();
        sharpe(&r)
    };
    let mut below = 0usize;
    let mut total = 0usize;
    for combo in combinations(s, s / 2) {
        let is_rows: Vec<usize> = combo.iter().flat_map(|&b| blocks[b].clone()).collect();
        let oos_rows: Vec<usize> = (0..s).filter(|b| !combo.contains(b)).flat_map(|b| blocks[b].clone()).collect();
        let is_scores: Vec<f64> = (0..k).map(|j| score(&is_rows, j)).collect();
        let best = (0..k).max_by(|&a, &b| is_scores[a].total_cmp(&is_scores[b])).unwrap_or(0);
        let oos: Vec<f64> = (0..k).map(|j| score(&oos_rows, j)).collect();
        let rank = oos.iter().filter(|&&v| v < oos[best]).count() as f64 + 1.0;
        let w = rank / (k as f64 + 1.0);
        let logit = (w / (1.0 - w)).ln();
        if logit <= 0.0 {
            below += 1;
        }
        total += 1;
    }
    below as f64 / total as f64
}

fn combinations(n: usize, r: usize) -> Vec<Vec<usize>> {
    fn rec(start: usize, n: usize, r: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == r {
            out.push(cur.clone());
            return;
        }
        for i in start..n {
            cur.push(i);
            rec(i + 1, n, r, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    rec(0, n, r, &mut Vec::new(), &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use fa_core::rng::Rng;

    #[test]
    fn normal_functions_are_consistent() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((norm_cdf(1.96) - 0.975).abs() < 1e-3);
        for p in [0.01, 0.2, 0.5, 0.9, 0.999] {
            assert!((norm_cdf(norm_ppf(p)) - p).abs() < 1e-4, "p={p}");
        }
    }

    #[test]
    fn deflation_penalises_many_trials() {
        let mut rng = Rng::seed(9);
        let r: Vec<f64> = (0..500).map(|_| 0.05 + rng.normal()).collect();
        let psr = probabilistic_sharpe(&r, 0.0);
        let dsr_few = deflated_sharpe(&r, 2, 0.002);
        let dsr_many = deflated_sharpe(&r, 10_000, 0.002);
        assert!(psr >= dsr_few && dsr_few > dsr_many, "{psr} {dsr_few} {dsr_many}");
    }

    #[test]
    fn pbo_high_for_noise_low_for_real_edge() {
        // Pure noise: PBO ≈ 0.5 in expectation; average over seeds to reduce variance.
        let pbo_noise = (0..6)
            .map(|seed| {
                let mut rng = Rng::seed(100 + seed);
                let noise: Vec<Vec<f64>> = (0..240).map(|_| (0..20).map(|_| rng.normal()).collect()).collect();
                pbo_cscv(&noise, 8)
            })
            .sum::<f64>()
            / 6.0;
        assert!(pbo_noise > 0.3 && pbo_noise < 0.7, "noise pbo {pbo_noise}");
        let mut rng = Rng::seed(21);
        let edge: Vec<Vec<f64>> =
            (0..240).map(|_| (0..20).map(|k| rng.normal() + if k == 0 { 0.8 } else { 0.0 }).collect()).collect();
        let pbo_edge = pbo_cscv(&edge, 8);
        assert!(pbo_edge < 0.1, "edge pbo {pbo_edge}");
    }
}
