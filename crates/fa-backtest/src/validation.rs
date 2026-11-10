//! Per-generation multiple-testing statistics over the trial registry (spec 41 §6.5).
//!
//! Every evaluation (decoder or strategy) is a [`Trial`] with its selection-window Sharpe, the
//! number of observations and its per-step returns compressed into [`N_BLOCKS`] block sums
//! (`validation.trial.returns`). [`generation_stats`] then gives, for the trial set of one
//! generation:
//!
//! - the **deflated Sharpe ratio** (Bailey & López de Prado) of the best trial: its PSR against
//!   the expected maximum Sharpe of `n_trials` trials with the observed variance of trial
//!   Sharpes, with skew/kurtosis from its block returns;
//! - the **PBO** via CSCV ([`crate::stats::pbo_cscv`]) over the trials' block returns.
//!
//! Block sums keep the registry small and make every trial the same length, which CSCV needs;
//! Sharpe is scale-free per block, so the PBO ranking is unaffected.

use serde::{Deserialize, Serialize};

use crate::stats;

/// Blocks a trial's returns are compressed into for the registry.
pub const N_BLOCKS: usize = 64;
/// CSCV partitions (spec 41 §7 `pbo.n_blocks`).
pub const PBO_BLOCKS: usize = 16;

/// One registered evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trial {
    pub candidate: String,
    /// Per-step Sharpe on the selection window.
    pub sharpe: f64,
    pub n_obs: usize,
    /// Block sums of the per-step returns ([`block_sums`]); empty when unknown.
    pub returns: Vec<f64>,
}

/// Statistics of one generation's trial set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationStats {
    pub n_trials: usize,
    pub sharpe_mean: f64,
    pub sharpe_var: f64,
    pub best_candidate: Option<String>,
    pub best_sharpe: f64,
    /// Expected max Sharpe of `n_trials` noise trials with `sharpe_var`.
    pub expected_max_sharpe: f64,
    /// Deflated Sharpe (probability) of the best trial; NaN with fewer than 2 observations.
    pub deflated_sharpe: f64,
    /// PBO over trials with `≥ pbo_blocks` blocks; NaN when fewer than 2 such trials.
    pub pbo: f64,
    pub pbo_blocks: usize,
    pub pbo_trials: usize,
}

/// Compresses `returns` into exactly `n_blocks` consecutive block sums (empty blocks = 0).
pub fn block_sums(returns: &[f64], n_blocks: usize) -> Vec<f64> {
    let n_blocks = n_blocks.max(1);
    let mut out = vec![0.0; n_blocks];
    if returns.is_empty() {
        return out;
    }
    for (i, r) in returns.iter().enumerate() {
        let b = i * n_blocks / returns.len();
        out[b.min(n_blocks - 1)] += r;
    }
    out
}

fn sample_var(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let m = stats::mean(x);
    x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64
}

fn skew_kurt(r: &[f64]) -> (f64, f64) {
    if r.len() < 3 {
        return (0.0, 3.0);
    }
    let n = r.len() as f64;
    let m = stats::mean(r);
    let m2 = r.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
    let m3 = r.iter().map(|x| (x - m).powi(3)).sum::<f64>() / n;
    let m4 = r.iter().map(|x| (x - m).powi(4)).sum::<f64>() / n;
    if m2 == 0.0 { (0.0, 3.0) } else { (m3 / m2.powf(1.5), m4 / (m2 * m2)) }
}

/// Deflated Sharpe from summary statistics: PSR of `sharpe` over `n_obs` observations (with
/// `skew`, `kurt` of the returns) against `sr_benchmark`.
pub fn deflated_sharpe_summary(sharpe: f64, n_obs: usize, skew: f64, kurt: f64, sr_benchmark: f64) -> f64 {
    if n_obs < 2 || !sharpe.is_finite() {
        return f64::NAN;
    }
    let denom = (1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe * sharpe).max(1e-12).sqrt();
    stats::norm_cdf((sharpe - sr_benchmark) * ((n_obs - 1) as f64).sqrt() / denom)
}

/// Statistics over `trials` (see module docs). `pbo_blocks` must be even and ≥ 2.
pub fn generation_stats(trials: &[Trial], pbo_blocks: usize) -> GenerationStats {
    let finite: Vec<&Trial> = trials.iter().filter(|t| t.sharpe.is_finite()).collect();
    let sharpes: Vec<f64> = finite.iter().map(|t| t.sharpe).collect();
    let n_trials = trials.len();
    let sharpe_mean = if sharpes.is_empty() { 0.0 } else { stats::mean(&sharpes) };
    let sharpe_var = sample_var(&sharpes);
    let best = finite.iter().copied().max_by(|a, b| a.sharpe.total_cmp(&b.sharpe));
    let expected_max = if n_trials >= 2 { stats::expected_max_sharpe(n_trials, sharpe_var) } else { 0.0 };
    let deflated = best.map_or(f64::NAN, |b| {
        let (skew, kurt) = skew_kurt(&b.returns);
        deflated_sharpe_summary(b.sharpe, b.n_obs, skew, kurt, expected_max)
    });
    let pbo_blocks = pbo_blocks.max(2) & !1;
    let eligible: Vec<&Trial> = finite.iter().copied().filter(|t| t.returns.len() >= pbo_blocks).collect();
    let pbo = if eligible.len() >= 2 {
        let t = eligible.iter().map(|t| t.returns.len()).min().unwrap_or(0);
        let perf: Vec<Vec<f64>> = (0..t).map(|row| eligible.iter().map(|tr| tr.returns[row]).collect()).collect();
        stats::pbo_cscv(&perf, pbo_blocks)
    } else {
        f64::NAN
    };
    GenerationStats {
        n_trials,
        sharpe_mean,
        sharpe_var,
        best_candidate: best.map(|b| b.candidate.clone()),
        best_sharpe: best.map_or(f64::NAN, |b| b.sharpe),
        expected_max_sharpe: expected_max,
        deflated_sharpe: deflated,
        pbo,
        pbo_blocks,
        pbo_trials: eligible.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fa_core::rng::Rng;

    fn trial(name: &str, seed: u64, edge: f64) -> Trial {
        let mut rng = Rng::seed(seed);
        let r: Vec<f64> = (0..640).map(|_| edge + 0.01 * rng.normal()).collect();
        Trial { candidate: name.into(), sharpe: stats::sharpe(&r), n_obs: r.len(), returns: block_sums(&r, N_BLOCKS) }
    }

    #[test]
    fn block_sums_preserve_total_and_length() {
        let r: Vec<f64> = (1..=10).map(f64::from).collect();
        let b = block_sums(&r, 4);
        assert_eq!(b.len(), 4);
        assert!((b.iter().sum::<f64>() - 55.0).abs() < 1e-12);
        assert_eq!(block_sums(&[1.0, 2.0], 4), vec![1.0, 0.0, 2.0, 0.0]);
        assert_eq!(block_sums(&[], 3), vec![0.0; 3]);
    }

    #[test]
    fn noise_generation_is_deflated_and_overfit() {
        let noise: Vec<Trial> = (0..30).map(|i| trial(&format!("n{i}"), 100 + i as u64, 0.0)).collect();
        let s = generation_stats(&noise, PBO_BLOCKS);
        assert_eq!((s.n_trials, s.pbo_trials, s.pbo_blocks), (30, 30, 16));
        assert!(s.best_sharpe > 0.0 && s.expected_max_sharpe > 0.0);
        // the best of 30 noise trials is not significantly better than the expected max
        assert!(s.deflated_sharpe < 0.9, "{s:?}");
        // one CSCV sample of 16 blocks is noisy (≈ 0.5 in expectation); it must stay well above
        // what a real edge gives below
        assert!(s.pbo > 0.1, "noise should look overfit: {s:?}");
        let noise_pbo = s.pbo;

        let mut real = noise.clone();
        real.push(trial("edge", 7, 0.01));
        let s = generation_stats(&real, PBO_BLOCKS);
        assert_eq!(s.best_candidate.as_deref(), Some("edge"));
        assert!(s.deflated_sharpe > 0.99, "{s:?}");
        assert!(s.pbo < 0.1 && s.pbo < noise_pbo, "{s:?}");
    }

    #[test]
    fn degenerate_sets_do_not_panic() {
        let s = generation_stats(&[], PBO_BLOCKS);
        assert_eq!(s.n_trials, 0);
        assert!(s.deflated_sharpe.is_nan() && s.pbo.is_nan() && s.best_candidate.is_none());
        let one = [Trial { candidate: "a".into(), sharpe: 0.3, n_obs: 50, returns: vec![] }];
        let s = generation_stats(&one, PBO_BLOCKS);
        assert_eq!(s.n_trials, 1);
        assert!(s.deflated_sharpe > 0.5 && s.pbo.is_nan() && s.pbo_trials == 0);
        let short = [
            Trial { candidate: "a".into(), sharpe: 0.3, n_obs: 5, returns: vec![0.1; 4] },
            Trial { candidate: "b".into(), sharpe: f64::NAN, n_obs: 0, returns: vec![] },
        ];
        let s = generation_stats(&short, PBO_BLOCKS);
        assert_eq!((s.n_trials, s.pbo_trials), (2, 0));
        assert_eq!(s.best_candidate.as_deref(), Some("a"));
    }
}
