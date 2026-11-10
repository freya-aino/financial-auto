//! Price-derived fallback observation for the live runtime.
//!
//! Spec 35 strategies read graph ops, decoder outputs and portfolio state. Until
//! `model-runtime` publishes regions, transitions and decoder outputs, the runtime has only
//! `md.trade.>` and (optionally) `latent.state.>`. [`FallbackContext`] maps the GP primitive
//! surface onto **empirical price statistics** so programs evaluate without rewriting them:
//!
//! | primitive | fallback (clearly a placeholder, not the spec'd semantics) |
//! |---|---|
//! | `analog_forecast` | mean log return per bar over the lookback |
//! | `forecast_quantile(q)` | normal-approx quantile of the lookback return distribution |
//! | `path_up_prob(k)` | fraction of k-bar windows in the lookback with a positive return |
//! | `stay_prob` | sign persistence of consecutive bar returns (or latent region dwell) |
//! | `region_entropy` | latent posterior std if a `LatentState` is known, else `std / (|mean| + std)` |
//! | `current_region` | region id of the latest `LatentState` expert 0, if any |
//! | `transition_prob`, `decoder_score` | 0 (no transition/decoder feed yet) |
//!
//! Portfolio primitives read the strategy's **virtual book**: its own last emitted exposure.
use std::collections::VecDeque;

use fa_core::Timescale;

use crate::StrategyContext;

/// Bounded trade-price history for one instrument.
#[derive(Debug, Clone, Default)]
pub struct PriceSeries {
    /// (event time ms, price), ascending.
    points: VecDeque<(i64, f64)>,
    max_points: usize,
}

impl PriceSeries {
    pub fn new(max_points: usize) -> Self {
        Self { points: VecDeque::new(), max_points: max_points.max(2) }
    }

    pub fn push(&mut self, at_ms: i64, price: f64) {
        if price <= 0.0 || !price.is_finite() {
            return;
        }
        // keep ascending order; late ticks are dropped
        if self.points.back().is_some_and(|(t, _)| at_ms < *t) {
            return;
        }
        self.points.push_back((at_ms, price));
        while self.points.len() > self.max_points {
            self.points.pop_front();
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn last(&self) -> Option<f64> {
        self.points.back().map(|p| p.1)
    }

    /// Last `n` bar closes at `timescale`, resampled from trades (last price per bucket,
    /// gaps carried forward). Fewer than `n` when history is short.
    pub fn closes(&self, timescale: Timescale, n: usize) -> Vec<f64> {
        let bucket_ms = timescale.duration().as_millis() as i64;
        let mut out: Vec<f64> = Vec::new();
        let mut cur_bucket: Option<i64> = None;
        for (t, p) in &self.points {
            let b = t.div_euclid(bucket_ms);
            match cur_bucket {
                None => out.push(*p),
                Some(cb) if b == cb => *out.last_mut().expect("non-empty") = *p,
                Some(cb) => {
                    let last = *out.last().expect("non-empty");
                    for _ in cb + 1..b {
                        out.push(last);
                    }
                    out.push(*p);
                }
            }
            cur_bucket = Some(b);
        }
        if out.len() > n {
            out.drain(..out.len() - n);
        }
        out
    }
}

/// Summary of the latest `LatentState` for an instrument (expert 0 of the strategy's timescale).
#[derive(Debug, Clone, PartialEq)]
pub struct LatentInfo {
    pub region_id: String,
    /// Mean posterior std of the latent (exp(log_var / 2)).
    pub uncertainty: f64,
    /// Consecutive states observed in the same region.
    pub dwell: u32,
}

/// A strategy's own virtual position for portfolio primitives.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VirtualBook {
    /// Last emitted exposure mean in [-1, 1].
    pub exposure: f64,
    /// Return of the instrument since the exposure's sign was last set, times exposure.
    pub unrealised: f64,
    /// Steps since the exposure's sign was last set.
    pub steps_in_trade: u32,
}

/// Price-only observation for one instrument at one step (see module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct FallbackContext {
    /// Log returns between consecutive bar closes, oldest first.
    returns: Vec<f64>,
    mean: f64,
    std: f64,
    latent: Option<LatentInfo>,
    book: VirtualBook,
}

impl FallbackContext {
    pub fn new(closes: &[f64], latent: Option<LatentInfo>, book: VirtualBook) -> Self {
        let returns: Vec<f64> =
            closes.windows(2).filter(|w| w[0] > 0.0 && w[1] > 0.0).map(|w| (w[1] / w[0]).ln()).collect();
        let n = returns.len() as f64;
        let mean = if n > 0.0 { returns.iter().sum::<f64>() / n } else { 0.0 };
        let std =
            if n > 1.0 { (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt() } else { 0.0 };
        Self { returns, mean, std, latent, book }
    }

    pub fn n_returns(&self) -> usize {
        self.returns.len()
    }

    pub fn mean_return(&self) -> f64 {
        self.mean
    }

    pub fn std_return(&self) -> f64 {
        self.std
    }
}

/// Acklam's rational approximation of the standard normal quantile (|err| < 1.2e-9).
pub fn norm_quantile(p: f64) -> f64 {
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
    let (lo, hi) = (0.02425, 1.0 - 0.02425);
    if p < lo {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= hi {
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

impl StrategyContext for FallbackContext {
    fn current_region(&self) -> Option<&str> {
        self.latent.as_ref().map(|l| l.region_id.as_str())
    }

    fn transition_prob(&self, _to_region: &str) -> f64 {
        0.0
    }

    fn stay_prob(&self) -> f64 {
        if let Some(l) = &self.latent {
            return f64::from(l.dwell) / (f64::from(l.dwell) + 1.0);
        }
        if self.returns.len() < 2 {
            return 0.5;
        }
        let same = self.returns.windows(2).filter(|w| w[0].signum() == w[1].signum()).count();
        same as f64 / (self.returns.len() - 1) as f64
    }

    fn path_up_prob(&self, k: u32) -> f64 {
        let k = k.max(1) as usize;
        if self.returns.len() < k {
            return 0.5;
        }
        let windows = self.returns.windows(k);
        let n = windows.len();
        let up = windows.filter(|w| w.iter().sum::<f64>() > 0.0).count();
        up as f64 / n as f64
    }

    fn region_entropy(&self) -> f64 {
        if let Some(l) = &self.latent {
            return l.uncertainty.clamp(0.0, 1.0);
        }
        let d = self.mean.abs() + self.std;
        if d > 0.0 { self.std / d } else { 1.0 }
    }

    fn forecast_quantile(&self, level: f64) -> f64 {
        self.mean + norm_quantile(level) * self.std
    }

    fn analog_forecast(&self) -> f64 {
        self.mean
    }

    fn decoder_score(&self, _family: &str) -> f64 {
        0.0
    }

    fn position(&self) -> f64 {
        self.book.exposure
    }

    fn unrealised_pnl(&self) -> f64 {
        self.book.unrealised
    }

    fn time_in_trade(&self) -> f64 {
        f64::from(self.book.steps_in_trade)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resamples_into_bars_with_carry_forward() {
        let mut s = PriceSeries::new(100);
        s.push(0, 100.0);
        s.push(500, 101.0); // same 1s bucket → close 101
        s.push(3_000, 103.0); // buckets 1 and 2 empty → carried forward
        assert_eq!(s.closes(Timescale::Second, 10), vec![101.0, 101.0, 101.0, 103.0]);
        assert_eq!(s.closes(Timescale::Second, 2), vec![101.0, 103.0]);
        assert_eq!(s.closes(Timescale::Minute, 10), vec![103.0]);
        s.push(2_000, 50.0); // late tick dropped
        assert_eq!(s.last(), Some(103.0));
    }

    #[test]
    fn fallback_statistics() {
        let closes = [100.0, 101.0, 102.0, 101.0, 103.0];
        let c = FallbackContext::new(&closes, None, VirtualBook::default());
        assert_eq!(c.n_returns(), 4);
        assert!(c.analog_forecast() > 0.0);
        assert_eq!(c.path_up_prob(1), 0.75);
        assert!((c.forecast_quantile(0.5) - c.mean_return()).abs() < 1e-9);
        assert!(c.forecast_quantile(0.9) > c.forecast_quantile(0.1));
        assert!(c.region_entropy() > 0.0 && c.region_entropy() <= 1.0);
        assert_eq!(c.stay_prob(), 1.0 / 3.0);
        assert_eq!(c.current_region(), None);

        let empty = FallbackContext::new(&[100.0], None, VirtualBook::default());
        assert_eq!(empty.analog_forecast(), 0.0);
        assert_eq!(empty.path_up_prob(3), 0.5);
        assert_eq!(empty.region_entropy(), 1.0);

        let l = LatentInfo { region_id: "r7".into(), uncertainty: 0.3, dwell: 3 };
        let with =
            FallbackContext::new(&closes, Some(l), VirtualBook { exposure: -0.4, unrealised: 0.01, steps_in_trade: 2 });
        assert_eq!(with.current_region(), Some("r7"));
        assert_eq!(with.stay_prob(), 0.75);
        assert_eq!(with.region_entropy(), 0.3);
        assert_eq!(with.position(), -0.4);
        assert_eq!(with.time_in_trade(), 2.0);
    }

    #[test]
    fn normal_quantile_reference_values() {
        assert!((norm_quantile(0.5)).abs() < 1e-9);
        assert!((norm_quantile(0.975) - 1.959964).abs() < 1e-5);
        assert!((norm_quantile(0.01) + 2.326348).abs() < 1e-5);
    }
}
