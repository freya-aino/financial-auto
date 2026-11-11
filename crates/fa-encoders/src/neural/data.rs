//! Market front-end of the neural families (spec 31 §4.2, modality `Market`): per-bar features,
//! next-step targets, the fitted input normalisation and the training/validation window sets.
//!
//! Per bar `t` (with `c₋` the previous close):
//! `[ln(c/c₋), ln(o/c₋), ln(h/l), (c−o)/(h−l), wick imbalance, ln(1+volume), ln(1+trades)]`.
//! Next-step targets of the window ending before bar `t` (spec 31 §6.5 `L_next_step`, spec 32
//! §6.3 residuals): `[ln(c/c₋), |ln(c/c₋)|, ln(h/l)]` of bar `t` — return, volatility, range.
//! Everything is z-scored with [`InputNorm`] fitted on the training windows and stored in the
//! weights, so a snapshot reproduces its encoder from the artifact alone.

use serde::{Deserialize, Serialize};

use crate::MarketBar;

pub const N_FEATURES: usize = 7;
pub const N_TARGETS: usize = 3;
pub const FEATURE_NAMES: [&str; N_FEATURES] = ["ret", "gap", "range", "body", "wick", "log_volume", "log_trades"];
pub const TARGET_NAMES: [&str; N_TARGETS] = ["next_ret", "next_abs_ret", "next_range"];
/// z-scores are clipped to ±this.
const CLIP: f32 = 6.0;

fn ln_ratio(a: f64, b: f64) -> f64 {
    if a > 0.0 && b > 0.0 { (a / b).ln() } else { 0.0 }
}

/// Raw features of every bar (oldest first).
pub fn bar_features(bars: &[MarketBar]) -> Vec<[f32; N_FEATURES]> {
    let mut prev = None::<f64>;
    bars.iter()
        .map(|b| {
            let pc = prev.unwrap_or(b.open);
            prev = Some(b.close);
            let hl = b.high - b.low;
            let (body, wick) = if hl > 0.0 {
                ((b.close - b.open) / hl, ((b.high - b.open.max(b.close)) - (b.open.min(b.close) - b.low)) / hl)
            } else {
                (0.0, 0.0)
            };
            [
                ln_ratio(b.close, pc) as f32,
                ln_ratio(b.open, pc) as f32,
                ln_ratio(b.high, b.low) as f32,
                body as f32,
                wick as f32,
                b.volume.max(0.0).ln_1p() as f32,
                (b.trade_count as f64).ln_1p() as f32,
            ]
        })
        .collect()
}

/// Raw next-step targets of the window `bars[..end]`, read from the features of bar `end`.
pub fn targets_at(feats: &[[f32; N_FEATURES]], end: usize) -> Option<[f32; N_TARGETS]> {
    let f = feats.get(end)?;
    Some([f[0], f[0].abs(), f[2]])
}

/// Fitted z-score statistics of features and targets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputNorm {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub target_mean: Vec<f32>,
    pub target_std: Vec<f32>,
    pub n: u64,
}

impl Default for InputNorm {
    fn default() -> Self {
        Self {
            mean: vec![0.0; N_FEATURES],
            std: vec![1.0; N_FEATURES],
            target_mean: vec![0.0; N_TARGETS],
            target_std: vec![1.0; N_TARGETS],
            n: 0,
        }
    }
}

fn welford(rows: impl Iterator<Item = Vec<f32>>, d: usize) -> (Vec<f32>, Vec<f32>, u64) {
    let (mut n, mut mean, mut m2) = (0u64, vec![0f64; d], vec![0f64; d]);
    for r in rows {
        n += 1;
        for (j, x) in r.iter().enumerate().take(d) {
            let x = f64::from(*x);
            let delta = x - mean[j];
            mean[j] += delta / n as f64;
            m2[j] += delta * (x - mean[j]);
        }
    }
    let std = m2.iter().map(|v| ((v / n.max(1) as f64).sqrt().max(1e-6)) as f32).collect();
    (mean.iter().map(|m| *m as f32).collect(), std, n)
}

impl InputNorm {
    /// Fits on the bars of every training series (features) and the training targets.
    pub fn fit(ds: &Dataset) -> Self {
        let rows = ds.series.iter().flat_map(|s| s.feats.iter().map(|f| f.to_vec()));
        let (mean, std, n) = welford(rows, N_FEATURES);
        let trows = ds.train.iter().filter_map(|&(s, e)| targets_at(&ds.series[s].feats, e).map(|t| t.to_vec()));
        let (target_mean, target_std, _) = welford(trows, N_TARGETS);
        Self { mean, std, target_mean, target_std, n }
    }

    pub fn feature(&self, j: usize, x: f32) -> f32 {
        ((x - self.mean[j]) / self.std[j]).clamp(-CLIP, CLIP)
    }

    pub fn target(&self, j: usize, x: f32) -> f32 {
        ((x - self.target_mean[j]) / self.target_std[j]).clamp(-CLIP, CLIP)
    }

    /// Normalised window `feats[..end]` as `[context][N_FEATURES]` row-major, left-padded with
    /// zeros (= the training mean) when shorter than `context`.
    pub fn window(&self, feats: &[[f32; N_FEATURES]], end: usize, context: usize) -> Vec<f32> {
        let end = end.min(feats.len());
        let start = end.saturating_sub(context);
        let pad = context - (end - start);
        let mut out = vec![0f32; pad * N_FEATURES];
        out.reserve(context * N_FEATURES);
        for f in &feats[start..end] {
            for (j, x) in f.iter().enumerate() {
                out.push(self.feature(j, *x));
            }
        }
        out
    }
}

/// One instrument's bars with their features.
#[derive(Debug, Clone)]
pub struct Series {
    pub instrument: String,
    pub bars: Vec<MarketBar>,
    pub feats: Vec<[f32; N_FEATURES]>,
    /// End time (ms) of every bar, ascending; empty when unknown (no cross-asset context).
    pub ends_ms: Vec<i64>,
}

impl Series {
    pub fn new(instrument: impl Into<String>, bars: Vec<MarketBar>) -> Self {
        let feats = bar_features(&bars);
        Self { instrument: instrument.into(), bars, feats, ends_ms: Vec::new() }
    }

    /// A series with bar end times (needed by the cross-asset context, spec 31 §6.4).
    pub fn with_times(instrument: impl Into<String>, bars: Vec<MarketBar>, ends_ms: Vec<i64>) -> Self {
        let mut s = Self::new(instrument, bars);
        if ends_ms.len() == s.bars.len() {
            s.ends_ms = ends_ms;
        }
        s
    }
}

/// Windows of a fit: `(series, end)` where the window is `bars[..end]` and its next-step target
/// is bar `end`. The last `val_fraction` of every series' windows (after an embargo of
/// `horizon` windows) is the validation set (spec 36 §6.1: train | validation, in time order).
#[derive(Debug, Clone)]
pub struct Dataset {
    pub series: Vec<Series>,
    pub train: Vec<(usize, usize)>,
    pub val: Vec<(usize, usize)>,
}

impl Dataset {
    /// `min_window` bars at least per window; `horizon` = steps the JEPA target looks ahead
    /// (windows need `end + horizon ≤ n`).
    pub fn build(series: Vec<Series>, min_window: usize, horizon: usize, val_fraction: f64) -> Self {
        let mut train = Vec::new();
        let mut val = Vec::new();
        let h = horizon.max(1);
        for (s, ser) in series.iter().enumerate() {
            let n = ser.bars.len();
            if n < min_window.max(2) + h {
                continue;
            }
            let ends: Vec<usize> = (min_window.max(2)..=n - h).collect();
            let n_val = ((ends.len() as f64) * val_fraction.clamp(0.0, 0.9)).round() as usize;
            let cut = ends.len() - n_val;
            let embargo = cut.saturating_sub(h);
            train.extend(ends[..embargo].iter().map(|e| (s, *e)));
            val.extend(ends[cut..].iter().map(|e| (s, *e)));
        }
        Self { series, train, val }
    }

    pub fn n_bars(&self) -> usize {
        self.series.iter().map(|s| s.bars.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bars(n: usize) -> Vec<MarketBar> {
        (0..n)
            .map(|i| {
                let p = 100.0 + (i as f64).sin();
                MarketBar { open: p, high: p + 0.5, low: p - 0.5, close: p + 0.1, volume: 10.0, trade_count: 3 }
            })
            .collect()
    }

    #[test]
    fn features_targets_windows_and_split() {
        let s = Series::new("a", bars(100));
        assert_eq!(s.feats.len(), 100);
        assert!(s.feats.iter().flatten().all(|x| x.is_finite()));
        let t = targets_at(&s.feats, 10).unwrap();
        assert_eq!(t[1], t[0].abs());
        assert!(targets_at(&s.feats, 100).is_none(), "no target past the last bar");
        let ds = Dataset::build(vec![s, Series::new("short", bars(5))], 8, 4, 0.2);
        assert!(ds.train.iter().chain(&ds.val).all(|(s, e)| *s == 0 && *e + 4 <= 100));
        let last_train = ds.train.last().unwrap().1;
        let first_val = ds.val.first().unwrap().1;
        assert!(first_val >= last_train + 4, "embargo between train and validation");
        let norm = InputNorm::fit(&ds);
        assert_eq!(norm.mean.len(), N_FEATURES);
        let w = norm.window(&ds.series[0].feats, 5, 8);
        assert_eq!(w.len(), 8 * N_FEATURES);
        assert!(w[..3 * N_FEATURES].iter().all(|x| *x == 0.0), "left padding");
    }
}
