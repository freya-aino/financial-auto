//! Baseline visual decoder (spec 34 §6.1 *Visual*, provisional): deterministic geometric chart
//! features computed from a bar window stand in for the chart / future-chart decoders until a
//! rendered-chart model exists. Families `visual.chart` (what the chart looks like now) and
//! `visual.future` (which way the chart geometry points), both scores in `[-1, 1]`.
//!
//! Formulas (window of `n ≥ MIN_BARS` bars, oldest first; the last bar is "now"):
//! - **support / resistance bands** over the bars *before* the last one: `R = max high`,
//!   `S = min low`, `W = R − S`. `breakout = tanh(4 · (close − R) / W)` when the close is above
//!   `R`, `tanh(4 · (close − S) / W)` (negative) when below `S`, `0` inside the band.
//! - **trend-line slope significance**: OLS of `ln close` on the bar index; `t = slope / se(slope)`;
//!   `trend = tanh(t / 3)`.
//! - **range compression** in `[0, 1]`: the range of the last `k = max(3, n / 5)` bars against
//!   the whole window's range, corrected for the shorter span (a random walk's range grows like
//!   `√k`): `compression = 1 − min(1, (range_k / range_n) · √(n / k))`. High when the recent bars
//!   are squeezed inside the window, `0` when the recent bars span the window or nothing moved.
//! - **future** (the `visual.future` directional score): `clamp(0.5 · breakout + 0.35 · trend +
//!   0.15 · sign(trend) · compression, −1, 1)`: a breakout points the way it broke, a trend
//!   continues, and a compression resolves in the trend's direction.
//!
//! Degenerate windows (fewer than `MIN_BARS` bars, zero range) score `0` everywhere.

use serde::{Deserialize, Serialize};

/// Decoder id published with every output of this module.
pub const DECODER_ID: &str = "visual-stat-v1";
/// Minimum bars for any non-zero score.
pub const MIN_BARS: usize = 8;

/// One OHLCV bar as the chart decoder sees it (same field meaning as `fa_encoders::MarketBar`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChartBar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl ChartBar {
    pub fn new(open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Self { open, high, low, close, volume }
    }
}

/// Geometric chart scores of one window.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct ChartScores {
    /// `[-1, 1]`: positive above resistance, negative below support, 0 inside the band.
    pub breakout: f64,
    /// `[-1, 1]`: squashed t-statistic of the log-price trend line.
    pub trend: f64,
    /// `[0, 1]`: recent range squeezed inside the window's range.
    pub compression: f64,
    /// `[-1, 1]`: directional combination (see module docs).
    pub future: f64,
    /// Resistance / support of the band (for inspection).
    pub resistance: f64,
    pub support: f64,
}

impl ChartScores {
    /// `visual.chart` quantiles: `[breakout, trend, compression]` (`quantiles[0]` is the score).
    pub fn chart_quantiles(&self) -> Vec<f64> {
        vec![self.breakout, self.trend, self.compression]
    }

    /// `visual.future` quantiles: `[future, breakout, trend, compression]`.
    pub fn future_quantiles(&self) -> Vec<f64> {
        vec![self.future, self.breakout, self.trend, self.compression]
    }
}

fn finite_or_zero(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

/// Scores of a bar window (oldest first).
pub fn chart_scores(bars: &[ChartBar]) -> ChartScores {
    let n = bars.len();
    if n < MIN_BARS || bars.iter().any(|b| !(b.close > 0.0 && b.high.is_finite() && b.low.is_finite())) {
        return ChartScores::default();
    }
    let last = bars[n - 1];
    let before = &bars[..n - 1];
    let resistance = before.iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max);
    let support = before.iter().map(|b| b.low).fold(f64::INFINITY, f64::min);
    let width = resistance - support;
    let breakout = if width > 0.0 {
        if last.close > resistance {
            (4.0 * (last.close - resistance) / width).tanh()
        } else if last.close < support {
            (4.0 * (last.close - support) / width).tanh()
        } else {
            0.0
        }
    } else {
        0.0
    };

    let trend = finite_or_zero((trend_t_stat(bars) / 3.0).tanh());

    let k = (n / 5).max(3).min(n);
    let range = |xs: &[ChartBar]| {
        let hi = xs.iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max);
        let lo = xs.iter().map(|b| b.low).fold(f64::INFINITY, f64::min);
        hi - lo
    };
    let range_n = range(bars);
    let range_k = range(&bars[n - k..]);
    let compression = if range_n > 0.0 {
        (1.0 - ((range_k / range_n) * (n as f64 / k as f64).sqrt()).min(1.0)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let future = (0.5 * breakout + 0.35 * trend + 0.15 * trend.signum() * compression).clamp(-1.0, 1.0);
    let future = if trend == 0.0 { (0.5 * breakout).clamp(-1.0, 1.0) } else { future };
    ChartScores { breakout, trend, compression, future, resistance, support }
}

/// t-statistic of the OLS slope of `ln close` against the bar index; 0 when undefined.
fn trend_t_stat(bars: &[ChartBar]) -> f64 {
    let n = bars.len();
    if n < 3 {
        return 0.0;
    }
    let ys: Vec<f64> = bars.iter().map(|b| b.close.ln()).collect();
    let xm = (n as f64 - 1.0) / 2.0;
    let ym = ys.iter().sum::<f64>() / n as f64;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (i, y) in ys.iter().enumerate() {
        let dx = i as f64 - xm;
        sxx += dx * dx;
        sxy += dx * (y - ym);
    }
    if sxx <= 0.0 {
        return 0.0;
    }
    let slope = sxy / sxx;
    let intercept = ym - slope * xm;
    let rss: f64 = ys.iter().enumerate().map(|(i, y)| (y - intercept - slope * i as f64).powi(2)).sum();
    let se = (rss / (n as f64 - 2.0) / sxx).sqrt();
    if se > 0.0 {
        slope / se
    } else if slope != 0.0 {
        // perfectly straight line: infinitely significant
        slope.signum() * 1e9
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(n: usize) -> Vec<ChartBar> {
        (0..n).map(|_| ChartBar::new(100.0, 100.5, 99.5, 100.0, 10.0)).collect()
    }

    /// A box between 99 and 101 for `n − 1` bars, then a last bar that closes at `last`.
    fn box_then(n: usize, last: f64) -> Vec<ChartBar> {
        let mut v: Vec<ChartBar> = (0..n - 1)
            .map(|i| {
                if i % 2 == 0 {
                    ChartBar::new(99.5, 101.0, 99.0, 100.5, 10.0)
                } else {
                    ChartBar::new(100.5, 101.0, 99.0, 99.5, 10.0)
                }
            })
            .collect();
        v.push(ChartBar::new(100.0, last.max(101.0), last.min(99.0), last, 20.0));
        v
    }

    #[test]
    fn flat_window_scores_zero() {
        let s = chart_scores(&flat(40));
        assert_eq!(s.breakout, 0.0);
        assert_eq!(s.trend, 0.0);
        assert_eq!(s.compression, 0.0);
        assert_eq!(s.future, 0.0);
        assert_eq!(chart_scores(&flat(MIN_BARS - 1)), ChartScores::default());
        assert_eq!(chart_scores(&[]), ChartScores::default());
    }

    #[test]
    fn breakout_up_and_down_are_mirrored_and_point_the_future() {
        let up = chart_scores(&box_then(40, 103.0));
        let down = chart_scores(&box_then(40, 97.0));
        assert_eq!((up.resistance, up.support), (101.0, 99.0));
        // 2 above a band of width 2 → tanh(4) ≈ 0.9993
        assert!(up.breakout > 0.99, "{up:?}");
        assert!(down.breakout < -0.99, "{down:?}");
        assert!(up.future > 0.45 && down.future < -0.45, "{up:?} {down:?}");
        assert!(up.future > 0.0 && down.future < 0.0);
        // inside the band: no breakout
        let inside = chart_scores(&box_then(40, 100.2));
        assert_eq!(inside.breakout, 0.0);
        assert!(inside.future.abs() < 0.2, "{inside:?}");
        // a marginal poke above resistance is a weak breakout
        let weak = chart_scores(&box_then(40, 101.2));
        assert!(weak.breakout > 0.0 && weak.breakout < up.breakout);
    }

    #[test]
    fn trend_follows_a_ramp_and_compression_detects_a_squeeze() {
        let ramp: Vec<ChartBar> = (0..40)
            .map(|i| {
                let p = 100.0 + i as f64;
                ChartBar::new(p - 0.2, p + 0.5, p - 0.5, p, 1.0)
            })
            .collect();
        let s = chart_scores(&ramp);
        assert!(s.trend > 0.99, "{s:?}");
        assert!(s.future > 0.3, "{s:?}");
        let fall: Vec<ChartBar> = ramp.iter().rev().copied().collect();
        let f = chart_scores(&fall);
        assert!(f.trend < -0.99 && f.future < -0.3, "{f:?}");
        // wide early range, then a tight squeeze near the middle: compression high, no breakout
        let mut squeeze: Vec<ChartBar> = (0..30)
            .map(|i| {
                if i % 2 == 0 {
                    ChartBar::new(100.0, 110.0, 90.0, 105.0, 1.0)
                } else {
                    ChartBar::new(105.0, 110.0, 90.0, 95.0, 1.0)
                }
            })
            .collect();
        squeeze.extend((0..10).map(|_| ChartBar::new(100.0, 100.3, 99.7, 100.0, 1.0)));
        let c = chart_scores(&squeeze);
        assert!(c.compression > 0.9, "{c:?}");
        assert_eq!(c.breakout, 0.0);
        assert_eq!(c.chart_quantiles().len(), 3);
        assert_eq!(c.future_quantiles()[0], c.future);
    }

    #[test]
    fn deterministic() {
        let b = box_then(60, 102.0);
        assert_eq!(chart_scores(&b), chart_scores(&b));
    }
}
