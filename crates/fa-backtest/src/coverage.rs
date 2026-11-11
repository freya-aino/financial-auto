//! Missing-data coverage of a replay window (spec 41 §8 "missing data in window: report marks
//! coverage; candidates with < min coverage get no fitness").
//!
//! Bars are resampled from trades, so an empty bar is not missing data by itself (a quiet
//! second has no trade). Missing data is a **gap**: a stretch between consecutive bars (or
//! between the reference interval's ends and the first / last bar) longer than the instrument's
//! tolerance `max(gap_bars × bar, min_gap)`, or `session_gap` for classes that trade in exchange
//! sessions (`equity`, `etf`, `index`: nights and weekends are closed markets, not holes). The
//! part of a gap beyond the tolerance counts as uncovered.
//!
//! * **Reference interval:** the replay window clipped to the data span of all selected series
//!   (earliest first bar … latest last bar), so an instrument that starts late or stops early
//!   against the others is marked; a window whose data covers only part of it is caught by the
//!   window coverage instead.
//! * **Instrument coverage** = 1 − uncovered / reference length.
//! * **Window coverage** = data span / window length for a bounded window (1 for an open one).
//! * **Cut:** an instrument below `min_coverage` is excluded from the replay (not traded, listed
//!   in the report); a window below `min_coverage`, or no instrument left, fails the replay with
//!   [`crate::ReplayError::Coverage`] — the candidate gets no fitness (evo-worker answers the
//!   job as failed and writes no trial row).

use serde::{Deserialize, Serialize};

use crate::replay::BarSeries;

/// Coverage thresholds (`ReplaySpec::coverage`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CoveragePolicy {
    /// Minimum coverage of an instrument (else excluded) and of the window (else no fitness).
    pub min_coverage: f64,
    /// Gap tolerance in bars of the replay's timescale.
    pub gap_bars: u32,
    /// Gap tolerance floor (ms): trade-resampled fast bars are sparse by nature.
    pub min_gap_ms: i64,
    /// Gap tolerance for session classes (ms): covers a weekend.
    pub session_gap_ms: i64,
}

impl Default for CoveragePolicy {
    fn default() -> Self {
        Self { min_coverage: 0.5, gap_bars: 5, min_gap_ms: 15 * 60_000, session_gap_ms: 72 * 3_600_000 }
    }
}

impl CoveragePolicy {
    /// No cut (every instrument kept, window never rejected); coverage still reported.
    pub fn off() -> Self {
        Self { min_coverage: 0.0, ..Self::default() }
    }

    /// Classes whose venues close outside exchange sessions.
    pub fn is_session_class(class: &str) -> bool {
        matches!(class, "equity" | "etf" | "index")
    }

    /// Gap tolerance (ms) of an instrument of `class` at bars of `step_ms`.
    pub fn tolerance_ms(&self, class: &str, step_ms: i64) -> i64 {
        let base = (i64::from(self.gap_bars) * step_ms.max(1)).max(self.min_gap_ms);
        if Self::is_session_class(class) { base.max(self.session_gap_ms) } else { base }
    }
}

/// Coverage of one instrument.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstrumentCoverage {
    pub instrument_id: String,
    pub bars: usize,
    /// `1 − uncovered / reference length`, in `[0, 1]`.
    pub coverage: f64,
    /// Longest stretch without a bar (ms), reference ends included.
    pub max_gap_ms: i64,
    pub tolerance_ms: i64,
    /// Below `min_coverage`: not traded.
    pub excluded: bool,
}

/// Coverage section of a replay report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CoverageReport {
    pub min_coverage: f64,
    /// Reference interval `[start, end]` (ms).
    pub reference_ms: (i64, i64),
    /// Data span / window length (1 for an open-ended window).
    pub window: f64,
    /// Mean instrument coverage (excluded instruments included).
    pub mean: f64,
    pub instruments: Vec<InstrumentCoverage>,
}

impl CoverageReport {
    pub fn excluded(&self) -> impl Iterator<Item = &str> {
        self.instruments.iter().filter(|c| c.excluded).map(|c| c.instrument_id.as_str())
    }
}

/// A window with ends this far apart is open-ended (tests use `[0, i64::MAX]`).
const OPEN_WINDOW_MS: i64 = 100 * 365 * 86_400_000;

/// Coverage of `series` in `window_ms` (bars of `step_ms`); `class_of` names each series' cost
/// class (session classes get the session tolerance).
pub fn measure(
    series: &[&BarSeries],
    window_ms: (i64, i64),
    step_ms: i64,
    class_of: impl Fn(&BarSeries) -> String,
    policy: &CoveragePolicy,
) -> CoverageReport {
    let (w0, w1) = window_ms;
    let in_window = |s: &BarSeries| -> Vec<i64> { s.ts_ms.iter().copied().filter(|t| *t >= w0 && *t <= w1).collect() };
    let stamps: Vec<Vec<i64>> = series.iter().map(|s| in_window(s)).collect();
    let first = stamps.iter().filter_map(|v| v.first()).min().copied();
    let last = stamps.iter().filter_map(|v| v.last()).max().copied();
    let open = w1.saturating_sub(w0) >= OPEN_WINDOW_MS;
    // bars are filtered to the window, so the window clipped to the data span is the data span
    let (r0, r1) = match (first, last) {
        (Some(f), Some(l)) => (f, l),
        _ => (w0, w1),
    };
    let ref_len = (r1 - r0).max(0);
    let window = match (first, last) {
        _ if open => 1.0,
        (Some(f), Some(l)) if w1 > w0 => ((l - f) as f64 / (w1 - w0) as f64).clamp(0.0, 1.0),
        (Some(_), Some(_)) => 1.0,
        _ => 0.0,
    };
    let instruments: Vec<InstrumentCoverage> = series
        .iter()
        .zip(&stamps)
        .map(|(s, ts)| {
            let tolerance_ms = policy.tolerance_ms(&class_of(s), step_ms);
            let (coverage, max_gap_ms) = if ts.is_empty() {
                (if ref_len == 0 { 1.0 } else { 0.0 }, ref_len)
            } else {
                let mut edges = Vec::with_capacity(ts.len() + 2);
                edges.push(r0);
                edges.extend(ts.iter().copied());
                edges.push(r1);
                let mut uncovered = 0i64;
                let mut max_gap = 0i64;
                for w in edges.windows(2) {
                    let gap = (w[1] - w[0]).max(0);
                    max_gap = max_gap.max(gap);
                    uncovered += (gap - tolerance_ms).max(0);
                }
                let c = if ref_len == 0 { 1.0 } else { 1.0 - uncovered as f64 / ref_len as f64 };
                (c.clamp(0.0, 1.0), max_gap)
            };
            InstrumentCoverage {
                instrument_id: s.instrument_id.clone(),
                bars: ts.len(),
                coverage,
                max_gap_ms,
                tolerance_ms,
                excluded: coverage < policy.min_coverage,
            }
        })
        .collect();
    let mean = if instruments.is_empty() {
        0.0
    } else {
        instruments.iter().map(|c| c.coverage).sum::<f64>() / instruments.len() as f64
    };
    CoverageReport { min_coverage: policy.min_coverage, reference_ms: (r0, r1), window, mean, instruments }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bars(id: &str, ts: impl IntoIterator<Item = i64>) -> BarSeries {
        let ts_ms: Vec<i64> = ts.into_iter().collect();
        BarSeries { instrument_id: id.into(), venue: "sim".into(), close: vec![1.0; ts_ms.len()], ts_ms }
    }

    const MIN: i64 = 60_000;

    #[test]
    fn full_series_is_covered_and_holes_are_marked() {
        let p = CoveragePolicy::default(); // tolerance 15 min at 1m bars
        let full = bars("a", (1..=600).map(|i| i * MIN));
        // b: a 5 h hole in the middle of a 10 h window
        let holed = bars("b", (1..=600).filter(|i| !(200..500).contains(i)).map(|i| i * MIN));
        let r = measure(&[&full, &holed], (0, 600 * MIN), MIN, |_| "crypto".into(), &p);
        assert!(r.instruments[0].coverage > 0.99 && !r.instruments[0].excluded, "{r:?}");
        let b = &r.instruments[1];
        // 300 min hole minus 15 min tolerance over a ~599 min reference
        assert!((b.coverage - (1.0 - 286.0 / 599.0)).abs() < 0.01, "{b:?}");
        assert!(!b.excluded);
        assert!((r.window - 599.0 / 600.0).abs() < 1e-9);
        // a 9 h hole excludes it
        let gone = bars("c", (1..=600).filter(|i| !(30..570).contains(i)).map(|i| i * MIN));
        let r = measure(&[&full, &gone], (0, 600 * MIN), MIN, |_| "crypto".into(), &p);
        assert!(r.instruments[1].excluded, "{r:?}");
        assert_eq!(r.excluded().collect::<Vec<_>>(), vec!["c"]);
    }

    #[test]
    fn sparse_fast_bars_and_closed_sessions_are_not_missing_data() {
        let p = CoveragePolicy::default();
        // 1 s bars from a trade every ~2 min: sparse, not missing
        let sparse = bars("k", (0..300).map(|i| i * 120_000 + 1000));
        let r = measure(&[&sparse], (0, 300 * 120_000), 1000, |_| "crypto".into(), &p);
        assert!(r.instruments[0].coverage > 0.99, "{r:?}");
        // equities: a 17.5 h overnight close is a session gap, not a hole
        let day = 86_400_000i64;
        let eq = bars("e", (0..5).flat_map(|d| (0..390).map(move |m| d * day + 13 * 3_600_000 + m * MIN)));
        let r = measure(&[&eq], (0, 5 * day), MIN, |_| "equity".into(), &p);
        assert!(r.instruments[0].coverage > 0.99, "{r:?}");
        let r = measure(&[&eq], (0, 5 * day), MIN, |_| "crypto".into(), &p);
        assert!(r.instruments[0].coverage < 0.5, "a 24/7 class with the same holes is not covered: {r:?}");
    }

    #[test]
    fn late_starting_instrument_and_short_window_data() {
        let p = CoveragePolicy::default();
        let full = bars("a", (1..=600).map(|i| i * MIN));
        let late = bars("b", (400..=600).map(|i| i * MIN));
        let r = measure(&[&full, &late], (0, 600 * MIN), MIN, |_| "crypto".into(), &p);
        assert!(r.instruments[1].excluded, "starts after two thirds of the data span: {r:?}");
        // data for only the last fifth of a bounded window
        let tail = bars("a", (480..=600).map(|i| i * MIN));
        let r = measure(&[&tail], (0, 600 * MIN), MIN, |_| "crypto".into(), &p);
        assert!(r.window < 0.25 && r.instruments[0].coverage > 0.99, "{r:?}");
        // open-ended windows measure the data span only
        let r = measure(&[&tail], (0, i64::MAX), MIN, |_| "crypto".into(), &p);
        assert_eq!(r.window, 1.0);
    }
}
