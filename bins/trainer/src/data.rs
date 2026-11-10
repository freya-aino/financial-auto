//! Training data: bars per timescale from VictoriaMetrics (`fa_trade_price`, D-049).
//!
//! `last_over_time(fa_trade_price{inst=…}[step])` at `step` resolution gives the last trade
//! price of every step bucket; buckets without a trade are absent. Bars are **price-only**
//! (`open` = previous close, `high`/`low` = max/min of open and close, `volume` = 1,
//! `trade_count` = 1): the statistical encoder's activity expert therefore sees a flat volume
//! profile until bar metrics are read too. The window start is aligned to the step so bar ends
//! coincide with model-runtime's bar closes (what `LatentVec.ts` holds), which the probe-set
//! alignment relies on.

use fa_core::Timescale;
use fa_encoders::MarketBar;
use fa_tsdb::{MAX_POINTS_PER_SERIES, Tsdb};
use serde::{Deserialize, Serialize};

/// Bars of one instrument, oldest first, with their bar-end timestamps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BarHistory {
    /// Instrument token as VictoriaMetrics labels it (dashless UUID).
    pub instrument: String,
    pub timescale: Timescale,
    pub start_ms: i64,
    pub end_ms: i64,
    pub ends_ms: Vec<i64>,
    pub bars: Vec<MarketBar>,
}

pub fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

/// `[start, end]` for `window_hours` ending at `end_ms`, aligned to the step and capped to
/// `MAX_POINTS_PER_SERIES` steps.
pub fn window(ts: Timescale, window_hours: f64, end_ms: i64) -> (i64, i64) {
    let step_ms = i64::try_from(ts.duration().as_millis()).unwrap_or(60_000).max(1);
    let end = end_ms - end_ms.rem_euclid(step_ms);
    let span = ((window_hours.max(0.0) * 3_600_000.0) as i64).min(MAX_POINTS_PER_SERIES * step_ms);
    let start = end - span;
    (start - start.rem_euclid(step_ms), end)
}

/// Price points → bars (oldest first).
pub fn bars_from_prices(points: &[(i64, f64)]) -> (Vec<i64>, Vec<MarketBar>) {
    let mut prev = None;
    let mut ends = Vec::with_capacity(points.len());
    let mut bars = Vec::with_capacity(points.len());
    for &(t, p) in points {
        let open = prev.unwrap_or(p);
        bars.push(MarketBar { open, high: open.max(p), low: open.min(p), close: p, volume: 1.0, trade_count: 1 });
        ends.push(t);
        prev = Some(p);
    }
    (ends, bars)
}

/// Loads the history of `instrument` (or of the instrument with the most points) over the window.
pub async fn load(
    tsdb: &Tsdb,
    ts: Timescale,
    instrument: Option<&str>,
    start_ms: i64,
    end_ms: i64,
) -> anyhow::Result<BarHistory> {
    let series = tsdb.trade_prices(instrument, start_ms / 1000, end_ms / 1000, fa_tsdb::step_of(ts)).await?;
    let Some(best) = series.into_iter().next() else {
        anyhow::bail!(
            "no fa_trade_price series{} between {start_ms} and {end_ms} at {ts}",
            instrument.map(|i| format!(" for {i}")).unwrap_or_default()
        );
    };
    let (ends_ms, bars) = bars_from_prices(&best.points);
    Ok(BarHistory { instrument: best.inst, timescale: ts, start_ms, end_ms, ends_ms, bars })
}

/// Dashed UUID → VM token and back-compatible comparison (`LatentVec.instrument_id` is dashed).
pub fn same_instrument(token: &str, instrument_id: &str) -> bool {
    token.eq_ignore_ascii_case(&instrument_id.replace('-', ""))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_is_step_aligned_and_capped() {
        let (s, e) = window(Timescale::Minute, 6.0, 1_700_000_123_456);
        assert_eq!(e % 60_000, 0);
        assert_eq!(s % 60_000, 0);
        assert_eq!(e - s, 6 * 3_600_000);
        let (s, e) = window(Timescale::Second, 24.0, 1_700_000_000_000);
        assert_eq!(e - s, MAX_POINTS_PER_SERIES * 1000, "capped to 20 000 steps");
    }

    #[test]
    fn price_points_become_bars() {
        let (ends, bars) = bars_from_prices(&[(60_000, 10.0), (120_000, 11.0), (240_000, 9.5)]);
        assert_eq!(ends, vec![60_000, 120_000, 240_000]);
        assert_eq!(bars[0].open, 10.0);
        assert_eq!(bars[1].open, 10.0);
        assert_eq!(bars[1].high, 11.0);
        assert_eq!(bars[2].low, 9.5);
        assert_eq!(bars[2].open, 11.0);
        assert!(same_instrument("01900000000070008000000000000001", "01900000-0000-7000-8000-000000000001"));
    }
}
