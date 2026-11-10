//! Evaluation data for `evo.eval.decoder` (spec 41 §5, spec 34 §6.5): VictoriaMetrics first,
//! the in-memory `md.trade.>` buffer as fallback.
//!
//! The primary source is `fa_tsdb::Tsdb::trade_prices` (`fa_trade_price` written by the
//! ingestor, D-038) over a **data window** — `Job.spec` `{"data_window": {"hours": H}}` or
//! `{"start_ms", "end_ms"}`, else the worker default — at the genome's timescale step. The
//! window is shrunk to at most `fa_tsdb::MAX_POINTS_PER_SERIES` steps. If VM is unreachable,
//! errors, or no instrument has enough points for the genome, the buffer is used and the
//! result's `data.source` says so (`"victoriametrics"` vs `"md.trade.> in-memory buffer
//! (fallback: <reason>)"`).

use fa_backtest::replay::BarSeries;
use fa_core::Timescale;
use fa_decoders::{baseline::Series, genome::DecoderGenome};
use fa_tsdb::{MAX_POINTS_PER_SERIES, Tsdb};
use serde::{Deserialize, Serialize};

use crate::market::{MarketBuffer, now_ms};

/// Window selector from `Job.spec` (all fields optional).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct DataWindow {
    pub hours: Option<f64>,
    pub start_ms: Option<i64>,
    pub end_ms: Option<i64>,
}

impl DataWindow {
    /// Resolves to `[start_ms, end_ms]`, capped to `MAX_POINTS_PER_SERIES` steps of `step_ms`
    /// (the start moves forward). `default_hours` applies when neither `hours` nor `start_ms` is set.
    pub fn resolve(&self, now: i64, default_hours: f64, step_ms: i64) -> (i64, i64) {
        let end = self.end_ms.unwrap_or(now).min(now);
        let start = match (self.start_ms, self.hours) {
            (Some(s), _) => s,
            (None, Some(h)) => end - (h.max(0.0) * 3_600_000.0) as i64,
            (None, None) => end - (default_hours.max(0.0) * 3_600_000.0) as i64,
        };
        let max_span = MAX_POINTS_PER_SERIES * step_ms.max(1);
        (start.max(end - max_span).min(end), end)
    }
}

/// Where the series came from and how the window was chosen.
#[derive(Debug, Clone, Serialize)]
pub struct DataSource {
    /// `victoriametrics` | `md.trade.> in-memory buffer (fallback: …)`.
    pub source: String,
    pub timescale: String,
    pub step_ms: i64,
    pub window_start_ms: i64,
    pub window_end_ms: i64,
    pub vm_url: Option<String>,
    pub vm_series: usize,
    pub vm_error: Option<String>,
    pub buffer_instruments: usize,
    pub buffer_trades: usize,
}

/// One usable instrument series plus its venue.
pub struct Loaded {
    pub series: Vec<Series>,
    pub venues: Vec<(String, String)>,
    pub source: DataSource,
    /// Longest series seen (for the "insufficient data" message).
    pub longest: usize,
}

/// Loads series for `genome`: VM over `window`, else the buffer. `need` is the minimum series
/// length a decoder needs; series shorter than that are dropped.
pub async fn load(
    tsdb: Option<&Tsdb>,
    market: &MarketBuffer,
    genome: &DecoderGenome,
    window: &DataWindow,
    default_hours: f64,
    need: usize,
) -> Loaded {
    let step_ms = i64::try_from(genome.timescale.duration().as_millis()).unwrap_or(1000);
    let now = now_ms();
    let (start_ms, end_ms) = window.resolve(now, default_hours, step_ms);
    let (buf_inst, buf_trades) = market.stats();
    let mut src = DataSource {
        source: String::new(),
        timescale: genome.timescale.to_string(),
        step_ms,
        window_start_ms: start_ms,
        window_end_ms: end_ms,
        vm_url: tsdb.map(|t| t.base_url().to_owned()),
        vm_series: 0,
        vm_error: None,
        buffer_instruments: buf_inst,
        buffer_trades: buf_trades,
    };
    let mut longest = 0usize;

    if let Some(vm) = tsdb {
        match vm
            .trade_prices(
                genome.instrument.as_deref(),
                start_ms / 1000,
                end_ms / 1000,
                fa_tsdb::step_of(genome.timescale),
            )
            .await
        {
            Ok(ts) => {
                src.vm_series = ts.len();
                let mut series = Vec::new();
                let mut venues = Vec::new();
                for t in ts {
                    let s = Series::resample(&t.inst, &t.points, step_ms);
                    longest = longest.max(s.len());
                    if s.len() >= need {
                        venues.push((t.inst.clone(), t.venue.clone()));
                        series.push(s);
                    }
                }
                if series.is_empty() {
                    src.vm_error =
                        Some(format!("{} series in window, longest {longest} steps < {need} needed", src.vm_series));
                } else {
                    src.source = "victoriametrics".into();
                    return Loaded { series, venues, source: src, longest };
                }
            }
            Err(e) => src.vm_error = Some(e.to_string()),
        }
    }

    // Fallback: in-memory buffer (fed from md.trade.> since the worker started).
    let snapshot = market.snapshot(genome.instrument.as_deref());
    let mut series = Vec::new();
    let mut venues = Vec::new();
    for t in snapshot {
        let s = Series::resample(&fa_decoders::instrument_token(&t.instrument_id), &t.trades, step_ms);
        longest = longest.max(s.len());
        if s.len() >= need {
            venues.push((s.instrument_id.clone(), t.venue.clone()));
            series.push(s);
        }
    }
    let reason = match (&src.vm_url, &src.vm_error) {
        (None, _) => "no VictoriaMetrics configured".to_owned(),
        (Some(_), Some(e)) => e.clone(),
        (Some(_), None) => "VictoriaMetrics returned nothing".to_owned(),
    };
    src.source = format!("md.trade.> in-memory buffer (fallback: {reason})");
    Loaded { series, venues, source: src, longest }
}

/// Bars for a strategy replay (same sources and window rules as [`load`]).
pub struct LoadedBars {
    pub bars: Vec<BarSeries>,
    pub source: DataSource,
    pub longest: usize,
    pub n_bars: usize,
}

/// Loads bar series at `timescale` for every instrument (or `instrument`) with at least
/// `need` bars: VictoriaMetrics `fa_trade_price` resampled to the timescale (last price per
/// bucket, stamped at the bucket close), else the in-memory buffer.
pub async fn load_bars(
    tsdb: Option<&Tsdb>,
    market: &MarketBuffer,
    timescale: Timescale,
    instrument: Option<&str>,
    window: &DataWindow,
    default_hours: f64,
    need: usize,
) -> LoadedBars {
    let step_ms = i64::try_from(timescale.duration().as_millis()).unwrap_or(1000);
    let now = now_ms();
    let (start_ms, end_ms) = window.resolve(now, default_hours, step_ms);
    let (buf_inst, buf_trades) = market.stats();
    let mut src = DataSource {
        source: String::new(),
        timescale: timescale.to_string(),
        step_ms,
        window_start_ms: start_ms,
        window_end_ms: end_ms,
        vm_url: tsdb.map(|t| t.base_url().to_owned()),
        vm_series: 0,
        vm_error: None,
        buffer_instruments: buf_inst,
        buffer_trades: buf_trades,
    };
    let mut longest = 0usize;
    if let Some(vm) = tsdb {
        match vm.trade_prices(instrument, start_ms / 1000, end_ms / 1000, fa_tsdb::step_of(timescale)).await {
            Ok(ts) => {
                src.vm_series = ts.len();
                let mut bars = Vec::new();
                for t in ts {
                    let b = BarSeries::from_points(&t.inst, &t.venue, &t.points, step_ms);
                    longest = longest.max(b.len());
                    if b.len() >= need {
                        bars.push(b);
                    }
                }
                if bars.is_empty() {
                    src.vm_error =
                        Some(format!("{} series in window, longest {longest} bars < {need} needed", src.vm_series));
                } else {
                    src.source = "victoriametrics".into();
                    let n_bars = bars.iter().map(BarSeries::len).sum();
                    return LoadedBars { bars, source: src, longest, n_bars };
                }
            }
            Err(e) => src.vm_error = Some(e.to_string()),
        }
    }
    let mut bars = Vec::new();
    for t in market.snapshot(instrument) {
        let b = BarSeries::from_points(&fa_decoders::instrument_token(&t.instrument_id), &t.venue, &t.trades, step_ms);
        longest = longest.max(b.len());
        if b.len() >= need {
            bars.push(b);
        }
    }
    let reason = match (&src.vm_url, &src.vm_error) {
        (None, _) => "no VictoriaMetrics configured".to_owned(),
        (Some(_), Some(e)) => e.clone(),
        (Some(_), None) => "VictoriaMetrics returned nothing".to_owned(),
    };
    src.source = format!("md.trade.> in-memory buffer (fallback: {reason})");
    let n_bars = bars.iter().map(BarSeries::len).sum();
    LoadedBars { bars, source: src, longest, n_bars }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_resolution_and_cap() {
        let now = 10_000_000_000;
        let w = DataWindow::default();
        assert_eq!(w.resolve(now, 2.0, 1000), (now - 7_200_000, now));
        let w = DataWindow { hours: Some(1.0), ..Default::default() };
        assert_eq!(w.resolve(now, 6.0, 60_000), (now - 3_600_000, now));
        // 24 h at 1 s = 86 400 points → capped at MAX_POINTS_PER_SERIES
        let w = DataWindow { hours: Some(24.0), ..Default::default() };
        assert_eq!(w.resolve(now, 6.0, 1000), (now - MAX_POINTS_PER_SERIES * 1000, now));
        let w = DataWindow { start_ms: Some(now - 5000), end_ms: Some(now + 999), hours: None };
        assert_eq!(w.resolve(now, 6.0, 1000), (now - 5000, now));
        let parsed: DataWindow = serde_json::from_str(r#"{"hours": 3}"#).unwrap();
        assert_eq!(parsed.hours, Some(3.0));
    }

    #[tokio::test]
    async fn falls_back_to_buffer_when_vm_is_unreachable() {
        let market = MarketBuffer::default();
        for i in 0..50 {
            market.push("0190-AB", "sim", i * 1000, 100.0 + i as f64);
        }
        let genome = DecoderGenome::parse("forecast:h=1;lb=4").unwrap();
        // unroutable port → connect error quickly
        let vm = Tsdb::new("http://127.0.0.1:9");
        let got = load(Some(&vm), &market, &genome, &DataWindow::default(), 1.0, 10).await;
        assert!(got.source.source.starts_with("md.trade.> in-memory buffer (fallback: "), "{}", got.source.source);
        assert!(got.source.vm_error.is_some());
        assert_eq!(got.series.len(), 1);
        assert_eq!(got.series[0].instrument_id, "0190ab");
        assert_eq!(got.venues[0].1, "sim");
        let none = load(None, &market, &genome, &DataWindow::default(), 1.0, 10).await;
        assert!(none.source.source.contains("no VictoriaMetrics configured"));
        let short = load(None, &market, &genome, &DataWindow::default(), 1.0, 500).await;
        assert!(short.series.is_empty() && short.longest == 50);
        let bars = load_bars(None, &market, Timescale::Second, None, &DataWindow::default(), 1.0, 10).await;
        assert_eq!(bars.bars.len(), 1);
        assert_eq!((bars.bars[0].venue.as_str(), bars.bars[0].len(), bars.n_bars), ("sim", 50, 50));
        assert!(bars.bars[0].ts_ms.windows(2).all(|w| w[1] > w[0]), "bars stamped at close, ascending");
    }
}
