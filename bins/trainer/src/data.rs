//! Training data: bars per timescale and instrument from VictoriaMetrics.
//!
//! **Bars first.** The ingestor writes connector bars as `fa_bar_{open,high,low,close,volume,vwap}`
//! (`tf` label, Yahoo writes `1m`). [`load_one`] reads them with [`fa_tsdb::Tsdb::bars`]; for a
//! timescale coarser than the bar timeframe the 1m bars are aggregated into step buckets
//! (first open, max high, min low, last close, summed volume). Real bars give the activity
//! expert a real volume profile.
//!
//! **Trades as fallback, per instrument.** An instrument without bars (the synthetic
//! heartbeat, any trade-only feed, every `1s` fit) falls back to
//! `last_over_time(fa_trade_price{inst=…}[step])`, turned into **price-only** bars (`open` =
//! previous close, `high`/`low` = max/min of open and close, `volume` = 1, `trade_count` = 1).
//! The manifest records which source fed each instrument (`data.instruments[].source = bars | trades`).
//!
//! The window start is aligned to the step so bar ends coincide with model-runtime's bar
//! closes (what `LatentVec.ts` holds), which the probe-set alignment relies on.

use std::collections::BTreeMap;

use fa_core::Timescale;
use fa_encoders::{InstrumentDataRef, MarketBar};
use fa_tsdb::{BarSeries, MAX_POINTS_PER_SERIES, Tsdb};
use serde::{Deserialize, Serialize};

/// Timeframe label of the connector bars the trainer reads (`fa_bar_*{tf="1m"}`).
pub const BAR_TF: &str = "1m";
const BAR_TF_MS: i64 = 60_000;
/// An instrument needs at least this many bars from a source to use it.
pub const MIN_BARS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    /// `fa_bar_*` OHLCV(+vwap).
    Bars,
    /// Price-only bars from `fa_trade_price`.
    Trades,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bars => "bars",
            Self::Trades => "trades",
        }
    }
}

/// Bars of one instrument, oldest first, with their bar-end timestamps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BarHistory {
    /// Instrument token as VictoriaMetrics labels it (dashless UUID).
    pub instrument: String,
    pub timescale: Timescale,
    pub source: Source,
    pub start_ms: i64,
    pub end_ms: i64,
    pub ends_ms: Vec<i64>,
    pub bars: Vec<MarketBar>,
}

impl BarHistory {
    pub fn data_ref(&self) -> InstrumentDataRef {
        InstrumentDataRef {
            instrument: self.instrument.clone(),
            source: self.source.as_str().into(),
            n_bars: self.bars.len(),
            start_ms: self.ends_ms.first().copied().unwrap_or(self.start_ms),
            end_ms: self.ends_ms.last().copied().unwrap_or(self.end_ms),
        }
    }
}

/// Which instruments a fit reads (`Job.spec.instruments`: a list or `"all"`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Instruments {
    /// `"all"`: every instrument with bars or trades in the window.
    All(AllTag),
    List(Vec<String>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AllTag {
    All,
}

impl Instruments {
    pub fn parse_cli(s: &str) -> Self {
        if s.trim().eq_ignore_ascii_case("all") {
            Self::All(AllTag::All)
        } else {
            Self::List(s.split(',').map(str::trim).filter(|x| !x.is_empty()).map(str::to_owned).collect())
        }
    }
}

pub fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

pub fn step_ms(ts: Timescale) -> i64 {
    i64::try_from(ts.duration().as_millis()).unwrap_or(60_000).max(1)
}

/// `[start, end]` for `window_hours` ending at `end_ms`, aligned to the step and capped to
/// `MAX_POINTS_PER_SERIES` steps.
pub fn window(ts: Timescale, window_hours: f64, end_ms: i64) -> (i64, i64) {
    let step_ms = step_ms(ts);
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

/// Connector bars → `MarketBar`s at `step_ms` (≥ the bar timeframe): bars are bucketed by
/// `ceil(ts / step) · step` (the bucket's end) and aggregated (first open, max high, min low,
/// last close, summed volume, `trade_count` = number of source bars). With `step_ms` equal to
/// the bar timeframe this is a 1:1 conversion.
pub fn bars_from_series(series: &BarSeries, step_ms: i64) -> (Vec<i64>, Vec<MarketBar>) {
    let step = step_ms.max(1);
    let mut buckets: BTreeMap<i64, MarketBar> = BTreeMap::new();
    for p in &series.points {
        let end = p.ts_ms + (step - p.ts_ms.rem_euclid(step)) % step;
        buckets
            .entry(end)
            .and_modify(|b| {
                b.high = b.high.max(p.high);
                b.low = b.low.min(p.low);
                b.close = p.close;
                b.volume += p.volume;
                b.trade_count += 1;
            })
            .or_insert(MarketBar {
                open: p.open,
                high: p.high.max(p.open).max(p.close),
                low: p.low.min(p.open).min(p.close),
                close: p.close,
                volume: p.volume,
                trade_count: 1,
            });
    }
    buckets.into_iter().unzip()
}

/// Whether connector bars can feed `ts`: the timescale must be at least the bar timeframe and
/// the window must fit into one `query_range` at bar resolution.
pub fn bars_usable(ts: Timescale, start_ms: i64, end_ms: i64) -> bool {
    step_ms(ts) >= BAR_TF_MS && (end_ms - start_ms) / BAR_TF_MS <= MAX_POINTS_PER_SERIES
}

async fn bar_history(
    tsdb: &Tsdb,
    ts: Timescale,
    instrument: Option<&str>,
    start_ms: i64,
    end_ms: i64,
) -> anyhow::Result<Vec<BarHistory>> {
    if !bars_usable(ts, start_ms, end_ms) {
        return Ok(Vec::new());
    }
    let series = tsdb.bars(instrument, BAR_TF, start_ms / 1000, end_ms / 1000).await?;
    Ok(series
        .iter()
        .map(|s| {
            let (ends_ms, bars) = bars_from_series(s, step_ms(ts));
            BarHistory {
                instrument: s.inst.clone(),
                timescale: ts,
                source: Source::Bars,
                start_ms,
                end_ms,
                ends_ms,
                bars,
            }
        })
        .filter(|h| h.bars.len() >= MIN_BARS)
        .collect())
}

async fn trade_history(
    tsdb: &Tsdb,
    ts: Timescale,
    instrument: Option<&str>,
    start_ms: i64,
    end_ms: i64,
) -> anyhow::Result<Vec<BarHistory>> {
    let series = tsdb.trade_prices(instrument, start_ms / 1000, end_ms / 1000, fa_tsdb::step_of(ts)).await?;
    Ok(series
        .into_iter()
        .map(|s| {
            let (ends_ms, bars) = bars_from_prices(&s.points);
            BarHistory { instrument: s.inst, timescale: ts, source: Source::Trades, start_ms, end_ms, ends_ms, bars }
        })
        .collect())
}

/// One instrument: bars when available (≥ [`MIN_BARS`]), else trades. `prefer` forces a source
/// (checkpoint resume reads exactly what the interrupted run read). `instrument = None` → the
/// instrument with the most bars (or trades).
pub async fn load_one(
    tsdb: &Tsdb,
    ts: Timescale,
    instrument: Option<&str>,
    start_ms: i64,
    end_ms: i64,
    prefer: Option<Source>,
) -> anyhow::Result<BarHistory> {
    if prefer != Some(Source::Trades)
        && let Some(h) = bar_history(tsdb, ts, instrument, start_ms, end_ms).await?.into_iter().next()
    {
        return Ok(h);
    }
    if prefer == Some(Source::Bars) {
        anyhow::bail!("no fa_bar_* bars for {} any more", instrument.unwrap_or("?"));
    }
    match trade_history(tsdb, ts, instrument, start_ms, end_ms).await?.into_iter().next() {
        Some(h) => Ok(h),
        None => anyhow::bail!(
            "no fa_bar_* or fa_trade_price series{} between {start_ms} and {end_ms} at {ts}",
            instrument.map(|i| format!(" for {i}")).unwrap_or_default()
        ),
    }
}

/// Every instrument with data in the window: bars where an instrument has them, trades
/// otherwise (per instrument). Sorted by instrument token for determinism.
pub async fn load_all(tsdb: &Tsdb, ts: Timescale, start_ms: i64, end_ms: i64) -> anyhow::Result<Vec<BarHistory>> {
    let mut by_inst: BTreeMap<String, BarHistory> = BTreeMap::new();
    for h in bar_history(tsdb, ts, None, start_ms, end_ms).await? {
        by_inst.insert(h.instrument.clone(), h);
    }
    for h in trade_history(tsdb, ts, None, start_ms, end_ms).await? {
        by_inst.entry(h.instrument.clone()).or_insert(h);
    }
    if by_inst.is_empty() {
        anyhow::bail!("no fa_bar_* or fa_trade_price series between {start_ms} and {end_ms} at {ts}");
    }
    Ok(by_inst.into_values().collect())
}

/// The histories a fit reads: an explicit list, `"all"`, or the single `instrument` (default:
/// the instrument with the most bars). `pinned` = `(token, source)` pairs of a checkpoint.
pub async fn load_set(
    tsdb: &Tsdb,
    ts: Timescale,
    instruments: Option<&Instruments>,
    instrument: Option<&str>,
    start_ms: i64,
    end_ms: i64,
    pinned: Option<&[(String, Source)]>,
) -> anyhow::Result<Vec<BarHistory>> {
    if let Some(pins) = pinned {
        let mut out = Vec::with_capacity(pins.len());
        for (inst, src) in pins {
            out.push(load_one(tsdb, ts, Some(inst), start_ms, end_ms, Some(*src)).await?);
        }
        return Ok(out);
    }
    match instruments {
        Some(Instruments::All(_)) => load_all(tsdb, ts, start_ms, end_ms).await,
        Some(Instruments::List(list)) if !list.is_empty() => {
            let mut out = Vec::with_capacity(list.len());
            for i in list {
                out.push(load_one(tsdb, ts, Some(i), start_ms, end_ms, None).await?);
            }
            out.sort_by(|a, b| a.instrument.cmp(&b.instrument));
            out.dedup_by(|a, b| a.instrument == b.instrument);
            Ok(out)
        }
        _ => Ok(vec![load_one(tsdb, ts, instrument, start_ms, end_ms, None).await?]),
    }
}

#[cfg(test)]
mod tests {
    use fa_tsdb::BarPoint;

    use super::*;

    #[test]
    fn window_is_step_aligned_and_capped() {
        let (s, e) = window(Timescale::Minute, 6.0, 1_700_000_123_456);
        assert_eq!(e % 60_000, 0);
        assert_eq!(s % 60_000, 0);
        assert_eq!(e - s, 6 * 3_600_000);
        let (s, e) = window(Timescale::Second, 24.0, 1_700_000_000_000);
        assert_eq!(e - s, MAX_POINTS_PER_SERIES * 1000, "capped to 20 000 steps");
        assert!(bars_usable(Timescale::Minute, s, e));
        assert!(!bars_usable(Timescale::Second, s, e), "1s fits fall back to trades");
        assert!(!bars_usable(Timescale::Hour, 0, 30 * 24 * 3_600_000), "too many 1m points");
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
    }

    #[test]
    fn connector_bars_convert_and_aggregate() {
        let p = |t: i64, o: f64, h: f64, l: f64, c: f64, v: f64| BarPoint {
            ts_ms: t,
            open: o,
            high: h,
            low: l,
            close: c,
            volume: v,
            vwap: None,
        };
        let s = BarSeries {
            inst: "x".into(),
            venue: "yahoo".into(),
            src: "yahoo".into(),
            tf: "1m".into(),
            points: vec![
                p(3_600_000, 10.0, 11.0, 9.0, 10.5, 100.0),
                p(3_660_000, 10.5, 12.0, 10.0, 11.0, 50.0),
                p(7_140_000, 11.0, 11.5, 8.0, 9.0, 25.0),
                p(7_200_000, 9.0, 9.5, 8.5, 9.2, 5.0),
            ],
        };
        let (ends, bars) = bars_from_series(&s, 60_000);
        assert_eq!(ends.len(), 4, "1:1 at the bar timeframe");
        assert_eq!(bars[0].volume, 100.0);
        assert_eq!(bars[1].close, 11.0);
        let (ends, bars) = bars_from_series(&s, 3_600_000);
        // 3_600_000 is itself an hour end; 3_660_000..=7_200_000 fall into the bucket ending 7_200_000
        assert_eq!(ends, vec![3_600_000, 7_200_000]);
        assert_eq!(bars[1].open, 10.5);
        assert_eq!(bars[1].high, 12.0);
        assert_eq!(bars[1].low, 8.0);
        assert_eq!(bars[1].close, 9.2);
        assert_eq!(bars[1].volume, 80.0);
        assert_eq!(bars[1].trade_count, 3);
    }

    #[test]
    fn instruments_spec_parses_list_or_all() {
        let all: Instruments = serde_json::from_str(r#""all""#).unwrap();
        assert_eq!(all, Instruments::All(AllTag::All));
        let list: Instruments = serde_json::from_str(r#"["a","b"]"#).unwrap();
        assert_eq!(list, Instruments::List(vec!["a".into(), "b".into()]));
        assert_eq!(Instruments::parse_cli("ALL"), all);
        assert_eq!(Instruments::parse_cli("a, b,"), list);
        assert!(serde_json::from_str::<Instruments>(r#""some""#).is_err());
    }
}
