//! Point-in-time reads for VictoriaMetrics (spec 20 §6.3, open question 4).
//!
//! VictoriaMetrics stores one timestamp per sample, the event time; it has no publish time.
//! A sample is taken as **known** at `event time + knowledge delay of its source` — the per-source
//! delays measured into `config/sources.yaml` (`backfill.knowledge_delay_secs`, read by
//! `fa_backtest::knowledge::KnowledgeDelays`, D-079). An as-of read returns only buckets whose
//! timestamp plus that delay is at or before `as_of`; since every bucket of a
//! `last_over_time(..[step])` read holds samples with event time ≤ the bucket time, no sample
//! published after `as_of` can leak in. Exact per-event availability (a late correction, an
//! out-of-order backfill) is only in the raw archive and the recorded bus (`fa_bus::replay`);
//! this module deliberately does not add an availability index (decision for Q4).
use std::collections::BTreeMap;

use fa_core::Timescale;

use crate::{BarSeries, TradeSeries, Tsdb, TsdbError, bar_window_s, step_of};

/// An as-of cut: `as_of_ms` plus the knowledge delay per source (matched on the series' `src`
/// label, then its `venue`), `default_delay_ms` for sources not listed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AsOf {
    pub as_of_ms: i64,
    pub delays_ms: BTreeMap<String, i64>,
    pub default_delay_ms: i64,
}

impl AsOf {
    /// A cut with no delays (event time = knowledge time).
    pub fn at(as_of_ms: i64) -> Self {
        Self { as_of_ms, ..Default::default() }
    }

    /// Adds a source's delay.
    pub fn delay(mut self, source: impl Into<String>, delay_ms: i64) -> Self {
        self.delays_ms.insert(source.into(), delay_ms.max(0));
        self
    }

    pub fn delay_of(&self, src: &str, venue: &str) -> i64 {
        self.delays_ms.get(src).or_else(|| self.delays_ms.get(venue)).copied().unwrap_or(self.default_delay_ms).max(0)
    }

    /// Whether a sample (or bucket) at `ts_ms` of `src`/`venue` was known at the cut.
    pub fn visible(&self, src: &str, venue: &str, ts_ms: i64) -> bool {
        ts_ms.saturating_add(self.delay_of(src, venue)) <= self.as_of_ms
    }

    /// The latest query end (unix seconds) that can hold a visible bucket.
    pub fn clamp_end_s(&self, end_s: i64) -> i64 {
        end_s.min(self.as_of_ms.div_euclid(1000))
    }

    /// Drops trade buckets not yet known at the cut (and series left empty).
    pub fn filter_trades(&self, series: Vec<TradeSeries>) -> Vec<TradeSeries> {
        series
            .into_iter()
            .map(|mut s| {
                let d = self.delay_of(&s.src, &s.venue);
                s.points.retain(|(t, _)| t.saturating_add(d) <= self.as_of_ms);
                s
            })
            .filter(|s| !s.points.is_empty())
            .collect()
    }

    /// Drops bars not yet known at the cut (and series left empty).
    pub fn filter_bars(&self, series: Vec<BarSeries>) -> Vec<BarSeries> {
        series
            .into_iter()
            .map(|mut s| {
                let d = self.delay_of(&s.src, &s.venue);
                s.points.retain(|p| p.ts_ms.saturating_add(d) <= self.as_of_ms);
                s
            })
            .filter(|s| !s.points.is_empty())
            .collect()
    }
}

impl Tsdb {
    /// [`Tsdb::trade_prices`] as known at `cut` (module docs).
    pub async fn trade_prices_as_of(
        &self,
        instrument: Option<&str>,
        start_s: i64,
        end_s: i64,
        step: &str,
        cut: &AsOf,
    ) -> Result<Vec<TradeSeries>, TsdbError> {
        let end_s = cut.clamp_end_s(end_s);
        if end_s < start_s {
            return Ok(Vec::new());
        }
        Ok(cut.filter_trades(self.trade_prices(instrument, start_s, end_s, step).await?))
    }

    /// [`Tsdb::bars`] as known at `cut` (module docs).
    pub async fn bars_as_of(
        &self,
        instrument: Option<&str>,
        tf: &str,
        start_s: i64,
        end_s: i64,
        cut: &AsOf,
    ) -> Result<Vec<BarSeries>, TsdbError> {
        let end_s = cut.clamp_end_s(end_s);
        if end_s < start_s {
            return Ok(Vec::new());
        }
        Ok(cut.filter_bars(self.bars(instrument, tf, start_s, end_s).await?))
    }

    /// `trade_prices_ms` (one instrument, merged across venues) as known at `cut`.
    pub async fn trade_prices_ms_as_of(
        &self,
        instrument: &str,
        start_ms: i64,
        end_ms: i64,
        bar: Timescale,
        cut: &AsOf,
    ) -> Result<Vec<(i64, f64)>, TsdbError> {
        let (start_s, end_s) = bar_window_s(start_ms, end_ms, bar);
        let series = self.trade_prices_as_of(Some(instrument), start_s, end_s, step_of(bar), cut).await?;
        let mut merged: BTreeMap<i64, f64> = BTreeMap::new();
        for s in &series {
            for (t, p) in &s.points {
                merged.entry(*t).or_insert(*p);
            }
        }
        Ok(merged.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BarPoint;

    /// xorshift64*: deterministic fixtures without a rand dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: u64) -> i64 {
            (self.next() % n) as i64
        }
    }

    /// Spec 20 criterion 2 for VictoriaMetrics: fixtures with late publications (per-source
    /// delays up to 10 min, random event times) probed at random cuts and around every
    /// publication instant — an as-of read never returns a point published after the cut, and
    /// returns every point published at or before it.
    #[test]
    fn as_of_never_returns_late_publications() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for _round in 0..200 {
            let sources = ["kraken", "yahoo", "sim", "fred"];
            let delays: Vec<i64> = sources.iter().map(|_| rng.below(600_000)).collect();
            let mut cut = AsOf::default();
            for (s, d) in sources.iter().zip(&delays) {
                cut = cut.delay(*s, *d);
            }
            let mut trades = Vec::new();
            let mut bars = Vec::new();
            for (si, s) in sources.iter().enumerate() {
                let mut pts: Vec<(i64, f64)> = (0..50).map(|_| (rng.below(3_600_000), 1.0)).collect();
                pts.sort_by_key(|p| p.0);
                pts.dedup_by_key(|p| p.0);
                trades.push(TradeSeries { inst: "i".into(), venue: s.to_string(), src: s.to_string(), points: pts.clone() });
                bars.push(BarSeries {
                    inst: "i".into(),
                    venue: format!("v{si}"),
                    src: s.to_string(),
                    tf: "1m".into(),
                    points: pts
                        .iter()
                        .map(|(t, _)| BarPoint { ts_ms: *t, open: 1., high: 1., low: 1., close: 1., volume: 0., vwap: None })
                        .collect(),
                });
            }
            let mut probes: Vec<i64> = (0..20).map(|_| rng.below(4_500_000)).collect();
            for (s, d) in trades.iter().zip(&delays) {
                for (t, _) in s.points.iter().take(5) {
                    probes.extend([t + d - 1, t + d, t + d + 1]);
                }
            }
            for at in probes {
                cut.as_of_ms = at;
                let seen = cut.filter_trades(trades.clone());
                for (s, d) in trades.iter().zip(&delays) {
                    let want: Vec<i64> = s.points.iter().map(|p| p.0).filter(|t| t + d <= at).collect();
                    let got: Vec<i64> = seen
                        .iter()
                        .find(|x| x.src == s.src)
                        .map(|x| x.points.iter().map(|p| p.0).collect())
                        .unwrap_or_default();
                    assert_eq!(got, want, "trades {} at {at} delay {d}", s.src);
                }
                for b in cut.filter_bars(bars.clone()) {
                    let d = cut.delay_of(&b.src, &b.venue);
                    assert!(b.points.iter().all(|p| p.ts_ms + d <= at), "bar published after the cut");
                }
                let visible_bars: usize = cut.filter_bars(bars.clone()).iter().map(|b| b.points.len()).sum();
                let visible_trades: usize = seen.iter().map(|s| s.points.len()).sum();
                assert_eq!(visible_bars, visible_trades, "bars keyed by src like trades");
            }
        }
    }

    #[test]
    fn delays_by_src_then_venue_then_default() {
        let cut = AsOf { as_of_ms: 10_000, default_delay_ms: 500, ..Default::default() }.delay("kraken", 2_000).delay("v", 1);
        assert_eq!(cut.delay_of("kraken", "v"), 2_000);
        assert_eq!(cut.delay_of("other", "v"), 1);
        assert_eq!(cut.delay_of("other", "w"), 500);
        assert!(cut.visible("kraken", "", 8_000) && !cut.visible("kraken", "", 8_001));
        assert_eq!(cut.clamp_end_s(99), 10);
        assert_eq!(AsOf::at(5).delay("x", -3).delay_of("x", ""), 0, "negative delays clamp to 0");
    }

    /// Live: `FA_VM_TEST=1 cargo test -p fa-tsdb live_as_of` (VM at `FA_VM_URL`) — bars written
    /// an hour ago are invisible at a cut before their knowledge time and visible after it; only
    /// its own series is written and deleted.
    #[tokio::test]
    async fn live_as_of_hides_unpublished_bars() {
        if std::env::var("FA_VM_TEST").ok().as_deref() != Some("1") {
            eprintln!("skipped: set FA_VM_TEST=1");
            return;
        }
        let url = std::env::var("FA_VM_URL").unwrap_or_else(|_| "http://127.0.0.1:8428".into());
        let vm = Tsdb::new(url);
        let token = crate::inst_token(&fa_core::InstrumentId::new().to_string());
        let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64;
        let base = now_ms - now_ms.rem_euclid(60_000) - 3_600_000;
        let mut samples = Vec::new();
        for i in 0..10i64 {
            for m in crate::metrics::BAR_FAMILY {
                samples.push(
                    crate::Sample::new(m, 100.0 + i as f64, base + i * 60_000)
                        .label("inst", token.clone())
                        .label("venue", "fa-tsdb-asof-test")
                        .label("src", "fa-tsdb-asof-test")
                        .label("tf", "1m"),
                );
            }
        }
        vm.write(&samples).await.unwrap();
        vm.force_flush().await.unwrap();
        let (start, end) = (base / 1000 - 60, base / 1000 + 11 * 60);
        // delay 120 s: at base + 5 min only the bars at base .. base + 3 min are known
        let cut = AsOf::at(base + 5 * 60_000).delay("fa-tsdb-asof-test", 120_000);
        let got = vm.bars_as_of(Some(&token), "1m", start, end, &cut).await.unwrap();
        let ts: Vec<i64> = got[0].points.iter().map(|p| (p.ts_ms - base) / 60_000).collect();
        assert_eq!(ts, [0, 1, 2, 3]);
        let all = vm.bars_as_of(Some(&token), "1m", start, end, &AsOf::at(now_ms)).await.unwrap();
        assert_eq!(all[0].points.len(), 10);
        let _ = vm.delete_series(r#"{venue="fa-tsdb-asof-test"}"#).await;
    }
}
