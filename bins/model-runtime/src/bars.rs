//! Rolling bars per (timescale, instrument) built from `md.trade.>` **or** `md.bar.<tf>.>`
//! (spec 31 §6.1 context, D-062).
//!
//! Bars are bucketed by `event_time` on fixed boundaries (`floor(t / step)`). From trades, a
//! bar closes when the first trade of a later bucket arrives; late trades fold into the forming
//! bar. From bus bars (a venue that publishes OHLCV, e.g. the Yahoo connector), bars whose
//! timeframe equals the series step close immediately, finer bars are **resampled upward**
//! (OHLC merged, volume and trade count summed, vwap volume-weighted) and the forming bar closes
//! as soon as its last sub-bar lands or a later bucket starts. Only the last `capacity` closed
//! bars are kept (the encoder's context length).

use std::collections::VecDeque;

use fa_core::Timescale;
use fa_encoders::MarketBar;

/// One bar with the bus-only fields the encoder row does not carry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BarRow {
    pub bar: MarketBar,
    /// Volume-weighted average price when the venue reports it (or it was accumulated from
    /// sub-bars that did); `None` for trade-built bars without a size.
    pub vwap: Option<f64>,
    /// Σ vwap·volume over the sub-bars that carried a vwap (to merge vwaps exactly).
    notional: f64,
    notional_volume: f64,
}

impl BarRow {
    pub fn from_trade(price: f64, size: f64) -> Self {
        let bar = MarketBar::from_trade(price, size);
        let has = size > 0.0;
        Self {
            bar,
            vwap: has.then_some(price),
            notional: if has { price * size } else { 0.0 },
            notional_volume: if has { size } else { 0.0 },
        }
    }

    /// A bus bar. `trade_count = 0` (venues like Yahoo report none) counts as one print so the
    /// activity expert's size-per-trade feature equals the bar volume instead of 0.
    pub fn from_bus(
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        trade_count: u64,
        vwap: Option<f64>,
    ) -> Self {
        let bar = MarketBar { open, high, low, close, volume, trade_count: trade_count.max(1) };
        let vwap = vwap.filter(|v| v.is_finite() && *v > 0.0);
        let (notional, notional_volume) = match vwap {
            Some(v) if volume > 0.0 => (v * volume, volume),
            _ => (0.0, 0.0),
        };
        Self { bar, vwap, notional, notional_volume }
    }

    pub fn add_trade(&mut self, price: f64, size: f64) {
        self.bar.add_trade(price, size);
        if size > 0.0 {
            self.notional += price * size;
            self.notional_volume += size;
            self.vwap = Some(self.notional / self.notional_volume);
        }
    }

    /// Merges a later sub-bar into this one (upward resampling).
    pub fn merge(&mut self, next: &BarRow) {
        self.bar.high = self.bar.high.max(next.bar.high);
        self.bar.low = self.bar.low.min(next.bar.low);
        self.bar.close = next.bar.close;
        self.bar.volume += next.bar.volume;
        self.bar.trade_count += next.bar.trade_count;
        self.notional += next.notional;
        self.notional_volume += next.notional_volume;
        if self.notional_volume > 0.0 {
            self.vwap = Some(self.notional / self.notional_volume);
        }
    }
}

#[derive(Debug, Clone)]
pub struct BarSeries {
    step_ms: i64,
    capacity: usize,
    closed: VecDeque<BarRow>,
    /// Bucket start of the last closed bar (later inputs for it or earlier are ignored by `on_bar`).
    last_closed_start: Option<i64>,
    /// Bucket start (ms) and the bar forming in it.
    forming: Option<(i64, BarRow)>,
}

/// What one trade or bus bar did to the series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarUpdate {
    /// The input extended the forming bar.
    Extended,
    /// A bar closed with this end time (ms). The input may have opened a new forming bar.
    Closed { bar_end_ms: i64 },
    /// The input lies before the last closed bar (or is coarser than the step) and was ignored.
    Ignored,
}

impl BarSeries {
    pub fn new(ts: Timescale, capacity: usize) -> Self {
        Self {
            step_ms: ts.duration().as_millis() as i64,
            capacity: capacity.max(2),
            closed: VecDeque::new(),
            last_closed_start: None,
            forming: None,
        }
    }

    fn push_closed(&mut self, start: i64, row: BarRow) {
        self.last_closed_start = Some(self.last_closed_start.map_or(start, |s| s.max(start)));
        self.closed.push_back(row);
        while self.closed.len() > self.capacity {
            self.closed.pop_front();
        }
    }

    pub fn on_trade(&mut self, t_ms: i64, price: f64, size: f64) -> BarUpdate {
        let bucket = t_ms.div_euclid(self.step_ms) * self.step_ms;
        match &mut self.forming {
            Some((start, bar)) if bucket <= *start => {
                bar.add_trade(price, size);
                BarUpdate::Extended
            }
            Some((start, bar)) => {
                let (fstart, done) = (*start, *bar);
                let bar_end_ms = fstart + self.step_ms;
                self.push_closed(fstart, done);
                self.forming = Some((bucket, BarRow::from_trade(price, size)));
                BarUpdate::Closed { bar_end_ms }
            }
            None => {
                self.forming = Some((bucket, BarRow::from_trade(price, size)));
                BarUpdate::Extended
            }
        }
    }

    /// A bus bar of timeframe `tf_ms` covering `[start_ms, start_ms + tf_ms)`. Bars coarser than
    /// the step, or older than the last closed bar, are ignored.
    pub fn on_bar(&mut self, start_ms: i64, tf_ms: i64, row: BarRow) -> BarUpdate {
        if tf_ms > self.step_ms || tf_ms <= 0 {
            return BarUpdate::Ignored;
        }
        let bucket = start_ms.div_euclid(self.step_ms) * self.step_ms;
        if self.last_closed_start.is_some_and(|s| bucket <= s) {
            return BarUpdate::Ignored;
        }
        let sub_end = start_ms + tf_ms;
        if tf_ms == self.step_ms {
            // a complete bar of this step: close it now
            match self.forming.take() {
                Some((fstart, fbar)) if fstart < bucket => {
                    // an older forming bar (from trades or partial sub-bars) closes first
                    self.push_closed(fstart, fbar);
                }
                Some((fstart, fbar)) if fstart > bucket => {
                    self.forming = Some((fstart, fbar));
                    return BarUpdate::Ignored;
                }
                _ => {}
            }
            self.push_closed(bucket, row);
            return BarUpdate::Closed { bar_end_ms: bucket + self.step_ms };
        }
        match &mut self.forming {
            Some((start, bar)) if bucket == *start => {
                bar.merge(&row);
                if sub_end >= *start + self.step_ms {
                    let (fstart, done) = (*start, *bar);
                    let bar_end_ms = fstart + self.step_ms;
                    self.forming = None;
                    self.push_closed(fstart, done);
                    BarUpdate::Closed { bar_end_ms }
                } else {
                    BarUpdate::Extended
                }
            }
            Some((start, _)) if bucket < *start => BarUpdate::Ignored,
            Some((start, bar)) => {
                let (fstart, done) = (*start, *bar);
                let bar_end_ms = fstart + self.step_ms;
                self.push_closed(fstart, done);
                self.forming = Some((bucket, row));
                if sub_end >= bucket + self.step_ms {
                    // the new sub-bar alone completes its bucket
                    self.forming = None;
                    self.push_closed(bucket, row);
                    return BarUpdate::Closed { bar_end_ms: bucket + self.step_ms };
                }
                BarUpdate::Closed { bar_end_ms }
            }
            None => {
                if sub_end >= bucket + self.step_ms {
                    self.push_closed(bucket, row);
                    BarUpdate::Closed { bar_end_ms: bucket + self.step_ms }
                } else {
                    self.forming = Some((bucket, row));
                    BarUpdate::Extended
                }
            }
        }
    }

    /// Closed bars, oldest first.
    pub fn closed(&self) -> Vec<MarketBar> {
        self.closed.iter().map(|r| r.bar).collect()
    }

    /// Closed bars plus the forming bar (a live preview of the current step).
    pub fn with_forming(&self) -> Vec<MarketBar> {
        let mut v = self.closed();
        if let Some((_, b)) = &self.forming {
            v.push(b.bar);
        }
        v
    }

    /// Closed rows with their vwap, oldest first.
    #[cfg(test)]
    pub fn closed_rows(&self) -> Vec<BarRow> {
        self.closed.iter().copied().collect()
    }

    #[cfg(test)]
    pub fn closed_len(&self) -> usize {
        self.closed.len()
    }
}

/// Parses a bar timeframe token (`1m`, `5m`, `15m`, `1h`, `1d`, `1w`, `30s`) to milliseconds.
pub fn parse_tf_ms(tf: &str) -> Option<i64> {
    let tf = tf.trim();
    let split = tf.find(|c: char| !c.is_ascii_digit())?;
    let n: i64 = tf[..split].parse().ok()?;
    let unit = match &tf[split..] {
        "s" => 1_000,
        "m" => 60_000,
        "h" => 3_600_000,
        "d" => 86_400_000,
        "w" => 604_800_000,
        _ => return None,
    };
    (n > 0).then_some(n * unit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_close_on_the_next_period_and_keep_capacity() {
        let mut s = BarSeries::new(Timescale::Minute, 3);
        assert_eq!(s.on_trade(5_000, 10.0, 1.0), BarUpdate::Extended);
        assert_eq!(s.on_trade(30_000, 12.0, 2.0), BarUpdate::Extended);
        assert_eq!(s.on_trade(61_000, 11.0, 1.0), BarUpdate::Closed { bar_end_ms: 60_000 });
        let c = s.closed();
        assert_eq!(c.len(), 1);
        assert_eq!((c[0].open, c[0].high, c[0].close, c[0].volume, c[0].trade_count), (10.0, 12.0, 12.0, 3.0, 2));
        // vwap of the trade-built bar: (10·1 + 12·2) / 3
        let r = s.closed_rows()[0];
        assert!((r.vwap.unwrap() - 34.0 / 3.0).abs() < 1e-12);
        assert_eq!(s.with_forming().len(), 2);
        for k in 2..8 {
            assert!(matches!(s.on_trade(k * 60_000 + 1, 10.0, 1.0), BarUpdate::Closed { .. }));
        }
        assert_eq!(s.closed_len(), 3);
        // a late trade folds into the forming bar instead of reopening the past
        assert_eq!(s.on_trade(1_000, 9.0, 1.0), BarUpdate::Extended);
        assert_eq!(s.with_forming().last().unwrap().low, 9.0);
    }

    fn bus(o: f64, h: f64, l: f64, c: f64, v: f64, vwap: Option<f64>) -> BarRow {
        BarRow::from_bus(o, h, l, c, v, 0, vwap)
    }

    #[test]
    fn bus_bars_of_the_step_close_immediately_with_volume() {
        let mut s = BarSeries::new(Timescale::Minute, 4);
        let m = 60_000;
        assert_eq!(s.on_bar(0, m, bus(10.0, 11.0, 9.0, 10.5, 100.0, Some(10.2))), BarUpdate::Closed { bar_end_ms: m });
        assert_eq!(s.on_bar(m, m, bus(10.5, 12.0, 10.0, 11.5, 250.0, None)), BarUpdate::Closed { bar_end_ms: 2 * m });
        let rows = s.closed_rows();
        assert_eq!(rows.len(), 2);
        assert_eq!((rows[0].bar.volume, rows[0].bar.trade_count, rows[0].vwap), (100.0, 1, Some(10.2)));
        assert_eq!((rows[1].bar.volume, rows[1].vwap), (250.0, None));
        // an out-of-order older bar is ignored; a coarser bar is ignored
        assert_eq!(s.on_bar(0, m, bus(1.0, 1.0, 1.0, 1.0, 1.0, None)), BarUpdate::Ignored);
        assert_eq!(s.on_bar(0, 3_600_000, bus(1.0, 1.0, 1.0, 1.0, 1.0, None)), BarUpdate::Ignored);
        assert_eq!(s.closed_len(), 2);
        assert_eq!(s.with_forming().len(), 2, "no forming bar between complete bus bars");
    }

    #[test]
    fn finer_bus_bars_resample_upward_summing_volume_and_weighting_vwap() {
        let mut s = BarSeries::new(Timescale::Hour, 4);
        let m = 60_000;
        let h = 3_600_000;
        // 60 one-minute bars fill the first hour: the last one closes it
        for i in 0..60 {
            let p = 100.0 + i as f64;
            let upd = s.on_bar(i * m, m, bus(p, p + 1.0, p - 1.0, p + 0.5, 10.0, Some(p)));
            if i < 59 {
                assert_eq!(upd, BarUpdate::Extended, "minute {i}");
                assert_eq!(s.with_forming().len(), 1);
            } else {
                assert_eq!(upd, BarUpdate::Closed { bar_end_ms: h });
            }
        }
        let r = s.closed_rows();
        assert_eq!(r.len(), 1);
        let b = r[0].bar;
        assert_eq!((b.open, b.high, b.low, b.close), (100.0, 160.0, 99.0, 159.5));
        assert_eq!(b.volume, 600.0);
        assert_eq!(b.trade_count, 60);
        // vwap = Σ p·10 / 600 = mean of 100..159 = 129.5
        assert!((r[0].vwap.unwrap() - 129.5).abs() < 1e-9);
        // a gap: the next bar lands in hour 3 → hour 2 never existed, hour 3 starts forming
        assert_eq!(s.on_bar(2 * h, m, bus(1.0, 1.0, 1.0, 1.0, 5.0, None)), BarUpdate::Extended);
        // a bar from hour 4 closes the partial hour 3 bar (with its partial volume)
        assert_eq!(s.on_bar(3 * h, m, bus(2.0, 2.0, 2.0, 2.0, 7.0, None)), BarUpdate::Closed { bar_end_ms: 3 * h });
        assert_eq!(s.closed()[1].volume, 5.0);
        // trades still fold into the forming (hour 4) bar
        assert_eq!(s.on_trade(3 * h + 30 * m, 3.0, 1.0), BarUpdate::Extended);
        let f = s.with_forming();
        assert_eq!(f.last().unwrap().volume, 8.0);
    }

    #[test]
    fn timeframe_tokens_parse() {
        assert_eq!(parse_tf_ms("1m"), Some(60_000));
        assert_eq!(parse_tf_ms("5m"), Some(300_000));
        assert_eq!(parse_tf_ms("1h"), Some(3_600_000));
        assert_eq!(parse_tf_ms("1d"), Some(86_400_000));
        assert_eq!(parse_tf_ms("30s"), Some(30_000));
        assert_eq!(parse_tf_ms("1w"), Some(604_800_000));
        assert_eq!(parse_tf_ms("x"), None);
        assert_eq!(parse_tf_ms("0m"), None);
        assert_eq!(parse_tf_ms("1y"), None);
    }
}
