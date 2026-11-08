//! Rolling bars per (timescale, instrument) built from `md.trade.>` (spec 31 §6.1 context).
//!
//! Bars are bucketed by `event_time` on fixed boundaries (`floor(t / step)`). A bar closes
//! when the first trade of a later bucket arrives; late trades fold into the forming bar.
//! Only the last `capacity` closed bars are kept (the encoder's context length).

use std::collections::VecDeque;

use fa_core::Timescale;
use fa_encoders::MarketBar;

#[derive(Debug, Clone)]
pub struct BarSeries {
    step_ms: i64,
    capacity: usize,
    closed: VecDeque<MarketBar>,
    /// Bucket start (ms) and the bar forming in it.
    forming: Option<(i64, MarketBar)>,
}

/// What one trade did to the series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarUpdate {
    /// The trade extended the forming bar.
    Extended,
    /// The trade opened a new bucket; the previous bar closed with this end time (ms).
    Closed { bar_end_ms: i64 },
}

impl BarSeries {
    pub fn new(ts: Timescale, capacity: usize) -> Self {
        Self {
            step_ms: ts.duration().as_millis() as i64,
            capacity: capacity.max(2),
            closed: VecDeque::new(),
            forming: None,
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
                let bar_end_ms = *start + self.step_ms;
                self.closed.push_back(*bar);
                while self.closed.len() > self.capacity {
                    self.closed.pop_front();
                }
                self.forming = Some((bucket, MarketBar::from_trade(price, size)));
                BarUpdate::Closed { bar_end_ms }
            }
            None => {
                self.forming = Some((bucket, MarketBar::from_trade(price, size)));
                BarUpdate::Extended
            }
        }
    }

    /// Closed bars, oldest first.
    pub fn closed(&self) -> Vec<MarketBar> {
        self.closed.iter().copied().collect()
    }

    /// Closed bars plus the forming bar (a live preview of the current step).
    pub fn with_forming(&self) -> Vec<MarketBar> {
        let mut v = self.closed();
        if let Some((_, b)) = &self.forming {
            v.push(*b);
        }
        v
    }

    #[cfg(test)]
    pub fn closed_len(&self) -> usize {
        self.closed.len()
    }
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
        assert_eq!(s.with_forming().len(), 2);
        for k in 2..8 {
            assert!(matches!(s.on_trade(k * 60_000 + 1, 10.0, 1.0), BarUpdate::Closed { .. }));
        }
        assert_eq!(s.closed_len(), 3);
        // a late trade folds into the forming bar instead of reopening the past
        assert_eq!(s.on_trade(1_000, 9.0, 1.0), BarUpdate::Extended);
        assert_eq!(s.with_forming().last().unwrap().low, 9.0);
    }
}
