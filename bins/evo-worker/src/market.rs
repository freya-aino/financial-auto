//! Bounded in-memory trade buffer fed from `md.trade.>` (stopgap evaluation data).
//!
//! The real evaluation path is VictoriaMetrics via `fa-tsdb` (spec 41 §5), which nobody
//! populates yet. Until then the worker keeps the most recent trades per instrument in RAM;
//! a freshly started worker therefore has to listen for a while before it can score
//! anything, and every worker sees a slightly different window.

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

use fa_bus::{Bus, subjects};
use fa_proto::v1::{Envelope, envelope::Payload};
use futures::StreamExt;

/// Trades kept per instrument.
pub const MAX_TRADES_PER_INSTRUMENT: usize = 50_000;
/// Instruments tracked (oldest-updated evicted first).
pub const MAX_INSTRUMENTS: usize = 64;

#[derive(Debug, Default)]
struct InstrumentBuffer {
    venue: String,
    /// `(event_time_ms, price)` in arrival order.
    trades: VecDeque<(i64, f64)>,
    last_ms: i64,
}

#[derive(Clone, Default)]
pub struct MarketBuffer {
    inner: Arc<Mutex<HashMap<String, InstrumentBuffer>>>,
}

/// Snapshot of one instrument's trades.
#[derive(Debug, Clone)]
pub struct InstrumentTrades {
    pub instrument_id: String,
    pub venue: String,
    pub trades: Vec<(i64, f64)>,
}

impl MarketBuffer {
    pub fn push(&self, instrument_id: &str, venue: &str, ts_ms: i64, price: f64) {
        if instrument_id.is_empty() || price.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return;
        }
        let mut map = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if !map.contains_key(instrument_id)
            && map.len() >= MAX_INSTRUMENTS
            && let Some(oldest) = map.iter().min_by_key(|(_, b)| b.last_ms).map(|(k, _)| k.clone())
        {
            map.remove(&oldest);
        }
        let b = map.entry(instrument_id.to_owned()).or_default();
        if b.venue.is_empty() {
            b.venue = venue.to_owned();
        }
        b.trades.push_back((ts_ms, price));
        b.last_ms = b.last_ms.max(ts_ms);
        while b.trades.len() > MAX_TRADES_PER_INSTRUMENT {
            b.trades.pop_front();
        }
    }

    /// All instruments (or only `filter`), with their trades in time order.
    pub fn snapshot(&self, filter: Option<&str>) -> Vec<InstrumentTrades> {
        let map = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut out: Vec<InstrumentTrades> = map
            .iter()
            .filter(|(id, _)| {
                filter.is_none_or(|f| f.eq_ignore_ascii_case(id) || f.replace('-', "") == id.replace('-', ""))
            })
            .map(|(id, b)| {
                let mut trades: Vec<(i64, f64)> = b.trades.iter().copied().collect();
                trades.sort_by_key(|t| t.0);
                InstrumentTrades { instrument_id: id.clone(), venue: b.venue.clone(), trades }
            })
            .collect();
        out.sort_by(|a, b| b.trades.len().cmp(&a.trades.len()).then_with(|| a.instrument_id.cmp(&b.instrument_id)));
        out
    }

    /// `(instruments, total trades)`.
    pub fn stats(&self) -> (usize, usize) {
        let map = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        (map.len(), map.values().map(|b| b.trades.len()).sum())
    }

    /// Consumes `md.trade.>` forever.
    pub async fn run(self, bus: Bus) {
        let mut sub = match bus.subscribe::<Envelope>(subjects::ALL_MD_TRADES.into()).await {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "cannot subscribe to {}", subjects::ALL_MD_TRADES);
                return;
            }
        };
        while let Some((subject, env)) = sub.next().await {
            let Some(Payload::Trade(t)) = env.payload else {
                continue;
            };
            let venue = subject.split('.').nth(2).unwrap_or("").to_owned();
            let ts = env
                .event_time
                .or(env.publish_time)
                .or(env.ingest_time)
                .map(|t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000)
                .unwrap_or_else(now_ms);
            self.push(&env.instrument_id, &venue, ts, t.price);
        }
        tracing::warn!("md.trade subscription ended");
    }
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_and_sorted() {
        let b = MarketBuffer::default();
        for i in 0..(MAX_TRADES_PER_INSTRUMENT + 10) {
            b.push("a", "sim", i as i64, 1.0 + i as f64);
        }
        b.push("b", "sim", 5, 2.0);
        b.push("b", "sim", 3, 1.0);
        b.push("", "sim", 1, 1.0);
        b.push("c", "sim", 1, -1.0);
        let snap = b.snapshot(None);
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].trades.len(), MAX_TRADES_PER_INSTRUMENT);
        assert_eq!(snap[1].trades, vec![(3, 1.0), (5, 2.0)]);
        assert_eq!(b.snapshot(Some("B")).len(), 1);
        assert_eq!(b.stats(), (2, MAX_TRADES_PER_INSTRUMENT + 2));
    }
}
