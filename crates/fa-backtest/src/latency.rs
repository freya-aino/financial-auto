//! Order latency in strategy replays (spec 41 §10 Q3: model order latency per venue, or assume
//! next-event execution?).
//!
//! **Decision:** a per-venue latency, on by default. A strategy decides at a bar close `t`; its
//! order is filled at the first price the venue knows at `t + latency(venue)` — the last close of
//! the (knowledge-shifted) series at or before that time — while the order size stays the one
//! decided at `t`. The difference between decision and execution price lands in the strategy's
//! P&L on the next mark (and is reported per trade as `exec_price`). With bars, a latency below
//! the bar length changes nothing (no intrabar price is known), so the model matters for the
//! 1 s timescale and is exact under the identity of `venue.rs` for coarser ones.
//!
//! **Values (measured on the deployed path):** a strategy signal reaches the venue through the
//! risk-engine's 1 s wall-clock decision grid (it is netted at the next whole second) and the
//! gateway: on the retained `ORDERS` stream (2026-11-10 19:48 – 11-11 15:00 UTC, 7 267 market
//! orders on 8 instruments of the `kraken`, `yahoo` and `sim` feeds) intent → fill took p50 1 ms,
//! p99 4 ms, max 19 ms, on every venue alike — the paper venue fills against its last trade.
//! The deciding term is the grid: a decision at a bar close `t` is filled at the venue's last
//! price at the next grid second, i.e. the close of the bar ending at `t + 1 s`, so every venue
//! gets [`DEFAULT_LATENCY_MS`]; the cost-aware rule's `confirm_secs` batching is modelled
//! separately by the trade rule. Venues and their latencies
//! can be overridden per replay (`ReplaySpec::latency`); [`LatencyModel::zero`] reproduces the
//! zero-latency fills of trials scored before the model existed (not comparable with newer
//! trials on the 1 s timescale).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Decision → fill latency of a paper venue (ms): one risk-engine decision period.
pub const DEFAULT_LATENCY_MS: i64 = 1_000;

/// Venue → order latency (ms).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatencyModel {
    #[serde(default)]
    pub by_venue: BTreeMap<String, i64>,
    pub default_ms: i64,
}

impl Default for LatencyModel {
    fn default() -> Self {
        Self::builtin()
    }
}

impl LatencyModel {
    /// Fills at the deciding close (pre-model trials).
    pub fn zero() -> Self {
        Self { by_venue: BTreeMap::new(), default_ms: 0 }
    }

    /// The measured deployed path: one decision period for every venue.
    pub fn builtin() -> Self {
        Self { by_venue: BTreeMap::new(), default_ms: DEFAULT_LATENCY_MS }
    }

    pub fn of(&self, venue: &str) -> i64 {
        self.by_venue.get(venue).copied().unwrap_or(self.default_ms).max(0)
    }

    pub fn is_zero(&self) -> bool {
        self.default_ms <= 0 && self.by_venue.values().all(|v| *v <= 0)
    }
}

/// Last price at or before `at_ms` in ascending `(ts, px)`; `None` before the first.
pub fn price_at(ts: &[i64], px: &[f64], at_ms: i64) -> Option<f64> {
    let n = ts.partition_point(|t| *t <= at_ms);
    (n > 0).then(|| px[n - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn venue_lookup_and_price_at() {
        let mut m = LatencyModel::builtin();
        assert_eq!(m.of("kraken"), DEFAULT_LATENCY_MS);
        m.by_venue.insert("sim".into(), 0);
        assert_eq!(m.of("sim"), 0);
        assert!(!m.is_zero() && LatencyModel::zero().is_zero());
        let (ts, px) = ([1000, 2000, 3000], [1.0, 2.0, 3.0]);
        assert_eq!(price_at(&ts, &px, 999), None);
        assert_eq!(price_at(&ts, &px, 2000), Some(2.0));
        assert_eq!(price_at(&ts, &px, 2999), Some(2.0));
        assert_eq!(price_at(&ts, &px, 10_000), Some(3.0));
        // serde: an absent field means the builtin model
        let m: LatencyModel = serde_json::from_str(r#"{"default_ms": 250}"#).unwrap();
        assert_eq!(m.of("x"), 250);
    }
}
