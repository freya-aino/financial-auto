//! Per-instrument feed health (spec 40 §6.5 "health halts", §8 "price feed spike / bad tick"):
//!
//! - **Stale feed**: an instrument without a trade for longer than `stale_freeze_bars` × the
//!   larger of the fastest strategy bar trading it and the instrument's own typical gap between
//!   trades (EWMA, so a minute-bar feed is not "stale" between its bars) is **frozen** — reduce
//!   only: no new risk, the position is kept or reduced; past `stale_flatten_secs` (0 = never)
//!   it is flattened.
//! - **Price jump**: a trade more than `jump_max` (log return) away from the last accepted mark
//!   is held back instead of becoming the mark; the instrument is frozen while the jump is
//!   unconfirmed. `jump_confirm` consecutive trades near the new level (within half the jump
//!   threshold of each other) confirm it and it becomes the mark; a trade back near the old
//!   mark discards it as a bad tick.

use std::collections::BTreeMap;

use fa_core::InstrumentId;
use serde::{Deserialize, Serialize};

/// `[feed]` in `risk.toml` (spec 40 §7 `risk.feed_stale_freeze`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FeedGuardConfig {
    /// Freeze after this many bars of the instrument's fastest strategy timescale (or of its own
    /// typical trade gap, whichever is longer) without a trade; 0 disables the freeze.
    pub stale_freeze_bars: f64,
    /// Floor of the freeze threshold in seconds (slow feeds, e.g. minute bars).
    pub stale_freeze_min_secs: f64,
    /// Flatten after this many seconds without a trade (0 = keep the position, never flatten).
    pub stale_flatten_secs: f64,
    /// Log-return jump that is held back as a possible bad tick; 0 disables the filter.
    pub jump_max: f64,
    /// Trades near the new level that confirm a jump.
    pub jump_confirm: u32,
}

impl Default for FeedGuardConfig {
    fn default() -> Self {
        Self {
            stale_freeze_bars: 3.0,
            stale_freeze_min_secs: 3.0,
            stale_flatten_secs: 0.0,
            jump_max: 0.15,
            jump_confirm: 3,
        }
    }
}

/// Verdict on one trade.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeVerdict {
    /// Use the price as the mark.
    Accept,
    /// A confirmed jump: use the price as the mark (it was held back before).
    Confirmed,
    /// Held back as a possible bad tick.
    Held,
    /// A held-back jump was discarded (price returned to the old level); this price is accepted.
    Discarded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FeedState {
    Fresh,
    /// No new orders; positions kept.
    Frozen,
    /// Flatten the instrument.
    Flatten,
}

#[derive(Debug, Clone, Default)]
struct Pending {
    price: f64,
    count: u32,
}

/// Tracks last trade times and unconfirmed jumps per instrument.
#[derive(Debug, Clone, Default)]
pub struct FeedGuard {
    pub cfg: FeedGuardConfig,
    last_trade_ms: BTreeMap<InstrumentId, i64>,
    /// EWMA of the gap between consecutive trades (ms).
    gap_ms: BTreeMap<InstrumentId, f64>,
    pending: BTreeMap<InstrumentId, Pending>,
}

impl FeedGuard {
    pub fn new(cfg: FeedGuardConfig) -> Self {
        Self { cfg, ..Default::default() }
    }

    /// One trade at `price` given the current accepted mark (same currency).
    pub fn on_trade(&mut self, inst: InstrumentId, at_ms: i64, price: f64, mark: Option<f64>) -> TradeVerdict {
        let last = self.last_trade_ms.entry(inst).or_insert(at_ms);
        let gap = at_ms - *last;
        *last = (*last).max(at_ms);
        if gap > 0 {
            let g = self.gap_ms.entry(inst).or_insert(gap as f64);
            *g = 0.9 * *g + 0.1 * gap as f64;
        }
        let Some(mark) = mark.filter(|m| *m > 0.0) else {
            self.pending.remove(&inst);
            return TradeVerdict::Accept;
        };
        let jump = self.cfg.jump_max;
        if jump <= 0.0 {
            return TradeVerdict::Accept;
        }
        let r = (price / mark).ln().abs();
        if let Some(p) = self.pending.get_mut(&inst) {
            if (price / p.price).ln().abs() <= jump / 2.0 {
                p.count += 1;
                p.price = price;
                if p.count >= self.cfg.jump_confirm.max(1) {
                    self.pending.remove(&inst);
                    return TradeVerdict::Confirmed;
                }
                return TradeVerdict::Held;
            }
            if r <= jump {
                self.pending.remove(&inst);
                return TradeVerdict::Discarded;
            }
            // yet another level: restart the confirmation there
            *p = Pending { price, count: 1 };
            return TradeVerdict::Held;
        }
        if r > jump {
            if self.cfg.jump_confirm <= 1 {
                return TradeVerdict::Confirmed;
            }
            self.pending.insert(inst, Pending { price, count: 1 });
            return TradeVerdict::Held;
        }
        TradeVerdict::Accept
    }

    /// Feed state of `inst` at `now_ms`, given its fastest strategy bar (ms; `None` = no
    /// strategy trades it, only the flatten threshold applies).
    pub fn state(&self, inst: &InstrumentId, now_ms: i64, bar_ms: Option<i64>) -> FeedState {
        let Some(last) = self.last_trade_ms.get(inst) else {
            return FeedState::Fresh; // never traded: the unknown-market rules apply instead
        };
        let age = (now_ms - last) as f64;
        if self.cfg.stale_flatten_secs > 0.0 && age > self.cfg.stale_flatten_secs * 1000.0 {
            return FeedState::Flatten;
        }
        if self.pending.contains_key(inst) {
            return FeedState::Frozen;
        }
        if let Some(bar) = bar_ms.filter(|_| self.cfg.stale_freeze_bars > 0.0) {
            let cadence = (bar as f64).max(self.gap_ms.get(inst).copied().unwrap_or(0.0));
            let freeze = (self.cfg.stale_freeze_bars * cadence).max(self.cfg.stale_freeze_min_secs * 1000.0);
            if age > freeze {
                return FeedState::Frozen;
            }
        }
        FeedState::Fresh
    }

    /// Instruments with an unconfirmed jump.
    pub fn held(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jumps_are_held_until_confirmed_or_discarded() {
        let i = InstrumentId::new();
        let mut g = FeedGuard::new(FeedGuardConfig { jump_max: 0.1, jump_confirm: 3, ..Default::default() });
        assert_eq!(g.on_trade(i, 0, 100.0, None), TradeVerdict::Accept);
        assert_eq!(g.on_trade(i, 1, 101.0, Some(100.0)), TradeVerdict::Accept);
        // a bad tick, then back to normal
        assert_eq!(g.on_trade(i, 2, 150.0, Some(101.0)), TradeVerdict::Held);
        assert_eq!(g.state(&i, 2, Some(1000)), FeedState::Frozen);
        assert_eq!(g.on_trade(i, 3, 101.5, Some(101.0)), TradeVerdict::Discarded);
        assert_eq!(g.state(&i, 3, Some(1000)), FeedState::Fresh);
        // a real move: three trades at the new level confirm it
        assert_eq!(g.on_trade(i, 4, 130.0, Some(101.5)), TradeVerdict::Held);
        assert_eq!(g.on_trade(i, 5, 131.0, Some(101.5)), TradeVerdict::Held);
        assert_eq!(g.on_trade(i, 6, 130.5, Some(101.5)), TradeVerdict::Confirmed);
        assert_eq!(g.held(), 0);
    }

    #[test]
    fn stale_feeds_freeze_then_optionally_flatten() {
        let i = InstrumentId::new();
        let mut g = FeedGuard::new(FeedGuardConfig {
            stale_freeze_bars: 3.0,
            stale_freeze_min_secs: 3.0,
            stale_flatten_secs: 60.0,
            ..Default::default()
        });
        g.on_trade(i, 0, 100.0, None);
        assert_eq!(g.state(&i, 2_000, Some(1_000)), FeedState::Fresh);
        assert_eq!(g.state(&i, 3_500, Some(1_000)), FeedState::Frozen);
        assert_eq!(g.state(&i, 3_500, Some(60_000)), FeedState::Fresh, "minute strategies tolerate 3 min");
        assert_eq!(g.state(&i, 3_500, None), FeedState::Fresh, "nobody trades it");
        assert_eq!(g.state(&i, 61_000, None), FeedState::Flatten);
        assert_eq!(g.state(&InstrumentId::new(), 61_000, Some(1_000)), FeedState::Fresh, "never seen");
        let keep = FeedGuard { cfg: FeedGuardConfig::default(), ..g.clone() };
        assert_eq!(keep.state(&i, 10_000_000, Some(1_000)), FeedState::Frozen, "default never flattens");
        // a feed that prints once a minute is not stale for a 1 s strategy between prints
        let j = InstrumentId::new();
        let mut m = FeedGuard::new(FeedGuardConfig::default());
        for k in 0..20 {
            m.on_trade(j, k * 60_000, 100.0, Some(100.0));
        }
        assert_eq!(m.state(&j, 19 * 60_000 + 59_000, Some(1_000)), FeedState::Fresh);
        assert_eq!(m.state(&j, 19 * 60_000 + 200_000, Some(1_000)), FeedState::Frozen);
    }
}
