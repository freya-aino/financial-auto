//! Backtest & validation (spec 41): deflated Sharpe ratio, PBO via CSCV, the replay driver
//! interface (replay runs through the live code path), [`eval`], the sequential
//! point-in-time scorer for series decoders, [`replay`], the pipeline replay of
//! `fa_strategies` programs with execution simulation through the [`cost`] model, and
//! [`validation`], the per-generation statistics over a trial set (deflated Sharpe, PBO), and
//! [`allocator_replay`], the portfolio replay of recorded strategy signals through one
//! allocator genome (allocator tournament, spec 40 §6.2), and [`holdout`], the locked-holdout
//! planner and holdout-aware selection windows (spec 41 §6.4), and [`knowledge`], the per-source
//! knowledge delays every event-time replay applies (spec 41 §6.2), and [`venue`], the paper
//! venue state ported from the broker-gateway so recorded live fills replay identically, and
//! [`trade_rule`], the live risk-engine's rebalance rule both fitness replays trade with, and
//! [`coverage`] (missing-data cut) and [`latency`] (per-venue order latency) of the strategy
//! replay.

pub mod allocator_replay;
pub mod cost;
pub mod coverage;
pub mod eval;
pub mod holdout;
pub mod knowledge;
pub mod latency;
pub mod latents;
pub mod replay;
pub mod stats;
pub mod trade_rule;
pub mod validation;
pub mod venue;

/// A replay source drives the same pipeline as live, in event-time order with
/// point-in-time (publish_time ≤ decision time) semantics.
pub trait ReplaySource {
    type Event;
    /// Next event by `publish_time`; `None` at end of range.
    fn next_event(&mut self) -> Option<(i64, Self::Event)>;
}

/// Consumer of replayed events (the live pipeline in replay mode).
pub trait ReplaySink<E> {
    fn on_event(&mut self, publish_time_ms: i64, event: E);
}

/// Drives a replay to completion; returns the number of events delivered.
/// Rejects out-of-order sources to prevent look-ahead.
pub fn run_replay<S: ReplaySource>(src: &mut S, sink: &mut impl ReplaySink<S::Event>) -> Result<u64, ReplayError> {
    let mut last = i64::MIN;
    let mut n = 0;
    while let Some((t, e)) = src.next_event() {
        if t < last {
            return Err(ReplayError::OutOfOrder { prev: last, got: t });
        }
        last = t;
        sink.on_event(t, e);
        n += 1;
    }
    Ok(n)
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ReplayError {
    #[error("replay source out of order: {got} after {prev}")]
    OutOfOrder { prev: i64, got: i64 },
    /// A `Selection` replay touched the locked holdout (spec 41 §6.4): rejected, not penalised.
    #[error("selection window intersects the locked {timescale} holdout [{holdout_start_ms}, {holdout_end_ms}]")]
    HoldoutIntersect { timescale: String, holdout_start_ms: i64, holdout_end_ms: i64 },
    #[error("no bars in the replay window")]
    NoData,
    /// Too much of the window is missing (spec 41 §8, [`coverage`]): no fitness.
    #[error("insufficient data coverage: window {window:.2}, best instrument {best_instrument:.2} < min {min:.2}")]
    Coverage { window: f64, best_instrument: f64, min: f64 },
    /// Recorded latents of another encoder snapshot than the one the replay declares (spec 41 §4
    /// `ReplaySpec.snapshot`); only `Diagnostic` replays may mix snapshots.
    #[error("latents of snapshot(s) `{found}` mixed into a replay declared on `{declared}`")]
    SnapshotMismatch { declared: String, found: String },
    #[error("candidate: {0}")]
    Candidate(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Src(Vec<(i64, u32)>);
    impl ReplaySource for Src {
        type Event = u32;
        fn next_event(&mut self) -> Option<(i64, u32)> {
            if self.0.is_empty() { None } else { Some(self.0.remove(0)) }
        }
    }
    struct Sink(Vec<u32>);
    impl ReplaySink<u32> for Sink {
        fn on_event(&mut self, _: i64, e: u32) {
            self.0.push(e);
        }
    }

    #[test]
    fn replay_in_order_and_rejects_lookahead() {
        let mut sink = Sink(vec![]);
        assert_eq!(run_replay(&mut Src(vec![(1, 1), (2, 2)]), &mut sink), Ok(2));
        assert_eq!(sink.0, vec![1, 2]);
        assert!(run_replay(&mut Src(vec![(2, 1), (1, 2)]), &mut sink).is_err());
    }
}
