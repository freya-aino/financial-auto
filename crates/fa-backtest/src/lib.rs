//! Backtest & validation statistics (spec 41): deflated Sharpe ratio, PBO via CSCV,
//! and the replay driver interface (replay runs through the live code path).

pub mod stats;

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
