//! Locked-holdout planning (spec 41 §6.4) and holdout-aware selection windows.
//!
//! The holdout of a timescale is the most recent completed segment at lock time, then fixed
//! (`validation.holdout` is immutable by trigger). [`plan_lock`] decides, for one timescale,
//! whether a holdout already covers it (idempotent: nothing to do), whether a new segment
//! should be locked (none yet, or the optional re-lock age has passed — a re-lock is a new,
//! non-overlapping row), and which segment: `[end − length, end)` with `end` = now rounded
//! down to the bar grid. The caller checks the segment holds enough data
//! ([`HoldoutPolicy::min_bars`]) before inserting it, so an empty holdout is never locked and a
//! missing one is reported instead of silently absent.
//!
//! Selection replays must not touch a holdout (`replay::check_holdout`). A now-relative
//! selection window that would intersect a freshly locked holdout is shifted to end one
//! embargo before the holdout starts ([`avoid_holdouts`]), which is the spec layout
//! `[selection][embargo][holdout][live]`, instead of failing every evaluation until the window
//! has moved past the holdout.

use fa_core::Timescale;
use serde::{Deserialize, Serialize};

use crate::replay::Holdout;

/// Locking policy of one timescale (spec 41 §7 `holdout.length.<timescale>`, `embargo.<timescale>`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HoldoutPolicy {
    pub length_ms: i64,
    /// Gap between a selection window and the holdout it precedes.
    pub embargo_ms: i64,
    /// Minimum bars of the timescale the segment must contain on its best-covered instrument.
    pub min_bars: usize,
    /// Re-lock (a new segment after the latest one) once the latest holdout ended this long ago;
    /// `None` = never (spec 41 Q5 is open; a re-lock invalidates gate results on the old one).
    pub relock_after_ms: Option<i64>,
}

impl HoldoutPolicy {
    /// Spec 41 §7 placeholders, scaled down for timescales whose history is still short: the
    /// caller may override every field from config.
    pub fn default_for(ts: Timescale) -> Self {
        const H: i64 = 3_600_000;
        const D: i64 = 24 * H;
        let (length_ms, embargo_ms) = match ts {
            Timescale::Second => (2 * H, 5 * 60_000),
            Timescale::Minute => (12 * H, H),
            Timescale::Hour => (90 * D, 2 * D),
            Timescale::Day => (365 * D, 10 * D),
            Timescale::Week => (3 * 365 * D, 70 * D),
        };
        Self { length_ms, embargo_ms, min_bars: 300, relock_after_ms: None }
    }
}

/// What [`plan_lock`] decided for one timescale.
#[derive(Debug, Clone, PartialEq)]
pub enum LockPlan {
    /// A holdout of the timescale exists and needs no re-lock.
    Locked(Holdout),
    /// Lock `[start_ms, end_ms)` (after checking its data coverage).
    Lock { start_ms: i64, end_ms: i64, relock: bool },
    /// Nothing can be locked yet (reason).
    Wait(String),
}

/// Plans the holdout of `ts` at `now_ms` given every locked row (any timescale).
pub fn plan_lock(ts: Timescale, policy: &HoldoutPolicy, now_ms: i64, existing: &[Holdout]) -> LockPlan {
    let bar = ts.duration().as_millis().max(1) as i64;
    let length = policy.length_ms.max(bar);
    let end = now_ms.div_euclid(bar) * bar;
    let latest = existing.iter().filter(|h| h.timescale == ts).max_by_key(|h| h.end_ms);
    match latest {
        None => LockPlan::Lock { start_ms: end - length, end_ms: end, relock: false },
        Some(h) => match policy.relock_after_ms {
            Some(age) if now_ms - h.end_ms >= age => {
                // the new segment never overlaps the old one
                let start = (end - length).max(h.end_ms);
                if end - start < bar {
                    LockPlan::Wait(format!("re-lock of {ts} due but no completed bar after {}", h.end_ms))
                } else {
                    LockPlan::Lock { start_ms: start, end_ms: end, relock: true }
                }
            }
            _ => LockPlan::Locked(h.clone()),
        },
    }
}

/// Shifts a selection window `(start, end)` of `ts` so it intersects no holdout of `ts`: the
/// window keeps its span and ends `embargo_ms` before the start of the holdout it would hit
/// (repeated while it hits an earlier one). Windows that intersect nothing are returned as is.
pub fn avoid_holdouts(window: (i64, i64), ts: Timescale, holdouts: &[Holdout], embargo_ms: i64) -> (i64, i64) {
    let (mut s, mut e) = window;
    let span = (e - s).max(0);
    let mut hs: Vec<&Holdout> = holdouts.iter().filter(|h| h.timescale == ts).collect();
    hs.sort_by_key(|h| std::cmp::Reverse(h.start_ms));
    // at most one pass per holdout, newest first
    for _ in 0..=hs.len() {
        let Some(h) = hs.iter().find(|h| s <= h.end_ms && e >= h.start_ms) else {
            break;
        };
        e = h.start_ms - embargo_ms.max(0) - 1;
        s = e - span;
    }
    (s, e)
}

/// Whether `(start, end)` intersects a holdout of `ts` (the `check_holdout` rule).
pub fn intersects(window: (i64, i64), ts: Timescale, holdouts: &[Holdout]) -> bool {
    holdouts.iter().any(|h| h.timescale == ts && window.0 <= h.end_ms && window.1 >= h.start_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: i64 = 3_600_000;

    fn h(ts: Timescale, s: i64, e: i64) -> Holdout {
        Holdout { timescale: ts, start_ms: s, end_ms: e, locked_at_ms: e }
    }

    #[test]
    fn first_lock_is_the_latest_completed_segment_and_relocking_is_idempotent() {
        let p = HoldoutPolicy { length_ms: 2 * H, embargo_ms: 0, min_bars: 1, relock_after_ms: None };
        let now = 100 * H + 1_234;
        let plan = plan_lock(Timescale::Second, &p, now, &[]);
        assert_eq!(plan, LockPlan::Lock { start_ms: 98 * H + 1_000, end_ms: 100 * H + 1_000, relock: false });
        let plan = plan_lock(Timescale::Minute, &p, now, &[]);
        assert_eq!(plan, LockPlan::Lock { start_ms: 98 * H, end_ms: 100 * H, relock: false }, "minute grid");
        // a locked row of the timescale → nothing to do, however often it is asked
        let locked = vec![h(Timescale::Second, 98 * H, 100 * H), h(Timescale::Minute, 0, H)];
        for t in [now, now + 50 * H, now + 5_000 * H] {
            assert_eq!(plan_lock(Timescale::Second, &p, t, &locked), LockPlan::Locked(locked[0].clone()));
        }
        // other timescales are independent
        assert!(matches!(plan_lock(Timescale::Hour, &p, now, &locked), LockPlan::Lock { .. }));
    }

    #[test]
    fn relock_after_age_never_overlaps_the_previous_segment() {
        let p = HoldoutPolicy { length_ms: 2 * H, embargo_ms: 0, min_bars: 1, relock_after_ms: Some(24 * H) };
        let old = vec![h(Timescale::Second, 98 * H, 100 * H)];
        assert!(matches!(plan_lock(Timescale::Second, &p, 110 * H, &old), LockPlan::Locked(_)));
        assert_eq!(
            plan_lock(Timescale::Second, &p, 124 * H, &old),
            LockPlan::Lock { start_ms: 122 * H, end_ms: 124 * H, relock: true }
        );
        // length longer than the time since the old end → starts exactly at the old end
        let p = HoldoutPolicy { length_ms: 48 * H, ..p };
        assert_eq!(
            plan_lock(Timescale::Second, &p, 124 * H, &old),
            LockPlan::Lock { start_ms: 100 * H, end_ms: 124 * H, relock: true }
        );
    }

    #[test]
    fn selection_windows_move_before_the_holdout_with_an_embargo() {
        let hs = vec![h(Timescale::Second, 10 * H, 12 * H), h(Timescale::Minute, 0, 100 * H)];
        // window [11h, 13h] hits the 1s holdout → ends 5 min before 10h, same 2 h span
        let emb = 5 * 60_000;
        let w = avoid_holdouts((11 * H, 13 * H), Timescale::Second, &hs, emb);
        assert_eq!(w, (10 * H - emb - 1 - 2 * H, 10 * H - emb - 1));
        assert!(!intersects(w, Timescale::Second, &hs));
        // no intersection → unchanged; other timescale's holdout is ignored
        assert_eq!(avoid_holdouts((13 * H, 15 * H), Timescale::Second, &hs, emb), (13 * H, 15 * H));
        // chained holdouts: shifting before the first lands in the second, shift again
        let chain = vec![h(Timescale::Second, 10 * H, 12 * H), h(Timescale::Second, 7 * H, 9 * H)];
        let w = avoid_holdouts((11 * H, 12 * H), Timescale::Second, &chain, 0);
        assert!(!intersects(w, Timescale::Second, &chain), "{w:?}");
        assert_eq!(w.1 - w.0, H);
        assert!(w.1 < 7 * H);
    }

    #[test]
    fn defaults_scale_with_the_timescale() {
        let s = HoldoutPolicy::default_for(Timescale::Second);
        let d = HoldoutPolicy::default_for(Timescale::Day);
        assert!(s.length_ms < d.length_ms && s.embargo_ms < d.embargo_ms);
        assert_eq!(s.relock_after_ms, None);
    }
}
