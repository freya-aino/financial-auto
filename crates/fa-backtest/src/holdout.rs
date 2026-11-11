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
//! selection window that would intersect a freshly locked holdout (or end inside the embargo
//! gap before it) is shifted to end one embargo before the holdout starts ([`avoid_holdouts`]),
//! which is the spec layout `[selection][embargo][holdout][live]`, instead of failing every
//! evaluation until the window has moved past the holdout.
//!
//! **Embargo rule** ([`selection_embargo_ms`], spec 41 §6.4/§7): the gap before a holdout is
//! `max(embargo.<timescale>, 2 × strategy horizon, 60 s)`. `embargo.<timescale>` is the
//! locker's configured floor (risk-engine `[holdout.embargo_secs]`), recorded on the lock
//! itself (`LockPlan::Lock { embargo_ms }` → `validation.holdout.embargo_ms`) so every reader
//! applies the value the holdout was locked with; rows without one use
//! [`HoldoutPolicy::default_for`]. `2 × horizon` is the per-candidate term (spec 41 §7 default:
//! a decision's outcome is realised one horizon later, twice that keeps labels and serial
//! correlation clear of the holdout; spec 36 §7 asks for ≥ the horizon). The larger wins: a
//! long-horizon strategy needs more than the timescale floor, a short one never less.

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
    /// Lock `[start_ms, end_ms)` (after checking its data coverage), recording the policy's
    /// `embargo_ms` with the row so selection windows keep that gap before it.
    Lock { start_ms: i64, end_ms: i64, relock: bool, embargo_ms: i64 },
    /// Nothing can be locked yet (reason).
    Wait(String),
}

/// Plans the holdout of `ts` at `now_ms` given every locked row (any timescale).
pub fn plan_lock(ts: Timescale, policy: &HoldoutPolicy, now_ms: i64, existing: &[Holdout]) -> LockPlan {
    let bar = ts.duration().as_millis().max(1) as i64;
    let length = policy.length_ms.max(bar);
    let end = now_ms.div_euclid(bar) * bar;
    let latest = existing.iter().filter(|h| h.timescale == ts).max_by_key(|h| h.end_ms);
    let embargo_ms = policy.embargo_ms.max(0);
    match latest {
        None => LockPlan::Lock { start_ms: end - length, end_ms: end, relock: false, embargo_ms },
        Some(h) => match policy.relock_after_ms {
            Some(age) if now_ms - h.end_ms >= age => {
                // the new segment never overlaps the old one
                let start = (end - length).max(h.end_ms);
                if end - start < bar {
                    LockPlan::Wait(format!("re-lock of {ts} due but no completed bar after {}", h.end_ms))
                } else {
                    LockPlan::Lock { start_ms: start, end_ms: end, relock: true, embargo_ms }
                }
            }
            _ => LockPlan::Locked(h.clone()),
        },
    }
}

/// Floor of every embargo: one minute.
pub const MIN_EMBARGO_MS: i64 = 60_000;

/// Gap a selection window of a strategy with horizon `horizon_ms` keeps before holdout `h`:
/// `max(h.embargo_ms (else the timescale default), 2 × horizon_ms, 60 s)` (module docs).
pub fn selection_embargo_ms(h: &Holdout, horizon_ms: i64) -> i64 {
    let configured = h.embargo_ms.unwrap_or_else(|| HoldoutPolicy::default_for(h.timescale).embargo_ms);
    configured.max(horizon_ms.saturating_mul(2)).max(MIN_EMBARGO_MS)
}

/// Shifts a selection window `(start, end)` of `ts` so it intersects no holdout of `ts` nor the
/// embargo gap before one: the window keeps its span and ends [`selection_embargo_ms`] before
/// the start of the holdout it would hit (repeated while it hits an earlier one). Windows that
/// touch nothing are returned as is.
pub fn avoid_holdouts(window: (i64, i64), ts: Timescale, holdouts: &[Holdout], horizon_ms: i64) -> (i64, i64) {
    let (mut s, mut e) = window;
    let span = (e - s).max(0);
    let mut hs: Vec<(&Holdout, i64)> =
        holdouts.iter().filter(|h| h.timescale == ts).map(|h| (h, selection_embargo_ms(h, horizon_ms))).collect();
    hs.sort_by_key(|(h, _)| std::cmp::Reverse(h.start_ms));
    // at most one pass per holdout, newest first
    for _ in 0..=hs.len() {
        let Some((h, emb)) = hs.iter().find(|(h, emb)| s <= h.end_ms && e >= h.start_ms - emb) else {
            break;
        };
        e = h.start_ms - emb - 1;
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
        Holdout { timescale: ts, start_ms: s, end_ms: e, locked_at_ms: e, embargo_ms: Some(0) }
    }

    #[test]
    fn first_lock_is_the_latest_completed_segment_and_relocking_is_idempotent() {
        let p = HoldoutPolicy { length_ms: 2 * H, embargo_ms: 0, min_bars: 1, relock_after_ms: None };
        let now = 100 * H + 1_234;
        let plan = plan_lock(Timescale::Second, &p, now, &[]);
        assert_eq!(
            plan,
            LockPlan::Lock { start_ms: 98 * H + 1_000, end_ms: 100 * H + 1_000, relock: false, embargo_ms: 0 }
        );
        let plan = plan_lock(Timescale::Minute, &p, now, &[]);
        assert_eq!(
            plan,
            LockPlan::Lock { start_ms: 98 * H, end_ms: 100 * H, relock: false, embargo_ms: 0 },
            "minute grid"
        );
        // the lock carries the policy's embargo
        let p5 = HoldoutPolicy { embargo_ms: 300_000, ..p };
        assert!(matches!(plan_lock(Timescale::Second, &p5, now, &[]), LockPlan::Lock { embargo_ms: 300_000, .. }));
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
            LockPlan::Lock { start_ms: 122 * H, end_ms: 124 * H, relock: true, embargo_ms: 0 }
        );
        // length longer than the time since the old end → starts exactly at the old end
        let p = HoldoutPolicy { length_ms: 48 * H, ..p };
        assert_eq!(
            plan_lock(Timescale::Second, &p, 124 * H, &old),
            LockPlan::Lock { start_ms: 100 * H, end_ms: 124 * H, relock: true, embargo_ms: 0 }
        );
    }

    #[test]
    fn selection_windows_move_before_the_holdout_with_an_embargo() {
        let mut h1 = h(Timescale::Second, 10 * H, 12 * H);
        h1.embargo_ms = Some(5 * 60_000);
        let hs = vec![h1.clone(), h(Timescale::Minute, 0, 100 * H)];
        // window [11h, 13h] hits the 1s holdout → ends 5 min (the recorded embargo, larger than
        // 2 × a 3 s horizon) before 10h, same 2 h span
        let emb = 5 * 60_000;
        let w = avoid_holdouts((11 * H, 13 * H), Timescale::Second, &hs, 3_000);
        assert_eq!(w, (10 * H - emb - 1 - 2 * H, 10 * H - emb - 1));
        assert!(!intersects(w, Timescale::Second, &hs));
        // no intersection → unchanged; other timescale's holdout is ignored
        assert_eq!(avoid_holdouts((13 * H, 15 * H), Timescale::Second, &hs, 3_000), (13 * H, 15 * H));
        // a window ending inside the embargo gap moves too
        let w = avoid_holdouts((8 * H, 10 * H - 60_000), Timescale::Second, &hs, 3_000);
        assert_eq!(w.1, 10 * H - emb - 1);
        // a long horizon outgrows the recorded embargo: 2 × 10 min = 20 min
        let w = avoid_holdouts((11 * H, 13 * H), Timescale::Second, &hs, 10 * 60_000);
        assert_eq!(w.1, 10 * H - 20 * 60_000 - 1);
        // chained holdouts: shifting before the first lands in the second, shift again
        let chain = vec![h(Timescale::Second, 10 * H, 12 * H), h(Timescale::Second, 7 * H, 9 * H)];
        let w = avoid_holdouts((11 * H, 12 * H), Timescale::Second, &chain, 0);
        assert!(!intersects(w, Timescale::Second, &chain), "{w:?}");
        assert_eq!(w.1 - w.0, H);
        assert!(w.1 < 7 * H - MIN_EMBARGO_MS);
    }

    #[test]
    fn selection_embargo_is_the_larger_of_the_lock_and_twice_the_horizon() {
        let mut x = h(Timescale::Minute, 0, H);
        x.embargo_ms = Some(H);
        assert_eq!(selection_embargo_ms(&x, 60_000), H, "configured 1 h beats 2 × 1 min");
        assert_eq!(selection_embargo_ms(&x, 45 * 60_000), 90 * 60_000, "2 × 45 min beats 1 h");
        x.embargo_ms = Some(0);
        assert_eq!(selection_embargo_ms(&x, 1_000), MIN_EMBARGO_MS, "never below a minute");
        // a row without a recorded embargo (manual lock, older row) uses the timescale default
        x.embargo_ms = None;
        assert_eq!(selection_embargo_ms(&x, 1_000), HoldoutPolicy::default_for(Timescale::Minute).embargo_ms);
    }

    #[test]
    fn defaults_scale_with_the_timescale() {
        let s = HoldoutPolicy::default_for(Timescale::Second);
        let d = HoldoutPolicy::default_for(Timescale::Day);
        assert!(s.length_ms < d.length_ms && s.embargo_ms < d.embargo_ms);
        assert_eq!(s.relock_after_ms, None);
    }
}
