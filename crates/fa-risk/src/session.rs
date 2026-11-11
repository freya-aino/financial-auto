//! The daily-loss "day" per trading calendar (spec 40 §6.5 "realised + unrealised P&L since
//! session start (UTC day for 24/7 venues, exchange day otherwise)", §10 Q4).
//!
//! Every instrument belongs to a **day group**: [`UTC_GROUP`] for 24/7 venues and anything
//! without a known calendar, else its exchange calendar (`us`, `xetr`, `fx`, …). A group's
//! trading day is an opaque key the caller derives from the calendar (the risk-engine: the date
//! of the next session close, so an exchange day runs from one close to the next and a
//! weekend or holiday belongs to the following session).
//!
//! The portfolio keeps **one** daily-loss measure, `1 − equity / day_start`, against one limit,
//! but `day_start` is no longer reset wholesale at UTC midnight:
//!
//! - instruments of a session group carry their own day: value at the group's last roll plus
//!   the cash flow of their fills since (so their day P&L is `value − value_at_roll + cash`);
//! - when a **session group rolls** (its exchange day ended), its finished day P&L is banked
//!   into `day_start` (`day_start += pnl_g`) and its instruments restart from their current
//!   value — the loss of a closed exchange day no longer counts, an open one keeps counting;
//! - when the **UTC day rolls**, `day_start = equity − Σ pnl of every session instrument`: the
//!   24/7 part (and book-level cash such as funding or payouts) restarts, the session days in
//!   progress carry on.
//!
//! With only 24/7 instruments this is exactly the former UTC-day rule. The caller clears the
//! daily-loss halt after a roll only when the recomputed loss is below the limit (the halt of
//! a session-day loss lasts until that exchange day ends).
use std::collections::BTreeMap;

use fa_core::InstrumentId;
use serde::{Deserialize, Serialize};

/// Day group of 24/7 venues, instruments without a calendar and book-level cash.
pub const UTC_GROUP: &str = "utc";

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct InstDay {
    group: String,
    /// EUR value of the position when the group's day began (or when tracking began).
    value_at_roll: f64,
    /// Σ cash flow of the instrument's fills since (EUR, negative = paid; incl. fees).
    cash_flow: f64,
}

/// One day roll.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Roll {
    pub group: String,
    pub from: i64,
    pub to: i64,
    /// Session groups: the finished day's P&L moved into `day_start`. UTC: the new `day_start`
    /// minus the old one.
    pub banked_eur: f64,
}

/// Day keys per group and the day P&L of every session instrument. Serialisable (persisted
/// with the risk state).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionDays {
    days: BTreeMap<String, i64>,
    insts: BTreeMap<InstrumentId, InstDay>,
}

impl SessionDays {
    /// Current day key per group.
    pub fn days(&self) -> &BTreeMap<String, i64> {
        &self.days
    }

    /// Instruments tracked on a session day, with their group.
    pub fn tracked(&self) -> impl Iterator<Item = (&InstrumentId, &str)> {
        self.insts.iter().map(|(i, d)| (i, d.group.as_str()))
    }

    /// A fill of `inst` in day group `group` during its trading day `day_key`:
    /// `value_before_eur` is the position's EUR value just before the fill (starts tracking an
    /// instrument first seen now), `cash_flow_eur` the fill's cash flow (−qty·price·fx − fee).
    /// UTC-group fills need nothing (they are part of the equity residual).
    pub fn on_fill(
        &mut self,
        inst: InstrumentId,
        group: &str,
        day_key: i64,
        value_before_eur: f64,
        cash_flow_eur: f64,
    ) {
        if group == UTC_GROUP || !cash_flow_eur.is_finite() {
            return;
        }
        self.days.entry(group.to_owned()).or_insert(day_key);
        let d = self.insts.entry(inst).or_insert_with(|| InstDay {
            group: group.to_owned(),
            value_at_roll: if value_before_eur.is_finite() { value_before_eur } else { 0.0 },
            cash_flow: 0.0,
        });
        d.cash_flow += cash_flow_eur;
    }

    /// Records `key` as the group's current day unless one is known (a restored state from
    /// before the calendars: the UTC day it last saw).
    pub fn ensure_day(&mut self, group: &str, key: i64) {
        self.days.entry(group.to_owned()).or_insert(key);
    }

    /// Day P&L of every session instrument (EUR) at `values` (an instrument without a value is
    /// valued at its roll value: no change known).
    pub fn session_pnl(&self, values: &BTreeMap<InstrumentId, f64>) -> f64 {
        self.insts.iter().map(|(i, d)| Self::pnl(d, values.get(i).copied())).sum()
    }

    fn pnl(d: &InstDay, value: Option<f64>) -> f64 {
        value.filter(|v| v.is_finite()).unwrap_or(d.value_at_roll) - d.value_at_roll + d.cash_flow
    }

    /// Advances the calendars. `group_of` maps an instrument to its day group, `key_of` a group
    /// to its current trading-day key; `values` are the EUR values of the held instruments.
    /// Returns the new `day_start` and the rolls (empty when nothing rolled; the first call only
    /// records the keys). Instruments that changed group (reference data) bank their day so far.
    pub fn advance(
        &mut self,
        group_of: impl Fn(&InstrumentId) -> String,
        key_of: impl Fn(&str) -> i64,
        values: &BTreeMap<InstrumentId, f64>,
        equity: f64,
        day_start: f64,
    ) -> (f64, Vec<Roll>) {
        let mut start = day_start;
        let mut rolls = Vec::new();
        // group membership: held instruments in a session group are tracked from their value now
        let ids: Vec<InstrumentId> = values.keys().chain(self.insts.keys()).copied().collect();
        for i in ids {
            let g = group_of(&i);
            match self.insts.get(&i) {
                Some(d) if d.group == g => {}
                Some(d) => {
                    // moved to another calendar: its day so far is banked, it restarts in the new one
                    start += Self::pnl(d, values.get(&i).copied());
                    self.insts.remove(&i);
                    if g != UTC_GROUP {
                        self.start_tracking(i, &g, values.get(&i).copied());
                    }
                }
                None if g != UTC_GROUP && values.get(&i).is_some_and(|v| *v != 0.0) => {
                    self.start_tracking(i, &g, values.get(&i).copied())
                }
                None => {}
            }
        }
        let mut groups: Vec<String> = self.insts.values().map(|d| d.group.clone()).collect();
        groups.push(UTC_GROUP.to_owned());
        groups.extend(self.days.keys().cloned());
        groups.sort();
        groups.dedup();
        // session groups first, so a UTC roll in the same call sees their fresh days
        for g in groups.iter().filter(|g| g.as_str() != UTC_GROUP) {
            let key = key_of(g);
            let prev = self.days.insert(g.clone(), key);
            let Some(from) = prev.filter(|p| *p != key) else { continue };
            let mut banked = 0.0;
            for (i, d) in self.insts.iter_mut().filter(|(_, d)| &d.group == g) {
                let v = values.get(i).copied().filter(|v| v.is_finite()).unwrap_or(d.value_at_roll);
                banked += v - d.value_at_roll + d.cash_flow;
                d.value_at_roll = v;
                d.cash_flow = 0.0;
            }
            start += banked;
            rolls.push(Roll { group: g.clone(), from, to: key, banked_eur: banked });
        }
        // flat instruments of a group that just rolled carry nothing any more
        self.insts.retain(|i, d| {
            d.value_at_roll != 0.0 || d.cash_flow != 0.0 || values.get(i).is_some_and(|v| *v != 0.0)
        });
        let key = key_of(UTC_GROUP);
        if let Some(from) = self.days.insert(UTC_GROUP.to_owned(), key).filter(|p| *p != key) {
            let new = equity - self.session_pnl(values);
            rolls.push(Roll { group: UTC_GROUP.to_owned(), from, to: key, banked_eur: new - start });
            start = new;
        }
        (start, rolls)
    }

    fn start_tracking(&mut self, i: InstrumentId, g: &str, value: Option<f64>) {
        let v = value.filter(|v| v.is_finite()).unwrap_or(0.0);
        self.insts.insert(i, InstDay { group: g.to_owned(), value_at_roll: v, cash_flow: 0.0 });
    }

    /// Restarts every day from now (book reseeded): nothing tracked, keys kept.
    pub fn reset(&mut self) {
        self.insts.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u128) -> InstrumentId {
        InstrumentId(uuid::Uuid::from_u128(n))
    }

    /// A crypto position (UTC) and a US equity (session day rolling at the US close).
    #[test]
    fn session_losses_count_until_their_exchange_day_ends() {
        let (btc, aapl) = (id(1), id(2));
        let group = |i: &InstrumentId| if *i == aapl { "us".to_owned() } else { UTC_GROUP.to_owned() };
        let mut s = SessionDays::default();
        let mut utc_day = 100;
        let mut us_day = 100;
        // day 100: cash 80 000, btc 10 000, aapl 10 000 → equity 100 000
        let mut vals = BTreeMap::from([(btc, 10_000.0), (aapl, 10_000.0)]);
        let (start, rolls) = s.advance(group, |g| if g == "us" { us_day } else { utc_day }, &vals, 100_000.0, 100_000.0);
        assert!(rolls.is_empty(), "first call only records the keys");
        assert_eq!(start, 100_000.0);
        // aapl loses 1 000 during the US session, btc gains 200
        vals.insert(aapl, 9_000.0);
        vals.insert(btc, 10_200.0);
        let eq = 99_200.0;
        // the UTC day rolls while the US exchange day is still running: its loss keeps counting
        utc_day = 101;
        let (start, rolls) = s.advance(group, |g| if g == "us" { us_day } else { utc_day }, &vals, eq, start);
        assert_eq!(rolls.len(), 1);
        assert_eq!(start, eq + 1_000.0, "day start excludes the running US session loss");
        assert!((1.0 - eq / start) > 0.0099, "the 1 000 US loss still counts after UTC midnight");
        // the US session closes: its day is banked, the loss stops counting
        us_day = 101;
        let (start, rolls) = s.advance(group, |g| if g == "us" { us_day } else { utc_day }, &vals, eq, start);
        assert_eq!(rolls.len(), 1);
        assert_eq!(rolls[0].banked_eur, -1_000.0);
        assert_eq!(start, eq);
    }

    #[test]
    fn utc_only_books_reduce_to_the_utc_day() {
        let btc = id(1);
        let mut s = SessionDays::default();
        let vals = BTreeMap::from([(btc, 5_000.0)]);
        let utc = |_: &InstrumentId| UTC_GROUP.to_owned();
        let (start, _) = s.advance(utc, |_| 7, &vals, 100.0, 120.0);
        assert_eq!(start, 120.0);
        let (start, rolls) = s.advance(utc, |_| 8, &vals, 95.0, start);
        assert_eq!((start, rolls.len()), (95.0, 1));
        assert_eq!(s.tracked().count(), 0);
    }

    #[test]
    fn fills_within_a_session_day_count_as_day_pnl() {
        let aapl = id(2);
        let us = |_: &InstrumentId| "us".to_owned();
        let mut s = SessionDays::default();
        let mut vals = BTreeMap::new();
        s.advance(us, |_| 1, &vals, 1_000.0, 1_000.0);
        // buy 500 EUR of aapl (fee 1): cash −501, value 500 → day P&L −1
        s.on_fill(aapl, "us", 1, 0.0, -501.0);
        vals.insert(aapl, 500.0);
        assert!((s.session_pnl(&vals) + 1.0).abs() < 1e-9);
        // it rises to 520: +19 on the day
        vals.insert(aapl, 520.0);
        assert!((s.session_pnl(&vals) - 19.0).abs() < 1e-9);
        // sold at 520 (fee 1): flat, cash +519 → day P&L +18; the roll banks it and forgets it
        s.on_fill(aapl, "us", 1, 520.0, 519.0);
        vals.insert(aapl, 0.0);
        let (start, rolls) = s.advance(us, |g| if g == "us" { 2 } else { 1 }, &vals, 1_018.0, 1_000.0);
        assert_eq!(rolls.len(), 1);
        assert!((start - 1_018.0).abs() < 1e-9);
        assert_eq!(s.tracked().count(), 0);
    }
}
