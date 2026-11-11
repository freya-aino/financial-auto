//! The daily-loss day per trading calendar (spec 40 §6.5, §10 Q4; `fa_risk::session`).
//!
//! Each instrument's day group comes from its exchange calendar (`fa_warehouse::calendar`,
//! loaded per instrument from its current listing by [`load_calendars`]): 24/7 venues and
//! instruments without a calendar are on the UTC day; `us`, `xetr` and `fx` days end at their
//! session close (the trading-day key is the date of the next close, so the overnight and a
//! weekend belong to the next session). The tick advances the calendars before checking
//! equity: a session group that rolled banks its finished day into the day-start equity, the
//! UTC roll restarts the 24/7 part; the daily-loss halt clears after a roll only when the
//! remaining loss is below the limit.
use std::collections::BTreeMap;

use chrono::{DateTime, Datelike, Duration, Utc};
use fa_core::InstrumentId;
use fa_risk::{
    HaltScope,
    session::UTC_GROUP,
};
use fa_warehouse::calendar::{CalendarRule, rule_for_instrument, rule_for_venue, session_for};
use sqlx::{PgPool, Row};

use super::{Engine, MS_PER_DAY};

/// Day group of a calendar rule.
pub fn group_of_rule(rule: Option<CalendarRule>) -> &'static str {
    match rule {
        Some(CalendarRule::Us) => "us",
        Some(CalendarRule::Xetr) => "xetr",
        Some(CalendarRule::Fx24x5) => "fx",
        Some(CalendarRule::AlwaysOpen) | None => UTC_GROUP,
    }
}

fn rule_of_group(group: &str) -> Option<CalendarRule> {
    match group {
        "us" => Some(CalendarRule::Us),
        "xetr" => Some(CalendarRule::Xetr),
        "fx" => Some(CalendarRule::Fx24x5),
        _ => None,
    }
}

/// `(key, valid until ms)` of a group's trading day at `ms`: the UTC day number for the UTC
/// group, else the date (days since CE) of the next session close after `ms`, valid until that
/// close.
pub fn trading_day(group: &str, ms: i64) -> (i64, i64) {
    let utc = (ms.div_euclid(MS_PER_DAY), (ms.div_euclid(MS_PER_DAY) + 1) * MS_PER_DAY);
    let Some(rule) = rule_of_group(group) else { return utc };
    let Some(t) = DateTime::<Utc>::from_timestamp_millis(ms) else { return utc };
    // the next close: at most a long weekend plus holidays ahead
    for k in -1..=14 {
        let d = t.date_naive() + Duration::days(k);
        if let Some(close) = session_for(rule, d).close
            && t < close
        {
            return (i64::from(d.num_days_from_ce()), close.timestamp_millis());
        }
    }
    utc
}

/// Calendar rule per instrument from its current listing: the venue's rule (`ref.venue.code`,
/// `mic`, `always_open`), else — aggregator venues such as Yahoo — the asset class and symbol
/// suffix (`fa_warehouse::calendar::rule_for_instrument`). Instruments without either stay on
/// the UTC day.
pub async fn load_calendars(pool: &PgPool) -> Result<BTreeMap<InstrumentId, CalendarRule>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT DISTINCT ON (i.id) i.id, i.asset_class, l.venue_symbol, v.code, v.mic, v.always_open \
           FROM ref.instrument i \
           JOIN ref.listing l ON l.instrument_id = i.id AND now() >= l.valid_from AND now() < l.valid_to \
           JOIN ref.venue v ON v.id = l.venue_id \
          WHERE i.status <> 'merged' AND now() >= i.valid_from AND now() < i.valid_to \
          ORDER BY i.id, v.code",
    )
    .fetch_all(pool)
    .await?;
    let mut out = BTreeMap::new();
    for r in rows {
        let code: String = r.get("code");
        let mic: Option<String> = r.get("mic");
        let class: String = r.get("asset_class");
        let symbol: String = r.get("venue_symbol");
        let rule = rule_for_venue(&code, mic.as_deref(), r.get("always_open"))
            .or_else(|| rule_for_instrument(&class.to_lowercase(), &symbol));
        if let Some(rule) = rule {
            out.insert(InstrumentId(r.get("id")), rule);
        }
    }
    Ok(out)
}

impl Engine {
    /// Calendar rule per instrument ([`load_calendars`]; absent = UTC day).
    pub fn set_calendars(&mut self, m: BTreeMap<InstrumentId, CalendarRule>) {
        let sessions = m.values().filter(|r| group_of_rule(Some(**r)) != UTC_GROUP).count();
        self.counters.insert("daily_loss.session_instruments", sessions as f64);
        self.calendars = m;
    }

    /// Day group of an instrument.
    pub(super) fn day_group(&self, i: &InstrumentId) -> &'static str {
        group_of_rule(self.calendars.get(i).copied())
    }

    /// The group's trading-day key at `now_ms` (cached until the day's end).
    pub(super) fn day_key(&mut self, group: &str, now_ms: i64) -> i64 {
        if let Some((k, until)) = self.day_keys.get(group)
            && now_ms < *until
        {
            return *k;
        }
        let (k, until) = trading_day(group, now_ms);
        self.day_keys.insert(group.to_owned(), (k, until));
        k
    }

    /// EUR value per held instrument (the equity's positions part).
    fn position_values(&self) -> BTreeMap<InstrumentId, f64> {
        self.ledger
            .positions
            .iter()
            .filter(|(_, p)| p.quantity != 0.0)
            .map(|(i, p)| (*i, p.quantity * self.marks.get(i).copied().unwrap_or(p.avg_cost_eur)))
            .collect()
    }

    /// A booked fill's cash flow on its instrument's session day (before the book applies it:
    /// `value_before` is the position's value just before the fill).
    pub(super) fn note_session_fill(&mut self, inst: InstrumentId, value_before: f64, cash_flow: f64, now_ms: i64) {
        let g = self.day_group(&inst);
        if g == UTC_GROUP {
            return;
        }
        let key = self.day_key(g, now_ms);
        self.session_days.on_fill(inst, g, key, value_before, cash_flow);
    }

    /// Advances every day group (spec 40 §10 Q4); replaces the plain UTC-day roll.
    pub(super) fn roll_days(&mut self, now_ms: i64) {
        let values = self.position_values();
        let mut groups: Vec<&'static str> = values.keys().map(|i| self.day_group(i)).collect();
        let tracked: Vec<InstrumentId> = self.session_days.tracked().map(|(i, _)| *i).collect();
        groups.extend(tracked.iter().map(|i| self.day_group(i)));
        groups.extend(["us", "xetr", "fx"].into_iter().filter(|g| self.session_days.days().contains_key(*g)));
        groups.push(UTC_GROUP);
        groups.sort_unstable();
        groups.dedup();
        let keys: BTreeMap<&str, i64> = groups.iter().map(|g| (*g, self.day_key(g, now_ms))).collect();
        let utc_day = keys[UTC_GROUP];
        let first = self.day.is_none();
        let equity = self.equity();
        let mut days = std::mem::take(&mut self.session_days);
        if let Some(d) = self.day {
            days.ensure_day(UTC_GROUP, d);
        }
        let calendars = &self.calendars;
        let (start, rolls) = days.advance(
            |i| group_of_rule(calendars.get(i).copied()).to_owned(),
            |g| keys.get(g).copied().unwrap_or(utc_day),
            &values,
            equity,
            self.risk.day_start_equity(),
        );
        self.session_days = days;
        self.day = Some(utc_day);
        if first || rolls.is_empty() {
            return;
        }
        let cleared = self.risk.roll_day_start(start, equity);
        self.day_start_equity = self.risk.day_start_equity();
        if cleared || !self.risk.halts().contains_key(&HaltScope::DailyLoss) {
            self.halts.remove(&HaltScope::DailyLoss);
        }
        self.risk_dirty = true;
        for r in &rolls {
            *self.counters.entry("daily_loss.rolls").or_insert(0.0) += 1.0;
            tracing::info!(group = %r.group, from = r.from, to = r.to, banked_eur = r.banked_eur, day_start = start, "daily-loss day rolled");
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    fn ms(y: i32, m: u32, d: u32, h: u32, min: u32) -> i64 {
        Utc.with_ymd_and_hms(y, m, d, h, min, 0).unwrap().timestamp_millis()
    }

    #[test]
    fn exchange_days_end_at_the_close_and_skip_weekends() {
        // US: 2026-11-13 is a Friday; the close is 21:00 UTC (EST)
        let (fri, until) = trading_day("us", ms(2026, 11, 13, 15, 0));
        assert_eq!(until, ms(2026, 11, 13, 21, 0));
        let (after_close, _) = trading_day("us", ms(2026, 11, 13, 21, 30));
        let (saturday, _) = trading_day("us", ms(2026, 11, 14, 12, 0));
        let (monday, _) = trading_day("us", ms(2026, 11, 16, 15, 0));
        assert_ne!(fri, after_close, "the day ends at the close");
        assert_eq!(after_close, saturday, "the weekend belongs to Monday's session");
        assert_eq!(saturday, monday);
        // 24/7 venues keep the UTC day
        let (a, _) = trading_day(UTC_GROUP, ms(2026, 11, 14, 23, 59));
        let (b, _) = trading_day(UTC_GROUP, ms(2026, 11, 15, 0, 1));
        assert_eq!(b, a + 1);
        // FX: the day ends 17:00 New York (22:00 UTC in winter)
        let (fx, until) = trading_day("fx", ms(2026, 11, 12, 21, 0));
        assert_eq!(until, ms(2026, 11, 12, 22, 0));
        assert_ne!(trading_day("fx", ms(2026, 11, 12, 22, 30)).0, fx);
    }
}
