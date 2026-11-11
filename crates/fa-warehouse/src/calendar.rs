//! Trading calendars and session state (spec 12 §3, §4 `session`, §6, §8, §9).
//!
//! Built-in rules cover the venues the platform touches today: US equities (`xnys`, `xnas`,
//! Yahoo's plain US symbols), Xetra (`xetr`), FX OTC 24/5 and always-open crypto venues. Rules
//! compute holidays (Easter-relative and observed fixed dates), early closes and DST without a
//! tz database. [`materialize`] writes the computed sessions into `ref.calendar_session`
//! (idempotent; rows carrying a `source_id`, i.e. imported from a venue publication or entered
//! by an operator, are never overwritten), and
//! [`session_state_at`] reads the table first and falls back to the rule; a venue with neither
//! is reported closed with `missing_calendar` (spec 12 §8: treat as closed for live orders).
use chrono::{DateTime, Datelike, Duration, NaiveDate, NaiveTime, TimeZone, Utc, Weekday};
use serde::Serialize;
use sqlx::{PgPool, Row};

use crate::WarehouseError;

/// `open/closed/auction/halt` of spec 12 §4 plus pre/post segments and 24/7.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Open,
    PreMarket,
    PostMarket,
    Auction,
    Closed,
    Holiday,
    AlwaysOpen,
}

impl SessionState {
    /// Whether a live order may be sent (regular session, auctions and 24/7 venues).
    pub fn tradable(self) -> bool {
        matches!(self, Self::Open | Self::Auction | Self::AlwaysOpen)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CalendarRule {
    /// NYSE / Nasdaq: 09:30–16:00 America/New_York, pre 04:00, post until 20:00.
    Us,
    /// Xetra: 09:00–17:30 Europe/Berlin with a closing auction until 17:35.
    Xetr,
    /// FX OTC: Sunday 17:00 New York to Friday 17:00 New York, closed 25 Dec and 1 Jan.
    Fx24x5,
    /// Crypto venues.
    AlwaysOpen,
}

/// Picks the rule for a venue (`ref.venue.code`, `mic`, `always_open`).
pub fn rule_for_venue(code: &str, mic: Option<&str>, always_open: bool) -> Option<CalendarRule> {
    if always_open {
        return Some(CalendarRule::AlwaysOpen);
    }
    match (mic.unwrap_or("").to_ascii_uppercase().as_str(), code) {
        ("XNYS" | "XNAS" | "ARCX" | "BATS", _) | (_, "xnys" | "xnas") => Some(CalendarRule::Us),
        ("XETR" | "XFRA", _) | (_, "xetr") => Some(CalendarRule::Xetr),
        (_, "fx_otc" | "oanda") => Some(CalendarRule::Fx24x5),
        (_, "crypto" | "kraken" | "bitvavo" | "coinbase" | "binance" | "coingecko") => Some(CalendarRule::AlwaysOpen),
        _ => None,
    }
}

/// Picks the rule for an instrument listed on an aggregator venue (Yahoo), where the venue says
/// nothing about the market: asset class and symbol suffix decide.
pub fn rule_for_instrument(asset_class: &str, symbol: &str) -> Option<CalendarRule> {
    match asset_class {
        "crypto" => Some(CalendarRule::AlwaysOpen),
        "fx" => Some(CalendarRule::Fx24x5),
        "equity" | "etf" | "index" => {
            if symbol.ends_with(".DE") || symbol.ends_with(".F") {
                Some(CalendarRule::Xetr)
            } else if !symbol.contains('.') {
                Some(CalendarRule::Us)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// One trading day.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Session {
    pub date: NaiveDate,
    pub open: Option<DateTime<Utc>>,
    pub close: Option<DateTime<Utc>>,
    /// `[{kind, start, end}]` with kind pre | regular | post | closing_auction.
    pub segments: Vec<Segment>,
    pub is_holiday: bool,
    pub holiday: Option<String>,
    pub early_close: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
pub struct Segment {
    pub kind: String,
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

/// Anonymous Gregorian Easter.
pub fn easter(year: i32) -> NaiveDate {
    let a = year % 19;
    let b = year / 100;
    let c = year % 100;
    let d = b / 4;
    let e = b % 4;
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - d - g + 15) % 30;
    let i = c / 4;
    let k = c % 4;
    let l = (32 + 2 * e + 2 * i - h - k) % 7;
    let m = (a + 11 * h + 22 * l) / 451;
    let month = (h + l - 7 * m + 114) / 31;
    let day = (h + l - 7 * m + 114) % 31 + 1;
    NaiveDate::from_ymd_opt(year, month as u32, day as u32).expect("valid easter date")
}

fn ymd(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).expect("valid date")
}

/// n-th (1-based) weekday of a month; n = -1 is the last.
fn nth_weekday(year: i32, month: u32, wd: Weekday, n: i32) -> NaiveDate {
    if n > 0 {
        let first = ymd(year, month, 1);
        let shift = (7 + wd.num_days_from_monday() as i64 - first.weekday().num_days_from_monday() as i64) % 7;
        first + Duration::days(shift + 7 * (n as i64 - 1))
    } else {
        let next = if month == 12 { ymd(year + 1, 1, 1) } else { ymd(year, month + 1, 1) };
        let last = next - Duration::days(1);
        let shift = (7 + last.weekday().num_days_from_monday() as i64 - wd.num_days_from_monday() as i64) % 7;
        last - Duration::days(shift)
    }
}

/// US federal-style observance: Saturday → Friday, Sunday → Monday.
fn observed(d: NaiveDate) -> NaiveDate {
    match d.weekday() {
        Weekday::Sat => d - Duration::days(1),
        Weekday::Sun => d + Duration::days(1),
        _ => d,
    }
}

/// Holidays of a rule in a year (full closures), sorted.
pub fn holidays(rule: CalendarRule, year: i32) -> Vec<(NaiveDate, &'static str)> {
    let mut out = match rule {
        CalendarRule::Us => {
            let mut v = vec![
                (nth_weekday(year, 1, Weekday::Mon, 3), "Martin Luther King Jr. Day"),
                (nth_weekday(year, 2, Weekday::Mon, 3), "Washington's Birthday"),
                (easter(year) - Duration::days(2), "Good Friday"),
                (nth_weekday(year, 5, Weekday::Mon, -1), "Memorial Day"),
                (observed(ymd(year, 7, 4)), "Independence Day"),
                (nth_weekday(year, 9, Weekday::Mon, 1), "Labor Day"),
                (nth_weekday(year, 11, Weekday::Thu, 4), "Thanksgiving Day"),
                (observed(ymd(year, 12, 25)), "Christmas Day"),
            ];
            // NYSE does not close on Friday 31 Dec when 1 Jan is a Saturday.
            let ny = ymd(year, 1, 1);
            if ny.weekday() != Weekday::Sat {
                v.push((observed(ny), "New Year's Day"));
            }
            if year >= 2022 {
                v.push((observed(ymd(year, 6, 19)), "Juneteenth"));
            }
            v
        }
        CalendarRule::Xetr => {
            let e = easter(year);
            vec![
                (ymd(year, 1, 1), "New Year's Day"),
                (e - Duration::days(2), "Good Friday"),
                (e + Duration::days(1), "Easter Monday"),
                (ymd(year, 5, 1), "Labour Day"),
                (ymd(year, 12, 24), "Christmas Eve"),
                (ymd(year, 12, 25), "Christmas Day"),
                (ymd(year, 12, 26), "Boxing Day"),
                (ymd(year, 12, 31), "New Year's Eve"),
            ]
        }
        CalendarRule::Fx24x5 => vec![(ymd(year, 1, 1), "New Year's Day"), (ymd(year, 12, 25), "Christmas Day")],
        CalendarRule::AlwaysOpen => vec![],
    };
    out.retain(|(d, _)| !matches!(d.weekday(), Weekday::Sat | Weekday::Sun) && d.year() == year);
    out.sort();
    out
}

/// US early closes (13:00 New York): day after Thanksgiving, 24 Dec and 3 Jul on weekdays.
fn us_early_close(d: NaiveDate) -> bool {
    let y = d.year();
    let after_thanksgiving = nth_weekday(y, 11, Weekday::Thu, 4) + Duration::days(1);
    (d == after_thanksgiving || (d.month() == 12 && d.day() == 24) || (d.month() == 7 && d.day() == 3))
        && !matches!(d.weekday(), Weekday::Sat | Weekday::Sun)
}

/// New York UTC offset in hours on a date (DST from the second Sunday of March to the first
/// Sunday of November).
fn ny_offset(d: NaiveDate) -> i64 {
    let start = nth_weekday(d.year(), 3, Weekday::Sun, 2);
    let end = nth_weekday(d.year(), 11, Weekday::Sun, 1);
    if d >= start && d < end { -4 } else { -5 }
}

/// Berlin UTC offset in hours on a date (CEST from the last Sunday of March to the last Sunday of
/// October).
fn berlin_offset(d: NaiveDate) -> i64 {
    let start = nth_weekday(d.year(), 3, Weekday::Sun, -1);
    let end = nth_weekday(d.year(), 10, Weekday::Sun, -1);
    if d >= start && d < end { 2 } else { 1 }
}

fn at_local(d: NaiveDate, h: u32, m: u32, offset_h: i64) -> DateTime<Utc> {
    let naive = d.and_time(NaiveTime::from_hms_opt(h, m, 0).expect("valid time"));
    Utc.from_utc_datetime(&naive) - Duration::hours(offset_h)
}

fn seg(kind: &str, start: DateTime<Utc>, end: DateTime<Utc>) -> Segment {
    Segment { kind: kind.to_owned(), start, end }
}

/// The session of a rule on a date. Weekends and holidays have no open/close.
pub fn session_for(rule: CalendarRule, date: NaiveDate) -> Session {
    let weekend = matches!(date.weekday(), Weekday::Sat | Weekday::Sun);
    let holiday = holidays(rule, date.year()).into_iter().find(|(d, _)| *d == date).map(|(_, n)| n.to_owned());
    let mut s = Session {
        date,
        open: None,
        close: None,
        segments: vec![],
        is_holiday: holiday.is_some(),
        holiday,
        early_close: false,
    };
    if rule == CalendarRule::AlwaysOpen {
        let start = Utc.from_utc_datetime(&date.and_time(NaiveTime::MIN));
        s.open = Some(start);
        s.close = Some(start + Duration::days(1));
        s.segments = vec![seg("regular", start, start + Duration::days(1))];
        return s;
    }
    if weekend || s.is_holiday {
        return s;
    }
    match rule {
        CalendarRule::Us => {
            let off = ny_offset(date);
            let early = us_early_close(date);
            let (open, close) = (at_local(date, 9, 30, off), at_local(date, if early { 13 } else { 16 }, 0, off));
            s.early_close = early;
            s.segments = vec![
                seg("pre", at_local(date, 4, 0, off), open),
                seg("regular", open, close),
                seg("post", close, at_local(date, if early { 17 } else { 20 }, 0, off)),
            ];
            (s.open, s.close) = (Some(open), Some(close));
        }
        CalendarRule::Xetr => {
            let off = berlin_offset(date);
            let (open, close) = (at_local(date, 9, 0, off), at_local(date, 17, 30, off));
            s.segments = vec![
                seg("opening_auction", at_local(date, 8, 50, off), open),
                seg("regular", open, close),
                seg("closing_auction", close, at_local(date, 17, 35, off)),
            ];
            (s.open, s.close) = (Some(open), Some(close));
        }
        CalendarRule::Fx24x5 => {
            // trading day D runs from D-1 17:00 New York to D 17:00 New York
            let prev = date - Duration::days(1);
            let (open, close) = (at_local(prev, 17, 0, ny_offset(prev)), at_local(date, 17, 0, ny_offset(date)));
            s.segments = vec![seg("regular", open, close)];
            (s.open, s.close) = (Some(open), Some(close));
        }
        CalendarRule::AlwaysOpen => unreachable!(),
    }
    s
}

/// Session state of a rule at an instant.
pub fn state_at(rule: CalendarRule, at: DateTime<Utc>) -> SessionState {
    if rule == CalendarRule::AlwaysOpen {
        return SessionState::AlwaysOpen;
    }
    // an instant can belong to the session of the UTC date or (FX) of the next one
    for d in [at.date_naive(), at.date_naive() + Duration::days(1), at.date_naive() - Duration::days(1)] {
        let s = session_for(rule, d);
        if let Some(st) = state_in(&s, at) {
            return st;
        }
    }
    let s = session_for(rule, at.date_naive());
    if s.is_holiday { SessionState::Holiday } else { SessionState::Closed }
}

/// State of an instant inside one session's segments (`None` when outside all of them).
pub fn state_in(s: &Session, at: DateTime<Utc>) -> Option<SessionState> {
    s.segments.iter().find(|g| g.start <= at && at < g.end).map(|g| match g.kind.as_str() {
        "pre" => SessionState::PreMarket,
        "post" => SessionState::PostMarket,
        "opening_auction" | "closing_auction" => SessionState::Auction,
        _ => SessionState::Open,
    })
}

/// Writes the rule's sessions for `[from, to]` into `ref.calendar_session` for a venue. Rows with a
/// `source_id` (venue publication, operator correction) are kept. Returns rows written.
pub async fn materialize(
    pool: &PgPool,
    venue_code: &str,
    rule: CalendarRule,
    from: NaiveDate,
    to: NaiveDate,
) -> Result<usize, WarehouseError> {
    let Some(venue_id) = sqlx::query_scalar::<_, uuid::Uuid>("SELECT id FROM ref.venue WHERE code = $1")
        .bind(venue_code)
        .fetch_optional(pool)
        .await?
    else {
        return Ok(0);
    };
    let mut n = 0;
    let mut d = from;
    while d <= to {
        let s = session_for(rule, d);
        let segs = serde_json::to_value(&s.segments).unwrap_or_default();
        let r = sqlx::query(
            "INSERT INTO ref.calendar_session (venue_id, date, open, close, segments, is_holiday) \
             VALUES ($1, $2, $3, $4, $5, $6) \
             ON CONFLICT (venue_id, date) DO UPDATE SET open = excluded.open, close = excluded.close, \
               segments = excluded.segments, is_holiday = excluded.is_holiday, recorded_at = now() \
             WHERE ref.calendar_session.source_id IS NULL",
        )
        .bind(venue_id)
        .bind(d)
        .bind(s.open)
        .bind(s.close)
        .bind(segs)
        .bind(s.is_holiday)
        .execute(pool)
        .await?;
        n += r.rows_affected() as usize;
        d += Duration::days(1);
    }
    Ok(n)
}

/// Session state answer with provenance.
#[derive(Debug, Clone, Serialize)]
pub struct SessionAnswer {
    pub venue: String,
    pub at: DateTime<Utc>,
    pub state: SessionState,
    /// `table` (ref.calendar_session), `rule` (built-in) or `missing_calendar`.
    pub basis: &'static str,
    pub rule: Option<CalendarRule>,
    pub session: Option<Session>,
}

/// `RefData::session` (spec 12 §4): the venue's state at `at`. Reads `ref.calendar_session`
/// first (always-open venues answer immediately), else the built-in rule, else closed with
/// basis `missing_calendar`.
pub async fn session_state_at(pool: &PgPool, venue_code: &str, at: DateTime<Utc>) -> Result<SessionAnswer, WarehouseError> {
    let venue = sqlx::query("SELECT id, mic, always_open FROM ref.venue WHERE code = $1")
        .bind(venue_code)
        .fetch_optional(pool)
        .await?;
    let (venue_id, mic, always_open) = match &venue {
        Some(r) => (Some(r.get::<uuid::Uuid, _>("id")), r.get::<Option<String>, _>("mic"), r.get::<bool, _>("always_open")),
        None => (None, None, false),
    };
    let rule = rule_for_venue(venue_code, mic.as_deref(), always_open);
    let answer = |state, basis, session| SessionAnswer { venue: venue_code.to_owned(), at, state, basis, rule, session };
    if rule == Some(CalendarRule::AlwaysOpen) {
        return Ok(answer(SessionState::AlwaysOpen, "rule", None));
    }
    if let Some(vid) = venue_id {
        for d in [at.date_naive(), at.date_naive() + Duration::days(1), at.date_naive() - Duration::days(1)] {
            let row = sqlx::query("SELECT open, close, segments, is_holiday FROM ref.calendar_session WHERE venue_id = $1 AND date = $2")
                .bind(vid)
                .bind(d)
                .fetch_optional(pool)
                .await?;
            if let Some(r) = row {
                let segments: Vec<Segment> = serde_json::from_value(r.get("segments")).unwrap_or_default();
                let s = Session {
                    date: d,
                    open: r.get("open"),
                    close: r.get("close"),
                    segments,
                    is_holiday: r.get("is_holiday"),
                    holiday: None,
                    early_close: false,
                };
                if let Some(st) = state_in(&s, at) {
                    return Ok(answer(st, "table", Some(s)));
                }
                if d == at.date_naive() && s.is_holiday {
                    return Ok(answer(SessionState::Holiday, "table", Some(s)));
                }
            }
        }
    }
    match rule {
        Some(r) => {
            let st = state_at(r, at);
            Ok(answer(st, "rule", Some(session_for(r, at.date_naive()))))
        }
        None => Ok(answer(SessionState::Closed, "missing_calendar", None)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utc(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc)
    }

    #[test]
    fn easter_dates() {
        assert_eq!(easter(2024), ymd(2024, 3, 31));
        assert_eq!(easter(2025), ymd(2025, 4, 20));
        assert_eq!(easter(2026), ymd(2026, 4, 5));
    }

    #[test]
    fn us_holidays_2026() {
        let h: Vec<_> = holidays(CalendarRule::Us, 2026).into_iter().map(|(d, _)| d.to_string()).collect();
        // NYSE 2026: Jan 1, Jan 19, Feb 16, Apr 3, May 25, Jun 19, Jul 3 (observed), Sep 7, Nov 26, Dec 25
        assert_eq!(
            h,
            [
                "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03",
                "2026-09-07", "2026-11-26", "2026-12-25"
            ]
        );
        // 2022: Jan 1 is a Saturday → no observed holiday on Fri 31 Dec 2021 nor in 2022
        assert!(!holidays(CalendarRule::Us, 2022).iter().any(|(_, n)| *n == "New Year's Day"));
    }

    #[test]
    fn us_session_handles_dst_and_early_close() {
        // winter: 09:30 EST = 14:30 UTC
        let s = session_for(CalendarRule::Us, ymd(2026, 1, 5));
        assert_eq!(s.open.unwrap(), utc("2026-01-05T14:30:00Z"));
        assert_eq!(s.close.unwrap(), utc("2026-01-05T21:00:00Z"));
        // summer: 09:30 EDT = 13:30 UTC
        assert_eq!(session_for(CalendarRule::Us, ymd(2026, 7, 6)).open.unwrap(), utc("2026-07-06T13:30:00Z"));
        // day after Thanksgiving closes 13:00 EST
        let f = session_for(CalendarRule::Us, ymd(2026, 11, 27));
        assert!(f.early_close);
        assert_eq!(f.close.unwrap(), utc("2026-11-27T18:00:00Z"));
        assert_eq!(state_at(CalendarRule::Us, utc("2026-01-05T15:00:00Z")), SessionState::Open);
        assert_eq!(state_at(CalendarRule::Us, utc("2026-01-05T12:00:00Z")), SessionState::PreMarket);
        assert_eq!(state_at(CalendarRule::Us, utc("2026-01-05T22:00:00Z")), SessionState::PostMarket);
        assert_eq!(state_at(CalendarRule::Us, utc("2026-01-10T15:00:00Z")), SessionState::Closed); // Saturday
        assert_eq!(state_at(CalendarRule::Us, utc("2026-12-25T15:00:00Z")), SessionState::Holiday);
    }

    #[test]
    fn xetra_and_fx_and_crypto() {
        // 2026-03-30 is after the last Sunday of March → CEST: 09:00 = 07:00 UTC
        assert_eq!(session_for(CalendarRule::Xetr, ymd(2026, 3, 30)).open.unwrap(), utc("2026-03-30T07:00:00Z"));
        assert_eq!(session_for(CalendarRule::Xetr, ymd(2026, 3, 27)).open.unwrap(), utc("2026-03-27T08:00:00Z"));
        assert_eq!(state_at(CalendarRule::Xetr, utc("2026-04-06T10:00:00Z")), SessionState::Holiday); // Easter Monday
        assert_eq!(state_at(CalendarRule::Xetr, utc("2026-01-05T16:32:00Z")), SessionState::Auction);
        // FX: open Sunday 22:00 UTC (winter), closed Saturday
        assert_eq!(state_at(CalendarRule::Fx24x5, utc("2026-01-11T22:30:00Z")), SessionState::Open);
        assert_eq!(state_at(CalendarRule::Fx24x5, utc("2026-01-10T12:00:00Z")), SessionState::Closed);
        assert_eq!(state_at(CalendarRule::Fx24x5, utc("2026-01-09T21:59:00Z")), SessionState::Open);
        assert_eq!(state_at(CalendarRule::Fx24x5, utc("2026-01-09T22:01:00Z")), SessionState::Closed);
        assert_eq!(state_at(CalendarRule::AlwaysOpen, utc("2026-12-25T03:00:00Z")), SessionState::AlwaysOpen);
        assert!(SessionState::AlwaysOpen.tradable() && !SessionState::Holiday.tradable());
    }

    #[test]
    fn rules_by_venue_and_instrument() {
        assert_eq!(rule_for_venue("xnas", Some("XNAS"), false), Some(CalendarRule::Us));
        assert_eq!(rule_for_venue("kraken", None, true), Some(CalendarRule::AlwaysOpen));
        assert_eq!(rule_for_venue("yahoo", None, false), None);
        assert_eq!(rule_for_instrument("crypto", "BTC-USD"), Some(CalendarRule::AlwaysOpen));
        assert_eq!(rule_for_instrument("equity", "SAP.DE"), Some(CalendarRule::Xetr));
        assert_eq!(rule_for_instrument("equity", "AAPL"), Some(CalendarRule::Us));
        assert_eq!(rule_for_instrument("fx", "EURUSD=X"), Some(CalendarRule::Fx24x5));
    }
}
