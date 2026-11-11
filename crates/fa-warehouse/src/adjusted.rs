//! Materialised adjusted daily bars (spec 12 §6 "option to materialize adjusted daily bars",
//! §7 `adjustment.materialize_daily`, Q2).
//!
//! Policy (Q2, provisional): raw series stay raw in VictoriaMetrics; adjustment is computed on
//! read by [`crate::corpactions::adjust_bars`] for any range, and additionally materialised into
//! `wh.adjusted_bar_1d` for the daily timescale of every instrument that has effective splits,
//! stock dividends or cash dividends, in both modes (`splits_only` for price continuity,
//! `total_return` for models that want dividends reinvested). Instruments without such actions
//! need no rows: their adjusted series is the raw one. Rows are back-adjusted to the latest bar
//! and carry the digest of the action set they were computed from ([`actions_digest`]); a new,
//! confirmed or cancelled action changes the digest and [`refresh`] recomputes the instrument.
//! The materialised rows are a current-knowledge cache: point-in-time reads for backtests keep
//! using `adjust_bars` over `corpactions::list_as_of(known_at)`.
use std::{collections::BTreeSet, future::Future};

use chrono::{DateTime, NaiveDate, Utc};
use serde::Serialize;
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::{
    WarehouseError,
    corpactions::{self, AdjustMode, BarPoint, CorporateAction},
};

/// Action kinds that change an adjusted series.
pub const ADJUSTING_KINDS: [&str; 4] = ["split", "reverse_split", "stock_dividend", "cash_dividend"];

/// `wh.adjusted_bar_1d.mode` token.
pub fn mode_str(mode: AdjustMode) -> &'static str {
    match mode {
        AdjustMode::SplitsOnly => "splits_only",
        AdjustMode::TotalReturn => "total_return",
    }
}

/// Deterministic digest (FNV-1a 64, hex) of the adjusting actions an adjusted series depends on:
/// id, kind, ex date, ratio, amount, status, cancellation. Order-independent.
pub fn actions_digest(actions: &[CorporateAction]) -> String {
    let mut keys: Vec<String> = actions
        .iter()
        .filter(|a| ADJUSTING_KINDS.contains(&a.kind.as_str()))
        .map(|a| {
            format!("{}|{}|{}|{:?}|{:?}|{}|{:?}", a.id, a.kind, a.ex_date, a.ratio, a.amount, a.status, a.cancelled_at)
        })
        .collect();
    keys.sort();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for k in &keys {
        for b in k.bytes().chain(std::iter::once(b'\n')) {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    format!("{h:016x}")
}

/// One materialised row.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AdjustedBar {
    pub day: NaiveDate,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub raw_close: f64,
    /// adjusted close / raw close.
    pub factor: f64,
}

/// Computes the adjusted rows for a raw daily series (pure; `raw` sorted by day, one bar per day).
pub fn adjusted_rows(raw: &[BarPoint], actions: &[CorporateAction], mode: AdjustMode) -> Vec<AdjustedBar> {
    let adj = corpactions::adjust_bars(raw, actions, mode);
    raw.iter()
        .zip(adj)
        .map(|(r, a)| AdjustedBar {
            day: r.ts.date_naive(),
            open: a.open,
            high: a.high,
            low: a.low,
            close: a.close,
            volume: a.volume,
            raw_close: r.close,
            factor: if r.close != 0.0 { a.close / r.close } else { 1.0 },
        })
        .collect()
}

/// Instruments with at least one adjusting action (any status; cancelled ones still need a
/// recompute so the rows go back to raw).
pub async fn instruments_with_actions(pool: &PgPool) -> Result<Vec<Uuid>, WarehouseError> {
    let kinds: Vec<String> = ADJUSTING_KINDS.iter().map(|k| (*k).to_owned()).collect();
    Ok(sqlx::query_scalar(
        "SELECT DISTINCT instrument_id FROM ref.corporate_action WHERE kind = ANY($1) ORDER BY instrument_id",
    )
    .bind(&kinds)
    .fetch_all(pool)
    .await?)
}

/// Digest stored for an instrument and mode (`None` when nothing is materialised).
pub async fn stored_digest(
    pool: &PgPool,
    instrument: Uuid,
    mode: AdjustMode,
) -> Result<Option<String>, WarehouseError> {
    Ok(sqlx::query_scalar(
        "SELECT actions_digest FROM wh.adjusted_bar_1d WHERE instrument_id = $1 AND mode = $2 ORDER BY day DESC LIMIT 1",
    )
    .bind(instrument)
    .bind(mode_str(mode))
    .fetch_optional(pool)
    .await?)
}

/// Replaces the materialised series of one instrument and mode with `raw` adjusted by the
/// actions known now. Returns the number of rows written.
pub async fn materialize_daily(
    pool: &PgPool,
    instrument: Uuid,
    mode: AdjustMode,
    raw: &[BarPoint],
) -> Result<usize, WarehouseError> {
    let actions = corpactions::list_as_of(pool, instrument, None).await?;
    materialize_with(pool, instrument, mode, raw, &actions).await
}

async fn materialize_with(
    pool: &PgPool,
    instrument: Uuid,
    mode: AdjustMode,
    raw: &[BarPoint],
    actions: &[CorporateAction],
) -> Result<usize, WarehouseError> {
    let digest = actions_digest(actions);
    let rows = adjusted_rows(raw, actions, mode);
    let mut tx = pool.begin().await?;
    sqlx::query("DELETE FROM wh.adjusted_bar_1d WHERE instrument_id = $1 AND mode = $2")
        .bind(instrument)
        .bind(mode_str(mode))
        .execute(&mut *tx)
        .await?;
    for r in &rows {
        sqlx::query(
            "INSERT INTO wh.adjusted_bar_1d (instrument_id, mode, day, open, high, low, close, volume, raw_close, factor, actions_digest) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) \
             ON CONFLICT (instrument_id, mode, day) DO UPDATE SET open = excluded.open, high = excluded.high, \
               low = excluded.low, close = excluded.close, volume = excluded.volume, raw_close = excluded.raw_close, \
               factor = excluded.factor, actions_digest = excluded.actions_digest, computed_at = now()",
        )
        .bind(instrument)
        .bind(mode_str(mode))
        .bind(r.day)
        .bind(r.open)
        .bind(r.high)
        .bind(r.low)
        .bind(r.close)
        .bind(r.volume)
        .bind(r.raw_close)
        .bind(r.factor)
        .bind(&digest)
        .execute(&mut *tx)
        .await?;
    }
    tx.commit().await?;
    Ok(rows.len())
}

/// Materialised rows of `[from, to]`, oldest first.
pub async fn read_daily(
    pool: &PgPool,
    instrument: Uuid,
    mode: AdjustMode,
    from: NaiveDate,
    to: NaiveDate,
) -> Result<Vec<AdjustedBar>, WarehouseError> {
    let rows = sqlx::query(
        "SELECT day, open, high, low, close, volume, raw_close, factor FROM wh.adjusted_bar_1d \
         WHERE instrument_id = $1 AND mode = $2 AND day BETWEEN $3 AND $4 ORDER BY day",
    )
    .bind(instrument)
    .bind(mode_str(mode))
    .bind(from)
    .bind(to)
    .fetch_all(pool)
    .await?;
    Ok(rows
        .iter()
        .map(|r| AdjustedBar {
            day: r.get("day"),
            open: r.get("open"),
            high: r.get("high"),
            low: r.get("low"),
            close: r.get("close"),
            volume: r.get("volume"),
            raw_close: r.get("raw_close"),
            factor: r.get("factor"),
        })
        .collect())
}

/// Outcome of [`refresh`].
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct RefreshReport {
    pub instruments: usize,
    /// (instrument, mode) series rewritten.
    pub series: usize,
    pub rows: usize,
    /// Instruments with actions but no raw daily bars (nothing to adjust yet).
    pub without_bars: usize,
    pub errors: Vec<String>,
}

/// Scheduled materialisation (`adjustment.materialize_daily = true`): for every instrument with
/// adjusting actions, fetches its raw daily bars through `fetch` (VictoriaMetrics in
/// control-api) and rewrites both modes. Run daily so new bars and new actions land within a day.
pub async fn refresh<F, Fut>(pool: &PgPool, mut fetch: F) -> Result<RefreshReport, WarehouseError>
where
    F: FnMut(Uuid) -> Fut,
    Fut: Future<Output = Result<Vec<BarPoint>, String>>,
{
    let mut rep = RefreshReport::default();
    for inst in instruments_with_actions(pool).await? {
        rep.instruments += 1;
        let mut raw = match fetch(inst).await {
            Ok(r) => r,
            Err(e) => {
                rep.errors.push(format!("{inst}: {e}"));
                continue;
            }
        };
        raw.sort_by_key(|b| b.ts);
        let mut seen = BTreeSet::new();
        raw.retain(|b| seen.insert(b.ts.date_naive())); // one bar per day
        if raw.is_empty() {
            rep.without_bars += 1;
            continue;
        }
        let actions = corpactions::list_as_of(pool, inst, None).await?;
        for mode in [AdjustMode::SplitsOnly, AdjustMode::TotalReturn] {
            rep.rows += materialize_with(pool, inst, mode, &raw, &actions).await?;
            rep.series += 1;
        }
    }
    Ok(rep)
}

/// Converts a daily bar into a [`BarPoint`] dated at 00:00 UTC of its trading day. `close_ms` is
/// the bar's close (the ingestor stamps bars at close = open + 1 day: Kraken 00:00 UTC, Yahoo US
/// equities 13:30 UTC on the next day) or the end of the hourly bucket that holds it; the trading
/// day is the UTC date one day before.
pub fn daily_point(close_ms: i64, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Option<BarPoint> {
    let end = DateTime::<Utc>::from_timestamp_millis(close_ms - 86_400_000)?;
    let ts = end.date_naive().and_hms_opt(0, 0, 0)?.and_utc();
    Some(BarPoint { ts, open, high, low, close, volume })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn bar(d: u32, c: f64) -> BarPoint {
        BarPoint {
            ts: Utc.with_ymd_and_hms(2026, 3, d, 0, 0, 0).unwrap(),
            open: c,
            high: c,
            low: c,
            close: c,
            volume: 100.0,
        }
    }

    fn split(d: u32, ratio: f64) -> CorporateAction {
        CorporateAction {
            id: Uuid::from_u128(u128::from(d)),
            instrument_id: Uuid::nil(),
            kind: "split".into(),
            ex_date: NaiveDate::from_ymd_opt(2026, 3, d).unwrap(),
            record_date: None,
            pay_date: None,
            ratio: Some(ratio),
            amount: None,
            ccy: None,
            source_id: None,
            announced_at: None,
            recorded_at: Utc::now(),
            status: "confirmed".into(),
            cancelled_at: None,
            old_symbol: None,
            new_symbol: None,
        }
    }

    #[test]
    fn rows_are_continuous_across_a_split_and_carry_factor() {
        let raw = [bar(2, 100.0), bar(3, 102.0), bar(4, 51.5), bar(5, 52.0)];
        let rows = adjusted_rows(&raw, &[split(4, 2.0)], AdjustMode::SplitsOnly);
        assert_eq!(rows.len(), 4);
        assert_eq!((rows[1].close, rows[1].raw_close, rows[1].factor), (51.0, 102.0, 0.5));
        assert_eq!(rows[1].volume, 200.0);
        assert_eq!((rows[2].close, rows[2].factor), (51.5, 1.0));
        assert_eq!(rows[0].day, NaiveDate::from_ymd_opt(2026, 3, 2).unwrap());
    }

    #[test]
    fn digest_is_order_independent_and_changes_with_cancellation() {
        let (a, b) = (split(4, 2.0), split(9, 3.0));
        assert_eq!(actions_digest(&[a.clone(), b.clone()]), actions_digest(&[b.clone(), a.clone()]));
        let mut c = a.clone();
        c.status = "cancelled".into();
        c.cancelled_at = Some(Utc::now());
        assert_ne!(actions_digest(&[a, b.clone()]), actions_digest(&[c, b]));
        let mut sym = split(10, 1.0);
        sym.kind = "symbol_change".into();
        assert_eq!(actions_digest(&[]), actions_digest(&[sym]), "non-adjusting kinds do not count");
    }

    #[test]
    fn daily_point_dates_the_trading_day() {
        // Kraken: open 4 Mar 00:00 UTC, close 5 Mar 00:00
        let end = Utc.with_ymd_and_hms(2026, 3, 5, 0, 0, 0).unwrap().timestamp_millis();
        let p = daily_point(end, 1.0, 2.0, 0.5, 1.5, 10.0).unwrap();
        assert_eq!(p.ts.date_naive(), NaiveDate::from_ymd_opt(2026, 3, 4).unwrap());
        // Yahoo US: open 4 Mar 14:30 UTC, close stamped 5 Mar 14:30, hourly bucket 15:00
        let end = Utc.with_ymd_and_hms(2026, 3, 5, 15, 0, 0).unwrap().timestamp_millis();
        assert_eq!(
            daily_point(end, 1.0, 1.0, 1.0, 1.0, 0.0).unwrap().ts.date_naive(),
            NaiveDate::from_ymd_opt(2026, 3, 4).unwrap()
        );
    }
}
