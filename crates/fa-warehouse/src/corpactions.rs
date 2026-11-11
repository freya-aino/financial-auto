//! Corporate actions, adjustment factors and symbol changes (spec 12 §3, §4
//! `adjustment_factor`, §6 "Adjusted series", §8, §9).
//!
//! Raw series stay raw (VictoriaMetrics); adjusted series are computed on read with
//! [`adjust_bars`] (spec 12 Q2 default: on read, price-only splits unless total return is asked
//! for). Point-in-time: [`list_as_of`] returns only actions with `recorded_at <= known_at`, and a
//! cancellation only hides an action for queries known after it.
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::WarehouseError;


#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorporateAction {
    pub id: Uuid,
    pub instrument_id: Uuid,
    /// split | reverse_split | stock_dividend | cash_dividend | spinoff | merger | symbol_change
    pub kind: String,
    pub ex_date: NaiveDate,
    pub record_date: Option<NaiveDate>,
    pub pay_date: Option<NaiveDate>,
    /// Splits: new shares per old share. Stock dividends: extra shares per share.
    pub ratio: Option<f64>,
    /// Cash dividends: amount per share in `ccy`.
    pub amount: Option<f64>,
    pub ccy: Option<String>,
    pub source_id: Option<String>,
    pub announced_at: Option<DateTime<Utc>>,
    pub recorded_at: DateTime<Utc>,
    pub status: String,
    pub cancelled_at: Option<DateTime<Utc>>,
    pub old_symbol: Option<String>,
    pub new_symbol: Option<String>,
}

/// Input for [`record`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NewCorporateAction {
    pub instrument_id: Uuid,
    pub kind: String,
    pub ex_date: NaiveDate,
    #[serde(default)]
    pub record_date: Option<NaiveDate>,
    #[serde(default)]
    pub pay_date: Option<NaiveDate>,
    #[serde(default)]
    pub ratio: Option<f64>,
    #[serde(default)]
    pub amount: Option<f64>,
    #[serde(default)]
    pub ccy: Option<String>,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub announced_at: Option<DateTime<Utc>>,
    /// announced | confirmed (default)
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
}

pub const KINDS: [&str; 7] = ["split", "reverse_split", "stock_dividend", "cash_dividend", "spinoff", "merger", "symbol_change"];

impl NewCorporateAction {
    /// Validates kind and the fields the kind needs.
    pub fn validate(&self) -> Result<(), String> {
        if !KINDS.contains(&self.kind.as_str()) {
            return Err(format!("kind must be one of {}", KINDS.join(", ")));
        }
        match self.kind.as_str() {
            "split" | "reverse_split" | "stock_dividend" if !self.ratio.is_some_and(|r| r > 0.0 && r.is_finite()) => {
                Err(format!("{} needs a positive ratio", self.kind))
            }
            "cash_dividend" if !self.amount.is_some_and(|a| a >= 0.0 && a.is_finite()) || self.ccy.is_none() => {
                Err("cash_dividend needs amount >= 0 and ccy".into())
            }
            _ => Ok(()),
        }
    }

    /// Natural key hashed into the deterministic id (`md5(key)::uuid` in Postgres).
    pub fn natural_key(&self) -> String {
        format!("{}|{}|{}|{}", self.instrument_id, self.kind, self.ex_date, self.source_id.as_deref().unwrap_or(""))
    }
}

/// Price multiplier a split-like action applies to prices *before* its ex date to express them
/// in post-action terms (2-for-1 → 0.5). `None` for actions that do not rescale prices.
pub fn split_price_factor(a: &CorporateAction) -> Option<f64> {
    let r = a.ratio.filter(|r| *r > 0.0)?;
    match a.kind.as_str() {
        "split" | "reverse_split" => Some(1.0 / r),
        "stock_dividend" => Some(1.0 / (1.0 + r)),
        _ => None,
    }
}

/// Whether an action is effective for a reader who knows facts up to `known_at`.
pub fn effective(a: &CorporateAction, known_at: DateTime<Utc>) -> bool {
    a.recorded_at <= known_at && !(a.status == "cancelled" && a.cancelled_at.is_some_and(|c| c <= known_at))
}

/// `RefData::adjustment_factor` (spec 12 §4) over a list: the product of split factors of
/// actions with `from < ex_date <= to`. Multiply a raw price observed at `from` by it to compare
/// with prices at `to`. Cash dividends are not included (price-only; see [`adjust_bars`]).
pub fn adjustment_factor(actions: &[CorporateAction], from: NaiveDate, to: NaiveDate) -> f64 {
    actions
        .iter()
        .filter(|a| a.ex_date > from && a.ex_date <= to)
        .filter_map(split_price_factor)
        .product()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AdjustMode {
    /// Splits and stock dividends only (price continuity).
    #[default]
    SplitsOnly,
    /// Splits plus cash dividends reinvested (total-return series).
    TotalReturn,
}

/// One OHLCV bar of a raw series.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BarPoint {
    pub ts: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Back-adjusts a raw bar series (sorted by time) to the basis of its last bar: every bar
/// before an ex date is scaled by the action's factor; volumes scale inversely for splits.
/// Total return also applies `1 − dividend / previous close` for cash dividends (dividend in
/// the series currency). The raw input is not modified (spec 12 §9).
pub fn adjust_bars(bars: &[BarPoint], actions: &[CorporateAction], mode: AdjustMode) -> Vec<BarPoint> {
    let mut out = bars.to_vec();
    let mut acts: Vec<&CorporateAction> = actions.iter().collect();
    acts.sort_by_key(|a| a.ex_date);
    for a in acts {
        // index of the first bar on/after the ex date
        let Some(cut) = out.iter().position(|b| b.ts.date_naive() >= a.ex_date) else { continue };
        if cut == 0 {
            continue;
        }
        let (pf, vf) = match (split_price_factor(a), a.kind.as_str()) {
            (Some(f), _) => (f, 1.0 / f),
            (None, "cash_dividend") if mode == AdjustMode::TotalReturn => {
                let prev = bars[cut - 1].close;
                let d = a.amount.unwrap_or(0.0);
                if prev <= 0.0 || d >= prev {
                    continue;
                }
                (1.0 - d / prev, 1.0)
            }
            _ => continue,
        };
        for b in &mut out[..cut] {
            b.open *= pf;
            b.high *= pf;
            b.low *= pf;
            b.close *= pf;
            b.volume *= vf;
        }
    }
    out
}

fn row_to_action(r: &sqlx::postgres::PgRow) -> CorporateAction {
    let num = |c: &str| r.get::<Option<f64>, _>(c);
    CorporateAction {
        id: r.get("id"),
        instrument_id: r.get("instrument_id"),
        kind: r.get("kind"),
        ex_date: r.get("ex_date"),
        record_date: r.get("record_date"),
        pay_date: r.get("pay_date"),
        ratio: num("ratio"),
        amount: num("amount"),
        ccy: r.get("ccy"),
        source_id: r.get("source_id"),
        announced_at: r.get("announced_at"),
        recorded_at: r.get("recorded_at"),
        status: r.get("status"),
        cancelled_at: r.get("cancelled_at"),
        old_symbol: r.get("old_symbol"),
        new_symbol: r.get("new_symbol"),
    }
}

/// Records an action idempotently (deterministic id `md5(natural key)::uuid`; the natural key is
/// instrument/kind/ex date/source). Returns the id and whether a new row was written.
pub async fn record(pool: &PgPool, a: &NewCorporateAction) -> Result<(Uuid, bool), WarehouseError> {
    let r = sqlx::query(
        "INSERT INTO ref.corporate_action (id, instrument_id, kind, ex_date, record_date, pay_date, ratio, amount, ccy, \
           source_id, announced_at, status, notes) \
         VALUES (md5($1)::uuid, $2, $3, $4, $5, $6, $7::float8::numeric, $8::float8::numeric, $9, $10, $11, $12, $13) \
         ON CONFLICT DO NOTHING",
    )
    .bind(a.natural_key())
    .bind(a.instrument_id)
    .bind(&a.kind)
    .bind(a.ex_date)
    .bind(a.record_date)
    .bind(a.pay_date)
    .bind(a.ratio)
    .bind(a.amount)
    .bind(&a.ccy)
    .bind(&a.source_id)
    .bind(a.announced_at)
    .bind(a.status.as_deref().unwrap_or("confirmed"))
    .bind(&a.notes)
    .execute(pool)
    .await?;
    let id: Uuid = sqlx::query_scalar("SELECT md5($1)::uuid").bind(a.natural_key()).fetch_one(pool).await?;
    Ok((id, r.rows_affected() > 0))
}

/// Marks an action cancelled (kept for as-of reads before the cancellation).
pub async fn cancel(pool: &PgPool, id: Uuid) -> Result<bool, WarehouseError> {
    let r = sqlx::query(
        "UPDATE ref.corporate_action SET status = 'cancelled', cancelled_at = now() WHERE id = $1 AND status <> 'cancelled'",
    )
    .bind(id)
    .execute(pool)
    .await?;
    Ok(r.rows_affected() > 0)
}

/// Actions of an instrument known at `known_at` (default now), oldest ex date first.
pub async fn list_as_of(
    pool: &PgPool,
    instrument: Uuid,
    known_at: Option<DateTime<Utc>>,
) -> Result<Vec<CorporateAction>, WarehouseError> {
    let known_at = known_at.unwrap_or_else(Utc::now);
    let rows = sqlx::query(
        "SELECT id, instrument_id, kind, ex_date, record_date, pay_date, ratio::float8 AS ratio, \
           amount::float8 AS amount, ccy, source_id, announced_at, recorded_at, status, cancelled_at, old_symbol, new_symbol \
         FROM ref.corporate_action WHERE instrument_id = $1 AND recorded_at <= $2 ORDER BY ex_date, kind",
    )
    .bind(instrument)
    .bind(known_at)
    .fetch_all(pool)
    .await?;
    Ok(rows.iter().map(row_to_action).filter(|a| effective(a, known_at)).collect())
}

/// [`adjustment_factor`] straight from Postgres, point-in-time.
pub async fn adjustment_factor_db(
    pool: &PgPool,
    instrument: Uuid,
    from: NaiveDate,
    to: NaiveDate,
    known_at: Option<DateTime<Utc>>,
) -> Result<f64, WarehouseError> {
    Ok(adjustment_factor(&list_as_of(pool, instrument, known_at).await?, from, to))
}

/// Outcome of [`symbol_change`].
#[derive(Debug, Clone, Serialize)]
pub struct SymbolChange {
    pub instrument_id: Uuid,
    pub venue: String,
    pub old_symbol: String,
    pub new_symbol: String,
    pub effective: DateTime<Utc>,
    pub action_id: Uuid,
    pub listings_closed: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum SymbolChangeError {
    #[error("not_found: {0}")]
    NotFound(String),
    #[error("invalid: {0}")]
    Invalid(String),
    #[error(transparent)]
    Warehouse(#[from] WarehouseError),
}

impl From<sqlx::Error> for SymbolChangeError {
    fn from(e: sqlx::Error) -> Self {
        Self::Warehouse(e.into())
    }
}

/// A ticker change on a venue (spec 12 §8 "identifier reused"): the listing `(venue, old)`
/// valid at `effective` is closed at `effective`, `(venue, new)` opens there for the same
/// instrument, matching `<source>_symbol` identifiers follow, and a `symbol_change` corporate
/// action records it. Refused when `new` is already listed for another instrument then.
pub async fn symbol_change(
    pool: &PgPool,
    venue_code: &str,
    old_symbol: &str,
    new_symbol: &str,
    effective: DateTime<Utc>,
) -> Result<SymbolChange, SymbolChangeError> {
    if old_symbol.trim().is_empty() || new_symbol.trim().is_empty() || old_symbol == new_symbol {
        return Err(SymbolChangeError::Invalid("old and new symbol must differ and be non-empty".into()));
    }
    let mut tx = pool.begin().await?;
    let venue_id: Uuid = sqlx::query_scalar("SELECT id FROM ref.venue WHERE code = $1")
        .bind(venue_code)
        .fetch_optional(&mut *tx)
        .await?
        .ok_or_else(|| SymbolChangeError::NotFound(format!("venue {venue_code}")))?;
    let cur = sqlx::query(
        "SELECT instrument_id, source_id, valid_from FROM ref.listing \
         WHERE venue_id = $1 AND venue_symbol = $2 AND valid_from <= $3 AND $3 < valid_to FOR UPDATE",
    )
    .bind(venue_id)
    .bind(old_symbol)
    .bind(effective)
    .fetch_optional(&mut *tx)
    .await?
    .ok_or_else(|| SymbolChangeError::NotFound(format!("listing {venue_code}:{old_symbol} at {effective}")))?;
    let instrument: Uuid = cur.get("instrument_id");
    let source: Option<String> = cur.get("source_id");
    let taken: Option<Uuid> = sqlx::query_scalar(
        "SELECT instrument_id FROM ref.listing WHERE venue_id = $1 AND venue_symbol = $2 AND valid_to > $3 AND instrument_id <> $4 LIMIT 1",
    )
    .bind(venue_id)
    .bind(new_symbol)
    .bind(effective)
    .bind(instrument)
    .fetch_optional(&mut *tx)
    .await?;
    if let Some(other) = taken {
        return Err(SymbolChangeError::Invalid(format!("{venue_code}:{new_symbol} is listed for instrument {other}")));
    }
    let closed = sqlx::query(
        "UPDATE ref.listing SET valid_to = $3 \
         WHERE venue_id = $1 AND venue_symbol = $2 AND valid_from <= $3 AND $3 < valid_to",
    )
    .bind(venue_id)
    .bind(old_symbol)
    .bind(effective)
    .execute(&mut *tx)
    .await?
    .rows_affected();
    sqlx::query(
        "INSERT INTO ref.listing (instrument_id, venue_id, venue_symbol, source_id, valid_from) VALUES ($1, $2, $3, $4, $5) \
         ON CONFLICT DO NOTHING",
    )
    .bind(instrument)
    .bind(venue_id)
    .bind(new_symbol)
    .bind(&source)
    .bind(effective)
    .execute(&mut *tx)
    .await?;
    // symbol-like identifiers (`<source>_symbol`, `yahoo_symbol`, ...) follow the listing
    let schemes: Vec<String> = sqlx::query_scalar(
        "UPDATE ref.identifier SET valid_to = $3 \
         WHERE instrument_id = $1 AND value = $2 AND scheme LIKE '%\\_symbol' AND valid_from <= $3 AND $3 < valid_to \
         RETURNING scheme",
    )
    .bind(instrument)
    .bind(old_symbol)
    .bind(effective)
    .fetch_all(&mut *tx)
    .await?;
    for scheme in &schemes {
        sqlx::query("INSERT INTO ref.identifier (instrument_id, scheme, value, valid_from) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING")
            .bind(instrument)
            .bind(scheme)
            .bind(new_symbol)
            .bind(effective)
            .execute(&mut *tx)
            .await?;
    }
    let a = NewCorporateAction {
        instrument_id: instrument,
        kind: "symbol_change".into(),
        ex_date: effective.date_naive(),
        source_id: source.clone(),
        announced_at: Some(Utc::now()),
        ..Default::default()
    };
    let action_id: Uuid = sqlx::query_scalar("SELECT md5($1)::uuid").bind(a.natural_key()).fetch_one(&mut *tx).await?;
    sqlx::query(
        "INSERT INTO ref.corporate_action (id, instrument_id, kind, ex_date, source_id, announced_at, venue_id, old_symbol, new_symbol) \
         VALUES ($1, $2, 'symbol_change', $3, $4, now(), $5, $6, $7) ON CONFLICT DO NOTHING",
    )
    .bind(action_id)
    .bind(instrument)
    .bind(a.ex_date)
    .bind(&source)
    .bind(venue_id)
    .bind(old_symbol)
    .bind(new_symbol)
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(SymbolChange {
        instrument_id: instrument,
        venue: venue_code.to_owned(),
        old_symbol: old_symbol.to_owned(),
        new_symbol: new_symbol.to_owned(),
        effective,
        action_id,
        listings_closed: closed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn act(kind: &str, ex: (i32, u32, u32), ratio: Option<f64>, amount: Option<f64>) -> CorporateAction {
        CorporateAction {
            id: Uuid::nil(),
            instrument_id: Uuid::nil(),
            kind: kind.into(),
            ex_date: NaiveDate::from_ymd_opt(ex.0, ex.1, ex.2).unwrap(),
            record_date: None,
            pay_date: None,
            ratio,
            amount,
            ccy: Some("USD".into()),
            source_id: None,
            announced_at: None,
            recorded_at: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
            status: "confirmed".into(),
            cancelled_at: None,
            old_symbol: None,
            new_symbol: None,
        }
    }

    fn bars(closes: &[f64]) -> Vec<BarPoint> {
        closes
            .iter()
            .enumerate()
            .map(|(i, c)| BarPoint {
                ts: Utc.with_ymd_and_hms(2026, 3, 1 + i as u32, 21, 0, 0).unwrap(),
                open: *c,
                high: *c,
                low: *c,
                close: *c,
                volume: 100.0,
            })
            .collect()
    }

    #[test]
    fn split_replayed_through_adjusted_series_is_continuous() {
        // 4-for-1 split effective 2026-03-04: raw 400, 404, 408 | 102, 103
        let raw = bars(&[400.0, 404.0, 408.0, 102.0, 103.0]);
        let split = act("split", (2026, 3, 4), Some(4.0), None);
        let adj = adjust_bars(&raw, std::slice::from_ref(&split), AdjustMode::SplitsOnly);
        let closes: Vec<f64> = adj.iter().map(|b| b.close).collect();
        assert_eq!(closes, [100.0, 101.0, 102.0, 102.0, 103.0]);
        assert_eq!(adj[0].volume, 400.0);
        // raw unchanged (spec 12 §9)
        assert_eq!(raw[0].close, 400.0);
        // adjacent returns never jump across the split
        for w in adj.windows(2) {
            assert!((w[1].close / w[0].close - 1.0).abs() < 0.02);
        }
        let d = |m, dd| NaiveDate::from_ymd_opt(2026, m, dd).unwrap();
        assert_eq!(adjustment_factor(std::slice::from_ref(&split), d(3, 1), d(3, 5)), 0.25);
        assert_eq!(adjustment_factor(std::slice::from_ref(&split), d(3, 4), d(3, 5)), 1.0);
    }

    #[test]
    fn reverse_split_and_total_return() {
        let raw = bars(&[1.0, 1.0, 10.0]);
        let rs = act("reverse_split", (2026, 3, 3), Some(0.1), None);
        assert_eq!(adjust_bars(&raw, &[rs], AdjustMode::SplitsOnly)[0].close, 10.0);
        let raw = bars(&[50.0, 50.0, 49.0]);
        let div = act("cash_dividend", (2026, 3, 3), None, Some(1.0));
        let px = adjust_bars(&raw, std::slice::from_ref(&div), AdjustMode::SplitsOnly);
        assert_eq!(px[0].close, 50.0, "price-only ignores dividends");
        let tr = adjust_bars(&raw, &[div], AdjustMode::TotalReturn);
        assert!((tr[1].close - 49.0).abs() < 1e-9, "50 × (1 − 1/50) = 49");
    }

    #[test]
    fn point_in_time_visibility() {
        let mut a = act("split", (2026, 3, 4), Some(2.0), None);
        let t = |d| Utc.with_ymd_and_hms(2026, 2, d, 0, 0, 0).unwrap();
        a.recorded_at = t(10);
        assert!(!effective(&a, t(9)), "announced late: earlier as-of does not see it");
        assert!(effective(&a, t(11)));
        a.status = "cancelled".into();
        a.cancelled_at = Some(t(20));
        assert!(effective(&a, t(15)), "visible before the cancellation was known");
        assert!(!effective(&a, t(21)));
    }

    #[test]
    fn validation_and_ids() {
        let mut n = NewCorporateAction {
            instrument_id: Uuid::nil(),
            kind: "split".into(),
            ex_date: NaiveDate::from_ymd_opt(2026, 3, 4).unwrap(),
            ..Default::default()
        };
        assert!(n.validate().is_err());
        n.ratio = Some(2.0);
        assert!(n.validate().is_ok());
        assert_eq!(n.natural_key(), n.clone().natural_key());
        n.kind = "dividend".into();
        assert!(n.validate().is_err());
    }
}
