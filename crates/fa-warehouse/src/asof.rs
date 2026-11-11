//! Point-in-time reference reads (spec 12 §4 `RefData`, §5 bitemporal-lite, §9; spec 20 §6.3).
//!
//! Every query takes the real-world instant `at` and an optional knowledge instant `known_at`
//! (default: now). A fact is visible when `valid_from <= at < valid_to` and
//! `recorded_at <= known_at`, so a backtest reading "as of t" passes `at = known_at = t` and
//! never sees mappings, specs or rates recorded later.
use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::WarehouseError;

/// Result of a symbol resolution (spec 12 §8: an ambiguous mapping is never picked silently).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum Resolved {
    None,
    One { instrument_id: Uuid },
    Ambiguous { instrument_ids: Vec<Uuid> },
}

impl Resolved {
    pub fn one(&self) -> Option<Uuid> {
        match self {
            Self::One { instrument_id } => Some(*instrument_id),
            _ => None,
        }
    }

    fn from_ids(mut ids: Vec<Uuid>) -> Self {
        ids.sort();
        ids.dedup();
        match ids.len() {
            0 => Self::None,
            1 => Self::One { instrument_id: ids[0] },
            _ => Self::Ambiguous { instrument_ids: ids },
        }
    }
}

fn known(k: Option<DateTime<Utc>>) -> DateTime<Utc> {
    k.unwrap_or_else(Utc::now)
}

/// `RefData::resolve` (spec 12 §4): venue code + venue symbol at `at` → instrument.
pub async fn resolve(
    pool: &PgPool,
    venue_code: &str,
    symbol: &str,
    at: DateTime<Utc>,
    known_at: Option<DateTime<Utc>>,
) -> Result<Resolved, WarehouseError> {
    let ids: Vec<Uuid> = sqlx::query_scalar(
        "SELECT l.instrument_id FROM ref.listing l JOIN ref.venue v ON v.id = l.venue_id \
         JOIN ref.instrument i ON i.id = l.instrument_id \
         WHERE v.code = $1 AND l.venue_symbol = $2 AND l.valid_from <= $3 AND $3 < l.valid_to \
           AND l.recorded_at <= $4 AND i.recorded_at <= $4 AND i.status <> 'merged'",
    )
    .bind(venue_code)
    .bind(symbol)
    .bind(at)
    .bind(known(known_at))
    .fetch_all(pool)
    .await?;
    Ok(Resolved::from_ids(ids))
}

/// Resolution by connector (source id) and symbol, the ingestion-side key (D-050).
pub async fn resolve_source_symbol(
    pool: &PgPool,
    source_id: &str,
    symbol: &str,
    at: DateTime<Utc>,
    known_at: Option<DateTime<Utc>>,
) -> Result<Resolved, WarehouseError> {
    let ids: Vec<Uuid> = sqlx::query_scalar(
        "SELECT l.instrument_id FROM ref.listing l JOIN ref.instrument i ON i.id = l.instrument_id \
         WHERE l.source_id = $1 AND l.venue_symbol = $2 AND l.valid_from <= $3 AND $3 < l.valid_to \
           AND l.recorded_at <= $4 AND i.status <> 'merged'",
    )
    .bind(source_id)
    .bind(symbol)
    .bind(at)
    .bind(known(known_at))
    .fetch_all(pool)
    .await?;
    Ok(Resolved::from_ids(ids))
}

/// Resolution by a global identifier (FIGI, ISIN, LEI, contract address ...).
pub async fn resolve_identifier(
    pool: &PgPool,
    scheme: &str,
    value: &str,
    at: DateTime<Utc>,
    known_at: Option<DateTime<Utc>>,
) -> Result<Resolved, WarehouseError> {
    let ids: Vec<Uuid> = sqlx::query_scalar(
        "SELECT instrument_id FROM ref.identifier WHERE scheme = $1 AND value = $2 \
         AND valid_from <= $3 AND $3 < valid_to AND recorded_at <= $4",
    )
    .bind(scheme)
    .bind(value)
    .bind(at)
    .bind(known(known_at))
    .fetch_all(pool)
    .await?;
    Ok(Resolved::from_ids(ids))
}

#[derive(Debug, Clone, Serialize)]
pub struct ListingAsOf {
    pub venue: String,
    pub mic: Option<String>,
    pub venue_symbol: String,
    pub source_id: Option<String>,
    pub valid_from: DateTime<Utc>,
    pub valid_to: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct IdentifierAsOf {
    pub scheme: String,
    pub value: String,
    pub valid_from: DateTime<Utc>,
    pub valid_to: DateTime<Utc>,
}

/// `RefData::instrument` (spec 12 §4) at a point in time.
#[derive(Debug, Clone, Serialize)]
pub struct InstrumentAsOf {
    pub id: Uuid,
    pub asset_class: String,
    pub kind: String,
    pub name: String,
    pub base_ccy: Option<String>,
    pub quote_ccy: Option<String>,
    pub issuer_id: Option<Uuid>,
    pub status: String,
    pub recorded_at: DateTime<Utc>,
    pub listings: Vec<ListingAsOf>,
    pub identifiers: Vec<IdentifierAsOf>,
}

/// Postgres `timestamptz` ±infinity decode as `None` through sqlx; map them to chrono bounds.
fn ts_or(r: &sqlx::postgres::PgRow, col: &str, inf: DateTime<Utc>) -> DateTime<Utc> {
    r.try_get::<Option<DateTime<Utc>>, _>(col).ok().flatten().unwrap_or(inf)
}

/// Instrument with the listings and identifiers valid at `at` and known at `known_at`.
pub async fn instrument(
    pool: &PgPool,
    id: Uuid,
    at: DateTime<Utc>,
    known_at: Option<DateTime<Utc>>,
) -> Result<Option<InstrumentAsOf>, WarehouseError> {
    let k = known(known_at);
    let Some(r) = sqlx::query(
        "SELECT id, asset_class, kind, name, base_ccy, quote_ccy, issuer_id, status, recorded_at FROM ref.instrument \
         WHERE id = $1 AND recorded_at <= $2 AND valid_from <= $3 AND $3 < valid_to",
    )
    .bind(id)
    .bind(k)
    .bind(at)
    .fetch_optional(pool)
    .await?
    else {
        return Ok(None);
    };
    // ±infinity bounds come back as text so they survive the round trip
    let listings = sqlx::query(
        "SELECT v.code, v.mic, l.venue_symbol, l.source_id, \
           CASE WHEN isfinite(l.valid_from) THEN l.valid_from END AS valid_from, \
           CASE WHEN isfinite(l.valid_to) THEN l.valid_to END AS valid_to \
         FROM ref.listing l JOIN ref.venue v ON v.id = l.venue_id \
         WHERE l.instrument_id = $1 AND l.valid_from <= $2 AND $2 < l.valid_to AND l.recorded_at <= $3 ORDER BY v.code",
    )
    .bind(id)
    .bind(at)
    .bind(k)
    .fetch_all(pool)
    .await?
    .iter()
    .map(|l| ListingAsOf {
        venue: l.get("code"),
        mic: l.get("mic"),
        venue_symbol: l.get("venue_symbol"),
        source_id: l.get("source_id"),
        valid_from: ts_or(l, "valid_from", DateTime::<Utc>::MIN_UTC),
        valid_to: ts_or(l, "valid_to", DateTime::<Utc>::MAX_UTC),
    })
    .collect();
    let identifiers = sqlx::query(
        "SELECT scheme, value, CASE WHEN isfinite(valid_from) THEN valid_from END AS valid_from, \
           CASE WHEN isfinite(valid_to) THEN valid_to END AS valid_to \
         FROM ref.identifier WHERE instrument_id = $1 AND valid_from <= $2 AND $2 < valid_to AND recorded_at <= $3 \
         ORDER BY scheme, value",
    )
    .bind(id)
    .bind(at)
    .bind(k)
    .fetch_all(pool)
    .await?
    .iter()
    .map(|i| IdentifierAsOf {
        scheme: i.get("scheme"),
        value: i.get("value"),
        valid_from: ts_or(i, "valid_from", DateTime::<Utc>::MIN_UTC),
        valid_to: ts_or(i, "valid_to", DateTime::<Utc>::MAX_UTC),
    })
    .collect();
    Ok(Some(InstrumentAsOf {
        id: r.get("id"),
        asset_class: r.get("asset_class"),
        kind: r.get("kind"),
        name: r.get("name"),
        base_ccy: r.get("base_ccy"),
        quote_ccy: r.get("quote_ccy"),
        issuer_id: r.get("issuer_id"),
        status: r.get("status"),
        recorded_at: r.get("recorded_at"),
        listings,
        identifiers,
    }))
}

/// `RefData::contract_spec` (spec 12 §4).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ContractSpec {
    pub instrument_id: Uuid,
    pub venue: String,
    pub tick_size: Option<f64>,
    pub lot_size: Option<f64>,
    pub min_notional: Option<f64>,
    pub multiplier: f64,
    pub expiry: Option<DateTime<Utc>>,
    pub settlement_ccy: Option<String>,
    pub shortable: Option<bool>,
    pub max_leverage: Option<f64>,
    pub fee_schedule_ref: Option<String>,
}

/// Contract specs valid at `at` (one per venue, or only `venue_code`'s).
pub async fn contract_specs(
    pool: &PgPool,
    id: Uuid,
    venue_code: Option<&str>,
    at: DateTime<Utc>,
    known_at: Option<DateTime<Utc>>,
) -> Result<Vec<ContractSpec>, WarehouseError> {
    let rows = sqlx::query(
        "SELECT DISTINCT ON (c.venue_id) c.instrument_id, v.code, c.tick_size::float8 AS tick_size, \
           c.lot_size::float8 AS lot_size, c.min_notional::float8 AS min_notional, c.multiplier::float8 AS multiplier, \
           c.expiry, c.settlement_ccy, c.shortable, c.max_leverage::float8 AS max_leverage, c.fee_schedule_ref \
         FROM ref.contract_spec c JOIN ref.venue v ON v.id = c.venue_id \
         WHERE c.instrument_id = $1 AND ($2::text IS NULL OR v.code = $2) \
           AND c.valid_from <= $3 AND $3 < c.valid_to AND c.recorded_at <= $4 \
         ORDER BY c.venue_id, c.valid_from DESC, c.recorded_at DESC",
    )
    .bind(id)
    .bind(venue_code)
    .bind(at)
    .bind(known(known_at))
    .fetch_all(pool)
    .await?;
    Ok(rows
        .iter()
        .map(|r| ContractSpec {
            instrument_id: r.get("instrument_id"),
            venue: r.get("code"),
            tick_size: r.get("tick_size"),
            lot_size: r.get("lot_size"),
            min_notional: r.get("min_notional"),
            multiplier: r.get::<Option<f64>, _>("multiplier").unwrap_or(1.0),
            expiry: r.get("expiry"),
            settlement_ccy: r.get("settlement_ccy"),
            shortable: r.get("shortable"),
            max_leverage: r.get("max_leverage"),
            fee_schedule_ref: r.get("fee_schedule_ref"),
        })
        .collect())
}

/// Input for [`upsert_contract_spec`]; a new `valid_from` closes the previous version.
#[derive(Debug, Clone, Default, serde::Deserialize, Serialize)]
pub struct NewContractSpec {
    pub instrument_id: Uuid,
    pub venue: String,
    pub tick_size: Option<f64>,
    pub lot_size: Option<f64>,
    pub min_notional: Option<f64>,
    pub multiplier: Option<f64>,
    pub expiry: Option<DateTime<Utc>>,
    pub settlement_ccy: Option<String>,
    pub shortable: Option<bool>,
    pub max_leverage: Option<f64>,
    pub fee_schedule_ref: Option<String>,
    pub valid_from: Option<DateTime<Utc>>,
}

/// Writes a contract-spec version: the version open at `valid_from` (default now) is closed there
/// and the new one starts (history kept, spec 12 bitemporal-lite). Returns false for an unknown venue.
pub async fn upsert_contract_spec(pool: &PgPool, s: &NewContractSpec) -> Result<bool, WarehouseError> {
    let from = s.valid_from.unwrap_or_else(Utc::now);
    let mut tx = pool.begin().await?;
    let Some(venue_id): Option<Uuid> =
        sqlx::query_scalar("SELECT id FROM ref.venue WHERE code = $1").bind(&s.venue).fetch_optional(&mut *tx).await?
    else {
        return Ok(false);
    };
    sqlx::query(
        "UPDATE ref.contract_spec SET valid_to = $3 WHERE instrument_id = $1 AND venue_id = $2 \
         AND valid_from < $3 AND $3 < valid_to",
    )
    .bind(s.instrument_id)
    .bind(venue_id)
    .bind(from)
    .execute(&mut *tx)
    .await?;
    sqlx::query(
        "INSERT INTO ref.contract_spec (instrument_id, venue_id, tick_size, lot_size, min_notional, multiplier, expiry, \
           settlement_ccy, shortable, max_leverage, fee_schedule_ref, valid_from) \
         VALUES ($1, $2, $3::float8::numeric, $4::float8::numeric, $5::float8::numeric, coalesce($6, 1)::float8::numeric, $7, $8, $9, \
           $10::float8::numeric, $11, $12) \
         ON CONFLICT (instrument_id, venue_id, valid_from) DO UPDATE SET tick_size = excluded.tick_size, \
           lot_size = excluded.lot_size, min_notional = excluded.min_notional, multiplier = excluded.multiplier, \
           expiry = excluded.expiry, settlement_ccy = excluded.settlement_ccy, shortable = excluded.shortable, \
           max_leverage = excluded.max_leverage, fee_schedule_ref = excluded.fee_schedule_ref, recorded_at = now()",
    )
    .bind(s.instrument_id)
    .bind(venue_id)
    .bind(s.tick_size)
    .bind(s.lot_size)
    .bind(s.min_notional)
    .bind(s.multiplier)
    .bind(s.expiry)
    .bind(&s.settlement_ccy)
    .bind(s.shortable)
    .bind(s.max_leverage)
    .bind(&s.fee_schedule_ref)
    .bind(from)
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(true)
}

/// Adds an identifier (FIGI/ISIN/LEI ...) valid from `valid_from` (default now). Refused (false)
/// when the same scheme/value is held by another instrument at that time (spec 12 §8).
pub async fn add_identifier(
    pool: &PgPool,
    instrument_id: Uuid,
    scheme: &str,
    value: &str,
    valid_from: Option<DateTime<Utc>>,
) -> Result<bool, WarehouseError> {
    let from = valid_from.unwrap_or_else(Utc::now);
    let other: Option<Uuid> = sqlx::query_scalar(
        "SELECT instrument_id FROM ref.identifier WHERE scheme = $1 AND value = $2 AND valid_to > $3 AND instrument_id <> $4 LIMIT 1",
    )
    .bind(scheme)
    .bind(value)
    .bind(from)
    .bind(instrument_id)
    .fetch_optional(pool)
    .await?;
    if other.is_some() {
        return Ok(false);
    }
    sqlx::query("INSERT INTO ref.identifier (instrument_id, scheme, value, valid_from) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING")
        .bind(instrument_id)
        .bind(scheme)
        .bind(value)
        .bind(from)
        .execute(pool)
        .await?;
    Ok(true)
}

/// `RefData::fx_to_eur` (spec 12 §4, §9): EUR per unit of `ccy` at `at` as known at `known_at`
/// (latest rate with `ts <= at`), with its timestamp and source. EUR is 1.
pub async fn fx_to_eur(
    pool: &PgPool,
    ccy: &str,
    at: DateTime<Utc>,
    known_at: Option<DateTime<Utc>>,
) -> Result<Option<(f64, DateTime<Utc>, String)>, WarehouseError> {
    if ccy.eq_ignore_ascii_case("EUR") {
        return Ok(Some((1.0, at, "identity".into())));
    }
    let r = sqlx::query(
        "SELECT rate_to_eur::float8 AS r, ts, source_id FROM ref.fx_rate WHERE ccy = $1 AND ts <= $2 AND recorded_at <= $3 \
         ORDER BY ts DESC, (source_id = 'ecb_daily') DESC LIMIT 1",
    )
    .bind(ccy.to_ascii_uppercase())
    .bind(at)
    .bind(known(known_at))
    .fetch_optional(pool)
    .await?;
    Ok(r.map(|r| (r.get("r"), r.get("ts"), r.get("source_id"))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_never_picks_silently() {
        let (a, b) = (Uuid::from_u128(1), Uuid::from_u128(2));
        assert_eq!(Resolved::from_ids(vec![]), Resolved::None);
        assert_eq!(Resolved::from_ids(vec![a, a]).one(), Some(a));
        assert_eq!(Resolved::from_ids(vec![b, a]), Resolved::Ambiguous { instrument_ids: vec![a, b] });
        assert_eq!(Resolved::from_ids(vec![b, a]).one(), None);
    }
}
