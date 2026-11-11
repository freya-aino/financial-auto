//! FX for EUR accounting (spec 12 §5 `ref.fx_rate`, §6 "ECB daily reference rate as official
//! EUR conversion for accounting"; spec 42 §3, §6, §8). Rows are keyed `(ccy, ts, source_id)`
//! with EUR as the implicit base: `rate_to_eur` is how many EUR one unit of `ccy` buys.
//!
//! [`parse_ecb_daily`] reads the ECB `eurofxref-daily.xml` (EUR → ccy quotes, one `Cube time`
//! block) into [`FxRate`]s with `source = ecb_daily`; [`upsert_rates`] writes them idempotently;
//! [`rate_as_of`] / [`load_recent`] read them back; [`FxTable`] is the pure in-memory as-of
//! table the risk-engine marks and fills against (latest rate with `as_of ≤ t`, staleness
//! flagged beyond a maximum age so spec 42 §8 "use last known rate flagged stale" holds).
//!
//! Intraday (spec 42 §6 "live mid for intraday"): an FX-pair instrument quoted against EUR
//! ([`EurCross`], from `ref.instrument` via [`load_eur_crosses`], e.g. Yahoo `EURUSD=X`) turns
//! each market-data price into an [`FxRate`] whose source is the feed's `catalog.source` id
//! (`yahoo`), so the same as-of table and `ref.fx_rate` hold daily and intraday rates side by
//! side and the latest one wins.
use std::collections::BTreeMap;

use chrono::{DateTime, NaiveDate, Utc};
use quick_xml::events::Event;
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};

use crate::WarehouseError;

/// `catalog.source` id / `ref.fx_rate.source_id` of the ECB daily reference rates
/// (spec 12 §7 `fx.accounting_source`).
pub const ECB_SOURCE: &str = "ecb_daily";
/// The ECB's daily reference-rate feed (published on TARGET working days around 16:00 CET).
pub const ECB_DAILY_URL: &str = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml";
/// Source token of the identity conversion for EUR-denominated amounts.
pub const EUR_SOURCE: &str = "eur";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FxRate {
    /// ISO currency code of the quote currency (`USD`, `GBP`, …).
    pub ccy: String,
    /// Validity instant (`ref.fx_rate.ts`); ECB daily rates are stored at 00:00 UTC of their day.
    pub as_of: DateTime<Utc>,
    /// EUR per one unit of `ccy`.
    pub rate_to_eur: f64,
    pub source: String,
}

#[derive(Debug, thiserror::Error)]
pub enum FxError {
    #[error("ecb xml: {0}")]
    Xml(String),
    #[error("ecb xml has no <Cube time=...> block")]
    NoDate,
    #[error("ecb xml has no <Cube currency rate/> entries")]
    NoRates,
}

/// One day of ECB reference rates: `rates` are EUR → ccy as published.
#[derive(Debug, Clone, PartialEq)]
pub struct EcbDaily {
    pub date: NaiveDate,
    pub rates: Vec<(String, f64)>,
}

impl EcbDaily {
    /// Inverts the published EUR → ccy quotes into `rate_to_eur` rows dated 00:00 UTC.
    pub fn to_rates(&self) -> Vec<FxRate> {
        let as_of = self.date.and_hms_opt(0, 0, 0).expect("midnight").and_utc();
        self.rates
            .iter()
            .filter(|(_, r)| *r > 0.0 && r.is_finite())
            .map(|(ccy, r)| FxRate { ccy: ccy.clone(), as_of, rate_to_eur: 1.0 / r, source: ECB_SOURCE.into() })
            .collect()
    }
}

/// Parses `eurofxref-daily.xml`. Tolerates namespaces and whitespace; the outer
/// `<Cube time='YYYY-MM-DD'>` dates every inner `<Cube currency rate/>`.
pub fn parse_ecb_daily(xml: &str) -> Result<EcbDaily, FxError> {
    let mut reader = quick_xml::Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut date: Option<NaiveDate> = None;
    let mut rates = Vec::new();
    loop {
        let ev = reader.read_event().map_err(|e| FxError::Xml(e.to_string()))?;
        match ev {
            Event::Start(ref e) | Event::Empty(ref e) if e.local_name().as_ref() == b"Cube" => {
                let (mut time, mut ccy, mut rate) = (None, None, None);
                for a in e.attributes().flatten() {
                    let v = String::from_utf8_lossy(&a.value).into_owned();
                    match a.key.local_name().as_ref() {
                        b"time" => time = Some(v),
                        b"currency" => ccy = Some(v),
                        b"rate" => rate = Some(v),
                        _ => {}
                    }
                }
                if let Some(t) = time {
                    date = Some(
                        NaiveDate::parse_from_str(&t, "%Y-%m-%d")
                            .map_err(|e| FxError::Xml(format!("time {t:?}: {e}")))?,
                    );
                }
                if let (Some(c), Some(r)) = (ccy, rate) {
                    let r: f64 = r.trim().parse().map_err(|e| FxError::Xml(format!("rate {r:?} for {c}: {e}")))?;
                    rates.push((c.trim().to_uppercase(), r));
                }
            }
            Event::Eof => break,
            _ => {}
        }
    }
    let date = date.ok_or(FxError::NoDate)?;
    if rates.is_empty() {
        return Err(FxError::NoRates);
    }
    Ok(EcbDaily { date, rates })
}

/// Upserts rates into `ref.fx_rate`; `(ccy, ts, source_id)` is the idempotency key, so a
/// re-fetch of the same day is a no-op. Returns the number of new rows.
pub async fn upsert_rates(pool: &PgPool, rates: &[FxRate]) -> Result<usize, WarehouseError> {
    let mut tx = pool.begin().await?;
    let mut n = 0;
    for r in rates {
        n += sqlx::query(
            "INSERT INTO ref.fx_rate (ccy, ts, rate_to_eur, source_id) VALUES ($1, $2, $3::float8::numeric, $4) \
             ON CONFLICT (ccy, ts, source_id) DO NOTHING",
        )
        .bind(&r.ccy)
        .bind(r.as_of)
        .bind(r.rate_to_eur)
        .bind(&r.source)
        .execute(&mut *tx)
        .await?
        .rows_affected() as usize;
    }
    tx.commit().await?;
    Ok(n)
}

fn row(r: &sqlx::postgres::PgRow) -> FxRate {
    FxRate { ccy: r.get("ccy"), as_of: r.get("ts"), rate_to_eur: r.get("rate"), source: r.get("source_id") }
}

/// Latest rate for `ccy` with `ts ≤ at` (any source; `source` filters to one).
pub async fn rate_as_of(
    pool: &PgPool,
    ccy: &str,
    at: DateTime<Utc>,
    source: Option<&str>,
) -> Result<Option<FxRate>, WarehouseError> {
    let r = sqlx::query(
        "SELECT ccy, ts, rate_to_eur::float8 AS rate, source_id FROM ref.fx_rate \
         WHERE ccy = $1 AND ts <= $2 AND ($3::text IS NULL OR source_id = $3) ORDER BY ts DESC LIMIT 1",
    )
    .bind(ccy.to_uppercase())
    .bind(at)
    .bind(source)
    .fetch_optional(pool)
    .await?;
    Ok(r.as_ref().map(row))
}

/// Rates of one source newer than `since` plus, per currency, the latest one at or before it
/// (so an [`FxTable`] built from the result answers as-of lookups inside the window).
pub async fn load_recent(pool: &PgPool, source: &str, since: DateTime<Utc>) -> Result<Vec<FxRate>, WarehouseError> {
    let rows = sqlx::query(
        "SELECT ccy, ts, rate_to_eur::float8 AS rate, source_id FROM ref.fx_rate WHERE source_id = $1 AND ts > $2 \
         UNION ALL \
         SELECT DISTINCT ON (ccy) ccy, ts, rate_to_eur::float8 AS rate, source_id FROM ref.fx_rate \
         WHERE source_id = $1 AND ts <= $2 ORDER BY ccy, ts DESC",
    )
    .bind(source)
    .bind(since)
    .fetch_all(pool)
    .await?;
    Ok(rows.iter().map(row).collect())
}

/// An FX-pair instrument with EUR on exactly one side: its price is an intraday EUR rate for
/// the other currency (spec 42 §6 `fx.intraday_source`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EurCross {
    pub instrument: fa_core::InstrumentId,
    /// The non-EUR currency.
    pub ccy: String,
    /// `EUR/ccy` (price = ccy per EUR, e.g. `EURUSD=X`) when true; `ccy/EUR` otherwise.
    pub eur_is_base: bool,
    /// `catalog.source` id of the feed (`ref.listing.source_id`), stored as `ref.fx_rate.source_id`.
    pub source: String,
}

impl EurCross {
    /// The EUR rate a price of this pair at `at` implies (`None` for a non-positive price).
    pub fn rate(&self, price: f64, at: DateTime<Utc>) -> Option<FxRate> {
        (price > 0.0 && price.is_finite()).then(|| FxRate {
            ccy: self.ccy.clone(),
            as_of: at,
            rate_to_eur: if self.eur_is_base { 1.0 / price } else { price },
            source: self.source.clone(),
        })
    }
}

/// FX-pair instruments (`asset_class = 'fx'`) with EUR on exactly one side and a current listing
/// with a source, one per instrument (the first source by id when several list it).
pub async fn load_eur_crosses(pool: &PgPool) -> Result<Vec<EurCross>, WarehouseError> {
    let rows = sqlx::query(
        "SELECT DISTINCT ON (i.id) i.id, upper(i.base_ccy) AS base, upper(i.quote_ccy) AS quote, l.source_id \
         FROM ref.instrument i JOIN ref.listing l ON l.instrument_id = i.id \
         WHERE i.asset_class = 'fx' AND i.base_ccy IS NOT NULL AND i.quote_ccy IS NOT NULL \
           AND (upper(i.base_ccy) = 'EUR') <> (upper(i.quote_ccy) = 'EUR') \
           AND l.source_id IS NOT NULL AND l.valid_from <= now() AND l.valid_to > now() \
         ORDER BY i.id, l.source_id",
    )
    .fetch_all(pool)
    .await?;
    Ok(rows
        .iter()
        .map(|r| {
            let (base, quote): (String, String) = (r.get("base"), r.get("quote"));
            let eur_is_base = base == fa_core::BASE_CURRENCY;
            EurCross {
                instrument: fa_core::InstrumentId(r.get("id")),
                ccy: if eur_is_base { quote } else { base },
                eur_is_base,
                source: r.get("source_id"),
            }
        })
        .collect())
}

/// Result of an as-of conversion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FxLookup {
    pub rate_to_eur: f64,
    pub source: String,
    /// Validity instant of the rate used (`None` for EUR or when no rate exists at all).
    pub as_of: Option<DateTime<Utc>>,
    /// The rate is older than the allowed age at `t`, or dated after `t` (no earlier rate
    /// existed), or missing altogether (rate 1.0, source `missing`).
    pub stale: bool,
}

/// In-memory as-of table: per currency, rates sorted by `as_of`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FxTable {
    by_ccy: BTreeMap<String, Vec<FxRate>>,
}

impl FxTable {
    pub fn from_rates(rates: impl IntoIterator<Item = FxRate>) -> Self {
        let mut t = Self::default();
        for r in rates {
            t.insert(r);
        }
        t
    }

    /// Inserts (or replaces the same `(ccy, as_of, source)`) keeping the per-currency order.
    pub fn insert(&mut self, r: FxRate) {
        let ccy = r.ccy.to_uppercase();
        let v = self.by_ccy.entry(ccy).or_default();
        if let Some(x) = v.iter_mut().find(|x| x.as_of == r.as_of && x.source == r.source) {
            x.rate_to_eur = r.rate_to_eur;
            return;
        }
        let pos = v.partition_point(|x| x.as_of <= r.as_of);
        v.insert(pos, r);
    }

    /// Every rate, per currency in `as_of` order.
    pub fn rates(&self) -> impl Iterator<Item = &FxRate> {
        self.by_ccy.values().flatten()
    }

    pub fn currencies(&self) -> impl Iterator<Item = &str> {
        self.by_ccy.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.by_ccy.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.by_ccy.values().all(Vec::is_empty)
    }

    /// Latest rate of `ccy` dated at or before `at`.
    pub fn latest_at(&self, ccy: &str, at: DateTime<Utc>) -> Option<&FxRate> {
        let v = self.by_ccy.get(&ccy.to_uppercase())?;
        let n = v.partition_point(|x| x.as_of <= at);
        if n == 0 { None } else { v.get(n - 1) }
    }

    /// Newest rate of `ccy` from `source`.
    pub fn newest_of(&self, ccy: &str, source: &str) -> Option<&FxRate> {
        self.by_ccy.get(&ccy.to_uppercase()).and_then(|v| v.iter().rev().find(|r| r.source == source))
    }

    /// Drops the rates of `source` dated before `before` (bounds an intraday series in memory).
    pub fn prune(&mut self, source: &str, before: DateTime<Utc>) {
        for v in self.by_ccy.values_mut() {
            v.retain(|r| r.source != source || r.as_of >= before);
        }
    }

    /// Newest rate of `ccy` regardless of time.
    pub fn newest(&self, ccy: &str) -> Option<&FxRate> {
        self.by_ccy.get(&ccy.to_uppercase()).and_then(|v| v.last())
    }

    /// `ccy` → EUR as of `at`: EUR itself is 1.0 (never stale); otherwise the latest rate dated
    /// `≤ at`, stale when older than `max_age`; without an earlier rate the earliest later one
    /// is used and flagged stale; without any rate 1.0 flagged stale (`source = missing`).
    pub fn convert(&self, ccy: &str, at: DateTime<Utc>, max_age: chrono::Duration) -> FxLookup {
        let ccy = ccy.trim().to_uppercase();
        if ccy.is_empty() || ccy == fa_core::BASE_CURRENCY {
            return FxLookup { rate_to_eur: 1.0, source: EUR_SOURCE.into(), as_of: None, stale: false };
        }
        if let Some(r) = self.latest_at(&ccy, at) {
            return FxLookup {
                rate_to_eur: r.rate_to_eur,
                source: r.source.clone(),
                as_of: Some(r.as_of),
                stale: at - r.as_of > max_age,
            };
        }
        match self.by_ccy.get(&ccy).and_then(|v| v.first()) {
            Some(r) => {
                FxLookup { rate_to_eur: r.rate_to_eur, source: r.source.clone(), as_of: Some(r.as_of), stale: true }
            }
            None => FxLookup { rate_to_eur: 1.0, source: "missing".into(), as_of: None, stale: true },
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    const FIXTURE: &str = include_str!("../testdata/eurofxref-daily.xml");

    #[test]
    fn ecb_fixture_parses_into_inverted_rates() {
        let d = parse_ecb_daily(FIXTURE).unwrap();
        assert_eq!(d.date, NaiveDate::from_ymd_opt(2026, 11, 7).unwrap());
        assert_eq!(d.rates.len(), 30);
        assert_eq!(d.rates[0], ("USD".into(), 1.1547));
        let rates = d.to_rates();
        let usd = rates.iter().find(|r| r.ccy == "USD").unwrap();
        assert!((usd.rate_to_eur - 1.0 / 1.1547).abs() < 1e-12);
        assert_eq!(usd.source, ECB_SOURCE);
        assert_eq!(usd.as_of, Utc.with_ymd_and_hms(2026, 11, 7, 0, 0, 0).unwrap());
        assert!(matches!(parse_ecb_daily("<Cube><Cube currency='USD' rate='1'/></Cube>"), Err(FxError::NoDate)));
        assert!(parse_ecb_daily("<Cube time='2026-11-07'><Cube currency='USD' rate='x'/></Cube>").is_err());
        assert!(matches!(parse_ecb_daily("<Cube time='2026-11-07'></Cube>"), Err(FxError::NoRates)));
        assert!(parse_ecb_daily("<Cube time='2026-11-07'><Cube currency='USD' rate='1'></Cub").is_err(), "truncated");
    }

    #[test]
    fn table_answers_as_of_and_flags_staleness() {
        let day = |d: u32| Utc.with_ymd_and_hms(2026, 11, d, 0, 0, 0).unwrap();
        let rate =
            |d: u32, r: f64| FxRate { ccy: "usd".into(), as_of: day(d), rate_to_eur: r, source: ECB_SOURCE.into() };
        let mut t = FxTable::from_rates([rate(5, 0.86), rate(3, 0.85), rate(4, 0.855)]);
        t.insert(rate(4, 0.8551)); // same key replaces
        assert_eq!(t.len(), 3);
        assert_eq!(t.currencies().collect::<Vec<_>>(), vec!["USD"]);
        let h72 = chrono::Duration::hours(72);
        let at = day(4) + chrono::Duration::hours(10);
        let l = t.convert("USD", at, h72);
        assert_eq!((l.rate_to_eur, l.stale, l.as_of), (0.8551, false, Some(day(4))));
        assert_eq!(t.convert("USD", day(9), h72).rate_to_eur, 0.86);
        assert!(t.convert("USD", day(9), h72).stale, "4 days after the last rate");
        assert!(!t.convert("USD", day(8), h72).stale, "exactly 72 h is still fresh");
        let early = t.convert("USD", day(1), h72);
        assert_eq!((early.rate_to_eur, early.stale), (0.85, true), "no earlier rate: earliest later one, stale");
        let eur = t.convert("EUR", day(1), h72);
        assert_eq!((eur.rate_to_eur, eur.stale, eur.source.as_str()), (1.0, false, EUR_SOURCE));
        assert_eq!(t.convert("", day(1), h72).rate_to_eur, 1.0);
        let none = t.convert("GBP", day(5), h72);
        assert_eq!((none.rate_to_eur, none.stale, none.source.as_str()), (1.0, true, "missing"));
        assert_eq!(t.newest("usd").unwrap().as_of, day(5));
        assert!(t.latest_at("USD", day(2)).is_none());
    }

    #[test]
    fn eur_cross_rates_merge_with_daily_rates() {
        let day = Utc.with_ymd_and_hms(2026, 11, 11, 0, 0, 0).unwrap();
        let x = EurCross {
            instrument: fa_core::InstrumentId(uuid::Uuid::nil()),
            ccy: "USD".into(),
            eur_is_base: true,
            source: "yahoo".into(),
        };
        let r = x.rate(1.25, day + chrono::Duration::hours(10)).unwrap();
        assert_eq!((r.rate_to_eur, r.ccy.as_str(), r.source.as_str()), (0.8, "USD", "yahoo"));
        assert!(x.rate(0.0, day).is_none() && x.rate(f64::NAN, day).is_none());
        let inv = EurCross { eur_is_base: false, ..x.clone() };
        assert_eq!(inv.rate(0.8, day).unwrap().rate_to_eur, 0.8);
        let ecb = FxRate { ccy: "USD".into(), as_of: day, rate_to_eur: 0.81, source: ECB_SOURCE.into() };
        let mut t = FxTable::from_rates([ecb, r]);
        let h72 = chrono::Duration::hours(72);
        assert_eq!(t.convert("USD", day + chrono::Duration::hours(9), h72).source, ECB_SOURCE);
        let l = t.convert("USD", day + chrono::Duration::hours(11), h72);
        assert_eq!((l.rate_to_eur, l.source.as_str()), (0.8, "yahoo"), "the intraday rate is newer");
        assert_eq!(t.newest_of("usd", ECB_SOURCE).unwrap().rate_to_eur, 0.81);
        t.prune("yahoo", day + chrono::Duration::hours(12));
        assert_eq!(t.len(), 1, "only the intraday row is pruned");
        assert_eq!(t.newest("USD").unwrap().source, ECB_SOURCE);
    }

    /// Needs `DATABASE_URL` (compose Postgres with migration 20261110000060 applied).
    #[tokio::test]
    async fn upsert_is_idempotent_and_as_of_reads_back() {
        let Ok(url) = std::env::var("DATABASE_URL") else { return };
        let pool = sqlx::postgres::PgPoolOptions::new().max_connections(2).connect(&url).await.unwrap();
        let ccy = format!("T{}", &uuid::Uuid::now_v7().simple().to_string()[..2]).to_uppercase();
        let d1 = Utc.with_ymd_and_hms(2099, 1, 1, 0, 0, 0).unwrap();
        let d2 = Utc.with_ymd_and_hms(2099, 1, 2, 0, 0, 0).unwrap();
        let rates = vec![
            FxRate { ccy: ccy.clone(), as_of: d1, rate_to_eur: 0.5, source: ECB_SOURCE.into() },
            FxRate { ccy: ccy.clone(), as_of: d2, rate_to_eur: 0.6, source: ECB_SOURCE.into() },
        ];
        assert_eq!(upsert_rates(&pool, &rates).await.unwrap(), 2);
        assert_eq!(upsert_rates(&pool, &rates).await.unwrap(), 0, "same (ccy, ts, source) is a no-op");
        let mid = d1 + chrono::Duration::hours(12);
        assert_eq!(rate_as_of(&pool, &ccy, mid, None).await.unwrap().unwrap().rate_to_eur, 0.5);
        assert_eq!(rate_as_of(&pool, &ccy, d2, Some(ECB_SOURCE)).await.unwrap().unwrap().rate_to_eur, 0.6);
        assert!(rate_as_of(&pool, &ccy, d1 - chrono::Duration::days(1), None).await.unwrap().is_none());
        let recent = load_recent(&pool, ECB_SOURCE, mid).await.unwrap();
        let mine: Vec<&FxRate> = recent.iter().filter(|r| r.ccy == ccy).collect();
        assert_eq!(mine.len(), 2, "newer row plus the latest at-or-before `since`: {mine:?}");
        let t = FxTable::from_rates(recent);
        assert_eq!(t.convert(&ccy, mid, chrono::Duration::hours(72)).rate_to_eur, 0.5);
        sqlx::query("DELETE FROM ref.fx_rate WHERE ccy = $1").bind(&ccy).execute(&pool).await.unwrap();
    }

    /// Read-only; needs `DATABASE_URL`. The live reference data lists Yahoo `EURUSD=X`.
    #[tokio::test]
    async fn eur_crosses_load_from_reference_data() {
        let Ok(url) = std::env::var("DATABASE_URL") else { return };
        let pool = sqlx::postgres::PgPoolOptions::new().max_connections(1).connect(&url).await.unwrap();
        let xs = load_eur_crosses(&pool).await.unwrap();
        for x in &xs {
            assert_ne!(x.ccy, fa_core::BASE_CURRENCY);
            assert!(!x.source.is_empty());
        }
    }
}
