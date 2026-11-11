//! Point-in-time fundamentals (`wh.fundamental`, spec 20 §5.1/§6.3): facts are vintaged by the
//! filing date (written by the ingestor from SEC EDGAR XBRL company facts, migration
//! `20261113000012`), so an as-of read returns, per `(period, unit)`, the latest vintage
//! published at or before `as_of` — a later restatement is invisible before its filing.
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

use crate::WarehouseError;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, sqlx::FromRow)]
pub struct Fundamental {
    pub issuer_cik: Option<String>,
    pub field: String,
    pub unit: Option<String>,
    pub period: String,
    pub period_start: Option<NaiveDate>,
    pub period_end: Option<NaiveDate>,
    pub value: f64,
    pub vintage: DateTime<Utc>,
    pub form: Option<String>,
    pub accession: Option<String>,
    pub source_id: String,
}

/// Normalises a CIK to the stored 10-digit form.
pub fn cik10(cik: &str) -> String {
    let d: String = cik.trim().trim_start_matches("CIK").chars().filter(char::is_ascii_digit).collect();
    format!("{:0>10}", d.trim_start_matches('0'))
}

/// `field` of issuer `cik` as known at `as_of`: per `(period, unit)` the latest vintage
/// `≤ as_of`, ordered by period end then period.
pub async fn as_of(
    pool: &PgPool,
    cik: &str,
    field: &str,
    as_of: DateTime<Utc>,
) -> Result<Vec<Fundamental>, WarehouseError> {
    Ok(sqlx::query_as::<_, Fundamental>(
        "SELECT * FROM (SELECT DISTINCT ON (period, coalesce(unit, '')) issuer_cik, field, unit, period, period_start, \
                period_end, value, vintage, form, accession, source_id \
              FROM wh.fundamental WHERE issuer_cik = $1 AND field = $2 AND vintage <= $3 \
              ORDER BY period, coalesce(unit, ''), vintage DESC) x \
         ORDER BY period_end NULLS LAST, period, unit",
    )
    .bind(cik10(cik))
    .bind(field)
    .bind(as_of)
    .fetch_all(pool)
    .await?)
}

/// Fields stored for an issuer with their fact counts.
pub async fn fields(pool: &PgPool, cik: &str) -> Result<Vec<(String, i64)>, WarehouseError> {
    Ok(sqlx::query_as("SELECT field, count(*) FROM wh.fundamental WHERE issuer_cik = $1 GROUP BY 1 ORDER BY 1")
        .bind(cik10(cik))
        .fetch_all(pool)
        .await?)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn cik_normalises() {
        assert_eq!(cik10("320193"), "0000320193");
        assert_eq!(cik10("CIK0000320193"), "0000320193");
    }

    /// spec 20 §9 "as-of queries never return data published after as_of", as a property over
    /// late publications: random restatements of random periods, queried at every instant around
    /// each vintage, must return exactly the latest vintage ≤ t per period (checked in Rust).
    #[tokio::test]
    async fn as_of_never_sees_late_publications() {
        let Some(url) = std::env::var("DATABASE_URL").ok() else { return };
        let pool = PgPool::connect(&url).await.unwrap();
        let cik = format!("99{:08}", Utc::now().timestamp_micros() % 100_000_000);
        let t0 = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut rnd = |n: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % n
        };
        let mut rows = Vec::new();
        for i in 0..60 {
            let period = format!("P{}", rnd(6));
            let vintage = t0 + chrono::Duration::days(rnd(400) as i64);
            rows.push((period, vintage, i as f64));
        }
        // unique (period, vintage): later duplicates are no-ops in the table too
        rows.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
        rows.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
        for (period, vintage, value) in &rows {
            sqlx::query(
                "INSERT INTO wh.fundamental (issuer_cik, field, unit, period, vintage, value, source_id) \
                 VALUES ($1, 'us-gaap:Test', 'USD', $2, $3, $4, 'sec_edgar')",
            )
            .bind(&cik)
            .bind(period)
            .bind(vintage)
            .bind(value)
            .execute(&pool)
            .await
            .unwrap();
        }
        let mut probes: Vec<DateTime<Utc>> = rows
            .iter()
            .flat_map(|r| [r.1 - chrono::Duration::microseconds(1), r.1, r.1 + chrono::Duration::hours(1)])
            .collect();
        probes.push(t0 - chrono::Duration::days(1));
        for t in probes {
            let got = as_of(&pool, &cik, "us-gaap:Test", t).await.unwrap();
            let mut want: BTreeMap<&str, (DateTime<Utc>, f64)> = BTreeMap::new();
            for (p, v, val) in rows.iter().filter(|r| r.1 <= t) {
                if want.get(p.as_str()).is_none_or(|w| w.0 < *v) {
                    want.insert(p, (*v, *val));
                }
            }
            assert!(got.iter().all(|f| f.vintage <= t), "late publication visible at {t}");
            let got_map: BTreeMap<&str, (DateTime<Utc>, f64)> =
                got.iter().map(|f| (f.period.as_str(), (f.vintage, f.value))).collect();
            assert_eq!(got_map, want, "at {t}");
        }
        assert_eq!(fields(&pool, &cik).await.unwrap(), vec![("us-gaap:Test".to_owned(), rows.len() as i64)]);
        sqlx::query("DELETE FROM wh.fundamental WHERE issuer_cik = $1").bind(&cik).execute(&pool).await.unwrap();
    }
}
