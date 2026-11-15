//! Postgres persistence of the organisational books: schema `acct` (migration
//! `20261115100000_accounting.sql`), amounts `NUMERIC(38,18)` like the trading ledger.
//! The journal is append-only: entries insert idempotently by `entry_ref` (same ref + same
//! postings = skipped; same ref + different postings = refused), and nothing updates or deletes
//! a posted entry. The chart syncs by upsert, except an account's `kind`, which is immutable —
//! changing it would flip the orientation of its whole history.

use std::collections::BTreeMap;

use sqlx::{PgPool, Row, postgres::PgPoolOptions};
use uuid::Uuid;

use crate::{Account, AcctError, Chart, Decimal, Entry, Journal, Kind, Posting, money};

#[derive(Debug, thiserror::Error)]
pub enum PgAcctError {
    #[error("postgres: {0}")]
    Sqlx(#[from] sqlx::Error),
    #[error(transparent)]
    Acct(#[from] AcctError),
    #[error("account {0}: kind is immutable ({1} in the database)")]
    KindChange(String, String),
}

#[derive(Clone)]
pub struct PgAccounting {
    pool: PgPool,
}

impl PgAccounting {
    pub async fn connect(url: &str) -> Result<Self, PgAcctError> {
        let pool = PgPoolOptions::new()
            .max_connections(4)
            .acquire_timeout(std::time::Duration::from_secs(3))
            .connect(url)
            .await?;
        Ok(Self::from_pool(pool))
    }

    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Upserts the chart (name, segment, external_ref); refuses a `kind` change.
    pub async fn sync_chart(&self, chart: &Chart) -> Result<(), PgAcctError> {
        for a in chart.accounts() {
            let row = sqlx::query(
                "INSERT INTO acct.account (id, name, kind, segment, external_ref)
                 VALUES ($1, $2, $3, $4, $5)
                 ON CONFLICT (id) DO UPDATE SET name = $2, segment = $4, external_ref = $5
                 RETURNING kind",
            )
            .bind(&a.id)
            .bind(&a.name)
            .bind(a.kind.as_str())
            .bind(&a.segment)
            .bind(&a.external_ref)
            .fetch_one(&self.pool)
            .await?;
            let kind: String = row.get(0);
            if kind != a.kind.as_str() {
                return Err(PgAcctError::KindChange(a.id.clone(), kind));
            }
        }
        Ok(())
    }

    /// Posts one entry; `Ok(false)` when its ref already holds the same postings.
    pub async fn post(&self, e: &Entry) -> Result<bool, PgAcctError> {
        let mut tx = self.pool.begin().await?;
        let inserted = sqlx::query(
            "INSERT INTO acct.entry (id, ts, entry_ref, memo) VALUES ($1, $2, $3, $4)
             ON CONFLICT (entry_ref) DO NOTHING",
        )
        .bind(e.id)
        .bind(e.ts)
        .bind(&e.entry_ref)
        .bind(&e.memo)
        .execute(&mut *tx)
        .await?
        .rows_affected();
        if inserted == 0 {
            let existing = postings_of(&mut tx, e.id).await?;
            tx.rollback().await?;
            return if existing == e.postings {
                Ok(false)
            } else {
                Err(AcctError::Entry(e.entry_ref.clone(), "ref reused with different postings").into())
            };
        }
        for (seq, p) in e.postings.iter().enumerate() {
            sqlx::query("INSERT INTO acct.posting (entry_id, seq, account, amount) VALUES ($1, $2, $3, $4)")
                .bind(e.id)
                .bind(i16::try_from(seq).map_err(|_| AcctError::Entry(e.entry_ref.clone(), "too many postings"))?)
                .bind(&p.account)
                .bind(p.amount)
                .execute(&mut *tx)
                .await?;
        }
        tx.commit().await?;
        Ok(true)
    }

    /// Raw (debit-positive) balance per account with postings.
    pub async fn raw_balances(&self) -> Result<BTreeMap<String, Decimal>, PgAcctError> {
        let rows =
            sqlx::query("SELECT account, SUM(amount) FROM acct.posting GROUP BY account").fetch_all(&self.pool).await?;
        Ok(rows.into_iter().map(|r| (r.get::<String, _>(0), money::round(r.get::<Decimal, _>(1)))).collect())
    }

    /// The stored chart (accounts with an unknown kind are skipped).
    pub async fn load_chart(&self) -> Result<Chart, PgAcctError> {
        let rows = sqlx::query("SELECT id, name, kind, segment, external_ref FROM acct.account ORDER BY id")
            .fetch_all(&self.pool)
            .await?;
        let mut chart = Chart::default();
        for r in rows {
            let Some(kind) = Kind::parse(r.get::<String, _>(2).as_str()) else { continue };
            chart.add(Account { id: r.get(0), name: r.get(1), kind, segment: r.get(3), external_ref: r.get(4) })?;
        }
        Ok(chart)
    }

    /// Rebuilds the in-memory journal from Postgres, replaying entries in `(ts, id)` order.
    pub async fn load(&self) -> Result<Journal, PgAcctError> {
        let chart = self.load_chart().await?;
        let mut journal = Journal::new(chart);
        let rows =
            sqlx::query("SELECT id, ts, entry_ref, memo FROM acct.entry ORDER BY ts, id").fetch_all(&self.pool).await?;
        for r in rows {
            let id: Uuid = r.get(0);
            let mut tx = self.pool.begin().await?;
            let postings = postings_of(&mut tx, id).await?;
            tx.rollback().await?;
            journal.post(Entry { id, ts: r.get(1), entry_ref: r.get(2), memo: r.get(3), postings })?;
        }
        Ok(journal)
    }
}

async fn postings_of(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    entry_id: Uuid,
) -> Result<Vec<Posting>, PgAcctError> {
    let rows = sqlx::query("SELECT account, amount FROM acct.posting WHERE entry_id = $1 ORDER BY seq")
        .bind(entry_id)
        .fetch_all(&mut **tx)
        .await?;
    Ok(rows.into_iter().map(|r| Posting { account: r.get(0), amount: money::round(r.get(1)) }).collect())
}

/// Live tests run only with `DATABASE_URL` set (a Postgres with the migrations applied). They
/// use throwaway refs and remove their own rows, so the shared schema stays clean.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{accounts, dec};

    async fn pg() -> Option<PgAccounting> {
        let url = std::env::var("DATABASE_URL").ok()?;
        PgAccounting::connect(&url).await.ok()
    }

    #[tokio::test]
    async fn chart_entries_and_rebuild_round_trip() {
        let Some(pg) = pg().await else {
            eprintln!("skipped: DATABASE_URL not set");
            return;
        };
        pg.sync_chart(&Chart::default_chart()).await.unwrap();
        let run = chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default().to_string();
        let e = Entry::new(
            chrono::Utc::now(),
            &format!("test:{run}:seed"),
            "test seed",
            vec![(accounts::TRADING_BOOK, dec(12.34)), (accounts::CONTRIBUTED, dec(-12.34))],
        )
        .unwrap();
        assert!(pg.post(&e).await.unwrap());
        assert!(!pg.post(&e).await.unwrap(), "idempotent by ref");
        let mut tampered = e.clone();
        tampered.postings[0].amount = dec(1.0);
        tampered.postings[1].amount = dec(-1.0);
        assert!(pg.post(&tampered).await.is_err(), "same ref, different postings");

        let bal = pg.raw_balances().await.unwrap();
        assert!(bal[accounts::TRADING_BOOK] >= dec(12.34));
        let j = pg.load().await.unwrap();
        assert_eq!(j.trial_balance(), Decimal::ZERO);
        assert!(j.entries().iter().any(|x| x.entry_ref == e.entry_ref));

        sqlx::query("DELETE FROM acct.entry WHERE entry_ref LIKE $1")
            .bind(format!("test:{run}:%"))
            .execute(pg.pool())
            .await
            .unwrap();
    }
}
