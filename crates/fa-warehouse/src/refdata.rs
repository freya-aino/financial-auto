//! Reference-data reads and the quarantine review loop (spec 12 §4 "CRUD for manual
//! overrides", spec 10 §5.3; D-054): instrument search over `ref.instrument` /
//! `ref.identifier` / `ref.listing`, the open quarantine queue, and resolving a quarantined
//! provisional instrument (activate in place or merge into an existing instrument).
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::WarehouseError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentifierRow {
    pub scheme: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListingRow {
    pub venue: String,
    pub venue_symbol: String,
    pub source_id: Option<String>,
}

/// One search hit: instrument plus its current identifiers and listings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstrumentHit {
    pub id: Uuid,
    pub asset_class: String,
    pub kind: String,
    pub name: String,
    pub status: String,
    pub base_ccy: Option<String>,
    pub quote_ccy: Option<String>,
    pub recorded_at: DateTime<Utc>,
    pub identifiers: Vec<IdentifierRow>,
    pub listings: Vec<ListingRow>,
}

/// Case-insensitive search of names, identifier values and venue symbols. An exact match on a
/// symbol or identifier ranks first, then prefix matches, then substring matches; `status`
/// filters (`active`, `provisional`, `merged`, …). Empty `q` lists the newest instruments.
pub async fn search_instruments(
    pool: &PgPool,
    q: &str,
    status: Option<&str>,
    limit: i64,
) -> Result<Vec<InstrumentHit>, WarehouseError> {
    let q = q.trim().to_lowercase();
    let pattern = format!("%{}%", q.replace('%', "\\%").replace('_', "\\_"));
    let prefix = format!("{}%", q.replace('%', "\\%").replace('_', "\\_"));
    let rows = sqlx::query(
        "WITH hits AS ( \
            SELECT i.id, \
                   MIN(CASE WHEN $1 = '' THEN 3 \
                            WHEN lower(coalesce(l.venue_symbol, '')) = $1 OR lower(coalesce(d.value, '')) = $1 THEN 0 \
                            WHEN lower(coalesce(l.venue_symbol, '')) LIKE $3 OR lower(coalesce(d.value, '')) LIKE $3 \
                                 OR lower(i.name) LIKE $3 THEN 1 \
                            ELSE 2 END) AS rank \
              FROM ref.instrument i \
              LEFT JOIN ref.listing l ON l.instrument_id = i.id AND now() >= l.valid_from AND now() < l.valid_to \
              LEFT JOIN ref.identifier d ON d.instrument_id = i.id AND now() >= d.valid_from AND now() < d.valid_to \
             WHERE ($1 = '' OR lower(i.name) LIKE $2 OR lower(coalesce(l.venue_symbol, '')) LIKE $2 \
                    OR lower(coalesce(d.value, '')) LIKE $2) \
               AND ($4::text IS NULL OR i.status = $4) \
             GROUP BY i.id) \
         SELECT i.id, i.asset_class, i.kind, i.name, i.status, i.base_ccy, i.quote_ccy, i.recorded_at, \
                coalesce((SELECT json_agg(json_build_object('scheme', d.scheme, 'value', d.value) ORDER BY d.scheme) \
                            FROM ref.identifier d WHERE d.instrument_id = i.id \
                             AND now() >= d.valid_from AND now() < d.valid_to), '[]'::json) AS identifiers, \
                coalesce((SELECT json_agg(json_build_object('venue', v.code, 'venue_symbol', l.venue_symbol, \
                                                            'source_id', l.source_id) ORDER BY v.code) \
                            FROM ref.listing l JOIN ref.venue v ON v.id = l.venue_id \
                           WHERE l.instrument_id = i.id AND now() >= l.valid_from AND now() < l.valid_to), '[]'::json) AS listings \
           FROM hits h JOIN ref.instrument i ON i.id = h.id \
          ORDER BY h.rank, i.recorded_at DESC, i.name \
          LIMIT $5",
    )
    .bind(&q)
    .bind(&pattern)
    .bind(&prefix)
    .bind(status)
    .bind(limit.clamp(1, 500))
    .fetch_all(pool)
    .await?;
    Ok(rows
        .iter()
        .map(|r| InstrumentHit {
            id: r.get("id"),
            asset_class: r.get("asset_class"),
            kind: r.get("kind"),
            name: r.get("name"),
            status: r.get("status"),
            base_ccy: r.get("base_ccy"),
            quote_ccy: r.get("quote_ccy"),
            recorded_at: r.get("recorded_at"),
            identifiers: serde_json::from_value(r.get::<Value, _>("identifiers")).unwrap_or_default(),
            listings: serde_json::from_value(r.get::<Value, _>("listings")).unwrap_or_default(),
        })
        .collect())
}

/// One `ingest.quarantine` row with the provisional instrument it stands for.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantineRow {
    pub source_id: String,
    pub venue_symbol: String,
    pub instrument_id: Option<Uuid>,
    pub instrument_name: Option<String>,
    pub instrument_status: Option<String>,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    pub count: i64,
    pub sample_raw_ref: Option<String>,
    pub status: String,
    pub resolved_at: Option<DateTime<Utc>>,
    pub resolution: Option<Value>,
}

/// Quarantine rows, newest activity first; `status` filters (`open` by default in the API).
pub async fn quarantine_list(
    pool: &PgPool,
    status: Option<&str>,
    limit: i64,
) -> Result<Vec<QuarantineRow>, WarehouseError> {
    let rows = sqlx::query(
        "SELECT q.source_id, q.venue_symbol, q.instrument_id, i.name AS instrument_name, i.status AS instrument_status, \
                q.first_seen, q.last_seen, q.count, q.sample_raw_ref, q.status, q.resolved_at, q.resolution \
           FROM ingest.quarantine q LEFT JOIN ref.instrument i ON i.id = q.instrument_id \
          WHERE ($1::text IS NULL OR q.status = $1) ORDER BY q.last_seen DESC LIMIT $2",
    )
    .bind(status)
    .bind(limit.clamp(1, 1000))
    .fetch_all(pool)
    .await?;
    Ok(rows.iter().map(quarantine_row).collect())
}

fn quarantine_row(r: &sqlx::postgres::PgRow) -> QuarantineRow {
    QuarantineRow {
        source_id: r.get("source_id"),
        venue_symbol: r.get("venue_symbol"),
        instrument_id: r.get("instrument_id"),
        instrument_name: r.get("instrument_name"),
        instrument_status: r.get("instrument_status"),
        first_seen: r.get("first_seen"),
        last_seen: r.get("last_seen"),
        count: r.get("count"),
        sample_raw_ref: r.get("sample_raw_ref"),
        status: r.get("status"),
        resolved_at: r.get("resolved_at"),
        resolution: r.get("resolution"),
    }
}

/// Open quarantine rows whose provisional instrument is `id` (normally one; several when more
/// than one source provisioned the same instrument id, which the UUIDv5 scheme prevents).
pub async fn quarantine_for_instrument(pool: &PgPool, id: Uuid) -> Result<Vec<QuarantineRow>, WarehouseError> {
    let rows = sqlx::query(
        "SELECT q.source_id, q.venue_symbol, q.instrument_id, i.name AS instrument_name, i.status AS instrument_status, \
                q.first_seen, q.last_seen, q.count, q.sample_raw_ref, q.status, q.resolved_at, q.resolution \
           FROM ingest.quarantine q LEFT JOIN ref.instrument i ON i.id = q.instrument_id \
          WHERE q.instrument_id = $1 ORDER BY q.last_seen DESC",
    )
    .bind(id)
    .fetch_all(pool)
    .await?;
    Ok(rows.iter().map(quarantine_row).collect())
}

/// What the reviewer decided. With `instrument_id` the provisional instrument is **merged**
/// into that existing one (listings and identifiers re-pointed, provisional row closed with
/// status `merged`); otherwise it is **activated** in place with the given facts. `ignore`
/// only closes the quarantine row (`ignored`) and leaves the instrument provisional.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Resolution {
    pub instrument_id: Option<Uuid>,
    pub figi: Option<String>,
    pub isin: Option<String>,
    pub asset_class: Option<String>,
    pub kind: Option<String>,
    pub name: Option<String>,
    pub base_ccy: Option<String>,
    pub quote_ccy: Option<String>,
    #[serde(default)]
    pub ignore: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolveOutcome {
    /// The instrument events now resolve to.
    pub instrument_id: Uuid,
    /// The provisional id that was merged away (merge only).
    pub merged_from: Option<Uuid>,
    pub action: String, // activated | merged | ignored
    pub quarantine_rows: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum ResolveError {
    #[error("no quarantine row for instrument {0}")]
    NotFound(Uuid),
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Warehouse(#[from] WarehouseError),
}

impl From<sqlx::Error> for ResolveError {
    fn from(e: sqlx::Error) -> Self {
        Self::Warehouse(e.into())
    }
}

/// Resolves the quarantine of provisional instrument `provisional` in one transaction.
pub async fn resolve_quarantine(
    pool: &PgPool,
    provisional: Uuid,
    res: &Resolution,
) -> Result<ResolveOutcome, ResolveError> {
    let mut tx = pool.begin().await?;
    let open: Vec<(String, String)> = sqlx::query_as(
        "SELECT source_id, venue_symbol FROM ingest.quarantine WHERE instrument_id = $1 AND status = 'open' FOR UPDATE",
    )
    .bind(provisional)
    .fetch_all(&mut *tx)
    .await?;
    if open.is_empty() {
        return Err(ResolveError::NotFound(provisional));
    }
    let exists: Option<(String,)> = sqlx::query_as("SELECT status FROM ref.instrument WHERE id = $1")
        .bind(provisional)
        .fetch_optional(&mut *tx)
        .await?;
    let Some((prov_status,)) = exists else {
        return Err(ResolveError::Invalid(format!("instrument {provisional} does not exist")));
    };

    let (target, action) = if res.ignore {
        (provisional, "ignored")
    } else if let Some(target) = res.instrument_id.filter(|t| *t != provisional) {
        let t: Option<(String,)> = sqlx::query_as("SELECT status FROM ref.instrument WHERE id = $1")
            .bind(target)
            .fetch_optional(&mut *tx)
            .await?;
        if t.is_none() {
            return Err(ResolveError::Invalid(format!("merge target {target} does not exist")));
        }
        if prov_status != "provisional" {
            return Err(ResolveError::Invalid(format!("instrument {provisional} is {prov_status}, not provisional")));
        }
        // Re-point listings and identifiers; drop the ones the target already has.
        sqlx::query(
            "DELETE FROM ref.listing l WHERE l.instrument_id = $1 AND EXISTS (SELECT 1 FROM ref.listing t \
               WHERE t.instrument_id = $2 AND t.venue_id = l.venue_id AND t.venue_symbol = l.venue_symbol AND t.valid_from = l.valid_from)",
        )
        .bind(provisional)
        .bind(target)
        .execute(&mut *tx)
        .await?;
        sqlx::query("UPDATE ref.listing SET instrument_id = $2 WHERE instrument_id = $1")
            .bind(provisional)
            .bind(target)
            .execute(&mut *tx)
            .await?;
        sqlx::query("UPDATE ref.identifier SET instrument_id = $2 WHERE instrument_id = $1")
            .bind(provisional)
            .bind(target)
            .execute(&mut *tx)
            .await?;
        sqlx::query("UPDATE ref.instrument SET status = 'merged', valid_to = now() WHERE id = $1")
            .bind(provisional)
            .execute(&mut *tx)
            .await?;
        sqlx::query("UPDATE ingest.backfill_cursor SET instrument_id = $2 WHERE instrument_id = $1 AND NOT EXISTS \
               (SELECT 1 FROM ingest.backfill_cursor c WHERE c.instrument_id = $2 AND c.source_id = ingest.backfill_cursor.source_id \
                 AND c.timescale = ingest.backfill_cursor.timescale AND c.range_start = ingest.backfill_cursor.range_start)")
            .bind(provisional)
            .bind(target)
            .execute(&mut *tx)
            .await?;
        (target, "merged")
    } else {
        sqlx::query(
            "UPDATE ref.instrument SET status = 'active', asset_class = coalesce($2, asset_class), kind = coalesce($3, kind), \
                    name = coalesce($4, name), base_ccy = coalesce($5, base_ccy), quote_ccy = coalesce($6, quote_ccy) WHERE id = $1",
        )
        .bind(provisional)
        .bind(&res.asset_class)
        .bind(&res.kind)
        .bind(&res.name)
        .bind(&res.base_ccy)
        .bind(&res.quote_ccy)
        .execute(&mut *tx)
        .await?;
        (provisional, "activated")
    };
    if !res.ignore {
        for (scheme, value) in [("figi", &res.figi), ("isin", &res.isin)] {
            if let Some(v) = value.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
                // A global identifier belongs to exactly one instrument: refuse to steal it silently.
                let owner: Option<(Uuid,)> = sqlx::query_as(
                    "SELECT instrument_id FROM ref.identifier WHERE scheme = $1 AND value = $2 \
                       AND now() >= valid_from AND now() < valid_to AND instrument_id <> $3 LIMIT 1",
                )
                .bind(scheme)
                .bind(v)
                .bind(target)
                .fetch_optional(&mut *tx)
                .await?;
                if let Some((other,)) = owner {
                    return Err(ResolveError::Invalid(format!(
                        "{scheme} {v} already belongs to instrument {other}; merge into it with instrument_id instead"
                    )));
                }
                sqlx::query("INSERT INTO ref.identifier (instrument_id, scheme, value) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING")
                    .bind(target)
                    .bind(scheme)
                    .bind(v)
                    .execute(&mut *tx)
                    .await?;
            }
        }
    }
    let resolution = json!({
        "action": action, "instrument_id": target, "merged_from": (action == "merged").then_some(provisional),
        "figi": res.figi, "isin": res.isin, "asset_class": res.asset_class, "kind": res.kind, "name": res.name,
    });
    let closed = sqlx::query(
        "UPDATE ingest.quarantine SET status = $2, resolved_at = now(), resolution = $3 WHERE instrument_id = $1 AND status = 'open'",
    )
    .bind(provisional)
    .bind(if res.ignore { "ignored" } else { "resolved" })
    .bind(&resolution)
    .execute(&mut *tx)
    .await?
    .rows_affected();
    tx.commit().await?;
    Ok(ResolveOutcome {
        instrument_id: target,
        merged_from: (action == "merged").then_some(provisional),
        action: action.into(),
        quarantine_rows: closed as usize,
    })
}

/// Live tests (`DATABASE_URL`, compose Postgres with migration `20261110000040` applied).
#[cfg(test)]
mod tests {
    use super::*;

    async fn pool() -> Option<PgPool> {
        let url = std::env::var("DATABASE_URL").ok()?;
        Some(sqlx::postgres::PgPoolOptions::new().max_connections(2).connect(&url).await.expect("connect"))
    }

    /// A provisional instrument with venue, listing, identifier and open quarantine row, as the
    /// Go resolver creates them (D-050). Returns (source, symbol, instrument id, venue id).
    async fn provision(pool: &PgPool, tag: &str) -> (String, String, Uuid, Uuid) {
        let source = format!("test_{tag}");
        let symbol = format!("TQ{tag}");
        let inst = Uuid::now_v7();
        let venue = Uuid::now_v7();
        sqlx::query(
            "INSERT INTO catalog.source (id, name, category, auth, status) VALUES ($1, 'Test', 'test', 'none', 'stub')",
        )
        .bind(&source)
        .execute(pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO ref.venue (id, code, name, timezone, kind) VALUES ($1, $2, 'Test venue', 'UTC', 'otc')",
        )
        .bind(venue)
        .bind(format!("v{tag}"))
        .execute(pool)
        .await
        .unwrap();
        sqlx::query("INSERT INTO ref.instrument (id, asset_class, kind, name, status) VALUES ($1, 'unknown', 'unknown', $2, 'provisional')")
            .bind(inst)
            .bind(format!("{symbol} ({source})"))
            .execute(pool)
            .await
            .unwrap();
        sqlx::query(
            "INSERT INTO ref.listing (instrument_id, venue_id, venue_symbol, source_id) VALUES ($1, $2, $3, $4)",
        )
        .bind(inst)
        .bind(venue)
        .bind(&symbol)
        .bind(&source)
        .execute(pool)
        .await
        .unwrap();
        sqlx::query("INSERT INTO ref.identifier (instrument_id, scheme, value) VALUES ($1, 'test_symbol', $2)")
            .bind(inst)
            .bind(&symbol)
            .execute(pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO ingest.quarantine (source_id, venue_symbol, instrument_id, status) VALUES ($1, $2, $3, 'open')")
            .bind(&source)
            .bind(&symbol)
            .bind(inst)
            .execute(pool)
            .await
            .unwrap();
        (source, symbol, inst, venue)
    }

    async fn cleanup(pool: &PgPool, source: &str, insts: &[Uuid], venue: Uuid) {
        sqlx::query("DELETE FROM ingest.quarantine WHERE source_id = $1").bind(source).execute(pool).await.unwrap();
        for i in insts {
            sqlx::query("DELETE FROM ref.identifier WHERE instrument_id = $1").bind(i).execute(pool).await.unwrap();
            sqlx::query("DELETE FROM ref.listing WHERE instrument_id = $1").bind(i).execute(pool).await.unwrap();
            sqlx::query("DELETE FROM ref.instrument WHERE id = $1").bind(i).execute(pool).await.unwrap();
        }
        sqlx::query("DELETE FROM ref.venue WHERE id = $1").bind(venue).execute(pool).await.unwrap();
        sqlx::query("DELETE FROM catalog.source WHERE id = $1").bind(source).execute(pool).await.unwrap();
    }

    #[tokio::test]
    async fn search_quarantine_and_activate() {
        let Some(pool) = pool().await else { return };
        let tag = Uuid::now_v7().simple().to_string();
        let (source, symbol, inst, venue) = provision(&pool, &tag).await;

        let hits = search_instruments(&pool, &symbol.to_lowercase(), None, 10).await.unwrap();
        assert_eq!(hits.len(), 1, "exact symbol match");
        assert_eq!(hits[0].id, inst);
        assert_eq!(hits[0].status, "provisional");
        assert_eq!(hits[0].listings[0].venue_symbol, symbol);
        assert_eq!(hits[0].identifiers[0].scheme, "test_symbol");
        let by_status = search_instruments(&pool, &tag[..8], Some("provisional"), 50).await.unwrap();
        assert!(by_status.iter().any(|h| h.id == inst));
        assert!(search_instruments(&pool, &symbol, Some("active"), 10).await.unwrap().is_empty());

        let open = quarantine_list(&pool, Some("open"), 1000).await.unwrap();
        let row = open.iter().find(|r| r.instrument_id == Some(inst)).expect("quarantine row listed");
        assert_eq!((row.source_id.as_str(), row.venue_symbol.as_str()), (source.as_str(), symbol.as_str()));
        assert_eq!(row.instrument_status.as_deref(), Some("provisional"));

        let isin = format!("TS{}", tag[..10].to_uppercase());
        let out = resolve_quarantine(
            &pool,
            inst,
            &Resolution {
                isin: Some(isin.clone()),
                asset_class: Some("equity".into()),
                kind: Some("equity".into()),
                name: Some("Resolved Inc".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!((out.action.as_str(), out.instrument_id, out.quarantine_rows), ("activated", inst, 1));
        let hit = &search_instruments(&pool, &symbol, None, 10).await.unwrap()[0];
        assert_eq!((hit.status.as_str(), hit.name.as_str(), hit.kind.as_str()), ("active", "Resolved Inc", "equity"));
        assert!(hit.identifiers.iter().any(|i| i.scheme == "isin" && i.value == isin));
        let rows = quarantine_for_instrument(&pool, inst).await.unwrap();
        assert_eq!(rows[0].status, "resolved");
        assert!(rows[0].resolved_at.is_some());
        assert!(matches!(
            resolve_quarantine(&pool, inst, &Resolution::default()).await,
            Err(ResolveError::NotFound(_))
        ));

        cleanup(&pool, &source, &[inst], venue).await;
    }

    #[tokio::test]
    async fn merge_into_existing_instrument() {
        let Some(pool) = pool().await else { return };
        let tag = Uuid::now_v7().simple().to_string();
        let (source, symbol, prov, venue) = provision(&pool, &tag).await;
        let target = Uuid::now_v7();
        sqlx::query("INSERT INTO ref.instrument (id, asset_class, kind, name, status) VALUES ($1, 'equity', 'equity', 'Target Corp', 'active')")
            .bind(target)
            .execute(&pool)
            .await
            .unwrap();

        let out = resolve_quarantine(
            &pool,
            prov,
            &Resolution { instrument_id: Some(target), figi: Some("BBG000B9XRY4".into()), ..Default::default() },
        )
        .await
        .unwrap();
        assert_eq!((out.action.as_str(), out.instrument_id, out.merged_from), ("merged", target, Some(prov)));
        let hits = search_instruments(&pool, &symbol, None, 10).await.unwrap();
        let t = hits.iter().find(|h| h.id == target).expect("target now carries the listing");
        assert!(t.listings.iter().any(|l| l.venue_symbol == symbol && l.source_id.as_deref() == Some(source.as_str())));
        assert!(t.identifiers.iter().any(|i| i.scheme == "figi"));
        assert!(t.identifiers.iter().any(|i| i.scheme == "test_symbol"));
        let (status,): (String,) = sqlx::query_as("SELECT status FROM ref.instrument WHERE id = $1")
            .bind(prov)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(status, "merged");
        // the provisional id no longer matches the live symbol (valid_to closed, no listing)
        assert!(!hits.iter().any(|h| h.id == prov && !h.listings.is_empty()));

        let bad = resolve_quarantine(&pool, Uuid::now_v7(), &Resolution::default()).await;
        assert!(matches!(bad, Err(ResolveError::NotFound(_))));
        // An identifier owned by another instrument cannot be claimed while activating.
        let (_, _, other, venue2) = provision(&pool, &format!("{tag}b")).await;
        let taken =
            resolve_quarantine(&pool, other, &Resolution { figi: Some("BBG000B9XRY4".into()), ..Default::default() })
                .await;
        assert!(matches!(taken, Err(ResolveError::Invalid(ref m)) if m.contains("already belongs")), "{taken:?}");
        cleanup(&pool, &format!("test_{tag}b"), &[other], venue2).await;

        cleanup(&pool, &source, &[prov, target], venue).await;
    }

    #[tokio::test]
    async fn ignore_closes_without_activating() {
        let Some(pool) = pool().await else { return };
        let tag = Uuid::now_v7().simple().to_string();
        let (source, _symbol, inst, venue) = provision(&pool, &tag).await;
        let out = resolve_quarantine(&pool, inst, &Resolution { ignore: true, ..Default::default() }).await.unwrap();
        assert_eq!(out.action, "ignored");
        let rows = quarantine_for_instrument(&pool, inst).await.unwrap();
        assert_eq!((rows[0].status.as_str(), rows[0].instrument_status.as_deref()), ("ignored", Some("provisional")));
        cleanup(&pool, &source, &[inst], venue).await;
    }
}
