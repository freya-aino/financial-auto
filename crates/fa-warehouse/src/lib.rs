//! PostgreSQL warehouse access (specs 12, 20; D-013): reference data, catalog, macro vintages,
//! the append-only [`audit`] log (spec 42), the instrument search / quarantine review
//! functions in [`refdata`] (D-054), the instrument → asset class / venue / currency / source
//! lookup in [`venues`] and the EUR accounting FX table in [`fx`] (D-061); point-in-time
//! reference reads in [`asof`], trading calendars in [`calendar`] and corporate actions,
//! adjusted series and symbol changes in [`corpactions`] (spec 12 §4); the `ops` schema (alert
//! history, job runs, config snapshots) in [`ops`] (spec 20 §5.4).
pub mod asof;
pub mod audit;
pub mod calendar;
pub mod corpactions;
pub mod fundamentals;
pub mod fx;
pub mod ops;
pub mod refdata;
pub mod venues;

use std::path::Path;

use chrono::{DateTime, Utc};
use fa_core::{AssetClass, InstrumentId};
use sqlx::{PgPool, postgres::PgPoolOptions};
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum WarehouseError {
    #[error("sql: {0}")]
    Sql(#[from] sqlx::Error),
    #[error("migrate: {0}")]
    Migrate(#[from] sqlx::migrate::MigrateError),
    #[error("reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing sources yaml: {0}")]
    Yaml(#[from] serde_yaml::Error),
}

#[derive(Clone)]
pub struct Warehouse {
    pool: PgPool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Instrument {
    pub id: InstrumentId,
    pub asset_class: AssetClass,
    pub name: String,
    pub figi: Option<String>,
    pub isin: Option<String>,
    pub currency: String,
}

/// One `config/sources.yaml` entry (spec 11; same shape as `go/pkg/catalog.Source`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Source {
    pub id: String,
    pub name: String,
    pub category: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub asset_classes: Vec<String>,
    #[serde(default)]
    pub transport: Vec<String>,
    pub auth: String,
    #[serde(default = "default_status")]
    pub status: String,
    #[serde(default)]
    pub tos_notes: String,
    #[serde(default)]
    pub notes: String,
}

fn default_status() -> String {
    "planned".into()
}

#[derive(Debug, Clone, serde::Deserialize)]
struct SourcesFile {
    #[allow(dead_code)]
    #[serde(default)]
    version: u32,
    sources: Vec<Source>,
}

/// Parses a `sources.yaml` document.
pub fn parse_sources(yaml: &str) -> Result<Vec<Source>, WarehouseError> {
    Ok(serde_yaml::from_str::<SourcesFile>(yaml)?.sources)
}

/// Reads and parses `config/sources.yaml`.
pub fn load_sources(path: &Path) -> Result<Vec<Source>, WarehouseError> {
    let text = std::fs::read_to_string(path)
        .map_err(|source| WarehouseError::Io { path: path.display().to_string(), source })?;
    parse_sources(&text)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SourceRow {
    pub id: String,
    pub name: String,
    pub category: String,
    pub status: String,
}

/// A macro observation as known at `vintage` (point-in-time, spec 41).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MacroVintage {
    pub series_id: String,
    pub period: String,
    pub value: f64,
    pub vintage: DateTime<Utc>,
}

impl Warehouse {
    pub async fn connect(url: &str) -> Result<Self, WarehouseError> {
        let pool = PgPoolOptions::new().max_connections(8).connect(url).await?;
        Ok(Self { pool })
    }

    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Upserts catalog rows from `sources.yaml` into `catalog.source` (Postgres is authoritative
    /// at runtime, the YAML is the seed). Returns the number of rows written.
    pub async fn sync_catalog(&self, sources: &[Source]) -> Result<usize, WarehouseError> {
        let mut tx = self.pool.begin().await?;
        for s in sources {
            sqlx::query(
                "INSERT INTO catalog.source \
                 (id, name, category, capabilities, asset_classes, transport, auth, status, tos_notes, notes) \
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) \
                 ON CONFLICT (id) DO UPDATE SET name = excluded.name, category = excluded.category, \
                 capabilities = excluded.capabilities, asset_classes = excluded.asset_classes, \
                 transport = excluded.transport, auth = excluded.auth, status = excluded.status, \
                 tos_notes = excluded.tos_notes, notes = excluded.notes, updated_at = now()",
            )
            .bind(&s.id)
            .bind(&s.name)
            .bind(&s.category)
            .bind(&s.capabilities)
            .bind(&s.asset_classes)
            .bind(&s.transport)
            .bind(&s.auth)
            .bind(&s.status)
            .bind(&s.tos_notes)
            .bind(&s.notes)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(sources.len())
    }

    /// Resolves a venue symbol to an internal instrument id.
    pub async fn resolve_symbol(&self, venue: &str, symbol: &str) -> Result<Option<InstrumentId>, WarehouseError> {
        let row: Option<(Uuid,)> = sqlx::query_as(
            "SELECT l.instrument_id FROM ref.listing l JOIN ref.venue v ON v.id = l.venue_id \
             WHERE v.code = $1 AND l.venue_symbol = $2 AND l.valid_from <= now() AND l.valid_to > now()",
        )
        .bind(venue)
        .bind(symbol)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|(u,)| InstrumentId(u)))
    }

    /// Catalog row by id.
    pub async fn source(&self, id: &str) -> Result<Option<SourceRow>, WarehouseError> {
        let row: Option<(String, String, String, String)> =
            sqlx::query_as("SELECT id, name, category, status FROM catalog.source WHERE id = $1")
                .bind(id)
                .fetch_optional(&self.pool)
                .await?;
        Ok(row.map(|(id, name, category, status)| SourceRow { id, name, category, status }))
    }

    pub async fn list_sources(&self) -> Result<Vec<SourceRow>, WarehouseError> {
        let rows: Vec<(String, String, String, String)> =
            sqlx::query_as("SELECT id, name, category, status FROM catalog.source ORDER BY id")
                .fetch_all(&self.pool)
                .await?;
        Ok(rows.into_iter().map(|(id, name, category, status)| SourceRow { id, name, category, status }).collect())
    }

    /// Latest value of each period of a series known at `as_of` (no look-ahead).
    pub async fn macro_as_of(
        &self,
        series_id: &str,
        as_of: DateTime<Utc>,
    ) -> Result<Vec<MacroVintage>, WarehouseError> {
        let rows: Vec<(String, String, f64, DateTime<Utc>)> = sqlx::query_as(
            "SELECT DISTINCT ON (period) series_id, period, value, vintage FROM wh.macro_observation \
             WHERE series_id = $1 AND vintage <= $2 ORDER BY period, vintage DESC",
        )
        .bind(series_id)
        .bind(as_of)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|(series_id, period, value, vintage)| MacroVintage { series_id, period, value, vintage })
            .collect())
    }
}

/// Live tests run only with `DATABASE_URL` set (compose Postgres, migrations applied). Each test
/// uses ids unique to the run and deletes what it inserted.
#[cfg(test)]
mod tests {
    use super::*;

    async fn wh() -> Option<Warehouse> {
        let url = std::env::var("DATABASE_URL").ok()?;
        Some(Warehouse::connect(&url).await.expect("connect"))
    }

    fn test_source(id: &str) -> Source {
        Source {
            id: id.into(),
            name: "Test source".into(),
            category: "test".into(),
            capabilities: vec!["data".into()],
            asset_classes: vec!["equity".into()],
            transport: vec!["rest".into()],
            auth: "none".into(),
            status: "stub".into(),
            tos_notes: String::new(),
            notes: "inserted by fa-warehouse tests".into(),
        }
    }

    #[test]
    fn repo_sources_yaml_parses() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/sources.yaml");
        let sources = load_sources(Path::new(path)).expect("config/sources.yaml parses");
        assert!(sources.len() >= 70);
        assert!(sources.iter().all(|s| ["planned", "stub", "live", "disabled"].contains(&s.status.as_str())));
    }

    #[tokio::test]
    async fn catalog_sync_upserts_and_lists() {
        let Some(wh) = wh().await else { return };
        let id = format!("test_{}", Uuid::now_v7().simple());
        let mut s = test_source(&id);
        assert_eq!(wh.sync_catalog(std::slice::from_ref(&s)).await.unwrap(), 1);
        s.status = "disabled".into();
        s.name = "Renamed".into();
        wh.sync_catalog(std::slice::from_ref(&s)).await.unwrap();
        let row = wh.source(&id).await.unwrap().expect("row");
        assert_eq!((row.name.as_str(), row.status.as_str()), ("Renamed", "disabled"));
        assert!(wh.list_sources().await.unwrap().iter().any(|r| r.id == id));
        sqlx::query("DELETE FROM catalog.source WHERE id = $1").bind(&id).execute(wh.pool()).await.unwrap();
        assert!(wh.source(&id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn symbol_resolution_honours_listing_validity() {
        let Some(wh) = wh().await else { return };
        let tag = Uuid::now_v7().simple().to_string();
        let venue = Uuid::now_v7();
        let inst = Uuid::now_v7();
        let code = format!("tv{tag}");
        sqlx::query(
            "INSERT INTO ref.venue (id, code, name, timezone, kind) VALUES ($1, $2, 'Test venue', 'UTC', 'exchange')",
        )
        .bind(venue)
        .bind(&code)
        .execute(wh.pool())
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO ref.instrument (id, asset_class, kind, name) VALUES ($1, 'equity', 'equity', 'Test Inc')",
        )
        .bind(inst)
        .execute(wh.pool())
        .await
        .unwrap();
        sqlx::query("INSERT INTO ref.listing (instrument_id, venue_id, venue_symbol) VALUES ($1, $2, 'TST')")
            .bind(inst)
            .bind(venue)
            .execute(wh.pool())
            .await
            .unwrap();
        // expired listing must not resolve
        sqlx::query(
            "INSERT INTO ref.listing (instrument_id, venue_id, venue_symbol, valid_from, valid_to) \
             VALUES ($1, $2, 'OLD', '2000-01-01', '2001-01-01')",
        )
        .bind(inst)
        .bind(venue)
        .execute(wh.pool())
        .await
        .unwrap();

        assert_eq!(wh.resolve_symbol(&code, "TST").await.unwrap(), Some(InstrumentId(inst)));
        assert_eq!(wh.resolve_symbol(&code, "OLD").await.unwrap(), None);
        assert_eq!(wh.resolve_symbol("no-such-venue", "TST").await.unwrap(), None);

        sqlx::query("DELETE FROM ref.listing WHERE venue_id = $1").bind(venue).execute(wh.pool()).await.unwrap();
        sqlx::query("DELETE FROM ref.instrument WHERE id = $1").bind(inst).execute(wh.pool()).await.unwrap();
        sqlx::query("DELETE FROM ref.venue WHERE id = $1").bind(venue).execute(wh.pool()).await.unwrap();
    }

    #[tokio::test]
    async fn macro_query_is_point_in_time() {
        let Some(wh) = wh().await else { return };
        let src = format!("test_{}", Uuid::now_v7().simple());
        wh.sync_catalog(&[test_source(&src)]).await.unwrap();
        let series = format!("SERIES_{src}");
        let v1 = DateTime::parse_from_rfc3339("2024-02-01T00:00:00Z").unwrap().with_timezone(&Utc);
        let v2 = DateTime::parse_from_rfc3339("2024-03-01T00:00:00Z").unwrap().with_timezone(&Utc);
        for (period, vintage, value, revision) in [
            ("2024-01", v1, 1.0, false),
            ("2024-01", v2, 1.5, true), // revised a month later
            ("2024-02", v2, 2.0, false),
        ] {
            sqlx::query(
                "INSERT INTO wh.macro_observation (source_id, series_id, period, vintage, value, revision) \
                 VALUES ($1, $2, $3, $4, $5, $6)",
            )
            .bind(&src)
            .bind(&series)
            .bind(period)
            .bind(vintage)
            .bind(value)
            .bind(revision)
            .execute(wh.pool())
            .await
            .unwrap();
        }
        let early = wh.macro_as_of(&series, v1 + chrono::Duration::days(1)).await.unwrap();
        assert_eq!(early.len(), 1);
        assert_eq!((early[0].period.as_str(), early[0].value), ("2024-01", 1.0));
        let late = wh.macro_as_of(&series, v2).await.unwrap();
        assert_eq!(late.len(), 2);
        assert_eq!(late[0].value, 1.5, "revision visible once released");
        assert_eq!(late[1].period, "2024-02");
        assert!(wh.macro_as_of(&series, v1 - chrono::Duration::days(1)).await.unwrap().is_empty());

        sqlx::query("DELETE FROM wh.macro_observation WHERE source_id = $1")
            .bind(&src)
            .execute(wh.pool())
            .await
            .unwrap();
        sqlx::query("DELETE FROM catalog.source WHERE id = $1").bind(&src).execute(wh.pool()).await.unwrap();
    }
}
