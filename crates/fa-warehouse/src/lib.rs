//! PostgreSQL warehouse access (specs 12, 20; D-013): reference data, catalog, macro vintages.
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

    pub fn pool(&self) -> &PgPool {
        &self.pool
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

    pub async fn list_sources(&self) -> Result<Vec<SourceRow>, WarehouseError> {
        let rows: Vec<(String, String, String, String)> =
            sqlx::query_as("SELECT id, name, category, status FROM catalog.source ORDER BY id")
                .fetch_all(&self.pool)
                .await?;
        Ok(rows.into_iter().map(|(id, name, category, status)| SourceRow { id, name, category, status }).collect())
    }

    /// Latest value of each period of a series known at `as_of` (no look-ahead).
    pub async fn macro_as_of(&self, series_id: &str, as_of: DateTime<Utc>) -> Result<Vec<MacroVintage>, WarehouseError> {
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
