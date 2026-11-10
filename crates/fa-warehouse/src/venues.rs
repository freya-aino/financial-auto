//! Instrument reference lookup for the risk chain (spec 12 §5 `ref.instrument` / `ref.listing`
//! / `ref.venue`, spec 40 §6.5 concentration caps per asset class and venue): per instrument its
//! asset class, kind, current venue, quote currency, status (`active` / `provisional`, D-050)
//! and the ingest source(s) feeding it. [`load_instruments`] reads the current rows;
//! [`RefTable`] is the pure in-memory map the risk-engine consults per tick and refreshes on
//! `ref.changed.*` / `ingest.unresolved.*` notices.
use std::collections::BTreeMap;

use fa_core::{AssetClass, InstrumentId};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::WarehouseError;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstrumentRef {
    pub id: InstrumentId,
    /// `ref.instrument.asset_class` parsed into the platform enum (`None` for `unknown` or
    /// unmapped values: callers fall back to their default class).
    pub asset_class: Option<AssetClass>,
    pub asset_class_raw: String,
    pub kind: String,
    /// Code of the venue of the current listing (`ref.venue.code`, the token used in subjects);
    /// `None` when the instrument has no current listing.
    pub venue: Option<String>,
    pub venue_id: Option<Uuid>,
    /// Quote currency (`ref.instrument.quote_ccy`); `None` = unknown (treated as EUR by callers).
    pub quote_ccy: Option<String>,
    /// `active` | `provisional` | …
    pub status: String,
    /// Distinct `ref.listing.source_id`s of the current listings, sorted.
    pub sources: Vec<String>,
}

impl InstrumentRef {
    pub fn provisional(&self) -> bool {
        self.status == "provisional"
    }

    /// The single source feeding this instrument, `None` when there is none or more than one
    /// (ambiguous: a health halt of one source then flattens everything, see risk-engine).
    pub fn source(&self) -> Option<&str> {
        match self.sources.as_slice() {
            [one] => Some(one),
            _ => None,
        }
    }
}

/// Maps `ref.instrument.asset_class` text to the platform enum (same spelling as serde).
pub fn parse_asset_class(s: &str) -> Option<AssetClass> {
    let norm = match s.trim().to_lowercase().as_str() {
        "equities" | "stock" | "stocks" => "equity".to_owned(),
        "forex" | "fx_pair" | "currency" => "fx".to_owned(),
        "cryptocurrency" | "crypto_spot" => "crypto".to_owned(),
        "commodities" => "commodity".to_owned(),
        "bond" | "bonds" | "rate" => "rates".to_owned(),
        "indices" => "index".to_owned(),
        "derivatives" | "future" | "option" | "perp" => "derivative".to_owned(),
        "prediction" | "prediction_contract" => "prediction_market".to_owned(),
        other => other.to_owned(),
    };
    serde_json::from_value(serde_json::Value::String(norm)).ok()
}

/// Every current instrument (not `merged`, valid now) with its current listing and sources.
pub async fn load_instruments(pool: &PgPool) -> Result<Vec<InstrumentRef>, WarehouseError> {
    let rows = sqlx::query(
        "SELECT i.id, i.asset_class, i.kind, i.quote_ccy, i.status, \
                l.venue_code, l.venue_id, \
                coalesce((SELECT array_agg(DISTINCT x.source_id ORDER BY x.source_id) FROM ref.listing x \
                           WHERE x.instrument_id = i.id AND x.source_id IS NOT NULL \
                             AND now() >= x.valid_from AND now() < x.valid_to), '{}'::text[]) AS sources \
           FROM ref.instrument i \
           LEFT JOIN LATERAL (SELECT v.code AS venue_code, v.id AS venue_id FROM ref.listing l JOIN ref.venue v ON v.id = l.venue_id \
                               WHERE l.instrument_id = i.id AND now() >= l.valid_from AND now() < l.valid_to \
                               ORDER BY v.code LIMIT 1) l ON true \
          WHERE i.status <> 'merged' AND now() >= i.valid_from AND now() < i.valid_to",
    )
    .fetch_all(pool)
    .await?;
    Ok(rows
        .iter()
        .map(|r| {
            let raw: String = r.get("asset_class");
            InstrumentRef {
                id: InstrumentId(r.get::<Uuid, _>("id")),
                asset_class: parse_asset_class(&raw),
                asset_class_raw: raw,
                kind: r.get("kind"),
                venue: r.get("venue_code"),
                venue_id: r.get("venue_id"),
                quote_ccy: r
                    .get::<Option<String>, _>("quote_ccy")
                    .map(|c| c.trim().to_uppercase())
                    .filter(|c| !c.is_empty()),
                status: r.get("status"),
                sources: r.get::<Vec<String>, _>("sources"),
            }
        })
        .collect())
}

/// Pure lookup table over [`InstrumentRef`]s.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RefTable {
    by_id: BTreeMap<InstrumentId, InstrumentRef>,
}

impl RefTable {
    pub fn from_refs(refs: impl IntoIterator<Item = InstrumentRef>) -> Self {
        Self { by_id: refs.into_iter().map(|r| (r.id, r)).collect() }
    }

    pub fn get(&self, id: &InstrumentId) -> Option<&InstrumentRef> {
        self.by_id.get(id)
    }

    pub fn insert(&mut self, r: InstrumentRef) {
        self.by_id.insert(r.id, r);
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &InstrumentRef> {
        self.by_id.values()
    }

    /// Instruments fed by `source` (single-source listings only).
    pub fn fed_by<'a>(&'a self, source: &'a str) -> impl Iterator<Item = InstrumentId> + 'a {
        self.by_id.values().filter(move |r| r.source() == Some(source)).map(|r| r.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_class_spellings_map_to_the_enum() {
        assert_eq!(parse_asset_class("equity"), Some(AssetClass::Equity));
        assert_eq!(parse_asset_class("Equities"), Some(AssetClass::Equity));
        assert_eq!(parse_asset_class("fx_pair"), Some(AssetClass::Fx));
        assert_eq!(parse_asset_class("crypto"), Some(AssetClass::Crypto));
        assert_eq!(parse_asset_class("prediction_market"), Some(AssetClass::PredictionMarket));
        assert_eq!(parse_asset_class("unknown"), None);
        assert_eq!(parse_asset_class(""), None);
    }

    #[test]
    fn table_lookups_and_source_ambiguity() {
        let mk = |venue: &str, sources: &[&str], status: &str| InstrumentRef {
            id: InstrumentId::new(),
            asset_class: Some(AssetClass::Equity),
            asset_class_raw: "equity".into(),
            kind: "equity".into(),
            venue: Some(venue.into()),
            venue_id: None,
            quote_ccy: Some("USD".into()),
            status: status.into(),
            sources: sources.iter().map(|s| (*s).to_owned()).collect(),
        };
        let a = mk("yahoo", &["yahoo"], "provisional");
        let b = mk("xetr", &["yahoo", "lemon"], "active");
        let t = RefTable::from_refs([a.clone(), b.clone()]);
        assert_eq!(t.len(), 2);
        assert!(t.get(&a.id).unwrap().provisional() && !t.get(&b.id).unwrap().provisional());
        assert_eq!(t.get(&a.id).unwrap().source(), Some("yahoo"));
        assert_eq!(t.get(&b.id).unwrap().source(), None, "two sources: ambiguous");
        assert_eq!(t.fed_by("yahoo").collect::<Vec<_>>(), vec![a.id]);
        assert!(t.get(&InstrumentId::new()).is_none());
    }

    /// Needs `DATABASE_URL` (compose Postgres).
    #[tokio::test]
    async fn loads_current_instruments_with_listing_and_sources() {
        let Ok(url) = std::env::var("DATABASE_URL") else { return };
        let pool = sqlx::postgres::PgPoolOptions::new().max_connections(2).connect(&url).await.unwrap();
        let tag = Uuid::now_v7().simple().to_string();
        let (inst, venue, src) = (Uuid::now_v7(), Uuid::now_v7(), format!("test_{tag}"));
        sqlx::query(
            "INSERT INTO catalog.source (id, name, category, auth, status) VALUES ($1, 'Test', 'test', 'none', 'stub')",
        )
        .bind(&src)
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO ref.venue (id, code, name, timezone, kind) VALUES ($1, $2, 'Test venue', 'UTC', 'exchange')",
        )
        .bind(venue)
        .bind(format!("v{tag}"))
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query("INSERT INTO ref.instrument (id, asset_class, kind, name, quote_ccy, status) VALUES ($1, 'crypto', 'crypto_spot', 'Test coin', 'usd', 'provisional')")
            .bind(inst)
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query(
            "INSERT INTO ref.listing (instrument_id, venue_id, venue_symbol, source_id) VALUES ($1, $2, 'TC', $3)",
        )
        .bind(inst)
        .bind(venue)
        .bind(&src)
        .execute(&pool)
        .await
        .unwrap();
        let refs = load_instruments(&pool).await.unwrap();
        let t = RefTable::from_refs(refs);
        let r = t.get(&InstrumentId(inst)).expect("loaded");
        assert_eq!(r.asset_class, Some(AssetClass::Crypto));
        assert_eq!(r.venue.as_deref(), Some(format!("v{tag}").as_str()));
        assert_eq!((r.quote_ccy.as_deref(), r.status.as_str()), (Some("USD"), "provisional"));
        assert_eq!(r.source(), Some(src.as_str()));
        sqlx::query("DELETE FROM ref.listing WHERE instrument_id = $1").bind(inst).execute(&pool).await.unwrap();
        sqlx::query("DELETE FROM ref.instrument WHERE id = $1").bind(inst).execute(&pool).await.unwrap();
        sqlx::query("DELETE FROM ref.venue WHERE id = $1").bind(venue).execute(&pool).await.unwrap();
        sqlx::query("DELETE FROM catalog.source WHERE id = $1").bind(&src).execute(&pool).await.unwrap();
    }
}
