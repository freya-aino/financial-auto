//! The graph store behind [`HubClient`] (ADR 0010 second half, D-108): every model-state graph
//! operation — reference mirror, snapshots/spaces/adapters, regions/transitions with their
//! event-time versions, lineage/hierarchy, per-space state, and the genome links — expressed as
//! the [`GraphStore`] trait, implemented by Postgres ([`PgGraph`], schema `graph`, migration
//! `20261115120000`) and by Helix ([`HelixGraph`], delegating to the original query layer).
//!
//! Routing mirrors D-101's vectors: [`configure_graph`] reads `[graph] backend` (default
//! `postgres`, env `FA_GRAPH_BACKEND`) and attaches a lazily connected [`PgGraph`] to every
//! client the process creates; `HubClient` methods then route to it, and Helix serves only what
//! has not been migrated (genomes/fitness/evolution, the graph explorer, meta, retention
//! aggregates — see the D-108 row for the list). Ids are `bigint`s from one shared sequence, so
//! the `u64` node ids of the API stay globally unique exactly as Helix `$id`s were; ids never
//! cross backends.
use std::sync::{Arc, RwLock};

use chrono::{DateTime, SecondsFormat, TimeZone, Utc};
use serde_json::{Map, Value as Json};
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::{Column, Row, TypeInfo};

use crate::queries::{
    AdapterState, EdgeRow, ExpertSpace, FeedbackLink, Instrument, LatentRegion, NewAdapterState, NewExpertSpace,
    NewInstrument, NewRegion, NewSnapshot, NodeRow, RegionGraph, Snapshot, SnapshotGraph, TrainedOnLink, Transition,
    TransitionUpsert, adapter_id,
};
use crate::reference::{INACTIVE, InstrumentReference, RefEntity, RefInstrument, RefSyncReport, RefVenue, sector_id};
use crate::regions::{
    LineageLink, NewCrossPrior, NewDescent, NewHierEdge, NewKernelModel, NewRegionEpisode, NewRegionSet, RegionLineage,
    RegionSetRow, RegionStatsUpdate,
};
use crate::versions::{RegionStatusVersion, TransitionVersion, edge_key};
use crate::{HubError, Result, schema};

/// Backend names accepted by `[graph] backend` / `FA_GRAPH_BACKEND`.
pub const BACKENDS: [&str; 2] = ["postgres", "helix"];

static DEFAULT_GRAPH: RwLock<Option<Arc<PgGraph>>> = RwLock::new(None);

/// Applies `[graph]` for this process (like [`crate::qdrant::configure_vectors`]): with backend
/// `postgres` a lazily connected [`PgGraph`] on `postgres.url` (`FA_POSTGRES_URL` wins) becomes
/// the default of every client created afterwards; `helix` detaches it. Returns the backend.
pub fn configure_graph(cfg: &fa_core::config::PlatformConfig) -> &'static str {
    let backend =
        std::env::var("FA_GRAPH_BACKEND").ok().filter(|b| !b.is_empty()).unwrap_or_else(|| cfg.graph.backend.clone());
    let pg = (backend != "helix").then(|| {
        let url =
            std::env::var("FA_POSTGRES_URL").ok().filter(|u| !u.is_empty()).unwrap_or_else(|| cfg.postgres.url.clone());
        Arc::new(PgGraph::connect_lazy(&url))
    });
    let name = if pg.is_some() { "postgres" } else { "helix" };
    *DEFAULT_GRAPH.write().unwrap_or_else(std::sync::PoisonError::into_inner) = pg;
    name
}

/// The process default set by [`configure_graph`] (`None` until a service configured it, so
/// plain `HubClient::new` outside a configured service keeps the historical Helix behaviour).
pub(crate) fn default_graph() -> Option<Arc<PgGraph>> {
    DEFAULT_GRAPH.read().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
}

fn pg_err(e: sqlx::Error) -> HubError {
    if let sqlx::Error::Database(db) = &e
        && db.code().as_deref() == Some("23505")
    {
        return HubError::Server {
            status: 409,
            code: "unique_constraint_violation".into(),
            msg: db.to_string(),
            retryable: Some(false),
        };
    }
    HubError::Graph(e.to_string())
}

fn t3339(t: DateTime<Utc>) -> String {
    t.to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn ms(t_ms: i64) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(t_ms).single().unwrap_or_else(Utc::now)
}

fn u(id: i64) -> u64 {
    id as u64
}

/// Every column of a row as a JSON property map (timestamps as RFC 3339), for the generic
/// `NodeRow` / `EdgeRow` shapes the explorer-style reads return.
fn props(row: &sqlx::postgres::PgRow, skip: &[&str]) -> Map<String, Json> {
    let mut m = Map::new();
    for c in row.columns() {
        let name = c.name();
        if skip.contains(&name) || name == "id" {
            continue;
        }
        let v = match c.type_info().name() {
            "TEXT" | "VARCHAR" => row.try_get::<Option<String>, _>(name).ok().flatten().map(Json::from),
            "INT4" => row.try_get::<Option<i32>, _>(name).ok().flatten().map(Json::from),
            "INT8" => row.try_get::<Option<i64>, _>(name).ok().flatten().map(Json::from),
            "FLOAT8" => row.try_get::<Option<f64>, _>(name).ok().flatten().map(Json::from),
            "BOOL" => row.try_get::<Option<bool>, _>(name).ok().flatten().map(Json::from),
            "TIMESTAMPTZ" => row.try_get::<Option<DateTime<Utc>>, _>(name).ok().flatten().map(|t| Json::from(t3339(t))),
            _ => None,
        };
        if let Some(v) = v {
            m.insert(name.to_owned(), v);
        }
    }
    m
}

/// The model-state graph surface shared by the Postgres and Helix backends. Argument and return
/// shapes are exactly those of the corresponding `HubClient` methods; node ids (`u64`) are only
/// meaningful within one backend.
#[allow(async_fn_in_trait)]
pub trait GraphStore {
    // reference mirror
    async fn upsert_ref_entities(&self, rows: &[RefEntity]) -> Result<usize>;
    async fn upsert_ref_venues(&self, rows: &[RefVenue]) -> Result<usize>;
    async fn upsert_ref_instruments(&self, rows: &[RefInstrument]) -> Result<RefSyncReport>;
    async fn retire_ref_instruments(&self, instrument_ids: &[String]) -> Result<usize>;
    async fn ref_instrument_ids(&self) -> Result<Vec<(String, String)>>;
    async fn instrument_reference(&self, instrument_id: &str) -> Result<Option<InstrumentReference>>;
    async fn upsert_instrument(&self, inst: &NewInstrument) -> Result<Instrument>;
    async fn get_instrument(&self, instrument_id: &str) -> Result<Option<Instrument>>;
    // snapshots, spaces, adapters
    async fn add_snapshot(&self, snap: &NewSnapshot) -> Result<Snapshot>;
    async fn ensure_snapshot(&self, snap: &NewSnapshot) -> Result<Snapshot>;
    async fn list_snapshots(&self) -> Result<Vec<Snapshot>>;
    async fn snapshot_by_id(&self, snapshot_id: &str) -> Result<Option<Snapshot>>;
    async fn snapshots_by_status(&self, timescale: &str, family: &str, status: &str) -> Result<Vec<Snapshot>>;
    async fn link_snapshot_parent(&self, parent: u64, child: u64, op: &str) -> Result<u64>;
    async fn snapshot_parent(&self, child: u64) -> Result<Option<Snapshot>>;
    async fn snapshot_children(&self, parent: u64) -> Result<Vec<Snapshot>>;
    async fn set_snapshot_status(&self, snapshot: u64, status: &str) -> Result<Snapshot>;
    async fn set_snapshot_previous_active(&self, snapshot: u64, previous: &str) -> Result<Snapshot>;
    async fn set_snapshot_manifest(&self, snapshot: u64, manifest_json: &str, carried_mass: f64) -> Result<Snapshot>;
    async fn add_adapter_state(&self, snapshot: u64, a: &NewAdapterState) -> Result<AdapterState>;
    async fn snapshot_adapters(&self, snapshot_id: &str, limit: usize) -> Result<Vec<AdapterState>>;
    async fn add_expert_space(&self, snapshot: u64, space: &NewExpertSpace) -> Result<ExpertSpace>;
    async fn ensure_expert_space(&self, snapshot: u64, space: &NewExpertSpace) -> Result<ExpertSpace>;
    async fn space_by_id(&self, expert_space_id: &str) -> Result<Option<ExpertSpace>>;
    async fn snapshot_spaces(&self, snapshot: u64) -> Result<Vec<ExpertSpace>>;
    // regions and transitions
    async fn add_region(&self, space: u64, region: &NewRegion) -> Result<LatentRegion>;
    async fn upsert_region(&self, space: u64, region: &NewRegion) -> Result<LatentRegion>;
    async fn region_by_id(&self, region_id: &str) -> Result<Option<LatentRegion>>;
    async fn mark_region_status(&self, region_id: &str, status: &str, related: &str) -> Result<LatentRegion>;
    async fn space_regions(&self, space: u64) -> Result<Vec<LatentRegion>>;
    async fn update_region_stats(&self, rows: &[RegionStatsUpdate]) -> Result<()>;
    async fn mark_region_died(&self, region_id: &str, reason: &str, related: &str) -> Result<()>;
    async fn add_transition(&self, from: u64, to: u64, t: &crate::queries::NewTransition) -> Result<Transition>;
    async fn upsert_transitions(&self, rows: &[TransitionUpsert]) -> Result<Vec<Transition>>;
    async fn region_transitions(&self, region: u64) -> Result<Vec<Transition>>;
    async fn region_neighbors(&self, region: u64) -> Result<Vec<LatentRegion>>;
    async fn match_regions(&self, prev: u64, next: u64, score: f64) -> Result<u64>;
    async fn region_graph(&self, space: u64) -> Result<RegionGraph>;
    async fn snapshot_graph(&self, snapshot_id: &str) -> Result<Option<SnapshotGraph>>;
    // lineage and hierarchy
    async fn upsert_descents(&self, rows: &[NewDescent]) -> Result<()>;
    async fn upsert_hier_edges(&self, rows: &[NewHierEdge]) -> Result<()>;
    async fn snapshot_hierarchy(&self, snapshot_id: &str) -> Result<Vec<EdgeRow>>;
    async fn lineage_members(&self, lineage_id: &str) -> Result<Vec<NodeRow>>;
    async fn region_lineage(&self, region_id: &str, depth: usize) -> Result<Option<RegionLineage>>;
    // per-space state
    async fn upsert_region_set(&self, r: &NewRegionSet) -> Result<()>;
    async fn region_set(&self, expert_space_id: &str) -> Result<Option<RegionSetRow>>;
    async fn upsert_kernel_model(&self, k: &NewKernelModel) -> Result<()>;
    async fn snapshot_kernels(&self, snapshot_id: &str) -> Result<Vec<NodeRow>>;
    async fn upsert_cross_priors(&self, rows: &[NewCrossPrior]) -> Result<()>;
    async fn cross_priors(&self, timescale: &str, limit: usize) -> Result<Vec<NodeRow>>;
    async fn upsert_region_episodes(&self, rows: &[NewRegionEpisode]) -> Result<()>;
    async fn region_episodes(&self, region_id: &str, limit: usize) -> Result<Vec<NodeRow>>;
    async fn prune_region_episodes(&self, timescale: &str, cutoff_ms: i64) -> Result<u64>;
    // event-time versions and as-of
    async fn transition_versions(
        &self,
        from: &str,
        to: &str,
        sampler: &str,
        as_of_ms: Option<i64>,
        limit: usize,
    ) -> Result<Vec<TransitionVersion>>;
    async fn region_status_versions(
        &self,
        region_id: &str,
        as_of_ms: Option<i64>,
        limit: usize,
    ) -> Result<Vec<RegionStatusVersion>>;
    async fn latest_transition_versions(&self, keys: &[String], t_ms: i64) -> Result<Vec<Option<TransitionVersion>>>;
    async fn latest_region_status_versions(
        &self,
        keys: &[String],
        t_ms: i64,
    ) -> Result<Vec<Option<RegionStatusVersion>>>;
    async fn prune_space_versions(&self, space: u64, cutoff_ms: i64) -> Result<u64>;
    async fn list_snapshots_as_of(&self, as_of_ms: i64) -> Result<Vec<Snapshot>>;
    async fn space_regions_as_of(&self, space: u64, as_of_ms: i64) -> Result<Vec<LatentRegion>>;
    async fn snapshot_graph_as_of(&self, snapshot_id: &str, as_of_ms: i64) -> Result<Option<SnapshotGraph>>;
    // genome links (business-id keyed; genome nodes themselves may live elsewhere)
    async fn link_trained_on(&self, links: &[TrainedOnLink]) -> Result<()>;
    async fn link_feedback_from(&self, links: &[FeedbackLink]) -> Result<()>;
    async fn trained_on_snapshots(&self, genome_id: &str) -> Result<Vec<Snapshot>>;
    async fn feedback_edges(&self, snapshot_id: &str) -> Result<Vec<EdgeRow>>;
}

/// The Postgres backend: schema `graph`, one pool shared by clones.
#[derive(Clone, Debug)]
pub struct PgGraph {
    pool: PgPool,
}

impl PgGraph {
    /// A lazily connecting pool (nothing is awaited here), sized like the other service pools.
    pub fn connect_lazy(url: &str) -> Self {
        let pool = PgPoolOptions::new()
            .max_connections(4)
            .acquire_timeout(std::time::Duration::from_secs(5))
            .connect_lazy(url)
            .unwrap_or_else(|_| PgPoolOptions::new().connect_lazy("postgres://invalid").expect("static url parses"));
        Self { pool }
    }

    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    async fn snapshot_row(&self, id: i64) -> Result<Snapshot> {
        let r = sqlx::query("SELECT * FROM graph.snapshot WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(pg_err)?;
        r.as_ref()
            .map(snapshot_of)
            .ok_or_else(|| HubError::NotFound { label: schema::node::SNAPSHOT.into(), id: id.to_string() })
    }
}

fn snapshot_of(r: &sqlx::postgres::PgRow) -> Snapshot {
    Snapshot {
        id: u(r.get("id")),
        snapshot_id: r.get("snapshot_id"),
        timescale: r.get("timescale"),
        encoder_family: r.get("encoder_family"),
        generation: r.get::<i32, _>("generation") as u32,
        garage_key: r.get("garage_key"),
        status: r.get("status"),
        created_at: Some(t3339(r.get("created_at"))),
        carried_mass: r.get("carried_mass"),
        manifest: r.get("manifest"),
        previous_active: r.get("previous_active"),
    }
}

fn space_of(r: &sqlx::postgres::PgRow) -> ExpertSpace {
    ExpertSpace {
        id: u(r.get("id")),
        expert_space_id: r.get("expert_space_id"),
        snapshot_id: r.get("snapshot_id"),
        index: r.get::<i32, _>("index") as u32,
        dim: r.get::<i32, _>("dim") as u32,
    }
}

fn region_of(r: &sqlx::postgres::PgRow) -> LatentRegion {
    LatentRegion {
        id: u(r.get("id")),
        region_id: r.get("region_id"),
        lineage_id: r.get("lineage_id"),
        expert_space_id: r.get("expert_space_id"),
        method: r.get("method"),
        radius: r.get("radius"),
        density: r.get("density"),
        status: r.get("status"),
        related_region_id: r.get("related_region_id"),
        created_at: Some(t3339(r.get("created_at"))),
        updated_at: r.get::<Option<DateTime<Utc>>, _>("updated_at").map(t3339),
    }
}

fn transition_of(r: &sqlx::postgres::PgRow) -> Transition {
    Transition {
        id: u(r.get("id")),
        from: u(r.get("from_id")),
        to: u(r.get("to_id")),
        from_region_id: Some(r.get("from_region_id")),
        to_region_id: Some(r.get("to_region_id")),
        sampler_id: r.get("sampler_id"),
        count: r.get("count"),
        mean: r.get("mean"),
        lower: r.get("lower"),
        upper: r.get("upper"),
        kernel_params: r.get("kernel_params"),
        updated_at: Some(t3339(r.get("updated_at"))),
    }
}

fn adapter_of(r: &sqlx::postgres::PgRow) -> AdapterState {
    AdapterState {
        id: u(r.get("id")),
        adapter_id: r.get("adapter_id"),
        snapshot_id: r.get("snapshot_id"),
        timescale: r.get("timescale"),
        version: r.get::<i64, _>("version") as u64,
        garage_key: r.get("garage_key"),
        drift: r.get("drift"),
        max_drift: r.get("max_drift"),
        frozen: r.get("frozen"),
        n_windows: r.get::<i64, _>("n_windows") as u64,
        window_start: r.get::<Option<DateTime<Utc>>, _>("window_start").map(t3339),
        window_end: r.get::<Option<DateTime<Utc>>, _>("window_end").map(t3339),
        updated_at: Some(t3339(r.get("updated_at"))),
    }
}

fn node_row(r: &sqlx::postgres::PgRow, label: &str) -> NodeRow {
    NodeRow { id: u(r.get("id")), label: label.to_owned(), properties: props(r, &[]) }
}

fn edge_row(r: &sqlx::postgres::PgRow, label: &str, from: u64, to: u64, skip: &[&str]) -> EdgeRow {
    EdgeRow { id: u(r.get("id")), label: label.to_owned(), from, to, properties: props(r, skip) }
}

impl GraphStore for PgGraph {
    async fn upsert_ref_entities(&self, rows: &[RefEntity]) -> Result<usize> {
        let mut tx = self.pool.begin().await.map_err(pg_err)?;
        for e in rows {
            sqlx::query(
                "INSERT INTO graph.ref_entity (entity_id, lei, name, country, sector, mirrored_at)
                 VALUES ($1, $2, $3, $4, $5, now())
                 ON CONFLICT (entity_id) DO UPDATE SET lei = $2, name = $3, country = $4, sector = $5, mirrored_at = now()",
            )
            .bind(&e.entity_id).bind(&e.lei).bind(&e.name).bind(&e.country).bind(&e.sector)
            .execute(&mut *tx).await.map_err(pg_err)?;
            let sid = sector_id(&e.sector);
            sqlx::query("DELETE FROM graph.ref_in_sector WHERE entity_id = $1 AND sector_id <> $2")
                .bind(&e.entity_id)
                .bind(&sid)
                .execute(&mut *tx)
                .await
                .map_err(pg_err)?;
            if !sid.is_empty() {
                sqlx::query("INSERT INTO graph.ref_sector (sector_id, name) VALUES ($1, $2) ON CONFLICT (sector_id) DO UPDATE SET name = $2")
                    .bind(&sid).bind(&e.sector).execute(&mut *tx).await.map_err(pg_err)?;
                sqlx::query(
                    "INSERT INTO graph.ref_in_sector (entity_id, sector_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                )
                .bind(&e.entity_id)
                .bind(&sid)
                .execute(&mut *tx)
                .await
                .map_err(pg_err)?;
            }
        }
        tx.commit().await.map_err(pg_err)?;
        Ok(rows.len())
    }

    async fn upsert_ref_venues(&self, rows: &[RefVenue]) -> Result<usize> {
        for v in rows {
            sqlx::query(
                "INSERT INTO graph.ref_venue (venue_id, code, mic, name, kind, timezone, mirrored_at)
                 VALUES ($1, $2, $3, $4, $5, $6, now())
                 ON CONFLICT (venue_id) DO UPDATE SET code = $2, mic = $3, name = $4, kind = $5, timezone = $6, mirrored_at = now()",
            )
            .bind(&v.venue_id).bind(&v.code).bind(&v.mic).bind(&v.name).bind(&v.kind).bind(&v.timezone)
            .execute(&self.pool).await.map_err(pg_err)?;
        }
        Ok(rows.len())
    }

    async fn upsert_ref_instruments(&self, rows: &[RefInstrument]) -> Result<RefSyncReport> {
        let mut rep = RefSyncReport::default();
        let mut tx = self.pool.begin().await.map_err(pg_err)?;
        for i in rows {
            sqlx::query(
                "INSERT INTO graph.ref_instrument (instrument_id, asset_class, kind, name, status, base_ccy, quote_ccy, mirrored_at)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, now())
                 ON CONFLICT (instrument_id) DO UPDATE SET asset_class = $2, kind = $3, name = $4, status = $5, base_ccy = $6, quote_ccy = $7, mirrored_at = now()",
            )
            .bind(&i.instrument_id).bind(&i.asset_class).bind(&i.kind).bind(&i.name).bind(&i.status).bind(&i.base_ccy).bind(&i.quote_ccy)
            .execute(&mut *tx).await.map_err(pg_err)?;
            for del in [
                "DELETE FROM graph.ref_issued_by WHERE instrument_id = $1",
                "DELETE FROM graph.ref_listed_on WHERE instrument_id = $1",
                "DELETE FROM graph.ref_same_exposure WHERE instrument_id = $1",
            ] {
                sqlx::query(del).bind(&i.instrument_id).execute(&mut *tx).await.map_err(pg_err)?;
            }
            if let Some(issuer) = &i.issuer_id {
                sqlx::query(
                    "INSERT INTO graph.ref_issued_by (instrument_id, entity_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                )
                .bind(&i.instrument_id)
                .bind(issuer)
                .execute(&mut *tx)
                .await
                .map_err(pg_err)?;
                rep.issued_by += 1;
            }
            for l in &i.listings {
                sqlx::query(
                    "INSERT INTO graph.ref_listed_on (instrument_id, venue_id, venue_symbol, source_id, listing_key)
                     VALUES ($1, $2, $3, $4, $5) ON CONFLICT DO NOTHING",
                )
                .bind(&i.instrument_id)
                .bind(&l.venue_id)
                .bind(&l.venue_symbol)
                .bind(&l.source_id)
                .bind(l.key())
                .execute(&mut *tx)
                .await
                .map_err(pg_err)?;
                rep.listed_on += 1;
            }
            for x in &i.exposures {
                sqlx::query(
                    "INSERT INTO graph.ref_same_exposure (instrument_id, to_instrument_id, relation, exposure_key)
                     VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING",
                )
                .bind(&i.instrument_id)
                .bind(&x.to_instrument_id)
                .bind(&x.relation)
                .bind(x.key())
                .execute(&mut *tx)
                .await
                .map_err(pg_err)?;
                rep.same_exposure += 1;
            }
        }
        tx.commit().await.map_err(pg_err)?;
        rep.instruments = rows.len();
        rep.batches = 1;
        Ok(rep)
    }

    async fn retire_ref_instruments(&self, instrument_ids: &[String]) -> Result<usize> {
        let mut tx = self.pool.begin().await.map_err(pg_err)?;
        sqlx::query("UPDATE graph.ref_instrument SET status = $2, mirrored_at = now() WHERE instrument_id = ANY($1)")
            .bind(instrument_ids)
            .bind(INACTIVE)
            .execute(&mut *tx)
            .await
            .map_err(pg_err)?;
        for del in [
            "DELETE FROM graph.ref_issued_by WHERE instrument_id = ANY($1)",
            "DELETE FROM graph.ref_listed_on WHERE instrument_id = ANY($1)",
            "DELETE FROM graph.ref_same_exposure WHERE instrument_id = ANY($1)",
        ] {
            sqlx::query(del).bind(instrument_ids).execute(&mut *tx).await.map_err(pg_err)?;
        }
        tx.commit().await.map_err(pg_err)?;
        Ok(instrument_ids.len())
    }

    async fn ref_instrument_ids(&self) -> Result<Vec<(String, String)>> {
        let rows = sqlx::query("SELECT instrument_id, status FROM graph.ref_instrument")
            .fetch_all(&self.pool)
            .await
            .map_err(pg_err)?;
        Ok(rows.iter().map(|r| (r.get("instrument_id"), r.get("status"))).collect())
    }

    async fn instrument_reference(&self, instrument_id: &str) -> Result<Option<InstrumentReference>> {
        let Some(i) = sqlx::query("SELECT * FROM graph.ref_instrument WHERE instrument_id = $1")
            .bind(instrument_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(pg_err)?
        else {
            return Ok(None);
        };
        let issuer = sqlx::query(
            "SELECT e.* FROM graph.ref_entity e JOIN graph.ref_issued_by b ON b.entity_id = e.entity_id WHERE b.instrument_id = $1",
        ).bind(instrument_id).fetch_all(&self.pool).await.map_err(pg_err)?;
        let listings = sqlx::query("SELECT * FROM graph.ref_listed_on WHERE instrument_id = $1")
            .bind(instrument_id)
            .fetch_all(&self.pool)
            .await
            .map_err(pg_err)?;
        let venues = sqlx::query(
            "SELECT v.* FROM graph.ref_venue v JOIN graph.ref_listed_on l ON l.venue_id = v.venue_id WHERE l.instrument_id = $1",
        ).bind(instrument_id).fetch_all(&self.pool).await.map_err(pg_err)?;
        let exposures = sqlx::query("SELECT * FROM graph.ref_same_exposure WHERE instrument_id = $1")
            .bind(instrument_id)
            .fetch_all(&self.pool)
            .await
            .map_err(pg_err)?;
        let sectors = sqlx::query(
            "SELECT s.* FROM graph.ref_sector s
             JOIN graph.ref_in_sector x ON x.sector_id = s.sector_id
             JOIN graph.ref_issued_by b ON b.entity_id = x.entity_id WHERE b.instrument_id = $1",
        )
        .bind(instrument_id)
        .fetch_all(&self.pool)
        .await
        .map_err(pg_err)?;
        let iid = u(i.get("id"));
        Ok(Some(InstrumentReference {
            instrument: Json::Object(props(&i, &[])),
            issuer: issuer.iter().map(|r| Json::Object(props(r, &[]))).collect(),
            listings: listings
                .iter()
                .map(|r| edge_row(r, schema::edge::LISTED_ON, iid, 0, &["instrument_id"]))
                .collect(),
            venues: venues.iter().map(|r| Json::Object(props(r, &[]))).collect(),
            exposures: exposures
                .iter()
                .map(|r| edge_row(r, schema::edge::SAME_EXPOSURE, iid, 0, &["instrument_id"]))
                .collect(),
            sectors: sectors.iter().map(|r| Json::Object(props(r, &[]))).collect(),
        }))
    }

    async fn upsert_instrument(&self, inst: &NewInstrument) -> Result<Instrument> {
        let r = sqlx::query(
            "INSERT INTO graph.ref_instrument (instrument_id, asset_class, kind, name)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT (instrument_id) DO UPDATE SET asset_class = $2, kind = $3, name = $4
             RETURNING id, instrument_id, asset_class, kind, name",
        )
        .bind(&inst.instrument_id)
        .bind(&inst.asset_class)
        .bind(&inst.kind)
        .bind(&inst.name)
        .fetch_one(&self.pool)
        .await
        .map_err(pg_err)?;
        Ok(Instrument {
            id: u(r.get("id")),
            instrument_id: r.get("instrument_id"),
            asset_class: r.get("asset_class"),
            kind: r.get("kind"),
            name: r.get("name"),
        })
    }

    async fn get_instrument(&self, instrument_id: &str) -> Result<Option<Instrument>> {
        let r = sqlx::query(
            "SELECT id, instrument_id, asset_class, kind, name FROM graph.ref_instrument WHERE instrument_id = $1",
        )
        .bind(instrument_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(pg_err)?;
        Ok(r.map(|r| Instrument {
            id: u(r.get("id")),
            instrument_id: r.get("instrument_id"),
            asset_class: r.get("asset_class"),
            kind: r.get("kind"),
            name: r.get("name"),
        }))
    }

    async fn add_snapshot(&self, snap: &NewSnapshot) -> Result<Snapshot> {
        let r = sqlx::query(
            "INSERT INTO graph.snapshot (snapshot_id, timescale, encoder_family, generation, garage_key, status)
             VALUES ($1, $2, $3, $4, $5, $6) RETURNING *",
        )
        .bind(&snap.snapshot_id)
        .bind(&snap.timescale)
        .bind(&snap.encoder_family)
        .bind(snap.generation as i32)
        .bind(&snap.garage_key)
        .bind(&snap.status)
        .fetch_one(&self.pool)
        .await
        .map_err(pg_err)?;
        Ok(snapshot_of(&r))
    }

    async fn ensure_snapshot(&self, snap: &NewSnapshot) -> Result<Snapshot> {
        sqlx::query(
            "INSERT INTO graph.snapshot (snapshot_id, timescale, encoder_family, generation, garage_key, status)
             VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (snapshot_id) DO NOTHING",
        )
        .bind(&snap.snapshot_id)
        .bind(&snap.timescale)
        .bind(&snap.encoder_family)
        .bind(snap.generation as i32)
        .bind(&snap.garage_key)
        .bind(&snap.status)
        .execute(&self.pool)
        .await
        .map_err(pg_err)?;
        let r = sqlx::query("SELECT * FROM graph.snapshot WHERE snapshot_id = $1")
            .bind(&snap.snapshot_id)
            .fetch_one(&self.pool)
            .await
            .map_err(pg_err)?;
        Ok(snapshot_of(&r))
    }

    async fn list_snapshots(&self) -> Result<Vec<Snapshot>> {
        let rows = sqlx::query("SELECT * FROM graph.snapshot ORDER BY snapshot_id")
            .fetch_all(&self.pool)
            .await
            .map_err(pg_err)?;
        Ok(rows.iter().map(snapshot_of).collect())
    }

    async fn snapshot_by_id(&self, snapshot_id: &str) -> Result<Option<Snapshot>> {
        let r = sqlx::query("SELECT * FROM graph.snapshot WHERE snapshot_id = $1")
            .bind(snapshot_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(pg_err)?;
        Ok(r.as_ref().map(snapshot_of))
    }

    async fn snapshots_by_status(&self, timescale: &str, family: &str, status: &str) -> Result<Vec<Snapshot>> {
        let rows = sqlx::query(
            "SELECT * FROM graph.snapshot WHERE timescale = $1 AND encoder_family = $2 AND status = $3
             ORDER BY created_at DESC, snapshot_id DESC",
        )
        .bind(timescale)
        .bind(family)
        .bind(status)
        .fetch_all(&self.pool)
        .await
        .map_err(pg_err)?;
        Ok(rows.iter().map(snapshot_of).collect())
    }

    async fn link_snapshot_parent(&self, parent: u64, child: u64, op: &str) -> Result<u64> {
        let r = sqlx::query(
            "INSERT INTO graph.snapshot_parent (parent_id, child_id, op) VALUES ($1, $2, $3)
             ON CONFLICT (parent_id, child_id) DO UPDATE SET op = $3 RETURNING id",
        )
        .bind(parent as i64)
        .bind(child as i64)
        .bind(op)
        .fetch_one(&self.pool)
        .await
        .map_err(pg_err)?;
        Ok(u(r.get("id")))
    }

    async fn snapshot_parent(&self, child: u64) -> Result<Option<Snapshot>> {
        let r = sqlx::query(
            "SELECT s.* FROM graph.snapshot s JOIN graph.snapshot_parent p ON p.parent_id = s.id WHERE p.child_id = $1 LIMIT 1",
        )
        .bind(child as i64).fetch_optional(&self.pool).await.map_err(pg_err)?;
        Ok(r.as_ref().map(snapshot_of))
    }

    async fn snapshot_children(&self, parent: u64) -> Result<Vec<Snapshot>> {
        let rows = sqlx::query(
            "SELECT s.* FROM graph.snapshot s JOIN graph.snapshot_parent p ON p.child_id = s.id WHERE p.parent_id = $1",
        )
        .bind(parent as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(pg_err)?;
        Ok(rows.iter().map(snapshot_of).collect())
    }

    async fn set_snapshot_status(&self, snapshot: u64, status: &str) -> Result<Snapshot> {
        let r = sqlx::query("UPDATE graph.snapshot SET status = $2 WHERE id = $1 RETURNING *")
            .bind(snapshot as i64)
            .bind(status)
            .fetch_optional(&self.pool)
            .await
            .map_err(pg_err)?;
        r.as_ref()
            .map(snapshot_of)
            .ok_or_else(|| HubError::NotFound { label: schema::node::SNAPSHOT.into(), id: snapshot.to_string() })
    }

    async fn set_snapshot_previous_active(&self, snapshot: u64, previous: &str) -> Result<Snapshot> {
        let r = sqlx::query("UPDATE graph.snapshot SET previous_active = $2 WHERE id = $1 RETURNING *")
            .bind(snapshot as i64)
            .bind(previous)
            .fetch_optional(&self.pool)
            .await
            .map_err(pg_err)?;
        r.as_ref()
            .map(snapshot_of)
            .ok_or_else(|| HubError::NotFound { label: schema::node::SNAPSHOT.into(), id: snapshot.to_string() })
    }

    async fn set_snapshot_manifest(&self, snapshot: u64, manifest_json: &str, carried_mass: f64) -> Result<Snapshot> {
        let r = sqlx::query("UPDATE graph.snapshot SET manifest = $2, carried_mass = $3 WHERE id = $1 RETURNING *")
            .bind(snapshot as i64)
            .bind(manifest_json)
            .bind(carried_mass)
            .fetch_optional(&self.pool)
            .await
            .map_err(pg_err)?;
        r.as_ref()
            .map(snapshot_of)
            .ok_or_else(|| HubError::NotFound { label: schema::node::SNAPSHOT.into(), id: snapshot.to_string() })
    }

    async fn add_adapter_state(&self, snapshot: u64, a: &NewAdapterState) -> Result<AdapterState> {
        let _ = self.snapshot_row(snapshot as i64).await?;
        let drift = a.drift.filter(|d| d.is_finite());
        let r = sqlx::query(
            "INSERT INTO graph.adapter_state (adapter_id, snapshot_id, timescale, version, garage_key, drift, max_drift, frozen, n_windows, window_start, window_end, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, now()) RETURNING *",
        )
        .bind(adapter_id(&a.snapshot_id, a.version)).bind(&a.snapshot_id).bind(&a.timescale)
        .bind(a.version as i64).bind(&a.garage_key).bind(drift).bind(a.max_drift).bind(a.frozen)
        .bind(a.n_windows as i64).bind(ms(a.window_start_ms)).bind(ms(a.window_end_ms))
        .fetch_one(&self.pool).await.map_err(pg_err)?;
        Ok(adapter_of(&r))
    }

    async fn snapshot_adapters(&self, snapshot_id: &str, limit: usize) -> Result<Vec<AdapterState>> {
        let rows =
            sqlx::query("SELECT * FROM graph.adapter_state WHERE snapshot_id = $1 ORDER BY version DESC LIMIT $2")
                .bind(snapshot_id)
                .bind(limit.max(1) as i64)
                .fetch_all(&self.pool)
                .await
                .map_err(pg_err)?;
        Ok(rows.iter().map(adapter_of).collect())
    }

    async fn add_expert_space(&self, snapshot: u64, space: &NewExpertSpace) -> Result<ExpertSpace> {
        let _ = self.snapshot_row(snapshot as i64).await?;
        let r = sqlx::query(
            "INSERT INTO graph.expert_space (expert_space_id, snapshot_id, \"index\", dim) VALUES ($1, $2, $3, $4) RETURNING *",
        )
        .bind(&space.expert_space_id).bind(&space.snapshot_id).bind(space.index as i32).bind(space.dim as i32)
        .fetch_one(&self.pool).await.map_err(pg_err)?;
        Ok(space_of(&r))
    }

    async fn ensure_expert_space(&self, snapshot: u64, space: &NewExpertSpace) -> Result<ExpertSpace> {
        let _ = self.snapshot_row(snapshot as i64).await?;
        sqlx::query(
            "INSERT INTO graph.expert_space (expert_space_id, snapshot_id, \"index\", dim) VALUES ($1, $2, $3, $4)
             ON CONFLICT (expert_space_id) DO NOTHING",
        )
        .bind(&space.expert_space_id)
        .bind(&space.snapshot_id)
        .bind(space.index as i32)
        .bind(space.dim as i32)
        .execute(&self.pool)
        .await
        .map_err(pg_err)?;
        let r = sqlx::query("SELECT * FROM graph.expert_space WHERE expert_space_id = $1")
            .bind(&space.expert_space_id)
            .fetch_one(&self.pool)
            .await
            .map_err(pg_err)?;
        Ok(space_of(&r))
    }

    async fn space_by_id(&self, expert_space_id: &str) -> Result<Option<ExpertSpace>> {
        let r = sqlx::query("SELECT * FROM graph.expert_space WHERE expert_space_id = $1")
            .bind(expert_space_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(pg_err)?;
        Ok(r.as_ref().map(space_of))
    }

    async fn snapshot_spaces(&self, snapshot: u64) -> Result<Vec<ExpertSpace>> {
        let rows = sqlx::query(
            "SELECT e.* FROM graph.expert_space e JOIN graph.snapshot s ON s.snapshot_id = e.snapshot_id WHERE s.id = $1 ORDER BY e.\"index\"",
        )
        .bind(snapshot as i64).fetch_all(&self.pool).await.map_err(pg_err)?;
        Ok(rows.iter().map(space_of).collect())
    }

    async fn add_region(&self, space: u64, region: &NewRegion) -> Result<LatentRegion> {
        let _ = sqlx::query_scalar::<_, i64>("SELECT id FROM graph.expert_space WHERE id = $1")
            .bind(space as i64)
            .fetch_optional(&self.pool)
            .await
            .map_err(pg_err)?
            .ok_or_else(|| HubError::NotFound { label: schema::node::EXPERT_SPACE.into(), id: space.to_string() })?;
        let r = sqlx::query(
            "INSERT INTO graph.region (region_id, lineage_id, expert_space_id, method, radius, density) VALUES ($1, $2, $3, $4, $5, $6) RETURNING *",
        )
        .bind(&region.region_id).bind(&region.lineage_id).bind(&region.expert_space_id)
        .bind(&region.method).bind(region.radius).bind(region.density)
        .fetch_one(&self.pool).await.map_err(pg_err)?;
        Ok(region_of(&r))
    }

    async fn upsert_region(&self, _space: u64, region: &NewRegion) -> Result<LatentRegion> {
        let mut tx = self.pool.begin().await.map_err(pg_err)?;
        let existing = sqlx::query("SELECT id, status FROM graph.region WHERE region_id = $1")
            .bind(&region.region_id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(pg_err)?;
        let row = if let Some(e) = existing {
            if e.get::<Option<String>, _>("status").as_deref() != Some("live") {
                version_status(&mut tx, &region.region_id, "live", "", "revived").await?;
            }
            sqlx::query(
                "UPDATE graph.region SET status = 'live', related_region_id = '', radius = $2, density = $3, updated_at = now()
                 WHERE region_id = $1 RETURNING *",
            )
            .bind(&region.region_id).bind(region.radius).bind(region.density)
            .fetch_one(&mut *tx).await.map_err(pg_err)?
        } else {
            version_status(&mut tx, &region.region_id, "live", "", "created").await?;
            sqlx::query(
                "INSERT INTO graph.region (region_id, lineage_id, expert_space_id, method, radius, density, status, related_region_id, updated_at)
                 VALUES ($1, $2, $3, $4, $5, $6, 'live', '', now()) RETURNING *",
            )
            .bind(&region.region_id).bind(&region.lineage_id).bind(&region.expert_space_id)
            .bind(&region.method).bind(region.radius).bind(region.density)
            .fetch_one(&mut *tx).await.map_err(pg_err)?
        };
        tx.commit().await.map_err(pg_err)?;
        Ok(region_of(&row))
    }

    async fn region_by_id(&self, region_id: &str) -> Result<Option<LatentRegion>> {
        let r = sqlx::query("SELECT * FROM graph.region WHERE region_id = $1")
            .bind(region_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(pg_err)?;
        Ok(r.as_ref().map(region_of))
    }

    async fn mark_region_status(&self, region_id: &str, status: &str, related: &str) -> Result<LatentRegion> {
        let mut tx = self.pool.begin().await.map_err(pg_err)?;
        let r = sqlx::query(
            "UPDATE graph.region SET status = $2, related_region_id = $3, updated_at = now() WHERE region_id = $1 RETURNING *",
        )
        .bind(region_id).bind(status).bind(related)
        .fetch_optional(&mut *tx).await.map_err(pg_err)?
        .ok_or_else(|| HubError::NotFound { label: schema::node::LATENT_REGION.into(), id: region_id.into() })?;
        version_status(&mut tx, region_id, status, related, "").await?;
        tx.commit().await.map_err(pg_err)?;
        Ok(region_of(&r))
    }

    async fn space_regions(&self, space: u64) -> Result<Vec<LatentRegion>> {
        let rows = sqlx::query(
            "SELECT r.* FROM graph.region r JOIN graph.expert_space e ON e.expert_space_id = r.expert_space_id WHERE e.id = $1",
        )
        .bind(space as i64).fetch_all(&self.pool).await.map_err(pg_err)?;
        Ok(rows.iter().map(region_of).collect())
    }

    async fn update_region_stats(&self, rows: &[RegionStatsUpdate]) -> Result<()> {
        for s in rows {
            sqlx::query(
                "UPDATE graph.region SET mass = $2, occupancy = $3, spread = $4, age = $5, updated_at = now() WHERE region_id = $1",
            )
            .bind(&s.region_id).bind(s.mass).bind(s.occupancy).bind(s.spread).bind(s.age.min(i64::MAX as u64) as i64)
            .execute(&self.pool).await.map_err(pg_err)?;
        }
        Ok(())
    }

    async fn mark_region_died(&self, region_id: &str, reason: &str, related: &str) -> Result<()> {
        let mut tx = self.pool.begin().await.map_err(pg_err)?;
        let n = sqlx::query(
            "UPDATE graph.region SET status = 'retired', retire_reason = $2, related_region_id = $3, died_at = now(), updated_at = now()
             WHERE region_id = $1",
        )
        .bind(region_id).bind(reason).bind(related)
        .execute(&mut *tx).await.map_err(pg_err)?.rows_affected();
        if n == 0 {
            return Err(HubError::NotFound { label: schema::node::LATENT_REGION.into(), id: region_id.into() });
        }
        version_status(&mut tx, region_id, "retired", related, reason).await?;
        tx.commit().await.map_err(pg_err)
    }

    async fn add_transition(&self, from: u64, to: u64, t: &crate::queries::NewTransition) -> Result<Transition> {
        let key = |id: u64| async move {
            sqlx::query_scalar::<_, String>("SELECT region_id FROM graph.region WHERE id = $1")
                .bind(id as i64)
                .fetch_optional(&self.pool)
                .await
                .map_err(pg_err)?
                .ok_or_else(|| HubError::NotFound { label: schema::node::LATENT_REGION.into(), id: id.to_string() })
        };
        let (f, tt) = (key(from).await?, key(to).await?);
        let r = sqlx::query(
            "INSERT INTO graph.transition (from_id, to_id, from_region_id, to_region_id, sampler_id, \"count\", mean, \"lower\", \"upper\", kernel_params, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, now()) RETURNING *",
        )
        .bind(from as i64).bind(to as i64).bind(&f).bind(&tt).bind(&t.sampler_id)
        .bind(t.count).bind(t.mean).bind(t.lower).bind(t.upper).bind(&t.kernel_params)
        .fetch_one(&self.pool).await.map_err(pg_err)?;
        Ok(transition_of(&r))
    }

    async fn upsert_transitions(&self, rows: &[TransitionUpsert]) -> Result<Vec<Transition>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let mut tx = self.pool.begin().await.map_err(pg_err)?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let ids = sqlx::query(
                "SELECT (SELECT id FROM graph.region WHERE region_id = $1) AS f, (SELECT id FROM graph.region WHERE region_id = $2) AS t",
            )
            .bind(&r.from_region_id).bind(&r.to_region_id)
            .fetch_one(&mut *tx).await.map_err(pg_err)?;
            let (Some(f), Some(t)) = (ids.get::<Option<i64>, _>("f"), ids.get::<Option<i64>, _>("t")) else {
                return Err(HubError::NotFound {
                    label: schema::node::LATENT_REGION.into(),
                    id: format!("{} -> {}", r.from_region_id, r.to_region_id),
                });
            };
            let tr = &r.transition;
            let row = sqlx::query(
                "INSERT INTO graph.transition (from_id, to_id, from_region_id, to_region_id, sampler_id, \"count\", mean, \"lower\", \"upper\", kernel_params, updated_at)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, now())
                 ON CONFLICT (from_region_id, to_region_id, sampler_id) DO UPDATE
                 SET \"count\" = $6, mean = $7, \"lower\" = $8, \"upper\" = $9, kernel_params = $10, updated_at = now()
                 RETURNING *",
            )
            .bind(f).bind(t).bind(&r.from_region_id).bind(&r.to_region_id).bind(&tr.sampler_id)
            .bind(tr.count).bind(tr.mean).bind(tr.lower).bind(tr.upper).bind(&tr.kernel_params)
            .fetch_one(&mut *tx).await.map_err(pg_err)?;
            sqlx::query(
                "INSERT INTO graph.transition_version (edge_key, from_region_id, to_region_id, sampler_id, \"count\", mean, \"lower\", \"upper\", kernel_params)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            )
            .bind(edge_key(&r.from_region_id, &r.to_region_id, &tr.sampler_id))
            .bind(&r.from_region_id).bind(&r.to_region_id).bind(&tr.sampler_id)
            .bind(tr.count).bind(tr.mean).bind(tr.lower).bind(tr.upper).bind(&tr.kernel_params)
            .execute(&mut *tx).await.map_err(pg_err)?;
            out.push(transition_of(&row));
        }
        tx.commit().await.map_err(pg_err)?;
        Ok(out)
    }

    async fn region_transitions(&self, region: u64) -> Result<Vec<Transition>> {
        let rows = sqlx::query("SELECT * FROM graph.transition WHERE from_id = $1")
            .bind(region as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(pg_err)?;
        Ok(rows.iter().map(transition_of).collect())
    }

    async fn region_neighbors(&self, region: u64) -> Result<Vec<LatentRegion>> {
        let rows = sqlx::query(
            "SELECT DISTINCT ON (r.id) r.* FROM graph.region r JOIN graph.transition t ON t.to_id = r.id WHERE t.from_id = $1",
        )
        .bind(region as i64).fetch_all(&self.pool).await.map_err(pg_err)?;
        Ok(rows.iter().map(region_of).collect())
    }

    async fn match_regions(&self, prev: u64, next: u64, score: f64) -> Result<u64> {
        let r = sqlx::query(
            "INSERT INTO graph.region_match (prev_id, next_id, prev_region_id, next_region_id, score)
             VALUES ($1, $2, COALESCE((SELECT region_id FROM graph.region WHERE id = $1), ''),
                     COALESCE((SELECT region_id FROM graph.region WHERE id = $2), ''), $3) RETURNING id",
        )
        .bind(prev as i64)
        .bind(next as i64)
        .bind(score)
        .fetch_one(&self.pool)
        .await
        .map_err(pg_err)?;
        Ok(u(r.get("id")))
    }

    async fn region_graph(&self, space: u64) -> Result<RegionGraph> {
        let regions = self.space_regions(space).await?;
        let transitions = sqlx::query(
            "SELECT t.* FROM graph.transition t JOIN graph.region r ON r.id = t.from_id
             JOIN graph.expert_space e ON e.expert_space_id = r.expert_space_id WHERE e.id = $1",
        )
        .bind(space as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(pg_err)?;
        Ok(RegionGraph { regions, transitions: transitions.iter().map(transition_of).collect() })
    }

    async fn snapshot_graph(&self, snapshot_id: &str) -> Result<Option<SnapshotGraph>> {
        let Some(snapshot) = self.snapshot_by_id(snapshot_id).await? else { return Ok(None) };
        let spaces = sqlx::query("SELECT * FROM graph.expert_space WHERE snapshot_id = $1 ORDER BY \"index\"")
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await
            .map_err(pg_err)?;
        let regions = sqlx::query(
            "SELECT r.* FROM graph.region r JOIN graph.expert_space e ON e.expert_space_id = r.expert_space_id WHERE e.snapshot_id = $1",
        )
        .bind(snapshot_id).fetch_all(&self.pool).await.map_err(pg_err)?;
        let transitions = sqlx::query(
            "SELECT t.* FROM graph.transition t JOIN graph.region r ON r.id = t.from_id
             JOIN graph.expert_space e ON e.expert_space_id = r.expert_space_id WHERE e.snapshot_id = $1",
        )
        .bind(snapshot_id)
        .fetch_all(&self.pool)
        .await
        .map_err(pg_err)?;
        Ok(Some(SnapshotGraph {
            snapshot,
            spaces: spaces.iter().map(space_of).collect(),
            regions: regions.iter().map(region_of).collect(),
            transitions: transitions.iter().map(transition_of).collect(),
        }))
    }

    async fn upsert_descents(&self, rows: &[NewDescent]) -> Result<()> {
        for d in rows {
            // skipped like the Helix probe when either region is missing
            sqlx::query(
                "INSERT INTO graph.descends_from (child_region_id, parent_region_id, kind, overlap, updated_at)
                 SELECT $1, $2, $3, $4, now()
                 WHERE EXISTS (SELECT 1 FROM graph.region WHERE region_id = $1) AND EXISTS (SELECT 1 FROM graph.region WHERE region_id = $2)
                 ON CONFLICT (child_region_id, parent_region_id, kind) DO UPDATE SET overlap = $4, updated_at = now()",
            )
            .bind(&d.child_region_id).bind(&d.parent_region_id).bind(&d.kind).bind(d.overlap)
            .execute(&self.pool).await.map_err(pg_err)?;
        }
        Ok(())
    }

    async fn upsert_hier_edges(&self, rows: &[NewHierEdge]) -> Result<()> {
        for h in rows {
            sqlx::query(
                "INSERT INTO graph.hier (fine_region_id, coarse_region_id, weight, p_coarse_given_fine, p_fine_given_coarse, support, pmi_bits, mi_bits, updated_at)
                 SELECT $1, $2, $3, $3, $4, $5, $6, $7, now()
                 WHERE EXISTS (SELECT 1 FROM graph.region WHERE region_id = $1) AND EXISTS (SELECT 1 FROM graph.region WHERE region_id = $2)
                 ON CONFLICT (fine_region_id, coarse_region_id) DO UPDATE
                 SET weight = $3, p_coarse_given_fine = $3, p_fine_given_coarse = $4, support = $5, pmi_bits = $6, mi_bits = $7, updated_at = now()",
            )
            .bind(&h.fine_region_id).bind(&h.coarse_region_id).bind(h.p_coarse_given_fine)
            .bind(h.p_fine_given_coarse).bind(h.support).bind(h.pmi_bits).bind(h.mi_bits)
            .execute(&self.pool).await.map_err(pg_err)?;
        }
        Ok(())
    }

    async fn snapshot_hierarchy(&self, snapshot_id: &str) -> Result<Vec<EdgeRow>> {
        let rows = sqlx::query(
            "SELECT h.*, rf.id AS fine_id, rc.id AS coarse_id FROM graph.hier h
             JOIN graph.region rf ON rf.region_id = h.fine_region_id
             JOIN graph.region rc ON rc.region_id = h.coarse_region_id
             JOIN graph.expert_space e ON e.expert_space_id = rf.expert_space_id
             WHERE e.snapshot_id = $1",
        )
        .bind(snapshot_id)
        .fetch_all(&self.pool)
        .await
        .map_err(pg_err)?;
        Ok(rows
            .iter()
            .map(|r| {
                edge_row(r, schema::edge::HIER, u(r.get("fine_id")), u(r.get("coarse_id")), &["fine_id", "coarse_id"])
            })
            .collect())
    }

    async fn lineage_members(&self, lineage_id: &str) -> Result<Vec<NodeRow>> {
        let rows = sqlx::query("SELECT * FROM graph.region WHERE lineage_id = $1")
            .bind(lineage_id)
            .fetch_all(&self.pool)
            .await
            .map_err(pg_err)?;
        Ok(rows.iter().map(|r| node_row(r, schema::node::LATENT_REGION)).collect())
    }

    async fn region_lineage(&self, region_id: &str, depth: usize) -> Result<Option<RegionLineage>> {
        let Some(start) = sqlx::query("SELECT * FROM graph.region WHERE region_id = $1")
            .bind(region_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(pg_err)?
        else {
            return Ok(None);
        };
        let region = node_row(&start, schema::node::LATENT_REGION);
        let lineage_id: String = start.get("lineage_id");
        let lineage = if lineage_id.is_empty() { Vec::new() } else { self.lineage_members(&lineage_id).await? };
        let mut out = [Vec::new(), Vec::new()];
        for (slot, up) in [(0usize, true), (1usize, false)] {
            let mut seen = vec![region_id.to_owned()];
            let mut frontier = vec![region_id.to_owned()];
            for level in 1..=depth.max(1) {
                if frontier.is_empty() {
                    break;
                }
                // ancestors: DescendsFrom child→parent walked out, RegionMatch prev→next walked in
                let sql = if up {
                    "SELECT parent_region_id AS next, child_region_id AS via, kind, overlap FROM graph.descends_from WHERE child_region_id = ANY($1)
                     UNION ALL
                     SELECT prev_region_id AS next, next_region_id AS via, 'carry' AS kind, score AS overlap FROM graph.region_match WHERE next_region_id = ANY($1)"
                } else {
                    "SELECT child_region_id AS next, parent_region_id AS via, kind, overlap FROM graph.descends_from WHERE parent_region_id = ANY($1)
                     UNION ALL
                     SELECT next_region_id AS next, prev_region_id AS via, 'carry' AS kind, score AS overlap FROM graph.region_match WHERE prev_region_id = ANY($1)"
                };
                let edges = sqlx::query(sql).bind(&frontier).fetch_all(&self.pool).await.map_err(pg_err)?;
                frontier = Vec::new();
                for e in &edges {
                    let next: String = e.get("next");
                    if next.is_empty() || seen.contains(&next) {
                        continue;
                    }
                    let Some(row) = sqlx::query("SELECT * FROM graph.region WHERE region_id = $1")
                        .bind(&next)
                        .fetch_optional(&self.pool)
                        .await
                        .map_err(pg_err)?
                    else {
                        continue;
                    };
                    seen.push(next.clone());
                    frontier.push(next.clone());
                    out[slot].push(LineageLink {
                        region: node_row(&row, schema::node::LATENT_REGION),
                        kind: e.get("kind"),
                        overlap: e.get("overlap"),
                        depth: level,
                        via: e.get("via"),
                    });
                }
            }
        }
        let [ancestors, descendants] = out;
        Ok(Some(RegionLineage { region, lineage, ancestors, descendants }))
    }

    async fn upsert_region_set(&self, r: &NewRegionSet) -> Result<()> {
        sqlx::query(
            "INSERT INTO graph.region_set (region_set_id, snapshot_id, timescale, expert_index, family, method, n_regions, cfg_hash, state, kernel_state, fit_time, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, now(), now())
             ON CONFLICT (region_set_id) DO UPDATE SET snapshot_id = $2, timescale = $3, expert_index = $4, family = $5,
                 method = $6, n_regions = $7, cfg_hash = $8, state = $9, kernel_state = $10, fit_time = now(), updated_at = now()",
        )
        .bind(&r.expert_space_id).bind(&r.snapshot_id).bind(&r.timescale).bind(r.expert_index as i32)
        .bind(&r.family).bind(&r.method).bind(r.n_regions as i32).bind(&r.cfg_hash).bind(&r.state).bind(&r.kernel_state)
        .execute(&self.pool).await.map_err(pg_err).map(drop)
    }

    async fn region_set(&self, expert_space_id: &str) -> Result<Option<RegionSetRow>> {
        let r = sqlx::query("SELECT * FROM graph.region_set WHERE region_set_id = $1")
            .bind(expert_space_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(pg_err)?;
        Ok(r.map(|r| RegionSetRow {
            id: u(r.get("id")),
            region_set_id: r.get("region_set_id"),
            snapshot_id: r.get("snapshot_id"),
            timescale: r.get("timescale"),
            expert_index: r.get::<i32, _>("expert_index") as u32,
            n_regions: r.get::<i32, _>("n_regions") as u32,
            cfg_hash: r.get("cfg_hash"),
            state: r.get("state"),
            kernel_state: r.get("kernel_state"),
            updated_at: Some(t3339(r.get("updated_at"))),
        }))
    }

    async fn upsert_kernel_model(&self, k: &NewKernelModel) -> Result<()> {
        sqlx::query(
            "INSERT INTO graph.kernel_model (kernel_id, expert_space_id, snapshot_id, timescale, expert_index, kernel_type, serving_sampler, scores, stationary, relaxation_steps, drift_kl, n_rows, observation_mass, live_regions, updated_at)
             VALUES ($1, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, now())
             ON CONFLICT (kernel_id) DO UPDATE SET snapshot_id = $2, timescale = $3, expert_index = $4, kernel_type = $5,
                 serving_sampler = $6, scores = $7, stationary = $8, relaxation_steps = $9, drift_kl = $10, n_rows = $11,
                 observation_mass = $12, live_regions = $13, updated_at = now()",
        )
        .bind(&k.expert_space_id).bind(&k.snapshot_id).bind(&k.timescale).bind(k.expert_index as i32)
        .bind(&k.kernel_type).bind(&k.serving_sampler).bind(&k.scores).bind(&k.stationary)
        .bind(k.relaxation_steps).bind(k.drift_kl).bind(k.n_rows as i32).bind(k.observation_mass).bind(k.live_regions as i32)
        .execute(&self.pool).await.map_err(pg_err).map(drop)
    }

    async fn snapshot_kernels(&self, snapshot_id: &str) -> Result<Vec<NodeRow>> {
        let rows = sqlx::query("SELECT * FROM graph.kernel_model WHERE snapshot_id = $1")
            .bind(snapshot_id)
            .fetch_all(&self.pool)
            .await
            .map_err(pg_err)?;
        Ok(rows.iter().map(|r| node_row(r, crate::regions::labels::KERNEL_MODEL)).collect())
    }

    async fn upsert_cross_priors(&self, rows: &[NewCrossPrior]) -> Result<()> {
        for c in rows {
            let unc = if c.uncertainty.is_finite() { c.uncertainty } else { f64::MAX };
            sqlx::query(
                "INSERT INTO graph.cross_prior (pair_id, timescale, source_instrument_id, target_instrument_id, \"n\", correlation, shrinkage, lead_lag, lead_corr, predictive_gain, uncertainty, score, \"rank\", updated_at)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, now())
                 ON CONFLICT (pair_id) DO UPDATE SET \"n\" = $5, correlation = $6, shrinkage = $7, lead_lag = $8,
                     lead_corr = $9, predictive_gain = $10, uncertainty = $11, score = $12, \"rank\" = $13, updated_at = now()",
            )
            .bind(c.pair_id()).bind(&c.timescale).bind(&c.source_instrument_id).bind(&c.target_instrument_id)
            .bind(c.n as i32).bind(c.correlation).bind(c.shrinkage).bind(c.lead_lag as i32)
            .bind(c.lead_corr).bind(c.predictive_gain).bind(unc).bind(c.score).bind(c.rank as i32)
            .execute(&self.pool).await.map_err(pg_err)?;
        }
        Ok(())
    }

    async fn cross_priors(&self, timescale: &str, limit: usize) -> Result<Vec<NodeRow>> {
        let rows = sqlx::query("SELECT * FROM graph.cross_prior WHERE timescale = $1 ORDER BY score DESC LIMIT $2")
            .bind(timescale)
            .bind(limit.max(1) as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(pg_err)?;
        Ok(rows.iter().map(|r| node_row(r, crate::regions::labels::CROSS_PRIOR)).collect())
    }

    async fn upsert_region_episodes(&self, rows: &[NewRegionEpisode]) -> Result<()> {
        for e in rows {
            sqlx::query(
                "INSERT INTO graph.region_episode (episode_id, region_id, expert_space_id, timescale, instrument_id, t_enter, t_exit, bars, max_weight, mean_weight)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                 ON CONFLICT (episode_id) DO UPDATE SET t_exit = $7, bars = $8, max_weight = $9, mean_weight = $10",
            )
            .bind(e.episode_id()).bind(&e.region_id).bind(&e.expert_space_id).bind(&e.timescale).bind(&e.instrument_id)
            .bind(ms(e.t_enter_ms)).bind(ms(e.t_exit_ms)).bind(e.bars as i32)
            .bind(if e.max_weight.is_finite() { e.max_weight } else { f64::MAX })
            .bind(if e.mean_weight.is_finite() { e.mean_weight } else { f64::MAX })
            .execute(&self.pool).await.map_err(pg_err)?;
        }
        Ok(())
    }

    async fn region_episodes(&self, region_id: &str, limit: usize) -> Result<Vec<NodeRow>> {
        let rows = sqlx::query("SELECT * FROM graph.region_episode WHERE region_id = $1 ORDER BY t_exit DESC LIMIT $2")
            .bind(region_id)
            .bind(limit.max(1) as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(pg_err)?;
        Ok(rows.iter().map(|r| node_row(r, crate::regions::labels::REGION_EPISODE)).collect())
    }

    async fn prune_region_episodes(&self, timescale: &str, cutoff_ms: i64) -> Result<u64> {
        let n = sqlx::query("DELETE FROM graph.region_episode WHERE timescale = $1 AND t_exit < $2")
            .bind(timescale)
            .bind(ms(cutoff_ms))
            .execute(&self.pool)
            .await
            .map_err(pg_err)?;
        Ok(n.rows_affected())
    }

    async fn transition_versions(
        &self,
        from: &str,
        to: &str,
        sampler: &str,
        as_of_ms: Option<i64>,
        limit: usize,
    ) -> Result<Vec<TransitionVersion>> {
        let rows = sqlx::query(
            "SELECT * FROM graph.transition_version WHERE edge_key = $1 AND ($2::timestamptz IS NULL OR ts <= $2)
             ORDER BY ts DESC, id DESC LIMIT $3",
        )
        .bind(edge_key(from, to, sampler))
        .bind(as_of_ms.map(ms))
        .bind(limit.max(1) as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(pg_err)?;
        Ok(rows.iter().map(transition_version_of).collect())
    }

    async fn region_status_versions(
        &self,
        region_id: &str,
        as_of_ms: Option<i64>,
        limit: usize,
    ) -> Result<Vec<RegionStatusVersion>> {
        let rows = sqlx::query(
            "SELECT * FROM graph.region_status_version WHERE region_id = $1 AND ($2::timestamptz IS NULL OR ts <= $2)
             ORDER BY ts DESC, id DESC LIMIT $3",
        )
        .bind(region_id)
        .bind(as_of_ms.map(ms))
        .bind(limit.max(1) as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(pg_err)?;
        Ok(rows.iter().map(status_version_of).collect())
    }

    async fn latest_transition_versions(&self, keys: &[String], t_ms: i64) -> Result<Vec<Option<TransitionVersion>>> {
        let rows = sqlx::query(
            "SELECT DISTINCT ON (edge_key) * FROM graph.transition_version WHERE edge_key = ANY($1) AND ts <= $2
             ORDER BY edge_key, ts DESC, id DESC",
        )
        .bind(keys)
        .bind(ms(t_ms))
        .fetch_all(&self.pool)
        .await
        .map_err(pg_err)?;
        let by: std::collections::HashMap<String, TransitionVersion> =
            rows.iter().map(|r| (r.get("edge_key"), transition_version_of(r))).collect();
        Ok(keys.iter().map(|k| by.get(k).cloned()).collect())
    }

    async fn latest_region_status_versions(
        &self,
        keys: &[String],
        t_ms: i64,
    ) -> Result<Vec<Option<RegionStatusVersion>>> {
        let rows = sqlx::query(
            "SELECT DISTINCT ON (region_id) * FROM graph.region_status_version WHERE region_id = ANY($1) AND ts <= $2
             ORDER BY region_id, ts DESC, id DESC",
        )
        .bind(keys)
        .bind(ms(t_ms))
        .fetch_all(&self.pool)
        .await
        .map_err(pg_err)?;
        let by: std::collections::HashMap<String, RegionStatusVersion> =
            rows.iter().map(|r| (r.get("region_id"), status_version_of(r))).collect();
        Ok(keys.iter().map(|k| by.get(k).cloned()).collect())
    }

    async fn prune_space_versions(&self, space: u64, cutoff_ms: i64) -> Result<u64> {
        // keep, per key, the newest version older than the cut-off (the value in force there)
        let t = sqlx::query(
            "DELETE FROM graph.transition_version v USING (
                 SELECT id, row_number() OVER (PARTITION BY edge_key ORDER BY ts DESC, id DESC) rn
                 FROM graph.transition_version
                 WHERE ts < $2 AND edge_key IN (
                     SELECT t.from_region_id || '|' || t.to_region_id || '|' || t.sampler_id
                     FROM graph.transition t JOIN graph.region r ON r.id = t.from_id
                     JOIN graph.expert_space e ON e.expert_space_id = r.expert_space_id WHERE e.id = $1)
             ) old WHERE v.id = old.id AND old.rn > 1",
        )
        .bind(space as i64)
        .bind(ms(cutoff_ms))
        .execute(&self.pool)
        .await
        .map_err(pg_err)?
        .rows_affected();
        let s = sqlx::query(
            "DELETE FROM graph.region_status_version v USING (
                 SELECT id, row_number() OVER (PARTITION BY region_id ORDER BY ts DESC, id DESC) rn
                 FROM graph.region_status_version
                 WHERE ts < $2 AND region_id IN (
                     SELECT r.region_id FROM graph.region r
                     JOIN graph.expert_space e ON e.expert_space_id = r.expert_space_id WHERE e.id = $1)
             ) old WHERE v.id = old.id AND old.rn > 1",
        )
        .bind(space as i64)
        .bind(ms(cutoff_ms))
        .execute(&self.pool)
        .await
        .map_err(pg_err)?
        .rows_affected();
        Ok(t + s)
    }

    async fn list_snapshots_as_of(&self, as_of_ms: i64) -> Result<Vec<Snapshot>> {
        let rows = sqlx::query("SELECT * FROM graph.snapshot WHERE created_at <= $1 ORDER BY snapshot_id")
            .bind(ms(as_of_ms))
            .fetch_all(&self.pool)
            .await
            .map_err(pg_err)?;
        Ok(rows.iter().map(snapshot_of).collect())
    }

    async fn space_regions_as_of(&self, space: u64, as_of_ms: i64) -> Result<Vec<LatentRegion>> {
        let rows = sqlx::query(
            "SELECT r.* FROM graph.region r JOIN graph.expert_space e ON e.expert_space_id = r.expert_space_id
             WHERE e.id = $1 AND r.created_at <= $2",
        )
        .bind(space as i64)
        .bind(ms(as_of_ms))
        .fetch_all(&self.pool)
        .await
        .map_err(pg_err)?;
        Ok(rows.iter().map(region_of).collect())
    }

    async fn snapshot_graph_as_of(&self, snapshot_id: &str, as_of_ms: i64) -> Result<Option<SnapshotGraph>> {
        let Some(g) = self.snapshot_graph(snapshot_id).await? else { return Ok(None) };
        if g.snapshot.created_at.as_deref().and_then(crate::retention::rfc3339_ms).is_none_or(|c| c > as_of_ms) {
            return Ok(None);
        }
        let regions: Vec<LatentRegion> = g
            .regions
            .into_iter()
            .filter(|r| r.created_at.as_deref().and_then(crate::retention::rfc3339_ms).is_some_and(|c| c <= as_of_ms))
            .collect();
        let visible: std::collections::HashSet<u64> = regions.iter().map(|r| r.id).collect();
        let transitions =
            g.transitions.into_iter().filter(|t| visible.contains(&t.from) && visible.contains(&t.to)).collect();
        Ok(Some(SnapshotGraph { snapshot: g.snapshot, spaces: g.spaces, regions, transitions }))
    }

    async fn link_trained_on(&self, links: &[TrainedOnLink]) -> Result<()> {
        for l in links {
            sqlx::query(
                "INSERT INTO graph.trained_on (genome_id, snapshot_id, generation) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
            )
            .bind(&l.genome_id).bind(&l.snapshot_id).bind(l.generation as i32)
            .execute(&self.pool).await.map_err(pg_err)?;
        }
        Ok(())
    }

    async fn link_feedback_from(&self, links: &[FeedbackLink]) -> Result<()> {
        for l in links {
            sqlx::query(
                "INSERT INTO graph.feedback_from (snapshot_id, genome_id, weight, output_family, score) VALUES ($1, $2, $3, $4, $5)
                 ON CONFLICT (snapshot_id, genome_id) DO UPDATE SET weight = $3, output_family = $4, score = $5",
            )
            .bind(&l.snapshot_id).bind(&l.genome_id).bind(l.weight).bind(&l.output_family).bind(l.score)
            .execute(&self.pool).await.map_err(pg_err)?;
        }
        Ok(())
    }

    async fn trained_on_snapshots(&self, genome_id: &str) -> Result<Vec<Snapshot>> {
        let rows = sqlx::query(
            "SELECT s.* FROM graph.snapshot s JOIN graph.trained_on t ON t.snapshot_id = s.snapshot_id WHERE t.genome_id = $1",
        )
        .bind(genome_id).fetch_all(&self.pool).await.map_err(pg_err)?;
        Ok(rows.iter().map(snapshot_of).collect())
    }

    async fn feedback_edges(&self, snapshot_id: &str) -> Result<Vec<EdgeRow>> {
        let rows = sqlx::query(
            "SELECT f.*, s.id AS snap_id FROM graph.feedback_from f
             LEFT JOIN graph.snapshot s ON s.snapshot_id = f.snapshot_id WHERE f.snapshot_id = $1",
        )
        .bind(snapshot_id)
        .fetch_all(&self.pool)
        .await
        .map_err(pg_err)?;
        Ok(rows
            .iter()
            .map(|r| {
                let from = r.get::<Option<i64>, _>("snap_id").map(u).unwrap_or_default();
                edge_row(r, schema::edge::FEEDBACK_FROM, from, 0, &["snap_id"])
            })
            .collect())
    }
}

async fn version_status(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    region_id: &str,
    status: &str,
    related: &str,
    reason: &str,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO graph.region_status_version (region_id, status, related_region_id, reason) VALUES ($1, $2, $3, $4)",
    )
    .bind(region_id).bind(status).bind(related).bind(reason)
    .execute(&mut **tx).await.map_err(pg_err).map(drop)
}

fn transition_version_of(r: &sqlx::postgres::PgRow) -> TransitionVersion {
    TransitionVersion {
        id: u(r.get("id")),
        edge_key: r.get("edge_key"),
        from_region_id: r.get("from_region_id"),
        to_region_id: r.get("to_region_id"),
        sampler_id: r.get("sampler_id"),
        count: r.get("count"),
        mean: r.get("mean"),
        lower: r.get("lower"),
        upper: r.get("upper"),
        kernel_params: r.get("kernel_params"),
        ts: Some(t3339(r.get("ts"))),
    }
}

fn status_version_of(r: &sqlx::postgres::PgRow) -> RegionStatusVersion {
    RegionStatusVersion {
        id: u(r.get("id")),
        region_id: r.get("region_id"),
        status: r.get("status"),
        related_region_id: r.get("related_region_id"),
        reason: r.get("reason"),
        ts: Some(t3339(r.get("ts"))),
    }
}

/// The Helix backend of [`GraphStore`]: a `HubClient` with no Postgres graph attached, so every
/// call runs the original Helix query layer. Used by the conformance suite and `graph_migrate`.
#[derive(Clone, Debug)]
pub struct HelixGraph(pub crate::HubClient);

impl HelixGraph {
    /// Wraps a client, detaching any Postgres graph so calls cannot route back.
    pub fn new(client: crate::HubClient) -> Self {
        Self(client.with_graph(None))
    }
}

impl GraphStore for HelixGraph {
    async fn upsert_ref_entities(&self, rows: &[RefEntity]) -> Result<usize> {
        self.0.upsert_ref_entities(rows).await
    }
    async fn upsert_ref_venues(&self, rows: &[RefVenue]) -> Result<usize> {
        self.0.upsert_ref_venues(rows).await
    }
    async fn upsert_ref_instruments(&self, rows: &[RefInstrument]) -> Result<RefSyncReport> {
        self.0.upsert_ref_instruments(rows).await
    }
    async fn retire_ref_instruments(&self, ids: &[String]) -> Result<usize> {
        self.0.retire_ref_instruments(ids).await
    }
    async fn ref_instrument_ids(&self) -> Result<Vec<(String, String)>> {
        self.0.ref_instrument_ids().await
    }
    async fn instrument_reference(&self, id: &str) -> Result<Option<InstrumentReference>> {
        self.0.instrument_reference(id).await
    }
    async fn upsert_instrument(&self, inst: &NewInstrument) -> Result<Instrument> {
        self.0.upsert_instrument(inst).await
    }
    async fn get_instrument(&self, id: &str) -> Result<Option<Instrument>> {
        self.0.get_instrument(id).await
    }
    async fn add_snapshot(&self, snap: &NewSnapshot) -> Result<Snapshot> {
        self.0.add_snapshot(snap).await
    }
    async fn ensure_snapshot(&self, snap: &NewSnapshot) -> Result<Snapshot> {
        self.0.ensure_snapshot(snap).await
    }
    async fn list_snapshots(&self) -> Result<Vec<Snapshot>> {
        self.0.list_snapshots().await
    }
    async fn snapshot_by_id(&self, id: &str) -> Result<Option<Snapshot>> {
        self.0.snapshot_by_id(id).await
    }
    async fn snapshots_by_status(&self, ts: &str, fam: &str, st: &str) -> Result<Vec<Snapshot>> {
        self.0.snapshots_by_status(ts, fam, st).await
    }
    async fn link_snapshot_parent(&self, p: u64, c: u64, op: &str) -> Result<u64> {
        self.0.link_snapshot_parent(p, c, op).await
    }
    async fn snapshot_parent(&self, c: u64) -> Result<Option<Snapshot>> {
        self.0.snapshot_parent(c).await
    }
    async fn snapshot_children(&self, p: u64) -> Result<Vec<Snapshot>> {
        self.0.snapshot_children(p).await
    }
    async fn set_snapshot_status(&self, s: u64, st: &str) -> Result<Snapshot> {
        self.0.set_snapshot_status(s, st).await
    }
    async fn set_snapshot_previous_active(&self, s: u64, pr: &str) -> Result<Snapshot> {
        self.0.set_snapshot_previous_active(s, pr).await
    }
    async fn set_snapshot_manifest(&self, s: u64, m: &str, cm: f64) -> Result<Snapshot> {
        self.0.set_snapshot_manifest(s, m, cm).await
    }
    async fn add_adapter_state(&self, s: u64, a: &NewAdapterState) -> Result<AdapterState> {
        self.0.add_adapter_state(s, a).await
    }
    async fn snapshot_adapters(&self, id: &str, n: usize) -> Result<Vec<AdapterState>> {
        self.0.snapshot_adapters(id, n).await
    }
    async fn add_expert_space(&self, s: u64, sp: &NewExpertSpace) -> Result<ExpertSpace> {
        self.0.add_expert_space(s, sp).await
    }
    async fn ensure_expert_space(&self, s: u64, sp: &NewExpertSpace) -> Result<ExpertSpace> {
        self.0.ensure_expert_space(s, sp).await
    }
    async fn space_by_id(&self, id: &str) -> Result<Option<ExpertSpace>> {
        self.0.space_by_id(id).await
    }
    async fn snapshot_spaces(&self, s: u64) -> Result<Vec<ExpertSpace>> {
        self.0.snapshot_spaces(s).await
    }
    async fn add_region(&self, sp: u64, r: &NewRegion) -> Result<LatentRegion> {
        self.0.add_region(sp, r).await
    }
    async fn upsert_region(&self, sp: u64, r: &NewRegion) -> Result<LatentRegion> {
        self.0.upsert_region(sp, r).await
    }
    async fn region_by_id(&self, id: &str) -> Result<Option<LatentRegion>> {
        self.0.region_by_id(id).await
    }
    async fn mark_region_status(&self, id: &str, st: &str, rel: &str) -> Result<LatentRegion> {
        self.0.mark_region_status(id, st, rel).await
    }
    async fn space_regions(&self, sp: u64) -> Result<Vec<LatentRegion>> {
        self.0.space_regions(sp).await
    }
    async fn update_region_stats(&self, rows: &[RegionStatsUpdate]) -> Result<()> {
        self.0.update_region_stats(rows).await
    }
    async fn mark_region_died(&self, id: &str, why: &str, rel: &str) -> Result<()> {
        self.0.mark_region_died(id, why, rel).await
    }
    async fn add_transition(&self, f: u64, t: u64, tr: &crate::queries::NewTransition) -> Result<Transition> {
        self.0.add_transition(f, t, tr).await
    }
    async fn upsert_transitions(&self, rows: &[TransitionUpsert]) -> Result<Vec<Transition>> {
        self.0.upsert_transitions(rows).await
    }
    async fn region_transitions(&self, r: u64) -> Result<Vec<Transition>> {
        self.0.region_transitions(r).await
    }
    async fn region_neighbors(&self, r: u64) -> Result<Vec<LatentRegion>> {
        self.0.region_neighbors(r).await
    }
    async fn match_regions(&self, p: u64, n: u64, s: f64) -> Result<u64> {
        self.0.match_regions(p, n, s).await
    }
    async fn region_graph(&self, sp: u64) -> Result<RegionGraph> {
        self.0.region_graph(sp).await
    }
    async fn snapshot_graph(&self, id: &str) -> Result<Option<SnapshotGraph>> {
        self.0.snapshot_graph(id).await
    }
    async fn upsert_descents(&self, rows: &[NewDescent]) -> Result<()> {
        self.0.upsert_descents(rows).await
    }
    async fn upsert_hier_edges(&self, rows: &[NewHierEdge]) -> Result<()> {
        self.0.upsert_hier_edges(rows).await
    }
    async fn snapshot_hierarchy(&self, id: &str) -> Result<Vec<EdgeRow>> {
        self.0.snapshot_hierarchy(id).await
    }
    async fn lineage_members(&self, id: &str) -> Result<Vec<NodeRow>> {
        self.0.lineage_members(id).await
    }
    async fn region_lineage(&self, id: &str, d: usize) -> Result<Option<RegionLineage>> {
        self.0.region_lineage(id, d).await
    }
    async fn upsert_region_set(&self, r: &NewRegionSet) -> Result<()> {
        self.0.upsert_region_set(r).await
    }
    async fn region_set(&self, id: &str) -> Result<Option<RegionSetRow>> {
        self.0.region_set(id).await
    }
    async fn upsert_kernel_model(&self, k: &NewKernelModel) -> Result<()> {
        self.0.upsert_kernel_model(k).await
    }
    async fn snapshot_kernels(&self, id: &str) -> Result<Vec<NodeRow>> {
        self.0.snapshot_kernels(id).await
    }
    async fn upsert_cross_priors(&self, rows: &[NewCrossPrior]) -> Result<()> {
        self.0.upsert_cross_priors(rows).await
    }
    async fn cross_priors(&self, ts: &str, n: usize) -> Result<Vec<NodeRow>> {
        self.0.cross_priors(ts, n).await
    }
    async fn upsert_region_episodes(&self, rows: &[NewRegionEpisode]) -> Result<()> {
        self.0.upsert_region_episodes(rows).await
    }
    async fn region_episodes(&self, id: &str, n: usize) -> Result<Vec<NodeRow>> {
        self.0.region_episodes(id, n).await
    }
    async fn prune_region_episodes(&self, ts: &str, cut: i64) -> Result<u64> {
        self.0.prune_region_episodes(ts, cut).await
    }
    async fn transition_versions(
        &self,
        f: &str,
        t: &str,
        s: &str,
        a: Option<i64>,
        n: usize,
    ) -> Result<Vec<TransitionVersion>> {
        self.0.transition_versions(f, t, s, a, n).await
    }
    async fn region_status_versions(&self, id: &str, a: Option<i64>, n: usize) -> Result<Vec<RegionStatusVersion>> {
        self.0.region_status_versions(id, a, n).await
    }
    async fn latest_transition_versions(&self, keys: &[String], t: i64) -> Result<Vec<Option<TransitionVersion>>> {
        self.0.latest_transition_versions_helix(keys, t).await
    }
    async fn latest_region_status_versions(&self, keys: &[String], t: i64) -> Result<Vec<Option<RegionStatusVersion>>> {
        self.0.latest_region_status_versions_helix(keys, t).await
    }
    async fn prune_space_versions(&self, sp: u64, cut: i64) -> Result<u64> {
        self.0.prune_space_versions(sp, cut).await
    }
    async fn list_snapshots_as_of(&self, t: i64) -> Result<Vec<Snapshot>> {
        self.0.list_snapshots_as_of(t).await
    }
    async fn space_regions_as_of(&self, sp: u64, t: i64) -> Result<Vec<LatentRegion>> {
        self.0.space_regions_as_of(sp, t).await
    }
    async fn snapshot_graph_as_of(&self, id: &str, t: i64) -> Result<Option<SnapshotGraph>> {
        self.0.snapshot_graph_as_of(id, t).await
    }
    async fn link_trained_on(&self, links: &[TrainedOnLink]) -> Result<()> {
        self.0.link_trained_on(links).await
    }
    async fn link_feedback_from(&self, links: &[FeedbackLink]) -> Result<()> {
        self.0.link_feedback_from(links).await
    }
    async fn trained_on_snapshots(&self, id: &str) -> Result<Vec<Snapshot>> {
        self.0.genome_trained_on(id).await
    }
    async fn feedback_edges(&self, id: &str) -> Result<Vec<EdgeRow>> {
        Ok(self.0.snapshot_feedback(id).await?.0)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn id_and_time_helpers() {
        assert_eq!(u(7), 7);
        let t = ms(1_700_000_000_000);
        assert!(t3339(t).starts_with("2023-11-14T22:13:20"));
        let e = pg_err(sqlx::Error::PoolTimedOut);
        assert!(matches!(e, HubError::Graph(_)));
        assert_eq!(json!(1).as_i64(), Some(1));
    }
}
