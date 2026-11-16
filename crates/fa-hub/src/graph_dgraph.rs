//! The Dgraph [`GraphStore`](crate::graph::GraphStore) backend (ADR 0011, D-110).
//!
//! Shapes that need care, because DQL has no equivalent of the SQL the Postgres backend used:
//!
//! - **Upsert by business key** replaces `INSERT … ON CONFLICT … DO UPDATE`. One upsert block binds
//!   `v as var(func: eq(<key>, …))` and mutates `uid(v)`, which Dgraph resolves to the existing
//!   node or to a fresh one. The uid is then read back with a second query rather than fished out
//!   of the mutation response: `uids` only carries blank nodes, and which of the two cases
//!   occurred is exactly what the caller must not have to care about.
//! - **`DISTINCT ON (key) … WHERE key = ANY($1) AND ts <= $2`** (newest version per key as of `t`)
//!   becomes one named query block per key in a single request — the same shape the Helix backend
//!   already used in `versions.rs`. ADR 0010 measured this at 209/505 ms against Postgres's
//!   8.9/10.4 ms; D-110 accepts that.
//! - **`row_number() OVER (PARTITION BY …)` inside `DELETE … USING`** (version pruning) has no DQL
//!   form at all: candidates are read, ranked in Rust, and deleted by uid.
//! - **`props()`**, which reflected `PgRow` columns into a JSON map, becomes a rename of the
//!   `<type>.<field>` predicates back to the bare field names the explorer and frontend expect.
use serde_json::{Map, Value as Json, json};

use crate::dgraph::{DgraphClient, u64_to_uid, uid_to_u64, unique_violation};
use crate::dgraph_schema as dsch;
use crate::queries::{
    AdapterState, EdgeRow, ExpertSpace, Instrument, LatentRegion, NewExpertSpace, NewInstrument, NewRegion,
    NewSnapshot, NodeRow, RegionGraph, Snapshot, SnapshotGraph, Transition,
};
use crate::reference::{InstrumentReference, RefEntity, RefInstrument, RefSyncReport, RefVenue};
use crate::regions::{
    LineageLink, NewCrossPrior, NewDescent, NewHierEdge, NewKernelModel, NewRegionEpisode, NewRegionSet,
    RegionLineage, RegionSetRow, RegionStatsUpdate,
};
use crate::versions::RegionStatusVersion;
use crate::versions::TransitionVersion;
use crate::{HubError, Result};

/// The Dgraph graph store. Cheap to clone.
#[derive(Clone, Debug)]
pub struct DgraphGraph {
    dg: DgraphClient,
}

/// Escape a DQL string literal. Business ids come from catalogues and model code, so they can
/// legitimately contain quotes; an unescaped one would be a query-injection bug, not a typo.
pub(crate) fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// `"a", "b"` — a DQL value list for `eq(pred, [..])`.
pub(crate) fn esc_list<'a>(vals: impl IntoIterator<Item = &'a str>) -> String {
    vals.into_iter().map(|v| format!("\"{}\"", esc(v))).collect::<Vec<_>>().join(", ")
}

pub(crate) fn str_of(j: &Json, k: &str) -> String {
    j.get(k).and_then(Json::as_str).unwrap_or_default().to_owned()
}

pub(crate) fn opt_str(j: &Json, k: &str) -> Option<String> {
    j.get(k).and_then(Json::as_str).map(str::to_owned)
}

pub(crate) fn f64_of(j: &Json, k: &str) -> f64 {
    j.get(k).and_then(Json::as_f64).unwrap_or_default()
}

pub(crate) fn opt_f64(j: &Json, k: &str) -> Option<f64> {
    j.get(k).and_then(Json::as_f64)
}

pub(crate) fn i64_of(j: &Json, k: &str) -> i64 {
    j.get(k).and_then(Json::as_i64).unwrap_or_default()
}

pub(crate) fn u32_of(j: &Json, k: &str) -> u32 {
    u32::try_from(i64_of(j, k)).unwrap_or_default()
}

pub(crate) fn u64_of(j: &Json, k: &str) -> u64 {
    u64::try_from(i64_of(j, k)).unwrap_or_default()
}

pub(crate) fn bool_of(j: &Json, k: &str) -> bool {
    j.get(k).and_then(Json::as_bool).unwrap_or_default()
}

/// The `uid` of a row as the `u64` node id the API hands out.
pub(crate) fn uid_of(j: &Json) -> u64 {
    j.get("uid").and_then(Json::as_str).and_then(uid_to_u64).unwrap_or_default()
}

/// Strip the `<type>.` prefix from every predicate, so a generic row looks like the Postgres one
/// (`region_id`, `status`, …) to `NodeRow` / `EdgeRow` consumers and to the frontend.
pub(crate) fn bare_props(j: &Json, prefix: &str, skip: &[&str]) -> Map<String, Json> {
    let mut m = Map::new();
    if let Some(obj) = j.as_object() {
        for (k, v) in obj {
            if k == "uid" || k == "dgraph.type" || v.is_null() {
                continue;
            }
            let bare = k.strip_prefix(prefix).unwrap_or(k);
            if skip.contains(&bare) || bare.contains('.') {
                continue;
            }
            m.insert(bare.to_owned(), v.clone());
        }
    }
    m
}

pub(crate) fn node_row(j: &Json, label: &str, prefix: &str) -> NodeRow {
    NodeRow { id: uid_of(j), label: label.to_owned(), properties: bare_props(j, prefix, &[]) }
}

// ── row conversions, one per `GraphStore` return shape ─────────────────────

pub(crate) fn snapshot_of(j: &Json) -> Snapshot {
    Snapshot {
        id: uid_of(j),
        snapshot_id: str_of(j, "snapshot.snapshot_id"),
        timescale: str_of(j, "snapshot.timescale"),
        encoder_family: str_of(j, "snapshot.encoder_family"),
        generation: u32_of(j, "snapshot.generation"),
        garage_key: str_of(j, "snapshot.garage_key"),
        status: str_of(j, "snapshot.status"),
        created_at: opt_str(j, "snapshot.created_at"),
        carried_mass: opt_f64(j, "snapshot.carried_mass"),
        manifest: opt_str(j, "snapshot.manifest"),
        previous_active: opt_str(j, "snapshot.previous_active"),
    }
}

fn space_of(j: &Json) -> ExpertSpace {
    ExpertSpace {
        id: uid_of(j),
        expert_space_id: str_of(j, "expert_space.expert_space_id"),
        snapshot_id: str_of(j, "expert_space.snapshot_id"),
        index: u32_of(j, "expert_space.index"),
        dim: u32_of(j, "expert_space.dim"),
    }
}

fn adapter_of(j: &Json) -> AdapterState {
    AdapterState {
        id: uid_of(j),
        adapter_id: str_of(j, "adapter_state.adapter_id"),
        snapshot_id: str_of(j, "adapter_state.snapshot_id"),
        timescale: str_of(j, "adapter_state.timescale"),
        version: u64_of(j, "adapter_state.version"),
        garage_key: str_of(j, "adapter_state.garage_key"),
        drift: opt_f64(j, "adapter_state.drift"),
        max_drift: f64_of(j, "adapter_state.max_drift"),
        frozen: bool_of(j, "adapter_state.frozen"),
        n_windows: u64_of(j, "adapter_state.n_windows"),
        window_start: opt_str(j, "adapter_state.window_start"),
        window_end: opt_str(j, "adapter_state.window_end"),
        updated_at: opt_str(j, "adapter_state.updated_at"),
    }
}

fn region_of(j: &Json) -> LatentRegion {
    LatentRegion {
        id: uid_of(j),
        region_id: str_of(j, "region.region_id"),
        lineage_id: str_of(j, "region.lineage_id"),
        expert_space_id: str_of(j, "region.expert_space_id"),
        method: str_of(j, "region.method"),
        radius: f64_of(j, "region.radius"),
        density: f64_of(j, "region.density"),
        status: opt_str(j, "region.status"),
        related_region_id: opt_str(j, "region.related_region_id"),
        created_at: opt_str(j, "region.created_at"),
        updated_at: opt_str(j, "region.updated_at"),
    }
}

fn transition_of(j: &Json) -> Transition {
    // `transition.from`/`.to` come back as a one-element array of `{uid}` under `expand(_all_)`.
    let edge_uid = |k: &str| {
        j.get(k)
            .and_then(|v| v.as_array().and_then(|a| a.first()).or(Some(v)))
            .map(uid_of)
            .unwrap_or_default()
    };
    Transition {
        id: uid_of(j),
        from: edge_uid("transition.from"),
        to: edge_uid("transition.to"),
        from_region_id: opt_str(j, "transition.from_region_id"),
        to_region_id: opt_str(j, "transition.to_region_id"),
        sampler_id: str_of(j, "transition.sampler_id"),
        count: i64_of(j, "transition.count"),
        mean: f64_of(j, "transition.mean"),
        lower: f64_of(j, "transition.lower"),
        upper: f64_of(j, "transition.upper"),
        kernel_params: str_of(j, "transition.kernel_params"),
        updated_at: opt_str(j, "transition.updated_at"),
    }
}

fn transition_version_of(j: &Json) -> TransitionVersion {
    TransitionVersion {
        id: uid_of(j),
        edge_key: str_of(j, "transition_version.edge_key"),
        from_region_id: str_of(j, "transition_version.from_region_id"),
        to_region_id: str_of(j, "transition_version.to_region_id"),
        sampler_id: str_of(j, "transition_version.sampler_id"),
        count: i64_of(j, "transition_version.count"),
        mean: f64_of(j, "transition_version.mean"),
        lower: f64_of(j, "transition_version.lower"),
        upper: f64_of(j, "transition_version.upper"),
        kernel_params: str_of(j, "transition_version.kernel_params"),
        ts: opt_str(j, "transition_version.ts"),
    }
}

fn instrument_of(j: &Json) -> Instrument {
    Instrument {
        id: uid_of(j),
        instrument_id: str_of(j, "ref_instrument.instrument_id"),
        asset_class: str_of(j, "ref_instrument.asset_class"),
        kind: str_of(j, "ref_instrument.kind"),
        name: str_of(j, "ref_instrument.name"),
    }
}

impl DgraphGraph {
    pub fn new(dg: DgraphClient) -> Self {
        Self { dg }
    }

    /// `FA_DGRAPH_URL`, else the compose default.
    pub fn from_env() -> Self {
        Self::new(DgraphClient::from_env())
    }

    pub fn client(&self) -> &DgraphClient {
        &self.dg
    }

    /// Apply the schema. Idempotent; called once per process before the first write.
    pub async fn ensure_schema(&self) -> Result<()> {
        self.dg.alter(dsch::SCHEMA).await
    }

    /// Rows of the `q` block of a query.
    pub(crate) async fn rows(&self, dql: &str) -> Result<Vec<Json>> {
        let d = self.dg.query(dql).await?;
        Ok(d.get("q").and_then(Json::as_array).cloned().unwrap_or_default())
    }

    /// The single row of the `q` block, if any.
    pub(crate) async fn one(&self, dql: &str) -> Result<Option<Json>> {
        Ok(self.rows(dql).await?.into_iter().next())
    }

    /// Every predicate of one node addressed by business key.
    pub(crate) async fn by_key(&self, ty: &str, key_pred: &str, key: &str) -> Result<Option<Json>> {
        let dql = format!(
            "{{ q(func: eq({key_pred}, \"{k}\")) @filter(type({ty})) {{ uid dgraph.type expand(_all_) }} }}",
            k = esc(key)
        );
        self.one(&dql).await
    }

    /// Every predicate of one node addressed by uid, checked against its type.
    pub(crate) async fn by_uid(&self, ty: &str, id: u64) -> Result<Option<Json>> {
        let dql = format!(
            "{{ q(func: uid({uid})) @filter(type({ty})) {{ uid dgraph.type expand(_all_) }} }}",
            uid = u64_to_uid(id)
        );
        self.one(&dql).await
    }

    /// Insert or update a node by business key, then return its uid.
    ///
    /// The uid is re-read instead of taken from the mutation response because Dgraph only reports
    /// uids it allocated: an update returns none, and the caller needs the id either way.
    pub(crate) async fn upsert_keyed(&self, ty: &str, key_pred: &str, key: &str, fields: Map<String, Json>) -> Result<u64> {
        let q = format!("{{ v as var(func: eq({key_pred}, \"{k}\")) }}", k = esc(key));
        let mut set = fields;
        set.insert("uid".into(), json!("uid(v)"));
        set.insert("dgraph.type".into(), json!(ty));
        set.insert(key_pred.to_owned(), json!(key));
        self.dg.upsert(&q, vec![Json::Object(set)], vec![]).await?;
        let row = self
            .by_key(ty, key_pred, key)
            .await?
            .ok_or_else(|| HubError::Shape { name: ty.into(), detail: format!("upsert of `{key}` wrote no node") })?;
        Ok(uid_of(&row))
    }

    /// Create a node that must not already exist, mirroring `INSERT` without `ON CONFLICT`:
    /// a duplicate business key is the 409 the Postgres backend raised from SQLSTATE 23505.
    pub(crate) async fn insert_keyed(&self, ty: &str, key_pred: &str, key: &str, fields: Map<String, Json>) -> Result<u64> {
        if self.by_key(ty, key_pred, key).await?.is_some() {
            return Err(unique_violation(&format!("{ty} `{key}`")));
        }
        self.upsert_keyed(ty, key_pred, key, fields).await
    }

    /// Set predicates on a node already addressed by uid.
    pub(crate) async fn set_on(&self, id: u64, fields: Map<String, Json>) -> Result<()> {
        let mut set = fields;
        set.insert("uid".into(), json!(u64_to_uid(id)));
        self.dg.mutate(vec![Json::Object(set)]).await.map(drop)
    }

    /// Delete whole nodes by uid.
    pub(crate) async fn delete_uids(&self, uids: &[u64]) -> Result<u64> {
        if uids.is_empty() {
            return Ok(0);
        }
        let del: Vec<Json> = uids.iter().map(|u| json!({"uid": u64_to_uid(*u)})).collect();
        let n = del.len() as u64;
        self.dg.delete(del).await?;
        Ok(n)
    }

    /// Append an immutable event-time row (the `*_version` tables are append-only).
    pub(crate) async fn append(&self, ty: &str, fields: Map<String, Json>) -> Result<()> {
        let mut set = fields;
        set.insert("uid".into(), json!("_:v"));
        set.insert("dgraph.type".into(), json!(ty));
        self.dg.mutate(vec![Json::Object(set)]).await.map(drop)
    }
}

/// RFC 3339 with millisecond precision, matching what the Postgres backend returned so callers
/// and fixtures see the same strings.
pub(crate) fn t3339(ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ms)
        .unwrap_or_else(chrono::Utc::now)
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

pub(crate) fn now3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// Parse an RFC 3339 timestamp back to epoch milliseconds, for the as-of comparisons.
pub(crate) fn ms_of(s: Option<&str>) -> Option<i64> {
    s.and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok()).map(|t| t.timestamp_millis())
}

/// `json!({..})` as a field map for the mutation helpers.
pub(crate) fn obj(j: Json) -> Map<String, Json> {
    j.as_object().cloned().unwrap_or_default()
}

/// The projection every full-node read uses.
pub(crate) const ALL: &str = "uid dgraph.type expand(_all_)";

/// Newest-first ordering that matches Postgres's `ORDER BY ts DESC, id DESC`. DQL cannot sort by
/// `uid`, so the tie-break is applied here; without it two versions written in the same
/// millisecond could come back in either order.
pub(crate) fn by_ts_desc(rows: &mut [Json], ts_pred: &str) {
    rows.sort_by(|a, b| {
        let ka = (ms_of(a.get(ts_pred).and_then(Json::as_str)), uid_of(a));
        let kb = (ms_of(b.get(ts_pred).and_then(Json::as_str)), uid_of(b));
        kb.cmp(&ka)
    });
}

impl DgraphGraph {
    /// Append a `RegionStatusVersion`, the counterpart of `version_status` in the Postgres backend.
    async fn version_status(&self, region_id: &str, status: &str, related: &str, reason: &str) -> Result<()> {
        self.append(
            "RegionStatusVersion",
            obj(json!({
                "region_status_version.region_id": region_id,
                "region_status_version.status": status,
                "region_status_version.related_region_id": related,
                "region_status_version.reason": reason,
                "region_status_version.ts": now3339(),
            })),
        )
        .await
    }

    /// Rows of one type whose `pred` equals any of `vals` (the `= ANY($1)` replacement).
    pub(crate) async fn rows_in(&self, ty: &str, pred: &str, vals: &[String]) -> Result<Vec<Json>> {
        if vals.is_empty() {
            return Ok(Vec::new());
        }
        let list = esc_list(vals.iter().map(String::as_str));
        self.rows(&format!("{{ q(func: eq({pred}, [{list}])) @filter(type({ty})) {{ {ALL} }} }}")).await
    }

    /// The region uid for a business id, or `NotFound` — the shape callers expect from the
    /// Postgres backend's `RETURNING`-less updates.
    async fn region_uid(&self, region_id: &str) -> Result<u64> {
        self.by_key("LatentRegion", "region.region_id", region_id)
            .await?
            .map(|r| uid_of(&r))
            .ok_or_else(|| HubError::NotFound { label: "LatentRegion".into(), id: region_id.into() })
    }

    async fn snapshot_at(&self, id: u64) -> Result<Snapshot> {
        self.by_uid("Snapshot", id)
            .await?
            .map(|r| snapshot_of(&r))
            .ok_or_else(|| HubError::NotFound { label: "Snapshot".into(), id: id.to_string() })
    }

    /// Regions of every space of a snapshot (the `JOIN expert_space` of the SQL reads).
    async fn snapshot_region_rows(&self, snapshot_id: &str) -> Result<Vec<Json>> {
        let spaces = self
            .rows(&format!(
                "{{ q(func: eq(expert_space.snapshot_id, \"{s}\")) @filter(type(ExpertSpace)) {{ uid expert_space.expert_space_id }} }}",
                s = esc(snapshot_id)
            ))
            .await?;
        let ids: Vec<String> = spaces.iter().map(|s| str_of(s, "expert_space.expert_space_id")).collect();
        self.rows_in("LatentRegion", "region.expert_space_id", &ids).await
    }
}

impl crate::graph::GraphStore for DgraphGraph {
    // ── reference mirror ───────────────────────────────────────────────────
    async fn upsert_ref_entities(&self, rows: &[RefEntity]) -> Result<usize> {
        for e in rows {
            self.upsert_keyed(
                "RefEntity",
                "ref_entity.entity_id",
                &e.entity_id,
                obj(json!({
                    "ref_entity.lei": e.lei, "ref_entity.name": e.name, "ref_entity.country": e.country,
                    "ref_entity.sector": e.sector, "ref_entity.mirrored_at": now3339(),
                })),
            )
            .await?;
            if !e.sector.is_empty() {
                let sector_id = crate::reference::sector_id(&e.sector);
                self.upsert_keyed(
                    "RefSector",
                    "ref_sector.sector_id",
                    &sector_id,
                    obj(json!({"ref_sector.name": e.sector})),
                )
                .await?;
                let key = format!("{}|{}", e.entity_id, sector_id);
                self.upsert_keyed(
                    "RefInSector",
                    "ref_in_sector.key",
                    &key,
                    obj(json!({"ref_in_sector.entity_id": e.entity_id, "ref_in_sector.sector_id": sector_id})),
                )
                .await?;
            }
        }
        Ok(rows.len())
    }

    async fn upsert_ref_venues(&self, rows: &[RefVenue]) -> Result<usize> {
        for v in rows {
            self.upsert_keyed(
                "RefVenue",
                "ref_venue.venue_id",
                &v.venue_id,
                obj(json!({
                    "ref_venue.code": v.code, "ref_venue.mic": v.mic, "ref_venue.name": v.name,
                    "ref_venue.kind": v.kind, "ref_venue.timezone": v.timezone, "ref_venue.mirrored_at": now3339(),
                })),
            )
            .await?;
        }
        Ok(rows.len())
    }

    async fn upsert_ref_instruments(&self, rows: &[RefInstrument]) -> Result<RefSyncReport> {
        let mut rep = RefSyncReport { batches: 1, ..Default::default() };
        for i in rows {
            self.upsert_keyed(
                "RefInstrument",
                "ref_instrument.instrument_id",
                &i.instrument_id,
                obj(json!({
                    "ref_instrument.asset_class": i.asset_class, "ref_instrument.kind": i.kind,
                    "ref_instrument.name": i.name, "ref_instrument.status": i.status,
                    "ref_instrument.base_ccy": i.base_ccy, "ref_instrument.quote_ccy": i.quote_ccy,
                    "ref_instrument.mirrored_at": now3339(),
                })),
            )
            .await?;
            rep.instruments += 1;
            if let Some(issuer) = i.issuer_id.as_deref().filter(|s| !s.is_empty()) {
                let key = format!("{}|{issuer}", i.instrument_id);
                self.upsert_keyed(
                    "RefIssuedBy",
                    "ref_issued_by.key",
                    &key,
                    obj(json!({
                        "ref_issued_by.instrument_id": i.instrument_id, "ref_issued_by.entity_id": issuer,
                    })),
                )
                .await?;
                rep.issued_by += 1;
            }
            for l in &i.listings {
                let listing_key = format!("{}|{}", l.venue_id, l.venue_symbol);
                let key = format!("{}|{listing_key}", i.instrument_id);
                self.upsert_keyed(
                    "RefListedOn",
                    "ref_listed_on.key",
                    &key,
                    obj(json!({
                        "ref_listed_on.instrument_id": i.instrument_id, "ref_listed_on.venue_id": l.venue_id,
                        "ref_listed_on.venue_symbol": l.venue_symbol, "ref_listed_on.source_id": l.source_id,
                        "ref_listed_on.listing_key": listing_key,
                    })),
                )
                .await?;
                rep.listed_on += 1;
            }
            for x in &i.exposures {
                let exposure_key = format!("{}|{}", x.to_instrument_id, x.relation);
                let key = format!("{}|{exposure_key}", i.instrument_id);
                self.upsert_keyed(
                    "RefSameExposure",
                    "ref_same_exposure.key",
                    &key,
                    obj(json!({
                        "ref_same_exposure.instrument_id": i.instrument_id,
                        "ref_same_exposure.to_instrument_id": x.to_instrument_id,
                        "ref_same_exposure.relation": x.relation,
                        "ref_same_exposure.exposure_key": exposure_key,
                    })),
                )
                .await?;
                rep.same_exposure += 1;
            }
        }
        Ok(rep)
    }

    async fn retire_ref_instruments(&self, instrument_ids: &[String]) -> Result<usize> {
        if instrument_ids.is_empty() {
            return Ok(0);
        }
        let mut n = 0;
        for id in instrument_ids {
            if let Some(row) = self.by_key("RefInstrument", "ref_instrument.instrument_id", id).await? {
                self.set_on(uid_of(&row), obj(json!({"ref_instrument.status": crate::reference::INACTIVE})))
                    .await?;
                n += 1;
            }
        }
        // a retired instrument loses its out-edges, as the SQL `DELETE`s did
        for (ty, pred) in [
            ("RefListedOn", "ref_listed_on.instrument_id"),
            ("RefIssuedBy", "ref_issued_by.instrument_id"),
            ("RefSameExposure", "ref_same_exposure.instrument_id"),
        ] {
            let rows = self.rows_in(ty, pred, instrument_ids).await?;
            let uids: Vec<u64> = rows.iter().map(uid_of).collect();
            self.delete_uids(&uids).await?;
        }
        Ok(n)
    }

    async fn ref_instrument_ids(&self) -> Result<Vec<(String, String)>> {
        let rows = self
            .rows("{ q(func: type(RefInstrument)) { ref_instrument.instrument_id ref_instrument.status } }")
            .await?;
        Ok(rows
            .iter()
            .map(|r| (str_of(r, "ref_instrument.instrument_id"), str_of(r, "ref_instrument.status")))
            .collect())
    }

    async fn instrument_reference(&self, instrument_id: &str) -> Result<Option<InstrumentReference>> {
        let Some(inst) = self.by_key("RefInstrument", "ref_instrument.instrument_id", instrument_id).await? else {
            return Ok(None);
        };
        let one = vec![instrument_id.to_owned()];
        let issued = self.rows_in("RefIssuedBy", "ref_issued_by.instrument_id", &one).await?;
        let issuer_ids: Vec<String> = issued.iter().map(|r| str_of(r, "ref_issued_by.entity_id")).collect();
        let issuer = self.rows_in("RefEntity", "ref_entity.entity_id", &issuer_ids).await?;
        let listed = self.rows_in("RefListedOn", "ref_listed_on.instrument_id", &one).await?;
        let venue_ids: Vec<String> = listed.iter().map(|r| str_of(r, "ref_listed_on.venue_id")).collect();
        let venues = self.rows_in("RefVenue", "ref_venue.venue_id", &venue_ids).await?;
        let exposures = self.rows_in("RefSameExposure", "ref_same_exposure.instrument_id", &one).await?;
        // sectors are resolved through the issuer, as the SQL join did
        let in_sector = self.rows_in("RefInSector", "ref_in_sector.entity_id", &issuer_ids).await?;
        let sector_ids: Vec<String> = in_sector.iter().map(|r| str_of(r, "ref_in_sector.sector_id")).collect();
        let sectors = self.rows_in("RefSector", "ref_sector.sector_id", &sector_ids).await?;
        let edge = |r: &Json, label: &str, prefix: &str| EdgeRow {
            id: uid_of(r),
            label: label.to_owned(),
            from: 0,
            to: 0,
            properties: bare_props(r, prefix, &["key"]),
        };
        Ok(Some(InstrumentReference {
            instrument: Json::Object(bare_props(&inst, "ref_instrument.", &[])),
            issuer: issuer.iter().map(|r| Json::Object(bare_props(r, "ref_entity.", &[]))).collect(),
            listings: listed.iter().map(|r| edge(r, "ListedOn", "ref_listed_on.")).collect(),
            venues: venues.iter().map(|r| Json::Object(bare_props(r, "ref_venue.", &[]))).collect(),
            exposures: exposures.iter().map(|r| edge(r, "SameExposure", "ref_same_exposure.")).collect(),
            sectors: sectors.iter().map(|r| Json::Object(bare_props(r, "ref_sector.", &[]))).collect(),
        }))
    }

    async fn upsert_instrument(&self, inst: &NewInstrument) -> Result<Instrument> {
        let id = self
            .upsert_keyed(
                "RefInstrument",
                "ref_instrument.instrument_id",
                &inst.instrument_id,
                obj(json!({
                    "ref_instrument.asset_class": inst.asset_class, "ref_instrument.kind": inst.kind,
                    "ref_instrument.name": inst.name,
                })),
            )
            .await?;
        let _ = id;
        self.get_instrument(&inst.instrument_id)
            .await?
            .ok_or_else(|| HubError::NotFound { label: "RefInstrument".into(), id: inst.instrument_id.clone() })
    }

    async fn get_instrument(&self, instrument_id: &str) -> Result<Option<Instrument>> {
        Ok(self
            .by_key("RefInstrument", "ref_instrument.instrument_id", instrument_id)
            .await?
            .map(|r| instrument_of(&r)))
    }

    // ── snapshots, spaces, adapters ────────────────────────────────────────
    async fn add_snapshot(&self, snap: &NewSnapshot) -> Result<Snapshot> {
        let id = self
            .insert_keyed("Snapshot", "snapshot.snapshot_id", &snap.snapshot_id, snapshot_fields(snap))
            .await?;
        self.snapshot_at(id).await
    }

    async fn ensure_snapshot(&self, snap: &NewSnapshot) -> Result<Snapshot> {
        // `ON CONFLICT DO NOTHING`: an existing row is kept exactly as it is.
        if let Some(row) = self.by_key("Snapshot", "snapshot.snapshot_id", &snap.snapshot_id).await? {
            return Ok(snapshot_of(&row));
        }
        let id = self
            .upsert_keyed("Snapshot", "snapshot.snapshot_id", &snap.snapshot_id, snapshot_fields(snap))
            .await?;
        self.snapshot_at(id).await
    }

    async fn list_snapshots(&self) -> Result<Vec<Snapshot>> {
        let rows = self.rows(&format!("{{ q(func: type(Snapshot)) {{ {ALL} }} }}")).await?;
        Ok(rows.iter().map(snapshot_of).collect())
    }

    async fn snapshot_by_id(&self, snapshot_id: &str) -> Result<Option<Snapshot>> {
        Ok(self.by_key("Snapshot", "snapshot.snapshot_id", snapshot_id).await?.map(|r| snapshot_of(&r)))
    }

    async fn snapshots_by_status(&self, timescale: &str, family: &str, status: &str) -> Result<Vec<Snapshot>> {
        let dql = format!(
            "{{ q(func: eq(snapshot.timescale, \"{t}\")) @filter(eq(snapshot.encoder_family, \"{f}\") AND eq(snapshot.status, \"{s}\")) {{ {ALL} }} }}",
            t = esc(timescale),
            f = esc(family),
            s = esc(status)
        );
        Ok(self.rows(&dql).await?.iter().map(snapshot_of).collect())
    }

    async fn link_snapshot_parent(&self, parent: u64, child: u64, op: &str) -> Result<u64> {
        // keyed on the snapshot business ids, not the uids: stable across backends, and it keeps
        // the row reachable by the id prefix the conformance cleanup matches on
        let (p, c) = (self.snapshot_at(parent).await?, self.snapshot_at(child).await?);
        let key = format!("{}|{}", p.snapshot_id, c.snapshot_id);
        self.upsert_keyed(
            "SnapshotParent",
            "snapshot_parent.key",
            &key,
            obj(json!({
                "snapshot_parent.parent": {"uid": u64_to_uid(parent)},
                "snapshot_parent.child": {"uid": u64_to_uid(child)},
                "snapshot_parent.op": op,
            })),
        )
        .await
    }

    async fn snapshot_parent(&self, child: u64) -> Result<Option<Snapshot>> {
        let dql = format!(
            "{{ q(func: type(SnapshotParent)) @filter(uid_in(snapshot_parent.child, {c})) {{ snapshot_parent.parent {{ {ALL} }} }} }}",
            c = u64_to_uid(child)
        );
        let rows = self.rows(&dql).await?;
        Ok(rows
            .first()
            .and_then(|r| r.get("snapshot_parent.parent").cloned())
            .and_then(|v| v.as_array().and_then(|a| a.first().cloned()).or(Some(v)))
            .map(|v| snapshot_of(&v)))
    }

    async fn snapshot_children(&self, parent: u64) -> Result<Vec<Snapshot>> {
        let dql = format!(
            "{{ q(func: type(SnapshotParent)) @filter(uid_in(snapshot_parent.parent, {p})) {{ snapshot_parent.child {{ {ALL} }} }} }}",
            p = u64_to_uid(parent)
        );
        let rows = self.rows(&dql).await?;
        Ok(rows
            .iter()
            .filter_map(|r| r.get("snapshot_parent.child"))
            .flat_map(|v| v.as_array().cloned().unwrap_or_else(|| vec![v.clone()]))
            .map(|v| snapshot_of(&v))
            .collect())
    }

    async fn set_snapshot_status(&self, snapshot: u64, status: &str) -> Result<Snapshot> {
        self.set_on(snapshot, obj(json!({"snapshot.status": status}))).await?;
        self.snapshot_at(snapshot).await
    }

    async fn set_snapshot_previous_active(&self, snapshot: u64, previous: &str) -> Result<Snapshot> {
        self.set_on(snapshot, obj(json!({"snapshot.previous_active": previous}))).await?;
        self.snapshot_at(snapshot).await
    }

    async fn set_snapshot_manifest(&self, snapshot: u64, manifest_json: &str, carried_mass: f64) -> Result<Snapshot> {
        let mass = if carried_mass.is_finite() { carried_mass } else { 0.0 };
        self.set_on(snapshot, obj(json!({"snapshot.manifest": manifest_json, "snapshot.carried_mass": mass})))
            .await?;
        self.snapshot_at(snapshot).await
    }

    async fn add_adapter_state(&self, snapshot: u64, a: &crate::queries::NewAdapterState) -> Result<AdapterState> {
        let _ = self.snapshot_at(snapshot).await?;
        let adapter_id = crate::queries::adapter_id(&a.snapshot_id, a.version);
        let id = self
            .insert_keyed(
                "AdapterState",
                "adapter_state.adapter_id",
                &adapter_id,
                obj(json!({
                    "adapter_state.snapshot_id": a.snapshot_id, "adapter_state.timescale": a.timescale,
                    "adapter_state.version": a.version, "adapter_state.garage_key": a.garage_key,
                    "adapter_state.drift": a.drift, "adapter_state.max_drift": a.max_drift,
                    "adapter_state.frozen": a.frozen, "adapter_state.n_windows": a.n_windows,
                    "adapter_state.window_start": t3339(a.window_start_ms),
                    "adapter_state.window_end": t3339(a.window_end_ms),
                    "adapter_state.updated_at": now3339(),
                })),
            )
            .await?;
        self.by_uid("AdapterState", id)
            .await?
            .map(|r| adapter_of(&r))
            .ok_or_else(|| HubError::NotFound { label: "AdapterState".into(), id: adapter_id })
    }

    async fn snapshot_adapters(&self, snapshot_id: &str, limit: usize) -> Result<Vec<AdapterState>> {
        let dql = format!(
            "{{ q(func: eq(adapter_state.snapshot_id, \"{s}\"), orderdesc: adapter_state.version, first: {n}) {{ {ALL} }} }}",
            s = esc(snapshot_id),
            n = limit.max(1)
        );
        Ok(self.rows(&dql).await?.iter().map(adapter_of).collect())
    }

    async fn add_expert_space(&self, snapshot: u64, space: &NewExpertSpace) -> Result<ExpertSpace> {
        let _ = self.snapshot_at(snapshot).await?;
        let id = self
            .insert_keyed("ExpertSpace", "expert_space.expert_space_id", &space.expert_space_id, space_fields(space))
            .await?;
        self.by_uid("ExpertSpace", id)
            .await?
            .map(|r| space_of(&r))
            .ok_or_else(|| HubError::NotFound { label: "ExpertSpace".into(), id: space.expert_space_id.clone() })
    }

    async fn ensure_expert_space(&self, snapshot: u64, space: &NewExpertSpace) -> Result<ExpertSpace> {
        if let Some(row) = self.by_key("ExpertSpace", "expert_space.expert_space_id", &space.expert_space_id).await? {
            return Ok(space_of(&row));
        }
        self.add_expert_space(snapshot, space).await
    }

    async fn space_by_id(&self, expert_space_id: &str) -> Result<Option<ExpertSpace>> {
        Ok(self.by_key("ExpertSpace", "expert_space.expert_space_id", expert_space_id).await?.map(|r| space_of(&r)))
    }

    async fn snapshot_spaces(&self, snapshot: u64) -> Result<Vec<ExpertSpace>> {
        let snap = self.snapshot_at(snapshot).await?;
        let dql = format!(
            "{{ q(func: eq(expert_space.snapshot_id, \"{s}\"), orderasc: expert_space.index) {{ {ALL} }} }}",
            s = esc(&snap.snapshot_id)
        );
        Ok(self.rows(&dql).await?.iter().map(space_of).collect())
    }

    // ── regions and transitions ────────────────────────────────────────────
    async fn add_region(&self, space: u64, region: &NewRegion) -> Result<LatentRegion> {
        self.by_uid("ExpertSpace", space)
            .await?
            .ok_or_else(|| HubError::NotFound { label: "ExpertSpace".into(), id: space.to_string() })?;
        // deliberately no status and no status version — see the conformance suite
        let id = self
            .insert_keyed(
                "LatentRegion",
                "region.region_id",
                &region.region_id,
                obj(json!({
                    "region.lineage_id": region.lineage_id, "region.expert_space_id": region.expert_space_id,
                    "region.method": region.method, "region.radius": region.radius, "region.density": region.density,
                    "region.created_at": now3339(),
                })),
            )
            .await?;
        self.by_uid("LatentRegion", id)
            .await?
            .map(|r| region_of(&r))
            .ok_or_else(|| HubError::NotFound { label: "LatentRegion".into(), id: region.region_id.clone() })
    }

    async fn upsert_region(&self, _space: u64, region: &NewRegion) -> Result<LatentRegion> {
        let existing = self.by_key("LatentRegion", "region.region_id", &region.region_id).await?;
        match &existing {
            Some(row) => {
                if opt_str(row, "region.status").as_deref() != Some("live") {
                    self.version_status(&region.region_id, "live", "", "revived").await?;
                }
            }
            None => self.version_status(&region.region_id, "live", "", "created").await?,
        }
        let mut fields = obj(json!({
            "region.status": "live", "region.related_region_id": "",
            "region.radius": region.radius, "region.density": region.density, "region.updated_at": now3339(),
        }));
        if existing.is_none() {
            fields.insert("region.lineage_id".into(), json!(region.lineage_id));
            fields.insert("region.expert_space_id".into(), json!(region.expert_space_id));
            fields.insert("region.method".into(), json!(region.method));
            fields.insert("region.created_at".into(), json!(now3339()));
        }
        let id = self.upsert_keyed("LatentRegion", "region.region_id", &region.region_id, fields).await?;
        self.by_uid("LatentRegion", id)
            .await?
            .map(|r| region_of(&r))
            .ok_or_else(|| HubError::NotFound { label: "LatentRegion".into(), id: region.region_id.clone() })
    }

    async fn region_by_id(&self, region_id: &str) -> Result<Option<LatentRegion>> {
        Ok(self.by_key("LatentRegion", "region.region_id", region_id).await?.map(|r| region_of(&r)))
    }

    async fn mark_region_status(&self, region_id: &str, status: &str, related: &str) -> Result<LatentRegion> {
        let uid = self.region_uid(region_id).await?;
        self.set_on(
            uid,
            obj(json!({"region.status": status, "region.related_region_id": related, "region.updated_at": now3339()})),
        )
        .await?;
        self.version_status(region_id, status, related, "").await?;
        self.by_uid("LatentRegion", uid)
            .await?
            .map(|r| region_of(&r))
            .ok_or_else(|| HubError::NotFound { label: "LatentRegion".into(), id: region_id.into() })
    }

    async fn space_regions(&self, space: u64) -> Result<Vec<LatentRegion>> {
        let Some(sp) = self.by_uid("ExpertSpace", space).await? else { return Ok(Vec::new()) };
        let esid = str_of(&sp, "expert_space.expert_space_id");
        let rows = self.rows_in("LatentRegion", "region.expert_space_id", &[esid]).await?;
        Ok(rows.iter().map(region_of).collect())
    }

    async fn update_region_stats(&self, rows: &[RegionStatsUpdate]) -> Result<()> {
        for r in rows {
            let uid = self.region_uid(&r.region_id).await?;
            let num = |v: f64| if v.is_finite() { v } else { f64::MAX };
            self.set_on(
                uid,
                obj(json!({
                    "region.mass": num(r.mass), "region.occupancy": num(r.occupancy),
                    "region.spread": num(r.spread), "region.age": r.age, "region.updated_at": now3339(),
                })),
            )
            .await?;
        }
        Ok(())
    }

    async fn mark_region_died(&self, region_id: &str, reason: &str, related: &str) -> Result<()> {
        let uid = self.region_uid(region_id).await?;
        self.set_on(
            uid,
            obj(json!({
                "region.status": "retired", "region.retire_reason": reason, "region.related_region_id": related,
                "region.died_at": now3339(), "region.updated_at": now3339(),
            })),
        )
        .await?;
        self.version_status(region_id, "retired", related, reason).await
    }

    async fn add_transition(&self, from: u64, to: u64, t: &crate::queries::NewTransition) -> Result<Transition> {
        let (f, tt) = (
            self.by_uid("LatentRegion", from)
                .await?
                .ok_or_else(|| HubError::NotFound { label: "LatentRegion".into(), id: from.to_string() })?,
            self.by_uid("LatentRegion", to)
                .await?
                .ok_or_else(|| HubError::NotFound { label: "LatentRegion".into(), id: to.to_string() })?,
        );
        let (fid, tid) = (str_of(&f, "region.region_id"), str_of(&tt, "region.region_id"));
        let key = dsch::edge_key(&fid, &tid, &t.sampler_id);
        let id = self
            .insert_keyed("Transition", "transition.key", &key, transition_fields(&fid, &tid, from, to, t))
            .await?;
        self.by_uid("Transition", id)
            .await?
            .map(|r| transition_of(&r))
            .ok_or_else(|| HubError::NotFound { label: "Transition".into(), id: key })
    }

    async fn upsert_transitions(&self, rows: &[crate::queries::TransitionUpsert]) -> Result<Vec<Transition>> {
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let from = self.region_uid(&r.from_region_id).await?;
            let to = self.region_uid(&r.to_region_id).await?;
            let key = dsch::edge_key(&r.from_region_id, &r.to_region_id, &r.transition.sampler_id);
            let id = self
                .upsert_keyed(
                    "Transition",
                    "transition.key",
                    &key,
                    transition_fields(&r.from_region_id, &r.to_region_id, from, to, &r.transition),
                )
                .await?;
            // every write appends an event-time version, as the Postgres trigger-free path did
            self.append(
                "TransitionVersion",
                obj(json!({
                    "transition_version.edge_key": key,
                    "transition_version.from_region_id": r.from_region_id,
                    "transition_version.to_region_id": r.to_region_id,
                    "transition_version.sampler_id": r.transition.sampler_id,
                    "transition_version.count": r.transition.count,
                    "transition_version.mean": r.transition.mean,
                    "transition_version.lower": r.transition.lower,
                    "transition_version.upper": r.transition.upper,
                    "transition_version.kernel_params": r.transition.kernel_params,
                    "transition_version.ts": now3339(),
                })),
            )
            .await?;
            if let Some(row) = self.by_uid("Transition", id).await? {
                out.push(transition_of(&row));
            }
        }
        Ok(out)
    }

    async fn region_transitions(&self, region: u64) -> Result<Vec<Transition>> {
        let dql = format!(
            "{{ q(func: type(Transition)) @filter(uid_in(transition.from, {r})) {{ {ALL} }} }}",
            r = u64_to_uid(region)
        );
        Ok(self.rows(&dql).await?.iter().map(transition_of).collect())
    }

    async fn region_neighbors(&self, region: u64) -> Result<Vec<LatentRegion>> {
        let dql = format!(
            "{{ q(func: type(Transition)) @filter(uid_in(transition.from, {r})) {{ transition.to {{ {ALL} }} }} }}",
            r = u64_to_uid(region)
        );
        let rows = self.rows(&dql).await?;
        let mut seen = Vec::new();
        let mut out = Vec::new();
        for r in rows.iter().filter_map(|r| r.get("transition.to")) {
            for v in r.as_array().cloned().unwrap_or_else(|| vec![r.clone()]) {
                let id = uid_of(&v);
                if id != 0 && !seen.contains(&id) {
                    seen.push(id);
                    out.push(region_of(&v));
                }
            }
        }
        Ok(out)
    }

    async fn match_regions(&self, prev: u64, next: u64, score: f64) -> Result<u64> {
        let prev_id = self.by_uid("LatentRegion", prev).await?.map(|r| str_of(&r, "region.region_id"));
        let next_id = self.by_uid("LatentRegion", next).await?.map(|r| str_of(&r, "region.region_id"));
        // keyed on its own predicate: using `prev_region_id` as the key would make `upsert_keyed`
        // overwrite it with the uid pair, and `region_lineage`'s carry edges read it by business id
        let key = format!("{prev}|{next}");
        self.upsert_keyed(
            "RegionMatch",
            "region_match.key",
            &key,
            obj(json!({
                "region_match.prev": {"uid": u64_to_uid(prev)}, "region_match.next": {"uid": u64_to_uid(next)},
                "region_match.prev_region_id": prev_id.unwrap_or_default(),
                "region_match.next_region_id": next_id.unwrap_or_default(),
                "region_match.score": score,
            })),
        )
        .await
    }

    async fn region_graph(&self, space: u64) -> Result<RegionGraph> {
        let regions = <Self as crate::graph::GraphStore>::space_regions(self, space).await?;
        let ids: Vec<String> = regions.iter().map(|r| r.region_id.clone()).collect();
        let rows = self.rows_in("Transition", "transition.from_region_id", &ids).await?;
        Ok(RegionGraph { regions, transitions: rows.iter().map(transition_of).collect() })
    }

    async fn snapshot_graph(&self, snapshot_id: &str) -> Result<Option<SnapshotGraph>> {
        let Some(snapshot) = <Self as crate::graph::GraphStore>::snapshot_by_id(self, snapshot_id).await? else {
            return Ok(None);
        };
        let spaces = self
            .rows(&format!(
                "{{ q(func: eq(expert_space.snapshot_id, \"{s}\"), orderasc: expert_space.index) {{ {ALL} }} }}",
                s = esc(snapshot_id)
            ))
            .await?;
        let region_rows = self.snapshot_region_rows(snapshot_id).await?;
        let ids: Vec<String> = region_rows.iter().map(|r| str_of(r, "region.region_id")).collect();
        let trans = self.rows_in("Transition", "transition.from_region_id", &ids).await?;
        Ok(Some(SnapshotGraph {
            snapshot,
            spaces: spaces.iter().map(space_of).collect(),
            regions: region_rows.iter().map(region_of).collect(),
            transitions: trans.iter().map(transition_of).collect(),
        }))
    }

    // ── lineage and hierarchy ──────────────────────────────────────────────
    async fn upsert_descents(&self, rows: &[NewDescent]) -> Result<()> {
        for d in rows {
            let key = format!("{}|{}|{}", d.child_region_id, d.parent_region_id, d.kind);
            self.upsert_keyed(
                "DescendsFrom",
                "descends_from.key",
                &key,
                obj(json!({
                    "descends_from.child_region_id": d.child_region_id,
                    "descends_from.parent_region_id": d.parent_region_id,
                    "descends_from.kind": d.kind, "descends_from.overlap": d.overlap,
                    "descends_from.updated_at": now3339(),
                })),
            )
            .await?;
        }
        Ok(())
    }

    async fn upsert_hier_edges(&self, rows: &[NewHierEdge]) -> Result<()> {
        for h in rows {
            // the SQL guarded on both regions existing and silently skipped otherwise
            let both = self.by_key("LatentRegion", "region.region_id", &h.fine_region_id).await?.is_some()
                && self.by_key("LatentRegion", "region.region_id", &h.coarse_region_id).await?.is_some();
            if !both {
                continue;
            }
            let key = format!("{}|{}", h.fine_region_id, h.coarse_region_id);
            self.upsert_keyed(
                "Hier",
                "hier.key",
                &key,
                obj(json!({
                    "hier.fine_region_id": h.fine_region_id, "hier.coarse_region_id": h.coarse_region_id,
                    "hier.weight": h.p_coarse_given_fine, "hier.p_coarse_given_fine": h.p_coarse_given_fine,
                    "hier.p_fine_given_coarse": h.p_fine_given_coarse, "hier.support": h.support,
                    "hier.pmi_bits": h.pmi_bits, "hier.mi_bits": h.mi_bits, "hier.updated_at": now3339(),
                })),
            )
            .await?;
        }
        Ok(())
    }

    async fn snapshot_hierarchy(&self, snapshot_id: &str) -> Result<Vec<EdgeRow>> {
        let region_rows = self.snapshot_region_rows(snapshot_id).await?;
        let ids: Vec<String> = region_rows.iter().map(|r| str_of(r, "region.region_id")).collect();
        let rows = self.rows_in("Hier", "hier.fine_region_id", &ids).await?;
        Ok(rows
            .iter()
            .map(|r| EdgeRow {
                id: uid_of(r),
                label: "Hier".into(),
                from: 0,
                to: 0,
                properties: bare_props(r, "hier.", &["key"]),
            })
            .collect())
    }

    async fn lineage_members(&self, lineage_id: &str) -> Result<Vec<NodeRow>> {
        let rows = self.rows_in("LatentRegion", "region.lineage_id", &[lineage_id.to_owned()]).await?;
        Ok(rows.iter().map(|r| node_row(r, "LatentRegion", "region.")).collect())
    }

    async fn region_lineage(&self, region_id: &str, depth: usize) -> Result<Option<RegionLineage>> {
        let Some(start) = self.by_key("LatentRegion", "region.region_id", region_id).await? else {
            return Ok(None);
        };
        let region = node_row(&start, "LatentRegion", "region.");
        let lineage_id = str_of(&start, "region.lineage_id");
        let lineage = if lineage_id.is_empty() {
            Vec::new()
        } else {
            <Self as crate::graph::GraphStore>::lineage_members(self, &lineage_id).await?
        };
        let mut out: [Vec<LineageLink>; 2] = [Vec::new(), Vec::new()];
        for (slot, up) in [(0usize, true), (1usize, false)] {
            let mut seen = vec![region_id.to_owned()];
            let mut frontier = vec![region_id.to_owned()];
            for level in 1..=depth.max(1) {
                if frontier.is_empty() {
                    break;
                }
                // ancestors: DescendsFrom child→parent, plus RegionMatch next→prev ("carry")
                let (d_pred, d_next, m_pred, m_next) = if up {
                    ("descends_from.child_region_id", "parent_region_id", "region_match.next_region_id", "prev")
                } else {
                    ("descends_from.parent_region_id", "child_region_id", "region_match.prev_region_id", "next")
                };
                let mut edges: Vec<(String, String, String, f64)> = Vec::new();
                for d in self.rows_in("DescendsFrom", d_pred, &frontier).await? {
                    edges.push((
                        str_of(&d, &format!("descends_from.{d_next}")),
                        str_of(&d, if up { "descends_from.child_region_id" } else { "descends_from.parent_region_id" }),
                        str_of(&d, "descends_from.kind"),
                        f64_of(&d, "descends_from.overlap"),
                    ));
                }
                for m in self.rows_in("RegionMatch", m_pred, &frontier).await? {
                    edges.push((
                        str_of(&m, &format!("region_match.{m_next}_region_id")),
                        str_of(&m, m_pred),
                        "carry".into(),
                        f64_of(&m, "region_match.score"),
                    ));
                }
                frontier = Vec::new();
                for (next, via, kind, overlap) in edges {
                    if next.is_empty() || seen.contains(&next) {
                        continue;
                    }
                    let Some(row) = self.by_key("LatentRegion", "region.region_id", &next).await? else { continue };
                    seen.push(next.clone());
                    frontier.push(next);
                    out[slot].push(LineageLink {
                        region: node_row(&row, "LatentRegion", "region."),
                        kind,
                        overlap,
                        depth: level,
                        via,
                    });
                }
            }
        }
        let [ancestors, descendants] = out;
        Ok(Some(RegionLineage { region, lineage, ancestors, descendants }))
    }

    // ── per-space state ────────────────────────────────────────────────────
    async fn upsert_region_set(&self, r: &NewRegionSet) -> Result<()> {
        // `region_set_id` is the expert space id, as in the SQL schema
        self.upsert_keyed(
            "RegionSet",
            "region_set.region_set_id",
            &r.expert_space_id,
            obj(json!({
                "region_set.snapshot_id": r.snapshot_id, "region_set.timescale": r.timescale,
                "region_set.expert_index": r.expert_index, "region_set.family": r.family,
                "region_set.method": r.method, "region_set.n_regions": r.n_regions,
                "region_set.cfg_hash": r.cfg_hash, "region_set.state": r.state,
                "region_set.kernel_state": r.kernel_state, "region_set.fit_time": now3339(),
                "region_set.updated_at": now3339(),
            })),
        )
        .await
        .map(drop)
    }

    async fn region_set(&self, expert_space_id: &str) -> Result<Option<RegionSetRow>> {
        Ok(self.by_key("RegionSet", "region_set.region_set_id", expert_space_id).await?.map(|j| RegionSetRow {
            id: uid_of(&j),
            region_set_id: str_of(&j, "region_set.region_set_id"),
            snapshot_id: str_of(&j, "region_set.snapshot_id"),
            timescale: str_of(&j, "region_set.timescale"),
            expert_index: u32_of(&j, "region_set.expert_index"),
            n_regions: u32_of(&j, "region_set.n_regions"),
            cfg_hash: str_of(&j, "region_set.cfg_hash"),
            state: str_of(&j, "region_set.state"),
            kernel_state: str_of(&j, "region_set.kernel_state"),
            updated_at: opt_str(&j, "region_set.updated_at"),
        }))
    }

    async fn upsert_kernel_model(&self, k: &NewKernelModel) -> Result<()> {
        self.upsert_keyed(
            "KernelModel",
            "kernel_model.kernel_id",
            &k.expert_space_id,
            obj(json!({
                "kernel_model.expert_space_id": k.expert_space_id, "kernel_model.snapshot_id": k.snapshot_id,
                "kernel_model.timescale": k.timescale, "kernel_model.expert_index": k.expert_index,
                "kernel_model.kernel_type": k.kernel_type, "kernel_model.serving_sampler": k.serving_sampler,
                "kernel_model.scores": k.scores, "kernel_model.stationary": k.stationary,
                "kernel_model.relaxation_steps": k.relaxation_steps, "kernel_model.drift_kl": k.drift_kl,
                "kernel_model.n_rows": k.n_rows, "kernel_model.observation_mass": k.observation_mass,
                "kernel_model.live_regions": k.live_regions, "kernel_model.updated_at": now3339(),
            })),
        )
        .await
        .map(drop)
    }

    async fn snapshot_kernels(&self, snapshot_id: &str) -> Result<Vec<NodeRow>> {
        let rows = self.rows_in("KernelModel", "kernel_model.snapshot_id", &[snapshot_id.to_owned()]).await?;
        Ok(rows.iter().map(|r| node_row(r, "KernelModel", "kernel_model.")).collect())
    }

    async fn upsert_cross_priors(&self, rows: &[NewCrossPrior]) -> Result<()> {
        for c in rows {
            let unc = if c.uncertainty.is_finite() { c.uncertainty } else { f64::MAX };
            self.upsert_keyed(
                "CrossPrior",
                "cross_prior.pair_id",
                &c.pair_id(),
                obj(json!({
                    "cross_prior.timescale": c.timescale,
                    "cross_prior.source_instrument_id": c.source_instrument_id,
                    "cross_prior.target_instrument_id": c.target_instrument_id,
                    "cross_prior.n": c.n, "cross_prior.correlation": c.correlation,
                    "cross_prior.shrinkage": c.shrinkage, "cross_prior.lead_lag": c.lead_lag,
                    "cross_prior.lead_corr": c.lead_corr, "cross_prior.predictive_gain": c.predictive_gain,
                    "cross_prior.uncertainty": unc, "cross_prior.score": c.score, "cross_prior.rank": c.rank,
                    "cross_prior.updated_at": now3339(),
                })),
            )
            .await?;
        }
        Ok(())
    }

    async fn cross_priors(&self, timescale: &str, limit: usize) -> Result<Vec<NodeRow>> {
        let dql = format!(
            "{{ q(func: eq(cross_prior.timescale, \"{t}\"), orderdesc: cross_prior.score, first: {n}) {{ {ALL} }} }}",
            t = esc(timescale),
            n = limit.max(1)
        );
        Ok(self.rows(&dql).await?.iter().map(|r| node_row(r, "CrossPrior", "cross_prior.")).collect())
    }

    async fn upsert_region_episodes(&self, rows: &[NewRegionEpisode]) -> Result<()> {
        for e in rows {
            let num = |v: f64| if v.is_finite() { v } else { f64::MAX };
            self.upsert_keyed(
                "RegionEpisode",
                "region_episode.episode_id",
                &e.episode_id(),
                obj(json!({
                    "region_episode.region_id": e.region_id, "region_episode.expert_space_id": e.expert_space_id,
                    "region_episode.timescale": e.timescale, "region_episode.instrument_id": e.instrument_id,
                    "region_episode.t_enter": t3339(e.t_enter_ms), "region_episode.t_exit": t3339(e.t_exit_ms),
                    "region_episode.bars": e.bars, "region_episode.max_weight": num(e.max_weight),
                    "region_episode.mean_weight": num(e.mean_weight),
                })),
            )
            .await?;
        }
        Ok(())
    }

    async fn region_episodes(&self, region_id: &str, limit: usize) -> Result<Vec<NodeRow>> {
        let dql = format!(
            "{{ q(func: eq(region_episode.region_id, \"{r}\"), orderdesc: region_episode.t_exit, first: {n}) {{ {ALL} }} }}",
            r = esc(region_id),
            n = limit.max(1)
        );
        Ok(self.rows(&dql).await?.iter().map(|r| node_row(r, "RegionEpisode", "region_episode.")).collect())
    }

    async fn prune_region_episodes(&self, timescale: &str, cutoff_ms: i64) -> Result<u64> {
        let dql = format!(
            "{{ q(func: eq(region_episode.timescale, \"{t}\")) @filter(lt(region_episode.t_exit, \"{c}\")) {{ uid }} }}",
            t = esc(timescale),
            c = t3339(cutoff_ms)
        );
        let uids: Vec<u64> = self.rows(&dql).await?.iter().map(uid_of).collect();
        self.delete_uids(&uids).await
    }

    // ── event-time versions and as-of reads ────────────────────────────────
    async fn transition_versions(
        &self,
        from: &str,
        to: &str,
        sampler: &str,
        as_of_ms: Option<i64>,
        limit: usize,
    ) -> Result<Vec<TransitionVersion>> {
        let key = dsch::edge_key(from, to, sampler);
        let filter = as_of_ms
            .map(|t| format!(" @filter(le(transition_version.ts, \"{}\"))", t3339(t)))
            .unwrap_or_default();
        let dql = format!(
            "{{ q(func: eq(transition_version.edge_key, \"{k}\")){filter} {{ {ALL} }} }}",
            k = esc(&key)
        );
        let mut rows = self.rows(&dql).await?;
        by_ts_desc(&mut rows, "transition_version.ts");
        Ok(rows.iter().take(limit.max(1)).map(transition_version_of).collect())
    }

    async fn region_status_versions(
        &self,
        region_id: &str,
        as_of_ms: Option<i64>,
        limit: usize,
    ) -> Result<Vec<RegionStatusVersion>> {
        let filter = as_of_ms
            .map(|t| format!(" @filter(le(region_status_version.ts, \"{}\"))", t3339(t)))
            .unwrap_or_default();
        let dql = format!(
            "{{ q(func: eq(region_status_version.region_id, \"{r}\")){filter} {{ {ALL} }} }}",
            r = esc(region_id)
        );
        let mut rows = self.rows(&dql).await?;
        by_ts_desc(&mut rows, "region_status_version.ts");
        Ok(rows.iter().take(limit.max(1)).map(region_status_version_of).collect())
    }

    async fn latest_transition_versions(&self, keys: &[String], t_ms: i64) -> Result<Vec<Option<TransitionVersion>>> {
        let rows = self.latest_per_key("TransitionVersion", "transition_version", "edge_key", keys, t_ms).await?;
        Ok(rows.iter().map(|r| r.as_ref().map(transition_version_of)).collect())
    }

    async fn latest_region_status_versions(
        &self,
        keys: &[String],
        t_ms: i64,
    ) -> Result<Vec<Option<RegionStatusVersion>>> {
        let rows =
            self.latest_per_key("RegionStatusVersion", "region_status_version", "region_id", keys, t_ms).await?;
        Ok(rows.iter().map(|r| r.as_ref().map(region_status_version_of)).collect())
    }

    async fn prune_space_versions(&self, space: u64, cutoff_ms: i64) -> Result<u64> {
        // `row_number() OVER (PARTITION BY key ORDER BY ts DESC)` has no DQL form: read the
        // candidates, rank them here, and delete everything but the newest per key.
        let regions = <Self as crate::graph::GraphStore>::space_regions(self, space).await?;
        let ids: Vec<String> = regions.iter().map(|r| r.region_id.clone()).collect();
        if ids.is_empty() {
            return Ok(0);
        }
        let mut dropped = 0;
        let trans = self.rows_in("Transition", "transition.from_region_id", &ids).await?;
        let keys: Vec<String> = trans.iter().map(|t| str_of(t, "transition.key")).collect();
        for (ty, prefix, key_field, key_list) in [
            ("TransitionVersion", "transition_version", "edge_key", keys),
            ("RegionStatusVersion", "region_status_version", "region_id", ids),
        ] {
            for key in &key_list {
                let dql = format!(
                    "{{ q(func: eq({prefix}.{key_field}, \"{k}\")) @filter(lt({prefix}.ts, \"{c}\")) {{ uid {prefix}.ts }} }}",
                    k = esc(key),
                    c = t3339(cutoff_ms)
                );
                let mut rows = self.rows(&dql).await?;
                if rows.len() <= 1 {
                    continue;
                }
                by_ts_desc(&mut rows, &format!("{prefix}.ts"));
                let stale: Vec<u64> = rows.iter().skip(1).map(uid_of).collect();
                dropped += self.delete_uids(&stale).await?;
            }
            let _ = ty;
        }
        Ok(dropped)
    }

    async fn list_snapshots_as_of(&self, as_of_ms: i64) -> Result<Vec<Snapshot>> {
        let dql = format!(
            "{{ q(func: type(Snapshot)) @filter(le(snapshot.created_at, \"{t}\")) {{ {ALL} }} }}",
            t = t3339(as_of_ms)
        );
        Ok(self.rows(&dql).await?.iter().map(snapshot_of).collect())
    }

    async fn space_regions_as_of(&self, space: u64, as_of_ms: i64) -> Result<Vec<LatentRegion>> {
        let Some(sp) = self.by_uid("ExpertSpace", space).await? else { return Ok(Vec::new()) };
        let esid = str_of(&sp, "expert_space.expert_space_id");
        let dql = format!(
            "{{ q(func: eq(region.expert_space_id, \"{e}\")) @filter(le(region.created_at, \"{t}\")) {{ {ALL} }} }}",
            e = esc(&esid),
            t = t3339(as_of_ms)
        );
        Ok(self.rows(&dql).await?.iter().map(region_of).collect())
    }

    async fn snapshot_graph_as_of(&self, snapshot_id: &str, as_of_ms: i64) -> Result<Option<SnapshotGraph>> {
        let Some(snapshot) = <Self as crate::graph::GraphStore>::snapshot_by_id(self, snapshot_id).await? else {
            return Ok(None);
        };
        // a snapshot created after `t` was not visible then
        if ms_of(snapshot.created_at.as_deref()).is_some_and(|c| c > as_of_ms) {
            return Ok(None);
        }
        let spaces = self
            .rows(&format!(
                "{{ q(func: eq(expert_space.snapshot_id, \"{s}\"), orderasc: expert_space.index) {{ {ALL} }} }}",
                s = esc(snapshot_id)
            ))
            .await?;
        let region_rows = self.snapshot_region_rows(snapshot_id).await?;
        let visible: Vec<&Json> = region_rows
            .iter()
            .filter(|r| ms_of(r.get("region.created_at").and_then(Json::as_str)).is_none_or(|c| c <= as_of_ms))
            .collect();
        let ids: Vec<String> = visible.iter().map(|r| str_of(r, "region.region_id")).collect();
        let trans = self.rows_in("Transition", "transition.from_region_id", &ids).await?;
        Ok(Some(SnapshotGraph {
            snapshot,
            spaces: spaces.iter().map(space_of).collect(),
            regions: visible.iter().map(|r| region_of(r)).collect(),
            transitions: trans.iter().map(transition_of).collect(),
        }))
    }

    // ── genome links by business id ────────────────────────────────────────
    async fn link_trained_on(&self, links: &[crate::queries::TrainedOnLink]) -> Result<()> {
        for l in links {
            let key = format!("{}|{}|{}", l.genome_id, l.snapshot_id, l.generation);
            self.upsert_keyed(
                "TrainedOn",
                "trained_on.key",
                &key,
                obj(json!({
                    "trained_on.genome_id": l.genome_id, "trained_on.snapshot_id": l.snapshot_id,
                    "trained_on.generation": l.generation,
                })),
            )
            .await?;
        }
        Ok(())
    }

    async fn link_feedback_from(&self, links: &[crate::queries::FeedbackLink]) -> Result<()> {
        for l in links {
            let key = format!("{}|{}", l.snapshot_id, l.genome_id);
            self.upsert_keyed(
                "FeedbackFrom",
                "feedback_from.key",
                &key,
                obj(json!({
                    "feedback_from.snapshot_id": l.snapshot_id, "feedback_from.genome_id": l.genome_id,
                    "feedback_from.weight": l.weight, "feedback_from.output_family": l.output_family,
                    "feedback_from.score": l.score,
                })),
            )
            .await?;
        }
        Ok(())
    }

    async fn trained_on_snapshots(&self, genome_id: &str) -> Result<Vec<Snapshot>> {
        let links = self.rows_in("TrainedOn", "trained_on.genome_id", &[genome_id.to_owned()]).await?;
        let ids: Vec<String> = links.iter().map(|l| str_of(l, "trained_on.snapshot_id")).collect();
        let rows = self.rows_in("Snapshot", "snapshot.snapshot_id", &ids).await?;
        Ok(rows.iter().map(snapshot_of).collect())
    }

    async fn feedback_edges(&self, snapshot_id: &str) -> Result<Vec<EdgeRow>> {
        let rows = self.rows_in("FeedbackFrom", "feedback_from.snapshot_id", &[snapshot_id.to_owned()]).await?;
        Ok(rows
            .iter()
            .map(|r| EdgeRow {
                id: uid_of(r),
                label: "FeedbackFrom".into(),
                from: 0,
                to: 0,
                properties: bare_props(r, "feedback_from.", &["key"]),
            })
            .collect())
    }
}

fn region_status_version_of(j: &Json) -> RegionStatusVersion {
    RegionStatusVersion {
        id: uid_of(j),
        region_id: str_of(j, "region_status_version.region_id"),
        status: str_of(j, "region_status_version.status"),
        related_region_id: str_of(j, "region_status_version.related_region_id"),
        reason: str_of(j, "region_status_version.reason"),
        ts: opt_str(j, "region_status_version.ts"),
    }
}

fn transition_fields(
    from_id: &str,
    to_id: &str,
    from: u64,
    to: u64,
    t: &crate::queries::NewTransition,
) -> Map<String, Json> {
    obj(json!({
        "transition.from": {"uid": u64_to_uid(from)}, "transition.to": {"uid": u64_to_uid(to)},
        "transition.from_region_id": from_id, "transition.to_region_id": to_id,
        "transition.sampler_id": t.sampler_id, "transition.count": t.count, "transition.mean": t.mean,
        "transition.lower": t.lower, "transition.upper": t.upper, "transition.kernel_params": t.kernel_params,
        "transition.updated_at": now3339(),
    }))
}

impl DgraphGraph {
    /// Newest row per key with `ts <= t`: one named block per key in a single request, which is
    /// the shape the Helix backend already used (`versions.rs`). This is the `DISTINCT ON` the ADR
    /// measured at 209/505 ms.
    async fn latest_per_key(
        &self,
        ty: &str,
        prefix: &str,
        key_field: &str,
        keys: &[String],
        t_ms: i64,
    ) -> Result<Vec<Option<Json>>> {
        let mut out = Vec::with_capacity(keys.len());
        for chunk in keys.chunks(32) {
            let mut blocks = String::new();
            for (i, k) in chunk.iter().enumerate() {
                blocks.push_str(&format!(
                    "v{i}(func: eq({prefix}.{key_field}, \"{k}\")) @filter(type({ty}) AND le({prefix}.ts, \"{t}\")) {{ {ALL} }}\n",
                    k = esc(k),
                    t = t3339(t_ms)
                ));
            }
            let data = self.dg.query(&format!("{{ {blocks} }}")).await?;
            for i in 0..chunk.len() {
                let mut rows = data.get(format!("v{i}")).and_then(Json::as_array).cloned().unwrap_or_default();
                by_ts_desc(&mut rows, &format!("{prefix}.ts"));
                out.push(rows.into_iter().next());
            }
        }
        Ok(out)
    }
}

fn snapshot_fields(s: &NewSnapshot) -> Map<String, Json> {
    obj(json!({
        "snapshot.timescale": s.timescale, "snapshot.encoder_family": s.encoder_family,
        "snapshot.generation": s.generation, "snapshot.garage_key": s.garage_key,
        "snapshot.status": s.status, "snapshot.created_at": now3339(),
    }))
}

fn space_fields(s: &NewExpertSpace) -> Map<String, Json> {
    obj(json!({
        "expert_space.snapshot_id": s.snapshot_id, "expert_space.index": s.index, "expert_space.dim": s.dim,
    }))
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dql_literals_are_escaped() {
        assert_eq!(esc(r#"a"b"#), r#"a\"b"#);
        assert_eq!(esc(r"a\b"), r"a\\b");
        assert_eq!(esc_list(["a", r#"b"c"#]), r#""a", "b\"c""#);
    }

    #[test]
    fn bare_props_strips_the_type_prefix() {
        let j = json!({"uid": "0x1", "dgraph.type": "LatentRegion", "region.region_id": "r1", "region.mass": 2.5});
        let m = bare_props(&j, "region.", &[]);
        assert_eq!(m.get("region_id"), Some(&json!("r1")));
        assert_eq!(m.get("mass"), Some(&json!(2.5)));
        assert!(!m.contains_key("uid"), "uid is the node id, not a property");
        assert!(!m.contains_key("dgraph.type"));
    }

    #[test]
    fn bare_props_skips_nulls_and_requested_fields() {
        let j = json!({"region.region_id": "r1", "region.status": Json::Null, "region.mass": 1.0});
        let m = bare_props(&j, "region.", &["mass"]);
        assert_eq!(m.len(), 1);
        assert!(m.contains_key("region_id"));
    }

    #[test]
    fn uids_become_node_ids() {
        assert_eq!(uid_of(&json!({"uid": "0x2a"})), 42);
        assert_eq!(uid_of(&json!({})), 0);
    }

    #[test]
    fn timestamps_round_trip() {
        let ms = 1_763_000_000_123;
        assert_eq!(ms_of(Some(&t3339(ms))), Some(ms));
        assert_eq!(ms_of(None), None);
        assert_eq!(ms_of(Some("not a timestamp")), None);
    }

    #[test]
    fn transition_edges_read_through_an_array() {
        let j = json!({"uid": "0x5", "transition.from": [{"uid": "0x1"}], "transition.to": [{"uid": "0x2"}]});
        let t = transition_of(&j);
        assert_eq!((t.id, t.from, t.to), (5, 1, 2));
    }
}

/// Rows per import mutation. Large enough that 62 k versions are ~63 requests rather than 62 k,
/// small enough to stay well inside Dgraph's request limits.
///
/// Note the blank-node labels must differ *within* a chunk: `_:v` repeated across objects of one
/// mutation names a single node, so a shared label silently collapses the whole chunk into one
/// row instead of erroring. Caught by the migration's row-count verification (62 594 → 1 858).
const IMPORT_CHUNK: usize = 1_000;

/// One `TransitionVersion` as read from the source, for [`DgraphGraph::dg_import_transition_versions`].
#[derive(Debug, Clone)]
pub struct TransitionVersionImport {
    pub edge_key: String,
    pub from_region_id: String,
    pub to_region_id: String,
    pub sampler_id: String,
    pub count: i64,
    pub mean: f64,
    pub lower: f64,
    pub upper: f64,
    pub kernel_params: String,
    pub ts: String,
}

/// Historical node timestamps as read from the source: the creation / update marks the runtime
/// API cannot carry.
///
/// Every live write stamps `now()`, as it must, so copying current values through `GraphStore`
/// *manufactures* `snapshot.created_at`, `region.created_at` and every other node timestamp —
/// they come out as the migration's wall-clock. Nothing errors and the row counts still match,
/// which is exactly how the first cutover (2026-11-16 20:48) left all five snapshots and 451
/// regions stamped at migration time: an as-of read for any earlier instant then sees no
/// snapshot at all, because `snapshot_graph_as_of` filters `le(snapshot.created_at, t)`.
#[derive(Debug, Clone)]
pub struct NodeTimesImport {
    /// Dgraph type of the node, e.g. `Snapshot`.
    pub ty: &'static str,
    /// Predicate holding its business key, e.g. `snapshot.snapshot_id`.
    pub key_pred: &'static str,
    pub key: String,
    /// `(predicate, RFC 3339)` pairs; empty values are skipped.
    pub times: Vec<(&'static str, String)>,
}

/// One `RegionStatusVersion` as read from the source.
#[derive(Debug, Clone)]
pub struct RegionStatusVersionImport {
    pub region_id: String,
    pub status: String,
    pub related_region_id: String,
    pub reason: String,
    pub ts: String,
}

/// Bulk import used only by the D-110 migration (`examples/graph_migrate.rs`).
///
/// `GraphStore` can *read* event-time history but has no way to write it: versions are appended as
/// a side effect of `upsert_transitions` / `mark_region_status`, which would stamp `now()` and lose
/// the original `ts`. A faithful copy therefore needs these, and they exist for the migration
/// rather than for the running platform — nothing else should append history directly.
impl DgraphGraph {
    /// Append `TransitionVersion` rows with their original timestamps, batched.
    ///
    /// Batching is not an optimisation here: a real book carried 62 594 version rows, and one
    /// HTTP mutation each would make the cutover a multi-minute wall of round trips. Each chunk is
    /// one atomic mutation, and the rows are append-only, so a re-run after a failure duplicates
    /// only the chunk boundary rather than corrupting anything.
    /// Set node timestamps ([`NodeTimesImport`]) to the values the source holds.
    ///
    /// The uid of each node is resolved by a read first and the mutation addresses it directly.
    /// An upsert block would be shorter but unsafe here: in Dgraph a `uid(v)` whose variable
    /// matched nothing *creates* a node, so one stale key would quietly add an empty node to the
    /// store instead of being skipped.
    pub async fn dg_import_node_times(&self, rows: &[NodeTimesImport]) -> Result<usize> {
        let mut by_type: std::collections::BTreeMap<(&str, &str), Vec<&NodeTimesImport>> = Default::default();
        for r in rows.iter().filter(|r| r.times.iter().any(|(_, t)| !t.is_empty())) {
            by_type.entry((r.ty, r.key_pred)).or_default().push(r);
        }
        let mut written = 0;
        for ((ty, key_pred), group) in by_type {
            for chunk in group.chunks(500) {
                let keys = chunk.iter().map(|r| format!("\"{}\"", esc(&r.key))).collect::<Vec<_>>().join(", ");
                let dql =
                    format!("{{ q(func: eq({key_pred}, [{keys}])) @filter(type({ty})) {{ uid {key_pred} }} }}");
                let found = self.rows(&dql).await?;
                let uids: std::collections::HashMap<&str, u64> = found
                    .iter()
                    .filter_map(|j| j.get(key_pred).and_then(Json::as_str).map(|k| (k, uid_of(j))))
                    .collect();
                let mut set = Vec::new();
                for r in chunk {
                    let Some(uid) = uids.get(r.key.as_str()) else {
                        continue; // not in the destination: the generic pass skipped it, nothing to stamp
                    };
                    let mut m = Map::new();
                    m.insert("uid".into(), json!(u64_to_uid(*uid)));
                    for (pred, t) in r.times.iter().filter(|(_, t)| !t.is_empty()) {
                        m.insert((*pred).to_owned(), json!(t));
                    }
                    set.push(Json::Object(m));
                }
                if set.is_empty() {
                    continue;
                }
                written += set.len();
                self.dg.mutate(set).await?;
            }
        }
        Ok(written)
    }

    pub async fn dg_import_transition_versions(&self, rows: &[TransitionVersionImport]) -> Result<usize> {
        let mut n = 0;
        for chunk in rows.chunks(IMPORT_CHUNK) {
            let set: Vec<Json> = chunk
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    json!({
                        "uid": format!("_:v{i}"), "dgraph.type": "TransitionVersion",
                        "transition_version.edge_key": v.edge_key,
                        "transition_version.from_region_id": v.from_region_id,
                        "transition_version.to_region_id": v.to_region_id,
                        "transition_version.sampler_id": v.sampler_id,
                        "transition_version.count": v.count,
                        "transition_version.mean": v.mean,
                        "transition_version.lower": v.lower,
                        "transition_version.upper": v.upper,
                        "transition_version.kernel_params": v.kernel_params,
                        "transition_version.ts": v.ts,
                    })
                })
                .collect();
            self.dg.mutate(set).await?;
            n += chunk.len();
        }
        Ok(n)
    }

    /// Append `RegionStatusVersion` rows with their original timestamps, batched.
    pub async fn dg_import_region_status_versions(&self, rows: &[RegionStatusVersionImport]) -> Result<usize> {
        let mut n = 0;
        for chunk in rows.chunks(IMPORT_CHUNK) {
            let set: Vec<Json> = chunk
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    json!({
                        "uid": format!("_:v{i}"), "dgraph.type": "RegionStatusVersion",
                        "region_status_version.region_id": v.region_id,
                        "region_status_version.status": v.status,
                        "region_status_version.related_region_id": v.related_region_id,
                        "region_status_version.reason": v.reason,
                        "region_status_version.ts": v.ts,
                    })
                })
                .collect();
            self.dg.mutate(set).await?;
            n += chunk.len();
        }
        Ok(n)
    }

    /// A `RegionMatch` addressed by business ids. The trait's `match_regions` takes node ids,
    /// which do not survive a backend change (D-108: ids never cross backends).
    pub async fn dg_import_region_match(&self, prev_region_id: &str, next_region_id: &str, score: f64) -> Result<()> {
        let key = format!("{prev_region_id}|{next_region_id}");
        self.upsert_keyed(
            "RegionMatch",
            "region_match.key",
            &key,
            obj(json!({
                "region_match.prev_region_id": prev_region_id,
                "region_match.next_region_id": next_region_id,
                "region_match.score": score,
            })),
        )
        .await
        .map(drop)
    }

    /// A `SnapshotParent` addressed by business ids, keeping `op` — which the trait's
    /// `snapshot_parent` read drops, because it returns the parent snapshot and not the edge.
    pub async fn dg_import_snapshot_parent(&self, parent_id: &str, child_id: &str, op: &str) -> Result<()> {
        let key = format!("{parent_id}|{child_id}");
        let (p, c) = (
            self.by_key("Snapshot", "snapshot.snapshot_id", parent_id).await?,
            self.by_key("Snapshot", "snapshot.snapshot_id", child_id).await?,
        );
        let (Some(p), Some(c)) = (p, c) else { return Ok(()) };
        self.upsert_keyed(
            "SnapshotParent",
            "snapshot_parent.key",
            &key,
            obj(json!({
                "snapshot_parent.parent": {"uid": u64_to_uid(uid_of(&p))},
                "snapshot_parent.child": {"uid": u64_to_uid(uid_of(&c))},
                "snapshot_parent.op": op,
            })),
        )
        .await
        .map(drop)
    }

    /// Drop event-time versions — all of them, or only those stamped at or after `since`.
    ///
    /// Two separate problems make this necessary for the migration, and both showed up only in the
    /// row-count verification:
    ///
    /// 1. Copying *current values* through the runtime API manufactures history, by design:
    ///    `upsert_transitions` and `upsert_region` each append a version stamped `now()`, because
    ///    that is what a live write must do. Importing the real history on top left both
    ///    (62 594 real + 1 795 manufactured).
    /// 2. The history import is append-only, so running the migration twice doubled it
    ///    (125 188 rows).
    ///
    /// Passing `None` makes the import *authoritative*: the source's history replaces whatever the
    /// destination has, so the migration converges on a re-run instead of accumulating. That is
    /// the right semantic for a cutover — the destination is not yet taking live writes — and it
    /// is why the migration must finish before the platform starts writing to Dgraph.
    pub async fn dg_drop_versions(&self, since: Option<&str>) -> Result<u64> {
        let mut n = 0;
        for (ty, prefix) in
            [("TransitionVersion", "transition_version"), ("RegionStatusVersion", "region_status_version")]
        {
            let filter = since
                .map(|t| format!(" @filter(ge({prefix}.ts, \"{}\"))", esc(t)))
                .unwrap_or_default();
            let dql = format!("{{ q(func: type({ty})){filter} {{ uid }} }}");
            let uids: Vec<u64> = self.rows(&dql).await?.iter().map(uid_of).collect();
            n += self.delete_uids(&uids).await?;
        }
        Ok(n)
    }

    /// Count the nodes of one type — the migration's row-count verification.
    pub async fn dg_count(&self, ty: &str) -> Result<u64> {
        let d = self.dg.query(&format!("{{ q(func: type({ty})) {{ count(uid) }} }}")).await?;
        Ok(d.pointer("/q/0/count").and_then(Json::as_u64).unwrap_or(0))
    }

    /// Oldest and newest value of one datetime predicate, for the migration's timestamp check.
    ///
    /// DQL has no `min()` over a predicate across a type, so this is two one-row ordered reads.
    pub async fn dg_time_range(&self, ty: &str, pred: &str) -> Result<(Option<String>, Option<String>)> {
        let one = |dir: &str| {
            format!("{{ q(func: type({ty}), order{dir}: {pred}, first: 1) @filter(has({pred})) {{ {pred} }} }}")
        };
        let read = |dql: String| async move {
            let d = self.dg.query(&dql).await?;
            Ok::<Option<String>, HubError>(
                d.pointer("/q/0").and_then(|j| j.get(pred)).and_then(Json::as_str).map(str::to_owned),
            )
        };
        Ok((read(one("asc")).await?, read(one("desc")).await?))
    }
}
