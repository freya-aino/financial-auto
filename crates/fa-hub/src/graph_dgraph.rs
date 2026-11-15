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
// The row conversions and mutation helpers below are consumed by `impl GraphStore for
// DgraphGraph`, which lands with the rest of D-110 phase 2; this allow comes off with it.
#![allow(dead_code)]

use serde_json::{Map, Value as Json, json};

use crate::dgraph::{DgraphClient, u64_to_uid, uid_to_u64, unique_violation};
use crate::dgraph_schema as dsch;
use crate::queries::{AdapterState, ExpertSpace, Instrument, LatentRegion, NodeRow, Snapshot, Transition};
use crate::versions::TransitionVersion;
use crate::{HubError, Result};

/// The Dgraph graph store. Cheap to clone.
#[derive(Clone, Debug)]
pub struct DgraphGraph {
    dg: DgraphClient,
}

/// Escape a DQL string literal. Business ids come from catalogues and model code, so they can
/// legitimately contain quotes; an unescaped one would be a query-injection bug, not a typo.
fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// `"a", "b"` — a DQL value list for `eq(pred, [..])`.
fn esc_list<'a>(vals: impl IntoIterator<Item = &'a str>) -> String {
    vals.into_iter().map(|v| format!("\"{}\"", esc(v))).collect::<Vec<_>>().join(", ")
}

fn str_of(j: &Json, k: &str) -> String {
    j.get(k).and_then(Json::as_str).unwrap_or_default().to_owned()
}

fn opt_str(j: &Json, k: &str) -> Option<String> {
    j.get(k).and_then(Json::as_str).map(str::to_owned)
}

fn f64_of(j: &Json, k: &str) -> f64 {
    j.get(k).and_then(Json::as_f64).unwrap_or_default()
}

fn opt_f64(j: &Json, k: &str) -> Option<f64> {
    j.get(k).and_then(Json::as_f64)
}

fn i64_of(j: &Json, k: &str) -> i64 {
    j.get(k).and_then(Json::as_i64).unwrap_or_default()
}

fn u32_of(j: &Json, k: &str) -> u32 {
    u32::try_from(i64_of(j, k)).unwrap_or_default()
}

fn u64_of(j: &Json, k: &str) -> u64 {
    u64::try_from(i64_of(j, k)).unwrap_or_default()
}

fn bool_of(j: &Json, k: &str) -> bool {
    j.get(k).and_then(Json::as_bool).unwrap_or_default()
}

/// The `uid` of a row as the `u64` node id the API hands out.
fn uid_of(j: &Json) -> u64 {
    j.get("uid").and_then(Json::as_str).and_then(uid_to_u64).unwrap_or_default()
}

/// Strip the `<type>.` prefix from every predicate, so a generic row looks like the Postgres one
/// (`region_id`, `status`, …) to `NodeRow` / `EdgeRow` consumers and to the frontend.
fn bare_props(j: &Json, prefix: &str, skip: &[&str]) -> Map<String, Json> {
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

fn node_row(j: &Json, label: &str, prefix: &str) -> NodeRow {
    NodeRow { id: uid_of(j), label: label.to_owned(), properties: bare_props(j, prefix, &[]) }
}

// ── row conversions, one per `GraphStore` return shape ─────────────────────

fn snapshot_of(j: &Json) -> Snapshot {
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
    async fn rows(&self, dql: &str) -> Result<Vec<Json>> {
        let d = self.dg.query(dql).await?;
        Ok(d.get("q").and_then(Json::as_array).cloned().unwrap_or_default())
    }

    /// The single row of the `q` block, if any.
    async fn one(&self, dql: &str) -> Result<Option<Json>> {
        Ok(self.rows(dql).await?.into_iter().next())
    }

    /// Every predicate of one node addressed by business key.
    async fn by_key(&self, ty: &str, key_pred: &str, key: &str) -> Result<Option<Json>> {
        let dql = format!(
            "{{ q(func: eq({key_pred}, \"{k}\")) @filter(type({ty})) {{ uid dgraph.type expand(_all_) }} }}",
            k = esc(key)
        );
        self.one(&dql).await
    }

    /// Every predicate of one node addressed by uid, checked against its type.
    async fn by_uid(&self, ty: &str, id: u64) -> Result<Option<Json>> {
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
    async fn upsert_keyed(&self, ty: &str, key_pred: &str, key: &str, fields: Map<String, Json>) -> Result<u64> {
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
    async fn insert_keyed(&self, ty: &str, key_pred: &str, key: &str, fields: Map<String, Json>) -> Result<u64> {
        if self.by_key(ty, key_pred, key).await?.is_some() {
            return Err(unique_violation(&format!("{ty} `{key}`")));
        }
        self.upsert_keyed(ty, key_pred, key, fields).await
    }

    /// Set predicates on a node already addressed by uid.
    async fn set_on(&self, id: u64, fields: Map<String, Json>) -> Result<()> {
        let mut set = fields;
        set.insert("uid".into(), json!(u64_to_uid(id)));
        self.dg.mutate(vec![Json::Object(set)]).await.map(drop)
    }

    /// Delete whole nodes by uid.
    async fn delete_uids(&self, uids: &[u64]) -> Result<u64> {
        if uids.is_empty() {
            return Ok(0);
        }
        let del: Vec<Json> = uids.iter().map(|u| json!({"uid": u64_to_uid(*u)})).collect();
        let n = del.len() as u64;
        self.dg.delete(del).await?;
        Ok(n)
    }

    /// Append an immutable event-time row (the `*_version` tables are append-only).
    async fn append(&self, ty: &str, fields: Map<String, Json>) -> Result<()> {
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

fn now3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// Parse an RFC 3339 timestamp back to epoch milliseconds, for the as-of comparisons.
fn ms_of(s: Option<&str>) -> Option<i64> {
    s.and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok()).map(|t| t.timestamp_millis())
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
