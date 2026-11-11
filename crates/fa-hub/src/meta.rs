//! Schema / migration version marker stored in Helix (spec 20 §8 "migration version stored in
//! Helix", §10 Q6).
//!
//! The standalone server is schemaless, so "the schema" is (a) the static index set of
//! [`crate::schema::static_indexes`] and (b) the shape of the rows the typed queries write. Both
//! are versioned by [`SCHEMA_VERSION`]; one `SchemaMeta` node (`meta_id = "fa-hub"`) records the
//! version Helix was last migrated to and a fingerprint of the index set.
//!
//! [`HubClient::migrate_schema`] (run by [`HubClient::ensure_schema_ready`] after the index
//! builds) applies every [`MIGRATIONS`] step above the stored version, in order, stamping after
//! each, and refuses with [`HubError::SchemaTooNew`] when a newer build already stamped a higher
//! version. The marker only ever moves forward.
//!
//! Adding a migration: bump [`SCHEMA_VERSION`], append a [`Migration`] and implement its data
//! step in `apply_migration` (index-only changes need no step: `ensure_schema` is idempotent and
//! the fingerprint records the new set).

use serde::{Deserialize, Serialize};

use crate::ast::{Batch, Cond, Input, NodeRef, Op, Pred};
use crate::schema::{node, static_indexes};
use crate::{HubClient, HubError, Result};

/// Data-model version this build writes and reads.
pub const SCHEMA_VERSION: u32 = 1;
/// `meta_id` of the fa-hub marker row.
pub const META_ID: &str = "fa-hub";

/// One forward migration step to `version`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Migration {
    pub version: u32,
    pub description: &'static str,
}

/// Every version, oldest first; the last one is [`SCHEMA_VERSION`].
pub const MIGRATIONS: &[Migration] = &[Migration {
    version: 1,
    description: "baseline: schema.hx indexes (D-043), LatentVec.vec<dim> per-dim indexes, SchemaMeta marker",
}];

/// The `SchemaMeta` row.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SchemaMarker {
    #[serde(rename = "$id")]
    pub id: u64,
    pub meta_id: String,
    pub version: u32,
    #[serde(default)]
    pub index_fingerprint: String,
    #[serde(default)]
    pub writer: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
}

/// FNV-1a (64-bit) over the canonical JSON of the static index set: changes whenever an index is
/// added, removed or redefined. Order-independent.
pub fn index_fingerprint() -> String {
    let mut specs: Vec<String> = static_indexes().into_iter().map(|s| s.into_json().to_string()).collect();
    specs.sort_unstable();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in specs.join("\n").bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

impl HubClient {
    /// The marker row, `None` when Helix was never stamped (pre-marker data or an empty server).
    pub async fn schema_marker(&self) -> Result<Option<SchemaMarker>> {
        let op = Op::nodes_with_label_where(node::SCHEMA_META, Pred::eq("meta_id", META_ID)).value_map(None);
        let mut resp = self.batch(&Batch::read().named("schema_marker").ret("row", op)).await?;
        resp.take_opt("row")
    }

    /// Apply pending migrations and stamp the marker (see the module docs). Indexes must already
    /// be active (`ensure_schema_ready` calls this after waiting for them). No write when the
    /// stored version and fingerprint already match.
    pub async fn migrate_schema(&self) -> Result<SchemaMarker> {
        let current = self.schema_marker().await?;
        let stored = current.as_ref().map_or(0, |m| m.version);
        if stored > SCHEMA_VERSION {
            return Err(HubError::SchemaTooNew { stored, supported: SCHEMA_VERSION });
        }
        let fingerprint = index_fingerprint();
        if let Some(m) = current.as_ref()
            && stored == SCHEMA_VERSION
            && m.index_fingerprint == fingerprint
        {
            return Ok(m.clone());
        }
        let mut last = current;
        for m in MIGRATIONS.iter().filter(|m| m.version > stored) {
            self.apply_migration(m).await?;
            last = Some(self.stamp_schema_version(m.version, &fingerprint).await?);
        }
        match last {
            Some(m) if m.version == SCHEMA_VERSION && m.index_fingerprint == fingerprint => Ok(m),
            // Same version, new index set: refresh the fingerprint only.
            _ => self.stamp_schema_version(SCHEMA_VERSION, &fingerprint).await,
        }
    }

    /// Data step of one migration. Version 1 is the baseline (nothing to rewrite).
    async fn apply_migration(&self, m: &Migration) -> Result<()> {
        match m.version {
            1 => Ok(()),
            v => Err(HubError::BadParam(format!("no data step implemented for schema version {v}"))),
        }
    }

    /// Upsert the marker to `version` (never lowers it: an existing row with a higher version is
    /// left alone and returned). Safe under concurrent boots: a lost create race (409 unique) is
    /// retried as an update.
    pub async fn stamp_schema_version(&self, version: u32, fingerprint: &str) -> Result<SchemaMarker> {
        for attempt in 0..2 {
            let existing = Op::nodes_with_label_where(node::SCHEMA_META, Pred::eq("meta_id", META_ID));
            let raise = Op::nodes(NodeRef::var("existing"))
                .where_(Pred::lte("version", version))
                .set_property("version", version)
                .set_property("index_fingerprint", fingerprint)
                .set_property("writer", env!("CARGO_PKG_VERSION"))
                .set_property("updated_at", Input::Now);
            let created = Op::add_n(
                node::SCHEMA_META,
                vec![
                    ("meta_id", META_ID.into()),
                    ("version", version.into()),
                    ("index_fingerprint", fingerprint.into()),
                    ("writer", env!("CARGO_PKG_VERSION").into()),
                    ("updated_at", Input::Now),
                ],
            );
            let batch = Batch::write()
                .named("stamp_schema_version")
                .var("existing", existing)
                .var_if("raised", Cond::VarNotEmpty("existing".into()), raise)
                .var_if("created", Cond::VarEmpty("existing".into()), created)
                .var("row", Op::nodes_with_label_where(node::SCHEMA_META, Pred::eq("meta_id", META_ID)))
                .ret("out", Op::nodes(NodeRef::var("row")).value_map(None));
            match self.batch(&batch).await {
                Ok(mut resp) => {
                    let mut rows: Vec<SchemaMarker> = resp.take("out")?;
                    if let Some(m) = rows.pop() {
                        return Ok(m);
                    }
                    // Read-your-writes not visible in the same batch: read back separately.
                    return self
                        .schema_marker()
                        .await?
                        .ok_or_else(|| HubError::NotFound { label: node::SCHEMA_META.into(), id: META_ID.into() });
                }
                Err(e) if e.is_unique_violation() && attempt == 0 => continue,
                Err(e) => return Err(e),
            }
        }
        unreachable!("loop returns on the second attempt")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrations_end_at_the_current_version_and_increase() {
        assert_eq!(MIGRATIONS.last().map(|m| m.version), Some(SCHEMA_VERSION));
        assert!(MIGRATIONS.windows(2).all(|w| w[0].version < w[1].version));
        assert_eq!(MIGRATIONS[0].version, 1);
    }

    #[test]
    fn fingerprint_is_stable_and_hex() {
        let a = index_fingerprint();
        assert_eq!(a, index_fingerprint());
        assert_eq!(a.len(), 16);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn marker_deserializes() {
        let m: SchemaMarker = serde_json::from_value(serde_json::json!({
            "$id": 4, "$label": "SchemaMeta", "meta_id": "fa-hub", "version": 1,
            "index_fingerprint": "00ff", "writer": "0.1.0", "updated_at": "2026-11-11T00:00:00.000Z"
        }))
        .unwrap();
        assert_eq!((m.version, m.index_fingerprint.as_str()), (1, "00ff"));
    }
}
