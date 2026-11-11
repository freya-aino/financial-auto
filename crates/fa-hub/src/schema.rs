//! The data model of `helix/schema.hx`, expressed as what the standalone server actually
//! needs declared: nothing for node/edge labels and properties (the server is schemaless and
//! infers them on write), and explicit index DDL for `UNIQUE INDEX` / `INDEX` fields and for
//! every vector property.
//!
//! Labels and property names below are the single source of truth for the Rust side; keep
//! them in sync with `helix/schema.hx`, which stays the human-readable catalog.

use crate::ast::{IndexSpec, Metric};

/// Node labels (`N::…` in `schema.hx`).
pub mod node {
    pub const INSTRUMENT: &str = "Instrument";
    pub const ENTITY: &str = "Entity";
    pub const VENUE: &str = "Venue";
    pub const SNAPSHOT: &str = "Snapshot";
    /// Online-adapter version of a snapshot (spec 36 §5 `AdapterState`): written by the
    /// trainer's `adapter` job next to the Garage artifact.
    pub const ADAPTER_STATE: &str = "AdapterState";
    pub const EXPERT_SPACE: &str = "ExpertSpace";
    pub const LATENT_REGION: &str = "LatentRegion";
    pub const WINDOW: &str = "Window";
    pub const GENOME: &str = "Genome";
    pub const FITNESS_RECORD: &str = "FitnessRecord";
    /// `V::LatentVec`: a node whose vector lives in a `vec<dim>` property (see [`super::latent_vec_property`]).
    pub const LATENT_VEC: &str = "LatentVec";
    /// `V::NewsDoc`: text embedding in `vec<dim>` (one index per dim, no tenant).
    pub const NEWS_DOC: &str = "NewsDoc";
    /// `V::MediaSegment`: media embedding in `vec<dim>` (one index per dim, no tenant).
    pub const MEDIA_SEGMENT: &str = "MediaSegment";
    /// Schema / migration version marker (spec 20 §8), one row per `meta_id` (see [`crate::meta`]).
    pub const SCHEMA_META: &str = "SchemaMeta";
}

/// Edge labels (`E::…` in `schema.hx`).
pub mod edge {
    pub const ISSUED_BY: &str = "IssuedBy";
    pub const LISTED_ON: &str = "ListedOn";
    pub const SAME_EXPOSURE: &str = "SameExposure";
    pub const HAS_SPACE: &str = "HasSpace";
    /// Snapshot lineage (spec 36 §5): parent snapshot → child snapshot, `op` = the job kind that
    /// produced the child (`pretrain | generation | adapter`).
    pub const SNAPSHOT_PARENT: &str = "SnapshotParent";
    /// Snapshot → its adapter versions.
    pub const HAS_ADAPTER: &str = "HasAdapter";
    pub const IN_SPACE: &str = "InSpace";
    pub const TRANSITION: &str = "Transition";
    pub const HIER: &str = "Hier";
    pub const REGION_MATCH: &str = "RegionMatch";
    pub const CONTAINS: &str = "Contains";
    pub const PARENT_OF: &str = "ParentOf";
    pub const SCORED: &str = "Scored";
    pub const TRAINED_ON: &str = "TrainedOn";
    pub const FEEDBACK_FROM: &str = "FeedbackFrom";
    pub const MENTIONS: &str = "Mentions";
}

/// Tenant property of every latent vector index: one HNSW partition per expert space.
pub const LATENT_TENANT: &str = "expert_space_id";

/// Property holding a latent vector of `dim` components. A vector index is keyed by
/// `(label, property)` and has a fixed dimension, so each dimension gets its own property
/// (and index) instead of one `LatentVec` type with mixed dimensions.
pub fn latent_vec_property(dim: usize) -> String {
    format!("vec{dim}")
}

/// Vector index for latents of one dimension, partitioned by expert space.
pub fn latent_vec_index(dim: usize) -> IndexSpec {
    IndexSpec::node_vector(node::LATENT_VEC, &latent_vec_property(dim), dim, Metric::Cosine, Some(LATENT_TENANT))
}

/// Vector index for the text / media embeddings (`NewsDoc`, `MediaSegment`) of one dimension.
/// Not tenant-partitioned: one index per `(label, dim)` holds every source.
pub fn embedding_index(label: &str, dim: usize) -> IndexSpec {
    IndexSpec::node_vector(label, &latent_vec_property(dim), dim, Metric::Cosine, None)
}

/// Static indexes (every `UNIQUE INDEX` / `INDEX` line of `schema.hx`, plus lookups the typed
/// queries rely on). Vector indexes are created on demand because their dimension is only
/// known at runtime ([`crate::HubClient::ensure_latent_index`]).
pub fn static_indexes() -> Vec<IndexSpec> {
    use IndexSpec as I;
    vec![
        I::node_equality(node::INSTRUMENT, "instrument_id", true),
        I::node_equality(node::ENTITY, "entity_id", true),
        I::node_equality(node::VENUE, "venue_id", true),
        I::node_equality(node::SNAPSHOT, "snapshot_id", true),
        I::node_equality(node::SNAPSHOT, "timescale", false),
        I::node_equality(node::ADAPTER_STATE, "adapter_id", true),
        I::node_equality(node::ADAPTER_STATE, "snapshot_id", false),
        I::node_equality(node::EXPERT_SPACE, "expert_space_id", true),
        I::node_equality(node::EXPERT_SPACE, "snapshot_id", false),
        I::node_equality(node::LATENT_REGION, "region_id", true),
        I::node_equality(node::LATENT_REGION, "lineage_id", false),
        I::node_equality(node::LATENT_REGION, "expert_space_id", false),
        I::node_equality(node::WINDOW, "window_id", true),
        I::node_equality(node::GENOME, "genome_id", true),
        I::node_equality(node::FITNESS_RECORD, "genome_id", false),
        I::node_equality(node::LATENT_VEC, LATENT_TENANT, false),
        I::node_equality(node::LATENT_VEC, "ref_id", false),
        I::node_equality(node::NEWS_DOC, "doc_id", false),
        I::node_equality(node::MEDIA_SEGMENT, "segment_id", false),
        I::node_equality(node::SCHEMA_META, "meta_id", true),
        I::node_equality(crate::regions::labels::REGION_SET, "region_set_id", true),
        I::node_equality(crate::regions::labels::REGION_SET, "snapshot_id", false),
        I::node_equality(crate::regions::labels::KERNEL_MODEL, "kernel_id", true),
        I::node_equality(crate::regions::labels::KERNEL_MODEL, "snapshot_id", false),
        I::node_equality(crate::regions::labels::CROSS_PRIOR, "pair_id", true),
        I::node_equality(crate::regions::labels::CROSS_PRIOR, "timescale", false),
        I::node_equality(crate::experts::labels::EXPERT, "expert_uid", true),
        I::node_equality(crate::experts::labels::EXPERT, "snapshot_id", false),
        I::edge_equality(edge::TRANSITION, "sampler_id"),
        I::node_equality(crate::versions::labels::TRANSITION_VERSION, "edge_key", false),
        I::node_equality(crate::versions::labels::REGION_STATUS_VERSION, "region_id", false),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value as Json, json};

    const SCHEMA_HX: &str = include_str!("../../../helix/schema.hx");

    /// `(label, field, unique)` of every `UNIQUE INDEX` / `INDEX` line of `schema.hx`, with the
    /// `N::` / `V::` block it belongs to.
    fn hx_indexes() -> Vec<(String, String, bool)> {
        let mut out = Vec::new();
        let mut label: Option<String> = None;
        for raw in SCHEMA_HX.lines() {
            let line = raw.split("//").next().unwrap_or("").trim();
            if let Some(rest) = line.strip_prefix("N::").or_else(|| line.strip_prefix("V::")) {
                label = Some(rest.trim_end_matches('{').trim().to_owned());
                continue;
            }
            if line.starts_with("E::") {
                label = None;
                continue;
            }
            let (unique, rest) = if let Some(r) = line.strip_prefix("UNIQUE INDEX ") {
                (true, r)
            } else if let Some(r) = line.strip_prefix("INDEX ") {
                (false, r)
            } else {
                continue;
            };
            let field = rest.split(':').next().unwrap_or("").trim().to_owned();
            out.push((label.clone().expect("index line inside a node block"), field, unique));
        }
        out
    }

    #[test]
    fn static_indexes_cover_every_index_line_of_schema_hx() {
        let have: Vec<Json> = static_indexes().into_iter().map(IndexSpec::into_json).collect();
        let hx = hx_indexes();
        assert!(hx.len() >= 14, "parser found too few index lines: {hx:?}");
        for (label, field, unique) in hx {
            let want = json!({ "node_equality": { "label": label, "property": field, "unique": unique } });
            assert!(
                have.contains(&want),
                "schema.hx `{label}.{field}` (unique={unique}) has no index in static_indexes()"
            );
        }
    }

    #[test]
    fn static_index_labels_are_declared_in_schema_hx() {
        for spec in static_indexes() {
            let j = spec.into_json();
            let (kind, body) = j.as_object().and_then(|o| o.iter().next()).expect("tagged spec");
            let label = body["label"].as_str().expect("label");
            let prefix = if kind.starts_with("edge") { "E::" } else { "" };
            assert!(
                SCHEMA_HX.contains(&format!("{prefix}{label} {{")) || SCHEMA_HX.contains(&format!("::{label} {{")),
                "{kind} index on `{label}` but schema.hx declares no such type"
            );
        }
    }

    #[test]
    fn vector_index_shapes() {
        let j = latent_vec_index(8).into_json();
        assert_eq!(j["node_vector"]["property"], "vec8");
        assert_eq!(j["node_vector"]["tenant_property"], LATENT_TENANT);
        let j = embedding_index(node::NEWS_DOC, 384).into_json();
        assert_eq!(j["node_vector"]["label"], "NewsDoc");
        assert_eq!(j["node_vector"]["dimension"], 384);
        assert!(j["node_vector"].get("tenant_property").is_none());
    }
}
