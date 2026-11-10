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
    pub const EXPERT_SPACE: &str = "ExpertSpace";
    pub const LATENT_REGION: &str = "LatentRegion";
    pub const WINDOW: &str = "Window";
    pub const GENOME: &str = "Genome";
    pub const FITNESS_RECORD: &str = "FitnessRecord";
    /// `V::LatentVec`: a node whose vector lives in a `vec<dim>` property (see [`super::latent_vec_property`]).
    pub const LATENT_VEC: &str = "LatentVec";
    pub const NEWS_DOC: &str = "NewsDoc";
    pub const MEDIA_SEGMENT: &str = "MediaSegment";
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
        I::edge_equality(edge::TRANSITION, "sampler_id"),
    ]
}
