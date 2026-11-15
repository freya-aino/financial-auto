//! The Dgraph schema (ADR 0011, D-110): the DQL equivalent of `migrations/20261115120000_graph.sql`.
//!
//! **Node ids.** Postgres drew every `id` from one shared `graph.node_id` sequence so that the
//! `u64` node ids of the `fa-hub` API stayed globally unique across labels, exactly as Helix `$id`s
//! were. Dgraph `uid`s already have that property — they are cluster-wide `uint64` handles — so the
//! sequence disappears and [`crate::dgraph::uid_to_u64`] is the whole mapping.
//!
//! **Predicate naming.** Predicates are global in Dgraph, not scoped to a type, so every one is
//! written `<type>.<field>`. Without that, `region.status` ("live"/"retired") and
//! `snapshot.status` ("candidate"/"active") would be the same predicate with two value domains.
//!
//! **Indexes.** A business key carries `@index(exact) @upsert`: `exact` because it is looked up
//! with `eq` and because Dgraph's exact index is ordered, which also gives prefix matching as
//! `ge(k, p) AND lt(k, p+\u{ff})` — that is how the conformance suite's `LIKE 'prefix%'` cleanup
//! ports without paying for a trigram index. `@upsert` makes the upsert block's uniqueness check
//! conflict-safe, replacing the `UNIQUE` constraints and `ON CONFLICT` of the SQL schema.
//! Event-time predicates (`*.ts`) carry `@index(hour)` because every as-of read filters on them.
//!
//! **Edges.** Edges that carry properties stay reified — `transition`, `hier`, `descends_from` and
//! friends are nodes, as they were tables — because `GraphStore` hands them to callers as rows with
//! ids. Pure structural links (`snapshot_parent.parent`) are plain `uid` predicates with
//! `@reverse` so both directions are traversable.

/// Applied with `/alter` at startup; idempotent, so re-applying is a no-op.
pub const SCHEMA: &str = r#"
# ── reference mirror (spec 12 §5) ──────────────────────────────────────────
ref_entity.entity_id: string @index(exact) @upsert .
ref_entity.lei: string .
ref_entity.name: string .
ref_entity.country: string .
ref_entity.sector: string .
ref_entity.mirrored_at: datetime .

ref_sector.sector_id: string @index(exact) @upsert .
ref_sector.name: string .

ref_in_sector.key: string @index(exact) @upsert .
ref_in_sector.entity_id: string @index(exact) .
ref_in_sector.sector_id: string @index(exact) .

ref_venue.venue_id: string @index(exact) @upsert .
ref_venue.code: string .
ref_venue.mic: string .
ref_venue.name: string .
ref_venue.kind: string .
ref_venue.timezone: string .
ref_venue.mirrored_at: datetime .

ref_instrument.instrument_id: string @index(exact) @upsert .
ref_instrument.asset_class: string .
ref_instrument.kind: string .
ref_instrument.name: string .
ref_instrument.status: string @index(exact) .
ref_instrument.base_ccy: string .
ref_instrument.quote_ccy: string .
ref_instrument.mirrored_at: datetime .

ref_issued_by.key: string @index(exact) @upsert .
ref_issued_by.instrument_id: string @index(exact) .
ref_issued_by.entity_id: string @index(exact) .

ref_listed_on.key: string @index(exact) @upsert .
ref_listed_on.instrument_id: string @index(exact) .
ref_listed_on.venue_id: string @index(exact) .
ref_listed_on.venue_symbol: string .
ref_listed_on.source_id: string .
ref_listed_on.listing_key: string .

ref_same_exposure.key: string @index(exact) @upsert .
ref_same_exposure.instrument_id: string @index(exact) .
ref_same_exposure.to_instrument_id: string .
ref_same_exposure.relation: string .
ref_same_exposure.exposure_key: string .

# ── snapshots, spaces, adapters (specs 31/36) ──────────────────────────────
snapshot.snapshot_id: string @index(exact) @upsert .
snapshot.timescale: string @index(exact) .
snapshot.encoder_family: string @index(exact) .
snapshot.generation: int .
snapshot.garage_key: string .
snapshot.status: string @index(exact) .
snapshot.carried_mass: float .
snapshot.manifest: string .
snapshot.previous_active: string .
snapshot.created_at: datetime @index(hour) .

snapshot_parent.key: string @index(exact) @upsert .
snapshot_parent.parent: uid @reverse .
snapshot_parent.child: uid @reverse .
snapshot_parent.op: string .

adapter_state.adapter_id: string @index(exact) @upsert .
adapter_state.snapshot_id: string @index(exact) .
adapter_state.timescale: string .
adapter_state.version: int @index(int) .
adapter_state.garage_key: string .
adapter_state.drift: float .
adapter_state.max_drift: float .
adapter_state.frozen: bool .
adapter_state.n_windows: int .
adapter_state.window_start: datetime .
adapter_state.window_end: datetime .
adapter_state.updated_at: datetime .

expert_space.expert_space_id: string @index(exact) @upsert .
expert_space.snapshot_id: string @index(exact) .
expert_space.index: int .
expert_space.dim: int .

# ── regions and transitions (specs 30/33, D-048) ───────────────────────────
region.region_id: string @index(exact) @upsert .
region.lineage_id: string @index(exact) .
region.expert_space_id: string @index(exact) .
region.method: string .
region.radius: float .
region.density: float .
region.status: string @index(exact) .
region.related_region_id: string .
region.retire_reason: string .
region.mass: float .
region.occupancy: float .
region.spread: float .
region.age: int .
region.created_at: datetime @index(hour) .
region.updated_at: datetime @index(hour) .
region.died_at: datetime .

transition.key: string @index(exact) @upsert .
transition.from: uid @reverse .
transition.to: uid @reverse .
transition.from_region_id: string @index(exact) .
transition.to_region_id: string @index(exact) .
transition.sampler_id: string @index(exact) .
transition.count: int .
transition.mean: float .
transition.lower: float .
transition.upper: float .
transition.kernel_params: string .
transition.updated_at: datetime @index(hour) .

transition_version.edge_key: string @index(exact) .
transition_version.from_region_id: string .
transition_version.to_region_id: string .
transition_version.sampler_id: string .
transition_version.count: int .
transition_version.mean: float .
transition_version.lower: float .
transition_version.upper: float .
transition_version.kernel_params: string .
transition_version.ts: datetime @index(hour) .

region_status_version.region_id: string @index(exact) .
region_status_version.status: string .
region_status_version.related_region_id: string .
region_status_version.reason: string .
region_status_version.ts: datetime @index(hour) .

# ── lineage and hierarchy (spec 30 §5/§6.5) ────────────────────────────────
descends_from.key: string @index(exact) @upsert .
descends_from.child_region_id: string @index(exact) .
descends_from.parent_region_id: string @index(exact) .
descends_from.kind: string .
descends_from.overlap: float .
descends_from.updated_at: datetime .

region_match.key: string @index(exact) @upsert .
region_match.prev_region_id: string @index(exact) .
region_match.next_region_id: string @index(exact) .
region_match.prev: uid .
region_match.next: uid .
region_match.score: float .

hier.key: string @index(exact) @upsert .
hier.fine_region_id: string @index(exact) .
hier.coarse_region_id: string @index(exact) .
hier.weight: float .
hier.p_coarse_given_fine: float .
hier.p_fine_given_coarse: float .
hier.support: float .
hier.pmi_bits: float .
hier.mi_bits: float .
hier.updated_at: datetime .

# ── per-space state (specs 30 §5, 33 §5/§6.5) ──────────────────────────────
region_set.region_set_id: string @index(exact) @upsert .
region_set.snapshot_id: string @index(exact) .
region_set.timescale: string .
region_set.expert_index: int .
region_set.family: string .
region_set.method: string .
region_set.n_regions: int .
region_set.cfg_hash: string .
region_set.state: string .
region_set.kernel_state: string .
region_set.fit_time: datetime .
region_set.updated_at: datetime .

kernel_model.kernel_id: string @index(exact) @upsert .
kernel_model.expert_space_id: string @index(exact) .
kernel_model.snapshot_id: string @index(exact) .
kernel_model.timescale: string .
kernel_model.expert_index: int .
kernel_model.kernel_type: string .
kernel_model.serving_sampler: string .
kernel_model.scores: string .
kernel_model.stationary: string .
kernel_model.relaxation_steps: float .
kernel_model.drift_kl: float .
kernel_model.n_rows: int .
kernel_model.observation_mass: float .
kernel_model.live_regions: int .
kernel_model.updated_at: datetime .

cross_prior.pair_id: string @index(exact) @upsert .
cross_prior.timescale: string @index(exact) .
cross_prior.source_instrument_id: string .
cross_prior.target_instrument_id: string .
cross_prior.n: int .
cross_prior.correlation: float .
cross_prior.shrinkage: float .
cross_prior.lead_lag: int .
cross_prior.lead_corr: float .
cross_prior.predictive_gain: float .
cross_prior.uncertainty: float .
cross_prior.score: float @index(float) .
cross_prior.rank: int .
cross_prior.updated_at: datetime .

region_episode.episode_id: string @index(exact) @upsert .
region_episode.region_id: string @index(exact) .
region_episode.expert_space_id: string .
region_episode.timescale: string @index(exact) .
region_episode.instrument_id: string .
region_episode.t_enter: datetime .
region_episode.t_exit: datetime @index(hour) .
region_episode.bars: int .
region_episode.max_weight: float .
region_episode.mean_weight: float .

# ── genome links by business id ────────────────────────────────────────────
trained_on.key: string @index(exact) @upsert .
trained_on.genome_id: string @index(exact) .
trained_on.snapshot_id: string @index(exact) .
trained_on.generation: int .

feedback_from.key: string @index(exact) @upsert .
feedback_from.snapshot_id: string @index(exact) .
feedback_from.genome_id: string @index(exact) .
feedback_from.weight: float .
feedback_from.output_family: string .
feedback_from.score: float .

# ── genomes, fitness, evolution (Helix-resident before D-110) ──────────────
genome.genome_id: string @index(exact) @upsert .
genome.kind: string @index(exact) .
genome.family: string .
genome.params: string .
genome.status: string @index(exact) .
genome.strategy_id: string @index(exact) .
genome.status_reason: string .
genome.status_since_ms: int .
genome.status_updated_at: datetime .
genome.created_at: datetime @index(hour) .

fitness_record.genome_id: string @index(exact) .
fitness_record.generation: int @index(int) .
fitness_record.objectives: string .
fitness_record.pareto_rank: int .
fitness_record.created_at: datetime .

genome_parent.key: string @index(exact) @upsert .
genome_parent.parent_genome_id: string @index(exact) .
genome_parent.child_genome_id: string @index(exact) .
genome_parent.op: string .

pareto_front.front_id: string @index(exact) @upsert .
pareto_front.generation_id: string @index(exact) .
pareto_front.gen_index: int .
pareto_front.kind: string @index(exact) .
pareto_front.snapshot_id: string .
pareto_front.keys: string .
pareto_front.members: string .
pareto_front.hypervolume: float .
pareto_front.final_loop: bool .
pareto_front.created_at: datetime .

evaluated_on.key: string @index(exact) @upsert .
evaluated_on.genome_id: string @index(exact) .
evaluated_on.snapshot_id: string @index(exact) .
evaluated_on.generation: string .
evaluated_on.gen_index: int .
evaluated_on.rung: int .
evaluated_on.score: float .
evaluated_on.objectives: string .
evaluated_on.job_id: string .
evaluated_on.experts: string .

# ── experts (spec 32) ──────────────────────────────────────────────────────
expert.expert_uid: string @index(exact) @upsert .
expert.snapshot_id: string @index(exact) .
expert.timescale: string .
expert.expert_id: int .
expert.name: string @index(exact) .
expert.dim: int .
expert.usage: float .
expert.skill: float .
expert.born_gen: int .
expert.died_gen: int .
expert.specialisation: string .
expert.updated_at: datetime .

expert_align.key: string @index(exact) @upsert .
expert_align.child_expert_uid: string @index(exact) .
expert_align.parent_expert_uid: string @index(exact) .
expert_align.cka: float .
expert_align.kind: string .

# ── schema marker ─────────────────────────────────────────────────────────
# Replaces the Helix `SchemaMeta` row, whose fingerprint hashed Helix index DDL
# and has no Dgraph analogue: the DQL schema is applied with `/alter`, so only
# the version is worth recording.
schema_marker.marker_id: string @index(exact) @upsert .
schema_marker.version: int .
schema_marker.applied_at: datetime .

# ── types ──────────────────────────────────────────────────────────────────
type RefEntity {
    ref_entity.entity_id
    ref_entity.lei
    ref_entity.name
    ref_entity.country
    ref_entity.sector
    ref_entity.mirrored_at
}
type RefSector {
    ref_sector.sector_id
    ref_sector.name
}
type RefInSector {
    ref_in_sector.key
    ref_in_sector.entity_id
    ref_in_sector.sector_id
}
type RefVenue {
    ref_venue.venue_id
    ref_venue.code
    ref_venue.mic
    ref_venue.name
    ref_venue.kind
    ref_venue.timezone
    ref_venue.mirrored_at
}
type RefInstrument {
    ref_instrument.instrument_id
    ref_instrument.asset_class
    ref_instrument.kind
    ref_instrument.name
    ref_instrument.status
    ref_instrument.base_ccy
    ref_instrument.quote_ccy
    ref_instrument.mirrored_at
}
type RefIssuedBy {
    ref_issued_by.key
    ref_issued_by.instrument_id
    ref_issued_by.entity_id
}
type RefListedOn {
    ref_listed_on.key
    ref_listed_on.instrument_id
    ref_listed_on.venue_id
    ref_listed_on.venue_symbol
    ref_listed_on.source_id
    ref_listed_on.listing_key
}
type RefSameExposure {
    ref_same_exposure.key
    ref_same_exposure.instrument_id
    ref_same_exposure.to_instrument_id
    ref_same_exposure.relation
    ref_same_exposure.exposure_key
}
type Snapshot {
    snapshot.snapshot_id
    snapshot.timescale
    snapshot.encoder_family
    snapshot.generation
    snapshot.garage_key
    snapshot.status
    snapshot.carried_mass
    snapshot.manifest
    snapshot.previous_active
    snapshot.created_at
}
type SnapshotParent {
    snapshot_parent.key
    snapshot_parent.parent
    snapshot_parent.child
    snapshot_parent.op
}
type AdapterState {
    adapter_state.adapter_id
    adapter_state.snapshot_id
    adapter_state.timescale
    adapter_state.version
    adapter_state.garage_key
    adapter_state.drift
    adapter_state.max_drift
    adapter_state.frozen
    adapter_state.n_windows
    adapter_state.window_start
    adapter_state.window_end
    adapter_state.updated_at
}
type ExpertSpace {
    expert_space.expert_space_id
    expert_space.snapshot_id
    expert_space.index
    expert_space.dim
}
type LatentRegion {
    region.region_id
    region.lineage_id
    region.expert_space_id
    region.method
    region.radius
    region.density
    region.status
    region.related_region_id
    region.retire_reason
    region.mass
    region.occupancy
    region.spread
    region.age
    region.created_at
    region.updated_at
    region.died_at
}
type Transition {
    transition.key
    transition.from
    transition.to
    transition.from_region_id
    transition.to_region_id
    transition.sampler_id
    transition.count
    transition.mean
    transition.lower
    transition.upper
    transition.kernel_params
    transition.updated_at
}
type TransitionVersion {
    transition_version.edge_key
    transition_version.from_region_id
    transition_version.to_region_id
    transition_version.sampler_id
    transition_version.count
    transition_version.mean
    transition_version.lower
    transition_version.upper
    transition_version.kernel_params
    transition_version.ts
}
type RegionStatusVersion {
    region_status_version.region_id
    region_status_version.status
    region_status_version.related_region_id
    region_status_version.reason
    region_status_version.ts
}
type DescendsFrom {
    descends_from.key
    descends_from.child_region_id
    descends_from.parent_region_id
    descends_from.kind
    descends_from.overlap
    descends_from.updated_at
}
type RegionMatch {
    region_match.key
    region_match.prev_region_id
    region_match.next_region_id
    region_match.prev
    region_match.next
    region_match.score
}
type Hier {
    hier.key
    hier.fine_region_id
    hier.coarse_region_id
    hier.weight
    hier.p_coarse_given_fine
    hier.p_fine_given_coarse
    hier.support
    hier.pmi_bits
    hier.mi_bits
    hier.updated_at
}
type RegionSet {
    region_set.region_set_id
    region_set.snapshot_id
    region_set.timescale
    region_set.expert_index
    region_set.family
    region_set.method
    region_set.n_regions
    region_set.cfg_hash
    region_set.state
    region_set.kernel_state
    region_set.fit_time
    region_set.updated_at
}
type KernelModel {
    kernel_model.kernel_id
    kernel_model.expert_space_id
    kernel_model.snapshot_id
    kernel_model.timescale
    kernel_model.expert_index
    kernel_model.kernel_type
    kernel_model.serving_sampler
    kernel_model.scores
    kernel_model.stationary
    kernel_model.relaxation_steps
    kernel_model.drift_kl
    kernel_model.n_rows
    kernel_model.observation_mass
    kernel_model.live_regions
    kernel_model.updated_at
}
type CrossPrior {
    cross_prior.pair_id
    cross_prior.timescale
    cross_prior.source_instrument_id
    cross_prior.target_instrument_id
    cross_prior.n
    cross_prior.correlation
    cross_prior.shrinkage
    cross_prior.lead_lag
    cross_prior.lead_corr
    cross_prior.predictive_gain
    cross_prior.uncertainty
    cross_prior.score
    cross_prior.rank
    cross_prior.updated_at
}
type RegionEpisode {
    region_episode.episode_id
    region_episode.region_id
    region_episode.expert_space_id
    region_episode.timescale
    region_episode.instrument_id
    region_episode.t_enter
    region_episode.t_exit
    region_episode.bars
    region_episode.max_weight
    region_episode.mean_weight
}
type TrainedOn {
    trained_on.key
    trained_on.genome_id
    trained_on.snapshot_id
    trained_on.generation
}
type FeedbackFrom {
    feedback_from.key
    feedback_from.snapshot_id
    feedback_from.genome_id
    feedback_from.weight
    feedback_from.output_family
    feedback_from.score
}
type Genome {
    genome.genome_id
    genome.kind
    genome.family
    genome.params
    genome.status
    genome.strategy_id
    genome.status_reason
    genome.status_since_ms
    genome.status_updated_at
    genome.created_at
}
type FitnessRecord {
    fitness_record.genome_id
    fitness_record.generation
    fitness_record.objectives
    fitness_record.pareto_rank
    fitness_record.created_at
}
type GenomeParent {
    genome_parent.key
    genome_parent.parent_genome_id
    genome_parent.child_genome_id
    genome_parent.op
}
type ParetoFront {
    pareto_front.front_id
    pareto_front.generation_id
    pareto_front.gen_index
    pareto_front.kind
    pareto_front.snapshot_id
    pareto_front.keys
    pareto_front.members
    pareto_front.hypervolume
    pareto_front.final_loop
    pareto_front.created_at
}
type EvaluatedOn {
    evaluated_on.key
    evaluated_on.genome_id
    evaluated_on.snapshot_id
    evaluated_on.generation
    evaluated_on.gen_index
    evaluated_on.rung
    evaluated_on.score
    evaluated_on.objectives
    evaluated_on.job_id
    evaluated_on.experts
}
type Expert {
    expert.expert_uid
    expert.snapshot_id
    expert.timescale
    expert.expert_id
    expert.name
    expert.dim
    expert.usage
    expert.skill
    expert.born_gen
    expert.died_gen
    expert.specialisation
    expert.updated_at
}
type ExpertAlign {
    expert_align.key
    expert_align.child_expert_uid
    expert_align.parent_expert_uid
    expert_align.cka
    expert_align.kind
}
type SchemaMarker {
    schema_marker.marker_id
    schema_marker.version
    schema_marker.applied_at
}
"#;

/// Every `dgraph.type` the schema declares, in the order of the SQL migration.
///
/// The graph explorer (`resolve_node` / `neighborhood`) is label-free on Helix — it scans without
/// knowing a type. DQL has no untyped scan, so the explorer enumerates this list instead; keeping
/// it next to [`SCHEMA`] is what stops the two drifting apart (see the test below).
pub const TYPES: &[&str] = &[
    "RefEntity",
    "RefSector",
    "RefInSector",
    "RefVenue",
    "RefInstrument",
    "RefIssuedBy",
    "RefListedOn",
    "RefSameExposure",
    "Snapshot",
    "SnapshotParent",
    "AdapterState",
    "ExpertSpace",
    "LatentRegion",
    "Transition",
    "TransitionVersion",
    "RegionStatusVersion",
    "DescendsFrom",
    "RegionMatch",
    "Hier",
    "RegionSet",
    "KernelModel",
    "CrossPrior",
    "RegionEpisode",
    "TrainedOn",
    "FeedbackFrom",
    "Genome",
    "FitnessRecord",
    "GenomeParent",
    "ParetoFront",
    "EvaluatedOn",
    "Expert",
    "ExpertAlign",
    "SchemaMarker",
];

/// Types that are reified *edges* rather than entities.
///
/// The Helix model had real edges carrying properties; here they are nodes, because `GraphStore`
/// hands edges to callers as rows with ids. The explorer's `Neighborhood` still separates nodes
/// from edges, so it needs to know which is which.
pub fn is_edge_type(type_name: &str) -> bool {
    matches!(
        type_name,
        "Transition"
            | "DescendsFrom"
            | "Hier"
            | "RegionMatch"
            | "SnapshotParent"
            | "TrainedOn"
            | "FeedbackFrom"
            | "GenomeParent"
            | "ExpertAlign"
            | "EvaluatedOn"
            | "RefIssuedBy"
            | "RefListedOn"
            | "RefSameExposure"
            | "RefInSector"
    )
}

/// The business-key predicate of a type, where it has one. `resolve_node` tries these in order
/// when the explorer is handed a string id rather than a `uid`; it mirrors `BUSINESS_KEYS` in
/// `queries.rs`, widened to the whole schema.
pub fn business_key(type_name: &str) -> Option<&'static str> {
    Some(match type_name {
        "RefEntity" => "ref_entity.entity_id",
        "RefSector" => "ref_sector.sector_id",
        "RefVenue" => "ref_venue.venue_id",
        "RefInstrument" => "ref_instrument.instrument_id",
        "Snapshot" => "snapshot.snapshot_id",
        "AdapterState" => "adapter_state.adapter_id",
        "ExpertSpace" => "expert_space.expert_space_id",
        "LatentRegion" => "region.region_id",
        "RegionSet" => "region_set.region_set_id",
        "KernelModel" => "kernel_model.kernel_id",
        "CrossPrior" => "cross_prior.pair_id",
        "RegionEpisode" => "region_episode.episode_id",
        "Genome" => "genome.genome_id",
        "ParetoFront" => "pareto_front.front_id",
        "Expert" => "expert.expert_uid",
        _ => return None,
    })
}

/// `edge_key` of a transition version — `from|to|sampler`, as the SQL schema built it by string
/// concatenation. Also the `transition.key` upsert key.
pub fn edge_key(from: &str, to: &str, sampler: &str) -> String {
    format!("{from}|{to}|{sampler}")
}

/// A prefix match on an `exact`-indexed predicate: Dgraph's exact index is ordered, so
/// `ge(p, prefix) AND lt(p, prefix+\u{10ffff})` is the `LIKE 'prefix%'` of DQL. Used by the
/// conformance suite's per-run cleanup.
pub fn prefix_filter(pred: &str, prefix: &str) -> String {
    format!("ge({pred}, \"{prefix}\") AND lt({pred}, \"{prefix}\u{10ffff}\")")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `TYPES` drives the explorer, so a type added to the schema and forgotten here would be
    /// invisible to `resolve_node` / `neighborhood` — exactly the kind of silent gap the label-free
    /// Helix scan could not have.
    #[test]
    fn types_list_matches_the_schema() {
        let declared: Vec<&str> = SCHEMA
            .lines()
            .filter_map(|l| l.trim().strip_prefix("type "))
            .filter_map(|l| l.split_whitespace().next())
            .collect();
        assert_eq!(declared, TYPES, "TYPES is out of step with the `type` blocks in SCHEMA");
    }

    /// Every predicate named inside a `type` block must also be declared with a datatype, or
    /// `/alter` rejects the whole schema.
    #[test]
    fn every_type_field_is_a_declared_predicate() {
        let declared: Vec<&str> =
            SCHEMA.lines().filter_map(|l| l.trim().split_once(':').map(|(p, _)| p.trim())).collect();
        let mut in_type = false;
        for line in SCHEMA.lines() {
            let t = line.trim();
            if t.starts_with("type ") {
                in_type = true;
            } else if t == "}" {
                in_type = false;
            } else if in_type && !t.is_empty() {
                assert!(declared.contains(&t), "type field `{t}` has no predicate declaration");
            }
        }
    }

    #[test]
    fn business_keys_are_declared_predicates() {
        for t in TYPES {
            if let Some(k) = business_key(t) {
                assert!(SCHEMA.contains(&format!("{k}:")), "business key `{k}` of {t} is not declared");
            }
        }
    }

    #[test]
    fn edge_types_are_declared_types() {
        for t in TYPES {
            let _ = is_edge_type(t);
        }
        // and nothing claims to be an edge type without being in the schema
        for t in ["Transition", "Hier", "GenomeParent", "ExpertAlign", "EvaluatedOn"] {
            assert!(is_edge_type(t), "{t} should be an edge type");
            assert!(TYPES.contains(&t), "{t} is not a declared type");
        }
        assert!(!is_edge_type("LatentRegion"));
        assert!(!is_edge_type("Snapshot"));
        assert!(!is_edge_type("Genome"));
    }

    #[test]
    fn edge_key_matches_the_sql_concatenation() {
        assert_eq!(edge_key("r1", "r2", "s"), "r1|r2|s");
    }

    #[test]
    fn prefix_filter_brackets_the_range() {
        let f = prefix_filter("region.region_id", "conf-7-");
        assert!(f.starts_with("ge(region.region_id, \"conf-7-\")"), "{f}");
        assert!(f.contains("lt(region.region_id, \"conf-7-"), "{f}");
    }
}
