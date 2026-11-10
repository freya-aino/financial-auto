// HelixQL query catalog (spec 20). Readable reference only: the standalone server (D-033) does
// not compile .hx queries into endpoints. Each QUERY below is implemented as a typed function of
// the same name (snake_case) on `fa_hub::HubClient` (crates/fa-hub/src/queries.rs), sent as a
// `/v2/query` operation-tree batch. `ID` params are Helix node ids (u64). Verified by
// `cargo test -p fa-hub --test live`.

// ── Reference ───────────────────────────────────────────────────────────
QUERY upsertInstrument(instrument_id: String, asset_class: String, kind: String, name: String) =>
    inst <- AddN<Instrument>({instrument_id: instrument_id, asset_class: asset_class, kind: kind, name: name})
    RETURN inst

QUERY getInstrument(instrument_id: String) =>
    inst <- N<Instrument>({instrument_id: instrument_id})
    RETURN inst

// ── Snapshots and expert spaces ─────────────────────────────────────────
QUERY addSnapshot(snapshot_id: String, timescale: String, encoder_family: String, generation: U32, garage_key: String, status: String) =>
    s <- AddN<Snapshot>({snapshot_id: snapshot_id, timescale: timescale, encoder_family: encoder_family, generation: generation, garage_key: garage_key, status: status})
    RETURN s

QUERY addExpertSpace(snapshot: ID, expert_space_id: String, snapshot_id: String, index: U32, dim: U32) =>
    space <- AddN<ExpertSpace>({expert_space_id: expert_space_id, snapshot_id: snapshot_id, index: index, dim: dim})
    edge <- AddE<HasSpace>::From(snapshot)::To(space)
    RETURN space

// Get-or-create by business id (model-runtime boot, idempotent across restarts).
QUERY ensureSnapshot(snapshot_id: String, timescale: String, encoder_family: String, generation: U32, garage_key: String, status: String) =>
    existing <- N<Snapshot>({snapshot_id: snapshot_id})
    created <- AddN<Snapshot>({snapshot_id: snapshot_id, timescale: timescale, encoder_family: encoder_family, generation: generation, garage_key: garage_key, status: status}) // only when existing is empty
    RETURN existing, created

QUERY ensureExpertSpace(snapshot: ID, expert_space_id: String, snapshot_id: String, index: U32, dim: U32) =>
    existing <- N<ExpertSpace>({expert_space_id: expert_space_id})
    created <- AddN<ExpertSpace>({expert_space_id: expert_space_id, snapshot_id: snapshot_id, index: index, dim: dim}) // only when existing is empty
    edge <- AddE<HasSpace>::From(snapshot)::To(created)
    RETURN existing, created

QUERY listSnapshots() =>
    snaps <- N<Snapshot>
    RETURN snaps

QUERY snapshotSpaces(snapshot: ID) =>
    spaces <- N<Snapshot>(snapshot)::Out<HasSpace>
    RETURN spaces

// ── Regions and transitions ─────────────────────────────────────────────
QUERY addRegion(space: ID, region_id: String, lineage_id: String, expert_space_id: String, method: String, radius: F64, density: F64) =>
    region <- AddN<LatentRegion>({region_id: region_id, lineage_id: lineage_id, expert_space_id: expert_space_id, method: method, radius: radius, density: density})
    edge <- AddE<InSpace>::From(region)::To(space)
    RETURN region

// Upsert by business id: an existing region (same id after a model-runtime restart) is revived
// (status live, radius/density refreshed); otherwise it is inserted with its InSpace edge.
QUERY upsertRegion(space: ID, region_id: String, lineage_id: String, expert_space_id: String, method: String, radius: F64, density: F64) =>
    existing <- N<LatentRegion>({region_id: region_id})
    updated <- existing::UPDATE({status: "live", related_region_id: "", radius: radius, density: density, updated_at: NOW}) // when existing is not empty
    created <- AddN<LatentRegion>({region_id: region_id, lineage_id: lineage_id, expert_space_id: expert_space_id, method: method, radius: radius, density: density, status: "live", related_region_id: ""}) // when existing is empty
    edge <- AddE<InSpace>::From(created)::To(space)
    RETURN updated, created

QUERY getRegion(region_id: String) =>
    region <- N<LatentRegion>({region_id: region_id})
    RETURN region

// Lifecycle marks (spec 30 §4): status merged | retired, related = absorbed / survivor region id.
QUERY markRegionStatus(region_id: String, status: String, related_region_id: String) =>
    region <- N<LatentRegion>({region_id: region_id})::UPDATE({status: status, related_region_id: related_region_id, updated_at: NOW})
    RETURN region

QUERY spaceRegions(space: ID) =>
    regions <- N<ExpertSpace>(space)::In<InSpace>
    RETURN regions

QUERY addTransition(from: ID, to: ID, sampler_id: String, count: I64, mean: F64, lower: F64, upper: F64, kernel_params: String) =>
    t <- AddE<Transition>({sampler_id: sampler_id, count: count, mean: mean, lower: lower, upper: upper, kernel_params: kernel_params})::From(from)::To(to)
    RETURN t

// Upsert one Transition edge by business ids (spec 33 §6 throttled posterior writes): the edge
// From(from_region_id) with the same to_region_id and sampler_id is updated in place, else added.
QUERY upsertTransition(from_region_id: String, to_region_id: String, sampler_id: String, count: I64, mean: F64, lower: F64, upper: F64, kernel_params: String) =>
    from <- N<LatentRegion>({region_id: from_region_id})
    to <- N<LatentRegion>({region_id: to_region_id})
    existing <- from::OutE<Transition>::WHERE(AND(_::{to_region_id}::EQ(to_region_id), _::{sampler_id}::EQ(sampler_id)))
    updated <- existing::UPDATE({count: count, mean: mean, lower: lower, upper: upper, kernel_params: kernel_params, updated_at: NOW}) // when existing is not empty
    created <- AddE<Transition>({from_region_id: from_region_id, to_region_id: to_region_id, sampler_id: sampler_id, count: count, mean: mean, lower: lower, upper: upper, kernel_params: kernel_params, updated_at: NOW})::From(from)::To(to) // when existing is empty
    RETURN updated, created

QUERY regionTransitions(region: ID) =>
    edges <- N<LatentRegion>(region)::OutE<Transition>
    RETURN edges

QUERY regionNeighbors(region: ID) =>
    next <- N<LatentRegion>(region)::Out<Transition>
    RETURN next

// Whole persisted view of one snapshot (spec 50 /api/latent/{snapshot}/regions, /api/mcmc/{snapshot}/graph).
QUERY snapshotGraph(snapshot_id: String) =>
    snap <- N<Snapshot>({snapshot_id: snapshot_id})
    spaces <- snap::Out<HasSpace>
    regions <- spaces::In<InSpace>
    transitions <- regions::OutE<Transition>
    RETURN snap, spaces, regions, transitions

QUERY matchRegions(prev: ID, next: ID, score: F64) =>
    m <- AddE<RegionMatch>({score: score})::From(prev)::To(next)
    RETURN m

// ── Latent vectors (centroids, observations, retrieval windows) ─────────
QUERY addLatent(vector: [F64], expert_space_id: String, kind: String, ref_id: String, instrument_id: String, uncertainty: F64) =>
    v <- AddV<LatentVec>(vector, {expert_space_id: expert_space_id, kind: kind, ref_id: ref_id, instrument_id: instrument_id, uncertainty: uncertainty})
    RETURN v

// Retrieval decoders (spec 34): nearest analogs; caller filters by expert_space_id and time distance.
QUERY searchLatent(vector: [F64], k: I64) =>
    hits <- SearchV<LatentVec>(vector, k)
    RETURN hits

// ── Evolution ───────────────────────────────────────────────────────────
QUERY addGenome(genome_id: String, kind: String, family: String, params: String) =>
    g <- AddN<Genome>({genome_id: genome_id, kind: kind, family: family, params: params})
    RETURN g

QUERY addLineage(parent: ID, child: ID, op: String) =>
    e <- AddE<ParentOf>({op: op})::From(parent)::To(child)
    RETURN e

QUERY addFitness(genome: ID, genome_id: String, generation: U32, objectives: String, pareto_rank: U32) =>
    rec <- AddN<FitnessRecord>({genome_id: genome_id, generation: generation, objectives: objectives, pareto_rank: pareto_rank})
    edge <- AddE<Scored>::From(genome)::To(rec)
    RETURN rec

// Business-id variants for evo-worker (D-048): genome rows are keyed by genome_id, not node id.
QUERY upsertGenome(genome_id: String, kind: String, family: String, params: String) =>
    existing <- N<Genome>({genome_id: genome_id})
    created <- AddN<Genome>({genome_id: genome_id, kind: kind, family: family, params: params}) // when existing is empty
    RETURN existing, created

QUERY linkParent(parent_id: String, child_id: String, op: String) =>
    parent <- N<Genome>({genome_id: parent_id})
    child <- N<Genome>({genome_id: child_id})
    e <- AddE<ParentOf>({op: op})::From(parent)::To(child)
    RETURN e

QUERY addFitnessRecord(genome_id: String, generation: U32, objectives: String, pareto_rank: U32) =>
    genome <- N<Genome>({genome_id: genome_id})
    rec <- AddN<FitnessRecord>({genome_id: genome_id, generation: generation, objectives: objectives, pareto_rank: pareto_rank})
    edge <- AddE<Scored>::From(genome)::To(rec)
    RETURN rec

QUERY genomeParents(genome: ID) =>
    parents <- N<Genome>(genome)::In<ParentOf>
    RETURN parents

QUERY genomeFitness(genome: ID) =>
    recs <- N<Genome>(genome)::Out<Scored>
    RETURN recs

// ── Generic graph reads (spec 50 §4.4) ──────────────────────────────────
// resolveNode: a node by any unique business id (region_id | snapshot_id | expert_space_id | genome_id | instrument_id).
QUERY resolveNode(business_id: String) =>
    n <- N::WHERE(OR(_::{region_id}::EQ(business_id), _::{snapshot_id}::EQ(business_id), _::{expert_space_id}::EQ(business_id), _::{genome_id}::EQ(business_id), _::{instrument_id}::EQ(business_id)))
    RETURN n

// neighborhood(node: ID, depth) and genomeAncestry(genome: ID, max_depth) are depth-unrolled
// batches (one Both / In<ParentOf> hop per entry); see fa-hub `queries.rs`.
//
// Named-query catalog exposed to `POST /api/graph/query` (fa-hub `catalog.rs`, whitelist, business-id
// params): getInstrument, listSnapshots, snapshotSpaces, snapshotGraph, spaceRegions, getRegion,
// regionTransitions, regionNeighbors, searchLatent, genomeParents, genomeFitness, genomeAncestry,
// resolveNode, neighborhood.
