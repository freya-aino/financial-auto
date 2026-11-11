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
    version <- AddN<RegionStatusVersion>({region_id: region_id, status: status, related_region_id: related_region_id, reason: "", ts: NOW}) // same batch, when region exists
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
    version <- AddN<TransitionVersion>({edge_key: from_region_id + "|" + to_region_id + "|" + sampler_id, from_region_id: from_region_id, to_region_id: to_region_id, sampler_id: sampler_id, count: count, mean: mean, lower: lower, upper: upper, kernel_params: kernel_params, ts: NOW}) // same batch
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

// ── Region lifecycle, lineage, hierarchy, kernels (specs 30 §5, 33 §5; fa-hub `regions.rs`) ──
QUERY updateRegionStats(region_id: String, mass: F64, occupancy: F64, spread: F64, age: I64) =>
    r <- N<LatentRegion>({region_id: region_id})::UPDATE({mass: mass, occupancy: occupancy, spread: spread, age: age, updated_at: NOW})
    RETURN r

QUERY markRegionDied(region_id: String, reason: String, related_region_id: String) =>
    r <- N<LatentRegion>({region_id: region_id})::UPDATE({status: "retired", retire_reason: reason, related_region_id: related_region_id, died_at: NOW})
    RETURN r

// upsertCentroid: drop the region's `centroid` LatentVec, then addLatent(kind: "centroid", ref_id: region_id).

// upsertDescents (batched, one entry group per row): child::OutE<DescendsFrom>::WHERE(parent_region_id, kind) updated, else AddE.
QUERY upsertDescents(child_region_id: String, parent_region_id: String, kind: String, overlap: F64) =>
    c <- N<LatentRegion>({region_id: child_region_id})
    p <- N<LatentRegion>({region_id: parent_region_id})
    e <- AddE<DescendsFrom>({child_region_id: child_region_id, parent_region_id: parent_region_id, kind: kind, overlap: overlap})::From(c)::To(p)
    RETURN e

// upsertHierEdges (batched): fine::OutE<Hier>::WHERE(coarse_region_id) updated, else AddE (fine → coarse).
QUERY upsertHierEdges(fine_region_id: String, coarse_region_id: String, p_coarse_given_fine: F64, p_fine_given_coarse: F64, support: F64, pmi_bits: F64, mi_bits: F64) =>
    f <- N<LatentRegion>({region_id: fine_region_id})
    c <- N<LatentRegion>({region_id: coarse_region_id})
    e <- AddE<Hier>({weight: p_coarse_given_fine, fine_region_id: fine_region_id, coarse_region_id: coarse_region_id, p_coarse_given_fine: p_coarse_given_fine, p_fine_given_coarse: p_fine_given_coarse, support: support, pmi_bits: pmi_bits, mi_bits: mi_bits})::From(f)::To(c)
    RETURN e

// upsertRegionSet / upsertKernelModel: get-or-create by region_set_id / kernel_id (= expert_space_id),
// linked ExpertSpace -HasRegionSet-> RegionSet / -HasKernel-> KernelModel on create.
QUERY regionSet(region_set_id: String) =>
    r <- N<RegionSet>({region_set_id: region_set_id})
    RETURN r

QUERY snapshotKernels(snapshot_id: String) =>
    k <- N<KernelModel>({snapshot_id: snapshot_id})
    RETURN k

QUERY snapshotHierarchy(snapshot_id: String) =>
    regions <- N<Snapshot>({snapshot_id: snapshot_id})::Out<HasSpace>::In<InSpace>
    edges <- regions::OutE<Hier>
    RETURN edges

QUERY lineageMembers(lineage_id: String) =>
    r <- N<LatentRegion>({lineage_id: lineage_id})
    RETURN r

// regionLineage(region_id, depth): depth-unrolled walk over OutE<DescendsFrom> + InE<RegionMatch>
// (ancestors) and InE<DescendsFrom> + OutE<RegionMatch> (descendants).

QUERY crossPriors(timescale: String) =>
    c <- N<CrossPrior>({timescale: timescale})::ORDER<Desc>(_::{score})
    RETURN c

// ── Latent vectors (centroids, observations, retrieval windows) ─────────
QUERY addLatent(vector: [F64], expert_space_id: String, kind: String, ref_id: String, instrument_id: String, uncertainty: F64) =>
    v <- AddV<LatentVec>(vector, {expert_space_id: expert_space_id, kind: kind, ref_id: ref_id, instrument_id: instrument_id, uncertainty: uncertainty})
    RETURN v

// Retrieval decoders (spec 34): nearest analogs; caller filters by expert_space_id and time distance.
QUERY searchLatent(vector: [F64], k: I64) =>
    hits <- SearchV<LatentVec>(vector, k)
    RETURN hits

// Spec 20 §4 knn(space, v, k, filter): vector length must equal ExpertSpace.dim; the search runs in
// the space's tenant partition of LatentVec.vec<dim>, filtered after the (over-fetched) top-k.
QUERY knn(expert_space_id: String, vector: [F64], k: I64, kind: String, instrument_id: String, as_of: Date) =>
    space <- N<ExpertSpace>({expert_space_id: expert_space_id}) // dim check
    hits <- SearchV<LatentVec>(vector, k)::WHERE(AND(_::{kind}::EQ(kind), _::{instrument_id}::EQ(instrument_id), _::{ts}::LTE(as_of)))
    RETURN hits

// Retention (spec 20 §6.5): fold observation latents older than `cutoff` into one kind=aggregate
// LatentVec per (space, ref_id) (obs_count, mean vector, mean uncertainty) and drop them.
QUERY pruneSpaceObservations(expert_space_id: String, cutoff: Date) =>
    old <- N<LatentVec>::WHERE(AND(_::{expert_space_id}::EQ(expert_space_id), _::{kind}::EQ("observation"), _::{ts}::LT(cutoff)))
    agg <- N<LatentVec>::WHERE(AND(_::{expert_space_id}::EQ(expert_space_id), _::{kind}::EQ("aggregate")))
    // per ref_id: agg::UPDATE({obs_count, vec<dim>, uncertainty, ts}) or AddN<LatentVec>({kind: "aggregate", ...})
    DROP old
    RETURN agg

// ── As-of reads (spec 20 §6.3: snapshot id and created_at <= t) ─────────
QUERY listSnapshotsAsOf(as_of: Date) =>
    snaps <- N<Snapshot>::WHERE(_::{created_at}::LTE(as_of))
    RETURN snaps

QUERY spaceRegionsAsOf(space: ID, as_of: Date) =>
    regions <- N<ExpertSpace>(space)::In<InSpace>::WHERE(_::{created_at}::LTE(as_of))
    RETURN regions

QUERY snapshotGraphAsOf(snapshot_id: String, as_of: Date) =>
    snap <- N<Snapshot>({snapshot_id: snapshot_id})::WHERE(_::{created_at}::LTE(as_of))
    spaces <- snap::Out<HasSpace>
    regions <- spaces::In<InSpace>::WHERE(_::{created_at}::LTE(as_of))
    transitions <- regions::OutE<Transition> // kept when the target region is visible too
    RETURN snap, spaces, regions, transitions

QUERY spaceLatentsAsOf(expert_space_id: String, kind: String, as_of: Date, limit: I64) =>
    rows <- N<LatentVec>::WHERE(AND(_::{expert_space_id}::EQ(expert_space_id), _::{kind}::EQ(kind), _::{ts}::LTE(as_of)))::ORDER<Desc>(_::{ts})::RANGE(0, limit)
    RETURN rows

// ── Event-time versions (spec 20 §6.3; fa-hub `versions`) ───────────────
// As-of reads keep an edge / region unchanged when updated_at <= t, else take the newest version <= t.
QUERY transitionVersions(from_region_id: String, to_region_id: String, sampler_id: String, as_of: Date, limit: I64) =>
    rows <- N<TransitionVersion>::WHERE(AND(_::{edge_key}::EQ(from_region_id + "|" + to_region_id + "|" + sampler_id), _::{ts}::LTE(as_of)))::ORDER<Desc>(_::{ts})::RANGE(0, limit)
    RETURN rows

QUERY regionStatusVersions(region_id: String, as_of: Date, limit: I64) =>
    rows <- N<RegionStatusVersion>::WHERE(AND(_::{region_id}::EQ(region_id), _::{ts}::LTE(as_of)))::ORDER<Desc>(_::{ts})::RANGE(0, limit)
    RETURN rows

// Retention: per edge key / region id of the space, versions with ts < cutoff except the newest.
QUERY pruneSpaceVersions(space: ID, cutoff: Date) =>
    old <- N<TransitionVersion>::WHERE(AND(_::{edge_key}::EQ(key), _::{ts}::LT(cutoff)))::ORDER<Desc>(_::{ts})::RANGE(1, N) // per key
    DROP old
    RETURN old

// ── Text / media embeddings (NewsDoc.vec<dim>, MediaSegment.vec<dim>) ────
QUERY addNewsDocs(doc_id: String, source_id: String, ts: Date, vector: [F64]) =>
    d <- AddV<NewsDoc>(vector, {doc_id: doc_id, source_id: source_id, ts: ts})
    RETURN d

QUERY searchNewsDocs(vector: [F64], k: I64, as_of: Date) =>
    hits <- SearchV<NewsDoc>(vector, k)::WHERE(_::{ts}::LTE(as_of))
    RETURN hits

QUERY addMediaSegments(segment_id: String, media_id: String, ts: Date, vector: [F64]) =>
    m <- AddV<MediaSegment>(vector, {segment_id: segment_id, media_id: media_id, ts: ts})
    RETURN m

QUERY searchMediaSegments(vector: [F64], k: I64, as_of: Date) =>
    hits <- SearchV<MediaSegment>(vector, k)::WHERE(_::{ts}::LTE(as_of))
    RETURN hits

// ── Schema version marker (spec 20 §8) ──────────────────────────────────
QUERY schemaMarker() =>
    m <- N<SchemaMeta>({meta_id: "fa-hub"})
    RETURN m

// Never lowers the stored version (UPDATE only WHERE version <= new version).
QUERY stampSchemaVersion(version: U32, fingerprint: String) =>
    existing <- N<SchemaMeta>({meta_id: "fa-hub"})
    raised <- existing::WHERE(_::{version}::LTE(version))::UPDATE({version: version, index_fingerprint: fingerprint, updated_at: NOW}) // when existing is not empty
    created <- AddN<SchemaMeta>({meta_id: "fa-hub", version: version, index_fingerprint: fingerprint}) // when existing is empty
    RETURN existing

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

// Lifecycle mirror (spec 35 §5): update the status properties, creating the genome like
// upsertGenome when it is missing.
QUERY setGenomeStatus(genome_id: String, kind: String, family: String, params: String, strategy_id: String, status: String, reason: String, since_ms: I64) =>
    existing <- N<Genome>({genome_id: genome_id})
    updated <- existing::UPDATE({status: status, strategy_id: strategy_id, status_reason: reason, status_since_ms: since_ms, status_updated_at: NOW}) // when existing is not empty
    created <- AddN<Genome>({genome_id: genome_id, kind: kind, family: family, params: params, status: status, strategy_id: strategy_id, status_reason: reason, status_since_ms: since_ms}) // when existing is empty
    RETURN updated, created

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

// ── Adapters (spec 36 §5) ───────────────────────────────────────────────
QUERY addAdapterState(snapshot: ID, adapter_id: String, snapshot_id: String, timescale: String, version: U64, garage_key: String, drift: F64, max_drift: F64, frozen: Boolean, n_windows: U64, window_start: Date, window_end: Date) =>
    a <- AddN<AdapterState>({adapter_id: adapter_id, snapshot_id: snapshot_id, timescale: timescale, version: version, garage_key: garage_key, drift: drift, max_drift: max_drift, frozen: frozen, n_windows: n_windows, window_start: window_start, window_end: window_end})
    edge <- AddE<HasAdapter>::From(snapshot)::To(a)
    RETURN a

QUERY latestAdapter(snapshot_id: String) =>
    a <- N<AdapterState>::WHERE(_::{snapshot_id}::EQ(snapshot_id))::ORDER<Desc>(_::{version})::RANGE(0, 1)
    RETURN a

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

// ── Expert identity (spec 32 §5, §6.5; fa-hub `experts.rs`) ──

QUERY upsertExpert(expert_uid: String, snapshot_id: String, timescale: String, expert_id: U32, dim: U32, usage: F64, skill: F64, born_gen: U32, specialisation: String) =>
    existing <- N<Expert>({expert_uid: expert_uid})
    updated <- existing::UPDATE({usage: usage, skill: skill, dim: dim, specialisation: specialisation, updated_at: NOW}) // when existing is not empty
    created <- AddN<Expert>({expert_uid: expert_uid, snapshot_id: snapshot_id, timescale: timescale, expert_id: expert_id, dim: dim, usage: usage, skill: skill, born_gen: born_gen, specialisation: specialisation}) // when existing is empty
    snap <- N<Snapshot>({snapshot_id: snapshot_id})
    edge <- AddE<ExpertOf>::From(created)::To(snap) // when created
    RETURN updated, created

QUERY alignExpert(child_uid: String, parent_uid: String, cka: F64, kind: String) =>
    child <- N<Expert>({expert_uid: child_uid})
    parent <- N<Expert>({expert_uid: parent_uid})
    edge <- AddE<AlignedTo>({cka: cka, kind: kind})::From(child)::To(parent)
    RETURN edge

QUERY snapshotExperts(snapshot_id: String) =>
    experts <- N<Expert>({snapshot_id: snapshot_id})
    RETURN experts

QUERY expertParents(expert_uid: String) =>
    parents <- N<Expert>({expert_uid: expert_uid})::Out<AlignedTo>
    RETURN parents
