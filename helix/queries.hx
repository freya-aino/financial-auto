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

QUERY spaceRegions(space: ID) =>
    regions <- N<ExpertSpace>(space)::In<InSpace>
    RETURN regions

QUERY addTransition(from: ID, to: ID, sampler_id: String, count: I64, mean: F64, lower: F64, upper: F64, kernel_params: String) =>
    t <- AddE<Transition>({sampler_id: sampler_id, count: count, mean: mean, lower: lower, upper: upper, kernel_params: kernel_params})::From(from)::To(to)
    RETURN t

QUERY regionTransitions(region: ID) =>
    edges <- N<LatentRegion>(region)::OutE<Transition>
    RETURN edges

QUERY regionNeighbors(region: ID) =>
    next <- N<LatentRegion>(region)::Out<Transition>
    RETURN next

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

QUERY genomeParents(genome: ID) =>
    parents <- N<Genome>(genome)::In<ParentOf>
    RETURN parents

QUERY genomeFitness(genome: ID) =>
    recs <- N<Genome>(genome)::Out<Scored>
    RETURN recs
