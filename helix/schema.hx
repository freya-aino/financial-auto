// HelixDB schema catalog (spec 20 §5.2). Human-readable source of truth for the data model;
// the standalone server (D-033) is schemaless and infers labels/properties on write, so this
// file is NOT loaded by HelixDB. Its executable mirror is crates/fa-hub/src/schema.rs:
//   - every UNIQUE INDEX / INDEX line  -> node_equality index in `schema::static_indexes()`
//   - every V:: type                  -> a `node_vector` index per dimension (`vec<dim>` property),
//                                        partitioned by `expert_space_id` (tenant_property)
//   - `Date DEFAULT NOW`              -> `date_time_now` expression on insert
// Keep both in sync; run `cargo test -p fa-hub --test live` against a running server.
//
// Mixed-dimension latents: one `LatentVec` label, vector stored in property `vec<dim>` with one
// index per dim, each partitioned per expert space. (Resolves the earlier open question.)

// ── Reference graph (mirrored from Postgres `ref`, spec 12 §5) ─────────
// Written by control-api `ref_mirror.rs` through fa-hub `reference.rs`: full sync at start and
// every FA_REF_MIRROR_FULL_SECS, incremental on `ref.changed.*`. Ids are the Postgres UUIDs;
// empty strings stand for NULL. Edges carry the key they are upserted by.
N::Instrument {
    UNIQUE INDEX instrument_id: String,
    asset_class: String,
    kind: String,
    name: String,
    status: String,      // active | provisional | merged | … ; "inactive" = no longer current in Postgres
    base_ccy: String,
    quote_ccy: String,
    mirrored_at: Date,
}

N::Entity {
    UNIQUE INDEX entity_id: String,
    lei: String,
    name: String,
    country: String,
    sector: String,
    mirrored_at: Date,
}

N::Venue {
    UNIQUE INDEX venue_id: String,
    code: String,        // ref.venue.code, the subject token
    mic: String,
    name: String,
    kind: String,        // exchange | broker | dex | otc
    timezone: String,
    mirrored_at: Date,
}

E::IssuedBy {
    From: Instrument,
    To: Entity,
    Properties: {
        instrument_id: String,
        entity_id: String,   // upsert key
        mirrored_at: Date,
    }
}

E::ListedOn {
    From: Instrument,
    To: Venue,
    Properties: {
        listing_key: String, // "<venue_id>|<venue_symbol>", upsert key
        instrument_id: String,
        venue_id: String,
        venue_symbol: String,
        source_id: String,
        mirrored_at: Date,
    }
}

// From ref.instrument_relation rows of kind same_economic_exposure (directional as stored).
E::SameExposure {
    From: Instrument,
    To: Instrument,
    Properties: {
        exposure_key: String,  // "<to_instrument_id>|<relation>", upsert key
        instrument_id: String,
        to_instrument_id: String,
        relation: String,      // "<from kind>-<to kind>", e.g. "crypto_spot-perp"
        mirrored_at: Date,
    }
}

// ── Encoder snapshots and expert spaces (specs 31, 32, 36) ──────────────
N::Snapshot {
    UNIQUE INDEX snapshot_id: String,
    timescale: String,
    encoder_family: String,
    generation: U32,
    garage_key: String,
    status: String, // candidate | active | retired
    created_at: Date DEFAULT NOW,
    // Written by the trainer for the spec 36 §6.4 promotion checklist (control-api reads them
    // when the Garage artifact is unavailable); absent on baseline rows.
    carried_mass: F64,
    manifest: String, // JSON `fa_encoders::SnapshotManifest`
    // Set by control-api on promote: snapshot_id of the active snapshot this one replaced;
    // rollback restores exactly that one (D-065).
    previous_active: String,
}

// Online-adapter versions of a snapshot (spec 36 §5 / §6.5): one node per adapter artifact
// (`fa-models/adapters/<snapshot>/<version>.json`), written by the trainer's `adapter` job.
N::AdapterState {
    UNIQUE INDEX adapter_id: String, // `<snapshot_id>@<version>`
    INDEX snapshot_id: String,
    timescale: String,
    version: U64,    // unix ms of the adapter fit (monotonic per snapshot)
    garage_key: String,
    drift: F64,      // mean |Δmean|/σ vs the snapshot weights
    max_drift: F64,  // guard threshold the job applied (manifest.params.adapter_guard)
    frozen: Boolean, // drift > max_drift
    n_windows: U64,
    window_start: Date,
    window_end: Date,
    updated_at: Date DEFAULT NOW,
}

E::HasAdapter {
    From: Snapshot,
    To: AdapterState,
    Properties: {}
}

N::ExpertSpace {
    UNIQUE INDEX expert_space_id: String,
    snapshot_id: String,
    index: U32,
    dim: U32,
}

E::HasSpace {
    From: Snapshot,
    To: ExpertSpace,
    Properties: {}
}

// One surviving expert of a learned (neural) snapshot (spec 32 §5, §6.5); written by the
// trainer after CKA matching. `expert_id` is the stable expert index, so `AlignedTo` chains are
// an expert's identity across generations.
N::Expert {
    UNIQUE INDEX expert_uid: String, // `<snapshot_id>/e<expert_id>`
    INDEX snapshot_id: String,
    timescale: String,
    expert_id: U32,
    dim: U32,
    usage: F64,          // validation router usage
    skill: F64,          // validation skill vs naive (spec 32 §6.4)
    specialisation: String, // JSON: gate mass per instrument
    born_gen: U32,
    died_gen: U32,       // set when no child expert continues it
    updated_at: Date DEFAULT NOW,
}

E::ExpertOf {
    From: Expert,
    To: Snapshot,
    Properties: {}
}

E::AlignedTo {
    From: Expert, // child (new generation)
    To: Expert,   // parent-generation expert it continues
    Properties: {
        cka: F64,
        kind: String, // carry
    }
}

E::SnapshotParent {
    From: Snapshot, // parent
    To: Snapshot,   // child (spec 36 lineage; written by the trainer)
    Properties: {
        op: String, // pretrain | generation | adapter
    }
}

// ── Latent regions and the Markov state graph (specs 30, 33) ────────────
N::LatentRegion {
    UNIQUE INDEX region_id: String,
    lineage_id: String, // stable across generations (D-007)
    expert_space_id: String,
    method: String, // codebook | density | mapper | ...
    radius: F64,
    density: F64,
    status: String, // live | merged | retired (model-runtime lifecycle, spec 30 §4)
    related_region_id: String, // merged: absorbed region; retired: survivor it merged into
    created_at: Date DEFAULT NOW,
    updated_at: Date,
    // Online statistics (model-runtime, fa-hub `update_region_stats`) and death marks.
    mass: F64,        // points assigned since birth
    occupancy: F64,   // decayed occupancy share in its set
    spread: F64,      // RMS distance of members to the centroid
    age: I64,         // observations of the set since birth
    died_at: Date,
    retire_reason: String, // merged | support | unmatched
}

E::InSpace {
    From: LatentRegion,
    To: ExpertSpace,
    Properties: {}
}

E::Transition {
    From: LatentRegion,
    To: LatentRegion,
    Properties: {
        from_region_id: String, // denormalised business ids so the edge can be upserted
        to_region_id: String,   // (lookup: From::OutE<Transition>::WHERE(to_region_id, sampler_id))
        sampler_id: String,
        count: I64,
        mean: F64,
        lower: F64,
        upper: F64,
        kernel_params: String, // JSON
        updated_at: Date,
    }
}

// Event-time history of values updated in place (spec 20 §6.3, fa-hub `versions`): one row per
// Transition write (same batch as the edge upsert) and per region lifecycle mark. As-of reads use
// the latest version with ts <= t; retention keeps everything newer than the observation horizon
// plus the newest older row per edge / region.
N::TransitionVersion {
    INDEX edge_key: String, // from_region_id|to_region_id|sampler_id
    from_region_id: String,
    to_region_id: String,
    sampler_id: String,
    count: I64,
    mean: F64,
    lower: F64,
    upper: F64,
    kernel_params: String, // JSON
    ts: Date DEFAULT NOW, // record time of the write
}

N::RegionStatusVersion {
    INDEX region_id: String,
    status: String, // live | merged | retired
    related_region_id: String,
    reason: String, // created | revived | merged | support | unmatched | ""
    ts: Date DEFAULT NOW,
}

E::Hier {
    From: LatentRegion, // fine timescale (spec 30 §6.5 REFINES)
    To: LatentRegion,   // coarse timescale, same expert index
    Properties: {
        weight: F64, // = p_coarse_given_fine
        fine_region_id: String,
        coarse_region_id: String,
        p_coarse_given_fine: F64,
        p_fine_given_coarse: F64,
        support: F64,
        pmi_bits: F64,
        mi_bits: F64, // mutual information of the whole set pair
        updated_at: Date,
    }
}

E::RegionMatch {
    From: LatentRegion,
    To: LatentRegion,
    Properties: {
        score: F64, // probe-set alignment score (D-007)
    }
}

// Region lineage (spec 30 §5 DESCENDS_FROM): child → parent, written by model-runtime for
// online splits/merges and for the alignment of a loaded snapshot's region_map.
E::DescendsFrom {
    From: LatentRegion, // child
    To: LatentRegion,   // parent
    Properties: {
        child_region_id: String,
        parent_region_id: String,
        kind: String, // carry | split | merge | partial
        overlap: F64, // probe Jaccard overlap (snapshot) or mass share (online split)
        updated_at: Date,
    }
}

// Persisted online region set + conjugate kernel of one expert space (spec 30 §5 RegionSet;
// region_set_id = expert_space_id). model-runtime restores both on restart.
N::RegionSet {
    UNIQUE INDEX region_set_id: String,
    INDEX snapshot_id: String,
    timescale: String,
    expert_index: U32,
    family: String,
    method: String,
    n_regions: U32,
    cfg_hash: String,
    state: String,        // JSON fa_patterns::OnlineRegionSet
    kernel_state: String, // JSON fa_mcmc::MarkovGraph
    fit_time: Date,
    updated_at: Date,
}

E::HasRegionSet {
    From: ExpertSpace,
    To: RegionSet,
    Properties: {}
}

// Kernel model of one expert space (spec 33 §5; kernel_id = expert_space_id).
N::KernelModel {
    UNIQUE INDEX kernel_id: String,
    INDEX snapshot_id: String,
    expert_space_id: String,
    timescale: String,
    expert_index: U32,
    kernel_type: String,     // dirichlet (geometric prior, forgetting)
    serving_sampler: String, // conjugate | smc
    scores: String,          // JSON sampler standings: log score, Brier, ECE, PIT KS, cost
    stationary: String,      // JSON [{region_id, mean, lower, upper, sd}]
    relaxation_steps: F64,
    drift_kl: F64,
    n_rows: U32,
    observation_mass: F64,
    live_regions: U32,
    updated_at: Date,
}

E::HasKernel {
    From: ExpertSpace,
    To: KernelModel,
    Properties: {}
}

// Cross-asset prior of one directed instrument pair at one timescale (spec 33 §6.5).
N::CrossPrior {
    UNIQUE INDEX pair_id: String, // <timescale>:<source>:<target>
    INDEX timescale: String,
    source_instrument_id: String,
    target_instrument_id: String,
    n: U32,
    correlation: F64, // Ledoit–Wolf shrunk
    shrinkage: F64,
    lead_lag: U32,
    lead_corr: F64,
    predictive_gain: F64, // Δ CRPS
    uncertainty: F64,
    score: F64,
    rank: U32,
    updated_at: Date,
}

// Latent vectors: region centroids, sampled state observations, retrieval windows.
V::LatentVec {
    INDEX expert_space_id: String,
    kind: String, // centroid | observation | window
    ref_id: String, // region_id / observation id / window id
    instrument_id: String,
    ts: Date,
    uncertainty: F64,
}

N::Window {
    UNIQUE INDEX window_id: String,
    timescale: String,
    instrument_set: String, // JSON array of instrument ids
    start_ts: Date,
    end_ts: Date,
}

E::Contains {
    From: LatentRegion,
    To: Window,
    Properties: {}
}

// ── Evolution: genomes, lineage, fitness (specs 34, 35, 36) ─────────────
N::Genome {
    UNIQUE INDEX genome_id: String,
    kind: String, // encoder | decoder | strategy | sampler | allocator
    family: String,
    params: String, // JSON
    created_at: Date DEFAULT NOW,
    // Strategies only (spec 35 §5): lifecycle status mirrored from ledger.strategy_state by the
    // risk-engine pipeline (setGenomeStatus); absent on genomes that never reached the lifecycle.
    status: String, // candidate | holdout_passed | shadow | active | paused | flattening | retired
    strategy_id: String,
    status_reason: String,
    status_since_ms: I64,
    status_updated_at: Date,
}

N::FitnessRecord {
    genome_id: String,
    generation: U32,
    objectives: String, // JSON: reconstruction, forecast, trading, cost
    pareto_rank: U32,
}

E::ParentOf {
    From: Genome,
    To: Genome,
    Properties: {
        op: String, // mutate | crossover | finetune
    }
}

E::Scored {
    From: Genome,
    To: FitnessRecord,
    Properties: {}
}

// Decoder genome → encoder snapshot whose latents it was trained/evaluated on (evo-worker,
// `evo.vary.decoder` parents of a generation run on Job.snapshot_id). Upsert key: snapshot_id.
E::TrainedOn {
    From: Genome,
    To: Snapshot,
    Properties: {
        genome_id: String,
        snapshot_id: String,
        generation: U32,
        updated_at: Date,
    }
}

// Encoder snapshot → elite decoder genome fed back into its fit (D-009; trainer, from
// manifest.feedback). Upsert key: genome_id.
E::FeedbackFrom {
    From: Snapshot,
    To: Genome,
    Properties: {
        snapshot_id: String,
        genome_id: String,
        weight: F64,         // quota weight of the elite in the feedback set (D-068)
        output_family: String,
        score: F64,
        updated_at: Date,
    }
}

// ── Text and media (specs 10, 31) ───────────────────────────────────────
// Embedding in property `vec<dim>` like LatentVec, one cosine index per (label, dim), not
// tenant-partitioned (fa-hub `schema::embedding_index`, created on first write).
V::NewsDoc {
    INDEX doc_id: String,
    ts: Date,
    source_id: String,
}

V::MediaSegment {
    INDEX segment_id: String,
    ts: Date,
    media_id: String,
}

E::Mentions {
    From: NewsDoc,
    To: Instrument,
    Properties: {
        confidence: F64,
    }
}

// ── Schema / migration version marker (spec 20 §8, §10 Q6) ──────────────
// One row per meta_id ("fa-hub"): the data-model version the last `ensure_schema_ready` migrated
// to and a fingerprint of the static index set. A build refuses to run against a newer version.
N::SchemaMeta {
    UNIQUE INDEX meta_id: String,
    version: U32,
    index_fingerprint: String,
    writer: String, // crate version of the build that stamped it
    updated_at: Date DEFAULT NOW,
}
