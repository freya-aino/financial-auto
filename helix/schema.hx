// HelixDB schema (spec 20 §5.2). UNVALIDATED: written from the HelixQL docs without the
// `helix` CLI available. Run `helix check` once the CLI is installed and fix what it reports.
//
// Vector types are static in HelixQL, but expert spaces are created at runtime with varying
// dims. v1 stores all latents in one `LatentVec` type and filters by `expert_space_id`.
// Revisit with the Helix scale benchmark (spec 20 open question 2).

// ── Reference graph (mirrored from Postgres `ref`, spec 12) ─────────────
N::Instrument {
    UNIQUE INDEX instrument_id: String,
    asset_class: String,
    kind: String,
    name: String,
}

N::Entity {
    UNIQUE INDEX entity_id: String,
    lei: String,
    name: String,
}

N::Venue {
    UNIQUE INDEX venue_id: String,
    mic: String,
    name: String,
}

E::IssuedBy {
    From: Instrument,
    To: Entity,
    Properties: {}
}

E::ListedOn {
    From: Instrument,
    To: Venue,
    Properties: {
        venue_symbol: String,
    }
}

E::SameExposure {
    From: Instrument,
    To: Instrument,
    Properties: {
        relation: String, // e.g. "spot-perp"
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

// ── Latent regions and the Markov state graph (specs 30, 33) ────────────
N::LatentRegion {
    UNIQUE INDEX region_id: String,
    lineage_id: String, // stable across generations (D-007)
    expert_space_id: String,
    method: String, // codebook | density | mapper | ...
    radius: F64,
    density: F64,
    created_at: Date DEFAULT NOW,
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
        sampler_id: String,
        count: I64,
        mean: F64,
        lower: F64,
        upper: F64,
        kernel_params: String, // JSON
        updated_at: Date,
    }
}

E::Hier {
    From: LatentRegion,
    To: LatentRegion,
    Properties: {
        weight: F64, // timescale a → timescale b containment
    }
}

E::RegionMatch {
    From: LatentRegion,
    To: LatentRegion,
    Properties: {
        score: F64, // probe-set alignment score (D-007)
    }
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

E::TrainedOn {
    From: Genome,
    To: Snapshot,
    Properties: {}
}

E::FeedbackFrom {
    From: Snapshot,
    To: Genome,
    Properties: {
        weight: F64, // elite decoder feedback into the next encoder generation (D-009)
    }
}

// ── Text and media (specs 10, 31) ───────────────────────────────────────
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
