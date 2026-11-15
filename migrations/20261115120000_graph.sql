-- The model-state graph in Postgres (ADR 0010 second half, D-108): typed tables replacing the
-- HelixDB nodes/edges that model-runtime, the trainer and control-api read and write. Ids come
-- from one shared sequence so the u64 node ids of the fa-hub API stay globally unique across
-- labels, as they were in Helix. The genome/evolution side stays in Helix for now; the
-- TrainedOn/FeedbackFrom links live here keyed by business ids (no node-id coupling).
CREATE SCHEMA IF NOT EXISTS graph;
CREATE SEQUENCE IF NOT EXISTS graph.node_id;

-- ── reference mirror (spec 12 §5) ──────────────────────────────────────────
CREATE TABLE graph.ref_entity (
    id          bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    entity_id   text NOT NULL UNIQUE,
    lei         text NOT NULL DEFAULT '',
    name        text NOT NULL DEFAULT '',
    country     text NOT NULL DEFAULT '',
    sector      text NOT NULL DEFAULT '',
    mirrored_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE graph.ref_sector (
    id        bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    sector_id text NOT NULL UNIQUE,
    name      text NOT NULL DEFAULT ''
);
CREATE TABLE graph.ref_in_sector (
    id        bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    entity_id text NOT NULL,
    sector_id text NOT NULL,
    UNIQUE (entity_id, sector_id)
);
CREATE TABLE graph.ref_venue (
    id          bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    venue_id    text NOT NULL UNIQUE,
    code        text NOT NULL DEFAULT '',
    mic         text NOT NULL DEFAULT '',
    name        text NOT NULL DEFAULT '',
    kind        text NOT NULL DEFAULT '',
    timezone    text NOT NULL DEFAULT '',
    mirrored_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE graph.ref_instrument (
    id            bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    instrument_id text NOT NULL UNIQUE,
    asset_class   text NOT NULL DEFAULT '',
    kind          text NOT NULL DEFAULT '',
    name          text NOT NULL DEFAULT '',
    status        text NOT NULL DEFAULT 'active',
    base_ccy      text NOT NULL DEFAULT '',
    quote_ccy     text NOT NULL DEFAULT '',
    mirrored_at   timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE graph.ref_issued_by (
    id            bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    instrument_id text NOT NULL,
    entity_id     text NOT NULL,
    UNIQUE (instrument_id, entity_id)
);
CREATE TABLE graph.ref_listed_on (
    id            bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    instrument_id text NOT NULL,
    venue_id      text NOT NULL,
    venue_symbol  text NOT NULL DEFAULT '',
    source_id     text NOT NULL DEFAULT '',
    listing_key   text NOT NULL,
    UNIQUE (instrument_id, listing_key)
);
CREATE TABLE graph.ref_same_exposure (
    id               bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    instrument_id    text NOT NULL,
    to_instrument_id text NOT NULL,
    relation         text NOT NULL DEFAULT '',
    exposure_key     text NOT NULL,
    UNIQUE (instrument_id, exposure_key)
);

-- ── snapshots, spaces, adapters (specs 31/36) ──────────────────────────────
CREATE TABLE graph.snapshot (
    id              bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    snapshot_id     text NOT NULL UNIQUE,
    timescale       text NOT NULL,
    encoder_family  text NOT NULL,
    generation      int NOT NULL DEFAULT 0,
    garage_key      text NOT NULL DEFAULT '',
    status          text NOT NULL DEFAULT 'candidate',
    carried_mass    float8,
    manifest        text,
    previous_active text,
    created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX snapshot_status_idx ON graph.snapshot (timescale, encoder_family, status);
CREATE TABLE graph.snapshot_parent (
    id        bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    parent_id bigint NOT NULL REFERENCES graph.snapshot (id) ON DELETE CASCADE,
    child_id  bigint NOT NULL REFERENCES graph.snapshot (id) ON DELETE CASCADE,
    op        text NOT NULL DEFAULT '',
    UNIQUE (parent_id, child_id)
);
CREATE TABLE graph.adapter_state (
    id           bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    adapter_id   text NOT NULL UNIQUE,
    snapshot_id  text NOT NULL,
    timescale    text NOT NULL DEFAULT '',
    version      bigint NOT NULL,
    garage_key   text NOT NULL DEFAULT '',
    drift        float8,
    max_drift    float8 NOT NULL DEFAULT 0,
    frozen       boolean NOT NULL DEFAULT false,
    n_windows    bigint NOT NULL DEFAULT 0,
    window_start timestamptz,
    window_end   timestamptz,
    updated_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX adapter_snapshot_idx ON graph.adapter_state (snapshot_id, version DESC);
CREATE TABLE graph.expert_space (
    id              bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    expert_space_id text NOT NULL UNIQUE,
    snapshot_id     text NOT NULL,
    index           int NOT NULL DEFAULT 0,
    dim             int NOT NULL DEFAULT 0
);
CREATE INDEX expert_space_snapshot_idx ON graph.expert_space (snapshot_id);

-- ── regions and transitions (specs 30/33, D-048) ───────────────────────────
CREATE TABLE graph.region (
    id                bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    region_id         text NOT NULL UNIQUE,
    lineage_id        text NOT NULL DEFAULT '',
    expert_space_id   text NOT NULL,
    method            text NOT NULL DEFAULT '',
    radius            float8 NOT NULL DEFAULT 0,
    density           float8 NOT NULL DEFAULT 0,
    status            text,
    related_region_id text,
    retire_reason     text,
    mass              float8,
    occupancy         float8,
    spread            float8,
    age               bigint,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz,
    died_at           timestamptz
);
CREATE INDEX region_space_idx ON graph.region (expert_space_id);
CREATE INDEX region_lineage_idx ON graph.region (lineage_id);
CREATE TABLE graph.transition (
    id             bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    from_id        bigint NOT NULL REFERENCES graph.region (id) ON DELETE CASCADE,
    to_id          bigint NOT NULL REFERENCES graph.region (id) ON DELETE CASCADE,
    from_region_id text NOT NULL,
    to_region_id   text NOT NULL,
    sampler_id     text NOT NULL,
    count          bigint NOT NULL DEFAULT 0,
    mean           float8 NOT NULL DEFAULT 0,
    lower          float8 NOT NULL DEFAULT 0,
    upper          float8 NOT NULL DEFAULT 0,
    kernel_params  text NOT NULL DEFAULT '',
    updated_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (from_region_id, to_region_id, sampler_id)
);
CREATE INDEX transition_from_idx ON graph.transition (from_id);
CREATE TABLE graph.transition_version (
    id             bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    edge_key       text NOT NULL,
    from_region_id text NOT NULL,
    to_region_id   text NOT NULL,
    sampler_id     text NOT NULL,
    count          bigint NOT NULL DEFAULT 0,
    mean           float8 NOT NULL DEFAULT 0,
    lower          float8 NOT NULL DEFAULT 0,
    upper          float8 NOT NULL DEFAULT 0,
    kernel_params  text NOT NULL DEFAULT '',
    ts             timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX transition_version_key_idx ON graph.transition_version (edge_key, ts DESC);
CREATE TABLE graph.region_status_version (
    id                bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    region_id         text NOT NULL,
    status            text NOT NULL,
    related_region_id text NOT NULL DEFAULT '',
    reason            text NOT NULL DEFAULT '',
    ts                timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX region_status_version_idx ON graph.region_status_version (region_id, ts DESC);

-- ── lineage and hierarchy (spec 30 §5/§6.5) ────────────────────────────────
CREATE TABLE graph.descends_from (
    id               bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    child_region_id  text NOT NULL,
    parent_region_id text NOT NULL,
    kind             text NOT NULL,
    overlap          float8 NOT NULL DEFAULT 0,
    updated_at       timestamptz NOT NULL DEFAULT now(),
    UNIQUE (child_region_id, parent_region_id, kind)
);
CREATE INDEX descends_parent_idx ON graph.descends_from (parent_region_id);
CREATE TABLE graph.region_match (
    id             bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    prev_id        bigint NOT NULL,
    next_id        bigint NOT NULL,
    prev_region_id text NOT NULL DEFAULT '',
    next_region_id text NOT NULL DEFAULT '',
    score          float8 NOT NULL DEFAULT 0
);
CREATE INDEX region_match_prev_idx ON graph.region_match (prev_region_id);
CREATE INDEX region_match_next_idx ON graph.region_match (next_region_id);
CREATE TABLE graph.hier (
    id                  bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    fine_region_id      text NOT NULL,
    coarse_region_id    text NOT NULL,
    weight              float8 NOT NULL DEFAULT 0,
    p_coarse_given_fine float8 NOT NULL DEFAULT 0,
    p_fine_given_coarse float8 NOT NULL DEFAULT 0,
    support             float8 NOT NULL DEFAULT 0,
    pmi_bits            float8 NOT NULL DEFAULT 0,
    mi_bits             float8 NOT NULL DEFAULT 0,
    updated_at          timestamptz NOT NULL DEFAULT now(),
    UNIQUE (fine_region_id, coarse_region_id)
);

-- ── per-space state (specs 30 §5, 33 §5/§6.5) ──────────────────────────────
CREATE TABLE graph.region_set (
    id            bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    region_set_id text NOT NULL UNIQUE,
    snapshot_id   text NOT NULL DEFAULT '',
    timescale     text NOT NULL DEFAULT '',
    expert_index  int NOT NULL DEFAULT 0,
    family        text NOT NULL DEFAULT '',
    method        text NOT NULL DEFAULT '',
    n_regions     int NOT NULL DEFAULT 0,
    cfg_hash      text NOT NULL DEFAULT '',
    state         text NOT NULL DEFAULT '',
    kernel_state  text NOT NULL DEFAULT '',
    fit_time      timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE graph.kernel_model (
    id               bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    kernel_id        text NOT NULL UNIQUE,
    expert_space_id  text NOT NULL DEFAULT '',
    snapshot_id      text NOT NULL DEFAULT '',
    timescale        text NOT NULL DEFAULT '',
    expert_index     int NOT NULL DEFAULT 0,
    kernel_type      text NOT NULL DEFAULT '',
    serving_sampler  text NOT NULL DEFAULT '',
    scores           text NOT NULL DEFAULT '',
    stationary       text NOT NULL DEFAULT '',
    relaxation_steps float8 NOT NULL DEFAULT 0,
    drift_kl         float8 NOT NULL DEFAULT 0,
    n_rows           int NOT NULL DEFAULT 0,
    observation_mass float8 NOT NULL DEFAULT 0,
    live_regions     int NOT NULL DEFAULT 0,
    updated_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX kernel_snapshot_idx ON graph.kernel_model (snapshot_id);
CREATE TABLE graph.cross_prior (
    id                   bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    pair_id              text NOT NULL UNIQUE,
    timescale            text NOT NULL,
    source_instrument_id text NOT NULL,
    target_instrument_id text NOT NULL,
    n                    int NOT NULL DEFAULT 0,
    correlation          float8 NOT NULL DEFAULT 0,
    shrinkage            float8 NOT NULL DEFAULT 0,
    lead_lag             int NOT NULL DEFAULT 0,
    lead_corr            float8 NOT NULL DEFAULT 0,
    predictive_gain      float8 NOT NULL DEFAULT 0,
    uncertainty          float8 NOT NULL DEFAULT 0,
    score                float8 NOT NULL DEFAULT 0,
    rank                 int NOT NULL DEFAULT 0,
    updated_at           timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX cross_prior_ts_idx ON graph.cross_prior (timescale, score DESC);
CREATE TABLE graph.region_episode (
    id              bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    episode_id      text NOT NULL UNIQUE,
    region_id       text NOT NULL,
    expert_space_id text NOT NULL DEFAULT '',
    timescale       text NOT NULL DEFAULT '',
    instrument_id   text NOT NULL DEFAULT '',
    t_enter         timestamptz NOT NULL,
    t_exit          timestamptz NOT NULL,
    bars            int NOT NULL DEFAULT 0,
    max_weight      float8 NOT NULL DEFAULT 0,
    mean_weight     float8 NOT NULL DEFAULT 0
);
CREATE INDEX episode_region_idx ON graph.region_episode (region_id, t_exit DESC);
CREATE INDEX episode_ts_idx ON graph.region_episode (timescale, t_exit);

-- ── genome links by business id (genomes themselves stay in Helix for now) ─
CREATE TABLE graph.trained_on (
    id          bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    genome_id   text NOT NULL,
    snapshot_id text NOT NULL,
    generation  int NOT NULL DEFAULT 0,
    UNIQUE (genome_id, snapshot_id, generation)
);
CREATE TABLE graph.feedback_from (
    id            bigint PRIMARY KEY DEFAULT nextval('graph.node_id'),
    snapshot_id   text NOT NULL,
    genome_id     text NOT NULL,
    weight        float8 NOT NULL DEFAULT 0,
    output_family text NOT NULL DEFAULT '',
    score         float8 NOT NULL DEFAULT 0,
    UNIQUE (snapshot_id, genome_id)
);
