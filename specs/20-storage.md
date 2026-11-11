# 20 — Storage

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium |
| **Owners (code)** | `crates/fa-hub` (HelixDB), `crates/fa-tsdb` (VictoriaMetrics), `crates/fa-warehouse` (Postgres), `crates/fa-blob` (Garage S3), `helix/`, `migrations/`, `compose.yaml` |
| **Depends on** | 10-ingestion, 12-reference-data, 21-bus-and-contracts, 30–36 (Helix contents), 42-ledger-and-accounting |

## 1. Purpose
Define which store holds what, how data is keyed and queried point-in-time, and how long it is kept. Four stores, each with one job: **HelixDB** (graph + vectors + lineage), **VictoriaMetrics** (time series), **PostgreSQL** (static records, warehouse, ledger), **Garage S3** (blobs). NATS JetStream is transport with bounded retention, not a store of record.

## 2. Scope
**In scope:** store responsibilities, Helix schema sketch, VictoriaMetrics modeling rules, Postgres schemas overview, Garage bucket layout, as-of query rules, retention and disk budget, backup hooks.
**Out of scope:** detailed ledger schema (42), reference tables (12), model internals (31–36).

## 3. Decisions
- **HelixDB** is the central hub for graph + vectors + lineage: instruments graph, latent regions, transitions, embeddings, genomes, lineage, snapshot metadata (ADR 0002).
- **VictoriaMetrics** for trades/bars/features/system metrics; ms timestamps, float/label model (ADR 0007).
- **PostgreSQL** warehouse for static records: reference data, fundamentals, filings metadata, macro releases with vintages, source catalog, orders/fills/positions ledger, audit log (ADR 0008).
- **Garage S3** for raw payload archive, media, model weights/snapshots, checkpoints (ADR 0009).
- Helix and VM hold **references** (Garage keys) and embeddings, never large blobs.
- Retention defined **per timescale**; as-of correctness is required for every query used by models or backtests.

## 4. Interfaces
Rust crates, each a thin typed wrapper:
```rust
// fa-hub
trait Hub {
    async fn upsert_region(&self, r: &LatentRegion) -> Result<RegionId>;
    async fn upsert_transition(&self, t: &Transition) -> Result<()>;
    async fn knn(&self, space: ExpertSpaceKey, v: &[f32], k: usize, filter: KnnFilter) -> Result<Vec<Hit>>;
    async fn lineage(&self, genome: GenomeId, depth: usize) -> Result<LineageGraph>;
    // + generated wrappers for every query in helix/queries.hx
}
// fa-tsdb
trait Tsdb {
    async fn write(&self, batch: &[Sample]) -> Result<()>;
    async fn range(&self, q: SeriesQuery, from: Ts, to: Ts, as_of: Option<Ts>) -> Result<Frame>;
}
// fa-warehouse: sqlx pools per schema; fa-blob: S3 get/put/list with content hashing
```
**Implemented as:** concrete clients, not traits — `fa_hub::HubClient` (`upsert_region`, `upsert_transitions`, `knn(space, v, k, &KnnFilter)`, `genome_ancestry`, one typed function per `queries.hx` query), `fa_tsdb::Tsdb` (`write`, `query_range`, `trade_prices`, `bars`; no `as_of` parameter, see §6.3), `fa_blob::BlobStore` (trait with `MemoryStore` / `S3BlobStore`; keys are ULID/id based rather than content-addressed; the payload SHA-256 is only the SigV4 body hash, and integrity at rest comes from the backup manifest).
- Go: VictoriaMetrics written by the ingestor via the JSON-line import API (`go/pkg/tsdb`, `/api/v1/import`); Postgres via `pgx`; Garage via S3 SDK (`go/pkg/archive`).
- HelixQL queries live in `helix/queries.hx`; schema in `helix/schema.hx`.

## 5. Data model

### 5.1 Responsibility matrix
| Data | Store | Key |
|---|---|---|
| Trades, quotes (L1), bars, book top-N, derived features | VictoriaMetrics | metric + labels (§6.2) |
| System/ops metrics | VictoriaMetrics | standard |
| Instruments/venues/identifiers/calendars/corp actions | Postgres `ref` (mirrored to Helix) | UUID |
| Fundamentals, macro vintages, filings metadata | Postgres `wh` | (entity, field, period, vintage) |
| Source catalog, connector cursors, quarantine | Postgres `catalog`, `ingest` | id |
| Ledger: intents/orders/fills/positions/cash | Postgres `ledger` (42) | ULID |
| Audit log (every control action, promotion, kill) | Postgres `audit` | ULID, append-only |
| Latent regions, transitions, embeddings, expert-space registry | Helix | snapshot-scoped ids |
| Genomes (decoders, strategies, encoders, samplers, allocators), lineage, fitness | Helix (+ metrics in VM) | GenomeId |
| Snapshot metadata (encoder generations, adapters) | Helix | SnapshotId |
| Raw payloads, media, weights, checkpoints, rendered decoder outputs | Garage | id-based key (§5.5) |

### 5.2 Helix schema sketch (original proposal; the current model is `helix/schema.hx`, see "Implemented today" below)
Nodes:
- `Instrument {id, asset_class, kind}` · `Entity {id, lei}` · `Venue {id}`
- `Snapshot {id, timescale, encoder_family, generation, created_at, garage_key, status}`
- `ExpertSpace {id, snapshot_id, index, dim}`
- `LatentRegion {id, expert_space_id, method, centroid V, radius/density, created_at}`
- `StateObservation {id, ts, instrument_id?, region_id, latent V, uncertainty}` (sampled, not every tick — see §6.4)
- `Window {id, instrument_set, timescale, start, end, latent V}` — retrieval-decoder targets (34)
- `Genome {id, kind: encoder|decoder|strategy|sampler|allocator, family, params json, created_at}`
- `FitnessRecord {genome_id, generation, objectives json, pareto_rank}`
- `NewsDoc {id, ts, embedding V}` · `MediaSegment {id, ts, embedding V}`

Edges:
- `ISSUED_BY`, `LISTED_ON`, `SAME_EXPOSURE` (from 12)
- `TRANSITION {region_from → region_to, kernel_params json, count, posterior_summary}` (33)
- `PARENT_OF {genome → genome, op: mutate|crossover|finetune}` (lineage)
- `TRAINED_ON / FEEDBACK_FROM {snapshot ↔ genome}` (36)
- `HIER {region(timescale a) → region(timescale b)}` (30)
- `CONTAINS {region → window}`, `MENTIONS {news → instrument}`
- `REGION_MATCH {region(snapshot n) → region(snapshot n+1), score}` (region identity across retraining — open, 30)

Vectors: indexed per `ExpertSpace` (one vector index per space/dim), plus text/media embedding indexes.

**Implemented today (D-043, D-048; `crates/fa-hub`):** the standalone server is schemaless, so the sketch above is realised as labels + indexes. Every `UNIQUE INDEX`/`INDEX` of `helix/schema.hx` is a `node_equality` index created idempotently by `ensure_schema` (a unit test parses `schema.hx` and fails on any index line without one). Vectors: one label per vector type (`LatentVec`, `NewsDoc`, `MediaSegment`), a `d`-dimensional vector stored in property `vec<d>`, one cosine index per `(label, d)` created on first use and remembered per client (no DDL per write); `LatentVec` indexes are tenant-partitioned by `expert_space_id`, so a search is always scoped to one expert space. The registered `ExpertSpace.dim` is authoritative: `add_latents_checked` and `knn(space, v, k, KnnFilter{kind, instrument_id, as_of_ms, since_ms, exclude_ref_ids, max_distance})` reject other lengths (`DimMismatch`), and empty / zero-norm / non-finite vectors are rejected client-side (`InvalidVector`). Every `QUERY` of `helix/queries.hx` has a same-named typed function (enforced by `tests/coverage.rs`). A `SchemaMeta {meta_id = "fa-hub", version, index_fingerprint}` node records the data-model version (§8, Q6). Departures from the sketch: state observations, window vectors, region centroids and aggregates are all `LatentVec` rows (`kind = observation|aggregate|centroid`); regions are additionally linked by `DescendsFrom{carry|split|merge|partial}` (region identity across snapshots, D-069) and per expert space carry a `RegionSet` (serialised online region set + kernel) and a `KernelModel`; cross-asset priors are `CrossPrior` nodes; neural experts are `Expert` nodes with `ExpertOf` / `AlignedTo{cka}` (D-073); `TransitionVersion` / `RegionStatusVersion` hold event-time history (§6.3); `AdapterState` hangs off `Snapshot` via `HasAdapter`. The reference mirror (`Instrument`/`Entity`/`Venue` with `IssuedBy`/`ListedOn`/`SameExposure`) is written by control-api from Postgres `ref` (full sync at start and every 6 h, incremental on `ref.changed.*`; `fa_hub::reference`, spec 12 §6). `TrainedOn` (decoder genome → the encoder snapshot of the `evo.vary.decoder` job its parents were evaluated on, evo-worker) and `FeedbackFrom{weight}` (snapshot → elite decoder genomes of its D-009 feedback set, trainer, from `manifest.feedback`) are written as upserts keyed by the business ids stored on the edge.

### 5.3 VictoriaMetrics modeling rules (implemented by `go/pkg/tsdb` and `fa-tsdb`; budget numbers are placeholders)
- Metric names: `fa_trade_price`, `fa_trade_size`, `fa_quote_{bid,ask}_{px,sz}`, `fa_bar_{open,high,low,close,volume,vwap}`, `fa_book_{bid,ask}_{px,sz}`, `fa_feat_<name>`.
- Labels (bounded): `inst` (UUID), `venue`, `tf` (bar timeframe), `src`; book adds `lvl` (0..N-1, N ≤ 20). **No** order ids, trade ids, or free text as labels.
- Cardinality budget: series = instruments × venues × fields × (levels | timeframes); target < 5 M active series **(placeholder)**; instruments beyond budget kept at bar resolution only.
- Book depth: store **top-N snapshots at fixed cadence** (e.g. 1 s) rather than every update; full-depth deltas stay only in Garage raw archive.
- Trades with identical ms timestamps: aggregated per ms (VWAP + summed size + count) in VM; exact trade sequence only in raw archive.
  **Implemented today (closes D-038's pending item):** `go/pkg/tsdb.Writer.AddTrade` (every trade envelope from `AddEnvelope`) keeps the newest millisecond of each `(inst, venue, src)` series open and folds further trades of that millisecond into it; the bucket is written when a newer millisecond arrives or after `TradeHold` (default = flush interval, 1 s) as `fa_trade_price` = VWAP Σp·s/Σs (last price when Σs = 0), `fa_trade_size` = Σs and the new `fa_trade_count` = number of trades. A trade older than its series' newest written millisecond can no longer be merged and is written as-is (`tsdb.late`); merges are counted in `tsdb.aggregated`. Verified live against VictoriaMetrics (`FA_VM_TEST=1 go test ./pkg/tsdb -run Live`: three trades in one ms → one sample 102 / 3 / 3) and in the running ingestor (`tsdb.aggregated` rising on Yahoo's synthetic bar trades).
- Precision: float64 values; prices that need exact decimal (ledger) never come from VM.
- `publish_time` vs `event_time`: VM timestamp = `event_time`; a parallel `fa_avail_lag_ms` series or as-of logic in `fa-tsdb` (§6.3) — neither exists; backtests apply per-source knowledge delays instead (§6.3).

### 5.4 Postgres schemas
`ref` (12), `catalog`, `ingest`, `wh` (fundamentals, macro vintages, filings), `ledger` (42), `audit`, `ops` (alert history, job-run history, config snapshots; §6.1). Migrations in `migrations/` (sqlx).

### 5.5 Garage buckets
| Bucket | Layout |
|---|---|
| `fa-raw` | `<source>/<yyyy>/<mm>/<dd>/<hh>/<ulid>.zst` |
| `fa-media` | `<source>/<media_id>/{video,audio,frames,transcript}/…` |
| `fa-models` | `snapshots/<snapshot_id>/…` (in use; `snapshots/<timescale>/<family>/<gen>/…` reserved), `adapters/…`, `decoders/…`, `genomes/<genome_id>/weights` |
| `fa-ckpt` | preemptible job checkpoints (60) |
| `fa-artifacts` | decoder outputs (rendered charts, generated text), reports |

**Implemented today (D-045, D-050):** `fa-blob::S3BlobStore` (hand-rolled SigV4, no SDK) and `go/pkg/archive` (aws-sdk-go-v2) write to Garage. Keys in use: `fa-raw/raw/<source>/<yyyy>/<mm>/<dd>/<hh>/<ulid>.raw.gz` (one gzip object per source and UTC minute, envelopes carry `raw_ref{key, offset, length}` into the decompressed stream; gzip instead of zstd for now) plus an index object `<ulid>.idx.jsonl.gz` per data object (offset, length, received time, content type, transport meta) that makes `ctl.ingest.replay {verify|publish}` byte-exact (D-067; failed uploads go to a bounded local spool, a full spool flags `raw_missing`); `fa-models/snapshots/<snapshot_id>/{manifest.json,weights.bin,…}` (spec 36), `fa-models/adapters/<snapshot_id>/…` (adapter states and rank-8 LoRA `*.lora.safetensors`, D-073), `fa-models/decoders/<gen>/<family>-<hash>-<job>.safetensors` + `.card.json` (D-068); `fa-ckpt/<job_id>/state.json` (+ `neural.safetensors`) for resumable training; `fa-artifacts/evals/<generation>/<job>.json` (eval reports), `fa-artifacts/strategies/<gen>.toml` and the managed live set `fa-artifacts/live/strategies.toml` (D-071), `fa-artifacts/mcmc/<kernel>/{obs/…,<ts>.bin}` and `mcmc/crossprior/state.json` (D-075). Credentials come from `config/secrets.toml` `[garage]` (D-020), mounted read-only into the compose services.

## 6. Behaviour & algorithms

### 6.1 Write paths
Ingestor → NATS → (a) VM writer (Go, batched), (b) Postgres writer for macro/filings/fundamentals, (c) raw already in Garage. Model runtime → Helix (regions, transitions, observations). Ledger writes are synchronous and transactional (42).

**Implemented today:** ingestor → VictoriaMetrics (`fa_trade_*`, `fa_quote_*`, `fa_bar_*`, `fa_book_*`, D-038/D-050), ingestor → Garage raw archive and Postgres `ref.*`/`ingest.*`/`catalog.source` (D-050); model-runtime → HelixDB snapshots, expert spaces, regions, transitions, latent vectors through a non-blocking sink (D-048); risk-engine → Postgres `ledger.*` and `audit.log` as the single ledger writer (D-047); evo-worker → Garage eval reports (D-045) and Helix genome lineage (D-049); trainer → Garage snapshots/checkpoints and Helix candidate snapshots, region matches and experts; risk-engine → Helix `Genome` lifecycle status (D-076); control-api → Postgres `ops.*` and `audit.log`. Every store is reachable from `/api/health`.

**Since 2026-11-11 (Postgres):** `wh` holds macro vintages (`wh.macro_observation`, written by the ingestor's `go/pkg/warehouse`), filings (`wh.filing`, migration `20261111000045`) and vintaged fundamentals (`wh.fundamental`, written by the ingestor from SEC EDGAR XBRL company facts — keyless — since migration `20261113000012`: keyed by issuer CIK, `field = <taxonomy>:<concept>` + `unit`, `period` = the fact's own `start..end` / instant `end`, `vintage` = filed date 22:00 ET (conservative), `form`/`accession`/`fiscal_year`/`fiscal_period` kept; idempotent on (CIK, field, unit, period, vintage, source), revisions are new vintages; `instrument_id` only when exactly one instrument carries the CIK identifier; verified live: 14 708 facts for the 5 configured CIKs, 2009–2026); `ref` is bitemporal-lite throughout with as-of indexes (migration `20261111000010`, spec 12); the ledger gained tax lots, lot closings, orphan fills and a valued-fill view (migration `20261111000030`, spec 42). As-of reads for Postgres are `fa_warehouse::asof` (spec 12), `Warehouse::macro_as_of` (latest vintage ≤ t, index `macro_observation_asof_idx`) and `fa_warehouse::fundamentals::as_of` (per `(period, unit)` the latest vintage ≤ t, index `fundamental_asof_idx`).

**`ops` schema (round 3, migration `20261113000013`; `fa_warehouse::ops`, writer control-api):** `ops.alert_event` stores every alert transition control-api publishes on `ops.alert` (spec 60 §6.5; idempotent on `(fingerprint, status, at)`); at start control-api restores `GET /api/alerts` `recent` and the active set from it (an alert whose latest transition is `firing` is active again, so vmalert's re-sends after a restart are refreshes, not new firings), and serves `GET /api/alerts/history?since=&alertname=&limit=`. `ops.job_run` is the job registry's history: a durable explicit-ack consumer `ops-job-history` on `JOB_RESULTS` (`jobs.result.>`, deliver all) stores each `fa.v1.JobResult` (kind from the subject, status name, worker, error, GPU ms, output refs, checkpoint, `result` as JSON when it parses, stream sequence; a redelivery only overwrites with a newer stream position), so results published while control-api is down are stored when it returns and the retained stream history is backfilled; `GET /api/ops/jobs?kind=&limit=`, `GET /api/ops/jobs/summary?hours=`. `ops.config_snapshot` keeps every distinct redacted configuration per `(service, section)` (md5 digest; checked at start and every 5 min, so a `PUT /api/config/runtime` lands within 5 min); `GET /api/ops/config-snapshots?service=&limit=&config=true`. Live jobs themselves stay in JetStream (`JOBS_<kind>` work queues).

### 6.2 Read paths
Training/backtest read VM ranges + Postgres vintages + Helix graph; live runtime reads mostly from the bus and Helix.

### 6.3 As-of rules
- A query `as_of = t` returns only data with `publish_time ≤ t`.
- Postgres: bitemporal columns (`valid_from/to`, `recorded_at`); macro rows keyed by vintage.
- VM: since VM stores event time only, sources with `publish_time > event_time` (bars finalized at close, delayed feeds) get a per-source availability lag. **Today** this is not in `fa-tsdb`: `config/sources.yaml` carries measured `knowledge_delay`s (e.g. Kraken 120 s, Yahoo 100 s) and `fa_backtest::knowledge` shifts series by them in event-time replays (D-079); exact per-event availability only via raw replay or the recorded bus (`fa_bus::replay`, 41).
- Helix: all nodes carry `created_at`; snapshot-scoped queries filter by snapshot id and `created_at ≤ t`.
  **Implemented today:** `fa-hub` as-of variants `list_snapshots_as_of`, `space_regions_as_of`, `snapshot_graph_as_of` (`created_at ≤ t`), `space_latents_as_of` and `knn(.., KnnFilter{as_of_ms})` (`ts ≤ t`), `search_news_docs` / `search_media_segments` (`ts ≤ t`). Limits: `ExpertSpace` has no `created_at` (visible with its snapshot). **Versions (round 3, `fa_hub::versions`):** `Transition` posteriors and region lifecycle marks are updated in place, so every such write also appends an event-time version in the same transaction: `TransitionVersion {edge_key = from|to|sampler, count, mean, lower, upper, kernel_params, ts}` from each `upsertTransition`, `RegionStatusVersion {region_id, status, related_region_id, reason, ts}` from `markRegionStatus`, death marks, region creation and revival. `ts` is the record time of the write (at most one flush, 10 s, after the observation, never before it). As-of reads (`snapshot_graph_as_of`, `space_regions_as_of`) keep a row unchanged when its `updated_at ≤ t`, otherwise overlay its newest version with `ts ≤ t`; with none, an edge is not visible and a region's status is unknown (`null`). Rows written before versioning therefore read correctly after their last write and as invisible/unknown before it. Region statistics (mass, occupancy, spread, age) and `Snapshot.status` remain current values. Verified live: `stat-v1-1m` read now vs 30 s earlier differs (1 046 vs 1 043 edges, total count 2 740 vs 2 734); ~200 version rows/min.

### 6.4 Volume control in Helix
Helix stores **state observations at decision cadence per timescale** (e.g., once per bar), not per tick, plus aggregated transition counts. Windows for retrieval are stored at a stride per timescale.

**Implemented today:** `fa-hub::retention` — `prune_space_observations(space, cutoff)` folds `LatentVec kind=observation` rows with `ts < cutoff` into one `kind=aggregate` row per `(space, ref_id = region)` (`obs_count`, running mean vector in `vec<d>` so aggregates stay k-NN searchable, mean uncertainty, `ts` = newest folded row) and drops them in the same write batch, in pages of 500; `prune_snapshot_observations` / `prune_observations` apply a `RetentionPolicy` (horizon per `Snapshot.timescale`, §6.5). Regions, transitions, centroids and windows are never pruned. **Scheduled (round 3):** model-runtime runs it (`bins/model-runtime/src/hub_retention.rs`) every `--hub-prune-secs` / `FA_HUB_PRUNE_SECS` (6 h; first run 10 min after start, `--hub-prune-delay-secs`; 0 disables) with `--hub-retention` / `FA_HUB_RETENTION` (default `1s=2d,1m=14d,1h=180d`; 1d/1w kept), on its own client off the bus path, counted in `hub.retention.*`. The same cutoff thins `TransitionVersion` / `RegionStatusVersion` rows to the newest one older than the cutoff per edge / region (`pruneSpaceVersions`), so an as-of read at the cutoff keeps the value then in force. Verified live: the first run (05:39 UTC) found nothing past the horizons (live data < 2 days).

### 6.5 Retention (placeholders)
| Data | Retention |
|---|---|
| Raw archive (Garage) | forever (compressed) — source of truth for replay |
| VM seconds-scale series | 90 days hot |
| VM minute bars | 2 years |
| VM hour/day bars | forever |
| VM book top-N | 30 days |
| Helix state observations | per timescale: 2× backfill depth; older pruned to region-level aggregates |
| Genomes & lineage | forever (small); weights of non-elite genomes GC'd after N generations |
| Media | video downsampled after 30 days; audio + transcripts forever |
| JetStream streams | see 21 |

### 6.6 Backups
Original proposal: nightly Postgres dump + Helix snapshot + VM snapshot → Garage `fa-backup`; Garage itself replicated when a second node exists (60). Superseded by the implementation below.

**Implemented today (spec 60 §6.6):** Postgres continuous WAL archiving (volume `pg-wal-archive`, `archive_timeout` 300 s) + a nightly PITR base backup and `pg_dump` from the `backup` compose sidecar, a HelixDB volume tar (hot in the sidecar, cold via `scripts/backup.sh`), rclone copies of `fa-models fa-ckpt fa-artifacts fa-backup`, all into `./backups/<ts>/` with a sha256 manifest; retention 7 scheduled backups with WAL pruned to the oldest kept base; weekly automatic restore drill. VictoriaMetrics is **not** backed up: its trade/bar series are rebuildable from the Garage raw archive (10) and its ops metrics are disposable; `vmbackup` would be the tool if that changes. Garage is copied, not replicated (single node).

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `vm.max_active_series` | 5,000,000 | cardinality budget |
| `vm.book_levels` | 10 | |
| `vm.book_cadence` | 1 s | |
| `helix.obs_cadence.<timescale>` | 1 per bar | |
| `helix.window_stride.<timescale>` | 1 bar | |
| `garage.media_downsample_after_d` | 30 | |
| `retention.*` | see §6.5 | |

## 8. Failure modes
| Failure | Response |
|---|---|
| VM cardinality explosion | writer rejects new series beyond budget, alert, instrument demoted to bars |
| Helix unavailable | live runtime keeps in-memory graph, buffers writes; strategies needing Helix-only queries pause. **Today:** model-runtime's sink drops on a bounded channel (`hub.dropped`, D-048) and reconnects; its region sets and kernels are restored from `RegionSet` on start (D-069); `fa_hub::WriteBuffer` is the generic bounded queue of prepared write batches (drop-oldest with a `dropped` counter, in-order flush that stops on transport/5xx/retryable errors and discards non-retryable 4xx as `rejected`) |
| Postgres unavailable | **trading halts** (ledger is mandatory); ingest of static records buffered |
| Garage unavailable | see 10 §8 (local spool) |
| Disk budget exceeded | retention tightened automatically per priority table; alert |
| Schema drift Helix ↔ Rust | generated query wrappers are compile-checked; migration version stored in Helix. **Today:** `ensure_schema_ready` builds indexes, then `migrate_schema` applies `fa_hub::meta::MIGRATIONS` above the stored `SchemaMeta.version` and stamps it (forward only); a build older than the stored version fails with `SchemaTooNew` |

## 9. Acceptance criteria
**Remaining (2026-11-11, code checked at e25e9b5):**
- **VM cardinality budget** (§5.3, §8, criterion 3): not enforced. `fa-tsdb` and `go/pkg/tsdb` whitelist labels, but nothing counts active series, rejects series beyond `vm.max_active_series` or demotes an instrument to bars; the criterion has never been measured.
- **VM retention per timescale** (§6.5): not enforced. VictoriaMetrics runs with `-retentionPeriod=100y` and no writer deletes old series (the compose comment saying writers enforce it is wrong); seconds-scale series, minute bars and book top-N are kept indefinitely. Helix observation retention (§6.4) is the only scheduled pruning.
- **Disk budget** (§8 "retention tightened automatically"): not built; open question 3 (retention values and disk budget) is still open.
- **Genome weight GC** (§6.5 "weights of non-elite genomes GC'd after N generations"): not built; `fa-models` and `fa-artifacts` only grow.
- **Media downsampling** (§6.5, `garage.media_downsample_after_d`): not built; no media producer exists yet (`MediaChunk` has no publisher, spec 10).
- **Helix k-NN latency** (criterion 4, open question 2): no benchmark at 10 M vectors has been run.
- **As-of for VictoriaMetrics** (criterion 2, open question 4): `fa-tsdb` has no `as_of`; backtests apply per-source knowledge delays instead (§6.3). Either accept that as the design (and close Q4) or add an availability index.
- **Restore** (criterion 5): Postgres (logical + PITR) and Helix restores are drilled (`scripts/restore-drill.sh`, weekly Postgres drill in the `backup` sidecar); VictoriaMetrics is excluded by design and its rebuild from the raw archive (`ctl.ingest.replay {publish}`) has not been drilled.
- Open question 1 (order-book modelling in VM) is moot until a `BookSnapshot` producer exists, but not decided.

- Every data type in §5.1 has exactly one store of record.
- As-of queries never return data published after `as_of` (property test on fixtures with late publications). — **met for Postgres fundamentals** (`fundamentals::tests::as_of_never_sees_late_publications`: random restatements, probed around every vintage, equal the latest vintage ≤ t), reference data (`tests/refdata_asof_live.rs`, spec 12) and Helix (§6.3); VictoriaMetrics has no publish time (§6.3: knowledge delays are applied in backtests, not in `fa-tsdb`).
- VM series count stays under budget under first-wave load.
- Helix k-NN over an expert space returns in < 50 ms p95 for 10M vectors **(placeholder)**.
- Full restore from backups reproduces Postgres, Helix and VM state.

## 10. Open questions
1. VictoriaMetrics modeling of order books and label cardinality — confirm §5.3 rules or move book data elsewhere (biggest technical risk in this spec).
2. HelixDB scale limits (vector count, write throughput, transactions) — needs a benchmark spike before 30–34 commit to Helix-only queries.
3. Retention values and disk budget for the workstation (blocks §6.5).
4. Is publish-time lag per source sufficient in VM, or do we need a separate availability index?
5. ~~Region identity across snapshots (`REGION_MATCH`) — owned by 30, impacts Helix schema.~~ Resolved by D-069: Hungarian probe alignment, `RegionMatch` (trainer) + `DescendsFrom{carry|split|merge|partial}` edges and inherited `lineage_id` (spec 30).
6. ~~Helix schema versioning / migration story.~~ Resolved in `fa-hub::meta`: `SCHEMA_VERSION` + ordered `MIGRATIONS` (data steps; index-only changes need none since `ensure_schema` is idempotent and the index-set fingerprint is recorded), one `SchemaMeta` marker node in Helix, forward-only stamping, newer-than-build refused. Still open: rollback of a data migration (none exists yet; restore from the §6.6 backup).

## 11. Deferred
- Multi-node Garage / VM cluster setup.
- Cold storage tiering beyond Garage.
