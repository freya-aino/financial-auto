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
- Go: VictoriaMetrics written by ingestor via remote-write / import API **(proposal)**; Postgres via `pgx`; Garage via S3 SDK.
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
| Raw payloads, media, weights, checkpoints, rendered decoder outputs | Garage | content-addressed key |

### 5.2 Helix schema sketch **(proposal)**
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

### 5.3 VictoriaMetrics modeling rules **(proposal)**
- Metric names: `fa_trade_price`, `fa_trade_size`, `fa_quote_{bid,ask}_{px,sz}`, `fa_bar_{open,high,low,close,volume,vwap}`, `fa_book_{bid,ask}_{px,sz}`, `fa_feat_<name>`.
- Labels (bounded): `inst` (UUID), `venue`, `tf` (bar timeframe), `src`; book adds `lvl` (0..N-1, N ≤ 20). **No** order ids, trade ids, or free text as labels.
- Cardinality budget: series = instruments × venues × fields × (levels | timeframes); target < 5 M active series **(placeholder)**; instruments beyond budget kept at bar resolution only.
- Book depth: store **top-N snapshots at fixed cadence** (e.g. 1 s) rather than every update; full-depth deltas stay only in Garage raw archive.
- Trades with identical ms timestamps: aggregated per ms (VWAP + summed size + count) in VM; exact trade sequence only in raw archive.
- Precision: float64 values; prices that need exact decimal (ledger) never come from VM.
- `publish_time` vs `event_time`: VM timestamp = `event_time`; a parallel `fa_avail_lag_ms` series **(proposal)** or as-of logic in `fa-tsdb` (§6.3).

### 5.4 Postgres schemas
`ref` (12), `catalog`, `ingest`, `wh` (fundamentals, macro vintages, filings), `ledger` (42), `audit`, `ops` (job registry, config snapshots). Migrations in `migrations/` (sqlx).

### 5.5 Garage buckets
| Bucket | Layout |
|---|---|
| `fa-raw` | `<source>/<yyyy>/<mm>/<dd>/<hh>/<ulid>.zst` |
| `fa-media` | `<source>/<media_id>/{video,audio,frames,transcript}/…` |
| `fa-models` | `snapshots/<timescale>/<family>/<gen>/…`, `adapters/…`, `genomes/<genome_id>/weights` |
| `fa-ckpt` | preemptible job checkpoints (60) |
| `fa-artifacts` | decoder outputs (rendered charts, generated text), reports |

**Implemented today (D-045, D-050):** `fa-blob::S3BlobStore` (hand-rolled SigV4, no SDK) and `go/pkg/archive` (aws-sdk-go-v2) write to Garage. Keys in use: `fa-raw/raw/<source>/<yyyy>/<mm>/<dd>/<hh>/<ulid>.raw.gz` (one gzip object per source and UTC minute, envelopes carry `raw_ref{key, offset, length}` into the decompressed stream; gzip instead of zstd for now), `fa-artifacts/evals/<generation>/<job>.json` (decoder eval reports). Credentials come from `config/secrets.toml` `[garage]` (D-020), mounted read-only into the compose services.

## 6. Behaviour & algorithms

### 6.1 Write paths
Ingestor → NATS → (a) VM writer (Go, batched), (b) Postgres writer for macro/filings/fundamentals, (c) raw already in Garage. Model runtime → Helix (regions, transitions, observations). Ledger writes are synchronous and transactional (42).

**Implemented today:** ingestor → VictoriaMetrics (`fa_trade_*`, `fa_quote_*`, `fa_bar_*`, `fa_book_*`, D-038/D-050), ingestor → Garage raw archive and Postgres `ref.*`/`ingest.*`/`catalog.source` (D-050); model-runtime → HelixDB snapshots, expert spaces, regions, transitions, latent vectors through a non-blocking sink (D-048); risk-engine → Postgres `ledger.*` and `audit.log` as the single ledger writer (D-047); evo-worker → Garage eval reports (D-045) and Helix genome lineage (D-049). Every store is reachable from `/api/health`.

### 6.2 Read paths
Training/backtest read VM ranges + Postgres vintages + Helix graph; live runtime reads mostly from the bus and Helix.

### 6.3 As-of rules
- A query `as_of = t` returns only data with `publish_time ≤ t`.
- Postgres: bitemporal columns (`valid_from/to`, `recorded_at`); macro rows keyed by vintage.
- VM: since VM stores event time only, sources with `publish_time > event_time` (bars finalized at close, delayed feeds) get a per-source availability lag applied in `fa-tsdb` **(proposal)**; exact per-event availability only via raw replay (41).
- Helix: all nodes carry `created_at`; snapshot-scoped queries filter by snapshot id and `created_at ≤ t`.

### 6.4 Volume control in Helix
Helix stores **state observations at decision cadence per timescale** (e.g., once per bar), not per tick, plus aggregated transition counts. Windows for retrieval are stored at a stride per timescale.

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
Nightly Postgres dump + Helix snapshot + VM snapshot → Garage `fa-backup` **(proposal)**; Garage itself replicated when a second node exists (60).

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
| Helix unavailable | live runtime keeps in-memory graph, buffers writes; strategies needing Helix-only queries pause |
| Postgres unavailable | **trading halts** (ledger is mandatory); ingest of static records buffered |
| Garage unavailable | see 10 §8 (local spool) |
| Disk budget exceeded | retention tightened automatically per priority table; alert |
| Schema drift Helix ↔ Rust | generated query wrappers are compile-checked; migration version stored in Helix |

## 9. Acceptance criteria
- Every data type in §5.1 has exactly one store of record.
- As-of queries never return data published after `as_of` (property test on fixtures with late publications).
- VM series count stays under budget under first-wave load.
- Helix k-NN over an expert space returns in < 50 ms p95 for 10M vectors **(placeholder)**.
- Full restore from backups reproduces Postgres, Helix and VM state.

## 10. Open questions
1. VictoriaMetrics modeling of order books and label cardinality — confirm §5.3 rules or move book data elsewhere (biggest technical risk in this spec).
2. HelixDB scale limits (vector count, write throughput, transactions) — needs a benchmark spike before 30–34 commit to Helix-only queries.
3. Retention values and disk budget for the workstation (blocks §6.5).
4. Is publish-time lag per source sufficient in VM, or do we need a separate availability index?
5. Region identity across snapshots (`REGION_MATCH`) — owned by 30, impacts Helix schema.
6. Helix schema versioning / migration story.

## 11. Deferred
- Multi-node Garage / VM cluster setup.
- Cold storage tiering beyond Garage.
