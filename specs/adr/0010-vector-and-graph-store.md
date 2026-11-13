# ADR 0010 — Vector and graph store: Qdrant for vectors, PostgreSQL for the graph

**Status:** vectors **accepted** (owner 2026-11-14: "qdrant is preferred, use it"; D-101) · graph **proposed** (PostgreSQL vs keeping Helix, undecided) · **Date:** 2026-11-13 · **Supersedes:** ADR 0002 for vectors (Helix stays the graph store until the graph half is decided)

## Decision (2026-11-14, vectors)
Latents, news and media embeddings move to Qdrant (`qdrant/qdrant:v1.20.0`, compose service `qdrant`, host port 6333). Implemented in `fa_hub::qdrant` behind the existing `HubClient` API, so no caller changed: a client with a `QdrantVectors` attached runs `add_latents(_checked)`, `knn`, `search_latent`, `space_latents(_as_of)`, `count_space_latents`, `upsert_centroid`'s replace, the news/media add+search and the retention fold on Qdrant; everything else stays in Helix, including `ExpertSpace.dim`, which remains the dimension authority. The backend is `[qdrant] vector_backend = "qdrant" | "helix"` (default `qdrant`; `FA_VECTOR_BACKEND`, `FA_QDRANT_URL` override), applied per process by `fa_hub::configure_vectors` in every service that opens a hub client. Collections per dimension (`latent_d<dim>`, `news_d<dim>`, `media_d<dim>`), cosine, payload indexes `expert_space_id` (keyword, `is_tenant`), `ts` (integer range, unix ms), `kind` / `instrument_id` / `ref_id` (keyword); filters run inside the query, so the Helix over-fetch is gone. Point ids are FNV-1a 64 of the business key (observation/window: space, kind, ref_id, instrument, ts; centroid/aggregate: space, kind, ref_id; embeddings: label, id), which makes redelivered writes idempotent. Qdrant stores cosine vectors normalised, so the payload keeps `norm` and reads restore the raw vector. Per the ephemeral rule (D-096) there was no data to migrate: no backfill, no dual-write; the step list below is kept for the graph half. Measured through `fa-hub` (`examples/knn_bench.rs --filtered`, dim 32, scratch Qdrant + Helix): p95 1.5 / 4.0 / 10.9 ms at 10 k / 50 k / 200 k, recall 1.000, 33–54 k inserts/s (segments below Qdrant's indexing threshold, i.e. exact segment scans; the 1 M / 10 M index numbers are in the table below).

## Context
ADR 0002 put graph and vectors into one HelixDB. The spec 20 §6.4 k-NN bench (2026-11-11, v0.0.14) measured p95 93 ms at 50 k 32-d vectors, latency that grows with the **total** vectors in the shared `vec<d>` index, 429 `index_backpressure` at 250 k pending members and ≈ 21 KB per vector; the criterion-4 target (placeholder) is < 50 ms p95 at 10 M. The owner asked to check Qdrant for workability and scalability, to look for a mature Go graph store, and to confirm whether the latency is a hardware limit.

**What the platform actually asks of the store** (`crates/fa-hub`, `helix/queries.hx`):

| Kind | Query shape | Where |
|---|---|---|
| Vector | `knn(space, v, k, {kind, instrument_id, as_of: ts ≤ t})`: cosine top-k inside one expert space (tenant), filter applied after an over-fetched top-k (×4, ≤ 4096) | `vectors.rs`, retrieval decoders (spec 34) |
| Vector | `searchNewsDocs` / `searchMediaSegments` with `ts ≤ as_of`; `spaceLatentsAsOf` (scan, newest first, ≤ 20 000 rows for the PCA view) | `vectors.rs`, `asof.rs` |
| Vector | retention: fold `observation` rows older than the cutoff into per-region `aggregate` vectors, delete | `retention.rs` |
| Graph | `snapshotGraph` / `snapshotGraphAsOf`: snapshot → spaces → regions (`created_at ≤ t`) → transitions, each edge overlaid with its newest `TransitionVersion ≤ t` | `queries.rs`, `asof.rs`, `versions.rs` |
| Graph | `regionTransitions`, `regionNeighbors`, `neighborhood(node, depth)` (1–n hops, depth-unrolled batches) | `queries.rs`, `catalog.rs` |
| Graph | `regionLineage(region, depth)`: ancestors / descendants over `DescendsFrom` + `RegionMatch`; `lineageMembers` | `regions.rs` |
| Graph | `genomeAncestry(genome, depth)` over `ParentOf`; `genomeFitness`, `genomeTrainedOn`, `snapshotFeedback`, `expertParents` (`AlignedTo`) | `queries.rs`, `evolution.rs`, `experts.rs` |
| Graph | reference mirror: `instrumentReference` = instrument → issuer → sector, listings → venues, `SameExposure`; full / incremental sync from Postgres `ref` | `reference.rs` |
| Graph | upserts: `upsertTransition` (+ version row), `upsertRegion`, `upsertDescents`, `upsertHierEdges`, `upsertGenome`, `linkParent`, … (lookup → update or create) | all writers |

No query combines a vector search and a traversal in one engine round trip; fa-hub already issues them as separate batches.

## Measurements (2026-11-13)
Host: 12 cores, 31 GB RAM, NVMe. Every engine ran as a scratch container (`fa-bench-*`, tmpfs or scratch dir) on a private Docker network with the client container in the same network; one query in flight, k = 10, cosine, 64-cluster Gaussian mixture (as in `knn_bench`). The host was shared with a concurrent agent (trainer + rustc; load average 4–20), so absolute numbers carry ≈ ±30 % noise; the rows marked "re-run" were taken at load 4–8. Harnesses: `crates/fa-hub/examples/knn_floor.rs`, `qdrant_bench.rs`, `knn_bench.rs` (Helix), `graph_bench.rs`.

### Hardware floor (in process, no database)
| dim 32 | 50 k | 1 M | 10 M |
|---|---|---|---|
| exact scan, 1 core, p50 | 0.43 ms | 10.6 ms | 110 ms |
| exact scan, 12 cores, p50 / p95 | 0.17 / 0.33 ms | 3.8 / 4.5 ms | **37.7 / 44.7 ms** |
| HNSW (`instant-distance`, M 32, ef 100) p50 / p95 | 0.11 / 0.17 ms | **1.02 / 1.07 ms** | not built (≈ 1 h) |
| HNSW recall@10 · build time | 0.996 · 4.9 s | 0.997 · 393 s | – |

dim 128 (under load 13–20): exact 12-core p95 0.89 ms (50 k) / 77 ms (1 M); HNSW p95 0.67 ms (50 k, recall 1.000) / 4.1 ms (1 M, recall 0.958, build 2 011 s).

### Vector stores, dim 32 (p50 / p95 ms, recall@10)
| Engine | 10 k | 50 k | 1 M | 10 M |
|---|---|---|---|---|
| HelixDB v0.0.14, one space (idle, re-run) | 13.6 / 20.8 | 47–49 / **69–72** (right after drain 67 / 136) | – (§6.4: 213–300 at 250 k) | – (≈ 200 GB) |
| HelixDB v0.0.16 (2026-11-11 image), idle | 13.0 / 17.4 | 32–38 / **54–63** (after drain 62 / 80) | – | – |
| Qdrant v1.20.0, one collection, no filter | – | 1.34 / 2.36 · 1.000 | 1.17 / 1.93 · 0.998 | 7.8 / 14.3 (first cold pass 15 / 140) · 0.983 |
| Qdrant, `space_id` filter (1 of 8 tenants) | – | 0.85 / 1.50 · 1.000 | 3.72 / 5.92 · 1.000 | 6.1 / **9.2** · 0.978 |
| Qdrant, `space_id` + `ts ≤ as_of` (as-of read) | – | 0.92 / 1.59 · 1.000 | 4.57 / 6.83 · 1.000 | 7.5 / **11.0** (p99 16) · 0.971 |
| Qdrant, collection per space (8 × 125 k at 1 M), `ts ≤ as_of` | – | 0.73 / 1.51 · 1.000 | 3.26 / 5.08 · 1.000 | – |

Recall at 10 M is from the first pass (the warm latency re-run used a different draw order, so its recall is not comparable). Collection-per-space at 1 M stayed below Qdrant's `indexing_threshold` (125 k × 32-d per collection ≈ 16 MB in 6 segments), i.e. it is a plain scan and still 5 ms p95.

| Write / resources | HelixDB v0.0.14 / v0.0.16 | Qdrant v1.20.0 |
|---|---|---|
| Insert | 7–22 k vectors/s accepted, index built asynchronously; 429 at 250 k pending | 74–85 k points/s (`wait=true`, 8 × 1 000) to 1 M; 41 k/s for 1 M → 10 M |
| Index ready | drain 3 s (10 k), 11 s (50 k) | 39 s at 1 M; 35 min at 10 M (host load 13–20) |
| Memory | 2.1–2.2 GB RSS at 60 k vectors (≈ 35 KB/vector) | 1.3 GB resident at 10 M (container peak 4.2 GB while optimising) |
| Disk | ≈ 21 KB/vector (§6.4) | 5.1 GB at 10 M incl. payload, indexes, WAL (≈ 510 B/vector) |

### Graph stores (~10× today's graph)
Synthetic graph: 100 snapshots × 4 spaces × 64 regions = 25 600 regions, 204 800 transitions with 614 400 versions, 31 794 `DescendsFrom`, 50 000 genomes in 500 generations with 99 800 `ParentOf`, 20 000 instruments / 8 000 issuers / 60 venues / 40 sectors / 29 955 listings. 200 measured queries after 20 warm-up, re-run at load 4–8. Same answers from both stores (e.g. 153 distinct ancestors, 24.8 lineage rows, 1 642 as-of edges per snapshot).

| Shape | PostgreSQL 17 (recursive CTE / LATERAL) p50 / p95 | Dgraph v24.1.4 standalone (DQL `@recurse`, reified edges) p50 / p95 |
|---|---|---|
| `genomeAncestry`, depth 10 | 0.58 / 0.69 ms | 2.84 / 7.18 ms |
| `regionLineage`, up + down, depth 6 | 0.51 / 0.65 ms | 2.46 / 3.41 ms |
| 2-hop transition neighbourhood with summed counts | 0.42 / 0.54 ms | 3.49 / 5.67 ms (client-side group-by) |
| `snapshotGraphAsOf` with newest version ≤ t per edge (1 642 edges) | 8.9 / 10.4 ms | 209 / 505 ms |
| `instrumentReference` | 0.37 / 0.47 ms | 1.55 / 2.48 ms |
| Bulk load · memory | 2.0 s · 431 MiB | 47 s · 3.3 GiB |

## Is the 93 ms a hardware limit?
**No.** On the same CPU an exact brute-force scan answers 1 M vectors in 4.5 ms p95 and even **10 M in 45 ms p95** (memory-bandwidth bound, ≈ 35 GB/s), an in-process HNSW answers 1 M in 1.1 ms, and Qdrant answers a filtered as-of k-NN over 10 M in 11 ms p95. HelixDB needs 54–72 ms at 50 k (93 ms in §6.4 under heavier host load) — about 300× the in-process HNSW and 200× an exact scan at the same size, and slower than an exact scan of 1 M. The cost is in Helix itself (index kept in an LSM object store at ≈ 21 KB and ≈ 35 KB RSS per vector, per-query I/O over the shared `vec<d>` index, post-filter of the tenant partition), and v0.0.16 improves it by only ≈ 20 %.

## Options
- **A. Keep HelixDB for both.** Works for today's volume-controlled spaces (≤ 10–20 k vectors per dimension index under 50 ms) and keeps one engine. Does not reach criterion 4 by two orders of magnitude; 0.0.x releases, asynchronous index with backpressure.
- **B. Qdrant for vectors, HelixDB for the graph.** Fixes k-NN; keeps a pre-1.0 engine and the reference mirror for graph reads that are small.
- **C. Qdrant for vectors, PostgreSQL for the graph.** Every graph shape ≤ 10 ms p95 at 10× size; Postgres is already operated, backed up and transactional (ADR 0008); `ref` joins directly, so the Helix reference mirror (`ref_mirror.rs`) disappears; as-of versions are ordinary indexed tables.
- **D. Qdrant for vectors, Dgraph for the graph.** Dgraph is the only mature Go graph database (Cayley and EliasDB are small or dormant), but here it is 4–20× slower than Postgres on every shape (50× on the as-of overlay), needs 3.3 GiB for this graph, adds a third data system, and its stewardship has changed hands since 2023.
- **E. pgvector in the same Postgres.** One system for both; not measured here. Filtered HNSW in pgvector is post-filtered (iterative scans), and the measured Qdrant filter path is what criterion 4 needs; can be revisited if Qdrant's operations cost proves too high.

## Recommendation
**Option C.** Qdrant (pinned image, currently `qdrant/qdrant:v1.20.0`) for latents, news and media embeddings: one collection per vector dimension (`latent_d<dim>`, `news_d<dim>`, `media_d<dim>`) with a `space_id` keyword index marked `is_tenant`, a `ts` integer range index and keyword indexes on `kind` / `instrument_id`; filters run inside the index, so fa-hub's over-fetch and post-filter go away and filtered queries return k hits. Collection-per-space is not needed (shared + tenant index already meets the target with 4–5× margin) and would multiply collections by snapshots × experts. PostgreSQL schema `hub` for the graph (snapshots, spaces, regions, transitions + versions, lineage, hierarchy, genomes, fitness, experts, adapters), next to `ref`. HelixDB is retired after the cutover. criterion 4 (< 50 ms p95 at 10 M) is met by Qdrant on this host as measured (11 ms filtered as-of).

## Migration sketch
1. **Traits.** Split `fa_hub::HubClient` (≈ 130 public async fns, 9 of them vector) into `VectorStore` (`add_latents_checked`, `knn`, `search_news_docs`, `search_media_segments`, `space_latents(_as_of)`, `count_space_latents`, retention fold) and `GraphStore` (everything else). Helix keeps implementing both (`HelixVectorStore`, `HelixGraphStore`) as the reference; the live tests become a contract suite run per adapter.
2. **Adapters.** `QdrantVectorStore` over REST with the existing `reqwest`/`serde_json` (no gRPC crates); point id = hash of `(space_id, kind, ref_id, ts)`, payload carries the Helix properties. `PgGraphStore` on `sqlx` with migrations in `migrations/` (`hub.*` tables, unique business ids, `(tid, ts DESC)` version index, recursive CTEs for lineage / ancestry / neighbourhood, the `catalog.rs` whitelist mapped to SQL). Reference reads join `ref.*` directly; `ref_mirror` is removed at the end.
3. **Compose.** Add a `qdrant` service (pinned, named volume, healthcheck, snapshots to Garage in the backup job); the `hub` schema migration runs with the others.
4. **Dual-write.** Writers (model-runtime `hub_sink`, trainer, evo-worker, risk-engine, control-api) go through a `TeeStore` (primary Helix, shadow new) behind a config flag; the D-089 disk log replays both sides. A reconciliation job compares counts, graph answers row by row and k-NN overlap@10 on sampled queries.
5. **Backfill.** One-shot copy of the existing Helix rows (snapshots, regions, transitions with versions, latents, genomes, lineage) into the new stores, then dual-write keeps them in step.
6. **Cutover.** Flip reads per path — control-api graph/latent routes first, then decoder k-NN — then flip the primary; Helix stays read-only for one release, then leaves compose. ADR 0002 becomes superseded; spec 20 §5.2/§6.4 and `helix/` are rewritten as the Postgres/Qdrant data model.

## Risks
- **Two stores instead of one:** cross-store consistency rests on business ids and write ordering; retention must prune both (vector fold in Qdrant, version thinning in Postgres). There is no single-engine vector-then-traverse query, but none exists in fa-hub today.
- **Postgres shares the ledger instance:** graph writes (transition upserts, versions) add WAL and vacuum load next to the ledger, which halts trading when Postgres is down (ADR 0008). Mitigate with a separate pool and limits, fillfactor/HOT for hot tables, or a separate database on the cluster.
- **Qdrant at 10 M:** index build took 35 min on a loaded host and the container peaked at 4.2 GB; plan `indexing_threshold`, optimizer threads, `on_disk` vectors or quantisation before the space sizes grow, and alert on collection status ≠ green.
- **Benchmark limits:** synthetic mixture data, mostly 32-d, one client, one in flight, tenant selectivity 1/8 and as-of at 70 %; host shared with another workload (±30 %); Qdrant 10 M recall from a single pass; Helix graph shapes were not re-benchmarked (fa-hub issues them as depth-unrolled HTTP batches); the 10 M HNSW floor was not built. Concurrency and mixed read/write load are untested.
- **Operational:** a new pinned image to track (Qdrant) and a migration of every fa-hub caller; until cutover, both paths must be kept correct.
