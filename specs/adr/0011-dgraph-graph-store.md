# ADR 0011 — Dgraph as the single graph store

**Status:** **rolled back 2026-11-16** (D-112), decision reopened · accepted 2026-11-15 (owner: "dgraph, and move the postgres graph data to dgraph as well") · **Supersedes:** ADR 0002 · **Amends:** ADR 0010 (graph half; the vector half stands)

## What happened on the cutover (2026-11-16, D-112)

The code was complete and the migration was faithful. The store was not usable.

`graph_migrate --from postgres --to dgraph` copied the live book — 5 snapshots, 15 expert spaces,
406 regions, 5 916 transitions, **897 799** `transition_version` rows, 875 region-status versions,
815 hierarchy edges, 19 136 episodes — and `verify` matched all 25 tables. The platform was then
started on Dgraph and served graph reads for about fifteen minutes.

Then `dgraph-alpha` restarted, and never recovered. Applying the import left it looping on
`opRollup`, climbing to 11–16 GiB, where it was killed and restarted every 60–100 s. The
single-voter Raft group never elected a leader (`/state`: `"leader": false`), so every query timed
out while `/health` still reported `healthy`. Measured three ways on this 31 GiB host, all
identical:

| Attempt | Peak memory | Outcome |
|---|---|---|
| Default (4 GiB cache, no limit) | 16 GiB | killed, restart loop |
| `--cache size-mb=1024`, `mem_limit 7g`, `GOMEMLIMIT 5GiB` | 7 GiB (the cap) | killed every ~65 s |
| `mem_limit 22g`, `GOMEMLIMIT 18GiB`, rest of the platform stopped | 16.6 GiB | killed every ~90 s |

ADR 0010 rejected Dgraph on latency and this ADR accepted that regression knowingly. The regression
was not the problem: **the store could not stay up under the volume the platform actually produces.**

And the volume is the real finding. Those 898 k version rows accumulated in ~26 hours of paper
trading — roughly 1.2 M rows a day, because a version is appended per transition per kernel update.
No graph store answers that forever without a retention or compaction policy for event-time
history; Postgres holds it today only because 900 k narrow rows are nothing to Postgres.

**Live state since the rollback:** `[graph] backend = "postgres"`, `HubClient.graph` is
`Option<Arc<PgGraph>>` again, the `graph` schema is intact and was never dropped, and
`dgraph-zero` / `dgraph-alpha` sit behind compose profile `dgraph`. Every line of the Dgraph
implementation stays in the tree — `graph_dgraph.rs`, `dgraph_schema.rs`, `dgraph_evolution.rs`,
the migration with its `--verify-only` / `--times-only` modes, and the live tests that drive the
store directly — so a retry costs a config value, not a rewrite.

**The decision is the owner's again,** in three shapes: (a) give event-time history a retention /
compaction policy and retry Dgraph against a graph that stops growing without bound; (b) run Dgraph
for current state and leave history in Postgres, accepting that as-of overlays lose their history
and that "one graph store" is no longer true; (c) keep the Postgres graph and retire this ADR.

## Decision (as accepted 2026-11-15)

One graph store: Dgraph (`dgraph/dgraph:v25.5.1`, Apache-2.0, Go), replacing **both** the HelixDB graph of ADR 0002 and the PostgreSQL `graph` schema of ADR 0010 / D-108. Compose runs `dgraph-zero` + `dgraph-alpha` (single node, `--replicas=1`). `[graph] backend` accepts `dgraph` (default) or `helix`; `postgres` is no longer a valid value. Vectors stay in Qdrant (D-101, D-109) — this ADR does not reopen that.

This is a **direct replacement**, chosen over a phased dual-backend period: `PgGraph`, `HelixGraph` and the `graph` schema all go, with no fallback.

## The regression this accepts
ADR 0010 evaluated Dgraph as its option D and **rejected it on measured latency**. Those measurements, on this host against the same synthetic graph at ~10× live size:

| Shape | PostgreSQL 17 p50 / p95 | Dgraph v24.1.4 p50 / p95 |
|---|---|---|
| `genomeAncestry`, depth 10 | 0.58 / 0.69 ms | 2.84 / 7.18 ms |
| `regionLineage`, depth 6 | 0.51 / 0.65 ms | 2.46 / 3.41 ms |
| 2-hop transition neighbourhood | 0.42 / 0.54 ms | 3.49 / 5.67 ms |
| **`snapshotGraphAsOf`** (1 642 edges) | **8.9 / 10.4 ms** | **209 / 505 ms** |
| `instrumentReference` | 0.37 / 0.47 ms | 1.55 / 2.48 ms |
| Bulk load · memory | 2.0 s · 431 MiB | 47 s · 3.3 GiB |

The owner was shown this table on 2026-11-15 and chose to proceed **without a re-benchmark on v25.5.1 and without keeping a fallback**. So this ADR does not claim Dgraph is faster, or that the gap has closed: it records that the platform accepts 4–20× slower graph reads, ~24–50× on the as-of overlay, and ~8× memory, in exchange for a single Go graph store.

The as-of overlay is slow for a structural reason, not a tuning one. DQL has no `DISTINCT ON` and no window functions, so "newest version ≤ t per key" cannot be one query; it is one named query block per key in a single request (chunked 32 at a time), which is the same shape the Helix backend used. `prune_space_versions` has the same problem — `row_number() OVER (PARTITION BY …)` inside `DELETE … USING` becomes read, rank in Rust, delete by uid.

**Re-measuring is the obvious follow-up.** `examples/graph_bench.rs` already drives Dgraph over DQL/HTTP with `--dgraph <url>` and replays exactly the shapes above, so reproducing the table on v25.5.1 is hours of work, not weeks.

## Why Dgraph specifically
The owner asked for a high-performance graph database written in Go. ADR 0010 had already surveyed that ground and reached the same conclusion this one does: Dgraph is the only mature Go graph database — Cayley is a graph layer over KV stores and effectively dormant, EliasDB is hobby scale, and the newer Go projects (NornicDB and friends) are single-maintainer and too young for a state store. The faster engines are all non-Go (FalkorDB, Kuzu, Apache AGE).

Stewardship has changed twice since 2023 (Dgraph Labs → Hypermode → Istari Digital, Oct 2025). v25 consolidated the formerly enterprise features into the single Apache-2.0 build: the `ee/` directory is gone from the repo and a running alpha reports `ee_features: ["backup_restore", "cdc"]`, so binary backup/restore needs no licence.

## Consequences

**Better than before.** The alpha image ships a shell, so the graph store gets a real healthcheck and `service_healthy` dependencies for the first time — Helix was distroless and could only be probed by TCP from control-api. Dgraph also exposes `/debug/prometheus_metrics`, where Helix had no metrics endpoint at all. Backups use Dgraph's native backup rather than tarring a stopped volume.

**Node ids.** Dgraph `uid`s are cluster-wide `uint64`, so they satisfy the "globally unique across labels" property that Postgres needed a shared `graph.node_id` sequence for. Ids still never cross backends.

**Predicate naming.** Dgraph predicates are global, not type-scoped, so every one is written `<type>.<field>`. Without that, `region.status` ("live"/"retired") and `snapshot.status` ("candidate"/"active") would be one predicate with two value domains.

**Reified edges.** Edges carrying properties are nodes (`Transition`, `Hier`, `GenomeParent`, `ExpertAlign`, …), because `GraphStore` hands edges to callers as rows with ids.

**The explorer lost its label-free scan.** `resolve_node` and `neighborhood` scanned without a label on Helix. DQL has neither an untyped scan nor untyped edge traversal, so `resolve_node` walks a declared type list and `neighborhood` follows the links the schema declares. A type added to the schema but missing from that list would be silently invisible to the explorer, so a unit test keeps the two in step.

**The meta marker shrank.** Helix's `SchemaMeta` fingerprint hashed Helix index DDL, which has no Dgraph analogue — the DQL schema is declarative and `/alter` is idempotent. `SchemaMarker` records only the version, still monotonic and still refusing a store written by a newer build.

**Vectors become Qdrant-only.** With Helix gone, `[qdrant] vector_backend = "helix"` has no implementation left.

## Migration
`examples/graph_migrate.rs --from postgres --to dgraph`, then `verify` compares all 25 tables and exits non-zero on a mismatch. The source is dropped afterwards, so the copy has to be complete, and three real losses were found only by that count check and not by any error:

- 62 594 `transition_version` rows arrived as 1 858 — in a JSON mutation a blank-node label names one node, so a repeated `_:v` collapsed each 1 000-row chunk into a single row.
- 7 `ref_venue` rows arrived as 2 — the generic pass walks the mirror through `instrument_reference`, so a venue with no listing was never visited.
- Then 64 389 versions for 62 594 source rows: copying current values through the runtime API *manufactures* history, because `upsert_transitions` and `upsert_region` append a version stamped `now()` as any live write must.

The history import is therefore authoritative — destination history is cleared before importing — which also makes the migration idempotent; without it a second run doubled history to 125 188 rows. **The migration must finish before the platform starts writing to Dgraph**, which is what makes "authoritative" safe.

Helix holds no genome, fitness, Pareto or expert rows at all (measured 2026-11-15: only its own `SchemaMeta`), so `helix → dgraph` carries no evolution copy path and refuses loudly if that ever stops being true.

Measured against a copy of the real book — 269 regions, 1 795 transitions, 62 594 versions, 4 545 episodes, 5 snapshots, 243 hierarchy edges — all 25 tables match, a re-run still matches, and `--dry-run` writes nothing.

## Alternatives
- **Keep PostgreSQL (ADR 0010's option C).** Faster on every measured shape, already operated, backed up and transactional. Rejected by the owner in favour of a single Go graph store.
- **Phased dual-backend cutover.** Rejected by the owner: direct replacement, no fallback.
- **Apache AGE** (openCypher inside the existing Postgres) would have kept one datastore and given the explorer Cypher, but it is C, not Go.
- **Dgraph's native HNSW for vectors** could later remove the `qdrant` service. Explicitly out of scope here (owner, 2026-11-15); D-101/D-109 stand.
