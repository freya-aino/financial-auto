# helix

The HelixDB data model: `schema.hx` (nodes, edges, vector types) and `queries.hx` (query
catalog). Governed by [spec 20 §5.2](../specs/20-storage.md), ADR 0002, D-012 and D-033.

**Status:** the `.hx` files are the **readable catalog** of the model, not something the server
loads. HelixDB now runs from the official standalone image (D-033), which exposes the
operation-tree API at `POST /v2/query` and does not compile HelixQL projects into named
endpoints (HelixQL is "legacy" in the upstream docs; the SDKs and `helix query` build the same
JSON operation tree). The executable counterpart of these files is
[`crates/fa-hub`](../crates/fa-hub/README.md):

| `.hx` construct | fa-hub |
|---|---|
| `N::Label { … }` / `E::Label { From, To, Properties }` | `schema::node::*` / `schema::edge::*` label constants; properties are inferred by the server on write |
| `UNIQUE INDEX field` / `INDEX field` | `schema::static_indexes()` → `create_index{node_equality{unique}}`, created by `HubClient::ensure_schema()` |
| `V::LatentVec { … }` | label `LatentVec`, vector in property `vec<dim>`, one `node_vector` index per dimension partitioned by `expert_space_id` (`ensure_latent_index(dim)`) |
| `Date DEFAULT NOW` | `{"expr":"date_time_now"}` on insert |
| `QUERY name(args) => … RETURN x` | `HubClient::name(args)` in `queries.rs` (snake_case, same semantics), one `/v2/query` batch each |

When changing the model, edit the `.hx` file first (it is the review surface), then mirror the
change in `fa-hub::schema` / `fa-hub::queries` and run the live tests.

**Who writes (D-048):** `bins/model-runtime` (`hub_sink.rs`) persists its snapshot
(`stat-v1-<ts>`), expert spaces (`<snapshot>-e<k>`), regions with lifecycle `status`,
coalesced `Transition` posteriors and closed-bar `LatentVec` observations (1m and slower);
`bins/control-api` (`graph_routes.rs`) serves spec 50 §4.4/§4.5 from these rows, including the
named-query whitelist of `fa-hub::catalog` behind `POST /api/graph/query`. evo-worker writes
genomes/lineage/fitness through `upsert_genome` / `link_parent` / `add_fitness_record` (pending).

## Contents
| Area | Nodes / vectors | Edges |
|---|---|---|
| Reference (mirror of Postgres `ref`) | Instrument, Entity, Venue | IssuedBy, ListedOn, SameExposure |
| Snapshots | Snapshot, ExpertSpace, AdapterState | HasSpace, SnapshotParent, HasAdapter |
| Regions / Markov graph | LatentRegion, Window, `V::LatentVec` | InSpace, Transition, Hier, RegionMatch, Contains |
| Evolution | Genome, FitnessRecord | ParentOf, Scored, TrainedOn, FeedbackFrom |
| Text / media | `V::NewsDoc`, `V::MediaSegment` | Mentions |

## Design notes
- **Mixed-dimension latents** (previously an open gap): a vector index is keyed by
  `(label, property)` with a fixed dimension, so `LatentVec` stores a `d`-dimensional vector in
  `vec<d>` and gets one index per `d`, each partitioned per expert space via `tenant_property =
  expert_space_id`. Searches are therefore always scoped to one expert space (what spec 34 wants).
- **Upserts:** no server-side upsert; `fa-hub` implements `upsert_*` / `ensure_*`
  (`ensureSnapshot`, `ensureExpertSpace`, `upsertRegion`, `upsertTransition`, `upsertGenome`) as
  conditional batches (lookup → `UPDATE`, else `AddN`/`AddE`). `Transition` edges carry
  `from_region_id`/`to_region_id` so an edge can be found again (`From::OutE<Transition>::WHERE`)
  and updated in place; `LatentRegion` has `status` (live | merged | retired) and
  `related_region_id` for the lifecycle marks (`markRegionStatus`). Plain `add*` queries rely on
  the unique indexes and fail with 409 `unique_constraint_violation` on duplicates.
- **Index DDL is asynchronous** and cannot share a batch with data writes; `fa-hub` waits for
  builds (`wait_for_index`) in its `*_ready` helpers.
- `NewsDoc` / `MediaSegment` embeddings use the same `vec<d>` convention, one un-partitioned
  cosine index per `(label, d)` created on first write (`add_news_docs` / `add_media_segments`,
  `search_news_docs` / `search_media_segments`); no ingestor writes them yet.
- **Dimensions:** `ExpertSpace.dim` is authoritative; `add_latents_checked` / `knn` reject other
  vector lengths. Index ensures are cached per client.
- **Version marker:** `N::SchemaMeta` (`meta_id = "fa-hub"`) holds the data-model version and an
  index-set fingerprint, stamped by `ensure_schema_ready` (`fa-hub::meta`, spec 20 §8).
- **Coverage:** `crates/fa-hub/tests/coverage.rs` fails when a `QUERY` here has no same-named
  function; `schema.rs` unit tests fail when an `INDEX` line has no index DDL.

## Running (D-033)
`just up` starts HelixDB from `ghcr.io/helixdb/helixdb:v0.0.14` on `127.0.0.1:6969`
(`GET /healthz`, `GET /readyz`, `GET /v2/diagnostics`, `POST /v2/query`). Data persists in the
`helix-data` volume. Verify the model end to end with
`FA_HELIX_URL=http://127.0.0.1:6969 cargo test -p fa-hub --test live`.

The `helix` CLI (`helix check` / `helix push dev`) is no longer part of the workflow.
