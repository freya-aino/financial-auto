# fa-hub

Typed Rust client for HelixDB's standalone server (`ghcr.io/helixdb/helixdb:v0.0.14`, compose
service `helix`, `http://127.0.0.1:6969`). It speaks `POST /v2/query`, the operation-tree
contract shared by the official Helix SDKs, and wraps every query of `helix/queries.hx` as a
typed function with the same name.

**Status:** implemented and verified against the live v0.0.14 image (unit tests + gated
integration tests, see below). Supersedes the D-012 assumption of CLI-compiled
`POST /<query_name>` endpoints (D-033).

## Owns
- `ast` – builder for operation trees (`Op`, `Pred`, `Batch`, `IndexSpec`, `Val`/`Input`).
- `schema` – labels, property names and the index DDL mirroring `helix/schema.hx`.
- `queries` – typed functions on `HubClient` (`add_snapshot`, `add_region`, `add_transition`,
  `search_latent`, `genome_parents`, … plus `region_graph`, `genome_ancestry`, `neighborhood`
  for spec 50 §4.4/§4.5). Idempotent writers for live services (D-048): `ensure_snapshot`,
  `ensure_expert_space` (get-or-create), `upsert_region`, `mark_region_status`,
  `upsert_transition(s)` (edge keyed by `from_region_id, to_region_id, sampler_id`),
  `snapshot_graph`, `resolve_node`; business-id variants for evo-worker: `upsert_genome`,
  `link_parent`, `add_fitness_record`, and `set_genome_status` (risk-engine's lifecycle mirror,
  D-076). Trainer (spec 36): `set_snapshot_manifest` (manifest JSON +
  `carried_mass` on the `Snapshot` node, read by control-api's promotion checklist),
  `add_adapter_state` / `snapshot_adapters` / `latest_adapter` (`AdapterState` node,
  `adapter_id = <snapshot_id>@<version>`, `HasAdapter` edge from the snapshot).
  Evolution edges: `link_trained_on` (`TrainedOn` genome → snapshot, upsert by `snapshot_id`;
  evo-worker's `evo.vary.decoder`), `link_feedback_from` (`FeedbackFrom{weight, output_family,
  score}` snapshot → elite decoder genome, upsert by `genome_id`; trainer, D-009),
  `genome_trained_on`, `snapshot_feedback`.
- `reference` – the Postgres `ref` mirror (spec 12 §5), written by control-api `ref_mirror.rs`:
  `upsert_ref_entities`, `upsert_ref_venues`, `upsert_ref_instruments` (nodes first, then each
  instrument's `IssuedBy` / `ListedOn` / `SameExposure` out-edges made equal to the given sets:
  stale edges dropped, current ones upserted by an edge key), `retire_ref_instruments`
  (`status = inactive`, out-edges dropped), `sync_reference(entities, venues, instruments, full)`
  (`full` retires mirrored instruments absent from the input), `ref_instrument_ids`,
  `instrument_reference`. 25 rows per write batch.
- `catalog` – the **whitelist** of read-only named queries behind `POST /api/graph/query`
  (`NAMED_QUERIES`, `HubClient::run_named(name, params)`); business-id parameters, no raw trees.
  Includes `instrumentReference`, `genomeTrainedOn` and `snapshotFeedback`.
- `HubClient::{new, with_timeout, from_env, url_from_env_or, health, ready, diagnostics, batch,
  query, ensure_schema, ensure_latent_index, wait_for_index, clear_caches}`.
- `vectors` – vector validation (`InvalidVector`: empty / zero-norm / non-finite), registered-dim
  checks (`space_dim`, `check_space_dim`, `DimMismatch`), `add_latents_checked` (validate + ensure
  index + write), `knn(space, v, k, &KnnFilter)` (kind / instrument / as-of / since / exclude /
  max-distance, over-fetched server-side filter; a dim without an index answers empty),
  `add_latents_batch` (prepared batch), text/media embeddings `add_news_docs`, `search_news_docs`,
  `add_media_segments`, `search_media_segments`. Vector-index ensures are cached per client
  (`ensure_latent_index_ready` makes no request once a dim is known active).
- `meta` – `SCHEMA_VERSION`, `MIGRATIONS`, `SchemaMeta` marker: `schema_marker`, `migrate_schema`
  (run by `ensure_schema_ready`; `SchemaTooNew` when Helix is ahead), `stamp_schema_version`.
- `asof` – spec 20 §6.3 variants: `list_snapshots_as_of`, `space_regions_as_of`,
  `snapshot_graph_as_of`, `space_latents_as_of`; `Transition` posteriors and region `status`
  are overlaid with their event-time versions (below).
- `versions` (D-078) – every `upsert_transitions` row and every region lifecycle mark appends a
  `TransitionVersion` / `RegionStatusVersion` node in the same write batch (`ts` = record time);
  `transition_versions`, `region_status_versions`, `transitions_as_of`, `regions_as_of`,
  `prune_space_versions`.
- `regions` (D-069) – `update_region_stats`, `mark_region_died`, `upsert_centroid`
  (`LatentVec kind=centroid`), `upsert_descents` (`DescendsFrom{carry|split|merge|partial}`),
  `upsert_hier_edges` (`Hier`), `upsert_region_set` / `region_set` (serialised online region set +
  kernel, restored on restart), `upsert_kernel_model` / `snapshot_kernels`, `upsert_cross_priors` /
  `cross_priors`, `snapshot_hierarchy`, `lineage_members`, `region_lineage`.
- `experts` (D-073) – `Expert` nodes (`expert_uid = <snapshot>/e<id>`), `ExpertOf`,
  `AlignedTo{cka}`: `upsert_expert`, `align_expert`, `retire_expert` (`died_gen`),
  `snapshot_experts`, `expert_parents`, `expert_alignments`.
- `retention` – `prune_space_observations`, `prune_snapshot_observations`, `prune_observations`
  with a `RetentionPolicy` (old `observation` latents folded into `kind=aggregate` rows per region;
  version rows older than the cut-off thinned to the one in force). model-runtime runs it on a
  schedule (`hub_retention.rs`, `--hub-prune-secs`).
- `buffer` – `WriteBuffer`: bounded drop-oldest queue of write batches with `flush` /
  `send_or_queue` and `BufferStats{queued, sent, dropped, rejected}` (spec 20 §8).

## Usage
```rust
use fa_hub::*;
use std::time::Duration;

let hub = HubClient::from_env();                       // FA_HELIX_URL or http://127.0.0.1:6969
hub.ensure_schema_ready(Duration::from_secs(30)).await?;   // unique/equality indexes, idempotent
hub.ensure_latent_index_ready(16, Duration::from_secs(30)).await?; // one vector index per dim

let snap = hub.add_snapshot(&NewSnapshot { /* … */ }).await?;      // -> Snapshot { id, … }
let space = hub.add_expert_space(snap.id, &NewExpertSpace { /* … */ }).await?;
let r1 = hub.add_region(space.id, &NewRegion { /* … */ }).await?;
let graph = hub.region_graph(space.id).await?;                      // regions + Transition edges
let hits = hub.search_latent(&space.expert_space_id, &query_vec, 20).await?; // k-NN, $distance asc

// Live-service writers (model-runtime `hub_sink`): idempotent across restarts.
let snap = hub.ensure_snapshot(&NewSnapshot { /* stat-v1-1m … */ }).await?;   // existing row returned as is
let space = hub.ensure_expert_space(snap.id, &NewExpertSpace { /* stat-v1-1m-e0 … */ }).await?;
let region = hub.upsert_region(space.id, &NewRegion { /* 1m-e0-r000 … */ }).await?; // revives a known id: status=live
hub.mark_region_status("1m-e0-r003", "retired", "1m-e0-r001").await?;      // merged | retired
hub.upsert_transitions(&[TransitionUpsert { from_region_id, to_region_id, transition }]).await?; // one write batch
let g = hub.snapshot_graph("stat-v1-1m").await?;                            // snapshot + spaces + regions + transitions
let hit = hub.resolve_node("1m-e0-r000").await?;                            // any unique business id, or a numeric $id

// evo-worker (business ids; D-048):
let g = hub.upsert_genome(&NewGenome { /* … */ }).await?;
hub.link_parent("parent-genome-id", "child-genome-id", "mutate").await?;
hub.add_fitness_record(&NewFitness { genome_id, generation, objectives, pareto_rank }).await?;

// Named queries (what control-api forwards from `POST /api/graph/query`):
let rows = hub.run_named("regionTransitions", &serde_json::json!({ "region_id": "1m-e0-r000" })).await?;

// Ad-hoc trees when no typed function fits:
let n = hub.query(Op::nodes_with_label("Snapshot").count()).await?;
let mut resp = hub.batch(&Batch::read()
    .var("src", Op::nodes(r1.id))
    .ret("next", Op::nodes(NodeRef::var("src")).out(Some("Transition")).value_map(None))).await?;
let next: Vec<LatentRegion> = resp.take("next")?;
```

## What the server needs declared
The server is **schemaless**: node/edge labels and property types are inferred on first write
(`N::`/`E::` blocks of `schema.hx` are documentation only). What must exist are **indexes**:
- `UNIQUE INDEX` / `INDEX` → `create_index{node_equality{label, property, unique}}`
  (`schema::static_indexes`, created by `ensure_schema`). Uniqueness is only enforced once the
  build has activated (`wait_for_receipts`); a duplicate then fails with HTTP 409
  `unique_constraint_violation`.
- `V::` vector types → `create_index{node_vector{label, property, dimension, metric,
  tenant_property}}`. An index is keyed by `(label, property)` with one fixed dimension, so
  latents of dimension `d` live in property `vec<d>` of label `LatentVec`
  (`schema::latent_vec_property`) and the index is partitioned by `expert_space_id`
  (`tenant_property`). `ensure_latent_index(d)` creates it; searches pass the expert-space id
  as `tenant_value`. This resolves the "mixed dims in one V type" gap noted in `helix/README.md`.
- Index DDL is **asynchronous** (`{"kind":"accepted","operation_id":…}`); poll with
  `get_index_operation` (`wait_for_index`). Re-creating an existing index returns
  `{"kind":"already_active"}`; a different definition for the same `(label, property)` is a
  409 `index_definition_conflict`.
- Concurrent DDL batches (several processes calling `ensure_schema` at boot) fail with 409
  `transaction_conflict` (`retryable: true`); `ensure_indexes` goes through `batch_retrying`,
  which retries retryable errors with backoff. Do not use it for non-idempotent data writes.
- DDL and data mutations **cannot share a write batch** (500 `internal_error: … mixes incompatible
  request transaction modes`); `ensure_*` run DDL-only batches.
- **Upserts** are conditional batches (lookup → `set_property`, else `add_n`/`add_e`, entries
  guarded by `var_empty`/`var_not_empty`). An `add_e` whose **target variable is empty is a
  server error** (500 `invalid_query: addE() requires at least one target vertex`), not an empty
  result, so edge creation is gated on a probe entry that is non-empty only when the target
  exists (`upsert_transitions`, `link_parent`). `set_property` works on edge streams
  (`edges{var}` → `set_property` → `edge_properties`).

## Wire contract (verified against v0.0.14)
Request: `POST /v2/query`, JSON. `request_type` must match the batch variant; entries are named
trees, `returns` selects what comes back, `parameters`/`parameter_types` are optional.
```json
{"request_type":"write","query_name":null,"query":{"write":{"entries":[
  {"query":{"name":"n","root":{"add_n":{"label":"Snapshot","properties":[
     ["snapshot_id",{"value":{"string":"s1"}}],
     ["generation",{"value":{"i64":3}}],
     ["created_at",{"expr":"date_time_now"}]]}}}},
  {"query":{"name":"row","root":{"value_map":{"input":{"nodes":{"reference":{"var":"n"}}}}}}}
],"returns":["row"]}}}
```
Response `200`:
```json
{"row":[{"$id":6,"$label":"Snapshot","snapshot_id":"s1","generation":3,"created_at":"2026-11-08T17:51:51.867Z"}]}
```
Read with a label scan, a filter and a traversal (`$label` is a virtual property):
```json
{"request_type":"read","query":{"read":{"entries":[
  {"query":{"name":"src","root":{"nodes_where":{"predicate":{"and":{"predicates":[
     {"eq":{"left":{"property":"$label"},"right":{"constant":{"string":"LatentRegion"}}}},
     {"eq":{"left":{"property":"region_id"},"right":{"param":"rid"}}}]}}}}}},
  {"query":{"name":"edges","root":{"edge_properties":{"input":{"out_e":{"input":{"nodes":{"reference":{"var":"src"}}},"label":"Transition"}}}}}}
],"returns":["edges"]}},"parameters":{"rid":"r1"}}
```
→ `{"edges":[{"$id":0,"$label":"Transition","$from":2,"$to":3,"count":5,"mean":0.25}]}`

Vector search (index `LatentVec.vec3`, tenant `expert_space_id`):
```json
{"request_type":"read","query":{"read":{"entries":[{"query":{"name":"hits","root":{"value_map":{
  "input":{"vector_search_nodes":{"label":"LatentVec","property":"vec3",
           "tenant_value":{"value":{"string":"sp1"}},
           "query_vector":{"value":{"f32_array":[1,0,0]}},"k":{"literal":5}}},
  "properties":["$id","$distance","ref_id"]}}}}],"returns":["hits"]}}}
```
→ `{"hits":[{"$distance":0.0,"$id":2,"ref_id":"w1"},{"$distance":0.00306,"$id":3,"ref_id":"w2"}]}`

Shapes:
- A bare node/edge stream returns `[{"$id": n}, …]`; use `value_map` (nodes: `$id`, `$label`,
  properties; vector hits add `$distance`) or `edge_properties` (adds `$from`, `$to`) for rows.
  `count` → number, `exists` → bool, DDL → receipt object.
- Request values are tagged (`{"string":…}`, `{"i64":…}`, `{"f64":…}`, `{"bool":…}`,
  `{"date_time": unix_ms}`, `{"f32_array":[…]}`, `{"f64_array":[…]}`, `{"string_array":[…]}`),
  wrapped as `{"value": …}` in property/input position or `{"constant": …}` in expressions;
  parameters are `{"expr":{"param":"name"}}` / `{"param":"name"}`. Response values are plain JSON;
  `date_time` renders as RFC 3339. A **typed** `date_time` parameter is validated but stored as a
  string, so fa-hub writes dates as inline `{"date_time": ms}` literals.
- Conditional entries: `"condition":{"var_not_empty":"x"}` / `{"var_empty":"x"}` (used for upserts).
- Errors: non-2xx with `{"error":"<code>","msg":"…","retryable":bool?}` — e.g. 400
  `invalid_query_json`, 404 `index_not_found`, 409 `unique_constraint_violation` /
  `index_definition_conflict`, 503 `writer_mode_required`. `HubError::Server` carries them.
- Optional headers: `x-helix-await-durable: false` (don't wait for WAL flush),
  `x-helix-require-writer`, `x-helix-warm` (reads). Not used yet.

Also available but unused: `GET /healthz`, `GET /readyz` (503 while indexes catch up),
`GET /v2/diagnostics`, and `POST /v2/cypher` / `/v2/sql` text endpoints.

## Who writes what (D-048)
| Writer | Rows |
|---|---|
| `bins/model-runtime` (`hub_sink`, `hub_retention`) | `Snapshot stat-v1-<ts>`, `ExpertSpace <snapshot>-e<k>`, `LatentRegion` (+ `status`, statistics, death marks), `Transition` (coalesced) + `TransitionVersion`, `RegionStatusVersion`, `LatentVec kind=observation` for 1m and slower and `kind=centroid`, `DescendsFrom`, `Hier`, `RegionSet`, `KernelModel`, `CrossPrior`; prunes observations and versions every 6 h |
| `bins/evo-worker` (`vary`, `vary_strategy`, `allocator`) | `Genome` via `upsert_genome`, `ParentOf` via `link_parent`, `FitnessRecord`+`Scored` via `add_fitness_record` |
| `bins/risk-engine` (`pipeline`) | `Genome` lifecycle status via `set_genome_status` (mirrors `ledger.strategy_state`) |
| `bins/trainer` | `Snapshot` candidates (+ `manifest`, `carried_mass`), `ExpertSpace`, `LatentRegion`, `SnapshotParent`, `RegionMatch`, `AdapterState` + `HasAdapter`, `Expert` + `ExpertOf`/`AlignedTo` for neural snapshots |
| `bins/control-api` (`graph_routes`, `latent_routes`, `mcmc_routes`, `evolution_routes`) | reads (`list_snapshots`, `snapshot_graph`, `resolve_node` + `neighborhood`, `genome_ancestry`, `run_named`, region lineage, kernels, hierarchy, cross priors) plus `set_snapshot_status` / `set_snapshot_previous_active` on promote / rollback |

## Tests
```sh
cargo test -p fa-hub                                   # unit tests + tests/coverage.rs (every queries.hx QUERY has a function)
FA_HELIX_URL=http://127.0.0.1:6969 cargo test -p fa-hub --test live            # against `just up`'s helix
FA_HELIX_URL=http://127.0.0.1:6969 cargo test -p fa-hub --test regions_live    # region lifecycle, lineage, kernels
FA_HELIX_URL=http://127.0.0.1:6969 cargo test -p fa-hub --test versions_live   # event-time versions and as-of overlay
```
Concurrent writers: `HubClient::batch` retries `409 transaction_conflict` (an aborted, uncommitted transaction) up to six times with a short backoff, so parallel writers and the parallel live suite do not fail spuriously.
The live tests create `test-<nanos>-…` rows and drop them afterwards; the indexes created by
`ensure_schema` and the `vec4` latent index remain. `live.rs` covers the typed queries, the
idempotent `ensure_*`/`upsert_*` writers, lifecycle marks, `snapshot_graph`, `resolve_node`, the
named-query catalog, the evo-worker business-id writers and the `AdapterState` round trip
(`adapter_state_round_trip_and_snapshot_manifest`).

## Governing specs
- [20-storage](../../specs/20-storage.md) §5.2, [ADR 0002](../../specs/adr/0002-helixdb-hub.md), D-012, D-033, D-043, D-048, D-067, D-069, D-073, D-078, D-084
- [helix/](../../helix/README.md) – the readable catalog this crate mirrors
