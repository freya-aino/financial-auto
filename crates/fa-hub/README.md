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
  for spec 50 §4.4/§4.5).
- `HubClient::{health, ready, diagnostics, batch, query, ensure_schema, ensure_latent_index,
  wait_for_index}`.

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

## Tests
```sh
cargo test -p fa-hub                                   # unit tests (JSON shapes, parsing)
FA_HELIX_URL=http://127.0.0.1:6969 cargo test -p fa-hub --test live   # against `just up`'s helix
```
The live tests create `test-<nanos>-…` rows and drop them afterwards; the indexes created by
`ensure_schema` and the `vec4` latent index remain.

## Governing specs
- [20-storage](../../specs/20-storage.md) §5.2, [ADR 0002](../../specs/adr/0002-helixdb-hub.md), D-012, D-033
- [helix/](../../helix/README.md) – the readable catalog this crate mirrors
