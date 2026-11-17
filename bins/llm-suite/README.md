# llm-suite

The platform's LLM service (D-117): news embeddings for the encoder's text expert, and the operator
agent the UI reaches at `/api/llm/*` through control-api. One OpenAI-compatible endpoint serves
both ([`fa-llm`](../../crates/fa-llm)).

**Status:** implemented — the embedder runs live on `news.item.>`, the agent loop answers with the
platform's own control API as its tools, external MCP servers can be registered from the UI
(D-120), and the encoder's text expert reads these vectors (`stat-v2`, D-119).

## Embedder

Every `news.item.>` envelope is embedded **once, on arrival**, and the vector is stored with the
item's own knowledge time (`fa_hub::vectors::add_news_docs` → Qdrant `NewsDoc`). The encoder then
reads stored vectors as of a window end and never calls an API on the model path, which is what
keeps spec 41's replay equivalence possible: an endpoint cannot be asked to return the same vector
twice.

| Property | How |
|---|---|
| Identity | `dedup_key` when the source sets one, else `news-<16 hex>` of source + text — content-addressed, so a republished item embeds once |
| Text | title + body, cut to 4 000 bytes on a char boundary; a `body_ref`-only item is embedded from its title rather than skipped (a headline is most of the signal, and fetching the blob would put object storage on the ingest path) |
| Batching | 32 items per request, or whatever is queued after a 2 s linger |
| Backpressure | bounded queue (`FA_LLM_QUEUE_CAP`, 4096); over it the **oldest** item is dropped and counted — a backlog costs old headlines, not the live path |
| Across restarts | `llm.news_embedding` holds one row per embedded item, so a restart neither re-pays nor writes a second vector |
| No endpoint, or a failed call | deterministic hashed byte-4-gram embedding, `provider = 'hashed'` in the row — the text expert exists before anyone enters a key, and a latent built on fallbacks is visible rather than quietly different |

`GET /api/llm/embeddings` (control-api) reports the mix: counts per provider and the fallback share.

## Agent

An agentic loop over control-api's routes — never a store directly, so the request audit, the
single-writer rule and every route's own refusals apply to the agent exactly as to an operator.

- **Tools** (`tools.rs`): 15 read tools (health, services, sources, risk, positions, P&L,
  strategies, orders, snapshots, regions, transitions, generations, jobs, decoder outputs, audit)
  and 4 mutating ones (`ingest_control`, `start_generation`, `cancel_job`, `strategy_action`).
- **Not tools, by construction:** promotion and rollback, trading mode, halt re-arming, the kill
  switch, flattening, payouts. A test asserts that no tool name or path can reach those routes.
- **Approval:** a mutating call **ends the turn**. The response carries the pending call, the UI
  shows it with its arguments, and the operator's approval is tied to *those* arguments — approving
  a stop of one connector cannot start another.
- **Live mode:** while `trading.mode` is `live` every mutating call is refused outright; reads stay.
- **Bounds:** `MAX_ROUNDS` = 6 tool rounds per turn, so a model looping on a failing tool costs a
  bounded number of calls.
- **Persistence:** conversations, messages and every tool call — refusals included — in the `llm`
  schema. A refusal is the more interesting row: it records what the agent wanted to do and was not
  allowed to.

## External MCP servers (D-120)

`mcp.rs` is a JSON-RPC client over the **streamable-HTTP** transport (`initialize`, `tools/list`,
`tools/call`) plus the registry that merges a server's tools into the agent's catalogue under
`mcp__<server>__<tool>`. Registrations live in `llm.mcp_server` and are managed from the agent tab:
`GET /api/llm/mcp`, `PUT /api/llm/mcp/{id}`, `DELETE /api/llm/mcp/{id}`,
`POST /api/llm/mcp/{id}/probe`.

| Property | How |
|---|---|
| Authority | the **operator** classifies the server `read` or `mutating`; `mutating` is the default and means every call needs per-call approval and none is possible in live mode |
| `annotations.readOnlyHint` | reported per tool in the UI as a *claim*, never used to grant authority — a server that wants to be trusted will set it |
| Same gate | an MCP call resolves through the same `tools::Caller::call`, so the approval rule, the live interlock and the `llm.tool_call` row apply unchanged |
| Results | flattened to text, cut at 16 KiB, handed to the model as a tool result under an `external` key; the system prompt says that what a server returns is data and never an instruction |
| Bounds | 48 tools per server (a catalogue of hundreds would crowd out the platform's own), 8 `tools/list` pages, 20 s probe / 60 s call timeouts |
| Probing | at start, after every registration change, and every 5 minutes — a server that restarts or drops a tool stops being offered it |
| Sessions | each call handshakes its own session, which costs a round trip and makes a restarted or expired server work anyway |
| Token | `PUT` accepts a bearer token, stored in `llm.mcp_server.token`; no route returns it and `GET` reports only `token_set` |
| `enabled = false` | keeps the registration and removes its tools from the catalogue — the cheap way to answer "was it that server?" |

Only streamable HTTP: stdio would mean the suite spawning processes inside its own container, a
much larger trust decision than calling a URL, and a local stdio server can sit behind one of the
HTTP bridges.

Unlike the endpoint key, the token is **not** in `config/secrets.toml`. That file is bind-mounted
into the service read-only and per inode, so an atomic rewrite of it is only picked up on restart —
correct for a key entered once, wrong for a server the operator registers and expects to work
immediately.

## Configuration

| Setting | Where |
|---|---|
| `base_url`, `model`, `embedding_model`, `embedding_dim` | `[llm]` in the platform config; `config/runtime.toml` when the UI writes it; `FA_LLM_BASE_URL` / `FA_LLM_MODEL` / `FA_LLM_EMBEDDING_MODEL` / `FA_LLM_EMBEDDING_DIM` override |
| `api_key` | `[llm] api_key` in the git-ignored `config/secrets.toml`, or `FA_LLM_API_KEY`. Written by `PUT /api/llm/endpoint`, never returned by any route |
| `FA_LLM_BIND` | this service's API (default `0.0.0.0:7710`); control-api proxies it, the UI never reaches it directly |
| `FA_CONTROL_API` | where the agent's tools call (default `http://control-api:7700`) |

A base URL without a key is allowed: local servers (llama.cpp, vLLM, Ollama's OpenAI shim) take
none, and requiring one would tie the suite to a paid provider.

The suite reads its configuration at start, so a change through the UI needs a restart
(`POST /api/services/llm-suite/restart`); the route's answer says so.

## Metrics

`/metrics` on `:9100` in compose (host `:9109`), scraped as `service="llm-suite"`:
`fa_llm_requests_total{kind,outcome}`, `fa_llm_tokens_total{kind}`, `fa_llm_request_seconds{kind}`,
plus the shared heartbeat counters.
