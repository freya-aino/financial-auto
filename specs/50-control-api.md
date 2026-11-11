# 50 — Control API

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium |
| **Owners (code)** | `bins/control-api` (Rust, axum) |
| **Depends on** | 20 Storage, 21 Bus & contracts, 40 Execution & risk, 42 Ledger, 60 Ops |

## 1. Purpose
A single HTTP + WebSocket surface through which the frontend (51) — and scripts — observe and operate the whole platform: services, sources, bus, graph/latent state, evolution, and trading. It aggregates the stores and the bus so the frontend never talks to NATS, Helix, Postgres, VictoriaMetrics or Garage directly.

## 2. Scope
**In scope:** REST endpoints, WebSocket streams, command dispatch to services (via NATS request/reply), read aggregation from stores, OpenAPI document.
**Out of scope:** authentication/authorization (local-only, ADR 0006), business logic (lives in owning crates/services).

## 3. Decisions
- **Rust + axum**, REST (JSON) + WebSocket.
- **No auth**, bound to `127.0.0.1` by default (ADR 0006).
- Frontend dev server proxies `/api` and `/ws` to control-api.
- Commands to services go over NATS request/reply on `ctl.<service>.<command>`; control-api does not hold business state.
- Every mutating endpoint writes an `audit_log` entry (42).

## 4. Interfaces
Conventions: `/api/v1/...` **(proposal: version prefix)**; JSON; time as RFC3339 UTC; ids as UUID strings; list endpoints paginate with `?cursor=&limit=`; OpenAPI served at `/api/openapi.json`. **Implemented (round 3, `bins/control-api/src/openapi.rs`):** OpenAPI 3.1 **generated from the route declarations** instead of `utoipa` annotations — every route module's source is embedded at compile time and its `.route("<path>", get(h).post(h2)…)` calls are parsed (test modules excluded), so no route can exist without an operation; real handlers win over the `routes.rs` 501 placeholders (later `install`ed routers layer in front), path parameters are declared, summaries come from the modules' doc tables (`| \`GET /api/x\` | … |`) or the handler name, tags = module, `x-handler` names the Rust function. Bodies are documented as generic JSON objects (the Rust handler is the schema of record). 113 paths / 121 operations at the time of writing.

### 4.1 Ops
| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | control-api + dependency health (NATS, Helix, Postgres, VM, Garage) |
| GET | `/api/services` | registered services, heartbeat, version, status |
| POST | `/api/services/{name}/start\|stop\|restart` | via compose/process supervisor **(proposal)** |
| GET/PUT | `/api/config/{section}` | read config; PUT only for runtime-tunable keys |
| GET | `/api/jobs` | work-queue jobs (eval, training, backfill) with status |
| POST | `/api/jobs/{id}/cancel` | |
| GET | `/api/gpu` | GPU queue: running job, priority classes, utilisation |

**Implemented today (D-060):** `POST /api/services/{name}/start|stop|restart` supervises compose services through the Docker Engine API (socket mounted into control-api only, project label filter). Infra services and control-api itself are refused with 409, `risk-engine` stop/restart needs `{"confirm": true}` (428 otherwise), every attempt is audited, and without a socket the routes answer `501 no_supervisor`. This closes open question 1 for compose deployments; spec 50 has no 501 routes left.

### 4.2 Sources & ingestion
| GET | `/api/sources` | catalog (11) with status, rate-limit usage, last event |
| GET | `/api/sources/{id}` | detail, recent errors, coverage |
| POST | `/api/sources/{id}/enable\|disable` | |
| POST | `/api/sources/{id}/backfill` | body: instruments, timescale, range |
| GET | `/api/instruments?q=` | reference data search (12) |

### 4.3 Bus tap
| WS | `/ws/bus?subjects=md.>,risk.event` | live tap of NATS subjects (decoded proto → JSON), server-side sampling/rate limit |
| GET | `/api/bus/streams` | JetStream streams, consumers, lag |

`/api/bus/streams` lists every stream (spec 21 §5 limits streams incl. `DLQ`, the `JOBS_<kind>` work queues incl. `JOBS_MCMC`) with `max_bytes` (the applied cap, `-1` = none), `planned_max_bytes` (the spec 21 §5 `max_mib` from `fa_bus::streams::table()`), `cap_applied` (false when `ensure_streams` kept a larger cap rather than purge) and `used_ratio` (bytes / cap; the UI's Streams panel shows usage against the cap), messages, bytes, sequences and per-consumer lag. control-api also runs the dead-letter forwarder (spec 21 §8): poison messages land on `dlq.<STREAM>.<subject>` and raise a `DeadLetter` alert on `ops.alert` (seen on the bus tap) and `fa_bus_dead_letters_total{stream,consumer,reason}`; `GET /api/alerts` lists vmalert alerts only, so a vmalert rule on that counter is the way to surface dead letters there.

### 4.4 Hub / graph
| GET | `/api/graph/neighborhood?node=&depth=` | Helix subgraph around a node |
| POST | `/api/graph/query` | run a named query from the Helix query catalog with params |
| GET | `/api/graph/lineage/{genome_id}` | ancestry tree |

### 4.5 Latent / MCMC
| GET | `/api/latent/snapshots` | encoder snapshots per timescale, status (active/candidate/retired) |
| GET | `/api/latent/{snapshot}/experts` | experts, routing stats, residual covariance matrix; `health` = the latest `latent.expert.health` of the snapshot (K_eff, residual correlation matrix, per-expert skill/usage on served windows; control-api's live collector keeps the last one per snapshot) |
| GET | `/api/latent/{snapshot}/projection?expert=&range=` | 2D/3D projection points (downsampled) |
| GET | `/api/latent/{snapshot}/regions` | regions with occupancy, stats |
| GET | `/api/mcmc/{snapshot}/graph?timescale=` | region graph with transition posteriors |
| WS | `/ws/latent?snapshot=&instruments=` | live current state / region per instrument |
| POST | `/api/latent/snapshots/{id}/promote\|rollback` | |

### 4.6 Evolution
| GET | `/api/evolution/populations` | decoder / strategy / allocator populations, generation |
| GET | `/api/evolution/populations/{id}/pareto?gen=` | Pareto front points + objectives |
| GET | `/api/evolution/genomes/{id}` | genome, fitness, eval reports (41) |
| GET | `/api/evolution/genomes/{id}/outputs` | decoder outputs (image URLs via Garage presign, text, forecasts) |
| POST | `/api/evolution/runs` | start generation with budget (GPU-hours); body as `POST /api/evolution/generations`: `kind`, `genomes`, `eta`, `min_budget_ms`, `max_budget_ms`, `max_generations`, `total_budget_ms`, `data_window_hours`, `mcmc_pairs` (cross-asset src→target genomes per decoder loop, D-074; absent = orchestrator default 4, 0 disables) forwarded as `ctl.evo.start` args |
| POST | `/api/evolution/runs/{id}/pause\|resume\|stop` | |
| WS | `/ws/evolution` | generation progress, survivors per halving round |

### 4.7 Trading
| GET | `/api/trading/positions?book=` | |
| GET | `/api/trading/orders`, `/fills`, `/pnl?range=&strategy=` | |
| GET | `/api/trading/strategies` | status, sleeve, shadow progress, gate status |
| POST | `/api/trading/strategies/{id}/pause\|resume\|flatten\|promote` | promote rejected unless gate green |
| POST | `/api/trading/orders` | manual order (goes through risk chain, 40) |
| POST | `/api/trading/flatten`, `/api/trading/kill-switch`, `/api/trading/kill-switch/rearm` | |
| GET/PUT | `/api/trading/limits` | PUT may only tighten |
| GET | `/api/trading/reconciliation` | latest runs + diffs |
| GET | `/api/audit?since=` | audit log |
| WS | `/ws/trading` | positions, P&L, risk events, fills |

### 4.8 WebSocket protocol
Envelope: `{ "stream": "...", "seq": n, "ts": "...", "data": {...} }`. Client may send `{ "op": "subscribe" | "unsubscribe", "topics": [...] }`. Server applies backpressure by dropping to latest-value for high-rate topics and signals `{ "op": "lagged", "dropped": n }`.

**Implemented today (D-055):** every route of §4 is implemented except `POST /api/services/{name}/start|stop|restart` (no process supervisor). The four sockets share one loop: `{stream, seq, ts, topic, data}` frames, `{op: subscribe|unsubscribe, topics}` filtering, latest-value coalescing for state streams (10 Hz latent/evolution, 5 Hz trading) and a bounded queue for the bus tap, `{op:"lagged",dropped}` after any drop. `PUT /api/config/runtime` is the only writable config section (bounded, persisted to git-ignored `config/runtime.toml`, audited). Routes are listed with their state in `bins/control-api/README.md`.

### 4.9 Alerts, MCMC queries, neural models, pipeline, halts, reference data (D-066..D-071)
| Method | Path | Description |
|---|---|---|
| GET | `/api/alerts?limit=` | active alerts, recent transitions, alert-pipeline health (`pipeline.ok` = Watchdog fresh; false is itself an alert) |
| POST | `/api/alerts/vmalert/api/v2/alerts`, `/api/alerts/webhook` | vmalert / Alertmanager receivers → `ops.alert` (JSON) + `audit.log` |
| GET | `/api/mcmc/{snapshot}/kernels` | kernel per expert space: serving sampler, prequential scores, stationary, relaxation, drift |
| GET | `/api/mcmc/{snapshot}/transition\|k-step\|neighbors\|paths\|path-prob\|hitting\|stationary` | spec 33 §4.2 queries, each a posterior `{mean, sd, lower, upper, median, level, n}` |
| GET | `/api/mcmc/{snapshot}/hierarchy`, `/api/mcmc/cross-priors?timescale=&limit=` | REFINES edges; ranked cross-asset priors (gain, lead-lag and score on expert residuals, spec 33 §6.5) |
| GET | `/api/latent/regions/{id}/lineage?depth=` | ancestors, descendants, lineage members (DescendsFrom) |
| GET | `/api/latent/snapshots/{id}` | snapshot, manifest, promotion checklist (incl. neural `ssl_regression`, `expert_k_eff`, `expert_resid_corr`, `expert_min_skill`), not-evaluated items (items now reported as checks are dropped from it), shadow gauges. A trained snapshot always carries `ssl_regression`: the parent-vs-candidate mixture-skill regression when the parent is neural, else an n/a pass (`value: null`, the detail names the statistical parent — there is no SSL objective to regress against) |
| POST/DELETE | `/api/latent/snapshots/{id}/shadow`, `POST .../shadow/stop` | shadow serving via `ctl.model` (audited) |
| GET | `/api/decoders/serving` | `ctl.decoders.status`: slots per (family, timescale), serving vs shadow, online n / loss, pins, disabled |
| GET | `/api/decoders/cards?limit=` | decoder cards in `fa-models/decoders/**/*.card.json`, newest first |
| POST | `/api/decoders/reload\|promote\|pin\|unpin\|off\|on` | `ctl.decoders.<verb>` with the JSON body as args; audited `decoders.<verb>` |
| GET | `/api/trading/pipeline`, POST `/api/trading/pipeline/run` | `ctl.risk.pipeline {status\|run}` (spec 35 §6.4 round: holdouts, actions, rejections, live set, policy) |
| GET | `/api/trading/allocators` | `ctl.risk.allocators`: live paper allocator tournament |
| GET | `/api/trading/halts` | halts persisted in `ledger.risk_state` merged with those seen on `risk.halt.*`; `rearmable` per scope |
| POST | `/api/trading/halts/{scope}/rearm` | `{"confirm": true, "reason"?}` → `fa.v1.Resume` on `risk.resume.<scope>` for any family risk-engine re-arms (portfolio, daily_loss, source[:id], model, broker, fx, …); 428 without confirm, 400 unknown family, 404 not halted; audited `halt.rearm` / `halt.rearm.refused` |
| GET | `/api/trading/risk-events?since=&limit=&kind=` | `RiskEvent`s, halts, resumes, re-arms, reconciliation diffs from `audit.log`, newest first, with counts |
| GET | `/api/refdata/instruments/{id}`, `/resolve`, `/session`, `/fx` (`at`, `known_at`) | point-in-time reference data (spec 12) |
| GET | `/api/refdata/venues`, `/api/refdata/calendar/{venue}?from=&to=` | venues with calendar rule and materialised range; sessions per day (`table` row, else built-in `rule`) |
| GET/POST | `/api/refdata/corporate-actions` (`?instrument_id=&known_at=`), POST `.../{id}/cancel`, `/symbol-change`, `/identifiers`, `/contract-specs`, `/calendar/{venue}/materialize` | corporate actions (as-of when an instrument is given) and audited overrides |
| GET | `/api/trading/lots`, `/statement?format=json\|csv`, `/pnl/breakdown`, `/orphans`; POST `/orphans/{id}/resolve`, `/payouts` | ledger reports (spec 42) |

The bus tap (`/ws/bus`) decodes `ops.alert` (JSON), `latent.membership.*`, `latent.lifecycle.*`, `latent.mcmc.*`, `latent.hier.*` and `latent.crossprior.*` (proto) and passes any other JSON payload through.

## 5. Data model
Stateless except a small cache. Reads: Postgres (catalog, ledger, audit, jobs), Helix (graph, latent, lineage), VictoriaMetrics (series via PromQL/MetricsQL passthrough `/api/metrics/query` **(proposal)**), Garage (presigned URLs for blobs).

## 6. Behaviour & algorithms
- Requests that mutate state are translated to NATS `ctl.*` requests with timeout; the owning service validates and replies; control-api returns its result verbatim.
- Large payloads (projections, graphs) are downsampled server-side to configured point budgets.
- Health aggregation polls dependencies every few seconds and caches the result.

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `api.bind` | 127.0.0.1:8080 | |
| `api.ctl_timeout` | 5 s | |
| `api.ws.max_rate_per_topic` | 20 msg/s | then latest-value |
| `api.projection.max_points` | 50 000 | |

## 8. Failure modes
| Failure | Response |
|---|---|
| Dependency down | endpoint returns 503 with dependency name; `/api/health` shows degraded |
| Owning service doesn't reply | 504; command not assumed applied |
| WS client slow | latest-value coalescing, `lagged` notice |

## 9. Acceptance criteria
- OpenAPI document covers every REST endpoint.

**Status (round 3):** generated from the route declarations, so it covers every endpoint by construction (`document_covers_the_known_routes`); the frontend's committed copy `frontend/src/lib/openapi.json` is checked against the generated document (`openapi_snapshot_is_current`; `FA_UPDATE_OPENAPI=1 cargo test -p control-api openapi` rewrites it) and every `/api/…` call of the frontend against it (vitest `openapi.test.ts`). Request/response schemas are generic objects — typed schemas would need annotating ~120 handlers.
- Every mutating endpoint creates an audit entry.
- Limit loosening via `PUT /api/trading/limits` returns 409.
- Bus tap streams decoded events for any subject pattern at bounded rate.
- control-api restart does not affect trading (no business state).

## 10. Open questions
1. Service start/stop: via Docker API, a process supervisor, or out of scope (compose only)? — blocks 4.1 service control.
2. Version prefix `/api/v1` now or later.
3. Typed client generation for the frontend (OpenAPI → TS) vs hand-written. *Decided (round 3): hand-written fetch layer (`frontend/src/lib/{api,console}.ts`, typed by hand) checked against the generated OpenAPI document: a vitest scans every `/api/…` literal in the frontend sources and fails on a path or method the document does not have. A generated TS client would only add value once the document carries real body schemas.*

## 11. Deferred
- Auth, TLS, multi-user (local only).
- Remote access.
