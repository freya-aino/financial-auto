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
Conventions: `/api/v1/...` **(proposal: version prefix)**; JSON; time as RFC3339 UTC; ids as UUID strings; list endpoints paginate with `?cursor=&limit=`; OpenAPI served at `/api/openapi.json` (generated via `utoipa` **(proposal)**).

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

### 4.2 Sources & ingestion
| GET | `/api/sources` | catalog (11) with status, rate-limit usage, last event |
| GET | `/api/sources/{id}` | detail, recent errors, coverage |
| POST | `/api/sources/{id}/enable\|disable` | |
| POST | `/api/sources/{id}/backfill` | body: instruments, timescale, range |
| GET | `/api/instruments?q=` | reference data search (12) |

### 4.3 Bus tap
| WS | `/ws/bus?subjects=md.>,risk.event` | live tap of NATS subjects (decoded proto → JSON), server-side sampling/rate limit |
| GET | `/api/bus/streams` | JetStream streams, consumers, lag |

### 4.4 Hub / graph
| GET | `/api/graph/neighborhood?node=&depth=` | Helix subgraph around a node |
| POST | `/api/graph/query` | run a named query from the Helix query catalog with params |
| GET | `/api/graph/lineage/{genome_id}` | ancestry tree |

### 4.5 Latent / MCMC
| GET | `/api/latent/snapshots` | encoder snapshots per timescale, status (active/candidate/retired) |
| GET | `/api/latent/{snapshot}/experts` | experts, routing stats, residual covariance matrix |
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
| POST | `/api/evolution/runs` | start generation with budget (GPU-hours) |
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
- Every mutating endpoint creates an audit entry.
- Limit loosening via `PUT /api/trading/limits` returns 409.
- Bus tap streams decoded events for any subject pattern at bounded rate.
- control-api restart does not affect trading (no business state).

## 10. Open questions
1. Service start/stop: via Docker API, a process supervisor, or out of scope (compose only)? — blocks 4.1 service control.
2. Version prefix `/api/v1` now or later.
3. Typed client generation for the frontend (OpenAPI → TS) vs hand-written.

## 11. Deferred
- Auth, TLS, multi-user (local only).
- Remote access.
