# control-api

axum REST + WebSocket control plane for the frontend. Local only, no auth (ADR 0006).

**Status:** ops, sources, bus tap, latent views, evolution and paper trading are implemented and verified against live NATS and the compose services (2026-11-07); ingest start/stop and job cancel verified against host-run Go services (2026-11-08). The live views are in-memory aggregations of bus traffic (D-032); the paper book is seeded from the persisted `paper` ledger in Postgres at startup (D-047; it still accumulates bus fills in memory afterwards, D-028) and `GET /api/trading/pnl`, `/api/trading/reconciliation` and `/api/audit` read Postgres directly (503 when it is unreachable). Live trading is impossible through this API (D-029).

## Run
```sh
cargo run -p control-api            # binds control_api.bind from config/platform.toml (127.0.0.1:7700)
cargo run -p control-api --example fake_publisher   # fake heartbeats + trades for UI work
cargo run -p control-api --example demo_publisher   # synthetic latent/decoder/jobs/signals/risk/trades for every view
```
Flags / env: `--config` / `FA_CONFIG` (default `config/platform.toml`), `--sources` / `FA_SOURCES` (default `config/sources.yaml`), `--frontend` / `FA_FRONTEND` (default `frontend/build`, served at `/` when present, SPA fallback to `index.html`).

Starts even when NATS is down: the client retries in the background, and `/api/health` reports `nats.connected: false`.

## Endpoints
| Route | State |
|---|---|
| `GET /api/health` | implemented: NATS connection state + TCP reachability of Helix, VictoriaMetrics, Postgres, Garage |
| `GET /api/services` | implemented: latest `Heartbeat` per service from `svc.heartbeat.>` |
| `GET /api/sources`, `GET /api/sources/{id}` | implemented: reads `config/sources.yaml` per request (Postgres-backed catalog comes later, spec 11) |
| `GET /api/config/timescales` | implemented |
| `WS /ws/bus?subjects=a,b` (or `?subject=`) | implemented: frames `{stream, seq, ts, data: {subject, size, decoded}}`; decodes `Heartbeat` and canonical `Envelope`s; max 200 msg/s per socket, overflow reported as `{op: "lagged", dropped}` |
| `GET /api/bus/streams` | implemented: `{streams: [{name, subjects, retention, storage, max_age_secs, messages, bytes, first_seq, last_seq, consumer_count, consumers: [{name, durable, filter_subjects, ack_pending, num_pending, redelivered, waiting, delivered_seq, ack_floor_seq}]}]}` from `fa_bus::streams::list_streams`; 503 without NATS. control-api also runs `ensure_streams` (spec 21 §5) once NATS connects |
| `GET /api/latent/timescales` | implemented: configured ladder + timescales with data |
| `GET /api/latent/timescales/{ts}/states\|regions\|transitions` | implemented: latest `LatentState` per instrument; region registry with lineage, status, occupancy; region graph `{nodes, edges, rows}` with posterior mean and interval |
| `GET /api/graph/neighborhood?node=&depth=` | implemented from HelixDB (D-048): `node` = numeric `$id` or any unique business id (region / snapshot / expert space / genome / instrument); `depth` 1–4 (default 1); `{center, depth, nodes, edges}`; 404 unknown node |
| `GET /api/graph/query`, `POST /api/graph/query` | implemented: GET lists the named-query whitelist (`fa_hub::catalog`); POST `{name, params}` runs one of them (business-id params, never raw operation trees) → `{name, result}`; 404 `unknown_query` (with `known`), 400 `bad_request`, 404 `not_found` |
| `GET /api/graph/lineage/{genome_id}` | implemented: `{genome, ancestors_by_depth[][], fitness[]}` from `ParentOf` / `Scored`; 404 unknown genome |
| `GET /api/latent/snapshots` | implemented from HelixDB: `{snapshots[], by_timescale{}}` (rows written by model-runtime's hub sink: `stat-v1-<ts>`, status `active`) |
| `GET /api/latent/{snapshot}/regions` | implemented from HelixDB: `{snapshot, spaces[], regions[], status_counts{live,merged,retired}, n_transitions}`; 404 unknown snapshot |
| `GET /api/mcmc/{snapshot}/graph?timescale=` | implemented from HelixDB: `{snapshot, timescale, spaces[], nodes[] (regions), edges[] (Transition: from/to region ids, count, mean, lower, upper, updated_at)}`; `timescale` must match the snapshot's (else 404) |
| `GET /api/decoders/outputs?family=&limit=` | implemented: recent `DecoderOutput`s; `family=text` matches `text.*` |
| `GET /api/evolution/jobs?kind=&limit=` (alias `/api/jobs`) | implemented: per-kind status counts, recent jobs with progress, score and objectives |
| `GET /api/evolution/gpu` (alias `/api/gpu`) | implemented from evo-orchestrator heartbeat counters (D-031) |
| `POST /api/evolution/generations` | implemented: `{kind, snapshot_id, genomes[], eta?, min_budget_ms?, max_budget_ms?, max_generations?, total_budget_ms?, data_window_hours?}` (D-049) → `ctl.evo.start` request; returns the generation id |
| `GET /api/trading/positions`, `/orders?fills=`, `/fills`, `/risk`, `/signals` | implemented: paper book (EUR), blotter, risk state (reported + paper), halts, latest exposure per strategy |
| `POST /api/trading/halt` (alias `/kill-switch`), `/resume` (alias `/kill-switch/rearm`) | implemented: `risk.halt.portfolio` / `risk.resume.portfolio` (D-030) |
| `POST /api/trading/paper-order` (alias `POST /api/trading/orders`) | implemented: `{instrument_id, side, quantity, limit_price?}`. Always PAPER; `mode` other than paper → 400; while halted → 409 (D-029) |
| `POST /api/ingest/{source}/start\|stop` | implemented: `ctl.ingest.start\|stop` request (`{source}` arg, 5 s timeout) to the ingestor; 200 `{ok, source, action, message}`, 404 unknown source / no connector, 409 already running / not running / catalog status `disabled`\|`planned`, 503 without NATS, 504 no reply |
| `POST /api/jobs/{id}/cancel` | implemented: `ctl.evo.cancel` request to the evo-orchestrator; 200 when the job was pending (the job list then shows it `failed` with error `cancelled by operator`), 404 when no generation owns it, 503 without NATS |
| `GET /api/trading/pnl?range=&strategy=&book=` | implemented: realised P&L / fees / turnover over persisted fills in `range` (`<n>m\|h\|d`, RFC 3339 or `all`; default `1d`), optional strategy attribution filter, latest position snapshot; 503 without Postgres |
| `GET /api/trading/reconciliation?limit=` | implemented: latest `reconciliation_run` rows (newest first) with diffs |
| `GET /api/audit?since=&limit=` | implemented: `audit.log` since `since` (same grammar, default `24h`), oldest first |
| Everything else in spec 50 §4 | `501 {"error": "not_implemented", "spec": "specs/50-control-api.md §4.x"}` |

Helix-backed routes (`src/graph_routes.rs`) answer `503 {"error": "helix_unavailable"}` when HelixDB cannot be reached and `502 {"error": "helix_error"}` on a server-side query error. The Helix URL is `helix.url` from the config, overridable with `FA_HELIX_URL`. These are the **persisted** view; the in-memory `/api/latent/timescales/...` views (D-032) stay as they are.

Error replies from Go services (`CtlReply{ok:false}`) are mapped by message prefix: `not_found:` → 404, `conflict:` → 409, `invalid:` → 400, else 422.

Deviation from spec 50: no `/api/v1` prefix yet (the spec marks it a proposal). Paths match D-018.

## Owns
HTTP/WS surface.

## Governing specs
- [50-control-api](../../specs/50-control-api.md)
