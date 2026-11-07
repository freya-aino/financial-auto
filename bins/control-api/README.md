# control-api

axum REST + WebSocket control plane for the frontend. Local only, no auth (ADR 0006).

**Status:** ops, sources, bus tap, latent views, evolution and paper trading are implemented and verified against live NATS and the compose services (2026-11-07). The live views are in-memory aggregations of bus traffic (D-032), and the paper book resets on restart (D-028). Live trading is impossible through this API (D-029).

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
| `GET /api/latent/timescales` | implemented: configured ladder + timescales with data |
| `GET /api/latent/timescales/{ts}/states\|regions\|transitions` | implemented: latest `LatentState` per instrument; region registry with lineage, status, occupancy; region graph `{nodes, edges, rows}` with posterior mean and interval |
| `GET /api/decoders/outputs?family=&limit=` | implemented: recent `DecoderOutput`s; `family=text` matches `text.*` |
| `GET /api/evolution/jobs?kind=&limit=` (alias `/api/jobs`) | implemented: per-kind status counts, recent jobs with progress, score and objectives |
| `GET /api/evolution/gpu` (alias `/api/gpu`) | implemented from evo-orchestrator heartbeat counters (D-031) |
| `POST /api/evolution/generations` | implemented: `{kind, snapshot_id, genomes[], eta?, min_budget_ms?, max_budget_ms?}` → `ctl.evo.start` request; returns the generation id |
| `GET /api/trading/positions`, `/orders?fills=`, `/fills`, `/risk`, `/signals` | implemented: paper book (EUR), blotter, risk state (reported + paper), halts, latest exposure per strategy |
| `POST /api/trading/halt` (alias `/kill-switch`), `/resume` (alias `/kill-switch/rearm`) | implemented: `risk.halt.portfolio` / `risk.resume.portfolio` (D-030) |
| `POST /api/trading/paper-order` (alias `POST /api/trading/orders`) | implemented: `{instrument_id, side, quantity, limit_price?}`. Always PAPER; `mode` other than paper → 400; while halted → 409 (D-029) |
| `POST /api/ingest/{source}/start\|stop` | 501 with explanation: the ingestor has no `ctl.ingest.*` handler yet |
| Everything else in spec 50 §4 | `501 {"error": "not_implemented", "spec": "specs/50-control-api.md §4.x"}` |

Deviation from spec 50: no `/api/v1` prefix yet (the spec marks it a proposal). Paths match D-018.

## Owns
HTTP/WS surface.

## Governing specs
- [50-control-api](../../specs/50-control-api.md)
