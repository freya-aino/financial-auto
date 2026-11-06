# control-api

axum REST + WebSocket control plane for the frontend. Local only, no auth (ADR 0006).

**Status:** v1 skeleton implemented and verified against live NATS (2026-11-06).

## Run
```sh
cargo run -p control-api            # binds control_api.bind from config/platform.toml (127.0.0.1:7700)
cargo run -p control-api --example fake_publisher   # fake heartbeats + trades for UI work
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
| Everything else in spec 50 §4 | `501 {"error": "not_implemented", "spec": "specs/50-control-api.md §4.x"}` |

Deviation from spec 50: no `/api/v1` prefix yet (the spec marks it a proposal). Paths match D-018.

## Owns
HTTP/WS surface.

## Governing specs
- [50-control-api](../../specs/50-control-api.md)
