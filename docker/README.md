# docker

Everything runs through Docker Compose (D-034): `compose.yaml` at the repo root covers the infrastructure (NATS, HelixDB, VictoriaMetrics, PostgreSQL, Garage) and, under profile `app`, every platform service plus the operator UI (D-026, D-033, D-035). This folder holds the Dockerfiles, the Garage config and its init script.

**Status:** compose working (verified 2026-11-06: nats, postgres, victoriametrics healthy; garage up, initialized by `garage-init.sh`).

## Services (ports per D-014, bound to 127.0.0.1)
| Service | Image | Ports | Health |
|---|---|---|---|
| nats | `nats:2.12-alpine` (`-js`) | 4222, monitor 8222 | `curl localhost:8222/healthz` |
| postgres | `postgres:17-alpine` (user/pass/db `fa`) | 5432 | `pg_isready` |
| victoriametrics | `victoriametrics/victoria-metrics:v1.155.0` | 8428 | `curl localhost:8428/health` |
| garage | `dxflrs/garage:v2.5.0` | S3 3900, admin 3903 | `docker compose exec garage /garage status` |
| helix | `ghcr.io/helixdb/helixdb:v0.0.14` (data in `/var/lib/helix`) | 6969 → 8080 | `curl localhost:6969/healthz` |

## Usage
```sh
just up        # docker compose up -d
just migrate   # apply migrations/ to postgres
just down
```

## Running the full stack (D-026)
The application services run in compose under the `app` profile. They use three images built from the repo root:

| Image | Dockerfile | Contains |
|---|---|---|
| `financial-auto/rust:dev` | `rust.Dockerfile` | `control-api`, `model-runtime`, `evo-worker`, `trainer`, `strategy-runtime`, `risk-engine` |
| `financial-auto/ui:dev` | `ui.Dockerfile` + `nginx.conf` | SvelteKit static build on unprivileged nginx, proxying `/api` and `/ws` to `control-api` |
| `financial-auto/go:dev` | `go.Dockerfile` | `ingestor`, `broker-gateway`, `evo-orchestrator` |

Inside the containers, services use `config/platform.docker.toml`, which has compose hostnames.

```sh
just app-build                 # docker compose --profile app build
just app-up                    # infra + services; UI at http://127.0.0.1:8080, API at http://127.0.0.1:7700
just app-ps / just app-logs [service]
just app-down                  # stops everything, keeps volumes
```

- **Default feed:** the ingestor starts `heartbeat,yahoo,hackernews,google_news` with Postgres reference resolution and the Garage raw archive (D-050). Edit `-connectors` in `compose.yaml` to change the set; `lemon`/`oanda` need keys in `config/secrets.toml`.
- **Migrations:** `control-api` applies the Postgres migrations on startup.
- **Scaling workers:** `just app-scale-workers 4`, or `docker compose --profile app up -d --scale evo-worker=4`. Workers are stateless and pull jobs from a NATS work queue. For more machines later, point more `evo-worker` containers at the same NATS.
- **Trainer:** a one-off CLI in the `tools` profile, not started by default. Run it with `just trainer pretrain --help` or `docker compose --profile tools run --rm trainer …`.
- **GPU:** install `nvidia-container-toolkit`, then uncomment the `deploy.resources.reservations.devices` blocks on `model-runtime`, `evo-worker` and `trainer`. GPU time is still arbitrated by `evo-orchestrator` (D-006). The Rust image is CPU-only today because `burn`/`candle` are off by default (D-004); enabling CUDA will need a CUDA base image.
- **HelixDB:** the `helix` service, reached at `http://helix:8080` inside compose and `127.0.0.1:6969` from the host. The image is distroless, so it has no in-container healthcheck; `control-api` only waits for it to start.
- **UI:** the `ui` service is the only thing serving the frontend in compose; the control-api container has no `frontend/build`, so `:7700` is API only. On the host, `cargo run -p control-api` still serves `frontend/build` when it exists.
- **No host processes:** don't run `cargo run -p control-api` at the same time, because it also binds port 7700.

## Garage first-time init
Garage needs a cluster layout, buckets and an access key before use. The script is idempotent:
```sh
docker/garage-init.sh
```
It does the following:
```sh
G() { docker compose exec -T garage /garage "$@"; }
G layout assign -z local -c 50G "$(G node id -q | cut -d@ -f1)"
G layout apply --version 1
G bucket create fa-raw   # likewise: fa-media fa-models fa-ckpt fa-artifacts fa-backup (spec 20 §5.5)
G key create fa-local
G bucket allow --read --write --owner fa-raw --key fa-local   # same for every bucket
G key info fa-local --show-secret   # copy into config/secrets.toml [garage]
```
`garage.toml` holds a fixed local-only `rpc_secret` and `admin_token` (ADR 0006). Don't reuse them anywhere else.

## HelixDB (D-033)
The standalone server image speaks `POST /v2/query` (operation-tree contract) plus `GET /healthz`, `GET /readyz` and `GET /v2/diagnostics`. `helix/schema.hx` and `helix/queries.hx` are not loaded by the image; `fa-hub` still targets the CLI-compiled `POST /<query_name>` endpoints (D-012), so its query layer has to move to `/v2/query` before Helix-backed routes work. Until then `/api/health` reports Helix reachable and nothing else uses it.

## Data
All state lives in named volumes (`financial-auto_{nats,helix,postgres,vm,garage-meta,garage-data}-data`). `docker compose down -v` wipes everything.
