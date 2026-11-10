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
- **Service control from the API/UI (D-060):** `POST /api/services/{name}/start|stop|restart` and `GET /api/services/{name}/container` work through the Docker socket, which `compose.yaml` mounts read-write into `control-api` **only** (`${FA_DOCKER_SOCK:-/var/run/docker.sock}` → `/var/run/docker.sock`, `FA_COMPOSE_PROJECT=financial-auto`). Infra services and control-api itself are refused (409); `risk-engine` stop/restart needs `{"confirm": true}`. See `bins/control-api/README.md` → Supervisor.
  - **Rootless Docker** (this machine): the daemon listens on `$XDG_RUNTIME_DIR/docker.sock` and `/var/run/docker.sock` is root-owned, so point compose at the right socket with `export FA_DOCKER_SOCK=$XDG_RUNTIME_DIR/docker.sock` or, better, a git-ignored `.env` at the repo root (compose reads it):
    ```sh
    FA_DOCKER_SOCK=/run/user/1000/docker.sock
    FA_DOCKER_GID=131
    ```
  - **Permissions:** the Rust image runs as uid 10001 (`rust.Dockerfile`), and the socket is `0660` with group `docker`, so control-api joins that group through `group_add: ["${FA_DOCKER_GID:-999}"]` instead of running as root. The gid must be the one the socket has *as the daemon sees it*: rootful → the host's `docker` group (`getent group docker`, often 999 or 998); rootless → the `docker` gid inside the daemon's user namespace, which the host sees as a subgid (here `100130`), so read it from a throwaway container: `docker run --rm -v "$FA_DOCKER_SOCK":/s alpine stat -c %g /s` (→ `131` here). Verified: without `group_add` the mount is there but `connect` fails (`EACCES`); with it `GET /_ping` answers `OK`.
  - **Host runs** (`cargo run -p control-api`) use `DOCKER_HOST`/`FA_DOCKER_SOCK` or the default sockets directly; with none reachable the routes answer `501 no_supervisor` and everything else keeps working.

## Metrics (spec 60 §4/§6.5, D-056)
Every service exposes Prometheus text on `GET /metrics` (`crates/fa-metrics`, re-exported as `fa_bus::metrics`; Go: `go/pkg/bus/metrics.go`). Inside compose each listens on `0.0.0.0:9100` (`FA_METRICS_BIND`), published on 127.0.0.1 as:

| Service | Host port (compose) | Host default when run natively (`FA_METRICS_BIND` unset) |
|---|---|---|
| control-api | 9101 | 127.0.0.1:9101 |
| model-runtime | 9102 | 127.0.0.1:9102 |
| strategy-runtime | 9103 | 127.0.0.1:9103 |
| risk-engine | 9104 | 127.0.0.1:9104 |
| ingestor | 9105 | 127.0.0.1:9105 |
| broker-gateway | 9106 | 127.0.0.1:9106 |
| evo-orchestrator | 9107 | 127.0.0.1:9107 |
| evo-worker | 9110–9119 (one per replica) | 127.0.0.1:9110 |

`FA_METRICS_BIND=off` disables the endpoint (the trainer sets this). Go services also take `-metrics <bind>`.

What is exported: `fa_heartbeat_counter{service,key}` (every heartbeat counter, mirrored at publish time, so nothing is instrumented twice), `fa_heartbeat_last_unix_seconds`, process gauges (`fa_process_uptime_seconds`, `fa_process_resident_memory_bytes` from `/proc/self/statm`, `fa_build_info{service,version}`), `fa_bus_consumer_num_pending|ack_pending|redelivered|waiting{stream,consumer}` for every JetStream pull consumer a service owns (evo-worker `JOBS_EVO`, ingestor `JOBS_INGEST`), and on control-api `fa_http_requests_total{method,path,status}` plus the `fa_http_request_duration_seconds` histogram (path = matched axum route, or the raw path with id-like segments collapsed to `{id}` where routers are chained through `fallback_service`).

**Scrape:** `victoriametrics` runs with `-promscrape.config=/etc/vm-scrape.yaml` (`docker/vm-scrape.yaml`, mounted read-only): static targets for each service, DNS discovery for the evo-worker replicas, plus VictoriaMetrics itself and Garage's admin `/metrics`. Check with `just metrics-up` or `curl 'http://127.0.0.1:8428/api/v1/query?query=up'` — every target should be `1`. `just metrics <service>` dumps one endpoint. NATS, Postgres and HelixDB have no Prometheus endpoint without exporters; their reachability is in control-api `/api/health`. Alerting thresholds are a documented proposal in `docker/vm-alerts.proposal.yaml` (no vmalert container yet).

## Backup and restore (spec 60 §6.6, spec 20 §6.6, D-056)
```sh
just backup                          # → backups/<UTC timestamp>/ (prints the directory)
just restore backups/<ts>            # asks for confirmation; --yes to skip
just restore backups/<ts> --postgres-db fa_restore_test   # drill: Postgres only, into a scratch DB
```
`scripts/backup.sh` writes `postgres/fa.dump` (`pg_dump -Fc` through `docker compose exec -T postgres`) and `postgres/globals.sql`, `helix/helix-data.tar.gz` (the `helix-data` volume, tarred by a throwaway alpine container while `helix` is stopped for a few seconds; `--no-stop-helix` for a hot copy), `garage/<bucket>/` for `fa-models fa-ckpt fa-artifacts fa-backup` (rclone in a container against the compose network, credentials from `config/secrets.toml` `[garage]`; `FA_BACKUP_BUCKETS` overrides the list), optionally `nats/<stream>/` with `FA_BACKUP_NATS=1` (`nats stream backup` via nats-box; off by default because the streams are replayable limits-retention buffers), and `manifest.json` with versions, sizes and sha256 of every file. `--skip postgres,helix,garage` leaves parts out.

`scripts/restore.sh DIR` verifies the checksums, asks for confirmation, stops the app services, recreates and `pg_restore`s the database, replaces the Helix volume contents, syncs the buckets back and prints the §6.4 recovery order; `just app-up` then applies migrations (control-api), reconciles the ledger (risk-engine) and waits for health. Nothing restores over the live databases without the prompt or `--yes`.

`backup.schedule` (03:00 daily, spec 60 §7) is a crontab line, not a container: `0 3 * * * cd /path/to/financial-auto && scripts/backup.sh >> backups/backup.log 2>&1`. `backups/` ignores itself (`backup.sh` drops a `.gitignore` there); copy it to a second disk or the `fa-backup` bucket off-machine (open question 60 §10.4).

## Startup / recovery order (spec 60 §6.4)
`just app-up` runs `scripts/healthwait.sh` after `up -d`: it waits for the five infra health endpoints, then control-api `/api/health` (NATS connected and every dependency reachable; migrations ran on start), then the risk-engine heartbeat in `/api/services` (ledger loaded, reconciliation armed — trading stays halted until then), and reports the other services' heartbeats as they appear. `just healthwait [timeout]` runs it on its own.

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
All state lives in named volumes (`financial-auto_{nats,helix,postgres,vm,garage-meta,garage-data}-data`). `docker compose down -v` wipes everything — take `just backup` first.
