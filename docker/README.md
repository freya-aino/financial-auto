# docker

Everything runs through Docker Compose (D-034): `compose.yaml` at the repo root covers the infrastructure (NATS, HelixDB, VictoriaMetrics, PostgreSQL, Garage) and, under profile `app`, every platform service plus the operator UI (D-026, D-033, D-035). This folder holds the Dockerfiles, the Garage config and its init script, the VictoriaMetrics scrape config and vmalert rules (`vm-scrape.yaml`, `vm-alerts.yaml`), the Postgres WAL-archiving files (`postgres/`) and the `backup` sidecar (`backup/`).

**Status:** compose working: infrastructure since 2026-11-06, the full `app` profile, vmalert, exporters and the backup sidecar running on this machine (rootless Docker) as of 2026-11-11 (D-056, D-060, D-066). No GPU on this machine: the images are CPU-only.

## Services (ports per D-014, bound to 127.0.0.1)
| Service | Image | Ports | Health |
|---|---|---|---|
| nats | `nats:2.12-alpine` (`-js`) | 4222, monitor 8222 | `curl localhost:8222/healthz` |
| postgres | `postgres:17-alpine` (user/pass/db `fa`) | 5432 | `pg_isready` |
| victoriametrics | `victoriametrics/victoria-metrics:v1.155.0` | 8428 | `curl localhost:8428/health` |
| garage | `dxflrs/garage:v2.5.0` | S3 3900, admin 3903 | `docker compose exec garage /garage status` |
| helix | `ghcr.io/helixdb/helixdb:v0.0.14` (data in `/var/lib/helix`) | 6969 → 8080 | `curl localhost:6969/healthz` |
| vmalert | `victoriametrics/vmalert:v1.155.0` (rules `vm-alerts.yaml`) | 8880 | `curl localhost:8880/health` |
| nats-exporter | `natsio/prometheus-nats-exporter:0.17.3` | — (7777 in compose) | scraped as job `nats` |
| postgres-exporter | `quay.io/prometheuscommunity/postgres-exporter:v0.17.1` | — (9187 in compose) | scraped as job `postgres` |
| backup | `financial-auto/backup:dev` (`backup/Dockerfile`) | — | crond running; `docker compose logs backup` |

Every long-running service except `helix` and the two exporters has a healthcheck: infra natively (garage: `/garage status`), the Rust services `curl` their `/metrics`, the distroless Go images run `healthprobe` (`go/services/healthprobe`) against theirs. `helix` is distroless without a probe binary; control-api's `fa_dependency_up{dep="helix"}` and the `DependencyDown` alert cover it. Postgres runs with WAL archiving (`docker/postgres/`).

## Usage
```sh
just up        # docker compose up -d
just migrate   # apply migrations/ to postgres (optional: control-api applies them on start)
just down
```
`just` is optional (and not installed on this machine): every recipe in the root `justfile` is a one-liner, so run its body directly, e.g. `docker compose up -d`, `docker compose --profile app build`, `docker compose --profile app up -d && scripts/healthwait.sh`, `docker compose --profile app --profile tools down`, `docker compose --profile app logs -f --tail 100 <service>`, `scripts/backup.sh`, `scripts/restore.sh <dir>`, `scripts/restore-drill.sh`, `scripts/alert-drill.sh`, `docker compose exec backup run.sh`. The `just` names below map to those lines.

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
- **GPU:** install `nvidia-container-toolkit`, then uncomment the `deploy.resources.reservations.devices` blocks on `model-runtime`, `evo-worker` and `trainer`. GPU time is still arbitrated by `evo-orchestrator` (D-006). The Rust image is CPU-only today: the ML crates run on candle's CPU backend and the `cuda` cargo feature is off (D-070); enabling CUDA needs `--features cuda` and a CUDA base / builder image. Untested: this machine has no GPU. Device selection: `FA_ML_DEVICE` (spec 60 §6.2).
- **HelixDB:** the `helix` service, reached at `http://helix:8080` inside compose and `127.0.0.1:6969` from the host. The image is distroless, so it has no in-container healthcheck; `control-api` only waits for it to start and probes it every 15 s (`fa_dependency_up`).
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

**Scrape:** `victoriametrics` runs with `-promscrape.config=/etc/vm-scrape.yaml` (`docker/vm-scrape.yaml`, mounted read-only): static targets for each service, DNS discovery for the evo-worker replicas, plus VictoriaMetrics itself and Garage's admin `/metrics`. Check with `just metrics-up` or `curl 'http://127.0.0.1:8428/api/v1/query?query=up'` — every target should be `1`. `just metrics <service>` dumps one endpoint. NATS, Postgres and HelixDB have no Prometheus endpoint without exporters; their reachability is in control-api `/api/health`. NATS and Postgres are covered by the `nats-exporter` / `postgres-exporter` services; HelixDB has no endpoint, so control-api probes every dependency each 15 s into `fa_dependency_up{dep}`. The `fa` jobs use `honor_labels: true`. Dashboards: VictoriaMetrics' vmui (`http://127.0.0.1:8428/vmui`) and vmalert's UI (`http://127.0.0.1:8880`); no Grafana.

## Logs (spec 60 §4, Q3)
Every Rust binary initialises logging through `fa_core::logging::init` and every Go service through `bus.NewLogger`: `FA_LOG_FORMAT=json` writes one JSON object per line (Rust: `timestamp`, `level`, `target`, the event fields flattened, `span`; Go: slog's `time`, `level`, `msg`, `svc` and the attributes), anything else the human-readable text format (default). `RUST_LOG` filters the Rust side (directives on top of a global `info`), `FA_LOG_LEVEL` (`debug|info|warn|error`) the Go side. In compose `FA_LOG_FORMAT: ${FA_LOG_FORMAT:-text}` is set for every app service, so `FA_LOG_FORMAT=json` in `.env` switches the whole stack on the next `up -d`.

**Log store (Q3, decided provisionally):** Docker's default `json-file` driver is the store — `docker compose logs [-f] <service>` (`just app-logs`, `just logs-json <service>` for the raw lines), `docker compose logs --since 1h | grep trace_id=<id>` to follow one trace across services. No VictoriaLogs/Loki service: a single-host stack gains little from it, and JSON lines can be shipped later without code changes. The daemon does not rotate `json-file` logs by default; bound them on the host in the daemon config (rootless: `~/.config/docker/daemon.json`, rootful: `/etc/docker/daemon.json`) with `{"log-driver": "json-file", "log-opts": {"max-size": "50m", "max-file": "5"}}` and restart the daemon (applies to containers created afterwards).

## Tracing (spec 60 §6.5)
Trace ids travel in the W3C `traceparent` NATS header (`crates/fa-bus/src/trace.rs`, `go/pkg/bus/trace.go`): `fa_bus::Bus::publish` and Go `Publisher.Publish|PublishCtx|PublishWithID` add one, continuing the current trace (`fa_bus::trace::scope` / `bus.WithTrace`) or starting a new one; `md.*` and heartbeats are not traced (high rate, they start no chain worth correlating). Consumers log the trace id at `debug` (`Bus::subscribe`, `bus.Consumed`), `Bus::subscribe_traced` hands the context to a handler. control-api starts (or continues, from an HTTP `traceparent`) one trace per request, returns it in the response header, stores `trace_id` in the request's audit rows and sends it on its `ctl.*` requests and trading publishes; the ingestor and evo-orchestrator log their `ctl` handling with `trace_id`. **No OpenTelemetry exporter or collector (decided provisionally):** the ids are already W3C-shaped, so an exporter can be added later without a wire change; today the trace id is a log and audit field.

## Secrets (spec 60 §6.7)
API keys live in git-ignored `config/secrets.toml` or `FA_<SOURCE>_<KEY>` env vars; config dumps redact `*key*|*secret*|*token*` (`fa_core::secrets::redact`). `scripts/secret-scan.sh` (`just secret-scan`; `just secret-hook` installs the pre-commit hook running `--staged`) fails on credential-shaped strings and forbidden files; the local-only Garage `rpc_secret` / `admin_token` below carry `secret-scan: allow`.

## Alerting (spec 60 §6.5)
`vmalert` evaluates `docker/vm-alerts.yaml` every 30 s and notifies control-api (`/api/alerts/vmalert/api/v2/alerts`), which publishes each transition on NATS `ops.alert` and writes `audit.log`. `just alerts` (= `GET /api/alerts`) shows active + recent alerts and `pipeline.ok` (the always-firing `Watchdog` arrived within 3 min). Edit the rules, check them with `docker run --rm -v $PWD/docker/vm-alerts.yaml:/r.yaml:ro victoriametrics/vmalert:v1.155.0 -rule=/r.yaml -dryRun`, reload with `docker compose kill -s HUP vmalert`. `just alert-drill` is the end-to-end check.

## Backup and restore (spec 60 §6.6, spec 20 §6.6, D-056)
```sh
just backup                          # → backups/<UTC timestamp>/ (prints the directory)
just restore backups/<ts>            # asks for confirmation; --yes to skip
just restore backups/<ts> --postgres-db fa_restore_test   # drill: Postgres only, into a scratch DB
```
`scripts/backup.sh` writes `postgres/fa.dump` (`pg_dump -Fc` through `docker compose exec -T postgres`) and `postgres/globals.sql`, `helix/helix-data.tar.gz` (the `helix-data` volume, tarred by a throwaway alpine container while `helix` is stopped for a few seconds; `--no-stop-helix` for a hot copy), `garage/<bucket>/` for `fa-models fa-ckpt fa-artifacts fa-backup` (rclone in a container against the compose network, credentials from `config/secrets.toml` `[garage]`; `FA_BACKUP_BUCKETS` overrides the list), optionally `nats/<stream>/` with `FA_BACKUP_NATS=1` (`nats stream backup` via nats-box; off by default because the streams are replayable limits-retention buffers), and `manifest.json` with versions, sizes and sha256 of every file. `--skip postgres,helix,garage` leaves parts out.

`scripts/restore.sh DIR` verifies the checksums, asks for confirmation, stops the app services, recreates and `pg_restore`s the database, replaces the Helix volume contents, syncs the buckets back and prints the §6.4 recovery order; `just app-up` then applies migrations (control-api), reconciles the ledger (risk-engine) and waits for health. Nothing restores over the live databases without the prompt or `--yes`.

**Scheduled backups, WAL archiving, drills (spec 60 §6.6):** `backup.schedule` runs inside compose. The `backup` sidecar (`docker/backup/`) runs `run.sh` at `FA_BACKUP_SCHEDULE` (default `0 3 * * *`, `FA_BACKUP_TZ` default UTC): a PITR base backup (`postgres/base/`), the dump, a hot Helix tar, the Garage buckets, an incremental `backups/raw-mirror/fa-raw`, then pruning (`FA_BACKUP_KEEP`=7 scheduled backups, WAL back to the oldest kept base). Postgres archives every WAL segment (gzip) into the `pg-wal-archive` volume (`docker/postgres/archive-wal.sh`, `archive_timeout` 300 s). Weekly (`FA_DRILL_SCHEDULE`) the sidecar restores the newest backup into scratch clusters (logical + PITR); `just restore-drill` does the same from the host plus a Helix boot check. `just backup-now` runs the sidecar backup immediately; PITR restore: `just restore backups/<ts> --pitr [--target …]` (runbook in `scripts/README.md`). Off-machine: `FA_BACKUP_OFFSITE_ENABLED=1` + `FA_BACKUP_OFFSITE_REMOTE=<rclone remote:path>` (disabled by default). `backups/` ignores itself.

## Host reboot (rootless Docker)
The rootless daemon is a systemd *user* unit: with `Linger=no` it starts only when the user logs in, and stops at logout. After a reboot the `restart: unless-stopped` policy brings every container back at the same moment (the daemon ignores compose `depends_on`), which is what happened on 2026-11-11 (power-off 22:40 CET, boot 02:37, login 02:43 → all containers started at 01:43 UTC; the ingestor's first start hit a Postgres that was not yet accepting connections and was restarted by the policy — it now retries for 60 s instead). To have the stack up without a login: `loginctl enable-linger $USER` (host configuration; on NixOS `users.users.<name>.linger = true`).

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
The standalone server image speaks `POST /v2/query` (operation-tree contract) plus `GET /healthz`, `GET /readyz` and `GET /v2/diagnostics`. `helix/schema.hx` and `helix/queries.hx` are not loaded by the image; `fa-hub` talks to `POST /v2/query` directly (D-033, D-048), so the Helix-backed control-api routes (graph, snapshots, regions, MCMC, lineage), model-runtime's hub sink and the evolution lineage all use it.

## Data
All state lives in named volumes: `financial-auto_{nats-data,helix-data,postgres-data,vm-data,garage-meta,garage-data}`, `financial-auto_pg-wal-archive` (archived WAL) and `financial-auto_broker-gateway-state` (calibrated costs, persisted working orders / positions `orders.json`; D-076, D-079); backups go to the `./backups` bind mount. `docker compose down -v` wipes everything — take `just backup` first.

## Reference (spec 60 Ops, retired 2026-11-11)
Spec 60 was retired once its acceptance criteria were met (or out of scope) and its open questions decided; its lasting content is here and in `scripts/README.md`. Code comments citing "spec 60 §n" refer to these sections.

**§1–3 Purpose and decisions.** The platform runs on one workstation in Docker Compose (infrastructure: NATS JetStream, HelixDB, VictoriaMetrics, PostgreSQL, Garage; profile `app`: every service, D-026) and scales horizontally later through JetStream work queues: stateless workers pull jobs with explicit ack, artifacts go to Garage, results and lineage to Helix/Postgres. One GPU shared by priority class; observability = logs (stdout), metrics (VictoriaMetrics), trace ids in NATS headers; backups of Postgres, Helix and Garage; secrets only in git-ignored `config/secrets.toml` or env. Out of scope: cloud deployment, auth/TLS (ADR 0006), multi-machine tooling, cloud bursting.

**§4 Interfaces.** Ports: table "Services" above and "Metrics"; control-api 7700, ui 8080, Vite 5173. Every service publishes `svc.heartbeat.<service>` every 5 s, serves `/metrics`, logs to stdout (format per "Logs"). Alerts: NATS `ops.alert` JSON `{status: firing|resolved, alertname, severity, summary, fingerprint, labels, annotations, starts_at, ends_at, generator_url, at, expired?}`, one message per transition; `GET /api/alerts` → `{active, recent, pipeline{ok, watchdog_seen_at, posts, last_post_at}}`. Work-queue envelope (spec 21): `Job{id, kind, priority_class, payload_ref|inline, budget, checkpoint_ref?, attempt}` → `JobResult{id, status, result_ref, metrics}`.

**§5 Data.** Job status: JetStream `JOBS_<kind>` + control-api memory + Postgres `ops.job_run`; checkpoints Garage `fa-ckpt/<job_id>/`; metrics VictoriaMetrics; logs Docker `json-file`; backups `./backups/<ts>/` + volume `pg-wal-archive`; alert state control-api memory + `ops.alert_event` + `audit.log` + VM `ALERTS`.

**§6.1 Process model.** ingestor (Go, 1, connectors as goroutines), broker-gateway (Go, 1 per mode; paper today), evo-orchestrator (Go, 1, hosts the GPU arbiter), model-runtime (Rust, 1 serving every timescale), strategy-runtime (Rust, 1), risk-engine (Rust, 1 per mode; paper), evo-worker (Rust, N: `just app-scale-workers N`), trainer (Rust, on demand / `trainer serve`), control-api (Rust, 1), ui (nginx, 1).

**§6.2 GPU scheduling.** `go/services/evo-orchestrator/arbiter.go` (D-006) owns the lease over `ctl.gpu.acquire|release` with notifications `ctl.gpu.preempt|granted`: strict priority live inference > online adapters > evolution evaluation > generation training; a higher class preempts a lower one, which checkpoints and yields; memory estimates pack concurrent jobs within `-gpu-mem-mb`. Devices: candle CPU by default, `cuda` cargo feature + `FA_ML_DEVICE`/`--device` (CPU fallback with a recorded note). Untested on real GPU hardware (none on this host); the arbiter logic is unit-tested.

**§6.3 Horizontal scaling.** Workers ack explicitly with ack-wait and in-progress acks; a crashed worker's job is redelivered after the ack wait and resumes from `checkpoint_ref` (trainer: Welford / candle state in `fa-ckpt`, verified identical to an uninterrupted fit, D-058/D-070; evo-worker evaluations are cheap and re-run from the start). `scripts/worker-kill-drill.sh` is the kill drill. Adding a machine = more `evo-worker`s against the same NATS/Garage/Helix.

**§6.4 Startup / recovery order** — section above; trading stays halted until the risk-engine has loaded the ledger and reconciled.

**§6.5 Observability** — "Metrics", "Logs", "Tracing", "Alerting" above; rule groups in `vm-alerts.yaml` (meta, services, infra, backup, bus, control-api, trading, ingestion). §6.6 Backups — "Backup and restore" above and `scripts/README.md`. §6.7 Secrets — "Secrets" above.

**§7 Tunables.** `ops.heartbeat_interval` 5 s, `ops.heartbeat_timeout` 15 s (alert at 45 s), `gpu.preempt_deadline` 60 s, `gpu.live_reserved_mem` 4 GiB, `jobs.ack_wait` 60 s for `JOBS_EVO` (in-progress ack every 20 s), `train.checkpoint_interval` 10 min, `backup.schedule` 03:00 (`FA_BACKUP_SCHEDULE`, `FA_BACKUP_TZ`), `backup.keep` 7 (`FA_BACKUP_KEEP`), `backup.drill_schedule` Sunday 04:30 (`FA_DRILL_SCHEDULE`), `postgres.archive_timeout` 300 s, `alerts.evaluation_interval` 30 s, `FA_LOG_FORMAT` text|json, `garage.media_budget` 100 GiB (below).

**§8 Failure modes.** GPU OOM → job fails and is retried smaller, live inference memory never preempted. Worker crash → redelivery, resume from checkpoint. Disk nearly full → `DiskSpaceLow` alert, retention prunes (spec 20), raw-media capture paused first. Infra restart → services reconnect with backoff, trading halts until health returns. Host reboot with rootless Docker → see "Host reboot" (all containers start at once; services tolerate starting dependencies; `loginctl enable-linger`). Alerting pipeline down → `pipeline.ok = false` is itself an alert. Power loss → §6.4 order, reconciliation mandatory.

**§9 Acceptance.** Met: `just up` healthy infra and `just app-up` for every service; evo-worker kill drill (`scripts/worker-kill-drill.sh`); every service in `/api/services` with heartbeat and scraped metrics; backup → restore (scratch database and volumes); `just restore-drill` incl. PITR; `just alert-drill`; secret scan passes on the tree (`just secret-scan`; `--history` still lists the old repo's `APIs/lemon/.env` JWT, a key to rotate). Out of scope: real-GPU preemption (no GPU), off-machine backup, restore on a second machine.

**§10 Decisions on the open questions.** Q1 GPU arbiter in evo-orchestrator (D-006). Q2 supervision: compose `restart: unless-stopped` + healthchecks, operated through control-api's Docker supervisor (D-060). Q3 log store: Docker `json-file` + `docker compose logs`, JSON lines on `FA_LOG_FORMAT=json`, host log rotation; no VictoriaLogs/Loki (provisional). Q4 off-machine backup: disabled rclone hook (`FA_BACKUP_OFFSITE_*`), out of scope. Q5 raw-media disk budget (provisional, no media producer exists yet): bucket `fa-media` capped at 100 GiB (`garage.media_budget`); video originals downsampled after 30 days (`garage.media_downsample_after_d`, spec 20) and deleted once downsampled, audio and transcripts kept; at 80 % of the budget or on `DiskSpaceLow` the media capture lanes pause first (before any market or text ingestion), resuming below 70 %. The non-media raw archive `fa-raw` stays keep-forever (spec 20) and is mirrored by the backup sidecar. Enforcement belongs to the first media connector (spec 10).
