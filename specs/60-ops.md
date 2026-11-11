# 60 — Operations

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium |
| **Owners (code)** | `compose.yaml`, `docker/`, `justfile`, `go/services/evo-orchestrator`, `bins/*` (health/metrics, incl. `strategy-runtime`, `risk-engine`), `config/` |
| **Depends on** | 01 Architecture, 20 Storage, 21 Bus & contracts, 36 Training lifecycle |

## 1. Purpose
Define how the platform runs on a single local workstation today and scales horizontally later: deployment, GPU sharing, job distribution, observability, backups, restart/recovery and secrets.

## 2. Scope
**In scope:** compose topology, process model, GPU scheduling, work-queue scaling contract, logs/metrics/traces, backups, recovery order, secrets handling.
**Out of scope:** cloud deployment, auth/TLS (ADR 0006).

## 3. Decisions
- **Local docker compose** for infrastructure: NATS (JetStream), HelixDB, VictoriaMetrics, PostgreSQL, Garage.
- Application services run natively in dev (`just dev`) and optionally containerised.
- **Single GPU now**; every training/evaluation component is distributable.
- **GPU priority queue**: live inference > online adapters > evolution evaluation > generation training; jobs are preemptible and checkpointed.
- **Horizontal scaling via NATS JetStream work queues**: stateless Rust workers pull jobs; artifacts in Garage; results and lineage to Helix/Postgres.
- Observability: **tracing → logs**, **metrics → VictoriaMetrics**.
- Backups of Postgres, Helix and Garage.
- Secrets only in git-ignored `config/secrets.toml` or environment variables.

## 4. Interfaces
- `compose.yaml` services and default ports **(proposal)**:
  | Service | Port |
  |---|---|
  | nats | 4222 (client), 8222 (monitor) |
  | helixdb | 6969 |
  | victoriametrics | 8428 |
  | postgres | 5432 |
  | garage | 3900 (S3), 3903 (admin) |
  | vmalert | 8880 (rules UI, API) |
  | backup (sidecar) | — (crond; writes `./backups`) |
  | nats-exporter / postgres-exporter | — (compose network only, 7777 / 9187) |
  | control-api | 7700 (D-014) |
  | ui (compose, nginx) | 8080 (D-035) |
  | frontend (vite dev) | 5173 |
- `justfile` recipes: `up`, `down`, `dev`, `check`, `test`, `fmt`, `proto`, `migrate`, `backup`, `restore`, `backup-now`, `restore-drill`, `alert-drill`, `alerts`.
- Alerts: NATS subject `ops.alert`, JSON `{status: firing|resolved, alertname, severity, summary, fingerprint, labels, annotations, starts_at, ends_at, generator_url, at, expired?}` — one message per transition, never per re-send; `GET /api/alerts` → `{active, recent, pipeline{ok, watchdog_seen_at, posts, last_post_at}}`; receivers `POST /api/alerts/vmalert/api/v2/alerts` (Alertmanager v2 array, what vmalert sends) and `POST /api/alerts/webhook` (Alertmanager webhook body).
- Every service exposes: heartbeat on `svc.heartbeat.<service>` (every 5 s, as in 21 and the code), Prometheus metrics endpoint scraped by VictoriaMetrics (`vmagent` or VM scrape config), structured JSON logs to stdout.
- Work-queue envelope (21): `Job { id, kind, priority_class, payload_ref (Garage key) | inline, budget, checkpoint_ref?, attempt }` → `JobResult { id, status, result_ref, metrics }`.

## 5. Data model
| Item | Store |
|---|---|
| Job records & status | Postgres `job` + JetStream work-queue streams `JOBS_<kind>` |
| Checkpoints | Garage `checkpoints/<job_id>/…` |
| Metrics | VictoriaMetrics |
| Logs | stdout → file/journald **(proposal: VictoriaLogs later)** |
| Backups | `./backups/<ts>/` (bind mount of the backup sidecar), archived WAL in volume `pg-wal-archive`; off-machine copy = disabled rclone hook (§6.6) |
| Alert state | control-api memory (active + last 200 transitions), `audit.log` rows `ops.alert.firing|resolved`, VictoriaMetrics `ALERTS` / `ALERTS_FOR_STATE` (vmalert remote write/read) |

## 6. Behaviour & algorithms

### 6.1 Process model
| Process | Lang | Instances |
|---|---|---|
| ingestor | Go | 1 (connectors as goroutines); shardable by source |
| broker-gateway | Go | 1 per mode (paper, live) |
| evo-orchestrator | Go | 1 |
| model-runtime | Rust | 1 per timescale group **(proposal)** |
| strategy-runtime | Rust | 1 (D-040) |
| risk-engine | Rust | 1 per mode; paper only today (D-040, D-041) |
| evo-worker | Rust | N (1 GPU worker + CPU workers today) |
| trainer | Rust | on demand, as GPU job |
| control-api | Rust | 1 |
| ui | nginx | 1, compose only (D-035) |

### 6.2 GPU scheduling
A GPU arbiter (part of `evo-orchestrator` **(proposal)**) owns the GPU lease:
1. **Live inference** (model-runtime) — always resident, reserved memory slice.
2. **Online adapters** — small, frequent updates.
3. **Evolution evaluation** — batched candidate evaluations, successive halving.
4. **Generation training** — long, checkpointed every N minutes.
Higher class arriving → lower class checkpoints and yields within a deadline. Jobs declare memory estimates; arbiter packs concurrent jobs when memory allows. Budget accounting in GPU-hours per generation (34).

**Device side (implemented):** the ML crates build on candle with the CPU backend by default; the `cuda` cargo feature (`fa-expert-space`, `fa-encoders`, `fa-decoders`, `trainer`, `model-runtime`) enables CUDA devices, off by default because the dev host has no GPU. The device is configuration: `FA_ML_DEVICE` (`cpu` | `cuda` | `cuda:<n>`), the trainer's `--device` / job `spec.device`; a CUDA request in a CPU build, or an unavailable ordinal, falls back to the CPU with a logged note recorded in `manifest.params.device_note`. Neural fits report `metrics.mem_estimate_mb`; the lease size is still the worker's `--gpu-lease-mb`. With a GPU: build the image with `--features cuda` (needs a CUDA toolchain in the builder image, not provided) and uncomment the compose `deploy.resources` blocks of `model-runtime` / `trainer`. Unverified on hardware.

### 6.3 Horizontal scaling
Workers are stateless: pull from JetStream work queues with explicit ack and ack-wait; heartbeat in-progress (`+WPI`); on failure the job is redelivered and resumes from `checkpoint_ref`. Adding a machine = start more `evo-worker`s pointing at the same NATS/Garage/Helix. No worker-local state is authoritative.

### 6.4 Startup / recovery order
1. Postgres, NATS, Garage, VictoriaMetrics, HelixDB (health-checked).
2. Migrations (`just migrate`), Helix schema deploy.
3. ledger + broker-gateway → **reconciliation** (42) before any trading.
4. risk (inside model-runtime/allocator path) loads persisted risk state incl. kill switch.
5. ingestor (resumes from last known cursors per source; gaps flagged for backfill).
6. model-runtime (loads active snapshots), evo-orchestrator, workers.
7. control-api, frontend.
Trading stays halted until steps 3–5 report healthy.

**Implemented today (D-056):** `just app-up` ends in `scripts/healthwait.sh`, which waits for infra health, then control-api `/api/health` (migrations applied, every store reachable), then the risk-engine heartbeat (ledger loaded from Postgres, D-047) before reporting the other services. Trading stays halted until risk-engine is up because it is the only publisher of approved orders.

### 6.5 Observability
- Metrics: ingestion rates/lag per source, bus consumer lag, inference latency per timescale, GPU utilisation/memory per class, job throughput, P&L/exposure, limit headroom, slippage.
- Tracing: `tracing` (Rust) and `slog`/OTel (Go) with trace ids propagated in NATS headers.
- Alerts: VM alerting rules (vmalert) → control-api receiver → `ops.alert` subject + audit → UI banner.

**Implemented today (D-056):** every service serves Prometheus text on `GET /metrics` (`crates/fa-metrics`, re-exported as `fa_bus::metrics`; Go `go/pkg/bus/metrics.go`): heartbeat counters mirrored as `fa_heartbeat_counter{service,key}`, process gauges, JetStream consumer lag, and control-api HTTP counts/latency. VictoriaMetrics scrapes all targets itself (`docker/vm-scrape.yaml`, `-promscrape.config`); host ports 9101..9107 and 9110. 

**Implemented today (alerting):** the `vmalert` compose service (`victoriametrics/vmalert:v1.155.0`, UI `127.0.0.1:8880`) evaluates `docker/vm-alerts.yaml` every 30 s against VictoriaMetrics, writes `ALERTS`/`ALERTS_FOR_STATE` back (and restores `for:` state from them on restart) and notifies control-api as if it were an Alertmanager (`-notifier.url=http://control-api:7700/api/alerts/vmalert`). `bins/control-api/src/alerts_routes.rs` keeps the active set in memory and acts on transitions only (vmalert re-sends every firing alert each evaluation): a new firing alert, a resolution (vmalert sends `endsAt` in the past) or an expiry (an active alert not refreshed past its `endsAt`, i.e. vmalert gone; `expired: true`) is published as JSON on `ops.alert` and written to `audit.log` (`actor 'system:vmalert'`, `action 'ops.alert.firing|resolved'`, `subject` = alertname). The always-firing `Watchdog` (severity `none`) is never published; its arrival sets `pipeline.ok` in `GET /api/alerts` and `fa_alerts_watchdog_age_seconds`. Also exported: `fa_alerts_active{severity}`, `fa_alerts_received_total`, `fa_alerts_transitions_total{status,severity}`. Rule groups: meta (Watchdog, AlertDrill), services (scrape down, missing targets, heartbeat stale > 45 s, crash loop = more than 2 process starts in 15 min, RSS), infra (`fa_dependency_up` — control-api's 15 s TCP probe of helix/VM/postgres/garage/nats, the only HelixDB signal —, exporter targets, free disk < 20 GiB, Postgres connections, NATS slow consumers), backup (WAL archive failing/stale, backup failed/stale, restore drill failed/stale), bus (work-queue backlog/redeliveries, any JetStream consumer > 10 000 behind, JetStream storage > 85 %, dead letters), control-api (5xx ratio, p99), trading (risk halt, stale FX), ingestion (pipeline stalled, TSDB write errors/drops, raw-archive errors, Helix write errors). `scripts/alert-drill.sh` (`just alert-drill`) is the end-to-end test: it writes `fa_alert_drill{drill_id}` into VictoriaMetrics, waits for the firing `ops.alert` message on NATS, checks `GET /api/alerts` and the audit row, stops writing and waits for the resolved message (verified: firing after 38 s, resolved after 151 s). The UI banner itself (spec 51) subscribes to `ops.alert` through the bus tap or polls `/api/alerts`.

**Implemented today (exporters):** `nats-exporter` (`natsio/prometheus-nats-exporter`, varz/connz/JetStream per stream and consumer — all consumer lag, not only the services' own work queues) and `postgres-exporter` (connections, sizes, WAL, locks) run in compose and are scraped as jobs `nats` / `postgres`, plus `vmalert`. The `fa` jobs use `honor_labels: true`, so a series' own `service` label is kept (no `exported_service`). Go services now also export `fa_heartbeat_published_total` / `fa_heartbeat_last_unix_seconds` like the Rust ones. Spec metrics still not instrumented: inference latency per timescale (model-runtime), GPU utilisation per class (only `gpu.used_mb`/`capacity_mb` from the orchestrator), limit headroom and slippage (risk-engine / broker-gateway heartbeats) — they belong to those services. No Grafana: VictoriaMetrics' built-in vmui (`127.0.0.1:8428/vmui`) and vmalert's UI are the dashboards.

### 6.6 Backups
- Postgres: nightly base backup + `pg_dump` + continuous WAL archiving (PITR).
- Helix: nightly snapshot of data directory (service paused or via native snapshot if available).
- Garage: bucket replication/rsync of critical buckets (models, raw archive) to a second disk.
- Restore drilled automatically every week (Postgres, in the sidecar) and on demand (`just restore-drill`, all parts).

**Implemented today (D-056):** `scripts/backup.sh` (`just backup`) writes `backups/<ts>/` with `pg_dump -Fc` + globals, a tar of the `helix-data` volume (helix stopped for seconds), rclone copies of the critical Garage buckets, optional `nats stream backup`, and a sha256 manifest; `scripts/restore.sh DIR` (`just restore`) verifies, confirms, stops the app services and restores in the §6.4 order (`--postgres-db` for scratch drills, verified).

**Implemented today (WAL archiving, scheduled backups, drills):**
- *WAL archiving:* the `postgres` service runs with `archive_mode=on`, `archive_timeout=300` (RPO ≤ 5 min while anything writes) and `archive_command=sh /etc/postgresql/fa/archive-wal.sh %p %f`, which gzips each finished segment into the named volume `pg-wal-archive` (`/wal-archive`; refuses to overwrite a different file, succeeds on an identical one, per the PostgreSQL archiving contract). `docker/postgres/pg_hba.conf` (selected with `-c hba_file=…`, the data directory's own file stays as initdb wrote it) adds network `replication` for user `fa` so the sidecar can run `pg_basebackup`. WAL archiver health (`pg_stat_archiver`) is pushed every 5 min as `fa_wal_archive_*`.
- *Scheduled backup in compose (no host cron):* the `backup` sidecar (`docker/backup/`, image `financial-auto/backup:dev` = `postgres:17-alpine` + rclone + curl) runs busybox `crond`: `run.sh` at `FA_BACKUP_SCHEDULE` (default `0 3 * * *`, `TZ=${FA_BACKUP_TZ:-UTC}`) writes `./backups/<ts>/` with a PITR base backup (`postgres/base/base.tar.gz` + `pg_wal.tar.gz` + `backup_label.txt`, `pg_basebackup -Ft -z -X stream`), `pg_dump -Fc` + globals + exact `row_counts.tsv`, a hot tar of the read-only `helix-data` volume (manifest `"helix_consistency": "hot"`; `scripts/backup.sh` still stops helix for a cold copy), the critical Garage buckets and a `manifest.json` with `"kind": "scheduled"` — the same layout `restore.sh` reads. On first start without a scheduled base backup it takes one at once, so archived WAL is usable immediately. Metrics `fa_backup_last_{status,success_timestamp_seconds,duration_seconds,bytes}` go to VictoriaMetrics.
- *Retention:* `prune.sh` keeps the newest `FA_BACKUP_KEEP` (7) scheduled backups and runs `pg_archivecleanup -x .gz` up to the START WAL file of the oldest kept base backup, so every point from that base backup to now stays recoverable; manual backups (`scripts/backup.sh`) are never pruned.
- *PITR restore:* `scripts/restore.sh <scheduled-backup> --pitr [--target 'YYYY-MM-DD HH:MM:SS+00']` stops the app services and postgres, saves the current data directory to `backups/pre-restore-<ts>/postgres-data.tar.gz`, replaces it with the base backup, appends `restore_command = 'gunzip -c /wal-archive/%f.gz > %p'` + `recovery_target_action = 'promote'` (+ `recovery_target_time`) and `recovery.signal`, starts postgres (replays WAL, promotes to a new timeline) and waits until it leaves recovery; `just app-up` continues the §6.4 order. `--pg-volume NAME` does the same into a scratch volume checked by a throwaway container instead (verified: 2 WAL files replayed, timeline 2, 35 tables; the scratch volume is emptied afterwards).
- *Restore drills (live data untouched):* `docker/backup/drill-postgres.sh` restores into throwaway clusters inside a container (unix sockets only): logical (`pg_restore` + table presence and row counts vs `row_counts.tsv`) and PITR (base backup + WAL archive mounted read-only, promoted; reports WAL files replayed and the last replayed commit time). The sidecar runs it weekly (`FA_DRILL_SCHEDULE`, default Sunday 04:30) on the newest scheduled backup; `scripts/restore-drill.sh [DIR] [--target T] [--switch-wal]` (`just restore-drill`) runs it in a throwaway backup-image container plus a HelixDB boot check (the tar in an anonymous volume, a scratch helix with `--network none`, `/readyz` must answer) and the manifest/Garage checks. Results: `fa_restore_drill_last_{status,success_timestamp_seconds}`. Verified: 83 files checksummed; logical 35/35 tables, row counts equal; PITR replayed 4 archived WAL files to 02:13:18 from a 02:09:35 base; to a target time it stopped at 02:09:59.86 for target 02:10:00; Helix answered `/readyz` on the restored volume (1177 files).
- *Off-machine copy:* configured but disabled hook in `run.sh`: with `FA_BACKUP_OFFSITE_ENABLED=1` and `FA_BACKUP_OFFSITE_REMOTE=<rclone remote:path>` (remote defined through `RCLONE_CONFIG_<NAME>_*` env) each backup and the WAL archive are copied there. External targets stay out of scope (§10 Q4).

### 6.7 Secrets
- API keys in `config/secrets.toml` (git-ignored) or env vars `FA_<SOURCE>_<KEY>`.
- `.gitignore` covers `config/secrets.toml`, `.env*`.
- Secrets never logged; config dumps redact keys matching `*key*|*secret*|*token*`.
- Pre-commit check for accidental secrets **(proposal)** — the old repo committed `APIs/lemon/.env`.

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `ops.heartbeat_interval` | 5 s | |
| `ops.heartbeat_timeout` | 15 s | service marked down |
| `gpu.preempt_deadline` | 60 s | lower class must yield |
| `gpu.live_reserved_mem` | 4 GiB | |
| `jobs.ack_wait` | 5 min | with progress heartbeats |
| `train.checkpoint_interval` | 10 min | |
| `backup.schedule` | 03:00 daily | `FA_BACKUP_SCHEDULE` (cron line), `FA_BACKUP_TZ` (default UTC) |
| `backup.keep` | 7 scheduled backups | `FA_BACKUP_KEEP`; WAL kept back to the oldest kept base backup |
| `backup.drill_schedule` | Sunday 04:30 | `FA_DRILL_SCHEDULE` |
| `postgres.archive_timeout` | 300 s | RPO bound for PITR |
| `alerts.evaluation_interval` | 30 s | vmalert `-evaluationInterval`; thresholds in `docker/vm-alerts.yaml` |

## 8. Failure modes
| Failure | Response |
|---|---|
| GPU OOM | job fails, retried with smaller batch; live inference memory reserved and never preempted |
| Worker crash | JetStream redelivery, resume from checkpoint |
| Disk nearly full | alert; retention jobs prune per 20; ingestion of raw media paused first |
| Infra container restart | services reconnect with backoff; trading halts until health restored. The ingestor retries its Postgres connect for 60 s at start instead of exiting |
| Host reboot / power loss with rootless Docker | the user-level dockerd starts only at login unless lingering is enabled; then `restart: unless-stopped` starts every container **at once** (compose `depends_on` is not applied by the daemon), so services must tolerate dependencies that are still starting. Seen 2026-11-11: host powered off 22:40 CET, booted 02:37, login 02:43 → all containers started 01:43 UTC; the ingestor's first start failed (Postgres not yet accepting) and was restarted by the policy. Hardening: every service has a healthcheck (Rust: curl `/metrics`, Go: `healthprobe`, garage: `garage status`), the ingestor retries Postgres; operators who want the stack up without login run `loginctl enable-linger <user>` (host setting, outside the repo) |
| Alerting pipeline down | `Watchdog` stops arriving → `/api/alerts` `pipeline.ok = false`; vmalert down → `InfraScrapeDown{job="vmalert"}` only fires once it is back, so the UI must treat `pipeline.ok = false` as an alert of its own |
| Power loss | recovery order §6.4; reconciliation mandatory |

## 9. Acceptance criteria
- `just up` brings all five infra services healthy; `just dev` starts all app services.
- Killing an evo-worker mid-job results in the job completing on another worker from its checkpoint.
- Submitting a live-inference-class job preempts a training job within the deadline.
- Every service appears in `/api/services` with heartbeat and exposes metrics scraped into VictoriaMetrics.
- `just backup` then `just restore` on a clean machine reproduces ledger and Helix state.
- `just restore-drill` passes on the newest scheduled backup, including PITR past the base backup, without touching live data.
- `just alert-drill` sees the firing and the resolved `ops.alert` message, the `/api/alerts` entry and the audit row.
- No secrets in the repository; secret scan passes.

## 10. Open questions
1. Where does the GPU arbiter live (orchestrator vs dedicated service)? — blocks scheduler implementation.
2. Process supervision for app services (systemd user units, compose, or `just dev` only)? — shared with 50 Q1.
3. Logs store: plain files vs VictoriaLogs.
4. Off-machine backup target (hook exists, disabled: `FA_BACKUP_OFFSITE_*`).
5. Disk budget for raw media archive and retention policy (with 20).

## 11. Deferred
- Multi-machine deployment tooling (Nomad/k8s).
- Cloud bursting for evolution.
- Auth/TLS.
