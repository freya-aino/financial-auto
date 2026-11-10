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
  | control-api | 7700 (D-014) |
  | ui (compose, nginx) | 8080 (D-035) |
  | frontend (vite dev) | 5173 |
- `justfile` recipes: `up`, `down`, `dev`, `check`, `test`, `fmt`, `proto`, `migrate`, `backup`, `restore`.
- Every service exposes: heartbeat on `svc.heartbeat.<service>` (every 5 s, as in 21 and the code), Prometheus metrics endpoint scraped by VictoriaMetrics (`vmagent` or VM scrape config), structured JSON logs to stdout.
- Work-queue envelope (21): `Job { id, kind, priority_class, payload_ref (Garage key) | inline, budget, checkpoint_ref?, attempt }` → `JobResult { id, status, result_ref, metrics }`.

## 5. Data model
| Item | Store |
|---|---|
| Job records & status | Postgres `job` + JetStream work-queue streams `JOBS_<kind>` |
| Checkpoints | Garage `checkpoints/<job_id>/…` |
| Metrics | VictoriaMetrics |
| Logs | stdout → file/journald **(proposal: VictoriaLogs later)** |
| Backups | local backup dir / second disk, Garage bucket `backups/` replicated off-machine **(proposal)** |

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
- Alerts (proposal): VM alerting rules → `ops.alert` subject → UI banner.

**Implemented today (D-056):** every service serves Prometheus text on `GET /metrics` (`crates/fa-metrics`, re-exported as `fa_bus::metrics`; Go `go/pkg/bus/metrics.go`): heartbeat counters mirrored as `fa_heartbeat_counter{service,key}`, process gauges, JetStream consumer lag, and control-api HTTP counts/latency. VictoriaMetrics scrapes all targets itself (`docker/vm-scrape.yaml`, `-promscrape.config`); host ports 9101..9107 and 9110. Alert rules exist only as a proposal file (`docker/vm-alerts.proposal.yaml`); no vmalert, no `ops.alert` yet.

### 6.6 Backups
- Postgres: nightly `pg_dump` + WAL archiving **(proposal)**.
- Helix: nightly snapshot of data directory (service paused or via native snapshot if available).
- Garage: bucket replication/rsync of critical buckets (models, raw archive) to a second disk.
- `just restore` tested quarterly **(proposal)**.

**Implemented today (D-056):** `scripts/backup.sh` (`just backup`) writes `backups/<ts>/` with `pg_dump -Fc` + globals, a tar of the `helix-data` volume (helix stopped for seconds), rclone copies of the critical Garage buckets, optional `nats stream backup`, and a sha256 manifest; `scripts/restore.sh DIR` (`just restore`) verifies, confirms, stops the app services and restores in the §6.4 order (`--postgres-db` for scratch drills, verified). WAL archiving and the off-machine copy are still open.

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
| `backup.schedule` | 03:00 daily | |

## 8. Failure modes
| Failure | Response |
|---|---|
| GPU OOM | job fails, retried with smaller batch; live inference memory reserved and never preempted |
| Worker crash | JetStream redelivery, resume from checkpoint |
| Disk nearly full | alert; retention jobs prune per 20; ingestion of raw media paused first |
| Infra container restart | services reconnect with backoff; trading halts until health restored |
| Power loss | recovery order §6.4; reconciliation mandatory |

## 9. Acceptance criteria
- `just up` brings all five infra services healthy; `just dev` starts all app services.
- Killing an evo-worker mid-job results in the job completing on another worker from its checkpoint.
- Submitting a live-inference-class job preempts a training job within the deadline.
- Every service appears in `/api/services` with heartbeat and exposes metrics scraped into VictoriaMetrics.
- `just backup` then `just restore` on a clean machine reproduces ledger and Helix state.
- No secrets in the repository; secret scan passes.

## 10. Open questions
1. Where does the GPU arbiter live (orchestrator vs dedicated service)? — blocks scheduler implementation.
2. Process supervision for app services (systemd user units, compose, or `just dev` only)? — shared with 50 Q1.
3. Logs store: plain files vs VictoriaLogs.
4. Off-machine backup target.
5. Disk budget for raw media archive and retention policy (with 20).

## 11. Deferred
- Multi-machine deployment tooling (Nomad/k8s).
- Cloud bursting for evolution.
- Auth/TLS.
