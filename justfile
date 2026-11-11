# Task runner (D-019). Install: `cargo install just`.
# Go, Node and buf fall back to Docker images when not installed locally (rootless Docker: no --user).

root := justfile_directory()
compose := "docker compose -f " + root + "/compose.yaml"
go_image := "golang:1.25-alpine"
node_image := "node:22-alpine"
buf_image := "bufbuild/buf:latest"
database_url := env_var_or_default("DATABASE_URL", "postgres://fa:fa@127.0.0.1:5432/fa")

default:
    @just --list

# ── infrastructure ──────────────────────────────────────────────────────

# Start NATS, Postgres, VictoriaMetrics, Garage
up:
    {{compose}} up -d
    {{compose}} ps

# Stop infrastructure (keeps volumes)
down:
    {{compose}} down

# One-time Garage layout, buckets and key
garage-init:
    {{root}}/docker/garage-init.sh

# Apply Postgres migrations (needs sqlx-cli: cargo install sqlx-cli --no-default-features --features rustls,postgres)
migrate:
    sqlx migrate run --source {{root}}/migrations --database-url {{database_url}}

# ── app stack in Docker (D-026) ─────────────────────────────────────────

# Build the Rust (incl. frontend) and Go service images
app-build:
    {{compose}} --profile app build

# Start infrastructure + all platform services; UI at http://127.0.0.1:8080, API at :7700.
# Then waits in the spec 60 §6.4 order (infra → control-api /api/health → risk-engine heartbeat).
app-up:
    {{compose}} --profile app up -d
    {{compose}} --profile app ps
    {{root}}/scripts/healthwait.sh

# Wait for the running stack to be healthy, printing the §6.4 recovery steps: just healthwait [timeout]
healthwait timeout="180":
    {{root}}/scripts/healthwait.sh {{timeout}}

# Stop everything (keeps volumes)
app-down:
    {{compose}} --profile app --profile tools down

# Follow logs: just app-logs [service]
app-logs *services:
    {{compose}} --profile app logs -f --tail 100 {{services}}

app-ps:
    {{compose}} --profile app ps

# Run more evaluation workers: just app-scale-workers 4
app-scale-workers n:
    {{compose}} --profile app up -d --scale evo-worker={{n}}

# Run the trainer CLI in a container: just trainer pretrain --help
trainer *args:
    {{compose}} --profile tools run --rm trainer {{args}}

# ── observability (spec 60 §6.5, D-056) ────────────────────────────────

# Every service's /metrics is scraped by VictoriaMetrics (docker/vm-scrape.yaml); shows the `up` series
metrics-up:
    curl -fsS 'http://127.0.0.1:8428/api/v1/query?query=up' | sed 's/},{/},\n{/g'

# Dump one service's /metrics from the host: just metrics risk-engine  (ports: docker/README.md)
metrics service="control-api":
    #!/usr/bin/env sh
    set -eu
    case "{{service}}" in
        control-api) p=9101 ;; model-runtime) p=9102 ;; strategy-runtime) p=9103 ;; risk-engine) p=9104 ;;
        ingestor) p=9105 ;; broker-gateway) p=9106 ;; evo-orchestrator) p=9107 ;; evo-worker) p=9110 ;;
        *) echo "unknown service {{service}}" >&2; exit 2 ;;
    esac
    curl -fsS "http://127.0.0.1:$p/metrics"

# ── backups (spec 60 §6.6, spec 20 §6.6, D-056) ─────────────────────────

# Postgres dump + HelixDB volume tar (helix paused for seconds) + Garage critical buckets → backups/<timestamp>/
# The scheduled backup (backup.schedule 03:00) runs in the `backup` compose sidecar, not host cron.
backup *args:
    {{root}}/scripts/backup.sh {{args}}

# Run the sidecar's scheduled backup now (PITR base backup + dump + hot Helix tar + buckets + pruning)
backup-now:
    {{compose}} exec backup run.sh

# Restore drill into throwaway containers, live data untouched: just restore-drill [backups/<ts>] [--target TIME]
restore-drill *args:
    {{root}}/scripts/restore-drill.sh {{args}}

# End-to-end alerting check: rule → vmalert → control-api → NATS ops.alert → audit.log
alert-drill *args:
    {{root}}/scripts/alert-drill.sh {{args}}

# Active and recent alerts as control-api sees them (vmalert UI: http://127.0.0.1:8880)
alerts:
    curl -fsS 'http://127.0.0.1:7700/api/alerts'

# Restore a backup directory in the §6.4 order (asks for confirmation; `--yes` skips it):
#   just restore backups/<ts>                       # overwrite live Postgres/Helix/Garage
#   just restore backups/<ts> --postgres-db fa_restore_test   # drill: Postgres only, into a scratch database
#   just restore backups/<ts> --pitr [--target '2026-11-11 03:00:00+00']   # point-in-time recovery (scheduled backups)
restore dir *args:
    {{root}}/scripts/restore.sh {{dir}} {{args}}

# ── contracts ───────────────────────────────────────────────────────────

# Lint protos and regenerate Go code (Rust codegen happens in fa-proto/build.rs)
proto:
    #!/usr/bin/env sh
    set -eu
    cd "{{root}}/proto"
    if command -v buf >/dev/null; then
        buf lint && buf generate
    else
        docker run --rm -e HOME=/tmp -v "{{root}}":/repo -w /repo/proto {{buf_image}} lint
        docker run --rm -e HOME=/tmp -v "{{root}}":/repo -w /repo/proto {{buf_image}} generate
    fi

# ── rust ────────────────────────────────────────────────────────────────

rust-check:
    cargo check --workspace --all-targets

rust-test:
    cargo test --workspace

rust-clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# ── go ──────────────────────────────────────────────────────────────────

# Run a go command natively or in Docker: just go build ./...
go *args:
    #!/usr/bin/env sh
    set -eu
    cd "{{root}}/go"
    if command -v go >/dev/null; then
        go {{args}}
    else
        docker run --rm -v "{{root}}/go":/src -w /src -v fa-gomod:/go/pkg/mod -v fa-gocache:/root/.cache/go-build {{go_image}} go {{args}}
    fi

go-build: (go "build" "./...") (go "vet" "./...")

go-test: (go "test" "./...")

# ── frontend ────────────────────────────────────────────────────────────

# Run pnpm natively or in Docker: just pnpm install
pnpm *args:
    #!/usr/bin/env sh
    set -eu
    cd "{{root}}/frontend"
    if command -v pnpm >/dev/null; then
        pnpm {{args}}
    else
        docker run --rm $([ -t 0 ] && echo -it) -v "{{root}}/frontend":/app -w /app -v fa-pnpm:/pnpm -e PNPM_HOME=/pnpm \
            -e COREPACK_ENABLE_DOWNLOAD_PROMPT=0 --network host {{node_image}} \
            sh -c 'corepack enable >/dev/null && pnpm {{args}}'
    fi

fe-install: (pnpm "install" "--frozen-lockfile")

# Vite dev server on :5173, proxying /api and /ws to control-api on :7700
fe-dev: (pnpm "dev")

fe-build: (pnpm "check") (pnpm "build")

# ── run ─────────────────────────────────────────────────────────────────

control-api:
    cargo run -p control-api

# Fake heartbeats + trades for UI work without the ingestor or model-runtime running
fake-publisher:
    cargo run -p control-api --example fake_publisher

# How to run the system locally
dev:
    @echo "1. just up            # infrastructure (first time also: just garage-init && just migrate)"
    @echo "2. just control-api   # http://127.0.0.1:7700 (serves frontend/build when present)"
    @echo "3. just fe-dev        # http://127.0.0.1:5173 with hot reload"
    @echo "4. go -C go run ./services/ingestor -catalog config/sources.yaml   # heartbeat feed"
    @echo "5. cargo run -p model-runtime / strategy-runtime / risk-engine / evo-worker"
    @echo "   Go: go -C go run ./services/broker-gateway, ./services/evo-orchestrator"
    @echo "   (or everything in compose: just app-up; synthetic data only: just fake-publisher)"

# Everything: Rust, Go, protos, frontend
check: rust-check rust-test rust-clippy proto go-build go-test fe-install fe-build
