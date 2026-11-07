# Task runner (D-019). Install: `cargo install just`.
# Go, Node and buf fall back to Docker images when not installed locally (rootless Docker: no --user).

root := justfile_directory()
compose := "docker compose -f " + root + "/deploy/compose.yaml"
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
    {{root}}/deploy/garage-init.sh

# Apply Postgres migrations (needs sqlx-cli: cargo install sqlx-cli --no-default-features --features rustls,postgres)
migrate:
    sqlx migrate run --source {{root}}/migrations --database-url {{database_url}}

# ── app stack in Docker (D-026) ─────────────────────────────────────────

# Build the Rust (incl. frontend) and Go service images
app-build:
    {{compose}} --profile app build

# Start infrastructure + all platform services; UI at http://127.0.0.1:7700
app-up:
    {{compose}} --profile app up -d
    {{compose}} --profile app ps

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

# Fake heartbeats + trades for UI work before ingestor/model-runtime exist
fake-publisher:
    cargo run -p control-api --example fake_publisher

# How to run the system locally
dev:
    @echo "1. just up            # infrastructure (first time also: just garage-init && just migrate)"
    @echo "2. just control-api   # http://127.0.0.1:7700 (serves frontend/build when present)"
    @echo "3. just fe-dev        # http://127.0.0.1:5173 with hot reload"
    @echo "4. ingestor + model-runtime: see go/services/ingestor and bins/model-runtime READMEs"
    @echo "   (until they exist: just fake-publisher)"

# Everything: Rust, Go, protos, frontend
check: rust-check rust-test rust-clippy proto go-build go-test fe-install fe-build
