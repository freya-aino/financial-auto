# syntax=docker/dockerfile:1.7
# Rust services image (D-026): control-api, model-runtime, evo-worker, trainer
# plus the built frontend. Build context = repo root.

# ── frontend ────────────────────────────────────────────────────────────
FROM node:22-alpine AS frontend
ENV PNPM_HOME=/pnpm COREPACK_ENABLE_DOWNLOAD_PROMPT=0
RUN corepack enable
WORKDIR /src/frontend
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN --mount=type=cache,id=fa-pnpm-store,target=/pnpm/store \
    pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

# ── rust build ──────────────────────────────────────────────────────────
FROM rust:1.98-slim-trixie AS builder
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates crates
COPY bins bins
COPY proto proto
COPY migrations migrations
RUN --mount=type=cache,id=fa-cargo-registry,target=/usr/local/cargo/registry \
    --mount=type=cache,id=fa-cargo-git,target=/usr/local/cargo/git \
    --mount=type=cache,id=fa-cargo-target,target=/src/target \
    cargo build --release --locked -p control-api -p model-runtime -p evo-worker -p trainer \
    && mkdir -p /out \
    && cp target/release/control-api target/release/model-runtime \
          target/release/evo-worker target/release/trainer /out/

# ── runtime ─────────────────────────────────────────────────────────────
FROM debian:trixie-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home /app fa
WORKDIR /app
COPY --from=builder /out/ /usr/local/bin/
COPY config/platform.toml config/platform.docker.toml config/sources.yaml config/
COPY --from=frontend /src/frontend/build frontend/build
ENV FA_CONFIG=/app/config/platform.docker.toml \
    FA_SOURCES=/app/config/sources.yaml \
    FA_FRONTEND=/app/frontend/build \
    RUST_LOG=info
USER fa
CMD ["control-api"]
