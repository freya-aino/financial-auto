# syntax=docker/dockerfile:1.7
# Rust services image (D-026): control-api, model-runtime, evo-worker, trainer, strategy-runtime, risk-engine.
# The UI is its own image (ui.Dockerfile, D-035). Build context = repo root.

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
    # The target cache outlives the source tree: if it was last filled from another checkout with
    # newer mtimes, cargo would reuse stale workspace crates. Stamping the workspace sources "now"
    # makes them always newer than any cached artifact; registry dependencies stay cached.
    find crates bins -type f -exec touch {} + \
    && cargo build --release --locked -p control-api -p model-runtime -p evo-worker -p trainer \
          -p strategy-runtime -p risk-engine \
    && mkdir -p /out \
    && cp target/release/control-api target/release/model-runtime \
          target/release/evo-worker target/release/trainer \
          target/release/strategy-runtime target/release/risk-engine /out/

# ── runtime ─────────────────────────────────────────────────────────────
FROM debian:trixie-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home /app fa
WORKDIR /app
COPY --from=builder /out/ /usr/local/bin/
COPY config/platform.toml config/platform.docker.toml config/sources.yaml config/
ENV FA_CONFIG=/app/config/platform.docker.toml \
    FA_SOURCES=/app/config/sources.yaml \
    RUST_LOG=info
USER fa
CMD ["control-api"]
