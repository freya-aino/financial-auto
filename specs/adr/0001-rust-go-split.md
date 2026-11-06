# ADR 0001 — Rust core, Go for parallel/scalable services

**Status:** accepted · **Date:** 2026-11-06

## Context
The platform mixes latency-sensitive numeric work (model inference, MCMC, risk), heavy ML training, and highly concurrent I/O (≈70 external APIs, broker connections, job orchestration). The user wants the system mostly in Rust, with Go for services that need to be parallel and scalable.

## Decision
- **Rust:** domain core (`fa-core`), ML (encoders, decoders, evolution workers), model runtime, MCMC, strategies, allocator, risk, ledger, backtest, control API.
- **Go:** connectors + `ingestor`, `broker-gateway` (execution adapters), `evo-orchestrator` (population scheduling). VictoriaMetrics (Go) is used as a service.
- The boundary is **NATS + protobuf** (ADR 0004, 0005); no FFI between Rust and Go.

## Consequences
- Two toolchains, two workspaces (`Cargo.toml`, `go.work`); shared contracts must live in `proto/`.
- Domain types are duplicated in generated form; hand-written domain logic stays on one side only.
- Risk checks are in Rust; the Go broker-gateway only executes **approved** intents (`order.approved.*`), so safety logic is not split.
- Go's goroutines suit hundreds of API connections and fan-out scheduling; Rust keeps numeric hot paths and memory control.

## Alternatives considered
- **All Rust:** viable (tokio) but user explicitly wants Go for scalable services; Go SDKs exist for more broker/exchange APIs.
- **Python services:** rejected — user chose pure Rust ML; no Python runtime in production.
- **FFI (cgo ↔ Rust):** rejected — couples build systems and crash domains.
