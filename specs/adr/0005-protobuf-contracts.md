# ADR 0005 — Protobuf as the Rust↔Go contract

**Status:** proposed · **Date:** 2026-11-06

## Context
Every cross-process message (canonical events, latent states, orders, jobs) is produced in one language and consumed in another. Contracts must be typed, versioned, compact and generated for both Rust and Go.

## Decision
- Schemas in `proto/fa/v1/*.proto`, managed with **buf** (`buf lint`, `buf breaking`, `buf generate`).
- Rust codegen via **prost** in `crates/fa-proto` (build.rs or buf plugin); Go via **protoc-gen-go** into `go/pkg/bus/pb`.
- Additive-only changes within `fa.v1`; breaking changes create `fa.v2`.

## Consequences
- Single source of truth for wire types; compile-time checking on both sides.
- Generated types are not domain types — Rust wraps them in `fa-core` where invariants matter.
- Binary payloads are not human-readable on the bus; the control API / UI decodes for display.

## Alternatives considered
- **JSON + JSON Schema:** readable, slower, weaker typing.
- **Cap'n Proto / FlatBuffers:** zero-copy, weaker Go/Rust tooling parity and ecosystem.
- **Avro + registry:** extra service, less natural in Rust.
