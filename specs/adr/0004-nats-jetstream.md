# ADR 0004 — NATS JetStream as message bus and job queue

**Status:** accepted (D-001) · **Date:** 2026-11-06

## Context
Rust and Go services need a low-latency pub/sub bus for market data and signals, durable streams for replay and audit, request/reply for control, and work queues so evolution/training/media jobs can scale horizontally (single GPU now, more machines later).

## Decision
Use **NATS with JetStream**: subjects per spec 21, limits-based streams for data, **work-queue streams** for jobs (stateless Rust workers pull), request/reply for `ctl.*`. Clients: `nats.go`, `async-nats` (Rust).

## Consequences
- One lightweight binary covers pub/sub, persistence, queues, KV; trivially runs in compose.
- Horizontal scaling = start more workers; clustering later via NATS cluster/leaf nodes.
- JetStream retention is bounded; long-term history lives in VictoriaMetrics/Garage/Postgres, not in NATS.
- At-least-once delivery → all consumers must be idempotent.

## Alternatives considered
- **Kafka/Redpanda:** stronger log semantics, heavier ops, weaker request/reply and work-queue ergonomics.
- **Redis Streams:** simple, weaker durability/clustering story.
- **gRPC job dispatch:** rejected by user in favour of NATS work queues.
