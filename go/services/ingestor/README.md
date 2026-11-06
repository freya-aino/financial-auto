# ingestor

Runs N connectors concurrently, normalizes to canonical events, archives raw payloads, publishes to NATS.

**Status:** implemented (v1) — runs enabled connectors (`-connectors`, default `heartbeat`) concurrently, normalizes, routes and publishes envelopes, per-source limiter, heartbeat counters `events.<id>` / `errors.<id>` / `unresolved.<id>`, graceful shutdown. Raw archive and JetStream streams pending.

## Owns
Connector supervisor process.

## Governing specs
- [10-ingestion](../../../specs/10-ingestion.md)
- [21-bus-and-contracts](../../../specs/21-bus-and-contracts.md)
