# ingestor

Runs N connectors concurrently, normalizes to canonical events, archives raw payloads, publishes to NATS.

**Status:** spec only — no code yet.

## Owns
Connector supervisor process.

## Governing specs
- [10-ingestion](../../../specs/10-ingestion.md)
- [21-bus-and-contracts](../../../specs/21-bus-and-contracts.md)
