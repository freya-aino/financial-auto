# connector

Connector interface (Stream / Poll / Backfill), rate limiting, retries.

**Status:** implemented — interfaces per spec 10 §4.1, registry, token-bucket limiter, `StaticResolver`, `NopArchive` (Garage writer and Postgres resolver pending).

## Owns
Go interface every connector implements.

## Governing specs
- [10-ingestion](../../../specs/10-ingestion.md)
