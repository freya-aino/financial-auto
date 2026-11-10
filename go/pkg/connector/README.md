# connector

Connector interface (Stream / Poll / Backfill / Normalize), rate limiting, retries, secrets.

**Status:** implemented — interfaces per spec 10 §4.1, registry, token-bucket limiter, plus:
- `Fetcher`: rate-limited HTTP with bounded retries (429/5xx/network, exponential backoff + jitter, spec 10 §6.3), fixed `UserAgent`, per-request `Prepare` hook for auth headers.
- `LoadSecrets` / `Secrets`: minimal reader for `config/secrets.toml` (`[garage]`, `[sources.<id>]`); `FA_<SOURCE>_<KEY>` environment variables take precedence (D-020). `SourceConfig.Secret(key)` for connectors.
- `CursorStore` (+ `MemCursors`): persisted progress per `(source, instrument?, scope)`; Postgres implementation in `go/pkg/refdata`.
- `Seen`: bounded in-process de-dup set.
- `Deps.Cursors`, `TransportMeta` keys (`endpoint`, `symbol`, `tf`, `backfill`, `url`).
- `StaticResolver` / `NopArchive` remain the no-infrastructure fallbacks (Postgres resolver: `go/pkg/refdata`, Garage writer: `go/pkg/archive`).

## Owns
Go interface every connector implements.

## Governing specs
- [10-ingestion](../../../specs/10-ingestion.md)
