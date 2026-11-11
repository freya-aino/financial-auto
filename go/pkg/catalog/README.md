# catalog

Loads and validates `config/sources.yaml`, syncs it into Postgres.

**Status:** implemented — `Load`/`Parse` + `Validate` of `config/sources.yaml` (unique ids, enum values for status/auth/capabilities/transport, `rate_limit` > 0, symbols, non-empty queries, timeframe ∈ `1m 5m 15m 1h 1d`); optional `rate_limit {per_second, burst}` (`Source.Limit()`, default 5/s burst 10 when absent) and connector `config:` block (`Source.Params()`: `symbols`, `queries`, `timeframe`, `extra`); the `backfill.knowledge_delay_secs` block (D-079) is not read here (`fa_backtest::knowledge` reads it); `SetStatus` / `ApplyStatuses` let the ingestor overlay runtime statuses (Postgres wins over the YAML seed, D-054); `BackfillDepth(timescale)` gives the spec 10 §3 default history depth (seconds 4 w, minutes 1 y, hours 5 y, days 50 y). `ByID`/`SetStatus` are safe for concurrent use.

## Owns
Catalog model and loader.

## Governing specs
- [11-source-catalog](../../../specs/11-source-catalog.md)
