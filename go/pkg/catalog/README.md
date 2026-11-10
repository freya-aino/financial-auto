# catalog

Loads and validates `config/sources.yaml`, syncs it into Postgres.

**Status:** implemented — loads and validates `config/sources.yaml` (unique ids, enum values); optional `rate_limit` field with a default when absent; `SetStatus` / `ApplyStatuses` let the ingestor overlay runtime statuses (Postgres wins over the YAML seed, D-054); `BackfillDepth(timescale)` gives the spec 10 §3 default history depth (seconds 4 w, minutes 1 y, hours 5 y, days 50 y). `ByID`/`SetStatus` are safe for concurrent use.

## Owns
Catalog model and loader.

## Governing specs
- [11-source-catalog](../../../specs/11-source-catalog.md)
