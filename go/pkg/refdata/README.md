# refdata

Postgres side of ingestion bookkeeping: symbol resolution over `ref.*`, catalog sync, run log and cursors (`ingest.*`). Driver: pgx v5.

**Status:** implemented, live-tested against the compose Postgres (`FA_POSTGRES_URL` gates the test).

- `Store.SyncCatalog`: upserts `config/sources.yaml` into `catalog.source` (status kept when the row exists — Postgres is authoritative at runtime, spec 11). Needed because every `ingest.*` row references `catalog.source`.
- `Resolver` (`connector.SymbolResolver`): `(source, venue_symbol)` → instrument UUID via `ref.listing` (then `ref.identifier`), cached. Unknown pairs are **provisioned**: `ref.venue` (code from `DefaultHints`, e.g. `yahoo`, `lemon`, `oanda`), `ref.instrument` with `status = 'provisional'` and the deterministic id `uuid5(Namespace, source + "\0" + symbol)`, `ref.listing`, `ref.identifier` (`scheme` e.g. `isin`, `yahoo_symbol`) and an `ingest.quarantine` row with `status = 'open'` so the manual-review queue shows what was auto-created (spec 12 §6 variant; D-050). DB errors are reported through `onErr` and leave the event unresolved.
- `Cursors` (`connector.CursorStore`): `ingest.backfill_cursor` rows keyed `(source, instrument|null, timescale = scope, range_start = epoch)`; `cursor`/`range_end` hold the progress time.
- `RunLog`: `ingest.connector_run` row per connector start (`running`) and stop (`stopped`, or `failed` when the connector ended on its own) with event/error deltas.

## Governing specs
- [10-ingestion](../../../specs/10-ingestion.md) §4.2, §6.4; [12-reference-data](../../../specs/12-reference-data.md); [20-storage](../../../specs/20-storage.md) §5.4
