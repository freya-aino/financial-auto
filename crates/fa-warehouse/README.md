# fa-warehouse

PostgreSQL warehouse access: reference data, catalog, fundamentals, macro vintages.

**Status:** implemented (thin) — PgPool wrapper, symbol resolution (validity-aware), catalog listing/lookup, `sync_catalog(&[Source])` upserting `config/sources.yaml` rows (parsed with `serde_yaml`, same shape as `go/pkg/catalog`) into `catalog.source`, point-in-time macro query, and the append-only `audit::append` / `audit::since` helpers over `audit.log` (spec 42). Since D-061 there are two more modules:
- `venues`: `load_instruments` → `RefTable` of `InstrumentRef {asset_class, kind, venue, quote_ccy, status, sources}` from the current `ref.instrument` / `ref.listing` / `ref.venue` rows, used by the risk chain for class/venue caps, provisional caps and the per-source flatten rule.
- `fx`: `parse_ecb_daily` (quick-xml, fixture `testdata/eurofxref-daily.xml`) → `FxRate`s with `source = ecb_daily`; `upsert_rates` writes them idempotently on `(ccy, ts, source_id)`; `rate_as_of` / `load_recent` read them; `FxTable::convert(ccy, t, max_age)` does the as-of conversion (latest rate with `ts ≤ t`, staleness flagged, EUR = 1.0).

All tested against the live DB when `DATABASE_URL` is set (unique ids per run, cleaned up).

## Owns
Repositories over `migrations/` schema.

## Governing specs
- [12-reference-data](../../specs/12-reference-data.md)
- [20-storage](../../specs/20-storage.md)
