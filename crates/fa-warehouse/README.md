# fa-warehouse

PostgreSQL warehouse access: reference data, catalog, fundamentals, macro vintages.

**Status:** implemented (thin) — PgPool wrapper, symbol resolution (validity-aware), catalog listing/lookup, `sync_catalog(&[Source])` upserting `config/sources.yaml` rows (parsed with `serde_yaml`, same shape as `go/pkg/catalog`) into `catalog.source`, point-in-time macro query, and the append-only `audit::append` / `audit::since` helpers over `audit.log` (spec 42). All tested against the live DB when `DATABASE_URL` is set (unique ids per run, cleaned up).

## Owns
Repositories over `migrations/` schema.

## Governing specs
- [12-reference-data](../../specs/12-reference-data.md)
- [20-storage](../../specs/20-storage.md)
