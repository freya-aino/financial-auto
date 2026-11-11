# fa-warehouse

PostgreSQL warehouse access: reference data, catalog, fundamentals, macro vintages.

**Status:** implemented — PgPool wrapper, symbol resolution (validity-aware), catalog listing/lookup, `sync_catalog(&[Source])` upserting `config/sources.yaml` rows (parsed with `serde_yaml`, same shape as `go/pkg/catalog`) into `catalog.source`, point-in-time macro query, and the append-only `audit::append` / `audit::since` helpers over `audit.log` (spec 42). Modules:
- `refdata` (D-054): instrument search over `ref.instrument`/`ref.identifier`/`ref.listing`, the quarantine queue and its resolution (activate in place or merge); `mirror_rows(pool, only)` (D-084) reads what the Helix reference mirror needs, valid now: instruments (any status, `merged` included) with their current listings and `same_economic_exposure` relations, entities and all venues (`only = None` full sync, `Some(ids)` those instruments and their issuers); control-api's `ref_mirror.rs` writes them through `fa_hub::reference`.
- `asof` (D-067, spec 12 §4/§5): point-in-time reads taking `at` and optional `known_at` (visible when `valid_from ≤ at < valid_to` and `recorded_at ≤ known_at`): `resolve`/`resolve_source_symbol`/`resolve_identifier` (none / one / ambiguous), `instrument`, `contract_specs`, `fx_to_eur`, plus `add_identifier`/`upsert_contract_spec`.
- `calendar` (D-067): built-in rules for US equities, Xetra, FX 24/5 and always-open crypto (holidays, early closes, DST without a tz database); `materialize` writes sessions into `ref.calendar_session` (source-carrying rows never overwritten); `session_state_at` reads the table, falls back to the rule, and reports a venue with neither as closed (`missing_calendar`).
- `corpactions` (D-067): `record`/`cancel`/`list_as_of` (point-in-time by `recorded_at`), `adjustment_factor(_db)`, `adjust_bars` (on read, price-only unless total return is asked for) and `symbol_change` (closes the old listing and opens the new one at the effective time; refused when the new ticker is then listed for another instrument).
- `fundamentals` (D-078): as-of reads of `wh.fundamental` (SEC EDGAR XBRL facts keyed by `cik10`, vintaged by filing date).
- `ops` (D-078, migration `20261113000013`): alert history, job runs and config snapshots written by control-api.
- `venues`: `load_instruments` → `RefTable` of `InstrumentRef {asset_class, kind, venue, quote_ccy, status, sources}` from the current `ref.instrument` / `ref.listing` / `ref.venue` rows, used by the risk chain for class/venue caps, provisional caps and the per-source flatten rule.
- `fx`: `parse_ecb_daily` (quick-xml, fixture `testdata/eurofxref-daily.xml`) → `FxRate`s with `source = ecb_daily`; `upsert_rates` writes them idempotently on `(ccy, ts, source_id)`; `rate_as_of` / `load_recent` read them; `FxTable::convert(ccy, t, max_age)` does the as-of conversion (latest rate with `ts ≤ t`, staleness flagged, EUR = 1.0).

All tested against the live DB when `DATABASE_URL` is set (unique ids per run, cleaned up).

## Owns
Repositories over `migrations/` schema.

## Governing specs
- [12-reference-data](../../specs/12-reference-data.md)
- [20-storage](../../specs/20-storage.md)
