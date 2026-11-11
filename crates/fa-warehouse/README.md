# fa-warehouse

PostgreSQL warehouse access: reference data, catalog, fundamentals, macro vintages.

**Status:** implemented — PgPool wrapper, symbol resolution (validity-aware), catalog listing/lookup, `sync_catalog(&[Source])` upserting `config/sources.yaml` rows (parsed with `serde_yaml`, same shape as `go/pkg/catalog`) into `catalog.source`, point-in-time macro query, and the append-only `audit::append` / `audit::since` helpers over `audit.log` (spec 42). Modules:
- `refdata` (D-054): instrument search over `ref.instrument`/`ref.identifier`/`ref.listing`, the quarantine queue and its resolution (activate in place or merge); `mirror_rows(pool, only)` (D-084) reads what the Helix reference mirror needs, valid now: instruments (any status, `merged` included) with their current listings and `same_economic_exposure` relations, entities and all venues (`only = None` full sync, `Some(ids)` those instruments and their issuers); control-api's `ref_mirror.rs` writes them through `fa_hub::reference`.
- `asof` (D-067, spec 12 §4/§5): point-in-time reads taking `at` and optional `known_at` (visible when `valid_from ≤ at < valid_to` and `recorded_at ≤ known_at`): `resolve`/`resolve_source_symbol`/`resolve_identifier` (none / one / ambiguous), `instrument`, `contract_specs`, `fx_to_eur`, plus `add_identifier`/`upsert_contract_spec`.
- `calendar` (D-067): built-in rules for US equities, Xetra, FX 24/5 and always-open crypto (holidays, early closes, DST without a tz database); `materialize` writes sessions into `ref.calendar_session` (source-carrying rows never overwritten, unchanged sessions not rewritten so their `recorded_at` stays); `refresh(pool, today, horizon)` is the scheduled refresh (every venue with a rule except 24/7 ones, next 400 days; control-api `refdata_maint.rs` runs it daily); `session_state_at` reads the table, falls back to the rule, and reports a venue with neither as closed (`missing_calendar`).
- `adjusted` (migration `20261114030000`): materialised adjusted daily bars in `wh.adjusted_bar_1d` (both modes, back-adjusted to the latest bar, with the raw close, the factor and the digest of the action set) for every instrument with splits or dividends: `adjusted_rows` (pure), `materialize_daily`, `read_daily`, `refresh(pool, fetch)` (control-api feeds it raw `fa_bar_*{tf="1d"}` from VictoriaMetrics, daily), `daily_point` (trading day = UTC date one day before the bar's close).
- `corpactions` (D-067): `record`/`cancel`/`list_as_of` (point-in-time by `recorded_at`), `adjustment_factor(_db)`, `adjust_bars` (on read, price-only unless total return is asked for) and `symbol_change` (closes the old listing and opens the new one at the effective time; refused when the new ticker is then listed for another instrument).
- `fundamentals` (D-078): as-of reads of `wh.fundamental` (SEC EDGAR XBRL facts keyed by `cik10`, vintaged by filing date).
- `ops` (D-078, migration `20261113000013`): alert history, job runs and config snapshots written by control-api.
- `venues`: `load_instruments` → `RefTable` of `InstrumentRef {asset_class, kind, venue, quote_ccy, status, sources}` from the current `ref.instrument` / `ref.listing` / `ref.venue` rows, used by the risk chain for class/venue caps, provisional caps and the per-source flatten rule.
- `fx`: `parse_ecb_daily` (quick-xml, fixture `testdata/eurofxref-daily.xml`) → `FxRate`s with `source = ecb_daily`; `upsert_rates` writes them idempotently on `(ccy, ts, source_id)`; `rate_as_of` / `load_recent` read them; `FxTable::convert(ccy, t, max_age)` does the as-of conversion (latest rate with `ts ≤ t`, staleness flagged, EUR = 1.0).

All tested against the live DB when `DATABASE_URL` is set (unique ids per run, cleaned up).

## Owns
Repositories over `migrations/` schema.

## Governing specs
- [Reference data reference](#reference-data-reference) below (spec 12, retired 2026-11-11; `spec 12 §x` in code comments refers to it, see git history)
- [20-storage](../../specs/20-storage.md)

## Reference data reference
Lasting content of spec 12 (Reference data), retired 2026-11-11.

**Purpose.** Give every tradeable or observable thing one stable identity and the static facts needed to trade and model it: identifiers, venues, contract specs, trading calendars, corporate actions, EUR FX rates. Fundamentals/filings content, prices (VictoriaMetrics) and latent representations (Helix) are out of scope.

**Decisions.** The internal instrument UUID is the only identity used downstream. Mappings (FIGI, ISIN, LEI, CIK, venue symbols) live in Postgres `ref` (authoritative) and are mirrored to Helix (instrument ↔ issuer ↔ venue ↔ sector). Calendars per venue, 24/7 for crypto. Corporate actions stored; raw and adjusted series both available. Contract specs: tick, lot, multiplier, expiry, settlement currency, shortability, leverage. FX for EUR conversion: the ECB daily reference rate (`ecb_daily`) is the accounting rate; intraday rates of EUR FX pairs (Yahoo `EURUSD=X`) update the risk-engine's as-of table live (`risk.toml [fx] intraday`, spec 42 §6). Unknown symbols are provisioned as `provisional` instruments with a deterministic UUIDv5 and an open quarantine row (D-050), classified by symbol syntax (`go/pkg/refdata.Classify`); auto-creation from a global identifier (OpenFIGI / CoinGecko lookups) is out of scope (external connections), so a provisioned instrument carries no global identifier until an operator adds one (`POST /api/refdata/identifiers`).

**Data model** (schema `ref`; bitemporal-lite: `valid_from/valid_to` + `recorded_at`, migration `20261111000010`):
| Table | Key fields |
|---|---|
| `instrument` | `id`, `asset_class`, `kind` (equity, etf, fx_pair, crypto_spot, perp, future, option, bond, prediction_contract, index, macro_series_proxy), `name`, `base_ccy`, `quote_ccy`, `issuer_id`, `status` (active, provisional, merged, …), validity |
| `identifier` | `instrument_id`, `scheme` (figi, isin, cik, lei, yahoo_symbol, kraken_symbol, …), `value`, validity |
| `venue` | `id`, `code` (subject token), `mic`, `name`, `timezone`, `kind`, `always_open` |
| `listing` | `instrument_id`, `venue_id`, `venue_symbol`, `source_id`, validity (ticker recycling) |
| `contract_spec` | tick/lot size, min notional, multiplier, expiry, settlement ccy, shortable, max leverage, `fee_schedule_ref`, validity |
| `calendar_session` | `venue_id`, `date`, `open`, `close`, `segments` (pre/regular/post/auction), `is_holiday`, `source_id` (venue-published rows win over rules) |
| `corporate_action` | kind, ex/record/pay date, ratio/amount, ccy, `source_id`, `announced_at`, `status` (announced, confirmed, cancelled), symbol change |
| `entity` | `id`, `lei`, `name`, `country`, `sector`, `parent_id` |
| `instrument_relation` | `same_economic_exposure`, `underlying`, `successor` |
| `fx_rate` | `ccy`, `ts`, `rate_to_eur`, `source_id` (`ecb_daily`; intraday feed ids written by the risk-engine) |
Plus `wh.adjusted_bar_1d` (materialised adjusted daily bars) and `wh.fundamental` (XBRL facts).

**RefData contract** (spec 12 §4) = free functions here: `asof::{resolve, resolve_source_symbol, resolve_identifier}` (none / one / ambiguous, never picked silently), `asof::instrument`, `asof::contract_specs`, `calendar::session_state_at`, `corpactions::adjustment_factor(_db)`, `asof::fx_to_eur` (with source). Go side: `go/pkg/refdata` resolver with a 1 h cache.

**Behaviour.**
- *Change propagation:* every reference update publishes `fa.v1.RefChanged` on `ref.changed.<entity>` (control-api overrides, quarantine resolution, provisioning, the daily calendar refresh when rows changed); the ingestor's resolver cache flushes, the risk-engine reloads its `RefTable`, control-api's Helix mirror re-syncs (`catalog.changed.<id>` for source status/config).
- *Adjusted series:* raw in VictoriaMetrics, never modified; `corpactions::adjust_bars` adjusts on read for any range (point-in-time with `list_as_of(known_at)`); daily bars of instruments with splits or dividends are also materialised (`adjusted`).
- *Calendars:* built-in rules materialised into `ref.calendar_session` daily for the next 400 days (idempotent); venue-published rows (`source_id`) are never overwritten; session state is consulted by risk (no orders into closed sessions), backtests and timescale alignment. Seeding from venue publications or broker APIs is out of scope (external).
- *Helix mirror* (control-api `ref_mirror.rs` via `mirror_rows` and `fa_hub::reference`): nodes `Instrument`, `Entity`, `Venue`, `Sector`; edges `IssuedBy`, `ListedOn`, `SameExposure`, `InSector` (entity → sector node per distinct `ref.entity.sector`); full sync at start and every 6 h, incremental on `ref.changed.*`.

**Tunables.**
| Name | Value | Where |
|---|---|---|
| `resolver.auto_create` | `provisional` (D-050) | `go/pkg/refdata` |
| `resolver.cache_ttl` | 1 h | `refdata.CacheTTL` |
| `fx.accounting_source` | `ecb_daily` | `risk.toml [fx] source` |
| `fx.live_source` | intraday EUR-pair trades over `ecb_daily` | `risk.toml [fx] intraday` (spec 42) |
| `calendar.refresh` | daily, 400-day horizon | control-api `FA_REFDATA_MAINT_SECS` (`0` disables), `calendar::REFRESH_HORIZON_DAYS` |
| `adjustment.materialize_daily` | true for instruments with actions | control-api `refdata_maint.rs` |

**Failure modes.**
| Failure | Response |
|---|---|
| Ambiguous mapping | `ambiguous` answer, quarantine, manual resolution; never picked silently |
| Missing calendar | venue treated as closed for live orders (`missing_calendar`) |
| Corporate action announced late | new row with `recorded_at`; earlier as-of reads do not see it; the materialised series is recomputed (digest changes) |
| FX rate missing / stale | risk uses the last rate flagged stale (`[fx] max_age_hours`, optional halt / per-currency block); a fill or fee without a rate is refused for booking (spec 42). There is no separate EOD-close step: refusing to book without a rate is the accounting block |
| Ticker recycled | `listing` validity |

**Acceptance (met):** every canonical market event references an instrument UUID (provisioned if new); as-of reads honour `recorded_at ≤ known_at` and `valid_from ≤ at < valid_to`; a split through the adjusted series is continuous and the raw series unchanged (also materialised); any non-EUR amount converts to EUR at any time with a recorded source; session state is 24/7 for crypto and honours exchange holidays. Tests: `tests/refdata_asof_live.rs`, unit tests per module.

**Out of scope (external connections excluded, 2026-11-11):** OpenFIGI / CoinGecko lookups on a miss, a corporate-action feed (actions are entered via `POST /api/refdata/corporate-actions`), calendar seeding from venue publications or broker APIs.

**Open questions, decided (provisional, 2026-11-11):**
1. *Authoritative source per field:* operator override (row with `source_id` or an API entry) > venue publication > broker API > FIGI > symbol-syntax inference; for contract specs venue/broker win over FIGI.
2. *Adjustment policy:* on read for every range and point-in-time use; materialised daily for instruments with actions; models default to split-adjusted (price-only), total return available in both paths.
3. *Derivatives in v1:* no option chains and no continuous futures (deferred); futures exist only as Yahoo `=F` instruments of kind `future`.
4. *Prediction-market and on-chain identity:* `ref.identifier` schemes `kalshi_ticker`, `polymarket_condition_id` and `contract_address` (`<chain>:<address>`), added when such a connector exists (keyed, out of scope today).
5. *Fee schedules:* live with execution and accounting (broker-gateway cost model, spec 40/42); `contract_spec.fee_schedule_ref` points at them, nothing is versioned here.
