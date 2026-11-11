# 12 — Reference data

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium |
| **Owners (code)** | `crates/fa-warehouse`, `crates/fa-core`, `migrations/`, `go/pkg/catalog`, reference connectors (OpenFIGI, GLEIF, exchange calendars) |
| **Depends on** | 10-ingestion, 11-source-catalog, 20-storage |

## 1. Purpose
Give every tradeable or observable thing a single stable identity and the static facts needed to trade and model it correctly: identifiers, venues, contract specifications, trading calendars, corporate actions, and FX rates for EUR conversion. Without this, ~70 sources cannot be joined and P&L cannot be computed.

## 2. Scope
**In scope:** instrument master, identifier mappings (OpenFIGI, ISIN, GLEIF-LEI, CIK, venue symbols), venues, asset classes, contract specs, trading calendars and sessions, corporate actions and adjustment factors, FX reference rates, issuer/entity master.
**Out of scope:** fundamentals and filings content (stored in warehouse but specified with ingestion 10/20), prices (VictoriaMetrics), latent representations (Helix, 30+).

## 3. Decisions
- **Internal instrument UUID** is the only identity used downstream.
- Mappings to **OpenFIGI / ISIN / GLEIF-LEI** and **per-venue symbols** stored in **Postgres** (authoritative) and **mirrored as Helix edges** (graph traversal: instrument↔issuer↔venue↔sector).
- **Trading calendars/sessions** per venue, including **24/7 crypto**.
- **Corporate actions** (splits, dividends, …) stored; both **raw and adjusted** series are available.
- **Contract specs**: tick size, lot size, multiplier, expiry, settlement currency, shortability/leverage limits.
- **FX rates** for **EUR** base-currency conversion (P&L, risk, fitness).
- Postgres is the warehouse for all static records.

## 4. Interfaces
- Rust: the `RefData` contract below (implemented as free functions over a `PgPool` in `fa-warehouse::{asof, calendar, corpactions}`, see §6; `refdata` holds search and quarantine review) —
  ```rust
  trait RefData {
      fn resolve(&self, venue: VenueId, symbol: &str, at: Timestamp) -> Option<InstrumentId>;
      fn instrument(&self, id: InstrumentId) -> Option<Instrument>;
      fn contract_spec(&self, id: InstrumentId, at: Timestamp) -> Option<ContractSpec>;
      fn session(&self, venue: VenueId, at: Timestamp) -> SessionState; // open/closed/auction/halt
      fn adjustment_factor(&self, id: InstrumentId, from: Timestamp, to: Timestamp) -> f64;
      fn fx_to_eur(&self, ccy: Currency, at: Timestamp) -> Option<Decimal>;
  }
  ```
- Go: `pkg/catalog` + `SymbolResolver` (10) backed by Postgres with in-memory cache.
- NATS: `ref.changed.<entity>` (`fa.v1.RefChanged`) on any reference update (cache invalidation; Helix sync not yet implemented), `catalog.changed.<source>` on catalog enable/disable.
- Control API: CRUD for manual overrides/mappings (51 ops view).

## 5. Data model (Postgres, schema `ref`)
| Table | Key fields |
|---|---|
| `instrument` | `id uuid`, `asset_class`, `kind` (equity, etf, fx_pair, crypto_spot, perp, future, option, bond, prediction_contract, index, macro_series_proxy), `name`, `base_ccy`, `quote_ccy`, `issuer_id`, `status`, `valid_from/valid_to` |
| `identifier` | `instrument_id`, `scheme` (figi, composite_figi, share_class_figi, isin, cusip?, cik, lei, coingecko_id, contract_address…), `value`, `valid_from/to` |
| `venue` | `id`, `mic` (if any), `name`, `timezone`, `kind` (exchange, broker, dex, otc) |
| `listing` | `instrument_id`, `venue_id`, `venue_symbol`, `valid_from/to`, `source_id` |
| `contract_spec` | `instrument_id`, `venue_id`, `tick_size`, `lot_size`, `min_notional`, `multiplier`, `expiry`, `settlement_ccy`, `shortable`, `max_leverage`, `fee_schedule_ref`, `valid_from/to` |
| `calendar_session` | `venue_id`, `date`, `open`, `close`, `segments jsonb` (pre/regular/post/auction), `is_holiday`; crypto venues flagged `always_open` |
| `corporate_action` | `instrument_id`, `kind`, `ex_date`, `record_date`, `pay_date`, `ratio`/`amount`, `ccy`, `source_id`, `announced_at` (point-in-time) |
| `entity` | `id`, `lei`, `name`, `country`, `sector`, `parent_id` |
| `fx_rate` | `ccy`, `ts`, `rate_to_eur`, `source_id` (ECB reference daily + intraday from FX feeds) |

All tables are **bitemporal-lite**: `valid_from/valid_to` (real-world validity) + `recorded_at` (when we learned it) so as-of queries in backtests are correct.

Helix mirror: nodes `Instrument`, `Entity`, `Venue`; edges `ISSUED_BY`, `LISTED_ON`, `SAME_ECONOMIC_EXPOSURE` (e.g., BTC spot ↔ BTC perp), `IN_SECTOR`. Declared in `helix/schema.hx` (`IssuedBy`, `ListedOn`, `SameExposure`; no `IN_SECTOR` yet) and written by control-api (`ref_mirror.rs`, below).

## 6. Behaviour & algorithms
- **Resolution:** connector symbol + venue + time → `listing` → instrument. On miss: OpenFIGI lookup (equities), CoinGecko id (crypto), manual queue otherwise. Auto-create only when ≥1 global identifier (FIGI/ISIN/contract address) matches, else quarantine — **superseded for now by provisional provisioning (D-050, below)**; OpenFIGI/CoinGecko lookups are not wired.
- **Adjusted series:** stored raw in VM; adjusted series computed on read (factor from `corporate_action`; implemented, `fa-warehouse::corpactions`), with option to materialize adjusted daily bars (not implemented).
- **Calendars:** built-in rules materialized into `ref.calendar_session` (implemented, below); seeding from venue publications and broker APIs is not (such rows would carry a `source_id` and win over the rules); holidays refreshed yearly; session state is consulted by risk (no orders into closed sessions), backtests and model timescale alignment.
- **FX:** ECB daily reference rate as official EUR conversion for accounting (42); intraday FX mid from FX feeds for live risk; source of each conversion recorded.
- **Change propagation:** update → `ref.changed.*` (or `catalog.changed.*`) → caches invalidated (the ingestor's resolver flushes, else 1 h TTL; risk-engine reloads its `RefTable`, debounced). Helix edges synced by control-api's reference mirror (below).

**Implemented today (D-050):** `go/pkg/refdata` resolves `(source, symbol)` through `ref.listing`/`ref.identifier`. Unknown pairs are **provisioned** as `status='provisional'` instruments (`uuid5(namespace, source\0symbol)`) with venue, listing and identifier rows plus an open `ingest.quarantine` row for manual review. This relaxes the "global identifier only" rule above so keyless sources can flow before FIGI/ISIN enrichment exists; `resolver.auto_create` is effectively `provisional`.

**Since 2026-11-11:** migration `20261111000010_refdata_asof` makes every `ref` table bitemporal-lite (`recorded_at` on venue/calendar/fx, validity on entity/venue), extends `corporate_action` (`status announced|confirmed|cancelled`, `cancelled_at`, `venue_id`, `old_symbol`/`new_symbol`, natural-key unique index), adds `ref.instrument_relation` (Postgres side of `SAME_ECONOMIC_EXPOSURE`), as-of indexes, `wh.fundamental` (vintaged) and seeds the reference venues `xnys`, `xnas`, `xetr`, `fx_otc`, `crypto`, `kraken`. Provisioning classifies by symbol syntax (`go/pkg/refdata.Classify`: Yahoo `XXXYYY=X` fx pair, `XXX-YYY` crypto, `^…` index, `…=F` future, exchange suffixes → listing currency; OANDA `EUR_USD`; Kraken `XBTEUR` / `XXBTZEUR` / `XBT/EUR` with `XBT` → `BTC`, `XDG` → `DOGE`; `BASE/QUOTE` on crypto venues), and the migration repaired the earlier rows (BTC-USD and EURUSD=X were provisioned as equities). The `RefData` interface of §4 is `fa-warehouse`: `asof::{resolve, resolve_source_symbol, resolve_identifier}` return `none | one | ambiguous` (never picked silently), `asof::instrument` / `contract_specs` / `fx_to_eur` filter `valid_from ≤ at < valid_to` and `recorded_at ≤ known_at`; `calendar` has built-in rules (US equities with holidays, early closes and DST; Xetra; FX 24/5; 24/7 crypto), `materialize` into `ref.calendar_session` (rows with a `source_id` are never overwritten) and `session_state_at` (table → rule → `missing_calendar` = closed); `corpactions` records actions idempotently, `adjustment_factor` / `adjust_bars` compute split-adjusted or total-return series on read (raw unchanged), `symbol_change` closes the old listing and symbol identifiers at the effective time and opens the new ones (ticker recycling via validity). control-api `refdata_routes` exposes the as-of reads (`GET /api/refdata/{instruments/{id},resolve,session,fx}`; the console adds `GET /api/refdata/{venues,calendar/{venue},corporate-actions}`, D-077) and the override CRUD (`POST /api/refdata/{corporate-actions,corporate-actions/{id}/cancel,symbol-change,identifiers,contract-specs,calendar/{venue}/materialize}`), each audited and announced as `fa.v1.RefChanged` on `ref.changed.<entity>`; quarantine resolution announces too. Calendars for `xnys`, `xnas`, `xetr` are materialized through 2027 on the live stack.

**Kraken classification (2026-11-11):** `kraken` has a provisioning hint (venue `kraken`, `crypto`/`crypto_spot`, scheme `kraken_symbol`) and `Classify` splits Kraken pair names, so new Kraken pairs get base/quote currencies (`XBTEUR` → BTC/EUR). The keyless macro/news/filing sources have no venue and provision no instruments. Migration `20261113000020_refdata_kraken_class` repairs the three Kraken rows provisioned as `unknown` (`XBTEUR`, `ETHEUR`, `SOLEUR`) in place with an `audit.log` entry each: `ref.instrument` has one row per id, and the class is an inference from the symbol rather than a real-world change, so `valid_from` and `recorded_at` are left alone (bumping `recorded_at` would hide the instrument from earlier `known_at` reads). The migration cannot publish `ref.changed.instrument`; risk-engine picks the new class up on its periodic RefTable reload (`refdata_refresh_secs`), on the next `ref.changed.*` notice or at restart.

**Helix mirror (2026-11-11):** control-api (`bins/control-api/src/ref_mirror.rs`, through `fa_warehouse::refdata::mirror_rows` and `fa_hub::reference`) writes the mirror: `Instrument{instrument_id = ref.instrument.id, asset_class, kind, name, status, base_ccy, quote_ccy}` for every instrument valid now (any status, `merged` included), `Entity{entity_id, lei, name, country, sector}`, `Venue{venue_id, code, mic, name, kind, timezone}`, and the instrument's out-edges `IssuedBy` (from `issuer_id`), `ListedOn{venue_symbol, source_id}` (current listings) and `SameExposure{relation = <from kind>-<to kind>}` (current `ref.instrument_relation` rows of kind `same_economic_exposure`, directional as stored; `underlying` / `successor` are not mirrored). Writes are idempotent upserts by business id with `mirrored_at`; each instrument's out-edges are made equal to Postgres (stale edges dropped). A **full sync** runs once Postgres and Helix are reachable and every `FA_REF_MIRROR_FULL_SECS` (default 6 h; `0` = start only); instruments no longer current stay as nodes with `status = inactive` and no out-edges. **Incremental** on `ref.changed.*` (debounced 2 s): `instrument` / `listing` notices with UUID ids re-mirror those instruments; merges, quarantine resolutions, venue / entity / relation and unparseable notices trigger a full sync; corporate actions, identifiers, contract specs and calendars are not mirrored. `catalog.changed.*` is not consumed (no mirrored property depends on a source's status); changes made without a notice (migrations) arrive with the next full sync. Metrics `fa_ref_mirror_syncs_total`, `fa_ref_mirror_failures_total`, `fa_ref_mirror_instruments`, `fa_ref_mirror_last_success_seconds`; the read side is the `instrumentReference` named query (`POST /api/graph/query`). `IN_SECTOR` is still not modelled (`Entity.sector` is a property).

**Round 3 (2026-11-11):** reference data now also backs the ledger's orphan handling and FX restatement (spec 42): an orphan fill's venue symbol is resolved through `ref.listing` as of the fill time (`none | one | ambiguous`, never picked silently), its currency from `ref.instrument.quote_ccy`, its rate and fee value from `ref.fx_rate` valid at the fill time (missing → left NULL, booking refused, never assumed EUR); `fa_ledger::restate` re-labels fills booked before an instrument's quote currency was known with the valid-time `ref.fx_rate` (the rate's `recorded_at` may be later; the restatement is an accounting correction, not an as-of replay). SEC XBRL facts in `wh.fundamental` are keyed by issuer CIK and link to an instrument only through a unique `ref.identifier` (`scheme = 'cik'`); no instrument carries a CIK yet (operator entry via `POST /api/refdata/identifiers`).

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `resolver.auto_create` | `global_id_only` | see §6; effectively `provisional` today (D-050), not configurable |
| `fx.accounting_source` | `ecb_daily` | risk-engine `risk.toml [fx] source` |
| `fx.live_source` | `oanda_mid` | not implemented: live risk also converts with `ecb_daily` |
| `calendar.refresh` | monthly | |
| `adjustment.materialize_daily` | false | |

## 8. Failure modes
| Failure | Response |
|---|---|
| Ambiguous mapping (one symbol → multiple instruments) | quarantine, manual resolution in UI; never pick silently |
| Missing calendar for venue | treat venue as closed for live orders; flag in ops |
| Corporate action announced late | new record with `recorded_at`; backtests as-of earlier time don't see it |
| FX rate missing | risk uses last rate with staleness flag; accounting blocks EOD close until present |
| Identifier reused (ticker recycled) | handled by `valid_from/to` on `listing` |

## 9. Acceptance criteria
- Every canonical market event references an existing instrument UUID.
- As-of query at time *t* returns only reference facts with `recorded_at ≤ t` and `valid_from ≤ t < valid_to`.
- A split replayed through adjusted series yields continuous prices; raw series unchanged.
- Any non-EUR amount can be converted to EUR for any historical timestamp with a recorded source.
- Session state API correctly reports 24/7 for crypto venues and holidays for exchanges.

**Status (2026-11-11):** every criterion has a test: listing/instrument as-of with `recorded_at` (`tests/refdata_asof_live.rs`), split continuity and raw-unchanged (`corpactions` unit + live), EUR conversion with source (`asof::fx_to_eur`, ECB daily), crypto 24/7 and exchange holidays (`calendar` unit + live). Still provisional: provisioned instruments carry no global identifier until an operator adds one (OpenFIGI/GLEIF lookups need keys or are rate-limited, not wired), and no connector feeds corporate actions yet (manual entry via the API).

## 10. Open questions
1. Authoritative source per field (FIGI vs exchange vs broker for contract specs) — blocks resolver precedence.
2. Adjustment policy: on-read vs materialized; total-return vs price-only adjustment for models — blocks 31 inputs.
3. Derivatives coverage (options chains, futures rolls/continuous contracts) in v1? — blocks schema completeness.
4. Prediction-market contracts (Kalshi/Polymarket) and on-chain tokens: identity scheme beyond contract address.
5. Fee schedules: where they live (here vs 42) and how they're versioned.

## 11. Deferred
- Options chains and Greeks.
- Index constituents history.
- Automated calendar scraping per venue.
