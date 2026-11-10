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
- Rust: `fa-warehouse::refdata` —
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
- NATS: `ref.changed.<entity>` events on any reference update (cache invalidation, Helix sync).
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

Helix mirror: nodes `Instrument`, `Entity`, `Venue`; edges `ISSUED_BY`, `LISTED_ON`, `SAME_ECONOMIC_EXPOSURE` (e.g., BTC spot ↔ BTC perp), `IN_SECTOR`.

## 6. Behaviour & algorithms
- **Resolution:** connector symbol + venue + time → `listing` → instrument. On miss: OpenFIGI lookup (equities), CoinGecko id (crypto), manual queue otherwise. **(proposal)** auto-create instrument only when ≥1 global identifier (FIGI/ISIN/contract address) matches; else quarantine.
- **Adjusted series:** stored raw in VM; adjusted series computed on read (factor from `corporate_action`) **(proposal)**, with option to materialize adjusted daily bars.
- **Calendars:** **(proposal)** seed from venue publications and broker APIs; holidays refreshed yearly; session state is consulted by risk (no orders into closed sessions), backtests and model timescale alignment.
- **FX:** ECB daily reference rate as official EUR conversion for accounting (42); intraday FX mid from FX feeds for live risk; source of each conversion recorded.
- **Change propagation:** update → `ref.changed.*` → caches invalidated, Helix edges synced.

**Implemented today (D-050):** `go/pkg/refdata` resolves `(source, symbol)` through `ref.listing`/`ref.identifier`. Unknown pairs are **provisioned** as `status='provisional'` instruments (`uuid5(namespace, source\0symbol)`) with venue, listing and identifier rows plus an open `ingest.quarantine` row for manual review. This relaxes the "global identifier only" rule above so keyless sources can flow before FIGI/ISIN enrichment exists; `resolver.auto_create` is effectively `provisional`.

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `resolver.auto_create` | `global_id_only` | see §6 |
| `fx.accounting_source` | `ecb_daily` | |
| `fx.live_source` | `oanda_mid` | |
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
