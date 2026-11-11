# catalog

Loads and validates `config/sources.yaml`, the inventory of every external API the platform may read from or trade through; the ingestor syncs it into Postgres (`catalog.source`).

**Status:** implemented — `Load`/`Parse` + `Validate` of `config/sources.yaml` (unique ids, enum values for status/auth/capabilities/transport/adapter, `rate_limit` > 0, symbols, non-empty queries, timeframe ∈ `1m 5m 15m 1h 1d`); optional `rate_limit {per_second, burst}` (`Source.Limit()`, default 5/s burst 10 when absent), connector `config:` block (`Source.Params()`: `symbols`, `queries`, `timeframe`, `extra`), `adapter:` (generic connector, `rss` | `sdmx`) and `backfill.knowledge_delay_secs` (`Source.Backfill`, D-079; also read by `fa_backtest::knowledge`); queries `ByID`, `ByCapability(data|execution)`, `ByCategory(cat)`, `All`; runtime edits `SetStatus` / `ApplyStatuses` (Postgres status wins over the YAML seed, D-054), `ApplyOverride(id, Override{RateLimit, Config})` (`ctl.ingest.configure`) and `Replace(next)` (`ctl.ingest.reload`: swaps in a re-read file, keeps runtime statuses, reports changed/added/removed ids; `RuntimeEqual` decides what needs a connector restart); `BackfillDepth(timescale)` gives the default history depth (seconds 4 w, minutes 1 y, hours 5 y, days 50 y). Safe for concurrent use. Tests: `TestRepoCatalogIsValid`, `TestRepoCatalogMatchesFixedIDList` (the 72 ids below, exact), capability/category, reload and override.

## Owns
Catalog model and loader.

## Governing specs
- [Source catalog reference](#source-catalog-reference) below (spec 11, retired 2026-11-11; `spec 11 §x` in code comments refers to it, see git history)
- [Ingestion reference](../../services/ingestor/README.md#ingestion-reference) (spec 10, retired)

## Source catalog reference
Lasting content of spec 11 (Source catalog), retired 2026-11-11.

**Decisions.** The catalog is data, not code: adding a source starts with an entry in `config/sources.yaml` (versioned seed), which the ingestor upserts into `catalog.source`; Postgres is authoritative at runtime for `status` and for runtime overrides of `rate_limit` / `config` (`catalog.source.rate_limit|config`, migration `20261114030000`), the YAML for the descriptive fields (re-upserted on every sync). Every API from the user's list is an entry ("yfinance" and "Yahoo Finance" are one entry, `yahoo`; "Alpaca" trading and "Alpaca Market Data" are separate entries sharing credentials). ToS/licensing is free text (`tos_notes`), not enforced. Media sources are entries like any other. Secrets are never in the catalog (git-ignored `config/secrets.toml`, D-020).

**Fields** (YAML entry / `catalog.source` row):
| Field | Notes |
|---|---|
| `id` | stable slug used in subjects and Garage keys |
| `name`, `category` | category ∈ broker, crypto_exchange, dex, crypto_aggregator_lib, prediction_market, market_data, fx_reference, crypto_data, onchain, macro, filings, fundamentals, identifiers, news, media |
| `capabilities` | data, execution (execution → a broker-gateway adapter, spec 40) |
| `asset_classes`, `transport`, `auth` | transport ∈ rest, websocket, fix, grpc, graphql, sse, socket, file, sdk, stream; auth ∈ none, api_key, oauth2, hmac, session, wallet, gateway |
| `rate_limit` | one token bucket per source `{per_second, burst}`, shared by live and backfill lanes; runtime override in Postgres |
| `config` | connector parameters `symbols`, `queries`, `timeframe`, `extra`; runtime override in Postgres |
| `adapter` | generic connector serving the entry when none is registered under its id: `rss` (`extra.feeds`), `sdmx` (`extra.base_url`, `queries`) |
| `backfill.knowledge_delay_secs` | measured delay until an event is on the bus (D-079) |
| `status` | planned → stub → live → disabled; Postgres wins; `status_note`, `status_changed_at` record why (auto-disable) |
| `tos_notes`, `notes` | free text |

**Status lifecycle.** `planned` (catalog only) → `stub` (connector exists, never run live) → `live` (meets the ingestion acceptance criteria) → `disabled` (by operator, `ctl.ingest.disable` / `POST /api/sources/{id}/disable`, or automatically after 24 h `down`, note recorded). Live today: `yahoo`, `hackernews`, `google_news`, `frankfurter`, `ecb`, `gdelt`, `sec_edgar`, `kraken`; stub: `lemon`, `oanda` (keyed); everything else planned.

**Interfaces.** Go: this package; `go/pkg/refdata.Store` (`SyncCatalog`, `Statuses`, `SetStatus`, `SetStatusNote`, `Overrides`, `SetOverride`). Control API: `GET /api/sources`, `GET /api/sources/{id}`, `PATCH /api/sources/{id}` (rate limit / config, `ctl.ingest.configure`), `POST /api/sources/{id}/enable|disable|backfill|reload`. NATS: `catalog.changed.<id>` (`fa.v1.RefChanged`, change `status` or `config`).

**Inventory (72 ids, fixed by `TestRepoCatalogMatchesFixedIDList`).** Brokers: interactive_brokers, saxo, alpaca, tastytrade, schwab, lemon, ig, oanda, ctrader, metatrader5. Crypto exchanges & DEX: bitvavo, kraken, coinbase, binance, bybit, okx, bitget, bitstamp, bitpanda, ccxt, hyperliquid, dydx. Prediction markets: kalshi, polymarket. Market data: fmp, eodhd, twelve_data, finnhub, alpha_vantage, tiingo, marketstack, alpaca_market_data, yahoo, dukascopy, nasdaq_data_link. FX reference: frankfurter, exchangerate_host, fixer, currencylayer. Crypto data / on-chain / DeFi: coingecko, coinmarketcap, coindesk_data, etherscan, blockscout, dune, the_graph, defillama. Macro: fred, ecb, eurostat, cbs_statline, us_treasury_fiscal, bls, world_bank, imf, oecd, dbnomics, eia, usda_quickstats, entsoe. Filings / identifiers: sec_edgar, simfin, openfigi, gleif. News: marketaux, newsapi, gdelt, hackernews, google_news. Media: youtube, earnings_webcasts, central_bank_pressers. Postgres additionally holds `ecb_daily` (risk-engine accounting FX, migration `20261110000060`), which has no YAML entry. Per-source facts of `planned` entries are first guesses, verified when a connector is built (verified facts are in each connector's README and the entry's `notes`).

**Generic adapters.** RSS/Atom (`go/connectors/rss`, shared parser also used by `google_news`) and SDMX-JSON (`go/connectors/ecb`, adapter `sdmx`, for ECB/OECD/IMF-style endpoints; Eurostat answers JSON-stat and would need its own parser). No catalog entry uses an adapter yet (that would enable a new source). A CCXT adapter is not built (see Q5).

**Tunables.**
| Name | Value | Where |
|---|---|---|
| `catalog.reload_on_change` | poll every 30 s | ingestor `-catalog-watch` (`0` disables) |
| `catalog.auto_disable_after_failures_h` | 24 h `down` | ingestor `-auto-disable-after` |

**Failure modes.**
| Failure | Response |
|---|---|
| YAML and Postgres diverge | YAML seeds new ids and descriptive fields; Postgres wins for status and overrides; `GET /api/sources` shows `status` (effective) and `yaml_status` |
| Source deprecated / API changed | `down` → halt → auto-disabled after 24 h with `status_note`, or disabled by the operator |
| Missing credentials | connector `Init` fails (`ErrNoAPIKey`), the source does not start |

**Acceptance (met):** every API of the user's list plus first-wave and media sources is in the YAML (fixed-id test); the catalog loads in Go and is served at `GET /api/sources`; no secrets in the catalog (credentials live in the git-ignored `config/secrets.toml`, D-020).

**Open questions, decided (provisional, 2026-11-11):**
1. *Per-source facts:* verified when a connector is built (done for the ten built ones); verifying the 62 planned entries needs external calls and is out of scope.
2. *Accessible execution venues:* out of scope (live brokers); paper trading only.
3. *Paid tiers:* none; only keyless or free endpoints, limits set from measured free-tier behaviour.
4. *MT5 bridge:* dropped in favour of cTrader/OANDA for FX; the entry stays `planned`.
5. *CCXT-Go as generic crypto adapter:* no — the Go port's maturity is unverified and it would add many new external venues; long-tail crypto follows the Kraken pattern (dedicated keyless connector) when needed.
