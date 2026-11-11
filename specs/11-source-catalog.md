# 11 — Source catalog

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium |
| **Owners (code)** | `config/sources.yaml`, `go/pkg/catalog`, `go/pkg/refdata` (Postgres sync), `migrations/` (`catalog` schema), `go/connectors/*` |
| **Depends on** | 10-ingestion, 12-reference-data, 40-execution-and-risk |

## 1. Purpose
Single inventory of every external API the platform may read from or trade through, with enough metadata to schedule connectors, respect rate limits, route orders and track status. The catalog is data, not code: adding a source starts here.

## 2. Scope
**In scope:** catalog schema, grouping, first-wave selection, status lifecycle, per-source caveats, seeding into Postgres.
**Out of scope:** connector implementation details (per-connector specs later), execution adapter behaviour (40).

## 3. Decisions
- Catalog lives in **`config/sources.yaml`** (seed, versioned in git) and is loaded into **Postgres** (`catalog.source`), which is authoritative at runtime (UI edits, status changes).
- **First wave:** Yahoo Finance, Lemon.markets, OANDA, Hacker News, Google News (`status: stub`). All others `planned`. Current state: `yahoo`, `hackernews`, `google_news` are `live`; `lemon`, `oanda` stay `stub` (implemented, no keys, never run live); the keyless second wave `frankfurter`, `ecb`, `gdelt`, `sec_edgar`, `kraken` is `live` (§6.2).
- Every API the user listed is in the catalog. "yfinance" and "Yahoo Finance" are one entry (`yahoo`). "Alpaca" (trading) and "Alpaca Market Data" are separate entries sharing credentials.
- ToS/licensing is a **free-text field only**; not enforced (deferred).
- Media sources (YouTube, earnings webcasts, central-bank pressers) are catalog entries like any other.

## 4. Interfaces
- `go/pkg/catalog`: `Load(path) (*Catalog, error)`, `Get(id)`, `ByCapability(cap)`, `ByCategory(cat)`; Postgres-backed `Store` with `Upsert`, `SetStatus`.
- Control API: `GET /api/sources`, `PATCH /api/sources/{id}` (status, enabled, rate limits).
- NATS: `catalog.changed.<id>`.

**Implemented:** `go/pkg/catalog` has `Load`/`Parse`, `Validate`, `ByID`, `SetStatus`, `ApplyStatuses`, `BackfillDepth` (no capability/category queries); the Postgres side is `go/pkg/refdata.Store` (`SyncCatalog` upsert that keeps an existing row's status, `Statuses`, `SetStatus`). control-api serves `GET /api/sources` (catalog + DB status + live health overlay), `GET /api/sources/{id}` (health, recent gaps and unresolved symbols, cursors) and `POST /api/sources/{id}/enable|disable|backfill` (→ `ctl.ingest.*`, audited) instead of `PATCH`; rate limits and `config` are edited in the YAML only. The ingestor publishes `fa.v1.RefChanged` on `catalog.changed.<id>` after `enable|disable`.

## 5. Data model
YAML entry / `catalog.source` row:

| Field | Type | Notes |
|---|---|---|
| `id` | string | stable slug, used in subjects and Garage keys |
| `name` | string | |
| `category` | enum | broker, crypto_exchange, dex, crypto_aggregator_lib, prediction_market, market_data, fx_reference, crypto_data, onchain, macro, filings, fundamentals, identifiers, news, media |
| `capabilities` | [data \| execution] | execution → broker-gateway adapter needed (40) |
| `asset_classes` | list | coarse |
| `transport` | list | rest, websocket, fix, grpc, graphql, sse, socket, file, sdk, stream |
| `auth` | enum | none, api_key, oauth2, hmac, session, wallet, gateway |
| `rate_limits` | list | **(proposal)** `{endpoint, requests, per}`; to be filled per source. Implemented as one token bucket per source, `rate_limit: {per_second, burst}` (default 5/s, burst 10), shared by live polling and backfill |
| `config` | map | connector parameters (D-050): `symbols`, `queries`, `timeframe` (`1m 5m 15m 1h 1d`), `extra` (string map) |
| `backfill.knowledge_delay_secs` | int | measured delay until an event is on the bus (D-079; kraken 120, yahoo 100), read by `fa_backtest::knowledge` |
| `status` | enum | planned → stub → live → disabled |
| `tos_notes` | text | |
| `notes` | text | platform caveats |

`rate_limit`, `config` and `backfill` live only in the YAML; `catalog.source` stores the descriptive fields and the status. Secrets are **never** in the catalog; they're referenced by `id` in git-ignored `config/secrets.toml` (ADR 0006).

## 6. Behaviour & algorithms

### 6.1 Status lifecycle
`planned` (catalog only) → `stub` (connector package exists, may emit fixtures) → `live` (passes acceptance in 10 §9) → `disabled` (manually or after persistent failure).

### 6.2 Catalog (grouped)
All per-source facts below are **unverified first guesses** and must be confirmed when the connector is specced.

**Brokers / multi-asset**
| id | Name | Data/Exec | Asset classes | Transport (guess) | Caveats |
|---|---|---|---|---|---|
| interactive_brokers | Interactive Brokers | D+E | equity, etf, option, future, fx, bond, crypto | socket, rest | Needs local TWS/IB Gateway or Client Portal Gateway |
| saxo | Saxo OpenAPI | D+E | multi-asset, cfd | rest, ws | OAuth2; sim env |
| alpaca | Alpaca | E+D | equity, etf, option, crypto | rest, ws | Paper endpoint; US-centric |
| tastytrade | Tastytrade | E+D | equity, option, future, crypto | rest, ws | |
| schwab | Charles Schwab | E+D | equity, etf, option, future | rest, ws | OAuth refresh expiry |
| lemon | Lemon.markets | E+D | equity, etf | rest | **First wave**; EU |
| ig | IG | E+D | cfd, fx, index, commodity | rest, Lightstreamer | |
| oanda | OANDA | E+D | fx, cfd | rest, http-stream | **First wave**; practice acct |
| ctrader | cTrader Open API | E+D | fx, cfd | protobuf/TCP | |
| metatrader5 | MetaTrader 5 | E+D | fx, cfd | Python SDK | Windows terminal only → bridge needed |

**Crypto exchanges & DEX**
| id | Name | Data/Exec | Asset classes | Transport (guess) | Caveats |
|---|---|---|---|---|---|
| bitvavo | Bitvavo | D+E | crypto | rest, ws | EUR-quoted |
| kraken | Kraken | D+E | crypto, derivs | rest, ws | |
| coinbase | Coinbase Advanced Trade | D+E | crypto | rest, ws | JWT keys |
| binance | Binance | D+E | crypto, derivs | rest, ws | Regional restrictions |
| bybit | Bybit | D+E | crypto, derivs | rest, ws | Regional restrictions |
| okx | OKX | D+E | crypto, derivs | rest, ws | |
| bitget | Bitget | D+E | crypto, derivs | rest, ws | |
| bitstamp | Bitstamp | D+E | crypto | rest, ws | |
| bitpanda | Bitpanda | D(+E?) | crypto, equity, commodity | rest | Trading support unverified |
| ccxt | CCXT | D+E | crypto | library (Go port) | Generic adapter for long tail |
| hyperliquid | Hyperliquid | D+E | perps, spot | rest, ws | Wallet signing |
| dydx | dYdX | D+E | perps | rest, ws, grpc | v4 chain; regional restrictions |

**Prediction markets**
| id | Name | Data/Exec | Asset classes | Transport (guess) | Caveats |
|---|---|---|---|---|---|
| kalshi | Kalshi | D+E | event contracts | rest, ws | Jurisdiction eligibility |
| polymarket | Polymarket | D+E | event contracts | rest, ws | Wallet; jurisdiction eligibility |

**Market data vendors**
| id | Name | Data/Exec | Asset classes | Transport (guess) | Caveats |
|---|---|---|---|---|---|
| fmp | Financial Modeling Prep | D | multi + fundamentals | rest, ws | |
| eodhd | EODHD | D | multi + fundamentals | rest, ws | |
| twelve_data | Twelve Data | D | multi | rest, ws | |
| finnhub | Finnhub | D | multi + news | rest, ws | |
| alpha_vantage | Alpha Vantage | D | multi + macro | rest | Low free limits |
| tiingo | Tiingo | D | multi + news | rest, ws | |
| marketstack | Marketstack | D | equity, index | rest | |
| alpaca_market_data | Alpaca Market Data | D | equity, option, crypto, news | rest, ws | Shares alpaca creds |
| yahoo | Yahoo Finance (yfinance) | D | multi + fundamentals | rest (unofficial) | **First wave**; may break |
| dukascopy | Dukascopy | D | fx, cfd | bi5 files | Historical ticks |
| nasdaq_data_link | Nasdaq Data Link | D | multi, alternative | rest | Per-dataset licensing |

**FX reference**
| id | Name | Transport | Caveats |
|---|---|---|---|
| frankfurter | Frankfurter.app | rest | ECB mirror |
| exchangerate_host | exchangerate.host | rest | Auth/free tier changed; verify |
| fixer | Fixer | rest | |
| currencylayer | CurrencyLayer | rest | |

**Crypto data, on-chain, DeFi**
| id | Name | Transport | Caveats |
|---|---|---|---|
| coingecko | CoinGecko | rest | Also crypto identity source |
| coinmarketcap | CoinMarketCap | rest | |
| coindesk_data | CoinDesk Data | rest, ws | Ex-CryptoCompare; verify |
| etherscan | Etherscan | rest | |
| blockscout | Blockscout | rest, graphql | Multi-chain |
| dune | Dune API | rest | Credit-based |
| the_graph | The Graph | graphql | Query fees |
| defillama | DefiLlama | rest | |

**Macro / statistics**
fred (use ALFRED vintages), ecb (SDMX; EUR reference rates), eurostat, cbs_statline (OData), us_treasury_fiscal, bls, world_bank, imf, oecd, dbnomics, eia, usda_quickstats, entsoe (XML). All data-only, mostly REST, many without auth.

**Fundamentals / filings / identifiers**
sec_edgar (User-Agent required), simfin, openfigi, gleif.

**News**
marketaux, newsapi (dev-only free tier), gdelt, hackernews (**first wave**), google_news (**first wave**, RSS).

**Broadcast media**
youtube (Data API + capture; ToS), earnings_webcasts (IR sites; discovery via earnings calendars), central_bank_pressers (Fed/ECB/BoE…).

**Implemented (2026-11-10, D-050):** `yahoo` (chart endpoint, 1m bars + synthetic trades), `hackernews` (Firebase + Algolia backfill) and `google_news` (RSS) are `live`; `lemon` and `oanda` are implemented against the public docs but stay `stub` (no keys).

**Implemented today (2026-11-11):** `frankfurter`, `ecb`, `gdelt`, `sec_edgar` and `kraken` (public market data only) are `live` keyless connectors with a `rate_limit` and `config:` block in `config/sources.yaml` (frankfurter `extra.{base,currencies}`, ecb `queries` = SDMX `<FLOW>/<KEY>`, gdelt `queries`, sec_edgar `extra.{ciks,user_agent}`, kraken `symbols`). Verified facts: Frankfurter moved to `api.frankfurter.dev/v1` (`.app` answers 301); the ECB portal's firewall answers HTML blocks to bursts and to some user agents (limit 1 request / 10 s); GDELT enforces one request per 5 s per IP with multi-minute 429 penalties (limit 1 / 10 s); EDGAR's `data.sec.gov` needs a declared User-Agent (`financial-auto ingestor` accepted, the generic ingestor UA refused with 403; `www.sec.gov` feeds refuse it); Kraken public REST needs no key and returns at most 720 OHLC intervals. `sec_edgar` also feeds `wh.fundamental` from XBRL company facts (D-078). Measured knowledge delays (D-079): kraken 120 s, yahoo 100 s (`backfill.knowledge_delay_secs`). CoinGecko was not added (catalog entry still `api_key`). Postgres additionally holds `ecb_daily` (migration `20261110000060`), the risk-engine's accounting FX source (`ref.fx_rate`), which is not in the YAML and has no connector.

### 6.3 Generic adapters **(proposal)**
- `ccxt` adapter covers long-tail crypto venues before dedicated connectors.
- SDMX adapter shared by ECB, Eurostat, IMF, OECD.
- RSS adapter shared by Google News and outlet feeds.

### 6.4 Prioritization **(proposal)** after first wave
1. EUR-relevant execution venues (Bitvavo, Kraken, IBKR) for paper trading breadth. (Kraken public market data done; execution not.)
2. Macro with vintages (FRED/ALFRED, ECB) for point-in-time macro. (ECB done.)
3. GDELT + Marketaux for news density. (GDELT done.)
4. Media pipeline (central-bank pressers first: scheduled, high signal).

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `catalog.reload_on_change` | true | not implemented: the YAML is read at ingestor start |
| `catalog.auto_disable_after_failures_h` | 24 | persistent failure → disabled; not implemented (failures degrade/halt, status changes are manual) |

## 8. Failure modes
| Failure | Response |
|---|---|
| YAML and Postgres diverge | YAML seeds only new ids; Postgres wins for existing; diff shown in UI (implemented for the status: `GET /api/sources` returns `status` = effective and `yaml_status` = seed; descriptive fields are re-upserted from the YAML on every sync) |
| Source deprecated / API changed | status → disabled, note recorded |
| Missing credentials | connector refuses to start (`Init` fails, e.g. lemon/oanda `ErrNoAPIKey`), status shown in ops view |

## 9. Acceptance criteria
**Remaining (not fully implemented):**
- The id-list check is weaker than specified: `TestRepoCatalogIsValid` counts (≥ 70) and checks the five first-wave ids, it does not compare against the fixed list of §6.2. (Checked by hand 2026-11-11: all 70 ids named in §6.2 are in `config/sources.yaml`, 72 entries in all.)
- Runtime edits of `rate_limit` / `config` (§4 `PATCH … rate limits`) do not exist; they are YAML-only and need an ingestor restart.
- `catalog.reload_on_change` and `catalog.auto_disable_after_failures_h` (§7) are not implemented; a failing source degrades and halts, it is never auto-disabled, and "source deprecated → disabled, note recorded" (§8) is a manual status change.
- `ByCapability` / `ByCategory` (§4) do not exist (no consumer needs them yet).
- Generic adapters (§6.3: ccxt, shared SDMX, shared RSS) are not built; `ecb` and `google_news` carry their own SDMX / RSS code.
- 62 of 72 entries are `planned` with unverified per-source facts (§6.2, §10 Q1); open questions §10 Q1–Q5 are undecided.

- `config/sources.yaml` contains every API from the user's list plus first-wave and media sources (checked by a test that compares against a fixed id list). Implemented check: `TestRepoCatalogIsValid` loads and validates the file, requires ≥ 70 sources and the five first-wave ids with status `stub` or `live`.
- Catalog loads in Go and is visible at `GET /api/sources`.
- No secrets in the catalog or git history.

## 10. Open questions
1. Verify per-source facts (transport, auth, limits, data vs execution, EU availability) — blocks connector prioritization.
2. Which execution venues are actually accessible from the user's jurisdiction/accounts — blocks 40 routing.
3. Paid tiers: which subscriptions will be bought — blocks rate-limit config.
4. MT5 bridge: worth it, or drop in favour of cTrader/OANDA for FX?
5. Is CCXT-Go mature enough to be the generic crypto adapter?

## 11. Deferred
- Per-connector specs beyond first wave.
- ToS/licensing enforcement.
- YouTube channel list, earnings-call discovery, keyword-driven discovery.
