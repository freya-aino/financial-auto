# 11 — Source catalog

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium |
| **Owners (code)** | `config/sources.yaml`, `go/pkg/catalog`, `migrations/` (`catalog` schema), `go/connectors/*` |
| **Depends on** | 10-ingestion, 12-reference-data, 40-execution-and-risk |

## 1. Purpose
Single inventory of every external API the platform may read from or trade through, with enough metadata to schedule connectors, respect rate limits, route orders and track status. The catalog is data, not code: adding a source starts here.

## 2. Scope
**In scope:** catalog schema, grouping, first-wave selection, status lifecycle, per-source caveats, seeding into Postgres.
**Out of scope:** connector implementation details (per-connector specs later), execution adapter behaviour (40).

## 3. Decisions
- Catalog lives in **`config/sources.yaml`** (seed, versioned in git) and is loaded into **Postgres** (`catalog.source`), which is authoritative at runtime (UI edits, status changes).
- **First wave:** Yahoo Finance, Lemon.markets, OANDA, Hacker News, Google News (`status: stub`). All others `planned`.
- Every API the user listed is in the catalog. "yfinance" and "Yahoo Finance" are one entry (`yahoo`). "Alpaca" (trading) and "Alpaca Market Data" are separate entries sharing credentials.
- ToS/licensing is a **free-text field only**; not enforced (deferred).
- Media sources (YouTube, earnings webcasts, central-bank pressers) are catalog entries like any other.

## 4. Interfaces
- `go/pkg/catalog`: `Load(path) (*Catalog, error)`, `Get(id)`, `ByCapability(cap)`, `ByCategory(cat)`; Postgres-backed `Store` with `Upsert`, `SetStatus`.
- Control API: `GET /api/sources`, `PATCH /api/sources/{id}` (status, enabled, rate limits).
- NATS: `catalog.changed.<id>`.

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
| `rate_limits` | list | **(proposal)** `{endpoint, requests, per}`; to be filled per source |
| `status` | enum | planned → stub → live → disabled |
| `tos_notes` | text | |
| `notes` | text | platform caveats |

Secrets are **never** in the catalog; they're referenced by `id` in git-ignored `config/secrets.toml` (ADR 0006).

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

### 6.3 Generic adapters **(proposal)**
- `ccxt` adapter covers long-tail crypto venues before dedicated connectors.
- SDMX adapter shared by ECB, Eurostat, IMF, OECD.
- RSS adapter shared by Google News and outlet feeds.

### 6.4 Prioritization **(proposal)** after first wave
1. EUR-relevant execution venues (Bitvavo, Kraken, IBKR) for paper trading breadth.
2. Macro with vintages (FRED/ALFRED, ECB) for point-in-time macro.
3. GDELT + Marketaux for news density.
4. Media pipeline (central-bank pressers first: scheduled, high signal).

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `catalog.reload_on_change` | true | |
| `catalog.auto_disable_after_failures_h` | 24 | persistent failure → disabled |

## 8. Failure modes
| Failure | Response |
|---|---|
| YAML and Postgres diverge | YAML seeds only new ids; Postgres wins for existing; diff shown in UI |
| Source deprecated / API changed | status → disabled, note recorded |
| Missing credentials | connector refuses to start, status shown in ops view |

## 9. Acceptance criteria
- `config/sources.yaml` contains every API from the user's list plus first-wave and media sources (checked by a test that compares against a fixed id list).
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
