# lemon

First-wave connector: lemon.markets market data (quotes, trades, OHLC backfill). Execution stays in broker-gateway.

**Status:** implemented, **unverified against the live API (no key)**. Written against the public API reference; fixtures in `testdata/` are the documented example payloads.

## Behaviour
- **Secrets:** `api_key` from `FA_LEMON_API_KEY` or `[sources.lemon]` in `config/secrets.toml` (D-020); `Init` fails with `ErrNoAPIKey` without it. Sent as `Authorization: Bearer`.
- **Config:** `config.symbols` = ISINs (required), `config.extra.mic` (default `XMUN`).
- **Poll** (every minute): `GET https://data.lemon.markets/v1/quotes/latest?isin=…&mic=…` → `Quote`; `…/trades/latest` → `Trade`; batches of 10 ISINs. Rows whose time did not advance since the last poll are not re-emitted.
- **Backfill:** `GET /v1/ohlc/{m1,h1,d1}?isin=&from=&to=&limit=100`, following `next` links (re-based onto the configured host) → `Bar` with `vwap = pbv / v`, `event_time` = bar close.
- Venue token `lemon`; subjects `md.{quote,trade}.lemon.<inst>`, `md.bar.<tf>.lemon.<inst>`.

## Tests
`lemon_test.go`: key handling (map and env), auth header, URL shape, quotes/trades mapping, unresolved ISIN, de-dup, paged OHLC backfill, API error surfacing.

## Governing specs
- [10-ingestion](../../../specs/10-ingestion.md), [11-source-catalog](../../../specs/11-source-catalog.md)
