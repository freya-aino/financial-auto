# oanda

First-wave connector: OANDA v20 (pricing stream → quotes, candles → bars).

**Status:** implemented, **unverified against the live API (no key)**. Written against the public v20 reference; fixtures in `testdata/` are the documented example payloads.

## Behaviour
- **Secrets:** `api_key` and `account_id` from `FA_OANDA_API_KEY` / `FA_OANDA_ACCOUNT_ID` or `[sources.oanda]` in `config/secrets.toml` (D-020); `Init` fails with `ErrNoAPIKey` / `ErrNoAccount`. `config.extra.env` = `practice` (default, `*-fxpractice.oanda.com`) or `live`.
- **Stream:** `GET /v3/accounts/<id>/pricing/stream?instruments=<config.symbols>`; each `PRICE` line → `Quote` (best bid/ask price, quoted liquidity as size) on `md.quote.oanda.<inst>`; `HEARTBEAT` lines dropped; reconnects after 5 s unless the server answers 401/403.
- **Backfill:** `GET /v3/instruments/<i>/candles?granularity=<M1|M5|M15|H1|D>&price=M&from=&to=` in ≤ 5000-candle windows; complete candles → `Bar` (`event_time` = close, `trade_count` = tick volume) on `md.bar.<tf>.oanda.<inst>`. Triggered by `jobs.ingest.backfill` jobs (`POST /api/sources/<id>/backfill`, D-054); the ingestor drives `Backfill` per instrument with a persisted range cursor.

## Tests
`oanda_test.go`: secrets precedence and env hosts, stream parsing + reconnect/401 stop, quote mapping, unresolved instrument, chunked candle backfill (incomplete candle skipped), API error surfacing.

## Governing specs
- [10-ingestion](../../../specs/10-ingestion.md), [11-source-catalog](../../../specs/11-source-catalog.md)
