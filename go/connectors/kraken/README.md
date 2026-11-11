# kraken

Kraken public market data (keyless REST): 1m bars, trades and L1 quotes.

**Status:** implemented; endpoints verified with curl 2026-11-11 (OHLC, Trades, Ticker 200 without a key). Execution is out of scope (broker-gateway, spec 40).

## Behaviour
- **Poll** (every minute), per `config.symbols` (Kraken pair names, default `XBTEUR`; timeframe `config.timeframe`, default `1m`, also `5m 15m 1h 4h 1d`):
  - `GET /0/public/OHLC?pair=<p>&interval=<min>` → `Bar` on `md.bar.<tf>.kraken.<inst>`, stamped at bar close (open + tf), with vwap and trade count. The still-forming last row is never emitted; live polls skip bars already emitted (watermark persisted as `ingest.backfill_cursor` row, instrument + scope `<tf>`, cursor = bar close).
  - `GET /0/public/Trades?pair=<p>&since=<ns>` → `Trade` on `md.trade.kraken.<inst>` (aggressor from `b`/`s`, `venue_trade_id` = Kraken `trade_id`, event_time = trade time parsed from the decimal text without float rounding). `since` follows the response's `last`; the first poll without a cursor asks for one poll interval of history. De-dup by trade id; newest emitted trade time persisted under scope `trades` and used as `since` after a restart.
  - `GET /0/public/Ticker?pair=<p>` → `Quote` on `md.quote.kraken.<inst>` (best bid/ask, lot volumes). The ticker has no timestamp, so event_time = ingest_time.
  - `publish_time = ingest_time` for all events (no availability time, spec 10 §6.2).
- Kraken's result keys differ from request pairs (`XBTEUR` → `XXBTZEUR`, `ETHEUR` → `XETHZEUR`); the configured symbol travels in `TransportMeta.symbol` and is what the resolver sees.
- **Backfill:** one `OHLC ... &since=<from-1>` request; rows with open time in `[From, To)` and already closed are emitted (`backfill=1`). **Limit:** Kraken returns only the latest 720 intervals (12 h of 1m, 30 d of 1h, ~2 y of 1d); older ranges come back empty. Deeper history would need the Trades endpoint paged by `since` (not implemented).
- **Errors:** a non-empty `error` array (e.g. `EQuery:Unknown asset pair`, answered with HTTP 200) is returned as a normalize error.
- **Resolution:** `Resolver.Resolve("kraken", pair)`; misses are reported as `Unresolved` (with Postgres they are provisioned, `go/pkg/refdata`).
- Rate limit from the catalog (`0.5/s, burst 2`; 3 requests per pair per minute; public limit ≈ 1 req/s). Catalog symbols: `XBTEUR`, `ETHEUR`, `SOLEUR`.
- **Knowledge delay:** per-minute polling behind the limiter makes trades reach the bus late (measured 2026-11-11 on the MD stream: p50 43 s, p99 87 s, max 117 s); the catalog's `backfill.knowledge_delay_secs: 120` is what `fa_backtest::knowledge` shifts Kraken series by in event-time replays (D-079).
- The paper broker-gateway prices Kraken instruments as crypto with Kraken's 40 bp taker fee (`go/services/broker-gateway/costs.yaml`).

## Tests
`kraken_test.go` on `testdata/` (live responses trimmed: 5 OHLC rows, 5 trades, one ticker, the unknown-pair error): poll → 4 bars + 5 trades + 1 quote, payload values, trade `since` cursor, de-dup on re-normalize, restart from persisted cursors, deterministic payloads, unresolved pair, error payload, backfill range filtering and unsupported timeframe.

## Governing specs
- [Ingestion reference (ex spec 10)](../../services/ingestor/README.md#ingestion-reference), [Source catalog reference (ex spec 11)](../../pkg/catalog/README.md#source-catalog-reference), [21-bus-and-contracts](../../../specs/21-bus-and-contracts.md)
