# yahoo

First-wave connector: Yahoo Finance chart endpoint (1m bars live, backfill by window).

**Status:** implemented, verified live 2026-11-10 (`query2.finance.yahoo.com` answered 200 to the ingestor's own user agent; `query1` answered 429 to a browser-like one). Unofficial endpoint, research use only.

## Behaviour
- **Poll** (every minute): `GET /v8/finance/chart/<symbol>?interval=<tf>&range=1d` per `config.symbols` (default `SPY`), timeframe `config.timeframe` (default `1m`; `1m 5m 15m 1h 1d`).
- **Normalize:** every completed bar → `Bar` on `md.bar.<tf>.yahoo.<inst>` **and** a synthetic `Trade` (price = close, size = volume, `venue_trade_id = bar:<unix>`) on `md.trade.yahoo.<inst>`, because the latent pipeline, strategy features and evo-worker eval (D-044) consume `md.trade.>`. `event_time` = bar close (open + tf); `publish_time = ingest_time` (spec 10 §6.2). Live polls skip bars already emitted (per-symbol watermark, persisted as an `ingest.backfill_cursor` row per instrument/timeframe) and the still-forming bar; null-padded minutes are dropped.
- **Backfill:** `period1/period2` windows (≤ 7 d for 1m, ≤ 60 d for 5m/15m, ≤ 730 d for 1h), all bars emitted, `TransportMeta.backfill=1`. Triggered by `jobs.ingest.backfill` jobs (`POST /api/sources/<id>/backfill`, D-054); the ingestor drives `Backfill` per instrument with a persisted range cursor.
- **Resolution:** `Resolver.Resolve("yahoo", symbol)`; misses are reported as `Unresolved` (with Postgres the resolver provisions them, see `go/pkg/refdata`).
- Rate limit from the catalog (`1/s, burst 3`).

## Tests
`yahoo_test.go` on `testdata/chart_aapl_1m.json` (live response trimmed to 5 bars): poll → 4 completed bars × 2 envelopes, de-dup, cursor persistence/resume, unresolved symbol, API error, backfill chunking (15 d → 3 requests).

## Governing specs
- [10-ingestion](../../../specs/10-ingestion.md), [11-source-catalog](../../../specs/11-source-catalog.md), [20-storage](../../../specs/20-storage.md) §5.3
