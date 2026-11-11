# googlenews

First-wave connector: Google News RSS search (text), catalog id `google_news`.

**Status:** implemented, verified live 2026-11-10 (`news.google.com/rss/search?q=Apple+stock&hl=en-US&gl=US&ceid=US:en` → 200, 100 items). The feed's copyright notice restricts use to personal, non-commercial readers (catalog `tos_notes`).

## Behaviour
- **Poll** (every 5 minutes): one feed per `config.queries` entry (default `"stock market"`), catalog rate limit 0.5/s.
- **Normalize:** each `<item>` → `NewsItem` on `news.item.google_news` with `title` (outlet suffix stripped), `url` (Google redirect link, kept as-is), `outlet` from `<source>`, `language` from the channel, `entities_raw ["query:<q>"]`, `dedup_key` = `<guid>`. `event_time = publish_time` = `pubDate`.
- **De-dup:** in-memory guid set plus a persisted cursor (`ingest.backfill_cursor`, timescale `items`) = newest `pubDate` published; items at or before it are skipped after a restart. The ingestor's cross-source news de-dup (normalized-URL hash, spec 10 §6.5) sees the Google redirect link, so it rarely matches the same story from another source.
- No backfill: the feed carries no history.

## Tests
`googlenews_test.go` on `testdata/search_apple_stock.xml` (live feed trimmed to 3 items): URL building, parsing, de-dup, cursor, restart, malformed XML.

## Governing specs
- [10-ingestion](../../../specs/10-ingestion.md), [11-source-catalog](../../../specs/11-source-catalog.md)
