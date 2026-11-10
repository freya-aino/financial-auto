# hackernews

First-wave connector: Hacker News (text).

**Status:** implemented, verified live 2026-11-10 (Firebase `topstories.json` 500 ids, `item/8863.json`, Algolia `search_by_date` all 200, no auth).

## Behaviour
- **Poll** (every minute): `newstories.json` + `topstories.json` (newest first), then `item/<id>.json` for ids not fetched before (≤ 30 per poll, one request each, catalog rate limit 5/s).
- **Normalize:** stories/jobs → `NewsItem` on `news.item.hackernews` (`title`, cleaned `body` for Ask HN text, `url` or the HN item link, `outlet news.ycombinator.com`, `entities_raw ["author:<by>"]`, `dedup_key hn:<id>`); comments, dead and deleted items dropped. `event_time = publish_time` = item time.
- **De-dup:** in-memory id sets (fetched / published) plus a persisted cursor (`ingest.backfill_cursor`, source `hackernews`, instrument null, timescale `items`) = newest story time published; after a restart stories at or before the cursor are skipped.
- **Backfill:** Algolia `search_by_date?tags=story&numericFilters=created_at_i>=from,created_at_i<to`, paged (100/page); no live de-dup applies. Triggered by `jobs.ingest.backfill` jobs (`POST /api/sources/<id>/backfill`, D-054); the ingestor drives `Backfill` per instrument with a persisted range cursor.

## Tests
`hackernews_test.go` on `testdata/{topstories,item_8863,algolia_search_by_date}.json`: poll + fetch bounding, de-dup across polls and restarts, cursor, field mapping, HTML cleaning, Algolia backfill.

## Governing specs
- [10-ingestion](../../../specs/10-ingestion.md), [11-source-catalog](../../../specs/11-source-catalog.md)
