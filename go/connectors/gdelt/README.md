# gdelt

GDELT DOC 2.0 article search (text), catalog id `gdelt`. No key.

**Status:** implemented; endpoint verified 2026-11-11 (`api.gdeltproject.org/api/v2/doc/doc?mode=artlist&format=json` → 200 with `articles[]`). GDELT throttles hard: more than one request per 5 s (per IP) answers `429` with a plain-text body, and the throttle can persist for minutes, so the catalog limits it to 1 request / 10 s (`rate_limit` 0.1/s, burst 1; D-067).

## Behaviour
- **Poll** (every 15 minutes, GDELT's update cadence): one `artlist` request per `config.queries` entry (default `"stock market" sourcelang:english`; the catalog adds `"Federal Reserve"`) with `timespan` (`config.extra.timespan`, default `1h`) and `maxrecords` (`config.extra.maxrecords`, default/max 250), newest first.
- **Normalize:** each article → `NewsItem` on `news.item.gdelt`: `title` (GDELT's spaced punctuation tidied), `url`, `outlet` = `domain`, `language` (GDELT name → ISO 639-1, else lower-cased name), `entities_raw ["query:<q>", "country:<sourcecountry>"]`, `dedup_key = "url:" + sha1(normalized URL)` (lower-cased scheme/host, `www.` dropped, fragment and `utm_*` removed, trailing `/` trimmed; spec 10 §6.5). `event_time = publish_time = seendate` (GDELT's first-seen time; it gives no publication time).
- **De-dup:** in-memory key set (the ingestor additionally de-duplicates news across sources by normalized URL, spec 10 §6.5) plus a persisted cursor (`ingest.backfill_cursor`, scope `items`) = newest `seendate` published; after a restart articles at or before it are skipped.
- **Backfill:** `startdatetime`/`enddatetime` windows (`config.extra.window`, default 24h, 250 records each) per query (or the job's instrument string as the query). The DOC API only covers about the last 3 months, so the range is clamped to `To − 90 days`; busy windows are truncated at 250 (no paging).
- **Errors:** any non-JSON body (rate-limit text, `Your query was too short or too long.` with status 200) is an error; `{}` is an empty result.

## Tests
`gdelt_test.go` on `testdata/artlist_stock_market.json` (live answer, 3 articles), `bad_query.txt`, `ratelimit.txt`, `empty.json`: poll URL, mapping, dedup, cursor/restart, deterministic payloads, error bodies, backfill windows and depth clamp, URL normalization.

## Governing specs
- [10-ingestion](../../../specs/10-ingestion.md), [11-source-catalog](../../../specs/11-source-catalog.md)
