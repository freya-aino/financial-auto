# rss

Generic RSS 2.0 / Atom adapter (catalog `adapter: rss`) and the shared feed parser `rss.Parse` (also used by `google_news`).

**Status:** implemented and tested against fixtures (`testdata/feed.rss`, `testdata/feed.atom`); no catalog entry uses the adapter yet (outlet feeds would be new sources).

- `Parse` reads RSS 2.0 (`pubDate`, `dc:date`, `<source>`) and Atom (`published`/`updated`, `rel=alternate` links) into `Feed{Title, Language, Items[]}`; `Item.Key()` = guid, else link; `ParseDate` accepts RFC 822/1123 and RFC 3339.
- Adapter: `connector.NewFor(id, "rss")` serves catalog entry `id`; `config.extra.feeds` (comma/space separated URLs, required), `poll_interval` (Go duration ≥ 10 s, default 5 m), `outlet` and `language` fallbacks. Every item becomes a `NewsItem` on `news.item.<id>` (dedup key = guid/link; the pipeline also de-duplicates across sources by normalized URL). One persisted cursor per feed (`ingest.backfill_cursor`, scope `feed:<url>`) skips items already published after a restart. Poll only.

## Governing specs
- [Source catalog reference](../../pkg/catalog/README.md#source-catalog-reference) (generic adapters), [Ingestion reference](../../services/ingestor/README.md#ingestion-reference)
