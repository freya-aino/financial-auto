# connector

Connector interface (Stream / Poll / Backfill / Normalize), rate limiting, retries, secrets.

**Status:** implemented — interfaces per spec 10 §4.1, registry, token-bucket limiter, plus:
- `Fetcher`: rate-limited HTTP with bounded retries (429/5xx/network; 3 retries, exponential backoff from 500 ms + jitter, spec 10 §6.3), fixed `UserAgent`, per-request `Prepare` hook for auth headers. The poll-level backoff and circuit breaker live in the ingestor's poll loop.
- `LoadSecrets` / `Secrets`: minimal reader for `config/secrets.toml` (`[garage]`, `[sources.<id>]`); `FA_<SOURCE>_<KEY>` environment variables take precedence (D-020). `SourceConfig.Secret(key)` for connectors.
- `CursorStore` (+ `MemCursors`): persisted progress per `(source, instrument?, scope)`; Postgres implementation in `go/pkg/refdata`.
- `BackfillCursorStore` (+ `MemBackfillCursors`): bounded backfill ranges keyed by `range_start`, used by the ingestor's `jobs.ingest.backfill` worker (D-054); Postgres implementation `refdata.BackfillCursors`.
- `MeteredLimiter`: wraps a `RateLimiter` and measures saturation (share of wall time spent blocked in `Wait` per window) for `ingest.health.<source>`; the ingestor wraps each source's live lane in one.
- `LaneLimiter` (`lanes.go`): one token bucket per source (catalog `rate_limit`) with a live lane (`Live()`) and a low-priority backfill lane (`Backfill()`): the backfill lane is capped at a share of the rate (`DefaultBackfillShare` 0.5), never takes a token while a live request waits and keeps one token of headroom when burst ≥ 2; both lanes pass the shared bucket, so the catalog limit holds for the sum. `SetLimit` applies a reloaded rate limit; `Counts` reports requests per lane.
- Registry: `Register(id, factory)` per connector, `RegisterAdapter(name, func(id) Connector)` for generic adapters (`rss`, `sdmx`), `NewFor(id, adapter)` / `HasConnector` resolve a catalog entry to its connector or adapter.
- `Seen`: bounded in-process de-dup set (the ingestor keeps one of 50 000 keys for cross-source news de-dup).
- `StableEventID` (`eventid.go`, D-067): deterministic ULID per canonical envelope — time part = event time (ingest time when absent), entropy = sha256 of source, instrument and the news dedup key or the deterministic payload encoding; ingest/publish times and the raw ref are excluded, so re-polls, backfill and archive replay of the same fact get the same id (sent as `Nats-Msg-Id`).
- `NormalizeURL` / `NewsDedupKey`: cross-source news key (spec 10 §6.5) `u:` + sha256 of the normalized URL (lowercase host without `www.`, no fragment, no `utm_*`/click-id params, sorted query, no trailing slash, http = https), fallback `t:` + hash of title, outlet and publish hour.
- `Deps.Cursors`, `TransportMeta` keys (`endpoint`, `symbol`, `tf`, `backfill`, `url`, `gap_fill` — set by the ingestor's gap filler so envelopes carry `gap_filled`).
- `StaticResolver` / `NopArchive` remain the no-infrastructure fallbacks (`NopArchive` is also used by replay so replayed payloads are never re-archived) (Postgres resolver: `go/pkg/refdata`, Garage writer: `go/pkg/archive`).

## Owns
Go interface every connector implements.

## Governing specs
- [Ingestion reference](../../services/ingestor/README.md#ingestion-reference) (spec 10, retired)
