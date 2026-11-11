# 10 — Ingestion

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium-high |
| **Owners (code)** | `go/pkg/connector`, `go/pkg/catalog`, `go/pkg/bus`, `go/pkg/archive`, `go/pkg/refdata`, `go/pkg/tsdb`, `go/pkg/warehouse`, `go/connectors/*`, `go/services/ingestor` |
| **Depends on** | 11-source-catalog, 12-reference-data, 20-storage, 21-bus-and-contracts |

## 1. Purpose
Pull every reachable data source (market data, broker feeds, FX, macro, on-chain, filings, news, broadcast media) into the platform as **canonical, point-in-time-correct events** on the NATS bus, and archive the raw payloads so any period can be replayed exactly. Ingestion is the only place where source-specific formats exist; everything downstream sees canonical protobuf.

## 2. Scope
**In scope:** connector framework (Go), stream/poll/backfill modes, rate limiting and retry, normalization to canonical events, raw-payload archival to Garage, instrument symbol resolution via reference data, media capture (video/audio/images) into `MediaChunk`s, connector health reporting, historical backfill per timescale.
**Out of scope:** feature engineering and embeddings (31), ASR/frame embedding *models* (31; ingestion only cuts and stores media), order execution (40 — broker-gateway owns execution even for venues that also supply data), web-search-driven discovery (deferred).

## 3. Decisions
- Connectors are written in **Go** and run inside the `ingestor` service (one goroutine group per connector instance); scalable horizontally by sharding connector instances across ingestor replicas.
- Three connector modes: **Stream** (WebSocket/FIX/SSE), **Poll** (REST on schedule), **Backfill** (bounded historical range).
- **Canonical events** (protobuf, `fa.v1`): `Trade`, `Quote`, `Bar`, `BookSnapshot`, `NewsItem`, `MediaChunk`, `MacroObservation`, `OnchainEvent`, `FilingEvent`, `FundamentalFact` (D-078).
- Every event carries **`event_time`, `publish_time`, `ingest_time`** (see §6.2) — point-in-time correctness is non-negotiable.
- **Raw payloads archived to Garage** before/alongside normalization; replay (41) re-normalizes from raw.
- Instrument identity resolved to the **internal instrument UUID** (12) at ingestion; unresolved symbols are quarantined, not dropped.
- **Media sources v1:** YouTube live/VOD finance channels, earnings-call webcasts/audio, central-bank press conferences → video frames + audio segments (ASR downstream in Rust).
- **Backfill depth per timescale:** seconds ≈ weeks, minutes ≈ 1 year, hours ≈ 5 years, days = maximum available.
- Web-search / keyword-driven discovery: only a generic `DiscoveryHook` interface now; content deferred.
- First wave connectors: **Yahoo Finance, Lemon.markets, OANDA, Hacker News, Google News**.

## 4. Interfaces

### 4.1 Connector interface (Go, sketch)
```go
type Connector interface {
    ID() string                          // matches sources.yaml id
    Capabilities() Capabilities          // stream/poll/backfill, event kinds
    Init(ctx context.Context, cfg SourceConfig, deps Deps) error
}

type Streamer interface {
    Stream(ctx context.Context, subs []Subscription, out chan<- RawMessage) error
}
type Poller interface {
    Poll(ctx context.Context, req PollRequest) ([]RawMessage, error)
    Schedule() PollSchedule
}
type Backfiller interface {
    Backfill(ctx context.Context, req BackfillRequest, out chan<- RawMessage) error
}
type Normalizer interface {
    Normalize(raw RawMessage) ([]*fav1.Envelope, []Unresolved, error)
}

type Deps struct {
    Limiter  RateLimiter     // per-source token bucket from catalog
    Resolver SymbolResolver  // venue symbol -> instrument UUID (12)
    Archive  RawArchive      // Garage writer
    Clock    Clock           // monotonic + wall, injectable for tests
    Cursors  CursorStore     // persisted live progress (ingest.backfill_cursor); nil = start from now
}
```
`RawMessage` = `{source_id, received_at, transport_meta, bytes, content_type}`.

### 4.2 Outputs
- NATS subjects per 21 (`md.*`, `news.*`, `media.*`, `macro.*`, `onchain.*`, `filing.*`, `ingest.health.*`).
- Garage: `raw/<source_id>/<yyyy>/<mm>/<dd>/<hh>/<ulid>.<ext>` (+ zstd). Implemented: one gzip object per (source, UTC minute) `…/<ulid>.raw.gz` plus an index sidecar `<ulid>.idx.jsonl.gz` (gzip instead of zstd, D-050/D-067).
- Garage: `media/<source_id>/<media_id>/{video|audio|frames}/…`.
- Postgres: connector run log, backfill progress cursors, quarantine table; slow canonical records in `wh.macro_observation`, `wh.filing`, `wh.fundamental` (`go/pkg/warehouse`, idempotent per vintage / accession).
- VictoriaMetrics: market-data samples (`fa_trade_*`, `fa_quote_*`, `fa_bar_*`, `fa_book_*`, spec 20 §5.3) and ingestion metrics (heartbeat counters scraped from the ingestor's `/metrics`).

### 4.3 Control
- `jobs.ingest.backfill` work queue (21) for backfill requests from the control API / trainer.
- `ctl.ingest.{start|stop|status}` request/reply with arg `source` (D-036; `reload` pending).
- `DiscoveryHook` interface: `Discover(ctx, query) ([]SourceTarget, error)` — no implementation in v1.

**Implemented today (D-054):** `ctl.ingest.{start,stop,status,enable,disable,backfill}` with arg `source` (`status` replies JSON with health, cursors and limiter usage); `enable|disable` persist `catalog.source.status` (Postgres wins over the YAML seed after the first sync). Health: JSON frames on `ingest.health.<source>` every 5 s (`ok|degraded|down`, ages, error rate, limiter saturation); an unexpected `down` raises `risk.halt.source:<id>` and recovery publishes `risk.resume.source:<id>`; `ingest.gap.<source>` and `ingest.unresolved.<source>` as in §4.2. Backfill runs from the `JOBS_INGEST` work queue (`jobs.ingest.backfill`, cursor rows per range so redeliveries resume).

**Since 2026-11-11 (contracts, determinism, replay):** health, gap and unresolved frames are the `fa.v1` messages `IngestHealth` / `IngestGap` / `IngestUnresolved` (`proto/fa/v1/ingest.proto`); `-health-json` / `FA_INGEST_HEALTH_JSON=1` restores the D-054 JSON, and every consumer (risk-engine, control-api through `fa_bus::ingest::frame_json`) accepts both (a JSON payload starts with `{`, never the first byte of a proto3 encoding), so `/api/sources` serves the same JSON shape. Recovery publishes `fa.v1.Resume` (wire-compatible with `Halt`). Provisioning publishes `RefChanged` on `ref.changed.instrument`, `ctl.ingest.enable|disable` publish `catalog.changed.<id>`, and the ingestor drops its resolver cache on any `ref.changed.*` or `catalog.changed.*` (spec 12 §6; cache TTL 1 h = `resolver.cache_ttl`); a `catalog.changed.<id>` also re-reads `catalog.source` statuses into the in-memory catalog (it does not start or stop connectors). `ctl.ingest.replay {source, start, end, mode=verify|publish}` re-normalizes the Garage raw archive (§9): `verify` replies a sha256 digest over the deterministic encodings, `publish` re-emits on `replay.<subject>` (core NATS only, no stream captures it). Backfill jobs of venue-less sources (macro series, CIKs, queries) no longer resolve symbols into instruments; their cursor scope is `<timescale>|<series>`.

## 5. Data model

### 5.1 Envelope (all canonical events)
| Field | Meaning |
|---|---|
| `event_id` | ULID, unique per canonical event |
| `source_id` | catalog id |
| `instrument_id` | internal UUID (nullable for news/macro/media) |
| `event_time` | when the thing happened in the world (trade time, bar end, article timestamp) |
| `publish_time` | when the source made it available (API availability / article publication / macro release) |
| `ingest_time` | when our ingestor received it |
| `raw_ref` | Garage key of raw payload + byte offset |
| `schema_version` | proto version |
| `payload` | oneof the canonical event types |

### 5.2 Canonical event highlights (full definitions in `proto/fa/v1/`)
- `Trade`: price, size, side (if known), venue trade id.
- `Quote`: bid/ask price & size (L1).
- `Bar`: timeframe, OHLCV, vwap (optional), trade count, `adjusted` flag (12).
- `BookSnapshot`: top-N levels each side, sequence number.
- `NewsItem`: title, body (or Garage ref), url, language, source outlet, tickers/entities mentioned (raw), dedup key.
- `MediaChunk`: media_id, kind (video/audio/image), segment start/end, Garage ref, origin (channel, event), transcript ref (filled later by ASR worker).
- `MacroObservation`: series id, period, value, **vintage/release time**, revision flag.
- `OnchainEvent`: chain, block, tx hash, event kind, decoded fields.
- `FilingEvent`: issuer (LEI/CIK), form type, filing time, accession, document refs.

### 5.3 Quarantine
Postgres `ingest.quarantine(source_id, venue_symbol, first_seen, last_seen, count, sample_raw_ref, status, instrument_id)`. Implemented variant (D-050, D-054): unknown `(source, symbol)` pairs of market-data sources are provisioned as `status='provisional'` instruments and recorded here as `open` with the provisional `instrument_id`; control-api resolves them (activate, merge, ignore).

## 6. Behaviour & algorithms

### 6.1 Pipeline per message
1. Receive raw → assign `ingest_time` (UTC, ns) → write to Garage raw archive (batched, async, at-least-once).
2. Normalize → canonical envelopes; resolve symbols via `SymbolResolver` (cached; misses are provisioned + quarantined and emit `ingest.unresolved.<source>`).
3. Validate (schema, monotonic sequence where applicable, sane prices — **(proposal)** reject price jumps > X σ only with flag, never silently drop). Not implemented in the ingestor; the risk-engine's per-instrument feed guard holds unconfirmed jumps (D-071).
4. Publish to NATS with `Nats-Msg-Id = event_id` (JetStream dedup).
5. Emit metrics; market data → VictoriaMetrics, macro/filings/facts → `wh.*`.

### 6.2 Time semantics
- All times UTC, nanosecond precision in proto, millisecond in VictoriaMetrics.
- Downstream models may only use an event at time *t* if `publish_time ≤ t` (and `ingest_time ≤ t` in live). Backtests use `publish_time`; where a source provides no publish time, `publish_time := ingest_time` in live and **(proposal)** `event_time + source_latency_estimate` in backfill, flagged `publish_time_estimated`. Implemented: connectors with a known release schedule estimate it and set `publish_time_estimated` (frankfurter and history-less ECB flows: 16:00 Europe/Berlin; SEC XBRL facts: filing date 22:00 ET); for market data the measured per-source knowledge delay is a catalog field (`backfill.knowledge_delay_secs`: kraken 120 s, yahoo 100 s) applied by `fa_backtest::knowledge` in replays, not by the ingestor (D-079).
- Macro and fundamentals: one event per **vintage** (revisions are new events, never overwrites).

### 6.3 Rate limiting & retry
- Token bucket per source (and per endpoint where catalog specifies), limits from `sources.yaml`.
- Exponential backoff with jitter; circuit breaker opens after N consecutive failures → health event `degraded`/`down` → consumed by risk health halts (40).
- Streams: auto-reconnect; on reconnect, gap-fill via REST/backfill where the source allows, marking gap-filled events. Implemented: reconnect only (`oanda`, the one real stream connector, never run live); the Envelope `gap_filled` flag exists but nothing sets it yet, bar gaps are reported on `ingest.gap.<source>`.

### 6.4 Backfill
- Requests carry `(source_id, instruments | query, timeframe, from, to)`; split into chunks; cursors persisted in Postgres so backfill is resumable and idempotent.
- Default depth per timescale per §3; scheduler runs backfill in low-priority lanes so live streams are never rate-starved.

**Implemented today (D-050):** `yahoo`, `hackernews` and `google_news` run live without keys; `lemon` and `oanda` are implemented against the public docs but unverified. Yahoo bars are stamped at bar close and also emit a synthetic `Trade` per completed bar so `md.trade.>` consumers see keyless data. Each connector implements `Backfill` with cursors in `ingest.backfill_cursor`; the `jobs.ingest.backfill` work queue drives it since D-054 (§4.3). Low-priority lanes are not implemented: one backfill job runs at a time and shares the source's token bucket with the live connector.

**Since 2026-11-11 (§6.1, §6.3, §6.5, §8):** the pipeline finalizes every envelope the same way for live, backfill and replay (`Finalize`): `event_id` is a **stable ULID** (`connector.StableEventID`: time part = event time, entropy = sha256 of source, instrument and the deterministic payload encoding, or the news dedup key), `schema_version = 1`, new flags `raw_missing` (archive configured but the payload could not be archived), `backfill`, `publish_time_estimated`, `gap_filled` (Envelope fields 9–12), and it publishes with `Nats-Msg-Id = event_id`, so JetStream drops re-polled duplicates within the stream duplicate window. News is additionally de-duplicated across sources by `NewsDedupKey` (sha256 of the normalized URL: lowercase host without `www.`, no fragment, no `utm_*`/click-id params, sorted query, no trailing slash, http = https; fallback title + outlet + hour bucket) in a bounded 50 000-key set (counter `dedup.<source>`). The raw archive writes an index sidecar (`<ulid>.idx.jsonl.gz`: offset, length, received_at, content type, transport meta) next to every data object, which is what makes replay exact; failed uploads go to a bounded local spool (`-spool-dir` / `FA_ARCHIVE_SPOOL`, 256 MiB) retried every 30 s, and a full spool makes `Put` refuse so envelopes carry `raw_missing` (counters `archive.spooled|respooled|raw_missing|spool_bytes`). Poll failures back off exponentially from 5 s with 20 % jitter, capped at the poll interval; after `breaker.failures_to_open` (5) consecutive failures the breaker opens and retries every `retry.max_backoff` (60 s) (counter `breaker_open.<source>`). Verified live: replaying a 75 s window reproduced every live kraken (15/15) and yahoo (6/6) message byte-identically (same `Nats-Msg-Id`, same bytes); two `verify` runs over the same range give the same digest.

**Implemented today (keyless second wave, 2026-11-11):** five more keyless connectors run live: `frankfurter` (ECB reference rates → `MacroObservation` per `<BASE>_<CCY>` and business day; rates are a daily fixing, not a venue bid/ask, so not `Quote`s; vintage = estimated 16:00 Europe/Berlin release, `publish_time_estimated`), `ecb` (SDMX REST with `includeHistory=true`: one `MacroObservation` per `validFrom` version, `revision` when the payload holds an earlier vintage of the period; history-less flows such as EXR get a deterministic estimated vintage; series id = `<FLOW>:<key dims>` so it stays one subject token), `gdelt` (DOC 2.0 artlist → `NewsItem`, dedup = sha1 of the normalized URL, time = `seendate`, backfill only over the API's ~3-month window, throttled to 1 request / 10 s), `sec_edgar` (`data.sec.gov` submissions per CIK → `FilingEvent`, time = `acceptanceDateTime`, declared User-Agent from `config.extra.user_agent`; XBRL company facts of whitelisted concepts → `FundamentalFact` on `filing.fact.sec_edgar` → `wh.fundamental`, one row per vintage, D-078) and `kraken` (public OHLC/Trades/Ticker → `Bar`/`Trade`/`Quote`; backfill limited to Kraken's latest 720 intervals). Macro observations are also written by the ingestor (with `-postgres`) to `wh.macro_observation` through `go/pkg/warehouse`, idempotent on `(source_id, series_id, period, vintage)` (spec 20 §6.1 (b)); filings are written to `wh.filing` (migration `20261111000045`, idempotent on `(source_id, accession)`; verified: 46 SEC filings from one backfill job).

### 6.5 News & text
- Dedup key = normalized URL hash, fallback title+outlet+time-bucket hash (implemented across sources in the ingestor pipeline, §6.4); near-duplicate (syndicated) detection is an open question.
- Entity/ticker tagging is downstream (31); ingestion only passes through source-provided tags.

### 6.6 Media
- Live streams recorded in fixed segments (**(proposal)** 30 s video, 30 s audio with 2 s overlap) → Garage → `MediaChunk` event.
- ASR and frame-embedding jobs are emitted to `jobs.media.*` work queues (21) and processed by Rust workers (31); results reference `media_id` + segment.
- Tooling for capture (e.g., yt-dlp/ffmpeg as subprocess) **(proposal)**; licensing/ToS noted in catalog only.

### 6.7 Health
Each connector publishes `ingest.health.<source_id>` (`fa.v1.IngestHealth`) every 5 s (implemented; D-054): state (ok/degraded/down), activity and event ages, error counts and rate, limiter saturation. Staleness thresholds per source feed the data-health halts (40).

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `archive.batch_bytes` | 8 MiB | raw archive batching |
| `archive.flush_interval` | 5 s | implemented: an object per (source, UTC minute), uploaded when the minute ends, at 8 MiB or on shutdown; spool 256 MiB, retried every 30 s |
| `retry.max_backoff` | 60 s | |
| `breaker.failures_to_open` | 5 | |
| `health.interval` | 10 s | implemented: 5 s (`HealthInterval`) |
| `staleness.<class>` | stream 30 s / poll 3× interval | feeds 40 halts; implemented as `down` after 3 × expected period (stream 30 s, poll = interval) |
| `media.segment_seconds` | 30 | |
| `backfill.depth.{s,m,h,d}` | 4w / 1y / 5y / max | per §3 |
| `resolver.cache_ttl` | 1 h | |

## 8. Failure modes
| Failure | Response |
|---|---|
| Source down / rate-limited | breaker, backoff, health `down`, downstream halts affected strategies |
| Stream gap | reconnect, gap-fill if possible, mark `gap_filled`; otherwise emit `ingest.gap` event |
| Unknown symbol | quarantine + alert; never guessed |
| Garage unavailable | buffer to local disk spool (bounded); if spool full, keep publishing live but flag `raw_missing` (implemented: `-spool-dir`) |
| NATS unavailable | bounded in-memory/disk buffer; then stop connector (no silent loss) |
| Schema change at source | normalization error rate spike → connector `degraded`, raw still archived for re-normalization |
| Clock skew | ingestor hosts NTP-synced; `ingest_time` vs `event_time` drift monitored |

## 9. Acceptance criteria
- Every canonical event on the bus has non-null `event_time`, `publish_time`, `ingest_time`, `raw_ref` (or `raw_missing` flag).
- Replaying raw archive for a period through the normalizers produces byte-identical canonical events (deterministic normalization).
- Killing and restarting an ingestor during backfill resumes without duplicates (JetStream dedup + cursors).
- First-wave connectors (Yahoo, Lemon, OANDA, HN, Google News) produce canonical events end-to-end in local compose.
- Unresolved symbols appear in quarantine and in the frontend ops view.
- Rate limits from catalog are never exceeded in a 24 h soak test.

**Status (2026-11-11):** raw ref / `raw_missing` — met (pipeline `Finalize`); byte-identical replay — met and verified live (`ctl.ingest.replay`); restart during backfill without duplicates — met by stable event ids + `Nats-Msg-Id` + per-range cursors; first wave — Yahoo, HN, Google News live, Lemon/OANDA need keys (out of scope); quarantine in the ops view — met (D-054); 24 h soak — not run (limiter is per source and shared with backfill, `limiter_saturation` is reported).

## 10. Open questions
1. Which sources stream vs. poll — needs per-source verification (blocks connector priorities, 11).
2. Near-duplicate / syndicated news detection method (blocks text pipeline quality, 31).
3. Media capture tooling and segment sizes; whether video is stored at full resolution or downsampled at ingest (blocks Garage budget, 20).
4. Order-book depth N per venue and snapshot frequency (blocks VM modeling, 20).
5. Publish-time estimation for backfilled data without availability timestamps (blocks backtest realism, 41).
6. Does the broker-gateway's own market data (e.g., OANDA, IBKR) flow through ingestor or gateway? **(proposal)** ingestor, single path.

## 11. Deferred
- Keyword lists and web-search-driven discovery of news/media (`DiscoveryHook` only).
- Per-source connector specs beyond first wave.
- Data licensing / ToS enforcement (catalog field only).
