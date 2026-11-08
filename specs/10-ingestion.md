# 10 — Ingestion

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium-high |
| **Owners (code)** | `go/pkg/connector`, `go/pkg/catalog`, `go/pkg/bus`, `go/connectors/*`, `go/services/ingestor` |
| **Depends on** | 11-source-catalog, 12-reference-data, 20-storage, 21-bus-and-contracts |

## 1. Purpose
Pull every reachable data source (market data, broker feeds, FX, macro, on-chain, filings, news, broadcast media) into the platform as **canonical, point-in-time-correct events** on the NATS bus, and archive the raw payloads so any period can be replayed exactly. Ingestion is the only place where source-specific formats exist; everything downstream sees canonical protobuf.

## 2. Scope
**In scope:** connector framework (Go), stream/poll/backfill modes, rate limiting and retry, normalization to canonical events, raw-payload archival to Garage, instrument symbol resolution via reference data, media capture (video/audio/images) into `MediaChunk`s, connector health reporting, historical backfill per timescale.
**Out of scope:** feature engineering and embeddings (31), ASR/frame embedding *models* (31; ingestion only cuts and stores media), order execution (40 — broker-gateway owns execution even for venues that also supply data), web-search-driven discovery (deferred).

## 3. Decisions
- Connectors are written in **Go** and run inside the `ingestor` service (one goroutine group per connector instance); scalable horizontally by sharding connector instances across ingestor replicas.
- Three connector modes: **Stream** (WebSocket/FIX/SSE), **Poll** (REST on schedule), **Backfill** (bounded historical range).
- **Canonical events** (protobuf, `fa.v1`): `Trade`, `Quote`, `Bar`, `BookSnapshot`, `NewsItem`, `MediaChunk`, `MacroObservation`, `OnchainEvent`, `FilingEvent`.
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
}
```
`RawMessage` = `{source_id, received_at, transport_meta, bytes, content_type}`.

### 4.2 Outputs
- NATS subjects per 21 (`md.*`, `news.*`, `media.*`, `macro.*`, `onchain.*`, `filing.*`, `ingest.health.*`).
- Garage: `raw/<source_id>/<yyyy>/<mm>/<dd>/<hh>/<ulid>.<ext>` (+ zstd).
- Garage: `media/<source_id>/<media_id>/{video|audio|frames}/…`.
- Postgres: connector run log, backfill progress cursors, quarantine table.
- VictoriaMetrics: ingestion metrics (lag, msgs/s, error rates).

### 4.3 Control
- `jobs.ingest.backfill` work queue (21) for backfill requests from the control API / trainer.
- `ctl.ingest.{start|stop|status}` request/reply with arg `source` (D-036; `reload` pending).
- `DiscoveryHook` interface: `Discover(ctx, query) ([]SourceTarget, error)` — no implementation in v1.

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
Postgres `ingest_quarantine(source_id, venue_symbol, first_seen, last_seen, count, sample_raw_ref, status)`.

## 6. Behaviour & algorithms

### 6.1 Pipeline per message
1. Receive raw → assign `ingest_time` (UTC, ns) → write to Garage raw archive (batched, async, at-least-once).
2. Normalize → canonical envelopes; resolve symbols via `SymbolResolver` (cached; misses go to quarantine and emit `ingest.unresolved.<source>`).
3. Validate (schema, monotonic sequence where applicable, sane prices — **(proposal)** reject price jumps > X σ only with flag, never silently drop).
4. Publish to NATS with `Nats-Msg-Id = event_id` (JetStream dedup).
5. Emit metrics.

### 6.2 Time semantics
- All times UTC, nanosecond precision in proto, millisecond in VictoriaMetrics.
- Downstream models may only use an event at time *t* if `publish_time ≤ t` (and `ingest_time ≤ t` in live). Backtests use `publish_time`; where a source provides no publish time, `publish_time := ingest_time` in live and **(proposal)** `event_time + source_latency_estimate` in backfill, flagged `publish_time_estimated`.
- Macro and fundamentals: one event per **vintage** (revisions are new events, never overwrites).

### 6.3 Rate limiting & retry
- Token bucket per source (and per endpoint where catalog specifies), limits from `sources.yaml`.
- Exponential backoff with jitter; circuit breaker opens after N consecutive failures → health event `degraded`/`down` → consumed by risk health halts (40).
- Streams: auto-reconnect; on reconnect, gap-fill via REST/backfill where the source allows, marking gap-filled events.

### 6.4 Backfill
- Requests carry `(source_id, instruments | query, timeframe, from, to)`; split into chunks; cursors persisted in Postgres so backfill is resumable and idempotent.
- Default depth per timescale per §3; scheduler runs backfill in low-priority lanes so live streams are never rate-starved.

### 6.5 News & text
- Dedup key = normalized URL hash, fallback title+outlet+time-bucket hash **(proposal)**; near-duplicate (syndicated) detection is an open question.
- Entity/ticker tagging is downstream (31); ingestion only passes through source-provided tags.

### 6.6 Media
- Live streams recorded in fixed segments (**(proposal)** 30 s video, 30 s audio with 2 s overlap) → Garage → `MediaChunk` event.
- ASR and frame-embedding jobs are emitted to `jobs.media.*` work queues (21) and processed by Rust workers (31); results reference `media_id` + segment.
- Tooling for capture (e.g., yt-dlp/ffmpeg as subprocess) **(proposal)**; licensing/ToS noted in catalog only.

### 6.7 Health
Each connector publishes `ingest.health.<source_id>` every 10 s: state (up/degraded/down), lag, last event time, error counts. Staleness thresholds per source feed the data-health halts (40).

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `archive.batch_bytes` | 8 MiB | raw archive batching |
| `archive.flush_interval` | 5 s | |
| `retry.max_backoff` | 60 s | |
| `breaker.failures_to_open` | 5 | |
| `health.interval` | 10 s | |
| `staleness.<class>` | stream 30 s / poll 3× interval | feeds 40 halts |
| `media.segment_seconds` | 30 | |
| `backfill.depth.{s,m,h,d}` | 4w / 1y / 5y / max | per §3 |
| `resolver.cache_ttl` | 1 h | |

## 8. Failure modes
| Failure | Response |
|---|---|
| Source down / rate-limited | breaker, backoff, health `down`, downstream halts affected strategies |
| Stream gap | reconnect, gap-fill if possible, mark `gap_filled`; otherwise emit `ingest.gap` event |
| Unknown symbol | quarantine + alert; never guessed |
| Garage unavailable | buffer to local disk spool (bounded); if spool full, keep publishing live but flag `raw_missing` |
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
