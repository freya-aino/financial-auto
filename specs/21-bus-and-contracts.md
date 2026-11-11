# 21 — Bus & contracts

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium-high |
| **Owners (code)** | `proto/`, `crates/fa-proto`, `crates/fa-bus`, `go/pkg/bus`, `compose.yaml` (NATS) |
| **Depends on** | 10-ingestion, 20-storage, 60-ops |

## 1. Purpose
Define how components talk: NATS JetStream subjects and streams for live data and control, work queues for distributable jobs, and protobuf as the single schema shared by Rust and Go. Anything crossing a process boundary goes through a contract defined here.

## 2. Scope
**In scope:** subject taxonomy, JetStream stream and consumer configs, retention, protobuf packages and versioning rules, job/result envelope for work queues, request/reply control conventions, backpressure.
**Out of scope:** HTTP API to the frontend (50), storage schemas (20).

## 3. Decisions
- **NATS JetStream** for live streams and work queues (ADR 0004).
- **Protobuf** in `proto/`, built with **buf**; Rust via **prost**, Go via **protoc-gen-go** (ADR 0005).
- Subject roots: `md.*`, `news.*`, `media.*`, `latent.*`, `signal.*`, `order.*`, `jobs.*` (+ `macro.*`, `onchain.*`, `filing.*`, `ingest.*`, `ctl.*`, `ref.*`, `catalog.*`, `risk.*`).
- **Horizontal scaling contract:** stateless Rust workers **pull** jobs from JetStream work-queue streams; artifacts to Garage; results/lineage to Helix.
- Message dedup via `Nats-Msg-Id = event_id` / `job_id`.

## 4. Interfaces

### 4.1 Subject taxonomy
| Subject | Payload | Producer → Consumer |
|---|---|---|
| `md.trade.<venue>.<inst>` | `Envelope{Trade}` | ingestor → runtime, VM writer |
| `md.quote.<venue>.<inst>` | `Envelope{Quote}` | |
| `md.bar.<tf>.<venue>.<inst>` | `Envelope{Bar}` | |
| `md.book.<venue>.<inst>` | `Envelope{BookSnapshot}` | |
| `news.item.<source>` | `Envelope{NewsItem}` | ingestor → text workers, runtime |
| `media.chunk.<source>` | `Envelope{MediaChunk}` | ingestor → media workers |
| `macro.obs.<source>.<series>` | `Envelope{MacroObservation}` | |
| `onchain.event.<chain>` | `Envelope{OnchainEvent}` | |
| `filing.event.<source>` | `Envelope{FilingEvent}` | |
| `latent.state.<timescale>.<instrument>` | `LatentState` (carries `snapshot_id`) | model-runtime → strategies, UI (D-039) |
| `latent.region.<timescale>` | `RegionEvent` (created/merged/split) | |
| `latent.transition.<timescale>` | `TransitionUpdate` | MCMC → strategies, UI |
| `latent.membership.<timescale>.<instrument>` | `RegionMembership` (soft, k = 4) | model-runtime → strategies, UI (D-069, spec 30 §4.2) |
| `latent.lifecycle.<timescale>` | `RegionLifecycleEvent` | model-runtime → Helix sink, UI (D-069) |
| `latent.mcmc.<timescale>.<expert>` | `KernelUpdate` | model-runtime → UI, strategies (D-069, spec 33 §4.3) |
| `latent.hier.<fine>.<coarse>` | `RegionHierarchy` | model-runtime → UI, Helix (D-069, spec 30 §6.5) |
| `latent.crossprior.<timescale>` | `CrossAssetPriors` | model-runtime → evo-orchestrator, UI (D-069, spec 33 §6.5) |
| `latent.expert.<timescale>.<instrument>` / `latent.expert.health` | `ExpertView` / `ExpertHealth` | model-runtime → decoders, UI (D-073, spec 32 §4.3) |
| `decoder.output.<family>` / `decoder.shadow.<family>` | `DecoderOutput` (refs) | evo/runtime → UI, strategies; shadow candidates on `decoder.shadow.*` (D-068) |
| `signal.exposure.<strategy>` | `ExposureDistribution` | strategies → allocator |
| `order.intent.<account>` | `OrderIntent` | allocator → risk |
| `order.approved.<account>` | `OrderIntent` | risk → broker-gateway |
| `order.event.<account>` | `OrderEvent` (ack/fill/cancel/reject) | broker-gateway → ledger, risk |
| `order.cancel.<account>` | `OrderCancel` (legacy `OrderIntent` still decodes) | operator / tools → broker-gateway (D-076; no service publishes it yet) |
| `risk.state` / `risk.halt.<scope>` | `RiskState` / `Halt` | risk-engine → all; scopes `portfolio`, `daily_loss`, `source`, `model`, `broker` (D-042) |
| `risk.resume.<scope>` | `Resume` (wire-compatible with `Halt`) | control-api / operator / ingestor → risk-engine; re-arms a halt (D-030) |
| `filing.fact.<source>` | `Envelope{FundamentalFact}` (field 29; XBRL company facts) | ingestor (`sec_edgar`) → `wh.fundamental` |
| `risk.event.<account>` | `RiskEvent` (intent rejected, halt raised/cleared, ...) | risk-engine → UI, audit; rejections are still mirrored as `OrderEvent{REJECTED}` |
| `jobs.<kind>` | `Job` | orchestrator/trainer → workers; `jobs.mcmc.refit` (`JOBS_MCMC`): model-runtime → its `mcmc` worker (spec 33 §6.3) |
| `jobs.result.<kind>` | `JobResult` | workers → orchestrator |
| `jobs.progress.<job_id>` | `JobProgress` | workers → UI |
| `ingest.health.<source>` / `ingest.gap.*` / `ingest.unresolved.*` | `IngestHealth` / `IngestGap` / `IngestUnresolved` (legacy JSON accepted) | ingestor → risk, UI |
| `ctl.<service>.<verb>` | `CtlRequest` / `CtlReply` | control-api ↔ services; today `ctl.ingest.{start,stop,status,enable,disable,backfill,replay}` (D-036, D-054, D-067), `ctl.evo.{start,cancel}` (D-037), `ctl.strategy.{reload,status,…}`, `ctl.train.cancel`, `ctl.model.{reload,shadow,adapter}`, `ctl.risk.{strategy,flatten,limits,pipeline,allocators}` (D-051, D-071), `ctl.decoders.{status,reload,pin,off,retrain}` (D-068, D-074) |
| `ctl.gpu.{acquire,release}` / `ctl.gpu.{granted,preempt}` | request/reply / notifications | workers, trainer ↔ GPU arbiter in evo-orchestrator (D-006, D-044) |
| `ref.changed.<entity>`, `catalog.changed.<id>` | `RefChanged` | ingestor, control-api → ingestor resolver (both subjects flush it; `catalog.changed` also reloads catalog statuses), risk-engine RefTable |
| `svc.heartbeat.<service>` | `Heartbeat` | all → UI |
| `ops.alert` | JSON `AlertEvent` (`status`, `alertname`, `severity`, `summary`, `fingerprint`, `labels`, `annotations`, `starts_at`, `ends_at`, `at`) | control-api (vmalert receiver, D-066), dead-letter forwarder → UI, audit; core-only, by design not proto (the vmalert/Alertmanager shape) |
| `dlq.<STREAM>.<original subject>` | the original payload + `Fa-Dlq-*` headers | dead-letter forwarder → `DLQ` stream (§8) |

Tokens (`<inst>`) use the instrument UUID without dashes (D-017; `InstrumentId::subject_token`, Go `bus.SubjectToken`) to keep subjects safe.

### 4.2 Proto packages
```
proto/fa/v1/
  envelope.proto     # Envelope, times, raw_ref
  market.proto       # Trade, Quote, Bar, BookSnapshot
  text.proto         # NewsItem, FilingEvent, FundamentalFact
  media.proto        # MediaChunk
  macro.proto        # MacroObservation, OnchainEvent
  latent.proto       # LatentState, RegionEvent, TransitionUpdate, ExpertSpaceRef, RegionMembership,
                     # RegionLifecycleEvent, KernelUpdate, RegionHierarchy, CrossAssetPriors,
                     # ExpertView, ExpertHealth (+ their parts)
  decoder.proto      # DecoderOutput, Analog
  strategy.proto     # ExposureDistribution, AssetExposure
  order.proto        # OrderIntent, OrderEvent, OrderCancel
  risk.proto         # RiskState, Halt, Resume, RiskEvent
  ingest.proto       # IngestHealth, IngestGap, IngestUnresolved, RefChanged
  jobs.proto         # Job, JobResult, JobProgress
  control.proto      # Heartbeat, ctl requests/replies
```

### 4.3 Job / result envelope
```proto
message Job {
  string job_id = 1;            // ULID, also Nats-Msg-Id
  string kind = 2;              // e.g. "evo.eval.decoder", "train.encoder", "media.asr"
  string parent_job_id = 3;
  uint32 priority = 4;          // maps to GPU priority class (60)
  string snapshot_id = 5;       // encoder snapshot the job runs against
  string genome_id = 6;         // optional
  bytes  spec = 7;              // kind-specific proto (Any-like, typed by kind)
  repeated string input_refs = 8;  // Garage keys
  google.protobuf.Timestamp deadline = 9;
  uint64 budget_gpu_ms = 10;
  string checkpoint_ref = 11;   // resume point for preempted jobs
  uint64 seed = 12;             // reproducibility
}
message JobResult {
  string job_id = 1;
  enum Status { STATUS_OK = 0; STATUS_FAILED = 1; STATUS_PREEMPTED = 2;
                STATUS_BUDGET_EXCEEDED = 3; STATUS_CANCELLED = 4; }  // buf STANDARD prefixes
  Status status = 2;
  bytes  result = 3;            // kind-specific (e.g. FitnessRecord)
  repeated string output_refs = 4;
  string checkpoint_ref = 5;
  uint64 used_gpu_ms = 6;
  string worker_id = 7;
  string error = 8;
  map<string, double> objectives = 9;   // multi-objective fitness (spec 34 §6.3)
  map<string, string> directions = 10;  // "min" | "max" per objective
}
```
`JobProgress` published on `jobs.progress.<job_id>` for UI.

## 5. Data model — JetStream streams
| Stream | Subjects | Retention | Max age | Notes |
|---|---|---|---|---|
| `MD` | `md.>` | limits | 24 h | live + short replay; long history in VM/Garage |
| `TEXT` | `news.>`, `filing.>` | limits | 7 d | |
| `MEDIA` | `media.>` | limits | 7 d | refs only, blobs in Garage |
| `MACRO` | `macro.>`, `onchain.>` | limits | 30 d | |
| `LATENT` | `latent.>`, `decoder.>` | limits | 7 d | |
| `SIGNAL` | `signal.>` | limits | 30 d | |
| `ORDERS` | `order.>`, `risk.>` | limits | 365 d, file storage, replicas when clustered | ledger is source of truth, stream is audit aid |
| `JOBS_<kind>` | `jobs.<kind>.>` | **work-queue** | until acked | one stream per kind (`JOBS_EVO`, `JOBS_TRAIN`, `JOBS_MCMC`); `fa_bus::streams::ensure_work_queue` |
| `JOB_RESULTS` | `jobs.result.>` | limits | 30 d | |
| `DLQ` | `dlq.>` | limits | 30 d | dead letters of every stream (§8) |

Consumers: durable pull consumers per service; `AckExplicit`; `MaxAckPending` set per consumer for backpressure; work queues use `AckWait` + heartbeat `in-progress` acks for long jobs.

**Today:** durable explicit-ack pull consumers exist only for the work queues (`JOBS_EVO`, `JOBS_TRAIN`, `JOBS_MCMC`: `ack_wait` 60 s, `max_deliver` 5, `AckKind::Progress` heartbeats during long jobs) and control-api's `ops-job-history` on `JOB_RESULTS`. Live consumers (model-runtime, strategies, risk-engine on `order.*`, broker-gateway, control-api's tap) use core NATS subscriptions, so the streams are a replay/audit record rather than their delivery path; JetStream reads elsewhere are ephemeral ordered consumers (model-runtime's `TEXT` news backlog at start, `fa_bus::replay` time-range reads for backtests). No consumer sets `MaxAckPending`.

**Implemented today (D-046):** the limits streams above are declared once in `crates/fa-bus/streams.json` (mirrored byte-for-byte in `go/pkg/bus/streams.json`) and ensured idempotently by control-api when NATS connects and by the ingestor at start; `GET /api/bus/streams` lists them. `ctl.*`, `svc.heartbeat.*`, `ingest.*`, `ref.changed.*`, `catalog.changed.*`, `jobs.progress.*` and `ops.alert` stay core-only.

**Disk budget (`max_mib`, 2026-11-11, closes open question 1):** every stream carries a byte cap in `streams.json` (`max_bytes`, limits discard-old), sized from live `nats stream info` after the 19:48 UTC data reset with ≥ 2× headroom over the projected full-retention volume; the LATENT rate was measured after D-069's soft memberships (≈ 44 MB/h off-hours, roughly double the pre-D-069 average of 10.7 MB/h):

| Stream | Measured (7.5 h) | Rate | Projected at max age | `max_mib` |
|---|---|---|---|---|
| `MD` | 4.5 MB | 0.7–2 MB/h off-hours | ≤ 0.5 GB / 24 h with market hours | 4096 |
| `TEXT` | 1.1 MB | 0.15 MB/h | 25 MB / 7 d | 2048 |
| `MEDIA` | 0 | — | — | 2048 |
| `MACRO` | 0.56 MB (backfill) | — | < 0.1 GB / 30 d | 2048 |
| `LATENT` | 80.5 MB | 44 MB/h (soft memberships) | 7.4 GB / 7 d off-hours, ≈ 2× in market hours | 24576 |
| `SIGNAL` | 31.6 MB | 4–10 MB/h | 3–7.4 GB / 30 d | 16384 |
| `ORDERS` | 6.0 MB | 0.4–0.8 MB/h | 3.5–7 GB / 365 d | 16384 |
| `JOB_RESULTS` | 0.5 MB | bursty | < 0.1 GB / 30 d | 2048 |
| `DLQ` | 0 | — | — | 1024 |

The caps sum to 69 GiB against the server's 202 GB JetStream storage. `ensure_streams` / `EnsureStreams` never apply a cap below a stream's current bytes (that would purge at once): the old cap stays and a warning names the stream, so a purge is always an operator decision. Neither side ever shortens a live stream's `max_age` (D-082: Go `safeMaxAge`/`mergeStreamConfig`, reported through `bus.OnLimitHeld`; Rust `fa_bus::streams::safe_max_age`, logged): a longer or unlimited age is applied, a shorter one (or a finite one on an unlimited stream) keeps the live value, because shortening retention purges everything older at once. `GET /api/bus/streams` reports `max_bytes` per stream.

**Since 2026-11-11 (typed contracts):** the JSON workarounds of D-030, D-042 and D-054 are replaced by `fa.v1` messages, generated for Rust (`fa-proto` build) and Go (`buf generate` via Docker, committed in `go/pkg/bus/pb`); `buf lint` and `buf breaking --against main` pass. Compatibility rules used: `Resume` keeps `Halt`'s field numbers 1–4, so consumers decoding `Halt` keep working; ingest frames are switched producer-side with consumers accepting both encodings (`fa_bus::ingest::{is_json, frame_json, health_status}`); `RiskEvent` is dual-published next to the old `OrderEvent{REJECTED}`. `Envelope` gained `raw_missing`, `publish_time_estimated`, `gap_filled`, `backfill` (fields 9–12). Subjects helpers: `fa_bus::subjects::{risk_event, ingest_health, ingest_gap, ingest_unresolved, ref_changed, catalog_changed}` and the Go mirrors in `go/pkg/bus/subjects.go`. The ingestor publishes canonical events with `Nats-Msg-Id = event_id` and stable ids (spec 10), which is what makes §3's dedup rule effective. The control-api bus tap decodes the new messages. Still JSON inside proto: `CtlReply.message` for `status`/`replay` replies and `JobResult.result` (by design, D-044); `order.cancel.<account>` carries `fa.v1.OrderCancel` since D-076 (a legacy `OrderIntent` payload still decodes). `OrderEvent` gained `venue_fill_id = 12`, `instrument_id = 13` and `venue_symbol = 14` (D-078): the paper gateway stamps every fill `pf-<unix ns>-<seq>`, and risk-engine's fill id becomes `intent:venue_order:vf:<id>` when present (legacy `intent:venue_order:<ms>` otherwise).

## 6. Behaviour & algorithms
- **Versioning:** package `fa.v1`; additive changes only (new fields, new messages); breaking change → `fa.v2` side by side. `buf breaking` against `main` in CI.
- **Ordering:** per-subject ordering relied on only within an instrument subject; consumers must tolerate out-of-order across subjects and use `event_time`.
- **Delivery:** at-least-once everywhere; consumers idempotent by `event_id`/`job_id`.
- **Backpressure:** slow consumers bounded by `MaxAckPending`; live runtime drops to latest-only for `md.quote` when behind **(proposal)**, never for `order.*`.
- **Job lifecycle:** submit → pulled → in-progress acks → result → ack. Preempted jobs publish `PREEMPTED` with `checkpoint_ref` and are re-queued by the orchestrator at the same priority.
- **Control:** `ctl.*` request/reply with 5 s default timeout; idempotent verbs.

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `stream.MD.max_age` | 24 h | |
| `consumer.max_ack_pending` | 10,000 (md) / 1 (gpu jobs per worker) | |
| `jobs.ack_wait` | 60 s + in-progress heartbeats | |
| `ctl.timeout` | 5 s | |

## 8. Failure modes
| Failure | Response |
|---|---|
| NATS down | producers buffer (bounded) then stop; risk sees heartbeat loss → halts trading |
| Consumer lag | metrics alert; quote coalescing; never drop orders |
| Poison message | moved to `dlq.<STREAM>.<subject>` (stream `DLQ`) after `max_deliver` redeliveries or a `Term` ack, `ops.alert` + `fa_bus_dead_letters_total` (implemented, see below) |
| Proto incompatibility | `buf breaking` blocks merge; runtime rejects unknown major version |
| Worker crash mid-job | ack timeout → redelivery; resume from last checkpoint |

## 9. Acceptance criteria
**Remaining (2026-11-11, code checked at e25e9b5):**
- **Exactly-once across worker crashes** (criterion 2): no test. Workers ack after publishing a result and `Nats-Msg-Id = job_id` deduplicates resubmission, but nothing kills a worker mid-job and checks the effect.
- **Scaling contract** (criterion 3): no test or measurement of two evo-workers on a CPU-bound fake job.
- **Backpressure** (§5 consumers, §6, §7 `consumer.max_ack_pending`): not implemented as specified — live consumers are core subscriptions (§5 "Today"), no `MaxAckPending` is set, and `md.quote` latest-only coalescing (§6, proposal) does not exist; open question 2 is undecided.
- **"Runtime rejects unknown major version"** (§8): nothing checks the package version at runtime (only `fa.v1` exists).
- **CI**: `buf breaking` is run by hand through the `bufbuild/buf` image; there is no CI job (§6 "in CI").
- Open questions 3 (typed job specs) and 4 (work queue per priority class vs consumer priority groups) are undecided; `max_mib` should be re-measured after a week of market hours (Q1).

- Rust and Go both build from the same `proto/` with `buf generate`; `buf lint` and `buf breaking` pass.
- A job published to `jobs.<kind>` is processed exactly once in effect (idempotent result) across worker crashes in a test.
- Adding a second evo-worker doubles throughput on a CPU-bound fake job (scaling contract works).
- Every subject in §4.1 has a proto payload type.

**Status (2026-11-11):** `buf lint` / `buf breaking` pass (run via the `bufbuild/buf` image, no CI yet); every §4.1 subject now has a proto payload type (`ops.alert` and the DLQ copies excepted by design); the exactly-once and scaling criteria are owned by the job-lifecycle work (D-064). Every stream is capped (`max_mib`, §5).

**Dead letters (§8, implemented):** `fa_bus::dlq::spawn_forwarder` (run by control-api right after `ensure_streams`) subscribes to the JetStream advisories `$JS.EVENT.ADVISORY.CONSUMER.MAX_DELIVERIES.>` and `…MSG_TERMINATED.>`, fetches the advised message by stream sequence and republishes it on `dlq.<STREAM>.<original subject>` with the original headers plus `Fa-Dlq-Stream`, `Fa-Dlq-Consumer`, `Fa-Dlq-Seq`, `Fa-Dlq-Subject`, `Fa-Dlq-Reason` (`max_deliveries` / `terminated[: reason]`), `Fa-Dlq-Deliveries` and `Nats-Msg-Id = dlq:<stream>:<seq>` (several forwarders deduplicate). Work-queue originals are then deleted (they would otherwise sit unacked forever); limits streams keep theirs. Each dead letter increments `fa_bus_dead_letters_total{stream,consumer,reason}` and at most once a minute per (stream, consumer) publishes a `DeadLetter` alert on `ops.alert` (D-066 shape). Consumers mark undecodable messages with `AckKind::Term` (the `mcmc.refit` worker does) so poison skips the remaining redeliveries. Verified against the compose NATS by `FA_NATS_TEST=1 cargo test -p fa-bus live_dead_letter` (a job nacked to `max_deliver` lands in `DLQ` with its headers and leaves the work queue). Go helpers: `bus.DeadLetter`, `bus.OpsAlert`.

## 10. Open questions
1. Retention per stream — confirm §5 values against disk budget (blocks 60). *Resolved 2026-11-11: `max_mib` per stream (§5), re-measure after a full week of market hours.*
2. Quote coalescing under backpressure acceptable for which consumers?
3. Typed job specs: `google.protobuf.Any` vs `kind` + bytes (current) — ergonomics vs safety.
4. One work-queue stream per priority class or NATS consumer priority groups?
5. ~~Subject tokenization for instruments (UUID vs short ids) — readability in UI vs safety.~~ Decided by D-017: dashless lowercase UUID (§4.1).

## 11. Deferred
- NATS clustering / leaf nodes for multi-machine scaling.
- Schema registry beyond git + buf.
