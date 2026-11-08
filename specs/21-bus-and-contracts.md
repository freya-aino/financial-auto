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
| `decoder.output.<family>` | `DecoderOutput` (refs) | evo/runtime → UI, strategies |
| `signal.exposure.<strategy>` | `ExposureDistribution` | strategies → allocator |
| `order.intent.<account>` | `OrderIntent` | allocator → risk |
| `order.approved.<account>` | `OrderIntent` | risk → broker-gateway |
| `order.event.<account>` | `OrderEvent` (ack/fill/cancel/reject) | broker-gateway → ledger, risk |
| `risk.state` / `risk.halt.<scope>` | `RiskState` / `Halt` | risk → all |
| `jobs.<kind>` | `Job` | orchestrator/trainer → workers |
| `jobs.result.<kind>` | `JobResult` | workers → orchestrator |
| `ingest.health.<source>` / `ingest.gap.*` / `ingest.unresolved.*` | health events | ingestor → risk, UI |
| `ctl.<service>.<verb>` | request/reply | control-api ↔ services |
| `ref.changed.*`, `catalog.changed.*` | change notices | |
| `svc.heartbeat.<service>` | `Heartbeat` | all → UI |

Tokens (`<inst>`) use the instrument UUID without dashes **(proposal)** to keep subjects safe.

### 4.2 Proto packages
```
proto/fa/v1/
  envelope.proto     # Envelope, times, raw_ref
  market.proto       # Trade, Quote, Bar, BookSnapshot
  text.proto         # NewsItem, FilingEvent
  media.proto        # MediaChunk
  macro.proto        # MacroObservation, OnchainEvent
  latent.proto       # LatentState, RegionEvent, TransitionUpdate, ExpertSpaceRef
  decoder.proto      # DecoderOutput
  strategy.proto     # ExposureDistribution
  order.proto        # OrderIntent, OrderEvent
  risk.proto         # RiskState, Halt
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
  enum Status { OK = 0; FAILED = 1; PREEMPTED = 2; BUDGET_EXCEEDED = 3; }
  Status status = 2;
  bytes  result = 3;            // kind-specific (e.g. FitnessRecord)
  repeated string output_refs = 4;
  string checkpoint_ref = 5;
  uint64 used_gpu_ms = 6;
  string worker_id = 7;
  string error = 8;
}
```
`JobProgress` published on `jobs.progress.<job_id>` for UI.

## 5. Data model — JetStream streams **(proposal)**
| Stream | Subjects | Retention | Max age | Notes |
|---|---|---|---|---|
| `MD` | `md.>` | limits | 24 h | live + short replay; long history in VM/Garage |
| `TEXT` | `news.>`, `filing.>` | limits | 7 d | |
| `MEDIA` | `media.>` | limits | 7 d | refs only, blobs in Garage |
| `MACRO` | `macro.>`, `onchain.>` | limits | 30 d | |
| `LATENT` | `latent.>`, `decoder.>` | limits | 7 d | |
| `SIGNAL` | `signal.>` | limits | 30 d | |
| `ORDERS` | `order.>`, `risk.>` | limits | 365 d, file storage, replicas when clustered | ledger is source of truth, stream is audit aid |
| `JOBS_<kind>` | `jobs.<kind>` | **work-queue** | until acked | one stream per kind/priority class |
| `JOB_RESULTS` | `jobs.result.>` | limits | 30 d | |

Consumers: durable pull consumers per service; `AckExplicit`; `MaxAckPending` set per consumer for backpressure; work queues use `AckWait` + heartbeat `in-progress` acks for long jobs.

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
| Poison message | moved to `DLQ.<stream>` after N redeliveries, alert |
| Proto incompatibility | `buf breaking` blocks merge; runtime rejects unknown major version |
| Worker crash mid-job | ack timeout → redelivery; resume from last checkpoint |

## 9. Acceptance criteria
- Rust and Go both build from the same `proto/` with `buf generate`; `buf lint` and `buf breaking` pass.
- A job published to `jobs.<kind>` is processed exactly once in effect (idempotent result) across worker crashes in a test.
- Adding a second evo-worker doubles throughput on a CPU-bound fake job (scaling contract works).
- Every subject in §4.1 has a proto payload type.

## 10. Open questions
1. Retention per stream — confirm §5 values against disk budget (blocks 60).
2. Quote coalescing under backpressure acceptable for which consumers?
3. Typed job specs: `google.protobuf.Any` vs `kind` + bytes (current) — ergonomics vs safety.
4. One work-queue stream per priority class or NATS consumer priority groups?
5. Subject tokenization for instruments (UUID vs short ids) — readability in UI vs safety.

## 11. Deferred
- NATS clustering / leaf nodes for multi-machine scaling.
- Schema registry beyond git + buf.
