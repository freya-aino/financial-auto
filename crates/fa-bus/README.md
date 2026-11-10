# fa-bus

NATS JetStream client: subject taxonomy, typed publish/subscribe, work-queue job/result envelopes, and the JetStream stream table.

**Status:** implemented — subject taxonomy + typed pub/sub (tested subjects); spec 21 §5 streams provisioned idempotently and listed for `GET /api/bus/streams` (verified against the compose NATS, 2026-11-10).

## Streams (`streams` module)
The spec 21 §5 table lives in [`streams.json`](streams.json) (`MD` 24 h, `TEXT`/`MEDIA`/`LATENT` 7 d, `MACRO`/`SIGNAL`/`JOB_RESULTS` 30 d, `ORDERS` 365 d; all limits retention, file storage) and is embedded here (`streams::STREAMS_JSON`) and as a verbatim copy in `go/pkg/bus/streams.json`; a unit test fails if the copy drifts, another verifies no two streams' subject filters overlap (including the `JOBS_EVO` work queue on `jobs.evo.>`, which NATS would reject).

- `ensure_streams(&js)` — get-or-create each stream, then update it when subjects, max age, storage or description differ; idempotent, returns the names it changed. control-api calls it once NATS is connected; the Go ingestor calls the mirror at startup.
- `ensure_work_queue(&js, kind, durable)` — `JOBS_<KIND>` work-queue stream on `jobs.<kind>.>` plus a durable pull consumer with `ack_wait` 60 s / `max_deliver` 5 (spec 21 §7). The evo-worker and evo-orchestrator still own `JOBS_EVO` with their own code; this helper is for them to adopt.
- `list_streams(&js)` — every stream with subjects, retention, messages, bytes, first/last seq and per-consumer `ack_pending` / `num_pending` / `redelivered` (the `/api/bus/streams` payload).
- `subjects_overlap(a, b)` — NATS wildcard overlap predicate used by the tests.

**Publishers do not change.** JetStream captures every core NATS publish whose subject matches a stream, so `Bus::publish` and the Go `Publisher` keep publishing plain messages; the live test publishes on `md.trade.test.<id>` and checks `MD`'s message count grows. Subjects outside the table (`ctl.*`, `svc.heartbeat.*`, `ingest.*`, `jobs.progress.*`) stay core-only on purpose.

Live test against the compose NATS: `FA_NATS_TEST=1 cargo test -p fa-bus -- --nocapture live` (also exercises drift repair and the work-queue helper on a throwaway `JOBS_FABUSTEST` stream it deletes afterwards).

## Owns
Subject constants, typed pub/sub, job pull loop, stream table + provisioning.

## Governing specs
- [21-bus-and-contracts](../../specs/21-bus-and-contracts.md)
- [60-ops](../../specs/60-ops.md)
