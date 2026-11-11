# fa-bus

NATS JetStream client: subject taxonomy, typed publish/subscribe, work-queue job/result envelopes, and the JetStream stream table.

**Status:** implemented — subject taxonomy + typed pub/sub (tested subjects); spec 21 §5 streams provisioned idempotently and listed for `GET /api/bus/streams` (verified against the compose NATS, 2026-11-10).

## Streams (`streams` module)
The spec 21 §5 table lives in [`streams.json`](streams.json) (`MD` 24 h / 4 GiB, `TEXT`/`MEDIA` 7 d / 2 GiB, `LATENT` 7 d / 24 GiB, `MACRO`/`JOB_RESULTS` 30 d / 2 GiB, `SIGNAL` 30 d / 16 GiB, `ORDERS` 365 d / 16 GiB, `DLQ` 30 d / 1 GiB; `max_mib` per stream since D-075; all limits retention, file storage) and is embedded here (`streams::STREAMS_JSON`) and as a verbatim copy in `go/pkg/bus/streams.json`; a unit test fails if the copy drifts, another verifies no two streams' subject filters overlap (including the `JOBS_EVO` work queue on `jobs.evo.>`, which NATS would reject).

- `ensure_streams(&js)` — get-or-create each stream, then update it when subjects, max age, max bytes, storage or description differ; idempotent, returns the names it changed. A byte cap below a stream's current usage is not applied (it would purge): the old cap stays and a warning names the stream. Likewise a shorter `max_age` (or a finite one on an unlimited stream) is never applied (`safe_max_age`, D-082): shortening retention purges everything older at once, so it is an operator action; growing or removing a limit is applied. control-api calls it once NATS is connected; the Go ingestor calls the mirror at startup.
- `ensure_work_queue(&js, kind, durable)` — `JOBS_<KIND>` work-queue stream on `jobs.<kind>.>` plus a durable pull consumer with `ack_wait` 60 s / `max_deliver` 5 / `max_ack_pending` 64 (spec 21 §7; jobs in flight across all workers of a kind — each worker pulls one at a time). Used by the trainer (`train`) and model-runtime (`mcmc`, D-075); evo-worker and evo-orchestrator still own `JOBS_EVO` with their own code.
- `list_streams(&js)` — every stream with subjects, retention, messages, bytes, first/last seq and per-consumer `ack_pending` / `num_pending` / `redelivered` (the `/api/bus/streams` payload).
- `subjects_overlap(a, b)` — NATS wildcard overlap predicate used by the tests.

**Publishers do not change.** JetStream captures every core NATS publish whose subject matches a stream, so `Bus::publish` and the Go `Publisher` keep publishing plain messages; the live test publishes on `md.trade.test.<id>` and checks `MD`'s message count grows. Subjects outside the table (`ctl.*`, `svc.heartbeat.*`, `ingest.*`, `ref.changed.*`, `catalog.changed.*`, `ops.*`, `jobs.progress.*`) stay core-only on purpose.

Live test against the compose NATS: `FA_NATS_TEST=1 cargo test -p fa-bus -- --nocapture live` (also exercises drift repair and the work-queue helper on a throwaway `JOBS_FABUSTEST` stream it deletes afterwards).

## Job loop (`jobs` module, spec 21 §4.3/§6, criteria 2–3)
`run_worker(consumer, js, WorkerConfig, handler)` is the horizontal-scaling contract in code: every worker of a kind shares one durable pull consumer and pulls **one job at a time**; per job it checks the contract version, decodes (undecodable → `Term` → dead letter), runs the handler with `in-progress` acks every `progress_every`, stores the `JobResult` through JetStream with `Nats-Msg-Id = result_msg_id(r)` (`result:<job_id>:<status>:<checkpoint_ref>`) and waits for the stream ack, then double-acks the job. A worker that dies before the result is stored leaves the job for redelivery after `ack_wait`; one that dies after storing it but before the ack makes the next worker publish an identical result, which the stream drops as a duplicate within its dedup window (2 min default) — beyond it, result consumers are idempotent by `job_id` (`ops.job_run`). `publish_result(js, subject, r)` → `duplicate?` is usable on its own (evo-worker still core-publishes; switching it is a one-line change). Live tests on test-scoped streams (`fabustest.*`, deleted afterwards, never `jobs.result.>`): `live_exactly_once_across_worker_crashes` (20 jobs; one worker dies after storing a result, one mid-job; two healthy workers drain the queue; every job ends with exactly one stored, correct result) and `live_two_workers_double_throughput` (40 CPU-bound ~60 ms jobs drained by one worker, then two: speed-up ≥ 1.7 required).

## Live consumers (`consume` module, spec 21 §5–§7, Q2)
`LiveConsumer::ensure(js, LiveSpec)` creates a durable, explicit-ack, deliver-new pull consumer with `max_ack_pending` (`LiveSpec::market_data`: 10 000, batches of 500): a slow consumer stops delivery instead of buffering, the backlog stays in the stream and survives restarts. `next_batch(wait)` coalesces when more than `behind_threshold` (1 000) messages wait behind the batch: `md.quote.*` / `md.book.*` subjects keep only their latest message per subject (superseded ones are acked unprocessed, `stats.coalesced`); every other subject is delivered in full — `CoalescePolicy::new` refuses trades, bars, latents, signals, jobs, `order.*` and `risk.*`. Live test `live_backpressure_and_coalescing` (scratch stream: 2 000 quotes on 5 subjects + 50 trades, `max_ack_pending` 100: never more than 100 unacked, every trade in order, the final quote of every subject, most intermediate quotes skipped). Go mirror: `go/pkg/bus/consume.go`.

## Contract version (`contract` module, spec 21 §6/§8)
`fa_proto::PACKAGE = "fa.v1"`, `MAJOR = 1`. A message without the `Fa-Schema` header is v1 (so v1 producers add no bytes); a future `fa.v2` producer sends `Fa-Schema: fa.v2`. `contract::check(headers)` rejects majors outside `SUPPORTED_MAJORS` and malformed values; `Bus::subscribe`, `Bus::subscribe_traced` and `jobs::run_worker` drop such messages (`accept`: warn + `fa_bus_rejected_schema_total{major}`) instead of mis-decoding them. Raw subscribers (`subscribe_raw`, hand-rolled JetStream consumers) call `contract::accept` themselves. Go mirror: `bus.CheckSchema` / `bus.Decode`. Contract CI: `scripts/buf-check.sh` (`just proto-check`) runs `buf lint` and `buf breaking` against `main`.

## Dead letters (`dlq` module, D-075)
`dlq::spawn_forwarder(client)` (run by control-api) listens to the JetStream `MAX_DELIVERIES` and `MSG_TERMINATED` advisories, fetches the original message by sequence and republishes it on `dlq.<STREAM>.<subject>` (captured by `DLQ`) with `Fa-Dlq-*` headers and `Nats-Msg-Id = dlq:<stream>:<seq>`; work-queue originals are then deleted, limits streams keep theirs. Each dead letter increments `fa_bus_dead_letters_total{stream,consumer,reason}` and raises a `DeadLetter` alert on `ops.alert` at most once per (stream, consumer) and minute; `DLQ`'s own advisories are ignored.

## Ingest frames (`ingest` module, D-067)
`ingest.health|gap|unresolved.<source>` and `ref.changed.*`/`catalog.changed.*` are `fa.v1` protos (`IngestHealth`, `IngestGap`, `IngestUnresolved`, `RefChanged`). Consumers accept both encodings (`is_json`: JSON starts with `{`), and `frame_json` turns either into the D-054 JSON shape control-api serves; `ref_changed_notice` builds a change notice.

## Metrics (D-056)
`fa_bus::metrics` re-exports [`fa-metrics`](../fa-metrics/README.md) so a service's whole observability hook is one line, `fa_bus::metrics::serve_from_env("<service>", env!("CARGO_PKG_VERSION"))`. `Bus::publish` mirrors every `svc.heartbeat.*` message's counters into `fa_heartbeat_counter{service,key}` before sending it, so heartbeat counters are scrapeable without a second instrumentation. `streams::spawn_consumer_lag(consumer)` (called by `ensure_work_queue`, and by evo-worker for its hand-rolled `JOBS_EVO` consumer) polls `consumer.info()` every 10 s into `fa_bus_consumer_num_pending|ack_pending|redelivered|waiting{stream,consumer}`.

## Replay reader (`replay` module, D-059)
Reads a JetStream stream **by time range** so a backtest can rebuild the latent graph and decoder outputs the live strategies saw (spec 41 §3/§6.1):

- `read_range(&js, stream, filter_subject, start_ms, end_ms, &ReadOptions)` — ordered ephemeral pull consumer with `deliver_policy = by_start_time(start)` and one filter subject, drained until the first message published after `end`, until the server reports nothing pending, or after `idle` (default 2 s) without a message; returns `RecordedMessage { subject, stream_seq, published_ms, payload }` oldest first. `published_ms` is the server receive time — the message's **knowledge time**, which point-in-time replays filter on. A range after the stream's last message returns at once (stream info is checked first); `max_messages` (default 500 000) caps a read.
- `read_latents(&js, start_ms, end_ms, &opts)` — the four `LATENT_FILTERS` (`latent.state.>`, `latent.region.>`, `latent.transition.>`, `decoder.output.>`) of the `LATENT` stream merged in `(published_ms, stream_seq)` order (`sort_records`).
- `LatentMessage::decode(subject, payload)` → `State | Region | Transition | DecoderOutput` with `snapshot_id()`, `timescale()`, `as_of_ms()`; `Recorded::decode` / `decode_all` for whole slices.
- `to_jsonl` / `from_jsonl` — one JSON object per line with a hex payload, the fixture format `crates/fa-backtest/testdata/latent_fixture.jsonl` uses.

Live test: `FA_NATS_TEST=1 cargo test -p fa-bus -- --nocapture live_read` publishes three batches into a scratch `FABUSREPLAY` stream with gaps between them, reads the middle batch back by time, checks filter subjects, caps and the empty-range fast path, then deletes the stream. Inspect a stream from the shell with the example: `cargo run -p fa-bus --example read_range -- LATENT 'latent.state.1s.>' -10m now --limit 20` (times: RFC 3339, epoch ms, `now`, `-<n>h|m|s`; `--json` decodes `jobs.result.>` payloads).

## Owns
Subject constants, typed pub/sub, job pull loop (`jobs`), live consumers with backpressure / coalescing (`consume`), contract-version check (`contract`), stream table + provisioning, dead-letter forwarder, ingest-frame codecs, time-range stream reader (`replay`), metrics re-export + heartbeat/consumer-lag bridges.

## Governing specs
- [21-bus-and-contracts](../../specs/21-bus-and-contracts.md)
- [Ops reference (ex spec 60)](../../docker/README.md#reference-spec-60-ops-retired-2026-11-11)
