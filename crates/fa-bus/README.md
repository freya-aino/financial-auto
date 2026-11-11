# fa-bus

NATS JetStream client: subject taxonomy, typed publish/subscribe, work-queue job/result envelopes, and the JetStream stream table.

**Status:** implemented — subject taxonomy + typed pub/sub (tested subjects); spec 21 §5 streams provisioned idempotently and listed for `GET /api/bus/streams` (verified against the compose NATS, 2026-11-10).

## Streams (`streams` module)
The spec 21 §5 table lives in [`streams.json`](streams.json) (`MD` 24 h / 4 GiB, `TEXT`/`MEDIA` 7 d / 2 GiB, `LATENT` 7 d / 24 GiB, `MACRO`/`JOB_RESULTS` 30 d / 2 GiB, `SIGNAL` 30 d / 16 GiB, `ORDERS` 365 d / 16 GiB, `DLQ` 30 d / 1 GiB; `max_mib` per stream since D-075; all limits retention, file storage) and is embedded here (`streams::STREAMS_JSON`) and as a verbatim copy in `go/pkg/bus/streams.json`; a unit test fails if the copy drifts, another verifies no two streams' subject filters overlap (including the `JOBS_EVO` work queue on `jobs.evo.>`, which NATS would reject).

- `ensure_streams(&js)` — get-or-create each stream, then update it when subjects, max age, max bytes, storage or description differ; idempotent, returns the names it changed. A byte cap below a stream's current usage is not applied (it would purge): the old cap stays and a warning names the stream. control-api calls it once NATS is connected; the Go ingestor calls the mirror at startup.
- `ensure_work_queue(&js, kind, durable)` — `JOBS_<KIND>` work-queue stream on `jobs.<kind>.>` plus a durable pull consumer with `ack_wait` 60 s / `max_deliver` 5 (spec 21 §7). Used by the trainer (`train`) and model-runtime (`mcmc`, D-075); evo-worker and evo-orchestrator still own `JOBS_EVO` with their own code.
- `list_streams(&js)` — every stream with subjects, retention, messages, bytes, first/last seq and per-consumer `ack_pending` / `num_pending` / `redelivered` (the `/api/bus/streams` payload).
- `subjects_overlap(a, b)` — NATS wildcard overlap predicate used by the tests.

**Publishers do not change.** JetStream captures every core NATS publish whose subject matches a stream, so `Bus::publish` and the Go `Publisher` keep publishing plain messages; the live test publishes on `md.trade.test.<id>` and checks `MD`'s message count grows. Subjects outside the table (`ctl.*`, `svc.heartbeat.*`, `ingest.*`, `ref.changed.*`, `catalog.changed.*`, `ops.*`, `jobs.progress.*`) stay core-only on purpose.

Live test against the compose NATS: `FA_NATS_TEST=1 cargo test -p fa-bus -- --nocapture live` (also exercises drift repair and the work-queue helper on a throwaway `JOBS_FABUSTEST` stream it deletes afterwards).

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
Subject constants, typed pub/sub, job pull loop, stream table + provisioning, dead-letter forwarder, ingest-frame codecs, time-range stream reader (`replay`), metrics re-export + heartbeat/consumer-lag bridges.

## Governing specs
- [21-bus-and-contracts](../../specs/21-bus-and-contracts.md)
- [60-ops](../../specs/60-ops.md)
