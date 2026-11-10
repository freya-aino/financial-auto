# bus

NATS publish helpers and Go protobuf types (generated into `pb/`).

**Status:** implemented — subject taxonomy mirroring `crates/fa-bus/src/subjects.rs`, envelope→subject routing, protobuf `Publisher` (with `Nats-Msg-Id` for jobs), heartbeat loop + counters. Generated code in `pb/` (buf, committed; D-002).

## Streams
`streams.go` mirrors `crates/fa-bus/src/streams.rs`: `EnsureStreams(ctx, js)` / `EnsureStreamsConn(ctx, nc)` create or update the spec 21 §5 streams idempotently (the ingestor calls it at startup and only warns on failure), `EnsureWorkQueue(ctx, js, kind, durable)` provisions `JOBS_<KIND>` with the spec's `ack_wait` 60 s / `max_deliver` 5, `SubjectsOverlap` is the test predicate. The table is the embedded `streams.json`, a verbatim copy of `crates/fa-bus/streams.json` (Go cannot embed across the module root); `cargo test -p fa-bus` fails when the two differ, and `TestStreamsFixtureMatchesRust` checks the same when run from a full checkout. Live test: `FA_NATS_TEST=1 go test ./pkg/bus -run Live -v`. Publishers need no change: core publishes on captured subjects are persisted by JetStream.

## Owns
Subject helpers mirrored from `fa-bus`; JetStream stream table mirror.

## Governing specs
- [21-bus-and-contracts](../../../specs/21-bus-and-contracts.md)
