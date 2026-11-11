# bus

NATS publish helpers and Go protobuf types (generated into `pb/`).

**Status:** implemented — subject taxonomy mirroring `crates/fa-bus/src/subjects.rs` (market data, text, macro, filings incl. `filing.fact.<source>`, latent, orders, `risk.halt|resume|event`, `ingest.health|gap|unresolved`, `ref.changed`, `catalog.changed`, jobs, heartbeats, `dlq.<STREAM>.<subject>`, `ctl`), envelope→subject routing (`SubjectFor`, every payload incl. `FundamentalFact`), protobuf `Publisher` (`PublishWithID` sets `Nats-Msg-Id`: job ids, and the ingestor's stable event ids so JetStream drops duplicates), heartbeat loop + counters. Generated code in `pb/` (buf, committed; D-002), including `ingest.proto` (`IngestHealth`/`IngestGap`/`IngestUnresolved`/`RefChanged`) and `risk.proto` (`Halt`/`Resume`/`RiskEvent`) of D-067; `TestExpertViewRoundTripsFromRust` decodes a Rust-written `ExpertView` fixture (`testdata/expert_view.bin`).

## Streams
`streams.go` mirrors `crates/fa-bus/src/streams.rs`: `EnsureStreams(ctx, js)` / `EnsureStreamsConn(ctx, nc)` create or update the spec 21 §5 streams idempotently (the ingestor calls it at startup and only warns on failure), including the per-stream disk cap `max_mib` (D-075; a cap below a stream's current size is not applied, since JetStream would purge at once). A configured `max_age` shorter than the live one (or finite on an unlimited stream) is never applied either (`safeMaxAge`; shortening retention purges everything older at once, the D-075 MD incident): growing or removing the limit is applied, shortening it is an operator action. Held limits go to the optional `OnLimitHeld` hook (the ingestor logs a warning); `mergeStreamConfig` is the pure, unit-tested merge, `EnsureWorkQueue(ctx, js, kind, durable)` provisions `JOBS_<KIND>` with the spec's `ack_wait` 60 s / `max_deliver` 5, `SubjectsOverlap` is the test predicate. The table is the embedded `streams.json`, a verbatim copy of `crates/fa-bus/streams.json` (Go cannot embed across the module root); `cargo test -p fa-bus` fails when the two differ, and `TestStreamsFixtureMatchesRust` checks the same when run from a full checkout. Live test: `FA_NATS_TEST=1 go test ./pkg/bus -run Live -v`. Publishers need no change: core publishes on captured subjects are persisted by JetStream.

## Metrics (D-056)
`metrics.go` is a hand-rolled Prometheus text exporter (no client_golang): `ServeMetrics(ctx, bind, service, version, counters, log)` serves `GET /metrics` with `fa_heartbeat_counter{service,key}` straight from the heartbeat `Counters`, process gauges (`fa_process_uptime_seconds`, `fa_process_resident_memory_bytes` from `/proc/self/statm`, `fa_build_info`) and `fa_bus_consumer_num_pending|ack_pending|redelivered|waiting{stream,consumer}` for every consumer `EnsureWorkQueue` created (`WatchConsumerLag` polls `Info()` every 10 s). `MetricsBindDefault(service)` honours `FA_METRICS_BIND` (`off` disables) and falls back to the host ports in `docker/README.md`; each service exposes it as `-metrics`.

## Owns
Subject helpers mirrored from `fa-bus`; JetStream stream table mirror; metrics exporter.

## Governing specs
- [21-bus-and-contracts](../../../specs/21-bus-and-contracts.md)
