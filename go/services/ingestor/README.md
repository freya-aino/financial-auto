# ingestor

Runs N connectors concurrently, normalizes to canonical events, archives raw payloads, publishes to NATS, writes trades to VictoriaMetrics.

**Status:** implemented (v1) — connectors are supervised and can be started/stopped at runtime over `ctl.ingest.*`; envelopes are routed and published; trades are batched into VictoriaMetrics; per-source limiter; heartbeat counters; graceful shutdown. Raw archive and JetStream streams pending.

## Implemented
- **Supervisor** (`supervisor.go`): starts a connector by source id when it is registered (`go/connectors/*`) and its catalog status is not `disabled`/`planned` (the synthetic `heartbeat` is allowed without a catalog entry). Stops cancel the connector's context and wait for it to exit.
- **Control** (request/reply with `CtlRequest`/`CtlReply`, spec 10 §4.3): `ctl.ingest.start` and `ctl.ingest.stop` with arg `source`, `ctl.ingest.status` (reply message = comma list of running sources). Error messages are prefixed `not_found:` (unknown source / no connector), `conflict:` (already running, not running, blocked status) or `invalid:`; control-api maps them to 404/409/400. The spec's per-source subject `ctl.ingest.<source_id>.<verb>` is replaced by the verb-only subject plus a `source` arg so it fits `bus.Ctl(service, verb)`.
- **TSDB writer** (`go/pkg/tsdb`): every published `Trade` becomes `fa_trade_price{inst,venue,src}` and `fa_trade_size{...}` samples at `event_time` (spec 20 §5.3), batched (500 points or 1 s) into `POST <tsdb-url>/api/v1/import`. Buffer is bounded (50 000 points) and drops when VictoriaMetrics is unreachable.
- **Heartbeat counters:** `connectors.running`, `connectors.up.<id>` (0/1 for every registered connector), `events.<id>`, `errors.<id>`, `unresolved.<id>`, `tsdb.written`, `tsdb.errors`, `tsdb.dropped`.

## Run
```sh
# host, after `just go-build` (binaries land in go/bin)
go/bin/ingestor -nats nats://127.0.0.1:4222 -catalog config/sources.yaml -connectors heartbeat -tsdb-url http://127.0.0.1:8428
```
| Flag | Env | Default | |
|---|---|---|---|
| `-nats` | `FA_NATS_URL` | `nats://127.0.0.1:4222` | |
| `-catalog` | `FA_CATALOG` | `config/sources.yaml` | |
| `-connectors` | `FA_CONNECTORS` | `heartbeat` | comma list started at boot; `""` starts nothing (use `ctl.ingest.start`) |
| `-tsdb-url` | `FA_TSDB_URL` | `http://127.0.0.1:8428` | VictoriaMetrics base URL; `""` disables the writer. In compose use `http://victoriametrics:8428` |
| `-heartbeat` | | `5s` | |

Start/stop from the control API: `curl -X POST http://127.0.0.1:7700/api/ingest/heartbeat/stop`. Check the series: `curl 'http://127.0.0.1:8428/api/v1/query?query=fa_trade_price'` (VictoriaMetrics hides the last 30 s by default, `-search.latencyOffset`).

## Owns
Connector supervisor process.

## Governing specs
- [10-ingestion](../../../specs/10-ingestion.md)
- [20-storage](../../../specs/20-storage.md) §5.3
- [21-bus-and-contracts](../../../specs/21-bus-and-contracts.md)
