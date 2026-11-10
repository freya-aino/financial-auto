# ingestor

Runs N connectors concurrently, archives raw payloads to Garage, normalizes to canonical events, publishes to NATS, writes market data to VictoriaMetrics, keeps reference data and run bookkeeping in Postgres.

**Status:** implemented (v1) — connectors are supervised and can be started/stopped at runtime over `ctl.ingest.*`; envelopes are routed and published; trades, quotes, bars and books are batched into VictoriaMetrics; raw payloads go to Garage (`-garage`); symbols resolve through Postgres with provisioning of unknown ones, cursors and `ingest.connector_run` rows (`-postgres`); secrets from `config/secrets.toml` / env (`-secrets`, D-020); per-source limiter; heartbeat counters; graceful shutdown. Live connectors: `heartbeat`, `yahoo`, `hackernews`, `google_news`; keyed: `lemon`, `oanda`.

## Implemented
- **Supervisor** (`supervisor.go`): starts a connector by source id when it is registered (`go/connectors/*`) and its catalog status is not `disabled`/`planned` (the synthetic `heartbeat` is allowed without a catalog entry). Stops cancel the connector's context and wait for it to exit.
- **Control** (request/reply with `CtlRequest`/`CtlReply`, spec 10 §4.3): `ctl.ingest.start` and `ctl.ingest.stop` with arg `source`, `ctl.ingest.status` (reply message = comma list of running sources). Error messages are prefixed `not_found:` (unknown source / no connector), `conflict:` (already running, not running, blocked status) or `invalid:`; control-api maps them to 404/409/400. The spec's per-source subject `ctl.ingest.<source_id>.<verb>` is replaced by the verb-only subject plus a `source` arg so it fits `bus.Ctl(service, verb)`.
- **TSDB writer** (`go/pkg/tsdb`): every published market envelope becomes samples at `event_time` (spec 20 §5.3): `fa_trade_{price,size}`, `fa_quote_{bid,ask}_{px,sz}`, `fa_bar_{open,high,low,close,volume,vwap}{tf}`, `fa_book_*{lvl}`, labels `inst,venue,src`; batched (500 points or 1 s) into `POST <tsdb-url>/api/v1/import`. Buffer is bounded (50 000 points) and drops when VictoriaMetrics is unreachable.
- **Raw archive** (`go/pkg/archive`): with `-garage`, every `RawMessage` is appended to a per-source, per-minute gzip object `raw/<source>/<yyyy>/<mm>/<dd>/<hh>/<ulid>.raw.gz` in bucket `fa-raw` and the envelopes carry `raw_ref{key, offset, length}` (offsets into the decompressed object). Credentials: `[garage]` in `config/secrets.toml` or `FA_GARAGE_ACCESS_KEY_ID` / `FA_GARAGE_SECRET_ACCESS_KEY`. Without `-garage`: `NopArchive`.
- **Reference data** (`go/pkg/refdata`): with `-postgres`, the catalog is synced into `catalog.source`, symbols resolve through `ref.listing`/`ref.identifier`, unknown `(source, symbol)` pairs are provisioned as `status='provisional'` instruments with a deterministic UUIDv5 plus an open `ingest.quarantine` row, every connector start/stop writes `ingest.connector_run`, and connectors persist progress in `ingest.backfill_cursor`. Without `-postgres`: in-memory `StaticResolver` (empty, so market connectors report everything unresolved) and in-memory cursors.
- **Secrets** (`go/pkg/connector`): `-secrets` file (`[sources.<id>]`), `FA_<SOURCE>_<KEY>` env overrides; passed to connectors as `SourceConfig.Secrets`.
- **Heartbeat counters:** `connectors.running`, `connectors.up.<id>` (0/1 for every registered connector), `events.<id>`, `errors.<id>`, `unresolved.<id>`, `tsdb.written|errors|dropped`, `archive.messages|objects|errors`, `refdata.errors`.

## Run
```sh
# host, after `just go-build` (binaries land in go/bin)
go/bin/ingestor -nats nats://127.0.0.1:4222 -catalog config/sources.yaml -connectors heartbeat,yahoo,hackernews,google_news \
  -tsdb-url http://127.0.0.1:8428 -postgres postgres://fa:fa@127.0.0.1:5432/fa -garage http://127.0.0.1:3900 -secrets config/secrets.toml
```
| Flag | Env | Default | |
|---|---|---|---|
| `-nats` | `FA_NATS_URL` | `nats://127.0.0.1:4222` | |
| `-catalog` | `FA_CATALOG` | `config/sources.yaml` | |
| `-connectors` | `FA_CONNECTORS` | `heartbeat` | comma list started at boot; `""` starts nothing (use `ctl.ingest.start`) |
| `-tsdb-url` | `FA_TSDB_URL` | `http://127.0.0.1:8428` | VictoriaMetrics base URL; `""` disables the writer. In compose use `http://victoriametrics:8428` |
| `-postgres` | `FA_POSTGRES_URL` | `""` | Postgres URL; enables the `refdata` resolver, cursors and run log. Compose: `postgres://fa:fa@postgres:5432/fa` |
| `-garage` | `FA_GARAGE_URL` | `""` | Garage S3 endpoint; enables the raw archive. Compose: `http://garage:3900` |
| `-secrets` | `FA_SECRETS` | `config/secrets.toml` | secrets file; missing file is fine when env vars carry the keys |
| `-heartbeat` | | `5s` | |

Start/stop from the control API: `curl -X POST http://127.0.0.1:7700/api/ingest/heartbeat/stop`. Check the series: `curl 'http://127.0.0.1:8428/api/v1/series?match[]=fa_bar_close'` (VictoriaMetrics hides the last 30 s by default, `-search.latencyOffset`). Provisioned instruments: `select * from ingest.quarantine`; runs: `select * from ingest.connector_run order by started_at desc`.

## Owns
Connector supervisor process.

## Governing specs
- [10-ingestion](../../../specs/10-ingestion.md)
- [20-storage](../../../specs/20-storage.md) §5.3
- [21-bus-and-contracts](../../../specs/21-bus-and-contracts.md)
