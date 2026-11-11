# go

Single Go module `github.com/fey/financial-auto/go` (D-016): the connector framework (`pkg/connector`, `pkg/catalog`), bus helpers, generated `fa.v1` types and metrics exporter (`pkg/bus`), the VictoriaMetrics writer (`pkg/tsdb`), the Garage raw archive and its replay reader (`pkg/archive`, aws-sdk-go-v2 s3), Postgres reference data / ingestion bookkeeping (`pkg/refdata`, pgx v5), the warehouse writer for macro observations, filings and XBRL facts (`pkg/warehouse`, `wh.*`), connectors (`connectors/*`: the generic RSS/Atom adapter `rss` and SDMX adapter `sdmx` (in `ecb`) for catalog entries with `adapter:`; the synthetic `heartbeat`; keyless and live: `yahoo`, `hackernews`, `googlenews`, `frankfurter`, `ecb`, `gdelt`, `secedgar`, `kraken`; keyed and never run live: `lemon`, `oanda`), and the services `ingestor`, `broker-gateway` (paper), `evo-orchestrator` and `healthprobe` (compose healthcheck of the distroless images).

Requires Go 1.25. Dependencies are pinned to releases that still support Go 1.25; newer nats.go and x/time releases require Go 1.26 (run `go get` with `GOTOOLCHAIN=local` so an upgrade that needs a newer toolchain fails instead of downloading one).

Ingestor flags beyond NATS/catalog: `-tsdb-url`, `-postgres` (`FA_POSTGRES_URL`), `-garage` (`FA_GARAGE_URL`), `-secrets` (`FA_SECRETS`, default `config/secrets.toml`), `-spool-dir` (`FA_ARCHIVE_SPOOL`), `-health-json` (`FA_INGEST_HEALTH_JSON=1`), `-backfill-lanes`, `-nats-spool-dir` (`FA_NATS_SPOOL`), `-catalog-watch`, `-auto-disable-after`, `-gap-fill`; see `services/ingestor/README.md` (which also holds the ingestion reference of the retired spec 10). The rate-limit soak harness: `FA_SOAK_DURATION=24h go test ./services/ingestor -run TestRateLimitSoak -timeout 25h` (local fake server). Every long-running service serves Prometheus text on `-metrics` (`FA_METRICS_BIND`, `off` disables; D-056). Live-infrastructure tests are gated by `FA_POSTGRES_URL` (`FA_TEST_POSTGRES_URL` for `pkg/warehouse`), `FA_GARAGE_URL` + `FA_GARAGE_ACCESS_KEY_ID` + `FA_GARAGE_SECRET_ACCESS_KEY`, `FA_NATS_TEST=1` and `FA_VM_TEST=1` (run the container with `--network host`).

```sh
# without a local Go toolchain (rootless Docker):
docker run --rm -v "$PWD/..":/repo -w /repo/go -e GOFLAGS=-buildvcs=false \
  -v fa-gomod:/go/pkg/mod -v fa-gocache:/root/.cache/go-build \
  golang:1.25-alpine sh -c 'go build ./... && go vet ./... && go test ./...'

# run (from repo root, NATS on localhost):
go -C go run ./services/ingestor -catalog config/sources.yaml
```
