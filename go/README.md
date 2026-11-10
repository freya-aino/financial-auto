# go

Single Go module `github.com/fey/financial-auto/go` (D-016): the connector framework (`pkg/connector`, `pkg/catalog`), bus helpers (`pkg/bus`), the VictoriaMetrics writer (`pkg/tsdb`), the Garage raw archive (`pkg/archive`, aws-sdk-go-v2 s3), Postgres reference data / ingestion bookkeeping (`pkg/refdata`, pgx v5), connectors (`connectors/*`: heartbeat, yahoo, hackernews, googlenews live; lemon, oanda keyed), `ingestor`, `broker-gateway` and `evo-orchestrator`.

Requires Go 1.25. Dependencies are pinned to releases that still support Go 1.25; newer nats.go and x/time releases require Go 1.26 (run `go get` with `GOTOOLCHAIN=local` so an upgrade that needs a newer toolchain fails instead of downloading one).

Ingestor flags beyond NATS/catalog: `-tsdb-url`, `-postgres` (`FA_POSTGRES_URL`), `-garage` (`FA_GARAGE_URL`), `-secrets` (`FA_SECRETS`, default `config/secrets.toml`); see `services/ingestor/README.md`. Live-infrastructure tests are gated by `FA_POSTGRES_URL` and `FA_GARAGE_URL` + `FA_GARAGE_ACCESS_KEY_ID` + `FA_GARAGE_SECRET_ACCESS_KEY` (run the container with `--network host`).

```sh
# without a local Go toolchain (rootless Docker):
docker run --rm -v "$PWD/..":/repo -w /repo/go -e GOFLAGS=-buildvcs=false \
  -v fa-gomod:/go/pkg/mod -v fa-gocache:/root/.cache/go-build \
  golang:1.25-alpine sh -c 'go build ./... && go vet ./... && go test ./...'

# run (from repo root, NATS on localhost):
go -C go run ./services/ingestor -catalog config/sources.yaml
```
