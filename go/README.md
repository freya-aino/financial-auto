# go

Single Go module `github.com/fey/financial-auto/go` (D-016): the connector framework, connectors, `ingestor`, `broker-gateway` and `evo-orchestrator`.

Requires Go 1.25. Dependencies are pinned to releases that still support Go 1.25; newer nats.go and x/time releases require Go 1.26.

```sh
# without a local Go toolchain (rootless Docker):
docker run --rm -v "$PWD/..":/repo -w /repo/go -e GOFLAGS=-buildvcs=false \
  -v fa-gomod:/go/pkg/mod -v fa-gocache:/root/.cache/go-build \
  golang:1.25-alpine sh -c 'go build ./... && go vet ./... && go test ./...'

# run (from repo root, NATS on localhost):
go -C go run ./services/ingestor -catalog config/sources.yaml
```
