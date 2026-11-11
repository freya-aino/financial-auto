# healthprobe

Compose healthcheck of the Go service images (spec 60 §6.4/§8, D-066). The images are distroless (no shell, curl or wget), so compose runs this binary inside the ingestor, broker-gateway and evo-orchestrator containers:

```yaml
healthcheck: { test: ["CMD", "healthprobe", "http://127.0.0.1:9100/metrics"] }
```

**Status:** implemented — GETs each URL argument (default `http://127.0.0.1:9100/metrics`) and exits 0 only when every one answers 2xx within `-timeout` (default 3 s); otherwise prints `unhealthy: …` and exits 1. Answering `/metrics` proves the process is up and its HTTP loop is not hung. Built into the shared Go image with the other services (`go build ./services/...`).

## Tests
`main_test.go`: 2xx passes; a 503 among several URLs, a timeout and a refused connection fail.

## Governing specs
- [Ops reference (ex spec 60)](../../../docker/README.md#reference-spec-60-ops-retired-2026-11-11)
