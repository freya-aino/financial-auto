# syntax=docker/dockerfile:1.7
# Go services image (D-026): ingestor, broker-gateway, evo-orchestrator, and healthprobe (the
# compose healthcheck of these distroless containers, D-066).
# Build context = repo root.

FROM golang:1.25-alpine AS builder
WORKDIR /src/go
ENV CGO_ENABLED=0 GOFLAGS=-buildvcs=false
COPY go/go.mod go/go.sum ./
RUN --mount=type=cache,id=fa-gomod,target=/go/pkg/mod go mod download
COPY go/ ./
RUN --mount=type=cache,id=fa-gomod,target=/go/pkg/mod \
    --mount=type=cache,id=fa-gocache,target=/root/.cache/go-build \
    go build -trimpath -ldflags="-s -w" -o /out/ ./services/...
# writable state dir for the nonroot user (broker-gateway's calibrated costs, named volume)
RUN mkdir -p /state

FROM gcr.io/distroless/static-debian12:nonroot
WORKDIR /app
COPY --from=builder /out/ /usr/local/bin/
COPY config/sources.yaml config/sources.yaml
COPY go/services/broker-gateway/costs.yaml config/costs.yaml
COPY --from=builder --chown=65532:65532 /state /var/lib/fa
ENTRYPOINT ["ingestor"]
