#!/usr/bin/env bash
# VictoriaMetrics rebuild drill (spec 20 §6.6 / criterion 5): VM is not backed up, its market
# series are rebuilt from the Garage raw archive. This proves it WITHOUT touching live data:
#   1. a scratch VictoriaMetrics container (tmpfs storage, host port 18428, removed on exit)
#   2. go/services/vm-rebuild: archive.Replay (the ctl.ingest.replay reader) → the connector's
#      normalizer → tsdb.Writer.AddEnvelope (the ingestor's write path) → scratch VM.
#      Garage, Postgres (read-only session, lookup only) and the live VM are only read; nothing is
#      published on NATS.
#   3. per source and metric: matched / missing / extra / value-mismatched samples vs the live VM.
# Usage: scripts/vm-rebuild-drill.sh [--sources kraken,yahoo] [--from -2h] [--to -10m] [--json] [--keep]
#   --keep   leave the scratch container running (fa-vm-drill, http://127.0.0.1:18428)
# Env: FA_VM_DRILL_PORT (18428), FA_VM_DRILL_TAIL (5m: archive read past --to for late-received events), FA_TSDB_URL (live VM, default http://127.0.0.1:8428),
#      FA_GARAGE_URL (http://127.0.0.1:3900), FA_POSTGRES_URL (postgres://fa:fa@127.0.0.1:5432/fa).
set -euo pipefail
. "$(dirname "$0")/lib.sh"

SOURCES=kraken,yahoo; FROM=-2h; TO=-10m; JSON=""; KEEP=0
while [ $# -gt 0 ]; do
    case $1 in
        --sources) SOURCES=$2; shift 2 ;;
        --from) FROM=$2; shift 2 ;;
        --to) TO=$2; shift 2 ;;
        --json) JSON=-json; shift ;;
        --keep) KEEP=1; shift ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

PORT=${FA_VM_DRILL_PORT:-18428}
NAME=fa-vm-drill
VM_IMAGE=victoriametrics/victoria-metrics:v1.155.0 # same version as compose.yaml
GO_IMAGE=golang:1.25-alpine

docker rm -f "$NAME" >/dev/null 2>&1 || true
cleanup() { [ "$KEEP" = 1 ] || docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

log "starting scratch VictoriaMetrics $NAME on 127.0.0.1:$PORT (tmpfs)"
docker run -d --name "$NAME" -p "127.0.0.1:$PORT:8428" --tmpfs /storage "$VM_IMAGE" \
    -storageDataPath=/storage -retentionPeriod=100y -httpListenAddr=:8428 >/dev/null
for _ in $(seq 1 60); do
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 0.5
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null || die "scratch VM did not come up"

log "rebuilding $SOURCES for $FROM .. $TO from the raw archive"
docker run --rm --network host -v "$ROOT":/repo -w /repo/go -e GOFLAGS=-buildvcs=false \
    -v fa-gomod:/go/pkg/mod -v fa-gocache:/root/.cache/go-build "$GO_IMAGE" \
    go run ./services/vm-rebuild -tail "${FA_VM_DRILL_TAIL:-5m}" \
        -sources "$SOURCES" -from "$FROM" -to "$TO" \
        -target "http://127.0.0.1:$PORT" \
        -compare "${FA_TSDB_URL:-http://127.0.0.1:8428}" \
        -garage "${FA_GARAGE_URL:-http://127.0.0.1:3900}" \
        -postgres "${FA_POSTGRES_URL:-postgres://fa:fa@127.0.0.1:5432/fa}" \
        -secrets /repo/config/secrets.toml -catalog /repo/config/sources.yaml $JSON
[ "$KEEP" = 1 ] && log "scratch VM kept: http://127.0.0.1:$PORT (docker rm -f $NAME)"
log "VM REBUILD DRILL DONE"
