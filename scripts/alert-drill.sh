#!/usr/bin/env bash
# End-to-end alerting drill (spec 60 §6.5): proves rule → vmalert → control-api → NATS `ops.alert`
# → audit.log without disturbing any service.
#   1. subscribes to `ops.alert` (nats-box on the compose network)
#   2. writes fa_alert_drill{drill_id=<id>} = 1 into VictoriaMetrics (/api/v1/import/prometheus)
#      every 20 s, which the `AlertDrill` rule in docker/vm-alerts.yaml turns into a firing alert
#   3. waits for the firing ops.alert message with that drill_id, checks GET /api/alerts and the
#      `ops.alert.firing` audit row, then stops writing and (unless --no-resolve) waits for the
#      resolved message (max_over_time window 2 m + evaluation, ~3 min)
# Usage: scripts/alert-drill.sh [--timeout SECONDS] [--no-resolve]
# Exit 0 = every step seen. Needs the stack up with the vmalert service (docker compose up -d vmalert).
set -euo pipefail
. "$(dirname "$0")/lib.sh"

TIMEOUT=180; RESOLVE=1
while [ $# -gt 0 ]; do
    case $1 in
        --timeout) TIMEOUT=$2; shift 2 ;;
        --no-resolve) RESOLVE=0; shift ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done
VM=${FA_VM_URL:-http://127.0.0.1:8428}
API=${FA_API_URL:-http://127.0.0.1:7700}
ID="drill-$(date -u +%Y%m%dT%H%M%SZ)"
OUT=$(mktemp); trap 'kill ${SUB:-} ${PUSH:-} 2>/dev/null || true; docker rm -f "fa-alert-drill-sub" >/dev/null 2>&1 || true; rm -f "$OUT"' EXIT

curl -fsS "$API/api/alerts" >/dev/null || die "control-api $API/api/alerts not reachable"
curl -fsS "http://127.0.0.1:8880/health" >/dev/null || die "vmalert not reachable on 127.0.0.1:8880 (docker compose up -d vmalert)"

log "subscribing to ops.alert"
docker run --rm --name fa-alert-drill-sub --network "$(network_name)" natsio/nats-box:latest \
    nats --server nats://nats:4222 sub ops.alert --raw > "$OUT" 2>/dev/null &
SUB=$!
sleep 2

log "writing fa_alert_drill{drill_id=\"$ID\"} = 1 to VictoriaMetrics"
( while :; do echo "fa_alert_drill{drill_id=\"$ID\"} 1" | curl -fsS --data-binary @- "$VM/api/v1/import/prometheus" >/dev/null || true; sleep 20; done ) &
PUSH=$!

wait_for() { # status
    local deadline=$((SECONDS + TIMEOUT))
    while [ $SECONDS -lt $deadline ]; do
        if grep -F "\"$ID\"" "$OUT" | grep -q "\"status\":\"$1\""; then return 0; fi
        sleep 2
    done
    return 1
}

t0=$SECONDS
wait_for firing || die "no firing ops.alert for $ID within ${TIMEOUT}s"
log "ops.alert firing received after $((SECONDS - t0)) s: $(grep -F "\"$ID\"" "$OUT" | head -1 | cut -c1-200)"
curl -fsS "$API/api/alerts" | grep -qF "\"$ID\"" || die "GET /api/alerts does not list $ID"
log "GET /api/alerts lists it"
n=$(compose exec -T postgres psql -U fa -d fa -Atc "select count(*) from audit.log where action = 'ops.alert.firing' and payload->'labels'->>'drill_id' = '$ID'")
[ "$n" -ge 1 ] || die "no ops.alert.firing audit row for $ID"
log "audit.log has the ops.alert.firing row"

kill "$PUSH" 2>/dev/null || true; PUSH=
if [ $RESOLVE = 1 ]; then
    log "stopped writing; waiting for the resolved message"
    t1=$SECONDS
    TIMEOUT=$((TIMEOUT + 180)) wait_for resolved || die "no resolved ops.alert for $ID"
    log "ops.alert resolved received after $((SECONDS - t1)) s"
fi
log "alert drill $ID OK"
