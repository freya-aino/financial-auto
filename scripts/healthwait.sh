#!/usr/bin/env bash
# Waits for the stack to come up in the spec 60 §6.4 order and prints each step (D-056).
#   1. infrastructure health: postgres, nats, garage, victoriametrics, helix
#   2. control-api /api/health: nats connected + every dependency reachable (migrations applied on start)
#   3. risk-engine heartbeat in /api/services (ledger loaded, reconciliation armed) — trading stays
#      halted until this reports; ingestor / broker-gateway / model-runtime are reported as they appear
# Usage: scripts/healthwait.sh [timeout-seconds]   (default 180; FA_CONTROL_API=http://127.0.0.1:7700)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
TIMEOUT=${1:-${FA_HEALTHWAIT_TIMEOUT:-180}}
API=${FA_CONTROL_API:-http://127.0.0.1:7700}
deadline=$(( $(date +%s) + TIMEOUT ))

wait_for() { # <label> <command...> ; retries until the deadline
    local label=$1; shift
    local t0=$(date +%s)
    until "$@" >/dev/null 2>&1; do
        [ "$(date +%s)" -lt "$deadline" ] || { echo "  ✗ $label (timeout after ${TIMEOUT}s)"; return 1; }
        sleep 2
    done
    echo "  ✓ $label ($(( $(date +%s) - t0 ))s)"
}

echo "spec 60 §6.4 step 1: infrastructure"
wait_for "postgres  pg_isready"      compose exec -T postgres pg_isready -U fa -d fa
wait_for "nats      :8222/healthz"   curl -fsS http://127.0.0.1:8222/healthz
wait_for "garage    :3903/health"    curl -fsS http://127.0.0.1:3903/health
wait_for "victoria  :8428/health"    curl -fsS http://127.0.0.1:8428/health
wait_for "helix     :6969/healthz"   curl -fsS http://127.0.0.1:6969/healthz

echo "step 2: migrations + control-api /api/health (nats connected, dependencies reachable)"
health_ok() {
    local h; h=$(curl -fsS "$API/api/health") || return 1
    echo "$h" | grep -q '"connected":true' &&
        for d in helix victoriametrics postgres garage; do echo "$h" | grep -q "\"$d\":true" || return 1; done
}
wait_for "control-api $API/api/health all true" health_ok

echo "step 3-6: service heartbeats (risk-engine gates trading; others reported as they appear)"
svc_present() { curl -fsS "$API/api/services" | grep -q "\"service\":\"$1\""; }
wait_for "risk-engine heartbeat (ledger loaded, reconciliation armed)" svc_present risk-engine
for s in broker-gateway ingestor model-runtime strategy-runtime evo-orchestrator evo-worker; do
    if svc_present "$s"; then echo "  ✓ $s heartbeat"; else echo "  … $s heartbeat not seen yet"; fi
done
echo "step 7: control-api up; UI at http://127.0.0.1:8080 (compose) — metrics: curl http://127.0.0.1:9101/metrics"
