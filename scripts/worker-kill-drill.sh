#!/usr/bin/env bash
# evo-worker kill drill (spec 60 §9): a job whose worker is SIGKILLed mid-run is redelivered by
# JetStream and finished by another worker. Runs entirely off the compose stack: a throwaway
# NATS container (`fa-worker-drill-nats`, 127.0.0.1:<port>, no
# volume; first free port from 14322 unless FA_DRILL_NATS_PORT) and two native evo-worker processes whose platform config points every other store at
# a closed port. Takes ~2 minutes (ack wait 60 s). `just worker-kill-drill`.
#
#   scripts/worker-kill-drill.sh [--keep]     --keep leaves the work dir (worker logs) behind
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
port=${FA_DRILL_NATS_PORT:-}
if [[ -z $port ]]; then # first free port from 14322
  for port in $(seq 14322 14399); do
    (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null || break
  done
fi
name=fa-worker-drill-nats
keep=${1:-}
dir=$(mktemp -d "${TMPDIR:-/tmp}/fa-worker-drill.XXXXXX")

cleanup() {
  docker rm -f "$name" >/dev/null 2>&1 || true
  if [[ $keep == --keep ]]; then echo "worker logs: $dir" >&2; else rm -rf "$dir"; fi
}
trap cleanup EXIT

echo "== building evo-worker and the drill driver" >&2
cargo build -q --manifest-path "$root/Cargo.toml" -p evo-worker -p control-api --example worker_kill_drill

echo "== scratch NATS on 127.0.0.1:$port" >&2
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --rm --name "$name" -p "127.0.0.1:$port:4222" nats:2-alpine -js >/dev/null
for _ in $(seq 1 50); do
  (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && break
  sleep 0.2
done

# Platform config: scratch NATS; Helix, VictoriaMetrics, Postgres and Garage on a closed port.
awk -v nats="nats://127.0.0.1:$port" '
  /^\[/ { section = $0 }
  /^url = / && section == "[nats]" { print "url = \"" nats "\""; next }
  /^url = / && section == "[postgres]" { print "url = \"postgres://fa:fa@127.0.0.1:9/fa\""; next }
  /^url = / { print "url = \"http://127.0.0.1:9\""; next }
  /^endpoint = / && section == "[garage]" { print "endpoint = \"http://127.0.0.1:9\""; next }
  { print }
' "$root/config/platform.toml" >"$dir/platform.toml"

echo "== drill (worker A holds the job, worker B buffers trades, A is killed)" >&2
cd "$root"
FA_DRILL_DIR="$dir" "$root/target/debug/examples/worker_kill_drill" \
  "nats://127.0.0.1:$port" "$root/target/debug/evo-worker" "$dir/platform.toml"
