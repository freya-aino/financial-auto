#!/usr/bin/env bash
# Ephemeral test stacks (D-096): every test run gets its own compose project `fa-eph-<run>` with
# fresh volumes, a local-only connector set and its own backup directory; `down` deletes exactly
# that project's containers, volumes and backups so test runs never accumulate on disk.
#
#   scripts/eph.sh up <run> [infra]   start (infra = without the app profile)
#   scripts/eph.sh down <run>         remove containers, volumes and scratch/backups-<run>
#   scripts/eph.sh ls                 list leftover fa-eph-* projects and volumes
#
# Host ports are the normal ones (D-014), so only one stack — the main one or one ephemeral run —
# can be up at a time. Env: FA_CONNECTORS (default synthetic,heartbeat), FA_EPH_DISK_MAX (85 %).
set -euo pipefail
. "$(dirname "$0")/lib.sh"

usage() { sed -n '5,8p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

run_name() {
    [[ ${1:-} =~ ^[a-z0-9][a-z0-9-]{0,30}$ ]] || die "run name must match [a-z0-9-], got '${1:-}'"
    echo "$1"
}

disk_pct() { df --output=pcent / | tail -1 | tr -dc 0-9; }

running_projects() { # compose projects of this repo with running containers
    docker ps --format '{{.Label "com.docker.compose.project"}}' | sort -u | grep -E '^(financial-auto|fa-eph-.+)$' || true
}

up() {
    local run; run=$(run_name "${1:-}"); local mode=${2:-app}
    local pct; pct=$(disk_pct)
    [ "$pct" -le "${FA_EPH_DISK_MAX:-85}" ] || die "disk at ${pct} % (limit ${FA_EPH_DISK_MAX:-85} %); free space first"
    local busy; busy=$(running_projects | grep -vx "fa-eph-$run" || true)
    [ -z "$busy" ] || die "another stack holds the ports: $busy (stop it first)"

    export COMPOSE_PROJECT_NAME="fa-eph-$run"
    export FA_BACKUP_DIR="$ROOT/scratch/backups-$run"
    export FA_CONNECTORS=${FA_CONNECTORS:-synthetic,heartbeat}
    mkdir -p "$FA_BACKUP_DIR"
    log "up $COMPOSE_PROJECT_NAME (connectors $FA_CONNECTORS, backups $FA_BACKUP_DIR, disk ${pct} %)"
    compose up -d --wait garage
    "$ROOT/docker/garage-init.sh" >/dev/null
    if [ "$mode" = infra ]; then
        compose up -d --wait
    else
        compose --profile app up -d
        "$ROOT/scripts/healthwait.sh" "${FA_HEALTHWAIT_TIMEOUT:-300}"
    fi
    log "stack fa-eph-$run is up; tear down with: scripts/eph.sh down $run"
}

down() {
    local run; run=$(run_name "${1:-}")
    export COMPOSE_PROJECT_NAME="fa-eph-$run"
    log "down $COMPOSE_PROJECT_NAME: containers, volumes, scratch/backups-$run"
    compose --profile app --profile tools down -v --remove-orphans
    local dir="$ROOT/scratch/backups-$run"
    if [ -d "$dir" ]; then # files are written by root inside the backup container
        docker run --rm --name "fa-eph-rm-$run" -v "$ROOT/scratch":/scratch alpine:3.20 rm -rf "/scratch/backups-$run"
    fi
    local left; left=$(docker volume ls -q --filter "label=com.docker.compose.project=fa-eph-$run")
    [ -z "$left" ] || die "volumes left behind: $left"
    log "removed; disk at $(disk_pct) %"
}

ls_runs() {
    echo "projects:"; docker compose ls -a --format table | awk 'NR==1 || /^fa-eph-/'
    echo "volumes:"; docker volume ls --format '{{.Name}}' | grep '^fa-eph-' || echo "  (none)"
    echo "backups:"; ls -1d "$ROOT"/scratch/backups-* 2>/dev/null || echo "  (none)"
}

case ${1:-} in
    up) shift; up "$@" ;;
    down) shift; down "$@" ;;
    ls) ls_runs ;;
    *) usage ;;
esac
