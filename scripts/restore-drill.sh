#!/usr/bin/env bash
# Restore drill (spec 60 §6.6): proves a backup directory restores WITHOUT touching live data.
# Nothing is written to the live databases, volumes or buckets; every restore goes into throwaway
# containers that are removed at the end (anonymous volumes / tmp dirs only).
#   1. manifest   sha256 of every file (scripts/lib.sh verify_manifest)
#   2. postgres   docker/backup/drill-postgres.sh in a throwaway backup-image container:
#                 logical (pg_restore into a scratch cluster, row counts) and, for scheduled backups,
#                 PITR (base backup + the pg-wal-archive volume mounted READ-ONLY, promoted, queried)
#   3. helix      the volume tar extracted into an anonymous volume, a scratch HelixDB container
#                 (no ports, no compose labels) started on it, /readyz must answer
#   4. garage     bucket copies covered by the manifest checksums; object counts reported
# Usage: scripts/restore-drill.sh [DIR] [--only manifest,postgres,helix,garage] [--target TIME] [--switch-wal]
#   DIR         default: newest backups/<ts> with a manifest
#   --target    PITR recovery_target_time, e.g. '2026-11-11 03:00:00+00' (default: end of archived WAL)
#   --switch-wal  ask live Postgres for pg_switch_wal() first so the newest changes are archived
#                 (the only interaction with a live service; read-only in effect)
# Exit 0 and "RESTORE DRILL OK" when every part passed; results also go to VictoriaMetrics
# (fa_restore_drill_last_status / _last_success_timestamp_seconds, alerts RestoreDrill*).
set -euo pipefail
. "$(dirname "$0")/lib.sh"

SRC=""; ONLY="manifest,postgres,helix,garage"; TARGET=""; SWITCH=0
while [ $# -gt 0 ]; do
    case $1 in
        --only) ONLY=$2; shift 2 ;;
        --target) TARGET=$2; shift 2 ;;
        --switch-wal) SWITCH=1; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        -*) die "unknown argument $1" ;;
        *) SRC=$1; shift ;;
    esac
done
want() { case ",$ONLY," in *",$1,"*) return 0 ;; esac; return 1; }
[ -n "$SRC" ] || SRC=$(latest_backup) || die "no backup directory with a manifest under backups/"
SRC=$(cd "$SRC" && pwd)
TAG="fa-drill-$$"
cleanup() { docker rm -f "$TAG-helix" "$TAG-helixdata" >/dev/null 2>&1 || true; }
trap cleanup EXIT
results=()
pass() { results+=("PASS $1"); log "PASS $1"; }
fail() { results+=("FAIL $1"); log "FAIL $1"; }

log "restore drill of $SRC ($(sed -n 's/.*"created_at": "\([^"]*\)".*/\1/p' "$SRC/manifest.json"), kind $(sed -n 's/.*"kind": "\([^"]*\)".*/\1/p' "$SRC/manifest.json" | head -1 || true))"

if want manifest; then
    if verify_manifest "$SRC"; then pass manifest; else fail manifest; fi
fi

if want postgres && [ -f "$SRC/postgres/fa.dump" ]; then
    docker image inspect "$BACKUP_IMAGE" >/dev/null 2>&1 || compose build backup >/dev/null
    [ $SWITCH = 1 ] && compose exec -T postgres psql -U fa -d fa -Atc 'select pg_switch_wal()' >/dev/null && sleep 3
    wal=(); docker volume inspect "$(volume_name pg-wal-archive)" >/dev/null 2>&1 && wal=(-v "$(volume_name pg-wal-archive)":/wal-archive:ro)
    if docker run --rm --name "$TAG-pg" --network "$(network_name)" -v "$SRC":/drill/backup:ro "${wal[@]}" \
        -e FA_DRILL_TARGET="$TARGET" "$BACKUP_IMAGE" drill-postgres.sh /drill/backup; then
        pass postgres
    else
        fail postgres
    fi
fi

if want helix && [ -f "$SRC/helix/helix-data.tar.gz" ]; then
    image=$(compose config --images 2>/dev/null | grep helixdb || true); image=${image%%$'\n'*}
    image=${FA_HELIX_IMAGE:-${image:-ghcr.io/helixdb/helixdb:v0.0.14}}
    log "helix: scratch $image on the backed-up volume"
    docker run -d --rm --name "$TAG-helixdata" -v /var/lib/helix -v "$SRC/helix":/backup:ro alpine:3.21 \
        sh -c 'tar xzf /backup/helix-data.tar.gz -C /var/lib/helix && touch /tmp/ready && sleep 3600' >/dev/null
    for _ in $(seq 60); do docker exec "$TAG-helixdata" test -f /tmp/ready 2>/dev/null && break; sleep 1; done
    docker run -d --rm --name "$TAG-helix" --network none --volumes-from "$TAG-helixdata" \
        -e HELIX_DATA_DIR=/var/lib/helix "$image" >/dev/null
    ok=0
    for _ in $(seq 30); do
        if docker run --rm --network "container:$TAG-helix" alpine:3.21 wget -qO- http://127.0.0.1:8080/readyz >/dev/null 2>&1; then ok=1; break; fi
        sleep 1
    done
    if [ $ok = 1 ]; then
        files=$(docker exec "$TAG-helixdata" sh -c 'find /var/lib/helix -type f | wc -l')
        pass "helix (/readyz answered on the restored volume, $files files)"
    else
        docker logs "$TAG-helix" 2>&1 | tail -10 >&2 || true
        fail helix
    fi
    cleanup
fi

if want garage && [ -d "$SRC/garage" ]; then
    summary=""
    for d in "$SRC"/garage/*/; do summary+="$(basename "$d")=$(find "$d" -type f | wc -l) "; done
    pass "garage (objects: ${summary% })"
fi

echo
printf '%s\n' "${results[@]}"
now=$(date +%s)
if printf '%s\n' "${results[@]}" | grep -q '^FAIL'; then
    echo 'fa_restore_drill_last_status 0' | curl -fsS --data-binary @- 'http://127.0.0.1:8428/api/v1/import/prometheus?extra_label=job=fa-backup' >/dev/null 2>&1 || true
    echo "RESTORE DRILL FAILED"; exit 1
fi
printf 'fa_restore_drill_last_status 1\nfa_restore_drill_last_success_timestamp_seconds %s\n' "$now" \
    | curl -fsS --data-binary @- 'http://127.0.0.1:8428/api/v1/import/prometheus?extra_label=job=fa-backup' >/dev/null 2>&1 || true
echo "RESTORE DRILL OK"
