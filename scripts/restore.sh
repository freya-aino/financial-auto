#!/usr/bin/env bash
# Restore a scripts/backup.sh directory, following the spec 60 §6.4 recovery order (D-056).
# Usage: scripts/restore.sh <backup-dir> [--yes] [--only postgres,helix,garage] [--postgres-db NAME]
#                           [--pitr [--target TIME]] [--pg-volume NAME]
#   --yes          skip the confirmation prompt
#   --only         restore only these parts (default: postgres,helix,garage; `raw` copies the
#                  fa-raw mirror backups/raw-mirror back into Garage)
#   --postgres-db  restore into this database instead of `fa` (scratch test; app services untouched)
#   --pitr         point-in-time recovery instead of pg_restore: the backup's PITR base backup
#                  (postgres/base, scheduled backups) + the pg-wal-archive volume, replayed up to
#                  --target 'YYYY-MM-DD HH:MM:SS+00' (default: end of archived WAL), then promoted
#                  (new timeline). The current data directory is first saved to
#                  backups/pre-restore-<ts>/postgres-data.tar.gz.
#   --pg-volume    with --pitr: restore into this volume instead of the live postgres-data volume and
#                  verify it in a throwaway container (drills; the live stack is not touched)
# Order: 1 stop app services (unless --postgres-db) → 2 postgres → 3 helix (stop, replace volume,
# start) → 4 garage buckets (rclone sync) → 5 print the §6.4 order; `just app-up` then runs
# migrations (control-api), reconciliation (risk-engine) and the health wait (scripts/healthwait.sh).
set -euo pipefail
. "$(dirname "$0")/lib.sh"

[ $# -ge 1 ] || { sed -n '2,19p' "$0"; exit 2; }
SRC=$(cd "$1" && pwd); shift
YES=0; ONLY="postgres,helix,garage"; PGDB="fa"; PITR=0; TARGET=""; PGVOL=""
while [ $# -gt 0 ]; do
    case $1 in
        --yes|-y) YES=1; shift ;;
        --only) ONLY=$2; shift 2 ;;
        --postgres-db) PGDB=$2; shift 2 ;;
        --pitr) PITR=1; shift ;;
        --target) TARGET=$2; shift 2 ;;
        --pg-volume) PGVOL=$2; shift 2 ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done
want() { case ",$ONLY," in *",$1,"*) return 0 ;; esac; return 1; }
[ -f "$SRC/manifest.json" ] || die "$SRC has no manifest.json"

# Verify checksums first: a restore from a corrupt file must not start.
log "verifying manifest checksums"
verify_manifest "$SRC" || die "manifest verification failed"

if [ $PITR = 1 ]; then
    [ -f "$SRC/postgres/base/base.tar.gz" ] || die "$SRC has no PITR base backup (postgres/base); use a scheduled backup"
    exec_pitr=1
fi
SCRATCH=0; [ "$PGDB" != "fa" ] && SCRATCH=1
[ -n "$PGVOL" ] && [ "$PGVOL" != "$(volume_name postgres-data)" ] && SCRATCH=1 && ONLY=postgres
echo "Restore from: $SRC ($(sed -n 's/.*"created_at": "\([^"]*\)".*/\1/p' "$SRC/manifest.json"))"
echo "Parts:        $ONLY"
if [ $SCRATCH = 1 ]; then
    if [ -n "$PGVOL" ]; then echo "Postgres:     PITR into scratch volume '$PGVOL' (live untouched; helix/garage skipped)"
    else echo "Postgres:     into scratch database '$PGDB' (live 'fa' untouched; helix/garage parts are skipped)"; fi
    ONLY=postgres
else
    echo "THIS OVERWRITES the live Postgres database 'fa', the HelixDB volume and the Garage buckets."
fi
if [ $YES != 1 ]; then
    read -r -p "Type 'restore' to continue: " ans
    [ "$ans" = "restore" ] || die "aborted"
fi

# 1. Stop the application services so nothing writes during the restore (infra stays up).
if [ $SCRATCH = 0 ]; then
    log "1/5 stopping app services (profile app)"
    compose --profile app stop control-api ui model-runtime strategy-runtime risk-engine evo-worker ingestor broker-gateway evo-orchestrator >/dev/null 2>&1 || true
fi

# 2a. Postgres PITR: replace the data directory with the base backup, replay archived WAL, promote.
pitr_into_volume() { # volume
    local vol=$1
    docker run --rm -v "$vol":/data -v "$SRC/postgres/base":/base:ro alpine:3.21 sh -euc '
        find /data -mindepth 1 -delete
        tar xzf /base/base.tar.gz -C /data
        [ -f /base/pg_wal.tar.gz ] && tar xzf /base/pg_wal.tar.gz -C /data/pg_wal
        {
            echo "# scripts/restore.sh --pitr $(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "restore_command = '"'"'gunzip -c /wal-archive/%f.gz > %p'"'"'"
            echo "recovery_target_action = '"'"'promote'"'"'"
            [ -n "$1" ] && echo "recovery_target_time = '"'"'$1'"'"'"
        } >> /data/postgresql.auto.conf
        touch /data/recovery.signal
        chown -R 70:70 /data && chmod 700 /data' pitr "$TARGET"
}
if [ "${exec_pitr:-0}" = 1 ] && want postgres; then
    if [ $SCRATCH = 1 ]; then
        log "2/5 postgres PITR → scratch volume $PGVOL (live postgres untouched)"
        docker volume inspect "$PGVOL" >/dev/null 2>&1 || docker volume create "$PGVOL" >/dev/null
        pitr_into_volume "$PGVOL"
        name="fa-pitr-check-$$"
        docker run -d --rm --name "$name" --network none -v "$PGVOL":/var/lib/postgresql/data \
            -v "$(volume_name pg-wal-archive)":/wal-archive:ro postgres:17-alpine >/dev/null
        for _ in $(seq 300); do
            [ "$(docker exec "$name" psql -U fa -d fa -Atc 'select pg_is_in_recovery()' 2>/dev/null)" = f ] && break; sleep 1
        done
        state=$(docker exec "$name" psql -U fa -d fa -Atc 'select pg_is_in_recovery()' 2>/dev/null || echo error)
        n=$(docker exec "$name" psql -U fa -d fa -Atc "select count(*) from information_schema.tables where table_schema not in ('pg_catalog','information_schema')" 2>/dev/null || echo 0)
        docker logs "$name" 2>&1 | grep -E 'restored log file|last completed transaction|recovery stopping|selected new timeline' | tail -4 >&2 || true
        docker stop "$name" >/dev/null
        [ "$state" = f ] || die "PITR into $PGVOL did not finish recovery ($state)"
        log "postgres PITR: promoted, $n tables in $PGVOL"
        if [ "${FA_KEEP_SCRATCH:-0}" != 1 ]; then # empty the scratch volume again (FA_KEEP_SCRATCH=1 keeps it)
            docker run --rm -v "$PGVOL":/data alpine:3.21 find /data -mindepth 1 -delete
            log "scratch volume $PGVOL emptied"
        fi
    else
        pre="$ROOT/backups/pre-restore-$(date -u +%Y%m%dT%H%M%SZ)"
        mkdir -p "$pre"
        log "2/5 postgres PITR: stopping postgres, saving the current data directory to $pre"
        compose stop postgres >/dev/null
        docker run --rm -v "$(volume_name postgres-data)":/data:ro -v "$pre":/out alpine:3.21 tar czf /out/postgres-data.tar.gz -C /data .
        pitr_into_volume "$(volume_name postgres-data)"
        compose up -d postgres >/dev/null
        log "postgres PITR: replaying WAL (docker compose logs -f postgres)"
        for _ in $(seq 600); do
            [ "$(compose exec -T postgres psql -U fa -d fa -Atc 'select pg_is_in_recovery()' 2>/dev/null)" = f ] && break; sleep 1
        done
        [ "$(compose exec -T postgres psql -U fa -d fa -Atc 'select pg_is_in_recovery()' 2>/dev/null)" = f ] || die "postgres still in recovery; see docker compose logs postgres"
        log "postgres PITR: promoted ($(compose logs postgres 2>&1 | grep -c 'restored log file') WAL files replayed)"
    fi
fi

# 2. Postgres: recreate the database and pg_restore the custom-format dump.
if [ "${exec_pitr:-0}" = 0 ] && want postgres && [ -f "$SRC/postgres/fa.dump" ]; then
    log "2/5 postgres → database $PGDB"
    compose exec -T postgres psql -U fa -d postgres -v ON_ERROR_STOP=1 -q \
        -c "select pg_terminate_backend(pid) from pg_stat_activity where datname = '$PGDB' and pid <> pg_backend_pid();" \
        -c "drop database if exists \"$PGDB\";" -c "create database \"$PGDB\" owner fa;" >/dev/null
    compose exec -T postgres pg_restore -U fa -d "$PGDB" --no-owner --no-privileges --exit-on-error < "$SRC/postgres/fa.dump"
    n=$(compose exec -T postgres psql -U fa -d "$PGDB" -Atc "select count(*) from information_schema.tables where table_schema not in ('pg_catalog','information_schema')")
    log "postgres: $n tables in $PGDB (backup had $(cat "$SRC/postgres/table_count.txt" 2>/dev/null || echo '?'))"
fi

# 3. HelixDB: stop, wipe the volume, untar, start.
if want helix && [ -f "$SRC/helix/helix-data.tar.gz" ]; then
    vol=$(volume_name helix-data)
    log "3/5 helix → volume $vol"
    compose stop helix >/dev/null
    docker run --rm -v "$vol":/data -v "$SRC/helix":/backup:ro alpine:3.21 \
        sh -c 'find /data -mindepth 1 -delete && tar xzf /backup/helix-data.tar.gz -C /data'
    compose start helix >/dev/null
fi

# 4. Garage: sync each backed-up bucket back (objects missing in the backup are deleted).
if want garage && [ -d "$SRC/garage" ]; then
    for d in "$SRC"/garage/*/; do
        b=$(basename "$d")
        log "4/5 garage → bucket $b"
        garage_rclone "$SRC/garage" -- sync "/backup/$b" "garage:$b" --s3-no-check-bucket -q || log "garage: bucket $b failed"
    done
fi

# 4b. Raw archive mirror (backups/raw-mirror/<bucket>, written by the backup sidecar): copied back
#     without deleting anything newer in Garage. Only with --only raw (it is shared by all backups).
if want raw && [ -d "$ROOT/backups/raw-mirror" ]; then
    for d in "$ROOT"/backups/raw-mirror/*/; do
        b=$(basename "$d")
        log "4/5 garage ← raw mirror $b"
        garage_rclone "$ROOT/backups/raw-mirror" -- copy "/backup/$b" "garage:$b" --s3-no-check-bucket -q || log "garage: mirror $b failed"
    done
fi

log "5/5 restore finished"
if [ $SCRATCH = 0 ]; then
    cat <<MSG
Recovery order (spec 60 §6.4) from here:
  1. infra healthy (postgres, nats, garage, victoriametrics, helix)   — running
  2. migrations: control-api applies migrations/ on start
  3. ledger + broker-gateway → reconciliation before trading           — risk-engine reconciles on start
  4. risk state incl. kill switch                                      — risk-engine loads ledger.strategy_state
  5. ingestor resumes from ingest.backfill_cursor
  6. model-runtime, evo-orchestrator, evo-worker
  7. control-api, ui
Run:  just app-up     (starts everything and waits via scripts/healthwait.sh)
MSG
fi
