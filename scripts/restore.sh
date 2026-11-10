#!/usr/bin/env bash
# Restore a scripts/backup.sh directory, following the spec 60 §6.4 recovery order (D-056).
# Usage: scripts/restore.sh <backup-dir> [--yes] [--only postgres,helix,garage] [--postgres-db NAME]
#   --yes          skip the confirmation prompt
#   --only         restore only these parts (default: postgres,helix,garage)
#   --postgres-db  restore into this database instead of `fa` (scratch test; app services untouched)
# Order: 1 stop app services (unless --postgres-db) → 2 postgres → 3 helix (stop, replace volume,
# start) → 4 garage buckets (rclone sync) → 5 print the §6.4 order; `just app-up` then runs
# migrations (control-api), reconciliation (risk-engine) and the health wait (scripts/healthwait.sh).
set -euo pipefail
. "$(dirname "$0")/lib.sh"

[ $# -ge 1 ] || { sed -n '2,9p' "$0"; exit 2; }
SRC=$(cd "$1" && pwd); shift
YES=0; ONLY="postgres,helix,garage"; PGDB="fa"
while [ $# -gt 0 ]; do
    case $1 in
        --yes|-y) YES=1; shift ;;
        --only) ONLY=$2; shift 2 ;;
        --postgres-db) PGDB=$2; shift 2 ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done
want() { case ",$ONLY," in *",$1,"*) return 0 ;; esac; return 1; }
[ -f "$SRC/manifest.json" ] || die "$SRC has no manifest.json"

# Verify checksums first: a restore from a corrupt file must not start.
log "verifying manifest checksums"
bad=0
while IFS=$'\t' read -r path sum; do
    [ -f "$SRC/$path" ] || { log "missing $path"; bad=1; continue; }
    [ "$(sha256_of "$SRC/$path")" = "$sum" ] || { log "checksum mismatch: $path"; bad=1; }
done < <(sed -n 's/.*"path": "\([^"]*\)".*"sha256": "\([0-9a-f]*\)".*/\1\t\2/p' "$SRC/manifest.json")
[ $bad = 0 ] || die "manifest verification failed"

SCRATCH=0; [ "$PGDB" != "fa" ] && SCRATCH=1
echo "Restore from: $SRC ($(sed -n 's/.*"created_at": "\([^"]*\)".*/\1/p' "$SRC/manifest.json"))"
echo "Parts:        $ONLY"
if [ $SCRATCH = 1 ]; then
    echo "Postgres:     into scratch database '$PGDB' (live 'fa' untouched; helix/garage parts are skipped)"
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

# 2. Postgres: recreate the database and pg_restore the custom-format dump.
if want postgres && [ -f "$SRC/postgres/fa.dump" ]; then
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
