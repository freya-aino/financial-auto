#!/usr/bin/env bash
# Backup of the local stack (spec 60 §6.6, spec 20 §6.6, D-056) into backups/<UTC timestamp>/:
#   postgres/fa.dump        pg_dump -Fc of database `fa` (ledger, warehouse, validation registry)
#   postgres/globals.sql    roles (pg_dumpall --globals-only)
#   helix/helix-data.tar.gz HelixDB data volume, taken while the helix service is stopped (~seconds)
#   garage/<bucket>/        rclone copy of the critical buckets (fa-models fa-ckpt fa-artifacts fa-backup)
#   nats/<stream>/          optional `nats stream backup` per JetStream stream (FA_BACKUP_NATS=1)
#   manifest.json           versions, sizes, sha256 of every file
# Usage: scripts/backup.sh [--dest DIR] [--skip postgres,helix,garage] [--no-stop-helix]
# Env:   FA_BACKUP_BUCKETS (space-separated, default above), FA_BACKUP_NATS=1, RCLONE_IMAGE
# Schedule (backup.schedule 03:00 daily, spec 60 §7) — add to `crontab -e`:
#   0 3 * * * cd /path/to/financial-auto && scripts/backup.sh >> backups/backup.log 2>&1
set -euo pipefail
. "$(dirname "$0")/lib.sh"

DEST_ROOT="$ROOT/backups"; SKIP=""; STOP_HELIX=1
while [ $# -gt 0 ]; do
    case $1 in
        --dest) DEST_ROOT=$2; shift 2 ;;
        --skip) SKIP=$2; shift 2 ;;
        --no-stop-helix) STOP_HELIX=0; shift ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done
skip() { case ",$SKIP," in *",$1,"*) return 0 ;; esac; return 1; }

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
DEST="$DEST_ROOT/$STAMP"
mkdir -p "$DEST"
[ -e "$DEST_ROOT/.gitignore" ] || printf '*\n' > "$DEST_ROOT/.gitignore" # backups never enter git
log "backup → $DEST"
FILES=() # relative paths for the manifest

# 1. Postgres: custom-format dump + globals (restore with pg_restore / psql).
if ! skip postgres; then
    mkdir -p "$DEST/postgres"
    log "postgres: pg_dump -Fc fa"
    compose exec -T postgres pg_dump -U fa -d fa -Fc --no-owner > "$DEST/postgres/fa.dump"
    compose exec -T postgres pg_dumpall -U fa --globals-only > "$DEST/postgres/globals.sql"
    compose exec -T postgres psql -U fa -d fa -Atc "select count(*) from information_schema.tables where table_schema not in ('pg_catalog','information_schema')" > "$DEST/postgres/table_count.txt"
    FILES+=(postgres/fa.dump postgres/globals.sql postgres/table_count.txt)
fi

# 2. HelixDB: tar the data volume through a throwaway alpine while the service is stopped.
if ! skip helix; then
    mkdir -p "$DEST/helix"
    vol=$(volume_name helix-data)
    if [ "$STOP_HELIX" = 1 ]; then log "helix: stopping service for a consistent snapshot"; compose stop helix >/dev/null; fi
    log "helix: tar volume $vol"
    docker run --rm -v "$vol":/data:ro -v "$DEST/helix":/backup alpine:3.21 \
        sh -c 'cd /data && tar czf /backup/helix-data.tar.gz . && find . -type f | sort > /backup/files.txt'
    if [ "$STOP_HELIX" = 1 ]; then compose start helix >/dev/null; log "helix: started"; fi
    FILES+=(helix/helix-data.tar.gz helix/files.txt)
fi

# 3. Garage: critical buckets copied object-by-object (rclone sync, S3 API; no service stop).
if ! skip garage; then
    for b in ${FA_BACKUP_BUCKETS:-fa-models fa-ckpt fa-artifacts fa-backup}; do
        mkdir -p "$DEST/garage/$b"
        log "garage: sync bucket $b"
        garage_rclone "$DEST/garage" -- sync "garage:$b" "/backup/$b" --s3-no-check-bucket -q || log "garage: bucket $b failed (missing or empty?)"
        while IFS= read -r f; do FILES+=("garage/$b/${f#"$DEST/garage/$b/"}"); done < <(find "$DEST/garage/$b" -type f)
    done
fi

# 4. NATS JetStream (optional): streams are replayable limits-retention buffers (spec 21), so this
#    is off by default. `nats stream backup` through nats-box writes one directory per stream.
if [ "${FA_BACKUP_NATS:-0}" = 1 ]; then
    mkdir -p "$DEST/nats"
    for st in $(curl -fsS 'http://127.0.0.1:8222/jsz?streams=true' | tr ',' '\n' | sed -n 's/.*"name":"\([A-Z_]*\)".*/\1/p' | sort -u); do
        log "nats: backup stream $st"
        docker run --rm --network "$(network_name)" -v "$DEST/nats":/backup natsio/nats-box:latest \
            nats --server nats://nats:4222 stream backup "$st" "/backup/$st" >/dev/null || log "nats: stream $st failed"
    done
    while IFS= read -r f; do FILES+=("nats/${f#"$DEST/nats/"}"); done < <(find "$DEST/nats" -type f)
fi

# 5. Manifest.
log "manifest"
images=$(compose images --format json 2>/dev/null | tr -d '\n' || echo '[]')
{
    printf '{\n  "created_at": "%s",\n  "git_rev": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    printf '  "postgres_version": "%s",\n' "$(compose exec -T postgres psql -U fa -d fa -Atc 'show server_version' 2>/dev/null || echo unknown)"
    printf '  "compose_images": %s,\n' "${images:-[]}"
    printf '  "files": [\n'
    first=1
    for f in "${FILES[@]}"; do
        [ -f "$DEST/$f" ] || continue
        [ $first = 1 ] || printf ',\n'; first=0
        printf '    {"path": "%s", "bytes": %s, "sha256": "%s"}' "$f" "$(size_of "$DEST/$f")" "$(sha256_of "$DEST/$f")"
    done
    printf '\n  ]\n}\n'
} > "$DEST/manifest.json"
log "done: $(du -sh "$DEST" | cut -f1) in $DEST ($(( ${#FILES[@]} )) files)"
echo "$DEST"
