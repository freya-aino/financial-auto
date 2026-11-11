#!/bin/sh
# Scheduled backup (spec 60 §6.6) run by the sidecar's crond, writing /backups/<UTC stamp>/ in the
# same layout as scripts/backup.sh plus a PITR base backup, so scripts/restore.sh and
# scripts/restore-drill.sh accept both:
#   postgres/base/{base.tar.gz,pg_wal.tar.gz,backup_manifest}  pg_basebackup -Ft -z -X stream (PITR base)
#   postgres/base/backup_label.txt                              START WAL LOCATION → WAL pruning point
#   postgres/fa.dump, globals.sql, table_count.txt, row_counts.tsv   logical dump (+ exact row counts)
#   helix/helix-data.tar.gz, files.txt                          HelixDB volume, mounted read-only (hot copy)
#   garage/<bucket>/                                            rclone copy of the critical buckets
#   garage-mirror.txt                                           object counts of /backups/raw-mirror/<bucket>
#                                                               (fa-raw, mirrored incrementally, never pruned)
#   manifest.json                                               kind "scheduled", sizes, sha256
# Then: metrics → VictoriaMetrics, retention pruning (prune.sh), optional off-machine copy (disabled).
# Env: FA_BACKUP_BUCKETS, FA_BACKUP_SKIP (postgres,helix,garage), FA_BACKUP_KEEP (prune.sh),
#      FA_BACKUP_OFFSITE_REMOTE + FA_BACKUP_OFFSITE_ENABLED=1 (rclone remote, e.g. "offsite:fa-backup").
set -eu
. /usr/local/lib/fa-backup/common.sh

started=$(date +%s)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
DEST="$BACKUP_ROOT/$STAMP"
SKIP=${FA_BACKUP_SKIP:-}
skip() { case ",$SKIP," in *",$1,"*) return 0 ;; esac; return 1; }

fail() {
    log "backup FAILED: $*"
    [ -d "$DEST" ] && echo "$*" > "$DEST/FAILED"
    printf 'fa_backup_last_status{kind="scheduled"} 0\nfa_backup_last_run_timestamp_seconds{kind="scheduled"} %s\n' "$(date +%s)" | push_metrics
    exit 1
}
trap 'fail "interrupted or command failed (line $LINENO)"' EXIT

mkdir -p "$DEST"
[ -e "$BACKUP_ROOT/.gitignore" ] || printf '*\n' > "$BACKUP_ROOT/.gitignore"
log "scheduled backup → $DEST"

if ! skip postgres; then
    mkdir -p "$DEST/postgres/base"
    log "postgres: pg_basebackup (tar, gzip, WAL streamed)"
    pg_basebackup -D "$DEST/postgres/base" -Ft -z -X stream -c fast -l "fa-scheduled-$STAMP" -w
    tar xzOf "$DEST/postgres/base/base.tar.gz" ./backup_label 2>/dev/null > "$DEST/postgres/base/backup_label.txt" \
        || tar xzOf "$DEST/postgres/base/base.tar.gz" backup_label > "$DEST/postgres/base/backup_label.txt"
    log "postgres: pg_dump -Fc"
    pg_dump -Fc --no-owner -f "$DEST/postgres/fa.dump"
    pg_dumpall --globals-only > "$DEST/postgres/globals.sql"
    psql -Atc "select count(*) from information_schema.tables where table_schema not in ('pg_catalog','information_schema')" > "$DEST/postgres/table_count.txt"
    # Exact row counts per base table; the drill compares the restored database against them.
    psql -AtF "$(printf '\t')" -c "select table_schema||'.'||table_name,
        (xpath('/row/c/text()', query_to_xml(format('select count(*) as c from %I.%I', table_schema, table_name), false, true, '')))[1]::text::bigint
        from information_schema.tables where table_type = 'BASE TABLE' and table_schema not in ('pg_catalog','information_schema') order by 1" \
        > "$DEST/postgres/row_counts.tsv"
fi

if ! skip helix && [ -d /helix ]; then
    mkdir -p "$DEST/helix"
    log "helix: tar of the read-only volume (hot copy; scripts/backup.sh stops helix for a cold one)"
    (cd /helix && tar czf "$DEST/helix/helix-data.tar.gz" . && find . -type f | sort > "$DEST/helix/files.txt")
fi

if ! skip garage; then
    if [ -n "$(garage_secret access_key_id 2>/dev/null)" ]; then
        for b in ${FA_BACKUP_BUCKETS:-fa-models fa-ckpt fa-artifacts fa-backup}; do
            mkdir -p "$DEST/garage/$b"
            log "garage: copy bucket $b"
            garage_rclone sync "garage:$b" "$DEST/garage/$b" --s3-no-check-bucket -q || log "garage: bucket $b failed (missing or empty?)"
        done
        # Raw archive (spec 10/20: the source to rebuild VictoriaMetrics and replay ingestion): its objects
        # are immutable ULID keys, so it is mirrored incrementally into /backups/raw-mirror (shared by all
        # backups, never pruned with them) instead of being copied into every backup directory.
        for b in ${FA_BACKUP_MIRROR_BUCKETS:-fa-raw}; do
            log "garage: incremental mirror of $b → raw-mirror/$b"
            garage_rclone copy "garage:$b" "$BACKUP_ROOT/raw-mirror/$b" --immutable --s3-no-check-bucket -q \
                || log "garage: mirror of $b failed"
            echo "$b $(find "$BACKUP_ROOT/raw-mirror/$b" -type f 2>/dev/null | wc -l)" >> "$DEST/garage-mirror.txt"
        done
    else
        log "garage: no [garage] credentials in secrets.toml, skipped"
    fi
fi

log "manifest"
files=$(cd "$DEST" && find . -type f ! -name manifest.json | sed 's|^\./||' | sort)
{
    printf '{\n  "kind": "scheduled",\n  "created_at": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "postgres_version": "%s",\n' "$(psql -Atc 'show server_version' 2>/dev/null || echo unknown)"
    printf '  "pitr": %s,\n  "helix_consistency": "hot",\n' "$([ -f "$DEST/postgres/base/base.tar.gz" ] && echo true || echo false)"
    printf '  "files": [\n'
    first=1
    for f in $files; do
        [ $first = 1 ] || printf ',\n'; first=0
        printf '    {"path": "%s", "bytes": %s, "sha256": "%s"}' "$f" "$(stat -c %s "$DEST/$f")" "$(sha256sum "$DEST/$f" | cut -d' ' -f1)"
    done
    printf '\n  ]\n}\n'
} > "$DEST/manifest.json"
trap - EXIT

bytes=$(du -sb "$DEST" | cut -f1)
now=$(date +%s)
log "done: $bytes bytes in $((now - started)) s"
cat <<METRICS | push_metrics
fa_backup_last_status{kind="scheduled"} 1
fa_backup_last_run_timestamp_seconds{kind="scheduled"} $now
fa_backup_last_success_timestamp_seconds{kind="scheduled"} $now
fa_backup_last_duration_seconds{kind="scheduled"} $((now - started))
fa_backup_last_bytes{kind="scheduled"} $bytes
METRICS

/usr/local/lib/fa-backup/prune.sh || log "prune failed"

# Off-machine copy: configured but disabled (spec 60 §10 Q4, external targets are out of scope).
if [ "${FA_BACKUP_OFFSITE_ENABLED:-0}" = 1 ] && [ -n "${FA_BACKUP_OFFSITE_REMOTE:-}" ]; then
    log "offsite: rclone copy → $FA_BACKUP_OFFSITE_REMOTE"
    rclone copy "$DEST" "$FA_BACKUP_OFFSITE_REMOTE/$STAMP" -q && rclone copy "$WAL_DIR" "$FA_BACKUP_OFFSITE_REMOTE/wal" -q \
        || log "offsite copy failed"
else
    log "offsite: disabled (FA_BACKUP_OFFSITE_ENABLED=1 and FA_BACKUP_OFFSITE_REMOTE enable it)"
fi
echo "$DEST"
