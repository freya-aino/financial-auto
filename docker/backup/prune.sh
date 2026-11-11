#!/bin/sh
# Retention (spec 60 §6.6/§7): keep the newest FA_BACKUP_KEEP scheduled backups (default 7) and
# drop archived WAL older than the oldest kept PITR base backup (pg_archivecleanup), so every
# point between that base backup and now stays recoverable. Manual backups (scripts/backup.sh,
# manifest without "kind": "scheduled") are never pruned. FA_BACKUP_PRUNE_DRY_RUN=1 only prints.
set -eu
. /usr/local/lib/fa-backup/common.sh
keep=${FA_BACKUP_KEEP:-7}
dry=${FA_BACKUP_PRUNE_DRY_RUN:-0}
all=$(scheduled_dirs)
n=$(printf '%s\n' "$all" | grep -c . || true)
drop=$((n > keep ? n - keep : 0))
for d in $(printf '%s\n' "$all" | head -n "$drop"); do
    case $(basename "$d") in [0-9]*T[0-9]*Z) ;; *) continue ;; esac # only stamp-named directories
    log "prune: backup $d"
    [ "$dry" = 1 ] || rm -rf "$d"
done
# Oldest kept base backup → its START WAL file; everything archived before it is unreachable.
oldest=""
for d in $(printf '%s\n' "$all" | tail -n +"$((drop + 1))"); do
    [ -f "$d/postgres/base/backup_label.txt" ] && { oldest=$d; break; }
done
if [ -n "$oldest" ] && [ -d "$WAL_DIR" ]; then
    wal=$(sed -n 's/^START WAL LOCATION: .*(file \([0-9A-F]\{24\}\)).*/\1/p' "$oldest/postgres/base/backup_label.txt")
    if [ -n "$wal" ]; then
        log "prune: WAL older than $wal (base of $(basename "$oldest"))"
        if [ "$dry" = 1 ]; then pg_archivecleanup -n -x .gz "$WAL_DIR" "$wal"; else pg_archivecleanup -x .gz "$WAL_DIR" "$wal"; fi
    fi
fi
printf 'fa_backup_scheduled_kept %s\nfa_wal_archive_bytes %s\nfa_wal_archive_files %s\n' \
    "$((n - drop))" "$(du -sb "$WAL_DIR" 2>/dev/null | cut -f1 || echo 0)" "$(ls "$WAL_DIR" 2>/dev/null | wc -l)" | push_metrics
