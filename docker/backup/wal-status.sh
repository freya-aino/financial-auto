#!/bin/sh
# Every 5 minutes: WAL archiver health from pg_stat_archiver into VictoriaMetrics (alerts
# WalArchiveFailing / WalArchiveStale in docker/vm-alerts.yaml).
set -eu
. /usr/local/lib/fa-backup/common.sh
row=$(psql -AtF ' ' -c "select archived_count, failed_count,
    coalesce(extract(epoch from last_archived_time)::bigint, 0), coalesce(extract(epoch from last_failed_time)::bigint, 0),
    (select setting from pg_settings where name = 'archive_mode')
    from pg_stat_archiver") || { echo 'fa_wal_archive_up 0' | push_metrics; exit 1; }
set -- $row
mode=0; [ "$5" = on ] && mode=1
cat <<METRICS | push_metrics
fa_wal_archive_up 1
fa_wal_archive_mode_on $mode
fa_wal_archive_archived_total $1
fa_wal_archive_failed_total $2
fa_wal_archive_last_archived_timestamp_seconds $3
fa_wal_archive_last_failed_timestamp_seconds $4
fa_wal_archive_bytes $(du -sb "$WAL_DIR" 2>/dev/null | cut -f1 || echo 0)
METRICS
