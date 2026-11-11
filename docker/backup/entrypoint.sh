#!/bin/sh
# Backup sidecar entrypoint: installs the crontab from env and runs busybox crond in the foreground.
#   FA_BACKUP_SCHEDULE  cron line for run.sh             (default "0 3 * * *", backup.schedule spec 60 §7; TZ applies)
#   FA_DRILL_SCHEDULE   cron line for the postgres drill (default "30 4 * * 0", weekly on the newest backup)
#   FA_BACKUP_ON_START  "missing" (default): run once at start when no scheduled PITR base exists yet,
#                       so archived WAL is usable at once; "always" | "never"
# Other commands pass through: `docker compose exec backup run.sh`, `… drill-postgres.sh /backups/<ts>`.
set -eu
. /usr/local/lib/fa-backup/common.sh
L=/usr/local/lib/fa-backup
export PATH="$L:$PATH"
[ $# -gt 0 ] && exec "$@"

mkdir -p "$BACKUP_ROOT"
env | grep -E '^(FA_|PG|TZ=)' | sed 's/^/export /; s/=\(.*\)$/="\1"/' > /etc/fa-backup.env
cat > /etc/crontabs/root <<CRON
${FA_BACKUP_SCHEDULE:-0 3 * * *} . /etc/fa-backup.env; $L/run.sh > /proc/1/fd/1 2>&1
${FA_DRILL_SCHEDULE:-30 4 * * 0} . /etc/fa-backup.env; $L/drill-latest.sh > /proc/1/fd/1 2>&1
*/5 * * * * . /etc/fa-backup.env; $L/wal-status.sh > /proc/1/fd/1 2>&1
CRON
log "crontab:"; cat /etc/crontabs/root >&2

i=0; until pg_isready -q || [ $i -ge 60 ]; do sleep 2; i=$((i + 1)); done
case ${FA_BACKUP_ON_START:-missing} in
    always) run.sh || true ;;
    missing)
        has=0; for d in $(scheduled_dirs); do [ -f "$d/postgres/base/base.tar.gz" ] && has=1; done
        [ $has = 1 ] || { log "no scheduled base backup yet: running one now"; run.sh || true; } ;;
esac
wal-status.sh || true
exec crond -f -l 6 -c /etc/crontabs
