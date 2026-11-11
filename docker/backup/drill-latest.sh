#!/bin/sh
# Weekly self-check: Postgres restore drill (logical + PITR) on the newest scheduled backup.
set -eu
. /usr/local/lib/fa-backup/common.sh
latest=$(scheduled_dirs | tail -1)
[ -n "$latest" ] || die "no scheduled backup to drill"
exec /usr/local/lib/fa-backup/drill-postgres.sh "$latest"
