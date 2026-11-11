# Shared helpers of the backup sidecar scripts (POSIX sh, busybox). Source, don't execute.
# shellcheck shell=sh
BACKUP_ROOT=${FA_BACKUP_ROOT:-/backups}
WAL_DIR=${FA_WAL_ARCHIVE_DIR:-/wal-archive}
VM_URL=${FA_VM_URL:-http://victoriametrics:8428}
export PGHOST=${PGHOST:-postgres} PGUSER=${PGUSER:-fa} PGPASSWORD=${PGPASSWORD:-fa} PGDATABASE=${PGDATABASE:-fa}

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }
die() { log "error: $*"; exit 1; }

# Pushes Prometheus text lines (stdin) into VictoriaMetrics; failures are logged, never fatal.
push_metrics() {
    curl -fsS --max-time 10 --data-binary @- "$VM_URL/api/v1/import/prometheus?extra_label=job=fa-backup" >/dev/null 2>&1 \
        || log "metrics push to $VM_URL failed"
}

# Reads `key = "value"` from the [garage] section of the mounted secrets.toml.
garage_secret() {
    f=${FA_SECRETS:-/run/fa/secrets.toml}
    [ -f "$f" ] || return 1
    sed -n '/^\[garage\]/,/^\[/p' "$f" | sed -n "s/^$1 *= *\"\([^\"]*\)\".*/\1/p" | head -1
}

# rclone with the compose Garage as remote `garage:` (credentials from secrets.toml).
garage_rclone() {
    RCLONE_CONFIG_GARAGE_TYPE=s3 RCLONE_CONFIG_GARAGE_PROVIDER=Other \
    RCLONE_CONFIG_GARAGE_ENDPOINT=${FA_GARAGE_URL:-http://garage:3900} \
    RCLONE_CONFIG_GARAGE_REGION=${FA_GARAGE_REGION:-garage} RCLONE_CONFIG_GARAGE_FORCE_PATH_STYLE=true \
    RCLONE_CONFIG_GARAGE_ACCESS_KEY_ID=$(garage_secret access_key_id) \
    RCLONE_CONFIG_GARAGE_SECRET_ACCESS_KEY=$(garage_secret secret_access_key) \
        rclone "$@"
}

# Scheduled backup directories (manifest kind "scheduled"), oldest first.
scheduled_dirs() {
    for d in "$BACKUP_ROOT"/2*Z; do
        [ -f "$d/manifest.json" ] && grep -q '"kind": "scheduled"' "$d/manifest.json" && echo "$d"
    done | sort
}
