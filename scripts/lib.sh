# Shared helpers for scripts/*.sh (spec 60 §6.4/§6.6, D-056). Source, don't execute.
# shellcheck shell=bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE=(docker compose -f "$ROOT/compose.yaml")

compose() { "${COMPOSE[@]}" "$@"; }

# Compose project name (top-level `name:` of compose.yaml) → volume/network prefixes.
project_name() { compose config 2>/dev/null | sed -n 's/^name: *//p' | head -1; }

# Docker volume name for a compose volume key (e.g. helix-data → financial-auto_helix-data).
volume_name() { echo "$(project_name)_$1"; }
network_name() { echo "$(project_name)_default"; }

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "error: $*"; exit 1; }

# Reads `key = "value"` from the [garage] section of config/secrets.toml (D-020).
garage_secret() {
    local key=$1 file="${FA_SECRETS:-$ROOT/config/secrets.toml}"
    [ -f "$file" ] || return 1
    sed -n '/^\[garage\]/,/^\[/p' "$file" | sed -n "s/^$key *= *\"\([^\"]*\)\".*/\1/p" | head -1
}

sha256_of() { sha256sum "$1" | cut -d' ' -f1; }
size_of() { stat -c %s "$1"; }

# rclone against the compose Garage (S3) through a throwaway container; args after `--` go to rclone.
# Usage: garage_rclone <host-dir> -- sync garage:fa-models /backup/fa-models
garage_rclone() {
    local dir=$1; shift; [ "$1" = "--" ] && shift
    local ak sk
    ak=$(garage_secret access_key_id) || die "config/secrets.toml has no [garage] access_key_id"
    sk=$(garage_secret secret_access_key) || die "config/secrets.toml has no [garage] secret_access_key"
    [ -n "$ak" ] && [ -n "$sk" ] || die "[garage] credentials in config/secrets.toml are empty (run docker/garage-init.sh)"
    docker run --rm --network "$(network_name)" -v "$dir":/backup \
        -e RCLONE_CONFIG_GARAGE_TYPE=s3 -e RCLONE_CONFIG_GARAGE_PROVIDER=Other \
        -e RCLONE_CONFIG_GARAGE_ENDPOINT="${FA_BACKUP_GARAGE_ENDPOINT:-http://garage:3900}" \
        -e RCLONE_CONFIG_GARAGE_REGION="${FA_GARAGE_REGION:-garage}" -e RCLONE_CONFIG_GARAGE_FORCE_PATH_STYLE=true \
        -e RCLONE_CONFIG_GARAGE_ACCESS_KEY_ID="$ak" -e RCLONE_CONFIG_GARAGE_SECRET_ACCESS_KEY="$sk" \
        "${RCLONE_IMAGE:-rclone/rclone:1.69}" "$@"
}

# Verifies every file of a backup directory against its manifest.json (sha256); returns 1 on any
# missing or mismatching file. Used by restore.sh and restore-drill.sh before anything is restored.
verify_manifest() {
    local src=$1 bad=0 n=0 path sum
    [ -f "$src/manifest.json" ] || { log "$src has no manifest.json"; return 1; }
    while IFS=$'\t' read -r path sum; do
        n=$((n + 1))
        [ -f "$src/$path" ] || { log "missing $path"; bad=1; continue; }
        [ "$(sha256_of "$src/$path")" = "$sum" ] || { log "checksum mismatch: $path"; bad=1; }
    done < <(sed -n 's/.*"path": "\([^"]*\)".*"sha256": "\([0-9a-f]*\)".*/\1\t\2/p' "$src/manifest.json")
    log "manifest: $n files checked"
    [ $bad = 0 ]
}

# Newest backup directory under backups/ that has a manifest (optionally only scheduled ones).
latest_backup() {
    local d
    for d in $(ls -1d "$ROOT"/backups/2*Z 2>/dev/null | sort -r); do
        [ -f "$d/manifest.json" ] || continue
        [ "${1:-}" = scheduled ] && ! grep -q '"kind": "scheduled"' "$d/manifest.json" && continue
        echo "$d"; return 0
    done
    return 1
}

BACKUP_IMAGE=${FA_BACKUP_IMAGE:-financial-auto/backup:dev}
