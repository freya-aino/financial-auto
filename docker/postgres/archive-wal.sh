#!/bin/sh
# Postgres archive_command (spec 60 §6.6): `sh /etc/postgresql/fa/archive-wal.sh %p %f`.
# Copies one finished WAL segment (or .history/.backup file) gzip-compressed into /wal-archive
# (named volume pg-wal-archive, shared read-only with the backup sidecar and the restore drill).
# Contract (PostgreSQL docs, "Setting Up WAL Archiving"): exit 0 only once the file is durably
# archived; never overwrite an existing archive file unless it is identical (then succeed, so a
# retry after a crash between copy and status update is harmless).
set -eu
src=$1 name=$2
dir=${FA_WAL_ARCHIVE_DIR:-/wal-archive}
dst="$dir/$name.gz"
if [ -f "$dst" ]; then
    if gunzip -c "$dst" | cmp -s - "$src"; then exit 0; fi
    echo "archive-wal: $dst exists with different content" >&2
    exit 1
fi
tmp="$dst.tmp.$$"
gzip -c "$src" > "$tmp"
sync "$tmp" 2>/dev/null || sync
mv "$tmp" "$dst"
