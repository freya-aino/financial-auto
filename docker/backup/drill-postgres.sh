#!/bin/sh
# Postgres restore drill (spec 60 §6.6): proves a backup directory restores, without touching the
# live server. Everything runs in throwaway clusters under a private temp dir, on unix sockets only
# (listen_addresses=''), as the unprivileged `postgres` user; the backup and the WAL archive are
# read-only inputs. Two checks:
#   logical  initdb → pg_restore postgres/fa.dump → every table of row_counts.tsv present, non-empty
#            tables non-empty, counts reported (the dump and the counts are seconds apart)
#   pitr     postgres/base/base.tar.gz + pg_wal.tar.gz + restore_command over the WAL archive →
#            recovery to FA_DRILL_TARGET (default: end of archived WAL), promoted, queried; reports
#            WAL files replayed and the last replayed commit time (proves replay past the base)
# Usage: drill-postgres.sh <backup-dir> [logical,pitr]     (runs as root, drops to postgres)
# Env:   FA_DRILL_TARGET='2026-11-11 03:00:00+00' (recovery_target_time), FA_WAL_ARCHIVE_DIR
# Output: a summary on stdout, "DRILL OK" / "DRILL FAILED" as the last line; exit status likewise.
set -eu
. /usr/local/lib/fa-backup/common.sh
SRC=${1:?usage: drill-postgres.sh <backup-dir> [logical,pitr]}
PARTS=${2:-logical,pitr}
want() { case ",$PARTS," in *",$1,"*) return 0 ;; esac; return 1; }
work=$(mktemp -d /tmp/fa-drill.XXXXXX)
chown postgres:postgres "$work"
as_pg() { gosu postgres "$@"; }
cleanup() {
    rc=$?
    [ $rc = 0 ] || [ "${finished:-0}" = 1 ] || { echo "drill: aborted (exit $rc)"; echo 'fa_restore_drill_last_status 0' | push_metrics; echo "DRILL FAILED"; }
    for d in logical pitr; do [ -f "$work/$d/postmaster.pid" ] && as_pg pg_ctl -D "$work/$d" -m immediate stop >/dev/null 2>&1; done
    rm -rf "$work"
}
trap cleanup EXIT
ok=1
note() { echo "drill: $*"; }
bad() { echo "drill: FAIL $*"; ok=0; }
sql() { as_pg psql -h "$work" -p "$1" -U "$2" -d "$3" -Atc "$4"; }

compare_counts() { # port user db
    [ -f "$SRC/postgres/row_counts.tsv" ] || { note "no row_counts.tsv (manual backup); table count only"; return 0; }
    tables=0 diff=0
    while IFS="$(printf '\t')" read -r t want_n; do
        tables=$((tables + 1))
        got=$(sql "$1" "$2" "$3" "select count(*) from $t" 2>/dev/null) || { bad "table $t missing"; continue; }
        if [ "$want_n" -gt 0 ] && [ "$got" -eq 0 ]; then bad "table $t empty (backup had $want_n)"; fi
        [ "$got" = "$want_n" ] || diff=$((diff + 1))
    done < "$SRC/postgres/row_counts.tsv"
    note "  $tables tables present, $diff with a different row count than the backup's count"
}

if want logical && [ -f "$SRC/postgres/fa.dump" ]; then
    note "logical: initdb + pg_restore fa.dump into a scratch cluster"
    as_pg initdb -D "$work/logical" -U fa --auth=trust -E UTF8 --locale=C >"$work/initdb.log" 2>&1
    as_pg pg_ctl -D "$work/logical" -l "$work/logical.log" -w -o "-p 55432 -k $work -c listen_addresses=''" start >/dev/null
    sql 55432 fa postgres "create database fa_drill" >/dev/null
    if as_pg pg_restore -h "$work" -p 55432 -U fa -d fa_drill --no-owner --no-privileges --exit-on-error "$SRC/postgres/fa.dump"; then
        n=$(sql 55432 fa fa_drill "select count(*) from information_schema.tables where table_schema not in ('pg_catalog','information_schema')")
        note "  restored: $n tables (backup: $(cat "$SRC/postgres/table_count.txt" 2>/dev/null || echo '?'))"
        compare_counts 55432 fa fa_drill
    else
        bad "pg_restore failed"
    fi
    as_pg pg_ctl -D "$work/logical" -m fast stop >/dev/null
fi

if want pitr && [ -f "$SRC/postgres/base/base.tar.gz" ]; then
    d="$work/pitr"
    note "pitr: base backup $(sed -n 's/^START TIME: //p' "$SRC/postgres/base/backup_label.txt" 2>/dev/null) + WAL archive $WAL_DIR"
    mkdir -m 700 "$d" # extracted as root (pg_basebackup writes 0600 files), then handed to postgres
    tar xzf "$SRC/postgres/base/base.tar.gz" -C "$d"
    [ -f "$SRC/postgres/base/pg_wal.tar.gz" ] && tar xzf "$SRC/postgres/base/pg_wal.tar.gz" -C "$d/pg_wal"
    chown -R postgres:postgres "$d"
    {
        echo "# fa restore drill"
        echo "restore_command = 'gunzip -c $WAL_DIR/%f.gz > %p'"
        echo "recovery_target_action = 'promote'"
        [ -n "${FA_DRILL_TARGET:-}" ] && echo "recovery_target_time = '$FA_DRILL_TARGET'"
        echo "archive_mode = off"
    } | as_pg tee -a "$d/postgresql.auto.conf" >/dev/null
    as_pg touch "$d/recovery.signal"
    if as_pg pg_ctl -D "$d" -l "$work/pitr.log" -w -t 900 -o "-p 55433 -k $work -c listen_addresses='' -c hba_file=$d/pg_hba.conf" start >/dev/null; then
        i=0
        while [ "$(sql 55433 fa postgres 'select pg_is_in_recovery()' 2>/dev/null)" != f ] && [ $i -lt 300 ]; do sleep 1; i=$((i + 1)); done
        if [ "$(sql 55433 fa postgres 'select pg_is_in_recovery()')" = f ]; then
            replayed=$(grep -c 'restored log file' "$work/pitr.log" || true)
            last=$(sed -n 's/.*last completed transaction was at log time \(.*\)$/\1/p' "$work/pitr.log" | tail -1)
            note "  promoted: $replayed archived WAL files replayed, last replayed commit at ${last:-unknown}"
            compare_counts 55433 fa fa
            [ "$replayed" -gt 0 ] || note "  (no archived WAL past the base backup yet: archive_timeout not elapsed)"
        else
            bad "pitr cluster still in recovery after 300 s"; tail -20 "$work/pitr.log"
        fi
        as_pg pg_ctl -D "$d" -m fast stop >/dev/null
    else
        bad "pitr cluster did not start"; tail -30 "$work/pitr.log"
    fi
elif want pitr; then
    note "pitr: no postgres/base in $SRC (manual pg_dump-only backup), skipped"
fi

finished=1
if [ $ok = 1 ]; then
    printf 'fa_restore_drill_last_success_timestamp_seconds %s\nfa_restore_drill_last_status 1\n' "$(date +%s)" | push_metrics
    echo "DRILL OK"
else
    echo 'fa_restore_drill_last_status 0' | push_metrics
    echo "DRILL FAILED"; exit 1
fi
