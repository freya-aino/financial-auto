# scripts

Operational shell scripts for spec 60 (D-056). All of them drive the compose stack from the repo root. The scheduled side (nightly backup, WAL status, weekly Postgres drill) lives in the `backup` compose sidecar (`docker/backup/`), not in host cron.

| Script | Purpose |
|---|---|
| `backup.sh [--dest DIR] [--skip parts] [--no-stop-helix]` | On-demand backup: Postgres `pg_dump -Fc`, HelixDB volume tar (service stopped for seconds → cold, consistent copy), Garage critical buckets via rclone, optional NATS stream backup, `manifest.json` with sha256 — into `backups/<timestamp>/`. `just backup`. The nightly scheduled backup is the sidecar's `run.sh` (`just backup-now` runs it at once). |
| `restore.sh DIR [--yes] [--only parts] [--postgres-db NAME] [--pitr [--target T]] [--pg-volume NAME]` | Restore in the §6.4 recovery order, with confirmation. `--postgres-db` restores the dump into a scratch database; `--pitr` does point-in-time recovery from a scheduled backup's base backup + the `pg-wal-archive` volume (saves the current data directory to `backups/pre-restore-<ts>/` first); `--pg-volume` runs the PITR into a scratch volume and checks it in a throwaway container (live untouched); `--only raw` copies the `backups/raw-mirror/` fa-raw mirror back. `just restore DIR`. |
| `restore-drill.sh [DIR] [--only parts] [--target T] [--switch-wal]` | Restore drill without touching live data: manifest checksums; Postgres logical + PITR restore in a throwaway backup-image container (WAL archive mounted read-only); HelixDB boot check on the restored volume in a scratch container (`--network none`, anonymous volume); Garage object counts. Pushes `fa_restore_drill_*` to VictoriaMetrics. `just restore-drill`. |
| `alert-drill.sh [--timeout S] [--no-resolve]` | End-to-end alerting test: writes `fa_alert_drill{drill_id}` into VictoriaMetrics, waits for the firing `ops.alert` message on NATS (rule `AlertDrill` → vmalert → control-api), checks `GET /api/alerts` and the `ops.alert.firing` audit row, then waits for the resolved message. `just alert-drill`. |
| `healthwait.sh [timeout]` | Waits infra → control-api `/api/health` → risk-engine heartbeat, printing the §6.4 steps; called by `just app-up`. |
| `lib.sh` | Shared helpers (compose wrapper, volume/network names, Garage credentials, rclone runner, `verify_manifest`, `latest_backup`). |

## PITR runbook (spec 60 §6.6)
1. Pick the scheduled backup whose base backup is **before** the target time: `ls backups/` (scheduled ones have `postgres/base/`). WAL is kept back to the oldest kept base backup (7 by default).
2. Rehearse first, live data untouched: `scripts/restore-drill.sh backups/<ts> --only postgres --target '2026-11-11 03:00:00+00'` (or `scripts/restore.sh backups/<ts> --pitr --target … --pg-volume fa-pitr-scratch --yes`).
3. Real restore: `scripts/restore.sh backups/<ts> --pitr --target '2026-11-11 03:00:00+00'` — stops the app services and postgres, saves the current data directory to `backups/pre-restore-<ts>/postgres-data.tar.gz`, lays down the base backup, replays archived WAL up to the target and promotes (new timeline; archiving continues on it).
4. `just app-up` (migrations, reconciliation, health wait). Without `--target` the replay runs to the end of the archived WAL (at most `archive_timeout` = 5 min behind the crash).

## Governing specs
- [60-ops](../specs/60-ops.md) §6.4, §6.5, §6.6
- [20-storage](../specs/20-storage.md) §6.6
