# scripts

Operational shell scripts for spec 60 (D-056). All of them drive the compose stack from the repo root.

| Script | Purpose |
|---|---|
| `backup.sh [--dest DIR] [--skip parts]` | Postgres `pg_dump -Fc`, HelixDB volume tar (service paused for seconds), Garage critical buckets via rclone, optional NATS stream backup, `manifest.json` with sha256 — into `backups/<timestamp>/`. `just backup`. |
| `restore.sh DIR [--yes] [--only parts] [--postgres-db NAME]` | Reverse of the above in the §6.4 recovery order, with confirmation; `--postgres-db` restores only Postgres into a scratch database for drills. `just restore DIR`. |
| `healthwait.sh [timeout]` | Waits infra → control-api `/api/health` → risk-engine heartbeat, printing the §6.4 steps; called by `just app-up`. |
| `lib.sh` | Shared helpers (compose wrapper, volume/network names, Garage credentials, rclone runner). |

Cron line for the nightly backup (`backup.schedule` 03:00): `0 3 * * * cd /path/to/financial-auto && scripts/backup.sh >> backups/backup.log 2>&1`.

## Governing specs
- [60-ops](../specs/60-ops.md) §6.4, §6.6
- [20-storage](../specs/20-storage.md) §6.6
