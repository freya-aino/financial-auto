# config

Runtime configuration: source catalog, platform config, timescale ladder. Secrets live in git-ignored `secrets.toml`.

**Status:** in use — `platform.toml` (host) and `platform.docker.toml` (compose hostnames) are read by every Rust service via `fa-core` (`FA_CONFIG`); `sources.yaml` is loaded by `go/pkg/catalog`, synced into Postgres `catalog.source` by the ingestor and served (with the Postgres status and live health overlaid) by control-api `/api/sources`; `secrets.toml` (git-ignored, documented by `secrets.example.toml`) is read by the ingestor (`[garage]`, `[sources.<id>]`, D-020).

## Source status precedence (spec 11 §8, D-054)
`status:` in `sources.yaml` is only the **seed**. The ingestor's catalog sync inserts new ids with the YAML status but never overwrites the status of an existing `catalog.source` row, and then applies the Postgres statuses to its in-memory catalog. `POST /api/sources/{id}/enable|disable` (→ `ctl.ingest.enable|disable`) write `live` / `disabled` to Postgres, so a disabled source stays disabled across restarts even if the YAML still says `live`; the response of `GET /api/sources` shows both (`status` = effective, `yaml_status` = seed). To change a status from the YAML again, update the row (`update catalog.source set status = ... where id = ...`) or disable/enable through the API.

## Environment variables shared by every service (spec 60)
| Variable | Default | Meaning |
|---|---|---|
| `FA_CONFIG` / `FA_SOURCES` / `FA_SECRETS` | `config/platform.toml`, `config/sources.yaml`, `config/secrets.toml` | file locations (compose uses `platform.docker.toml`) |
| `FA_METRICS_BIND` | `127.0.0.1:91xx` per service (table in `docker/README.md`); compose sets `0.0.0.0:9100` | bind of the Prometheus `GET /metrics` endpoint; `off` disables it (D-056). Go services also accept `-metrics` |
| `FA_BACKUP_BUCKETS`, `FA_BACKUP_NATS` | `fa-models fa-ckpt fa-artifacts fa-backup`, `0` | `scripts/backup.sh` bucket list / include JetStream streams |

## Owns
`sources.yaml`, `platform.toml`, `platform.docker.toml`, `secrets.example.toml`.

## Governing specs
- [11-source-catalog](../specs/11-source-catalog.md)
- [60-ops](../specs/60-ops.md)
