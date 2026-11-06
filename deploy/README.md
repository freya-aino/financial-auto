# deploy

Local deployment: compose for NATS, VictoriaMetrics, PostgreSQL and Garage. HelixDB runs via its own CLI (D-012).

**Status:** compose working (verified 2026-11-06: nats, postgres, victoriametrics healthy; garage up, initialized by `garage-init.sh`).

## Services (ports per D-014, bound to 127.0.0.1)
| Service | Image | Ports | Health |
|---|---|---|---|
| nats | `nats:2.12-alpine` (`-js`) | 4222, monitor 8222 | `curl localhost:8222/healthz` |
| postgres | `postgres:17-alpine` (user/pass/db `fa`) | 5432 | `pg_isready` |
| victoriametrics | `victoriametrics/victoria-metrics:v1.155.0` | 8428 | `curl localhost:8428/health` |
| garage | `dxflrs/garage:v2.5.0` | S3 3900, admin 3903 | `docker compose exec garage /garage status` |
| HelixDB | via `helix` CLI | 6969 | see below |

## Usage
```sh
just up        # docker compose -f deploy/compose.yaml up -d
just migrate   # apply migrations/ to postgres
just down
```

## Garage first-time init
Garage needs a cluster layout, buckets and an access key before use. The script is idempotent:
```sh
deploy/garage-init.sh
```
It does the following:
```sh
G() { docker compose -f deploy/compose.yaml exec -T garage /garage "$@"; }
G layout assign -z local -c 50G "$(G node id -q | cut -d@ -f1)"
G layout apply --version 1
G bucket create fa-raw   # likewise: fa-media fa-models fa-ckpt fa-artifacts fa-backup (spec 20 §5.5)
G key create fa-local
G bucket allow --read --write --owner fa-raw --key fa-local   # same for every bucket
G key info fa-local --show-secret   # copy into config/secrets.toml [garage]
```
`garage.toml` holds a fixed local-only `rpc_secret` and `admin_token` (ADR 0006). Don't reuse them anywhere else.

## HelixDB (not in compose, D-012)
HelixDB builds a project-specific container from `helix/` (schema + queries), so it's driven by its CLI:
```sh
curl -sSL https://install.helix-db.com | bash   # installs the `helix` CLI
cd helix
helix check          # validate schema.hx / queries.hx
helix push dev       # build + run locally (HTTP on 6969)
helix status
```
Command names follow the HelixDB docs at the time of writing. They're unverified here because the CLI isn't installed yet.

## Data
All state lives in named volumes (`financial-auto_{nats,postgres,vm,garage-meta,garage-data}-data`). `docker compose down -v` wipes everything.
