# config

Runtime configuration: source catalog, platform config (incl. the timescale ladder `timescales` D-011, the store backends `[qdrant]` D-101 / `[graph]` D-108 and the organisational-books split `[accounting]` D-104/D-106). Secrets live in git-ignored `secrets.toml`, operator overrides in git-ignored `runtime.toml`.

**Status:** in use — `platform.toml` (host) and `platform.docker.toml` (compose hostnames) are read by every Rust service via `fa-core` (`FA_CONFIG`); `runtime.toml` (git-ignored) is written next to the platform file by `PUT /api/config/runtime` and merged over its `[runtime]` section on load (spec 50 §4.1); `sources.yaml` is loaded by `go/pkg/catalog`, synced into Postgres `catalog.source` by the ingestor (re-read on change and by `ctl.ingest.reload`; runtime `rate_limit` / `config` overrides from `PATCH /api/sources/{id}` are kept in Postgres and win over the file, like the status) and served (with the Postgres status and live health overlaid) by control-api `/api/sources`; its per-source `knowledge_delay_secs` (Kraken 120 s, Yahoo 100 s) shift event-time replays in `fa_backtest::knowledge` (D-079). `secrets.toml` (git-ignored, documented by `secrets.example.toml`; `docker/garage-init.sh` prints the Garage key to put under `[garage]`, and imports an existing `[garage]` key into a fresh Garage so ephemeral stacks keep working credentials, D-096) is read by the ingestor and evo-orchestrator (`[garage]`, `[sources.<id>]`) and by every Rust service through `fa_core::secrets` (`FA_SECRETS`, `FA_GARAGE_*` / `FA_<SOURCE>_<KEY>` env overrides, values redacted in logs; D-020, D-045); compose mounts it read-only into the services and the `backup` sidecar.

Backends in the platform tomls: `[qdrant] url, vector_backend = "qdrant"|"helix"` picks the vector store for latents, news and media embeddings (ADR 0010, D-101; applied by `fa_hub::configure_vectors`), `[graph] backend = "postgres"|"helix"` the model-state graph store (schema `graph` on the platform Postgres by default, D-108; `fa_hub::configure_graph`) — both validated at load, env-overridable below. `[accounting]` (D-104) holds the revenue-split fractions and notional opex prices of the organisational books, owner-confirmed at 20/30/0 and `base_eur_per_day 1.0` / `gpu_eur_per_hour 0.5` / `storage_eur_per_gib_month 0.02` (D-106); control-api's daily close reads it.

## Source status precedence (spec 11 §8, D-054)
`status:` in `sources.yaml` is only the **seed**. The ingestor's catalog sync inserts new ids with the YAML status but never overwrites the status of an existing `catalog.source` row, and then applies the Postgres statuses to its in-memory catalog. `POST /api/sources/{id}/enable|disable` (→ `ctl.ingest.enable|disable`) write `live` / `disabled` to Postgres, so a disabled source stays disabled across restarts even if the YAML still says `live`; the response of `GET /api/sources` shows both (`status` = effective, `yaml_status` = seed). To change a status from the YAML again, update the row (`update catalog.source set status = ... where id = ...`) or disable/enable through the API.

## Environment variables shared by every service (spec 60)
| Variable | Default | Meaning |
|---|---|---|
| `FA_CONFIG` / `FA_SOURCES` / `FA_SECRETS` | `config/platform.toml`, `config/sources.yaml`, `config/secrets.toml` | file locations (compose uses `platform.docker.toml`) |
| `FA_METRICS_BIND` | `127.0.0.1:91xx` per service (table in `docker/README.md`); compose sets `0.0.0.0:9100` | bind of the Prometheus `GET /metrics` endpoint; `off` disables it (D-056). Go services also accept `-metrics` |
| `FA_BACKUP_BUCKETS`, `FA_BACKUP_NATS` | `fa-models fa-ckpt fa-artifacts fa-backup`, `0` | `scripts/backup.sh` bucket list / include JetStream streams |
| `FA_MARKET_DATA` | `real` (compose; `scripts/eph.sh` sets `synthetic`) | which market data the paper-trading path acts on (D-103): `real` drops the synthetic venues (`sim`) in model-runtime, strategy-runtime, risk-engine and the paper broker; `synthetic` the inverse; `all` both. Ingestion stays neutral |
| `FA_QDRANT_URL`, `FA_VECTOR_BACKEND` | `[qdrant]` values | override the vector store url / backend (`qdrant`\|`helix`, D-101) |
| `FA_GRAPH_BACKEND` | `[graph].backend` (`postgres`) | override the model-state graph backend (`postgres`\|`helix`, D-108) |

## Configuration kept elsewhere
| File | Read by |
|---|---|
| `bins/risk-engine/risk.toml` | risk-engine (`FA_RISK`): limits, allocator, lifecycle pipeline, FX, rebalance, exec |
| `go/services/broker-gateway/costs.yaml` | broker-gateway, risk-engine, evo-worker (`FA_COSTS`): venue cost model |
| `crates/fa-strategies/baselines.toml` | strategy-runtime: hand-written baseline population |
| `docker/vm-scrape.yaml`, `docker/vm-alerts.yaml` | VictoriaMetrics scrape targets, vmalert rules (D-056, D-066) |

## Owns
`sources.yaml`, `platform.toml`, `platform.docker.toml`, `secrets.example.toml`; the git-ignored `secrets.toml` and `runtime.toml` are local.

## Governing specs
- [Source catalog reference (ex spec 11)](../go/pkg/catalog/README.md#source-catalog-reference)
- [Ops reference (ex spec 60)](../docker/README.md#reference-spec-60-ops-retired-2026-11-11)
