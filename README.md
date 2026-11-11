# financial-auto

Automated multi-asset trading platform. Real-time multimodal market, macro, on-chain and media data is embedded by per-timescale expert encoders into decorrelated **expert spaces**. **Regions** of those latent spaces form a Markov state graph with Bayesian transitions. A population of **decoders** and **strategies** evolves over that graph, and promoted strategies trade through a meta-allocator under hard risk limits.

**Status:** integrated platform, paper only. Every service runs in compose on real keyless feeds (Kraken, Yahoo, Frankfurter, ECB, SEC EDGAR, GDELT, Hacker News, Google News) plus the synthetic heartbeat; trades, bars and quotes land in VictoriaMetrics, raw payloads in Garage, reference data, macro/filings, the paper ledger, audit and ops history in Postgres, regions/transitions/lineage in HelixDB, and the bus has its JetStream streams (with caps and a dead-letter stream). Statistical baselines run behind every stage; neural encoder and decoder families (candle, CPU; D-068, D-070, D-073) train, evolve and serve with shadow promotion, but no neural snapshot meets the promotion checklist yet and the host has no GPU. Ops: vmalert alerting into control-api, exporters, a backup sidecar with PITR. No live broker: the broker-gateway is paper only. Start at [`specs/README.md`](specs/README.md); implementation decisions (D-001..D-080) are in [`specs/DECISIONS.md`](specs/DECISIONS.md).

## Quick start
Needs Rust and Docker. Go, Node and buf run in Docker when not installed. Install `just` with `cargo install just`.

```sh
just up                      # NATS, HelixDB, Postgres, VictoriaMetrics, Garage, vmalert, exporters, backup (compose.yaml)
just garage-init             # first time only: Garage layout, buckets and key
just check                   # Rust + Go + frontend build/lint/test
just fe-build                # build the UI, served by control-api
cargo run -p control-api     # http://127.0.0.1:7700 (applies Postgres migrations on start)
cargo run -p model-runtime   # likewise strategy-runtime, risk-engine, evo-worker
just dev                     # prints the host run order; `just app-up` runs everything in compose (UI :8080)
```

## Layout
| Path | What |
|---|---|
| `specs/` | Specifications, ADRs, maturity overview |
| `crates/` | Rust libraries (core, proto, bus, storage clients for Helix/Garage/VictoriaMetrics/Postgres, metrics, encoders, expert spaces, regions, MCMC, decoders, evolution, strategies, allocator, risk, ledger, backtest) |
| `bins/` | Rust services: `model-runtime`, `strategy-runtime`, `risk-engine`, `evo-worker`, `control-api`, `trainer` (CLI) |
| `go/` | Go: connector framework, connectors (heartbeat, Kraken, Yahoo, Frankfurter, ECB, SEC EDGAR, GDELT, Hacker News, Google News; Lemon and OANDA unverified without keys), `ingestor`, `broker-gateway`, `evo-orchestrator`, `healthprobe` (compose healthcheck of the Go images) |
| `proto/` | Protobuf contracts shared by Rust and Go |
| `helix/` | HelixDB schema and queries |
| `frontend/` | SvelteKit operator UI |
| `compose.yaml` | Everything runs through it: infra (NATS, HelixDB, VictoriaMetrics, PostgreSQL, Garage) and ops (`vmalert`, `nats-exporter`, `postgres-exporter`, `backup`) by default; all platform services and the `ui` service under profile `app`; the `trainer` CLI under profile `tools` |
| `docker/` | Dockerfiles for the Rust, Go and UI images, nginx and Garage config, Garage init script, VictoriaMetrics scrape config and vmalert rules, Postgres (WAL archiving) and backup sidecar images |
| `config/` | Source catalog, platform config (secrets in git-ignored `config/secrets.toml`, runtime overrides in `config/runtime.toml`) |
| `scripts/` | Backup, restore (incl. PITR), restore-drill, alert-drill and health-wait scripts (`just backup`, `just restore DIR`, `just restore-drill`, `just alert-drill`, `just healthwait`) |
| `migrations/` | PostgreSQL schemas: catalog, ref, ingest, wh, ledger, audit, validation, ops |

Every component directory has a `README.md` linking to the specs that govern it.
