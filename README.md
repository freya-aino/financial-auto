# financial-auto

Automated multi-asset trading platform. Real-time multimodal market, macro, on-chain and media data is embedded by per-timescale expert encoders into decorrelated **expert spaces**. **Regions** of those latent spaces form a Markov state graph with Bayesian transitions. A population of **decoders** and **strategies** evolves over that graph, and promoted strategies trade through a meta-allocator under hard risk limits.

**Status:** integrated platform, paper only. Every service runs in compose; paper trading runs on real keyless feeds by default (Kraken, Yahoo, Frankfurter, ECB, SEC EDGAR, GDELT, Hacker News, Google News; `FA_MARKET_DATA`, D-103), with a local synthetic market for development and each test run on its own ephemeral stack (D-096). Trades, bars and quotes land in VictoriaMetrics, raw payloads in Garage, latent/news/media vectors in Qdrant (D-101), the model-state graph in Postgres behind a `GraphStore` trait (D-108; genomes, fitness and the graph explorer still live in HelixDB — the 2026-11-16 Dgraph cutover was rolled back the same day, D-112/ADR 0011), reference data, macro/filings, the paper ledger, the organisational books (fa-accounting with the confirmed 20/30/0 split, D-104/D-106), audit and ops history in Postgres, and the bus has its JetStream streams (caps, a dead-letter stream, durable live consumers in every service, D-097). Statistical baselines run behind every stage; neural encoder and decoder families (candle, CPU) train end to end on reconstruction with an expert-diversity term (D-099), evolve and serve with shadow promotion — 12/12 synthetic runs pass the tightened checklist (D-102) and the first live 1m generation passes checklist item 2 on real bars (D-118, rejected on `carried_mass` / `mcmc_ece`, which need a neural parent), promotion is still the owner's, and the host has no GPU. The risk engine ticks deterministically with Postgres off the decision path (D-105); its full-day replay re-verification is pending. Ops: vmalert alerting into control-api, exporters, scheduled storage maintenance, a backup sidecar with PITR (incl. Qdrant snapshots). No live broker: the broker-gateway is paper only. Start at [`specs/README.md`](specs/README.md); implementation decisions (D-001..D-115) are in [`specs/DECISIONS.md`](specs/DECISIONS.md).

## Quick start
Needs Rust and Docker. Go, Node and buf run in Docker when not installed. Install `just` with `cargo install just`.

```sh
just up                      # NATS, HelixDB, Qdrant, Postgres, VictoriaMetrics, Garage, vmalert, exporters, backup (compose.yaml)
just garage-init             # first time only: Garage layout, buckets and key
just check                   # Rust + Go + frontend build/lint/test (toolchains run in Docker; never cargo on the host)
just fe-build                # build the UI, served by control-api
just control-api             # http://127.0.0.1:7700 (applies Postgres migrations on start)
just cargo run -p model-runtime   # likewise strategy-runtime, risk-engine, evo-worker
just dev                     # prints the host run order; `just app-up` runs everything in compose (UI :8080)
just eph-up t1 && just eph-down t1   # ephemeral test stack on the synthetic market (D-096)
```

## Layout
| Path | What |
|---|---|
| `specs/` | Specifications, ADRs, maturity overview |
| `crates/` | Rust libraries (core, proto, bus, storage clients for Qdrant/Helix/Garage/VictoriaMetrics/Postgres, metrics, the OpenAI-compatible LLM client, encoders, expert spaces, regions, MCMC, decoders, evolution, strategies, allocator, risk, ledger, accounting, backtest) |
| `bins/` | Rust services: `model-runtime`, `strategy-runtime`, `risk-engine`, `evo-worker`, `control-api`, `llm-suite` (news embeddings + operator agent, D-117), `trainer` (CLI + the `trainer-worker` service, D-113) |
| `go/` | Go: connector framework, connectors (heartbeat, synthetic market, Kraken, Yahoo, Frankfurter, ECB, SEC EDGAR, GDELT, Hacker News, Google News, generic RSS; Lemon and OANDA unverified without keys), `ingestor`, `broker-gateway`, `evo-orchestrator`, `vm-rebuild` (VictoriaMetrics rebuild drill from the raw archive), `healthprobe` (compose healthcheck of the Go images) |
| `proto/` | Protobuf contracts shared by Rust and Go |
| `helix/` | HelixDB schema and queries |
| `frontend/` | SvelteKit operator UI |
| `compose.yaml` | Everything runs through it: infra (NATS, HelixDB, Qdrant, VictoriaMetrics, PostgreSQL, Garage) and ops (`vmalert`, `nats-exporter`, `postgres-exporter`, `backup`) by default; all platform services and the `ui` service under profile `app`; the `trainer` CLI under profile `tools` |
| `docker/` | Dockerfiles for the Rust, Go and UI images, nginx and Garage config, Garage init script, VictoriaMetrics scrape config and vmalert rules, Postgres (WAL archiving) and backup sidecar images |
| `config/` | Source catalog, platform config incl. the `[accounting]` split and opex prices (secrets in git-ignored `config/secrets.toml`, runtime overrides in `config/runtime.toml`) |
| `scripts/` | Backup, restore (incl. PITR), restore-drill, alert-drill, health-wait and ephemeral-stack scripts (`just backup`, `just restore DIR`, `just restore-drill`, `just alert-drill`, `just healthwait`, `just eph-up\|eph-down\|eph-ls`) |
| `migrations/` | PostgreSQL schemas: catalog, ref, ingest, wh, ledger, acct, audit, validation, ops, graph |

Every component directory has a `README.md` linking to the specs that govern it.
