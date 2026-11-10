# financial-auto

Automated multi-asset trading platform. Real-time multimodal market, macro, on-chain and media data is embedded by per-timescale expert encoders into decorrelated **expert spaces**. **Regions** of those latent spaces form a Markov state graph with Bayesian transitions. A population of **decoders** and **strategies** evolves over that graph, and promoted strategies trade through a meta-allocator under hard risk limits.

**Status:** integrated baseline. Every service runs in compose on real keyless feeds (Yahoo bars, Hacker News, Google News) plus the synthetic heartbeat; trades, bars and quotes land in VictoriaMetrics, raw payloads in Garage, reference data and the paper ledger in Postgres, regions/transitions/latents in HelixDB, and the bus has its JetStream streams. Evolution runs multi-generation NSGA-II over baseline CPU decoders. The ML substance is still statistical baselines: no neural encoders/decoders, no trainer, no live broker. Start at [`specs/README.md`](specs/README.md); implementation decisions (D-001..D-050) are in [`specs/DECISIONS.md`](specs/DECISIONS.md).

## Quick start
Needs Rust and Docker. Go, Node and buf run in Docker when not installed. Install `just` with `cargo install just`.

```sh
just up                      # NATS, HelixDB, Postgres, VictoriaMetrics, Garage (compose.yaml)
just check                   # Rust + Go + frontend build/lint/test
just fe-build                # build the UI, served by control-api
cargo run -p control-api     # http://127.0.0.1:7700 (applies Postgres migrations on start)
cargo run -p model-runtime   # likewise strategy-runtime, risk-engine, evo-worker
just dev                     # prints the host run order; `just app-up` runs everything in compose
```

## Layout
| Path | What |
|---|---|
| `specs/` | Specifications, ADRs, maturity overview |
| `crates/` | Rust libraries (core, storage clients, ML, evolution, risk, ledger, backtest) |
| `bins/` | Rust services: `model-runtime`, `strategy-runtime`, `risk-engine`, `evo-worker`, `control-api`, `trainer` (CLI) |
| `go/` | Go: connector framework, connectors, `ingestor`, `broker-gateway`, `evo-orchestrator` |
| `proto/` | Protobuf contracts shared by Rust and Go |
| `helix/` | HelixDB schema and queries |
| `migrations/` | PostgreSQL warehouse + ledger schema |
| `frontend/` | SvelteKit operator UI |
| `compose.yaml` | Everything runs through it: infra (NATS, HelixDB, VictoriaMetrics, PostgreSQL, Garage) + all platform services and the `ui` service under profile `app` |
| `docker/` | Dockerfiles for the Rust, Go and UI images, nginx and Garage config, Garage init script |
| `config/` | Source catalog, platform config (secrets in git-ignored `config/secrets.toml`) |

Every component directory has a `README.md` linking to the specs that govern it.
