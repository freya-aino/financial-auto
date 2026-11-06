# financial-auto

Automated multi-asset trading platform. Real-time multimodal market, macro, on-chain and media data is embedded by per-timescale expert encoders into decorrelated **expert spaces**. **Regions** of those latent spaces form a Markov state graph with Bayesian transitions. A population of **decoders** and **strategies** evolves over that graph, and promoted strategies trade through a meta-allocator under hard risk limits.

**Status:** specification phase. No code yet. Start at [`specs/README.md`](specs/README.md).

## Layout
| Path | What |
|---|---|
| `specs/` | Specifications, ADRs, maturity overview |
| `crates/` | Rust libraries (core, storage clients, ML, evolution, risk, ledger, backtest) |
| `bins/` | Rust services: `model-runtime`, `trainer`, `evo-worker`, `control-api` |
| `go/` | Go: connector framework, connectors, `ingestor`, `broker-gateway`, `evo-orchestrator` |
| `proto/` | Protobuf contracts shared by Rust and Go |
| `helix/` | HelixDB schema and queries |
| `migrations/` | PostgreSQL warehouse + ledger schema |
| `frontend/` | SvelteKit operator UI |
| `deploy/` | Local compose (NATS, HelixDB, VictoriaMetrics, PostgreSQL, Garage) |
| `config/` | Source catalog, platform config (secrets in git-ignored `config/secrets.toml`) |

Every component directory has a `README.md` linking to the specs that govern it.
