# strategy-runtime

Live service: runs a population of GP strategies over `md.trade.>`, the latent graph (`latent.state.>`, `latent.region.>`, `latent.transition.>`) and decoder outputs (`decoder.output.>`), and publishes `ExposureDistribution` on `signal.exposure.<strategy_id>`.

**Status:** running — baseline population (hand-written, **not evolved**) from [`crates/fa-strategies/baselines.toml`](../../crates/fa-strategies/baselines.toml), including two programs on the spec 35 §4.2 graph primitives; graph and decoder primitives read the real bus data for the strategy's timescale/instrument and fall back to price statistics otherwise; hot reload and pause/resume over `ctl.strategy.*`; heartbeat `svc.heartbeat.strategy-runtime` with population, feed and per-strategy gauges.

## How it works
- Each strategy declares `timescale`, `steps` (horizon in bars), `lookback_bars` and an optional `step_secs` cadence (default: one bar). Trades are resampled into bars at the strategy's timescale; the GP program is evaluated per instrument through `fa_strategies::features::LiveContext`: a point-in-time `graph::GraphAt` view of the latent graph at the strategy's timescale, a `decoders::DecodersAt` view of the instrument's decoder outputs (recency `--decoder-max-age-secs`, default 6 h) and the price-only `FallbackContext`. Primitives only see messages with event time ≤ the step.
- Closed bars are fed back into the graph (`LatentGraph::record_close`) so `region_feature(r, mean_fwd_return|vol|mass)` accumulates realised returns per region online.
- Portfolio primitives read the strategy's own virtual book (last emitted exposure, steps in trade, unrealised return of that exposure).
- Outputs are clipped to `mean ∈ [-1, 1]`, `stddev ∈ [0, 1]`; NaN → 0 exposure (spec 35 §8). A **paused** strategy keeps stepping but emits `mean = 0, stddev = 1` so the allocator flattens it.
- Deterministic core in `src/runtime.rs` (tested: cadence, graph-fed primitives, region returns, reload swap, pause); I/O in `src/main.rs`.

## Control (`ctl.strategy.<verb>`, `CtlRequest` → `CtlReply`)
| Verb | Args | Effect |
|---|---|---|
| `reload` | `artifact_key` (Garage key under `fa-artifacts/`, e.g. `strategies/strategies.3.toml`, written by the evo pipeline) or `path` (local file) | loads the population with `fa_blob::S3BlobStore::from_config` (credentials from `fa_core::Secrets::load()`), validates every program, swaps the population **atomically** (invalid file → running population untouched, reply `invalid: …`). Strategies whose id survives keep their virtual book and cadence. |
| `pause` / `resume` | `strategy_id` | flatten / un-flatten one strategy (`not_found: …` for unknown ids) |

```sh
cargo run -p strategy-runtime --example ctl -- pause strategy_id=baseline-momentum-1s
cargo run -p strategy-runtime --example ctl -- reload artifact_key=strategies/strategies.3.toml
```

## Heartbeat counters
`strategies`, `paused`, `reloads`, `instruments`, `trades_seen`, `latent_seen`, `regions_seen`, `transitions_seen`, `closes_seen`, `regions`, `transition_rows`, `decoder_outputs_seen`, `decoder_outputs`, `signals_emitted`, `signals_published`, **`primitives.graph` / `primitives.decoder` / `primitives.fallback`** (how many primitive evaluations came from the real graph, from decoder outputs, or from the price fallback), and per strategy `strategy.<id>.mean|std|steps|paused|primitives.graph|primitives.decoder|primitives.fallback`. Shown by `GET /api/services`.

## Run
```sh
cargo run -p strategy-runtime                                  # built-in baselines, config/platform.toml
cargo run -p strategy-runtime -- --strategies my-population.toml
FA_CONFIG=config/platform.docker.toml strategy-runtime        # in the Rust image (compose)
```
Running one on the host while the compose service is up emits every strategy id twice: `docker compose stop strategy-runtime` first. Strategy file format: see `baselines.toml`; programs are S-expressions over the GP DSL (`crates/fa-strategies/README.md`).

## Governing specs
- [35-strategies](../../specs/35-strategies.md)
- [33-mcmc-graph](../../specs/33-mcmc-graph.md) §4.2, [34-decoder-population](../../specs/34-decoder-population.md)
- [21-bus-and-contracts](../../specs/21-bus-and-contracts.md)
