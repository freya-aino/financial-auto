# strategy-runtime

Live service: runs a population of GP strategies over `md.trade.>` (+ `latent.state.>` when present) and publishes `ExposureDistribution` on `signal.exposure.<strategy_id>`.

**Status:** running — baseline population (hand-written, **not evolved**) from [`crates/fa-strategies/baselines.toml`](../../crates/fa-strategies/baselines.toml); price-only fallback features until regions/transitions/decoder outputs are published; heartbeat `svc.heartbeat.strategy-runtime` with counters (`strategies`, `instruments`, `trades_seen`, `latent_seen`, `signals_emitted`, `signals_published`).

## How it works
- Each strategy declares `timescale`, `steps` (horizon in bars), `lookback_bars` and an optional `step_secs` cadence (default: one bar). Trades are resampled into bars at the strategy's timescale; the GP program is evaluated per instrument through `fa_strategies::features::FallbackContext`, which maps the graph/decoder primitives onto empirical price statistics (documented in that module, clearly a placeholder).
- Portfolio primitives read the strategy's own virtual book (last emitted exposure, steps in trade, unrealised return of that exposure).
- Outputs are clipped to `mean ∈ [-1, 1]`, `stddev ∈ [0, 1]`; NaN → 0 exposure (spec 35 §8).
- Deterministic core in `src/runtime.rs` (tested); I/O in `src/main.rs`.

## Run
```sh
cargo run -p strategy-runtime                                  # built-in baselines, config/platform.toml
cargo run -p strategy-runtime -- --strategies my-population.toml
FA_CONFIG=config/platform.docker.toml strategy-runtime        # in the Rust image (compose)
```
Strategy file format: see `baselines.toml`; programs are S-expressions over the GP DSL (`crates/fa-strategies/src/sexpr.rs`).

## Governing specs
- [35-strategies](../../specs/35-strategies.md)
- [21-bus-and-contracts](../../specs/21-bus-and-contracts.md)
