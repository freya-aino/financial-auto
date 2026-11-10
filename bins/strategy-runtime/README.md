# strategy-runtime

Live service: runs a population of GP strategies over `md.trade.>`, the latent graph (`latent.state.>`, `latent.region.>`, `latent.transition.>`) and decoder outputs (`decoder.output.>`), and publishes `ExposureDistribution` on `signal.exposure.<strategy_id>`.

**Status:** feature-complete for spec 35 §4.2 / §8 (D-063) — baseline population (hand-written, **not evolved**) from [`crates/fa-strategies/baselines.toml`](../../crates/fa-strategies/baselines.toml), including two programs on the graph primitives and two on `text_score` / `visual_score`; or an evolved population from Garage (`--strategies garage:…` / `latest`); every primitive reads the real bus data for the strategy's timescale/instrument (latent graph, `rnn_prior` / `retrieval` / `text.*` / `visual.*` decoder outputs) and falls back to price statistics otherwise; retrieval analogs older than the in-memory history are priced from VictoriaMetrics; step-budget and NaN auto-pause, starved-primitive flag; hot reload, pause/resume and `status` over `ctl.strategy.*`; heartbeat `svc.heartbeat.strategy-runtime` with population, feed, analog, failure-mode and per-strategy gauges.

## How it works
- Each strategy declares `timescale`, `steps` (horizon in bars), `lookback_bars` and an optional `step_secs` cadence (default: one bar). Trades are resampled into bars at the strategy's timescale; the GP program is evaluated per instrument through `fa_strategies::features::LiveContext`: a point-in-time `graph::GraphAt` view of the latent graph at the strategy's timescale, a `decoders::DecodersAt` view of the instrument's decoder outputs (recency `--decoder-max-age-secs`, default 6 h) and the price-only `FallbackContext`. Primitives only see messages with event time ≤ the step.
- Closed bars are fed back into the graph (`LatentGraph::record_close`) so `region_feature(r, mean_fwd_return|vol|mass)` accumulates realised returns per region online.
- Portfolio primitives read the strategy's own virtual book (last emitted exposure, steps in trade, unrealised return of that exposure).
- Outputs are clipped to `mean ∈ [-1, 1]`, `stddev ∈ [0, 1]`. A **paused** strategy (operator) keeps stepping but emits `mean = 0, stddev = 1` so the allocator flattens it.
- **Decoder primitives.** `text_score(topic)` / `visual_score(kind)` read `quantiles[0]` of the latest fresh `decoder.output.text.<topic>` / `visual.<kind>` output (model-runtime contract: score in `[-1, 1]`, `output_refs` `inst:<uuid>` or `inst:*`); an instrument-specific output wins over a market-wide one while it is fresh. `(score family)` takes the dotted families (`(score text.news)`). `fc_quantile(q, k)` / `fc_dist(k)` prefer the forecast output whose horizon (`h:<steps>` ref, else `h=` in the genome text) is closest to `k`.
- **Analog outcomes (point-in-time).** `analog_outcome(k)` uses `p(window_end)` and `p(window_end + k·bar)` per retrieval analog, counted only once that outcome window has **completed at or before the step time**. The in-memory history holds `--price-history` trades per instrument (default 100 000). Analogs it cannot price are skipped at that step (`analog.unresolved`) and queued (bounded, keyed by `(instrument, window_end, k, bar)`); a background task (`src/prefetch.rs`, every `--analog-prefetch-ms`, `--analog-prefetch-batch` keys per round) reads `fa_trade_price` around them from VictoriaMetrics (`fa_tsdb::Tsdb::trade_prices_ms`, last bucket close at or before each end, never past the step time) into a bounded cache (`--analog-cache`, LRU-by-insertion, misses retried after 2 min), and the next step resolves them (`analog.resolved_tsdb`). The pure evaluation stays synchronous.
- **Failure modes (spec 35 §8).** Each strategy's evaluation per step is timed against `--step-budget-ms` (default 50): a slower step is **discarded** (nothing published, books unchanged: the allocator keeps the previous exposure) and counted (`budget_exceeded.<id>`); after `--breach-limit` (N, default 5) consecutive breaches the strategy is **auto-paused**. An instrument whose program produced NaN/inf (any node; counted by `fa_strategies::take_non_finite`) gets `mean = 0, stddev = 1`; after `--nan-limit` (N, default 5) consecutive such steps the strategy is auto-paused. Auto-paused strategies are not evaluated and emit `mean = 0, stddev = 1` until `ctl.strategy.resume` (which clears the streaks). A referenced primitive that has answered only from the fallback for more than `--starved-after-mins` (M, default 10) sets `strategy.<id>.starved = 1` (names in `status`).
- Deterministic core in `src/runtime.rs` (tested: cadence, graph-fed primitives, region returns, reload swap, pause, budget / NaN auto-pause, starvation, text/visual baselines, analog cache bounds); population sources in `src/source.rs` (tested with a local file and `fa_blob::MemoryStore`); analog prefetch in `src/prefetch.rs` (tested against a mock price source); I/O in `src/main.rs`.

## Population (`--strategies`, env `FA_STRATEGIES`)
| Value | Population |
|---|---|
| *(unset)*, `baselines` | built-in `baselines.toml` |
| `path/to/file.toml` | local file |
| `garage:fa-artifacts/strategies/<gen>.toml` | that object (bucket prefix optional) |
| `latest` | the greatest key under `fa-artifacts/strategies/` ending in `.toml` (key order) |

Any failure at start (missing file, invalid program, Garage unreachable or unconfigured) logs a warning and runs the baselines.

## Control (`ctl.strategy.<verb>`, `CtlRequest` → `CtlReply`)
| Verb | Args | Effect |
|---|---|---|
| `reload` | `artifact_key` (Garage key under `fa-artifacts/`, e.g. `strategies/strategies.3.toml`, written by the evo pipeline, or `latest`) or `path` (local file) | loads the population with `fa_blob::S3BlobStore::from_config` (credentials from `fa_core::Secrets::load()`), validates every program, swaps the population **atomically** (invalid file → running population untouched, reply `invalid: …`). Strategies whose id survives keep their virtual book and cadence. |
| `pause` / `resume` | `strategy_id` | flatten / un-flatten one strategy (`not_found: …` for unknown ids); `resume` also clears an auto-pause |
| `status` | – | `message` = JSON: `population`, `reloads`, `instruments`, `settings`, total `sources`, `analog` counters, `decoder_families`, and per strategy `id`, `genome_id`, spec genes, both programs, `paused`, `paused_reason` (text), `starved`, `starved_primitives`, `stats` (steps, last mean/std, `last_eval_us`, budget / NaN totals and streaks, analog counts), `sources`, `primitives{real, fallback_only_secs}` |

```sh
cargo run -p strategy-runtime --example ctl -- pause strategy_id=baseline-momentum-1s
cargo run -p strategy-runtime --example ctl -- reload artifact_key=strategies/strategies.3.toml
cargo run -p strategy-runtime --example ctl -- status
```

## Heartbeat counters
`strategies`, `paused`, `auto_paused`, `auto_pauses`, `starved`, `budget_exceeded`, `nan_outputs`, `reloads`, `instruments`, `trades_seen`, `latent_seen`, `regions_seen`, `transitions_seen`, `closes_seen`, `regions`, `transition_rows`, `decoder_outputs_seen`, `decoder_outputs`, `signals_emitted`, `signals_published`, **`primitives.graph` / `primitives.decoder` / `primitives.fallback`** (how many primitive evaluations came from the real graph, from decoder outputs, or from the price fallback), **`analog.unresolved`** (analog evaluations skipped for missing prices; an analog is counted again on every step until it resolves), **`analog.resolved_tsdb`** / `analog.resolved_memory` (analogs priced from the VictoriaMetrics cache / the in-memory history), `analog.cache`, `analog.pending`, `analog.fetches`, `analog.fetch_resolved`, `analog.fetch_missing`, `analog.fetch_errors`, the effective `settings.*`, `budget_exceeded.<id>`, and per strategy `strategy.<id>.mean|std|steps|paused|paused_reason|starved|budget_exceeded|nan_outputs|eval_us|primitives.graph|primitives.decoder|primitives.fallback|analog.unresolved|analog.resolved_tsdb`. Heartbeat values are numbers, so **`paused_reason` is a code**: 0 running, 1 operator, 2 step budget, 3 NaN/inf; the text is in `ctl.strategy.status`. Shown by `GET /api/services`.

## Settings
| Flag | Env | Default | |
|---|---|---|---|
| `--step-budget-ms` | `FA_STRATEGY_STEP_BUDGET_MS` | 50 | per strategy per step (spec 35 §7 proposes 1 ms; 50 leaves room for debug builds and many instruments) |
| `--breach-limit` | `FA_STRATEGY_BREACH_LIMIT` | 5 | N consecutive budget breaches → auto-pause |
| `--nan-limit` | `FA_STRATEGY_NAN_LIMIT` | 5 | N consecutive NaN steps → auto-pause |
| `--starved-after-mins` | `FA_STRATEGY_STARVED_AFTER_MINS` | 10 | M minutes of fallback-only → `starved` |
| `--price-history` | `FA_STRATEGY_PRICE_HISTORY` | 100 000 | trades kept per instrument |
| `--analog-cache` | `FA_STRATEGY_ANALOG_CACHE` | 4 096 | analog price cache entries |
| `--analog-prefetch-batch` / `--analog-prefetch-ms` | | 32 / 500 | prefetch round size / interval |
| `--decoder-max-age-secs` | `FA_STRATEGY_DECODER_MAX_AGE_SECS` | 21 600 | decoder output recency window |
| `--vm-url` | `FA_VM_URL` | `[victoriametrics] url` | VictoriaMetrics for analog prices |

## Run
```sh
cargo run -p strategy-runtime                                  # built-in baselines, config/platform.toml
cargo run -p strategy-runtime -- --strategies my-population.toml
FA_CONFIG=config/platform.docker.toml strategy-runtime        # in the Rust image (compose)
```
Running one on the host while the compose service is up emits every strategy id twice: `docker compose stop strategy-runtime` first (and `start` it afterwards).

To check the text / visual consumer without the model-runtime producer, publish contract-conformant frames (instruments discovered from `md.trade.>` for 5 s, or `--inst <uuid>`):
```sh
cargo run -p strategy-runtime --example publish_decoder -- --secs 60    # text.news|narrative, visual.chart per instrument; text.summary, visual.future market-wide
``` Strategy file format: see `baselines.toml`; programs are S-expressions over the GP DSL (`crates/fa-strategies/README.md`).

## Governing specs
- [35-strategies](../../specs/35-strategies.md)
- [33-mcmc-graph](../../specs/33-mcmc-graph.md) §4.2, [34-decoder-population](../../specs/34-decoder-population.md)
- [21-bus-and-contracts](../../specs/21-bus-and-contracts.md)
