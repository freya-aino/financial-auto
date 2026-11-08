# model-runtime

Live service: bus → encoders → expert spaces → regions/MCMC → decoders/strategies → signals.

**Status:** baseline latent pipeline running — consumes `md.trade.>`, builds rolling bars per timescale and instrument, and publishes `LatentState`, `RegionEvent` and `TransitionUpdate` derived from the live trade stream. Encoders are the deterministic **statistical baseline** (no neural models, D-004); decoders and strategies are not wired yet.

## What runs (per timescale in `config/platform.toml`, D-011)
| Stage | Trait (swap point) | Baseline implementation |
|---|---|---|
| Bars | — (`src/bars.rs`) | `BarSeries`: fixed `event_time` buckets, last `context_steps` closed bars + the forming bar |
| Encoder | `fa_encoders::EncoderFamily` | `StatisticalEncoder`: 3 experts (returns, volatility, activity) × 8 bounded z-score features, `log_var = −ln(n_bars)`; snapshot id `stat-v1-<ts>` |
| Router | `fa_expert_space::Router` | `EvidenceSoftmaxRouter` (softmax over latent energy) |
| Regions | `fa_patterns::OnlineRegionDiscovery` (one set per expert) | `OnlineRegionSet`: streaming leader clustering, ids `<ts>-e<k>-r<nnn>`, lineage = id at birth, merge/retire events |
| Transitions | `pipeline::TransitionModel` (one per expert) | `DirichletTransitions` over `fa_mcmc::MarkovGraph` (Dirichlet rows, per-timescale forgetting, 90 % credible interval from 256 draws, sampler `conjugate`) |

Steps: a **closed bar** is one step of the timescale's chain — state + region membership are published and the transition `region(t−1) → region(t)` is observed per expert and its posterior row published. Between closes a **preview** step re-encodes the window including the forming bar every `--preview-secs` (default 5 s) so slow timescales (1h, 1d, 1w) show a live state and regions; previews never touch the transition posteriors.

## Bus
- `latent.state.<ts>.<instrument token>` → `LatentState` (per instrument, latest wins in control-api)
- `latent.region.<ts>` → `RegionEvent` (created / merged / retired)
- `latent.transition.<ts>` → `TransitionUpdate` (posterior mean + credible interval per successor)
- `svc.heartbeat.model-runtime` counters: `trades_total`, `states_published[.ts]`, `region_events[.ts]`, `transitions_published[.ts]`, `bars_closed.<ts>`, `regions_live.<ts>`, `transition_rows.<ts>`

## Run
```sh
cargo run -p model-runtime                       # host, reads config/platform.toml
cargo run -p model-runtime -- --preview-secs 2 --region-radius 0.8 --max-regions 32
curl http://127.0.0.1:7700/api/latent/timescales/1m/states      # via control-api (D-032)
curl http://127.0.0.1:7700/api/latent/timescales/1m/transitions
```
With the ingestor's `heartbeat` connector (1 trade/s on one simulated instrument) the 1s and 1m views fill within seconds / about two minutes; transitions at 1h and above need real elapsed time.

## Owns
Real-time inference loop per timescale.

## Governing specs
- [01-architecture](../../specs/01-architecture.md)
- [30-latent-regions](../../specs/30-latent-regions.md)
- [31-expert-encoders](../../specs/31-expert-encoders.md)
- [32-expert-spaces](../../specs/32-expert-spaces.md)
- [33-mcmc-graph](../../specs/33-mcmc-graph.md)
- [35-strategies](../../specs/35-strategies.md)
