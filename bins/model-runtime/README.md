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
- `svc.heartbeat.model-runtime` counters: `trades_total`, `states_published[.ts]`, `region_events[.ts]`, `transitions_published[.ts]`, `bars_closed.<ts>`, `regions_live.<ts>`, `transition_rows.<ts>`, and from the hub sink `hub.writes`, `hub.errors`, `hub.dropped`, `hub.coalesced`, `hub.pending`, `hub.skipped_zero_vec` (all-zero first-bar latents are not stored: the cosine index rejects them)

## HelixDB (`src/hub_sink.rs`, D-048)
The pipeline output is also persisted into HelixDB through [`fa-hub`](../../crates/fa-hub/README.md), from a background task fed by a **bounded channel** (`--hub-queue`, default 4096) so bus publishing is never blocked. What is written, keyed by business ids so restarts are idempotent:

| Event | Helix rows | When |
|---|---|---|
| start | `Snapshot stat-v1-<ts>` (status `active`) + `ExpertSpace <snapshot>-e<k>` (dim 8) | once per process (`ensure_*`, get-or-create) |
| region created | `LatentRegion` (`region_id`, `lineage_id`, `method=leader`, `radius`, `status=live`) + `InSpace` | on the event (upsert: the baseline's ids `<ts>-e<k>-r<nnn>` are process-local, D-039, so a restart revives rows of the same id) |
| region merged / retired | `LatentRegion.status = merged \| retired`, `related_region_id` | on the event |
| transition row published | `Transition` edge `from → to` with `count` (observations seen by this process), `mean`, `lower`, `upper`, `kernel_params`, `updated_at` | **coalesced**: at most one write per edge per `--hub-flush-secs` (default 10); only successors observed at least once get an edge (prior-only mass stays in the in-memory view) |
| closed bar | one `LatentVec` per expert (`kind=observation`, `ref_id=region_id`, `instrument_id`, `ts`, `uncertainty = sqrt(mean exp(log_var))`, vector in `vec8`) | `--hub-latent-timescales` (default `1m,1h,1d,1w`; 1s is not stored) |

Flags: `--hub-url` (default `FA_HELIX_URL`, else `helix.url` from the config), `--hub-flush-secs`, `--hub-latent-timescales`, `--hub-queue`, `--no-hub`. Helix unreachable → one warning, writes are dropped (`hub.dropped`) and the connection is retried every 30 s; the live pipeline keeps running. Read it back through control-api: `GET /api/latent/snapshots`, `GET /api/latent/stat-v1-1m/regions`, `GET /api/mcmc/stat-v1-1m/graph`.

## Run
```sh
cargo run -p model-runtime                       # host, reads config/platform.toml
cargo run -p model-runtime -- --preview-secs 2 --region-radius 0.8 --max-regions 32
cargo run -p model-runtime -- --hub-flush-secs 5 --hub-latent-timescales 1m,1h   # Helix sink tuning; --no-hub disables it
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
- [20-storage](../../specs/20-storage.md) §5.2 (Helix rows written by `hub_sink`, D-048)
