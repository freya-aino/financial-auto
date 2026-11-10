# model-runtime

Live service: bus → encoders → expert spaces → regions/MCMC → decoders/strategies → signals.

**Status:** feature-complete at the service level with baselines behind every stage (D-062). Consumes `md.trade.>`, `md.bar.>` and `news.item.>`; builds rolling bars per timescale and instrument (from bus OHLCV bars where a venue publishes them, else from trades); publishes `LatentState`, `RegionEvent`, `TransitionUpdate`, and `DecoderOutput`s of the baseline **text** (`text.news`, `text.summary`) and **visual** (`visual.chart`, `visual.future`) decoders. Encoders are the deterministic **statistical** family (no neural models, D-004): the in-process baseline, or a **trained snapshot** loaded from Garage and hot-swapped on promotion with **kernel transfer** (spec 33 §6.6); a candidate can run in **shadow** next to the active snapshot (spec 36 §6.4) and **online adapters** are applied as the trainer produces them (spec 36 §6.5). Strategies consume the outputs in `strategy-runtime`.

## Snapshot lifecycle (`src/snapshot.rs`, spec 36 §6.4)
- **Start:** per timescale, the `active` Helix `Snapshot` with `encoder_family = statistical` is resolved. A row with a `garage_key` is loaded from Garage (`manifest.json` + `weights.bin` → `StatisticalEncoder::from_manifest`, `region_map.json` → region sets seeded with the snapshot's regions and lineages); no active row, a row without an artifact (the baseline `stat-v1-<ts>`), Helix down or no Garage credentials (`config/secrets.toml`) → the in-process baseline, registered as `active` (gen 0 when first created; `ensure_snapshot` keeps an existing row). Seeded regions are announced as `Created` events (bus + Helix). Log line: `encoder resolved ts=… snapshot=… source=garage|baseline note=…`.
- **`ctl.model.reload`** (request/reply `CtlRequest{verb: reload, args: timescale, snapshot_id?}`): resolves the named snapshot (default: the active one), `TimescalePipeline::swap_encoder` replaces the encoder, rebuilds the region sets from the new `region_map` (matched lineages continue; unmatched regions are new lineages, new births continue the series), **transfers the transition kernel** and clears the per-instrument chain positions; the new snapshot and its seeded regions are re-announced. Reply: `1m: <from> -> <to> (gen N, garage|baseline, K seeded regions, carried mass x, kernel carried y over R rows)`, `unchanged <id>`, or an error (unknown snapshot, wrong timescale, no artifact) in which case the current encoder keeps serving. control-api's `POST /api/latent/snapshots/{id}/promote|rollback` sends it after changing the Helix status. Heartbeat counters `reloads`, `reloads.<ts>`, `kernel_transfer.<ts>.carried` (cumulative carried observation mass), `kernel_transfer.<ts>.rows|dropped` (last swap).
- **Kernel transfer** (spec 33 §6.6, `fa_mcmc::MarkovGraph::transfer`): per expert, every region of the new map with a `parent_region_id` inherits the parent's Dirichlet row: a successor's (decayed) count lands on the new regions carrying that successor in proportion to their match scores (splits divide, merges sum), the whole row is scaled by `score · carry_discount` (`PipelineConfig::carry_discount`, default 0.5: transferred evidence counts less than fresh observations); successors without a destination are dropped, unmatched regions start from the symmetric prior, every new region is prior support. No map (rollback to the baseline) → posteriors restart.
- **Shadow** (`ctl.model.shadow {timescale, snapshot_id | none}`, `src/shadow.rs`): the candidate is resolved like a reload and served by a second `TimescalePipeline` on the same bars (closed bars and previews). Its states go to `latent.shadow.<ts>.<instrument token>` (inside `LATENT` via `latent.>`); its region events and transition rows stay in memory (not published, not in Helix). Reply JSON `{ok, timescale, active, shadow, generation, source, seeded_regions, carried_mass, replaced, subject}` (`none` → `{ok, shadow: null, stopped}`). Heartbeat gauges: `shadow.<ts>.snapshot` (the candidate's **generation**: heartbeat counters are numbers only, so the id is carried as `shadow.<ts>.snapshot_id.<snapshot_id> = 1`), `shadow.<ts>.states`, `shadow.<ts>.minutes`, `shadow.<ts>.region_agreement` (share of compared expert steps where the shadow region continues the active one: equal `region_lineage_id`, or the shadow region's `parent_region_id` in its `region_map` is the active region), `shadow.<ts>.carried_mass` (`region_map.carried_mass()`); all removed when shadowing stops. A reload of the active snapshot leaves the shadow running.
- **Adapters** (`ctl.model.adapter {timescale}`, and automatically on every OK `jobs.result.train.adapter` whose `snapshot_id` is serving): `SnapshotLoader::adapter` reads `artifact_key` from the result (else the newest `fa-models/adapters/<snapshot>/<ms>.json`), refuses `frozen` artifacts and other snapshots/timescales, and builds the encoder from the artifact's re-fitted `stats` (a trained snapshot keeps its manifest params); `TimescalePipeline::set_encoder` swaps only the encoder (regions, posteriors and chain positions stay). Counter `adapters_applied.<ts>`, gauge `adapter.<ts>.ts_ms`; a reload resets the adapter.

## What runs (per timescale in `config/platform.toml`, D-011)
| Stage | Trait (swap point) | Baseline implementation |
|---|---|---|
| Bars | — (`src/bars.rs`) | `BarSeries`: fixed `event_time` buckets, last `context_steps` closed bars + the forming bar; rows carry OHLCV, trade count and vwap (vwap stays in the series: the encoder row has no vwap field) |
| Encoder | `fa_encoders::EncoderFamily` | `StatisticalEncoder`: 3 experts (returns, volatility, activity) × 8 bounded z-score features, `log_var = −ln(n_bars)`; snapshot id `stat-v1-<ts>` |
| Router | `fa_expert_space::Router` | `EvidenceSoftmaxRouter` (softmax over latent energy) |
| Regions | `fa_patterns::OnlineRegionDiscovery` (one set per expert) | `OnlineRegionSet`: streaming leader clustering, ids `<ts>-e<k>-r<nnn>`, lineage = id at birth, merge/retire events |
| Transitions | `pipeline::TransitionModel` (one per expert) | `DirichletTransitions` over `fa_mcmc::MarkovGraph` (Dirichlet rows, per-timescale forgetting, 90 % credible interval from 256 draws, sampler `conjugate`), kernel transfer on swap |
| Text decoder | `fa_decoders::text_baseline` (`src/decoders.rs`) | lexicon + negation scorer, instrument matching against the `ref` snapshot, `text.news` / `text.summary` |
| Visual decoder | `fa_decoders::visual_baseline` (`src/decoders.rs`) | breakout / trend / compression of the last 60 closed bars, `visual.chart` / `visual.future` |

### Bar sources
- **Trades** (`md.trade.>`): every trade folds into the forming bar of its bucket; the first trade of a later bucket closes it (unchanged; `sim` and streaming venues).
- **Bus bars** (`md.bar.<tf>.<venue>.<inst>`, `Envelope{Bar}` stamped at the bar **close**, `event_time = open + tf`, the Yahoo connector's convention): once an instrument publishes bars, every timescale whose step is ≥ the finest bar timeframe builds its ladder from them (the series restarts once on the switch, so the connector's synthetic per-bar trade is not double counted); trades of that instrument are then ignored at those timescales and keep feeding finer ones (1s). A bar of the step's timeframe closes immediately; finer bars are **resampled upward** (open of the first, max high, min low, close of the last, volume and trade count summed, vwap volume-weighted) and the bar closes when its last sub-bar lands or a later bucket starts; bars at or before the last closed bucket are ignored. `trade_count = 0` (Yahoo) counts as one print. The activity expert therefore sees real volume, dollar volume and size per print.
- Heartbeat: `bars_in.<tf>` (bus bars received per timeframe token), gauges `bars_source.<ts>.bus` / `bars_source.<ts>.trades` (instruments per source; numbers only, so the source is in the key).

### Decoders (contract consumed by `fa_strategies::decoders`, D-062)
Every output is a `DecoderOutput` on `decoder.output.<family>` with `quantiles[0]` = the score, `snapshot_id` = the timescale's serving encoder, `timescale`, `as_of`, `genome_id` empty:

| family | decoder_id | when | `quantiles` | `output_refs` |
|---|---|---|---|---|
| `text.news` | `text-stat-v1` | on each new news item for its keys, and every `--text-refresh-secs` (60) for every key with items in range | `[score ∈ [-1, 1]]` | `inst:<uuid>` or `inst:*`, `topic:news`, `src:<source>`…, `n:<items in window>` |
| `text.summary` | `text-stat-v1` | same | `[novelty ∈ [0, 1], lexicon hits, trailing items]` | `inst:…`, `topic:summary`, `src:…`, `n:…` |
| `visual.chart` | `visual-stat-v1` | each closed bar (`--visual-timescales`, default `1m,1h,1d,1w`), ≥ 8 bars | `[breakout, trend, compression]` | `inst:<uuid>`, `kind:chart`, `src:bus\|trades`, `n:<bars>` |
| `visual.future` | `visual-stat-v1` | same | `[future, breakout, trend, compression]` | `inst:<uuid>`, `kind:future`, `src:…`, `n:…` |

- **Text** (`--text-timescales`, default `1m,1h,1d,1w`): window per timescale 1m → 1 h, 1h → 1 d, 1d → 1 w, 1w → 4 w; trailing window of the same length before it for the novelty score. Items are matched to instruments (`fa_decoders::text_baseline::InstrumentIndex`) by upper-case ticker tokens (`SPY`, `$SPY`, the base of a pair `BTC-USD` → `BTC`), name phrases (legal suffixes and parenthesised notes dropped; a name that is just the ticker matches only as a symbol) and `entities_raw` (`query:Apple stock`); no match → market-wide `inst:*`. The reference snapshot (`ref.instrument` names, `ref.identifier` values, `ref.listing` symbols via `fa_warehouse::refdata::search_instruments`, ≤ 500 instruments) is loaded at start and every 5 min (`--pg-url`, default `FA_POSTGRES_URL` / `postgres.url`). At start the `TEXT` stream's retained items are replayed through an ordered JetStream consumer (after the first reference snapshot, ≤ 15 s wait) without publishing, then one output per key is published and live items follow; items are deduplicated by `dedup_key` / `event_id`. Scoring formulas are in `fa_decoders::text_baseline`.
- **Visual**: formulas in `fa_decoders::visual_baseline` (breakout vs the support/resistance band of the window before the last bar, OLS t-statistic of the log-price trend, range compression, `future = 0.5·breakout + 0.35·trend + 0.15·sign(trend)·compression`).
- Heartbeat: `decoder_outputs.<family>`, `news.items|matched|market|duplicates`, `news.keys`, `ref_instruments`.

Steps: a **closed bar** is one step of the timescale's chain — state + region membership are published and the transition `region(t−1) → region(t)` is observed per expert and its posterior row published. Between closes a **preview** step re-encodes the window including the forming bar every `--preview-secs` (default 5 s) so slow timescales (1h, 1d, 1w) show a live state and regions; previews never touch the transition posteriors.

## Bus
- `latent.state.<ts>.<instrument token>` → `LatentState` (per instrument, latest wins in control-api)
- `latent.shadow.<ts>.<instrument token>` → `LatentState` of the shadow snapshot (only while shadowing)
- `decoder.output.text.news|text.summary|visual.chart|visual.future` → `DecoderOutput`
- consumes `md.trade.>`, `md.bar.>`, `news.item.>` (JetStream `TEXT` backlog first), `jobs.result.train.>`, `ctl.model.reload|shadow|adapter`
- `latent.region.<ts>` → `RegionEvent` (created / merged / retired)
- `latent.transition.<ts>` → `TransitionUpdate` (posterior mean + credible interval per successor)
- `svc.heartbeat.model-runtime` counters: `trades_total`, `states_published[.ts]`, `region_events[.ts]`, `transitions_published[.ts]`, `bars_closed.<ts>`, `regions_live.<ts>`, `transition_rows.<ts>`, and from the hub sink `hub.writes`, `hub.errors`, `hub.dropped`, `hub.coalesced`, `hub.pending`, `hub.skipped_zero_vec` (all-zero first-bar latents are not stored: the cosine index rejects them)

## HelixDB (`src/hub_sink.rs`, D-048)
The pipeline output is also persisted into HelixDB through [`fa-hub`](../../crates/fa-hub/README.md), from a background task fed by a **bounded channel** (`--hub-queue`, default 4096) so bus publishing is never blocked. What is written, keyed by business ids so restarts are idempotent:

| Event | Helix rows | When |
|---|---|---|
| start / reload | `Snapshot` (the served snapshot: `stat-v1-<ts>` baseline with status `active`, or the loaded trained snapshot with its generation and `garage_key`) + `ExpertSpace <snapshot>-e<k>` (dim 8) | once per process and on every `ctl.model.reload` (`ensure_*`, get-or-create) |
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
cargo run -p model-runtime -- --visual-timescales 1m,1h --text-timescales 1h --text-refresh-secs 30
cargo run -p model-runtime --example ctl -- shadow timescale=1m snapshot_id=<candidate>   # or snapshot_id=none
cargo run -p model-runtime --example ctl -- reload timescale=1m snapshot_id=<id>        # kernel transfer
cargo run -p model-runtime --example ctl -- adapter timescale=1m
curl 'http://127.0.0.1:7700/api/decoders/outputs?family=text'                         # or family=visual
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
