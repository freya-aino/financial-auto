# evo-worker

Stateless worker evaluating decoder genomes pulled from the `JOBS_EVO` NATS work queue (`jobs.evo.>`).

**Status:** evaluates `evo.eval.decoder` jobs end to end with baseline CPU decoders on **VictoriaMetrics data** (`fa_trade_price` via `fa-tsdb`, in-memory `md.trade.>` buffer as fallback) — genome parsing, point-in-time CRPS/coverage/trading-value scoring, multi-objective `JobResult` (JSON + typed `objectives`/`directions` maps), optional `latent.state.>` conditioning of the retrieval decoder, `JobProgress`, GPU-arbiter lease with preemption handling, `DecoderOutput` publishing, eval report persistence to Garage (`fa-artifacts/evals/<generation>/<job>.json`) when credentials are configured. Handles **`evo.vary.decoder`** jobs (next-population variation with `fa-evolution`, lineage to Helix). Other job kinds get `STATUS_FAILED` ("not implemented").

## What it does per job
1. Decodes `Job`; genome = `Job.spec` if present, else `Job.genome_id`. The spec is either the genome itself (kv or JSON, D-044) or an envelope `{"genome": "...", "data_window": {"hours": 6}}` / `{"data_window": {"start_ms", "end_ms"}}` (the orchestrator sends the envelope when `data_window_hours` was given). Format: [`crates/fa-decoders/README.md`](../../crates/fa-decoders/README.md) (`forecast:h=5;lb=32;…`, `retrieval:k=20;tau=30;…`, or a bare id that is hash-derived). Separate genes with `;`, never `,`.
2. Acquires a GPU lease from the arbiter (`ctl.gpu.acquire`, class `evolution_eval`, `--gpu-lease-mb`, default 512; `0` disables). Queued leases wait for `ctl.gpu.granted` until the job deadline; an unreachable arbiter is tolerated (CPU work).
3. Loads the evaluation window (see *Evaluation data*): `fa_tsdb::Tsdb::trade_prices` for every instrument (or the genome's `inst`) at the genome's timescale step, else the buffer. Builds the decoder with the latest `latent.state.>` context (`fa_decoders::baseline::build_with`) and runs `fa_backtest::eval::evaluate`: forecast at every step using only past data (and latents as of that step), scored against the realised horizon return. Stops at `budget_gpu_ms` (partial evaluation, `budget.truncated=true`), on `ctl.gpu.preempt` for its lease (→ `STATUS_PREEMPTED` with `checkpoint_ref`; redelivered jobs re-evaluate from scratch because evaluation is cheap) or when the window is exhausted.
4. Publishes `JobProgress` on `jobs.progress.<job_id>` every 500 ms, in-progress JetStream acks every 20 s (`ack_wait` 60 s, `max_deliver` 5), one message in flight per worker.
5. Publishes `DecoderOutput` on `decoder.output.rnn_prior` (forecast quantiles) / `decoder.output.retrieval` (analogs with similarity, window end, time distance) for up to 4 instruments, then the `JobResult` on `jobs.result.evo.eval.decoder`, then acks. If the result cannot be published the message is left unacked for redelivery.

### Result payload (`evo.eval.decoder/v0`, D-025 compatible)
```json
{"protocol":"evo.eval.decoder/v0","score":0.39,
 "objectives":{"crps":6e-4,"crps_skill":0.39,"coverage":0.94,"coverage_err":0.04,"hit_rate":0.56,
               "trade_sharpe":0.13,"trade_psr":0.69,"compute_cost":2.5,"n_eval":18},
 "directions":{"crps":"min","crps_skill":"max","coverage_err":"min","hit_rate":"max","trade_sharpe":"max","trade_psr":"max","compute_cost":"min"},
 "genome":{...},
 "data":{"source":"victoriametrics","timescale":"1s","step_ms":1000,"window_start_ms":..,"window_end_ms":..,"vm_url":"http://127.0.0.1:8428",
         "vm_series":1,"vm_error":null,"latents":{"instruments":1,"points":195},"instruments":[{"instrument_id":"0190…0001","venue":"sim","n_steps":4254,"n_eval":4232,...}]},
 "budget":{"budget_ms":2000,"elapsed_ms":11,"truncated":false},"stop":"complete",
 "report_ref":{"planned":"fa-artifacts/evals/<generation>/<job>.json","persisted":true}}
```
`score` = `crps_skill` (the scalar fallback); the orchestrator ranks with NSGA-II over `objectives`/`directions`, which the worker also sets as the typed `JobResult.objectives` / `JobResult.directions` proto maps. control-api shows `objectives` in the Evolution Lab. Objective definitions: `crates/fa-backtest/src/eval.rs`. `data.source` is `victoriametrics` or `md.trade.> in-memory buffer (fallback: <reason>)`.

### `evo.vary.decoder` (next population)
`Job.spec` = `{"parents": [{"genome": "...", "objectives": {...}, "directions": {...}} | "genome string", ...], "n_children": 3, "seed": 7, "generation": 1, "budget": {"total_gpu_ms", "min_budget_gpu_ms", "eta", "rungs"}?}` (parents best first). The worker parses each parent (`DecoderGenome`), maps the numeric genes to `fa_evolution::Genome.params`, runs `fa_evolution::vary::vary` (same-family uniform crossover with p = 0.5, Gaussian mutation σ = 0.2 relative; overridable with `sigma` / `crossover_prob`), clamps the children back with `with_params` and returns them as canonical kv strings, de-duplicated against the parents. `n_children = 0` with `budget` sizes the population with `fa_evolution::plan_halving` (spec 34 §6.4): `children = initial_candidates − parents`. Result: `{"protocol":"evo.vary.decoder/v0","children":[...],"n_parents","n_children","plan"?,"lineage":[{"child","parents","op"}],"helix":{"persisted",...}}`. Lineage goes to Helix best effort (`upsert_genome` for parents and children, `link_parent` edges, `add_fitness_record` with the parents' objectives; `FA_HELIX_URL` or `[helix] url`); failures only warn. No GPU lease is taken.

### Report persistence
At startup the worker loads `config/secrets.toml` (`$FA_SECRETS`; `FA_GARAGE_ACCESS_KEY_ID` / `FA_GARAGE_SECRET_ACCESS_KEY` override, D-020) and builds an `fa_blob::S3BlobStore` against `[garage] endpoint` of the platform config. Every OK result's JSON (pretty-printed, `application/json`) is then written to `report_ref.planned` and the published result says `"persisted": true`. Without credentials the worker logs one warning at startup and results carry `"persisted": false`; a failed write is a warning too and never fails the job. Create the key with `docker/garage-init.sh` and paste it into `config/secrets.toml` `[garage]`.

## Evaluation data
**Primary: VictoriaMetrics** (`[victoriametrics] url` of the platform config, `FA_VM_URL` override, `--no-tsdb` disables). `fa_tsdb::Tsdb::trade_prices` reads `last_over_time(fa_trade_price{inst=…}[step])` over the data window — the job's `data_window`, else `--data-window-hours` / `FA_EVO_DATA_WINDOW_HOURS` (default 6 h) ending now — at the genome's timescale step (`inst` label = UUID without dashes, as the ingestor writes it, D-038). The window is shrunk to ≤ 20 000 steps (VM's per-series point limit). A genome needs `warmup + horizon + 1` steps (forecast: `lb + 2h + 1`; retrieval: `2·lb + gap + 2h + 3`).

**Fallback: in-memory buffer.** The worker still subscribes to `md.trade.>` (last 50 000 trades per instrument, 64 instruments). When VM is unreachable, errors, or no series in the window is long enough, the buffer is used and `data.source` says `md.trade.> in-memory buffer (fallback: <reason>)`; a job only fails when both sources are too short (the error lists the VM error and buffer size).

**Latent conditioning.** The worker subscribes to `latent.state.>` and keeps the latest 20 000 `LatentState`s per instrument (concatenated expert means). Each eval job gets a snapshot as `DecodeCtx`; the retrieval decoder adds `lat · cos(latent as of query, latent as of candidate)` to its similarity (gene `lat`, default 0.5) whenever both latents exist — looked up as of each window end, so it stays point-in-time and deterministic. `data.latents` reports how many latents were available; heartbeat counters `latents.instruments` / `latents.points`.

## Run
```sh
cargo run -p evo-worker                # host, against compose NATS + VM; FA_CONFIG=config/platform.toml
cargo run -p evo-worker -- --gpu-lease-mb 0 --data-window-hours 2
# then start a generation through control-api (max_generations > 1 loops halving → variation → halving):
curl -X POST http://127.0.0.1:7700/api/evolution/generations -H 'content-type: application/json' \
  -d '{"kind":"evo.eval.decoder","snapshot_id":"snap-test","genomes":["forecast:h=3;lb=16","retrieval:h=3;lb=8;k=10","g1"],"eta":2,"min_budget_ms":2000}'
curl http://127.0.0.1:7700/api/evolution/jobs ; curl http://127.0.0.1:7700/api/decoders/outputs
```
Only one consumer may read a work queue, so stop the compose `evo-worker` (`docker compose stop evo-worker`) when running one on the host, or they compete for jobs.

## Owns
`bins/evo-worker/src/{main,job,data,latent,vary,market,gpu}.rs`: JetStream consumer loop, job evaluation, evaluation-data loading (VM + buffer), latent buffer, variation jobs, trade buffer, arbiter lease client.

## Governing specs
- [21-bus-and-contracts](../../specs/21-bus-and-contracts.md), [34-decoder-population](../../specs/34-decoder-population.md), [36-training-lifecycle](../../specs/36-training-lifecycle.md), [41-backtest-and-validation](../../specs/41-backtest-and-validation.md), [60-ops](../../specs/60-ops.md); decisions D-006, D-025, D-038, D-044.
