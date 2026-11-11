# evo-orchestrator

Schedules decoder/strategy populations onto evo-workers under a GPU-hour budget and hosts the GPU arbiter (D-006).

**Status:** implemented — GPU arbiter with strict priority and preemption (`ctl.gpu.acquire|release`, notifications `ctl.gpu.preempt|granted`); generations started via `ctl.evo.start`: successive halving ranked **per niche by non-dominated front and hypervolume contribution** over `JobResult.objectives`/`directions` (scalar `score` fallback), loop-end elites by **NSGA-II (≤ 3 objectives) / NSGA-III (≥ 4)**, per-rung **Pareto summaries** on `jobs.result.evo.generation` and heartbeat counters, and a **generation loop** (`max_generations` > 1): elites + `evo.vary.<family>` children form the next population — the same loop for decoders (`evo.eval.decoder`), strategies (`evo.eval.strategy`) and allocators (`evo.eval.allocator`, spec 40 §6.2 tournament). Job cancel via `ctl.evo.cancel` emits `STATUS_CANCELLED`; `ctl.evo.pause|resume|stop` control a whole generation. Summaries carry the trial registry's **validation** statistics (deflated Sharpe, PBO) and, for strategy and allocator generations, the Garage key of the promoted front file (D-052).

## Control (request/reply, `CtlRequest`/`CtlReply`)
| Subject | Args | Reply |
|---|---|---|
| `ctl.gpu.acquire` | `lease_id, class, mem_mb` | `granted` / `queued` |
| `ctl.gpu.release` | `lease_id` | `released` |
| `ctl.evo.start` | `kind, snapshot_id, genomes` (newline-separated, D-044)`, eta, min_budget_ms, max_budget_ms, max_generations, total_budget_ms, data_window_hours, mcmc_pairs` | generation id |
| `ctl.evo.cancel` | `job_id` | `cancelled <id>`; `not_found: …` when no generation has the job pending |
| `ctl.evo.pause` | `generation_id` | `paused <id>`: no new rung (or variation job) is dispatched; running jobs finish and are recorded. `conflict:` when already paused / done |
| `ctl.evo.resume` | `generation_id` | `resumed <id>`: a rung deferred while paused is submitted now. `conflict:` when not paused |
| `ctl.evo.reeval` | `snapshot_id?, timescale?, k?, data_window_hours?, dry_run?` | job id of the `evo.reeval.decoder` job (elite decoders re-evaluated frozen, transfer records on their cards; below) |
| `ctl.evo.stop` | `generation_id` | `stopped <id> (n job(s) cancelled)`: every pending job gets `JobResult{STATUS_CANCELLED}` on `jobs.result.<kind>`, leases released, final summary with `state: "stopped"`; the elites are the best candidates evaluated so far |

- `max_generations` (default: `-max-generations` / `FA_EVO_MAX_GENERATIONS`, default 1): halving loops. 1 = a single successive halving (previous behaviour). control-api forwards it from `POST /api/evolution/generations` (D-049).
- `total_budget_ms` (optional): when set, the next population is sized by the worker with `fa_evolution::plan_halving` (spec 34 §6.4) instead of keeping the start size.
- `data_window_hours` (optional): forwarded to eval jobs as `Job.spec = {"genome", "data_window": {"hours"}}`.

Cancel removes the job from its generation (it counts as a failed candidate, so the rung can still complete; cancelling the pending variation job ends the generation), releases the GPU lease named after the job id (convention until workers report lease ids) and publishes `JobResult{status: CANCELLED, error: "cancelled by operator", worker_id: "evo-orchestrator"}` on `jobs.result.<kind>`. A result the worker publishes later for that job id is ignored.

## Generation loop
1. **Halving:** rungs from `Plan(n, eta, min_budget, max_budget)`; every rung submits one `jobs.<kind>` job per alive genome. `Job.spec` is always the envelope `{"genome", "generation": <loop>, "rung": <rung>, "data_window"?: {"hours"}}` (the worker records loop and rung in the trial registry; strategy genomes are TOML inline tables, which never parse as JSON, so the two forms cannot be confused). When a rung completes the candidates are ranked (below) and the best `ceil(n/eta)` advance with `eta`× budget.
2. **Rung ranking (`Rank`/`Halve`, `halving.go` + `pareto.go` + `nsga.go`, spec 34 §6.2 step 4):** candidates are sorted by ID first (arrival order does not matter) and grouped by **niche** (`Niche`: the decoder output family of the genome — `forecast`/`nn_forecast` → `rnn_prior`, … — and one niche for everything else). Inside a niche: non-dominated fronts over the minimised objective vectors (`max` objectives negated, missing values `+∞`), inside a front by **hypervolume contribution** (the scalarised rung score: objectives min–max normalised over the niche, exclusive HV of each member w.r.t. reference 1.1 over the objectives every front member defines — exact for ≤ 2 objectives, a deterministic SplitMix64 Monte-Carlo estimate with 8192 samples otherwise), then crowding distance, then score. The niches' orders are interleaved by relative position `(i+½)/n`, so `Halve` keeps about `1/η` of every family. Candidates without objectives (failed, cancelled, or scalar-only `{"score"}` results) come last; with no objectives at all the old scalar order applies. Objectives are read from the `JobResult.objectives`/`directions` proto maps first, then from the result JSON (D-044).
3. **Loop end (`SelectElites`, spec 34 §6.2 step 6):** elites = `ceil(N/eta)` of every genome evaluated in the loop (latest objectives; failed genomes never carry over), `ceil(k·n_f/n)` per niche chosen by **NSGA-II for ≤ 3 objectives, NSGA-III for ≥ 4** (Das–Dennis reference directions with the smallest division count giving ≥ k directions, ideal/nadir normalisation of the fronts taken, closest member of the least crowded direction first, deterministic ties), listed in `Rank` order. Decoder results carry 8 ranked objectives, so decoder elites are NSGA-III selections. `nsga.go` / `pareto.go` port `crates/fa-evolution/src/{lib,pareto}.rs`; both sides are tested against `testdata/nsga_fixture.json` (fronts, crowding, `nsga2_select`, hypervolume contributions and order, `nsga3_select`; here, under `go/`, so the Docker toolchain sees it; the Rust tests read it from this path). If more loops remain, a `jobs.evo.vary.<family>` job (`VaryKind("evo.eval.decoder") = "evo.vary.decoder"`, `VaryKind("evo.eval.strategy") = "evo.vary.strategy"`, `VaryKind("evo.eval.allocator") = "evo.vary.allocator"`) carries `{"parents": [{genome, objectives, directions, score}], "n_children": N − elites, "seed", "generation", "budget"?}`; its result `{"children": [...]}` plus the elites become the next population, re-planned with the same rungs. A failed/cancelled/unusable variation result ends the generation (logged, `vary.failed`).

## Neural decoders, MCMC pairs and budget (spec 34 §6.2–6.4)
- **Warm start:** the orchestrator keeps each genome's latest bundle (`neural.weights {persisted, bucket, key}` of its eval result). An `nn_*` genome's next rung and next loop get `"warm_start": "<bucket>/<key>"` of its own bundle; a child of `evo.vary.decoder` gets its first parent's (`lineage[].parents_as_sent`, the ids this orchestrator sent; `parents` are the worker's canonical forms).
- **Rung fidelity:** `nn_*` eval envelopes carry `"train_steps_fraction": η^-(R−1−r)` — rung `r` of `R` trains that share of the genome's SGD steps, the last rung the full count.
- **MCMC pairs:** `latent.crossprior.>` (`CrossAssetPriors`, D-069) is kept per timescale (`pairs.go`); every decoder loop (first and after each variation) adds up to `mcmc_pairs` (`ctl.evo.start` arg, default `-mcmc-pairs` / `FA_EVO_MCMC_PAIRS` = 4) `nn_forecast` genomes `…;inst=<target>;src=<source>;ts=<ts>` built from the population's first `nn_forecast` genome (else a small GRU template), best proposal score first; pairs never count towards `n_children`.
- **Budget:** `used_gpu_ms` of every result is summed per loop and per generation; with `total_budget_ms`, a rung whose survivors × per-candidate budget would push the loop beyond 110 % of it is skipped and the loop ends on the partial results (`budget.truncated`, note). Summaries carry `budget`, `warm_started`, `pairs_seeded`; heartbeat `budget.used_ms|loop_used_ms|truncations`, `warm_start.jobs`, `pairs.seeded`, `crossprior.messages|proposals`.

## Decoder bookkeeping jobs (`lineage.go`, spec 34 §5 / §8)
- **Fronts → Helix:** every final (loop-end) summary of a decoder generation is sent as an `evo.front.decoder` job (`{generation_id, generation, kind, snapshot_id, keys, fronts: [[{genome, rank, crowding, hv, score}]], final}`); the worker stores it as a `ParetoFront` node (`<generation>/<loop>`) with `InFront` edges to the member genomes. Fire and forget (counter `fronts.jobs`).
- **Elite re-evaluation:** `ctl.evo.reeval` sends one `evo.reeval.decoder` job; and when `latent.lifecycle.<ts>` shows a new snapshot id for a timescale (a swap; the first sighting only records it), a job for that timescale and snapshot is sent `FA_EVO_REEVAL_DELAY_MIN` minutes later (default 120, `0` disables) with a data window of that length, so it reads only the new snapshot's latents. Counter `reeval.jobs`.

## Pareto summaries
After every rung (and once more at loop end over all evaluated genomes) a `generation.summary` JSON is published as `JobResult{job_id: <generation id>, status: OK, worker_id: "evo-orchestrator", result: <json>}` on **`jobs.result.evo.generation`** (so `jobs.result.>` consumers decode it as a normal result; control-api lists it under kind `evo.generation`):
```json
{"type":"generation.summary","generation_id":"01M7…","kind":"evo.eval.decoder","generation":0,"max_generations":2,"rung":0,"rungs":4,"final":false,
 "candidates":6,"unranked":0,"objective_keys":["compute_cost","coverage_err","crps","crps_skill","hit_rate","trade_psr","trade_sharpe"],
 "front_sizes":[5,1],"pareto_front":[{"genome":"forecast:h=3;lb=16","rank":0,"crowding":1.7976931348623157e+308,"objectives":{...},"score":0.326}],
 "survivors":["forecast:h=3;lb=16","g1","retrieval:h=2;lb=12;k=20;tau=5"],"fronts":[[...],[...]],"next_population":6,"vary_job_id":"01M7…","at":"…","elapsed_ms":4300}
```
`crowding` `+∞` is encoded as `1.7976931348623157e+308` (max float64); `hv` is the member's exclusive hypervolume contribution inside its front; `snapshot_id` names the encoder snapshot the evaluations read. `survivors` are the next rung's genomes, or the elites when `final`. Further fields: `state` (`running` | `paused` | `stopped` | `done`), `validation` = the latest `generation_stats` object a worker returned in an eval result of this generation (`{n_trials, sharpe_mean, sharpe_var, best_candidate, best_sharpe, expected_max_sharpe, deflated_sharpe, pbo, pbo_trials, pbo_blocks}`, spec 41 §6.5; the worker recomputes it over the Postgres trial registry after every evaluation, so the orchestrator needs no database and no extra job), and on the final summary of a strategy generation `artifact_key`.

## Promotion hand-off (spec 35 §6.4)
When a `evo.eval.strategy` generation ends (last loop, or `ctl.evo.stop` is **not** promoted), the final Pareto front is written to Garage as **`fa-artifacts/strategies/<generation_id>.toml`** (`promote.go`: a `strategies.toml` population file — `instruments = []` and `strategy = [ {…}, … ]` with the genomes' inline tables, one per front member, commented with rank and score) and the key is announced as `artifact_key` in the final `generation.summary`; the risk-engine's strategy pipeline (D-071) reads the newest such file, admits members that pass its PBO/DSR and holdout gates into `fa-artifacts/live/strategies.toml` and reloads strategy-runtime. The Garage client is `go/pkg/archive` (aws-sdk-go-v2, path style) with `[garage] access_key_id / secret_access_key` from `-secrets` / `FA_SECRETS` (default `config/secrets.toml`) against `-garage` / `FA_GARAGE_URL` (default `http://127.0.0.1:3900`; in compose set `FA_GARAGE_URL=http://garage:3900`). Without credentials the summary carries no `artifact_key` and the note says why. **Allocator generations** (`evo.eval.allocator`) are archived the same way (`PromoteFronts` dispatches on the kind; `PromoteStrategies` remains as an alias) to **`fa-artifacts/allocators/<generation_id>.toml`**: a header with the `ctl.risk.limits` payload of the active member (`set={"allocator":{"name":…,"params":{…}}}`), an `[active]` table (the rank-0 member with the best score = deflated Sharpe) and one `[[allocator]]` table per front member with `rank`, `score`, `genome` (the `alloc:` text), `name` and `params` (inline table, the `fa_allocator::genome::AllocatorSpec` form). Non-`alloc:` genomes are skipped. risk-engine accepts that payload at runtime as `ctl.risk.limits {set: {allocator: {name, params}}}` (D-064). Decoder generations are not archived here (their eval reports already live under `fa-artifacts/evals/`).

## Heartbeat counters
(also on `/metrics` as `fa_heartbeat_counter{service="evo-orchestrator"}`) `gpu.capacity_mb` (from `-gpu-mem-mb`, D-031), `gpu.used_mb`, `gpu.active`, `gpu.queued`, `generations.started`, `generations.done`, `generations.active`, `generations.loops` (variation rounds started), `generations.paused`, `generations.stopped`, `jobs.cancelled`, `vary.jobs`, `vary.failed`, `promotions.artifacts`, `errors.decode`, and the latest summary as gauges: `pareto.front_size`, `pareto.fronts`, `pareto.candidates`, `pareto.unranked`, `pareto.generation`, `pareto.rung`, `validation.n_trials`, `validation.deflated_sharpe`, `validation.pbo`.

## Run
```sh
go/bin/evo-orchestrator -nats nats://127.0.0.1:4222 -gpu-mem-mb 24000 -max-generations 2
```
| Flag | Env | Default |
|---|---|---|
| `-nats` | `FA_NATS_URL` | `nats://127.0.0.1:4222` |
| `-gpu-mem-mb` | `FA_GPU_MEM_MB` | `24000` (MB managed by the arbiter and reported as capacity) |
| `-max-generations` | `FA_EVO_MAX_GENERATIONS` | `1` (default `max_generations` for `ctl.evo.start`) |
| `-garage` | `FA_GARAGE_URL` | `http://127.0.0.1:3900` (promoted strategy populations; empty disables) |
| `-secrets` | `FA_SECRETS` | `config/secrets.toml` (`[garage]` key for the artifact writes, D-020) |
| `-mcmc-pairs` | `FA_EVO_MCMC_PAIRS` | `4` (default `mcmc_pairs`: cross-asset `nn_forecast` genomes per decoder loop; 0 disables) |
| `-metrics` | `FA_METRICS_BIND` | host port of `docker/README.md` (Prometheus `/metrics`; compose `0.0.0.0:9100`, published as `127.0.0.1:9107`; `off` disables) |

## Tests
`TestNSGAMatchesSharedFixture` reads `testdata/nsga_fixture.json` (shared with `crates/fa-evolution`, which reads it from here), so the plain `just go-test` (only `go/` mounted) runs every test.

## Owns
Generation scheduler (`scheduler.go`), ranking and halving (`halving.go`, `nsga.go`), promotion artifacts for strategy and allocator fronts (`promote.go`), GPU arbiter (`arbiter.go`, D-006).

## Governing specs
- Spec 34 (retired): [fa-decoders README](../../../crates/fa-decoders/README.md#decoder-population-lasting-content-of-the-retired-spec-34) (§6.2–6.5 here)
- [35-strategies](../../../specs/35-strategies.md) §6.3–6.4, [Execution & risk (ex spec 40)](../../../bins/risk-engine/README.md#execution-and-risk-lasting-content-of-the-retired-spec-40) §6.2, [41-backtest-and-validation](../../../specs/41-backtest-and-validation.md) §6.5
- [Ops reference (ex spec 60)](../../../docker/README.md#reference-spec-60-ops-retired-2026-11-11); decisions D-006, D-025, D-037, D-044, D-049, D-052
