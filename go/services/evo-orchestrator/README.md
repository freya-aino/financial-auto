# evo-orchestrator

Schedules decoder/strategy populations onto evo-workers under a GPU-hour budget and hosts the GPU arbiter (D-006).

**Status:** implemented — GPU arbiter with strict priority and preemption (`ctl.gpu.acquire|release`, notifications `ctl.gpu.preempt|granted`); generations started via `ctl.evo.start`: successive halving ranked by **NSGA-II over `JobResult.objectives`/`directions`** (scalar `score` fallback), per-rung **Pareto summaries** on `jobs.result.evo.generation` and heartbeat counters, and a **generation loop** (`max_generations` > 1): elites + `evo.vary.<kind>` children form the next population. Job cancel via `ctl.evo.cancel` emits `STATUS_CANCELLED`.

## Control (request/reply, `CtlRequest`/`CtlReply`)
| Subject | Args | Reply |
|---|---|---|
| `ctl.gpu.acquire` | `lease_id, class, mem_mb` | `granted` / `queued` |
| `ctl.gpu.release` | `lease_id` | `released` |
| `ctl.evo.start` | `kind, snapshot_id, genomes` (newline-separated, D-044)`, eta, min_budget_ms, max_budget_ms, max_generations, total_budget_ms, data_window_hours` | generation id |
| `ctl.evo.cancel` | `job_id` | `cancelled <id>`; `not_found: …` when no generation has the job pending |

- `max_generations` (default: `-max-generations` / `FA_EVO_MAX_GENERATIONS`, default 1): halving loops. 1 = a single successive halving (previous behaviour). control-api does not forward this argument yet; set the flag/env for a multi-generation run.
- `total_budget_ms` (optional): when set, the next population is sized by the worker with `fa_evolution::plan_halving` (spec 34 §6.4) instead of keeping the start size.
- `data_window_hours` (optional): forwarded to eval jobs as `Job.spec = {"genome", "data_window": {"hours"}}`.

Cancel removes the job from its generation (it counts as a failed candidate, so the rung can still complete; cancelling the pending variation job ends the generation), releases the GPU lease named after the job id (convention until workers report lease ids) and publishes `JobResult{status: CANCELLED, error: "cancelled by operator", worker_id: "evo-orchestrator"}` on `jobs.result.<kind>`. A result the worker publishes later for that job id is ignored.

## Generation loop
1. **Halving:** rungs from `Plan(n, eta, min_budget, max_budget)`; every rung submits one `jobs.<kind>` job per alive genome. When a rung completes the candidates are ranked (below) and the best `ceil(n/eta)` advance with `eta`× budget.
2. **Ranking (`Rank`/`Halve`, `halving.go` + `nsga.go`):** candidates are sorted by ID first (arrival order does not matter), then NSGA-II — non-dominated fronts over the minimised objective vectors (`max` objectives negated, missing values `+∞`), inside a front by crowding distance descending — so every prefix equals the `NSGA2Select` survivor set. Candidates without objectives (failed, cancelled, or scalar-only `{"score"}` results) come last; with no objectives at all the old scalar order applies. Objectives are read from the `JobResult.objectives`/`directions` proto maps first, then from the result JSON (D-044). `nsga.go` is a port of `crates/fa-evolution/src/lib.rs`; both are tested against `crates/fa-evolution/testdata/nsga_fixture.json`.
3. **Loop end:** elites = top `ceil(N/eta)` of every genome evaluated in the loop (latest objectives; failed genomes never carry over). If more loops remain, a `jobs.evo.vary.<family>` job (`VaryKind("evo.eval.decoder") = "evo.vary.decoder"`) carries `{"parents": [{genome, objectives, directions, score}], "n_children": N − elites, "seed", "generation", "budget"?}`; its result `{"children": [...]}` plus the elites become the next population, re-planned with the same rungs. A failed/cancelled/unusable variation result ends the generation (logged, `vary.failed`).

## Pareto summaries
After every rung (and once more at loop end over all evaluated genomes) a `generation.summary` JSON is published as `JobResult{job_id: <generation id>, status: OK, worker_id: "evo-orchestrator", result: <json>}` on **`jobs.result.evo.generation`** (so `jobs.result.>` consumers decode it as a normal result; control-api lists it under kind `evo.generation`):
```json
{"type":"generation.summary","generation_id":"01M7…","kind":"evo.eval.decoder","generation":0,"max_generations":2,"rung":0,"rungs":4,"final":false,
 "candidates":6,"unranked":0,"objective_keys":["compute_cost","coverage_err","crps","crps_skill","hit_rate","trade_psr","trade_sharpe"],
 "front_sizes":[5,1],"pareto_front":[{"genome":"forecast:h=3;lb=16","rank":0,"crowding":1.7976931348623157e+308,"objectives":{...},"score":0.326}],
 "survivors":["forecast:h=3;lb=16","g1","retrieval:h=2;lb=12;k=20;tau=5"],"fronts":[[...],[...]],"next_population":6,"vary_job_id":"01M7…","at":"…","elapsed_ms":4300}
```
`crowding` `+∞` is encoded as `1.7976931348623157e+308` (max float64). `survivors` are the next rung's genomes, or the elites when `final`.

## Heartbeat counters
`gpu.capacity_mb` (from `-gpu-mem-mb`, D-031), `gpu.used_mb`, `gpu.active`, `gpu.queued`, `generations.started`, `generations.done`, `generations.active`, `generations.loops` (variation rounds started), `jobs.cancelled`, `vary.jobs`, `vary.failed`, `errors.decode`, and the latest summary as gauges: `pareto.front_size`, `pareto.fronts`, `pareto.candidates`, `pareto.unranked`, `pareto.generation`, `pareto.rung`.

## Run
```sh
go/bin/evo-orchestrator -nats nats://127.0.0.1:4222 -gpu-mem-mb 24000 -max-generations 2
```
| Flag | Env | Default |
|---|---|---|
| `-nats` | `FA_NATS_URL` | `nats://127.0.0.1:4222` |
| `-gpu-mem-mb` | `FA_GPU_MEM_MB` | `24000` (MB managed by the arbiter and reported as capacity) |
| `-max-generations` | `FA_EVO_MAX_GENERATIONS` | `1` (default `max_generations` for `ctl.evo.start`) |

## Tests
`TestNSGAMatchesSharedFixture` reads `../../../crates/fa-evolution/testdata/nsga_fixture.json`, so run the Go tests with the repository root mounted:
```sh
docker run --rm -v "$PWD":/repo -w /repo/go -v fa-gomod:/go/pkg/mod -v fa-gocache:/root/.cache/go-build \
  -e GOFLAGS=-buildvcs=false golang:1.25-alpine go test ./services/evo-orchestrator/...
```

## Owns
Generation scheduler (`scheduler.go`), ranking and halving (`halving.go`, `nsga.go`), GPU arbiter (`arbiter.go`, D-006).

## Governing specs
- [34-decoder-population](../../../specs/34-decoder-population.md) (§6.2–6.5)
- [60-ops](../../../specs/60-ops.md); decisions D-006, D-025, D-037, D-044
