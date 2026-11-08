# evo-orchestrator

Schedules decoder/strategy populations onto evo-workers under a GPU-hour budget.

**Status:** implemented (skeleton) — GPU arbiter with strict priority and preemption (`ctl.gpu.acquire|release`, notifications `ctl.gpu.preempt|granted`), successive-halving generations started via `ctl.evo.start`, jobs on `jobs.<kind>`, results from `jobs.result.>`, job cancel via `ctl.evo.cancel`. Scores are provisional JSON `{"score": x}`; NSGA Pareto ranking pending.

## Control (request/reply, `CtlRequest`/`CtlReply`)
| Subject | Args | Reply |
|---|---|---|
| `ctl.gpu.acquire` | `lease_id, class, mem_mb` | `granted` / `queued` |
| `ctl.gpu.release` | `lease_id` | `released` |
| `ctl.evo.start` | `kind, snapshot_id, genomes, eta, min_budget_ms, max_budget_ms` | generation id |
| `ctl.evo.cancel` | `job_id` | `cancelled <id>`; `not_found: …` when no generation has the job pending |

Cancel removes the job from its generation (it counts as a failed candidate, so the rung can still complete), releases the GPU lease named after the job id (convention until workers report lease ids), and publishes `JobResult{status: FAILED, error: "cancelled by operator", worker_id: "evo-orchestrator"}` on `jobs.result.<kind>` so control-api's job list shows it. The proto has no `CANCELLED` status yet. A result the worker publishes later for that job id is ignored.

## Heartbeat counters
`gpu.capacity_mb` (from `-gpu-mem-mb`, D-031), `gpu.used_mb`, `gpu.active`, `gpu.queued`, `generations.started`, `generations.done`, `jobs.cancelled`, `errors.decode`.

## Run
```sh
go/bin/evo-orchestrator -nats nats://127.0.0.1:4222 -gpu-mem-mb 24000
```
| Flag | Env | Default |
|---|---|---|
| `-nats` | `FA_NATS_URL` | `nats://127.0.0.1:4222` |
| `-gpu-mem-mb` | `FA_GPU_MEM_MB` | `24000` (MB managed by the arbiter and reported as capacity) |

## Owns
Generation scheduler, successive halving, GPU arbiter (D-006).

## Governing specs
- [34-decoder-population](../../../specs/34-decoder-population.md)
- [60-ops](../../../specs/60-ops.md)
