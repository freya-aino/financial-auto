# evo-orchestrator

Schedules decoder/strategy populations onto evo-workers under a GPU-hour budget.

**Status:** implemented (skeleton) — GPU arbiter with strict priority and preemption (`ctl.gpu.acquire|release`, notifications `ctl.gpu.preempt|granted`), successive-halving generations started via `ctl.evo.start`, jobs on `jobs.<kind>`, results from `jobs.result.>`. Scores are provisional JSON `{"score": x}`; NSGA Pareto ranking pending.

## Owns
Generation scheduler, successive halving.

## Governing specs
- [34-decoder-population](../../../specs/34-decoder-population.md)
- [60-ops](../../../specs/60-ops.md)
