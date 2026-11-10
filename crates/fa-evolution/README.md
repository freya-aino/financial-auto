# fa-evolution

Genomes, variation operators, NSGA-II selection, successive halving and the objective-vector conventions shared with the Go orchestrator.

**Status:** implemented (v1) — `Genome`, Gaussian mutation, uniform crossover, population variation (`vary`, generic over the `Evolvable` / `Operators` traits so decoder and strategy genomes share tournament and crossover gating), NSGA-II non-dominated sort / crowding distance / selection, named-objective → minimisation mapping (`objectives`), successive halving + budget plan (`plan_halving`). All tested; the NSGA-II functions are the **reference implementation** for `go/services/evo-orchestrator/nsga.go`.

## Modules
| Module | What |
|---|---|
| `lib` | `Genome`, `Fitness`, `GaussianMutation`, `UniformCrossover`, `dominates`, `non_dominated_sort`, `crowding_distance`, `nsga2_select`, `successive_halving_keep`, `plan_halving` |
| `objectives` | `Direction`, `ranked_keys(directions)`, `minimised(objectives, directions, keys)` (max → negated, missing/non-finite → `+∞`), `pareto_ranks` |
| `vary` | `vary_with<G: Evolvable, O: Operators<G>>(parents, n_children, seed, crossover_prob, &ops) -> Vec<Child<G>>`: binary tournament on rank position (parents best first), crossover only between members of the same `family()`, then `ops.mutate`; `Child { genome, parents, op }` names the real parents and the operator (`crossover+<op>` or `<op>`). `vary(parents: &[Genome], …, &VarySpec)` is the decoder instance (`DecoderOps`: uniform crossover + Gaussian mutation); `evo-worker` plugs `fa_strategies::ops` in for `evo.vary.strategy`. Deterministic in `seed` |

## Shared NSGA-II fixture
`testdata/nsga_fixture.json` holds objective sets with their expected fronts, crowding distances (`"inf"` = +∞) and `nsga2_select` results plus a named-objective mapping case. `tests::matches_shared_fixture` here and `TestNSGAMatchesSharedFixture` in `go/services/evo-orchestrator/nsga_test.go` both read it, so the Rust reference and the Go port cannot drift. Semantics both sides keep: stable sorts (ties keep input order), crowding extremes `+∞`, a non-finite objective span contributes nothing to interior distances.

## Owns
Rust-side evolution logic used by `evo-worker` (`evo.vary.decoder`, `evo.vary.strategy`) and mirrored by `evo-orchestrator` (ranking); `control-api` reuses the NSGA-II functions for `GET /api/evolution/populations/{id}/pareto`.

## Governing specs
- [34-decoder-population](../../specs/34-decoder-population.md) (§6.2 variation, §6.3 objectives, §6.4 budget, §6.5 orchestration)
- [35-strategies](../../specs/35-strategies.md)
