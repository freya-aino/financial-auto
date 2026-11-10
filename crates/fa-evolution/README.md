# fa-evolution

Genomes, variation operators, NSGA-II selection, successive halving and the objective-vector conventions shared with the Go orchestrator.

**Status:** implemented (v1) — `Genome`, Gaussian mutation, uniform crossover, population variation (`vary`), NSGA-II non-dominated sort / crowding distance / selection, named-objective → minimisation mapping (`objectives`), successive halving + budget plan (`plan_halving`). All tested; the NSGA-II functions are the **reference implementation** for `go/services/evo-orchestrator/nsga.go`.

## Modules
| Module | What |
|---|---|
| `lib` | `Genome`, `Fitness`, `GaussianMutation`, `UniformCrossover`, `dominates`, `non_dominated_sort`, `crowding_distance`, `nsga2_select`, `successive_halving_keep`, `plan_halving` |
| `objectives` | `Direction`, `ranked_keys(directions)`, `minimised(objectives, directions, keys)` (max → negated, missing/non-finite → `+∞`), `pareto_ranks` |
| `vary` | `vary(parents, n_children, seed, &VarySpec)`: binary tournament on rank position (parents best first), same-family uniform crossover with `crossover_prob`, Gaussian mutation; lineage points at the real parents; deterministic in `seed` |

## Shared NSGA-II fixture
`testdata/nsga_fixture.json` holds objective sets with their expected fronts, crowding distances (`"inf"` = +∞) and `nsga2_select` results plus a named-objective mapping case. `tests::matches_shared_fixture` here and `TestNSGAMatchesSharedFixture` in `go/services/evo-orchestrator/nsga_test.go` both read it, so the Rust reference and the Go port cannot drift. Semantics both sides keep: stable sorts (ties keep input order), crowding extremes `+∞`, a non-finite objective span contributes nothing to interior distances.

## Owns
Rust-side evolution logic used by `evo-worker` (`evo.vary.decoder`) and mirrored by `evo-orchestrator` (ranking).

## Governing specs
- [34-decoder-population](../../specs/34-decoder-population.md) (§6.2 variation, §6.3 objectives, §6.4 budget, §6.5 orchestration)
- [35-strategies](../../specs/35-strategies.md)
