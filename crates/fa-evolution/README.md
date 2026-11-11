# fa-evolution

Genomes, variation operators, NSGA-II / NSGA-III selection, hypervolume rung scores, successive halving and the objective-vector conventions shared with the Go orchestrator.

**Status:** implemented (v1) — `Genome`, Gaussian mutation, uniform crossover, population variation (`vary`, generic over the `Evolvable` / `Operators` traits so decoder, strategy and allocator genomes share tournament and crossover gating), NSGA-II non-dominated sort / crowding distance / selection, named-objective → minimisation mapping (`objectives`), hypervolume contributions and NSGA-III (`pareto`), successive halving + budget plan (`plan_halving`). All tested; the NSGA functions and the hypervolume are the **reference implementation** for `go/services/evo-orchestrator/{nsga,pareto}.go`.

## Modules
| Module | What |
|---|---|
| `lib` | `Genome`, `Fitness`, `GaussianMutation`, `UniformCrossover`, `dominates`, `non_dominated_sort`, `crowding_distance`, `nsga2_select`, `successive_halving_keep`, `plan_halving` |
| `pareto` | `normalise` (min–max per objective, missing → reference), `hv_contributions` (exclusive hypervolume w.r.t. reference 1.1: exact for ≤ 2 objectives, deterministic SplitMix64 Monte Carlo with 8192 samples otherwise), `front_points` (only the objectives every front member defines), `hv_rank` (fronts, then HV contribution, then crowding), `reference_directions` (Das–Dennis) / `divisions_for`, `nsga3_select` (Deb & Jain 2014 niching with ideal/nadir normalisation, closest member first, deterministic ties), `select` (NSGA-II ≤ 3 objectives, NSGA-III ≥ 4) — reference for `go/services/evo-orchestrator/pareto.go` |
| `objectives` | `Direction`, `ranked_keys(directions)`, `minimised(objectives, directions, keys)` (max → negated, missing/non-finite → `+∞`), `pareto_ranks` |
| `vary` | `vary_with<G: Evolvable, O: Operators<G>>(parents, n_children, seed, crossover_prob, &ops) -> Vec<Child<G>>`: binary tournament on rank position (parents best first), crossover only between members of the same `family()`, then `ops.mutate`; `Child { genome, parents, op }` names the real parents and the operator (`crossover+<op>` or `<op>`). `vary(parents: &[Genome], …, &VarySpec)` is the decoder instance (`DecoderOps`: uniform crossover + Gaussian mutation over `params` and the `expert_mask` bits, which carry the decoder genome's `mask` gene, spec 34 §6.2); `evo-worker` plugs `fa_strategies::ops` in for `evo.vary.strategy` and `fa_allocator::genome::AllocatorOps` for `evo.vary.allocator`. Deterministic in `seed` |

## Shared NSGA-II fixture
`go/services/evo-orchestrator/testdata/nsga_fixture.json` (kept under `go/` so `just go-test`, which mounts only `go/`, finds it) holds objective sets with their expected fronts, crowding distances (`"inf"` = +∞) and `nsga2_select` results plus a named-objective mapping case, and (key `pareto`) hypervolume orders / contributions and `nsga3_select` results (written by `FA_PRINT_PARETO_FIXTURE=1 cargo test -p fa-evolution matches_shared_pareto -- --nocapture`). `tests::matches_shared_fixture` / `pareto::fixture::matches_shared_pareto_fixture` here and `TestNSGAMatchesSharedFixture` / `TestParetoMatchesSharedFixture` in `go/services/evo-orchestrator` read it, so the Rust reference and the Go port cannot drift. Semantics both sides keep: stable sorts (ties keep input order), crowding extremes `+∞`, a non-finite objective span contributes nothing to interior distances.

## Owns
Rust-side evolution logic used by `evo-worker` (`evo.vary.decoder`, `evo.vary.strategy`, `evo.vary.allocator`, `plan_halving` sizing) and mirrored by `evo-orchestrator` (ranking); `control-api` reuses the NSGA-II functions for `GET /api/evolution/populations/{id}/pareto`.

## Governing specs
- Spec 34 (retired): [fa-decoders README](../fa-decoders/README.md#decoder-population-lasting-content-of-the-retired-spec-34) (§6.2 variation and selection, §6.3 objectives, Q6 normalisation, §6.4 budget, §6.5 orchestration)
- [35-strategies](../../specs/35-strategies.md), [Execution & risk (ex spec 40)](../../bins/risk-engine/README.md#execution-and-risk-lasting-content-of-the-retired-spec-40) §6.2 (allocator tournament)
