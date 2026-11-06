# fa-evolution

Genomes, mutation/crossover, ES+SGD candidate training, NSGA Pareto selection.

**Status:** implemented (v1) — genomes, Gaussian mutation, uniform crossover, NSGA-II sort/crowding/selection, successive halving + budget plan (tested).

## Owns
Rust-side evaluation logic used by `evo-worker`.

## Governing specs
- [34-decoder-population](../../specs/34-decoder-population.md)
- [35-strategies](../../specs/35-strategies.md)
