# fa-mcmc

Markov state graph over latent regions with Bayesian transition kernels and pluggable samplers.

**Status:** implemented (v1) — Markov graph with Dirichlet posteriors + forgetting, credible intervals, conjugate path sampler, log-score calibration, rows rebuilt from published summaries, matrix powers, and **kernel transfer across snapshots** (tested).

## Kernel transfer (spec 33 §6.6, D-062)
`MarkovGraph::transfer(&old, &carries, &new_support, shrink) -> Transfer { graph, carried, dropped, rows_carried }` builds the graph of a new region set from the old one. `carries` are `RegionCarry { new_id, old_id, score }` (from a `region_map`: a region with a `parent_region_id` and its Jaccard `score`). For a carried source row, each old successor count `c` is mapped onto the new regions carrying that successor in proportion to their scores (a split divides, a merge sums), and the row is scaled by `score(source) · shrink` (`carry_discount`); `unknown` mass is carried as is; successors without a destination are dropped (`dropped`); unmatched new regions get no row (they start from the symmetric prior); every new id plus `new_support` is registered as live support so it keeps prior mass in every row. Tests: carried rows with score·shrink scaling and row sums of 1, unmatched → prior, splits/merges, zero score/shrink. model-runtime calls it from `TimescalePipeline::swap_encoder`.

## Owns
`Sampler` trait, kernel posteriors, cross-asset priors.

## Governing specs
- [33-mcmc-graph](../../specs/33-mcmc-graph.md)
