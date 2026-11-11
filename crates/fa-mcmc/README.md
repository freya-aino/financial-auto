# fa-mcmc

Markov state graph over latent regions with Bayesian transition kernels and pluggable samplers.

**Status:** implemented — Dirichlet Markov graph with the geometric structural prior, forgetting, soft update, credible intervals, log-score calibration, rows rebuilt from published summaries, matrix powers and **kernel transfer across snapshots**; six competing kernels in a calibration arena; NUTS / Gibbs batch refits; graph-level posterior inference; cross-asset priors (D-062, D-069, D-075, D-080; all tested, spec 33 §12).

## Modules
| Module | What |
|---|---|
| `lib` | `MarkovGraph` (Dirichlet rows, `set_centroid` geometric prior, `observe` / `observe_soft`, `summarize`, `transfer`), `TransitionMatrix` (`k_step`, `set_row_from_summary`), `Sampler` / `ConjugateSampler`, `log_score`, `hitting_time` |
| `kernel` | `KernelSampler` trait (`observe[_ctx]`, `predict[_ctx]`, `summarize[_ctx]`, `apply_refit`, `export_state` / `import_state`), `ConjugateExact`, `SmcParticle` (particles per row, tempered weights, resample + Gibbs move) |
| `rich` | `KernelCtx{instrument, prev, dwell, x}` and the richer kernels `SecondOrder` (`second_order`), `SemiMarkov` (`semi_markov`, Beta hazards per dwell bin), `HierAsset` (`hier_asset`, hierarchical Dirichlet across assets), `LogitCov` (`logit_cov`, logistic-normal on the covariates) |
| `calibration` | `Calibration` (prequential EW log score, Brier, ECE, PIT KS, cost), `Arena` (Pareto front, serving sampler with hysteresis; `export_calibration` / `import_calibration` → `ArenaCalibrationState`, persisted in `RegionSet.kernel_state`) |
| `nuts` | slice NUTS with dual averaging, split R-hat, ESS, divergences |
| `refit` | `jobs.mcmc.refit` computation (`run`: NUTS per `logit_cov` row, Gibbs for `hier_asset`; accepted with R-hat ≤ 1.05 / ≤ 1 % divergent / ESS ≥ 50, Gibbs R-hat ≤ 1.1), `ObsRecord` / `ObsDump` / `ObsWindow`, `.bin` format `FAMCMC01` (`encode` / `decode`), `COVARIATES` (`ln_dwell, vol_ratio, novelty, router_weight, peer_switch, coarse_ln_dwell`, `COV_DIM` = 6; older 4-covariate windows load zero-padded) |
| `inference` | `Posterior`: every spec 33 §4.2 query as a `Dist` over kernel draws (`transition_prob`, `k_step`, `neighbors`, `sample_paths`, `path_prob`, `hitting`, `stationary`, `relaxation`) |
| `cross_asset` | `CrossAssetPriors`: Ledoit–Wolf correlation, lead-lag, Δ CRPS gain on returns or expert residuals, ranked proposals, `proposals_with_evidence` (decoder win/loss evidence) |

## Kernel transfer (spec 33 §6.6, D-062)
`MarkovGraph::transfer(&old, &carries, &new_support, shrink) -> Transfer { graph, carried, dropped, rows_carried }` builds the graph of a new region set from the old one. `carries` are `RegionCarry { new_id, old_id, score }` (from a `region_map`: a region with a `parent_region_id` and its Jaccard `score`). For a carried source row, each old successor count `c` is mapped onto the new regions carrying that successor in proportion to their scores (a split divides, a merge sums), and the row is scaled by `score(source) · shrink` (`carry_discount`); `unknown` mass is carried as is; successors without a destination are dropped (`dropped`); unmatched new regions get no row (they start from the prior, geometric once centroids are registered); every new id plus `new_support` is registered as live support so it keeps prior mass in every row. Tests: carried rows with score·shrink scaling and row sums of 1, unmatched → prior, splits/merges, zero score/shrink. model-runtime calls it from `TimescalePipeline::swap_encoder`, except for an expert whose map is flagged `regime_break` (carried probe mass < 0.5, spec 30 §8): that kernel restarts from the prior.

## Owns
`Sampler` / `KernelSampler` traits, kernel posteriors and their competition, batch refits, graph-level inference, cross-asset priors.

## Governing specs
- [33-mcmc-graph](../../specs/33-mcmc-graph.md)
