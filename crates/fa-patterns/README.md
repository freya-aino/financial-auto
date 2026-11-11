# fa-patterns

Latent region discovery and region graph maintenance.

**Status:** implemented (D-007, D-058, D-069, D-075, D-080) — Region/lineage types, the batch `RegionDiscovery` trait (`discover`, `discover_with_residuals`, `assign`) and the online `OnlineRegionDiscovery` trait (`observe`, `stats`, `export_state` / `restore_state`); all discovery methods pass the shared conformance test (`conformance::assert_conformance`).

## Modules
| Module | What |
|---|---|
| `online` | `OnlineRegionSet`: streaming leader clustering with birth / merge / **split** (streaming 2-means per region) / **death by decayed support** lifecycle events, deterministic ids, soft memberships (k nearest, Gaussian kernel), novelty = distance / radius, serialisable state (persisted by model-runtime in Helix `RegionSet`); batch `LeaderClustering` |
| `methods` | pure-Rust batch discovery for any encoder output: `DensityClusters` (HDBSCAN), `MapperCover` (PCA filter, overlapping cover, nerve components), `CodebookRegions` (k-means++ codebook; Gabriel-neighbour codes merge while a KS test of their pooled next-step residuals gives p ≥ 0.05, `ks_pvalue`; geometric merge without residuals); `discovery_for_method(density\|mapper\|codebook\|leader, radius, max_regions)`, `fit_assign`. Neural families route through `fa_encoders::neural::regions::discovery_for` (spec 30 §6.2) |
| `lineage`, `hungarian` | `align`: Jaccard probe overlap, Hungarian carry ≥ `carry_threshold`, other pairs ≥ `split_threshold` as `DescendsFrom{carry\|split\|merge\|partial}`, births, deaths, regime break (carried mass < 0.5) |
| `hierarchy` | `HierarchyTracker`: cross-timescale `REFINES` edges by bias-corrected mutual information |
| `drift` | `DriftMonitor`: occupancy KL vs the fitted masses, `refit_due` |
| `conformance` | shared fit / assign / align test on synthetic Gaussian mixtures (spec 30 §9.1) |

## Region maps (`region_map.rs`, spec 30 §6.4, D-058)
- `FittedRegion` (region_id, lineage_id, centroid, radius, mass, parent_region_id, score) and `RegionMap` (per expert: regions, matched count, `carried_mass`) are the `regions.json` / `region_map.json` of a trained snapshot (written by `trainer`).
- `align_expert(AlignInput)` (`align_expert_with` for a non-default `split_threshold`, 0.2) runs `lineage::align` on the parent's and the new snapshot's assignment of the same probe windows: a Hungarian match ≥ `carry_threshold` **inherits the parent's `lineage_id`**; per-region `ancestors`, per-expert `dead_parents`, `split_parents` and `regime_break` are recorded in the map (`align_by_probes` remains as the carry-only view); region ids stay unique per snapshot (`<snapshot>-e<k>-r<nnn>`, `region_map::region_id`).
- `LeaderClustering::fit_assign(points)` discovers a region set and assigns every point (inside a radius, else the nearest centroid), deterministic in point order; shared by the trainer's region fit and its feedback search.
- `OnlineRegionSet::seeded(cfg, snapshot, expert, prefix, &[FittedRegion])` starts a live set from a map, so `model-runtime` continues the snapshot's lineages after a load or hot swap; later births continue the ordinal series.

## Owns
Region discovery per encoder family, region identity across snapshots, cross-timescale edges, occupancy drift.

## Governing specs
- [30-latent-regions](../../specs/30-latent-regions.md)
