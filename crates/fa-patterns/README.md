# fa-patterns

Latent region discovery and region graph maintenance.

**Status:** partial — Region/lineage types, `RegionDiscovery` trait, probe-set alignment (D-007), and the implemented **online leader clustering** (`online.rs`): `OnlineRegionSet` (streaming membership with birth / merge / retire lifecycle events, deterministic ids, novelty score) behind the `OnlineRegionDiscovery` stage trait, plus batch `LeaderClustering` behind `RegionDiscovery`. Density (HDBSCAN-like) and Mapper methods for the neural families are not implemented.

## Region maps (`region_map.rs`, spec 30 §6.4, D-058)
- `FittedRegion` (region_id, lineage_id, centroid, radius, mass, parent_region_id, score) and `RegionMap` (per expert: regions, matched count, `carried_mass`) are the `regions.json` / `region_map.json` of a trained snapshot (written by `trainer`).
- `align_expert(AlignInput)` runs `align_by_probes` on the parent's and the new snapshot's assignment of the same probe windows: a match above `carry_threshold` **inherits the parent's `lineage_id`**; region ids stay unique per snapshot (`<snapshot>-e<k>-r<nnn>`, `region_map::region_id`).
- `LeaderClustering::fit_assign(points)` discovers a region set and assigns every point (inside a radius, else the nearest centroid), deterministic in point order; shared by the trainer's region fit and its feedback search.
- `OnlineRegionSet::seeded(cfg, snapshot, expert, prefix, &[FittedRegion])` starts a live set from a map, so `model-runtime` continues the snapshot's lineages after a load or hot swap; later births continue the ordinal series.

## Owns
Region discovery per encoder family, cross-timescale edges.

## Governing specs
- [30-latent-regions](../../specs/30-latent-regions.md)
