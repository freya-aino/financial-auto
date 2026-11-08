# fa-patterns

Latent region discovery and region graph maintenance.

**Status:** partial — Region/lineage types, `RegionDiscovery` trait, probe-set alignment (D-007), and the implemented **online leader clustering** (`online.rs`): `OnlineRegionSet` (streaming membership with birth / merge / retire lifecycle events, deterministic ids, novelty score) behind the `OnlineRegionDiscovery` stage trait, plus batch `LeaderClustering` behind `RegionDiscovery`. Density (HDBSCAN-like) and Mapper methods for the neural families are not implemented.

## Owns
Region discovery per encoder family, cross-timescale edges.

## Governing specs
- [30-latent-regions](../../specs/30-latent-regions.md)
