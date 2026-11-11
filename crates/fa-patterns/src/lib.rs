//! Latent regions: discovery trait, lineage, and region identity across retraining (spec 30, D-007).
//!
//! [`online`] holds the streaming leader-clustering baseline (`OnlineRegionSet`) that the
//! live runtime uses for region discovery and membership; [`methods`] holds the family-specific
//! batch discoveries of spec 30 §6.2 (HDBSCAN density, Mapper, codebook) for any encoder output. [`lineage`] aligns region sets across refits (Hungarian
//! matching on probe overlap, [`hungarian`]) and classifies carry / split / merge / birth /
//! death; [`region_map`] stores the result per snapshot. [`hierarchy`] estimates the
//! cross-timescale `REFINES` edges, [`drift`] the occupancy drift monitor, and [`conformance`]
//! is the shared test every `RegionDiscovery` implementation must pass (spec 30 §9.1).

pub mod conformance;
pub mod drift;
pub mod hierarchy;
pub mod hungarian;
pub mod lineage;
pub mod methods;
pub mod online;
pub mod region_map;

pub use drift::{DriftMonitor, DriftReport};
pub use hierarchy::{HierEdge, HierarchyCfg, HierarchyTracker, HierarchyUpdate};
pub use lineage::{AlignCfg, Alignment, Descent, DescentKind, align};
pub use methods::{CodebookRegions, DensityClusters, MapperCover, discovery_for_method};
pub use online::{
    Assignment, LeaderClustering, OnlineRegionCfg, OnlineRegionDiscovery, OnlineRegionSet, RegionLifecycle, RegionStat,
    RetireReason,
};
pub use region_map::{
    AlignInput, Ancestor, ExpertRegionMap, FittedRegion, ParentIds, RegionMap, align_expert, align_expert_with,
};
use serde::{Deserialize, Serialize};

pub type RegionId = String;
pub type LineageId = String;

/// A region of one expert space within one encoder snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Region {
    pub id: RegionId,
    /// Stable identity across generations (D-007).
    pub lineage_id: LineageId,
    pub snapshot_id: String,
    pub expert_index: u32,
    pub centroid: Vec<f32>,
    pub radius: f32,
    pub method: DiscoveryMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryMethod {
    /// Codebook entries/combinations of discrete-latent families.
    Codebook,
    /// Density clustering on continuous manifolds.
    Density,
    /// Mapper / topological cover.
    Mapper,
    /// Streaming leader clustering (pure-Rust baseline, [`online`]).
    Leader,
}

/// Region discovery; each encoder family declares its own implementation (spec 30).
pub trait RegionDiscovery: Send + Sync {
    fn method(&self) -> DiscoveryMethod;
    /// Discover regions from latent points of one expert space.
    fn discover(&self, points: &[Vec<f32>]) -> Vec<Vec<f32>>;
    /// Assign a point to the index of a discovered region, or `None` if outside all regions.
    fn assign(&self, centroids: &[Vec<f32>], point: &[f32]) -> Option<usize> {
        nearest(centroids, point)
    }
}

pub fn nearest(centroids: &[Vec<f32>], p: &[f32]) -> Option<usize> {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, c.iter().zip(p).map(|(a, b)| (a - b) * (a - b)).sum::<f32>()))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

/// Result of matching regions of a new snapshot to the previous one.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionMatch {
    pub new_index: usize,
    /// Matched old region index, or `None` → new lineage.
    pub old_index: Option<usize>,
    /// Jaccard overlap of probe-window assignments.
    pub score: f64,
}

/// Probe-set alignment (D-007 fallback): a fixed set of probe windows is assigned to
/// regions under the old and the new snapshot, and regions are matched by **Hungarian
/// assignment** on the Jaccard overlap of their probe sets (only pairs at or above `min_score`
/// take part). Matched new regions carry the old identity; every other new region that holds a
/// probe is returned with `old_index: None` (new lineage). Splits, merges, births and deaths
/// are in [`lineage::align`].
pub fn align_by_probes(old_assign: &[Option<usize>], new_assign: &[Option<usize>], min_score: f64) -> Vec<RegionMatch> {
    let n_new = new_assign.iter().flatten().map(|i| i + 1).max().unwrap_or(0);
    let cfg = AlignCfg { carry_threshold: min_score, ..AlignCfg::default() };
    let a = lineage::align(old_assign, new_assign, n_new, &cfg);
    let mut held = vec![false; n_new];
    for n in new_assign.iter().flatten() {
        held[*n] = true;
    }
    (0..n_new)
        .filter(|&n| held[n])
        .map(|n| match a.carried[n] {
            Some(o) => RegionMatch {
                new_index: n,
                old_index: Some(o),
                score: a.descents.iter().find(|d| d.new_index == n && d.old_index == o).map_or(0.0, |d| d.overlap),
            },
            None => RegionMatch { new_index: n, old_index: None, score: 0.0 },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_picks_closest() {
        let c = vec![vec![0.0, 0.0], vec![5.0, 5.0]];
        assert_eq!(nearest(&c, &[4.0, 4.5]), Some(1));
        assert_eq!(nearest(&[], &[1.0]), None);
    }

    #[test]
    fn alignment_handles_relabelling_and_new_regions() {
        // Old regions 0,1. New snapshot relabels them as 1,0 and adds region 2.
        let old = [Some(0), Some(0), Some(0), Some(1), Some(1), None, Some(1)];
        let new = [Some(1), Some(1), Some(1), Some(0), Some(0), Some(2), Some(2)];
        let m = align_by_probes(&old, &new, 0.3);
        assert_eq!(m.len(), 3);
        assert_eq!(m[0].old_index, Some(1)); // new 0 ← old 1
        assert_eq!(m[1].old_index, Some(0)); // new 1 ← old 0
        assert_eq!(m[1].score, 1.0);
        assert_eq!(m[2].old_index, None); // new 2 too weak a match → new lineage
    }
}
