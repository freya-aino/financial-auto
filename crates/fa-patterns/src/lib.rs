//! Latent regions: discovery trait, lineage, and region identity across retraining (spec 30, D-007).
//!
//! [`online`] holds the streaming leader-clustering baseline (`OnlineRegionSet`) that the
//! live runtime uses for region discovery and membership until the family-specific
//! density/Mapper methods exist.

pub mod online;

use std::collections::HashMap;

pub use online::{
    Assignment, LeaderClustering, OnlineRegionCfg, OnlineRegionDiscovery, OnlineRegionSet, RegionLifecycle,
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
/// regions under the old and the new snapshot. Regions are matched by maximising the
/// Jaccard overlap of their probe sets (greedy, highest overlap first). Pairs below
/// `min_score` are left unmatched and get a new lineage.
pub fn align_by_probes(old_assign: &[Option<usize>], new_assign: &[Option<usize>], min_score: f64) -> Vec<RegionMatch> {
    assert_eq!(old_assign.len(), new_assign.len(), "probe sets must be the same windows");
    let mut old_sets: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut new_sets: HashMap<usize, Vec<usize>> = HashMap::new();
    for (w, (o, n)) in old_assign.iter().zip(new_assign).enumerate() {
        if let Some(o) = o {
            old_sets.entry(*o).or_default().push(w);
        }
        if let Some(n) = n {
            new_sets.entry(*n).or_default().push(w);
        }
    }
    let mut pairs = Vec::new();
    for (&n, ns) in &new_sets {
        for (&o, os) in &old_sets {
            let inter = ns.iter().filter(|w| os.contains(w)).count();
            if inter > 0 {
                let union = ns.len() + os.len() - inter;
                pairs.push((inter as f64 / union as f64, n, o));
            }
        }
    }
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut used_new = std::collections::HashSet::new();
    let mut used_old = std::collections::HashSet::new();
    let mut out = Vec::new();
    for (score, n, o) in pairs {
        if score < min_score || used_new.contains(&n) || used_old.contains(&o) {
            continue;
        }
        used_new.insert(n);
        used_old.insert(o);
        out.push(RegionMatch { new_index: n, old_index: Some(o), score });
    }
    let mut unmatched: Vec<usize> = new_sets.keys().copied().filter(|n| !used_new.contains(n)).collect();
    unmatched.sort_unstable();
    out.extend(unmatched.into_iter().map(|n| RegionMatch { new_index: n, old_index: None, score: 0.0 }));
    out.sort_by_key(|m| m.new_index);
    out
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
