//! Region sets of a trained snapshot and their alignment to the parent (spec 30 §6.4, D-007).
//!
//! The trainer fits a region set per expert on the probe windows of a new snapshot
//! ([`FittedRegion`], written as `regions.json`) and aligns it to the parent's regions with
//! [`crate::align_by_probes`]: both snapshots assign the same probe windows, regions are matched
//! by Jaccard overlap above `carry_threshold`, and a matched region **inherits the parent's
//! `lineage_id`** (its `region_id` stays unique per snapshot, as the Helix `LatentRegion` index
//! requires). Unmatched regions open a new lineage. The result, [`RegionMap`], is stored as
//! `region_map.json` next to the manifest and seeds the live runtime's region sets
//! ([`crate::OnlineRegionSet::seeded`]) when the snapshot is loaded or hot-swapped.

use serde::{Deserialize, Serialize};

use crate::{DiscoveryMethod, Region, RegionMatch, align_by_probes};

/// One region of a fitted set, with its lineage after alignment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FittedRegion {
    pub region_id: String,
    pub lineage_id: String,
    pub expert_index: u32,
    pub centroid: Vec<f32>,
    pub radius: f32,
    /// Fraction of probe windows assigned to this region.
    pub mass: f64,
    /// Parent region this one inherited its lineage from (`None` → new lineage).
    #[serde(default)]
    pub parent_region_id: Option<String>,
    /// Jaccard overlap with the parent region (0 when unmatched).
    #[serde(default)]
    pub score: f64,
}

impl FittedRegion {
    pub fn to_region(&self, snapshot_id: &str) -> Region {
        Region {
            id: self.region_id.clone(),
            lineage_id: self.lineage_id.clone(),
            snapshot_id: snapshot_id.to_owned(),
            expert_index: self.expert_index,
            centroid: self.centroid.clone(),
            radius: self.radius,
            method: DiscoveryMethod::Leader,
        }
    }
}

/// Alignment of one expert's regions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertRegionMap {
    pub expert_index: u32,
    pub regions: Vec<FittedRegion>,
    /// Number of parent regions that were assigned at least one probe.
    pub parent_regions: usize,
    pub matched: usize,
    /// Probe mass landing in regions that kept a lineage (spec 36 §6.4 `min_carry_mass`).
    pub carried_mass: f64,
}

/// `region_map.json`: the fitted regions of every expert and how they map onto the parent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionMap {
    pub snapshot_id: String,
    #[serde(default)]
    pub parent_snapshot_id: Option<String>,
    pub probe_set_version: String,
    pub n_probes: usize,
    pub carry_threshold: f64,
    pub experts: Vec<ExpertRegionMap>,
    /// Why no parent assignment was available, when the map carries no lineage at all.
    #[serde(default)]
    pub note: Option<String>,
}

impl RegionMap {
    pub fn to_json(&self) -> serde_json::Result<Vec<u8>> {
        serde_json::to_vec_pretty(self)
    }

    pub fn from_json(bytes: &[u8]) -> serde_json::Result<Self> {
        serde_json::from_slice(bytes)
    }

    pub fn expert(&self, k: u32) -> Option<&ExpertRegionMap> {
        self.experts.iter().find(|e| e.expert_index == k)
    }

    /// Mass-weighted carry over all experts.
    pub fn carried_mass(&self) -> f64 {
        if self.experts.is_empty() {
            return 0.0;
        }
        self.experts.iter().map(|e| e.carried_mass).sum::<f64>() / self.experts.len() as f64
    }

    pub fn n_regions(&self) -> usize {
        self.experts.iter().map(|e| e.regions.len()).sum()
    }

    pub fn n_matched(&self) -> usize {
        self.experts.iter().map(|e| e.matched).sum()
    }
}

/// Region id of the `i`-th region of expert `k` in `snapshot_id`: `<snapshot_id>-e<k>-r<nnn>`.
pub fn region_id(snapshot_id: &str, expert_index: u32, ordinal: usize) -> String {
    format!("{snapshot_id}-e{expert_index}-r{ordinal:03}")
}

/// Parent regions as the aligner needs them: `(region_id, lineage_id)` per parent index.
pub type ParentIds = Vec<(String, String)>;

/// Input of [`align_expert`] for one expert.
#[derive(Debug, Clone, Copy)]
pub struct AlignInput<'a> {
    pub snapshot_id: &'a str,
    pub expert_index: u32,
    /// Centroids of the new region set (index = new region index).
    pub new_centroids: &'a [Vec<f32>],
    pub radius: f32,
    /// New region index of every probe window (`None` = outside every region).
    pub new_assign: &'a [Option<usize>],
    /// Parent region index of the same probe windows (same length, or empty = no parent).
    pub old_assign: &'a [Option<usize>],
    /// `(region_id, lineage_id)` of parent region `i`.
    pub old_ids: &'a ParentIds,
    /// Jaccard overlap for lineage carry-over.
    pub carry_threshold: f64,
}

/// Aligns the new regions of one expert to the parent's. Masses come from `new_assign`.
pub fn align_expert(input: AlignInput<'_>) -> ExpertRegionMap {
    let AlignInput {
        snapshot_id,
        expert_index,
        new_centroids,
        radius,
        new_assign,
        old_assign,
        old_ids,
        carry_threshold,
    } = input;
    let n = new_assign.len().max(1) as f64;
    let mut counts = vec![0usize; new_centroids.len()];
    for a in new_assign.iter().flatten() {
        if let Some(c) = counts.get_mut(*a) {
            *c += 1;
        }
    }
    let matches: Vec<RegionMatch> = if old_assign.len() == new_assign.len() && !old_ids.is_empty() {
        align_by_probes(old_assign, new_assign, carry_threshold)
    } else {
        Vec::new()
    };
    let mut carried = 0usize;
    let mut matched = 0usize;
    let regions = new_centroids
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let id = region_id(snapshot_id, expert_index, i);
            let m = matches.iter().find(|m| m.new_index == i && m.old_index.is_some());
            let (lineage, parent, score) = match m.and_then(|m| old_ids.get(m.old_index?).map(|o| (o, m.score))) {
                Some(((old_id, old_lineage), score)) => {
                    carried += counts[i];
                    matched += 1;
                    (old_lineage.clone(), Some(old_id.clone()), score)
                }
                None => (id.clone(), None, 0.0),
            };
            FittedRegion {
                region_id: id,
                lineage_id: lineage,
                expert_index,
                centroid: c.clone(),
                radius,
                mass: counts[i] as f64 / n,
                parent_region_id: parent,
                score,
            }
        })
        .collect();
    let parent_regions = {
        let mut seen: Vec<usize> = old_assign.iter().flatten().copied().collect();
        seen.sort_unstable();
        seen.dedup();
        seen.len()
    };
    ExpertRegionMap { expert_index, regions, parent_regions, matched, carried_mass: carried as f64 / n }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matched_regions_inherit_lineage_and_unmatched_open_new_ones() {
        // Parent regions p0, p1 (lineages L0, L1). New set: region 1 ≈ p0, region 0 ≈ p1, region 2 new.
        let old = [Some(0), Some(0), Some(0), Some(1), Some(1), None, Some(1), None];
        let new = [Some(1), Some(1), Some(1), Some(0), Some(0), Some(2), Some(2), Some(2)];
        let centroids = vec![vec![1.0], vec![0.0], vec![5.0]];
        let parents: ParentIds = vec![("p0".into(), "L0".into()), ("p1".into(), "L1".into())];
        let m = align_expert(AlignInput {
            snapshot_id: "snap",
            expert_index: 0,
            new_centroids: &centroids,
            radius: 1.0,
            new_assign: &new,
            old_assign: &old,
            old_ids: &parents,
            carry_threshold: 0.3,
        });
        assert_eq!(m.regions.len(), 3);
        assert_eq!(m.regions[0].region_id, "snap-e0-r000");
        assert_eq!(m.regions[0].lineage_id, "L1");
        assert_eq!(m.regions[0].parent_region_id.as_deref(), Some("p1"));
        assert_eq!(m.regions[1].lineage_id, "L0");
        assert_eq!(m.regions[1].score, 1.0);
        assert_eq!(m.regions[2].lineage_id, "snap-e0-r002", "new lineage");
        assert_eq!(m.regions[2].parent_region_id, None);
        assert_eq!(m.matched, 2);
        assert_eq!(m.parent_regions, 2);
        assert!((m.carried_mass - 5.0 / 8.0).abs() < 1e-12);
        assert!((m.regions[2].mass - 3.0 / 8.0).abs() < 1e-12);

        // No parent → every region opens a lineage, nothing carried.
        let m0 = align_expert(AlignInput {
            snapshot_id: "snap",
            expert_index: 1,
            new_centroids: &centroids,
            radius: 1.0,
            new_assign: &new,
            old_assign: &[],
            old_ids: &ParentIds::new(),
            carry_threshold: 0.5,
        });
        assert_eq!(m0.matched, 0);
        assert_eq!(m0.carried_mass, 0.0);
        assert!(m0.regions.iter().all(|r| r.lineage_id == r.region_id));

        let map = RegionMap {
            snapshot_id: "snap".into(),
            parent_snapshot_id: Some("parent".into()),
            probe_set_version: "probe-1m-8-x".into(),
            n_probes: 8,
            carry_threshold: 0.3,
            experts: vec![m, m0],
            note: None,
        };
        let back = RegionMap::from_json(&map.to_json().unwrap()).unwrap();
        assert_eq!(back, map);
        assert_eq!(back.n_regions(), 6);
        assert_eq!(back.n_matched(), 2);
        assert!((back.carried_mass() - 5.0 / 16.0).abs() < 1e-12);
        assert_eq!(back.expert(1).unwrap().regions[0].to_region("snap").method, DiscoveryMethod::Leader);
    }
}
