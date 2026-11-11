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

use crate::{
    DiscoveryMethod, Region,
    lineage::{AlignCfg, DEFAULT_REGIME_BREAK_MASS, DEFAULT_SPLIT_THRESHOLD, DescentKind, align},
};

/// One lineage edge of a fitted region (`DESCENDS_FROM`, spec 30 §5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ancestor {
    pub region_id: String,
    pub lineage_id: String,
    /// Jaccard overlap of the probe sets.
    pub overlap: f64,
    pub kind: DescentKind,
}

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
    /// Every parent region this one descends from (carry, split, merge, partial), with overlap.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ancestors: Vec<Ancestor>,
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
    /// Parent regions without any descendant (`died_at` at promotion, spec 30 §6.4 step 6).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dead_parents: Vec<String>,
    /// Parent regions that split into ≥ 2 new regions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub split_parents: Vec<String>,
    /// Carried mass below 0.5 with a parent present: a regime break of the representation
    /// (spec 30 §8). The runtime then resets the transition priors instead of transferring.
    #[serde(default)]
    pub regime_break: bool,
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

    /// Any expert's alignment was a representation regime break.
    pub fn regime_break(&self) -> bool {
        self.experts.iter().any(|e| e.regime_break)
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

/// Aligns the new regions of one expert to the parent's with the default split threshold
/// (0.2); see [`align_expert_with`].
pub fn align_expert(input: AlignInput<'_>) -> ExpertRegionMap {
    align_expert_with(input, DEFAULT_SPLIT_THRESHOLD)
}

/// Aligns the new regions of one expert to the parent's (Hungarian carry, splits, merges,
/// births, deaths: [`crate::lineage::align`]). Masses come from `new_assign`.
pub fn align_expert_with(input: AlignInput<'_>, split_threshold: f64) -> ExpertRegionMap {
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
    let with_parent = old_assign.len() == new_assign.len() && !old_ids.is_empty();
    let cfg = AlignCfg { carry_threshold, split_threshold, regime_break_mass: DEFAULT_REGIME_BREAK_MASS };
    let alignment = with_parent.then(|| align(old_assign, new_assign, new_centroids.len(), &cfg));
    let mut carried = 0usize;
    let mut matched = 0usize;
    let regions = new_centroids
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let id = region_id(snapshot_id, expert_index, i);
            let ancestors: Vec<Ancestor> = alignment
                .as_ref()
                .map(|a| {
                    a.ancestors(i)
                        .filter_map(|d| {
                            old_ids.get(d.old_index).map(|(rid, lin)| Ancestor {
                                region_id: rid.clone(),
                                lineage_id: lin.clone(),
                                overlap: d.overlap,
                                kind: d.kind,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            let carry = ancestors.iter().find(|a| a.kind == DescentKind::Carry);
            let (lineage, parent, score) = match carry {
                Some(a) => {
                    carried += counts[i];
                    matched += 1;
                    (a.lineage_id.clone(), Some(a.region_id.clone()), a.overlap)
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
                ancestors,
            }
        })
        .collect();
    let parent_regions = {
        let mut seen: Vec<usize> = old_assign.iter().flatten().copied().collect();
        seen.sort_unstable();
        seen.dedup();
        seen.len()
    };
    let ids = |v: &[usize]| v.iter().filter_map(|o| old_ids.get(*o).map(|x| x.0.clone())).collect::<Vec<_>>();
    let (dead_parents, split_parents, regime_break) = match &alignment {
        Some(a) => (ids(&a.dead), ids(&a.splits), a.regime_break),
        None => (Vec::new(), Vec::new(), false),
    };
    ExpertRegionMap {
        expert_index,
        regions,
        parent_regions,
        matched,
        carried_mass: carried as f64 / n,
        dead_parents,
        split_parents,
        regime_break,
    }
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
        assert_eq!(m.regions[0].ancestors.len(), 1);
        assert_eq!(m.regions[0].ancestors[0].kind, DescentKind::Carry);
        assert!(m.dead_parents.is_empty());
        assert!(!m.regime_break, "5/8 of the probe mass carried");
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
