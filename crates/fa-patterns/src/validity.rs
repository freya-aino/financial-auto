//! Region-set collapse guard (spec 30 §8: "discovery collapses into 1 region or hits
//! `max_regions_per_set` → mark the set invalid, keep the previous region set active, alert").
//!
//! A region set is **invalid** when it is degenerate as a discrete vocabulary:
//! - [`Collapse::Single`]: at most one region holds the data (every point maps to one state, so
//!   the MCMC graph has nothing to transition between);
//! - [`Collapse::AtCap`]: the set reached the cap `max_regions_per_set` (the method truncated
//!   its output or the online set stopped birthing, so regions no longer follow the data).
//!
//! [`check_set`] is the pure rule; [`RegionMap::collapsed`] applies it to every expert of a
//! trained snapshot's map, [`check_online`] to a live set after warm-up. The consequences are the
//! caller's: model-runtime refuses a hot swap onto a map with a collapsed expert (the previous
//! snapshot and its sets keep serving), seeds a collapsed expert empty when there is nothing to
//! keep (start, shadow), and raises `regions_invalid.<ts>.e<k>` (alert `RegionSetInvalid`).

use serde::{Deserialize, Serialize};

use crate::RegionMap;

/// Why a region set is invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Collapse {
    /// ≤ 1 region.
    Single,
    /// `n_regions ≥ max_regions_per_set`.
    AtCap,
}

impl Collapse {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single_region",
            Self::AtCap => "at_cap",
        }
    }
}

/// The rule: `n_regions ≤ 1` → [`Collapse::Single`]; `n_regions ≥ cap` (cap ≥ 2) →
/// [`Collapse::AtCap`]; else valid. A cap of 0 disables the cap check.
pub fn check_set(n_regions: usize, cap: usize) -> Option<Collapse> {
    if n_regions <= 1 {
        Some(Collapse::Single)
    } else if cap >= 2 && n_regions >= cap {
        Some(Collapse::AtCap)
    } else {
        None
    }
}

/// A live (online) set: judged only after `min_observations` (a fresh set legitimately starts
/// with one region).
pub fn check_online(live_regions: usize, observations: u64, min_observations: u64, cap: usize) -> Option<Collapse> {
    if observations < min_observations {
        return None;
    }
    check_set(live_regions, cap)
}

impl RegionMap {
    /// Experts whose fitted set is collapsed under `cap` (`max_regions_per_set`), by index.
    pub fn collapsed(&self, cap: usize) -> Vec<(u32, Collapse)> {
        self.experts.iter().filter_map(|e| check_set(e.regions.len(), cap).map(|c| (e.expert_index, c))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExpertRegionMap, FittedRegion};

    fn map(sizes: &[usize]) -> RegionMap {
        RegionMap {
            snapshot_id: "s".into(),
            parent_snapshot_id: None,
            probe_set_version: "p".into(),
            n_probes: 10,
            carry_threshold: 0.5,
            experts: sizes
                .iter()
                .enumerate()
                .map(|(k, n)| ExpertRegionMap {
                    expert_index: k as u32,
                    regions: (0..*n)
                        .map(|i| FittedRegion {
                            region_id: format!("s-e{k}-r{i:03}"),
                            lineage_id: format!("s-e{k}-r{i:03}"),
                            expert_index: k as u32,
                            centroid: vec![i as f32],
                            radius: 1.0,
                            mass: 1.0 / *n as f64,
                            parent_region_id: None,
                            score: 0.0,
                            ancestors: vec![],
                        })
                        .collect(),
                    parent_regions: 0,
                    matched: 0,
                    carried_mass: 0.0,
                    dead_parents: vec![],
                    split_parents: vec![],
                    regime_break: false,
                })
                .collect(),
            note: None,
        }
    }

    #[test]
    fn single_region_and_cap_are_invalid() {
        assert_eq!(check_set(0, 64), Some(Collapse::Single));
        assert_eq!(check_set(1, 64), Some(Collapse::Single));
        assert_eq!(check_set(2, 64), None);
        assert_eq!(check_set(63, 64), None);
        assert_eq!(check_set(64, 64), Some(Collapse::AtCap));
        assert_eq!(check_set(500, 0), None, "cap 0 disables the cap check");
        assert_eq!(check_online(1, 10, 2000, 64), None, "warm-up");
        assert_eq!(check_online(1, 2000, 2000, 64), Some(Collapse::Single));
        assert_eq!(map(&[5, 1, 64, 3]).collapsed(64), vec![(1, Collapse::Single), (2, Collapse::AtCap)]);
        assert!(map(&[5, 3]).collapsed(64).is_empty());
    }
}
