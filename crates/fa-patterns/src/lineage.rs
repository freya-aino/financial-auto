//! Region identity across refits and snapshots (spec 30 §6.4, D-007 fallback).
//!
//! Both region sets assign the same fixed probe windows. The overlap matrix
//! `O[i,j] = |old i ∩ new j| / |old i ∪ new j|` (Jaccard of the probe sets) is solved with the
//! Hungarian algorithm ([`crate::hungarian`]) on the entries at or above `carry_threshold`, so
//! the carried pairs maximise the total overlap (not the greedy highest-first choice). Every
//! remaining pair at or above `split_threshold` becomes a lineage edge (`DESCENDS_FROM`):
//!
//! | situation | kind | lineage of the new region |
//! |---|---|---|
//! | Hungarian pair with `O ≥ carry_threshold` | `carry` | inherits the old lineage |
//! | old region with ≥ 2 new partners (`O ≥ split_threshold`) | `split` | new lineage (unless it is the carried child) |
//! | new region with ≥ 2 old partners | `merge` | new lineage (unless one partner is carried) |
//! | single partner with `split ≤ O < carry` | `partial` | new lineage |
//!
//! Old regions without any partner `≥ split_threshold` **die**; new regions without any partner
//! are **born**. When the carried probe mass is below `regime_break_mass` (0.5) the alignment is
//! a regime break of the representation (spec 30 §8): consumers reset their priors.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::hungarian::max_weight_assignment;

/// Thresholds of the alignment (spec 30 §7).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AlignCfg {
    /// Jaccard overlap for identity carry-over (`carry_threshold`, default 0.5).
    pub carry_threshold: f64,
    /// Overlap to count as split/merge partner (`split_threshold`, default 0.2).
    pub split_threshold: f64,
    /// Carried probe mass below which the alignment is a representation regime break (0.5).
    pub regime_break_mass: f64,
}

pub const DEFAULT_SPLIT_THRESHOLD: f64 = 0.2;
pub const DEFAULT_REGIME_BREAK_MASS: f64 = 0.5;

impl Default for AlignCfg {
    fn default() -> Self {
        Self {
            carry_threshold: 0.5,
            split_threshold: DEFAULT_SPLIT_THRESHOLD,
            regime_break_mass: DEFAULT_REGIME_BREAK_MASS,
        }
    }
}

/// How a new region descends from an old one (`DESCENDS_FROM.kind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DescentKind {
    Carry,
    Split,
    Merge,
    Partial,
}

impl DescentKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DescentKind::Carry => "carry",
            DescentKind::Split => "split",
            DescentKind::Merge => "merge",
            DescentKind::Partial => "partial",
        }
    }
}

/// One lineage edge: new region `new_index` descends from old region `old_index`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Descent {
    pub new_index: usize,
    pub old_index: usize,
    /// Jaccard overlap of the two probe sets.
    pub overlap: f64,
    /// Probe windows in both regions.
    pub shared: usize,
    pub kind: DescentKind,
}

/// Result of [`align`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Alignment {
    /// Old region carried by each new region (index = new region), `None` → new lineage.
    pub carried: Vec<Option<usize>>,
    /// Every lineage edge, sorted by `(new_index, old_index)`.
    pub descents: Vec<Descent>,
    /// Old regions (that were assigned at least one probe) without any descendant.
    pub dead: Vec<usize>,
    /// New regions without any ancestor.
    pub born: Vec<usize>,
    /// Old regions that split (≥ 2 descendants).
    pub splits: Vec<usize>,
    /// New regions that merged ≥ 2 old regions.
    pub merges: Vec<usize>,
    /// Fraction of probe windows whose new region carries a lineage.
    pub carried_mass: f64,
    pub regime_break: bool,
}

impl Alignment {
    pub fn ancestors(&self, new_index: usize) -> impl Iterator<Item = &Descent> {
        self.descents.iter().filter(move |d| d.new_index == new_index)
    }
}

/// Aligns two assignments of the same probe windows (`old_assign[w]`, `new_assign[w]`: region
/// index of window `w` under each set, `None` = outside every region). `n_new` is the number of
/// new regions (some may hold no probe).
pub fn align(old_assign: &[Option<usize>], new_assign: &[Option<usize>], n_new: usize, cfg: &AlignCfg) -> Alignment {
    assert_eq!(old_assign.len(), new_assign.len(), "probe sets must be the same windows");
    let n_old = old_assign.iter().flatten().map(|i| i + 1).max().unwrap_or(0);
    let n_new = n_new.max(new_assign.iter().flatten().map(|i| i + 1).max().unwrap_or(0));
    let mut old_size = vec![0usize; n_old];
    let mut new_size = vec![0usize; n_new];
    let mut inter: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for (o, n) in old_assign.iter().zip(new_assign) {
        if let Some(o) = o {
            old_size[*o] += 1;
        }
        if let Some(n) = n {
            new_size[*n] += 1;
        }
        if let (Some(o), Some(n)) = (o, n) {
            *inter.entry((*o, *n)).or_default() += 1;
        }
    }
    let jaccard = |o: usize, n: usize, shared: usize| shared as f64 / (old_size[o] + new_size[n] - shared) as f64;
    // Hungarian on the above-threshold overlaps (rows = new, cols = old).
    let mut w = vec![vec![0.0; n_old]; n_new];
    for (&(o, n), &s) in &inter {
        let j = jaccard(o, n, s);
        if j >= cfg.carry_threshold {
            w[n][o] = j;
        }
    }
    let carried = max_weight_assignment(&w);

    let mut partners_old = vec![0usize; n_old];
    let mut partners_new = vec![0usize; n_new];
    let mut pairs = Vec::new();
    for (&(o, n), &s) in &inter {
        let j = jaccard(o, n, s);
        if j >= cfg.split_threshold || carried[n] == Some(o) {
            partners_old[o] += 1;
            partners_new[n] += 1;
            pairs.push((n, o, j, s));
        }
    }
    let mut descents: Vec<Descent> = pairs
        .into_iter()
        .map(|(n, o, overlap, shared)| {
            let kind = if carried[n] == Some(o) {
                DescentKind::Carry
            } else if partners_old[o] >= 2 {
                DescentKind::Split
            } else if partners_new[n] >= 2 {
                DescentKind::Merge
            } else {
                DescentKind::Partial
            };
            Descent { new_index: n, old_index: o, overlap, shared, kind }
        })
        .collect();
    descents.sort_by_key(|d| (d.new_index, d.old_index));

    let dead = (0..n_old).filter(|&o| old_size[o] > 0 && partners_old[o] == 0).collect();
    let born = (0..n_new).filter(|&n| partners_new[n] == 0).collect();
    let splits = (0..n_old).filter(|&o| partners_old[o] >= 2).collect();
    let merges = (0..n_new).filter(|&n| partners_new[n] >= 2).collect();
    let total = new_assign.len().max(1) as f64;
    let carried_probes: usize =
        new_assign.iter().flatten().filter(|n| carried.get(**n).is_some_and(Option::is_some)).count();
    let carried_mass = carried_probes as f64 / total;
    let has_parent = old_assign.iter().any(Option::is_some);
    Alignment {
        carried,
        descents,
        dead,
        born,
        splits,
        merges,
        carried_mass,
        regime_break: has_parent && carried_mass < cfg.regime_break_mass,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[i64]) -> Vec<Option<usize>> {
        v.iter().map(|x| if *x < 0 { None } else { Some(*x as usize) }).collect()
    }

    #[test]
    fn carry_split_merge_birth_death() {
        // old 0 (8 probes) splits into new 0 (5) and new 1 (3);
        // old 1 (3) + old 2 (3) merge into new 2 (8); old 3 (2) dies; new 3 (2) is born.
        let old = s(&[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, -1, -1, -1, -1]);
        let new = s(&[0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, -1, -1, 3, 3, 2, 2]);
        let a = align(&old, &new, 4, &AlignCfg::default());
        assert_eq!(a.carried, vec![Some(0), None, None, None], "only new 0 (J = 5/8) carries");
        let kinds: Vec<(usize, usize, DescentKind)> =
            a.descents.iter().map(|d| (d.new_index, d.old_index, d.kind)).collect();
        assert_eq!(
            kinds,
            vec![
                (0, 0, DescentKind::Carry),
                (1, 0, DescentKind::Split),
                (2, 1, DescentKind::Merge),
                (2, 2, DescentKind::Merge),
            ]
        );
        assert_eq!(a.dead, vec![3]);
        assert_eq!(a.born, vec![3]);
        assert_eq!(a.splits, vec![0]);
        assert_eq!(a.merges, vec![2]);
        assert!((a.carried_mass - 5.0 / 20.0).abs() < 1e-12);
        assert!(a.regime_break, "only 25 % of the probe mass kept its identity");
    }

    #[test]
    fn relabelling_keeps_every_identity() {
        let old = s(&[0, 0, 0, 1, 1, 1, 2, 2]);
        let new = s(&[2, 2, 2, 0, 0, 0, 1, 1]);
        let a = align(&old, &new, 3, &AlignCfg::default());
        assert_eq!(a.carried, vec![Some(1), Some(2), Some(0)]);
        assert!(a.dead.is_empty() && a.born.is_empty());
        assert_eq!(a.carried_mass, 1.0);
        assert!(!a.regime_break);
        assert!(a.descents.iter().all(|d| d.kind == DescentKind::Carry && d.overlap == 1.0));
    }

    #[test]
    fn hungarian_carries_distinct_partners() {
        // new 0 overlaps old 0 (J=4/13) and old 1 (J=5/9); new 1 overlaps old 0 (J=.5):
        // the assignment carries both new regions, each from a distinct old region.
        let old = s(&[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]);
        let new = s(&[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1]);
        let a = align(&old, &new, 2, &AlignCfg { carry_threshold: 0.3, ..AlignCfg::default() });
        assert_eq!(a.carried, vec![Some(1), Some(0)], "{a:?}");
    }

    #[test]
    fn no_parent_is_not_a_regime_break() {
        let a = align(&s(&[-1, -1]), &s(&[0, 1]), 2, &AlignCfg::default());
        assert_eq!(a.born, vec![0, 1]);
        assert!(!a.regime_break);
        assert_eq!(a.carried_mass, 0.0);
    }
}
