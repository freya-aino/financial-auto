//! Cross-timescale hierarchy edges (spec 30 §6.5, `REFINES` / Helix `Hier`).
//!
//! For a pair of timescales (fine `s`, coarse `s+1`) and one expert index, every closed fine bar
//! of an instrument is paired with that instrument's current coarse region. The tracker keeps
//! the joint counts `n(fine, coarse)` and estimates
//!
//! - `P(coarse | fine)` and `P(fine | coarse)` per pair,
//! - the mutual information `I(fine; coarse)` in bits, bias-corrected (Miller–Madow), and the
//!   pointwise mutual information of each pair.
//!
//! When `I ≥ hier_mi_min`, pairs with at least `min_support` co-occurrences and
//! `P(coarse | fine) ≥ min_cond` become hierarchy edges. Counts are halved when their total
//! exceeds `max_total`, so the estimate follows the current regime.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HierarchyCfg {
    /// Mutual-information cutoff in bits (`hier_mi_min`).
    pub mi_min_bits: f64,
    /// Minimum co-occurrence count of an edge.
    pub min_support: f64,
    /// Minimum `P(coarse | fine)` of an edge.
    pub min_cond: f64,
    /// Halve every count when the total exceeds this (forgetting).
    pub max_total: f64,
}

impl Default for HierarchyCfg {
    fn default() -> Self {
        Self { mi_min_bits: 0.05, min_support: 20.0, min_cond: 0.05, max_total: 200_000.0 }
    }
}

/// One `REFINES` edge: fine region → coarse region.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierEdge {
    pub fine: String,
    pub coarse: String,
    pub p_coarse_given_fine: f64,
    pub p_fine_given_coarse: f64,
    /// Co-occurrence count (after forgetting).
    pub support: f64,
    /// Pointwise mutual information in bits.
    pub pmi_bits: f64,
}

/// Snapshot of the tracker's estimate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchyUpdate {
    /// Bias-corrected mutual information in bits (≥ 0).
    pub mutual_information_bits: f64,
    /// Total co-occurrences behind the estimate.
    pub n: f64,
    /// Edges that pass the cutoffs; empty when the MI is below `mi_min_bits`.
    pub edges: Vec<HierEdge>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HierarchyTracker {
    cfg: HierarchyCfg,
    joint: BTreeMap<(String, String), f64>,
    total: f64,
}

impl HierarchyTracker {
    pub fn new(cfg: HierarchyCfg) -> Self {
        Self { cfg, joint: BTreeMap::new(), total: 0.0 }
    }

    pub fn observe(&mut self, fine: &str, coarse: &str, weight: f64) {
        if weight <= 0.0 || fine.is_empty() || coarse.is_empty() {
            return;
        }
        *self.joint.entry((fine.to_owned(), coarse.to_owned())).or_default() += weight;
        self.total += weight;
        if self.total > self.cfg.max_total {
            for v in self.joint.values_mut() {
                *v *= 0.5;
            }
            self.joint.retain(|_, v| *v >= 1e-3);
            self.total = self.joint.values().sum();
        }
    }

    /// Forgets every pair that involves `region` (retired on either side).
    pub fn forget(&mut self, region: &str) {
        self.joint.retain(|(f, c), _| f != region && c != region);
        self.total = self.joint.values().sum();
    }

    pub fn total(&self) -> f64 {
        self.total
    }

    pub fn update(&self) -> HierarchyUpdate {
        let n = self.total;
        if n <= 0.0 {
            return HierarchyUpdate { mutual_information_bits: 0.0, n: 0.0, edges: Vec::new() };
        }
        let mut pf: BTreeMap<&str, f64> = BTreeMap::new();
        let mut pc: BTreeMap<&str, f64> = BTreeMap::new();
        for ((f, c), v) in &self.joint {
            *pf.entry(f).or_default() += v;
            *pc.entry(c).or_default() += v;
        }
        let ln2 = std::f64::consts::LN_2;
        let mut mi = 0.0;
        for ((f, c), v) in &self.joint {
            if *v > 0.0 {
                mi += v / n * (v * n / (pf[f.as_str()] * pc[c.as_str()])).ln();
            }
        }
        // Miller–Madow bias correction of the plug-in estimate.
        let bias = (pf.len().saturating_sub(1) * pc.len().saturating_sub(1)) as f64 / (2.0 * n);
        let mi_bits = ((mi - bias) / ln2).max(0.0);
        let edges = if mi_bits >= self.cfg.mi_min_bits {
            self.joint
                .iter()
                .filter(|(_, v)| **v >= self.cfg.min_support)
                .map(|((f, c), v)| HierEdge {
                    fine: f.clone(),
                    coarse: c.clone(),
                    p_coarse_given_fine: v / pf[f.as_str()],
                    p_fine_given_coarse: v / pc[c.as_str()],
                    support: *v,
                    pmi_bits: (v * n / (pf[f.as_str()] * pc[c.as_str()])).ln() / ln2,
                })
                .filter(|e| e.p_coarse_given_fine >= self.cfg.min_cond)
                .collect()
        } else {
            Vec::new()
        };
        HierarchyUpdate { mutual_information_bits: mi_bits, n, edges }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_regimes_give_edges_and_independence_gives_none() {
        // Fine regions a1, a2 live inside coarse A; b1 inside B.
        let mut h = HierarchyTracker::new(HierarchyCfg::default());
        for _ in 0..100 {
            h.observe("a1", "A", 1.0);
            h.observe("a2", "A", 1.0);
            h.observe("b1", "B", 1.0);
        }
        h.observe("a1", "B", 1.0); // noise
        let u = h.update();
        assert!(u.mutual_information_bits > 0.8, "{u:?}");
        let e: Vec<(&str, &str)> = u.edges.iter().map(|e| (e.fine.as_str(), e.coarse.as_str())).collect();
        assert_eq!(e, vec![("a1", "A"), ("a2", "A"), ("b1", "B")]);
        assert!((u.edges[0].p_coarse_given_fine - 100.0 / 101.0).abs() < 1e-12);
        assert!(u.edges[2].pmi_bits > 1.0);

        let mut ind = HierarchyTracker::new(HierarchyCfg::default());
        let mut rng = fa_core::rng::Rng::seed(2);
        for _ in 0..4000 {
            let f = ["x", "y", "z"][rng.below(3)];
            let c = ["P", "Q"][rng.below(2)];
            ind.observe(f, c, 1.0);
        }
        let u = ind.update();
        assert!(u.mutual_information_bits < 0.01, "{u:?}");
        assert!(u.edges.is_empty());
        ind.forget("x");
        assert!(ind.total() < 4000.0);
    }

    #[test]
    fn counts_are_halved_past_the_cap() {
        let mut h = HierarchyTracker::new(HierarchyCfg { max_total: 100.0, ..HierarchyCfg::default() });
        for _ in 0..150 {
            h.observe("f", "c", 1.0);
        }
        assert!(h.total() <= 100.0);
    }
}
