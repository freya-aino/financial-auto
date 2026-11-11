//! Predictive coherence of regions (spec 30 §6.1 point 3, open question 2).
//!
//! A region is **predictively coherent** when the next-step residuals of points inside it are
//! distributed differently from those outside it: the two-sample Kolmogorov–Smirnov test
//! ([`crate::ks_pvalue`]) of the region's residuals against the residuals of every other region
//! gives `p < coherence_alpha` (default 0.01).
//!
//! **Decision (open question 2): a soft score, not a mandatory criterion.** Discovery never drops
//! or merges a region for being incoherent outside the FSQ codebook merge (which tests pairs of
//! neighbouring codes, §6.2); instead every region carries its coherence and consumers weight by
//! it. Making it mandatory would tie the region count to the residual sample size (thin regions
//! can never reject) and break lineage continuity whenever a regime's residual law drifts toward
//! the global one. Consumers today: model-runtime's expert residual `r − Σ_k w_k μ_k(region)`
//! uses a region's conditional mean only while the region is coherent (an incoherent region has
//! no predictive view, so it contributes 0), which in turn feeds the cross-asset priors; the
//! share of coherent regions per expert space is a heartbeat gauge.
//!
//! [`CoherenceTracker`] is the online form (rolling windows per region and globally, scores
//! cached and refreshed every `refresh` new observations of a region); [`coherence_of`] is the
//! batch form for a fitted set (labels + residuals of the probe windows).

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::ks_pvalue;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CoherenceCfg {
    /// `coherence_alpha`: a region is coherent when the KS p-value is below it.
    pub alpha: f64,
    /// Residuals a region needs inside it (and outside it) before it is tested.
    pub min_n: usize,
    /// Residuals kept per region.
    pub region_window: usize,
    /// Residuals kept over all regions (the "outside" sample is drawn from it).
    pub global_window: usize,
    /// New observations of a region before its cached score is recomputed.
    pub refresh: u32,
}

impl Default for CoherenceCfg {
    fn default() -> Self {
        Self { alpha: 0.01, min_n: 20, region_window: 256, global_window: 2048, refresh: 16 }
    }
}

/// Coherence score of one region.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Coherence {
    /// KS p-value of inside vs outside residuals.
    pub p_value: f64,
    /// Residuals inside the region.
    pub n: usize,
    /// `p_value < alpha`.
    pub coherent: bool,
}

fn test(inside: &mut [f64], outside: &mut [f64], cfg: &CoherenceCfg) -> Option<Coherence> {
    if inside.len() < cfg.min_n || outside.len() < cfg.min_n {
        return None;
    }
    inside.sort_by(f64::total_cmp);
    outside.sort_by(f64::total_cmp);
    let p = ks_pvalue(inside, outside);
    Some(Coherence { p_value: p, n: inside.len(), coherent: p < cfg.alpha })
}

/// Batch coherence of a fitted set: `labels[i]` the region of probe `i` (`None` = noise),
/// `residuals[i]` its next-step residual (NaN = unknown). `None` for a region without enough
/// residuals inside or outside.
pub fn coherence_of(
    labels: &[Option<usize>],
    residuals: &[f64],
    n_regions: usize,
    cfg: &CoherenceCfg,
) -> Vec<Option<Coherence>> {
    (0..n_regions)
        .map(|r| {
            let (mut inside, mut outside) = (Vec::new(), Vec::new());
            for (l, x) in labels.iter().zip(residuals).filter(|(_, x)| x.is_finite()) {
                match l {
                    Some(k) if *k == r => inside.push(*x),
                    Some(_) => outside.push(*x),
                    None => {}
                }
            }
            test(&mut inside, &mut outside, cfg)
        })
        .collect()
}

/// Online coherence per region of one expert space.
#[derive(Debug, Clone, Default)]
pub struct CoherenceTracker {
    cfg: CoherenceCfg,
    regions: HashMap<String, VecDeque<f64>>,
    /// `(region, residual)` over all regions, newest last.
    global: VecDeque<(String, f64)>,
    /// Cached score and observations since it was computed.
    cache: HashMap<String, (Option<Coherence>, u32)>,
}

impl CoherenceTracker {
    pub fn new(cfg: CoherenceCfg) -> Self {
        Self { cfg, ..Self::default() }
    }

    pub fn cfg(&self) -> &CoherenceCfg {
        &self.cfg
    }

    /// The next-step residual realised after a step in `region`.
    pub fn observe(&mut self, region: &str, residual: f64) {
        if !residual.is_finite() || region.is_empty() {
            return;
        }
        let w = self.regions.entry(region.to_owned()).or_default();
        w.push_back(residual);
        while w.len() > self.cfg.region_window {
            w.pop_front();
        }
        self.global.push_back((region.to_owned(), residual));
        while self.global.len() > self.cfg.global_window {
            self.global.pop_front();
        }
        if let Some(c) = self.cache.get_mut(region) {
            c.1 += 1;
        }
    }

    /// Coherence of `region` (cached; recomputed after `refresh` new observations).
    pub fn score(&mut self, region: &str) -> Option<Coherence> {
        if let Some((c, fresh)) = self.cache.get(region)
            && *fresh < self.cfg.refresh
        {
            return *c;
        }
        let mut inside: Vec<f64> = self.regions.get(region)?.iter().copied().collect();
        let mut outside: Vec<f64> = self.global.iter().filter(|(r, _)| r != region).map(|x| x.1).collect();
        let c = test(&mut inside, &mut outside, &self.cfg);
        self.cache.insert(region.to_owned(), (c, 0));
        c
    }

    /// Is `region` coherent now (`false` while untested)?
    pub fn coherent(&mut self, region: &str) -> bool {
        self.score(region).is_some_and(|c| c.coherent)
    }

    /// Tested regions and how many of them are coherent.
    pub fn summary(&mut self) -> (usize, usize) {
        let ids: Vec<String> = self.regions.keys().cloned().collect();
        let scores: Vec<Coherence> = ids.iter().filter_map(|r| self.score(r)).collect();
        (scores.len(), scores.iter().filter(|c| c.coherent).count())
    }

    /// A retired region leaves the tracker (its residuals stay in the global window until they age out).
    pub fn forget(&mut self, region: &str) {
        self.regions.remove(region);
        self.cache.remove(region);
    }
}

#[cfg(test)]
mod tests {
    use fa_core::rng::Rng;

    use super::*;

    #[test]
    fn a_region_with_its_own_residual_law_is_coherent_and_a_copy_of_the_global_one_is_not() {
        let mut rng = Rng::seed(3);
        let mut t = CoherenceTracker::new(CoherenceCfg::default());
        for _ in 0..400 {
            t.observe("a", rng.normal());
            t.observe("b", rng.normal());
            t.observe("shifted", 1.0 + rng.normal());
        }
        let s = t.score("shifted").unwrap();
        assert!(s.coherent && s.p_value < 1e-6, "{s:?}");
        // `a` is compared with `b` ∪ `shifted`: the shifted third makes it differ too, so test
        // against a pool without it.
        let mut u = CoherenceTracker::new(CoherenceCfg::default());
        for _ in 0..400 {
            u.observe("a", rng.normal());
            u.observe("b", rng.normal());
        }
        assert!(!u.coherent("a"), "{:?}", u.score("a"));
        assert_eq!(u.summary(), (2, 0));
        assert!(t.score("never").is_none());
        u.forget("a");
        assert!(u.score("a").is_none());
    }

    #[test]
    fn batch_form_matches_the_definition() {
        let mut rng = Rng::seed(5);
        let mut labels = Vec::new();
        let mut res = Vec::new();
        for i in 0..600 {
            let k = i % 3;
            labels.push(Some(k));
            res.push(if k == 2 { 2.0 + rng.normal() } else { rng.normal() });
        }
        labels.push(None);
        res.push(100.0);
        let c = coherence_of(&labels, &res, 4, &CoherenceCfg::default());
        assert!(c[2].unwrap().coherent);
        assert!(c[3].is_none(), "empty region is untested");
        assert_eq!(c[0].unwrap().n, 200);
    }
}
