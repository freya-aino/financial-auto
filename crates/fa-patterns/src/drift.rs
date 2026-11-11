//! Region occupancy drift monitor (spec 30 §6.6).
//!
//! Online adapters shift latents slightly while memberships stay computed against the
//! snapshot's region set. The monitor compares the live occupancy distribution over regions
//! with a reference (the snapshot's fitted region masses, or the live occupancy captured once
//! after warm-up when the snapshot has no fitted regions) by `KL(live ‖ reference)` in nats,
//! with additive smoothing so regions missing on either side stay finite. Above
//! `drift_refit_kl` a partial refit is due.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Additive smoothing mass per region.
const EPS: f64 = 1e-4;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftMonitor {
    /// `drift_refit_kl`.
    pub threshold: f64,
    reference: Option<BTreeMap<String, f64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DriftReport {
    /// `KL(live ‖ reference)` in nats.
    pub kl: f64,
    pub refit_due: bool,
}

impl DriftMonitor {
    /// A monitor without reference: the first [`Self::capture`] sets it.
    pub fn new(threshold: f64) -> Self {
        Self { threshold, reference: None }
    }

    pub fn with_reference(threshold: f64, reference: impl IntoIterator<Item = (String, f64)>) -> Self {
        let mut m = Self::new(threshold);
        m.capture(reference);
        m
    }

    pub fn has_reference(&self) -> bool {
        self.reference.is_some()
    }

    /// Sets the reference distribution (normalised here).
    pub fn capture(&mut self, reference: impl IntoIterator<Item = (String, f64)>) {
        let r: BTreeMap<String, f64> = reference.into_iter().filter(|(_, v)| *v > 0.0).collect();
        if !r.is_empty() {
            self.reference = Some(r);
        }
    }

    /// Drift of the live occupancy against the reference; `None` before a reference exists.
    pub fn check(&self, live: &[(String, f64)]) -> Option<DriftReport> {
        let reference = self.reference.as_ref()?;
        let mut keys: Vec<&str> = reference.keys().map(String::as_str).collect();
        keys.extend(live.iter().map(|(k, _)| k.as_str()));
        keys.sort_unstable();
        keys.dedup();
        let live_map: BTreeMap<&str, f64> = live.iter().map(|(k, v)| (k.as_str(), v.max(0.0))).collect();
        let norm = |get: &dyn Fn(&str) -> f64| -> Vec<f64> {
            let raw: Vec<f64> = keys.iter().map(|k| get(k) + EPS).collect();
            let s: f64 = raw.iter().sum();
            raw.into_iter().map(|v| v / s).collect()
        };
        let p = norm(&|k| live_map.get(k).copied().unwrap_or(0.0));
        let q = norm(&|k| reference.get(k).copied().unwrap_or(0.0));
        let kl: f64 = p.iter().zip(&q).map(|(p, q)| p * (p / q).ln()).sum::<f64>().max(0.0);
        Some(DriftReport { kl, refit_due: kl > self.threshold })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_distribution_has_no_drift_and_a_shift_triggers_refit() {
        let m = DriftMonitor::with_reference(0.1, [("a".to_owned(), 0.5), ("b".to_owned(), 0.5)]);
        let r = m.check(&[("a".into(), 0.5), ("b".into(), 0.5)]).unwrap();
        assert!(r.kl < 1e-9 && !r.refit_due);
        let r = m.check(&[("a".into(), 0.9), ("b".into(), 0.1)]).unwrap();
        assert!(r.kl > 0.3 && r.refit_due, "{r:?}");
        // a brand-new region holding most of the mass is drift as well
        let r = m.check(&[("c".into(), 0.8), ("a".into(), 0.2)]).unwrap();
        assert!(r.refit_due);
        assert!(DriftMonitor::new(0.1).check(&[("a".into(), 1.0)]).is_none());
    }
}
