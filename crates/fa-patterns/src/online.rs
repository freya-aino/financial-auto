//! Online region discovery: streaming leader clustering (spec 30 §6.2/§6.3).
//!
//! This is the pure-Rust baseline `RegionDiscovery` for continuous latents while the
//! density/Mapper methods of the neural families are unavailable (D-004). Every point is
//! assigned to the nearest live region if it lies within `radius`; otherwise a new region
//! is born. Centroids track a running mean with a floor on the learning rate so regions
//! keep adapting. Two regions whose centroids drift within `merge_radius` are merged
//! (the lighter one is retired). Region ids are deterministic in the order of creation and
//! the lineage id is the id at birth (D-007: a single snapshot never re-fits, so there is
//! nothing to align yet).

use serde::{Deserialize, Serialize};

use crate::{DiscoveryMethod, LineageId, Region, RegionDiscovery, RegionId, nearest};

/// Tunables of the online leader clustering (spec 30 §7, scaled to the latent's units).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OnlineRegionCfg {
    /// A point farther than this from every live centroid opens a new region.
    pub radius: f32,
    /// Two live centroids closer than this are merged.
    pub merge_radius: f32,
    /// Hard cap (`max_regions_per_set`); beyond it points join the nearest region.
    pub max_regions: usize,
    /// Floor of the centroid update rate, so old regions keep tracking drift.
    pub min_learning_rate: f32,
    /// `distance / radius` at or above which a point is flagged out-of-region.
    pub novelty_threshold: f32,
}

impl Default for OnlineRegionCfg {
    fn default() -> Self {
        Self { radius: 1.0, merge_radius: 0.35, max_regions: 64, min_learning_rate: 0.02, novelty_threshold: 3.0 }
    }
}

/// Lifecycle change produced by one observation (spec 30 §6.4 vocabulary).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegionLifecycle {
    Created {
        region: Region,
    },
    /// `survivor` absorbed `absorbed`; the absorbed region is retired.
    Merged {
        survivor: RegionId,
        lineage_id: LineageId,
        absorbed: RegionId,
    },
    Retired {
        region_id: RegionId,
        lineage_id: LineageId,
        merged_into: RegionId,
    },
}

/// Hard assignment of one point plus the lifecycle events it caused.
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub region_id: RegionId,
    pub lineage_id: LineageId,
    /// Euclidean distance to the assigned centroid before the update.
    pub distance: f32,
    /// `distance / radius`; ≥ `novelty_threshold` means out-of-region.
    pub novelty: f32,
    pub events: Vec<RegionLifecycle>,
}

impl Assignment {
    pub fn is_novel(&self, cfg: &OnlineRegionCfg) -> bool {
        self.novelty >= cfg.novelty_threshold
    }
}

/// Stage boundary for online membership: assigns live latents to regions and reports
/// lifecycle changes. Implemented by [`OnlineRegionSet`]; neural families plug in here.
pub trait OnlineRegionDiscovery: Send + Sync {
    fn method(&self) -> DiscoveryMethod;
    fn observe(&mut self, point: &[f32]) -> Assignment;
    /// Live regions.
    fn regions(&self) -> Vec<&Region>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Slot {
    region: Region,
    /// Number of points assigned so far (soft mass; decays nothing).
    mass: f64,
    alive: bool,
}

/// One region set for one (snapshot, timescale, expert) triple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineRegionSet {
    cfg: OnlineRegionCfg,
    snapshot_id: String,
    expert_index: u32,
    /// Prefix of every region id, e.g. `1m-e0`.
    id_prefix: String,
    slots: Vec<Slot>,
    next_ordinal: u32,
}

impl OnlineRegionSet {
    pub fn new(
        cfg: OnlineRegionCfg,
        snapshot_id: impl Into<String>,
        expert_index: u32,
        id_prefix: impl Into<String>,
    ) -> Self {
        Self {
            cfg,
            snapshot_id: snapshot_id.into(),
            expert_index,
            id_prefix: id_prefix.into(),
            slots: Vec::new(),
            next_ordinal: 0,
        }
    }

    /// A set pre-populated with the regions of a trained snapshot (`region_map.json`,
    /// [`crate::FittedRegion`]): ids, lineages and centroids are taken as given, so live
    /// memberships continue the snapshot's lineages (D-007). `mass` seeds the centroid's inertia
    /// (at least one point). New regions born later continue the ordinal sequence.
    pub fn seeded(
        cfg: OnlineRegionCfg,
        snapshot_id: impl Into<String>,
        expert_index: u32,
        id_prefix: impl Into<String>,
        seeds: &[crate::FittedRegion],
    ) -> Self {
        let mut set = Self::new(cfg, snapshot_id, expert_index, id_prefix);
        for s in seeds {
            let region = Region {
                id: s.region_id.clone(),
                lineage_id: s.lineage_id.clone(),
                snapshot_id: set.snapshot_id.clone(),
                expert_index,
                centroid: s.centroid.clone(),
                radius: cfg.radius,
                method: DiscoveryMethod::Leader,
            };
            set.slots.push(Slot { region, mass: s.mass.max(1.0), alive: true });
            // ids are `<prefix>-r<nnn>`: keep the next ordinal above every seeded one
            if let Some(n) = s.region_id.rsplit("-r").next().and_then(|n| n.parse::<u32>().ok()) {
                set.next_ordinal = set.next_ordinal.max(n + 1);
            }
        }
        set.next_ordinal = set.next_ordinal.max(seeds.len() as u32);
        set
    }

    pub fn cfg(&self) -> &OnlineRegionCfg {
        &self.cfg
    }

    pub fn live_count(&self) -> usize {
        self.slots.iter().filter(|s| s.alive).count()
    }

    /// Mass (points assigned) of a live region.
    pub fn mass(&self, region_id: &str) -> Option<f64> {
        self.slots.iter().find(|s| s.alive && s.region.id == region_id).map(|s| s.mass)
    }

    fn live_indices(&self) -> Vec<usize> {
        self.slots.iter().enumerate().filter(|(_, s)| s.alive).map(|(i, _)| i).collect()
    }

    fn create(&mut self, point: &[f32]) -> usize {
        let id = format!("{}-r{:03}", self.id_prefix, self.next_ordinal);
        self.next_ordinal += 1;
        let region = Region {
            id: id.clone(),
            lineage_id: id,
            snapshot_id: self.snapshot_id.clone(),
            expert_index: self.expert_index,
            centroid: point.to_vec(),
            radius: self.cfg.radius,
            method: DiscoveryMethod::Leader,
        };
        self.slots.push(Slot { region, mass: 1.0, alive: true });
        self.slots.len() - 1
    }

    /// Merges live regions within `merge_radius` of slot `i` into the heavier of the pair.
    /// Returns the index of the slot that now holds the point.
    fn merge_around(&mut self, mut i: usize, events: &mut Vec<RegionLifecycle>) -> usize {
        loop {
            let partner = self.live_indices().into_iter().filter(|&j| j != i).find(|&j| {
                distance(&self.slots[i].region.centroid, &self.slots[j].region.centroid) < self.cfg.merge_radius
            });
            let Some(j) = partner else { return i };
            let (survivor, absorbed) = if self.slots[j].mass > self.slots[i].mass { (j, i) } else { (i, j) };
            let (ms, ma) = (self.slots[survivor].mass, self.slots[absorbed].mass);
            let total = ms + ma;
            let absorbed_centroid = self.slots[absorbed].region.centroid.clone();
            for (c, a) in self.slots[survivor].region.centroid.iter_mut().zip(&absorbed_centroid) {
                *c = ((f64::from(*c) * ms + f64::from(*a) * ma) / total) as f32;
            }
            self.slots[survivor].mass = total;
            self.slots[absorbed].alive = false;
            let (s, a) = (self.slots[survivor].region.clone(), self.slots[absorbed].region.clone());
            events.push(RegionLifecycle::Merged {
                survivor: s.id.clone(),
                lineage_id: s.lineage_id.clone(),
                absorbed: a.id.clone(),
            });
            events.push(RegionLifecycle::Retired { region_id: a.id, lineage_id: a.lineage_id, merged_into: s.id });
            i = survivor;
        }
    }
}

impl OnlineRegionDiscovery for OnlineRegionSet {
    fn method(&self) -> DiscoveryMethod {
        DiscoveryMethod::Leader
    }

    fn observe(&mut self, point: &[f32]) -> Assignment {
        let mut events = Vec::new();
        let live = self.live_indices();
        let nearest_live = live
            .iter()
            .map(|&i| (i, distance(&self.slots[i].region.centroid, point)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        let (idx, distance) = match nearest_live {
            Some((i, d)) if d <= self.cfg.radius || live.len() >= self.cfg.max_regions => {
                let slot = &mut self.slots[i];
                slot.mass += 1.0;
                let lr = (1.0 / slot.mass as f32).max(self.cfg.min_learning_rate);
                for (c, p) in slot.region.centroid.iter_mut().zip(point) {
                    *c += lr * (p - *c);
                }
                (self.merge_around(i, &mut events), d)
            }
            Some((_, d)) => {
                let i = self.create(point);
                events.push(RegionLifecycle::Created { region: self.slots[i].region.clone() });
                (i, d)
            }
            None => {
                let i = self.create(point);
                events.push(RegionLifecycle::Created { region: self.slots[i].region.clone() });
                (i, 0.0)
            }
        };
        let r = &self.slots[idx].region;
        Assignment {
            region_id: r.id.clone(),
            lineage_id: r.lineage_id.clone(),
            distance,
            novelty: if self.cfg.radius > 0.0 { distance / self.cfg.radius } else { 0.0 },
            events,
        }
    }

    fn regions(&self) -> Vec<&Region> {
        self.slots.iter().filter(|s| s.alive).map(|s| &s.region).collect()
    }
}

/// Batch leader clustering (one deterministic pass in input order) behind the
/// [`RegionDiscovery`] fit/assign contract.
#[derive(Debug, Clone, Copy)]
pub struct LeaderClustering {
    pub radius: f32,
    pub max_regions: usize,
}

impl RegionDiscovery for LeaderClustering {
    fn method(&self) -> DiscoveryMethod {
        DiscoveryMethod::Leader
    }

    fn discover(&self, points: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let cfg = OnlineRegionCfg {
            radius: self.radius,
            merge_radius: 0.0,
            max_regions: self.max_regions,
            ..Default::default()
        };
        let mut set = OnlineRegionSet::new(cfg, "fit", 0, "fit");
        for p in points {
            set.observe(p);
        }
        set.regions().into_iter().map(|r| r.centroid.clone()).collect()
    }

    fn assign(&self, centroids: &[Vec<f32>], point: &[f32]) -> Option<usize> {
        nearest(centroids, point).filter(|&i| distance(&centroids[i], point) <= self.radius)
    }
}

pub fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use fa_core::rng::Rng;

    use super::*;

    /// Synthetic test data: a Gaussian mixture with 4 well-separated components in R^3.
    fn mixture(n_per: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<usize>) {
        let centers = [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]];
        let mut rng = Rng::seed(seed);
        let mut pts = Vec::new();
        let mut labels = Vec::new();
        for i in 0..n_per {
            let k = i % centers.len();
            pts.push(centers[k].iter().map(|c| (*c + 0.3 * rng.normal()) as f32).collect());
            labels.push(k);
        }
        (pts, labels)
    }

    #[test]
    fn leader_clustering_recovers_mixture_components() {
        let (pts, labels) = mixture(400, 1);
        let d = LeaderClustering { radius: 2.0, max_regions: 64 };
        let centroids = d.discover(&pts);
        assert_eq!(centroids.len(), 4);
        // Purity: every true component maps to exactly one discovered centroid.
        let mut seen = std::collections::HashMap::new();
        let ok = pts
            .iter()
            .zip(&labels)
            .filter(|(p, l)| *seen.entry(**l).or_insert_with(|| d.assign(&centroids, p)) == d.assign(&centroids, p))
            .count();
        assert!(ok as f64 / pts.len() as f64 >= 0.9);
        assert_eq!(d.assign(&centroids, &[50.0, 50.0, 50.0]), None);
    }

    #[test]
    fn online_set_births_merges_and_ids_are_deterministic() {
        let cfg = OnlineRegionCfg { radius: 1.0, merge_radius: 0.7, max_regions: 8, ..Default::default() };
        let mut set = OnlineRegionSet::new(cfg, "snap", 2, "1m-e2");
        let a = set.observe(&[0.0, 0.0]);
        assert_eq!(a.region_id, "1m-e2-r000");
        assert_eq!(a.lineage_id, "1m-e2-r000");
        assert!(matches!(a.events[0], RegionLifecycle::Created { .. }));
        let b = set.observe(&[1.6, 0.0]);
        assert_eq!(b.region_id, "1m-e2-r001");
        assert!((b.novelty - 1.6).abs() < 1e-6);
        assert_eq!(set.live_count(), 2);
        // Pull r001 towards r000: first anchor it at 1.1, then feed points at 0.6 (still
        // nearer to r001 than to r000) until its centroid drifts inside `merge_radius`.
        let mut merged = false;
        for i in 0..140 {
            let x = if i < 20 { 1.1 } else { 0.6 };
            let r = set.observe(&[x, 0.0]);
            merged |= r.events.iter().any(|e| matches!(e, RegionLifecycle::Merged { .. }));
        }
        assert!(merged);
        assert_eq!(set.live_count(), 1);
        assert_eq!(set.regions()[0].expert_index, 2);
        assert_eq!(set.regions()[0].snapshot_id, "snap");
    }

    #[test]
    fn seeded_set_keeps_ids_and_lineages_and_continues_ordinals() {
        let seeds = vec![
            crate::FittedRegion {
                region_id: "snap-e0-r000".into(),
                lineage_id: "1m-e0-r004".into(),
                expert_index: 0,
                centroid: vec![0.0, 0.0],
                radius: 1.0,
                mass: 0.6,
                parent_region_id: Some("1m-e0-r004".into()),
                score: 0.8,
            },
            crate::FittedRegion {
                region_id: "snap-e0-r001".into(),
                lineage_id: "snap-e0-r001".into(),
                expert_index: 0,
                centroid: vec![5.0, 0.0],
                radius: 1.0,
                mass: 0.4,
                parent_region_id: None,
                score: 0.0,
            },
        ];
        let cfg = OnlineRegionCfg { radius: 1.0, merge_radius: 0.3, max_regions: 8, ..Default::default() };
        let mut set = OnlineRegionSet::seeded(cfg, "snap", 0, "snap-e0", &seeds);
        assert_eq!(set.live_count(), 2);
        let a = set.observe(&[0.2, 0.0]);
        assert_eq!(a.region_id, "snap-e0-r000");
        assert_eq!(a.lineage_id, "1m-e0-r004", "inherited lineage");
        assert!(a.events.is_empty());
        let b = set.observe(&[10.0, 0.0]);
        assert_eq!(b.region_id, "snap-e0-r002", "new regions continue after the seeds");
        assert_eq!(b.lineage_id, "snap-e0-r002");
    }

    #[test]
    fn cap_forces_nearest_assignment() {
        let cfg = OnlineRegionCfg { radius: 0.1, merge_radius: 0.0, max_regions: 2, ..Default::default() };
        let mut set = OnlineRegionSet::new(cfg, "s", 0, "p");
        set.observe(&[0.0]);
        set.observe(&[10.0]);
        let r = set.observe(&[9.0]);
        assert_eq!(r.region_id, "p-r001");
        assert!(r.is_novel(set.cfg()));
        assert_eq!(set.live_count(), 2);
    }
}
