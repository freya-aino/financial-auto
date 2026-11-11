//! Online region discovery: streaming leader clustering (spec 30 §6.2/§6.3).
//!
//! This is the pure-Rust baseline `RegionDiscovery` for continuous latents while the
//! density/Mapper methods of the neural families are unavailable (D-004). Every point is
//! assigned to the nearest live region if it lies within `radius`; otherwise a new region
//! is born. Centroids track a running mean with a floor on the learning rate so regions
//! keep adapting. Region lifecycle (spec 30 §6.4 vocabulary, applied online):
//!
//! - **birth**: a point outside every radius (below the `max_regions` cap);
//! - **merge**: two centroids drift within `merge_radius` (the lighter one is retired);
//! - **split**: every region runs a streaming 2-means on its own points; when the two
//!   sub-centroids drift `split_distance · radius` apart and both hold `min_split_mass` points,
//!   the lighter mode becomes a new region (new lineage, `DESCENDS_FROM{kind: split}`);
//! - **death by support**: occupancy is an exponentially decayed count (`support_halflife`
//!   observations); a region older than `min_age` observations whose occupancy share falls
//!   below `min_support` (`min_mass`) is retired.
//!
//! Region ids are deterministic in the order of creation and the lineage id is the id at birth
//! (or inherited from a trained snapshot's `region_map`, [`OnlineRegionSet::seeded`]). Soft
//! membership (spec 30 §4.1 `Membership`) weights the `k` nearest regions by a Gaussian kernel of
//! width `radius`; novelty is `distance / radius` to the nearest centroid.

use serde::{Deserialize, Serialize};

use crate::{DiscoveryMethod, LineageId, Region, RegionDiscovery, RegionId, nearest};

/// Tunables of the online leader clustering (spec 30 §7, scaled to the latent's units).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
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
    /// Minimum occupancy share of a live region (`min_mass`); 0 disables death by support.
    pub min_support: f64,
    /// Half-life of the occupancy count, in observations of this set.
    pub support_halflife: f64,
    /// A region is only retired for low support after this many observations of the set since its birth.
    pub min_age: u64,
    /// Sub-centroid separation (in radii) at which a region splits; 0 disables online splits.
    pub split_distance: f32,
    /// Points each sub-mode must hold before a split.
    pub min_split_mass: f64,
    /// Regions in a soft membership (`k` nearest).
    pub membership_k: usize,
}

impl Default for OnlineRegionCfg {
    fn default() -> Self {
        Self {
            radius: 1.0,
            merge_radius: 0.35,
            max_regions: 64,
            min_learning_rate: 0.02,
            novelty_threshold: 3.0,
            min_support: 0.002,
            support_halflife: 20_000.0,
            min_age: 2_000,
            split_distance: 1.5,
            min_split_mass: 30.0,
            membership_k: 4,
        }
    }
}

/// Why a region was retired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetireReason {
    /// Absorbed by a merge (`merged_into` is the survivor).
    Merged,
    /// Occupancy share below `min_support` (no successor).
    Support,
}

impl RetireReason {
    pub fn as_str(self) -> &'static str {
        match self {
            RetireReason::Merged => "merged",
            RetireReason::Support => "support",
        }
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
    /// `parent` split off `child` (a new region and lineage descending from the parent).
    Split {
        parent: RegionId,
        parent_lineage_id: LineageId,
        child: Region,
        /// Points of each sub-mode at the split `(parent, child)`.
        masses: (f64, f64),
    },
    Retired {
        region_id: RegionId,
        lineage_id: LineageId,
        /// Survivor of a merge; empty for death by support.
        merged_into: RegionId,
        reason: RetireReason,
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
    /// Soft membership over the `membership_k` nearest regions (weights sum to 1), computed
    /// against the centroids before the update. Empty for the first point of a set.
    pub memberships: Vec<(RegionId, f32)>,
    pub events: Vec<RegionLifecycle>,
}

impl Assignment {
    pub fn is_novel(&self, cfg: &OnlineRegionCfg) -> bool {
        self.novelty >= cfg.novelty_threshold
    }
}

/// Descriptive statistics of a live region (spec 30 §4.1 `Region`, §5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionStat {
    pub region_id: RegionId,
    pub lineage_id: LineageId,
    pub centroid: Vec<f32>,
    pub radius: f32,
    /// Points assigned since birth (merges add up).
    pub mass: f64,
    /// Decayed occupancy share among live regions (sums to 1 over the set).
    pub occupancy: f64,
    /// RMS distance of assigned points to the centroid (density proxy: smaller is denser).
    pub spread: f64,
    /// Observations of the set since the region's birth.
    pub age: u64,
}

/// Stage boundary for online membership: assigns live latents to regions and reports
/// lifecycle changes. Implemented by [`OnlineRegionSet`]; neural families plug in here.
pub trait OnlineRegionDiscovery: Send + Sync {
    fn method(&self) -> DiscoveryMethod;
    fn observe(&mut self, point: &[f32]) -> Assignment;
    /// Live regions.
    fn regions(&self) -> Vec<&Region>;
    /// Statistics of every live region (empty when the family does not track them).
    fn stats(&self) -> Vec<RegionStat> {
        Vec::new()
    }
    /// Serialisable state, so a restarted runtime continues the same regions (spec 30 §8:
    /// regions are never only in memory). `None` when the family cannot export.
    fn export_state(&self) -> Option<serde_json::Value> {
        None
    }
    /// Restores a state produced by [`Self::export_state`]; `false` when it does not fit.
    fn restore_state(&mut self, _state: &serde_json::Value) -> bool {
        false
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SubMode {
    centroid: Vec<f32>,
    mass: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Slot {
    region: Region,
    /// Number of points assigned so far (merges add up; never decays).
    mass: f64,
    alive: bool,
    /// Exponentially decayed count of assignments (support).
    #[serde(default)]
    occupancy: f64,
    /// Set observation counter at birth.
    #[serde(default)]
    born_step: u64,
    /// Running mean of squared distances of assigned points.
    #[serde(default)]
    sq_dist: f64,
    /// Streaming 2-means over the region's points (split detection).
    #[serde(default)]
    sub: Vec<SubMode>,
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
    /// Observations of this set so far.
    #[serde(default)]
    step: u64,
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
            step: 0,
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
        let n = seeds.len().max(1) as f64;
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
            // Seeded regions start with the occupancy their probe mass implies, so a region the
            // live data never visits still has to wait `min_age` and decay before it dies.
            let occupancy = s.mass.max(1.0 / n) * set.horizon();
            set.slots.push(Slot {
                region,
                mass: s.mass.max(1.0),
                alive: true,
                occupancy,
                born_step: 0,
                sq_dist: 0.0,
                sub: Vec::new(),
            });
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

    /// Soft membership of `point` without updating anything (spec 30 §4.1 `assign`).
    pub fn membership(&self, point: &[f32]) -> (Vec<(RegionId, f32)>, f32) {
        let live = self.live_indices();
        let mut d: Vec<(usize, f32)> =
            live.iter().map(|&i| (i, distance(&self.slots[i].region.centroid, point))).collect();
        d.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        let novelty = d.first().map_or(f32::INFINITY, |x| x.1 / self.cfg.radius.max(f32::MIN_POSITIVE));
        (self.soft_weights(&d), novelty)
    }

    /// Steady-state total occupancy `1 / (1 − λ)` for the configured half-life.
    fn horizon(&self) -> f64 {
        1.0 / (1.0 - self.decay())
    }

    fn decay(&self) -> f64 {
        if self.cfg.support_halflife > 0.0 { 0.5f64.powf(1.0 / self.cfg.support_halflife) } else { 0.0 }
    }

    fn soft_weights(&self, sorted: &[(usize, f32)]) -> Vec<(RegionId, f32)> {
        let k = self.cfg.membership_k.max(1);
        let r2 = 2.0 * f64::from(self.cfg.radius).powi(2).max(1e-12);
        let top = &sorted[..sorted.len().min(k)];
        let Some(d0) = top.first().map(|x| f64::from(x.1)) else { return Vec::new() };
        // Shift by the nearest distance so far-away points do not underflow to all zeros.
        let w: Vec<f64> = top.iter().map(|(_, d)| (-(f64::from(*d).powi(2) - d0 * d0) / r2).exp()).collect();
        let total: f64 = w.iter().sum();
        top.iter().zip(w).map(|((i, _), w)| (self.slots[*i].region.id.clone(), (w / total) as f32)).collect()
    }

    fn live_indices(&self) -> Vec<usize> {
        self.slots.iter().enumerate().filter(|(_, s)| s.alive).map(|(i, _)| i).collect()
    }

    fn new_region(&mut self, point: &[f32]) -> Region {
        let id = format!("{}-r{:03}", self.id_prefix, self.next_ordinal);
        self.next_ordinal += 1;
        Region {
            id: id.clone(),
            lineage_id: id,
            snapshot_id: self.snapshot_id.clone(),
            expert_index: self.expert_index,
            centroid: point.to_vec(),
            radius: self.cfg.radius,
            method: DiscoveryMethod::Leader,
        }
    }

    fn create(&mut self, point: &[f32]) -> usize {
        let region = self.new_region(point);
        self.slots.push(Slot {
            region,
            mass: 1.0,
            alive: true,
            occupancy: 1.0,
            born_step: self.step,
            sq_dist: 0.0,
            sub: Vec::new(),
        });
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
            self.slots[survivor].occupancy += self.slots[absorbed].occupancy;
            self.slots[survivor].sq_dist =
                (self.slots[survivor].sq_dist * ms + self.slots[absorbed].sq_dist * ma) / total;
            self.slots[survivor].sub.clear();
            self.slots[absorbed].alive = false;
            let (s, a) = (self.slots[survivor].region.clone(), self.slots[absorbed].region.clone());
            events.push(RegionLifecycle::Merged {
                survivor: s.id.clone(),
                lineage_id: s.lineage_id.clone(),
                absorbed: a.id.clone(),
            });
            events.push(RegionLifecycle::Retired {
                region_id: a.id,
                lineage_id: a.lineage_id,
                merged_into: s.id,
                reason: RetireReason::Merged,
            });
            i = survivor;
        }
    }

    /// Streaming 2-means inside slot `i`; splits off the lighter mode when the modes separate.
    /// Returns the slot that holds `point` afterwards.
    fn track_modes(&mut self, i: usize, point: &[f32], events: &mut Vec<RegionLifecycle>) -> usize {
        if self.cfg.split_distance <= 0.0 || self.live_count() >= self.cfg.max_regions {
            return i;
        }
        let lr_floor = self.cfg.min_learning_rate;
        let radius = self.cfg.radius;
        let slot = &mut self.slots[i];
        match slot.sub.len() {
            0 => slot.sub.push(SubMode { centroid: point.to_vec(), mass: 1.0 }),
            1 => {
                if distance(&slot.sub[0].centroid, point) > 0.25 * radius {
                    slot.sub.push(SubMode { centroid: point.to_vec(), mass: 1.0 });
                } else {
                    let m = &mut slot.sub[0];
                    m.mass += 1.0;
                    let lr = (1.0 / m.mass as f32).max(lr_floor);
                    for (c, p) in m.centroid.iter_mut().zip(point) {
                        *c += lr * (p - *c);
                    }
                }
            }
            _ => {
                let k = usize::from(distance(&slot.sub[1].centroid, point) < distance(&slot.sub[0].centroid, point));
                let m = &mut slot.sub[k];
                m.mass += 1.0;
                let lr = (1.0 / m.mass as f32).max(lr_floor);
                for (c, p) in m.centroid.iter_mut().zip(point) {
                    *c += lr * (p - *c);
                }
            }
        }
        if slot.sub.len() < 2 {
            return i;
        }
        let sep = distance(&slot.sub[0].centroid, &slot.sub[1].centroid);
        let (m0, m1) = (slot.sub[0].mass, slot.sub[1].mass);
        if sep < self.cfg.split_distance * radius || m0.min(m1) < self.cfg.min_split_mass {
            return i;
        }
        let (keep, go) = if m0 >= m1 { (0, 1) } else { (1, 0) };
        let keep_c = slot.sub[keep].centroid.clone();
        let go_c = slot.sub[go].centroid.clone();
        let (mk, mg) = (slot.sub[keep].mass, slot.sub[go].mass);
        let share = mg / (mk + mg);
        slot.region.centroid = keep_c;
        slot.sub.clear();
        let parent = slot.region.clone();
        let (p_mass, p_occ) = (slot.mass, slot.occupancy);
        slot.mass = (p_mass * (1.0 - share)).max(1.0);
        slot.occupancy = p_occ * (1.0 - share);
        let child = self.new_region(&go_c);
        let p_in_child = distance(&go_c, point) < distance(&self.slots[i].region.centroid, point);
        self.slots.push(Slot {
            region: child.clone(),
            mass: (p_mass * share).max(1.0),
            alive: true,
            occupancy: p_occ * share,
            born_step: self.step,
            sq_dist: 0.0,
            sub: Vec::new(),
        });
        events.push(RegionLifecycle::Split {
            parent: parent.id,
            parent_lineage_id: parent.lineage_id,
            child,
            masses: (mk, mg),
        });
        if p_in_child { self.slots.len() - 1 } else { i }
    }

    /// Retires live regions whose occupancy share fell below `min_support` (after `min_age`).
    /// Never retires the last live region or the slot `keep`.
    fn retire_unsupported(&mut self, keep: usize, events: &mut Vec<RegionLifecycle>) {
        if self.cfg.min_support <= 0.0 {
            return;
        }
        let total: f64 = self.slots.iter().filter(|s| s.alive).map(|s| s.occupancy).sum();
        if total <= 0.0 {
            return;
        }
        let step = self.step;
        let mut live = self.live_count();
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if !slot.alive || i == keep || live <= 1 {
                continue;
            }
            if step.saturating_sub(slot.born_step) >= self.cfg.min_age && slot.occupancy / total < self.cfg.min_support
            {
                slot.alive = false;
                live -= 1;
                events.push(RegionLifecycle::Retired {
                    region_id: slot.region.id.clone(),
                    lineage_id: slot.region.lineage_id.clone(),
                    merged_into: String::new(),
                    reason: RetireReason::Support,
                });
            }
        }
    }
}

/// Support sweeps run every this many observations (cheap, but not per point).
const SUPPORT_SWEEP_EVERY: u64 = 64;

impl OnlineRegionDiscovery for OnlineRegionSet {
    fn method(&self) -> DiscoveryMethod {
        DiscoveryMethod::Leader
    }

    fn observe(&mut self, point: &[f32]) -> Assignment {
        let mut events = Vec::new();
        self.step += 1;
        let decay = self.decay();
        for s in self.slots.iter_mut().filter(|s| s.alive) {
            s.occupancy *= decay;
        }
        let live = self.live_indices();
        let mut sorted: Vec<(usize, f32)> =
            live.iter().map(|&i| (i, distance(&self.slots[i].region.centroid, point))).collect();
        sorted.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        let memberships = self.soft_weights(&sorted);
        let nearest_live = sorted.first().copied();
        let (idx, distance) = match nearest_live {
            Some((i, d)) if d <= self.cfg.radius || live.len() >= self.cfg.max_regions => {
                let slot = &mut self.slots[i];
                slot.mass += 1.0;
                slot.occupancy += 1.0;
                slot.sq_dist += (f64::from(d).powi(2) - slot.sq_dist) / slot.mass;
                let lr = (1.0 / slot.mass as f32).max(self.cfg.min_learning_rate);
                for (c, p) in slot.region.centroid.iter_mut().zip(point) {
                    *c += lr * (p - *c);
                }
                let i = self.track_modes(i, point, &mut events);
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
        if self.step.is_multiple_of(SUPPORT_SWEEP_EVERY) {
            self.retire_unsupported(idx, &mut events);
        }
        let r = &self.slots[idx].region;
        Assignment {
            region_id: r.id.clone(),
            lineage_id: r.lineage_id.clone(),
            distance,
            novelty: if self.cfg.radius > 0.0 { distance / self.cfg.radius } else { 0.0 },
            memberships,
            events,
        }
    }

    fn regions(&self) -> Vec<&Region> {
        self.slots.iter().filter(|s| s.alive).map(|s| &s.region).collect()
    }

    fn stats(&self) -> Vec<RegionStat> {
        let total: f64 = self.slots.iter().filter(|s| s.alive).map(|s| s.occupancy).sum();
        self.slots
            .iter()
            .filter(|s| s.alive)
            .map(|s| RegionStat {
                region_id: s.region.id.clone(),
                lineage_id: s.region.lineage_id.clone(),
                centroid: s.region.centroid.clone(),
                radius: s.region.radius,
                mass: s.mass,
                occupancy: if total > 0.0 { s.occupancy / total } else { 0.0 },
                spread: s.sq_dist.sqrt(),
                age: self.step.saturating_sub(s.born_step),
            })
            .collect()
    }

    fn export_state(&self) -> Option<serde_json::Value> {
        serde_json::to_value(self).ok()
    }

    fn restore_state(&mut self, state: &serde_json::Value) -> bool {
        match serde_json::from_value::<OnlineRegionSet>(state.clone()) {
            Ok(mut s) if s.snapshot_id == self.snapshot_id && s.expert_index == self.expert_index => {
                // Keep the configured tunables; the state brings regions, masses and counters.
                s.cfg = self.cfg;
                *self = s;
                true
            }
            _ => false,
        }
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
        // One deterministic pass: no online splits or support deaths in a batch fit.
        let cfg = OnlineRegionCfg {
            radius: self.radius,
            merge_radius: 0.0,
            max_regions: self.max_regions,
            min_support: 0.0,
            split_distance: 0.0,
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

impl LeaderClustering {
    /// Discovers the region set of `points` and assigns every point: inside a region's radius
    /// when possible, else to the nearest centroid (so a fit assigns every probe window).
    /// Deterministic in point order; what the trainer's region fit and its hyper-parameter
    /// search share.
    pub fn fit_assign(&self, points: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<Option<usize>>) {
        let centroids = self.discover(points);
        let assign = points.iter().map(|p| self.assign(&centroids, p).or_else(|| nearest(&centroids, p))).collect();
        (centroids, assign)
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
        let cfg = OnlineRegionCfg {
            radius: 1.0,
            merge_radius: 0.7,
            max_regions: 8,
            split_distance: 0.0,
            ..Default::default()
        };
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
                ancestors: vec![],
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
                ancestors: vec![],
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

    #[test]
    fn fit_assign_covers_every_point_deterministically() {
        let disc = LeaderClustering { radius: 1.0, max_regions: 8 };
        let pts: Vec<Vec<f32>> = (0..40).map(|i| vec![(i % 4) as f32 * 5.0, 0.0]).collect();
        let (c, a) = disc.fit_assign(&pts);
        assert_eq!(c.len(), 4);
        assert!(a.iter().all(Option::is_some));
        assert_eq!((c.clone(), a.clone()), disc.fit_assign(&pts));
        let far = vec![vec![0.0, 0.0], vec![100.0, 0.0]];
        let tight = LeaderClustering { radius: 0.1, max_regions: 1 };
        let (c, a) = tight.fit_assign(&far);
        assert_eq!(c.len(), 1);
        assert_eq!(a, vec![Some(0), Some(0)], "outside every radius → nearest centroid");
    }

    #[test]
    fn diverging_modes_split_with_a_new_lineage() {
        let cfg = OnlineRegionCfg { radius: 1.0, merge_radius: 0.3, min_split_mass: 20.0, ..Default::default() };
        let mut set = OnlineRegionSet::new(cfg, "s", 0, "p");
        let mut rng = fa_core::rng::Rng::seed(4);
        let mut split = None;
        // Two modes start 0.4 apart and drift to 1.9 apart; the region must split once they
        // are 1.5 radii apart, before leader clustering would open a region by distance alone.
        for t in 0..1200 {
            let sep = 0.4 + 1.5 * (t as f32 / 1200.0);
            let side = if t % 2 == 0 { -0.5 } else { 0.5 };
            let p = [side * sep + 0.03 * rng.normal() as f32, 0.03 * rng.normal() as f32];
            let a = set.observe(&p);
            for e in a.events {
                if let RegionLifecycle::Split { parent, child, masses, .. } = e {
                    split.get_or_insert((parent, child.id.clone(), child.lineage_id.clone(), masses, sep));
                }
            }
        }
        let (parent, child, lineage, masses, sep) = split.expect("a split");
        assert_eq!(parent, "p-r000");
        assert_eq!(child, lineage, "the child opens its own lineage");
        assert!(masses.0 >= 20.0 && masses.1 >= 20.0);
        assert!((1.4..1.9).contains(&sep), "split at separation {sep}");
        assert!(set.live_count() >= 2);
    }

    #[test]
    fn unsupported_regions_die_after_min_age() {
        let cfg = OnlineRegionCfg {
            radius: 1.0,
            min_support: 0.01,
            support_halflife: 200.0,
            min_age: 300,
            split_distance: 0.0,
            ..Default::default()
        };
        let mut set = OnlineRegionSet::new(cfg, "s", 0, "p");
        set.observe(&[0.0]);
        let outlier = set.observe(&[10.0]);
        assert_eq!(outlier.region_id, "p-r001");
        let mut died = None;
        for i in 0..2000 {
            let a = set.observe(&[0.01 * (i % 7) as f32]);
            for e in a.events {
                if let RegionLifecycle::Retired { region_id, reason, merged_into, .. } = e {
                    died.get_or_insert((region_id, reason, merged_into, i));
                }
            }
        }
        let (id, reason, into, at) = died.expect("the outlier region dies");
        assert_eq!((id.as_str(), reason, into.as_str()), ("p-r001", RetireReason::Support, ""));
        assert!(at >= 290, "not before min_age: {at}");
        assert_eq!(set.live_count(), 1);
        let st = set.stats();
        assert_eq!(st.len(), 1);
        assert!((st[0].occupancy - 1.0).abs() < 1e-9);
        assert!(st[0].spread < 0.1);
    }

    #[test]
    fn soft_membership_weights_nearest_regions_and_state_round_trips() {
        let cfg = OnlineRegionCfg { radius: 1.0, split_distance: 0.0, ..Default::default() };
        let mut set = OnlineRegionSet::new(cfg, "s", 1, "p");
        set.observe(&[0.0, 0.0]);
        set.observe(&[2.0, 0.0]);
        set.observe(&[0.0, 2.0]);
        let a = set.observe(&[0.9, 0.0]);
        assert_eq!(a.memberships.len(), 3);
        let total: f32 = a.memberships.iter().map(|x| x.1).sum();
        assert!((total - 1.0).abs() < 1e-5);
        assert_eq!(a.memberships[0].0, "p-r000", "nearest first");
        assert!(a.memberships[0].1 > a.memberships[1].1);
        let (m, novelty) = set.membership(&[50.0, 0.0]);
        assert!(novelty > 40.0 && (m.iter().map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-5);

        let state = set.export_state().unwrap();
        let mut back = OnlineRegionSet::new(cfg, "s", 1, "p");
        assert!(back.restore_state(&state));
        assert_eq!(back.stats(), set.stats());
        assert_eq!(back.observe(&[4.0, 4.0]).region_id, "p-r003", "ordinals continue");
        let mut other = OnlineRegionSet::new(cfg, "other-snapshot", 1, "p");
        assert!(!other.restore_state(&state), "a state of another snapshot is refused");
    }
}
