//! The fit itself (spec 36 §6.2/§6.3, spec 30 §6.4): normalisation statistics → region sets on
//! the probe windows → alignment to the parent → snapshot bundle. Pure with respect to the bus
//! and the stores: [`Fit`] owns the resumable state, the caller drives it in steps so it can
//! checkpoint, yield to a preemption or stop at the budget between steps.
//!
//! Phases
//! 1. `Stats`: [`fa_encoders::StatisticalEncoder::fit_into`] over the window ends, in chunks
//!    ([`CHUNK`] windows per [`Fit::step`]). The Welford accumulator is the checkpoint.
//! 2. `Regions`: the fitted encoder embeds every probe window; `LeaderClustering` finds one
//!    region set per expert (radius `region_radius`, cap `max_regions`), deterministic in window
//!    order; every probe is assigned.
//! 3. `Align`: the parent's assignment of the same probe windows (from its stored `LatentVec`
//!    observations, see [`ParentAssignment`]) is matched with `fa_patterns::align_expert`.
//! 4. `Bundle`: manifest, weights, regions and region map.

use std::collections::{BTreeMap, HashMap};

use fa_core::Timescale;
use fa_encoders::{
    DataRef, EXPERT_NAMES, EncoderFamily, EncoderInput, FitConfig, NormAccumulator, NormStats, ProbeSetRef,
    STATISTICAL_FAMILY, SnapshotManifest, StatisticalEncoder,
};
use fa_patterns::{AlignInput, FittedRegion, LeaderClustering, ParentIds, RegionDiscovery, RegionMap, align_expert};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::data::BarHistory;

/// Windows accumulated per [`Fit::step`].
pub const CHUNK: usize = 64;

/// What a job asks for (`Job.spec` JSON and the CLI flags map onto this).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitSpec {
    pub timescale: Timescale,
    /// History length; the window ends at job start.
    #[serde(default = "default_hours")]
    pub window_hours: f64,
    /// VM instrument token or dashed UUID; default: the instrument with the most points.
    #[serde(default)]
    pub instrument: Option<String>,
    /// Snapshot to align with (and, for `generation`, to derive from); default: the active one.
    #[serde(default)]
    pub parent_snapshot_id: Option<String>,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub fit: FitConfig,
    #[serde(default = "default_radius")]
    pub region_radius: f32,
    #[serde(default = "default_max_regions")]
    pub max_regions: usize,
    /// Jaccard overlap for lineage carry-over (spec 30 §7 `carry_threshold`).
    #[serde(default = "default_carry")]
    pub carry_threshold: f64,
    /// At most this many probe windows (evenly spaced over the fit windows).
    #[serde(default = "default_probes")]
    pub probe_max: usize,
}

fn default_hours() -> f64 {
    6.0
}
fn default_radius() -> f32 {
    1.0
}
fn default_max_regions() -> usize {
    64
}
fn default_carry() -> f64 {
    0.5
}
fn default_probes() -> usize {
    512
}

impl FitSpec {
    pub fn new(timescale: Timescale) -> Self {
        Self {
            timescale,
            window_hours: default_hours(),
            instrument: None,
            parent_snapshot_id: None,
            seed: 0,
            fit: FitConfig::default(),
            region_radius: default_radius(),
            max_regions: default_max_regions(),
            carry_threshold: default_carry(),
            probe_max: default_probes(),
        }
    }
}

/// The parent's region assignment of the probe windows, per expert: `assign[k][w]` is the index
/// into `ids[k]` of the parent region probe `w` fell in (looked up by bar-end timestamp among
/// the parent's `LatentVec` observations), `None` when the parent never observed that window.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ParentAssignment {
    pub snapshot_id: String,
    pub ids: Vec<ParentIds>,
    pub assign: Vec<Vec<Option<usize>>>,
    /// Observations found / probes.
    pub covered: usize,
    pub note: Option<String>,
}

impl ParentAssignment {
    /// Builds the assignment from `(expert_index, region_id, lineage_id, bar_end_ms)` observations.
    pub fn from_observations(
        snapshot_id: &str,
        n_experts: usize,
        probe_ends: &[i64],
        obs: impl IntoIterator<Item = (u32, String, String, i64)>,
    ) -> Self {
        let pos: HashMap<i64, usize> = probe_ends.iter().enumerate().map(|(i, t)| (*t, i)).collect();
        let mut ids: Vec<ParentIds> = vec![Vec::new(); n_experts];
        let mut assign = vec![vec![None; probe_ends.len()]; n_experts];
        let mut covered = vec![false; probe_ends.len()];
        for (k, region_id, lineage_id, ts) in obs {
            let (Some(&w), Some(list)) = (pos.get(&ts), ids.get_mut(k as usize)) else { continue };
            let idx = match list.iter().position(|(r, _)| *r == region_id) {
                Some(i) => i,
                None => {
                    list.push((region_id, lineage_id));
                    list.len() - 1
                }
            };
            assign[k as usize][w] = Some(idx);
            covered[w] = true;
        }
        Self {
            snapshot_id: snapshot_id.to_owned(),
            ids,
            assign,
            covered: covered.iter().filter(|c| **c).count(),
            note: None,
        }
    }

    pub fn none(reason: impl Into<String>) -> Self {
        Self { note: Some(reason.into()), ..Self::default() }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Phase {
    Stats {
        next: usize,
        acc: NormAccumulator,
    },
    /// Statistics fitted; regions, alignment and the bundle remain ([`Fit::finish`]).
    Regions {
        stats: NormStats,
    },
    Done,
}

/// Resumable fit state: everything but the bars (which are re-read from the same window).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fit {
    pub spec: FitSpec,
    pub snapshot_id: String,
    pub generation: u32,
    pub parent_snapshot_id: Option<String>,
    /// Window ends (exclusive bar indices) the statistics are fitted on.
    pub ends: Vec<usize>,
    /// Probe windows: indices into `ends`.
    pub probes: Vec<usize>,
    pub phase: Phase,
    pub started_ms: i64,
    #[serde(default)]
    pub feedback: Option<Value>,
}

/// Output of a completed fit.
#[derive(Debug, Clone, PartialEq)]
pub struct Bundle {
    pub manifest: SnapshotManifest,
    pub stats: NormStats,
    pub regions: Vec<Vec<FittedRegion>>,
    pub region_map: RegionMap,
}

impl Bundle {
    pub fn weights(&self) -> Vec<u8> {
        self.stats.to_bytes()
    }
}

impl Fit {
    pub fn new(
        spec: FitSpec,
        snapshot_id: String,
        generation: u32,
        parent_snapshot_id: Option<String>,
        history: &BarHistory,
        started_ms: i64,
    ) -> Self {
        let ends = StatisticalEncoder::fit_window_ends(history.bars.len(), &spec.fit);
        let probes = evenly(ends.len(), spec.probe_max.max(1));
        Self {
            spec,
            snapshot_id,
            generation,
            parent_snapshot_id,
            ends,
            probes,
            phase: Phase::Stats { next: 0, acc: NormAccumulator::default() },
            started_ms,
            feedback: None,
        }
    }

    pub fn n_windows(&self) -> usize {
        self.ends.len()
    }

    /// Bar-end timestamps of the probe windows.
    pub fn probe_ends_ms(&self, history: &BarHistory) -> Vec<i64> {
        self.probes.iter().filter_map(|&p| history.ends_ms.get(self.ends[p] - 1).copied()).collect()
    }

    /// Fraction of the statistics phase done (the long part).
    pub fn fraction(&self) -> f32 {
        match &self.phase {
            Phase::Stats { next, .. } if !self.ends.is_empty() => 0.9 * (*next as f32 / self.ends.len() as f32),
            Phase::Stats { .. } => 0.0,
            Phase::Regions { .. } => 0.9,
            Phase::Done => 1.0,
        }
    }

    /// One chunk of the statistics phase. Returns `true` while there is more to do.
    pub fn step(&mut self, history: &BarHistory) -> anyhow::Result<bool> {
        let Phase::Stats { next, acc } = &mut self.phase else { return Ok(false) };
        if self.ends.is_empty() {
            anyhow::bail!(
                "{} bars are fewer than min_window {} at {}",
                history.bars.len(),
                self.spec.fit.min_window,
                self.spec.timescale
            );
        }
        let enc = StatisticalEncoder::new(self.spec.timescale);
        let stop = (*next + CHUNK).min(self.ends.len());
        enc.fit_into(&history.bars, &self.ends[*next..stop], acc)?;
        *next = stop;
        if stop >= self.ends.len() {
            self.phase = Phase::Regions { stats: acc.finish() };
            return Ok(false);
        }
        Ok(true)
    }

    /// Finishes the fit: regions on the probe windows, alignment, bundle. Call after
    /// [`Self::step`] returned `false` with the statistics accumulated.
    pub fn finish(
        &mut self,
        history: &BarHistory,
        parent: &ParentAssignment,
        created_at: String,
    ) -> anyhow::Result<Bundle> {
        let stats = match &self.phase {
            Phase::Regions { stats } => stats.clone(),
            Phase::Stats { .. } => anyhow::bail!("statistics phase incomplete; keep stepping"),
            Phase::Done => anyhow::bail!("fit already finished"),
        };
        let enc = StatisticalEncoder::with_stats(self.spec.timescale, self.snapshot_id.clone(), stats.clone());
        let n_experts = EXPERT_NAMES.len();

        // Probe latents per expert.
        let mut latents: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(self.probes.len()); n_experts];
        let mut spread = vec![0.0f64; n_experts];
        for &p in &self.probes {
            let end = self.ends[p];
            let view = enc.encode(&EncoderInput::from_bars(history.ends_ms[end - 1], &history.bars[..end]))?;
            for (k, e) in view.experts.iter().enumerate() {
                spread[k] += e.mean.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / e.mean.len() as f64;
                latents[k].push(e.mean.clone());
            }
        }
        let n_probes = self.probes.len().max(1) as f64;
        let probe_ends = self.probe_ends_ms(history);

        let disc = LeaderClustering { radius: self.spec.region_radius, max_regions: self.spec.max_regions };
        let mut experts = Vec::with_capacity(n_experts);
        let mut regions = Vec::with_capacity(n_experts);
        for (k, points) in latents.iter().enumerate() {
            let centroids = disc.discover(points);
            let new_assign: Vec<Option<usize>> = points
                .iter()
                .map(|p| disc.assign(&centroids, p).or_else(|| fa_patterns::nearest(&centroids, p)))
                .collect();
            let empty = Vec::new();
            let old_assign = parent.assign.get(k).unwrap_or(&empty);
            let no_ids = ParentIds::new();
            let old_ids = parent.ids.get(k).unwrap_or(&no_ids);
            let m = align_expert(AlignInput {
                snapshot_id: &self.snapshot_id,
                expert_index: k as u32,
                new_centroids: &centroids,
                radius: self.spec.region_radius,
                new_assign: &new_assign,
                old_assign,
                old_ids,
                carry_threshold: self.spec.carry_threshold,
            });
            regions.push(m.regions.clone());
            experts.push(m);
        }
        let region_map = RegionMap {
            snapshot_id: self.snapshot_id.clone(),
            parent_snapshot_id: self.parent_snapshot_id.clone(),
            probe_set_version: probe_version(self.spec.timescale, &probe_ends),
            n_probes: self.probes.len(),
            carry_threshold: self.spec.carry_threshold,
            experts,
            note: parent.note.clone(),
        };

        let mut metrics: BTreeMap<String, f64> = BTreeMap::new();
        metrics.insert("n_bars".into(), history.bars.len() as f64);
        metrics.insert("n_windows".into(), self.ends.len() as f64);
        metrics.insert("n_probes".into(), self.probes.len() as f64);
        metrics.insert("parent_probe_coverage".into(), parent.covered as f64 / n_probes);
        metrics.insert("carried_mass".into(), region_map.carried_mass());
        metrics.insert("regions".into(), region_map.n_regions() as f64);
        metrics.insert("regions_matched".into(), region_map.n_matched() as f64);
        for (k, name) in EXPERT_NAMES.iter().enumerate() {
            metrics.insert(format!("latent_rms_{name}"), (spread[k] / n_probes).sqrt());
            metrics.insert(format!("regions_{name}"), regions[k].len() as f64);
        }
        let manifest = SnapshotManifest {
            snapshot_id: self.snapshot_id.clone(),
            encoder_family: STATISTICAL_FAMILY.into(),
            timescale: self.spec.timescale,
            generation: self.generation,
            parent_snapshot_id: self.parent_snapshot_id.clone(),
            created_at,
            params: enc.params(&self.spec.fit),
            probe_set: ProbeSetRef {
                version: region_map.probe_set_version.clone(),
                instrument: history.instrument.clone(),
                n_windows: probe_ends.len(),
                start_ms: probe_ends.first().copied().unwrap_or_default(),
                end_ms: probe_ends.last().copied().unwrap_or_default(),
                hash: fnv(&probe_ends),
            },
            data: DataRef {
                source: "victoriametrics:fa_trade_price".into(),
                instrument: history.instrument.clone(),
                start_ms: history.start_ms,
                end_ms: history.end_ms,
                n_bars: history.bars.len(),
            },
            metrics,
            feedback: self.feedback.clone(),
            seed: self.spec.seed,
        };
        self.phase = Phase::Done;
        Ok(Bundle { manifest, stats, regions, region_map })
    }
}

/// `k` indices evenly spread over `0..n` (all of them when `n ≤ k`).
pub fn evenly(n: usize, k: usize) -> Vec<usize> {
    if n <= k {
        return (0..n).collect();
    }
    (0..k).map(|i| (i * (n - 1)) / (k - 1).max(1)).collect()
}

fn fnv(ends: &[i64]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for t in ends {
        for b in t.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    format!("{h:016x}")
}

pub fn probe_version(ts: Timescale, ends: &[i64]) -> String {
    format!("probe-{ts}-{}-{}", ends.len(), &fnv(ends)[..8])
}

/// Snapshot id of a new fit: `stat-v1-<ts>-g<gen>-<unix seconds>`.
pub fn snapshot_id(ts: Timescale, generation: u32, now_ms: i64) -> String {
    format!("{}-{ts}-g{generation}-{}", StatisticalEncoder::VERSION, now_ms / 1000)
}

#[cfg(test)]
mod tests {
    use fa_core::rng::Rng;

    use super::*;

    fn history(n: usize, seed: u64) -> BarHistory {
        let mut rng = Rng::seed(seed);
        let mut p = 100.0;
        let pts: Vec<(i64, f64)> = (0..n)
            .map(|i| {
                p *= 1.0 + 0.002 * rng.normal();
                ((i as i64 + 1) * 60_000, p)
            })
            .collect();
        let (ends_ms, bars) = crate::data::bars_from_prices(&pts);
        BarHistory {
            instrument: "inst".into(),
            timescale: Timescale::Minute,
            start_ms: 0,
            end_ms: n as i64 * 60_000,
            ends_ms,
            bars,
        }
    }

    fn run(fit: &mut Fit, h: &BarHistory, parent: &ParentAssignment) -> Bundle {
        while fit.step(h).unwrap() {}
        fit.finish(h, parent, "t".into()).unwrap()
    }

    #[test]
    fn fit_is_deterministic_resumable_and_aligns_to_a_parent() {
        let h = history(400, 3);
        let spec = FitSpec { probe_max: 100, ..FitSpec::new(Timescale::Minute) };
        let mut a = Fit::new(spec.clone(), "snap".into(), 0, None, &h, 0);
        assert_eq!(a.n_windows(), 393);
        assert_eq!(a.probes.len(), 100);
        let none = ParentAssignment::none("no parent");
        let ba = run(&mut a, &h, &none);
        assert_eq!(ba.manifest.metrics["n_windows"], 393.0);
        assert_eq!(ba.region_map.carried_mass(), 0.0);
        assert_eq!(ba.region_map.note.as_deref(), Some("no parent"));
        assert!(ba.region_map.n_regions() >= 3);
        assert_eq!(ba.manifest.probe_set.n_windows, 100);

        // Resume: checkpoint after one chunk, continue from the serialised state.
        let mut b = Fit::new(spec.clone(), "snap".into(), 0, None, &h, 0);
        assert!(b.step(&h).unwrap());
        let json = serde_json::to_vec(&b).unwrap();
        let mut resumed: Fit = serde_json::from_slice(&json).unwrap();
        assert!(matches!(resumed.phase, Phase::Stats { next: CHUNK, .. }));
        let bb = run(&mut resumed, &h, &none);
        assert_eq!(bb.stats, ba.stats, "resumed fit equals the uninterrupted one");
        assert_eq!(bb.regions, ba.regions);

        // Parent = the same snapshot's own assignment → everything carries its lineage.
        let probe_ends = a.probe_ends_ms(&h);
        let obs = ba.regions.iter().enumerate().flat_map(|(k, regs)| {
            let map = ba.region_map.expert(k as u32).unwrap();
            // reconstruct the probe assignment from masses is not possible; re-derive by nearest centroid
            let centroids: Vec<Vec<f32>> = regs.iter().map(|r| r.centroid.clone()).collect();
            let enc = StatisticalEncoder::with_stats(Timescale::Minute, "snap", ba.stats.clone());
            let ends = a.ends.clone();
            let probes = a.probes.clone();
            let h = h.clone();
            let regs = regs.clone();
            let _ = map;
            probes.into_iter().map(move |p| {
                let end = ends[p];
                let v = enc.encode(&EncoderInput::from_bars(h.ends_ms[end - 1], &h.bars[..end])).unwrap();
                let i = fa_patterns::nearest(&centroids, &v.experts[k].mean).unwrap();
                (k as u32, format!("old-{}", regs[i].region_id), format!("L-{}", regs[i].region_id), h.ends_ms[end - 1])
            })
        });
        let parent = ParentAssignment::from_observations("parent", 3, &probe_ends, obs);
        assert_eq!(parent.covered, 100);
        let mut c = Fit::new(spec, "snap2".into(), 1, Some("parent".into()), &h, 0);
        let bc = run(&mut c, &h, &parent);
        assert!(
            bc.region_map.carried_mass() > 0.99,
            "{:?}",
            bc.region_map.experts.iter().map(|e| e.carried_mass).collect::<Vec<_>>()
        );
        assert!(bc.regions.iter().flatten().filter(|r| r.mass > 0.0).all(|r| r.lineage_id.starts_with("L-")));
        assert_eq!(bc.manifest.parent_snapshot_id.as_deref(), Some("parent"));
        assert_eq!(bc.manifest.generation, 1);
        assert_eq!(bc.manifest.metrics["parent_probe_coverage"], 1.0);
    }

    #[test]
    fn helpers() {
        assert_eq!(evenly(5, 10), vec![0, 1, 2, 3, 4]);
        assert_eq!(evenly(11, 3), vec![0, 5, 10]);
        assert_eq!(snapshot_id(Timescale::Minute, 2, 1_700_000_000_123), "stat-v1-1m-g2-1700000000");
        assert!(probe_version(Timescale::Hour, &[1, 2]).starts_with("probe-1h-2-"));
        let a = ParentAssignment::from_observations(
            "p",
            1,
            &[10, 20],
            vec![(0, "r".into(), "l".into(), 20), (0, "r".into(), "l".into(), 99)],
        );
        assert_eq!(a.covered, 1);
        assert_eq!(a.assign[0], vec![None, Some(0)]);
        assert_eq!(a.ids[0], vec![("r".to_owned(), "l".to_owned())]);
    }
}
