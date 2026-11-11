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
    AdapterGuard, DataRef, EXPERT_NAMES, EncoderFamily, EncoderInput, FitConfig, NormAccumulator, NormStats,
    ProbeSetRef, STATISTICAL_FAMILY, SnapshotManifest,
};
use fa_patterns::{AlignInput, FittedRegion, LeaderClustering, ParentIds, RegionMap, align_expert};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    data::{BarHistory, Instruments, Source},
    search::{self, Candidate, SearchSpec},
};

/// Windows accumulated per [`Fit::step`].
pub const CHUNK: usize = 64;

/// Normalisation options (`Job.spec.params`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FitParams {
    /// Pool per-instrument statistics with **equal weight per instrument** instead of by
    /// window count, so every instrument's z-scores sit on the same footing and a long or
    /// high-frequency history does not dominate the fitted scale. (The served encoder has one
    /// `NormStats`, so a literally per-instrument z-score is not servable; this is the
    /// pooled form of it. Per-instrument window counts are recorded in `params.norm`.)
    #[serde(default)]
    pub per_instrument_norm: bool,
}

/// What a job asks for (`Job.spec` JSON and the CLI flags map onto this).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitSpec {
    pub timescale: Timescale,
    /// History length; the window ends at job start.
    #[serde(default = "default_hours")]
    pub window_hours: f64,
    /// Single instrument (VM token or dashed UUID); default: the instrument with the most bars.
    #[serde(default)]
    pub instrument: Option<String>,
    /// Multi-instrument fit: a list of instruments or `"all"`; overrides `instrument`.
    #[serde(default)]
    pub instruments: Option<Instruments>,
    #[serde(default)]
    pub params: FitParams,
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
    /// At most this many probe windows (spread over the instruments, evenly within each).
    #[serde(default = "default_probes")]
    pub probe_max: usize,
    /// Feedback search of `generation` jobs ([`crate::search`]).
    #[serde(default)]
    pub search: SearchSpec,
    /// Adapter drift guard recorded in `manifest.params.adapter_guard` (default: spec 36 §7).
    #[serde(default)]
    pub adapter_guard: Option<AdapterGuard>,
    /// Encoder family: `statistical` (default) or a neural family (`fsq_hybrid`, `jepa_vae`;
    /// [`crate::neural_fit`]).
    #[serde(default)]
    pub family: Option<String>,
    /// Neural genome overrides merged into the per-timescale defaults (`steps`, `backbone`,
    /// `context_steps`, `d_model`, `layers`, `moe: {max_experts, expert_dim, top_k, ...}`, `lr`, ...).
    #[serde(default)]
    pub neural: Option<Value>,
    /// Compute device of a neural fit (`cpu`, `cuda[:n]`); default `FA_ML_DEVICE`, else cpu.
    #[serde(default)]
    pub device: Option<String>,
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
            instruments: None,
            params: FitParams::default(),
            parent_snapshot_id: None,
            seed: 0,
            fit: FitConfig::default(),
            region_radius: default_radius(),
            max_regions: default_max_regions(),
            carry_threshold: default_carry(),
            probe_max: default_probes(),
            search: SearchSpec::default(),
            adapter_guard: None,
            family: None,
            neural: None,
            device: None,
        }
    }

    /// Parameters a fit uses without a search (pretrain, or `search.candidates = 1`).
    pub fn default_candidate(&self) -> Candidate {
        Candidate::defaults(self.timescale, self.region_radius, self.max_regions)
    }
}

/// A probe window: instrument token and bar-end timestamp.
pub type ProbeKey = (String, i64);

/// The parent's region assignment of the probe windows, per expert: `assign[k][w]` is the index
/// into `ids[k]` of the parent region probe `w` fell in (looked up by instrument and bar-end
/// timestamp among the parent's `LatentVec` observations), `None` when the parent never
/// observed that window.
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
    /// Builds the assignment from `(expert_index, region_id, lineage_id, instrument_token, bar_end_ms)`.
    pub fn from_observations(
        snapshot_id: &str,
        n_experts: usize,
        probes: &[ProbeKey],
        obs: impl IntoIterator<Item = (u32, String, String, String, i64)>,
    ) -> Self {
        let pos: HashMap<(&str, i64), usize> =
            probes.iter().enumerate().map(|(i, (inst, t))| ((inst.as_str(), *t), i)).collect();
        let mut ids: Vec<ParentIds> = vec![Vec::new(); n_experts];
        let mut assign = vec![vec![None; probes.len()]; n_experts];
        let mut covered = vec![false; probes.len()];
        for (k, region_id, lineage_id, inst, ts) in obs {
            let token = inst.replace('-', "").to_ascii_lowercase();
            let (Some(&w), Some(list)) = (pos.get(&(token.as_str(), ts)), ids.get_mut(k as usize)) else { continue };
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
    /// Per-instrument Welford accumulators; `inst`/`next` = the next window to accumulate.
    Stats {
        inst: usize,
        next: usize,
        accs: Vec<NormAccumulator>,
    },
    /// Statistics fitted (per instrument and pooled); search, regions, alignment and the bundle
    /// remain ([`Fit::finish`]).
    Regions {
        per_instrument: Vec<NormStats>,
        stats: NormStats,
    },
    Done,
}

/// Resumable fit state: everything but the bars (which are re-read from the same window and
/// sources).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fit {
    pub spec: FitSpec,
    pub snapshot_id: String,
    pub generation: u32,
    pub parent_snapshot_id: Option<String>,
    /// Instrument tokens, in history order.
    pub instruments: Vec<String>,
    /// Window ends (exclusive bar indices) per instrument the statistics are fitted on.
    pub ends: Vec<Vec<usize>>,
    /// Probe windows: `(instrument index, index into ends[instrument])`.
    pub probes: Vec<(usize, usize)>,
    pub phase: Phase,
    pub started_ms: i64,
    #[serde(default)]
    pub feedback: Option<Value>,
    /// Parameters chosen by the feedback search (`None` → [`FitSpec::default_candidate`]).
    #[serde(default)]
    pub chosen: Option<Candidate>,
}

/// Output of a completed fit.
#[derive(Debug, Clone, PartialEq)]
pub struct Bundle {
    pub manifest: SnapshotManifest,
    pub stats: NormStats,
    pub regions: Vec<Vec<FittedRegion>>,
    pub region_map: RegionMap,
    /// safetensors weights of a neural snapshot (then `stats` is empty).
    pub neural_weights: Option<Vec<u8>>,
}

impl Bundle {
    /// `weights.bin`: `FASTAT01` statistics, or the neural safetensors.
    pub fn weights(&self) -> Vec<u8> {
        match &self.neural_weights {
            Some(w) => w.clone(),
            None => self.stats.to_bytes(),
        }
    }
}

/// Probe windows spread over instruments: `probe_max / n` per instrument (≥ 1), evenly spaced
/// within each, instrument-major order.
pub fn probe_windows(ends: &[Vec<usize>], probe_max: usize) -> Vec<(usize, usize)> {
    let per = (probe_max.max(1) / ends.len().max(1)).max(1);
    ends.iter().enumerate().flat_map(|(i, e)| evenly(e.len(), per).into_iter().map(move |w| (i, w))).collect()
}

impl Fit {
    pub fn new(
        spec: FitSpec,
        snapshot_id: String,
        generation: u32,
        parent_snapshot_id: Option<String>,
        histories: &[BarHistory],
        started_ms: i64,
    ) -> Self {
        let ends: Vec<Vec<usize>> = histories.iter().map(|h| search::window_ends(h, &spec.fit)).collect();
        let probes = probe_windows(&ends, spec.probe_max);
        Self {
            phase: Phase::Stats { inst: 0, next: 0, accs: vec![NormAccumulator::default(); histories.len()] },
            instruments: histories.iter().map(|h| h.instrument.clone()).collect(),
            spec,
            snapshot_id,
            generation,
            parent_snapshot_id,
            ends,
            probes,
            started_ms,
            feedback: None,
            chosen: None,
        }
    }

    pub fn n_windows(&self) -> usize {
        self.ends.iter().map(Vec::len).sum()
    }

    /// Instrument and bar-end timestamp of every probe window.
    pub fn probe_keys(&self, histories: &[BarHistory]) -> Vec<ProbeKey> {
        self.probes
            .iter()
            .filter_map(|&(i, p)| {
                let h = histories.get(i)?;
                h.ends_ms.get(self.ends[i][p] - 1).map(|t| (h.instrument.clone(), *t))
            })
            .collect()
    }

    fn done_windows(&self) -> usize {
        match &self.phase {
            Phase::Stats { inst, next, .. } => self.ends.iter().take(*inst).map(Vec::len).sum::<usize>() + next,
            _ => self.n_windows(),
        }
    }

    /// Fraction of the statistics phase done (the long part).
    pub fn fraction(&self) -> f32 {
        match &self.phase {
            Phase::Stats { .. } if self.n_windows() > 0 => 0.8 * (self.done_windows() as f32 / self.n_windows() as f32),
            Phase::Stats { .. } => 0.0,
            Phase::Regions { .. } => 0.8,
            Phase::Done => 1.0,
        }
    }

    /// One chunk of the statistics phase. Returns `true` while there is more to do.
    pub fn step(&mut self, histories: &[BarHistory]) -> anyhow::Result<bool> {
        if self.n_windows() == 0 {
            anyhow::bail!(
                "{} bars over {} instrument(s) are fewer than min_window {} at {}",
                histories.iter().map(|h| h.bars.len()).sum::<usize>(),
                histories.len(),
                self.spec.fit.min_window,
                self.spec.timescale
            );
        }
        let Phase::Stats { inst, next, accs } = &mut self.phase else { return Ok(false) };
        while *inst < self.ends.len() && *next >= self.ends[*inst].len() {
            *inst += 1;
            *next = 0;
        }
        if *inst < self.ends.len() {
            let enc = fa_encoders::StatisticalEncoder::new(self.spec.timescale);
            let ends = &self.ends[*inst];
            let stop = (*next + CHUNK).min(ends.len());
            enc.fit_into(&histories[*inst].bars, &ends[*next..stop], &mut accs[*inst])?;
            *next = stop;
            if *inst + 1 < self.ends.len() || stop < ends.len() {
                return Ok(true);
            }
        }
        let per_instrument: Vec<NormStats> = accs.iter().map(NormAccumulator::finish).collect();
        let stats = NormStats::pool(&per_instrument, self.spec.params.per_instrument_norm)
            .ok_or_else(|| anyhow::anyhow!("no windows accumulated"))?;
        self.phase = Phase::Regions { per_instrument, stats };
        Ok(false)
    }

    /// Runs the feedback search and adopts its choice (statistics at the chosen context with
    /// gains, `chosen`, `feedback.search`).
    pub fn apply_search(
        &mut self,
        histories: &[BarHistory],
        anchor: &Candidate,
        skill: &search::ParentSkill,
    ) -> anyhow::Result<()> {
        let Phase::Regions { per_instrument, .. } = &self.phase else {
            anyhow::bail!("search needs the fitted statistics");
        };
        let out = search::run(
            self.spec.timescale,
            histories,
            &self.ends,
            search::default_context(self.spec.timescale),
            per_instrument,
            self.spec.params.per_instrument_norm,
            anchor,
            &self.spec.search,
            skill,
            self.spec.seed,
        )?;
        let per_instrument = per_instrument.clone();
        self.phase = Phase::Regions { per_instrument, stats: out.stats };
        self.chosen = Some(out.chosen);
        let mut fb = self.feedback.take().unwrap_or_else(|| json!({}));
        if !fb.is_object() {
            fb = json!({ "decoder_stats": fb });
        }
        fb["search"] = out.trace;
        self.feedback = Some(fb);
        Ok(())
    }

    /// Finishes the fit: regions on the probe windows, alignment, bundle. Call after
    /// [`Self::step`] returned `false` with the statistics accumulated.
    pub fn finish(
        &mut self,
        histories: &[BarHistory],
        parent: &ParentAssignment,
        created_at: String,
    ) -> anyhow::Result<Bundle> {
        let (per_instrument, stats) = match &self.phase {
            Phase::Regions { per_instrument, stats } => (per_instrument.clone(), stats.clone()),
            Phase::Stats { .. } => anyhow::bail!("statistics phase incomplete; keep stepping"),
            Phase::Done => anyhow::bail!("fit already finished"),
        };
        let ts = self.spec.timescale;
        let cand = self.chosen.clone().unwrap_or_else(|| self.spec.default_candidate());
        let enc = search::encoder(ts, &self.snapshot_id, &stats, cand.context_steps);
        let n_experts = EXPERT_NAMES.len();

        // Probe latents per expert.
        let mut latents: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(self.probes.len()); n_experts];
        let mut spread = vec![0.0f64; n_experts];
        for &(i, p) in &self.probes {
            let (h, end) = (&histories[i], self.ends[i][p]);
            let view = enc.encode(&EncoderInput::from_bars(h.ends_ms[end - 1], &h.bars[..end]))?;
            for (k, e) in view.experts.iter().enumerate() {
                spread[k] += e.mean.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / e.mean.len() as f64;
                latents[k].push(e.mean.clone());
            }
        }
        let n_probes = self.probes.len().max(1) as f64;
        let probe_keys = self.probe_keys(histories);
        let probe_ts: Vec<i64> = probe_keys.iter().map(|(_, t)| *t).collect();

        let disc = LeaderClustering { radius: cand.region_radius, max_regions: cand.max_regions };
        let mut experts = Vec::with_capacity(n_experts);
        let mut regions = Vec::with_capacity(n_experts);
        for (k, points) in latents.iter().enumerate() {
            let (centroids, new_assign) = disc.fit_assign(points);
            let empty = Vec::new();
            let old_assign = parent.assign.get(k).unwrap_or(&empty);
            let no_ids = ParentIds::new();
            let old_ids = parent.ids.get(k).unwrap_or(&no_ids);
            let m = align_expert(AlignInput {
                snapshot_id: &self.snapshot_id,
                expert_index: k as u32,
                new_centroids: &centroids,
                radius: cand.region_radius,
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
            probe_set_version: probe_version(ts, &probe_ts),
            n_probes: self.probes.len(),
            carry_threshold: self.spec.carry_threshold,
            experts,
            note: parent.note.clone(),
        };

        let n_bars: usize = histories.iter().map(|h| h.bars.len()).sum();
        let n_from = |src: Source| histories.iter().filter(|h| h.source == src).count() as f64;
        let mut metrics: BTreeMap<String, f64> = BTreeMap::new();
        metrics.insert("n_bars".into(), n_bars as f64);
        metrics.insert("n_windows".into(), self.n_windows() as f64);
        metrics.insert("n_probes".into(), self.probes.len() as f64);
        metrics.insert("n_instruments".into(), histories.len() as f64);
        metrics.insert("instruments_bars".into(), n_from(Source::Bars));
        metrics.insert("instruments_trades".into(), n_from(Source::Trades));
        metrics.insert("parent_probe_coverage".into(), parent.covered as f64 / n_probes);
        metrics.insert("carried_mass".into(), region_map.carried_mass());
        metrics.insert("regions".into(), region_map.n_regions() as f64);
        metrics.insert("regions_matched".into(), region_map.n_matched() as f64);
        for (k, name) in EXPERT_NAMES.iter().enumerate() {
            metrics.insert(format!("latent_rms_{name}"), (spread[k] / n_probes).sqrt());
            metrics.insert(format!("regions_{name}"), regions[k].len() as f64);
        }
        if let Some(score) = self.feedback.as_ref().and_then(|f| f["search"]["chosen_score"].as_f64()) {
            metrics.insert("feedback_score".into(), score);
        }

        let mut params = enc.params(&self.spec.fit);
        params["region_radius"] = json!(cand.region_radius);
        params["max_regions"] = json!(cand.max_regions);
        params["search_params"] = json!(cand);
        params["norm"] = json!({
            "pooling": if self.spec.params.per_instrument_norm { "equal_per_instrument" } else { "by_window_count" },
            "per_instrument_norm": self.spec.params.per_instrument_norm,
            "gain": stats.gain,
            "per_instrument_windows": self.instruments.iter().zip(&per_instrument)
                .map(|(i, s)| json!({ "instrument": i, "n_windows": s.n_windows })).collect::<Vec<_>>(),
        });
        self.spec.adapter_guard.clone().unwrap_or_default().write_params(&mut params);

        let data_refs: Vec<_> = histories.iter().map(BarHistory::data_ref).collect();
        let source = match (n_from(Source::Bars) > 0.0, n_from(Source::Trades) > 0.0) {
            (true, false) => "victoriametrics:fa_bar",
            (false, true) => "victoriametrics:fa_trade_price",
            _ => "victoriametrics:fa_bar+fa_trade_price",
        };
        let instrument_label =
            if histories.len() == 1 { histories[0].instrument.clone() } else { format!("multi:{}", histories.len()) };
        let manifest = SnapshotManifest {
            snapshot_id: self.snapshot_id.clone(),
            encoder_family: STATISTICAL_FAMILY.into(),
            timescale: ts,
            generation: self.generation,
            parent_snapshot_id: self.parent_snapshot_id.clone(),
            created_at,
            params,
            probe_set: ProbeSetRef {
                version: region_map.probe_set_version.clone(),
                instrument: instrument_label.clone(),
                n_windows: probe_keys.len(),
                start_ms: probe_ts.iter().copied().min().unwrap_or_default(),
                end_ms: probe_ts.iter().copied().max().unwrap_or_default(),
                hash: fnv_keys(&probe_keys),
            },
            data: DataRef {
                source: source.into(),
                instrument: instrument_label,
                start_ms: histories.iter().map(|h| h.start_ms).min().unwrap_or_default(),
                end_ms: histories.iter().map(|h| h.end_ms).max().unwrap_or_default(),
                n_bars,
                instruments: data_refs,
            },
            metrics,
            feedback: self.feedback.clone(),
            seed: self.spec.seed,
        };
        self.phase = Phase::Done;
        Ok(Bundle { manifest, stats, regions, region_map, neural_weights: None })
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

pub fn fnv_keys(keys: &[ProbeKey]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (inst, t) in keys {
        for b in inst.as_bytes().iter().copied().chain(t.to_le_bytes()) {
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
    format!("{}-{ts}-g{generation}-{}", fa_encoders::StatisticalEncoder::VERSION, now_ms / 1000)
}

#[cfg(test)]
mod tests {
    use fa_core::rng::Rng;

    use super::*;

    fn history(n: usize, seed: u64, inst: &str, vol: f64) -> BarHistory {
        let mut rng = Rng::seed(seed);
        let mut p = 100.0;
        let pts: Vec<(i64, f64)> = (0..n)
            .map(|i| {
                p *= 1.0 + vol * rng.normal();
                ((i as i64 + 1) * 60_000, p)
            })
            .collect();
        let (ends_ms, bars) = crate::data::bars_from_prices(&pts);
        BarHistory {
            instrument: inst.into(),
            timescale: Timescale::Minute,
            source: Source::Trades,
            start_ms: 0,
            end_ms: n as i64 * 60_000,
            ends_ms,
            bars,
        }
    }

    fn run(fit: &mut Fit, h: &[BarHistory], parent: &ParentAssignment) -> Bundle {
        while fit.step(h).unwrap() {}
        fit.finish(h, parent, "t".into()).unwrap()
    }

    #[test]
    fn fit_is_deterministic_resumable_and_aligns_to_a_parent() {
        let h = vec![history(400, 3, "inst", 0.002)];
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
        assert_eq!(ba.manifest.data.instruments[0].source, "trades");
        assert_eq!(ba.manifest.data.source, "victoriametrics:fa_trade_price");
        assert_eq!(ba.manifest.adapter_guard(), AdapterGuard::default());

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
        let keys = a.probe_keys(&h);
        let enc = search::encoder(Timescale::Minute, "snap", &ba.stats, search::default_context(Timescale::Minute));
        let mut obs = Vec::new();
        for (k, regs) in ba.regions.iter().enumerate() {
            let centroids: Vec<Vec<f32>> = regs.iter().map(|r| r.centroid.clone()).collect();
            for &(i, p) in &a.probes {
                let end = a.ends[i][p];
                let v = enc.encode(&EncoderInput::from_bars(h[i].ends_ms[end - 1], &h[i].bars[..end])).unwrap();
                let r = &regs[fa_patterns::nearest(&centroids, &v.experts[k].mean).unwrap()];
                obs.push((
                    k as u32,
                    format!("old-{}", r.region_id),
                    format!("L-{}", r.region_id),
                    "INST".to_owned(),
                    h[i].ends_ms[end - 1],
                ));
            }
        }
        let parent = ParentAssignment::from_observations("parent", 3, &keys, obs);
        assert_eq!(parent.covered, 100, "instrument tokens compare case-insensitively");
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
    fn multi_instrument_fit_pools_statistics_and_spreads_probes() {
        let hs = vec![history(300, 1, "aaa", 0.001), history(120, 2, "bbb", 0.02)];
        let mut by_count = FitSpec { probe_max: 40, ..FitSpec::new(Timescale::Minute) };
        by_count.instruments = Some(Instruments::List(vec!["aaa".into(), "bbb".into()]));
        let mut f = Fit::new(by_count.clone(), "s".into(), 0, None, &hs, 0);
        assert_eq!(f.n_windows(), 293 + 113);
        assert_eq!(f.probes.iter().filter(|(i, _)| *i == 0).count(), 20);
        assert_eq!(f.probes.iter().filter(|(i, _)| *i == 1).count(), 20);
        let b = run(&mut f, &hs, &ParentAssignment::none("x"));
        assert_eq!(b.manifest.data.instrument, "multi:2");
        assert_eq!(b.manifest.data.instruments.len(), 2);
        assert_eq!(b.manifest.metrics["n_instruments"], 2.0);
        assert_eq!(b.manifest.params["norm"]["pooling"], "by_window_count");

        // Pooling by count equals one accumulator over every window of both instruments.
        let enc = fa_encoders::StatisticalEncoder::new(Timescale::Minute);
        let mut one = NormAccumulator::default();
        for (h, e) in hs.iter().zip(&f.ends) {
            enc.fit_into(&h.bars, e, &mut one).unwrap();
        }
        let direct = one.finish();
        for (row_a, row_b) in b.stats.mean.iter().flatten().zip(direct.mean.iter().flatten()) {
            assert!((row_a - row_b).abs() < 1e-4, "{row_a} vs {row_b}");
        }
        for (row_a, row_b) in b.stats.std.iter().flatten().zip(direct.std.iter().flatten()) {
            assert!((row_a - row_b).abs() < 1e-3, "{row_a} vs {row_b}");
        }

        // per_instrument_norm: equal weights → differs from count pooling when the instruments differ.
        let eq = FitSpec { params: FitParams { per_instrument_norm: true }, ..by_count };
        let mut g = Fit::new(eq, "s".into(), 0, None, &hs, 0);
        let be = run(&mut g, &hs, &ParentAssignment::none("x"));
        assert_eq!(be.manifest.params["norm"]["pooling"], "equal_per_instrument");
        assert_ne!(be.stats.mean, b.stats.mean);
        let per = [direct.clone()];
        assert_eq!(NormStats::pool(&per, true).unwrap().mean, direct.mean, "one instrument: pooling is identity");
    }

    #[test]
    fn generation_search_choice_lands_in_manifest_params_and_feedback() {
        let hs = vec![history(260, 5, "aaa", 0.003)];
        let mut spec = FitSpec { probe_max: 50, ..FitSpec::new(Timescale::Minute) };
        spec.search = SearchSpec { candidates: 4, context_steps: vec![16, 64], eval_max: 200, ..SearchSpec::default() };
        spec.seed = 3;
        let mut f = Fit::new(spec.clone(), "g".into(), 1, Some("p".into()), &hs, 0);
        while f.step(&hs).unwrap() {}
        f.feedback = Some(json!({ "decoder_stats": { "deflated_sharpe": 0.1 } }));
        let anchor = spec.default_candidate();
        f.apply_search(&hs, &anchor, &search::ParentSkill::default()).unwrap();
        let chosen = f.chosen.clone().unwrap();
        let b = f.finish(&hs, &ParentAssignment::none("x"), "t".into()).unwrap();
        assert_eq!(b.manifest.params["search_params"], serde_json::to_value(&chosen).unwrap());
        assert_eq!(b.manifest.params["context_steps"], chosen.context_steps);
        let fb = b.manifest.feedback.as_ref().unwrap();
        assert_eq!(fb["decoder_stats"]["deflated_sharpe"], 0.1, "previous summary kept");
        assert_eq!(fb["search"]["candidates"].as_array().unwrap().len(), 4);
        assert!(b.manifest.metrics.contains_key("feedback_score"));
        // the artifact reproduces the chosen encoder (context + gains)
        let enc = fa_encoders::StatisticalEncoder::from_manifest(&b.manifest, &b.weights()).unwrap();
        assert_eq!(enc.sizing().context_steps, chosen.context_steps);
        assert_eq!(enc.norm().unwrap(), &b.stats.folded(), "gains folded into weights.bin (FASTAT01)");
        assert_eq!(&b.weights()[..8], b"FASTAT01");
        let probe = EncoderInput::from_bars(hs[0].ends_ms[99], &hs[0].bars[..100]);
        let in_memory = search::encoder(Timescale::Minute, "g", &b.stats, chosen.context_steps);
        let means = |e: &fa_encoders::StatisticalEncoder| {
            e.encode(&probe).unwrap().experts.into_iter().map(|x| x.mean).collect::<Vec<_>>()
        };
        assert_eq!(means(&enc), means(&in_memory));
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
            &[("i".into(), 10), ("i".into(), 20)],
            vec![
                (0, "r".into(), "l".into(), "i".into(), 20),
                (0, "r".into(), "l".into(), "i".into(), 99),
                (0, "r".into(), "l".into(), "other".into(), 10),
            ],
        );
        assert_eq!(a.covered, 1);
        assert_eq!(a.assign[0], vec![None, Some(0)]);
        assert_eq!(a.ids[0], vec![("r".to_owned(), "l".to_owned())]);
        assert_eq!(probe_windows(&[vec![0; 10], vec![0; 3]], 4), vec![(0, 0), (0, 9), (1, 0), (1, 2)]);
    }
}
