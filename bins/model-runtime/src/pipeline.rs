//! Per-timescale latent pipeline: encoder → router → regions → transitions (specs 30–33).
//!
//! The stages are trait objects so the baselines can be swapped for learned models:
//! [`EncoderFamily`] (fa-encoders), [`Router`] (fa-expert-space),
//! [`OnlineRegionDiscovery`] (fa-patterns, one set per expert) and [`TransitionModel`]
//! (backed by `fa_mcmc::MarkovGraph` in a sampler arena, one per expert,
//! [`crate::regions::ArenaTransitions`]). The region and transition stages only see the
//! encoder's per-expert `mean` vectors, so any `EncoderFamily` plugs in. The pipeline is pure
//! with respect to the bus: `step` returns the protobuf messages to publish (states, soft
//! memberships, region events and lifecycle messages, transition rows, kernel summaries).

use std::collections::HashMap;

use fa_core::{Timescale, rng::Rng};
use fa_encoders::{EncoderFamily, EncoderInput, MarketBar, StatisticalEncoder};
use fa_expert_space::{EvidenceSoftmaxRouter, Router};
use fa_mcmc::{MarkovGraph, RegionCarry, calibration::SamplerStanding};
use fa_patterns::{
    DriftMonitor, OnlineRegionCfg, OnlineRegionDiscovery, OnlineRegionSet, RegionLifecycle, RegionMap, RegionStat,
};
use fa_proto::{
    Timestamp,
    v1::{
        ExpertLatent, ExpertMembership, ExpertSpaceRef, KernelUpdate, LatentState, RegionEvent, RegionLifecycleEvent,
        RegionMembership, RegionWeight, TransitionProb, TransitionUpdate, region_event,
    },
};

use crate::regions::{ArenaTransitions, KernelCfg, kernel_update, lifecycle_map, lifecycle_online};

/// Tunables of the baseline pipeline (spec 30 §7, spec 33 §7).
#[derive(Debug, Clone, Copy)]
pub struct PipelineConfig {
    pub regions: OnlineRegionCfg,
    /// Symmetric Dirichlet prior per successor (`alpha_0`).
    pub alpha_0: f64,
    /// Central credible-interval mass of published transition probabilities.
    pub ci_level: f64,
    /// Dirichlet draws for the credible interval (`mc_draws`).
    pub mc_draws: usize,
    pub seed: u64,
    /// Kernel transfer on swap (spec 33 §6.6 `carry_discount`): carried observation mass is
    /// multiplied by `score · carry_discount` so transferred evidence counts less than fresh.
    pub carry_discount: f64,
    /// Sampler arena, kernel summaries, drift monitor, geometric prior (spec 33 §6.1, §6.4).
    pub kernel: KernelCfg,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            regions: OnlineRegionCfg::default(),
            alpha_0: 1.0,
            ci_level: 0.9,
            mc_draws: 256,
            seed: 7,
            carry_discount: 0.5,
            kernel: KernelCfg::default(),
        }
    }
}

/// Forgetting factor per timescale (spec 33 §7 `gamma_forget`).
pub fn gamma_forget(ts: Timescale) -> f64 {
    match ts {
        Timescale::Second => 0.999,
        Timescale::Minute => 0.9995,
        Timescale::Hour => 0.999,
        Timescale::Day => 0.995,
        Timescale::Week => 0.99,
    }
}

/// Bayesian transition tracking over one region set (spec 33 §4.1 `TransitionKernel`, hard memberships).
pub trait TransitionModel: Send {
    fn sampler_id(&self) -> &str;
    /// A region was created: it receives prior mass in every row (spec 33 §5).
    fn register_region(&mut self, id: &str);
    /// A region was retired (merged away).
    fn retire_region(&mut self, id: &str);
    fn observe(&mut self, from: &str, to: &str);
    /// Posterior mean + credible interval of the row `from`.
    fn summarize(&mut self, from: &str) -> Vec<TransitionProb>;
    fn n_rows(&self) -> usize;
    /// The underlying Dirichlet graph, when the model has one (kernel transfer, inference).
    fn graph(&self) -> Option<&MarkovGraph> {
        None
    }
    /// Region centroid for the geometric prior (spec 33 §6.1).
    fn set_centroid(&mut self, _id: &str, _centroid: &[f32]) {}
    /// Calibration standings of the competing samplers (spec 33 §6.4).
    fn standings(&self) -> Vec<SamplerStanding> {
        Vec::new()
    }
    /// Replaces the evidence (kernel transfer, restore from Helix).
    fn replace_graph(&mut self, _g: MarkovGraph) {}
}

/// One encoding step.
pub struct StepInput<'a> {
    /// Instrument UUID as carried in the envelope.
    pub instrument_id: &'a str,
    pub as_of_ms: i64,
    /// Bars of the context window, oldest first (the last may be a forming bar).
    pub bars: &'a [MarketBar],
    /// True when the last bar is closed: a step of the timescale's Markov chain. Only then
    /// are transitions observed; previews between closes update states and regions only.
    pub bar_closed: bool,
}

#[derive(Debug, Default)]
pub struct StepOutput {
    pub state: Option<LatentState>,
    pub region_events: Vec<RegionEvent>,
    pub transitions: Vec<TransitionUpdate>,
    /// Soft memberships of every expert (`latent.membership.<ts>.<instrument>`).
    pub membership: Option<RegionMembership>,
    /// Lifecycle with lineage and event time (`latent.lifecycle.<ts>`).
    pub lifecycle: Vec<RegionLifecycleEvent>,
    /// Graph-level kernel summaries due at this step (`latent.mcmc.<ts>.<expert>`).
    pub kernels: Vec<KernelUpdate>,
}

/// What the Helix sink persists per expert space (spec 30 §8, spec 33 §8).
#[derive(Debug, Clone)]
pub struct ExpertPersist {
    pub space: ExpertSpaceRef,
    /// Encoder family (`serde` name of `fa_encoders::FamilyId`).
    pub family: String,
    /// Serialised region set (`None` when the discovery cannot export).
    pub region_state: Option<serde_json::Value>,
    /// Serialised serving kernel graph.
    pub kernel_state: Option<String>,
    pub stats: Vec<RegionStat>,
}

/// What [`TimescalePipeline::swap_encoder`] did.
#[derive(Debug, Default)]
pub struct SwapReport {
    /// `Created` events for every seeded region of the new snapshot.
    pub events: Vec<RegionEvent>,
    /// Observation mass carried into the new posteriors (spec 33 §6.6), summed over experts.
    pub carried: f64,
    /// Old observation mass without a destination.
    pub dropped: f64,
    /// Rows that received carried mass.
    pub rows_carried: usize,
    /// Lifecycle of the alignment (carried / split / merged / born regions, dead parents).
    pub lifecycle: Vec<RegionLifecycleEvent>,
    /// The map was a representation regime break: posteriors restarted from the prior.
    pub regime_break: bool,
}

/// Region sets and transition models, one of each per expert.
type Stages = (Vec<Box<dyn OnlineRegionDiscovery>>, Vec<Box<dyn TransitionModel>>);

pub struct TimescalePipeline {
    ts: Timescale,
    encoder: Box<dyn EncoderFamily>,
    router: Box<dyn Router>,
    regions: Vec<Box<dyn OnlineRegionDiscovery>>,
    transitions: Vec<Box<dyn TransitionModel>>,
    /// Region per expert at the last closed bar, per instrument.
    last_closed: HashMap<String, Vec<String>>,
    /// Occupancy drift per expert (spec 30 §6.6).
    drift: Vec<DriftMonitor>,
    /// Observations per expert (drift warm-up).
    observed: Vec<u64>,
    /// Event time of the last `KernelUpdate` per expert.
    last_kernel_ms: Vec<i64>,
    kcfg: KernelCfg,
    rng: Rng,
    /// `novelty_threshold` of the region sets (out-of-region flag of memberships).
    novelty_threshold: f32,
}

impl TimescalePipeline {
    /// Baseline wiring: statistical encoder, energy-softmax router, leader clustering, Dirichlet rows.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn baseline(ts: Timescale, cfg: &PipelineConfig) -> Self {
        Self::with_encoder(ts, cfg, Box::new(StatisticalEncoder::new(ts)), None)
    }

    /// Any encoder (a loaded snapshot, spec 36) with the baseline router / regions / transitions.
    /// With a `region_map` (the snapshot's `region_map.json`) the region sets start seeded with
    /// its regions, so matched lineages continue (D-007); otherwise they start empty.
    pub fn with_encoder(
        ts: Timescale,
        cfg: &PipelineConfig,
        encoder: Box<dyn EncoderFamily>,
        region_map: Option<&RegionMap>,
    ) -> Self {
        let n_experts = encoder.sizing().max_experts as usize;
        let (regions, transitions) = Self::stages(ts, cfg, encoder.snapshot_id(), n_experts, region_map);
        Self {
            ts,
            encoder,
            router: Box::new(EvidenceSoftmaxRouter::default()),
            regions,
            transitions,
            last_closed: HashMap::new(),
            drift: Self::drift_monitors(cfg, n_experts, region_map),
            observed: vec![0; n_experts],
            last_kernel_ms: vec![i64::MIN; n_experts],
            kcfg: cfg.kernel,
            rng: Rng::seed(cfg.seed ^ 0xD1A6),
            novelty_threshold: cfg.regions.novelty_threshold,
        }
    }

    /// Drift references: the fitted region masses of a trained snapshot, else captured live.
    fn drift_monitors(cfg: &PipelineConfig, n_experts: usize, region_map: Option<&RegionMap>) -> Vec<DriftMonitor> {
        (0..n_experts)
            .map(|k| match region_map.and_then(|m| m.expert(k as u32)) {
                Some(e) => DriftMonitor::with_reference(
                    cfg.kernel.drift_refit_kl,
                    e.regions.iter().map(|r| (r.region_id.clone(), r.mass)),
                ),
                None => DriftMonitor::new(cfg.kernel.drift_refit_kl),
            })
            .collect()
    }

    fn transition_model(
        cfg: &PipelineConfig,
        ts: Timescale,
        k: usize,
        graph: Option<MarkovGraph>,
    ) -> Box<dyn TransitionModel> {
        let g = graph.unwrap_or_else(|| MarkovGraph::new(cfg.alpha_0, gamma_forget(ts)));
        Box::new(ArenaTransitions::from_graph(
            g,
            cfg.ci_level,
            cfg.mc_draws,
            &cfg.kernel,
            cfg.seed ^ ((k as u64 + 1) * 0x9E37_79B9),
        ))
    }

    fn stages(
        ts: Timescale,
        cfg: &PipelineConfig,
        snapshot: &str,
        n_experts: usize,
        region_map: Option<&RegionMap>,
    ) -> Stages {
        let regions = (0..n_experts)
            .map(|k| {
                // Region ids: `<ts>-e<k>-r<nnn>` for the in-process baseline, `<snapshot>-e<k>-r<nnn>` for a
                // trained snapshot (its `regions.json` uses that prefix, so new births continue the series).
                let prefix = if region_map.is_some() { format!("{snapshot}-e{k}") } else { format!("{ts}-e{k}") };
                let set = match region_map.and_then(|m| m.expert(k as u32)) {
                    Some(e) => OnlineRegionSet::seeded(cfg.regions, snapshot, k as u32, prefix, &e.regions),
                    None => OnlineRegionSet::new(cfg.regions, snapshot, k as u32, prefix),
                };
                Box::new(set) as Box<dyn OnlineRegionDiscovery>
            })
            .collect();
        let transitions = (0..n_experts)
            .map(|k| {
                let mut t = Self::transition_model(cfg, ts, k, None);
                if let Some(e) = region_map.and_then(|m| m.expert(k as u32)) {
                    for r in &e.regions {
                        t.register_region(&r.region_id);
                        if cfg.kernel.geo_prior {
                            t.set_centroid(&r.region_id, &r.centroid);
                        }
                    }
                }
                t
            })
            .collect();
        (regions, transitions)
    }

    /// `Created` events for every region currently live (used once at start to announce the
    /// regions a loaded snapshot was seeded with).
    pub fn seeded_region_events(&self) -> Vec<RegionEvent> {
        self.regions
            .iter()
            .enumerate()
            .flat_map(|(k, set)| {
                set.regions()
                    .into_iter()
                    .map(|r| self.region_event(k as u32, RegionLifecycle::Created { region: r.clone() }))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Lifecycle messages of the loaded snapshot's alignment (announced once at start; the
    /// `DescendsFrom` upserts are idempotent).
    pub fn seeded_lifecycle(&self, region_map: Option<&RegionMap>, as_of_ms: i64) -> Vec<RegionLifecycleEvent> {
        let Some(map) = region_map else { return Vec::new() };
        let parent = map.parent_snapshot_id.clone();
        map.experts
            .iter()
            .flat_map(|e| {
                let space = self.space_ref(e.expert_index);
                let ps = parent.as_ref().map(|p| ExpertSpaceRef { snapshot_id: p.clone(), ..space.clone() });
                lifecycle_map(&space, ps.as_ref(), as_of_ms, e)
            })
            .collect()
    }

    /// Hot-swaps the encoder (spec 36 §6.4 promotion / rollback) without restarting: the region
    /// sets are rebuilt from the new snapshot's `region_map` (seeded) or start empty, the
    /// transition posteriors are **transferred** through the map (spec 33 §6.6,
    /// [`MarkovGraph::transfer`]: a seeded region with a `parent_region_id` inherits the parent's
    /// row scaled by `score · carry_discount`, unmatched regions start from the prior; without a
    /// map the posteriors restart) and the per-instrument chain positions are cleared. Returns
    /// `Created` events for every seeded region so the bus and Helix learn the new snapshot's
    /// regions, plus what the transfer carried.
    pub fn swap_encoder(
        &mut self,
        cfg: &PipelineConfig,
        encoder: Box<dyn EncoderFamily>,
        region_map: Option<&RegionMap>,
    ) -> SwapReport {
        let n_experts = encoder.sizing().max_experts as usize;
        let previous = self.encoder.snapshot_id().to_owned();
        self.encoder = encoder;
        let (regions, mut transitions) = Self::stages(self.ts, cfg, self.encoder.snapshot_id(), n_experts, region_map);
        let mut report = SwapReport::default();
        if let Some(map) = region_map {
            report.regime_break = map.regime_break();
            for (k, model) in transitions.iter_mut().enumerate() {
                let Some(e) = map.expert(k as u32) else { continue };
                // Spec 30 §8: an alignment that kept < 50 % of the probe mass is a regime break
                // of the representation; the kernel restarts from the (geometric) prior.
                if e.regime_break {
                    continue;
                }
                let Some(old) = self.transitions.get(k).and_then(|t| t.graph()) else { continue };
                let carries: Vec<RegionCarry> = e
                    .regions
                    .iter()
                    .filter_map(|r| {
                        r.parent_region_id.as_ref().map(|p| RegionCarry {
                            new_id: r.region_id.clone(),
                            old_id: p.clone(),
                            score: r.score,
                        })
                    })
                    .collect();
                let support: Vec<String> = e.regions.iter().map(|r| r.region_id.clone()).collect();
                let mut t = MarkovGraph::transfer(old, &carries, &support, cfg.carry_discount);
                report.carried += t.carried;
                report.dropped += t.dropped;
                report.rows_carried += t.rows_carried;
                if cfg.kernel.geo_prior {
                    for r in &e.regions {
                        t.graph.set_centroid(&r.region_id, &r.centroid);
                    }
                }
                *model = Self::transition_model(cfg, self.ts, k, Some(t.graph));
            }
        }
        self.regions = regions;
        self.transitions = transitions;
        self.last_closed.clear();
        self.drift = Self::drift_monitors(cfg, n_experts, region_map);
        self.observed = vec![0; n_experts];
        self.last_kernel_ms = vec![i64::MIN; n_experts];
        report.events = self.seeded_region_events();
        if let Some(map) = region_map {
            let now =
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64);
            for e in &map.experts {
                let space = self.space_ref(e.expert_index);
                let ps = ExpertSpaceRef { snapshot_id: previous.clone(), ..space.clone() };
                report.lifecycle.extend(lifecycle_map(&space, Some(&ps), now, e));
            }
        }
        report
    }

    /// Per-expert state for Helix (region set, kernel, statistics).
    pub fn persist_rows(&self) -> Vec<ExpertPersist> {
        self.regions
            .iter()
            .zip(&self.transitions)
            .enumerate()
            .map(|(k, (set, model))| ExpertPersist {
                space: self.space_ref(k as u32),
                family: serde_json::to_value(self.encoder.descriptor().id)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_owned))
                    .unwrap_or_default(),
                region_state: set.export_state(),
                kernel_state: model.graph().and_then(|g| serde_json::to_string(g).ok()),
                stats: set.stats(),
            })
            .collect()
    }

    /// Restores expert `k` from a persisted region set and kernel (spec 30 §8: a restarted
    /// runtime continues the same regions and posteriors). Returns `false` when the region state
    /// does not fit this set (other snapshot, other family); the kernel is restored only then.
    pub fn restore_expert(&mut self, k: usize, region_state: &serde_json::Value, kernel_state: &str) -> bool {
        let Some(set) = self.regions.get_mut(k) else { return false };
        if !set.restore_state(region_state) {
            return false;
        }
        if let Ok(g) = serde_json::from_str::<MarkovGraph>(kernel_state) {
            self.transitions[k].replace_graph(g);
        }
        for s in self.regions[k].stats() {
            self.transitions[k].register_region(&s.region_id);
            if self.kcfg.geo_prior {
                self.transitions[k].set_centroid(&s.region_id, &s.centroid);
            }
        }
        true
    }

    pub fn n_experts(&self) -> usize {
        self.regions.len()
    }

    /// Replaces only the encoder (an online adapter of the same snapshot, spec 36 §6.5): region
    /// sets, transition posteriors and chain positions are kept. The adapter must carry the same
    /// snapshot id and expert count; otherwise it is refused.
    pub fn set_encoder(&mut self, encoder: Box<dyn EncoderFamily>) -> Result<(), String> {
        if encoder.snapshot_id() != self.encoder.snapshot_id() {
            return Err(format!(
                "adapter is for snapshot {} but {} is serving",
                encoder.snapshot_id(),
                self.encoder.snapshot_id()
            ));
        }
        if encoder.sizing().max_experts != self.encoder.sizing().max_experts {
            return Err("adapter expert count differs from the serving encoder".into());
        }
        self.encoder = encoder;
        Ok(())
    }

    pub fn timescale(&self) -> Timescale {
        self.ts
    }

    pub fn snapshot_id(&self) -> &str {
        self.encoder.snapshot_id()
    }

    pub fn context_steps(&self) -> usize {
        self.encoder.sizing().context_steps as usize
    }

    pub fn live_regions(&self) -> usize {
        self.regions.iter().map(|r| r.regions().len()).sum()
    }

    pub fn transition_rows(&self) -> usize {
        self.transitions.iter().map(|t| t.n_rows()).sum()
    }

    fn space(&self, expert_index: u32) -> Option<ExpertSpaceRef> {
        Some(self.space_ref(expert_index))
    }

    pub fn space_ref(&self, expert_index: u32) -> ExpertSpaceRef {
        ExpertSpaceRef { snapshot_id: self.snapshot_id().to_owned(), timescale: self.ts.to_string(), expert_index }
    }

    pub fn step(&mut self, input: &StepInput<'_>) -> Result<StepOutput, fa_encoders::EncoderError> {
        let mut view = self.encoder.encode(&EncoderInput::from_bars(input.as_of_ms, input.bars))?;
        // A neural encoder ships its learned top-k gates (spec 32 §6.1); only encoders without a
        // learned gate (the statistical family) are routed by the baseline router.
        if view.router_weights.len() != view.experts.len() {
            view.router_weights = self.router.route(&view.experts);
        }

        let mut out = StepOutput::default();
        let mut current = Vec::with_capacity(view.experts.len());
        let mut experts = Vec::with_capacity(view.experts.len());
        let mut memberships = Vec::with_capacity(view.experts.len());
        let novelty_threshold = self.novelty_threshold;
        for e in &view.experts {
            let k = e.expert_index as usize;
            let a = self.regions[k].observe(&e.mean);
            self.observed[k] += 1;
            let space = self.space_ref(e.expert_index);
            let stats = if a.events.is_empty() { Vec::new() } else { self.regions[k].stats() };
            for ev in a.events {
                match &ev {
                    RegionLifecycle::Created { region } => {
                        self.transitions[k].register_region(&region.id);
                        if self.kcfg.geo_prior {
                            self.transitions[k].set_centroid(&region.id, &region.centroid);
                        }
                    }
                    RegionLifecycle::Split { child, .. } => {
                        self.transitions[k].register_region(&child.id);
                        if self.kcfg.geo_prior {
                            self.transitions[k].set_centroid(&child.id, &child.centroid);
                        }
                    }
                    RegionLifecycle::Retired { region_id, .. } => self.transitions[k].retire_region(region_id),
                    RegionLifecycle::Merged { .. } => {}
                }
                out.lifecycle.push(lifecycle_online(&space, input.as_of_ms, &ev, |id| {
                    stats.iter().find(|s| s.region_id == id).cloned()
                }));
                out.region_events.push(self.region_event(e.expert_index, ev));
            }
            let weights = if a.memberships.iter().any(|(r, _)| *r == a.region_id) {
                a.memberships.iter().map(|(r, w)| RegionWeight { region_id: r.clone(), weight: *w }).collect()
            } else {
                // a birth (or a split that moved the point): the new region holds it alone
                vec![RegionWeight { region_id: a.region_id.clone(), weight: 1.0 }]
            };
            memberships.push(ExpertMembership {
                expert_index: e.expert_index,
                weights,
                novelty: a.novelty,
                out_of_region: a.novelty >= novelty_threshold,
            });
            current.push(a.region_id.clone());
            experts.push(ExpertLatent {
                expert_index: e.expert_index,
                mean: e.mean.clone(),
                log_var: e.log_var.clone(),
                region_id: a.region_id,
                region_lineage_id: a.lineage_id,
            });
        }

        if input.bar_closed {
            if let Some(prev) = self.last_closed.remove(input.instrument_id) {
                for (k, (from, to)) in prev.iter().zip(&current).enumerate() {
                    let space = self.space(k as u32);
                    let model = &mut self.transitions[k];
                    model.observe(from, to);
                    out.transitions.push(TransitionUpdate {
                        space,
                        from_region_id: from.clone(),
                        to: model.summarize(from),
                        sampler_id: model.sampler_id().to_owned(),
                        as_of: Some(ms_to_ts(input.as_of_ms)),
                    });
                }
            }
            self.last_closed.insert(input.instrument_id.to_owned(), current);
            out.kernels = self.kernels_due(input.as_of_ms);
        }

        out.membership = Some(RegionMembership {
            snapshot_id: view.snapshot_id.clone(),
            timescale: self.ts.to_string(),
            instrument_id: input.instrument_id.to_owned(),
            as_of: Some(ms_to_ts(input.as_of_ms)),
            experts: memberships,
        });

        out.state = Some(LatentState {
            snapshot_id: view.snapshot_id,
            timescale: self.ts.to_string(),
            instrument_id: input.instrument_id.to_owned(),
            as_of: Some(ms_to_ts(input.as_of_ms)),
            experts,
            router_weights: view.router_weights,
        });
        Ok(out)
    }

    /// `KernelUpdate`s of the experts whose cadence elapsed (event time), with refreshed
    /// geometric-prior centroids and the drift check.
    fn kernels_due(&mut self, as_of_ms: i64) -> Vec<KernelUpdate> {
        if self.kcfg.kernel_every_ms <= 0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        for k in 0..self.regions.len() {
            if self.last_kernel_ms[k] != i64::MIN && as_of_ms - self.last_kernel_ms[k] < self.kcfg.kernel_every_ms {
                continue;
            }
            self.last_kernel_ms[k] = as_of_ms;
            let stats = self.regions[k].stats();
            if self.kcfg.geo_prior {
                for s in &stats {
                    self.transitions[k].set_centroid(&s.region_id, &s.centroid);
                }
            }
            let live: Vec<(String, f64)> = stats.iter().map(|s| (s.region_id.clone(), s.occupancy)).collect();
            if !self.drift[k].has_reference() && self.observed[k] >= self.kcfg.drift_warmup {
                self.drift[k].capture(live.clone());
            }
            let drift = self.drift[k].check(&live);
            let space = self.space_ref(k as u32);
            if let Some(u) = kernel_update(
                &space,
                as_of_ms,
                self.transitions[k].as_ref(),
                stats.len(),
                drift,
                self.kcfg.kernel_draws,
                &mut self.rng,
            ) {
                out.push(u);
            }
        }
        out
    }

    fn region_event(&self, expert_index: u32, ev: RegionLifecycle) -> RegionEvent {
        let (kind, region_id, lineage_id, related) = match ev {
            RegionLifecycle::Created { region } => (region_event::Kind::Created, region.id, region.lineage_id, vec![]),
            RegionLifecycle::Merged { survivor, lineage_id, absorbed } => {
                (region_event::Kind::Merged, survivor, lineage_id, vec![absorbed])
            }
            // the child is a new region (created on the bus and in Helix) descending from the parent
            RegionLifecycle::Split { parent, child, .. } => {
                (region_event::Kind::Split, child.id, child.lineage_id, vec![parent])
            }
            RegionLifecycle::Retired { region_id, lineage_id, merged_into, .. } => {
                (region_event::Kind::Retired, region_id, lineage_id, vec![merged_into])
            }
        };
        RegionEvent {
            kind: kind as i32,
            space: self.space(expert_index),
            region_id,
            lineage_id,
            related_region_ids: related,
        }
    }
}

pub fn ms_to_ts(ms: i64) -> Timestamp {
    Timestamp { seconds: ms.div_euclid(1000), nanos: (ms.rem_euclid(1000) * 1_000_000) as i32 }
}

#[cfg(test)]
mod tests {
    use fa_core::rng::Rng;

    use super::*;

    /// Synthetic test bars (seeded random walk, not market data).
    fn bars(n: usize, seed: u64) -> Vec<MarketBar> {
        let mut rng = Rng::seed(seed);
        let mut p = 50.0;
        (0..n)
            .map(|_| {
                let mut b = MarketBar::from_trade(p, 1.0);
                for _ in 0..5 {
                    p *= 1.0 + 0.002 * rng.normal();
                    b.add_trade(p, 2.0);
                }
                b
            })
            .collect()
    }

    #[test]
    fn closed_steps_produce_states_regions_and_transitions() {
        let mut pl = TimescalePipeline::baseline(Timescale::Minute, &PipelineConfig::default());
        let all = bars(40, 1);
        let mut n_states = 0;
        let mut n_created = 0;
        let mut n_trans = 0;
        for end in 2..=all.len() {
            let out = pl
                .step(&StepInput {
                    instrument_id: "inst",
                    as_of_ms: end as i64 * 60_000,
                    bars: &all[..end],
                    bar_closed: true,
                })
                .unwrap();
            let s = out.state.unwrap();
            assert_eq!(s.timescale, "1m");
            assert_eq!(s.experts.len(), 3);
            assert!(s.experts.iter().all(|e| !e.region_id.is_empty() && e.region_id.starts_with("1m-e")));
            assert!((s.router_weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            n_states += 1;
            n_created += out.region_events.iter().filter(|e| e.kind == region_event::Kind::Created as i32).count();
            n_trans += out.transitions.len();
            for t in &out.transitions {
                assert_eq!(t.sampler_id, "conjugate");
                let total: f64 = t.to.iter().map(|p| p.mean).sum();
                assert!((total - 1.0).abs() < 1e-9);
                assert!(t.to.iter().all(|p| p.lower <= p.mean && p.mean <= p.upper));
            }
        }
        assert_eq!(n_states, 39);
        assert!(n_created >= 3, "at least one region per expert");
        assert_eq!(n_trans, 38 * 3, "one transition row per expert per closed step after the first");
        assert!(pl.live_regions() >= 3 && pl.transition_rows() >= 3);
    }

    #[test]
    fn swapping_the_encoder_seeds_regions_and_transfers_the_kernel() {
        let cfg = PipelineConfig::default();
        let mut pl = TimescalePipeline::baseline(Timescale::Minute, &cfg);
        let all = bars(30, 4);
        for end in 2..=all.len() {
            pl.step(&StepInput {
                instrument_id: "i",
                as_of_ms: end as i64 * 60_000,
                bars: &all[..end],
                bar_closed: true,
            })
            .unwrap();
        }
        assert!(pl.transition_rows() > 0);
        let stats = StatisticalEncoder::fit(Timescale::Minute, &all, &fa_encoders::FitConfig::default()).unwrap();
        let enc = StatisticalEncoder::with_stats(Timescale::Minute, "snap-x", stats);
        // every live expert-0 region of the baseline is carried into the new snapshot (score 0.9)
        let seeds: Vec<fa_patterns::FittedRegion> = pl.regions[0]
            .regions()
            .iter()
            .enumerate()
            .map(|(n, r)| fa_patterns::FittedRegion {
                region_id: format!("snap-x-e0-r{n:03}"),
                lineage_id: r.lineage_id.clone(),
                expert_index: 0,
                centroid: r.centroid.clone(),
                radius: 1.0,
                mass: 1.0,
                parent_region_id: Some(r.id.clone()),
                score: 0.9,
                ancestors: vec![fa_patterns::Ancestor {
                    region_id: r.id.clone(),
                    lineage_id: r.lineage_id.clone(),
                    overlap: 0.9,
                    kind: fa_patterns::DescentKind::Carry,
                }],
            })
            .collect();
        let n_seeds = seeds.len();
        let parents: Vec<String> = seeds.iter().filter_map(|r| r.parent_region_id.clone()).collect();
        let map = RegionMap {
            snapshot_id: "snap-x".into(),
            parent_snapshot_id: Some("stat-v1-1m".into()),
            probe_set_version: "p".into(),
            n_probes: 1,
            carry_threshold: 0.5,
            experts: vec![fa_patterns::ExpertRegionMap {
                expert_index: 0,
                regions: seeds,
                parent_regions: n_seeds,
                matched: n_seeds,
                carried_mass: 1.0,
                dead_parents: vec![],
                split_parents: vec![],
                regime_break: false,
            }],
            note: None,
        };
        let old = pl.transitions[0].graph().unwrap().clone();
        let old_live_mass: f64 = parents.iter().filter_map(|p| old.row(p)).flat_map(|r| r.counts.values()).sum::<f64>();
        let old_rows = parents.iter().filter(|p| old.row(p).is_some()).count();
        assert!(old_live_mass > 0.0 && old_rows > 0);
        let report = pl.swap_encoder(&cfg, Box::new(enc), Some(&map));
        let events = &report.events;
        assert_eq!(pl.snapshot_id(), "snap-x");
        assert_eq!(events.len(), n_seeds);
        assert_eq!(events[0].region_id, "snap-x-e0-r000");
        assert_eq!(events[0].lineage_id, "1m-e0-r000");
        assert_eq!(events[0].space.as_ref().unwrap().snapshot_id, "snap-x");
        assert_eq!(pl.live_regions(), n_seeds);
        // kernel transfer: every live row is carried with score (0.9) · carry_discount (0.5);
        // successors that were retired before the swap have no destination and are dropped
        assert_eq!(report.rows_carried, old_rows);
        assert!(report.carried > 0.0 && report.carried <= old_live_mass * 0.45 + 1e-9, "{report:?}");
        let g = pl.transitions[0].graph().unwrap();
        let new_mass: f64 = g.regions().filter_map(|r| g.row(r)).flat_map(|r| r.counts.values()).sum();
        assert!((new_mass - report.carried).abs() < 1e-9);
        for from in g.regions() {
            assert!(from.starts_with("snap-x-e0-"), "{from}");
            let total: f64 = g.posterior_mean(from).iter().map(|x| x.1).sum();
            assert!((total - 1.0).abs() < 1e-9);
        }
        assert_eq!(pl.transition_rows(), old_rows, "experts 1 and 2 have no map and restart from the prior");
        assert!(pl.transitions[1].graph().unwrap().regions().count() == 0);
        let out =
            pl.step(&StepInput { instrument_id: "i", as_of_ms: 31 * 60_000, bars: &all, bar_closed: true }).unwrap();
        assert!(out.transitions.is_empty(), "chain position cleared: first closed bar observes nothing");
        let st = out.state.unwrap();
        assert_eq!(st.snapshot_id, "snap-x");
        assert!(st.experts.iter().all(|e| e.region_id.starts_with("snap-x-e")), "{:?}", st.experts);
    }

    #[test]
    fn swap_without_a_map_restarts_the_posteriors() {
        let cfg = PipelineConfig::default();
        let mut pl = TimescalePipeline::baseline(Timescale::Minute, &cfg);
        let all = bars(20, 9);
        for end in 2..=all.len() {
            pl.step(&StepInput {
                instrument_id: "i",
                as_of_ms: end as i64 * 60_000,
                bars: &all[..end],
                bar_closed: true,
            })
            .unwrap();
        }
        assert!(pl.transition_rows() > 0);
        let stats = StatisticalEncoder::fit(Timescale::Minute, &all, &fa_encoders::FitConfig::default()).unwrap();
        let report =
            pl.swap_encoder(&cfg, Box::new(StatisticalEncoder::with_stats(Timescale::Minute, "snap-y", stats)), None);
        assert!(report.events.is_empty() && report.carried == 0.0);
        assert_eq!(pl.transition_rows(), 0);
    }

    #[test]
    fn set_encoder_keeps_regions_and_refuses_other_snapshots() {
        let cfg = PipelineConfig::default();
        let mut pl = TimescalePipeline::baseline(Timescale::Minute, &cfg);
        let all = bars(20, 3);
        for end in 2..=all.len() {
            pl.step(&StepInput {
                instrument_id: "i",
                as_of_ms: end as i64 * 60_000,
                bars: &all[..end],
                bar_closed: true,
            })
            .unwrap();
        }
        let (regions, rows) = (pl.live_regions(), pl.transition_rows());
        let stats = StatisticalEncoder::fit(Timescale::Minute, &all, &fa_encoders::FitConfig::default()).unwrap();
        let other = StatisticalEncoder::with_stats(Timescale::Minute, "snap-z", stats.clone());
        assert!(pl.set_encoder(Box::new(other)).is_err());
        let same = StatisticalEncoder::with_stats(Timescale::Minute, pl.snapshot_id().to_owned(), stats);
        pl.set_encoder(Box::new(same)).unwrap();
        assert_eq!((pl.live_regions(), pl.transition_rows()), (regions, rows));
        let out =
            pl.step(&StepInput { instrument_id: "i", as_of_ms: 21 * 60_000, bars: &all, bar_closed: true }).unwrap();
        assert!(!out.transitions.is_empty(), "chain position kept across an adapter");
    }

    #[test]
    fn previews_do_not_observe_transitions() {
        let mut pl = TimescalePipeline::baseline(Timescale::Hour, &PipelineConfig::default());
        let all = bars(10, 2);
        for end in 1..=all.len() {
            let out =
                pl.step(&StepInput { instrument_id: "i", as_of_ms: 0, bars: &all[..end], bar_closed: false }).unwrap();
            assert!(out.transitions.is_empty());
            assert!(out.state.is_some());
        }
        assert_eq!(pl.transition_rows(), 0);
    }

    #[test]
    fn steps_publish_memberships_lifecycle_and_kernel_summaries() {
        let cfg = PipelineConfig {
            kernel: KernelCfg { kernel_every_ms: 10 * 60_000, smc_particles: 16, ..KernelCfg::default() },
            ..PipelineConfig::default()
        };
        let mut pl = TimescalePipeline::baseline(Timescale::Minute, &cfg);
        let all = bars(60, 5);
        let mut kernels = Vec::new();
        let mut born = 0;
        for end in 2..=all.len() {
            let out = pl
                .step(&StepInput {
                    instrument_id: "i",
                    as_of_ms: end as i64 * 60_000,
                    bars: &all[..end],
                    bar_closed: true,
                })
                .unwrap();
            let m = out.membership.expect("membership every step");
            assert_eq!(m.experts.len(), 3);
            for e in &m.experts {
                let w: f32 = e.weights.iter().map(|w| w.weight).sum();
                assert!((w - 1.0).abs() < 1e-4, "{e:?}");
                assert!(e.novelty >= 0.0);
            }
            born += out
                .lifecycle
                .iter()
                .filter(|l| l.kind == fa_proto::v1::region_lifecycle_event::Kind::Born as i32)
                .count();
            assert!(out.lifecycle.iter().all(|l| l.as_of.is_some() && l.space.is_some()));
            kernels.extend(out.kernels);
        }
        assert_eq!(born, out_created(&pl));
        // first closed step emits for every expert, then every 10 bars: 59 closed steps → 6 rounds
        assert_eq!(kernels.len(), 3 * 6, "{}", kernels.len());
        let last = kernels.last().unwrap();
        assert_eq!(last.samplers.len(), 2);
        assert!(last.samplers.iter().any(|s| s.serving));
        assert!(!last.stationary.is_empty());
        assert!(last.observation_mass > 0.0);
    }

    fn out_created(pl: &TimescalePipeline) -> usize {
        // births = live regions + those merged/retired since; with this data nothing retired
        pl.live_regions()
    }

    #[test]
    fn persisted_state_restores_regions_and_kernel() {
        let cfg = PipelineConfig::default();
        let mut pl = TimescalePipeline::baseline(Timescale::Minute, &cfg);
        let all = bars(30, 6);
        for end in 2..=all.len() {
            pl.step(&StepInput {
                instrument_id: "i",
                as_of_ms: end as i64 * 60_000,
                bars: &all[..end],
                bar_closed: true,
            })
            .unwrap();
        }
        let rows = pl.persist_rows();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].family, "stat_baseline");
        let mut fresh = TimescalePipeline::baseline(Timescale::Minute, &cfg);
        for (k, r) in rows.iter().enumerate() {
            assert!(fresh.restore_expert(k, r.region_state.as_ref().unwrap(), r.kernel_state.as_ref().unwrap()));
        }
        assert_eq!(fresh.live_regions(), pl.live_regions());
        assert_eq!(fresh.transition_rows(), pl.transition_rows());
        let g0 = pl.transitions[0].graph().unwrap();
        let g1 = fresh.transitions[0].graph().unwrap();
        assert!((g0.total_mass() - g1.total_mass()).abs() < 1e-9);
        // a state of another snapshot is refused
        let stats = StatisticalEncoder::fit(Timescale::Minute, &all, &fa_encoders::FitConfig::default()).unwrap();
        let mut other = TimescalePipeline::with_encoder(
            Timescale::Minute,
            &cfg,
            Box::new(StatisticalEncoder::with_stats(Timescale::Minute, "snap-other", stats)),
            None,
        );
        assert!(!other.restore_expert(0, rows[0].region_state.as_ref().unwrap(), "{}"));
    }

    #[test]
    fn a_regime_break_restarts_the_posteriors_and_reports_lifecycle() {
        let cfg = PipelineConfig::default();
        let mut pl = TimescalePipeline::baseline(Timescale::Minute, &cfg);
        let all = bars(30, 8);
        for end in 2..=all.len() {
            pl.step(&StepInput {
                instrument_id: "i",
                as_of_ms: end as i64 * 60_000,
                bars: &all[..end],
                bar_closed: true,
            })
            .unwrap();
        }
        let old_id = pl.regions[0].regions()[0].id.clone();
        let stats = StatisticalEncoder::fit(Timescale::Minute, &all, &fa_encoders::FitConfig::default()).unwrap();
        let map = RegionMap {
            snapshot_id: "snap-b".into(),
            parent_snapshot_id: Some(pl.snapshot_id().to_owned()),
            probe_set_version: "p".into(),
            n_probes: 10,
            carry_threshold: 0.5,
            experts: vec![fa_patterns::ExpertRegionMap {
                expert_index: 0,
                regions: vec![fa_patterns::FittedRegion {
                    region_id: "snap-b-e0-r000".into(),
                    lineage_id: "snap-b-e0-r000".into(),
                    expert_index: 0,
                    centroid: vec![0.1; 8],
                    radius: 1.0,
                    mass: 1.0,
                    parent_region_id: None,
                    score: 0.0,
                    ancestors: vec![fa_patterns::Ancestor {
                        region_id: old_id.clone(),
                        lineage_id: old_id.clone(),
                        overlap: 0.3,
                        kind: fa_patterns::DescentKind::Partial,
                    }],
                }],
                parent_regions: 1,
                matched: 0,
                carried_mass: 0.0,
                dead_parents: vec!["gone".into()],
                split_parents: vec![],
                regime_break: true,
            }],
            note: None,
        };
        let report = pl.swap_encoder(
            &cfg,
            Box::new(StatisticalEncoder::with_stats(Timescale::Minute, "snap-b", stats)),
            Some(&map),
        );
        assert!(report.regime_break);
        assert_eq!(report.carried, 0.0);
        assert_eq!(pl.transition_rows(), 0, "posteriors restarted from the prior");
        use fa_proto::v1::region_lifecycle_event::Kind;
        let kinds: Vec<i32> = report.lifecycle.iter().map(|l| l.kind).collect();
        assert_eq!(kinds, vec![Kind::Born as i32, Kind::Died as i32]);
        assert_eq!(report.lifecycle[0].ancestors[0].kind, "partial");
        assert!(report.lifecycle.iter().all(|l| l.regime_break));
        assert_eq!(
            report.lifecycle[1].space.as_ref().unwrap().snapshot_id,
            "stat-v1-1m",
            "dead parent in the old space"
        );
        let seeded = pl.seeded_lifecycle(Some(&map), 0);
        assert_eq!(seeded.len(), 2);
    }
}
