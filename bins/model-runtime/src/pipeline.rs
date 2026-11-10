//! Per-timescale latent pipeline: encoder → router → regions → transitions (specs 30–33).
//!
//! The stages are trait objects so the baselines can be swapped for learned models:
//! [`EncoderFamily`] (fa-encoders), [`Router`] (fa-expert-space),
//! [`OnlineRegionDiscovery`] (fa-patterns, one set per expert) and [`TransitionModel`]
//! (here, backed by `fa_mcmc::MarkovGraph`, one graph per expert). The pipeline is pure
//! with respect to the bus: `step` returns the protobuf messages to publish.

use std::collections::HashMap;

use fa_core::{Timescale, rng::Rng};
use fa_encoders::{EncoderFamily, EncoderInput, MarketBar, StatisticalEncoder};
use fa_expert_space::{EvidenceSoftmaxRouter, Router};
use fa_mcmc::MarkovGraph;
use fa_patterns::{OnlineRegionCfg, OnlineRegionDiscovery, OnlineRegionSet, RegionLifecycle, RegionMap};
use fa_proto::{
    Timestamp,
    v1::{ExpertLatent, ExpertSpaceRef, LatentState, RegionEvent, TransitionProb, TransitionUpdate, region_event},
};

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
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self { regions: OnlineRegionCfg::default(), alpha_0: 1.0, ci_level: 0.9, mc_draws: 256, seed: 7 }
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
}

/// Dirichlet–multinomial rows with forgetting, intervals by Monte Carlo (`fa_mcmc::ConjugateSampler` family).
pub struct DirichletTransitions {
    graph: MarkovGraph,
    rng: Rng,
    ci_level: f64,
    draws: usize,
}

impl DirichletTransitions {
    pub fn new(alpha_0: f64, decay: f64, ci_level: f64, draws: usize, seed: u64) -> Self {
        Self { graph: MarkovGraph::new(alpha_0, decay), rng: Rng::seed(seed), ci_level, draws }
    }
}

impl TransitionModel for DirichletTransitions {
    fn sampler_id(&self) -> &str {
        "conjugate"
    }

    fn register_region(&mut self, id: &str) {
        self.graph.register_region(id);
    }

    fn retire_region(&mut self, id: &str) {
        self.graph.retire_region(id);
    }

    fn observe(&mut self, from: &str, to: &str) {
        self.graph.observe(from, to);
    }

    fn summarize(&mut self, from: &str) -> Vec<TransitionProb> {
        self.graph
            .summarize(from, self.ci_level, self.draws, &mut self.rng)
            .into_iter()
            .map(|(region_id, s)| TransitionProb { region_id, mean: s.mean, lower: s.lower, upper: s.upper })
            .collect()
    }

    fn n_rows(&self) -> usize {
        self.graph.regions().count()
    }
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
        }
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
                let mut t = DirichletTransitions::new(
                    cfg.alpha_0,
                    gamma_forget(ts),
                    cfg.ci_level,
                    cfg.mc_draws,
                    cfg.seed ^ ((k as u64 + 1) * 0x9E37_79B9),
                );
                if let Some(e) = region_map.and_then(|m| m.expert(k as u32)) {
                    for r in &e.regions {
                        t.register_region(&r.region_id);
                    }
                }
                Box::new(t) as Box<dyn TransitionModel>
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

    /// Hot-swaps the encoder (spec 36 §6.4 promotion / rollback) without restarting: the region
    /// sets are rebuilt from the new snapshot's `region_map` (seeded) or start empty, the
    /// transition posteriors restart (kernel transfer, spec 33 §6.6, is not implemented) and the
    /// per-instrument chain positions are cleared. Returns `Created` events for every seeded
    /// region so the bus and Helix learn the new snapshot's regions.
    pub fn swap_encoder(
        &mut self,
        cfg: &PipelineConfig,
        encoder: Box<dyn EncoderFamily>,
        region_map: Option<&RegionMap>,
    ) -> Vec<RegionEvent> {
        let n_experts = encoder.sizing().max_experts as usize;
        self.encoder = encoder;
        let (regions, transitions) = Self::stages(self.ts, cfg, self.encoder.snapshot_id(), n_experts, region_map);
        self.regions = regions;
        self.transitions = transitions;
        self.last_closed.clear();
        self.seeded_region_events()
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
        Some(ExpertSpaceRef {
            snapshot_id: self.snapshot_id().to_owned(),
            timescale: self.ts.to_string(),
            expert_index,
        })
    }

    pub fn step(&mut self, input: &StepInput<'_>) -> Result<StepOutput, fa_encoders::EncoderError> {
        let mut view = self.encoder.encode(&EncoderInput::from_bars(input.as_of_ms, input.bars))?;
        view.router_weights = self.router.route(&view.experts);

        let mut out = StepOutput::default();
        let mut current = Vec::with_capacity(view.experts.len());
        let mut experts = Vec::with_capacity(view.experts.len());
        for e in &view.experts {
            let k = e.expert_index as usize;
            let a = self.regions[k].observe(&e.mean);
            for ev in a.events {
                match &ev {
                    RegionLifecycle::Created { region } => self.transitions[k].register_region(&region.id),
                    RegionLifecycle::Retired { region_id, .. } => self.transitions[k].retire_region(region_id),
                    RegionLifecycle::Merged { .. } => {}
                }
                out.region_events.push(self.region_event(e.expert_index, ev));
            }
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
        }

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

    fn region_event(&self, expert_index: u32, ev: RegionLifecycle) -> RegionEvent {
        let (kind, region_id, lineage_id, related) = match ev {
            RegionLifecycle::Created { region } => (region_event::Kind::Created, region.id, region.lineage_id, vec![]),
            RegionLifecycle::Merged { survivor, lineage_id, absorbed } => {
                (region_event::Kind::Merged, survivor, lineage_id, vec![absorbed])
            }
            RegionLifecycle::Retired { region_id, lineage_id, merged_into } => {
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
    fn swapping_the_encoder_seeds_regions_from_the_map_and_restarts_the_chain() {
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
        let seed = fa_patterns::FittedRegion {
            region_id: "snap-x-e0-r000".into(),
            lineage_id: "1m-e0-r000".into(),
            expert_index: 0,
            centroid: vec![0.0; 8],
            radius: 1.0,
            mass: 1.0,
            parent_region_id: Some("1m-e0-r000".into()),
            score: 0.9,
        };
        let map = RegionMap {
            snapshot_id: "snap-x".into(),
            parent_snapshot_id: Some("stat-v1-1m".into()),
            probe_set_version: "p".into(),
            n_probes: 1,
            carry_threshold: 0.5,
            experts: vec![fa_patterns::ExpertRegionMap {
                expert_index: 0,
                regions: vec![seed],
                parent_regions: 1,
                matched: 1,
                carried_mass: 1.0,
            }],
            note: None,
        };
        let events = pl.swap_encoder(&cfg, Box::new(enc), Some(&map));
        assert_eq!(pl.snapshot_id(), "snap-x");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].region_id, "snap-x-e0-r000");
        assert_eq!(events[0].lineage_id, "1m-e0-r000");
        assert_eq!(events[0].space.as_ref().unwrap().snapshot_id, "snap-x");
        assert_eq!(pl.live_regions(), 1);
        assert_eq!(pl.transition_rows(), 0, "old posteriors dropped; nothing observed yet");
        let out =
            pl.step(&StepInput { instrument_id: "i", as_of_ms: 31 * 60_000, bars: &all, bar_closed: true }).unwrap();
        assert!(out.transitions.is_empty(), "chain position cleared: first closed bar observes nothing");
        let st = out.state.unwrap();
        assert_eq!(st.snapshot_id, "snap-x");
        assert!(st.experts.iter().all(|e| e.region_id.starts_with("snap-x-e")), "{:?}", st.experts);
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
}
