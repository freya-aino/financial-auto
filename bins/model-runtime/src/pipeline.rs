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
use fa_mcmc::{MarkovGraph, RegionCarry};
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
    /// Kernel transfer on swap (spec 33 §6.6 `carry_discount`): carried observation mass is
    /// multiplied by `score · carry_discount` so transferred evidence counts less than fresh.
    pub carry_discount: f64,
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
    /// The underlying Dirichlet graph, when the model has one (used for kernel transfer).
    fn graph(&self) -> Option<&MarkovGraph> {
        None
    }
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
        Self::from_graph(MarkovGraph::new(alpha_0, decay), ci_level, draws, seed)
    }

    /// Wraps a graph produced by [`MarkovGraph::transfer`].
    pub fn from_graph(graph: MarkovGraph, ci_level: f64, draws: usize, seed: u64) -> Self {
        Self { graph, rng: Rng::seed(seed), ci_level, draws }
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

    fn graph(&self) -> Option<&MarkovGraph> {
        Some(&self.graph)
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
        self.encoder = encoder;
        let (regions, mut transitions) = Self::stages(self.ts, cfg, self.encoder.snapshot_id(), n_experts, region_map);
        let mut report = SwapReport::default();
        if let Some(map) = region_map {
            for (k, model) in transitions.iter_mut().enumerate() {
                let Some(e) = map.expert(k as u32) else { continue };
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
                let t = MarkovGraph::transfer(old, &carries, &support, cfg.carry_discount);
                report.carried += t.carried;
                report.dropped += t.dropped;
                report.rows_carried += t.rows_carried;
                *model = Box::new(DirichletTransitions::from_graph(
                    t.graph,
                    cfg.ci_level,
                    cfg.mc_draws,
                    cfg.seed ^ ((k as u64 + 1) * 0x9E37_79B9),
                ));
            }
        }
        self.regions = regions;
        self.transitions = transitions;
        self.last_closed.clear();
        report.events = self.seeded_region_events();
        report
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
}
