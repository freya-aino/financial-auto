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
use fa_patterns::{OnlineRegionCfg, OnlineRegionDiscovery, OnlineRegionSet, RegionLifecycle};
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
    pub fn baseline(ts: Timescale, cfg: &PipelineConfig) -> Self {
        let encoder = StatisticalEncoder::new(ts);
        let n_experts = encoder.sizing().max_experts as usize;
        let snapshot = encoder.snapshot_id().to_owned();
        let regions = (0..n_experts)
            .map(|k| {
                Box::new(OnlineRegionSet::new(cfg.regions, snapshot.clone(), k as u32, format!("{ts}-e{k}")))
                    as Box<dyn OnlineRegionDiscovery>
            })
            .collect();
        let transitions = (0..n_experts)
            .map(|k| {
                Box::new(DirichletTransitions::new(
                    cfg.alpha_0,
                    gamma_forget(ts),
                    cfg.ci_level,
                    cfg.mc_draws,
                    cfg.seed ^ ((k as u64 + 1) * 0x9E37_79B9),
                )) as Box<dyn TransitionModel>
            })
            .collect();
        Self {
            ts,
            encoder: Box::new(encoder),
            router: Box::new(EvidenceSoftmaxRouter::default()),
            regions,
            transitions,
            last_closed: HashMap::new(),
        }
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
