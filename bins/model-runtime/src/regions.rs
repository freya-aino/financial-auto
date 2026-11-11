//! Region and transition stages beyond membership (specs 30, 33): the arena-backed transition
//! model, region lifecycle messages, graph-level kernel summaries and the cross-timescale /
//! cross-asset trackers.
//!
//! - [`ArenaTransitions`]: the Dirichlet kernel with the geometric prior, served by the
//!   calibration arena (`fa_mcmc::calibration::Arena`: `conjugate` and `smc` scored
//!   prequentially, one serves).
//! - [`lifecycle_online`] / [`lifecycle_map`]: `RegionLifecycleEvent`s for online births, splits,
//!   merges, deaths and for a loaded snapshot's alignment (carry / split / merge / birth, dead
//!   parents) — published on `latent.lifecycle.<ts>` and persisted as `DescendsFrom` edges.
//! - [`kernel_update`]: the `KernelUpdate` of one expert space (stationary distribution with
//!   intervals, relaxation time, sampler standings, drift), published on
//!   `latent.mcmc.<ts>.<expert>`.
//! - [`CrossScale`]: `REFINES` hierarchy between adjacent timescales (per expert index) and the
//!   cross-asset priors per timescale.

use std::collections::{BTreeMap, HashMap};

use fa_core::{Timescale, rng::Rng};
use fa_mcmc::{
    MarkovGraph, ProbSummary,
    calibration::{Arena, ArenaCfg, SamplerStanding},
    cross_asset::{CrossAssetCfg, CrossAssetPriors as CrossTracker},
    inference::Posterior,
    kernel::{ConjugateExact, KernelCtx, KernelSampler, SmcParticle},
    refit::RefitOutcome,
    rich::{HierAsset, LogitCov, SecondOrder, SemiMarkov},
};
use fa_patterns::{
    DescentKind, DriftReport, ExpertRegionMap, HierarchyCfg, HierarchyTracker, RegionLifecycle, RegionStat,
};
use fa_proto::v1::{
    CrossAssetPair, CrossAssetPriors, ExpertSpaceRef, HierEdge, KernelUpdate, LatentState, RegionAncestor, RegionDist,
    RegionHierarchy, RegionLifecycleEvent, SamplerScore, region_lifecycle_event::Kind as LifeKind,
};

use crate::pipeline::{TransitionModel, ms_to_ts};

/// Tunables of the kernel stage (spec 33 §7).
#[derive(Debug, Clone, Copy)]
pub struct KernelCfg {
    /// Particles per source row of the SMC sampler (`smc_particles`; 0 disables SMC).
    pub smc_particles: usize,
    pub arena: ArenaCfg,
    /// Event-time cadence of `KernelUpdate`s per expert space (0 disables them).
    pub kernel_every_ms: i64,
    /// Posterior kernel draws behind a `KernelUpdate`.
    pub kernel_draws: usize,
    /// `drift_refit_kl`.
    pub drift_refit_kl: f64,
    /// Observations of a set before a baseline (no fitted masses) captures its drift reference.
    pub drift_warmup: u64,
    /// Geometric structural prior (spec 33 §6.1).
    pub geo_prior: bool,
    /// The richer kernels of spec 33 §6.2 (`second_order`, `semi_markov`, `hier_asset`,
    /// `logit_cov`) join the arena.
    pub rich_kernels: bool,
    /// `batch_refit_interval` in event time (0 disables `jobs.mcmc.refit`).
    pub refit_every_ms: i64,
    /// Transitions in a window before its first refit (and a quarter of it fresh for later ones).
    pub refit_min_obs: usize,
    /// Observation window per expert space.
    pub refit_window: usize,
}

impl Default for KernelCfg {
    fn default() -> Self {
        Self {
            smc_particles: 64,
            arena: ArenaCfg::default(),
            kernel_every_ms: 60_000,
            kernel_draws: 32,
            drift_refit_kl: 0.1,
            drift_warmup: 2_000,
            geo_prior: true,
            rich_kernels: true,
            refit_every_ms: 3_600_000,
            refit_min_obs: 500,
            refit_window: 20_000,
        }
    }
}

/// Dirichlet rows with forgetting and the geometric prior; `conjugate`, `smc` and the richer
/// kernels (`second_order`, `semi_markov`, `hier_asset` with Gibbs refits, `logit_cov` with NUTS
/// refits) compete in a calibration arena and the serving one publishes (spec 33 §6.2, §6.4).
pub struct ArenaTransitions {
    arena: Arena,
    ci_level: f64,
}

/// Key of the sampler states inside the persisted kernel JSON (ignored by `MarkovGraph` readers).
const ARENA_STATES: &str = "arena_states";

impl ArenaTransitions {
    pub fn from_graph(graph: MarkovGraph, ci_level: f64, draws: usize, kcfg: &KernelCfg, seed: u64) -> Self {
        let mut samplers: Vec<Box<dyn KernelSampler>> = vec![Box::new(ConjugateExact::new(graph.clone(), draws, seed))];
        if kcfg.smc_particles > 0 {
            samplers.push(Box::new(SmcParticle::new(graph.clone(), kcfg.smc_particles, seed ^ 0x5EED)));
        }
        if kcfg.rich_kernels {
            let d = draws.min(64);
            samplers.push(Box::new(SecondOrder::new(graph.clone(), 4.0, d, seed ^ 0x20D)));
            samplers.push(Box::new(SemiMarkov::new(graph.clone(), 2.0, d, seed ^ 0x5E31)));
            samplers.push(Box::new(HierAsset::new(graph.clone(), 10.0, d, seed ^ 0x41E2)));
            samplers.push(Box::new(LogitCov::new(graph, 32)));
        }
        Self { arena: Arena::new(samplers, kcfg.arena, seed ^ 0xA7E4A), ci_level }
    }
}

fn probs(v: Vec<(String, ProbSummary)>) -> Vec<fa_proto::v1::TransitionProb> {
    v.into_iter()
        .map(|(region_id, ProbSummary { mean, lower, upper })| fa_proto::v1::TransitionProb {
            region_id,
            mean,
            lower,
            upper,
        })
        .collect()
}

impl TransitionModel for ArenaTransitions {
    fn sampler_id(&self) -> &str {
        self.arena.serving().name()
    }
    fn register_region(&mut self, id: &str) {
        self.arena.register_region(id);
    }
    fn retire_region(&mut self, id: &str) {
        self.arena.retire_region(id);
    }
    fn observe(&mut self, from: &str, to: &str) {
        self.arena.observe(from, to);
    }
    fn summarize(&mut self, from: &str) -> Vec<fa_proto::v1::TransitionProb> {
        let level = self.ci_level;
        probs(self.arena.serving_mut().summarize(from, level))
    }
    fn observe_ctx(&mut self, from: &str, to: &str, ctx: &KernelCtx) {
        self.arena.observe_ctx(from, to, ctx);
    }
    fn summarize_ctx(&mut self, from: &str, ctx: &KernelCtx) -> Vec<fa_proto::v1::TransitionProb> {
        let level = self.ci_level;
        probs(self.arena.serving_mut().summarize_ctx(from, level, ctx))
    }
    fn kernel_state(&self) -> Option<String> {
        let mut v = serde_json::to_value(self.arena.serving().graph()).ok()?;
        let states = self.arena.export_states();
        if let Some(obj) = v.as_object_mut()
            && !states.is_empty()
        {
            obj.insert(ARENA_STATES.into(), serde_json::to_value(states).ok()?);
        }
        serde_json::to_string(&v).ok()
    }
    fn restore_kernel(&mut self, state: &str) {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(state) else { return };
        let Ok(g) = serde_json::from_value::<MarkovGraph>(v.clone()) else { return };
        self.arena.replace_graph(&g);
        if let Some(states) = v.get(ARENA_STATES).and_then(|s| serde_json::from_value(s.clone()).ok()) {
            self.arena.import_states(&states);
        }
    }
    fn apply_refit(&mut self, o: &RefitOutcome) -> Vec<String> {
        self.arena.apply_refit(o)
    }
    fn refit_refs(&self) -> Vec<String> {
        self.arena.refit_refs()
    }
    fn n_rows(&self) -> usize {
        self.arena.serving().graph().regions().count()
    }
    fn graph(&self) -> Option<&MarkovGraph> {
        Some(self.arena.serving().graph())
    }
    fn set_centroid(&mut self, id: &str, centroid: &[f32]) {
        self.arena.set_centroid(id, centroid);
    }
    fn standings(&self) -> Vec<SamplerStanding> {
        self.arena.standings()
    }
    fn replace_graph(&mut self, g: MarkovGraph) {
        self.arena.replace_graph(&g);
    }
}

fn ancestor(region_id: &str, lineage_id: &str, overlap: f64, kind: DescentKind) -> RegionAncestor {
    RegionAncestor {
        region_id: region_id.to_owned(),
        lineage_id: lineage_id.to_owned(),
        overlap,
        kind: kind.as_str().to_owned(),
    }
}

/// Lifecycle message of one online event. `stat` looks up the statistics of a live region.
pub fn lifecycle_online(
    space: &ExpertSpaceRef,
    as_of_ms: i64,
    ev: &RegionLifecycle,
    stat: impl Fn(&str) -> Option<RegionStat>,
) -> RegionLifecycleEvent {
    let mut out =
        RegionLifecycleEvent { space: Some(space.clone()), as_of: Some(ms_to_ts(as_of_ms)), ..Default::default() };
    let fill = |out: &mut RegionLifecycleEvent, id: &str| {
        if let Some(s) = stat(id) {
            out.centroid = s.centroid;
            out.mass = s.mass;
            out.occupancy = s.occupancy;
        }
    };
    match ev {
        RegionLifecycle::Created { region } => {
            out.kind = LifeKind::Born as i32;
            out.region_id = region.id.clone();
            out.lineage_id = region.lineage_id.clone();
            out.reason = "online".into();
            out.centroid = region.centroid.clone();
            out.mass = 1.0;
        }
        RegionLifecycle::Split { parent, parent_lineage_id, child, masses } => {
            out.kind = LifeKind::Split as i32;
            out.region_id = child.id.clone();
            out.lineage_id = child.lineage_id.clone();
            out.reason = "online".into();
            let share = masses.1 / (masses.0 + masses.1).max(1e-12);
            out.ancestors.push(ancestor(parent, parent_lineage_id, share, DescentKind::Split));
            fill(&mut out, &child.id);
            if out.centroid.is_empty() {
                out.centroid = child.centroid.clone();
            }
        }
        RegionLifecycle::Merged { survivor, lineage_id, absorbed } => {
            out.kind = LifeKind::Merged as i32;
            out.region_id = survivor.clone();
            out.lineage_id = lineage_id.clone();
            out.reason = "online".into();
            out.ancestors.push(ancestor(absorbed, "", 0.0, DescentKind::Merge));
            fill(&mut out, survivor);
        }
        RegionLifecycle::Retired { region_id, lineage_id, merged_into, reason } => {
            out.kind = LifeKind::Died as i32;
            out.region_id = region_id.clone();
            out.lineage_id = lineage_id.clone();
            out.reason = reason.as_str().into();
            out.related_region_id = merged_into.clone();
        }
    }
    out
}

/// Lifecycle messages of a snapshot's alignment for one expert: every new region (carried,
/// split, merged or born, with its ancestors) and every dead parent region (in the parent's
/// space).
pub fn lifecycle_map(
    space: &ExpertSpaceRef,
    parent_space: Option<&ExpertSpaceRef>,
    as_of_ms: i64,
    e: &ExpertRegionMap,
) -> Vec<RegionLifecycleEvent> {
    let mut out: Vec<RegionLifecycleEvent> = e
        .regions
        .iter()
        .map(|r| {
            let kind = if r.parent_region_id.is_some() {
                LifeKind::Carried
            } else if r.ancestors.iter().any(|a| a.kind == DescentKind::Split) {
                LifeKind::Split
            } else if r.ancestors.iter().any(|a| a.kind == DescentKind::Merge) {
                LifeKind::Merged
            } else {
                LifeKind::Born
            };
            RegionLifecycleEvent {
                kind: kind as i32,
                space: Some(space.clone()),
                region_id: r.region_id.clone(),
                lineage_id: r.lineage_id.clone(),
                as_of: Some(ms_to_ts(as_of_ms)),
                ancestors: r
                    .ancestors
                    .iter()
                    .map(|a| ancestor(&a.region_id, &a.lineage_id, a.overlap, a.kind))
                    .collect(),
                reason: "snapshot".into(),
                centroid: r.centroid.clone(),
                mass: r.mass,
                occupancy: r.mass,
                related_region_id: String::new(),
                regime_break: e.regime_break,
            }
        })
        .collect();
    if let Some(ps) = parent_space {
        out.extend(e.dead_parents.iter().map(|id| RegionLifecycleEvent {
            kind: LifeKind::Died as i32,
            space: Some(ps.clone()),
            region_id: id.clone(),
            as_of: Some(ms_to_ts(as_of_ms)),
            reason: "unmatched".into(),
            regime_break: e.regime_break,
            ..Default::default()
        }));
    }
    out
}

/// Stationary entries kept in a `KernelUpdate` (largest mass first).
const STATIONARY_TOP: usize = 32;

/// Graph-level summary of one expert space (spec 33 §4.3 `KernelUpdate`).
pub fn kernel_update(
    space: &ExpertSpaceRef,
    as_of_ms: i64,
    model: &dyn TransitionModel,
    live_regions: usize,
    drift: Option<DriftReport>,
    draws: usize,
    rng: &mut Rng,
) -> Option<KernelUpdate> {
    let g = model.graph()?;
    let samplers = model
        .standings()
        .into_iter()
        .map(|s| SamplerScore {
            sampler_id: s.sampler_id,
            log_score: s.scores.log_score,
            brier: s.scores.brier,
            ece: s.scores.ece,
            pit_ks: s.scores.pit_ks,
            cost_us: s.scores.cost_us,
            evals: s.scores.evals,
            pareto: s.pareto,
            serving: s.serving,
        })
        .collect();
    let (stationary, relax) = if g.regions().next().is_some() {
        let post = Posterior::draw(g, draws.max(4), 0.9, rng);
        let st = post
            .stationary()
            .into_iter()
            .take(STATIONARY_TOP)
            .map(|(region_id, d)| RegionDist { region_id, mean: d.mean, lower: d.lower, upper: d.upper, sd: d.sd })
            .collect();
        (st, Some(post.relaxation(1e4)))
    } else {
        (Vec::new(), None)
    };
    Some(KernelUpdate {
        space: Some(space.clone()),
        as_of: Some(ms_to_ts(as_of_ms)),
        serving_sampler_id: model.sampler_id().to_owned(),
        samplers,
        stationary,
        relaxation_steps: relax.map_or(0.0, |r| r.median),
        relaxation_lower: relax.map_or(0.0, |r| r.lower),
        relaxation_upper: relax.map_or(0.0, |r| r.upper),
        n_rows: g.regions().count() as u64,
        observation_mass: g.total_mass(),
        live_regions: live_regions as u32,
        drift_kl: drift.map_or(0.0, |d| d.kl),
        refit_due: drift.is_some_and(|d| d.refit_due),
        draws: draws as u32,
    })
}

/// Drift-driven partial refit (spec 30 §6.6): a `train.adapter` job is requested when an expert
/// space's occupancy drift stays above `drift_refit_kl` for `sustain` consecutive kernel
/// summaries (a single spike during a market shock does not refit, spec 30 §8), at most once per
/// `cooldown_ms` per timescale.
#[derive(Debug, Default)]
pub struct DriftTrigger {
    pub sustain: u32,
    pub cooldown_ms: i64,
    streak: HashMap<(String, u32), u32>,
    last_fire: HashMap<String, i64>,
}

impl DriftTrigger {
    pub fn new(sustain: u32, cooldown_ms: i64) -> Self {
        Self { sustain: sustain.max(1), cooldown_ms, ..Self::default() }
    }

    /// Feeds one kernel summary; returns the adapter job spec when a refit is due now.
    pub fn observe(&mut self, k: &KernelUpdate, now_ms: i64) -> Option<serde_json::Value> {
        let space = k.space.as_ref()?;
        let key = (space.timescale.clone(), space.expert_index);
        let streak = self.streak.entry(key).or_default();
        *streak = if k.refit_due { *streak + 1 } else { 0 };
        if *streak < self.sustain {
            return None;
        }
        if self.last_fire.get(&space.timescale).is_some_and(|t| now_ms - t < self.cooldown_ms) {
            return None;
        }
        self.last_fire.insert(space.timescale.clone(), now_ms);
        *streak = 0;
        let ts: Timescale = space.timescale.parse().ok()?;
        let hours = (32.0 * ts.duration().as_secs_f64() / 3600.0).max(1.0);
        Some(serde_json::json!({
            "snapshot_id": space.snapshot_id, "timescale": space.timescale, "window_hours": hours,
            "reason": "region_drift", "expert_index": space.expert_index, "drift_kl": k.drift_kl,
        }))
    }
}

/// Hierarchy and cross-asset trackers across the served timescales.
pub struct CrossScale {
    /// Served timescales, finest first.
    ladder: Vec<Timescale>,
    hcfg: HierarchyCfg,
    /// `(fine, coarse, expert)` → tracker with the snapshot pair it was built for.
    hier: BTreeMap<(Timescale, Timescale, u32), (HierarchyTracker, String, String)>,
    /// Latest regions per expert of `(timescale, instrument)` with the snapshot.
    current: HashMap<(Timescale, String), (String, Vec<String>)>,
    last_close: HashMap<(Timescale, String), f64>,
    cross: BTreeMap<Timescale, CrossTracker>,
    ccfg: CrossAssetCfg,
    /// Region-conditional mean next-bar return per (timescale, expert, region): (mean, n).
    region_ret: HashMap<(Timescale, u32, String), (f64, f64)>,
    /// Regions and router weights per (timescale, instrument) at the last closed bar.
    prev_view: HashMap<(Timescale, String), (Vec<String>, Vec<f32>)>,
}

/// Running-mean horizon of the region-conditional returns (then an EWMA).
const REGION_RET_HORIZON: f64 = 200.0;

impl CrossScale {
    pub fn new(mut ladder: Vec<Timescale>, hcfg: HierarchyCfg, ccfg: CrossAssetCfg) -> Self {
        ladder.sort();
        ladder.dedup();
        Self {
            ladder,
            hcfg,
            hier: BTreeMap::new(),
            current: HashMap::new(),
            last_close: HashMap::new(),
            cross: BTreeMap::new(),
            ccfg,
            region_ret: HashMap::new(),
            prev_view: HashMap::new(),
        }
    }

    /// Expert residual of a closed-bar return (spec 33 §6.5): `r − Σ_k w_k · μ_k(region_k)` with
    /// the regions and router weights of the previous close; then the region means learn `r`.
    fn expert_residual(&mut self, ts: Timescale, inst: &str, ret: f64, regions: &[String], weights: &[f32]) -> f64 {
        let key = (ts, inst.to_owned());
        let resid = match self.prev_view.get(&key) {
            Some((prev, w)) => {
                let wsum: f64 = w.iter().map(|x| f64::from(*x)).sum::<f64>().max(1e-12);
                let pred: f64 = prev
                    .iter()
                    .zip(w)
                    .enumerate()
                    .map(|(k, (r, wk))| {
                        let (m, n) = self.region_ret.get(&(ts, k as u32, r.clone())).copied().unwrap_or((0.0, 0.0));
                        // shrink thin regions toward 0 (no view)
                        f64::from(*wk) / wsum * m * n / (n + 5.0)
                    })
                    .sum();
                for (k, r) in prev.iter().enumerate() {
                    let e = self.region_ret.entry((ts, k as u32, r.clone())).or_insert((0.0, 0.0));
                    e.1 = (e.1 + 1.0).min(REGION_RET_HORIZON);
                    e.0 += (ret - e.0) / e.1;
                }
                ret - pred
            }
            None => ret,
        };
        self.prev_view.insert(key, (regions.to_vec(), weights.to_vec()));
        resid
    }

    /// Persistable state of the cross-asset trackers (spec 33 §6.5; restored on start).
    pub fn export_cross(&self) -> serde_json::Value {
        let cross: BTreeMap<String, &CrossTracker> = self.cross.iter().map(|(t, c)| (t.to_string(), c)).collect();
        let closes: Vec<(String, &String, f64)> =
            self.last_close.iter().map(|((t, i), p)| (t.to_string(), i, *p)).collect();
        let means: Vec<(String, u32, &String, f64, f64)> =
            self.region_ret.iter().map(|((t, k, r), (m, n))| (t.to_string(), *k, r, *m, *n)).collect();
        serde_json::json!({ "version": 1, "cross": cross, "last_close": closes, "region_ret": means })
    }

    /// Restores [`CrossScale::export_cross`]; returns the instruments restored per timescale.
    pub fn restore_cross(&mut self, v: &serde_json::Value) -> usize {
        let Some(cross) =
            v.get("cross").and_then(|c| serde_json::from_value::<BTreeMap<String, CrossTracker>>(c.clone()).ok())
        else {
            return 0;
        };
        let mut n = 0;
        for (t, mut c) in cross {
            if let Ok(ts) = t.parse::<Timescale>() {
                c.set_cfg(self.ccfg);
                n += c.instruments();
                self.cross.insert(ts, c);
            }
        }
        if let Some(closes) =
            v.get("last_close").and_then(|c| serde_json::from_value::<Vec<(String, String, f64)>>(c.clone()).ok())
        {
            for (t, i, p) in closes {
                if let Ok(ts) = t.parse::<Timescale>() {
                    self.last_close.insert((ts, i), p);
                }
            }
        }
        if let Some(means) = v
            .get("region_ret")
            .and_then(|c| serde_json::from_value::<Vec<(String, u32, String, f64, f64)>>(c.clone()).ok())
        {
            for (t, k, r, m, cnt) in means {
                if let Ok(ts) = t.parse::<Timescale>() {
                    self.region_ret.insert((ts, k, r), (m, cnt));
                }
            }
        }
        n
    }

    fn coarser(&self, ts: Timescale) -> Option<Timescale> {
        self.ladder.iter().copied().find(|t| *t > ts)
    }

    /// One pipeline step. Previews update the current region; closed bars feed the hierarchy
    /// (with the instrument's current coarse region) and the cross-asset returns.
    pub fn observe(&mut self, ts: Timescale, state: &LatentState, closed: bool, close: Option<f64>, as_of_ms: i64) {
        let inst = state.instrument_id.clone();
        let regions: Vec<String> = state.experts.iter().map(|e| e.region_id.clone()).collect();
        self.current.insert((ts, inst.clone()), (state.snapshot_id.clone(), regions.clone()));
        if !closed {
            return;
        }
        if let Some(coarse) = self.coarser(ts)
            && let Some((coarse_snap, coarse_regions)) = self.current.get(&(coarse, inst.clone()))
        {
            for (k, (f, c)) in regions.iter().zip(coarse_regions).enumerate() {
                let entry = self.hier.entry((ts, coarse, k as u32)).or_insert_with(|| {
                    (HierarchyTracker::new(self.hcfg), state.snapshot_id.clone(), coarse_snap.clone())
                });
                if entry.1 != state.snapshot_id || entry.2 != *coarse_snap {
                    *entry = (HierarchyTracker::new(self.hcfg), state.snapshot_id.clone(), coarse_snap.clone());
                }
                entry.0.observe(f, c, 1.0);
            }
        }
        if let Some(p) = close.filter(|p| *p > 0.0 && p.is_finite())
            && let Some(prev) = self.last_close.insert((ts, inst.clone()), p)
        {
            let ccfg = self.ccfg;
            let ret = (p / prev).ln();
            let resid = self.expert_residual(ts, &inst, ret, &regions, &state.router_weights);
            self.cross
                .entry(ts)
                .or_insert_with(|| CrossTracker::new(ccfg))
                .push_with_residual(&inst, as_of_ms, ret, resid);
        }
    }

    /// A retired region leaves every hierarchy estimate.
    pub fn forget(&mut self, region_id: &str) {
        for (t, _, _) in self.hier.values_mut() {
            t.forget(region_id);
        }
    }

    pub fn hierarchy(&self, as_of_ms: i64) -> Vec<RegionHierarchy> {
        self.hier
            .iter()
            .filter(|(_, (t, _, _))| t.total() > 0.0)
            .map(|((fine, coarse, k), (t, fs, cs))| {
                let u = t.update();
                RegionHierarchy {
                    fine: Some(ExpertSpaceRef {
                        snapshot_id: fs.clone(),
                        timescale: fine.to_string(),
                        expert_index: *k,
                    }),
                    coarse: Some(ExpertSpaceRef {
                        snapshot_id: cs.clone(),
                        timescale: coarse.to_string(),
                        expert_index: *k,
                    }),
                    as_of: Some(ms_to_ts(as_of_ms)),
                    mutual_information_bits: u.mutual_information_bits,
                    n: u.n,
                    edges: u
                        .edges
                        .into_iter()
                        .map(|e| HierEdge {
                            fine_region_id: e.fine,
                            coarse_region_id: e.coarse,
                            p_coarse_given_fine: e.p_coarse_given_fine,
                            p_fine_given_coarse: e.p_fine_given_coarse,
                            support: e.support,
                            pmi_bits: e.pmi_bits,
                        })
                        .collect(),
                }
            })
            .collect()
    }

    pub fn cross_priors(&self, as_of_ms: i64) -> Vec<CrossAssetPriors> {
        self.cross
            .iter()
            .map(|(ts, c)| {
                let all = c.all_pairs();
                let evaluated = all.len();
                let proposals = c
                    .proposals()
                    .into_iter()
                    .map(|p| CrossAssetPair {
                        source_instrument_id: p.source,
                        target_instrument_id: p.target,
                        n: p.n as u32,
                        correlation: p.correlation,
                        shrinkage: p.shrinkage,
                        lead_lag: p.lead_lag as u32,
                        lead_corr: p.lead_corr,
                        predictive_gain: p.predictive_gain,
                        uncertainty: if p.uncertainty.is_finite() { p.uncertainty } else { f64::MAX },
                        score: p.score,
                    })
                    .collect();
                CrossAssetPriors {
                    timescale: ts.to_string(),
                    as_of: Some(ms_to_ts(as_of_ms)),
                    proposals,
                    instruments: c.instruments() as u32,
                    pairs_evaluated: evaluated as u32,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use fa_patterns::{Ancestor, FittedRegion, Region, RetireReason};
    use fa_proto::v1::ExpertLatent;

    use super::*;

    fn space(s: &str) -> ExpertSpaceRef {
        ExpertSpaceRef { snapshot_id: s.into(), timescale: "1m".into(), expert_index: 0 }
    }

    #[test]
    fn online_events_map_to_lifecycle_kinds() {
        let region = Region {
            id: "r1".into(),
            lineage_id: "r1".into(),
            snapshot_id: "s".into(),
            expert_index: 0,
            centroid: vec![1.0],
            radius: 1.0,
            method: fa_patterns::DiscoveryMethod::Leader,
        };
        let none = |_: &str| None;
        let b = lifecycle_online(&space("s"), 5, &RegionLifecycle::Created { region: region.clone() }, none);
        assert_eq!((b.kind, b.reason.as_str(), b.centroid.len()), (LifeKind::Born as i32, "online", 1));
        let sp = lifecycle_online(
            &space("s"),
            5,
            &RegionLifecycle::Split {
                parent: "r0".into(),
                parent_lineage_id: "L0".into(),
                child: region,
                masses: (30.0, 10.0),
            },
            none,
        );
        assert_eq!(sp.kind, LifeKind::Split as i32);
        assert_eq!(sp.ancestors[0].kind, "split");
        assert!((sp.ancestors[0].overlap - 0.25).abs() < 1e-12);
        let d = lifecycle_online(
            &space("s"),
            5,
            &RegionLifecycle::Retired {
                region_id: "r2".into(),
                lineage_id: "r2".into(),
                merged_into: String::new(),
                reason: RetireReason::Support,
            },
            none,
        );
        assert_eq!((d.kind, d.reason.as_str()), (LifeKind::Died as i32, "support"));
    }

    #[test]
    fn map_alignment_yields_carry_split_and_dead_parents() {
        let fr = |id: &str, parent: Option<&str>, anc: Vec<Ancestor>| FittedRegion {
            region_id: id.into(),
            lineage_id: parent.map_or(id, |_| "L").into(),
            expert_index: 0,
            centroid: vec![0.5],
            radius: 1.0,
            mass: 0.5,
            parent_region_id: parent.map(Into::into),
            score: 0.7,
            ancestors: anc,
        };
        let a = |id: &str, kind| Ancestor { region_id: id.into(), lineage_id: "L".into(), overlap: 0.4, kind };
        let e = ExpertRegionMap {
            expert_index: 0,
            regions: vec![
                fr("n0", Some("p0"), vec![a("p0", DescentKind::Carry)]),
                fr("n1", None, vec![a("p0", DescentKind::Split)]),
            ],
            parent_regions: 2,
            matched: 1,
            carried_mass: 0.5,
            dead_parents: vec!["p1".into()],
            split_parents: vec!["p0".into()],
            regime_break: false,
        };
        let ev = lifecycle_map(&space("new"), Some(&space("old")), 9, &e);
        let kinds: Vec<i32> = ev.iter().map(|e| e.kind).collect();
        assert_eq!(kinds, vec![LifeKind::Carried as i32, LifeKind::Split as i32, LifeKind::Died as i32]);
        assert_eq!(ev[2].space.as_ref().unwrap().snapshot_id, "old");
        assert_eq!(ev[2].reason, "unmatched");
        assert_eq!(lifecycle_map(&space("new"), None, 9, &e).len(), 2);
    }

    #[test]
    fn kernel_update_summarises_the_graph() {
        let kcfg = KernelCfg { arena: ArenaCfg { min_evals: 10, ..ArenaCfg::default() }, ..KernelCfg::default() };
        let mut t = ArenaTransitions::from_graph(MarkovGraph::new(1.0, 1.0), 0.9, 64, &kcfg, 1);
        for r in ["a", "b"] {
            t.register_region(r);
        }
        t.set_centroid("a", &[0.0]);
        t.set_centroid("b", &[1.0]);
        let mut rng = Rng::seed(2);
        let mut cur = "a";
        for _ in 0..300 {
            let next = if rng.uniform() < 0.8 {
                cur
            } else if cur == "a" {
                "b"
            } else {
                "a"
            };
            t.observe(cur, next);
            cur = next;
        }
        let k = kernel_update(&space("s"), 1000, &t, 2, Some(DriftReport { kl: 0.2, refit_due: true }), 16, &mut rng)
            .unwrap();
        assert_eq!(k.samplers.len(), 6);
        assert_eq!(k.samplers.iter().filter(|s| s.serving).count(), 1);
        assert_eq!(k.serving_sampler_id, t.sampler_id());
        assert_eq!(k.stationary.len(), 2);
        let total: f64 = k.stationary.iter().map(|s| s.mean).sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert!(k.relaxation_steps > 2.0 && k.relaxation_steps < 10.0, "{k:?}");
        assert!(k.refit_due && k.n_rows == 2 && k.observation_mass > 200.0);
        let summary = t.summarize("a");
        assert!((summary.iter().map(|p| p.mean).sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cross_scale_builds_hierarchy_and_cross_priors() {
        let mut cs = CrossScale::new(
            vec![Timescale::Hour, Timescale::Minute],
            HierarchyCfg::default(),
            CrossAssetCfg { min_obs: 20, ..CrossAssetCfg::default() },
        );
        let st = |ts: &str, inst: &str, region: &str| LatentState {
            snapshot_id: format!("snap-{ts}"),
            timescale: ts.into(),
            instrument_id: inst.into(),
            experts: vec![ExpertLatent { expert_index: 0, region_id: region.into(), ..Default::default() }],
            ..Default::default()
        };
        let mut rng = Rng::seed(3);
        let mut lead = 100.0;
        let mut follow = 100.0;
        let mut last_lead_ret = 0.0;
        for t in 0..200i64 {
            let coarse = if (t / 50) % 2 == 0 { "H0" } else { "H1" };
            cs.observe(Timescale::Hour, &st("1h", "x", coarse), false, None, t);
            let fine = if coarse == "H0" { "m0" } else { "m1" };
            let r = 0.01 * rng.normal();
            lead *= f64::exp(r);
            follow *= f64::exp(0.9 * last_lead_ret + 0.001 * rng.normal());
            last_lead_ret = r;
            cs.observe(Timescale::Minute, &st("1m", "x", fine), true, Some(lead), t * 60_000);
            cs.observe(Timescale::Minute, &st("1m", "y", fine), true, Some(follow), t * 60_000);
        }
        let h = cs.hierarchy(0);
        assert_eq!(h.len(), 1);
        assert!(h[0].mutual_information_bits > 0.5);
        assert_eq!(h[0].fine.as_ref().unwrap().timescale, "1m");
        assert_eq!(h[0].coarse.as_ref().unwrap().snapshot_id, "snap-1h");
        assert_eq!(h[0].edges.len(), 2);
        cs.forget("m0");
        assert_eq!(cs.hierarchy(0)[0].edges.len(), 0, "MI of a single fine region is 0");
        let c = cs.cross_priors(0);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].timescale, "1m");
        assert_eq!(c[0].instruments, 2);
        let top = &c[0].proposals[0];
        assert_eq!((top.source_instrument_id.as_str(), top.target_instrument_id.as_str(), top.lead_lag), ("x", "y", 1));
    }

    #[test]
    fn drift_trigger_needs_a_sustained_drift_and_respects_the_cooldown() {
        let mut t = DriftTrigger::new(3, 1000);
        let k = |due: bool| KernelUpdate {
            space: Some(ExpertSpaceRef { snapshot_id: "s".into(), timescale: "1m".into(), expert_index: 1 }),
            refit_due: due,
            drift_kl: 0.3,
            ..Default::default()
        };
        assert!(t.observe(&k(true), 0).is_none());
        assert!(t.observe(&k(false), 1).is_none(), "streak broken");
        assert!(t.observe(&k(true), 2).is_none());
        assert!(t.observe(&k(true), 3).is_none());
        let spec = t.observe(&k(true), 4).expect("third consecutive");
        assert_eq!(spec["reason"], "region_drift");
        assert_eq!(spec["window_hours"], 1.0);
        for i in 0..3 {
            assert!(t.observe(&k(true), 5 + i).is_none(), "cooldown");
        }
        assert!(t.observe(&k(true), 2000).is_some());
    }
}
