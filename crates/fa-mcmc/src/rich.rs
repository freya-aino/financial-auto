//! Richer transition kernels (spec 33 §6.2) competing with the Dirichlet baselines in the
//! calibration arena ([`crate::calibration::Arena`]).
//!
//! Every kernel keeps the shared Dirichlet evidence ([`MarkovGraph`], so kernel transfer,
//! persistence and the graph-level queries work whichever kernel serves) and adds its own
//! structure on top, conditioned on the [`KernelCtx`] of the step:
//!
//! | kernel | `name()` | structure | fit |
//! |---|---|---|---|
//! | [`SecondOrder`] | `second_order` | state = (previous, current) region; Dirichlet rows per pair with the first-order posterior as prior (`α = s·p₁ + n₂`), so sparse pairs back off to first order | closed form, online |
//! | [`SemiMarkov`] | `semi_markov` | explicit dwell: per region and dwell bin a Beta hazard of leaving (prior centred on the first-order self-transition), exits by the first-order row without self | closed form, online |
//! | [`HierAsset`] | `hier_asset` | per-asset rows `θ_{a,i} ~ Dir(c·β_i)` sharing a global `β_i` (hierarchical Dirichlet) | online predictive `(n_{a,ij} + c·β_ij)/(n_{a,i} + c)`; `β`, `c` by **Gibbs** on `jobs.mcmc.refit` ([`crate::refit`]) |
//! | [`LogitCov`] | `logit_cov` | logistic-normal transitions on the six covariates of [`crate::refit::COVARIATES`] (ln dwell, volatility ratio, novelty, router weight, `peer_switch`, `coarse_ln_dwell`; D-080): `p(j | i, x) = softmax(W_i x)` | **NUTS** on `jobs.mcmc.refit`; online = Monte Carlo average over the posterior draws, first-order fallback for rows without a fit |
//!
//! Intervals are central quantiles over posterior draws of each kernel's own parameters.

use std::collections::{BTreeMap, HashMap};

use fa_core::rng::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    MarkovGraph, ProbSummary, RegionKey, UNKNOWN,
    kernel::KernelSampler,
    refit::{COV_DIM, OTHER, RefitOutcome},
};

/// What a kernel may condition on at one step (spec 33 §4.1 `KernelCtx`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct KernelCtx {
    /// Asset (instrument id) of the chain.
    pub instrument: String,
    /// Region before `from` (a different region; `None` at the chain start).
    pub prev: Option<RegionKey>,
    /// Closed bars spent in `from` so far, including the current one (≥ 1; 0 = unknown).
    pub dwell: u32,
    /// Covariates known at `from` (see [`crate::refit::COVARIATES`]).
    pub x: [f64; COV_DIM],
}

/// Posterior mean + central interval from draws aligned with `keys`.
pub fn summarize_draws(
    keys: &[RegionKey],
    mean: &[f64],
    draws: &[Vec<f64>],
    level: f64,
) -> Vec<(RegionKey, ProbSummary)> {
    let lo = (1.0 - level) / 2.0;
    keys.iter()
        .enumerate()
        .map(|(j, k)| {
            let mut xs: Vec<f64> = draws.iter().map(|d| d[j]).collect();
            xs.sort_by(f64::total_cmp);
            let q = |p: f64| {
                if xs.is_empty() {
                    mean[j]
                } else {
                    xs[((p * (xs.len() - 1) as f64).round() as usize).min(xs.len() - 1)]
                }
            };
            (k.clone(), ProbSummary { mean: mean[j], lower: q(lo).min(mean[j]), upper: q(1.0 - lo).max(mean[j]) })
        })
        .collect()
}

fn normalise(v: &mut [(RegionKey, f64)]) {
    let s: f64 = v.iter().map(|x| x.1).sum();
    if s > 0.0 {
        for x in v.iter_mut() {
            x.1 /= s;
        }
    }
}

fn decay_add(row: &mut BTreeMap<RegionKey, f64>, decay: f64, to: &str, w: f64) {
    for c in row.values_mut() {
        *c *= decay;
    }
    *row.entry(to.to_owned()).or_insert(0.0) += w;
}

/// Dirichlet draws of a row with concentrations `alpha` (keys aligned).
fn dirichlet_draws(alpha: &[f64], n: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let a: Vec<f64> = alpha.iter().map(|x| x.max(1e-9)).collect();
    (0..n).map(|_| rng.dirichlet(&a)).collect()
}

macro_rules! base_impl {
    () => {
        fn graph(&self) -> &MarkovGraph {
            &self.base
        }
        fn graph_mut(&mut self) -> &mut MarkovGraph {
            &mut self.base
        }
        fn observe(&mut self, from: &str, to: &str) {
            self.observe_ctx(from, to, &KernelCtx::default());
        }
        fn predict(&mut self, from: &str) -> Vec<(RegionKey, f64)> {
            self.predict_ctx(from, &KernelCtx::default())
        }
        fn summarize(&mut self, from: &str, level: f64) -> Vec<(RegionKey, ProbSummary)> {
            self.summarize_ctx(from, level, &KernelCtx::default())
        }
    };
}

/// Prunes decayed context rows (keys `<prev|asset>\u{1f}<from>`) alongside the shared evidence:
/// entries below `min_edge`, rows left empty and rows whose `from` row was removed.
fn prune_ctx_rows(rows: &mut HashMap<String, BTreeMap<RegionKey, f64>>, min_edge: f64, removed: &[RegionKey]) {
    rows.retain(|key, row| {
        row.retain(|_, c| *c >= min_edge);
        let from = key.rsplit('\u{1f}').next().unwrap_or(key);
        !row.is_empty() && !removed.iter().any(|r| r == from)
    });
}

// ---------------------------------------------------------------------------------------------
// Higher order

/// Second-order kernel: rows per (previous, current) region with first-order back-off.
pub struct SecondOrder {
    base: MarkovGraph,
    rows: HashMap<String, BTreeMap<RegionKey, f64>>,
    /// Prior strength `s` of the first-order back-off.
    pub strength: f64,
    rng: Rng,
    draws: usize,
}

fn pair_key(prev: &str, from: &str) -> String {
    format!("{prev}\u{1f}{from}")
}

impl SecondOrder {
    pub fn new(base: MarkovGraph, strength: f64, draws: usize, seed: u64) -> Self {
        Self { base, rows: HashMap::new(), strength: strength.max(1e-3), rng: Rng::seed(seed), draws: draws.max(16) }
    }

    /// Dirichlet concentrations of the row given the context.
    fn alpha(&self, from: &str, ctx: &KernelCtx) -> Vec<(RegionKey, f64)> {
        let p1 = self.base.predictive(from);
        let Some(row) = ctx.prev.as_deref().and_then(|p| self.rows.get(&pair_key(p, from))) else {
            let mass = self.base.row_mass(from) + self.base.prior * p1.len() as f64;
            return p1.into_iter().map(|(k, p)| (k, p * mass)).collect();
        };
        let mut out: BTreeMap<RegionKey, f64> = p1.into_iter().map(|(k, p)| (k, self.strength * p)).collect();
        for (k, c) in row {
            *out.entry(k.clone()).or_insert(0.0) += c;
        }
        out.into_iter().collect()
    }

    pub fn pair_rows(&self) -> usize {
        self.rows.len()
    }
}

impl KernelSampler for SecondOrder {
    fn name(&self) -> &str {
        "second_order"
    }
    base_impl!();
    fn prune(&mut self, min_edge: f64) -> crate::PruneReport {
        let rep = self.base.prune(min_edge);
        prune_ctx_rows(&mut self.rows, min_edge, &rep.rows);
        rep
    }
    fn replace_graph(&mut self, g: MarkovGraph) {
        self.base = g;
        self.rows.clear();
    }
    fn observe_ctx(&mut self, from: &str, to: &str, ctx: &KernelCtx) {
        self.base.observe(from, to);
        if let Some(prev) = ctx.prev.as_deref() {
            let decay = self.base.decay;
            decay_add(self.rows.entry(pair_key(prev, from)).or_default(), decay, to, 1.0);
        }
    }
    fn predict_ctx(&mut self, from: &str, ctx: &KernelCtx) -> Vec<(RegionKey, f64)> {
        let mut a = self.alpha(from, ctx);
        normalise(&mut a);
        a
    }
    fn summarize_ctx(&mut self, from: &str, level: f64, ctx: &KernelCtx) -> Vec<(RegionKey, ProbSummary)> {
        let a = self.alpha(from, ctx);
        let keys: Vec<RegionKey> = a.iter().map(|x| x.0.clone()).collect();
        let al: Vec<f64> = a.iter().map(|x| x.1).collect();
        let s: f64 = al.iter().sum();
        let mean: Vec<f64> = al.iter().map(|x| x / s).collect();
        let d = dirichlet_draws(&al, self.draws, &mut self.rng);
        summarize_draws(&keys, &mean, &d, level)
    }
    fn export_state(&self) -> Option<serde_json::Value> {
        // only rows with evidence worth keeping (bounded persistence)
        let rows: BTreeMap<&String, &BTreeMap<RegionKey, f64>> =
            self.rows.iter().filter(|(_, r)| r.values().sum::<f64>() >= 0.5).collect();
        serde_json::to_value(serde_json::json!({ "rows": rows })).ok()
    }
    fn import_state(&mut self, v: &serde_json::Value) {
        if let Some(rows) = v.get("rows").and_then(|r| serde_json::from_value(r.clone()).ok()) {
            self.rows = rows;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Semi-Markov

/// Dwell bins (closed bars): 1, 2, 3, 4, 5–6, 7–9, 10–14, 15–24, 25–49, 50+.
pub const DWELL_BINS: [u32; 10] = [1, 2, 3, 4, 5, 7, 10, 15, 25, 50];

pub fn dwell_bin(d: u32) -> usize {
    let d = d.max(1);
    DWELL_BINS.iter().rposition(|lo| d >= *lo).unwrap_or(0)
}

/// Explicit-duration kernel: Beta hazard of leaving per region and dwell bin.
pub struct SemiMarkov {
    base: MarkovGraph,
    /// region → per dwell bin (leave, stay) decayed counts.
    hazard: HashMap<RegionKey, [(f64, f64); DWELL_BINS.len()]>,
    /// Prior strength of the hazard (pseudo-observations), centred on the first-order self rate.
    pub strength: f64,
    rng: Rng,
    draws: usize,
}

impl SemiMarkov {
    pub fn new(base: MarkovGraph, strength: f64, draws: usize, seed: u64) -> Self {
        Self { base, hazard: HashMap::new(), strength: strength.max(1e-3), rng: Rng::seed(seed), draws: draws.max(16) }
    }

    /// (self probability under first order, exit distribution without self, prior hazard).
    fn parts(&self, from: &str) -> (Vec<(RegionKey, f64)>, f64) {
        let p1 = self.base.predictive(from);
        let p_self = p1.iter().find(|x| x.0 == from).map_or(0.0, |x| x.1);
        let mut exit: Vec<(RegionKey, f64)> = p1.into_iter().filter(|x| x.0 != from).collect();
        if exit.is_empty() {
            exit.push((UNKNOWN.to_owned(), 1.0));
        }
        normalise(&mut exit);
        (exit, p_self.clamp(1e-3, 1.0 - 1e-3))
    }

    /// Beta parameters of the hazard of leaving `from` at dwell `d`.
    fn hazard_beta(&self, from: &str, d: u32, p_self: f64) -> (f64, f64) {
        let (a0, b0) = (self.strength * (1.0 - p_self), self.strength * p_self);
        let (l, s) = self.hazard.get(from).map_or((0.0, 0.0), |h| h[dwell_bin(d)]);
        (a0 + l, b0 + s)
    }

    /// Posterior-mean survival `P(dwell ≥ d)` for d = 1..=`horizon` (time-in-trade logic, spec 35).
    pub fn survival(&self, from: &str, horizon: u32) -> Vec<f64> {
        let (_, p_self) = self.parts(from);
        let mut s = 1.0;
        (1..=horizon)
            .map(|d| {
                let out = s;
                let (a, b) = self.hazard_beta(from, d, p_self);
                s *= 1.0 - a / (a + b);
                out
            })
            .collect()
    }
}

impl KernelSampler for SemiMarkov {
    fn name(&self) -> &str {
        "semi_markov"
    }
    base_impl!();
    fn replace_graph(&mut self, g: MarkovGraph) {
        self.base = g;
        self.hazard.clear();
    }
    fn prune(&mut self, min_edge: f64) -> crate::PruneReport {
        let rep = self.base.prune(min_edge);
        for k in &rep.rows {
            self.hazard.remove(k);
        }
        rep
    }
    fn observe_ctx(&mut self, from: &str, to: &str, ctx: &KernelCtx) {
        self.base.observe(from, to);
        let decay = self.base.decay;
        let h = self.hazard.entry(from.to_owned()).or_insert([(0.0, 0.0); DWELL_BINS.len()]);
        let b = dwell_bin(ctx.dwell);
        for x in h.iter_mut() {
            x.0 *= decay;
            x.1 *= decay;
        }
        if to == from {
            h[b].1 += 1.0;
        } else {
            h[b].0 += 1.0;
        }
    }
    fn predict_ctx(&mut self, from: &str, ctx: &KernelCtx) -> Vec<(RegionKey, f64)> {
        let (exit, p_self) = self.parts(from);
        let (a, b) = self.hazard_beta(from, ctx.dwell, p_self);
        let h = a / (a + b);
        let mut out: Vec<(RegionKey, f64)> = exit.into_iter().map(|(k, q)| (k, h * q)).collect();
        out.push((from.to_owned(), 1.0 - h));
        out
    }
    fn summarize_ctx(&mut self, from: &str, level: f64, ctx: &KernelCtx) -> Vec<(RegionKey, ProbSummary)> {
        let mean = self.predict_ctx(from, ctx);
        let (exit, p_self) = self.parts(from);
        let (a, b) = self.hazard_beta(from, ctx.dwell, p_self);
        let exit_alpha: Vec<f64> = {
            let mass = (self.base.row_mass(from) + self.base.prior * exit.len() as f64).max(1.0);
            exit.iter().map(|x| x.1 * mass).collect()
        };
        let draws: Vec<Vec<f64>> = (0..self.draws)
            .map(|_| {
                let (x, y) = (self.rng.gamma(a.max(1e-9)), self.rng.gamma(b.max(1e-9)));
                let h = x / (x + y).max(1e-300);
                let q = self.rng.dirichlet(&exit_alpha.iter().map(|v| v.max(1e-9)).collect::<Vec<_>>());
                q.into_iter().map(|v| h * v).chain(std::iter::once(1.0 - h)).collect()
            })
            .collect();
        let keys: Vec<RegionKey> = mean.iter().map(|x| x.0.clone()).collect();
        let m: Vec<f64> = mean.iter().map(|x| x.1).collect();
        summarize_draws(&keys, &m, &draws, level)
    }
    fn export_state(&self) -> Option<serde_json::Value> {
        serde_json::to_value(serde_json::json!({ "hazard": self.hazard })).ok()
    }
    fn import_state(&mut self, v: &serde_json::Value) {
        if let Some(h) = v.get("hazard").and_then(|r| serde_json::from_value(r.clone()).ok()) {
            self.hazard = h;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Hierarchical across assets

/// Asset-specific rows sharing a global kernel through a hierarchical Dirichlet.
pub struct HierAsset {
    base: MarkovGraph,
    assets: HashMap<String, BTreeMap<RegionKey, f64>>,
    /// Gibbs posterior mean of the global row `β_i` (keys incl. [`OTHER`]).
    beta: HashMap<RegionKey, Vec<(RegionKey, f64)>>,
    /// Concentration `c` of the asset rows around `β`.
    pub conc: f64,
    pub refits: u64,
    /// Garage key of the last applied refit (restored on restart).
    pub refit_ref: Option<String>,
    rng: Rng,
    draws: usize,
}

impl HierAsset {
    pub fn new(base: MarkovGraph, conc: f64, draws: usize, seed: u64) -> Self {
        Self {
            base,
            assets: HashMap::new(),
            beta: HashMap::new(),
            conc: conc.max(1e-3),
            refits: 0,
            refit_ref: None,
            rng: Rng::seed(seed),
            draws: draws.max(16),
        }
    }

    /// The global row: Gibbs `β` (the [`OTHER`] share spread by the pooled posterior over the
    /// remaining successors) or, before any refit, the pooled posterior mean.
    fn global(&self, from: &str) -> Vec<(RegionKey, f64)> {
        let pooled = self.base.predictive(from);
        let Some(beta) = self.beta.get(from) else { return pooled };
        spread_other(beta, &pooled)
    }

    fn alpha(&self, from: &str, ctx: &KernelCtx) -> Vec<(RegionKey, f64)> {
        let g = self.global(from);
        let row = self.assets.get(&pair_key(&ctx.instrument, from));
        let mut out: BTreeMap<RegionKey, f64> = g.into_iter().map(|(k, p)| (k, self.conc * p)).collect();
        if let Some(r) = row {
            for (k, c) in r {
                *out.entry(k.clone()).or_insert(0.0) += c;
            }
        }
        out.into_iter().collect()
    }
}

/// Maps a row over `keys ∪ {OTHER}` onto concrete successors: the [`OTHER`] mass is spread by
/// `fallback` over the successors the row does not name ([`UNKNOWN`] when there are none).
pub fn spread_other(row: &[(RegionKey, f64)], fallback: &[(RegionKey, f64)]) -> Vec<(RegionKey, f64)> {
    let other = row.iter().find(|x| x.0 == OTHER).map_or(0.0, |x| x.1);
    let mut out: BTreeMap<RegionKey, f64> = row.iter().filter(|x| x.0 != OTHER).cloned().collect();
    let rest: Vec<&(RegionKey, f64)> = fallback.iter().filter(|x| !out.contains_key(&x.0)).collect();
    let rest_mass: f64 = rest.iter().map(|x| x.1).sum();
    if rest.is_empty() || rest_mass <= 0.0 {
        *out.entry(UNKNOWN.to_owned()).or_insert(0.0) += other;
    } else {
        for (k, p) in rest {
            *out.entry(k.clone()).or_insert(0.0) += other * p / rest_mass;
        }
    }
    let mut v: Vec<(RegionKey, f64)> = out.into_iter().collect();
    normalise(&mut v);
    v
}

impl KernelSampler for HierAsset {
    fn name(&self) -> &str {
        "hier_asset"
    }
    base_impl!();
    fn replace_graph(&mut self, g: MarkovGraph) {
        self.base = g;
        self.assets.clear();
        self.beta.clear();
    }
    fn prune(&mut self, min_edge: f64) -> crate::PruneReport {
        let rep = self.base.prune(min_edge);
        prune_ctx_rows(&mut self.assets, min_edge, &rep.rows);
        for k in &rep.rows {
            self.beta.remove(k);
        }
        rep
    }
    fn observe_ctx(&mut self, from: &str, to: &str, ctx: &KernelCtx) {
        self.base.observe(from, to);
        if !ctx.instrument.is_empty() {
            let decay = self.base.decay;
            decay_add(self.assets.entry(pair_key(&ctx.instrument, from)).or_default(), decay, to, 1.0);
        }
    }
    fn predict_ctx(&mut self, from: &str, ctx: &KernelCtx) -> Vec<(RegionKey, f64)> {
        let mut a = self.alpha(from, ctx);
        normalise(&mut a);
        a
    }
    fn summarize_ctx(&mut self, from: &str, level: f64, ctx: &KernelCtx) -> Vec<(RegionKey, ProbSummary)> {
        let a = self.alpha(from, ctx);
        let keys: Vec<RegionKey> = a.iter().map(|x| x.0.clone()).collect();
        let al: Vec<f64> = a.iter().map(|x| x.1).collect();
        let s: f64 = al.iter().sum();
        let mean: Vec<f64> = al.iter().map(|x| x / s).collect();
        let d = dirichlet_draws(&al, self.draws, &mut self.rng);
        summarize_draws(&keys, &mean, &d, level)
    }
    fn apply_refit(&mut self, o: &RefitOutcome) -> bool {
        let Some(h) = o.hier.as_ref().filter(|h| h.accepted) else { return false };
        for r in &h.rows {
            self.beta.insert(r.from.clone(), r.keys.iter().cloned().zip(r.beta_mean.iter().copied()).collect());
        }
        self.conc = h.conc_mean.max(1e-3);
        self.refits += 1;
        if !o.samples_ref.is_empty() {
            self.refit_ref = Some(o.samples_ref.clone());
        }
        true
    }
    fn export_state(&self) -> Option<serde_json::Value> {
        let assets: BTreeMap<&String, &BTreeMap<RegionKey, f64>> =
            self.assets.iter().filter(|(_, r)| r.values().sum::<f64>() >= 0.5).collect();
        serde_json::to_value(serde_json::json!({ "assets": assets, "refit_ref": self.refit_ref })).ok()
    }
    fn import_state(&mut self, v: &serde_json::Value) {
        if let Some(a) = v.get("assets").and_then(|r| serde_json::from_value(r.clone()).ok()) {
            self.assets = a;
        }
        self.refit_ref = v.get("refit_ref").and_then(|r| r.as_str()).map(str::to_owned);
    }
}

// ---------------------------------------------------------------------------------------------
// Covariate-conditioned (logistic-normal)

/// Posterior draws of one row's softmax weights, `K × (1 + COV_DIM)` per draw.
#[derive(Debug, Clone)]
struct LogitServing {
    keys: Vec<RegionKey>,
    draws: Vec<Vec<f32>>,
    x_mean: [f64; COV_DIM],
    x_sd: [f64; COV_DIM],
}

impl LogitServing {
    fn probs(&self, w: &[f32], xs: &[f64; COV_DIM + 1], out: &mut [f64]) {
        let p = COV_DIM + 1;
        let mut m = f64::NEG_INFINITY;
        for (k, o) in out.iter_mut().enumerate() {
            let z: f64 = (0..p).map(|d| f64::from(w[k * p + d]) * xs[d]).sum();
            *o = z;
            m = m.max(z);
        }
        let mut s = 0.0;
        for o in out.iter_mut() {
            *o = (*o - m).exp();
            s += *o;
        }
        for o in out.iter_mut() {
            *o /= s;
        }
    }

    fn design(&self, x: &[f64; COV_DIM]) -> [f64; COV_DIM + 1] {
        let mut xs = [1.0; COV_DIM + 1];
        for d in 0..COV_DIM {
            xs[d + 1] = ((x[d] - self.x_mean[d]) / self.x_sd[d].max(1e-6)).clamp(-6.0, 6.0);
        }
        xs
    }
}

/// Multinomial-logistic kernel on covariates, NUTS-fitted (spec 33 §6.2 covariate-conditioned).
pub struct LogitCov {
    base: MarkovGraph,
    rows: HashMap<RegionKey, LogitServing>,
    /// Draws used per online prediction.
    pub online_draws: usize,
    pub refits: u64,
    pub refit_ref: Option<String>,
}

impl LogitCov {
    pub fn new(base: MarkovGraph, online_draws: usize) -> Self {
        Self { base, rows: HashMap::new(), online_draws: online_draws.max(4), refits: 0, refit_ref: None }
    }

    pub fn fitted_rows(&self) -> usize {
        self.rows.len()
    }

    /// Per-draw probability vectors over the row keys (`None` without a fit).
    fn draw_probs(&self, from: &str, ctx: &KernelCtx) -> Option<(&LogitServing, Vec<Vec<f64>>)> {
        let row = self.rows.get(from)?;
        let xs = row.design(&ctx.x);
        let step = (row.draws.len() / self.online_draws).max(1);
        let probs = row
            .draws
            .iter()
            .step_by(step)
            .map(|w| {
                let mut p = vec![0.0; row.keys.len()];
                row.probs(w, &xs, &mut p);
                p
            })
            .collect();
        Some((row, probs))
    }
}

impl KernelSampler for LogitCov {
    fn name(&self) -> &str {
        "logit_cov"
    }
    base_impl!();
    fn prune(&mut self, min_edge: f64) -> crate::PruneReport {
        let rep = self.base.prune(min_edge);
        for k in &rep.rows {
            self.rows.remove(k);
        }
        rep
    }
    fn replace_graph(&mut self, g: MarkovGraph) {
        self.base = g;
        self.rows.clear();
    }
    fn observe_ctx(&mut self, from: &str, to: &str, _ctx: &KernelCtx) {
        // the logistic weights are refitted on the observation window; the shared evidence
        // keeps the first-order fallback current
        self.base.observe(from, to);
    }
    fn predict_ctx(&mut self, from: &str, ctx: &KernelCtx) -> Vec<(RegionKey, f64)> {
        let Some((row, probs)) = self.draw_probs(from, ctx) else { return self.base.predictive(from) };
        let n = probs.len().max(1) as f64;
        let mean: Vec<(RegionKey, f64)> = row
            .keys
            .iter()
            .enumerate()
            .map(|(j, k)| (k.clone(), probs.iter().map(|p| p[j]).sum::<f64>() / n))
            .collect();
        spread_other(&mean, &self.base.predictive(from))
    }
    fn summarize_ctx(&mut self, from: &str, level: f64, ctx: &KernelCtx) -> Vec<(RegionKey, ProbSummary)> {
        let fallback = self.base.predictive(from);
        let Some((row, probs)) = self.draw_probs(from, ctx) else {
            let mut rng = Rng::seed(0x10C1);
            return self.base.summarize(from, level, 64, &mut rng);
        };
        let keys = row.keys.clone();
        let mapped: Vec<Vec<(RegionKey, f64)>> = probs
            .iter()
            .map(|p| spread_other(&keys.iter().cloned().zip(p.iter().copied()).collect::<Vec<_>>(), &fallback))
            .collect();
        let out_keys: Vec<RegionKey> = mapped[0].iter().map(|x| x.0.clone()).collect();
        let draws: Vec<Vec<f64>> = mapped.iter().map(|m| m.iter().map(|x| x.1).collect()).collect();
        let n = draws.len() as f64;
        let mean: Vec<f64> = (0..out_keys.len()).map(|j| draws.iter().map(|d| d[j]).sum::<f64>() / n).collect();
        summarize_draws(&out_keys, &mean, &draws, level)
    }
    fn apply_refit(&mut self, o: &RefitOutcome) -> bool {
        let mut changed = false;
        for r in o.logit.iter().filter(|r| r.accepted && !r.draws.is_empty()) {
            self.rows.insert(
                r.from.clone(),
                LogitServing { keys: r.keys.clone(), draws: r.draws.clone(), x_mean: r.x_mean, x_sd: r.x_sd },
            );
            changed = true;
        }
        if changed {
            self.refits += 1;
            if !o.samples_ref.is_empty() {
                self.refit_ref = Some(o.samples_ref.clone());
            }
        }
        changed
    }
    fn export_state(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({ "refit_ref": self.refit_ref }))
    }
    fn import_state(&mut self, v: &serde_json::Value) {
        self.refit_ref = v.get("refit_ref").and_then(|r| r.as_str()).map(str::to_owned);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(prev: Option<&str>, dwell: u32, inst: &str) -> KernelCtx {
        KernelCtx { instrument: inst.into(), prev: prev.map(Into::into), dwell, x: [0.0; COV_DIM] }
    }

    fn p(v: &[(RegionKey, f64)], k: &str) -> f64 {
        v.iter().find(|x| x.0 == k).map_or(0.0, |x| x.1)
    }

    #[test]
    fn second_order_learns_what_first_order_cannot() {
        // a → b → (c if came from a, else a); first order sees b → {a, c} 50/50.
        let mut so = SecondOrder::new(MarkovGraph::new(1.0, 1.0), 4.0, 64, 1);
        for _ in 0..300 {
            so.observe_ctx("a", "b", &ctx(Some("c"), 1, "x"));
            so.observe_ctx("b", "c", &ctx(Some("a"), 1, "x"));
            so.observe_ctx("c", "b", &ctx(Some("b"), 1, "x"));
            so.observe_ctx("b", "a", &ctx(Some("c"), 1, "x"));
        }
        let after_a = so.predict_ctx("b", &ctx(Some("a"), 1, "x"));
        let first = so.graph().predictive("b");
        assert!(p(&after_a, "c") > 0.95, "{after_a:?}");
        assert!((p(&first, "c") - 0.5).abs() < 0.05);
        let s = so.summarize_ctx("b", 0.9, &ctx(Some("a"), 1, "x"));
        assert!(s.iter().all(|(_, x)| x.lower <= x.mean && x.mean <= x.upper));
        // unseen pair backs off to first order
        let unseen = so.predict_ctx("b", &ctx(Some("zz"), 1, "x"));
        assert!((p(&unseen, "c") - p(&first, "c")).abs() < 1e-9);
        let st = so.export_state().unwrap();
        let mut so2 = SecondOrder::new(so.graph().clone(), 4.0, 64, 2);
        so2.import_state(&st);
        assert_eq!(so2.pair_rows(), so.pair_rows());
    }

    #[test]
    fn semi_markov_learns_a_fixed_dwell() {
        // stays exactly 3 bars in each region, then moves: a a a b b b a a a ...
        let mut sm = SemiMarkov::new(MarkovGraph::new(1.0, 1.0), 2.0, 64, 3);
        let seq: Vec<&str> = (0..600).map(|t| if (t / 3) % 2 == 0 { "a" } else { "b" }).collect();
        let mut dwell = 1;
        for w in seq.windows(2) {
            sm.observe_ctx(w[0], w[1], &ctx(None, dwell, "x"));
            dwell = if w[0] == w[1] { dwell + 1 } else { 1 };
        }
        let at1 = sm.predict_ctx("a", &ctx(None, 1, "x"));
        let at3 = sm.predict_ctx("a", &ctx(None, 3, "x"));
        assert!(p(&at1, "a") > 0.95, "{at1:?}");
        assert!(p(&at3, "b") > 0.95, "{at3:?}");
        let surv = sm.survival("a", 5);
        assert!(surv[0] == 1.0 && surv[2] > 0.9 && surv[3] < 0.1, "{surv:?}");
        let s = sm.summarize_ctx("a", 0.9, &ctx(None, 3, "x"));
        assert!((s.iter().map(|x| x.1.mean).sum::<f64>() - 1.0).abs() < 1e-9);
        assert_eq!(dwell_bin(0), 0);
        assert_eq!(dwell_bin(6), 4);
        assert_eq!(dwell_bin(500), 9);
    }

    #[test]
    fn hier_asset_separates_assets_and_shrinks_sparse_ones() {
        let mut h = HierAsset::new(MarkovGraph::new(1.0, 1.0), 10.0, 64, 4);
        for _ in 0..400 {
            h.observe_ctx("a", "b", &ctx(None, 1, "x"));
            h.observe_ctx("a", "c", &ctx(None, 1, "y"));
        }
        h.observe_ctx("a", "b", &ctx(None, 1, "z"));
        let x = h.predict_ctx("a", &ctx(None, 1, "x"));
        let y = h.predict_ctx("a", &ctx(None, 1, "y"));
        let z = h.predict_ctx("a", &ctx(None, 1, "z"));
        assert!(p(&x, "b") > 0.95 && p(&y, "c") > 0.95);
        assert!(p(&z, "b") > 0.4 && p(&z, "c") > 0.3, "sparse asset borrows the global row: {z:?}");
        let st = h.export_state().unwrap();
        let mut h2 = HierAsset::new(h.graph().clone(), 10.0, 64, 5);
        h2.import_state(&st);
        assert!((p(&h2.predict_ctx("a", &ctx(None, 1, "x")), "b") - p(&x, "b")).abs() < 1e-9);
    }

    #[test]
    fn spread_other_keeps_mass() {
        let row = vec![("a".to_owned(), 0.6), (OTHER.to_owned(), 0.4)];
        let fb = vec![("a".to_owned(), 0.5), ("b".to_owned(), 0.3), ("c".to_owned(), 0.2)];
        let v = spread_other(&row, &fb);
        assert!((p(&v, "b") - 0.24).abs() < 1e-9 && (p(&v, "c") - 0.16).abs() < 1e-9);
        let v = spread_other(&row, &[("a".to_owned(), 1.0)]);
        assert!((p(&v, UNKNOWN) - 0.4).abs() < 1e-9);
    }
}
