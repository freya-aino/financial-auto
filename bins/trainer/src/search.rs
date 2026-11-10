//! Decoder → encoder feedback for the statistical family (spec 34 §6.6, spec 36 §6.3, D-058).
//!
//! **What this is, honestly.** The statistical encoder has no loss and no gradients, so the
//! spec's default feedback (backprop through frozen elite decoders) and its alternatives
//! (distillation, sample reweighting) have nothing to act on. What the family *does* have is a
//! handful of hyper-parameters, so a `generation` job runs a **small deterministic
//! hyper-parameter search** and scores each candidate with an in-process **proxy**:
//!
//! 1. **Region predictability** (primary): the candidate encoder embeds a sample of window ends
//!    of every fitted instrument, leader clustering finds its regions per expert, and a
//!    Dirichlet transition model (`fa_mcmc::MarkovGraph`, the same conjugate kernel the live
//!    runtime uses) is fitted on the first half of each instrument's region sequence. The score
//!    is the out-of-sample `fa_mcmc::log_score` (mean log predictive probability of realised
//!    transitions, the sampler calibration score of spec 33) on the second half **minus** the
//!    log score of the marginal region frequencies — i.e. the information the state graph adds
//!    per step, in nats. A one-region partition scores ≈ 0, not a perfect log score.
//! 2. **Decoder skill of the parent** (tie-breaker): when decoders evaluated on the parent
//!    report skill (`crps_skill` of `validation.trial` rows keyed by the parent's
//!    `snapshot_id`, else the latest `validation.generation_stats.deflated_sharpe`), a candidate
//!    is penalised by `skill_weight · max(skill, 0) · distance(candidate, parent params)`:
//!    representations that decoders already exploit are moved only when predictability pays
//!    for it. With no or negative skill the term is 0.
//!
//! This is a stand-in, not the feedback path of D-009: it selects among representations by a
//! property decoders need (a predictable state graph) and by the decoders' own verdict on the
//! parent, but no decoder gradient or soft target reaches the encoder. Neural families replace
//! it with backprop/distillation behind the same `manifest.feedback` record.
//!
//! Search space (all optional in `FitSpec.search`): `context_steps`, the feature squash scale
//! `s`, per-expert feature weights `w_e`, region radius and max regions. `s` and `w_e` are baked
//! into `NormStats.gain = w_e / s`. Candidate 0 is the parent's (or default) parameters; the rest
//! are drawn from the grid by `Rng::seed(seed)` — the same seed and data give the same trace.

use std::collections::BTreeSet;

use fa_core::{Timescale, rng::Rng};
use fa_encoders::{
    EXPERT_NAMES, EncoderFamily, EncoderInput, FitConfig, NormAccumulator, NormStats, STATISTICAL_FAMILY,
    SnapshotManifest, StatisticalEncoder, default_sizing,
};
use fa_mcmc::{MarkovGraph, log_score};
use fa_patterns::LeaderClustering;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::data::BarHistory;

/// Hyper-parameters of one statistical-family candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub context_steps: u32,
    /// Divides every z-score before the squash (`tanh(z / (2 s))`).
    pub squash_scale: f64,
    /// Per-expert feature weight (`EXPERT_NAMES` order).
    pub expert_weights: Vec<f64>,
    pub region_radius: f32,
    pub max_regions: usize,
}

impl Candidate {
    pub fn defaults(ts: Timescale, region_radius: f32, max_regions: usize) -> Self {
        Self {
            context_steps: default_sizing(ts).context_steps,
            squash_scale: 1.0,
            expert_weights: vec![1.0; EXPERT_NAMES.len()],
            region_radius,
            max_regions,
        }
    }

    /// The parameters a manifest recorded (`params.search_params`, else its top-level
    /// `context_steps` and defaults).
    pub fn from_manifest(m: &SnapshotManifest, fallback: &Candidate) -> Candidate {
        if let Some(c) = m.params.get("search_params").and_then(|v| serde_json::from_value(v.clone()).ok()) {
            return c;
        }
        let mut c = fallback.clone();
        if let Some(ctx) = m.params.get("context_steps").and_then(Value::as_u64) {
            c.context_steps = ctx as u32;
        }
        c
    }

    /// `NormStats.gain`: `w_e / s`.
    pub fn gains(&self) -> Vec<f32> {
        let s = self.squash_scale.max(1e-6);
        self.expert_weights.iter().map(|w| (w / s) as f32).collect()
    }

    /// Normalised distance in log-parameter space (0 = identical).
    pub fn distance(&self, other: &Candidate) -> f64 {
        let l = |a: f64, b: f64| (a.max(1e-9) / b.max(1e-9)).ln().abs();
        let w = self.expert_weights.iter().zip(&other.expert_weights).map(|(a, b)| l(*a, *b)).sum::<f64>()
            / self.expert_weights.len().max(1) as f64;
        (l(f64::from(self.context_steps), f64::from(other.context_steps))
            + l(self.squash_scale, other.squash_scale)
            + w
            + l(f64::from(self.region_radius), f64::from(other.region_radius))
            + l(self.max_regions as f64, other.max_regions as f64))
            / 5.0
    }
}

/// `FitSpec.search`: the grid and the proxy settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchSpec {
    /// Candidates evaluated (candidate 0 = the parent's parameters). 1 disables the search.
    #[serde(default = "d_candidates")]
    pub candidates: usize,
    /// Empty → `{¼, ½, 1} × default_sizing(ts).context_steps` (≥ 16).
    #[serde(default)]
    pub context_steps: Vec<u32>,
    #[serde(default = "d_scales")]
    pub squash_scales: Vec<f64>,
    #[serde(default = "d_weights")]
    pub expert_weights: Vec<f64>,
    #[serde(default = "d_radii")]
    pub region_radii: Vec<f32>,
    #[serde(default = "d_max_regions")]
    pub max_regions: Vec<usize>,
    /// Weight of the parent-skill distance penalty.
    #[serde(default = "d_skill_weight")]
    pub skill_weight: f64,
    /// Windows embedded per candidate (spread over the instruments).
    #[serde(default = "d_eval_max")]
    pub eval_max: usize,
}

fn d_candidates() -> usize {
    12
}
fn d_scales() -> Vec<f64> {
    vec![0.5, 1.0, 2.0]
}
fn d_weights() -> Vec<f64> {
    vec![0.5, 1.0, 1.5]
}
fn d_radii() -> Vec<f32> {
    vec![0.6, 0.8, 1.0, 1.3]
}
fn d_max_regions() -> Vec<usize> {
    vec![16, 32, 64]
}
fn d_skill_weight() -> f64 {
    0.05
}
fn d_eval_max() -> usize {
    1200
}

impl Default for SearchSpec {
    fn default() -> Self {
        Self {
            candidates: d_candidates(),
            context_steps: Vec::new(),
            squash_scales: d_scales(),
            expert_weights: d_weights(),
            region_radii: d_radii(),
            max_regions: d_max_regions(),
            skill_weight: d_skill_weight(),
            eval_max: d_eval_max(),
        }
    }
}

impl SearchSpec {
    fn contexts(&self, ts: Timescale) -> Vec<u32> {
        if !self.context_steps.is_empty() {
            return self.context_steps.iter().map(|c| (*c).max(2)).collect();
        }
        let d = default_sizing(ts).context_steps;
        let mut v: Vec<u32> = [d / 4, d / 2, d].into_iter().map(|c| c.max(16)).collect();
        v.dedup();
        v
    }

    /// Candidate list: `anchor` first, then distinct draws from the grid.
    pub fn candidates(&self, ts: Timescale, anchor: &Candidate, seed: u64) -> Vec<Candidate> {
        let mut out = vec![anchor.clone()];
        let ctx = self.contexts(ts);
        let pick = |rng: &mut Rng, n: usize| rng.below(n.max(1));
        let mut rng = Rng::seed(seed ^ 0x5eed_fa5e_a5c4_0001);
        let mut tries = 0;
        while out.len() < self.candidates.max(1) && tries < 64 * self.candidates.max(1) {
            tries += 1;
            let c = Candidate {
                context_steps: ctx[pick(&mut rng, ctx.len())],
                squash_scale: self.squash_scales.get(pick(&mut rng, self.squash_scales.len())).copied().unwrap_or(1.0),
                expert_weights: (0..EXPERT_NAMES.len())
                    .map(|_| self.expert_weights.get(pick(&mut rng, self.expert_weights.len())).copied().unwrap_or(1.0))
                    .collect(),
                region_radius: self.region_radii.get(pick(&mut rng, self.region_radii.len())).copied().unwrap_or(1.0),
                max_regions: self.max_regions.get(pick(&mut rng, self.max_regions.len())).copied().unwrap_or(64),
            };
            if !out.contains(&c) {
                out.push(c);
            }
        }
        out
    }
}

/// Decoder skill reported for the parent snapshot (tie-breaker input).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ParentSkill {
    /// The value used (`crps_skill` mean, or `deflated_sharpe`).
    pub value: Option<f64>,
    /// `validation.trial` (keyed by snapshot_id) | `validation.generation_stats` (latest, not
    /// keyed by snapshot) | `none`.
    pub source: String,
    /// Whether the value is attributable to the parent snapshot.
    pub keyed: bool,
    pub n_trials: i64,
    #[serde(default)]
    pub detail: Value,
}

/// Region predictability of one candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evaluation {
    pub candidate: Candidate,
    /// Mean over experts of the information gain (nats / step).
    pub predictability: f64,
    pub per_expert: Vec<f64>,
    pub regions: Vec<usize>,
    pub n_test_transitions: usize,
    pub distance_to_parent: f64,
    pub penalty: f64,
    pub score: f64,
}

/// Search output: the chosen candidate, its statistics, and the trace for `manifest.feedback`.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub chosen: Candidate,
    /// Statistics fitted at the chosen `context_steps`, with gains applied.
    pub stats: NormStats,
    pub trace: Value,
}

/// Information gain of a Dirichlet transition model over the marginal region frequencies on
/// held-out transitions: the second half of every sequence is the test set.
pub fn predictability(sequences: &[Vec<usize>]) -> (f64, usize) {
    let key = |r: usize| format!("r{r}");
    let mut chain = MarkovGraph::new(0.5, 1.0);
    let mut marginal = MarkovGraph::new(0.5, 1.0);
    let states: BTreeSet<usize> = sequences.iter().flatten().copied().collect();
    for s in &states {
        chain.register_region(&key(*s));
        marginal.register_region(&key(*s));
    }
    let mut test = Vec::new();
    let mut test_marginal = Vec::new();
    for seq in sequences {
        let split = seq.len() / 2;
        for (i, w) in seq.windows(2).enumerate() {
            if i + 1 < split {
                chain.observe(&key(w[0]), &key(w[1]));
                marginal.observe("*", &key(w[1]));
            } else if i + 1 >= split {
                test.push((key(w[0]), key(w[1])));
                test_marginal.push(("*".to_owned(), key(w[1])));
            }
        }
    }
    if test.is_empty() {
        return (0.0, 0);
    }
    // Unseen rows fall back to the marginal so a source state without training rows is not
    // scored at the 1e-9 floor.
    let mut fallback = 0.0;
    let mut n = 0usize;
    for (pair, mpair) in test.iter().zip(&test_marginal) {
        let ls = if chain.row(&pair.0).is_some() {
            log_score(&chain, std::slice::from_ref(pair))
        } else {
            log_score(&marginal, std::slice::from_ref(mpair))
        };
        fallback += ls;
        n += 1;
    }
    let chain_ls = fallback / n as f64;
    let marg_ls = log_score(&marginal, &test_marginal);
    (chain_ls - marg_ls, test.len())
}

/// A statistical encoder with `stats` and `context_steps`, through the public manifest path
/// (exactly how model-runtime rebuilds a snapshot).
pub fn encoder(ts: Timescale, snapshot_id: &str, stats: &NormStats, context_steps: u32) -> StatisticalEncoder {
    let m = SnapshotManifest {
        snapshot_id: snapshot_id.into(),
        encoder_family: STATISTICAL_FAMILY.into(),
        timescale: ts,
        generation: 0,
        parent_snapshot_id: None,
        created_at: String::new(),
        params: json!({ "context_steps": context_steps }),
        probe_set: Default::default(),
        data: Default::default(),
        metrics: Default::default(),
        feedback: None,
        seed: 0,
    };
    StatisticalEncoder::from_manifest(&m, &stats.to_bytes()).expect("statistical manifest with 3x8 stats")
}

/// Per-instrument statistics at `context_steps` (the raw features depend on the context).
pub fn fit_stats(
    ts: Timescale,
    histories: &[BarHistory],
    ends: &[Vec<usize>],
    context_steps: u32,
) -> anyhow::Result<Vec<NormStats>> {
    let identity = NormStats {
        n_windows: 1,
        mean: vec![vec![0.0; fa_encoders::EXPERT_DIM]; EXPERT_NAMES.len()],
        std: vec![vec![1.0; fa_encoders::EXPERT_DIM]; EXPERT_NAMES.len()],
        gain: Vec::new(),
    };
    let enc = encoder(ts, "search", &identity, context_steps);
    histories
        .iter()
        .zip(ends)
        .map(|(h, e)| {
            let mut acc = NormAccumulator::default();
            enc.fit_into(&h.bars, e, &mut acc)?;
            Ok(acc.finish())
        })
        .collect()
}

/// Evaluation windows: per instrument up to `cap / n` window ends, evenly spaced and in order.
fn eval_ends(ends: &[Vec<usize>], cap: usize) -> Vec<Vec<usize>> {
    let per = (cap / ends.len().max(1)).max(8);
    ends.iter().map(|e| crate::fit::evenly(e.len(), per).into_iter().map(|i| e[i]).collect()).collect()
}

/// Scores one candidate on the evaluation windows.
pub fn evaluate(
    ts: Timescale,
    histories: &[BarHistory],
    eval: &[Vec<usize>],
    stats: &NormStats,
    c: &Candidate,
) -> anyhow::Result<(f64, Vec<f64>, Vec<usize>, usize)> {
    let enc = encoder(ts, "search", &stats.clone().with_gain(c.gains()), c.context_steps);
    let n_experts = EXPERT_NAMES.len();
    // latents[k] pooled over instruments (in order), lens = windows per instrument
    let mut latents: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_experts];
    let mut lens = Vec::with_capacity(histories.len());
    for (h, e) in histories.iter().zip(eval) {
        for &end in e {
            let view = enc.encode(&EncoderInput::from_bars(h.ends_ms[end - 1], &h.bars[..end]))?;
            for (k, x) in view.experts.into_iter().enumerate() {
                latents[k].push(x.mean);
            }
        }
        lens.push(e.len());
    }
    let disc = LeaderClustering { radius: c.region_radius, max_regions: c.max_regions };
    let mut per_expert = Vec::with_capacity(n_experts);
    let mut regions = Vec::with_capacity(n_experts);
    let mut n_test = 0;
    for points in &latents {
        let (centroids, assign) = disc.fit_assign(points);
        let mut seqs = Vec::with_capacity(lens.len());
        let mut at = 0;
        for &l in &lens {
            seqs.push(assign[at..at + l].iter().map(|a| a.unwrap_or(usize::MAX)).collect::<Vec<_>>());
            at += l;
        }
        let (gain, n) = predictability(&seqs);
        per_expert.push(gain);
        regions.push(centroids.len());
        n_test = n;
    }
    let mean = per_expert.iter().sum::<f64>() / per_expert.len().max(1) as f64;
    Ok((mean, per_expert, regions, n_test))
}

/// Runs the search. `ends[i]` are the fit window ends of `histories[i]`; `pooled_default` the
/// statistics already fitted at the default context (re-used for that context);
/// `equal_weight` pools per-instrument statistics with equal weights (`per_instrument_norm`).
#[allow(clippy::too_many_arguments)]
pub fn run(
    ts: Timescale,
    histories: &[BarHistory],
    ends: &[Vec<usize>],
    default_context: u32,
    per_instrument_default: &[NormStats],
    equal_weight: bool,
    anchor: &Candidate,
    spec: &SearchSpec,
    skill: &ParentSkill,
    seed: u64,
) -> anyhow::Result<Outcome> {
    let candidates = spec.candidates(ts, anchor, seed);
    let eval = eval_ends(ends, spec.eval_max);
    let mut stats_by_ctx: Vec<(u32, NormStats)> = Vec::new();
    let pooled = |parts: &[NormStats]| {
        NormStats::pool(parts, equal_weight).ok_or_else(|| anyhow::anyhow!("no windows to pool statistics from"))
    };
    stats_by_ctx.push((default_context, pooled(per_instrument_default)?));
    let skill_term = skill.value.filter(|v| v.is_finite()).map_or(0.0, |v| v.max(0.0));
    let mut evals = Vec::with_capacity(candidates.len());
    for c in &candidates {
        let stats = match stats_by_ctx.iter().find(|(ctx, _)| *ctx == c.context_steps) {
            Some((_, s)) => s.clone(),
            None => {
                let s = pooled(&fit_stats(ts, histories, ends, c.context_steps)?)?;
                stats_by_ctx.push((c.context_steps, s.clone()));
                s
            }
        };
        let (p, per_expert, regions, n_test) = evaluate(ts, histories, &eval, &stats, c)?;
        let distance = c.distance(anchor);
        let penalty = spec.skill_weight * skill_term * distance;
        evals.push(Evaluation {
            candidate: c.clone(),
            predictability: p,
            per_expert,
            regions,
            n_test_transitions: n_test,
            distance_to_parent: distance,
            penalty,
            score: p - penalty,
        });
    }
    // Highest score; ties → closer to the parent → earlier candidate.
    let best = evals
        .iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| {
            a.score.total_cmp(&b.score).then(b.distance_to_parent.total_cmp(&a.distance_to_parent)).then(j.cmp(i))
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let chosen = evals[best].candidate.clone();
    let stats = stats_by_ctx
        .iter()
        .find(|(ctx, _)| *ctx == chosen.context_steps)
        .map(|(_, s)| s.clone())
        .expect("chosen context was evaluated")
        .with_gain(chosen.gains());
    let trace = json!({
        "method": "deterministic hyper-parameter search (statistical family stand-in for backprop/distillation, spec 34 §6.6)",
        "proxy": "score = region predictability (held-out fa_mcmc::log_score of a Dirichlet transition model minus the marginal log score, nats/step, mean over experts) − skill_weight · max(parent_skill, 0) · distance(candidate, parent params)",
        "seed": seed,
        "eval_windows": eval.iter().map(Vec::len).sum::<usize>(),
        "parent_skill": skill,
        "skill_weight": spec.skill_weight,
        "anchor": anchor,
        "candidates": evals,
        "chosen_index": best,
        "chosen": chosen,
        "anchor_score": evals.first().map(|e| e.score),
        "chosen_score": evals[best].score,
    });
    Ok(Outcome { chosen, stats, trace })
}

/// Fit window ends with the default `FitConfig` for a history (shared with `fit`).
pub fn window_ends(h: &BarHistory, fit: &FitConfig) -> Vec<usize> {
    StatisticalEncoder::fit_window_ends(h.bars.len(), fit)
}

/// Default context of a timescale's encoder.
pub fn default_context(ts: Timescale) -> u32 {
    StatisticalEncoder::new(ts).sizing().context_steps
}

#[cfg(test)]
mod tests {
    use fa_encoders::MarketBar;

    use super::*;
    use crate::data::Source;

    /// Region sequences of a 3-state Markov chain that stays with probability `p`.
    fn chain(p: f64, n: usize, seed: u64) -> Vec<usize> {
        let mut rng = Rng::seed(seed);
        let mut s = 0usize;
        (0..n)
            .map(|_| {
                if rng.uniform() > p {
                    s = (s + 1 + rng.below(2)) % 3;
                }
                s
            })
            .collect()
    }

    #[test]
    fn predictability_is_monotone_in_chain_persistence() {
        let gains: Vec<f64> = [0.34, 0.6, 0.8, 0.95]
            .iter()
            .map(|p| predictability(&[chain(*p, 3000, 7), chain(*p, 3000, 8)]).0)
            .collect();
        assert!(gains[0].abs() < 0.02, "memoryless chain carries ~no information: {gains:?}");
        assert!(gains.windows(2).all(|w| w[1] > w[0]), "monotone in persistence: {gains:?}");
        // a single region carries no information either
        let (g, n) = predictability(&[vec![0; 100]]);
        assert!(g.abs() < 1e-9 && n > 0);
        assert_eq!(predictability(&[vec![1]]), (0.0, 0));
    }

    /// Bars whose volatility follows a 2-state regime chain with stay probability `p`.
    fn regime_bars(p: f64, n: usize, seed: u64) -> BarHistory {
        let mut rng = Rng::seed(seed);
        let mut price: f64 = 100.0;
        let mut hi = false;
        let mut bars = Vec::with_capacity(n);
        for _ in 0..n {
            if rng.uniform() > p {
                hi = !hi;
            }
            let vol = if hi { 0.01 } else { 0.0005 };
            let open = price;
            price *= 1.0 + vol * rng.normal();
            let (h, l) = (open.max(price) * (1.0 + vol), open.min(price) * (1.0 - vol));
            let volume = if hi { 5000.0 } else { 500.0 } * (1.0 + rng.uniform());
            bars.push(MarketBar { open, high: h, low: l, close: price, volume, trade_count: 1 });
        }
        BarHistory {
            instrument: format!("synthetic-{seed}"),
            timescale: Timescale::Minute,
            source: Source::Bars,
            start_ms: 0,
            end_ms: n as i64 * 60_000,
            ends_ms: (1..=n as i64).map(|i| i * 60_000).collect(),
            bars,
        }
    }

    fn setup(p: f64) -> (Vec<BarHistory>, Vec<Vec<usize>>, Vec<NormStats>) {
        let hs = vec![regime_bars(p, 500, 1), regime_bars(p, 500, 2)];
        let fit = FitConfig::default();
        let ends: Vec<Vec<usize>> = hs.iter().map(|h| window_ends(h, &fit)).collect();
        let stats = fit_stats(Timescale::Minute, &hs, &ends, 32).unwrap();
        (hs, ends, stats)
    }

    #[test]
    fn search_is_deterministic_and_records_its_trace() {
        let (hs, ends, stats) = setup(0.97);
        let anchor = Candidate { context_steps: 32, ..Candidate::defaults(Timescale::Minute, 1.0, 64) };
        let spec = SearchSpec { candidates: 5, context_steps: vec![16, 32], eval_max: 300, ..SearchSpec::default() };
        let skill =
            ParentSkill { value: Some(0.4), source: "test".into(), keyed: true, n_trials: 3, detail: Value::Null };
        let run1 = run(Timescale::Minute, &hs, &ends, 32, &stats, false, &anchor, &spec, &skill, 9).unwrap();
        let run2 = run(Timescale::Minute, &hs, &ends, 32, &stats, false, &anchor, &spec, &skill, 9).unwrap();
        assert_eq!(run1, run2, "same seed + data → same trace and choice");
        let cands = run1.trace["candidates"].as_array().unwrap();
        assert_eq!(cands.len(), 5);
        assert_eq!(cands[0]["candidate"], serde_json::to_value(&anchor).unwrap(), "candidate 0 = parent params");
        assert_eq!(cands[0]["penalty"], 0.0);
        let best = run1.trace["chosen_index"].as_u64().unwrap() as usize;
        let max = cands.iter().map(|c| c["score"].as_f64().unwrap()).fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(cands[best]["score"].as_f64().unwrap(), max);
        assert_eq!(run1.stats.gain.len() == 3, run1.chosen.gains().iter().any(|g| (*g - 1.0).abs() > 1e-9));
        // another seed explores other candidates
        let run3 = run(Timescale::Minute, &hs, &ends, 32, &stats, false, &anchor, &spec, &skill, 10).unwrap();
        assert_ne!(run3.trace["candidates"], run1.trace["candidates"]);
        // a large skill weight pins the choice to the parent's parameters
        let pinned = SearchSpec { skill_weight: 1e6, ..spec.clone() };
        let r = run(Timescale::Minute, &hs, &ends, 32, &stats, false, &anchor, &pinned, &skill, 9).unwrap();
        assert_eq!(r.chosen, anchor);
        // candidates are distinct and drawn from the grid
        let list = spec.candidates(Timescale::Minute, &anchor, 9);
        assert!(list[1..].iter().all(|c| [16, 32].contains(&c.context_steps)));
        assert!(list.iter().enumerate().all(|(i, c)| !list[..i].contains(c)), "distinct");
    }

    #[test]
    fn proxy_prefers_persistent_regimes() {
        let anchor =
            Candidate { context_steps: 16, region_radius: 0.8, ..Candidate::defaults(Timescale::Minute, 1.0, 16) };
        let score = |p: f64| {
            let (hs, ends, _) = setup(p);
            let stats = NormStats::pool(&fit_stats(Timescale::Minute, &hs, &ends, 16).unwrap(), false).unwrap();
            let eval = eval_ends(&ends, 600);
            evaluate(Timescale::Minute, &hs, &eval, &stats, &anchor).unwrap().0
        };
        let (lo, hi) = (score(0.5), score(0.98));
        assert!(hi > lo, "persistent volatility regimes are more predictable: {lo} vs {hi}");
        assert!(hi > 0.0);
    }

    #[test]
    fn candidate_helpers() {
        let a = Candidate::defaults(Timescale::Minute, 1.0, 64);
        assert_eq!(a.context_steps, 480);
        assert_eq!(a.gains(), vec![1.0, 1.0, 1.0]);
        assert_eq!(a.distance(&a), 0.0);
        let b = Candidate { squash_scale: 2.0, expert_weights: vec![1.0, 1.0, 2.0], ..a.clone() };
        assert_eq!(b.gains(), vec![0.5, 0.5, 1.0]);
        assert!(b.distance(&a) > 0.0 && (b.distance(&a) - a.distance(&b)).abs() < 1e-12);
        assert_eq!(SearchSpec::default().contexts(Timescale::Minute), vec![120, 240, 480]);
        assert_eq!(default_context(Timescale::Minute), 480);
    }
}
