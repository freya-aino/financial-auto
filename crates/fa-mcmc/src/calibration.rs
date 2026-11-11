//! Calibration suite and sampler competition (spec 33 §6.4).
//!
//! Scoring is **prequential**: before a sampler observes a transition it predicts the next
//! region, and the realised region is scored against that prediction. Scores are
//! exponentially weighted (`decay` per evaluation) so they follow the current regime:
//!
//! - **log score**: mean `ln p(realised)` (higher is better);
//! - **Brier**: `Σ_k (p_k − 1[k = realised])²` (lower is better);
//! - **ECE**: expected calibration error over 10 probability bins of every `(p_k, outcome)`
//!   pair (lower is better; a reliability diagram is available from [`Calibration::reliability`]);
//! - **PIT KS**: Kolmogorov–Smirnov distance from uniform of the randomised PIT of the
//!   realised category (categories ordered by predicted probability), over the last 512
//!   evaluations;
//! - **cost**: wall-clock nanoseconds per predict + observe (EWMA).
//!
//! A realised region that is missing from a prediction is scored as `unknown` (the
//! "somewhere not seen yet" successor every row keeps).
//!
//! [`Arena`] runs several [`KernelSampler`]s on the same stream. The Pareto front over
//! (log score ↑, ECE ↓, cost ↓) is computed from the scores; the serving sampler is the front
//! member with the best log score among those with ECE ≤ `max_ece` (any front member when none
//! qualifies), switched only after `min_evals` evaluations and when it beats the current one by
//! `margin` nats (hysteresis).

use std::collections::VecDeque;

use fa_core::rng::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    RegionKey, UNKNOWN,
    kernel::{KernelSampler, timed},
};

const BINS: usize = 10;
const PIT_WINDOW: usize = 512;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    decay: f64,
    weight: f64,
    log: f64,
    brier: f64,
    /// per bin: (Σ w·p, Σ w·outcome, Σ w)
    bins: [(f64, f64, f64); BINS],
    pit: VecDeque<f64>,
    cost_ns: f64,
    pub evals: u64,
}

/// Snapshot of a [`Calibration`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationScores {
    pub log_score: f64,
    pub brier: f64,
    pub ece: f64,
    pub pit_ks: f64,
    pub cost_us: f64,
    pub evals: u64,
}

impl Calibration {
    pub fn new(decay: f64) -> Self {
        Self {
            decay: decay.clamp(0.0, 1.0),
            weight: 0.0,
            log: 0.0,
            brier: 0.0,
            bins: [(0.0, 0.0, 0.0); BINS],
            pit: VecDeque::with_capacity(PIT_WINDOW),
            cost_ns: 0.0,
            evals: 0,
        }
    }

    /// Scores one prediction against the realised region; `cost_ns < 0` records no cost.
    pub fn record(&mut self, pred: &[(RegionKey, f64)], realised: &str, cost_ns: f64, rng: &mut Rng) {
        let hit = if pred.iter().any(|(k, _)| k == realised) { realised } else { UNKNOWN };
        let p_real = pred.iter().find(|(k, _)| k == hit).map_or(0.0, |x| x.1);
        let d = self.decay;
        self.weight = self.weight * d + 1.0;
        let w = 1.0 / self.weight;
        self.log += w * (p_real.max(1e-9).ln() - self.log);
        let mut brier: f64 = pred.iter().map(|(k, p)| if k == hit { (p - 1.0).powi(2) } else { p * p }).sum();
        if !pred.iter().any(|(k, _)| k == hit) {
            brier += 1.0;
        }
        self.brier += w * (brier - self.brier);
        for b in &mut self.bins {
            b.0 *= d;
            b.1 *= d;
            b.2 *= d;
        }
        for (k, p) in pred {
            let i = ((p * BINS as f64) as usize).min(BINS - 1);
            self.bins[i].0 += p;
            self.bins[i].1 += if k == hit { 1.0 } else { 0.0 };
            self.bins[i].2 += 1.0;
        }
        // randomised PIT over categories ordered by descending probability (ties by key)
        let mut ordered: Vec<&(RegionKey, f64)> = pred.iter().collect();
        ordered.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut below = 0.0;
        for (k, p) in ordered {
            if k == hit {
                break;
            }
            below += p;
        }
        let u = (below + rng.uniform() * p_real).clamp(0.0, 1.0);
        if self.pit.len() == PIT_WINDOW {
            self.pit.pop_front();
        }
        self.pit.push_back(u);
        self.evals += 1;
        if cost_ns >= 0.0 {
            self.add_cost(cost_ns);
        }
    }

    /// Folds one cost measurement (ns) into the EWMA.
    pub fn add_cost(&mut self, cost_ns: f64) {
        self.cost_ns = if self.cost_ns == 0.0 { cost_ns } else { 0.98 * self.cost_ns + 0.02 * cost_ns };
    }

    /// `(mean predicted, observed frequency, weight)` per bin.
    pub fn reliability(&self) -> Vec<(f64, f64, f64)> {
        self.bins.iter().filter(|b| b.2 > 0.0).map(|b| (b.0 / b.2, b.1 / b.2, b.2)).collect()
    }

    pub fn scores(&self) -> CalibrationScores {
        let total: f64 = self.bins.iter().map(|b| b.2).sum();
        let ece = if total > 0.0 {
            self.bins.iter().filter(|b| b.2 > 0.0).map(|b| (b.2 / total) * ((b.0 - b.1) / b.2).abs()).sum()
        } else {
            0.0
        };
        let mut u: Vec<f64> = self.pit.iter().copied().collect();
        u.sort_by(f64::total_cmp);
        let n = u.len() as f64;
        let pit_ks = u
            .iter()
            .enumerate()
            .map(|(i, x)| ((i + 1) as f64 / n - x).abs().max((x - i as f64 / n).abs()))
            .fold(0.0, f64::max);
        CalibrationScores {
            log_score: self.log,
            brier: self.brier,
            ece,
            pit_ks,
            cost_us: self.cost_ns / 1000.0,
            evals: self.evals,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ArenaCfg {
    /// Evaluations before the serving sampler may change.
    pub min_evals: u64,
    /// Log-score advantage (nats) required to switch.
    pub margin: f64,
    /// Calibration ceiling for serving.
    pub max_ece: f64,
    /// Score decay per evaluation.
    pub decay: f64,
}

impl Default for ArenaCfg {
    fn default() -> Self {
        Self { min_evals: 200, margin: 0.01, max_ece: 0.1, decay: 0.999 }
    }
}

/// Score of one sampler in the arena.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplerStanding {
    pub sampler_id: String,
    pub scores: CalibrationScores,
    pub pareto: bool,
    pub serving: bool,
}

/// Several samplers on one stream; one serves (spec 33 §6.4).
pub struct Arena {
    samplers: Vec<Box<dyn KernelSampler>>,
    cal: Vec<Calibration>,
    serving: usize,
    cfg: ArenaCfg,
    rng: Rng,
    /// Serving changes so far.
    pub switches: u64,
}

impl Arena {
    /// `samplers[0]` serves first.
    pub fn new(samplers: Vec<Box<dyn KernelSampler>>, cfg: ArenaCfg, seed: u64) -> Self {
        assert!(!samplers.is_empty(), "an arena needs a sampler");
        let cal = samplers.iter().map(|_| Calibration::new(cfg.decay)).collect();
        Self { samplers, cal, serving: 0, cfg, rng: Rng::seed(seed), switches: 0 }
    }

    pub fn serving(&self) -> &dyn KernelSampler {
        self.samplers[self.serving].as_ref()
    }

    pub fn serving_mut(&mut self) -> &mut dyn KernelSampler {
        self.samplers[self.serving].as_mut()
    }

    pub fn samplers(&self) -> impl Iterator<Item = &dyn KernelSampler> {
        self.samplers.iter().map(|s| s.as_ref())
    }

    pub fn register_region(&mut self, id: &str) {
        for s in &mut self.samplers {
            s.register_region(id);
        }
    }

    pub fn retire_region(&mut self, id: &str) {
        for s in &mut self.samplers {
            s.retire_region(id);
        }
    }

    pub fn set_centroid(&mut self, id: &str, c: &[f32]) {
        for s in &mut self.samplers {
            s.graph_mut().set_centroid(id, c);
        }
    }

    /// Replaces every sampler's evidence (kernel transfer); scores are kept.
    pub fn replace_graph(&mut self, g: &crate::MarkovGraph) {
        for s in &mut self.samplers {
            s.replace_graph(g.clone());
        }
    }

    /// Prequential step: every sampler predicts, is scored on `to`, then observes it.
    pub fn observe(&mut self, from: &str, to: &str) {
        for (s, c) in self.samplers.iter_mut().zip(&mut self.cal) {
            let ((), ns) = timed(|| {
                let pred = s.predict(from);
                s.observe(from, to);
                c.record(&pred, to, -1.0, &mut self.rng);
            });
            // cost is attributed after the fact (the closure cannot time itself)
            c.add_cost(ns);
        }
        self.reselect();
    }

    fn dominates(a: &CalibrationScores, b: &CalibrationScores) -> bool {
        let ge = a.log_score >= b.log_score && a.ece <= b.ece && a.cost_us <= b.cost_us;
        let gt = a.log_score > b.log_score || a.ece < b.ece || a.cost_us < b.cost_us;
        ge && gt
    }

    fn front(&self) -> Vec<bool> {
        let s: Vec<CalibrationScores> = self.cal.iter().map(Calibration::scores).collect();
        (0..s.len()).map(|i| !(0..s.len()).any(|j| j != i && Self::dominates(&s[j], &s[i]))).collect()
    }

    fn reselect(&mut self) {
        if self.cal.iter().any(|c| c.evals < self.cfg.min_evals) {
            return;
        }
        let front = self.front();
        let s: Vec<CalibrationScores> = self.cal.iter().map(Calibration::scores).collect();
        let pick = |strict: bool| {
            (0..s.len())
                .filter(|&i| front[i] && (!strict || s[i].ece <= self.cfg.max_ece))
                .max_by(|&a, &b| s[a].log_score.total_cmp(&s[b].log_score))
        };
        let Some(best) = pick(true).or_else(|| pick(false)) else { return };
        if best != self.serving && s[best].log_score > s[self.serving].log_score + self.cfg.margin {
            self.serving = best;
            self.switches += 1;
        }
    }

    pub fn standings(&self) -> Vec<SamplerStanding> {
        let front = self.front();
        self.samplers
            .iter()
            .zip(&self.cal)
            .enumerate()
            .map(|(i, (s, c))| SamplerStanding {
                sampler_id: s.name().to_owned(),
                scores: c.scores(),
                pareto: front[i],
                serving: i == self.serving,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        MarkovGraph,
        kernel::{ConjugateExact, SmcParticle},
    };

    #[test]
    fn calibrated_predictions_score_well_and_overconfident_ones_do_not() {
        let mut rng = Rng::seed(1);
        let mut good = Calibration::new(1.0);
        let mut bad = Calibration::new(1.0);
        let truth = [("a".to_owned(), 0.7), ("b".to_owned(), 0.3)];
        let over = [("a".to_owned(), 0.99), ("b".to_owned(), 0.01)];
        for _ in 0..4000 {
            let real = if rng.uniform() < 0.7 { "a" } else { "b" };
            good.record(&truth, real, 100.0, &mut rng);
            bad.record(&over, real, 100.0, &mut rng);
        }
        let (g, b) = (good.scores(), bad.scores());
        assert!(g.log_score > b.log_score);
        assert!(g.brier < b.brier);
        assert!(g.ece < 0.03 && b.ece > 0.2, "{g:?} {b:?}");
        assert!(g.pit_ks < 0.1, "calibrated PIT is uniform: {g:?}");
        assert!((g.cost_us - 0.1).abs() < 1e-9);
        assert!(!good.reliability().is_empty());
        // an unseen region counts as `unknown`
        let mut c = Calibration::new(1.0);
        c.record(&[("a".into(), 0.5), (UNKNOWN.into(), 0.5)], "zzz", 0.0, &mut rng);
        assert!((c.scores().log_score - 0.5f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn arena_scores_both_samplers_and_serves_one() {
        let samplers: Vec<Box<dyn KernelSampler>> = vec![
            Box::new(ConjugateExact::new(MarkovGraph::new(1.0, 0.999), 64, 1)),
            Box::new(SmcParticle::new(MarkovGraph::new(1.0, 0.999), 64, 2)),
        ];
        let mut arena = Arena::new(samplers, ArenaCfg { min_evals: 50, ..ArenaCfg::default() }, 3);
        let mut rng = Rng::seed(4);
        let mut cur = "a";
        for _ in 0..800 {
            let next = match (cur, rng.uniform()) {
                ("a", u) if u < 0.8 => "a",
                ("a", _) => "b",
                (_, u) if u < 0.6 => "a",
                _ => "b",
            };
            arena.observe(cur, next);
            cur = next;
        }
        let st = arena.standings();
        assert_eq!(st.len(), 2);
        assert_eq!(st.iter().filter(|s| s.serving).count(), 1);
        assert!(st.iter().any(|s| s.pareto));
        for s in &st {
            assert_eq!(s.scores.evals, 800);
            assert!(s.scores.log_score > -1.0 && s.scores.log_score < 0.0, "{s:?}");
            assert!(s.scores.ece < 0.1, "{s:?}");
            assert!(s.scores.cost_us > 0.0);
        }
        assert!(arena.serving().graph().row_mass("a") > 0.0);
    }
}
