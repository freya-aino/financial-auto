//! Sequential point-in-time evaluation of a [`SeriesDecoder`] (spec 41 §6, spec 34 §6.3).
//!
//! The decoder is asked for a forecast at every step `t` of the selection window (using only
//! data `≤ t`) and scored against the realised `horizon`-step log return. This is scoring,
//! not walk-forward *retraining* (which spec 41 excludes from evolution): the baseline
//! decoders have no trainable state. Reported objectives:
//!
//! | key | definition | direction |
//! |---|---|---|
//! | `crps` | mean CRPS (quantile approximation) of the forecast distribution | min |
//! | `crps_skill` | `1 − crps / crps_naive`, naive = zero-return point forecast (CRPS = mean |y|) | max |
//! | `coverage` | share of realised returns inside `[q_min, q_max]` | — |
//! | `coverage_err` | `|coverage − (level_max − level_min)|` (calibration) | min |
//! | `hit_rate` | share of steps where sign(point forecast) = sign(y) (y ≠ 0) | max |
//! | `trade_sharpe` | per-step Sharpe of the naive sign policy, net of `cost_bp` per unit turnover | max |
//! | `trade_psr` | probabilistic Sharpe ratio of that policy vs 0 | max |
//! | `compute_cost` | log10 of arithmetic ops per forecast (deterministic) | min |
//! | `n_eval` | steps scored | — |
//!
//! The provisional scalar `score` for the orchestrator (D-025) is `crps_skill`.

use std::time::Instant;

use fa_decoders::{
    baseline::{Series, SeriesDecoder, SeriesForecast},
    crps_quantiles,
};
use serde::{Deserialize, Serialize};

use crate::stats;

/// Evaluation controls.
#[derive(Default)]
pub struct EvalSpec {
    /// Transaction cost in basis points per unit of position change (naive policy).
    pub cost_bp: f64,
    /// Stop after this many steps (`None` = whole window).
    pub max_steps: Option<usize>,
    /// Stop when this returns `true` (budget / preemption). Checked every step.
    pub should_stop: Option<Box<dyn Fn(usize) -> bool + Send + Sync>>,
    /// Called every `progress_every` steps with `(done, total)`.
    pub on_progress: Option<Box<dyn Fn(usize, usize) + Send + Sync>>,
    pub progress_every: usize,
}

impl EvalSpec {
    /// 1 bp cost per unit turnover, progress every 64 steps, no limits.
    pub fn standard() -> Self {
        Self { cost_bp: 1.0, max_steps: None, should_stop: None, on_progress: None, progress_every: 64 }
    }
}

/// Why the evaluation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stop {
    /// Window exhausted.
    Complete,
    /// `max_steps` reached.
    MaxSteps,
    /// `should_stop` fired (budget or preemption).
    Interrupted,
    /// Not enough data for a single forecast.
    NoData,
}

/// Result of evaluating one decoder on one series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub instrument_id: String,
    pub n_series: usize,
    pub n_eval: usize,
    pub first_step: usize,
    pub last_step: usize,
    pub stop: Stop,
    pub crps: f64,
    pub crps_naive: f64,
    pub crps_skill: f64,
    pub coverage: f64,
    pub nominal_coverage: f64,
    pub coverage_err: f64,
    pub hit_rate: f64,
    pub trade_mean: f64,
    pub trade_sharpe: f64,
    pub trade_psr: f64,
    pub compute_cost: f64,
    /// Wall-clock ms spent (not deterministic; reported, never used for ranking).
    pub elapsed_ms: f64,
    /// Forecast made at the last evaluated step (for `DecoderOutput`).
    #[serde(skip)]
    pub last_forecast: Option<SeriesForecast>,
    /// Timestamp (ms) of the last evaluated step.
    pub last_ts_ms: i64,
}

impl EvalReport {
    /// Objectives as `name → value`, all finite (NaN → neutral 0).
    pub fn objectives(&self) -> Vec<(&'static str, f64)> {
        let f = |x: f64| if x.is_finite() { x } else { 0.0 };
        vec![
            ("crps", f(self.crps)),
            ("crps_skill", f(self.crps_skill)),
            ("coverage", f(self.coverage)),
            ("coverage_err", f(self.coverage_err)),
            ("hit_rate", f(self.hit_rate)),
            ("trade_sharpe", f(self.trade_sharpe)),
            ("trade_psr", f(self.trade_psr)),
            ("compute_cost", f(self.compute_cost)),
            ("n_eval", self.n_eval as f64),
        ]
    }

    /// Provisional scalar for the orchestrator (D-025): CRPS skill, higher is better.
    pub fn score(&self) -> f64 {
        if self.crps_skill.is_finite() { self.crps_skill } else { f64::NEG_INFINITY }
    }
}

/// Runs the decoder over `series` from its warm-up to the last step whose outcome is known.
pub fn evaluate(decoder: &dyn SeriesDecoder, series: &Series, spec: &EvalSpec) -> EvalReport {
    let started = Instant::now();
    let g = decoder.genome();
    let h = g.horizon;
    let first = decoder.warmup();
    let n = series.len();
    let mut rep = EvalReport {
        instrument_id: series.instrument_id.clone(),
        n_series: n,
        n_eval: 0,
        first_step: first,
        last_step: first,
        stop: Stop::NoData,
        crps: f64::NAN,
        crps_naive: f64::NAN,
        crps_skill: f64::NAN,
        coverage: f64::NAN,
        nominal_coverage: g.levels.last().copied().unwrap_or(1.0) - g.levels.first().copied().unwrap_or(0.0),
        coverage_err: f64::NAN,
        hit_rate: f64::NAN,
        trade_mean: f64::NAN,
        trade_sharpe: f64::NAN,
        trade_psr: f64::NAN,
        compute_cost: decoder.ops_per_forecast(n).max(1.0).log10(),
        elapsed_ms: 0.0,
        last_forecast: None,
        last_ts_ms: series.ts_ms.last().copied().unwrap_or(0),
    };
    if n <= first + h {
        rep.elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        return rep;
    }
    let last = n - 1 - h;
    let total = (last + 1 - first).min(spec.max_steps.unwrap_or(usize::MAX));

    let (mut crps_sum, mut naive_sum, mut covered, mut hits, mut signed) = (0.0, 0.0, 0usize, 0usize, 0usize);
    let mut pnl: Vec<f64> = Vec::with_capacity(total);
    let mut pos = 0.0;
    let mut stop = Stop::Complete;
    let mut done = 0usize;
    for t in first..=last {
        if done >= total {
            stop = Stop::MaxSteps;
            break;
        }
        if spec.should_stop.as_ref().is_some_and(|f| f(done)) {
            stop = Stop::Interrupted;
            break;
        }
        let Some(f) = decoder.forecast(series, t) else {
            continue;
        };
        let Some(y) = series.forward_return(t, h) else {
            break;
        };
        crps_sum += crps_quantiles(&f.quantiles, &f.levels, y);
        naive_sum += y.abs();
        let lo = f.quantiles.first().copied().unwrap_or(f.point);
        let hi = f.quantiles.last().copied().unwrap_or(f.point);
        if (lo..=hi).contains(&y) {
            covered += 1;
        }
        if y != 0.0 && f.point != 0.0 {
            signed += 1;
            if (y > 0.0) == (f.point > 0.0) {
                hits += 1;
            }
        }
        let new_pos = f.point.signum();
        let cost = (new_pos - pos).abs() * spec.cost_bp * 1e-4;
        pnl.push(new_pos * y - cost);
        pos = new_pos;
        done += 1;
        rep.last_step = t;
        rep.last_ts_ms = series.ts_ms[t];
        rep.last_forecast = Some(f);
        if done.is_multiple_of(spec.progress_every.max(1))
            && let Some(cb) = &spec.on_progress
        {
            cb(done, total);
        }
    }
    rep.n_eval = done;
    rep.stop = if done == 0 { Stop::NoData } else { stop };
    if done > 0 {
        let d = done as f64;
        rep.crps = crps_sum / d;
        rep.crps_naive = naive_sum / d;
        rep.crps_skill = if rep.crps_naive > 0.0 { 1.0 - rep.crps / rep.crps_naive } else { 0.0 };
        rep.coverage = covered as f64 / d;
        rep.coverage_err = (rep.coverage - rep.nominal_coverage).abs();
        rep.hit_rate = if signed > 0 { hits as f64 / signed as f64 } else { 0.5 };
        rep.trade_mean = stats::mean(&pnl);
        rep.trade_sharpe = stats::sharpe(&pnl);
        rep.trade_psr = if pnl.len() > 2 { stats::probabilistic_sharpe(&pnl, 0.0) } else { 0.5 };
    }
    rep.elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
    rep
}

/// Pools per-instrument reports into one, weighting means by `n_eval`.
pub fn pool(reports: &[EvalReport]) -> Option<EvalReport> {
    let with_data: Vec<&EvalReport> = reports.iter().filter(|r| r.n_eval > 0).collect();
    let total: usize = with_data.iter().map(|r| r.n_eval).sum();
    if total == 0 {
        return None;
    }
    let w = |f: fn(&EvalReport) -> f64| with_data.iter().map(|r| f(r) * r.n_eval as f64).sum::<f64>() / total as f64;
    let mut out = with_data[0].clone();
    out.instrument_id =
        if with_data.len() == 1 { out.instrument_id } else { format!("{} instruments", with_data.len()) };
    out.n_series = with_data.iter().map(|r| r.n_series).sum();
    out.n_eval = total;
    out.stop = with_data.iter().map(|r| r.stop).find(|s| *s == Stop::Interrupted).unwrap_or(Stop::Complete);
    out.crps = w(|r| r.crps);
    out.crps_naive = w(|r| r.crps_naive);
    out.crps_skill = if out.crps_naive > 0.0 { 1.0 - out.crps / out.crps_naive } else { 0.0 };
    out.coverage = w(|r| r.coverage);
    out.coverage_err = (out.coverage - out.nominal_coverage).abs();
    out.hit_rate = w(|r| r.hit_rate);
    out.trade_mean = w(|r| r.trade_mean);
    out.trade_sharpe = w(|r| r.trade_sharpe);
    out.trade_psr = w(|r| r.trade_psr);
    out.compute_cost = w(|r| r.compute_cost);
    out.elapsed_ms = with_data.iter().map(|r| r.elapsed_ms).sum();
    out.last_forecast = None;
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fa_core::rng::Rng;
    use fa_decoders::{baseline::build, genome::DecoderGenome};

    /// SYNTHETIC: random walk with a weak positive drift; not market data.
    fn walk(n: usize, seed: u64, drift: f64) -> Series {
        let mut rng = Rng::seed(seed);
        let mut p = 100.0;
        let trades: Vec<(i64, f64)> = (0..n)
            .map(|i| {
                p *= 1.0 + drift + 0.002 * rng.normal();
                (i as i64 * 1000, p)
            })
            .collect();
        Series::resample("synthetic", &trades, 1000)
    }

    #[test]
    fn evaluation_is_deterministic_and_reasonable() {
        let s = walk(500, 3, 0.0);
        let d = build(&DecoderGenome::parse("forecast:h=3,lb=32").unwrap());
        let a = evaluate(d.as_ref(), &s, &EvalSpec::standard());
        let b = evaluate(d.as_ref(), &s, &EvalSpec::standard());
        assert_eq!(a.n_eval, b.n_eval);
        assert_eq!(a.crps, b.crps);
        assert_eq!(a.stop, Stop::Complete);
        assert_eq!(a.n_eval, 500 - 3 - d.warmup());
        assert!(a.crps > 0.0 && a.crps_skill > -1.0 && a.crps_skill <= 1.0, "{a:?}");
        assert!((0.0..=1.0).contains(&a.coverage));
        assert!(a.coverage > 0.6, "90% interval should cover most steps, got {}", a.coverage);
        assert!(a.compute_cost > 0.0);
        assert!(a.objectives().iter().all(|(_, v)| v.is_finite()));
    }

    #[test]
    fn stop_hooks_and_max_steps() {
        let s = walk(400, 4, 0.0);
        let d = build(&DecoderGenome::parse("retrieval:h=2,lb=8,k=5").unwrap());
        let r = evaluate(d.as_ref(), &s, &EvalSpec { max_steps: Some(10), ..EvalSpec::standard() });
        assert_eq!((r.n_eval, r.stop), (10, Stop::MaxSteps));
        let spec = EvalSpec { should_stop: Some(Box::new(|done| done >= 3)), ..EvalSpec::standard() };
        let r = evaluate(d.as_ref(), &s, &spec);
        assert_eq!((r.n_eval, r.stop), (3, Stop::Interrupted));
        assert!(r.last_forecast.as_ref().is_some_and(|f| !f.analogs.is_empty()));
    }

    #[test]
    fn no_data_report_is_marked() {
        let s = walk(10, 5, 0.0);
        let d = build(&DecoderGenome::parse("forecast:h=3,lb=32").unwrap());
        let r = evaluate(d.as_ref(), &s, &EvalSpec::standard());
        assert_eq!((r.n_eval, r.stop), (0, Stop::NoData));
        assert_eq!(r.score(), f64::NEG_INFINITY);
        assert!(pool(&[r]).is_none());
    }

    #[test]
    fn pooling_weights_by_steps() {
        let d = build(&DecoderGenome::parse("forecast:h=2,lb=16").unwrap());
        let a = evaluate(d.as_ref(), &walk(200, 6, 0.0), &EvalSpec::standard());
        let b = evaluate(d.as_ref(), &walk(400, 7, 0.0), &EvalSpec::standard());
        let p = pool(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(p.n_eval, a.n_eval + b.n_eval);
        let expect = (a.crps * a.n_eval as f64 + b.crps * b.n_eval as f64) / p.n_eval as f64;
        assert!((p.crps - expect).abs() < 1e-12);
    }
}
