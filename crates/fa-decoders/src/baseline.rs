//! Baseline, CPU-only, deterministic decoders over a single price series (spec 34 §6.1,
//! provisional). They need no ML backend; they exist so the evolution loop can be exercised
//! end to end with real scores. Optional conditioning: [`build_with`] hands the retrieval
//! decoder a [`DecodeCtx`] of `latent.state.*` latents, which it uses as an extra similarity
//! term (gene `lat`) looked up **as of** each window end, so no future latent leaks in.
//!
//! Both decoders are **point-in-time**: `forecast(series, t)` only reads `series[..=t]` (and
//! latents with `as_of ≤ ts[t]`).

use std::sync::Arc;

use crate::{
    Analog, DecodeCtx, analog_forecast, analog_weight, cosine,
    genome::{BaselineFamily, DecoderGenome, Metric},
};

/// A time-ordered, evenly resampled log-price series.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Series {
    pub instrument_id: String,
    /// Timestamps (ms) of each step.
    pub ts_ms: Vec<i64>,
    /// Natural log of the price at each step.
    pub log_price: Vec<f64>,
    /// Step length in ms (timescale).
    pub step_ms: i64,
}

impl Series {
    pub fn len(&self) -> usize {
        self.log_price.len()
    }

    pub fn is_empty(&self) -> bool {
        self.log_price.is_empty()
    }

    /// Realised log return from `t` to `t + h`, if available.
    pub fn forward_return(&self, t: usize, h: usize) -> Option<f64> {
        let end = t.checked_add(h)?;
        (end < self.len()).then(|| self.log_price[end] - self.log_price[t])
    }

    /// Builds a series from `(ts_ms, price)` trades by taking the last price per
    /// `step_ms` bucket. Gaps (empty buckets) are skipped, not filled.
    pub fn resample(instrument_id: &str, trades: &[(i64, f64)], step_ms: i64) -> Self {
        let step_ms = step_ms.max(1);
        let mut ts_ms = Vec::new();
        let mut log_price = Vec::new();
        let mut cur_bucket: Option<i64> = None;
        for &(t, p) in trades {
            if p.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                continue;
            }
            let bucket = t.div_euclid(step_ms);
            if cur_bucket == Some(bucket) {
                *ts_ms.last_mut().unwrap() = t;
                *log_price.last_mut().unwrap() = p.ln();
            } else if cur_bucket.is_none_or(|b| bucket > b) {
                cur_bucket = Some(bucket);
                ts_ms.push(t);
                log_price.push(p.ln());
            }
            // out-of-order older trades are ignored
        }
        Self { instrument_id: instrument_id.to_owned(), ts_ms, log_price, step_ms }
    }
}

/// Forecast of the `horizon`-step forward log return made at one step.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SeriesForecast {
    pub quantiles: Vec<f64>,
    pub levels: Vec<f64>,
    /// Point forecast (median unless the family defines otherwise).
    pub point: f64,
    /// Retrieval family only.
    pub analogs: Vec<Analog>,
}

/// A decoder that forecasts a series from its own past.
pub trait SeriesDecoder: Send + Sync {
    fn genome(&self) -> &DecoderGenome;
    /// Earliest step index at which a forecast is possible.
    fn warmup(&self) -> usize;
    /// Forecast at step `t` using only `series[..=t]`. `None` if `t < warmup()`.
    fn forecast(&self, series: &Series, t: usize) -> Option<SeriesForecast>;
    /// Relative inference cost (arithmetic operations per forecast, order of magnitude),
    /// used as the `compute_cost` objective's deterministic component.
    fn ops_per_forecast(&self, series_len: usize) -> f64;
}

/// Instantiates the decoder a genome describes, without latent conditioning.
pub fn build(genome: &DecoderGenome) -> Box<dyn SeriesDecoder> {
    build_with(genome, Arc::new(DecodeCtx::default()))
}

/// Instantiates the decoder with a conditioning context (used by the retrieval family).
pub fn build_with(genome: &DecoderGenome, ctx: Arc<DecodeCtx>) -> Box<dyn SeriesDecoder> {
    match genome.family {
        BaselineFamily::Forecast => Box::new(QuantileForecaster { genome: genome.clone() }),
        BaselineFamily::Retrieval => Box::new(AnalogRetriever { genome: genome.clone(), ctx }),
    }
}

/// Weighted empirical quantile of `(value, weight)` pairs (linear interpolation on the
/// cumulative weight). `pairs` are sorted in place by value.
pub fn weighted_quantile(pairs: &mut [(f64, f64)], level: f64) -> Option<f64> {
    if pairs.is_empty() {
        return None;
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = pairs.iter().map(|p| p.1).sum();
    if total.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return None;
    }
    let target = level.clamp(0.0, 1.0) * total;
    let mut cum = 0.0;
    for (i, &(v, w)) in pairs.iter().enumerate() {
        let prev = cum;
        cum += w;
        if cum >= target {
            if i == 0 || w <= 0.0 {
                return Some(v);
            }
            let frac = ((target - prev) / w).clamp(0.0, 1.0);
            let (pv, _) = pairs[i - 1];
            return Some(pv + (v - pv) * frac);
        }
    }
    pairs.last().map(|p| p.0)
}

// ------------------------------------------------------------------ forecast

/// Empirical quantile forecaster: the distribution of the next `h`-step log return is the
/// exponentially weighted empirical distribution of past overlapping `h`-step returns in
/// the lookback, centred on a shrunk EW drift and stretched by `scale`.
pub struct QuantileForecaster {
    genome: DecoderGenome,
}

impl SeriesDecoder for QuantileForecaster {
    fn genome(&self) -> &DecoderGenome {
        &self.genome
    }

    fn warmup(&self) -> usize {
        self.genome.lookback + self.genome.horizon
    }

    fn forecast(&self, s: &Series, t: usize) -> Option<SeriesForecast> {
        let g = &self.genome;
        let h = g.horizon;
        if t < self.warmup() || t >= s.len() {
            return None;
        }
        // Overlapping h-step returns ending at or before t.
        let first_end = t + 1 - g.lookback;
        let decay = (2f64).ln() / g.kernel;
        let mut pairs: Vec<(f64, f64)> = (first_end..=t)
            .map(|end| {
                let r = s.log_price[end] - s.log_price[end - h];
                let age = (t - end) as f64;
                (r, (-decay * age).exp())
            })
            .collect();
        let wsum: f64 = pairs.iter().map(|p| p.1).sum();
        let mean = pairs.iter().map(|(r, w)| r * w).sum::<f64>() / wsum;
        let drift = g.shrink * mean;
        let quantiles: Vec<f64> = g
            .levels
            .iter()
            .map(|&l| {
                let q = weighted_quantile(&mut pairs, l).unwrap_or(mean);
                drift + g.scale * (q - mean)
            })
            .collect();
        let point = weighted_quantile(&mut pairs, 0.5).map(|m| drift + g.scale * (m - mean)).unwrap_or(drift);
        Some(SeriesForecast { quantiles, levels: g.levels.clone(), point, analogs: Vec::new() })
    }

    fn ops_per_forecast(&self, _series_len: usize) -> f64 {
        let lb = self.genome.lookback as f64;
        lb * (self.genome.levels.len() as f64 + lb.log2().max(1.0))
    }
}

// ----------------------------------------------------------------- retrieval

/// Analog retrieval: the query is the z-scored shape of the last `lookback` returns; every
/// earlier window whose outcome is fully known (`min_gap` leakage guard) is a candidate;
/// the `k` best by `similarity · exp(−Δt/τ)` vote, weighted, on what followed them.
/// With latents in the context for the instrument, `similarity` becomes
/// `(1 − lat) · shape + lat · cos(latent as of query, latent as of candidate)` whenever both
/// latents exist (otherwise the shape similarity alone).
pub struct AnalogRetriever {
    genome: DecoderGenome,
    ctx: Arc<DecodeCtx>,
}

impl AnalogRetriever {
    /// Whether the context can condition this series at all.
    pub fn conditioned(&self, s: &Series) -> bool {
        self.genome.latent_weight > 0.0 && self.ctx.latents.contains_key(&crate::instrument_token(&s.instrument_id))
    }

    fn combined_similarity(
        &self,
        s: &Series,
        query_shape: &[f64],
        query_latent: Option<&[f32]>,
        e: usize,
        shape: &[f64],
    ) -> f64 {
        let base = self.similarity(query_shape, shape);
        let Some(q) = query_latent else { return base };
        match self.ctx.latent_as_of(&s.instrument_id, s.ts_ms[e]) {
            Some(l) => {
                let w = self.genome.latent_weight;
                (1.0 - w) * base + w * cosine(q, &l.vec)
            }
            None => base,
        }
    }

    fn shape(s: &Series, end: usize, lb: usize) -> Option<Vec<f64>> {
        if end < lb {
            return None;
        }
        let rets: Vec<f64> = (end - lb + 1..=end).map(|i| s.log_price[i] - s.log_price[i - 1]).collect();
        let m = rets.iter().sum::<f64>() / lb as f64;
        let var = rets.iter().map(|r| (r - m) * (r - m)).sum::<f64>() / lb as f64;
        let sd = var.sqrt();
        Some(if sd > 0.0 { rets.iter().map(|r| (r - m) / sd).collect() } else { vec![0.0; lb] })
    }

    fn similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        match self.genome.metric {
            Metric::Cosine => {
                let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
                let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
                if na > 0.0 && nb > 0.0 { (dot / (na * nb)).clamp(-1.0, 1.0) } else { 0.0 }
            }
            Metric::Euclid => {
                let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
                1.0 / (1.0 + d / (a.len() as f64).sqrt())
            }
        }
    }
}

impl SeriesDecoder for AnalogRetriever {
    fn genome(&self) -> &DecoderGenome {
        &self.genome
    }

    fn warmup(&self) -> usize {
        // query window + at least two fully-resolved candidate windows behind the gap
        2 * self.genome.lookback + self.genome.min_gap + self.genome.horizon + 2
    }

    fn forecast(&self, s: &Series, t: usize) -> Option<SeriesForecast> {
        let g = &self.genome;
        if t < self.warmup() || t >= s.len() {
            return None;
        }
        let query = Self::shape(s, t, g.lookback)?;
        let query_latent = if self.conditioned(s) {
            self.ctx.latent_as_of(&s.instrument_id, s.ts_ms[t]).map(|l| l.vec.as_slice())
        } else {
            None
        };
        // Candidate window ends `e` with outcome window [e, e+h] ending ≤ t − min_gap.
        let last_end = t.checked_sub(g.min_gap.max(g.horizon))?.checked_sub(g.horizon)?;
        let step_days = s.step_ms as f64 / 86_400_000.0;
        let mut scored: Vec<Analog> = (g.lookback..=last_end)
            .filter_map(|e| {
                let shape = Self::shape(s, e, g.lookback)?;
                let sim = self.combined_similarity(s, &query, query_latent, e, &shape);
                let outcome = s.forward_return(e, g.horizon)?;
                Some(Analog {
                    window_id: format!("{}@{}", s.instrument_id, s.ts_ms[e]),
                    similarity: sim,
                    time_distance_days: (t - e) as f64 * step_days,
                    outcome,
                })
            })
            .collect();
        if scored.is_empty() {
            return None;
        }
        scored.sort_by(|a, b| analog_weight(b, g.tau_days).total_cmp(&analog_weight(a, g.tau_days)));
        scored.truncate(g.k);
        let point = analog_forecast(&scored, g.tau_days).unwrap_or(0.0);
        let mut pairs: Vec<(f64, f64)> = scored.iter().map(|a| (a.outcome, analog_weight(a, g.tau_days))).collect();
        let fallback = scored.iter().map(|a| a.outcome).sum::<f64>() / scored.len() as f64;
        let quantiles: Vec<f64> = g
            .levels
            .iter()
            .map(|&l| {
                weighted_quantile(&mut pairs, l).unwrap_or_else(|| {
                    // all weights zero (e.g. negative similarities only): unweighted
                    let mut eq: Vec<(f64, f64)> = scored.iter().map(|a| (a.outcome, 1.0)).collect();
                    weighted_quantile(&mut eq, l).unwrap_or(fallback)
                })
            })
            .collect();
        Some(SeriesForecast { quantiles, levels: g.levels.clone(), point, analogs: scored })
    }

    fn ops_per_forecast(&self, series_len: usize) -> f64 {
        // one shape + similarity per candidate window
        2.0 * series_len as f64 * self.genome.lookback as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fa_core::rng::Rng;

    fn synthetic(n: usize, seed: u64) -> Series {
        // SYNTHETIC test data: a random walk, not market data.
        let mut rng = Rng::seed(seed);
        let mut p = 100.0;
        let trades: Vec<(i64, f64)> = (0..n)
            .map(|i| {
                p *= 1.0 + 0.001 * rng.normal();
                (i as i64 * 1000, p)
            })
            .collect();
        Series::resample("inst-test", &trades, 1000)
    }

    #[test]
    fn resample_takes_last_price_per_bucket_and_skips_gaps() {
        let s = Series::resample("i", &[(0, 1.0), (500, 2.0), (1000, 4.0), (3000, 8.0), (2500, 100.0)], 1000);
        assert_eq!(s.ts_ms, vec![500, 1000, 3000]);
        assert!((s.log_price[0] - 2f64.ln()).abs() < 1e-12);
        assert_eq!(s.len(), 3);
        assert_eq!(s.forward_return(2, 1), None);
    }

    #[test]
    fn weighted_quantile_interpolates() {
        let mut p = vec![(3.0, 1.0), (1.0, 1.0), (2.0, 1.0)];
        assert_eq!(weighted_quantile(&mut p, 0.0), Some(1.0));
        assert_eq!(weighted_quantile(&mut p, 1.0), Some(3.0));
        let mid = weighted_quantile(&mut p, 0.5).unwrap();
        assert!((1.0..=3.0).contains(&mid));
        assert_eq!(weighted_quantile(&mut [], 0.5), None);
    }

    #[test]
    fn forecaster_is_point_in_time_and_monotone() {
        let s = synthetic(400, 1);
        let g = DecoderGenome::parse("forecast:h=3,lb=30,levels=0.1/0.5/0.9").unwrap();
        let d = build(&g);
        assert!(d.forecast(&s, d.warmup() - 1).is_none());
        let f = d.forecast(&s, 200).unwrap();
        assert!(f.quantiles[0] <= f.quantiles[1] && f.quantiles[1] <= f.quantiles[2]);
        // Truncating the future must not change the forecast at t.
        let mut shorter = s.clone();
        shorter.log_price.truncate(201);
        shorter.ts_ms.truncate(201);
        assert_eq!(d.forecast(&shorter, 200).unwrap(), f);
    }

    #[test]
    fn retriever_respects_leakage_gap() {
        let s = synthetic(600, 2);
        let g = DecoderGenome::parse("retrieval:h=4,lb=12,k=10,gap=6").unwrap();
        let d = build(&g);
        let t = 500;
        let f = d.forecast(&s, t).unwrap();
        assert_eq!(f.analogs.len(), 10);
        for a in &f.analogs {
            // analog window end e satisfies e + h ≤ t − gap  ⇔  Δt ≥ gap + h (in steps/days)
            let dt_steps = (a.time_distance_days * 86_400_000.0 / 1000.0).round() as usize;
            assert!(dt_steps >= 6 + 4, "analog too recent: {dt_steps} steps");
            assert!(a.similarity <= 1.0 + 1e-9);
        }
        assert!(f.quantiles.windows(2).all(|w| w[0] <= w[1] + 1e-12));
    }

    #[test]
    fn latent_context_changes_ranking_deterministically_and_point_in_time() {
        use crate::LatentPoint;
        let s = synthetic(400, 3);
        let g = DecoderGenome::parse("retrieval:h=2,lb=8,k=5,gap=2,lat=0.9").unwrap();
        let t = 300;
        let plain = build(&g).forecast(&s, t).unwrap();
        // Latents: one regime vector for early windows, the query's regime for a late band.
        let mut ctx = DecodeCtx::default();
        for i in (0..400).step_by(5) {
            let v = if (150..200).contains(&i) || i >= 290 { vec![1.0, 0.0] } else { vec![0.0, 1.0] };
            ctx.push(&s.instrument_id, LatentPoint { as_of_ms: s.ts_ms[i], vec: v });
        }
        // a future-only latent must not matter at t
        ctx.push(&s.instrument_id, LatentPoint { as_of_ms: s.ts_ms[t] + 1, vec: vec![-1.0, -1.0] });
        let ctx = Arc::new(ctx);
        let cond = build_with(&g, ctx.clone()).forecast(&s, t).unwrap();
        assert_eq!(cond, build_with(&g, ctx.clone()).forecast(&s, t).unwrap(), "deterministic");
        assert_ne!(cond.analogs, plain.analogs, "latents must influence retrieval");
        for a in &cond.analogs {
            let dt_steps = (a.time_distance_days * 86_400_000.0 / 1000.0).round() as usize;
            let e = t - dt_steps;
            // windows ending in the early band or in the current regime (≥ 290, ≤ t − gap − h)
            assert!((150..205).contains(&e) || e >= 290, "analog at step {e} is outside the matching regime");
        }
        // Truncating the future (series and latents) does not change the forecast.
        let mut shorter = s.clone();
        shorter.log_price.truncate(t + 1);
        shorter.ts_ms.truncate(t + 1);
        assert_eq!(build_with(&g, ctx).forecast(&shorter, t).unwrap(), cond);
        // other instruments' latents are ignored
        let mut other = DecodeCtx::default();
        other.push("someone-else", LatentPoint { as_of_ms: 0, vec: vec![1.0] });
        assert_eq!(build_with(&g, Arc::new(other)).forecast(&s, t).unwrap(), plain);
    }

    #[test]
    fn build_matches_family() {
        let f = build(&DecoderGenome::derive("x")).genome().family;
        assert_eq!(f, DecoderGenome::derive("x").family);
    }
}
