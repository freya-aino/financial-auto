//! Statistical baseline encoder (pure Rust, deterministic, no learned parameters).
//!
//! Stands in for the neural families while the ML backends are off (D-004). It turns a
//! window of bars at one timescale into a fixed-dimensional latent per expert. The three
//! "experts" are fixed statistical views (returns, volatility, activity), each `EXPERT_DIM`
//! bounded features in `(-1, 1)` built from in-window z-scores. The posterior log-variance
//! encodes estimation uncertainty from the window length, so a short history shows up as
//! a wide posterior. Nothing here is a model forecast: the latent is a descriptive summary
//! of the window and is swapped for a learned encoder behind [`EncoderFamily`].

use fa_core::Timescale;
use fa_expert_space::{ExpertLatent, ExpertView};
use serde::{Deserialize, Serialize};

use crate::{
    EncoderError, EncoderFamily, EncoderInput, FamilyDescriptor, FamilyId, LatentKind, PriorKind, ScaleSizing,
    default_sizing,
    snapshot::{NormAccumulator, NormStats, STATISTICAL_FAMILY, SnapshotManifest},
};

/// One bar at the encoder's timescale.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MarketBar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub trade_count: u64,
}

impl MarketBar {
    pub fn from_trade(price: f64, size: f64) -> Self {
        Self { open: price, high: price, low: price, close: price, volume: size, trade_count: 1 }
    }

    pub fn add_trade(&mut self, price: f64, size: f64) {
        self.high = self.high.max(price);
        self.low = self.low.min(price);
        self.close = price;
        self.volume += size;
        self.trade_count += 1;
    }

    /// Market feature row of [`EncoderInput::market`]: `[open, high, low, close, volume, trade_count]`.
    pub fn to_row(&self) -> Vec<f32> {
        vec![
            self.open as f32,
            self.high as f32,
            self.low as f32,
            self.close as f32,
            self.volume as f32,
            self.trade_count as f32,
        ]
    }

    pub fn from_row(row: &[f32]) -> Result<Self, EncoderError> {
        if row.len() != MARKET_ROW_LEN {
            return Err(EncoderError::Shape(format!(
                "market row has {} features, expected {MARKET_ROW_LEN}",
                row.len()
            )));
        }
        Ok(Self {
            open: f64::from(row[0]),
            high: f64::from(row[1]),
            low: f64::from(row[2]),
            close: f64::from(row[3]),
            volume: f64::from(row[4]),
            trade_count: row[5].max(0.0) as u64,
        })
    }
}

/// Length of one market feature row (see [`MarketBar::to_row`]).
pub const MARKET_ROW_LEN: usize = 6;
/// Latent dimension of every baseline expert.
pub const EXPERT_DIM: usize = 8;
/// The fixed expert views, in `expert_index` order.
pub const EXPERT_NAMES: [&str; 3] = ["returns", "volatility", "activity"];
/// Bars of the short sub-window used for short/long ratios.
const SHORT: usize = 5;

impl EncoderInput {
    /// Builds the market block from bars (oldest first).
    pub fn from_bars(as_of_ms: i64, bars: &[MarketBar]) -> Self {
        Self { as_of_ms, market: bars.iter().map(MarketBar::to_row).collect(), media: Vec::new(), instrument: None }
    }
}

/// Deterministic statistical encoder for one timescale.
///
/// Without fitted statistics (`StatisticalEncoder::new`) it is the parameter-free gen-0
/// baseline `stat-v1-<ts>`. A trained snapshot ([`StatisticalEncoder::fit`] →
/// [`StatisticalEncoder::from_manifest`]) carries [`NormStats`]: every raw feature is
/// z-scored against the training distribution before the bounded squash, so the latent is
/// centred on the fitted history instead of on the fixed in-window scale.
#[derive(Debug, Clone)]
pub struct StatisticalEncoder {
    descriptor: FamilyDescriptor,
    sizing: ScaleSizing,
    snapshot_id: String,
    norm: Option<NormStats>,
}

/// How [`StatisticalEncoder::fit`] walks the history: one window per `stride` bars, each
/// ending at a bar close, at least `min_window` bars long.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FitConfig {
    pub stride: usize,
    pub min_window: usize,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self { stride: 1, min_window: 8 }
    }
}

impl StatisticalEncoder {
    /// Snapshot ids are versioned by hand: the encoder has no trained weights, so any change
    /// to the feature definitions must bump this.
    pub const VERSION: &'static str = "stat-v1";

    pub fn new(ts: Timescale) -> Self {
        let descriptor = FamilyDescriptor {
            id: FamilyId::StatBaseline,
            latent: LatentKind::Continuous,
            prior: PriorKind::Gaussian,
            region_discovery: "leader",
            timescales: vec![ts],
        };
        let sizing =
            ScaleSizing { max_experts: EXPERT_NAMES.len() as u32, latent_dim: EXPERT_DIM as u32, ..default_sizing(ts) };
        Self { descriptor, sizing, snapshot_id: format!("{}-{ts}", Self::VERSION), norm: None }
    }

    /// The gen-0 baseline id of a timescale (`stat-v1-<ts>`).
    pub fn baseline_id(ts: Timescale) -> String {
        format!("{}-{ts}", Self::VERSION)
    }

    /// A fitted encoder under its own snapshot id.
    pub fn with_stats(ts: Timescale, snapshot_id: impl Into<String>, norm: NormStats) -> Self {
        Self { snapshot_id: snapshot_id.into(), norm: Some(norm), ..Self::new(ts) }
    }

    /// Rebuilds the encoder of a snapshot from its manifest and `weights.bin`
    /// ([`NormStats::to_bytes`]). `params.context_steps` overrides the default sizing.
    pub fn from_manifest(manifest: &SnapshotManifest, weights: &[u8]) -> Result<Self, EncoderError> {
        if manifest.encoder_family != STATISTICAL_FAMILY {
            return Err(EncoderError::Manifest(format!(
                "family `{}` is not the statistical encoder",
                manifest.encoder_family
            )));
        }
        let norm = NormStats::from_bytes(weights)?;
        if norm.n_experts() != EXPERT_NAMES.len() || norm.dim() != EXPERT_DIM {
            return Err(EncoderError::Manifest(format!(
                "weights are {}x{}, encoder is {}x{EXPERT_DIM}",
                norm.n_experts(),
                norm.dim(),
                EXPERT_NAMES.len()
            )));
        }
        let mut enc = Self::with_stats(manifest.timescale, manifest.snapshot_id.clone(), norm);
        if let Some(ctx) = manifest.params.get("context_steps").and_then(serde_json::Value::as_u64) {
            enc.sizing.context_steps = ctx.clamp(2, u64::from(u32::MAX)) as u32;
        }
        Ok(enc)
    }

    pub fn timescale(&self) -> Timescale {
        self.sizing.timescale
    }

    pub fn norm(&self) -> Option<&NormStats> {
        self.norm.as_ref()
    }

    /// Family hyperparameters recorded in a manifest (`params`).
    pub fn params(&self, fit: &FitConfig) -> serde_json::Value {
        serde_json::json!({
            "version": Self::VERSION,
            "context_steps": self.sizing.context_steps,
            "experts": EXPERT_NAMES,
            "expert_dim": EXPERT_DIM,
            "normalised": self.norm.is_some(),
            "fit": fit,
        })
    }

    /// Raw (un-normalised) features of one window: `[expert][dim]`, each in (−1, 1).
    pub fn raw_features(&self, bars: &[MarketBar]) -> Result<Vec<Vec<f32>>, EncoderError> {
        if bars.is_empty() {
            return Err(EncoderError::Shape("empty bar window".into()));
        }
        let f = Features::new(self.window(bars));
        Ok(vec![f.returns_expert(), f.volatility_expert(), f.activity_expert()])
    }

    /// Window ends visited by [`Self::fit`] over `n` bars: `min_window, min_window + stride, …, n`.
    pub fn fit_window_ends(n: usize, cfg: &FitConfig) -> Vec<usize> {
        let stride = cfg.stride.max(1);
        let first = cfg.min_window.max(2);
        if n < first {
            return Vec::new();
        }
        let mut ends: Vec<usize> = (first..=n).step_by(stride).collect();
        if ends.last() != Some(&n) {
            ends.push(n);
        }
        ends
    }

    /// Accumulates the raw features of the windows ending at `ends` (indices into `bars`,
    /// exclusive) into `acc`. Split across calls to checkpoint a long fit.
    pub fn fit_into(&self, bars: &[MarketBar], ends: &[usize], acc: &mut NormAccumulator) -> Result<(), EncoderError> {
        for &end in ends {
            let end = end.min(bars.len());
            acc.push(&self.raw_features(&bars[..end])?);
        }
        Ok(())
    }

    /// Fits the normalisation statistics over every window of `bars` (oldest first).
    pub fn fit(ts: Timescale, bars: &[MarketBar], cfg: &FitConfig) -> Result<NormStats, EncoderError> {
        let enc = Self::new(ts);
        let mut acc = NormAccumulator::default();
        enc.fit_into(bars, &Self::fit_window_ends(bars.len(), cfg), &mut acc)?;
        if acc.n == 0 {
            return Err(EncoderError::Shape(format!(
                "{} bars are fewer than min_window {}",
                bars.len(),
                cfg.min_window
            )));
        }
        Ok(acc.finish())
    }

    /// Context actually used: the last `context_steps` bars.
    fn window<'a>(&self, bars: &'a [MarketBar]) -> &'a [MarketBar] {
        let n = self.sizing.context_steps as usize;
        &bars[bars.len().saturating_sub(n)..]
    }
}

impl EncoderFamily for StatisticalEncoder {
    fn descriptor(&self) -> &FamilyDescriptor {
        &self.descriptor
    }

    fn sizing(&self) -> ScaleSizing {
        self.sizing
    }

    fn snapshot_id(&self) -> &str {
        &self.snapshot_id
    }

    fn encode(&self, input: &EncoderInput) -> Result<ExpertView, EncoderError> {
        if input.market.is_empty() {
            return Err(EncoderError::Shape("empty bar window".into()));
        }
        let bars: Vec<MarketBar> = input.market.iter().map(|r| MarketBar::from_row(r)).collect::<Result<_, _>>()?;
        let w = self.window(&bars);
        let mut latents = self.raw_features(w)?;
        if let Some(norm) = &self.norm {
            for (e, row) in latents.iter_mut().enumerate() {
                for (d, x) in row.iter_mut().enumerate() {
                    *x = squash(norm.apply(e, d, f64::from(*x))) as f32;
                }
            }
        }
        // Estimation uncertainty of in-window z-scores shrinks like 1/n (variance), so the
        // posterior log-variance is −ln(n); one bar → unit variance.
        let log_var = -(w.len() as f32).ln();
        let experts = latents
            .into_iter()
            .enumerate()
            .map(|(i, mean)| ExpertLatent {
                expert_index: i as u32,
                mean,
                log_var: vec![log_var; EXPERT_DIM],
                region_id: None,
                region_lineage_id: None,
            })
            .collect();
        Ok(ExpertView {
            snapshot_id: self.snapshot_id.clone(),
            timescale: self.sizing.timescale,
            as_of_ms: input.as_of_ms,
            expert_mask: vec![true; EXPERT_NAMES.len()],
            experts,
            router_weights: Vec::new(), // filled by the router stage
        })
    }
}

/// Derived series of one window; every feature is bounded by [`squash`].
struct Features<'a> {
    bars: &'a [MarketBar],
    /// Log returns between consecutive closes (len n−1).
    rets: Vec<f64>,
    /// Bar range relative to close.
    ranges: Vec<f64>,
    volumes: Vec<f64>,
    counts: Vec<f64>,
}

impl<'a> Features<'a> {
    fn new(bars: &'a [MarketBar]) -> Self {
        let rets = bars.windows(2).map(|p| safe_log_ratio(p[1].close, p[0].close)).collect();
        let ranges = bars.iter().map(|b| if b.close > 0.0 { (b.high - b.low) / b.close } else { 0.0 }).collect();
        let volumes = bars.iter().map(|b| b.volume).collect();
        let counts = bars.iter().map(|b| b.trade_count as f64).collect();
        Self { bars, rets, ranges, volumes, counts }
    }

    fn last(&self) -> &MarketBar {
        &self.bars[self.bars.len() - 1]
    }

    fn returns_expert(&self) -> Vec<f32> {
        let r = &self.rets;
        let last_r = r.last().copied().unwrap_or(0.0);
        let (m, s) = mean_std(r);
        let n = r.len() as f64;
        let short: f64 = r.iter().rev().take(SHORT).sum();
        let closes: Vec<f64> = self.bars.iter().map(|b| b.close).collect();
        let (lo, hi) = closes.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), c| (lo.min(*c), hi.max(*c)));
        let pos = if hi > lo { 2.0 * (self.last().close - lo) / (hi - lo) - 1.0 } else { 0.0 };
        let ups = if r.is_empty() { 0.0 } else { 2.0 * r.iter().filter(|x| **x > 0.0).count() as f64 / n - 1.0 };
        let b = self.last();
        let body = if b.high > b.low { (b.close - b.open) / (b.high - b.low) } else { 0.0 };
        to_f32(&[
            squash(zscore(last_r, m, s)),
            squash(if s > 0.0 { m / s * n.sqrt() } else { 0.0 }),
            squash(if s > 0.0 { short / (s * (SHORT.min(r.len().max(1)) as f64).sqrt()) } else { 0.0 }),
            autocorr1(r),
            squash(skewness(r)),
            pos,
            ups,
            body,
        ])
    }

    fn volatility_expert(&self) -> Vec<f32> {
        let abs_r: Vec<f64> = self.rets.iter().map(|x| x.abs()).collect();
        let (ma, sa) = mean_std(&abs_r);
        let last_abs = abs_r.last().copied().unwrap_or(0.0);
        let short_vol = rms(&self.rets[self.rets.len().saturating_sub(SHORT)..]);
        let long_vol = rms(&self.rets);
        let (mr, sr) = mean_std(&self.ranges);
        let last_range = self.ranges.last().copied().unwrap_or(0.0);
        let dd = max_drawdown(self.bars);
        let (_, s) = mean_std(&self.rets);
        let b = self.last();
        let wick = if b.high > b.low {
            ((b.high - b.open.max(b.close)) - (b.open.min(b.close) - b.low)) / (b.high - b.low)
        } else {
            0.0
        };
        to_f32(&[
            squash(zscore(last_abs, ma, sa)),
            squash(safe_log_ratio(short_vol, long_vol)),
            squash(zscore(last_range, mr, sr)),
            squash(if mr > 0.0 { last_range / mr - 1.0 } else { 0.0 }),
            squash(if ma > 0.0 { sa / ma - 1.0 } else { 0.0 }),
            squash(if s > 0.0 { dd / (s * (self.bars.len() as f64).sqrt()) } else { 0.0 }),
            squash(kurtosis_excess(&self.rets) / 3.0),
            wick,
        ])
    }

    fn activity_expert(&self) -> Vec<f32> {
        let v = &self.volumes;
        let (mv, sv) = mean_std(v);
        let (mc, sc) = mean_std(&self.counts);
        let last = self.last();
        let short_v = mean(&v[v.len().saturating_sub(SHORT)..]);
        let abs_r: Vec<f64> = self.rets.iter().map(|x| x.abs()).collect();
        let v_tail = &v[v.len() - abs_r.len()..];
        let dollar: Vec<f64> = self.bars.iter().map(|b| b.close * b.volume).collect();
        let (md, sd) = mean_std(&dollar);
        let sum_v: f64 = v.iter().sum();
        let n = v.len() as f64;
        let concentration = if sum_v > 0.0 && n > 1.0 {
            (v.iter().copied().fold(0.0, f64::max) / sum_v - 1.0 / n) * n / (n - 1.0)
        } else {
            0.0
        };
        let sizes: Vec<f64> =
            self.bars.iter().map(|b| if b.trade_count > 0 { b.volume / b.trade_count as f64 } else { 0.0 }).collect();
        let (ms, ss) = mean_std(&sizes);
        to_f32(&[
            squash(zscore(last.volume, mv, sv)),
            squash(zscore(last.trade_count as f64, mc, sc)),
            squash(safe_log_ratio(short_v, mv)),
            fa_expert_space::pearson(v_tail, &abs_r),
            fa_expert_space::pearson(v_tail, &self.rets),
            squash(zscore(dollar[dollar.len() - 1], md, sd)),
            squash(2.0 * concentration),
            squash(zscore(sizes[sizes.len() - 1], ms, ss)),
        ])
    }
}

/// `tanh(x / 2)`: bounded, ≈ linear near 0, a 2σ z-score maps to ≈ 0.76.
pub fn squash(x: f64) -> f64 {
    if x.is_finite() { (x / 2.0).tanh() } else { 0.0 }
}

fn to_f32(xs: &[f64; EXPERT_DIM]) -> Vec<f32> {
    xs.iter().map(|x| *x as f32).collect()
}

fn safe_log_ratio(a: f64, b: f64) -> f64 {
    if a > 0.0 && b > 0.0 { (a / b).ln() } else { 0.0 }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { 0.0 } else { xs.iter().sum::<f64>() / xs.len() as f64 }
}

/// Population mean and standard deviation (0 when fewer than 2 samples).
fn mean_std(xs: &[f64]) -> (f64, f64) {
    let m = mean(xs);
    if xs.len() < 2 {
        return (m, 0.0);
    }
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64;
    (m, var.sqrt())
}

fn zscore(x: f64, m: f64, s: f64) -> f64 {
    if s > 0.0 { (x - m) / s } else { 0.0 }
}

fn rms(xs: &[f64]) -> f64 {
    if xs.is_empty() { 0.0 } else { (xs.iter().map(|x| x * x).sum::<f64>() / xs.len() as f64).sqrt() }
}

fn autocorr1(xs: &[f64]) -> f64 {
    if xs.len() < 3 {
        return 0.0;
    }
    fa_expert_space::pearson(&xs[..xs.len() - 1], &xs[1..])
}

fn standardized_moment(xs: &[f64], k: i32) -> f64 {
    let (m, s) = mean_std(xs);
    if s == 0.0 || xs.len() < 3 {
        return 0.0;
    }
    xs.iter().map(|x| ((x - m) / s).powi(k)).sum::<f64>() / xs.len() as f64
}

fn skewness(xs: &[f64]) -> f64 {
    standardized_moment(xs, 3)
}

fn kurtosis_excess(xs: &[f64]) -> f64 {
    if xs.len() < 4 { 0.0 } else { standardized_moment(xs, 4) - 3.0 }
}

/// Largest peak-to-trough relative decline of closes within the window (≥ 0).
fn max_drawdown(bars: &[MarketBar]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut dd = 0.0;
    for b in bars {
        peak = peak.max(b.close);
        if peak > 0.0 {
            dd = f64::max(dd, 1.0 - b.close / peak);
        }
    }
    dd
}

#[cfg(test)]
mod tests {
    use fa_core::rng::Rng;

    use super::*;

    /// Synthetic test bars: a seeded random walk (not market data).
    fn walk(n: usize, seed: u64, vol: f64) -> Vec<MarketBar> {
        let mut rng = Rng::seed(seed);
        let mut price = 100.0;
        (0..n)
            .map(|_| {
                let open = price;
                let mut b = MarketBar::from_trade(open, 1.0);
                for _ in 0..10 {
                    price *= 1.0 + vol * rng.normal();
                    b.add_trade(price, 1.0 + rng.uniform() * 9.0);
                }
                b
            })
            .collect()
    }

    #[test]
    fn encodes_three_bounded_experts_deterministically() {
        let enc = StatisticalEncoder::new(Timescale::Minute);
        let bars = walk(120, 3, 0.001);
        let input = EncoderInput::from_bars(1_000, &bars);
        let a = enc.encode(&input).unwrap();
        let b = enc.encode(&input).unwrap();
        assert_eq!(a.experts.len(), EXPERT_NAMES.len());
        for (x, y) in a.experts.iter().zip(&b.experts) {
            assert_eq!(x.mean, y.mean);
            assert_eq!(x.mean.len(), EXPERT_DIM);
            assert!(x.mean.iter().all(|v| v.is_finite() && v.abs() <= 1.0), "{:?}", x.mean);
        }
        assert_eq!(a.snapshot_id, "stat-v1-1m");
        assert!((a.experts[0].log_var[0] - (-(120f32).ln())).abs() < 1e-6);
    }

    #[test]
    fn single_bar_window_is_finite_and_uncertain() {
        let enc = StatisticalEncoder::new(Timescale::Hour);
        let bars = [MarketBar::from_trade(10.0, 1.0)];
        let v = enc.encode(&EncoderInput::from_bars(0, &bars)).unwrap();
        assert!(v.experts.iter().all(|e| e.mean.iter().all(|x| x.is_finite())));
        assert_eq!(v.experts[0].log_var[0], 0.0);
        assert!(enc.encode(&EncoderInput::default()).is_err());
    }

    #[test]
    fn volatility_expert_reacts_to_a_range_spike() {
        let enc = StatisticalEncoder::new(Timescale::Minute);
        let mut bars = walk(60, 5, 0.0005);
        let calm = enc.encode(&EncoderInput::from_bars(0, &bars)).unwrap();
        let last = bars.last_mut().unwrap();
        last.high = last.close * 1.05;
        last.low = last.close * 0.95;
        let spiky = enc.encode(&EncoderInput::from_bars(0, &bars)).unwrap();
        // feature 2 of the volatility expert is the range z-score
        assert!(spiky.experts[1].mean[2] > 0.9 && spiky.experts[1].mean[2] > calm.experts[1].mean[2]);
    }

    #[test]
    fn fitted_snapshot_is_reproducible_from_its_artifact() {
        let bars = walk(300, 11, 0.001);
        let cfg = FitConfig { stride: 3, min_window: 16 };
        let stats = StatisticalEncoder::fit(Timescale::Minute, &bars, &cfg).unwrap();
        let ends = StatisticalEncoder::fit_window_ends(300, &cfg);
        assert_eq!(ends.first(), Some(&16));
        assert_eq!(ends.last(), Some(&300));
        assert_eq!(stats.n_windows, ends.len() as u64);
        assert_eq!(stats.n_experts(), 3);
        assert!(stats.std.iter().flatten().all(|s| s.is_finite() && *s >= 0.0));
        // chunked accumulation equals one pass
        let enc = StatisticalEncoder::new(Timescale::Minute);
        let mut acc = NormAccumulator::default();
        enc.fit_into(&bars, &ends[..40], &mut acc).unwrap();
        enc.fit_into(&bars, &ends[40..], &mut acc).unwrap();
        assert_eq!(acc.finish(), stats);

        let manifest = SnapshotManifest {
            snapshot_id: "stat-v1-1m-g0-test".into(),
            encoder_family: STATISTICAL_FAMILY.into(),
            timescale: Timescale::Minute,
            generation: 0,
            parent_snapshot_id: None,
            created_at: String::new(),
            params: serde_json::json!({"context_steps": 64}),
            probe_set: Default::default(),
            data: Default::default(),
            metrics: Default::default(),
            feedback: None,
            seed: 0,
        };
        let fitted = StatisticalEncoder::from_manifest(&manifest, &stats.to_bytes()).unwrap();
        assert_eq!(fitted.snapshot_id(), "stat-v1-1m-g0-test");
        assert_eq!(fitted.sizing().context_steps, 64);
        let input = EncoderInput::from_bars(0, &bars);
        let a = fitted.encode(&input).unwrap();
        let mut same = StatisticalEncoder::with_stats(Timescale::Minute, "x", stats.clone());
        same.sizing.context_steps = 64;
        let b = same.encode(&input).unwrap();
        assert_eq!(a.experts[0].mean, b.experts[0].mean, "same stats + context → same latent");
        assert!(a.experts.iter().all(|e| e.mean.iter().all(|v| v.is_finite() && v.abs() < 1.0)));
        let raw = StatisticalEncoder::new(Timescale::Minute).encode(&input).unwrap();
        assert_ne!(a.experts[1].mean, raw.experts[1].mean, "normalisation changes the latent");
        let mut other = manifest.clone();
        other.encoder_family = "fsq_hybrid".into();
        assert!(matches!(StatisticalEncoder::from_manifest(&other, &stats.to_bytes()), Err(EncoderError::Manifest(_))));
        assert!(StatisticalEncoder::fit(Timescale::Minute, &bars[..4], &cfg).is_err());
    }

    #[test]
    fn rejects_malformed_rows() {
        let enc = StatisticalEncoder::new(Timescale::Day);
        let bad = EncoderInput { as_of_ms: 0, market: vec![vec![1.0, 2.0]], media: vec![], instrument: None };
        assert!(matches!(enc.encode(&bad), Err(EncoderError::Shape(_))));
    }
}
