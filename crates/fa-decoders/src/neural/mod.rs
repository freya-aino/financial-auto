//! Trainable neural decoder families on candle (spec 34 §4.1, §6.1; ADR 0003 pure-Rust ML).
//!
//! Eight families extend the D-044 genome ([`crate::genome::BaselineFamily`]):
//!
//! | genome family | output family | input | target / loss |
//! |---|---|---|---|
//! | `nn_forecast` | `rnn_prior` | return window (+ latent → GRU initial state) | `z = r/(σ√h)` quantiles, pinball |
//! | `nn_retrieval` | `retrieval` | return window + latent → embedding | soft-kNN CRPS over leakage-safe analogs |
//! | `nn_chart` | `visual.chart` | latent (return window if unconditioned) | raster of the lookback window, column CE |
//! | `nn_future` | `visual.future` | return window + latent | raster of the next `h` steps, column CE |
//! | `nn_news` | `text.news` | return window + latent | news embedding of `(t − tw, t]`, 1 − cos |
//! | `nn_narrative` | `text.narrative` | return window + news embedding + latent | news embedding of `(t, t + tw]`, 1 − cos |
//! | `nn_summary` | `text.summary` | latent (return window if unconditioned) | templated summary slots ([`crate::summary`]), per-slot CE |
//! | `nn_frame` | `visual.frame` | return window + latent | frame-timeline embedding of `(t − tw, t]` ([`data::frame_features`]), 1 − cos |
//!
//! Reconstruction families start at their training-mean baseline (zero output weights, bias =
//! the mean target in output space) and validate on a purged tail (D-085).
//!
//! [`train`] fits a genome on everything **before the train cutoff** (`train` gene: share of the
//! window by time; a sample is used only once its target is known, `outcome_end ≤ cutoff`), with
//! AdamW minibatch SGD on the CPU (`cuda` feature: CUDA device when present). The resulting
//! [`NeuralDecoder`] forecasts only **after** the cutoff ([`NeuralDecoder::with_eval_from`]), so
//! `fa_backtest::eval` scores it out of sample with the D-049 objectives; reconstruction families
//! are scored by [`NeuralDecoder::recon_eval`] (`recon`, `recon_skill` vs the training-mean target).
//! Weights + metadata travel as one safetensors **bundle** ([`NeuralDecoder::to_bundle`],
//! [`NeuralDecoder::from_bundle`]); [`feedback`] loads elite bundles as frozen modules for the
//! decoder → encoder auxiliary loss (D-009).

pub mod data;
pub mod feedback;
pub mod net;
mod train;

use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex},
};

use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};

/// candle types of the public API (the trainer passes latents as a `Tensor`).
pub use candle_core::{DType, Device, Tensor, Var};

pub use self::train::{TrainOptions, TrainReport, train};
use self::{
    data::{Layout, Norm, TextTimeline},
    net::Net,
};
use crate::{
    Analog, DecodeCtx,
    baseline::{Series, SeriesDecoder, SeriesForecast, weighted_quantile},
    genome::{BaselineFamily, DecoderGenome, GenomeError},
    render,
};

/// Bundle format tag (`__meta__.format`).
pub const BUNDLE_FORMAT: &str = "fa-decoder-bundle/v1";
/// Name of the metadata tensor (UTF-8 JSON as `u8`) inside a bundle.
pub const META_TENSOR: &str = "__meta__";
/// Version of the latent normaliser a bundle was trained with (`BundleMeta.norm_version`).
/// 1 = the first-generation normaliser (no std floor, no clipping; bundles written before the
/// field existed deserialise as 1); 2 = std floored at [`data::LATENT_STD_FLOOR`] and values
/// clipped to ±[`data::LATENT_CLIP`] in training, serving and feedback alike (D-068). Bundles
/// below this version are refused by [`NeuralDecoder::from_bundle`], skipped by
/// [`feedback::select_top_k`] and re-trained ([`feedback::stale_cards`]).
pub const NORM_VERSION: u32 = 2;

fn legacy_norm_version() -> u32 {
    1
}
/// Rows per forward pass at inference.
const INFER_BATCH: usize = 1024;

#[derive(Debug, thiserror::Error)]
pub enum NeuralError {
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("genome: {0}")]
    Genome(#[from] GenomeError),
    #[error("`{0}` is not a neural family")]
    NotNeural(String),
    #[error("insufficient training data: {0}")]
    InsufficientData(String),
    #[error("bundle: {0}")]
    Bundle(String),
    #[error("interrupted after {0} steps")]
    Interrupted(usize),
    #[error("stale bundle: latent normaliser v{0} < v{NORM_VERSION} (re-train the genome)")]
    StaleNormaliser(u32),
}

/// Everything about a trained decoder except its weights (stored as JSON inside the bundle and
/// as the Garage card next to it).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleMeta {
    pub format: String,
    /// Canonical kv genome.
    pub genome: String,
    /// Genome family (`nn_forecast`, …).
    pub family: String,
    /// `decoder.output.<family>` family.
    pub output_family: String,
    pub layout: Layout,
    pub norm: Norm,
    /// Latent normaliser version ([`NORM_VERSION`]; absent = 1, the first generation).
    #[serde(default = "legacy_norm_version")]
    pub norm_version: u32,
    /// Expert layout `(expert_index, len)` of the **full** latent the decoder was trained on
    /// (before its expert mask); empty for unconditioned decoders and pre-mask bundles.
    #[serde(default)]
    pub latent_layout: Vec<(u32, u32)>,
    /// Mean training target (the reconstruction baseline).
    pub target_mean: Vec<f32>,
    /// Text families: mean / std of `dot(target, sentiment_direction)` over training targets.
    #[serde(default)]
    pub sentiment: (f64, f64),
    /// Step of the training series (ms).
    pub step_ms: i64,
    /// Train / evaluation boundary (ms).
    pub cutoff_ms: i64,
    pub train: TrainReport,
    /// Evaluation objectives (filled by the evaluator; empty until then).
    #[serde(default)]
    pub objectives: BTreeMap<String, f64>,
    /// Family score: `crps_skill` for forecast families, `recon_skill` otherwise.
    #[serde(default)]
    pub score: f64,
    #[serde(default)]
    pub job_id: String,
    #[serde(default)]
    pub generation: String,
    /// Re-evaluations of the frozen decoder on later data / encoder snapshots (spec 34 §8,
    /// `evo.reeval.decoder`), oldest first; the latest one sets the feedback decay
    /// ([`Self::transfer_factor`]).
    #[serde(default)]
    pub transfer: Vec<Transfer>,
}

/// One re-evaluation of a trained decoder (frozen weights) after its training cutoff.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Transfer {
    /// Encoder snapshot whose latents the re-evaluation read (`live` when unknown).
    pub snapshot_id: String,
    /// Family score on the re-evaluation window (`crps_skill` / `recon_skill`).
    pub score: f64,
    /// The score at training ([`BundleMeta::score`]).
    pub base_score: f64,
    /// `score / base_score` (1 = no loss; ≤ 0 = no longer beats the naive baseline).
    pub ratio: f64,
    pub n_eval: usize,
    pub window_start_ms: i64,
    pub window_end_ms: i64,
    pub at_ms: i64,
    #[serde(default)]
    pub job_id: String,
}

/// Default tolerated loss of a re-evaluated elite (spec 36 §6.4 item 5 `transfer_tol`).
pub const TRANSFER_TOL: f64 = 0.10;

impl BundleMeta {
    /// Latest re-evaluation, if any.
    pub fn latest_transfer(&self) -> Option<&Transfer> {
        self.transfer.last()
    }

    /// Feedback-weight decay of a decoder (spec 34 §8 "feedback weight decays for decoders
    /// failing transfer"): 1 before any re-evaluation, else the latest `ratio` clipped to
    /// `[0, 1]` — a decoder that kept its skill keeps its weight, one that lost half of it gets
    /// half, one that no longer beats naive gets none.
    pub fn transfer_factor(&self) -> f64 {
        self.latest_transfer().map_or(1.0, |t| if t.ratio.is_finite() { t.ratio.clamp(0.0, 1.0) } else { 0.0 })
    }

    /// Whether the latest re-evaluation lost at most `tol` of the training score (`true`
    /// before any re-evaluation).
    pub fn transfers(&self, tol: f64) -> bool {
        self.latest_transfer().is_none_or(|t| t.ratio >= 1.0 - tol)
    }

    /// Whether the bundle uses the current latent normaliser.
    pub fn is_current(&self) -> bool {
        self.norm_version >= NORM_VERSION
    }

    /// Dimension of the full (unmasked) latent the decoder reads from (0 = unconditioned).
    pub fn full_latent_dim(&self) -> usize {
        if self.layout.latent == 0 {
            return 0;
        }
        let full: usize = self.latent_layout.iter().map(|(_, l)| *l as usize).sum();
        if full > 0 { full } else { self.layout.latent }
    }
}

/// Reconstruction quality on the evaluation window.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct ReconReport {
    pub n: usize,
    /// Mean family loss of the decoder (column CE in nats, or `1 − cos`).
    pub loss: f64,
    /// Same loss of the training-mean target.
    pub baseline: f64,
    /// `1 − loss / baseline` (max).
    pub skill: f64,
}

/// What a served decoder says about the last step of a series.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Served {
    /// Horizon-return quantiles (forecast families; empty otherwise).
    pub quantiles: Vec<f64>,
    pub levels: Vec<f64>,
    /// Score in [-1, 1] for the `text.*` / `visual.*` contract (`quantiles[0]` there).
    pub score: f64,
    /// Further components published after the score.
    pub components: Vec<f64>,
    pub analogs: Vec<Analog>,
    /// Raw reconstruction (raster probabilities / embedding) for online scoring.
    pub recon: Vec<f32>,
    /// Decoded text (`text.summary`: the summary sentence of the most likely slot phrases).
    pub text: String,
}

type ForecastCache = HashMap<(String, usize, i64), Arc<Vec<Option<SeriesForecast>>>>;

/// A trained (or loaded) neural decoder.
pub struct NeuralDecoder {
    genome: DecoderGenome,
    pub meta: BundleMeta,
    net: Net,
    params: Vec<(String, Tensor)>,
    dev: Device,
    ctx: Arc<DecodeCtx>,
    text: Arc<TextTimeline>,
    eval_from_ms: Option<i64>,
    cache: Mutex<ForecastCache>,
}

impl NeuralDecoder {
    pub fn genome(&self) -> &DecoderGenome {
        &self.genome
    }

    pub fn family(&self) -> BaselineFamily {
        self.genome.family
    }

    /// Decoder id on `DecoderOutput.decoder_id`: `nn-<family>-<hash of the genome>`.
    pub fn decoder_id(&self) -> String {
        format!("nn-{}-{:08x}", self.genome.family.as_str(), crate::genome::stable_hash(&self.meta.genome) as u32)
    }

    /// Forecasts (and [`SeriesDecoder::forecast`]) only at steps with `ts ≥ ms`.
    pub fn with_eval_from(mut self, ms: Option<i64>) -> Self {
        self.eval_from_ms = ms;
        self.clear_cache();
        self
    }

    /// Replaces the latent / text context (serving, re-evaluation); the latents are masked to
    /// the genome's experts.
    pub fn with_context(mut self, ctx: Arc<DecodeCtx>, text: Arc<TextTimeline>) -> Self {
        self.ctx = if self.genome.expert_mask != 0 { Arc::new(ctx.masked(self.genome.expert_mask)) } else { ctx };
        self.text = text;
        self.clear_cache();
        self
    }

    fn clear_cache(&self) {
        if let Ok(mut c) = self.cache.lock() {
            c.clear();
        }
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    pub(crate) fn net(&self) -> &Net {
        &self.net
    }

    /// Number of scalar parameters.
    pub fn n_params(&self) -> usize {
        self.params.iter().map(|(_, t)| t.elem_count()).sum()
    }

    /// Input row at step `t` with the decoder's own latent / text context (latents of the `src`
    /// instrument for a cross-asset genome; the context is already expert-masked).
    pub fn input_at(&self, s: &Series, t: usize) -> Option<(Vec<f32>, f64)> {
        let lat =
            self.ctx.latent_as_of(data::latent_instrument(&self.genome, s), *s.ts_ms.get(t)?).map(|l| l.vec.as_slice());
        data::input_at(&self.genome, &self.meta.layout, &self.meta.norm, s, t, lat, &self.text)
    }

    /// The decoder's own (masked) latent as of step `t` from its context.
    pub fn own_latent(&self, s: &Series, t: usize) -> Option<Vec<f32>> {
        if self.meta.layout.latent == 0 {
            return None;
        }
        self.ctx
            .latent_as_of(data::latent_instrument(&self.genome, s), *s.ts_ms.get(t)?)
            .map(|l| l.vec.clone())
            .filter(|v| v.len() == self.meta.layout.latent)
    }

    /// Family head applied to raw outputs: quantiles in `z` units, normalised embeddings, raster
    /// probabilities (column-major) or normalised text embeddings.
    pub fn head(&self, raw: &Tensor) -> Result<Tensor, NeuralError> {
        Ok(match self.genome.family {
            BaselineFamily::NnRetrieval | BaselineFamily::NnNews | BaselineFamily::NnNarrative | BaselineFamily::NnFrame => {
                net::l2_normalize(raw)?
            }
            BaselineFamily::NnSummary => net::slots_logp(raw, &crate::summary::slot_sizes())?.exp()?,
            BaselineFamily::NnChart | BaselineFamily::NnFuture => {
                let b = raw.dim(0)?;
                net::raster_logp(raw, self.genome.nn.res)?
                    .exp()?
                    .reshape((b, self.genome.nn.res * self.genome.nn.res))?
            }
            _ => net::monotone(raw)?,
        })
    }

    /// Batched forward + head over input rows.
    pub fn predict_rows(&self, rows: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, NeuralError> {
        let width = self.meta.layout.input();
        let mut out = Vec::with_capacity(rows.len());
        for chunk in rows.chunks(INFER_BATCH) {
            let refs: Vec<&[f32]> = chunk.iter().map(Vec::as_slice).collect();
            let x = net::rows_tensor(&refs, width, &self.dev)?;
            let y = self.head(&self.net.forward(&x)?)?;
            out.extend(y.to_dtype(DType::F32)?.to_vec2::<f32>()?);
        }
        Ok(out)
    }

    fn levels(&self) -> &[f64] {
        &self.genome.levels
    }

    /// Quantile at level `l` by linear interpolation of `(levels, qs)`.
    fn interp(levels: &[f64], qs: &[f64], l: f64) -> f64 {
        if qs.is_empty() {
            return 0.0;
        }
        if l <= levels[0] {
            return qs[0];
        }
        for i in 1..levels.len() {
            if l <= levels[i] {
                let f = (l - levels[i - 1]) / (levels[i] - levels[i - 1]);
                return qs[i - 1] + f * (qs[i] - qs[i - 1]);
            }
        }
        *qs.last().unwrap_or(&0.0)
    }

    /// Forecast of one row's head output at `σ`.
    fn forecast_of(&self, y: &[f32], sigma: f64) -> SeriesForecast {
        let g = &self.genome;
        let scale = sigma * (g.horizon as f64).sqrt();
        let quantiles: Vec<f64> = match g.family {
            BaselineFamily::NnFuture => {
                let res = g.nn.res;
                render::column_quantiles(&y[(res - 1) * res..], self.levels(), g.nn.band * scale)
            }
            _ => y.iter().map(|z| f64::from(*z) * scale).collect(),
        };
        let point = Self::interp(self.levels(), &quantiles, 0.5);
        SeriesForecast { quantiles, levels: g.levels.clone(), point, analogs: Vec::new() }
    }

    /// Learned retrieval at `t` from per-step embeddings / z-outcomes / σ of the same series.
    fn retrieve(
        &self,
        s: &Series,
        t: usize,
        embs: &[Option<Vec<f32>>],
        zs: &[Option<f32>],
        sigma_t: f64,
    ) -> Option<SeriesForecast> {
        let g = &self.genome;
        let q = embs.get(t)?.as_ref()?;
        let last_end = t.checked_sub(g.min_gap.max(g.horizon))?.checked_sub(g.horizon)?;
        let first = last_end.saturating_sub(g.nn.pool).max(g.lookback);
        let temp = g.nn.temp;
        let mut scored: Vec<(f64, Analog, f64)> = (first..=last_end)
            .filter_map(|e| {
                let c = embs.get(e)?.as_ref()?;
                let z = f64::from((*zs.get(e)?)?);
                let sim: f64 = q.iter().zip(c).map(|(a, b)| f64::from(a * b)).sum::<f64>().clamp(-1.0, 1.0);
                let dt_days = (s.ts_ms[t] - s.ts_ms[e]) as f64 / 86_400_000.0;
                let w = ((sim - 1.0) / temp).exp() * (-dt_days / g.tau_days).exp();
                let outcome = s.forward_return(e, g.horizon)?;
                let a = Analog {
                    window_id: format!("{}@{}", s.instrument_id, s.ts_ms[e]),
                    similarity: sim,
                    time_distance_days: dt_days,
                    outcome,
                };
                Some((w, a, z))
            })
            .collect();
        if scored.is_empty() {
            return None;
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(g.k);
        let scale = sigma_t * (g.horizon as f64).sqrt();
        let wsum: f64 = scored.iter().map(|x| x.0).sum();
        let mut pairs: Vec<(f64, f64)> = scored.iter().map(|(w, _, z)| (z * scale, w.max(1e-300))).collect();
        let quantiles: Vec<f64> = g.levels.iter().map(|&l| weighted_quantile(&mut pairs, l).unwrap_or(0.0)).collect();
        let point = if wsum > 0.0 { scored.iter().map(|(w, _, z)| w * z * scale).sum::<f64>() / wsum } else { 0.0 };
        Some(SeriesForecast {
            quantiles,
            levels: g.levels.clone(),
            point,
            analogs: scored.into_iter().map(|x| x.1).collect(),
        })
    }

    /// All forecasts of a series (index = step), computed in batches once per series.
    fn forecasts(&self, s: &Series) -> Arc<Vec<Option<SeriesForecast>>> {
        let key = (s.instrument_id.clone(), s.len(), s.ts_ms.last().copied().unwrap_or(0));
        if let Some(v) = self.cache.lock().ok().and_then(|c| c.get(&key).cloned()) {
            return v;
        }
        let v = Arc::new(self.compute_forecasts(s).unwrap_or_else(|e| {
            tracing_warn(&format!("neural forecast failed: {e}"));
            vec![None; s.len()]
        }));
        if let Ok(mut c) = self.cache.lock() {
            if c.len() > 64 {
                c.clear();
            }
            c.insert(key, v.clone());
        }
        v
    }

    fn compute_forecasts(&self, s: &Series) -> Result<Vec<Option<SeriesForecast>>, NeuralError> {
        let n = s.len();
        let g = &self.genome;
        let mut out = vec![None; n];
        let from = self.eval_from_ms.unwrap_or(i64::MIN);
        if g.family == BaselineFamily::NnRetrieval {
            // candidates need embeddings before the evaluation window too
            let mut idx = Vec::new();
            let mut rows = Vec::new();
            let mut sig = vec![None; n];
            for (t, slot) in sig.iter_mut().enumerate().skip(g.lookback) {
                if let Some((x, sg)) = self.input_at(s, t) {
                    idx.push(t);
                    rows.push(x);
                    *slot = Some(sg);
                }
            }
            let ys = self.predict_rows(&rows)?;
            let mut embs: Vec<Option<Vec<f32>>> = vec![None; n];
            for (t, y) in idx.into_iter().zip(ys) {
                embs[t] = Some(y);
            }
            let zs: Vec<Option<f32>> = (0..n).map(|t| sig[t].and_then(|sg| data::z_at(s, t, g.horizon, sg))).collect();
            for (t, sg) in sig.iter().enumerate().skip(g.lookback) {
                if s.ts_ms[t] < from {
                    continue;
                }
                if let Some(sg) = sg {
                    out[t] = self.retrieve(s, t, &embs, &zs, *sg);
                }
            }
            return Ok(out);
        }
        let mut idx = Vec::new();
        let mut rows = Vec::new();
        let mut sigmas = Vec::new();
        for t in g.lookback..n {
            if s.ts_ms[t] < from {
                continue;
            }
            if let Some((x, sg)) = self.input_at(s, t) {
                idx.push(t);
                rows.push(x);
                sigmas.push(sg);
            }
        }
        let ys = self.predict_rows(&rows)?;
        for ((t, y), sg) in idx.into_iter().zip(ys).zip(sigmas) {
            out[t] = Some(self.forecast_of(&y, sg));
        }
        Ok(out)
    }

    /// Reconstruction quality on steps with `ts ≥ from_ms` whose target is known (at most
    /// `max_rows`, evenly strided). `None` for families without a reconstruction target.
    pub fn recon_eval(
        &self,
        series: &[Series],
        from_ms: i64,
        max_rows: usize,
    ) -> Result<Option<ReconReport>, NeuralError> {
        let g = &self.genome;
        if !g.family.reconstructs() {
            return Ok(None);
        }
        let mut rows = Vec::new();
        let mut targets = Vec::new();
        for s in series {
            for t in g.lookback..s.len() {
                if s.ts_ms[t] < from_ms {
                    continue;
                }
                let Some((x, sg)) = self.input_at(s, t) else { continue };
                let Some((y, _)) = data::target_at(g, s, t, sg, &self.text) else { continue };
                rows.push(x);
                targets.push(y);
            }
        }
        if rows.is_empty() {
            return Ok(Some(ReconReport::default()));
        }
        let stride = rows.len().div_ceil(max_rows.max(1));
        let rows: Vec<Vec<f32>> = rows.into_iter().step_by(stride).collect();
        let targets: Vec<Vec<f32>> = targets.into_iter().step_by(stride).collect();
        let preds = self.predict_rows(&rows)?;
        let base = &self.meta.target_mean;
        let (mut loss, mut bl) = (0.0, 0.0);
        for (p, y) in preds.iter().zip(&targets) {
            loss += self.row_loss(p, y);
            bl += self.row_loss(base, y);
        }
        let n = preds.len() as f64;
        let (loss, baseline) = (loss / n, bl / n);
        let skill = if baseline > 0.0 { 1.0 - loss / baseline } else { 0.0 };
        Ok(Some(ReconReport { n: preds.len(), loss, baseline, skill }))
    }

    /// Family loss of one prediction (head output) against one target, in plain f64.
    pub fn row_loss(&self, pred: &[f32], target: &[f32]) -> f64 {
        match self.genome.family {
            BaselineFamily::NnChart | BaselineFamily::NnFuture => {
                let res = self.genome.nn.res;
                -pred.iter().zip(target).map(|(p, t)| f64::from(*t) * f64::from(p.max(1e-12)).ln()).sum::<f64>()
                    / res as f64
            }
            BaselineFamily::NnSummary => {
                -pred.iter().zip(target).map(|(p, t)| f64::from(*t) * f64::from(p.max(1e-12)).ln()).sum::<f64>()
                    / crate::summary::SLOTS.len() as f64
            }
            _ => {
                let dot: f64 = pred.iter().zip(target).map(|(a, b)| f64::from(a * b)).sum();
                let na: f64 = pred.iter().map(|a| f64::from(a * a)).sum::<f64>().sqrt();
                let nb: f64 = target.iter().map(|a| f64::from(a * a)).sum::<f64>().sqrt();
                if na > 0.0 && nb > 0.0 { 1.0 - dot / (na * nb) } else { 1.0 }
            }
        }
    }

    /// The decoder's latent from `(expert_index, mean)` pairs of a latent state: the experts of
    /// its mask, concatenated in index order. `None` when the decoder is unconditioned or the
    /// result does not have the trained dimension.
    pub fn select_latent<'a>(&self, experts: impl IntoIterator<Item = (u32, &'a [f32])>) -> Option<Vec<f32>> {
        if self.meta.layout.latent == 0 {
            return None;
        }
        crate::masked_latent(experts, self.genome.expert_mask).filter(|v| v.len() == self.meta.layout.latent)
    }

    /// Serving: the decoder's output at the last step of `s` with an explicit latent / text
    /// context (model-runtime). `latent` is the decoder's own (already masked, see
    /// [`Self::select_latent`]). `None` when the series is too short.
    pub fn serve(&self, s: &Series, latent: Option<&[f32]>, text: &TextTimeline) -> Option<Served> {
        self.serve_at(s, s.len().checked_sub(1)?, latent, text)
    }

    /// [`Self::serve`] at step `t` (reads `s[..=t]` and the news up to `ts[t]` only).
    pub fn serve_at(&self, s: &Series, t: usize, latent: Option<&[f32]>, text: &TextTimeline) -> Option<Served> {
        let g = &self.genome;
        if t >= s.len() {
            return None;
        }
        let (x, sigma) = data::input_at(g, &self.meta.layout, &self.meta.norm, s, t, latent, text)?;
        let y = self.predict_rows(&[x]).ok()?.pop()?;
        let mut out = Served { levels: g.levels.clone(), ..Default::default() };
        match g.family {
            BaselineFamily::NnForecast | BaselineFamily::NnFuture => {
                let f = self.forecast_of(&y, sigma);
                out.score = if g.family == BaselineFamily::NnFuture {
                    let res = g.nn.res;
                    2.0 * render::prob_up(&y[(res - 1) * res..]) - 1.0
                } else {
                    (f.point / (sigma * (g.horizon as f64).sqrt())).tanh()
                };
                out.components = f.quantiles.clone();
                out.quantiles = f.quantiles;
                out.recon = y;
            }
            BaselineFamily::NnRetrieval => {
                // embeddings of the series' own history (latents of the past are not kept by
                // the server; candidates use the given latent only for the query row)
                let mut embs: Vec<Option<Vec<f32>>> = vec![None; s.len()];
                let mut zs = vec![None; s.len()];
                let mut idx = Vec::new();
                let mut rows = Vec::new();
                for (e, z) in zs.iter_mut().enumerate().take(t).skip(g.lookback) {
                    if let Some((xe, se)) = data::input_at(g, &self.meta.layout, &self.meta.norm, s, e, None, text) {
                        *z = data::z_at(s, e, g.horizon, se);
                        idx.push(e);
                        rows.push(xe);
                    }
                }
                for (e, ye) in idx.into_iter().zip(self.predict_rows(&rows).ok()?) {
                    embs[e] = Some(ye);
                }
                embs[t] = Some(y);
                let f = self.retrieve(s, t, &embs, &zs, sigma)?;
                out.score = (f.point / (sigma * (g.horizon as f64).sqrt())).tanh();
                out.quantiles = f.quantiles;
                out.analogs = f.analogs;
            }
            BaselineFamily::NnChart => {
                out.score = render::direction(&y, g.nn.res);
                if let Some((target, _)) = data::target_at(g, s, t, sigma, text) {
                    out.components = vec![self.row_loss(&y, &target)];
                }
                out.recon = y;
            }
            BaselineFamily::NnSummary => {
                // event intensity in [0, 1] (the baseline's novelty scale); components follow the
                // model-runtime contract (`text.*`: components[0] = 1 − loss), then the signed
                // move expectation and the slot classes
                out.score = crate::summary::intensity_score(&y);
                let classes: Vec<f64> = crate::summary::argmax(&y).into_iter().map(|c| c as f64).collect();
                out.components = match data::target_at(g, s, t, sigma, text) {
                    Some((target, _)) => [1.0 - self.row_loss(&y, &target), crate::summary::move_score(&y)]
                        .into_iter()
                        .chain(classes)
                        .collect(),
                    None => Vec::new(),
                };
                out.text = crate::summary::summary_text(&y);
                out.recon = y;
            }
            BaselineFamily::NnFrame => {
                // frame embeddings have no direction: score 0, the embedding is the output; the
                // online loss only against a frame timeline (a news timeline gives no target)
                if let Some((target, _)) = data::target_at(g, s, t, sigma, text) {
                    out.components = vec![self.row_loss(&y, &target)];
                }
                out.recon = y;
            }
            BaselineFamily::NnNews | BaselineFamily::NnNarrative => {
                let dir = data::sentiment_direction(g.nn.tdim);
                let d: f64 = y.iter().zip(&dir).map(|(a, b)| f64::from(a * b)).sum();
                let (m, sd) = self.meta.sentiment;
                out.score = ((d - m) / sd.max(1e-6)).tanh();
                if g.family == BaselineFamily::NnNews
                    && let Some((target, _)) = data::target_at(g, s, t, sigma, text)
                {
                    out.components = vec![1.0 - self.row_loss(&y, &target)];
                }
                out.recon = y;
            }
            _ => return None,
        }
        Some(out)
    }

    /// Serialises weights + metadata into one safetensors blob.
    pub fn to_bundle(&self) -> Result<Vec<u8>, NeuralError> {
        let meta = serde_json::to_vec(&self.meta).map_err(|e| NeuralError::Bundle(e.to_string()))?;
        let mut map: HashMap<String, Tensor> = self
            .params
            .iter()
            .map(|(k, t)| Ok((k.clone(), t.to_device(&Device::Cpu)?)))
            .collect::<Result<_, candle_core::Error>>()?;
        let n = meta.len();
        map.insert(META_TENSOR.to_owned(), Tensor::from_vec(meta, n, &Device::Cpu)?);
        let path = std::env::temp_dir().join(format!(
            "fa-bundle-{}-{}.safetensors",
            std::process::id(),
            BUNDLE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        candle_core::safetensors::save(&map, &path)?;
        let bytes = std::fs::read(&path).map_err(|e| NeuralError::Bundle(e.to_string()));
        let _ = std::fs::remove_file(&path);
        bytes
    }

    /// Only the metadata of a bundle (cheap: no network is built).
    pub fn meta_of(bytes: &[u8]) -> Result<BundleMeta, NeuralError> {
        let map = candle_core::safetensors::load_buffer(bytes, &Device::Cpu)?;
        Self::meta_from(&map)
    }

    fn meta_from(map: &HashMap<String, Tensor>) -> Result<BundleMeta, NeuralError> {
        let raw = map.get(META_TENSOR).ok_or_else(|| NeuralError::Bundle("no __meta__ tensor".into()))?;
        let meta: BundleMeta =
            serde_json::from_slice(&raw.to_vec1::<u8>()?).map_err(|e| NeuralError::Bundle(e.to_string()))?;
        if meta.format != BUNDLE_FORMAT {
            return Err(NeuralError::Bundle(format!("unknown format `{}`", meta.format)));
        }
        Ok(meta)
    }

    /// Loads a bundle as a **frozen** decoder (weights are constants: no gradient flows into
    /// them, gradients still flow through them into the inputs). Bundles trained with an older
    /// latent normaliser are refused ([`NeuralError::StaleNormaliser`]); use
    /// [`Self::from_bundle_any`] to inspect them.
    pub fn from_bundle(bytes: &[u8], dev: &Device) -> Result<Self, NeuralError> {
        let d = Self::from_bundle_any(bytes, dev)?;
        if !d.meta.is_current() {
            return Err(NeuralError::StaleNormaliser(d.meta.norm_version));
        }
        Ok(d)
    }

    /// [`Self::from_bundle`] without the normaliser-version check (warm starts, inspection).
    pub fn from_bundle_any(bytes: &[u8], dev: &Device) -> Result<Self, NeuralError> {
        let map = candle_core::safetensors::load_buffer(bytes, dev)?;
        let meta = Self::meta_from(&map)?;
        let genome = DecoderGenome::parse(&meta.genome)?;
        if !genome.family.is_neural() {
            return Err(NeuralError::NotNeural(meta.genome.clone()));
        }
        let params: Vec<(String, Tensor)> =
            map.iter().filter(|(k, _)| k.as_str() != META_TENSOR).map(|(k, t)| (k.clone(), t.clone())).collect();
        let vb = VarBuilder::from_tensors(map, net::DTYPE, dev);
        let net = Net::new(vb, meta.layout, &genome.nn)?;
        Ok(Self {
            genome,
            meta,
            net,
            params,
            dev: dev.clone(),
            ctx: Arc::new(DecodeCtx::default()),
            text: Arc::new(TextTimeline::new()),
            eval_from_ms: None,
            cache: Mutex::new(HashMap::new()),
        })
    }
}

static BUNDLE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn tracing_warn(msg: &str) {
    eprintln!("fa-decoders: {msg}");
}

impl SeriesDecoder for NeuralDecoder {
    fn genome(&self) -> &DecoderGenome {
        &self.genome
    }

    fn warmup(&self) -> usize {
        self.genome.lookback + 1
    }

    fn forecast(&self, series: &Series, t: usize) -> Option<SeriesForecast> {
        if !self.genome.family.forecasts() {
            return None;
        }
        self.forecasts(series).get(t).cloned().flatten()
    }

    fn ops_per_forecast(&self, _series_len: usize) -> f64 {
        let mut f = self.net.flops();
        if self.genome.family == BaselineFamily::NnRetrieval {
            f += 2.0 * (self.genome.nn.pool * self.genome.nn.emb) as f64;
        }
        f
    }
}

#[cfg(test)]
mod tests;
