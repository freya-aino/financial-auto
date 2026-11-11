//! Training of a neural genome (hybrid ES + SGD, spec 34 §3: evolution picks the genome, this is
//! the SGD part). Point-in-time: only samples whose target is known by the train cutoff are used.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Instant,
};

use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use fa_core::rng::Rng;
use serde::{Deserialize, Serialize};

use super::{
    BUNDLE_FORMAT, BundleMeta, NeuralDecoder, NeuralError,
    data::{self, Layout, Norm, TextTimeline},
    net::{self, Net},
};
use crate::{
    DecodeCtx,
    baseline::Series,
    genome::{BaselineFamily, DecoderGenome},
};

/// Training controls.
pub struct TrainOptions {
    /// Train / evaluation boundary; `None` = `train_frac` of the pooled time span.
    pub cutoff_ms: Option<i64>,
    /// Cap on training rows (evenly strided).
    pub max_samples: usize,
    /// Checked every few steps; `true` ends training early (`TrainReport.interrupted`).
    pub should_stop: Option<Box<dyn Fn() -> bool + Send + Sync>>,
    /// Parent bundle: variables with the same name and shape start from its weights (§6.2.3).
    pub warm_start: Option<Vec<u8>>,
    /// SGD steps instead of the genome's `steps` (successive-halving rungs train a fraction of
    /// the full budget, spec 34 §6.2 step 4).
    pub steps: Option<usize>,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self { cutoff_ms: None, max_samples: 20_000, should_stop: None, warm_start: None, steps: None }
    }
}

/// What training did.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct TrainReport {
    pub train_rows: usize,
    pub val_rows: usize,
    pub steps: usize,
    /// Step whose weights were kept (best validation loss; 0 = the initial weights).
    #[serde(default)]
    pub best_step: usize,
    /// Mean family loss on (a sample of) the training rows after training.
    pub train_loss: f64,
    /// Same before the first step (initial weights).
    pub initial_loss: f64,
    /// Mean family loss on the last 10 % of the training rows (by time), held out from SGD.
    pub val_loss: f64,
    pub elapsed_ms: f64,
    pub latent_dim: usize,
    /// Share of training rows that had a latent.
    pub latent_coverage: f64,
    pub interrupted: bool,
    /// Variables copied from a warm-start bundle.
    pub warm_started: usize,
    pub device: String,
}

/// `(context, raw latent, target, t_ms, outcome_end_ms)` before latent standardisation.
type RawRow = (Vec<f32>, Option<Vec<f32>>, Vec<f32>, i64, i64);

struct Row {
    x: Vec<f32>,
    y: Vec<f32>,
    t_ms: i64,
    end_ms: i64,
}

/// Fits `g` on `series` (+ latents in `ctx`, news in `text`). Errors when fewer than 32 rows have
/// a known target before the cutoff.
pub fn train(
    g: &DecoderGenome,
    series: &[Series],
    ctx: Arc<DecodeCtx>,
    text: Arc<TextTimeline>,
    opts: &TrainOptions,
) -> Result<NeuralDecoder, NeuralError> {
    let started = Instant::now();
    if !g.family.is_neural() {
        return Err(NeuralError::NotNeural(g.to_string()));
    }
    // expert mask (spec 34 §4.2): the decoder only ever sees the selected experts
    let latent_layout: Vec<(u32, u32)> = series
        .iter()
        .map(|s| ctx.layout(data::latent_instrument(g, s)))
        .find(|l| !l.is_empty())
        .map(<[_]>::to_vec)
        .unwrap_or_default();
    let ctx = if g.expert_mask != 0 { Arc::new(ctx.masked(g.expert_mask)) } else { ctx };
    let latent = if g.latent_weight > 0.0 { data::latent_dim(series, &ctx, g.source.as_deref()) } else { 0 };
    let lay = Layout::of(g, latent);
    let (t0, t1) = series
        .iter()
        .filter_map(|s| Some((*s.ts_ms.first()?, *s.ts_ms.last()?)))
        .fold((i64::MAX, i64::MIN), |(a, b), (x, y)| (a.min(x), b.max(y)));
    if t0 > t1 {
        return Err(NeuralError::InsufficientData("no series".into()));
    }
    let cutoff = opts.cutoff_ms.unwrap_or(t0 + ((t1 - t0) as f64 * g.nn.train_frac) as i64);

    // 1. raw rows (latent kept raw until the normaliser is fitted)
    let empty = Norm::default();
    let mut raw: Vec<RawRow> = Vec::new();
    let unconditioned = Layout { latent: 0, ..lay };
    for s in series {
        for t in g.lookback..s.len() {
            if s.ts_ms[t] > cutoff {
                break;
            }
            let lat = ctx
                .latent_as_of(data::latent_instrument(g, s), s.ts_ms[t])
                .map(|l| l.vec.clone())
                .filter(|v| v.len() == latent);
            if lay.seq == 0 && lay.extra == 0 && lat.is_none() {
                continue; // latent-only input (conditioned nn_chart) without a latent: nothing to learn from
            }
            let Some((x, sg)) = data::input_at(g, &unconditioned, &empty, s, t, None, &text) else { continue };
            let Some((y, end_ms)) = data::target_at(g, s, t, sg, &text) else { continue };
            if end_ms > cutoff {
                continue;
            }
            raw.push((x, lat, y, s.ts_ms[t], end_ms));
        }
    }
    if raw.len() < 32 && lay.seq == 0 && lay.extra == 0 && latent > 0 {
        // latent-only input (conditioned nn_chart) but too few latents in the training window:
        // fall back to the unconditioned decoder (return window → chart)
        let mut g0 = g.clone();
        g0.latent_weight = 0.0;
        return train(&g0, series, ctx, text, opts);
    }
    if raw.len() < 32 {
        return Err(NeuralError::InsufficientData(format!(
            "{} rows with a known target before the cutoff {cutoff} (need ≥ 32; family {}, {} series over [{t0}, {t1}], text items {} ({} matched to an instrument; scope {}) over {:?})",
            raw.len(),
            g.family.as_str(),
            series.len(),
            text.len(),
            text.matched(),
            if g.nn.news_inst { "instrument" } else { "market" },
            text.span()
        )));
    }
    raw.sort_by_key(|r| r.3);
    let stride = raw.len().div_ceil(opts.max_samples.max(32));
    let raw: Vec<_> = raw.into_iter().step_by(stride).collect();
    let lats: Vec<&[f32]> = raw.iter().filter_map(|r| r.1.as_deref()).collect();
    let latent_coverage = lats.len() as f64 / raw.len() as f64;
    let norm = Norm::fit(&lats, latent);
    let rows: Vec<Row> = raw
        .into_iter()
        .map(|(mut x, lat, y, t_ms, end_ms)| {
            x.extend(norm.latent_block(lat.as_deref(), latent));
            Row { x, y, t_ms, end_ms }
        })
        .collect();
    let n_val = (rows.len() / 10).max(8);
    let n_train = rows.len() - n_val;

    // 2. target statistics (reconstruction baseline, sentiment normalisation)
    let tw = lay.target;
    let mut target_mean = vec![0.0f32; tw];
    for r in &rows[..n_train] {
        for (m, y) in target_mean.iter_mut().zip(&r.y) {
            *m += y / n_train as f32;
        }
    }
    let sentiment = if g.family.uses_text() {
        let dir = data::sentiment_direction(g.nn.tdim);
        let ds: Vec<f64> =
            rows[..n_train].iter().map(|r| r.y.iter().zip(&dir).map(|(a, b)| f64::from(a * b)).sum()).collect();
        let m = ds.iter().sum::<f64>() / ds.len() as f64;
        let sd = (ds.iter().map(|d| (d - m).powi(2)).sum::<f64>() / ds.len() as f64).sqrt();
        (m, sd.max(1e-6))
    } else {
        (0.0, 1.0)
    };

    // 3. network
    let dev = net::device();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, net::DTYPE, &dev);
    let network = Net::new(vb, lay, &g.nn)?;
    let seed = g.nn.seed ^ crate::genome::stable_hash(g.family.as_str());
    net::seeded_init(&varmap, seed)?;
    let warm_started = match &opts.warm_start {
        Some(bytes) => warm_start(&varmap, bytes, &dev)?,
        None => 0,
    };
    let mut opt =
        AdamW::new(varmap.all_vars(), ParamsAdamW { lr: g.nn.lr, weight_decay: g.nn.wd, ..Default::default() })?;

    let step_ms = series.iter().map(|s| s.step_ms).find(|s| *s > 0).unwrap_or(1000);
    let gap_ms = (g.min_gap.max(g.horizon) as i64) * step_ms;
    let ctx_loss = LossCtx { g, lay, rows: &rows, gap_ms, dev: &dev };
    let eval_idx: Vec<usize> = (0..n_train).step_by(n_train.div_ceil(512).max(1)).collect();
    let initial_loss = ctx_loss.mean_loss(&network, &eval_idx)?;

    // 4. SGD
    let mut rng = Rng::seed(seed.wrapping_add(1));
    let bs = g.nn.batch.min(n_train);
    let mut steps = 0usize;
    let mut interrupted = false;
    // Early stopping on the held-out tail: the weights with the best validation loss win.
    let val_idx: Vec<usize> = (n_train..rows.len()).collect();
    let total_steps = opts.steps.unwrap_or(g.nn.steps).max(1);
    let check_every = (total_steps / 10).clamp(10, 200);
    let mut best = (ctx_loss.mean_loss(&network, &val_idx)?, 0usize, snapshot(&varmap)?);
    for step in 0..total_steps {
        if step % 8 == 0 && opts.should_stop.as_ref().is_some_and(|f| f()) {
            interrupted = true;
            break;
        }
        let batch: Vec<usize> = (0..bs).map(|_| rng.below(n_train)).collect();
        let Some(loss) = ctx_loss.batch_loss(&network, &batch, Some((&mut rng, n_train)))? else { continue };
        opt.backward_step(&loss)?;
        steps += 1;
        if steps.is_multiple_of(check_every) || step + 1 == total_steps {
            let v = ctx_loss.mean_loss(&network, &val_idx)?;
            if v < best.0 || !best.0.is_finite() {
                best = (v, steps, snapshot(&varmap)?);
            }
        }
    }
    restore(&varmap, &best.2)?;
    let best_step = best.1;
    let train_loss = ctx_loss.mean_loss(&network, &eval_idx)?;
    let val_loss = ctx_loss.mean_loss(&network, &val_idx)?;

    let params: Vec<(String, Tensor)> = {
        let data = varmap.data().lock().map_err(|e| NeuralError::Bundle(e.to_string()))?;
        let mut v: Vec<(String, Tensor)> = data.iter().map(|(k, v)| (k.clone(), v.as_tensor().detach())).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    };
    let report = TrainReport {
        train_rows: n_train,
        val_rows: n_val,
        steps,
        best_step,
        train_loss,
        initial_loss,
        val_loss,
        elapsed_ms: started.elapsed().as_secs_f64() * 1e3,
        latent_dim: latent,
        latent_coverage,
        interrupted,
        warm_started,
        device: format!("{dev:?}"),
    };
    let meta = BundleMeta {
        format: BUNDLE_FORMAT.into(),
        genome: g.to_string(),
        family: g.family.as_str().into(),
        output_family: g.family.output_family().as_str().into(),
        layout: lay,
        norm,
        norm_version: super::NORM_VERSION,
        latent_layout: if latent > 0 { latent_layout } else { Vec::new() },
        target_mean,
        sentiment,
        step_ms,
        cutoff_ms: cutoff,
        train: report,
        objectives: Default::default(),
        score: 0.0,
        job_id: String::new(),
        generation: String::new(),
    };
    // Rebuild the network on the detached tensors so the decoder holds no optimiser graph.
    let map: HashMap<String, Tensor> = params.iter().cloned().collect();
    let frozen = Net::new(VarBuilder::from_tensors(map, net::DTYPE, &dev), lay, &g.nn)?;
    let mut genome = g.clone();
    genome.id = meta.genome.clone();
    Ok(NeuralDecoder {
        genome,
        meta,
        net: frozen,
        params,
        dev,
        ctx,
        text,
        eval_from_ms: Some(cutoff),
        cache: Mutex::new(HashMap::new()),
    })
}

fn snapshot(varmap: &VarMap) -> Result<Vec<(String, Tensor)>, NeuralError> {
    let data = varmap.data().lock().map_err(|e| NeuralError::Bundle(e.to_string()))?;
    data.iter().map(|(k, v)| Ok((k.clone(), v.as_tensor().copy()?))).collect()
}

fn restore(varmap: &VarMap, snap: &[(String, Tensor)]) -> Result<(), NeuralError> {
    let data = varmap.data().lock().map_err(|e| NeuralError::Bundle(e.to_string()))?;
    for (k, t) in snap {
        if let Some(v) = data.get(k) {
            v.set(t)?;
        }
    }
    Ok(())
}

/// Copies every variable of the bundle whose name and shape match. Returns the count.
fn warm_start(varmap: &VarMap, bytes: &[u8], dev: &Device) -> Result<usize, NeuralError> {
    let parent = candle_core::safetensors::load_buffer(bytes, dev)?;
    let data = varmap.data().lock().map_err(|e| NeuralError::Bundle(e.to_string()))?;
    let mut n = 0;
    for (name, var) in data.iter() {
        if let Some(t) = parent.get(name)
            && t.dims() == var.dims()
        {
            var.set(&t.to_dtype(net::DTYPE)?)?;
            n += 1;
        }
    }
    Ok(n)
}

struct LossCtx<'a> {
    g: &'a DecoderGenome,
    lay: Layout,
    rows: &'a [Row],
    gap_ms: i64,
    dev: &'a Device,
}

impl LossCtx<'_> {
    fn xs(&self, idx: &[usize]) -> candle_core::Result<Tensor> {
        let r: Vec<&[f32]> = idx.iter().map(|i| self.rows[*i].x.as_slice()).collect();
        net::rows_tensor(&r, self.lay.input(), self.dev)
    }

    fn ys(&self, idx: &[usize]) -> candle_core::Result<Tensor> {
        let r: Vec<&[f32]> = idx.iter().map(|i| self.rows[*i].y.as_slice()).collect();
        net::rows_tensor(&r, self.lay.target, self.dev)
    }

    /// Family loss of a batch. Retrieval draws its candidate pool from `pool` (rng, n_train);
    /// without it (evaluation) the candidates are the batch itself. `None` when no query of the
    /// batch has a leakage-safe candidate.
    fn batch_loss(
        &self,
        network: &Net,
        idx: &[usize],
        pool: Option<(&mut Rng, usize)>,
    ) -> Result<Option<Tensor>, NeuralError> {
        let g = self.g;
        let out = network.forward(&self.xs(idx)?)?;
        Ok(Some(match g.family {
            BaselineFamily::NnForecast => net::pinball_loss(&net::monotone(&out)?, &self.ys(idx)?, &g.levels)?,
            BaselineFamily::NnChart | BaselineFamily::NnFuture => {
                net::raster_ce(&net::raster_logp(&out, g.nn.res)?, &self.ys(idx)?, g.nn.res)?
            }
            BaselineFamily::NnNews | BaselineFamily::NnNarrative => net::cosine_loss(&out, &self.ys(idx)?)?,
            BaselineFamily::NnRetrieval => {
                let cand: Vec<usize> = match pool {
                    Some((rng, n)) => (0..(4 * idx.len()).min(n)).map(|_| rng.below(n)).collect(),
                    None => idx.to_vec(),
                };
                // keep queries with at least one candidate whose outcome ended ≥ gap before them
                let mut keep = Vec::new();
                let mut bias = Vec::new();
                for (qi, &q) in idx.iter().enumerate() {
                    let tq = self.rows[q].t_ms;
                    let row: Vec<f32> = cand
                        .iter()
                        .map(|&c| {
                            let r = &self.rows[c];
                            if c != q && r.end_ms <= tq - self.gap_ms {
                                (-((tq - r.t_ms) as f64 / 86_400_000.0) / g.tau_days) as f32
                            } else {
                                -1e9
                            }
                        })
                        .collect();
                    if row.iter().any(|b| *b > -1e8) {
                        keep.push(qi);
                        bias.extend(row);
                    }
                }
                if keep.is_empty() {
                    return Ok(None);
                }
                let keep_t =
                    Tensor::from_vec(keep.iter().map(|k| *k as u32).collect::<Vec<_>>(), keep.len(), self.dev)?;
                let q = net::l2_normalize(&out.index_select(&keep_t, 0)?)?;
                let c = net::l2_normalize(&network.forward(&self.xs(&cand)?)?)?;
                let yq = self.ys(&keep.iter().map(|k| idx[*k]).collect::<Vec<_>>())?;
                let zc = self.ys(&cand)?.t()?.contiguous()?;
                let bias = Tensor::from_vec(bias, (keep.len(), cand.len()), self.dev)?;
                net::soft_knn_crps(&q, &c, &yq, &zc, &bias, g.nn.temp)?
            }
            _ => return Err(NeuralError::NotNeural(g.to_string())),
        }))
    }

    /// Mean loss over `idx` in chunks (no gradient use). NaN when empty.
    fn mean_loss(&self, network: &Net, idx: &[usize]) -> Result<f64, NeuralError> {
        let (mut sum, mut n) = (0.0, 0usize);
        for chunk in idx.chunks(256) {
            if let Some(l) = self.batch_loss(network, chunk, None)? {
                sum += f64::from(l.to_scalar::<f32>()?) * chunk.len() as f64;
                n += chunk.len();
            }
        }
        Ok(if n > 0 { sum / n as f64 } else { f64::NAN })
    }
}
