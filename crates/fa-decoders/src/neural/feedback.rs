//! Decoder → encoder feedback (D-009, spec 34 §6.6): the top-k elite decoders per family are
//! loaded as **frozen** modules and give the encoder trainer an auxiliary loss on its latents,
//! `L_gen = Σ_d w_d · L_d(decoder_d(context, latent), target_d)`, differentiable w.r.t. the
//! latents only (the decoder weights are constants).
//!
//! Typical trainer loop (spec 36):
//! ```ignore
//! use fa_decoders::neural::feedback::{self, EliteCard, FeedbackSet};
//! // 1. list the cards  fa-models/decoders/**/*.card.json  (fa_blob::BlobStore::list + get)
//! let cards: Vec<EliteCard> = /* serde_json::from_slice(&bytes) for each card */;
//! // 2. top-k per family whose decoders read latents of this dimension
//! let elites = feedback::select_top_k(&cards, 8, Some(latent_dim));
//! // 3. fetch the bundles (card.bucket / card.key) and weight them (equal family quota)
//! let weighted = feedback::quota_weights(&elites);
//! let bundles: Vec<(Vec<u8>, f64)> = /* (blob.get(card key), w) */;
//! let set = FeedbackSet::load(&bundles, latent_dim, &device)?;
//! // 4. per batch: context + targets from the price series / news, then the loss on the latents
//! let batch = set.prepare(&steps /* [(&Series, step)] */, &text_timeline)?;
//! let aux = set.aux_loss(&latents /* [B, latent_dim], requires grad */, &batch)?;
//! let loss = (ssl_loss + aux.total.affine(lambda_gen, 0.0)?)?;
//! ```
//! Rows whose context or target is not available (start of a series, no news in the window)
//! are left out per decoder, so `latents` always has one row per requested step.

use std::collections::BTreeMap;

use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use super::{
    BundleMeta, NeuralDecoder, NeuralError,
    data::{self, TextTimeline},
    net,
};
use crate::{baseline::Series, genome::BaselineFamily};

/// Bucket of decoder bundles and cards (`fa_blob::buckets::MODELS`).
pub const BUCKET: &str = "fa-models";
/// Key prefix of all decoder bundles / cards.
pub const PREFIX: &str = "decoders/";
/// Suffix of the JSON card next to each bundle.
pub const CARD_SUFFIX: &str = ".card.json";

/// Objective directions used for elite ranking (`true` = maximise).
pub const OBJECTIVE_DIRECTIONS: [(&str, bool); 8] = [
    ("crps", false),
    ("crps_skill", true),
    ("coverage_err", false),
    ("hit_rate", true),
    ("trade_sharpe", true),
    ("trade_psr", true),
    ("compute_cost", false),
    ("recon_skill", true),
];

/// Bundle key: `decoders/<generation>/<family>-<genome hash>-<job_id>.safetensors`.
pub fn bundle_key(generation: &str, meta: &BundleMeta, job_id: &str) -> String {
    let generation = if generation.is_empty() { "adhoc" } else { generation };
    format!(
        "{PREFIX}{generation}/{}-{:08x}-{job_id}.safetensors",
        meta.family,
        crate::genome::stable_hash(&meta.genome) as u32
    )
}

/// Card key of a bundle key.
pub fn card_key(bundle_key: &str) -> String {
    format!("{}{CARD_SUFFIX}", bundle_key.trim_end_matches(".safetensors"))
}

/// The JSON card stored next to a bundle: metadata (incl. objectives) + where the weights are.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EliteCard {
    pub meta: BundleMeta,
    pub bucket: String,
    pub key: String,
    /// Wall time of the evaluation (ms).
    pub created_ms: i64,
}

impl EliteCard {
    pub fn output_family(&self) -> &str {
        &self.meta.output_family
    }

    fn objective_vec(&self) -> Vec<f64> {
        OBJECTIVE_DIRECTIONS
            .iter()
            .map(|(k, max)| match self.meta.objectives.get(*k) {
                Some(v) if v.is_finite() => {
                    if *max {
                        -v
                    } else {
                        *v
                    }
                }
                _ => f64::INFINITY,
            })
            .collect()
    }
}

fn dominates(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
}

/// Pareto rank (0 = non-dominated) of each vector (minimised).
pub fn pareto_ranks(objs: &[Vec<f64>]) -> Vec<usize> {
    let n = objs.len();
    let mut rank = vec![usize::MAX; n];
    let mut left: Vec<usize> = (0..n).collect();
    let mut r = 0;
    while !left.is_empty() {
        let front: Vec<usize> =
            left.iter().copied().filter(|&i| !left.iter().any(|&j| j != i && dominates(&objs[j], &objs[i]))).collect();
        for &i in &front {
            rank[i] = r;
        }
        left.retain(|i| !front.contains(i));
        r += 1;
    }
    rank
}

/// Top-`k` cards per output family: Pareto rank over [`OBJECTIVE_DIRECTIONS`] within the family,
/// then family score (`crps_skill` / `recon_skill`), newest first on ties. Only decoders that beat
/// their naive baseline (`score > 0`) and use the current latent normaliser
/// ([`super::NORM_VERSION`]) qualify; with `latent_dim`, only latent-conditioned decoders whose
/// **full** (pre-mask) latent has exactly that dimension and that read the step's own instrument
/// (no cross-asset `src`), as the feedback path needs. Result is grouped by family.
pub fn select_top_k(cards: &[EliteCard], k: usize, latent_dim: Option<usize>) -> Vec<EliteCard> {
    let mut by_family: BTreeMap<&str, Vec<&EliteCard>> = BTreeMap::new();
    for c in cards {
        if !(c.meta.score.is_finite() && c.meta.score > 0.0) || !c.meta.is_current() {
            continue;
        }
        if let Some(d) = latent_dim
            && (c.meta.full_latent_dim() != d || c.meta.layout.latent == 0 || is_cross_asset(c))
        {
            continue;
        }
        by_family.entry(c.output_family()).or_default().push(c);
    }
    let mut out = Vec::new();
    for (_, fam) in by_family {
        let objs: Vec<Vec<f64>> = fam.iter().map(|c| c.objective_vec()).collect();
        let ranks = pareto_ranks(&objs);
        let mut idx: Vec<usize> = (0..fam.len()).collect();
        idx.sort_by(|&a, &b| {
            ranks[a]
                .cmp(&ranks[b])
                .then(fam[b].meta.score.total_cmp(&fam[a].meta.score))
                .then(fam[b].created_ms.cmp(&fam[a].created_ms))
        });
        out.extend(idx.into_iter().take(k).map(|i| fam[i].clone()));
    }
    out
}

fn is_cross_asset(c: &EliteCard) -> bool {
    crate::genome::DecoderGenome::parse(&c.meta.genome).is_ok_and(|g| g.source.is_some())
}

/// Cards trained with an older latent normaliser (first-generation elites, D-068) that would
/// otherwise qualify (`score > 0`): they are never served nor used for feedback; their genomes
/// are re-trained (model-runtime `ctl.decoders.retrain`). Best score first.
pub fn stale_cards(cards: &[EliteCard]) -> Vec<EliteCard> {
    let mut v: Vec<EliteCard> = cards
        .iter()
        .filter(|c| !c.meta.is_current() && c.meta.score.is_finite() && c.meta.score > 0.0)
        .cloned()
        .collect();
    v.sort_by(|a, b| b.meta.score.total_cmp(&a.meta.score).then(b.created_ms.cmp(&a.created_ms)));
    v
}

/// Feedback weights `w_d` ∝ family quota × rank (§6.6): every family present gets an equal share,
/// split inside the family by `1/(1 + position)` (best first, as ordered by [`select_top_k`]).
pub fn quota_weights(elites: &[EliteCard]) -> Vec<(EliteCard, f64)> {
    let mut by_family: BTreeMap<String, Vec<&EliteCard>> = BTreeMap::new();
    for c in elites {
        by_family.entry(c.output_family().to_owned()).or_default().push(c);
    }
    let nf = by_family.len().max(1) as f64;
    let mut out = Vec::new();
    for (_, fam) in by_family {
        let raw: Vec<f64> = (0..fam.len()).map(|i| 1.0 / (1.0 + i as f64)).collect();
        let s: f64 = raw.iter().sum();
        out.extend(fam.into_iter().zip(raw).map(|(c, r)| (c.clone(), r / s / nf)));
    }
    out
}

/// One frozen elite.
pub struct FrozenElite {
    pub decoder: NeuralDecoder,
    pub weight: f64,
    lat_mean: Tensor,
    lat_std: Tensor,
    /// Columns of the encoder latent this elite reads (its expert mask); `None` = all.
    cols: Option<Tensor>,
}

/// The loaded elite set.
pub struct FeedbackSet {
    pub elites: Vec<FrozenElite>,
    latent_dim: usize,
    /// `(decoder id or index, reason)` of bundles that were not loaded.
    pub skipped: Vec<(String, String)>,
}

/// Per-decoder rows of one batch.
struct Part {
    rows: Tensor,
    context: Option<Tensor>,
    target: Tensor,
    times: Vec<(i64, i64)>,
}

/// Context and targets of one batch, per elite (see [`FeedbackSet::prepare`]).
pub struct FeedbackBatch {
    pub n: usize,
    parts: Vec<Option<Part>>,
}

impl FeedbackBatch {
    /// Rows used per elite (`0` = the elite sits this batch out).
    pub fn rows_per_elite(&self) -> Vec<usize> {
        self.parts.iter().map(|p| p.as_ref().map_or(0, |p| p.rows.dims1().unwrap_or(0))).collect()
    }
}

/// The auxiliary loss and its parts.
pub struct FeedbackLoss {
    /// `Σ w_d · L_d` (scalar; differentiable w.r.t. the latents).
    pub total: Tensor,
    /// `(decoder id, family, loss, weight)` per elite that contributed.
    pub parts: Vec<(String, String, f32, f64)>,
}

impl FeedbackSet {
    /// Loads `(bundle bytes, weight)` pairs as frozen decoders on `dev`. Bundles that are not
    /// latent-conditioned or whose latent dimension differs from `latent_dim` are skipped (and
    /// listed in [`Self::skipped`]); weights are renormalised to sum 1 over the loaded ones.
    pub fn load(bundles: &[(Vec<u8>, f64)], latent_dim: usize, dev: &Device) -> Result<Self, NeuralError> {
        let mut elites = Vec::new();
        let mut skipped = Vec::new();
        for (i, (bytes, w)) in bundles.iter().enumerate() {
            let d = match NeuralDecoder::from_bundle(bytes, dev) {
                Ok(d) => d,
                Err(e) => {
                    skipped.push((format!("#{i}"), e.to_string()));
                    continue;
                }
            };
            if d.meta.full_latent_dim() != latent_dim || latent_dim == 0 || d.meta.layout.latent == 0 {
                skipped.push((d.decoder_id(), format!("latent dimension {} ≠ {latent_dim}", d.meta.full_latent_dim())));
                continue;
            }
            if d.genome().source.is_some() {
                skipped.push((d.decoder_id(), "cross-asset decoder (reads another instrument's latent)".into()));
                continue;
            }
            let own = d.meta.layout.latent;
            let cols = if d.genome().expert_mask != 0 || own != latent_dim {
                let c = crate::mask_columns(&d.meta.latent_layout, latent_dim, d.genome().expert_mask);
                if c.len() != own {
                    skipped.push((
                        d.decoder_id(),
                        format!("expert mask selects {} columns, decoder reads {own}", c.len()),
                    ));
                    continue;
                }
                Some(Tensor::from_vec(c, own, dev)?)
            } else {
                None
            };
            let lat_mean = Tensor::from_vec(d.meta.norm.lat_mean.clone(), (1, own), dev)?;
            let lat_std = Tensor::from_vec(d.meta.norm.lat_std.clone(), (1, own), dev)?;
            elites.push(FrozenElite { decoder: d, weight: w.max(0.0), lat_mean, lat_std, cols });
        }
        let s: f64 = elites.iter().map(|e| e.weight).sum();
        if s > 0.0 {
            elites.iter_mut().for_each(|e| e.weight /= s);
        }
        Ok(Self { elites, latent_dim, skipped })
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    pub fn is_empty(&self) -> bool {
        self.elites.is_empty()
    }

    /// Builds context and targets for `steps` (`(series, step index)`, row `i` of the latents the
    /// trainer will pass belongs to `steps[i]`) from the price series and the news timeline.
    pub fn prepare(&self, steps: &[(&Series, usize)], text: &TextTimeline) -> Result<FeedbackBatch, NeuralError> {
        let mut parts = Vec::with_capacity(self.elites.len());
        for e in &self.elites {
            let d = &e.decoder;
            let g = d.genome();
            let lay = d.meta.layout;
            let ctx_w = lay.context();
            let mut rows = Vec::new();
            let mut ctx = Vec::new();
            let mut tgt = Vec::new();
            let mut times = Vec::new();
            for (i, (s, t)) in steps.iter().enumerate() {
                let ctx_lay = data::Layout { latent: 0, ..lay };
                let Some((x, sigma)) = data::input_at(g, &ctx_lay, &data::Norm::default(), s, *t, None, text) else {
                    continue;
                };
                let Some((y, end)) = data::target_at(g, s, *t, sigma, text) else { continue };
                rows.push(i as u32);
                ctx.extend(x);
                tgt.extend(y);
                times.push((s.ts_ms[*t], end));
            }
            if rows.is_empty() {
                parts.push(None);
                continue;
            }
            let n = rows.len();
            let dev = d.device();
            parts.push(Some(Part {
                rows: Tensor::from_vec(rows, n, dev)?,
                context: if ctx_w > 0 { Some(Tensor::from_vec(ctx, (n, ctx_w), dev)?) } else { None },
                target: Tensor::from_vec(tgt, (n, lay.target), dev)?,
                times,
            }));
        }
        Ok(FeedbackBatch { n: steps.len(), parts })
    }

    /// `L_gen` for `latents [n, latent_dim]` (any float dtype; computed in f32). The gradient of
    /// `total` reaches `latents` through every elite that had rows in the batch.
    pub fn aux_loss(&self, latents: &Tensor, batch: &FeedbackBatch) -> Result<FeedbackLoss, NeuralError> {
        let (n, dim) = latents.dims2()?;
        if n != batch.n || dim != self.latent_dim {
            return Err(NeuralError::Bundle(format!(
                "latents are [{n}, {dim}], the batch / elites expect [{}, {}]",
                batch.n, self.latent_dim
            )));
        }
        let latents = latents.to_dtype(DType::F32)?;
        let mut total = Tensor::zeros((), DType::F32, latents.device())?;
        let mut parts = Vec::new();
        for (e, part) in self.elites.iter().zip(&batch.parts) {
            let Some(p) = part else { continue };
            let d = &e.decoder;
            let g = d.genome();
            // same standardisation + clipping as the decoder saw in training (data::Norm::latent_block)
            let rows = latents.index_select(&p.rows, 0)?;
            let rows = match &e.cols {
                Some(c) => rows.index_select(c, 1)?,
                None => rows,
            };
            let lat = rows
                .broadcast_sub(&e.lat_mean)?
                .broadcast_div(&e.lat_std)?
                .clamp(-data::LATENT_CLIP, data::LATENT_CLIP)?;
            let r = lat.dim(0)?;
            let flag = Tensor::ones((r, 1), DType::F32, lat.device())?;
            let x = match &p.context {
                Some(c) => Tensor::cat(&[c, &lat, &flag], 1)?,
                None => Tensor::cat(&[&lat, &flag], 1)?,
            };
            let raw = d.net().forward(&x)?;
            let loss = match g.family {
                BaselineFamily::NnForecast => net::pinball_loss(&net::monotone(&raw)?, &p.target, &g.levels)?,
                BaselineFamily::NnChart | BaselineFamily::NnFuture => {
                    net::raster_ce(&net::raster_logp(&raw, g.nn.res)?, &p.target, g.nn.res)?
                }
                BaselineFamily::NnNews | BaselineFamily::NnNarrative => net::cosine_loss(&raw, &p.target)?,
                BaselineFamily::NnRetrieval => {
                    // in-batch analogs, same leakage gap as training
                    let gap_ms = (g.min_gap.max(g.horizon) as i64) * d.meta.step_ms;
                    let mut bias = Vec::with_capacity(r * r);
                    let mut keep = Vec::new();
                    for (qi, (tq, _)) in p.times.iter().enumerate() {
                        let row: Vec<f32> = p
                            .times
                            .iter()
                            .enumerate()
                            .map(|(ci, (tc, ec))| {
                                if qi != ci && *ec <= tq - gap_ms {
                                    (-((tq - tc) as f64 / 86_400_000.0) / g.tau_days) as f32
                                } else {
                                    -1e9
                                }
                            })
                            .collect();
                        if row.iter().any(|b| *b > -1e8) {
                            keep.push(qi as u32);
                            bias.extend(row);
                        }
                    }
                    if keep.is_empty() {
                        continue;
                    }
                    let k = keep.len();
                    let keep = Tensor::from_vec(keep, k, lat.device())?;
                    let c = net::l2_normalize(&raw)?;
                    let q = c.index_select(&keep, 0)?;
                    let yq = p.target.index_select(&keep, 0)?;
                    let zc = p.target.t()?.contiguous()?;
                    let bias = Tensor::from_vec(bias, (k, r), lat.device())?;
                    net::soft_knn_crps(&q, &c, &yq, &zc, &bias, g.nn.temp)?
                }
                _ => continue,
            };
            parts.push((d.decoder_id(), d.meta.output_family.clone(), loss.to_scalar::<f32>()?, e.weight));
            total = (total + loss.affine(e.weight, 0.0)?)?;
        }
        Ok(FeedbackLoss { total, parts })
    }
}
