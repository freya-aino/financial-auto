//! The unified decoder interface of spec 34 §4.1, implemented by **every** genome family: the
//! CPU baselines (`forecast`, `retrieval`, wrapped [`crate::baseline::SeriesDecoder`]s) and the eight
//! trained candle families ([`crate::neural::NeuralDecoder`]).
//!
//! | spec 34 §4.1 | here |
//! |---|---|
//! | `family()` | [`Decoder::family`] (output family, `decoder.output.<family>`) |
//! | `forward(&ExpertViewBatch, &DecodeCtx)` | [`Decoder::forward`] over a point-in-time [`DecodeView`]: the series up to step `t`, the expert-space contract ([`ExpertView`], spec 32) restricted to the genome's **expert mask**, and the news timeline |
//! | `loss(out, target)` | [`Decoder::target`] (what step `t` is scored against, and when it becomes known) + [`Decoder::loss`] (CRPS of the quantiles, raster cross-entropy, `1 − cos`) |
//! | `Module` (save/load) | neural: safetensors bundles; baselines: the genome string is the whole model |
//! | cost estimate | [`Decoder::cost`] (arithmetic ops per forward) |
//!
//! Expert masks: bit `i` of `DecoderGenome::expert_mask` selects expert index `i`, `0` reads every
//! expert. A neural decoder takes the masked concatenation from the view
//! ([`crate::neural::NeuralDecoder::select_latent`]); baselines condition on their own
//! point-in-time [`crate::DecodeCtx`] (masked at construction, [`crate::baseline::build_with`]),
//! because the retrieval baseline also needs the latents of past candidate windows.

use std::sync::Arc;

use fa_expert_space::ExpertView;

use crate::{
    Analog, DecodeCtx, Family,
    baseline::{Series, SeriesDecoder, build_with},
    crps_quantiles,
    genome::DecoderGenome,
    neural::{NeuralDecoder, data, data::TextTimeline},
};

/// Point-in-time input of one decoder call.
pub struct DecodeView<'a> {
    /// Price series; the decoder reads `series[..=t]` only.
    pub series: &'a Series,
    pub t: usize,
    /// Expert-space contract as of `ts[t]` (spec 32). `None` = the decoder's own context.
    pub experts: Option<&'a ExpertView>,
    /// News known by `ts[t]` (text families).
    pub text: &'a TextTimeline,
}

impl<'a> DecodeView<'a> {
    pub fn new(series: &'a Series, t: usize, text: &'a TextTimeline) -> Self {
        Self { series, t, experts: None, text }
    }

    pub fn with_experts(mut self, v: &'a ExpertView) -> Self {
        self.experts = Some(v);
        self
    }
}

/// What a decoder produced for one view.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecodeOutput {
    /// Horizon log-return quantiles (forecast families; empty otherwise).
    pub quantiles: Vec<f64>,
    pub quantile_levels: Vec<f64>,
    /// Score in [-1, 1] (the `visual.*` / `text.*` output contract, `quantiles[0]` on the bus).
    pub score: f64,
    /// Retrieved analogs (retrieval families).
    pub analogs: Vec<Analog>,
    /// Reconstruction (raster probabilities / normalised embedding; reconstruction families).
    pub recon: Vec<f32>,
}

/// What a step is scored against.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodeTarget {
    /// Realised horizon log return.
    Return(f64),
    /// Reconstruction target (raster / embedding), same layout as [`DecodeOutput::recon`].
    Recon(Vec<f32>),
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum DecodeError {
    #[error("no output at step {0} (warm-up, missing context or series too short)")]
    NoOutput(usize),
    #[error("the expert view lacks the experts this decoder's mask selects")]
    MissingExperts,
}

/// Spec 34 §4.1 decoder.
pub trait Decoder: Send + Sync {
    /// `DecoderOutput.decoder_id`.
    fn id(&self) -> String;
    /// Output family (`decoder.output.<family>`).
    fn family(&self) -> Family;
    fn genome(&self) -> &DecoderGenome;
    /// Experts read (`0` = all).
    fn expert_mask(&self) -> u32 {
        self.genome().expert_mask
    }
    fn forward(&self, view: &DecodeView<'_>) -> Result<DecodeOutput, DecodeError>;
    /// Target of step `t` and the time it becomes known (`None` when not available).
    fn target(&self, series: &Series, t: usize, text: &TextTimeline) -> Option<(DecodeTarget, i64)>;
    /// Family loss of an output against a target (lower is better; NaN when they do not match).
    fn loss(&self, out: &DecodeOutput, target: &DecodeTarget) -> f64;
    /// Arithmetic operations per forward (the deterministic part of `compute_cost`).
    fn cost(&self, series_len: usize) -> f64;
}

/// CRPS of output quantiles against a realised return.
fn return_loss(out: &DecodeOutput, target: &DecodeTarget) -> f64 {
    match target {
        DecodeTarget::Return(y) if !out.quantiles.is_empty() && out.quantiles.len() == out.quantile_levels.len() => {
            crps_quantiles(&out.quantiles, &out.quantile_levels, *y)
        }
        _ => f64::NAN,
    }
}

/// A baseline [`SeriesDecoder`] behind the unified interface.
pub struct Baseline(pub Box<dyn SeriesDecoder>);

impl Decoder for Baseline {
    fn id(&self) -> String {
        let g = self.0.genome();
        format!("{}-{:08x}", g.family.as_str(), crate::genome::stable_hash(&g.to_string()) as u32)
    }

    fn family(&self) -> Family {
        self.0.genome().family.output_family()
    }

    fn genome(&self) -> &DecoderGenome {
        self.0.genome()
    }

    fn forward(&self, view: &DecodeView<'_>) -> Result<DecodeOutput, DecodeError> {
        let f = self.0.forecast(view.series, view.t).ok_or(DecodeError::NoOutput(view.t))?;
        let g = self.0.genome();
        let scale = data::sigma_at(view.series, view.t, g.lookback).unwrap_or(1.0) * (g.horizon as f64).sqrt();
        Ok(DecodeOutput {
            score: (f.point / scale.max(1e-12)).tanh(),
            quantiles: f.quantiles,
            quantile_levels: f.levels,
            analogs: f.analogs,
            recon: Vec::new(),
        })
    }

    fn target(&self, series: &Series, t: usize, _text: &TextTimeline) -> Option<(DecodeTarget, i64)> {
        let h = self.0.genome().horizon;
        Some((DecodeTarget::Return(series.forward_return(t, h)?), *series.ts_ms.get(t + h)?))
    }

    fn loss(&self, out: &DecodeOutput, target: &DecodeTarget) -> f64 {
        return_loss(out, target)
    }

    fn cost(&self, series_len: usize) -> f64 {
        self.0.ops_per_forecast(series_len)
    }
}

impl Decoder for NeuralDecoder {
    fn id(&self) -> String {
        self.decoder_id()
    }

    fn family(&self) -> Family {
        self.genome().family.output_family()
    }

    fn genome(&self) -> &DecoderGenome {
        NeuralDecoder::genome(self)
    }

    fn forward(&self, view: &DecodeView<'_>) -> Result<DecodeOutput, DecodeError> {
        let latent = match view.experts {
            Some(v) if self.meta.layout.latent > 0 => Some(
                self.select_latent(v.experts.iter().map(|e| (e.expert_index, e.mean.as_slice())))
                    .ok_or(DecodeError::MissingExperts)?,
            ),
            Some(_) => None,
            None => self.own_latent(view.series, view.t),
        };
        let s =
            self.serve_at(view.series, view.t, latent.as_deref(), view.text).ok_or(DecodeError::NoOutput(view.t))?;
        Ok(DecodeOutput {
            quantiles: s.quantiles,
            quantile_levels: if self.genome().family.forecasts() { s.levels } else { Vec::new() },
            score: s.score,
            analogs: s.analogs,
            recon: s.recon,
        })
    }

    fn target(&self, series: &Series, t: usize, text: &TextTimeline) -> Option<(DecodeTarget, i64)> {
        let g = NeuralDecoder::genome(self);
        if g.family.reconstructs() {
            let sigma = data::sigma_at(series, t, g.lookback)?;
            let (y, end) = data::target_at(g, series, t, sigma, text)?;
            return Some((DecodeTarget::Recon(y), end));
        }
        let h = g.horizon;
        Some((DecodeTarget::Return(series.forward_return(t, h)?), *series.ts_ms.get(t + h)?))
    }

    fn loss(&self, out: &DecodeOutput, target: &DecodeTarget) -> f64 {
        match target {
            DecodeTarget::Recon(y) if !out.recon.is_empty() && out.recon.len() == y.len() => {
                self.row_loss(&out.recon, y)
            }
            DecodeTarget::Recon(_) => f64::NAN,
            DecodeTarget::Return(_) => return_loss(out, target),
        }
    }

    fn cost(&self, series_len: usize) -> f64 {
        SeriesDecoder::ops_per_forecast(self, series_len)
    }
}

/// The decoder a **baseline** genome describes, behind the unified interface (neural genomes
/// come from [`crate::neural::train`] or a bundle, [`NeuralDecoder::from_bundle`]; for them this
/// is the untrained stand-in of the same output contract, as [`build_with`]).
pub fn build_decoder(genome: &DecoderGenome, ctx: Arc<DecodeCtx>) -> Box<dyn Decoder> {
    Box::new(Baseline(build_with(genome, ctx)))
}

#[cfg(test)]
mod tests {
    use fa_core::rng::Rng;
    use fa_expert_space::ExpertLatent;

    use super::*;
    use crate::{
        LatentPoint,
        genome::BaselineFamily,
        neural::{TrainOptions, train},
    };

    const INST: &str = "0190aaaa";

    /// SYNTHETIC regime series with a two-expert latent: expert 0 carries the regime, expert 1
    /// noise; regime-worded news every 10 steps, tagged with the instrument every other time;
    /// a regime-dependent synthetic "frame" (byte pattern) every 5 steps on the frame timeline.
    fn data(n: usize) -> (Series, DecodeCtx, TextTimeline, Vec<ExpertView>, TextTimeline) {
        let mut rng = Rng::seed(5);
        let mut lp = 4.6;
        let (mut ts, mut log_price) = (Vec::new(), Vec::new());
        let mut ctx = DecodeCtx::default();
        ctx.set_layout(INST, vec![(0, 2), (1, 3)]);
        let mut text = TextTimeline::new();
        let mut frames = TextTimeline::frames();
        let mut views = Vec::new();
        for i in 0..n {
            let r = if (i / 150) % 2 == 0 { 1.0 } else { -1.0 };
            lp += 0.0005 * r + 0.001 * rng.normal();
            ts.push(i as i64 * 1000);
            log_price.push(lp);
            let e0 = vec![(r + 0.1 * rng.normal()) as f32, 0.3];
            let e1 = vec![(0.5 * rng.normal()) as f32, (0.5 * rng.normal()) as f32, 0.1];
            ctx.push(INST, LatentPoint { as_of_ms: i as i64 * 1000, vec: [e0.clone(), e1.clone()].concat() });
            views.push(ExpertView {
                snapshot_id: "syn".into(),
                timescale: fa_core::Timescale::Second,
                as_of_ms: i as i64 * 1000,
                expert_mask: vec![true, true],
                experts: vec![
                    ExpertLatent {
                        expert_index: 1,
                        mean: e1,
                        log_var: vec![],
                        region_id: None,
                        region_lineage_id: None,
                    },
                    ExpertLatent {
                        expert_index: 0,
                        mean: e0,
                        log_var: vec![],
                        region_id: None,
                        region_lineage_id: None,
                    },
                ],
                router_weights: vec![],
            });
            if i % 10 == 0 {
                let words = if r > 0.0 {
                    "stocks rally strong gains record surge"
                } else {
                    "shares plunge weak losses slump crash"
                };
                let tags = if i % 20 == 0 { vec![INST.to_owned()] } else { Vec::new() };
                text.push_tagged(i as i64 * 1000, words, "", &tags);
            }
            if i % 5 == 0 {
                let pixel: u8 = if r > 0.0 { 200 } else { 40 };
                let bytes: Vec<u8> =
                    (0..512).map(|j| pixel.wrapping_add((j % 7) as u8 * if r > 0.0 { 3 } else { 11 })).collect();
                frames.push_features(i as i64 * 1000, crate::neural::data::frame_features(&bytes), &[]);
            }
        }
        text.finish();
        frames.finish();
        (Series { instrument_id: INST.into(), ts_ms: ts, log_price, step_ms: 1000 }, ctx, text, views, frames)
    }

    /// Every genome family implements the §4.1 interface: forward at a point in time, a target
    /// and a finite family loss, a positive cost, and (neural) the same output after a bundle
    /// round trip; masked decoders read only their experts.
    #[test]
    fn every_family_conforms() {
        let (s, ctx, news, views, frames) = data(1800);
        let ctx = Arc::new(ctx);
        let news = Arc::new(news);
        let frames = Arc::new(frames);
        let genomes = [
            "forecast:h=5;lb=32",
            "retrieval:h=5;lb=16;k=10;lat=0.5",
            "nn_forecast:h=5;lb=16;width=8;steps=60;mask=1",
            "nn_retrieval:h=5;lb=16;emb=6;steps=40;pool=256",
            "nn_chart:lb=16;res=6;steps=40;mask=0x2",
            "nn_future:h=5;lb=16;res=6;steps=40",
            "nn_news:lb=16;tdim=32;tw=1;steps=40;ninst=1",
            "nn_narrative:lb=16;tdim=32;tw=1;steps=40",
            "nn_summary:lb=16;tdim=32;tw=1;steps=60",
            "nn_frame:lb=16;tdim=32;tw=0.2;steps=40",
        ];
        let mut families = std::collections::HashSet::new();
        for raw in genomes {
            let g = DecoderGenome::parse(raw).unwrap();
            let text = if g.family.uses_frames() { frames.clone() } else { news.clone() };
            let d: Box<dyn Decoder> = if g.family.is_neural() {
                Box::new(
                    train(&g, std::slice::from_ref(&s), ctx.clone(), text.clone(), &TrainOptions::default()).unwrap(),
                )
            } else {
                build_decoder(&g, ctx.clone())
            };
            assert_eq!(d.family(), g.family.output_family(), "{raw}");
            assert_eq!(d.expert_mask(), g.expert_mask);
            assert!(d.cost(s.len()) > 0.0, "{raw}: cost");
            families.insert(d.family());
            let mut scored = 0;
            for t in (1500..1780).step_by(7) {
                let view = DecodeView::new(&s, t, &text).with_experts(&views[t]);
                let Ok(out) = d.forward(&view) else { continue };
                assert!((-1.0..=1.0).contains(&out.score), "{raw}: score {}", out.score);
                assert!(out.quantiles.windows(2).all(|w| w[0] <= w[1] + 1e-12), "{raw}: {:?}", out.quantiles);
                let Some((y, known)) = d.target(&s, t, &text) else { continue };
                assert!(known >= s.ts_ms[t], "{raw}: target known before the step");
                let l = d.loss(&out, &y);
                assert!(l.is_finite() && l >= 0.0, "{raw}: loss {l}");
                scored += 1;
            }
            assert!(scored > 5, "{raw}: {scored} scored steps");
            if g.family.is_neural() {
                let n =
                    train(&g, std::slice::from_ref(&s), ctx.clone(), text.clone(), &TrainOptions::default()).unwrap();
                let back = NeuralDecoder::from_bundle(&n.to_bundle().unwrap(), &crate::neural::Device::Cpu).unwrap();
                let view = DecodeView::new(&s, 1700, &text).with_experts(&views[1700]);
                assert_eq!(n.forward(&view), back.forward(&view), "{raw}: bundle round trip");
                if g.expert_mask == 1 {
                    assert_eq!(n.meta.layout.latent, 2, "mask 1 reads expert 0 only");
                    assert_eq!(n.meta.latent_layout, vec![(0, 2), (1, 3)]);
                    assert_eq!(n.meta.full_latent_dim(), 5);
                }
                if g.expert_mask == 2 {
                    assert_eq!(n.meta.layout.latent, 3, "mask 2 reads expert 1 only");
                }
                if g.family == BaselineFamily::NnForecast {
                    // a view without the masked expert cannot be decoded
                    let mut v = views[1700].clone();
                    v.experts.retain(|e| e.expert_index == 1);
                    assert_eq!(
                        n.forward(&DecodeView::new(&s, 1700, &text).with_experts(&v)),
                        Err(DecodeError::MissingExperts)
                    );
                }
            }
        }
        assert_eq!(families.len(), 8, "every output family of spec 34 §3");
        assert!(crate::Family::ALL.iter().all(|f| families.contains(f)));
    }
}
