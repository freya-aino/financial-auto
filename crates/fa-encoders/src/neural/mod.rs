//! Neural encoder families on candle (spec 31 §6.2, ADR 0003, D-010).
//!
//! * [`data`]: market front-end (per-bar features, next-step targets, fitted normalisation,
//!   train/validation windows);
//! * [`model`]: backbones (`tcn`, `gru`) + learned expert space + family heads
//!   (`fsq_hybrid`, `hier_vae_flow`, `jepa_vae`, `latent_diffusion`);
//! * [`prior`]: the flow-matching (family 2) and latent-diffusion (family 4) priors;
//! * [`regions`]: the density (mean-shift) region discovery families 2 and 4 declare;
//! * [`adapter`]: LoRA online adapters (spec 36 §6.5);
//! * [`train`]: the deterministic, resumable training loop with the spec 32 objective, the D-007
//!   anchor and the D-009 [`train::DecoderFeedback`] hook;
//! * [`NeuralEncoder`]: a trained snapshot behind [`EncoderFamily`] for the live runtime.
//!
//! Weights are safetensors (`weights.bin` of the D-058 snapshot folder): every parameter as
//! `p.<name>`, metadata `format = fa-neural-v1`, `family`, `config` (genome JSON), `norm` (input
//! normalisation) and `active` (expert ids after pruning, in output order).
//!
//! The device comes from config (`FA_ML_DEVICE` / `--device`: `cpu` default, `cuda[:n]` with the
//! `cuda` cargo feature); without the feature every request resolves to the CPU.

pub mod adapter;
pub mod data;
pub mod model;
pub mod prior;
pub mod regions;
pub mod train;

use candle_core::Device;
use fa_core::Timescale;
use fa_expert_space::{
    ExpertLatent, ExpertView,
    moe::top_k_gates,
    nn::{device_from_str, from_safetensors},
};

pub use self::{
    data::{Dataset, InputNorm, N_FEATURES, N_TARGETS, Series},
    model::{Backbone, NeuralConfig, NeuralFamily, NeuralModel},
    train::{Anchors, DecoderFeedback, EvalReport, FeedbackBatch, LossLog, NeuralTrainer, model_weights},
};
use crate::{
    EncoderError, EncoderFamily, EncoderInput, FamilyDescriptor, FamilyId, LatentKind, MarketBar, PriorKind,
    ScaleSizing, default_sizing, snapshot::SnapshotManifest,
};

/// The device of `FA_ML_DEVICE` (default `cpu`) and a note when the request was not honoured.
pub fn device_from_env() -> (Device, Option<String>) {
    device_from_str(&std::env::var("FA_ML_DEVICE").unwrap_or_default())
}

/// Encoder families this module can load from a manifest.
pub fn is_neural_family(family: &str) -> bool {
    NeuralFamily::ALL.iter().any(|f| f.as_str() == family)
}

fn err(e: candle_core::Error) -> EncoderError {
    EncoderError::Manifest(e.to_string())
}

/// A trained neural snapshot serving one timescale.
pub struct NeuralEncoder {
    model: NeuralModel,
    descriptor: FamilyDescriptor,
    sizing: ScaleSizing,
    snapshot_id: String,
}

impl std::fmt::Debug for NeuralEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuralEncoder")
            .field("snapshot_id", &self.snapshot_id)
            .field("family", &self.model.cfg.family)
            .field("experts", &self.model.moe.active)
            .finish()
    }
}

/// Region discovery / prior / latent kind each family declares (spec 31 §3).
fn descriptor(family: NeuralFamily, ts: Timescale) -> FamilyDescriptor {
    let (id, latent, prior, region_discovery) = match family {
        NeuralFamily::FsqHybrid => (FamilyId::FsqHybrid, LatentKind::Hybrid, PriorKind::Codebook, "codebook"),
        NeuralFamily::HierVaeFlow => {
            (FamilyId::HierVaeFlow, LatentKind::Continuous, PriorKind::FlowMatching, "density")
        }
        NeuralFamily::JepaVae => (FamilyId::JepaVae, LatentKind::Continuous, PriorKind::Gaussian, "mapper"),
        NeuralFamily::LatentDiffusion => {
            (FamilyId::LatentDiffusion, LatentKind::Continuous, PriorKind::Diffusion, "density")
        }
    };
    FamilyDescriptor { id, latent, prior, region_discovery, timescales: vec![ts] }
}

impl NeuralEncoder {
    pub fn new(model: NeuralModel, ts: Timescale, snapshot_id: impl Into<String>) -> Self {
        let sizing = ScaleSizing {
            context_steps: model.cfg.context_steps as u32,
            d_model: model.cfg.d_model as u32,
            layers: model.cfg.layers as u32,
            max_experts: model.moe.n_active() as u32,
            latent_dim: model.cfg.moe.expert_dim as u32,
            ..default_sizing(ts)
        };
        Self { descriptor: descriptor(model.cfg.family, ts), sizing, snapshot_id: snapshot_id.into(), model }
    }

    /// Rebuilds a snapshot from its manifest and safetensors `weights.bin` on `dev`.
    pub fn from_manifest_on(manifest: &SnapshotManifest, weights: &[u8], dev: &Device) -> Result<Self, EncoderError> {
        if !is_neural_family(&manifest.encoder_family) {
            return Err(EncoderError::Manifest(format!("family `{}` is not a neural family", manifest.encoder_family)));
        }
        let (vals, meta) = from_safetensors(weights, dev).map_err(err)?;
        let model = train::model_from_parts(&vals, &meta, dev).map_err(err)?;
        if model.cfg.family.as_str() != manifest.encoder_family {
            return Err(EncoderError::Manifest(format!(
                "weights are {} but the manifest says {}",
                model.cfg.family.as_str(),
                manifest.encoder_family
            )));
        }
        if !model.cfg.family.allows(manifest.timescale) {
            return Err(EncoderError::Manifest(format!(
                "{} is only enabled at day/week timescales (D-010), not {}",
                manifest.encoder_family, manifest.timescale
            )));
        }
        Ok(Self::new(model, manifest.timescale, manifest.snapshot_id.clone()))
    }

    /// [`Self::from_manifest_on`] on the configured device ([`device_from_env`]).
    pub fn from_manifest(manifest: &SnapshotManifest, weights: &[u8]) -> Result<Self, EncoderError> {
        Self::from_manifest_on(manifest, weights, &device_from_env().0)
    }

    pub fn model(&self) -> &NeuralModel {
        &self.model
    }

    pub fn family(&self) -> NeuralFamily {
        self.model.cfg.family
    }

    /// The same snapshot with a LoRA adapter (`fa-lora-v1` bytes) folded into its weights
    /// (spec 36 §6.5); the snapshot id is kept (adapters version a snapshot, they are not one).
    pub fn with_lora(&self, bytes: &[u8]) -> Result<Self, EncoderError> {
        let (lora, _) = fa_expert_space::lora::LoraSet::from_bytes(bytes, self.model.device()).map_err(err)?;
        for t in &lora.targets {
            if self.model.ps.tensor(t).is_none() {
                return Err(EncoderError::Manifest(format!("adapter targets `{t}`, which this snapshot lacks")));
            }
        }
        let model = adapter::merged(&self.model, &lora).map_err(err)?;
        Ok(Self::new(model, self.sizing.timescale, self.snapshot_id.clone()))
    }

    /// Novelty under the family's learned prior for windows `bars[..end]` (higher = more
    /// novel): `−log p(z_top)` under the flow prior (`hier_vae_flow`), the unconditional
    /// denoising error of the full latent (`latent_diffusion`); `None` for families without one.
    pub fn prior_score(&self, bars: &[MarketBar], ends: &[usize]) -> Result<Option<Vec<f64>>, EncoderError> {
        let m = &self.model;
        if !matches!(m.cfg.family, NeuralFamily::HierVaeFlow | NeuralFamily::LatentDiffusion) {
            return Ok(None);
        }
        let feats = data::bar_features(bars);
        let mut out = Vec::with_capacity(ends.len());
        for chunk in ends.chunks(128) {
            let wins: Vec<(&[[f32; N_FEATURES]], usize)> = chunk.iter().map(|&e| (feats.as_slice(), e)).collect();
            let x = m.input(&wins).map_err(err)?;
            let f = m.forward(&x, None).map_err(err)?;
            match m.cfg.family {
                NeuralFamily::HierVaeFlow => {
                    let z = f.top.as_ref().expect("top latent").mu.to_vec2::<f32>().map_err(err)?;
                    let lp = prior::flow_log_density(&m.ps, &z, 16, m.device()).map_err(err)?;
                    out.extend(lp.into_iter().map(|v| -v));
                }
                _ => {
                    let full = m.full_latent(&f.out.mu).map_err(err)?;
                    out.extend(prior::diffusion_nll(&m.ps, &full, m.cfg.diffusion_steps, 11).map_err(err)?);
                }
            }
        }
        Ok(Some(out))
    }

    /// `latent_diffusion`: `n` sampled future expert latents (`[n][expert][dim]`, `horizon()`
    /// steps after the window `bars[..end]`), DDIM from seeded noise; `None` for other families.
    pub fn sample_futures(
        &self,
        bars: &[MarketBar],
        end: usize,
        n: usize,
        seed: u64,
    ) -> Result<Option<Vec<Vec<Vec<f32>>>>, EncoderError> {
        let m = &self.model;
        if m.cfg.family != NeuralFamily::LatentDiffusion {
            return Ok(None);
        }
        let feats = data::bar_features(bars);
        let x = m.input(&[(feats.as_slice(), end)]).map_err(err)?;
        let f = m.forward(&x, None).map_err(err)?;
        let c = m.full_latent(&f.out.mu).map_err(err)?.repeat((n.max(1), 1)).map_err(err)?;
        let x0 = prior::ddim_sample(&m.ps, &c, m.cfg.diffusion_steps, 20, seed).map_err(err)?;
        Ok(Some(m.split_full(&x0).map_err(err)?.to_vec3::<f32>().map_err(err)?))
    }

    /// Encodes many windows of one series at once (probe sets, alignment): `ends` are exclusive
    /// bar indices. Returns `(mu, logvar, gates, fsq code)` per window and expert.
    pub fn encode_windows(&self, bars: &[MarketBar], ends: &[usize]) -> Result<Vec<Encoded>, EncoderError> {
        let feats = data::bar_features(bars);
        let mut out = Vec::with_capacity(ends.len());
        for chunk in ends.chunks(256) {
            let wins: Vec<(&[[f32; N_FEATURES]], usize)> = chunk.iter().map(|&e| (feats.as_slice(), e)).collect();
            out.extend(self.encode_inputs(&wins)?);
        }
        Ok(out)
    }

    fn encode_inputs(&self, wins: &[(&[[f32; N_FEATURES]], usize)]) -> Result<Vec<Encoded>, EncoderError> {
        let x = self.model.input(wins).map_err(err)?;
        let f = self.model.forward(&x, None).map_err(err)?;
        let mu = f.out.mu.to_vec3::<f32>().map_err(err)?;
        let lv = f.out.logvar.to_vec3::<f32>().map_err(err)?;
        let probs = f.out.probs.to_vec2::<f32>().map_err(err)?;
        let pred = f.out.pred.to_vec3::<f32>().map_err(err)?;
        let codes = match &f.out.codes {
            Some(c) => Some(c.to_vec2::<f32>().map_err(err)?),
            None => None,
        };
        Ok((0..wins.len())
            .map(|i| Encoded {
                mu: mu[i].clone(),
                pred: pred[i].clone(),
                logvar: lv[i].clone(),
                gates: top_k_gates(&probs[i], self.model.cfg.moe.top_k),
                probs: probs[i].clone(),
                codes: codes.as_ref().map(|c| c[i].iter().map(|v| *v as u32).collect()),
            })
            .collect())
    }
}

/// One encoded window: per expert `mu`/`logvar`, router gates and FSQ codes.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoded {
    pub mu: Vec<Vec<f32>>,
    /// Per expert next-step prediction (normalised targets).
    pub pred: Vec<Vec<f32>>,
    pub logvar: Vec<Vec<f32>>,
    /// Top-k gates renormalised (the runtime's `router_weights`).
    pub gates: Vec<f32>,
    pub probs: Vec<f32>,
    pub codes: Option<Vec<u32>>,
}

impl EncoderFamily for NeuralEncoder {
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
        Ok(self.encode_with_extras(input)?.0)
    }

    fn encode_with_extras(
        &self,
        input: &EncoderInput,
    ) -> Result<(ExpertView, Option<crate::ExpertExtras>), EncoderError> {
        if input.market.is_empty() {
            return Err(EncoderError::Shape("empty bar window".into()));
        }
        let bars: Vec<MarketBar> = input.market.iter().map(|r| MarketBar::from_row(r)).collect::<Result<_, _>>()?;
        let feats = data::bar_features(&bars);
        let e = self.encode_inputs(&[(feats.as_slice(), feats.len())])?.pop().expect("one window");
        // Fewer bars than the context: widen the posterior by the missing share (log scale).
        let missing =
            self.model.cfg.context_steps as f32 / (bars.len().min(self.model.cfg.context_steps).max(1) as f32);
        let widen = missing.ln();
        let n = feats.len();
        let realised_prev = (n >= 2)
            .then(|| data::targets_at(&feats, n - 1))
            .flatten()
            .map(|t| t.iter().enumerate().map(|(j, v)| self.model.norm.target(j, *v)).collect::<Vec<f32>>());
        let prior_score = self.prior_score(&bars, &[n])?.and_then(|v| v.first().copied());
        let extras = crate::ExpertExtras {
            targets: data::TARGET_NAMES.iter().map(|s| (*s).to_owned()).collect(),
            next_step: e.pred.clone(),
            codes: e.codes.clone(),
            prior_score,
            realised_prev,
            probs: e.probs.clone(),
        };
        let experts =
            e.mu.into_iter()
                .zip(e.logvar)
                .enumerate()
                .map(|(i, (mean, lv))| ExpertLatent {
                    expert_index: i as u32,
                    mean,
                    log_var: lv.into_iter().map(|v| v + widen).collect(),
                    region_id: None,
                    region_lineage_id: None,
                })
                .collect::<Vec<_>>();
        let view = ExpertView {
            snapshot_id: self.snapshot_id.clone(),
            timescale: self.sizing.timescale,
            as_of_ms: input.as_of_ms,
            expert_mask: vec![true; experts.len()],
            experts,
            router_weights: e.gates,
        };
        Ok((view, Some(extras)))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::snapshot::{DataRef, ProbeSetRef};

    /// Synthetic regime-switching bars (not market data): calm/volatile blocks with a volume
    /// signature, so the encoder has structure to find.
    pub(crate) fn synthetic(n: usize, seed: u64) -> Vec<MarketBar> {
        let mut rng = fa_core::rng::Rng::seed(seed);
        let mut p = 100.0;
        let mut regime = 0usize;
        (0..n)
            .map(|i| {
                if i % 40 == 0 {
                    regime = rng.below(2);
                }
                let vol = if regime == 0 { 0.001 } else { 0.006 };
                let drift = if regime == 0 { 0.0003 } else { -0.0005 };
                let open = p;
                p *= 1.0 + drift + vol * rng.normal();
                let hi = open.max(p) * (1.0 + vol * rng.uniform());
                let lo = open.min(p) * (1.0 - vol * rng.uniform());
                let volume = if regime == 0 { 100.0 } else { 400.0 } * (0.5 + rng.uniform());
                MarketBar { open, high: hi, low: lo, close: p, volume, trade_count: (volume / 10.0) as u64 }
            })
            .collect()
    }

    /// Parameter values of weights bytes (the safetensors header orders metadata arbitrarily).
    fn params(bytes: &[u8]) -> BTreeMap<String, Vec<f32>> {
        let (vals, _) = from_safetensors(bytes, &Device::Cpu).unwrap();
        vals.into_iter().map(|(k, v)| (k, v.flatten_all().unwrap().to_vec1::<f32>().unwrap())).collect()
    }

    fn small(family: NeuralFamily) -> NeuralConfig {
        NeuralConfig {
            context_steps: 24,
            d_model: 16,
            layers: 3,
            steps: 120,
            batch: 32,
            moe: fa_expert_space::moe::MoeConfig {
                max_experts: 3,
                expert_dim: 4,
                fsq_levels: if family == NeuralFamily::FsqHybrid { vec![5, 5] } else { vec![] },
                ..Default::default()
            },
            ..NeuralConfig::new(family)
        }
    }

    fn ts_of(family: NeuralFamily) -> Timescale {
        if family.allows(Timescale::Minute) { Timescale::Minute } else { Timescale::Day }
    }

    fn manifest(id: &str, family: NeuralFamily) -> SnapshotManifest {
        SnapshotManifest {
            snapshot_id: id.into(),
            encoder_family: family.as_str().into(),
            timescale: ts_of(family),
            generation: 0,
            parent_snapshot_id: None,
            created_at: String::new(),
            params: serde_json::Value::Null,
            probe_set: ProbeSetRef::default(),
            data: DataRef::default(),
            metrics: BTreeMap::new(),
            feedback: None,
            seed: 0,
        }
    }

    /// Conformance (spec 31 §9.1): shapes, determinism with seed, training lowers the loss,
    /// save/load round trip reproduces the encoder, resume equals an uninterrupted run.
    #[test]
    fn families_train_save_load_and_resume_deterministically() {
        let dev = Device::Cpu;
        for family in NeuralFamily::ALL {
            let cfg = small(family);
            let ds = Dataset::build(
                vec![Series::new("a", synthetic(600, 1)), Series::new("b", synthetic(400, 2))],
                cfg.min_window,
                cfg.horizon(),
                0.15,
            );
            let mut tr = NeuralTrainer::new(cfg.clone(), &ds, 7, &dev).unwrap();
            let before = tr.evaluate(&ds, &ds.val).unwrap();
            tr.train_steps(&ds, 120).unwrap();
            let after = tr.evaluate(&ds, &ds.val).unwrap();
            assert!(after.loss.total < before.loss.total, "{family:?}: {} → {}", before.loss.total, after.loss.total);
            assert_eq!(after.health.resid_corr.len(), 3);
            assert!(after.health.k_eff >= 1.0);

            // same seed, same data → same weights (bitwise on CPU)
            let mut again = NeuralTrainer::new(cfg.clone(), &ds, 7, &dev).unwrap();
            again.train_steps(&ds, 120).unwrap();
            assert_eq!(params(&again.weights().unwrap()), params(&tr.weights().unwrap()), "{family:?} deterministic");

            // checkpoint after 50 steps, resume, finish → same as uninterrupted
            let mut a = NeuralTrainer::new(cfg.clone(), &ds, 7, &dev).unwrap();
            a.train_steps(&ds, 50).unwrap();
            let ck = a.checkpoint().unwrap();
            let mut b = NeuralTrainer::from_checkpoint(&ck, &dev).unwrap();
            assert_eq!(b.step, 50);
            b.train_steps(&ds, 70).unwrap();
            assert_eq!(params(&b.weights().unwrap()), params(&tr.weights().unwrap()), "{family:?} resume");

            // the artifact reproduces the encoder
            let m = manifest("snap", family);
            let enc = NeuralEncoder::from_manifest_on(&m, &tr.weights().unwrap(), &dev).unwrap();
            let bars = synthetic(80, 3);
            let v = enc.encode(&EncoderInput::from_bars(5, &bars)).unwrap();
            assert_eq!(v.experts.len(), 3);
            assert_eq!(v.experts[0].mean.len(), 4);
            assert!(v.experts.iter().flat_map(|e| &e.mean).all(|x| x.is_finite() && x.abs() <= 1.0));
            assert!((v.router_weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert_eq!(v.router_weights.iter().filter(|w| **w > 0.0).count(), 2, "top-2 gates");
            let direct = NeuralEncoder::new(tr.model.clone(), ts_of(family), "snap");
            assert_eq!(direct.encode(&EncoderInput::from_bars(5, &bars)).unwrap().experts[1].mean, v.experts[1].mean);
            let batch = enc.encode_windows(&bars, &[80]).unwrap();
            // `EncoderInput` rows are f32 prices, the batch path reads the f64 bars: equal up to rounding
            for (x, y) in batch[0].mu[1].iter().zip(&v.experts[1].mean) {
                assert!((x - y).abs() < 1e-3, "{family:?} batch vs single: {x} vs {y}");
            }
            if family == NeuralFamily::FsqHybrid {
                assert!(batch[0].codes.is_some());
            }
            // short history → wider posterior
            let short = enc.encode(&EncoderInput::from_bars(5, &bars[..6])).unwrap();
            assert!(short.experts[0].log_var[0] > v.experts[0].log_var[0]);
            // wrong family in the manifest is refused
            let other = manifest(
                "snap",
                if family == NeuralFamily::FsqHybrid { NeuralFamily::JepaVae } else { NeuralFamily::FsqHybrid },
            );
            assert!(NeuralEncoder::from_manifest_on(&other, &tr.weights().unwrap(), &dev).is_err());
        }
    }

    #[test]
    fn pruning_and_warm_start() {
        let dev = Device::Cpu;
        let cfg = small(NeuralFamily::JepaVae);
        let ds = Dataset::build(vec![Series::new("a", synthetic(300, 4))], cfg.min_window, cfg.horizon(), 0.2);
        let mut tr = NeuralTrainer::new(cfg.clone(), &ds, 1, &dev).unwrap();
        tr.train_steps(&ds, 20).unwrap();
        let kept = tr.prune(&[0.5, 0.01, 0.49], 2);
        assert_eq!(kept, vec![0, 2]);
        let w = tr.weights().unwrap();
        let enc = NeuralEncoder::from_manifest_on(&manifest("p", NeuralFamily::JepaVae), &w, &dev).unwrap();
        assert_eq!(enc.sizing().max_experts, 2);
        let mut child = NeuralTrainer::new(cfg, &ds, 2, &dev).unwrap();
        assert_eq!(child.warm_start(&w).unwrap(), child.model.ps.len(), "every tensor warm-started");
        assert_eq!(
            child.model.ps.tensor("bb.in.w").unwrap().to_vec2::<f32>().unwrap(),
            tr.model.ps.tensor("bb.in.w").unwrap().to_vec2::<f32>().unwrap()
        );
    }

    /// Families 2 and 4 beyond the shared conformance: the learned priors score and sample, the
    /// D-010 timescale gate holds, and the declared region discovery conforms (spec 30 §9.1).
    #[test]
    fn hierarchical_flow_and_latent_diffusion_priors() {
        let dev = Device::Cpu;
        let novel = |bars: &mut Vec<MarketBar>| {
            // an extreme, never-seen regime at the end of the series
            for b in bars.iter_mut().rev().take(30) {
                b.high *= 1.2;
                b.low *= 0.8;
                b.volume *= 50.0;
            }
        };
        for family in [NeuralFamily::HierVaeFlow, NeuralFamily::LatentDiffusion] {
            let cfg = NeuralConfig { steps: 200, ..small(family) };
            let ds = Dataset::build(vec![Series::new("a", synthetic(700, 5))], cfg.min_window, cfg.horizon(), 0.15);
            let mut tr = NeuralTrainer::new(cfg.clone(), &ds, 3, &dev).unwrap();
            let before = tr.evaluate(&ds, &ds.val).unwrap();
            tr.train_steps(&ds, 200).unwrap();
            let after = tr.evaluate(&ds, &ds.val).unwrap();
            assert!(
                after.loss.prior < before.loss.prior,
                "{family:?} prior: {} → {}",
                before.loss.prior,
                after.loss.prior
            );
            assert!(after.loss.prior.is_finite() && after.loss.prior > 0.0);
            let enc = NeuralEncoder::from_manifest_on(&manifest("p", family), &tr.weights().unwrap(), &dev).unwrap();
            assert_eq!(enc.descriptor().region_discovery, "density");
            let bars = synthetic(200, 6);
            let ends: Vec<usize> = (40..=200).step_by(8).collect();
            let typical = enc.prior_score(&bars, &ends).unwrap().unwrap();
            assert!(typical.iter().all(|v| v.is_finite()));
            let typical = typical.iter().sum::<f64>() / typical.len() as f64;
            match family {
                NeuralFamily::HierVaeFlow => {
                    // the flow models the aggregate posterior of the top latent: the encoded
                    // windows are far more likely than the same latents moved 4σ away
                    let feats = data::bar_features(&bars);
                    let wins: Vec<(&[[f32; N_FEATURES]], usize)> =
                        ends.iter().map(|&e| (feats.as_slice(), e)).collect();
                    let f = tr.model.forward(&tr.model.input(&wins).unwrap(), None).unwrap();
                    let z = f.top.unwrap().mu.to_vec2::<f32>().unwrap();
                    let d = z[0].len();
                    let mean: Vec<f32> = (0..d).map(|j| z.iter().map(|r| r[j]).sum::<f32>() / z.len() as f32).collect();
                    let sd: Vec<f32> = (0..d)
                        .map(|j| {
                            (z.iter().map(|r| (r[j] - mean[j]).powi(2)).sum::<f32>() / z.len() as f32).sqrt().max(0.05)
                        })
                        .collect();
                    let far: Vec<Vec<f32>> =
                        z.iter().map(|r| r.iter().zip(&sd).map(|(x, s)| x + 4.0 * s).collect()).collect();
                    let lp_far = prior::flow_log_density(&tr.model.ps, &far, 16, &dev).unwrap();
                    let novel = -lp_far.iter().sum::<f64>() / lp_far.len() as f64;
                    assert!(novel > typical + 1.0, "{family:?}: shifted latents score {novel} vs typical {typical}");
                }
                _ => {
                    let mut odd = bars.clone();
                    novel(&mut odd);
                    let outlier = enc.prior_score(&odd, &[200]).unwrap().unwrap()[0];
                    assert!(outlier > typical, "{family:?}: novel window scores higher ({outlier} vs {typical})");
                }
            }
            match family {
                NeuralFamily::LatentDiffusion => {
                    let fut = enc.sample_futures(&bars, 120, 4, 1).unwrap().unwrap();
                    assert_eq!((fut.len(), fut[0].len(), fut[0][0].len()), (4, 3, 4));
                    assert!(fut.iter().flatten().flatten().all(|v| v.is_finite()));
                    assert_eq!(enc.sample_futures(&bars, 120, 4, 1).unwrap().unwrap(), fut, "seeded");
                    // D-010: not at minutes
                    let mut m = manifest("p", family);
                    m.timescale = Timescale::Minute;
                    assert!(NeuralEncoder::from_manifest_on(&m, &tr.weights().unwrap(), &dev).is_err());
                }
                _ => assert!(enc.sample_futures(&bars, 120, 2, 1).unwrap().is_none()),
            }
        }
        let jepa = NeuralTrainer::new(
            small(NeuralFamily::JepaVae),
            &Dataset::build(vec![Series::new("a", synthetic(200, 1))], 16, 4, 0.2),
            1,
            &dev,
        )
        .unwrap();
        let enc = NeuralEncoder::new(jepa.model.clone(), Timescale::Minute, "j");
        assert!(enc.prior_score(&synthetic(80, 2), &[80]).unwrap().is_none());
    }

    /// Spec 36 §6.5: a LoRA adapter trained on a shifted regime lowers the SSL loss there,
    /// stays a low-rank change of the frozen snapshot, is reproducible, and the runtime path
    /// (`with_lora`) serves exactly the adapted view.
    #[test]
    fn lora_adapter_adapts_a_frozen_snapshot() {
        use fa_expert_space::lora::LoraView;
        let dev = Device::Cpu;
        let cfg = small(NeuralFamily::FsqHybrid);
        let ds = Dataset::build(vec![Series::new("a", synthetic(500, 8))], cfg.min_window, cfg.horizon(), 0.15);
        let mut tr = NeuralTrainer::new(cfg.clone(), &ds, 2, &dev).unwrap();
        tr.train_steps(&ds, 120).unwrap();
        let w0 = tr.model.ps.tensor("bb.out.w").unwrap().to_vec2::<f32>().unwrap();
        // the live stream: same generator, twice the volatility and volume
        let shifted: Vec<MarketBar> = synthetic(300, 9)
            .into_iter()
            .map(|b| MarketBar { high: b.high * 1.004, low: b.low * 0.996, volume: b.volume * 3.0, ..b })
            .collect();
        let live = Dataset::build(vec![Series::new("a", shifted)], cfg.min_window, cfg.horizon(), 0.0);
        let lc = adapter::LoraConfig { steps: 80, rank: 4, ..Default::default() };
        let (lora, rep) = adapter::train_lora(&tr.model, &live, &lc, 5, &dev).unwrap();
        assert!(rep.loss_after < rep.loss_before, "{rep:?}");
        assert!(rep.delta_norm > 0.0 && rep.delta_norm < 1.0, "{rep:?}");
        assert!(rep.latent_shift > 0.0, "{rep:?}");
        assert_eq!(rep.targets, vec!["bb.out.w", "moe.proj0.w", "moe.proj1.w", "moe.proj2.w"]);
        assert_eq!(tr.model.ps.tensor("bb.out.w").unwrap().to_vec2::<f32>().unwrap(), w0, "snapshot frozen");
        let (again, _) = adapter::train_lora(&tr.model, &live, &lc, 5, &dev).unwrap();
        assert_eq!(
            again.ps.tensor("bb.out.w.lora_a").unwrap().to_vec2::<f32>().unwrap(),
            lora.ps.tensor("bb.out.w.lora_a").unwrap().to_vec2::<f32>().unwrap()
        );
        let base = NeuralEncoder::new(tr.model.clone(), Timescale::Minute, "s");
        let adapted = base.with_lora(&lora.to_bytes(Default::default()).unwrap()).unwrap();
        assert_eq!(fa_encoders_snapshot_id(&adapted), "s");
        let bars = synthetic(80, 3);
        let x = tr.model.input(&[(data::bar_features(&bars).as_slice(), 80)]).unwrap();
        let view = LoraView { base: &tr.model.ps, lora: &lora };
        let want = tr.model.forward_w(&view, &x, None, None).unwrap().out.mu.to_vec3::<f32>().unwrap();
        let got = adapted.encode_windows(&bars, &[80]).unwrap();
        for (a, b) in got[0].mu.iter().flatten().zip(want[0].iter().flatten()) {
            assert!((a - b).abs() < 1e-4);
        }
        assert_ne!(got[0].mu, base.encode_windows(&bars, &[80]).unwrap()[0].mu);
    }

    fn fa_encoders_snapshot_id(e: &NeuralEncoder) -> &str {
        EncoderFamily::snapshot_id(e)
    }
}
