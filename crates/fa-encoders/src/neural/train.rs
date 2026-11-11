//! Training loop of the neural families (spec 31 §6.5, spec 32 §6.3, spec 36 §6.2–§6.3).
//!
//! `L = λ_next Σ_k L_pred,k + λ_mix L_mix + λ_dec L_decor + λ_bal L_balance + β KL + λ_ssl L_ssl
//!      + λ_anchor L_anchor + λ_fb L_feedback`
//!
//! * `L_ssl`: reconstruction of the last bars (fsq_hybrid) or the JEPA latent prediction against
//!   the EMA target encoder (jepa_vae);
//! * `L_anchor` (D-007): soft-min distance of each expert's latents to the **previous generation's
//!   region centroids** of the matched parent expert ([`Anchors`]);
//! * `L_feedback` (D-009): auxiliary losses of frozen elite decoders on the expert latents,
//!   supplied through the [`DecoderFeedback`] hook.
//!
//! The run is deterministic for a seed (minibatches drawn from `Rng::seed(seed ^ step)`, router
//! noise likewise) and resumable: [`NeuralTrainer::checkpoint`] stores parameters, Adam moments,
//! the EMA target and the step, so a resumed job continues bit for bit.

use std::collections::{BTreeMap, HashMap};

use candle_core::{Device, Result, Tensor};
use fa_core::rng::Rng;
use fa_expert_space::{
    health::HealthReport,
    moe::{decorrelation_loss_gated, gaussian_kl, load_balance_loss, router_noise},
    nn::{Adam, from_safetensors, mse, scalar, to_safetensors},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{
    data::{Dataset, InputNorm, N_FEATURES, N_TARGETS, targets_at},
    model::{Batch, NeuralConfig, NeuralFamily, NeuralModel, Weights, anchor_loss, backbone_forward, recon_target},
};

/// What an elite decoder sees of a batch (spec 34 §6.6).
pub struct FeedbackBatch<'a> {
    /// Expert means `[B, K, dim]` (output order of the snapshot being trained).
    pub mu: &'a Tensor,
    /// Router probabilities `[B, K]`.
    pub probs: &'a Tensor,
    /// Normalised next-step targets `[B, T]`.
    pub targets: &'a Tensor,
    /// Expert id (pre-pruning index) of each output position.
    pub experts: &'a [usize],
    /// `(series index in the dataset, window end)` of each row: the latent of row `i` is as of
    /// bar `end − 1` of series `s` (decoders build their own context from it).
    pub windows: &'a [(usize, usize)],
}

/// **D-009 hook**: decoder → encoder feedback as auxiliary losses of *frozen* elite decoders
/// (top-k per family of the previous generation) evaluated on the expert latents during
/// generation training. Implementations hold the frozen decoder weights; the loss must be
/// differentiable with respect to `batch.mu` only (decoder parameters stay constant). Returning
/// `None` skips the term for this batch. The trainer ships a frozen linear-probe stand-in;
/// neural decoders (`fa-decoders::neural`) plug in here.
pub trait DecoderFeedback: Send + Sync {
    /// Stable name for the manifest (`feedback.decoders[].name`).
    fn name(&self) -> &str;
    fn aux_loss(&self, batch: &FeedbackBatch<'_>) -> Result<Option<Tensor>>;
    /// What the feedback is made of (recorded in `manifest.feedback`).
    fn summary(&self) -> Value {
        json!({ "name": self.name() })
    }
}

/// D-007 anchoring targets: the parent's region centroids per output position (`None` = no
/// anchor for that expert: unmatched, or the parent has another latent geometry).
#[derive(Debug, Clone, Default)]
pub struct Anchors {
    pub centroids: Vec<Option<Tensor>>,
}

impl Anchors {
    pub fn from_vecs(per_expert: &[Option<Vec<Vec<f32>>>], dev: &Device) -> Result<Self> {
        let centroids = per_expert
            .iter()
            .map(|c| match c {
                Some(rows) if !rows.is_empty() => {
                    let d = rows[0].len();
                    Ok(Some(Tensor::from_vec(
                        rows.iter().flatten().copied().collect::<Vec<_>>(),
                        (rows.len(), d),
                        dev,
                    )?))
                }
                _ => Ok(None),
            })
            .collect::<Result<_>>()?;
        Ok(Self { centroids })
    }

    pub fn n_anchored(&self) -> usize {
        self.centroids.iter().filter(|c| c.is_some()).count()
    }
}

/// Loss terms of one step / one evaluation (host values).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LossLog {
    pub step: u64,
    pub total: f64,
    pub pred: f64,
    pub mix: f64,
    pub decor: f64,
    pub balance: f64,
    pub kl: f64,
    pub ssl: f64,
    pub anchor: f64,
    pub feedback: f64,
    #[serde(default)]
    pub skill_hinge: f64,
    /// Flow-matching / diffusion loss of the family prior (families 2 and 4).
    #[serde(default)]
    pub prior: f64,
    /// Effective `λ_dec` of the step (differs from the genome with dual ascent).
    #[serde(default)]
    pub lambda_dec: f64,
    pub grad_norm: f64,
}

/// Validation result: losses + spec 32 §6.4 health.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    pub loss: LossLog,
    pub health: HealthReport,
    /// Mixture next-step MSE / naive MSE.
    pub mix_skill: f64,
    /// Mixture skill per next-step target (`1 − mse_t / naive_t`).
    #[serde(default)]
    pub target_skill: Vec<f64>,
    /// Variance of the mixture's scalarised residual (mean over targets): an estimate of the
    /// irreducible scalarised noise `V` every expert shares (spec 32 §6.3 feasibility bound).
    #[serde(default)]
    pub mix_resid_var: f64,
    pub n: usize,
}

pub struct NeuralTrainer {
    pub model: NeuralModel,
    pub opt: Adam,
    pub step: u64,
    pub seed: u64,
    ema: Option<BTreeMap<String, Tensor>>,
    pub anchors: Anchors,
    pub feedback: Vec<Box<dyn DecoderFeedback>>,
    pub log: Vec<LossLog>,
    /// Current `λ_dec` under dual ascent (`decor_dual_lr > 0`); `None` = the genome's value.
    pub lambda_dec: Option<f64>,
}

const META_FORMAT: &str = "fa-neural-v1";
/// Prior noise of evaluations (deterministic validation losses).
const EVAL_PRIOR_SEED: u64 = 0x00E7_A1E7;

impl NeuralTrainer {
    /// Fresh trainer; the input normalisation is fitted on `ds`.
    pub fn new(cfg: NeuralConfig, ds: &Dataset, seed: u64, dev: &Device) -> Result<Self> {
        let model = NeuralModel::new(cfg, InputNorm::fit(ds), seed, dev)?;
        Ok(Self::from_model(model, seed))
    }

    pub fn from_model(model: NeuralModel, seed: u64) -> Self {
        let ema = (model.cfg.family == NeuralFamily::JepaVae).then(|| {
            model
                .backbone_names()
                .into_iter()
                .filter_map(|n| Some((n.clone(), model.ps.tensor(&n)?.detach().copy().ok()?)))
                .collect()
        });
        let opt = Adam::new(model.cfg.lr, model.cfg.weight_decay);
        Self {
            model,
            opt,
            step: 0,
            seed,
            ema,
            anchors: Anchors::default(),
            feedback: Vec::new(),
            log: Vec::new(),
            lambda_dec: None,
        }
    }

    /// Warm start (spec 36 §6.3) from parent weights with the same parameter shapes; returns how
    /// many tensors were loaded. The EMA target restarts from the loaded backbone.
    pub fn warm_start(&mut self, parent_weights: &[u8]) -> Result<usize> {
        let (vals, _) = from_safetensors(parent_weights, self.model.device())?;
        let stripped: HashMap<String, Tensor> =
            vals.into_iter().filter_map(|(k, v)| k.strip_prefix("p.").map(|s| (s.to_owned(), v))).collect();
        let n = self.model.ps.load(&stripped)?;
        if self.ema.is_some() {
            self.ema = Some(
                self.model
                    .backbone_names()
                    .into_iter()
                    .filter_map(|k| Some((k.clone(), self.model.ps.tensor(&k)?.copy().ok()?)))
                    .collect(),
            );
        }
        Ok(n)
    }

    /// Builds a normalised batch of windows.
    pub fn batch(&self, ds: &Dataset, idx: &[(usize, usize)]) -> Result<Batch> {
        let m = &self.model;
        let wins: Vec<(&[[f32; N_FEATURES]], usize)> =
            idx.iter().map(|&(s, e)| (ds.series[s].feats.as_slice(), e)).collect();
        let x = m.input(&wins)?;
        let mut ys = Vec::with_capacity(idx.len() * N_TARGETS);
        for &(s, e) in idx {
            let t = targets_at(&ds.series[s].feats, e).unwrap_or([0.0; N_TARGETS]);
            ys.extend(t.iter().enumerate().map(|(j, v)| m.norm.target(j, *v)));
        }
        let y = Tensor::from_vec(ys, (idx.len(), N_TARGETS), m.device())?;
        let recon = if m.cfg.reconstructs() { Some(recon_target(&x, m.cfg.recon_steps)?) } else { None };
        let future = if m.cfg.needs_future() {
            let h = m.cfg.horizon();
            let fw: Vec<(&[[f32; N_FEATURES]], usize)> =
                idx.iter().map(|&(s, e)| (ds.series[s].feats.as_slice(), e + h)).collect();
            Some(m.input(&fw)?)
        } else {
            None
        };
        Ok(Batch { x, y, recon, future, windows: idx.to_vec() })
    }

    /// Loss of a batch (`train` adds router noise).
    fn losses(&self, b: &Batch, noise_seed: Option<u64>) -> Result<(Tensor, LossLog, super::model::Forward)> {
        self.losses_w(&self.model.ps, b, noise_seed)
    }

    /// [`Self::losses`] with the parameters read through `w` (a low-rank adapted view).
    pub fn losses_w(
        &self,
        w: &dyn Weights,
        b: &Batch,
        noise_seed: Option<u64>,
    ) -> Result<(Tensor, LossLog, super::model::Forward)> {
        let m = &self.model;
        let c = &m.cfg;
        let k = m.moe.n_active();
        let bsz = b.x.dim(0)?;
        let noise = match noise_seed {
            Some(s) if c.moe.router_noise > 0.0 => {
                Some(router_noise(bsz, c.moe.max_experts, c.moe.router_noise, s, m.device())?)
            }
            _ => None,
        };
        // prior / reparameterisation noise: seeded per step in training, fixed in evaluation
        let prior_seed = noise_seed.map_or(EVAL_PRIOR_SEED, |s| s ^ 0x9E37_79B9);
        let top_eps = match (c.family, noise_seed) {
            (NeuralFamily::HierVaeFlow, Some(_)) => {
                Some(router_noise(bsz, c.top_dim, 1.0, prior_seed ^ 1, m.device())?)
            }
            _ => None,
        };
        let f = m.forward_w(w, &b.x, noise.as_ref(), top_eps.as_ref())?;
        let resid = f.out.pred.broadcast_sub(&b.y.unsqueeze(1)?)?; // [B, K, T]
        // Σ_k L_pred,k: every expert is scored on every window (each must carry information,
        // spec 32 §6.3 gen 0) plus, with weight `gated_pred`, on the windows it is routed to
        // (gate-weighted, the classic MoE specialist term that also teaches the router).
        let per = resid.sqr()?.mean(2)?; // [B, K]
        let pred_all = per.mean_all()?.affine(k as f64, 0.0)?;
        let pred_gated = (&per * &f.out.probs)?.sum(1)?.mean_all()?;
        let pred = (pred_all.affine(1.0 - c.gated_pred, 0.0)? + pred_gated.affine(c.gated_pred * k as f64, 0.0)?)?;
        let mix = mse(&NeuralModel::mixture(&f)?, &b.y)?;
        let decor = if c.decor_scalar {
            decorrelation_loss_gated(&resid.mean_keepdim(2)?, &f.out.probs)?
        } else {
            decorrelation_loss_gated(&resid, &f.out.probs)?
        };
        let lambda_dec = self.lambda_dec.unwrap_or(c.moe.lambda_dec);
        let balance = load_balance_loss(&f.out.probs)?;
        let mut kl = gaussian_kl(&f.out.mu_raw, &f.out.logvar)?;
        if let Some(t) = &f.top {
            kl = (kl + gaussian_kl(&t.mu.unsqueeze(1)?, &t.logvar.unsqueeze(1)?)?)?;
        }
        let head = m.ssl_head_w(w, &f)?;
        let ssl = match c.family {
            NeuralFamily::JepaVae => {
                let fut = b.future.as_ref().expect("future window");
                let target = match &self.ema {
                    Some(e) => backbone_forward(e as &dyn Weights, c, fut)?,
                    None => backbone_forward(w, c, fut)?,
                }
                .detach();
                mse(&head, &target)?
            }
            _ => mse(&head, b.recon.as_ref().expect("recon target"))?,
        };
        // family priors (stop-gradient into the encoder: they model the latents, the encoder is
        // trained by reconstruction / prediction)
        let prior = match c.family {
            NeuralFamily::HierVaeFlow => {
                let z1 = f.top.as_ref().expect("top latent").mu.detach();
                Some(super::prior::flow_matching_loss(w, &z1, prior_seed)?)
            }
            NeuralFamily::LatentDiffusion => {
                let fut = b.future.as_ref().expect("future window");
                let ff = m.forward_w(w, fut, None, None)?;
                let x0 = m.full_latent(&ff.out.mu)?.detach();
                let cond = m.full_latent(&f.out.mu)?.detach();
                Some(super::prior::diffusion_loss(w, &x0, &cond, c.diffusion_steps, c.cond_drop, prior_seed)?)
            }
            _ => None,
        };
        let mut total = (pred.affine(c.lambda_next, 0.0)?
            + mix.affine(c.lambda_mix, 0.0)?
            + decor.affine(lambda_dec, 0.0)?
            + balance.affine(c.moe.lambda_bal, 0.0)?
            + kl.affine(c.moe.beta_kl, 0.0)?
            + ssl.affine(c.lambda_ssl, 0.0)?)?;
        let prior_v = match &prior {
            Some(p) => {
                total = (total + p.affine(c.lambda_prior, 0.0)?)?;
                scalar(p)?
            }
            None => 0.0,
        };
        let mut log = LossLog {
            step: self.step,
            pred: scalar(&pred)?,
            mix: scalar(&mix)?,
            decor: scalar(&decor)?,
            balance: scalar(&balance)?,
            kl: scalar(&kl)?,
            ssl: scalar(&ssl)?,
            lambda_dec,
            prior: prior_v,
            ..Default::default()
        };
        if c.lambda_ncl > 0.0 && k > 1 {
            let sp = f.out.pred.mean(2)?; // [B, K]
            let spread = sp.broadcast_sub(&sp.mean_keepdim(1)?)?.sqr()?.sum(1)?.mean_all()?;
            total = (total - spread.affine(c.lambda_ncl, 0.0)?)?;
        }
        if c.lambda_skill > 0.0 {
            // routed error vs routed naive error per expert, gates detached (the router is not
            // rewarded for hiding an expert)
            let p = f.out.probs.detach();
            let den = (p.sum(0)? + 1e-6)?; // [K]
            let mse_k = (&per * &p)?.sum(0)?.div(&den)?;
            let y2 = b.y.sqr()?.mean_keepdim(1)?; // [B, 1]
            let naive_k = (p.broadcast_mul(&y2)?.sum(0)?.div(&den)? + 1e-6)?;
            let hinge = mse_k.div(&naive_k)?.affine(1.0, c.skill_floor - 1.0)?.relu()?.sum_all()?;
            log.skill_hinge = scalar(&hinge)?;
            total = (total + hinge.affine(c.lambda_skill, 0.0)?)?;
        }
        if c.lambda_anchor > 0.0 && self.anchors.n_anchored() > 0 {
            let mut acc: Option<Tensor> = None;
            for (pos, cent) in self.anchors.centroids.iter().enumerate().take(k) {
                let Some(cent) = cent else { continue };
                let mu = f.out.mu.narrow(1, pos, 1)?.squeeze(1)?;
                let l = anchor_loss(&mu, cent, c.anchor_tau)?;
                acc = Some(match acc {
                    Some(a) => (a + l)?,
                    None => l,
                });
            }
            if let Some(a) = acc {
                let a = a.affine(1.0 / self.anchors.n_anchored() as f64, 0.0)?;
                log.anchor = scalar(&a)?;
                total = (total + a.affine(c.lambda_anchor, 0.0)?)?;
            }
        }
        if c.lambda_feedback > 0.0 && !self.feedback.is_empty() {
            let fb = FeedbackBatch {
                mu: &f.out.mu,
                probs: &f.out.probs,
                targets: &b.y,
                experts: &m.moe.active,
                windows: &b.windows,
            };
            let mut acc: Option<Tensor> = None;
            for d in &self.feedback {
                if let Some(l) = d.aux_loss(&fb)? {
                    acc = Some(match acc {
                        Some(a) => (a + l)?,
                        None => l,
                    });
                }
            }
            if let Some(a) = acc {
                log.feedback = scalar(&a)?;
                total = (total + a.affine(c.lambda_feedback, 0.0)?)?;
            }
        }
        log.total = scalar(&total)?;
        Ok((total, log, f))
    }

    /// `n` optimisation steps; returns the last step's log.
    pub fn train_steps(&mut self, ds: &Dataset, n: usize) -> Result<LossLog> {
        if ds.train.is_empty() {
            candle_core::bail!("no training windows");
        }
        let mut last = LossLog::default();
        for _ in 0..n {
            let mut rng = Rng::seed(self.seed ^ (self.step + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let bsz = self.model.cfg.batch.min(ds.train.len()).max(1);
            let idx: Vec<(usize, usize)> = (0..bsz).map(|_| ds.train[rng.below(ds.train.len())]).collect();
            let b = self.batch(ds, &idx)?;
            let (loss, mut log, _) = self.losses(&b, Some(rng.next_u64()))?;
            log.grad_norm = self.opt.step(&self.model.ps, &loss)?;
            self.step += 1;
            let c = &self.model.cfg;
            if c.decor_dual_lr > 0.0 {
                let lam = self.lambda_dec.unwrap_or(c.moe.lambda_dec);
                self.lambda_dec = Some((lam + c.decor_dual_lr * (log.decor - c.decor_eps)).max(0.0));
            }
            if let Some(ema) = &mut self.ema {
                let tau = self.model.cfg.ema;
                for (name, t) in ema.iter_mut() {
                    if let Some(p) = self.model.ps.tensor(name) {
                        *t = (t.affine(tau, 0.0)? + p.detach().affine(1.0 - tau, 0.0)?)?;
                    }
                }
            }
            if self.step.is_multiple_of(25) || self.log.is_empty() {
                self.log.push(log.clone());
            }
            last = log;
        }
        Ok(last)
    }

    /// Full-batch refit of the per-expert next-step heads (`moe.pred<e>`) on frozen latents of
    /// up to `max_windows` training windows: `Σ_k mse_k + λ_dec · L_decor(scalarised, gated) +
    /// λ_skill · hinge`, Adam at the genome's learning rate. Returns the last objective value.
    pub fn refit_heads(&mut self, ds: &Dataset, steps: usize, max_windows: usize) -> Result<f64> {
        if steps == 0 || ds.train.is_empty() {
            return Ok(0.0);
        }
        let stride = (ds.train.len() / max_windows.max(1)).max(1);
        let idx: Vec<(usize, usize)> = ds.train.iter().step_by(stride).copied().collect();
        let mut mus = Vec::new();
        let mut probs = Vec::new();
        let mut ys = Vec::new();
        for chunk in idx.chunks(256) {
            let b = self.batch(ds, chunk)?;
            let f = self.model.forward(&b.x, None)?;
            mus.push(f.out.mu.detach());
            probs.push(f.out.probs.detach());
            ys.push(b.y);
        }
        let mu = Tensor::cat(&mus, 0)?; // [N, K, d]
        let p = Tensor::cat(&probs, 0)?;
        let y = Tensor::cat(&ys, 0)?;
        let c = self.model.cfg.clone();
        let names: Vec<String> = self.model.moe.active.iter().map(|e| format!("moe.pred{e}")).collect();
        let mut heads = fa_expert_space::nn::ParamStore::new(self.seed, self.model.device());
        let mut init = HashMap::new();
        for n in &names {
            for suf in ["w", "b"] {
                let k = format!("{n}.{suf}");
                let t = self.model.ps.w(&k)?;
                heads.get(&k, t.dims(), fa_expert_space::nn::Init::Zeros)?;
                init.insert(k, t);
            }
        }
        heads.load(&init)?;
        let mut opt = Adam::new(c.lr, 0.0);
        let k = names.len();
        let y2 = (y.sqr()?.mean_keepdim(1)? + 1e-6)?; // [N, 1]
        let mut last = 0.0;
        for _ in 0..steps {
            let preds: Vec<Tensor> = names
                .iter()
                .enumerate()
                .map(|(pos, n)| fa_expert_space::nn::linear_w(&heads, n, &mu.narrow(1, pos, 1)?.squeeze(1)?))
                .collect::<Result<_>>()?;
            let pred = Tensor::stack(&preds, 1)?; // [N, K, T]
            let resid = pred.broadcast_sub(&y.unsqueeze(1)?)?;
            let per = resid.sqr()?.mean(2)?; // [N, K]
            let mut loss = per.mean_all()?.affine(k as f64, 0.0)?;
            loss = (loss + decorrelation_loss_gated(&resid.mean_keepdim(2)?, &p)?.affine(c.moe.lambda_dec, 0.0)?)?;
            if c.lambda_skill > 0.0 {
                let den = (p.sum(0)? + 1e-6)?;
                let mse_k = (&per * &p)?.sum(0)?.div(&den)?;
                let naive_k = p.broadcast_mul(&y2)?.sum(0)?.div(&den)?;
                let hinge = mse_k.div(&naive_k)?.affine(1.0, c.skill_floor - 1.0)?.relu()?.sum_all()?;
                loss = (loss + hinge.affine(c.lambda_skill, 0.0)?)?;
            }
            last = scalar(&loss)?;
            opt.step(&heads, &loss)?;
        }
        let fitted: HashMap<String, Tensor> = heads.snapshot()?.into_iter().collect();
        self.model.ps.load(&fitted)?;
        Ok(last)
    }

    pub fn done(&self) -> bool {
        self.step as usize >= self.model.cfg.steps
    }

    /// Losses and expert-space health on `windows` (validation by default), in chunks.
    pub fn evaluate(&self, ds: &Dataset, windows: &[(usize, usize)]) -> Result<EvalReport> {
        let k = self.model.moe.n_active();
        let top_k = self.model.cfg.moe.top_k;
        let mut resid: Vec<Vec<f64>> = vec![Vec::new(); k];
        let mut sq: Vec<Vec<f64>> = vec![Vec::new(); k];
        let mut y2: Vec<f64> = Vec::new();
        let mut gates: Vec<Vec<f32>> = Vec::new();
        let mut usage = vec![0.0; k];
        let (mut naive, mut mixsq, mut n) = (0.0, 0.0, 0usize);
        let mut t_naive = [0.0f64; N_TARGETS];
        let mut t_sq = [0.0f64; N_TARGETS];
        let mut mix_scalar: Vec<f64> = Vec::new();
        let mut sum = LossLog::default();
        let mut chunks = 0.0;
        for chunk in windows.chunks(256) {
            let b = self.batch(ds, chunk)?;
            let (_, log, f) = self.losses(&b, None)?;
            let w = chunk.len() as f64;
            for (acc, v) in [
                (&mut sum.total, log.total),
                (&mut sum.pred, log.pred),
                (&mut sum.mix, log.mix),
                (&mut sum.decor, log.decor),
                (&mut sum.balance, log.balance),
                (&mut sum.kl, log.kl),
                (&mut sum.ssl, log.ssl),
                (&mut sum.anchor, log.anchor),
                (&mut sum.feedback, log.feedback),
                (&mut sum.prior, log.prior),
            ] {
                *acc += v * w;
            }
            chunks += w;
            let r = f.out.pred.broadcast_sub(&b.y.unsqueeze(1)?)?.to_vec3::<f32>()?; // [B][K][T]
            let p = f.out.probs.to_vec2::<f32>()?;
            let y = b.y.to_vec2::<f32>()?;
            let mixp = NeuralModel::mixture(&f)?.to_vec2::<f32>()?;
            for (bi, row) in r.iter().enumerate() {
                for (e, rt) in row.iter().enumerate() {
                    resid[e].push(rt.iter().map(|v| f64::from(*v)).sum::<f64>() / N_TARGETS as f64);
                    sq[e].push(rt.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / N_TARGETS as f64);
                    usage[e] += f64::from(p[bi][e]);
                }
                gates.push(fa_expert_space::moe::top_k_gates(&p[bi], top_k));
                let yy = y[bi].iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / N_TARGETS as f64;
                y2.push(yy);
                naive += yy;
                mixsq +=
                    y[bi].iter().zip(&mixp[bi]).map(|(a, b)| f64::from(a - b).powi(2)).sum::<f64>() / N_TARGETS as f64;
                for t in 0..N_TARGETS {
                    t_naive[t] += f64::from(y[bi][t]).powi(2);
                    t_sq[t] += f64::from(y[bi][t] - mixp[bi][t]).powi(2);
                }
                mix_scalar
                    .push(y[bi].iter().zip(&mixp[bi]).map(|(a, b)| f64::from(a - b)).sum::<f64>() / N_TARGETS as f64);
                n += 1;
            }
        }
        if n == 0 {
            return Ok(EvalReport::default());
        }
        let nf = n as f64;
        for v in [
            &mut sum.total,
            &mut sum.pred,
            &mut sum.mix,
            &mut sum.decor,
            &mut sum.balance,
            &mut sum.kl,
            &mut sum.ssl,
            &mut sum.anchor,
            &mut sum.feedback,
            &mut sum.prior,
        ] {
            *v /= chunks.max(1.0);
        }
        sum.step = self.step;
        // spec 32 §6.3/§6.4: residual correlation over windows where both experts are active,
        // skill of each expert over the windows routed to it (top-k gate weighted)
        let health = HealthReport::compute_gated(&resid, &sq, &y2, &gates, usage.iter().map(|u| u / nf).collect(), 30);
        let ms = mix_scalar.iter().sum::<f64>() / nf;
        let mix_resid_var = mix_scalar.iter().map(|v| (v - ms).powi(2)).sum::<f64>() / nf;
        let target_skill =
            t_sq.iter().zip(&t_naive).map(|(s, n0)| if *n0 > 0.0 { 1.0 - s / n0 } else { 0.0 }).collect();
        Ok(EvalReport {
            loss: sum,
            health,
            mix_skill: if naive > 0.0 { 1.0 - mixsq / naive } else { 0.0 },
            target_skill,
            mix_resid_var,
            n,
        })
    }

    /// Emergent expert count (spec 32 §6.1): drops experts whose validation usage is below
    /// `prune_usage_below`, keeping at least `min_keep` (by usage). Returns the kept expert ids.
    pub fn prune(&mut self, usage: &[f64], min_keep: usize) -> Vec<usize> {
        let thr = self.model.cfg.moe.prune_usage_below;
        let active = self.model.moe.active.clone();
        let mut order: Vec<usize> = (0..active.len()).collect();
        order.sort_by(|a, b| usage[*b].total_cmp(&usage[*a]).then(a.cmp(b)));
        let mut keep: Vec<usize> = order
            .iter()
            .enumerate()
            .filter(|(rank, i)| usage[**i] >= thr || *rank < min_keep)
            .map(|(_, i)| *i)
            .collect();
        keep.sort_unstable();
        let kept: Vec<usize> = keep.iter().map(|&i| active[i]).collect();
        self.model.moe.select(kept.clone());
        kept
    }

    /// Resumable state (parameters, Adam moments, EMA target, step, norm, config).
    pub fn checkpoint(&self) -> Result<Vec<u8>> {
        let mut t = self.model.ps.named("p.");
        t.extend(self.opt.named_state());
        if let Some(e) = &self.ema {
            t.extend(e.iter().map(|(k, v)| (format!("ema.{k}"), v.clone())));
        }
        let mut meta = weights_meta(&self.model);
        meta.insert("kind".into(), "checkpoint".into());
        meta.insert("step".into(), self.step.to_string());
        meta.insert("seed".into(), self.seed.to_string());
        meta.insert("adam_t".into(), self.opt.t.to_string());
        if let Some(l) = self.lambda_dec {
            meta.insert("lambda_dec".into(), format!("{l:e}"));
        }
        meta.insert("log".into(), serde_json::to_string(&self.log).unwrap_or_default());
        to_safetensors(t, meta)
    }

    /// Restores a trainer from [`Self::checkpoint`] bytes (anchors / feedback are re-attached by the caller).
    pub fn from_checkpoint(bytes: &[u8], dev: &Device) -> Result<Self> {
        let (vals, meta) = from_safetensors(bytes, dev)?;
        let model = model_from_parts(&vals, &meta, dev)?;
        let seed = meta.get("seed").and_then(|s| s.parse().ok()).unwrap_or(0);
        let mut tr = Self::from_model(model, seed);
        tr.step = meta.get("step").and_then(|s| s.parse().ok()).unwrap_or(0);
        tr.opt.restore(meta.get("adam_t").and_then(|s| s.parse().ok()).unwrap_or(0), &vals);
        if tr.ema.is_some() {
            tr.ema = Some(
                vals.iter().filter_map(|(k, v)| k.strip_prefix("ema.").map(|n| (n.to_owned(), v.clone()))).collect(),
            );
        }
        tr.log = meta.get("log").and_then(|s| serde_json::from_str(s).ok()).unwrap_or_default();
        tr.lambda_dec = meta.get("lambda_dec").and_then(|s| s.parse().ok());
        Ok(tr)
    }

    /// `weights.bin` of the snapshot: parameters + metadata, no optimiser state.
    pub fn weights(&self) -> Result<Vec<u8>> {
        model_weights(&self.model)
    }
}

pub(crate) fn weights_meta(m: &NeuralModel) -> HashMap<String, String> {
    HashMap::from([
        ("format".to_owned(), META_FORMAT.to_owned()),
        ("family".to_owned(), m.cfg.family.as_str().to_owned()),
        ("config".to_owned(), serde_json::to_string(&m.cfg).unwrap_or_default()),
        ("norm".to_owned(), serde_json::to_string(&m.norm).unwrap_or_default()),
        ("active".to_owned(), serde_json::to_string(&m.moe.active).unwrap_or_default()),
    ])
}

pub fn model_weights(m: &NeuralModel) -> Result<Vec<u8>> {
    let mut meta = weights_meta(m);
    meta.insert("kind".into(), "weights".into());
    to_safetensors(m.ps.named("p."), meta)
}

/// Rebuilds a model from weights / checkpoint tensors and metadata.
pub(crate) fn model_from_parts(
    vals: &HashMap<String, Tensor>,
    meta: &HashMap<String, String>,
    dev: &Device,
) -> Result<NeuralModel> {
    let bad = |m: &str| candle_core::Error::Msg(format!("neural weights: {m}"));
    if meta.get("format").map(String::as_str) != Some(META_FORMAT) {
        return Err(bad("missing fa-neural-v1 format tag"));
    }
    let cfg: NeuralConfig = serde_json::from_str(meta.get("config").ok_or_else(|| bad("no config"))?)
        .map_err(|e| bad(&format!("config: {e}")))?;
    let norm: InputNorm = serde_json::from_str(meta.get("norm").ok_or_else(|| bad("no norm"))?)
        .map_err(|e| bad(&format!("norm: {e}")))?;
    let active: Vec<usize> = meta.get("active").and_then(|a| serde_json::from_str(a).ok()).unwrap_or_default();
    let mut model = NeuralModel::new(cfg, norm, 0, dev)?;
    let stripped: HashMap<String, Tensor> =
        vals.iter().filter_map(|(k, v)| k.strip_prefix("p.").map(|s| (s.to_owned(), v.clone()))).collect();
    let n = model.ps.load(&stripped)?;
    if n != model.ps.len() {
        return Err(bad(&format!("{n} of {} parameters present", model.ps.len())));
    }
    if !active.is_empty() {
        model.moe.select(active);
    }
    Ok(model)
}
