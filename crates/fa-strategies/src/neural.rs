//! Neural policies (spec 35 §4.3): a small MLP over the strategy's per-asset observation —
//! expert latents (masked by genome) and the same graph / decoder / portfolio primitives the GP
//! programs read — that outputs `(mean = tanh(o₀), std = softplus(o₁))` per asset.
//!
//! The network is applied to every asset of the set with shared weights (set-equivariant: the
//! output for an asset depends only on its own observation; the portfolio-level inputs
//! `exposure_gross` and `drawdown` carry the strategy-wide state). Inference is plain `f32`
//! Rust (deterministic, no ML runtime in the live path); training runs on candle in
//! [`train`](crate::neural::train) (feature `train`) and the forward pass there is pinned to this
//! one by a test.
//!
//! **Text form** (one line, part of the strategy genome): `mlp:f=<hex feature mask>;e=<hex expert
//! mask>;l=<latent dims>;h=<w1>x<w2>;a=tanh|relu;w=<base64 f32 LE weights>`. Weights are stored
//! exactly, so `parse(&net.to_string()) == net` and the genome id is a hash of the phenotype.
//!
//! **Variation** (evo-worker `evo.vary.strategy`, §6.3): Gaussian weight perturbation (ES),
//! topology mutation (a hidden unit added with zero outgoing weights — function-preserving — or
//! removed; a layer added as a near-identity or removed), input-mask flips (a feature or an
//! expert's latents switched on with zero weights, or off), arithmetic crossover between nets of
//! the same topology, and the SGD fine-tune of [`train`] on a training window that precedes the
//! selection window.

use std::{
    fmt,
    sync::{Arc, Mutex},
};

use fa_core::{Horizon, InstrumentId, rng::Rng};
use serde::{Deserialize, Serialize};

use crate::{AssetExposure, ExposureDistribution, ProbSrc, Strategy, StrategyContext};

#[cfg(feature = "train")]
pub mod train;

/// Observation features, in input order (scaled to roughly `[-1, 1]`).
pub const FEATURES: [&str; 20] = [
    "stay_prob",
    "path_up_1",
    "path_up_h",
    "region_entropy",
    "novelty",
    "dwell",
    "region_mean_fwd_return",
    "region_vol",
    "fc_dist",
    "fc_iqr",
    "prob_up",
    "analog_outcome",
    "analog_age",
    "text_score",
    "visual_score",
    "position",
    "unrealised_pnl",
    "time_in_trade",
    "exposure_gross",
    "drawdown",
];

/// All features on.
pub const ALL_FEATURES: u32 = (1 << FEATURES.len()) - 1;
/// Latent mean dimensions per expert available to the policy.
pub const MAX_LATENT_DIMS: usize = 8;
/// Expert spaces addressable by the expert mask.
pub const MAX_EXPERTS: u32 = 8;
/// Hidden width bounds.
pub const MIN_WIDTH: usize = 2;
pub const MAX_WIDTH: usize = 64;
/// At most this many hidden layers.
pub const MAX_LAYERS: usize = 3;

/// Hidden activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Act {
    Tanh,
    Relu,
}

impl Act {
    pub fn as_str(self) -> &'static str {
        match self {
            Act::Tanh => "tanh",
            Act::Relu => "relu",
        }
    }

    fn apply(self, x: f32) -> f32 {
        match self {
            Act::Tanh => x.tanh(),
            Act::Relu => x.max(0.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NetError {
    #[error("neural policy text must start with `mlp:`")]
    Prefix,
    #[error("bad field `{0}`")]
    Field(String),
    #[error("missing field `{0}`")]
    Missing(&'static str),
    #[error("weights: expected {expected} values, got {got}")]
    Weights { expected: usize, got: usize },
    #[error("bad base64 weights")]
    Base64,
    #[error("non-finite weight")]
    NonFinite,
    #[error("topology out of bounds: {0}")]
    Topology(String),
}

/// A policy network: input masks, hidden layers and flat weights.
///
/// Weight layout: for each layer (inputs → hidden₁ → … → 2 outputs) the matrix row-major
/// `[out][in]` followed by the bias `[out]`.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyNet {
    /// Bit `i` = [`FEATURES`]`[i]` is an input.
    pub feature_mask: u32,
    /// Bit `e` = the latent mean of expert `e` (first `latent_dims` dims) is an input.
    pub expert_mask: u32,
    pub latent_dims: usize,
    pub hidden: Vec<usize>,
    pub act: Act,
    pub weights: Vec<f32>,
}

/// Network output for one asset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolicyOut {
    pub mean: f64,
    pub std: f64,
}

/// `softplus(x) = ln(1 + eˣ)`, overflow-safe.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

impl PolicyNet {
    pub fn n_features(&self) -> usize {
        (self.feature_mask & ALL_FEATURES).count_ones() as usize
    }

    pub fn n_experts(&self) -> usize {
        (self.expert_mask & ((1 << MAX_EXPERTS) - 1)).count_ones() as usize
    }

    /// Input width.
    pub fn input_dim(&self) -> usize {
        self.n_features() + self.n_experts() * self.latent_dims
    }

    /// Layer shapes `(in, out)` from input to the 2-wide output.
    pub fn layers(&self) -> Vec<(usize, usize)> {
        let mut dims = vec![self.input_dim()];
        dims.extend(&self.hidden);
        dims.push(2);
        dims.windows(2).map(|w| (w[0], w[1])).collect()
    }

    /// Parameters for the topology.
    pub fn n_params(&self) -> usize {
        self.layers().iter().map(|(i, o)| i * o + o).sum()
    }

    /// Checks topology bounds and the weight count.
    pub fn validate(&self) -> Result<(), NetError> {
        if self.hidden.is_empty() || self.hidden.len() > MAX_LAYERS {
            return Err(NetError::Topology(format!("{} hidden layers", self.hidden.len())));
        }
        if let Some(w) = self.hidden.iter().find(|w| !(MIN_WIDTH..=MAX_WIDTH).contains(*w)) {
            return Err(NetError::Topology(format!("width {w}")));
        }
        if self.latent_dims > MAX_LATENT_DIMS {
            return Err(NetError::Topology(format!("{} latent dims", self.latent_dims)));
        }
        if self.input_dim() == 0 {
            return Err(NetError::Topology("no inputs".into()));
        }
        if self.weights.len() != self.n_params() {
            return Err(NetError::Weights { expected: self.n_params(), got: self.weights.len() });
        }
        if self.weights.iter().any(|w| !w.is_finite()) {
            return Err(NetError::NonFinite);
        }
        Ok(())
    }

    /// Fresh network with scaled-uniform (Glorot) weights, small output layer (outputs start near
    /// `mean ≈ 0`, `std ≈ softplus(0)`); deterministic in the `Rng`.
    pub fn random(
        rng: &mut Rng,
        feature_mask: u32,
        expert_mask: u32,
        latent_dims: usize,
        hidden: Vec<usize>,
        act: Act,
    ) -> Self {
        let mut n = Self {
            feature_mask: feature_mask & ALL_FEATURES,
            expert_mask: expert_mask & ((1 << MAX_EXPERTS) - 1),
            latent_dims: latent_dims.min(MAX_LATENT_DIMS),
            hidden,
            act,
            weights: Vec::new(),
        };
        let layers = n.layers();
        let last = layers.len() - 1;
        for (li, (fan_in, fan_out)) in layers.into_iter().enumerate() {
            let limit = (6.0 / (fan_in + fan_out).max(1) as f64).sqrt() * if li == last { 0.1 } else { 1.0 };
            for _ in 0..fan_in * fan_out {
                n.weights.push(((rng.uniform() * 2.0 - 1.0) * limit) as f32);
            }
            n.weights.extend(std::iter::repeat_n(0.0, fan_out));
        }
        n
    }

    /// Raw outputs `(o₀, o₁)` for one input vector (`x.len() == input_dim()`).
    pub fn forward_raw(&self, x: &[f32]) -> (f32, f32) {
        let mut cur: Vec<f32> = x.to_vec();
        let mut off = 0;
        let layers = self.layers();
        let last = layers.len() - 1;
        for (li, (fan_in, fan_out)) in layers.into_iter().enumerate() {
            let w = &self.weights[off..off + fan_in * fan_out];
            let b = &self.weights[off + fan_in * fan_out..off + fan_in * fan_out + fan_out];
            off += fan_in * fan_out + fan_out;
            let mut next = Vec::with_capacity(fan_out);
            for o in 0..fan_out {
                let row = &w[o * fan_in..(o + 1) * fan_in];
                let z = row.iter().zip(&cur).map(|(a, b)| a * b).sum::<f32>() + b[o];
                next.push(if li == last { z } else { self.act.apply(z) });
            }
            cur = next;
        }
        (cur[0], cur[1])
    }

    /// `(mean, std)` with the spec 35 §4.3 heads: `tanh` and `softplus` (std clipped to [0, 1]).
    /// Non-finite inputs are zeroed; a non-finite output yields `mean = 0`, `std = 1`.
    pub fn forward(&self, x: &[f32]) -> PolicyOut {
        let clean: Vec<f32> = x.iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect();
        let (o0, o1) = self.forward_raw(&clean);
        let mean = f64::from(o0.tanh());
        let std = f64::from(softplus(o1)).min(1.0);
        if mean.is_finite() && std.is_finite() { PolicyOut { mean, std } } else { PolicyOut { mean: 0.0, std: 1.0 } }
    }

    /// Input vector for one asset: the masked [`FEATURES`] then the masked experts' latents.
    pub fn inputs(&self, ctx: &dyn StrategyContext, horizon: u32) -> Vec<f32> {
        let mut x = Vec::with_capacity(self.input_dim());
        let all = feature_vector(ctx, horizon);
        for (i, v) in all.iter().enumerate() {
            if self.feature_mask & (1 << i) != 0 {
                x.push(*v);
            }
        }
        for e in 0..MAX_EXPERTS {
            if self.expert_mask & (1 << e) == 0 {
                continue;
            }
            let lat = ctx.latent(e).unwrap_or_default();
            for d in 0..self.latent_dims {
                // latent means are roughly unit-scaled by the encoders; squash outliers
                x.push(lat.get(d).copied().map_or(0.0, |v| (v / 3.0).tanh() * 3.0));
            }
        }
        x
    }

    /// Estimated operations per asset and step ([`crate::op_cost`] units): `2·MACs` of the
    /// layers plus 10 per observation feature read (the primitives it shares with GP) and 1 per
    /// latent input.
    pub fn flops(&self) -> f64 {
        let macs: usize = self.layers().iter().map(|(i, o)| i * o).sum();
        2.0 * macs as f64 + 10.0 * self.n_features() as f64 + (self.n_experts() * self.latent_dims) as f64
    }

    /// Short topology label (`16x8-tanh`).
    pub fn topology(&self) -> String {
        format!("{}-{}", self.hidden.iter().map(ToString::to_string).collect::<Vec<_>>().join("x"), self.act.as_str())
    }

    /// Parses the text form (module docs).
    pub fn parse(text: &str) -> Result<Self, NetError> {
        let body = text.trim().strip_prefix("mlp:").ok_or(NetError::Prefix)?;
        let (mut f, mut e, mut l, mut h, mut a, mut w) = (None, None, None, None, None, None);
        for part in body.split(';').filter(|p| !p.is_empty()) {
            let (k, v) = part.split_once('=').ok_or_else(|| NetError::Field(part.into()))?;
            let bad = || NetError::Field(part.into());
            match k {
                "f" => f = Some(u32::from_str_radix(v, 16).map_err(|_| bad())?),
                "e" => e = Some(u32::from_str_radix(v, 16).map_err(|_| bad())?),
                "l" => l = Some(v.parse::<usize>().map_err(|_| bad())?),
                "h" => {
                    h = Some(v.split('x').map(|x| x.parse::<usize>()).collect::<Result<Vec<_>, _>>().map_err(|_| bad())?)
                }
                "a" => {
                    a = Some(match v {
                        "tanh" => Act::Tanh,
                        "relu" => Act::Relu,
                        _ => return Err(bad()),
                    })
                }
                "w" => w = Some(decode_weights(v)?),
                _ => return Err(bad()),
            }
        }
        let net = Self {
            feature_mask: f.ok_or(NetError::Missing("f"))?,
            expert_mask: e.unwrap_or(0),
            latent_dims: l.unwrap_or(0),
            hidden: h.ok_or(NetError::Missing("h"))?,
            act: a.unwrap_or(Act::Tanh),
            weights: w.ok_or(NetError::Missing("w"))?,
        };
        if net.feature_mask & !ALL_FEATURES != 0 || net.expert_mask >= (1 << MAX_EXPERTS) {
            return Err(NetError::Topology("mask bits out of range".into()));
        }
        net.validate()?;
        Ok(net)
    }

    // ───────────────────────────────── variation (spec 35 §6.3) ─────────────────────────────────

    /// ES step: every weight `+= sigma · scale · N(0, 1)`, `scale` = the layer's Glorot limit.
    pub fn perturb(&self, rng: &mut Rng, sigma: f64) -> Self {
        let mut c = self.clone();
        let mut off = 0;
        for (fan_in, fan_out) in self.layers() {
            let scale = (2.0 / (fan_in + fan_out).max(1) as f64).sqrt();
            for w in &mut c.weights[off..off + fan_in * fan_out + fan_out] {
                *w += (sigma * scale * rng.normal()) as f32;
            }
            off += fan_in * fan_out + fan_out;
        }
        c
    }

    /// Splits the flat weights into per-layer `(W [out][in], b [out])`.
    pub fn unpack(&self) -> Vec<(Vec<Vec<f32>>, Vec<f32>)> {
        let mut out = Vec::new();
        let mut off = 0;
        for (fan_in, fan_out) in self.layers() {
            let w = (0..fan_out).map(|o| self.weights[off + o * fan_in..off + (o + 1) * fan_in].to_vec()).collect();
            let b = self.weights[off + fan_in * fan_out..off + fan_in * fan_out + fan_out].to_vec();
            off += fan_in * fan_out + fan_out;
            out.push((w, b));
        }
        out
    }

    fn pack(&mut self, layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>) {
        self.weights = layers.into_iter().flat_map(|(w, b)| w.into_iter().flatten().chain(b)).collect();
    }

    /// Topology mutation: widen a hidden layer by one unit (zero outgoing weights: the function is
    /// unchanged), narrow it by dropping its smallest-norm unit, insert a near-identity layer or
    /// drop a layer (the layer after absorbs it by re-initialisation of the joint).
    pub fn mutate_topology(&self, rng: &mut Rng) -> Self {
        let mut c = self.clone();
        let mut layers = self.unpack();
        let li = rng.below(self.hidden.len());
        match rng.below(4) {
            0 if self.hidden[li] < MAX_WIDTH => {
                // widen hidden layer li: new row in layer li, new zero column in layer li + 1
                let fan_in = layers[li].0[0].len();
                let limit = (6.0 / (fan_in + self.hidden[li] + 1) as f64).sqrt();
                layers[li].0.push((0..fan_in).map(|_| ((rng.uniform() * 2.0 - 1.0) * limit) as f32).collect());
                layers[li].1.push(0.0);
                for row in &mut layers[li + 1].0 {
                    row.push(0.0);
                }
                c.hidden[li] += 1;
            }
            1 if self.hidden[li] > MIN_WIDTH => {
                // narrow: drop the unit with the smallest outgoing weight norm
                let norm = |u: usize| layers[li + 1].0.iter().map(|r| r[u] * r[u]).sum::<f32>();
                let u = (0..self.hidden[li]).min_by(|a, b| norm(*a).total_cmp(&norm(*b))).unwrap_or(0);
                layers[li].0.remove(u);
                layers[li].1.remove(u);
                for row in &mut layers[li + 1].0 {
                    row.remove(u);
                }
                c.hidden[li] -= 1;
            }
            2 if self.hidden.len() < MAX_LAYERS => {
                // insert an identity-initialised layer after hidden layer li
                let w = self.hidden[li];
                let ident: Vec<Vec<f32>> = (0..w).map(|o| (0..w).map(|i| if i == o { 1.0 } else { 0.0 }).collect()).collect();
                layers.insert(li + 1, (ident, vec![0.0; w]));
                c.hidden.insert(li + 1, w);
            }
            3 if self.hidden.len() > 1 => {
                // drop hidden layer li: compose the next layer onto the previous layer's outputs
                // linearly (exact for a linear unit, an approximation through the activation)
                let (wa, ba) = layers[li].clone();
                let (wb, bb) = layers[li + 1].clone();
                let w: Vec<Vec<f32>> = wb
                    .iter()
                    .map(|rb| (0..wa[0].len()).map(|i| rb.iter().zip(&wa).map(|(x, ra)| x * ra[i]).sum()).collect())
                    .collect();
                let b: Vec<f32> =
                    wb.iter().zip(&bb).map(|(rb, bo)| rb.iter().zip(&ba).map(|(x, y)| x * y).sum::<f32>() + bo).collect();
                layers.splice(li..li + 2, [(w, b)]);
                c.hidden.remove(li);
            }
            _ => return self.perturb(rng, 0.1),
        }
        c.pack(layers);
        c
    }

    /// Input-mask flip: one feature (or one expert's latents) switched on with zero weights —
    /// the function is unchanged until training or ES moves them — or switched off (its input
    /// columns dropped). Never removes the last input.
    pub fn mutate_inputs(&self, rng: &mut Rng, experts: u32) -> Self {
        let mut c = self.clone();
        let mut layers = self.unpack();
        // input column of feature bit i / expert e in the current layout
        let feature_col = |mask: u32, i: usize| (mask & ((1u32 << i) - 1)).count_ones() as usize;
        let experts = experts.clamp(1, MAX_EXPERTS);
        let flip_expert = self.latent_dims > 0 && rng.uniform() < 0.25;
        if flip_expert {
            let e = rng.below(experts as usize);
            let base = self.n_features() + (self.expert_mask & ((1 << e) - 1)).count_ones() as usize * self.latent_dims;
            if self.expert_mask & (1 << e) != 0 {
                if self.input_dim() <= self.latent_dims {
                    return self.perturb(rng, 0.1);
                }
                for row in &mut layers[0].0 {
                    row.drain(base..base + self.latent_dims);
                }
                c.expert_mask &= !(1 << e);
            } else {
                for row in &mut layers[0].0 {
                    for _ in 0..self.latent_dims {
                        row.insert(base, 0.0);
                    }
                }
                c.expert_mask |= 1 << e;
            }
        } else {
            let i = rng.below(FEATURES.len());
            let col = feature_col(self.feature_mask, i);
            if self.feature_mask & (1 << i) != 0 {
                if self.input_dim() <= 1 {
                    return self.perturb(rng, 0.1);
                }
                for row in &mut layers[0].0 {
                    row.remove(col);
                }
                c.feature_mask &= !(1 << i);
            } else {
                for row in &mut layers[0].0 {
                    row.insert(col, 0.0);
                }
                c.feature_mask |= 1 << i;
            }
        }
        c.pack(layers);
        c
    }

    /// Arithmetic crossover: `α·a + (1 − α)·b` per layer when the topologies match, else `a`.
    pub fn crossover(&self, other: &Self, rng: &mut Rng) -> Self {
        if self.layers() != other.layers()
            || self.feature_mask != other.feature_mask
            || self.expert_mask != other.expert_mask
        {
            return self.clone();
        }
        let mut c = self.clone();
        let mut off = 0;
        for (fan_in, fan_out) in self.layers() {
            let alpha = rng.uniform() as f32;
            for k in off..off + fan_in * fan_out + fan_out {
                c.weights[k] = alpha * self.weights[k] + (1.0 - alpha) * other.weights[k];
            }
            off += fan_in * fan_out + fan_out;
        }
        c
    }
}

/// One recorded policy input (training samples, [`NeuralStrategy::recorder`]).
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureRow {
    pub as_of_ms: i64,
    pub instrument: InstrumentId,
    pub x: Vec<f32>,
    /// Exposure mean the policy emitted for it.
    pub mean: f32,
}

/// Shared sink of [`FeatureRow`]s.
pub type Recorder = Arc<Mutex<Vec<FeatureRow>>>;

/// A neural policy as a [`Strategy`] (spec 35 §4.3).
#[derive(Debug, Clone)]
pub struct NeuralStrategy {
    pub id: String,
    pub genome_id: String,
    pub horizon: Horizon,
    pub net: PolicyNet,
    /// When set, every step's inputs are appended (the replay that collects SGD samples).
    pub recorder: Option<Recorder>,
}

impl Strategy for NeuralStrategy {
    fn id(&self) -> &str {
        &self.id
    }

    fn horizon(&self) -> Horizon {
        self.horizon
    }

    fn ops_per_asset(&self) -> f64 {
        self.net.flops()
    }

    fn genome_id(&self) -> &str {
        &self.genome_id
    }

    fn step(&mut self, as_of_ms: i64, assets: &[(InstrumentId, &dyn StrategyContext)]) -> ExposureDistribution {
        let mut exposures = Vec::with_capacity(assets.len());
        for (inst, ctx) in assets {
            let x = self.net.inputs(*ctx, self.horizon.steps);
            let o = self.net.forward(&x);
            if let Some(r) = &self.recorder
                && let Ok(mut rows) = r.lock()
            {
                rows.push(FeatureRow { as_of_ms, instrument: *inst, x, mean: o.mean as f32 });
            }
            exposures.push(AssetExposure { instrument: *inst, mean: o.mean, stddev: o.std });
        }
        ExposureDistribution {
            strategy_id: self.id.clone(),
            genome_id: self.genome_id.clone(),
            horizon: self.horizon,
            as_of_ms,
            exposures,
        }
    }
}

impl fmt::Display for PolicyNet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "mlp:f={:x};e={:x};l={};h={};a={};w={}",
            self.feature_mask,
            self.expert_mask,
            self.latent_dims,
            self.hidden.iter().map(ToString::to_string).collect::<Vec<_>>().join("x"),
            self.act.as_str(),
            encode_weights(&self.weights)
        )
    }
}

/// The full feature vector (all [`FEATURES`]) of one asset at one step.
pub fn feature_vector(ctx: &dyn StrategyContext, horizon: u32) -> [f32; 20] {
    let h = horizon.max(1);
    let cur = crate::RegionRef::Cur(0);
    let t = |x: f64| x.tanh() as f32;
    let q75 = ctx.fc_quantile(0.75, h);
    let q25 = ctx.fc_quantile(0.25, h);
    [
        (ctx.stay_prob() * 2.0 - 1.0) as f32,
        ((ctx.path_up_prob(1) - 0.5) * 2.0) as f32,
        ((ctx.path_up_prob(h) - 0.5) * 2.0) as f32,
        t(ctx.region_entropy()),
        t(ctx.novelty(0)),
        t(ctx.dwell(cur) / 10.0),
        t(ctx.region_feature(cur, crate::RegionFeature::MeanFwdReturn) * 200.0),
        t(ctx.region_feature(cur, crate::RegionFeature::Vol) * 100.0),
        t(ctx.fc_dist(h) * 200.0),
        t((q75 - q25) * 100.0),
        ((ctx.prob_gt(ProbSrc::Fc(h), 0.0) - 0.5) * 2.0) as f32,
        t(ctx.analog_outcome(h) * 200.0),
        t(ctx.analog_age() / 30.0),
        t(ctx.text_score("news")),
        t(ctx.visual_score("future")),
        t(ctx.position()),
        t(ctx.unrealised_pnl() * 20.0),
        t(ctx.time_in_trade() / 20.0),
        t(ctx.exposure_gross() / 2.0),
        t(ctx.drawdown() * 20.0),
    ]
}

// ─────────────────────────────────────────── base64 ───────────────────────────────────────────

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// URL-safe base64 (no padding) of the weights as little-endian `f32`.
pub fn encode_weights(w: &[f32]) -> String {
    let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut out = String::with_capacity(bytes.len() * 4 / 3 + 4);
    for chunk in bytes.chunks(3) {
        let n = chunk.len();
        let v = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        for i in 0..=n {
            out.push(B64[((v >> (18 - 6 * i)) & 63) as usize] as char);
        }
    }
    out
}

/// Inverse of [`encode_weights`].
pub fn decode_weights(s: &str) -> Result<Vec<f32>, NetError> {
    let mut val = [0u8; 256];
    for (i, c) in B64.iter().enumerate() {
        val[*c as usize] = i as u8 + 1;
    }
    let mut bytes = Vec::with_capacity(s.len() * 3 / 4);
    let digits: Vec<u32> = s
        .bytes()
        .map(|c| match val[c as usize] {
            0 => Err(NetError::Base64),
            v => Ok(u32::from(v - 1)),
        })
        .collect::<Result<_, _>>()?;
    for chunk in digits.chunks(4) {
        if chunk.len() == 1 {
            return Err(NetError::Base64);
        }
        let mut v = 0u32;
        for (i, d) in chunk.iter().enumerate() {
            v |= d << (18 - 6 * i);
        }
        for i in 0..chunk.len() - 1 {
            bytes.push((v >> (16 - 8 * i)) as u8);
        }
    }
    if bytes.len() % 4 != 0 {
        return Err(NetError::Base64);
    }
    Ok(bytes.chunks(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{FallbackContext, VirtualBook};

    fn net(seed: u64) -> PolicyNet {
        PolicyNet::random(&mut Rng::seed(seed), ALL_FEATURES, 0b1, 4, vec![8, 4], Act::Tanh)
    }

    #[test]
    fn text_form_round_trips_exactly() {
        let n = net(1);
        assert_eq!(n.input_dim(), 24);
        assert_eq!(n.n_params(), 24 * 8 + 8 + 8 * 4 + 4 + 4 * 2 + 2);
        n.validate().unwrap();
        let t = n.to_string();
        assert!(t.starts_with("mlp:f=fffff;e=1;l=4;h=8x4;a=tanh;w=") && !t.contains(' '), "{t}");
        assert_eq!(PolicyNet::parse(&t).unwrap(), n);
        for w in [vec![], vec![1.5f32], vec![1.0, -2.0], vec![0.1, 0.2, f32::MIN_POSITIVE]] {
            assert_eq!(decode_weights(&encode_weights(&w)).unwrap(), w);
        }
        assert_eq!(PolicyNet::parse("gp:x"), Err(NetError::Prefix));
        assert!(matches!(PolicyNet::parse("mlp:f=1;h=4;w=AAAA"), Err(NetError::Base64 | NetError::Weights { .. })));
        assert!(PolicyNet::parse(&t.replace("h=8x4", "h=8x5")).is_err(), "weight count is checked");
    }

    #[test]
    fn forward_is_bounded_and_finite() {
        let n = net(2);
        let closes: Vec<f64> = (0..40).map(|i| 100.0 + (i as f64 * 0.3).sin()).collect();
        let ctx = FallbackContext::new(&closes, None, VirtualBook::default());
        let x = n.inputs(&ctx, 3);
        assert_eq!(x.len(), n.input_dim());
        let o = n.forward(&x);
        assert!((-1.0..=1.0).contains(&o.mean) && (0.0..=1.0).contains(&o.std), "{o:?}");
        let o = n.forward(&vec![f32::NAN; n.input_dim()]);
        assert!(o.mean.is_finite() && o.std.is_finite());
        // a fresh net starts near flat (small output layer)
        assert!(o.mean.abs() < 0.5);
    }

    #[test]
    fn function_preserving_mutations_keep_outputs() {
        let n = net(3);
        let x: Vec<f32> = (0..n.input_dim()).map(|i| (i as f32 * 0.37).sin()).collect();
        let base = n.forward_raw(&x);
        let mut rng = Rng::seed(4);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..200 {
            let c = n.mutate_topology(&mut rng);
            c.validate().unwrap();
            seen.insert(c.topology());
            assert_eq!(PolicyNet::parse(&c.to_string()).unwrap(), c);
            if c.hidden.len() == n.hidden.len() && c.hidden[0] == n.hidden[0] + 1 && c.hidden[1] == n.hidden[1] {
                let o = c.forward_raw(&x);
                assert!((o.0 - base.0).abs() < 1e-5 && (o.1 - base.1).abs() < 1e-5, "widening preserves the function");
            }
        }
        assert!(seen.len() >= 3, "{seen:?}");
        for _ in 0..200 {
            let c = n.mutate_inputs(&mut rng, 2);
            c.validate().unwrap();
            assert_ne!((c.feature_mask, c.expert_mask), (n.feature_mask, n.expert_mask));
            if c.input_dim() > n.input_dim() {
                // switched-on inputs start with zero weights: same output when the old inputs are kept
                let mut xi = Vec::new();
                let all: Vec<f32> = (0..FEATURES.len()).map(|i| (i as f32 * 0.37).sin()).collect();
                for (i, v) in all.iter().enumerate() {
                    if c.feature_mask & (1 << i) != 0 {
                        xi.push(if n.feature_mask & (1 << i) != 0 { *v } else { 9.0 });
                    }
                }
                let lat: Vec<f32> = (FEATURES.len()..n.input_dim()).map(|i| (i as f32 * 0.37).sin()).collect();
                for e in 0..MAX_EXPERTS {
                    if c.expert_mask & (1 << e) != 0 {
                        if n.expert_mask & (1 << e) != 0 {
                            xi.extend(&lat);
                        } else {
                            xi.extend(std::iter::repeat_n(9.0, c.latent_dims));
                        }
                    }
                }
                let o = c.forward_raw(&xi);
                assert!((o.0 - base.0).abs() < 1e-5, "{o:?} vs {base:?}");
            }
        }
        let p = n.perturb(&mut rng, 0.1);
        assert_ne!(p.weights, n.weights);
        let x2 = n.crossover(&p, &mut rng);
        assert_eq!(x2.layers(), n.layers());
        let other = PolicyNet::random(&mut rng, 0b11, 0, 0, vec![4], Act::Relu);
        assert_eq!(n.crossover(&other, &mut rng), n, "different topologies do not cross");
    }
}
