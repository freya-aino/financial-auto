//! Network bodies and family losses on candle (spec 34 §4.1 `forward` / `loss`).
//!
//! The body is an MLP (`depth` GELU layers of `width`) or, with `gru=1`, a GRU over the return
//! window whose initial state is `tanh(W·[extra | latent | flag])` (the "latent → initial state of
//! an RNN prior" of §6.1) followed by a `depth − 1` layer head. Heads per family:
//! monotone quantiles (cumulative softplus increments), L2-normalised embeddings (retrieval,
//! text), per-column log-softmax rasters (visual).

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{GRU, Linear, Module, VarBuilder, VarMap, rnn::GRUState, rnn::RNN};

use crate::{genome::NeuralGenes, neural::data::Layout};

enum Cell {
    Mlp,
    Gru { gru: GRU, h0: Option<Linear> },
}

/// Body + output layer of one decoder.
pub struct Net {
    cell: Cell,
    hidden: Vec<Linear>,
    out: Linear,
    lay: Layout,
    width: usize,
}

impl Net {
    /// Builds the network from `vb` (a `VarMap`-backed builder while training, a frozen
    /// tensor map when serving / for feedback). Names live under `net.`.
    pub fn new(vb: VarBuilder, lay: Layout, g: &NeuralGenes) -> Result<Self> {
        let vb = vb.pp("net");
        let w = g.width;
        let use_gru = g.gru && lay.seq > 1;
        let (cell, mut in_dim, n_hidden) = if use_gru {
            let gru = candle_nn::rnn::gru(1, w, Default::default(), vb.pp("gru"))?;
            let rest = lay.input() - lay.seq;
            let h0 = if rest > 0 { Some(candle_nn::linear(rest, w, vb.pp("h0"))?) } else { None };
            (Cell::Gru { gru, h0 }, w, g.depth.saturating_sub(1))
        } else {
            (Cell::Mlp, lay.input(), g.depth)
        };
        let mut hidden = Vec::with_capacity(n_hidden);
        for i in 0..n_hidden {
            hidden.push(candle_nn::linear(in_dim, w, vb.pp(format!("l{i}")))?);
            in_dim = w;
        }
        let out = candle_nn::linear(in_dim, lay.out, vb.pp("out"))?;
        Ok(Self { cell, hidden, out, lay, width: w })
    }

    pub fn layout(&self) -> Layout {
        self.lay
    }

    /// `[B, input] → [B, out]` (raw, before the family head).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = match &self.cell {
            Cell::Mlp => x.clone(),
            Cell::Gru { gru, h0 } => {
                let b = x.dim(0)?;
                let seq = x.narrow(1, 0, self.lay.seq)?.unsqueeze(2)?.contiguous()?;
                let init = match h0 {
                    Some(l) => {
                        let rest = x.narrow(1, self.lay.seq, x.dim(1)? - self.lay.seq)?;
                        l.forward(&rest)?.tanh()?
                    }
                    None => Tensor::zeros((b, self.width), x.dtype(), x.device())?,
                };
                let states = gru.seq_init(&seq, &GRUState { h: init })?;
                states.last().map(|s| s.h.clone()).ok_or_else(|| candle_core::Error::Msg("empty sequence".into()))?
            }
        };
        for l in &self.hidden {
            h = l.forward(&h)?.gelu()?;
        }
        self.out.forward(&h)
    }

    /// Trainable parameters (rough inference FLOPs = 2 × params, × seq for the GRU part).
    pub fn flops(&self) -> f64 {
        let lin = |l: &Linear| l.weight().elem_count() as f64 + l.bias().map_or(0.0, |b| b.elem_count() as f64);
        let mut f: f64 = self.hidden.iter().map(lin).sum::<f64>() + lin(&self.out);
        if let Cell::Gru { h0, .. } = &self.cell {
            let w = self.width as f64;
            f += self.lay.seq as f64 * (3.0 * w * (w + 1.0) + 6.0 * w);
            f += h0.as_ref().map_or(0.0, lin);
        }
        2.0 * f
    }
}

/// `softplus(x) = relu(x) + ln(1 + e^{−|x|})` (stable).
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    x.relu()? + x.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?
}

/// Monotone quantiles from raw outputs `[B, K]`: `q_0 = o_0`, `q_i = q_{i−1} + softplus(o_i)`.
pub fn monotone(o: &Tensor) -> Result<Tensor> {
    let k = o.dim(1)?;
    if k == 1 {
        return Ok(o.clone());
    }
    let first = o.narrow(1, 0, 1)?;
    let rest = softplus(&o.narrow(1, 1, k - 1)?)?;
    Tensor::cat(&[first, rest], 1)?.cumsum(1)
}

/// Row-wise L2 normalisation.
pub fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let n = x.sqr()?.sum_keepdim(1)?.affine(1.0, 1e-12)?.sqrt()?;
    x.broadcast_div(&n)
}

/// Pinball loss of quantiles `q [B, K]` at `levels [K]` against `y [B, 1]` (mean over B·K).
pub fn pinball_loss(q: &Tensor, y: &Tensor, levels: &[f64]) -> Result<Tensor> {
    let lv: Vec<f32> = levels.iter().map(|l| *l as f32).collect();
    let l = Tensor::from_vec(lv, (1, levels.len()), q.device())?;
    let d = y.broadcast_sub(q)?; // y − q
    let pos = d.relu()?.broadcast_mul(&l)?;
    let neg = d.neg()?.relu()?.broadcast_mul(&l.affine(-1.0, 1.0)?)?;
    (pos + neg)?.mean_all()
}

/// Per-column log-softmax of raster logits `[B, res·res]` → `[B, res, res]` (col, row).
pub fn raster_logp(o: &Tensor, res: usize) -> Result<Tensor> {
    let b = o.dim(0)?;
    candle_nn::ops::log_softmax(&o.reshape((b, res, res))?, D::Minus1)
}

/// Mean per-column cross-entropy `−Σ_r p·log q` of target columns `[B, res·res]`.
pub fn raster_ce(logp: &Tensor, target: &Tensor, res: usize) -> Result<Tensor> {
    let b = logp.dim(0)?;
    let t = target.reshape((b, res, res))?;
    (t * logp)?.sum_all()?.affine(-1.0 / (b * res) as f64, 0.0)
}

/// Mean `1 − cos` between predicted and target embeddings `[B, E]`.
pub fn cosine_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    let p = l2_normalize(pred)?;
    let t = l2_normalize(target)?;
    (p * t)?.sum(1)?.mean_all()?.affine(-1.0, 1.0)
}

/// Soft-kNN CRPS (learned retrieval). `q [B, E]`, `c [M, E]` normalised embeddings, `yq [B, 1]`
/// query outcomes, `zc [1, M]` candidate outcomes, `bias [B, M]` = `−Δt/τ` for valid pairs and a
/// large negative number for pairs violating the leakage gap. Weights `w = softmax(cos/temp + bias)`;
/// loss = mean over queries of the weighted-ensemble CRPS `Σ w_j|z_j − y| − ½ Σ_jk w_j w_k |z_j − z_k|`.
pub fn soft_knn_crps(q: &Tensor, c: &Tensor, yq: &Tensor, zc: &Tensor, bias: &Tensor, temp: f64) -> Result<Tensor> {
    let sim = q.matmul(&c.t()?)?.affine(1.0 / temp, 0.0)?;
    let w = candle_nn::ops::softmax(&(sim + bias)?, D::Minus1)?; // [B, M]
    let a = zc.broadcast_sub(yq)?.abs()?; // [B, M]
    let term1 = (&w * a)?.sum(1)?; // [B]
    let zt = zc.t()?; // [M, 1]
    let dmat = zt.broadcast_sub(zc)?.abs()?; // [M, M]
    let wd = w.matmul(&dmat)?; // [B, M]
    let term2 = (wd * &w)?.sum(1)?; // [B]
    (term1 - term2.affine(0.5, 0.0)?)?.mean_all()
}

/// Deterministic (seeded) re-initialisation of every variable: weights `U(−1/√fan_in, 1/√fan_in)`,
/// biases 0. candle's own initialisers draw from the thread RNG, which would make two trainings
/// of the same genome differ.
pub fn seeded_init(varmap: &VarMap, seed: u64) -> Result<()> {
    let data = varmap.data().lock().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    let mut rng = fa_core::rng::Rng::seed(seed);
    for name in names {
        let var = &data[name];
        let dims = var.dims().to_vec();
        let n: usize = dims.iter().product();
        let vals: Vec<f32> = if dims.len() >= 2 {
            let bound = 1.0 / (dims[1] as f64).sqrt();
            (0..n).map(|_| ((rng.uniform() * 2.0 - 1.0) * bound) as f32).collect()
        } else {
            vec![0.0; n]
        };
        var.set(&Tensor::from_vec(vals, dims, var.device())?)?;
    }
    Ok(())
}

/// The compute device: CUDA 0 when built with `cuda` and a device is present, else the CPU.
pub fn device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::cuda_if_available(0) {
            return d;
        }
    }
    Device::Cpu
}

/// `[rows, width]` f32 tensor from row vectors.
pub fn rows_tensor(rows: &[&[f32]], width: usize, dev: &Device) -> Result<Tensor> {
    let mut flat = Vec::with_capacity(rows.len() * width);
    for r in rows {
        debug_assert_eq!(r.len(), width);
        flat.extend_from_slice(r);
    }
    Tensor::from_vec(flat, (rows.len(), width), dev)
}

pub const DTYPE: DType = DType::F32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotone_quantiles_and_pinball() {
        let dev = Device::Cpu;
        let o = Tensor::new(&[[0.0f32, -3.0, 2.0], [1.0, 0.0, 0.0]], &dev).unwrap();
        let q = monotone(&o).unwrap().to_vec2::<f32>().unwrap();
        for row in &q {
            assert!(row[0] < row[1] && row[1] < row[2], "{row:?}");
        }
        let qt = Tensor::new(&[[0.0f32, 0.0]], &dev).unwrap();
        let y = Tensor::new(&[[1.0f32]], &dev).unwrap();
        let l = pinball_loss(&qt, &y, &[0.25, 0.75]).unwrap().to_scalar::<f32>().unwrap();
        assert!((l - 0.5).abs() < 1e-6, "{l}");
    }

    #[test]
    fn soft_knn_prefers_matching_outcomes() {
        let dev = Device::Cpu;
        let q = Tensor::new(&[[1.0f32, 0.0]], &dev).unwrap();
        let c = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &dev).unwrap();
        let y = Tensor::new(&[[1.0f32]], &dev).unwrap();
        let bias = Tensor::zeros((1, 2), DType::F32, &dev).unwrap();
        let good = Tensor::new(&[[1.0f32, -1.0]], &dev).unwrap();
        let bad = Tensor::new(&[[-1.0f32, 1.0]], &dev).unwrap();
        let lg = soft_knn_crps(&q, &c, &y, &good, &bias, 0.1).unwrap().to_scalar::<f32>().unwrap();
        let lb = soft_knn_crps(&q, &c, &y, &bad, &bias, 0.1).unwrap().to_scalar::<f32>().unwrap();
        assert!(lg < lb, "{lg} vs {lb}");
        // a masked candidate gets no weight
        let mask = Tensor::new(&[[-1e9f32, 0.0]], &dev).unwrap();
        let lm = soft_knn_crps(&q, &c, &y, &good, &mask, 0.1).unwrap().to_scalar::<f32>().unwrap();
        assert!((lm - 2.0).abs() < 1e-4, "{lm}");
    }

    #[test]
    fn raster_ce_is_minimal_at_target() {
        let dev = Device::Cpu;
        let t = Tensor::new(&[[1.0f32, 0.0, 0.0, 1.0]], &dev).unwrap();
        let near = raster_logp(&Tensor::new(&[[5.0f32, -5.0, -5.0, 5.0]], &dev).unwrap(), 2).unwrap();
        let far = raster_logp(&Tensor::new(&[[-5.0f32, 5.0, 5.0, -5.0]], &dev).unwrap(), 2).unwrap();
        let a = raster_ce(&near, &t, 2).unwrap().to_scalar::<f32>().unwrap();
        let b = raster_ce(&far, &t, 2).unwrap().to_scalar::<f32>().unwrap();
        assert!(a < 0.01 && b > 5.0, "{a} {b}");
    }
}
