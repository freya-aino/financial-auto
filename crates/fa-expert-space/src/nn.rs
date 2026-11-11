//! Minimal neural building blocks on candle (ADR 0003, pure Rust) shared by the learned expert
//! space ([`crate::moe`]) and the neural encoder families (`fa-encoders::neural`).
//!
//! Why not `candle_nn::VarMap`: its initialisers draw from an unseeded thread RNG, which breaks the
//! spec 31 §9.6 / spec 36 §6.6 reproducibility requirement (same data + genome + seed → same
//! weights). [`ParamStore`] initialises every parameter from a [`fa_core::rng::Rng`] seeded by
//! `(seed, parameter name)`, keeps parameters in name order, and round-trips through safetensors.
//! [`Adam`] keeps its moments as named tensors so a preempted job checkpoints and resumes the
//! optimiser bit for bit (spec 36 §9.5).

use std::collections::{BTreeMap, HashMap};

use candle_core::{D, DType, Device, Result, Tensor, Var};
use fa_core::rng::Rng;

/// Compute device from a config string: `cpu` (default), `cuda`, `cuda:<n>`. Without the `cuda`
/// cargo feature (the default build: no GPU on the dev host) every request resolves to the CPU;
/// the second value says why when the request was not honoured.
pub fn device_from_str(spec: &str) -> (Device, Option<String>) {
    let s = spec.trim().to_ascii_lowercase();
    if s.is_empty() || s == "cpu" {
        return (Device::Cpu, None);
    }
    if let Some(rest) = s.strip_prefix("cuda") {
        let ordinal: usize = rest.trim_start_matches(':').parse().unwrap_or(0);
        if !cfg!(feature = "cuda") {
            return (Device::Cpu, Some(format!("`{spec}` requested but built without the `cuda` feature; using cpu")));
        }
        return match Device::new_cuda(ordinal) {
            Ok(d) => (d, None),
            Err(e) => (Device::Cpu, Some(format!("cuda:{ordinal} unavailable ({e}); using cpu"))),
        };
    }
    (Device::Cpu, Some(format!("unknown device `{spec}`; using cpu")))
}

/// `cpu` / `cuda:<n>` label of a device (manifest `params.device`).
pub fn device_label(d: &Device) -> String {
    match d {
        Device::Cpu => "cpu".into(),
        Device::Cuda(_) => "cuda".into(),
        Device::Metal(_) => "metal".into(),
    }
}

/// How a parameter starts.
#[derive(Debug, Clone, Copy)]
pub enum Init {
    Zeros,
    Ones,
    Const(f64),
    /// `N(0, std²)`.
    Normal(f64),
    /// Kaiming-uniform style: `U(−b, b)` with `b = gain / sqrt(fan_in)`.
    Uniform {
        fan_in: usize,
        gain: f64,
    },
}

/// FNV-1a of a name (per-parameter seed).
fn fnv(name: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Named trainable parameters, deterministic initialisation, safetensors round trip.
#[derive(Debug, Clone)]
pub struct ParamStore {
    vars: BTreeMap<String, Var>,
    device: Device,
    seed: u64,
}

impl ParamStore {
    pub fn new(seed: u64, device: &Device) -> Self {
        Self { vars: BTreeMap::new(), device: device.clone(), seed }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The parameter `name`, created with `init` on first use. Creating a parameter is a pure
    /// function of `(seed, name, shape, init)`.
    pub fn get(&mut self, name: &str, shape: &[usize], init: Init) -> Result<Tensor> {
        if let Some(v) = self.vars.get(name) {
            if v.dims() != shape {
                candle_core::bail!("parameter {name}: shape {:?} != requested {shape:?}", v.dims());
            }
            return Ok(v.as_tensor().clone());
        }
        let n: usize = shape.iter().product();
        let mut rng = Rng::seed(self.seed ^ fnv(name));
        let data: Vec<f32> = match init {
            Init::Zeros => vec![0.0; n],
            Init::Ones => vec![1.0; n],
            Init::Const(c) => vec![c as f32; n],
            Init::Normal(std) => (0..n).map(|_| (rng.normal() * std) as f32).collect(),
            Init::Uniform { fan_in, gain } => {
                let b = gain / (fan_in.max(1) as f64).sqrt();
                (0..n).map(|_| ((rng.uniform() * 2.0 - 1.0) * b) as f32).collect()
            }
        };
        let v = Var::from_tensor(&Tensor::from_vec(data, shape, &self.device)?)?;
        let t = v.as_tensor().clone();
        self.vars.insert(name.to_owned(), v);
        Ok(t)
    }

    /// Existing parameter (no creation).
    pub fn tensor(&self, name: &str) -> Option<Tensor> {
        self.vars.get(name).map(|v| v.as_tensor().clone())
    }

    pub fn vars(&self) -> impl Iterator<Item = (&String, &Var)> {
        self.vars.iter()
    }

    pub fn len(&self) -> usize {
        self.vars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }

    /// Total scalar parameter count.
    pub fn n_params(&self) -> usize {
        self.vars.values().map(|v| v.elem_count()).sum()
    }

    /// Detached copies of every parameter (an EMA target, a frozen parent).
    pub fn snapshot(&self) -> Result<BTreeMap<String, Tensor>> {
        self.vars.iter().map(|(k, v)| Ok((k.clone(), v.as_tensor().detach().copy()?))).collect()
    }

    /// Overwrites parameters present in `values` (same shape); returns how many were set.
    /// Names in `values` that are not parameters of this store are ignored, so a parent with
    /// extra heads can warm-start a child.
    pub fn load(&mut self, values: &HashMap<String, Tensor>) -> Result<usize> {
        let mut n = 0;
        for (name, v) in &self.vars {
            if let Some(src) = values.get(name)
                && src.dims() == v.dims()
            {
                v.set(&src.to_device(&self.device)?.to_dtype(DType::F32)?)?;
                n += 1;
            }
        }
        Ok(n)
    }

    /// Inserts parameters that do not exist yet (loading a saved model before any forward pass).
    pub fn insert_all(&mut self, values: &HashMap<String, Tensor>, prefix: &str) -> Result<()> {
        for (name, t) in values {
            if let Some(stripped) = name.strip_prefix(prefix) {
                let t = t.to_device(&self.device)?.to_dtype(DType::F32)?;
                self.vars.insert(stripped.to_owned(), Var::from_tensor(&t)?);
            }
        }
        Ok(())
    }

    /// All parameters as `(prefix + name, tensor)`.
    pub fn named(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.vars.iter().map(|(k, v)| (format!("{prefix}{k}"), v.as_tensor().clone())).collect()
    }
}

/// safetensors bytes of named tensors plus string metadata.
pub fn to_safetensors(tensors: Vec<(String, Tensor)>, meta: HashMap<String, String>) -> Result<Vec<u8>> {
    let cpu: Vec<(String, Tensor)> =
        tensors.into_iter().map(|(k, t)| Ok((k, t.to_device(&Device::Cpu)?))).collect::<Result<_>>()?;
    safetensors::tensor::serialize(cpu, Some(meta)).map_err(|e| candle_core::Error::Msg(format!("safetensors: {e}")))
}

/// Tensors and metadata of safetensors bytes.
pub fn from_safetensors(bytes: &[u8], device: &Device) -> Result<(HashMap<String, Tensor>, HashMap<String, String>)> {
    let meta = safetensors::SafeTensors::read_metadata(bytes)
        .map_err(|e| candle_core::Error::Msg(format!("safetensors header: {e}")))?
        .1
        .metadata()
        .clone()
        .unwrap_or_default();
    Ok((candle_core::safetensors::load_buffer(bytes, device)?, meta))
}

/// `y = x W + b` over the last dim (`W: [in, out]`).
#[derive(Debug, Clone)]
pub struct Linear {
    pub w: Tensor,
    pub b: Option<Tensor>,
}

impl Linear {
    pub fn new(ps: &mut ParamStore, name: &str, d_in: usize, d_out: usize, bias: bool) -> Result<Self> {
        let w = ps.get(&format!("{name}.w"), &[d_in, d_out], Init::Uniform { fan_in: d_in, gain: 1.0 })?;
        let b = if bias { Some(ps.get(&format!("{name}.b"), &[d_out], Init::Zeros)?) } else { None };
        Ok(Self { w, b })
    }

    /// Same weights rebound from a detached map (frozen copy).
    pub fn frozen(&self, values: &BTreeMap<String, Tensor>, name: &str) -> Result<Self> {
        let get =
            |k: &str| values.get(k).cloned().ok_or_else(|| candle_core::Error::Msg(format!("frozen copy lacks {k}")));
        Ok(Self {
            w: get(&format!("{name}.w"))?,
            b: if self.b.is_some() { Some(get(&format!("{name}.b"))?) } else { None },
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.broadcast_matmul(&self.w)?;
        match &self.b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

/// Adam(W) with explicit, serialisable state.
#[derive(Debug, Clone)]
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    /// Global gradient-norm clip (0 = off).
    pub clip_norm: f64,
    pub t: u64,
    m: BTreeMap<String, Tensor>,
    v: BTreeMap<String, Tensor>,
}

impl Adam {
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            clip_norm: 1.0,
            t: 0,
            m: BTreeMap::new(),
            v: BTreeMap::new(),
        }
    }

    /// One update of every parameter of `ps` from the gradients of `loss`. Returns the global
    /// gradient norm before clipping.
    pub fn step(&mut self, ps: &ParamStore, loss: &Tensor) -> Result<f64> {
        let grads = loss.backward()?;
        let mut sq = 0.0f64;
        let mut present = Vec::new();
        for (name, var) in ps.vars() {
            if let Some(g) = grads.get(var.as_tensor()) {
                sq += f64::from(g.sqr()?.sum_all()?.to_scalar::<f32>()?);
                present.push((name.clone(), var.clone(), g.clone()));
            }
        }
        let norm = sq.sqrt();
        let scale = if self.clip_norm > 0.0 && norm > self.clip_norm { self.clip_norm / norm } else { 1.0 };
        self.t += 1;
        let (b1, b2) = (self.beta1, self.beta2);
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        for (name, var, g) in present {
            let g = if scale < 1.0 { g.affine(scale, 0.0)? } else { g };
            let m = match self.m.get(&name) {
                Some(m) => (m.affine(b1, 0.0)? + g.affine(1.0 - b1, 0.0)?)?,
                None => g.affine(1.0 - b1, 0.0)?,
            };
            let v = match self.v.get(&name) {
                Some(v) => (v.affine(b2, 0.0)? + g.sqr()?.affine(1.0 - b2, 0.0)?)?,
                None => g.sqr()?.affine(1.0 - b2, 0.0)?,
            };
            let mhat = m.affine(1.0 / bc1, 0.0)?;
            let vhat = v.affine(1.0 / bc2, 0.0)?;
            let upd = (mhat / (vhat.sqrt()? + self.eps)?)?;
            let w = var.as_tensor();
            let decayed =
                if self.weight_decay > 0.0 { w.affine(1.0 - self.lr * self.weight_decay, 0.0)? } else { w.clone() };
            var.set(&(decayed - upd.affine(self.lr, 0.0)?)?)?;
            self.m.insert(name.clone(), m.detach());
            self.v.insert(name, v.detach());
        }
        Ok(norm)
    }

    /// Moments as named tensors (`adam.m.<param>`, `adam.v.<param>`) for a checkpoint.
    pub fn named_state(&self) -> Vec<(String, Tensor)> {
        self.m
            .iter()
            .map(|(k, t)| (format!("adam.m.{k}"), t.clone()))
            .chain(self.v.iter().map(|(k, t)| (format!("adam.v.{k}"), t.clone())))
            .collect()
    }

    /// Restores moments from a checkpoint map (keys as written by [`Self::named_state`]).
    pub fn restore(&mut self, t: u64, values: &HashMap<String, Tensor>) {
        self.t = t;
        for (k, v) in values {
            if let Some(p) = k.strip_prefix("adam.m.") {
                self.m.insert(p.to_owned(), v.clone());
            } else if let Some(p) = k.strip_prefix("adam.v.") {
                self.v.insert(p.to_owned(), v.clone());
            }
        }
    }
}

/// Mean squared error over all elements.
pub fn mse(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    (a - b)?.sqr()?.mean_all()
}

/// Softmax over the last dimension.
pub fn softmax_last(x: &Tensor) -> Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?;
    let e = x.broadcast_sub(&max)?.exp()?;
    e.broadcast_div(&e.sum_keepdim(D::Minus1)?)
}

/// Scalar value of a 0-d / 1-element tensor.
pub fn scalar(t: &Tensor) -> Result<f64> {
    Ok(f64::from(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?.first().copied().unwrap_or(f32::NAN)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_is_deterministic_and_roundtrips() {
        let dev = Device::Cpu;
        let mut a = ParamStore::new(7, &dev);
        let mut b = ParamStore::new(7, &dev);
        let la = Linear::new(&mut a, "l", 4, 3, true).unwrap();
        let lb = Linear::new(&mut b, "l", 4, 3, true).unwrap();
        assert_eq!(la.w.to_vec2::<f32>().unwrap(), lb.w.to_vec2::<f32>().unwrap());
        let mut c = ParamStore::new(8, &dev);
        let lc = Linear::new(&mut c, "l", 4, 3, true).unwrap();
        assert_ne!(la.w.to_vec2::<f32>().unwrap(), lc.w.to_vec2::<f32>().unwrap());
        let bytes = to_safetensors(a.named("p."), HashMap::from([("k".into(), "v".into())])).unwrap();
        let (vals, meta) = from_safetensors(&bytes, &dev).unwrap();
        assert_eq!(meta["k"], "v");
        let mut d = ParamStore::new(0, &dev);
        d.insert_all(&vals, "p.").unwrap();
        assert_eq!(d.tensor("l.w").unwrap().to_vec2::<f32>().unwrap(), la.w.to_vec2::<f32>().unwrap());
        assert_eq!(c.load(&d.snapshot().unwrap().into_iter().collect()).unwrap(), 2);
        assert_eq!(c.tensor("l.w").unwrap().to_vec2::<f32>().unwrap(), la.w.to_vec2::<f32>().unwrap());
        assert_eq!(device_from_str("cpu").1, None);
        if !cfg!(feature = "cuda") {
            assert!(matches!(device_from_str("cuda:0"), (Device::Cpu, Some(_))));
        }
    }

    #[test]
    fn adam_fits_a_line_and_state_resumes_exactly() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec((0..32).map(|i| i as f32 / 16.0 - 1.0).collect::<Vec<_>>(), (32, 1), &dev).unwrap();
        let y = x.affine(3.0, 0.5).unwrap();
        let run = |steps: usize, split: Option<usize>| {
            let mut ps = ParamStore::new(1, &dev);
            let lin = Linear::new(&mut ps, "l", 1, 1, true).unwrap();
            let mut opt = Adam::new(0.05, 0.0);
            for i in 0..steps {
                if Some(i) == split {
                    // checkpoint → fresh optimiser restored from state
                    let st: HashMap<String, Tensor> = opt.named_state().into_iter().collect();
                    let t = opt.t;
                    opt = Adam::new(0.05, 0.0);
                    opt.restore(t, &st);
                }
                let loss = mse(&lin.forward(&x).unwrap(), &y).unwrap();
                opt.step(&ps, &loss).unwrap();
            }
            (
                scalar(&mse(&lin.forward(&x).unwrap(), &y).unwrap()).unwrap(),
                ps.tensor("l.w").unwrap().to_vec2::<f32>().unwrap(),
            )
        };
        let (loss, w) = run(400, None);
        assert!(loss < 1e-3, "{loss}");
        let (_, w2) = run(400, Some(150));
        assert_eq!(w, w2, "restored optimiser state continues bit for bit");
    }
}
