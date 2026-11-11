//! Low-rank adapters (LoRA, Hu et al. 2021) for the online adaptation of a frozen snapshot
//! (spec 36 §6.5): for a target weight `W ∈ R^{in×out}` the adapted weight is
//! `W' = W + (α / r) · A B` with `A ∈ R^{in×r}` (seeded `N(0, 1/in)`) and `B ∈ R^{r×out}`
//! (zeros), so a fresh adapter is the identity. Only `A`, `B` train; `W` stays the snapshot's.
//!
//! [`LoraView`] is a [`Weights`] source that serves `W'` for adapted names and the frozen base
//! for everything else, so any forward pass written against [`Weights`] runs adapted without
//! code changes. [`LoraSet::merge_into`] folds the adapter into a parameter store for serving
//! (no per-step overhead), [`LoraSet::delta_norm`] is the `‖ΔW‖_F / ‖W‖_F` drift guard.

use std::collections::{BTreeMap, HashMap};

use candle_core::{Device, Result, Tensor};

use crate::nn::{Init, ParamStore, Weights, from_safetensors, to_safetensors};

/// safetensors metadata tag of an adapter.
pub const LORA_FORMAT: &str = "fa-lora-v1";

/// Trainable low-rank factors of a set of target weights.
#[derive(Debug, Clone)]
pub struct LoraSet {
    pub rank: usize,
    pub alpha: f64,
    /// Adapted weight names (e.g. `bb.out.w`, `moe.proj0.w`), sorted.
    pub targets: Vec<String>,
    /// `<target>.lora_a` / `<target>.lora_b`.
    pub ps: ParamStore,
}

impl LoraSet {
    /// Fresh (identity) adapter for `targets` whose shapes come from `base`.
    pub fn new(
        base: &dyn Weights,
        targets: &[String],
        rank: usize,
        alpha: f64,
        seed: u64,
        dev: &Device,
    ) -> Result<Self> {
        let mut ps = ParamStore::new(seed, dev);
        let mut names: Vec<String> = targets.to_vec();
        names.sort();
        names.dedup();
        let r = rank.max(1);
        for t in &names {
            let (d_in, d_out) = base.w(t)?.dims2()?;
            ps.get(&format!("{t}.lora_a"), &[d_in, r], Init::Normal(1.0 / (d_in.max(1) as f64).sqrt()))?;
            ps.get(&format!("{t}.lora_b"), &[r, d_out], Init::Zeros)?;
        }
        Ok(Self { rank: r, alpha, targets: names, ps })
    }

    pub fn scale(&self) -> f64 {
        self.alpha / self.rank as f64
    }

    /// `ΔW = (α/r) A B` of a target (differentiable w.r.t. `A`, `B`).
    pub fn delta(&self, target: &str) -> Result<Option<Tensor>> {
        match (self.ps.tensor(&format!("{target}.lora_a")), self.ps.tensor(&format!("{target}.lora_b"))) {
            (Some(a), Some(b)) => Ok(Some(a.matmul(&b)?.affine(self.scale(), 0.0)?)),
            _ => Ok(None),
        }
    }

    /// `Σ ‖ΔW‖_F / Σ ‖W‖_F` over the targets (spec 36 §6.5 "norm of ΔW" guard).
    pub fn delta_norm(&self, base: &dyn Weights) -> Result<f64> {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for t in &self.targets {
            if let Some(d) = self.delta(t)? {
                num += f64::from(d.sqr()?.sum_all()?.to_scalar::<f32>()?).sqrt();
                den += f64::from(base.w(t)?.sqr()?.sum_all()?.to_scalar::<f32>()?).sqrt();
            }
        }
        Ok(if den > 0.0 { num / den } else { 0.0 })
    }

    /// `Σ ‖A‖² + ‖B‖²`, the L2 anchor toward the identity adapter.
    pub fn l2(&self) -> Result<Tensor> {
        let mut acc: Option<Tensor> = None;
        for (_, v) in self.ps.vars() {
            let s = v.as_tensor().sqr()?.sum_all()?;
            acc = Some(match acc {
                Some(a) => (a + s)?,
                None => s,
            });
        }
        match acc {
            Some(a) => Ok(a),
            None => Tensor::zeros((), candle_core::DType::F32, self.ps.device()),
        }
    }

    /// Folds `ΔW` into `store` (`W ← W + ΔW`); returns how many weights changed.
    pub fn merge_into(&self, store: &mut ParamStore) -> Result<usize> {
        let mut vals: HashMap<String, Tensor> = HashMap::new();
        for t in &self.targets {
            if let (Some(w), Some(d)) = (store.tensor(t), self.delta(t)?) {
                vals.insert(t.clone(), (w + d.detach())?);
            }
        }
        store.load(&vals)
    }

    /// safetensors bytes (`lora.<target>.lora_a|b`, metadata `format`, `rank`, `alpha`, `targets`).
    pub fn to_bytes(&self, extra: HashMap<String, String>) -> Result<Vec<u8>> {
        let mut meta = extra;
        meta.insert("format".into(), LORA_FORMAT.into());
        meta.insert("rank".into(), self.rank.to_string());
        meta.insert("alpha".into(), format!("{:e}", self.alpha));
        meta.insert("targets".into(), serde_json::to_string(&self.targets).unwrap_or_default());
        to_safetensors(self.ps.named("lora."), meta)
    }

    /// Reads [`Self::to_bytes`]; returns the set and its metadata.
    pub fn from_bytes(bytes: &[u8], dev: &Device) -> Result<(Self, HashMap<String, String>)> {
        let (vals, meta) = from_safetensors(bytes, dev)?;
        if meta.get("format").map(String::as_str) != Some(LORA_FORMAT) {
            candle_core::bail!("not an {LORA_FORMAT} adapter");
        }
        let rank = meta.get("rank").and_then(|s| s.parse().ok()).unwrap_or(1);
        let alpha = meta.get("alpha").and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let targets: Vec<String> = meta.get("targets").and_then(|s| serde_json::from_str(s).ok()).unwrap_or_default();
        let mut ps = ParamStore::new(0, dev);
        ps.insert_all(&vals, "lora.")?;
        Ok((Self { rank, alpha, targets, ps }, meta))
    }
}

/// Adapted weights: `base + ΔW` for the adapter's targets, the (detached) base otherwise.
pub struct LoraView<'a> {
    pub base: &'a dyn Weights,
    pub lora: &'a LoraSet,
}

impl Weights for LoraView<'_> {
    fn w(&self, name: &str) -> Result<Tensor> {
        let b = self.base.w(name)?.detach();
        match self.lora.delta(name)? {
            Some(d) => b + d,
            None => Ok(b),
        }
    }
}

/// Detached copies of the adapted weights (for an EMA target or a frozen comparison).
pub fn adapted_snapshot(base: &ParamStore, lora: &LoraSet) -> Result<BTreeMap<String, Tensor>> {
    let mut out = base.snapshot()?;
    for t in &lora.targets {
        if let (Some(w), Some(d)) = (out.get(t).cloned(), lora.delta(t)?) {
            out.insert(t.clone(), (w + d)?.detach());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Adam, Linear, linear_w, mse, scalar};

    #[test]
    fn fresh_adapter_is_identity_and_learns_a_low_rank_shift() {
        let dev = Device::Cpu;
        let mut base = ParamStore::new(1, &dev);
        Linear::new(&mut base, "l", 6, 4, true).unwrap();
        let x = Tensor::from_vec(
            (0..64 * 6).map(|i| ((i * 37 % 17) as f32 / 8.0) - 1.0).collect::<Vec<_>>(),
            (64, 6),
            &dev,
        )
        .unwrap();
        let lora = LoraSet::new(&base, &["l.w".to_owned()], 2, 2.0, 3, &dev).unwrap();
        let view = LoraView { base: &base, lora: &lora };
        let y0 = linear_w(&base, "l", &x).unwrap();
        assert_eq!(linear_w(&view, "l", &x).unwrap().to_vec2::<f32>().unwrap(), y0.to_vec2::<f32>().unwrap());
        assert_eq!(lora.delta_norm(&base).unwrap(), 0.0);
        // target: base output + a rank-1 shift
        let u = Tensor::from_vec(vec![1f32, -1.0, 0.5, 0.0, 0.0, 2.0], (6, 1), &dev).unwrap();
        let v = Tensor::from_vec(vec![0.5f32, 0.0, -0.5, 1.0], (1, 4), &dev).unwrap();
        let target = (&y0 + x.matmul(&u.matmul(&v).unwrap()).unwrap()).unwrap().detach();
        let before = scalar(&mse(&linear_w(&view, "l", &x).unwrap(), &target).unwrap()).unwrap();
        let mut opt = Adam::new(0.05, 0.0);
        for _ in 0..300 {
            let view = LoraView { base: &base, lora: &lora };
            let loss = mse(&linear_w(&view, "l", &x).unwrap(), &target).unwrap();
            opt.step(&lora.ps, &loss).unwrap();
        }
        let view = LoraView { base: &base, lora: &lora };
        let after = scalar(&mse(&linear_w(&view, "l", &x).unwrap(), &target).unwrap()).unwrap();
        assert!(after < before * 0.05, "{before} → {after}");
        assert!(lora.delta_norm(&base).unwrap() > 0.0);
        // base untouched by training; merge reproduces the view; bytes round trip
        assert_eq!(linear_w(&base, "l", &x).unwrap().to_vec2::<f32>().unwrap(), y0.to_vec2::<f32>().unwrap());
        let (back, meta) = LoraSet::from_bytes(&lora.to_bytes(HashMap::new()).unwrap(), &dev).unwrap();
        assert_eq!(meta["format"], LORA_FORMAT);
        assert_eq!((back.rank, back.alpha, back.targets.clone()), (2, 2.0, vec!["l.w".to_owned()]));
        // a cloned ParamStore shares Var storage: merge into an independent copy
        let mut fresh = ParamStore::new(0, &dev);
        fresh.insert_all(&base.snapshot().unwrap().into_iter().collect(), "").unwrap();
        assert_eq!(back.merge_into(&mut fresh).unwrap(), 1);
        let ym = linear_w(&fresh, "l", &x).unwrap().to_vec2::<f32>().unwrap();
        let yv = linear_w(&view, "l", &x).unwrap().to_vec2::<f32>().unwrap();
        for (a, b) in ym.iter().flatten().zip(yv.iter().flatten()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
