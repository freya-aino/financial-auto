//! Learned latent priors of spec 31 §6.2 families 2 and 4 (small CPU scale).
//!
//! * **Flow matching** (`hier_vae_flow`, Lipman et al. 2023): a velocity field `v(z, t)` on the
//!   top latent trained with the conditional flow-matching loss `‖v(z_t, t) − (z₁ − z₀)‖²`,
//!   `z_t = (1 − t) z₀ + t z₁`, `z₀ ~ N(0, I)`, `z₁` = the (detached) top latent. It models the
//!   aggregate posterior, so [`flow_log_density`] gives the exact (up to ODE discretisation)
//!   density used as a novelty score: `log p₁(z₁) = log N(z₀) − ∫₀¹ ∇·v(z_t, t) dt`, integrated
//!   backwards with Euler steps and the divergence by central differences (the top latent is a
//!   handful of dimensions).
//! * **Latent diffusion** (`latent_diffusion`, DDPM ε-prediction with a cosine schedule): a
//!   denoiser `ε(x_t, t, c)` over the full expert latent `x` (expert means at their expert-id
//!   slots, `K_max · expert_dim`), conditioned on the current latent `c`, trained on the latent
//!   of the window `horizon` steps ahead (detached; the autoencoder is trained by
//!   reconstruction). A share `cond_drop` of rows is trained with `c = 0`, so the same network
//!   is the unconditional prior ([`diffusion_nll`], a denoising-error score) and the future
//!   sampler ([`ddim_sample`], deterministic DDIM from seeded noise).

use candle_core::{D, DType, Device, Result, Tensor};
use fa_core::rng::Rng;
use fa_expert_space::nn::{Linear, ParamStore, Weights, linear_w};

/// Sinusoidal time-embedding width of the denoiser.
pub const T_EMB: usize = 8;

// ── flow matching ───────────────────────────────────────────────────────────

pub fn init_flow(ps: &mut ParamStore, dim: usize, hidden: usize) -> Result<()> {
    Linear::new(ps, "fm.l1", dim + 1, hidden, true)?;
    Linear::new(ps, "fm.l2", hidden, hidden, true)?;
    Linear::new(ps, "fm.l3", hidden, dim, true)?;
    Ok(())
}

/// `v(z, t)`: `z [B, d]`, `t [B, 1]`.
pub fn velocity(w: &dyn Weights, z: &Tensor, t: &Tensor) -> Result<Tensor> {
    let x = Tensor::cat(&[z, t], 1)?;
    let h = linear_w(w, "fm.l1", &x)?.gelu()?;
    let h = linear_w(w, "fm.l2", &h)?.gelu()?;
    linear_w(w, "fm.l3", &h)
}

/// Conditional flow-matching loss for targets `z1 [B, d]` (detached by the caller), with seeded
/// `z₀` and `t`.
pub fn flow_matching_loss(w: &dyn Weights, z1: &Tensor, seed: u64) -> Result<Tensor> {
    let (b, d) = z1.dims2()?;
    let dev = z1.device();
    let mut rng = Rng::seed(seed);
    let z0 = Tensor::from_vec((0..b * d).map(|_| rng.normal() as f32).collect::<Vec<_>>(), (b, d), dev)?;
    let t = Tensor::from_vec((0..b).map(|_| rng.uniform() as f32).collect::<Vec<_>>(), (b, 1), dev)?;
    let zt = (z0.broadcast_mul(&t.affine(-1.0, 1.0)?)? + z1.broadcast_mul(&t)?)?;
    let target = (z1 - &z0)?;
    (velocity(w, &zt, &t)? - target)?.sqr()?.mean_all()
}

/// `log p(z)` under the flow prior for each row of `z` (`steps` Euler steps from t = 1 to 0).
pub fn flow_log_density(w: &dyn Weights, z: &[Vec<f32>], steps: usize, dev: &Device) -> Result<Vec<f64>> {
    let n = z.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let d = z[0].len();
    let steps = steps.max(1);
    let dt = 1.0 / steps as f64;
    let eps = 1e-3f32;
    let mut cur: Vec<Vec<f32>> = z.to_vec();
    let mut div_int = vec![0.0f64; n];
    for s in 0..steps {
        let t = 1.0 - s as f64 * dt; // integrate z backwards from data (t=1) to noise (t=0)
        // batch: the point and ± eps along each axis
        let rows = n * (2 * d + 1);
        let mut flat = Vec::with_capacity(rows * d);
        for p in &cur {
            flat.extend_from_slice(p);
            for j in 0..d {
                for sign in [1.0f32, -1.0] {
                    let mut q = p.clone();
                    q[j] += sign * eps;
                    flat.extend(q);
                }
            }
        }
        let zt = Tensor::from_vec(flat, (rows, d), dev)?;
        let tt = Tensor::full(t as f32, (rows, 1), dev)?;
        let v = velocity(w, &zt, &tt)?.to_vec2::<f32>()?;
        for i in 0..n {
            let base = i * (2 * d + 1);
            let mut div = 0.0f64;
            for j in 0..d {
                let plus = v[base + 1 + 2 * j][j];
                let minus = v[base + 2 + 2 * j][j];
                div += f64::from(plus - minus) / (2.0 * f64::from(eps));
            }
            div_int[i] += div * dt;
            for j in 0..d {
                cur[i][j] -= (f64::from(v[base][j]) * dt) as f32;
            }
        }
    }
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    Ok(cur
        .iter()
        .zip(&div_int)
        .map(|(z0, div)| {
            let q: f64 = z0.iter().map(|x| f64::from(*x).powi(2)).sum();
            -0.5 * (q + d as f64 * ln2pi) - div
        })
        .collect())
}

/// Samples of the flow prior (forward Euler from seeded noise).
pub fn flow_sample(w: &dyn Weights, n: usize, d: usize, steps: usize, seed: u64, dev: &Device) -> Result<Tensor> {
    let mut rng = Rng::seed(seed);
    let mut z = Tensor::from_vec((0..n * d).map(|_| rng.normal() as f32).collect::<Vec<_>>(), (n, d), dev)?;
    let steps = steps.max(1);
    for s in 0..steps {
        let t = Tensor::full((s as f64 / steps as f64) as f32, (n, 1), dev)?;
        z = (&z + velocity(w, &z, &t)?.affine(1.0 / steps as f64, 0.0)?)?;
    }
    Ok(z)
}

// ── latent diffusion ────────────────────────────────────────────────────────

pub fn init_diffusion(ps: &mut ParamStore, dim: usize, hidden: usize) -> Result<()> {
    Linear::new(ps, "dm.l1", 2 * dim + T_EMB, hidden, true)?;
    Linear::new(ps, "dm.l2", hidden, hidden, true)?;
    Linear::new(ps, "dm.l3", hidden, dim, true)?;
    Ok(())
}

/// Cosine schedule `ᾱ(t)` for `t ∈ 1..=T` (Nichol & Dhariwal 2021), clipped away from 0.
pub fn alpha_bar(t: usize, steps: usize) -> f64 {
    let s = 0.008;
    let f = |x: f64| ((x / steps as f64 + s) / (1.0 + s) * std::f64::consts::FRAC_PI_2).cos().powi(2);
    (f(t as f64) / f(0.0)).clamp(1e-4, 1.0)
}

fn time_embedding(ts: &[usize], steps: usize, dev: &Device) -> Result<Tensor> {
    let mut v = Vec::with_capacity(ts.len() * T_EMB);
    for &t in ts {
        let x = t as f64 / steps as f64;
        for k in 0..T_EMB / 2 {
            let f = (k as f64 + 1.0) * std::f64::consts::PI;
            v.push((f * x).sin() as f32);
            v.push((f * x).cos() as f32);
        }
    }
    Tensor::from_vec(v, (ts.len(), T_EMB), dev)
}

/// `ε̂(x_t, t, c)`.
pub fn denoise(w: &dyn Weights, xt: &Tensor, c: &Tensor, ts: &[usize], steps: usize) -> Result<Tensor> {
    let emb = time_embedding(ts, steps, xt.device())?;
    let x = Tensor::cat(&[xt, c, &emb], 1)?;
    let h = linear_w(w, "dm.l1", &x)?.gelu()?;
    let h = linear_w(w, "dm.l2", &h)?.gelu()?;
    linear_w(w, "dm.l3", &h)
}

/// ε-prediction loss of targets `x0 [B, D]` given conditions `c [B, D]` (both detached by the
/// caller); a seeded share `cond_drop` of rows sees `c = 0`.
pub fn diffusion_loss(
    w: &dyn Weights,
    x0: &Tensor,
    c: &Tensor,
    steps: usize,
    cond_drop: f64,
    seed: u64,
) -> Result<Tensor> {
    let (b, d) = x0.dims2()?;
    let dev = x0.device();
    let mut rng = Rng::seed(seed);
    let ts: Vec<usize> = (0..b).map(|_| 1 + rng.below(steps.max(1))).collect();
    let keep: Vec<f32> = (0..b).map(|_| if rng.uniform() < cond_drop { 0.0 } else { 1.0 }).collect();
    let eps = Tensor::from_vec((0..b * d).map(|_| rng.normal() as f32).collect::<Vec<_>>(), (b, d), dev)?;
    let (sa, sb): (Vec<f32>, Vec<f32>) = ts
        .iter()
        .map(|&t| {
            let a = alpha_bar(t, steps);
            (a.sqrt() as f32, (1.0 - a).sqrt() as f32)
        })
        .unzip();
    let sa = Tensor::from_vec(sa, (b, 1), dev)?;
    let sb = Tensor::from_vec(sb, (b, 1), dev)?;
    let xt = (x0.broadcast_mul(&sa)? + eps.broadcast_mul(&sb)?)?;
    let c = c.broadcast_mul(&Tensor::from_vec(keep, (b, 1), dev)?)?;
    (denoise(w, &xt, &c, &ts, steps)? - eps)?.sqr()?.mean_all()
}

/// Unconditional denoising score of each row of `x [B, D]`: mean ε-prediction error over a
/// fixed grid of `t` with fixed noise (lower = more typical under the prior; a novelty score).
pub fn diffusion_nll(w: &dyn Weights, x: &Tensor, steps: usize, seed: u64) -> Result<Vec<f64>> {
    let (b, d) = x.dims2()?;
    let dev = x.device();
    let zeros = Tensor::zeros((b, d), DType::F32, dev)?;
    let grid: Vec<usize> = (1..=steps).step_by((steps / 8).max(1)).collect();
    let mut acc = vec![0.0f64; b];
    let mut rng = Rng::seed(seed);
    for &t in &grid {
        let eps = Tensor::from_vec((0..b * d).map(|_| rng.normal() as f32).collect::<Vec<_>>(), (b, d), dev)?;
        let a = alpha_bar(t, steps);
        let xt = (x.affine(a.sqrt(), 0.0)? + eps.affine((1.0 - a).sqrt(), 0.0)?)?;
        let err = (denoise(w, &xt, &zeros, &vec![t; b], steps)? - eps)?.sqr()?.mean(D::Minus1)?.to_vec1::<f32>()?;
        for (a, e) in acc.iter_mut().zip(err) {
            *a += f64::from(e) / grid.len() as f64;
        }
    }
    Ok(acc)
}

/// Deterministic DDIM (η = 0) samples `x₀` given conditions `c [B, D]` from seeded noise.
pub fn ddim_sample(w: &dyn Weights, c: &Tensor, steps: usize, sample_steps: usize, seed: u64) -> Result<Tensor> {
    let (b, d) = c.dims2()?;
    let dev = c.device();
    let mut rng = Rng::seed(seed);
    let mut x = Tensor::from_vec((0..b * d).map(|_| rng.normal() as f32).collect::<Vec<_>>(), (b, d), dev)?;
    let stride = (steps / sample_steps.max(1)).max(1);
    let mut ts: Vec<usize> = (1..=steps).rev().step_by(stride).collect();
    if ts.last() != Some(&1) {
        ts.push(1);
    }
    for (i, &t) in ts.iter().enumerate() {
        let a = alpha_bar(t, steps);
        let eps = denoise(w, &x, c, &vec![t; b], steps)?;
        let x0 = ((&x - eps.affine((1.0 - a).sqrt(), 0.0)?)?.affine(1.0 / a.sqrt(), 0.0)?).clamp(-1.5f32, 1.5f32)?;
        x = match ts.get(i + 1) {
            Some(&tn) => {
                let an = alpha_bar(tn, steps);
                (x0.affine(an.sqrt(), 0.0)? + eps.affine((1.0 - an).sqrt(), 0.0)?)?
            }
            None => x0,
        };
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use fa_expert_space::nn::{Adam, scalar};

    use super::*;

    #[test]
    fn flow_prior_learns_a_shifted_gaussian_and_scores_its_density() {
        let dev = Device::Cpu;
        let mut ps = ParamStore::new(4, &dev);
        init_flow(&mut ps, 2, 32).unwrap();
        let mut rng = Rng::seed(1);
        // data: N((1.5, -1), 0.3²)
        let data: Vec<f32> =
            (0..512).flat_map(|_| [1.5 + 0.3 * rng.normal() as f32, -1.0 + 0.3 * rng.normal() as f32]).collect();
        let z1 = Tensor::from_vec(data, (512, 2), &dev).unwrap();
        let mut opt = Adam::new(1e-2, 0.0);
        for s in 0..600 {
            let l = flow_matching_loss(&ps, &z1, 100 + s).unwrap();
            opt.step(&ps, &l).unwrap();
        }
        let lp = flow_log_density(&ps, &[vec![1.5, -1.0], vec![-1.5, 1.0]], 20, &dev).unwrap();
        assert!(lp[0] > lp[1] + 5.0, "mode is far more likely than the mirrored point: {lp:?}");
        // the true log density at the mode is ln(1/(2π·0.09)) ≈ 0.57
        assert!((lp[0] - 0.57).abs() < 1.0, "{lp:?}");
        let s = flow_sample(&ps, 256, 2, 20, 7, &dev).unwrap().to_vec2::<f32>().unwrap();
        let m0 = s.iter().map(|r| r[0]).sum::<f32>() / 256.0;
        assert!((m0 - 1.5).abs() < 0.2, "{m0}");
        assert!(scalar(&flow_matching_loss(&ps, &z1, 1).unwrap()).unwrap().is_finite());
    }

    #[test]
    fn diffusion_prior_samples_the_conditional_future() {
        let dev = Device::Cpu;
        let (d, steps) = (2, 30);
        let mut ps = ParamStore::new(2, &dev);
        init_diffusion(&mut ps, d, 64).unwrap();
        let mut rng = Rng::seed(3);
        // future = −current (+ small noise): a conditional map the sampler must reproduce
        let n = 512;
        let c: Vec<f32> = (0..n * d).map(|_| (rng.uniform() * 1.6 - 0.8) as f32).collect();
        let x: Vec<f32> = c.iter().map(|v| -v + 0.02 * rng.normal() as f32).collect();
        let c = Tensor::from_vec(c, (n, d), &dev).unwrap();
        let x0 = Tensor::from_vec(x, (n, d), &dev).unwrap();
        let mut opt = Adam::new(3e-3, 0.0);
        for s in 0..1500 {
            let l = diffusion_loss(&ps, &x0, &c, steps, 0.2, s).unwrap();
            opt.step(&ps, &l).unwrap();
        }
        let cq = Tensor::from_vec(vec![0.6f32, -0.4, -0.6, 0.4], (2, d), &dev).unwrap();
        let s = ddim_sample(&ps, &cq, steps, 15, 9).unwrap().to_vec2::<f32>().unwrap();
        assert!((s[0][0] + 0.6).abs() < 0.25 && (s[0][1] - 0.4).abs() < 0.25, "{s:?}");
        assert!((s[1][0] - 0.6).abs() < 0.25 && (s[1][1] + 0.4).abs() < 0.25, "{s:?}");
        let again = ddim_sample(&ps, &cq, steps, 15, 9).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(s, again, "seeded sampling is deterministic");
        // typical (in-distribution) latents score better than far outliers
        let probe = Tensor::from_vec(vec![0.3f32, -0.3, 6.0, 6.0], (2, d), &dev).unwrap();
        let nll = diffusion_nll(&ps, &probe, steps, 1).unwrap();
        assert!(nll[0] < nll[1], "{nll:?}");
        assert!(alpha_bar(1, steps) > 0.99 && alpha_bar(steps, steps) < 0.01);
    }
}
