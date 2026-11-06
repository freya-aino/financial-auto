//! Small deterministic RNG (xoshiro256**) plus samplers used by MCMC and evolution.
//! Seeded explicitly so every job is reproducible (spec 36).

#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn seed(seed: u64) -> Self {
        let mut sm = seed;
        let mut next = || {
            sm = sm.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self { s: [next(), next(), next(), next()] }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in [0, n).
    pub fn below(&mut self, n: usize) -> usize {
        (self.uniform() * n as f64) as usize % n.max(1)
    }

    /// Standard normal (Box–Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Gamma(shape, 1) via Marsaglia–Tsang.
    pub fn gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            let u = self.uniform().max(f64::MIN_POSITIVE);
            return self.gamma(shape + 1.0) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.uniform();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }

    /// Dirichlet(alpha) sample.
    pub fn dirichlet(&mut self, alpha: &[f64]) -> Vec<f64> {
        let g: Vec<f64> = alpha.iter().map(|&a| self.gamma(a)).collect();
        let s: f64 = g.iter().sum();
        g.into_iter().map(|x| x / s).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        let mut a = Rng::seed(7);
        let mut b = Rng::seed(7);
        for _ in 0..100 {
            let x = a.uniform();
            assert_eq!(x, b.uniform());
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn gamma_mean_close_to_shape() {
        let mut r = Rng::seed(1);
        let n = 20_000;
        let m: f64 = (0..n).map(|_| r.gamma(3.0)).sum::<f64>() / n as f64;
        assert!((m - 3.0).abs() < 0.1, "mean {m}");
    }
}
