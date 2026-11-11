//! Density region discovery (spec 30, `fa_patterns::DiscoveryMethod::Density`) declared by the
//! continuous-prior families 2 (`hier_vae_flow`) and 4 (`latent_diffusion`): regions are the
//! **modes of a Gaussian kernel density** of an expert's latents, found by mean shift.
//!
//! Deterministic and order-free: every point (or an evenly strided subset of at most
//! `max_seeds`) climbs to its mode with fixed-bandwidth mean shift; modes closer than
//! `bandwidth / 2` merge (the denser one wins); modes holding fewer than `min_share` of the
//! points are dropped; centroids are sorted lexicographically, so a shuffled copy of the points
//! gives the same regions. Points are assigned to the nearest mode within `3 · bandwidth`.

use fa_patterns::{DiscoveryMethod, RegionDiscovery};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DensityModes {
    /// Kernel bandwidth (latent units; the trainer passes the region radius).
    pub bandwidth: f32,
    pub max_regions: usize,
    /// Share of the points a mode must attract to become a region.
    pub min_share: f64,
    /// Seeds climbed per fit (strided over the sorted points).
    pub max_seeds: usize,
}

impl DensityModes {
    pub fn new(bandwidth: f32, max_regions: usize) -> Self {
        Self { bandwidth, max_regions, min_share: 0.01, max_seeds: 512 }
    }

    fn shift(&self, points: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
        let inv = 1.0 / (2.0 * f64::from(self.bandwidth).powi(2));
        let mut num = vec![0.0f64; x.len()];
        let mut den = 0.0f64;
        for p in points {
            let d2: f64 = p.iter().zip(x).map(|(a, b)| f64::from(a - b).powi(2)).sum();
            let w = (-d2 * inv).exp();
            if w < 1e-12 {
                continue;
            }
            den += w;
            for (n, v) in num.iter_mut().zip(p) {
                *n += w * f64::from(*v);
            }
        }
        if den <= 0.0 { x.to_vec() } else { num.iter().map(|v| (v / den) as f32).collect() }
    }

    fn climb(&self, points: &[Vec<f32>], start: &[f32]) -> Vec<f32> {
        let mut x = start.to_vec();
        let tol = 1e-4 * self.bandwidth;
        for _ in 0..200 {
            let y = self.shift(points, &x);
            let moved: f32 = y.iter().zip(&x).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
            x = y;
            if moved < tol {
                break;
            }
        }
        x
    }
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

impl RegionDiscovery for DensityModes {
    fn method(&self) -> DiscoveryMethod {
        DiscoveryMethod::Density
    }

    fn discover(&self, points: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if points.is_empty() {
            return Vec::new();
        }
        // canonical order → the seed subset does not depend on the input order
        let mut sorted: Vec<&Vec<f32>> = points.iter().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let stride = sorted.len().div_ceil(self.max_seeds.max(1)).max(1);
        let pts: Vec<Vec<f32>> = sorted.iter().map(|p| (*p).clone()).collect();
        let mut modes: Vec<(Vec<f32>, usize)> = Vec::new();
        let merge = (self.bandwidth / 2.0).powi(2);
        for seed in sorted.iter().step_by(stride) {
            let m = self.climb(&pts, seed);
            match modes.iter_mut().find(|(c, _)| dist2(c, &m) < merge) {
                Some((_, n)) => *n += 1,
                None => modes.push((m, 1)),
            }
        }
        // support of each mode = points nearest to it
        let mut support = vec![0usize; modes.len()];
        for p in points {
            if let Some(i) = fa_patterns::nearest(&modes.iter().map(|m| m.0.clone()).collect::<Vec<_>>(), p) {
                support[i] += 1;
            }
        }
        let min = (self.min_share * points.len() as f64).ceil() as usize;
        let mut keep: Vec<(Vec<f32>, usize)> =
            modes.into_iter().zip(support).filter(|(_, s)| *s >= min.max(1)).map(|((c, _), s)| (c, s)).collect();
        keep.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)));
        keep.truncate(self.max_regions.max(1));
        // rounding makes the centroids bit-stable across seed orders
        let mut out: Vec<Vec<f32>> =
            keep.into_iter().map(|(c, _)| c.iter().map(|v| (v * 1e4).round() / 1e4).collect()).collect();
        out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    fn assign(&self, centroids: &[Vec<f32>], point: &[f32]) -> Option<usize> {
        let i = fa_patterns::nearest(centroids, point)?;
        (dist2(&centroids[i], point) <= (3.0 * self.bandwidth).powi(2)).then_some(i)
    }
}

/// The region discovery a family declares (spec 31 §3, spec 30): density modes for the
/// continuous-prior families, the leader clustering otherwise (codebook / mapper discovery are
/// not implemented; their families fall back to the leader baseline).
pub fn discovery_for(family: super::NeuralFamily, radius: f32, max_regions: usize) -> Box<dyn RegionDiscovery> {
    match family {
        super::NeuralFamily::HierVaeFlow | super::NeuralFamily::LatentDiffusion => {
            Box::new(DensityModes::new(radius, max_regions))
        }
        _ => Box::new(fa_patterns::LeaderClustering { radius, max_regions }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec 30 §9.1 through the shared conformance suite.
    #[test]
    fn density_modes_conform() {
        let r = fa_patterns::conformance::assert_conformance(&DensityModes::new(1.0, 64), 6.0, 0.3);
        assert_eq!(r.n_regions, 6);
    }

    #[test]
    fn families_declare_their_discovery() {
        use super::super::NeuralFamily;
        assert_eq!(discovery_for(NeuralFamily::HierVaeFlow, 1.0, 8).method(), DiscoveryMethod::Density);
        assert_eq!(discovery_for(NeuralFamily::LatentDiffusion, 1.0, 8).method(), DiscoveryMethod::Density);
        assert_eq!(discovery_for(NeuralFamily::JepaVae, 1.0, 8).method(), DiscoveryMethod::Leader);
        let r =
            fa_patterns::conformance::assert_conformance(&*discovery_for(NeuralFamily::FsqHybrid, 2.0, 64), 6.0, 0.3);
        assert!(r.passes());
    }
}
