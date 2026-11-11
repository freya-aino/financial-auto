//! Shared conformance test for [`RegionDiscovery`] implementations (spec 30 §9.1).
//!
//! Every encoder family's discovery must, on synthetic Gaussian-mixture latents:
//! 1. recover ≥ 90 % of the true clusters (a cluster is recovered when one discovered region
//!    holds ≥ 80 % of its points and ≥ 80 % of that region's points come from the cluster);
//! 2. be deterministic (same points → same centroids, §9.3);
//! 3. keep identity under [`crate::lineage::align`] when re-fitted on a shuffled-label copy of
//!    the probe set (carried mass ≥ 0.9, §9.4).
//!
//! Family crates call [`assert_conformance`] from their own tests; the data generator is
//! synthetic and seeded (never market data).

use fa_core::rng::Rng;

use crate::{RegionDiscovery, lineage};

/// Synthetic mixture: `k` isotropic Gaussian clusters in `dim` dimensions whose centres lie
/// `separation` apart along distinct axes (and their negatives), `noise` standard deviation.
pub fn gaussian_mixture(
    k: usize,
    dim: usize,
    n_per: usize,
    separation: f64,
    noise: f64,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut rng = Rng::seed(seed);
    let centres: Vec<Vec<f64>> = (0..k)
        .map(|c| {
            let mut v = vec![0.0; dim.max(1)];
            if c > 0 {
                let axis = (c - 1) % dim.max(1);
                let sign = if ((c - 1) / dim.max(1)).is_multiple_of(2) { 1.0 } else { -1.0 };
                v[axis] = sign * separation;
            }
            v
        })
        .collect();
    let mut points = Vec::with_capacity(k * n_per);
    let mut labels = Vec::with_capacity(k * n_per);
    for i in 0..k * n_per {
        let c = i % k;
        points.push(centres[c].iter().map(|x| (x + noise * rng.normal()) as f32).collect());
        labels.push(c);
    }
    (points, labels)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConformanceReport {
    pub n_regions: usize,
    /// Fraction of true clusters recovered.
    pub recovered: f64,
    pub deterministic: bool,
    /// Carried probe mass when aligning the fit with a fit of the reversed point order.
    pub self_alignment_mass: f64,
}

impl ConformanceReport {
    pub fn passes(&self) -> bool {
        self.recovered >= 0.9 && self.deterministic && self.self_alignment_mass >= 0.9
    }
}

fn assignments<D: RegionDiscovery + ?Sized>(d: &D, centroids: &[Vec<f32>], points: &[Vec<f32>]) -> Vec<Option<usize>> {
    points.iter().map(|p| d.assign(centroids, p)).collect()
}

/// Runs the three checks on given points and labels.
pub fn check<D: RegionDiscovery + ?Sized>(d: &D, points: &[Vec<f32>], labels: &[usize]) -> ConformanceReport {
    let centroids = d.discover(points);
    let assign = assignments(d, &centroids, points);
    let k = labels.iter().max().map_or(0, |m| m + 1);
    let mut joint = vec![vec![0usize; centroids.len()]; k];
    let mut region_size = vec![0usize; centroids.len()];
    let mut cluster_size = vec![0usize; k];
    for (a, &l) in assign.iter().zip(labels) {
        cluster_size[l] += 1;
        if let Some(r) = a {
            joint[l][*r] += 1;
            region_size[*r] += 1;
        }
    }
    let recovered = (0..k)
        .filter(|&c| {
            (0..centroids.len()).any(|r| {
                let j = joint[c][r] as f64;
                j >= 0.8 * cluster_size[c] as f64 && j >= 0.8 * region_size[r] as f64
            })
        })
        .count();
    let deterministic = d.discover(points) == centroids;
    // Identity: a fit on the reversed order must carry ≥ 90 % of the probe mass.
    let reversed: Vec<Vec<f32>> = points.iter().rev().cloned().collect();
    let c2 = d.discover(&reversed);
    let assign2 = assignments(d, &c2, points);
    let a = lineage::align(&assign, &assign2, c2.len(), &lineage::AlignCfg::default());
    ConformanceReport {
        n_regions: centroids.len(),
        recovered: if k == 0 { 1.0 } else { recovered as f64 / k as f64 },
        deterministic,
        self_alignment_mass: a.carried_mass,
    }
}

/// The standard scenario (6 clusters in 8 dimensions, 4σ apart); panics with the report when
/// the discovery fails. `separation` is in the discovery's own latent units.
pub fn assert_conformance<D: RegionDiscovery + ?Sized>(d: &D, separation: f64, noise: f64) -> ConformanceReport {
    let (points, labels) = gaussian_mixture(6, 8, 200, separation, noise, 17);
    let r = check(d, &points, &labels);
    assert!(r.passes(), "RegionDiscovery {:?} fails spec 30 §9.1 conformance: {r:?}", d.method());
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LeaderClustering;

    #[test]
    fn leader_clustering_conforms() {
        let r = assert_conformance(&LeaderClustering { radius: 2.0, max_regions: 64 }, 6.0, 0.3);
        assert_eq!(r.n_regions, 6);
    }

    #[test]
    fn a_degenerate_discovery_fails() {
        struct One;
        impl RegionDiscovery for One {
            fn method(&self) -> crate::DiscoveryMethod {
                crate::DiscoveryMethod::Density
            }
            fn discover(&self, _: &[Vec<f32>]) -> Vec<Vec<f32>> {
                vec![vec![0.0; 8]]
            }
        }
        let (p, l) = gaussian_mixture(6, 8, 50, 6.0, 0.3, 1);
        let r = check(&One, &p, &l);
        assert!(!r.passes());
        assert!(r.recovered < 0.2);
    }
}
