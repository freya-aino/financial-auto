//! Family-specific region discovery (spec 30 §6.2) on **any encoder output**: every method sees
//! only the per-expert latent `mean` vectors, so statistical and neural families plug in alike.
//!
//! | method | type | spec 30 §6.2 family | algorithm |
//! |---|---|---|---|
//! | [`DensityClusters`] | `density` | hierarchical VAE + flow prior, JEPA + VAE head, latent diffusion | HDBSCAN: core distances (k-NN), mutual-reachability MST (Prim), single-linkage dendrogram, condensed tree with `min_cluster_size`, excess-of-mass selection |
//! | [`MapperCover`] | `mapper` | JEPA embedding | Mapper: filter = first principal component, overlapping interval cover, single-linkage clusters per interval, nerve graph; regions = connected components of the nerve |
//! | [`CodebookRegions`] | `codebook` | FSQ/VQ hybrid | a VQ codebook (deterministic k-means++ / Lloyd) over the latents; used codes are candidate regions, codes with overlapping spread merge (union-find), a code whose members are clearly bimodal splits |
//!
//! All three are deterministic and independent of the input order (points are sorted
//! canonically first; centroids are emitted sorted), as spec 30 §9.3 requires, and pass the
//! shared [`crate::conformance`] suite (§9.1). Points are assigned to the nearest centroid within
//! `assign_radius` (`None` = always the nearest).

use std::cmp::Ordering;

use fa_core::rng::Rng;

use crate::{DiscoveryMethod, RegionDiscovery, nearest};

fn d2(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum()
}

fn canonical(points: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut v = points.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    v
}

fn mean_of(points: &[&Vec<f32>]) -> Vec<f32> {
    let dim = points.first().map_or(0, |p| p.len());
    let mut m = vec![0.0f64; dim];
    for p in points {
        for (a, v) in m.iter_mut().zip(p.iter()) {
            *a += f64::from(*v);
        }
    }
    let n = points.len().max(1) as f64;
    m.into_iter().map(|v| (v / n) as f32).collect()
}

/// Centroids of labelled groups, rounded (bit-stable) and sorted.
fn centroids(points: &[Vec<f32>], labels: &[Option<usize>], max_regions: usize) -> Vec<Vec<f32>> {
    let k = labels.iter().flatten().max().map_or(0, |m| m + 1);
    let mut groups: Vec<Vec<&Vec<f32>>> = vec![Vec::new(); k];
    for (p, l) in points.iter().zip(labels) {
        if let Some(l) = l {
            groups[*l].push(p);
        }
    }
    let mut out: Vec<(usize, Vec<f32>)> = groups
        .iter()
        .filter(|g| !g.is_empty())
        .map(|g| (g.len(), mean_of(g).into_iter().map(|v| (v * 1e4).round() / 1e4).collect()))
        .collect();
    out.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)));
    out.truncate(max_regions.max(1));
    let mut c: Vec<Vec<f32>> = out.into_iter().map(|x| x.1).collect();
    c.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    c
}

fn assign_within(centroids: &[Vec<f32>], point: &[f32], radius: Option<f32>) -> Option<usize> {
    let i = nearest(centroids, point)?;
    match radius {
        Some(r) => (d2(&centroids[i], point) <= f64::from(r).powi(2)).then_some(i),
        None => Some(i),
    }
}

struct UnionFind(Vec<usize>);

impl UnionFind {
    fn new(n: usize) -> Self {
        Self((0..n).collect())
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.0[x] != x {
            self.0[x] = self.0[self.0[x]];
            x = self.0[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            // smaller root wins: deterministic labels
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            self.0[hi] = lo;
        }
    }
    /// Dense component labels in first-occurrence order.
    fn labels(&mut self) -> Vec<usize> {
        let n = self.0.len();
        let mut map = std::collections::HashMap::new();
        (0..n)
            .map(|i| {
                let r = self.find(i);
                let next = map.len();
                *map.entry(r).or_insert(next)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Density (HDBSCAN)

/// HDBSCAN density clustering (spec 30 §6.2 density families).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DensityClusters {
    /// `k` of the core distance.
    pub min_samples: usize,
    /// Smallest cluster in the condensed tree (also scales with `min_share`).
    pub min_cluster_size: usize,
    /// Minimum cluster size as a share of the points (the larger of the two applies).
    pub min_share: f64,
    pub max_regions: usize,
    /// At most this many points are clustered (an even stride over the canonical order).
    pub max_points: usize,
    pub assign_radius: Option<f32>,
}

impl DensityClusters {
    pub fn new(max_regions: usize) -> Self {
        Self {
            min_samples: 5,
            min_cluster_size: 10,
            min_share: 0.01,
            max_regions,
            max_points: 4000,
            assign_radius: None,
        }
    }

    /// Cluster label per point (`None` = noise).
    pub fn labels(&self, points: &[Vec<f32>]) -> Vec<Option<usize>> {
        let n = points.len();
        if n < 2 {
            return vec![Some(0); n];
        }
        let k = self.min_samples.clamp(1, n - 1);
        // core distance = distance to the k-th neighbour
        let core: Vec<f64> = (0..n)
            .map(|i| {
                let mut ds: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| d2(&points[i], &points[j])).collect();
                ds.select_nth_unstable_by(k - 1, f64::total_cmp);
                ds[k - 1].sqrt()
            })
            .collect();
        // Prim's MST on mutual reachability
        let mut in_tree = vec![false; n];
        let mut best = vec![f64::INFINITY; n];
        let mut from = vec![0usize; n];
        let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(n - 1);
        let mut cur = 0;
        in_tree[0] = true;
        for _ in 1..n {
            for j in 0..n {
                if in_tree[j] {
                    continue;
                }
                let mr = d2(&points[cur], &points[j]).sqrt().max(core[cur]).max(core[j]);
                if mr < best[j] {
                    best[j] = mr;
                    from[j] = cur;
                }
            }
            let next = (0..n)
                .filter(|&j| !in_tree[j])
                .min_by(|&a, &b| best[a].total_cmp(&best[b]).then(a.cmp(&b)))
                .unwrap_or(0);
            edges.push((best[next], from[next], next));
            in_tree[next] = true;
            cur = next;
        }
        edges.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        // single-linkage dendrogram: node n + m merges two components at distance d
        let mut uf = UnionFind::new(n);
        let mut node_of: Vec<usize> = (0..n).collect(); // component root → dendrogram node
        let mut children: Vec<(usize, usize, f64)> = Vec::with_capacity(n - 1);
        let mut size: Vec<usize> = vec![1; n];
        for (d, a, b) in edges {
            let (ra, rb) = (uf.find(a), uf.find(b));
            let (na, nb) = (node_of[ra], node_of[rb]);
            let id = n + children.len();
            children.push((na, nb, d));
            size.push(size[na] + size[nb]);
            uf.union(ra, rb);
            let r = uf.find(ra);
            node_of[r] = id;
        }
        let mcs = self.min_cluster_size.max((self.min_share * n as f64).ceil() as usize).max(2);
        let root = n + children.len() - 1;
        // condensed tree
        let mut parent: Vec<Option<usize>> = vec![None];
        let mut birth: Vec<f64> = vec![0.0];
        let mut stability: Vec<f64> = vec![0.0];
        let mut fell: Vec<(usize, usize)> = Vec::with_capacity(n); // (point, cluster)
        let leaves = |node: usize| -> Vec<usize> {
            let mut out = Vec::new();
            let mut st = vec![node];
            while let Some(x) = st.pop() {
                if x < n {
                    out.push(x);
                } else {
                    let (a, b, _) = children[x - n];
                    st.push(a);
                    st.push(b);
                }
            }
            out
        };
        let mut stack = vec![(root, 0usize)];
        while let Some((node, c)) = stack.pop() {
            if node < n {
                fell.push((node, c));
                continue;
            }
            let (a, b, d) = children[node - n];
            let lambda = if d > 0.0 { 1.0 / d } else { 1e12 };
            let (sa, sb) = (size[a], size[b]);
            let grow = lambda - birth[c];
            match (sa >= mcs, sb >= mcs) {
                (true, true) => {
                    stability[c] += (sa + sb) as f64 * grow;
                    for child in [a, b] {
                        parent.push(Some(c));
                        birth.push(lambda);
                        stability.push(0.0);
                        stack.push((child, parent.len() - 1));
                    }
                }
                (true, false) | (false, true) => {
                    let (big, small, ss) = if sa >= mcs { (a, b, sb) } else { (b, a, sa) };
                    stability[c] += ss as f64 * grow;
                    for p in leaves(small) {
                        fell.push((p, c));
                    }
                    stack.push((big, c));
                }
                (false, false) => {
                    stability[c] += (sa + sb) as f64 * grow;
                    for p in leaves(a).into_iter().chain(leaves(b)) {
                        fell.push((p, c));
                    }
                }
            }
        }
        // excess of mass, bottom-up (children have larger ids than their parents)
        let nc = parent.len();
        let mut selected = vec![false; nc];
        let mut sel_stab = vec![0.0; nc];
        let mut kids: Vec<Vec<usize>> = vec![Vec::new(); nc];
        for (c, p) in parent.iter().enumerate().skip(1) {
            if let Some(p) = p {
                kids[*p].push(c);
            }
        }
        for c in (1..nc).rev() {
            let child_sum: f64 = kids[c].iter().map(|k| sel_stab[*k]).sum();
            if stability[c] >= child_sum || kids[c].is_empty() {
                selected[c] = true;
                sel_stab[c] = stability[c];
                let mut st = kids[c].clone();
                while let Some(x) = st.pop() {
                    selected[x] = false;
                    st.extend(kids[x].iter().copied());
                }
            } else {
                sel_stab[c] = child_sum;
            }
        }
        if !selected.iter().any(|s| *s) {
            selected[0] = true; // one cluster only
        }
        let ids: Vec<usize> = (0..nc).filter(|c| selected[*c]).collect();
        let mut labels = vec![None; n];
        for (p, c) in fell {
            let mut x = Some(c);
            while let Some(cl) = x {
                if selected[cl] {
                    labels[p] = ids.iter().position(|i| *i == cl);
                    break;
                }
                x = parent[cl];
            }
        }
        labels
    }
}

impl RegionDiscovery for DensityClusters {
    fn method(&self) -> DiscoveryMethod {
        DiscoveryMethod::Density
    }
    fn discover(&self, points: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let sorted = canonical(points);
        let stride = sorted.len().div_ceil(self.max_points.max(1)).max(1);
        let pts: Vec<Vec<f32>> = sorted.into_iter().step_by(stride).collect();
        let labels = self.labels(&pts);
        centroids(&pts, &labels, self.max_regions)
    }
    fn assign(&self, centroids: &[Vec<f32>], point: &[f32]) -> Option<usize> {
        assign_within(centroids, point, self.assign_radius)
    }
}

// ---------------------------------------------------------------------------------------------
// Mapper

/// Mapper / topological cover (spec 30 §6.2 JEPA embedding).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapperCover {
    pub intervals: usize,
    /// Fractional overlap of neighbouring intervals.
    pub overlap: f64,
    /// Single-linkage threshold within an interval (`None` = 2 × median 5-NN distance).
    pub link: Option<f64>,
    /// Components below this share of the points are not regions (their points join the nearest).
    pub min_share: f64,
    pub max_regions: usize,
    pub max_points: usize,
    pub assign_radius: Option<f32>,
}

impl MapperCover {
    pub fn new(max_regions: usize) -> Self {
        Self {
            intervals: 10,
            overlap: 0.3,
            link: None,
            min_share: 0.01,
            max_regions,
            max_points: 3000,
            assign_radius: None,
        }
    }

    /// First principal direction by power iteration (deterministic start).
    fn filter(points: &[Vec<f32>]) -> Vec<f64> {
        let dim = points[0].len();
        let n = points.len() as f64;
        let mean: Vec<f64> = (0..dim).map(|d| points.iter().map(|p| f64::from(p[d])).sum::<f64>() / n).collect();
        let mut v: Vec<f64> = (0..dim).map(|d| 1.0 + d as f64 * 0.1).collect();
        for _ in 0..100 {
            let mut w = vec![0.0; dim];
            for p in points {
                let proj: f64 = p.iter().zip(&mean).zip(&v).map(|((x, m), vv)| (f64::from(*x) - m) * vv).sum();
                for d in 0..dim {
                    w[d] += proj * (f64::from(p[d]) - mean[d]);
                }
            }
            let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm <= 1e-12 {
                break;
            }
            v = w.into_iter().map(|x| x / norm).collect();
        }
        // sign convention: largest component positive
        let flip = v.iter().copied().max_by(|a, b| a.abs().total_cmp(&b.abs())).unwrap_or(1.0) < 0.0;
        points
            .iter()
            .map(|p| {
                let s: f64 = p.iter().zip(&mean).zip(&v).map(|((x, m), vv)| (f64::from(*x) - m) * vv).sum();
                if flip { -s } else { s }
            })
            .collect()
    }

    /// Component label per point (`None` = in a component below `min_share`).
    pub fn labels(&self, points: &[Vec<f32>]) -> Vec<Option<usize>> {
        let n = points.len();
        if n == 0 {
            return Vec::new();
        }
        let f = Self::filter(points);
        let (lo, hi) = f.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), x| (a.min(*x), b.max(*x)));
        let link = self.link.unwrap_or_else(|| {
            let mut nn: Vec<f64> = (0..n)
                .step_by((n / 300).max(1))
                .map(|i| {
                    let mut ds: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| d2(&points[i], &points[j])).collect();
                    let k = 5.min(ds.len()).max(1) - 1;
                    ds.select_nth_unstable_by(k, f64::total_cmp);
                    ds[k].sqrt()
                })
                .collect();
            nn.sort_by(f64::total_cmp);
            2.0 * nn.get(nn.len() / 2).copied().unwrap_or(1.0)
        });
        let k = self.intervals.max(1) as f64;
        let width = ((hi - lo) / (k - (k - 1.0) * self.overlap)).max(1e-9);
        let step = width * (1.0 - self.overlap);
        // the nerve: points in the same interval cluster are linked; overlaps connect intervals
        let mut uf = UnionFind::new(n);
        for i in 0..self.intervals.max(1) {
            let a = lo + i as f64 * step;
            let b = a + width;
            let members: Vec<usize> = (0..n).filter(|&j| f[j] >= a - 1e-12 && f[j] <= b + 1e-12).collect();
            let mut local = UnionFind::new(members.len());
            for x in 0..members.len() {
                for y in x + 1..members.len() {
                    if d2(&points[members[x]], &points[members[y]]) <= link * link {
                        local.union(x, y);
                    }
                }
            }
            let ll = local.labels();
            let mut first: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
            for (x, l) in ll.iter().enumerate() {
                match first.get(l) {
                    Some(&f0) => uf.union(members[f0], members[x]),
                    None => {
                        first.insert(*l, x);
                    }
                }
            }
        }
        let comp = uf.labels();
        let mut size = std::collections::HashMap::new();
        for c in &comp {
            *size.entry(*c).or_insert(0usize) += 1;
        }
        let min = ((self.min_share * n as f64).ceil() as usize).max(2);
        let mut dense = std::collections::HashMap::new();
        comp.iter()
            .map(|c| {
                (size[c] >= min).then(|| {
                    let next = dense.len();
                    *dense.entry(*c).or_insert(next)
                })
            })
            .collect()
    }
}

impl RegionDiscovery for MapperCover {
    fn method(&self) -> DiscoveryMethod {
        DiscoveryMethod::Mapper
    }
    fn discover(&self, points: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let sorted = canonical(points);
        let stride = sorted.len().div_ceil(self.max_points.max(1)).max(1);
        let pts: Vec<Vec<f32>> = sorted.into_iter().step_by(stride).collect();
        let labels = self.labels(&pts);
        centroids(&pts, &labels, self.max_regions)
    }
    fn assign(&self, centroids: &[Vec<f32>], point: &[f32]) -> Option<usize> {
        assign_within(centroids, point, self.assign_radius)
    }
}

// ---------------------------------------------------------------------------------------------
// Codebook

/// Codebook regions (spec 30 §6.2 FSQ/VQ hybrid).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CodebookRegions {
    pub codebook_size: usize,
    /// A code is "used" when it holds at least this share of the points.
    pub min_usage: f64,
    /// Codes merge when their centroid distance ≤ `merge_factor · (rms spread a + rms spread b)`.
    pub merge_factor: f64,
    /// A code splits when its 2-means halves are ≥ `split_gap` pooled sds apart.
    pub split_gap: f64,
    pub max_regions: usize,
    pub lloyd_iters: usize,
    pub assign_radius: Option<f32>,
}

impl CodebookRegions {
    pub fn new(max_regions: usize) -> Self {
        Self {
            codebook_size: 32,
            min_usage: 0.005,
            merge_factor: 1.0,
            split_gap: 4.0,
            max_regions,
            lloyd_iters: 30,
            assign_radius: None,
        }
    }

    /// k-means++ (seeded, deterministic) + Lloyd on canonically ordered points.
    fn kmeans(points: &[Vec<f32>], k: usize, iters: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<usize>) {
        let n = points.len();
        let k = k.min(n).max(1);
        let mut rng = Rng::seed(seed);
        let mut cents: Vec<Vec<f32>> = vec![points[0].clone()];
        let mut dmin: Vec<f64> = points.iter().map(|p| d2(p, &cents[0])).collect();
        while cents.len() < k {
            let total: f64 = dmin.iter().sum();
            if total <= 0.0 {
                break;
            }
            let mut u = rng.uniform() * total;
            let mut pick = n - 1;
            for (i, d) in dmin.iter().enumerate() {
                u -= d;
                if u <= 0.0 {
                    pick = i;
                    break;
                }
            }
            cents.push(points[pick].clone());
            for (i, p) in points.iter().enumerate() {
                dmin[i] = dmin[i].min(d2(p, &points[pick]));
            }
        }
        let mut assign = vec![0usize; n];
        for _ in 0..iters {
            let mut changed = false;
            for (i, p) in points.iter().enumerate() {
                let a = nearest(&cents, p).unwrap_or(0);
                changed |= a != assign[i];
                assign[i] = a;
            }
            let dim = points[0].len();
            let mut sums = vec![vec![0.0f64; dim]; cents.len()];
            let mut counts = vec![0usize; cents.len()];
            for (p, a) in points.iter().zip(&assign) {
                counts[*a] += 1;
                for (s, v) in sums[*a].iter_mut().zip(p) {
                    *s += f64::from(*v);
                }
            }
            for (c, (s, m)) in cents.iter_mut().zip(sums.iter().zip(&counts)) {
                if *m > 0 {
                    *c = s.iter().map(|v| (v / *m as f64) as f32).collect();
                }
            }
            if !changed {
                break;
            }
        }
        (cents, assign)
    }

    fn rms(points: &[&Vec<f32>], c: &[f32]) -> f64 {
        (points.iter().map(|p| d2(p, c)).sum::<f64>() / points.len().max(1) as f64).sqrt()
    }

    /// Region label per point.
    pub fn labels(&self, points: &[Vec<f32>]) -> Vec<Option<usize>> {
        let n = points.len();
        if n == 0 {
            return Vec::new();
        }
        let (cents, assign) = Self::kmeans(points, self.codebook_size, self.lloyd_iters, 0xC0DE);
        let k = cents.len();
        let members: Vec<Vec<&Vec<f32>>> =
            (0..k).map(|c| points.iter().zip(&assign).filter(|(_, a)| **a == c).map(|(p, _)| p).collect()).collect();
        let min = (self.min_usage * n as f64).ceil() as usize;
        let used: Vec<bool> = members.iter().map(|m| m.len() >= min.max(1)).collect();
        let spread: Vec<f64> = (0..k).map(|c| Self::rms(&members[c], &cents[c])).collect();
        // merge codes with overlapping spread
        let mut uf = UnionFind::new(k);
        for a in 0..k {
            for b in a + 1..k {
                if used[a] && used[b] && d2(&cents[a], &cents[b]).sqrt() <= self.merge_factor * (spread[a] + spread[b])
                {
                    uf.union(a, b);
                }
            }
        }
        let code_region = uf.labels();
        // split a code whose members are clearly bimodal (2-means gap)
        let mut extra = code_region.iter().max().map_or(0, |m| m + 1);
        let mut point_region: Vec<Option<usize>> = assign.iter().map(|a| used[*a].then_some(code_region[*a])).collect();
        for c in (0..k).filter(|c| used[*c] && members[*c].len() >= 4) {
            let pts: Vec<Vec<f32>> = members[c].iter().map(|p| (*p).clone()).collect();
            let (two, lab) = Self::kmeans(&pts, 2, 20, 0x5B11);
            if two.len() < 2 {
                continue;
            }
            let halves: Vec<Vec<&Vec<f32>>> =
                (0..2).map(|h| pts.iter().zip(&lab).filter(|(_, l)| **l == h).map(|(p, _)| p).collect()).collect();
            if halves.iter().any(|h| h.len() < min.max(2)) {
                continue;
            }
            let pooled =
                ((Self::rms(&halves[0], &two[0]).powi(2) + Self::rms(&halves[1], &two[1]).powi(2)) / 2.0).sqrt();
            if d2(&two[0], &two[1]).sqrt() >= self.split_gap * pooled.max(1e-9) {
                let idx: Vec<usize> = (0..n).filter(|i| assign[*i] == c).collect();
                for (j, i) in idx.iter().enumerate() {
                    if lab[j] == 1 {
                        point_region[*i] = Some(extra);
                    }
                }
                extra += 1;
            }
        }
        // dense relabel
        let mut map = std::collections::HashMap::new();
        point_region
            .into_iter()
            .map(|r| {
                r.map(|r| {
                    let next = map.len();
                    *map.entry(r).or_insert(next)
                })
            })
            .collect()
    }
}

impl RegionDiscovery for CodebookRegions {
    fn method(&self) -> DiscoveryMethod {
        DiscoveryMethod::Codebook
    }
    fn discover(&self, points: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let pts = canonical(points);
        let labels = self.labels(&pts);
        centroids(&pts, &labels, self.max_regions)
    }
    fn assign(&self, centroids: &[Vec<f32>], point: &[f32]) -> Option<usize> {
        assign_within(centroids, point, self.assign_radius)
    }
}

/// The discovery for a method name (`fa_encoders::FamilyDescriptor::region_discovery`):
/// `density`, `mapper`, `codebook`, else the leader baseline with `radius`.
pub fn discovery_for_method(method: &str, radius: f32, max_regions: usize) -> Box<dyn RegionDiscovery> {
    match method {
        "density" => Box::new(DensityClusters::new(max_regions)),
        "mapper" => Box::new(MapperCover::new(max_regions)),
        "codebook" => Box::new(CodebookRegions::new(max_regions)),
        _ => Box::new(crate::LeaderClustering { radius, max_regions }),
    }
}

/// Fits and assigns every point (nearest centroid as a fallback), like
/// `LeaderClustering::fit_assign`, for any discovery.
pub fn fit_assign(d: &dyn RegionDiscovery, points: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<Option<usize>>) {
    let c = d.discover(points);
    let a = points.iter().map(|p| d.assign(&c, p).or_else(|| nearest(&c, p))).collect();
    (c, a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conformance::{assert_conformance, check, gaussian_mixture};

    #[test]
    fn density_clusters_conform() {
        let r = assert_conformance(&DensityClusters::new(64), 6.0, 0.3);
        assert_eq!(r.n_regions, 6);
    }

    #[test]
    fn mapper_conforms() {
        let r = assert_conformance(&MapperCover::new(64), 6.0, 0.3);
        assert_eq!(r.n_regions, 6);
    }

    #[test]
    fn codebook_conforms() {
        let r = assert_conformance(&CodebookRegions::new(64), 6.0, 0.3);
        assert_eq!(r.n_regions, 6);
    }

    #[test]
    fn methods_work_on_other_scales_and_dimensions() {
        // a neural-looking latent: 16 dims, unit-ish scale, unequal cluster sizes
        let (mut p, mut l) = gaussian_mixture(4, 16, 150, 2.0, 0.1, 3);
        let (p2, l2) = gaussian_mixture(4, 16, 40, 2.0, 0.1, 4);
        p.extend(p2);
        l.extend(l2);
        for d in [
            Box::new(DensityClusters::new(64)) as Box<dyn RegionDiscovery>,
            Box::new(MapperCover::new(64)),
            Box::new(CodebookRegions::new(64)),
        ] {
            let r = check(&*d, &p, &l);
            assert!(r.passes(), "{:?}: {r:?}", d.method());
        }
    }

    #[test]
    fn hdbscan_marks_sparse_noise_and_factory_maps_methods() {
        let (mut p, _) = gaussian_mixture(2, 2, 100, 10.0, 0.2, 9);
        p.push(vec![50.0, 50.0]);
        let labels = DensityClusters::new(8).labels(&p);
        assert_eq!(labels.iter().flatten().max(), Some(&1));
        assert_eq!(labels.last(), Some(&None), "an isolated point is noise");
        assert_eq!(discovery_for_method("density", 1.0, 8).method(), DiscoveryMethod::Density);
        assert_eq!(discovery_for_method("mapper", 1.0, 8).method(), DiscoveryMethod::Mapper);
        assert_eq!(discovery_for_method("codebook", 1.0, 8).method(), DiscoveryMethod::Codebook);
        assert_eq!(discovery_for_method("leader", 1.0, 8).method(), DiscoveryMethod::Leader);
        let (c, a) = fit_assign(&*discovery_for_method("codebook", 1.0, 8), &p);
        assert!(c.len() >= 2 && a.iter().all(Option::is_some));
    }
}
