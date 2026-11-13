//! Hardware floor for k-NN on this host (ADR 0010): what the CPU does with no database in the way.
//! Measures, in process and per size step:
//! - exact brute-force scan (cosine on pre-normalised vectors = dot product), one query at a time,
//!   single-threaded and split across all cores with rayon;
//! - an in-process HNSW (`instant-distance`, ef_construction 100, ef_search 100, M 32): build time,
//!   query latency and recall@k against the exact scan.
//!
//! Same Gaussian-mixture data as `knn_bench` (64 clusters, σ 0.35). No server involved:
//! ```sh
//! cargo run --release -p fa-hub --example knn_floor -- --dim 32 --n 50000,1000000 --queries 300
//! ```
//! `--no-hnsw` skips the index (e.g. for the 10 M brute-force row).
use std::time::{Duration, Instant};

use instant_distance::{Builder, Point, Search};
use rayon::prelude::*;

struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.u().ln()).sqrt() * (std::f64::consts::TAU * self.u()).cos()
    }
}

fn mixture(rng: &mut Rng, clusters: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..clusters).map(|_| (0..dim).map(|_| rng.normal() as f32).collect()).collect()
}

fn draw(rng: &mut Rng, centers: &[Vec<f32>]) -> Vec<f32> {
    let c = &centers[(rng.u() * centers.len() as f64) as usize % centers.len()];
    let mut v: Vec<f32> = c.iter().map(|x| x + 0.35 * rng.normal() as f32).collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
    v
}

#[derive(Clone)]
struct P(Vec<f32>);
impl Point for P {
    fn distance(&self, other: &Self) -> f32 {
        1.0 - self.0.iter().zip(&other.0).map(|(a, b)| a * b).sum::<f32>()
    }
}

/// Top-k (distance, index) of `q` over `data[lo..hi]` (row-major, `dim` wide).
fn scan(data: &[f32], dim: usize, lo: usize, hi: usize, q: &[f32], k: usize) -> Vec<(f32, usize)> {
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    let mut worst = f32::INFINITY;
    for i in lo..hi {
        let row = &data[i * dim..(i + 1) * dim];
        let d = 1.0 - row.iter().zip(q).map(|(a, b)| a * b).sum::<f32>();
        if best.len() < k || d < worst {
            let pos = best.partition_point(|x| x.0 <= d);
            best.insert(pos, (d, i));
            best.truncate(k);
            worst = best.last().map_or(f32::INFINITY, |x| x.0);
        }
    }
    best
}

fn scan_par(data: &[f32], dim: usize, n: usize, q: &[f32], k: usize) -> Vec<(f32, usize)> {
    let parts = rayon::current_num_threads() * 4;
    let step = n.div_ceil(parts);
    let mut all: Vec<(f32, usize)> = (0..parts)
        .into_par_iter()
        .flat_map_iter(|p| scan(data, dim, (p * step).min(n), ((p + 1) * step).min(n), q, k))
        .collect();
    all.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    all.truncate(k);
    all
}

fn pct(sorted: &[Duration], p: f64) -> f64 {
    sorted[((sorted.len() as f64 - 1.0) * p).round() as usize].as_secs_f64() * 1e3
}

fn main() {
    let mut dim = 32usize;
    let mut sizes = vec![50_000usize, 1_000_000];
    let mut queries = 300usize;
    let mut k = 10usize;
    let mut hnsw = true;
    let mut serial_queries = 30usize;
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        let mut v = || it.next().expect("flag value").replace('_', "");
        match f.as_str() {
            "--dim" => dim = v().parse().expect("dim"),
            "--n" => sizes = v().split(',').map(|s| s.parse().expect("n")).collect(),
            "--queries" => queries = v().parse().expect("queries"),
            "--serial-queries" => serial_queries = v().parse().expect("serial-queries"),
            "--k" => k = v().parse().expect("k"),
            "--no-hnsw" => hnsw = false,
            other => panic!("unknown flag {other}"),
        }
    }
    sizes.sort_unstable();
    let threads = rayon::current_num_threads();
    println!("dim={dim} k={k} queries={queries} threads={threads}");
    println!(
        "{:>9} {:>11} {:>11} {:>11} {:>9} {:>9} {:>9} {:>9} {:>8}",
        "n", "scan1 p50", "scanN p50", "scanN p95", "build s", "hnsw p50", "hnsw p95", "hnsw p99", "recall"
    );
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let centers = mixture(&mut rng, 64, dim);
    let mut data: Vec<f32> = Vec::new();
    for &n in &sizes {
        while data.len() < n * dim {
            data.extend(draw(&mut rng, &centers));
        }
        let qs: Vec<Vec<f32>> = (0..queries).map(|_| draw(&mut rng, &centers)).collect();
        // single-threaded exact scan (a few queries; it is linear and stable)
        let mut s1: Vec<Duration> = qs
            .iter()
            .take(serial_queries.min(queries))
            .map(|q| {
                let t = Instant::now();
                std::hint::black_box(scan(&data, dim, 0, n, q, k));
                t.elapsed()
            })
            .collect();
        s1.sort_unstable();
        // all-core exact scan, one query at a time; also the ground truth
        let mut sn = Vec::with_capacity(queries);
        let truth: Vec<Vec<usize>> = qs
            .iter()
            .map(|q| {
                let t = Instant::now();
                let r = scan_par(&data, dim, n, q, k);
                sn.push(t.elapsed());
                r.into_iter().map(|x| x.1).collect()
            })
            .collect();
        sn.sort_unstable();
        let (mut build, mut h50, mut h95, mut h99, mut recall) = (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        if hnsw {
            let pts: Vec<P> = data.chunks_exact(dim).take(n).map(|c| P(c.to_vec())).collect();
            let vals: Vec<usize> = (0..n).collect();
            let t = Instant::now();
            let map = Builder::default().ef_construction(100).ef_search(100).seed(7).build(pts, vals);
            build = t.elapsed().as_secs_f64();
            let mut search = Search::default();
            let mut lat = Vec::with_capacity(queries);
            let (mut hit, mut want) = (0usize, 0usize);
            for (q, tr) in qs.iter().zip(&truth) {
                let qp = P(q.clone());
                let t = Instant::now();
                let got: Vec<usize> = map.search(&qp, &mut search).take(k).map(|item| *item.value).collect();
                lat.push(t.elapsed());
                hit += got.iter().filter(|g| tr.contains(g)).count();
                want += tr.len();
            }
            lat.sort_unstable();
            (h50, h95, h99) = (pct(&lat, 0.5), pct(&lat, 0.95), pct(&lat, 0.99));
            recall = hit as f64 / want.max(1) as f64;
        }
        println!(
            "{:>9} {:>11.3} {:>11.3} {:>11.3} {:>9.1} {:>9.3} {:>9.3} {:>9.3} {:>8.3}",
            n,
            pct(&s1, 0.5),
            pct(&sn, 0.5),
            pct(&sn, 0.95),
            build,
            h50,
            h95,
            h99,
            recall
        );
    }
}
