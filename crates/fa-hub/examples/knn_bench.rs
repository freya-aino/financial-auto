//! Helix k-NN benchmark (spec 20 criterion 4, open question 2): grows ONE expert space through the
//! real write path (`add_latents_checked`) and measures `knn(space, v, k, &KnnFilter)` latency at
//! every size step, plus recall@k against an exact client-side scan.
//!
//! Runs only against a scratch server, never the live hub on port 6969:
//! ```sh
//! docker run -d --name fa-helix-bench -p 127.0.0.1:16969:8080 \
//!   -e HELIX_DATA_DIR=/var/lib/helix --tmpfs /var/lib/helix:size=8g ghcr.io/helixdb/helixdb:v0.0.14
//! cargo run --release -p fa-hub --example knn_bench -- --url http://127.0.0.1:16969 \
//!   --dim 32 --n 10000,50000,100000,250000,500000 --queries 300 --k 10
//! docker rm -f fa-helix-bench
//! ```
//! Helix builds its vector index asynchronously (writes return before the vectors are linked into
//! the graph; a write past ~250 000 pending members answers 429 `index_backpressure`), so
//! `--settle-secs` waits after each insert phase before querying.
//! Vectors are a Gaussian mixture (64 clusters, like region-structured latents) so the index sees
//! realistic structure; queries are fresh draws from the same mixture. Output: one table row per
//! size (insert rate, p50/p95/p99/max latency, recall@k) and a log-n / linear fit extrapolated to
//! `--extrapolate` (default 10 M).
use std::time::{Duration, Instant};

use fa_hub::ast::IndexSpec;
use fa_hub::{HubClient, KnnFilter, NewExpertSpace, NewLatent, NewSnapshot};

struct Args {
    url: String,
    dim: usize,
    sizes: Vec<usize>,
    queries: usize,
    k: usize,
    batch: usize,
    concurrency: usize,
    clusters: usize,
    extrapolate: f64,
    filtered: bool,
    settle_secs: u64,
    drain_secs: u64,
    query_space: Option<String>,
    seed: u64,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        url: String::new(),
        dim: 32,
        sizes: vec![10_000, 50_000, 100_000],
        queries: 300,
        k: 10,
        batch: 500,
        concurrency: 8,
        clusters: 64,
        extrapolate: 10_000_000.0,
        filtered: false,
        settle_secs: 0,
        drain_secs: 0,
        query_space: None,
        seed: 0x9e37_79b9_7f4a_7c15,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        let num = |s: String| s.replace('_', "").parse::<usize>().map_err(|e| format!("{flag}: {e}"));
        match flag.as_str() {
            "--url" => a.url = val()?,
            "--dim" => a.dim = num(val()?)?,
            "--n" => a.sizes = val()?.split(',').map(|s| num(s.to_owned())).collect::<Result<_, _>>()?,
            "--queries" => a.queries = num(val()?)?,
            "--k" => a.k = num(val()?)?,
            "--batch" => a.batch = num(val()?)?,
            "--concurrency" => a.concurrency = num(val()?)?,
            "--clusters" => a.clusters = num(val()?)?,
            "--extrapolate" => a.extrapolate = num(val()?)? as f64,
            "--filtered" => a.filtered = true,
            "--settle-secs" => a.settle_secs = num(val()?)? as u64,
            "--drain-secs" => a.drain_secs = num(val()?)? as u64,
            "--query-space" => a.query_space = Some(val()?),
            "--seed" => a.seed = num(val()?)? as u64,
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if a.url.is_empty() {
        return Err("--url is required (a scratch Helix, never the live one)".into());
    }
    if a.url.contains(":6969") {
        return Err("refusing port 6969: that is the live compose Helix; use a scratch container".into());
    }
    a.sizes.sort_unstable();
    Ok(a)
}

/// xorshift64* + Box–Muller: deterministic and dependency-free.
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

struct Mixture {
    centers: Vec<Vec<f32>>,
}
impl Mixture {
    fn new(rng: &mut Rng, clusters: usize, dim: usize) -> Self {
        Self { centers: (0..clusters).map(|_| (0..dim).map(|_| rng.normal() as f32).collect()).collect() }
    }
    fn draw(&self, rng: &mut Rng) -> Vec<f32> {
        let c = &self.centers[(rng.u() * self.centers.len() as f64) as usize % self.centers.len()];
        c.iter().map(|x| x + 0.35 * rng.normal() as f32).collect()
    }
}

fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b) {
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    1.0 - d / (na.sqrt() * nb.sqrt())
}

fn pct(sorted: &[Duration], p: f64) -> f64 {
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[i].as_secs_f64() * 1e3
}

/// Least squares y = a + b·x; returns (a, b, r²).
fn fit(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    let n = xs.len() as f64;
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    let b = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let a = my - b * mx;
    let ss_tot: f64 = ys.iter().map(|y| (y - my).powi(2)).sum();
    let ss_res: f64 = xs.iter().zip(ys).map(|(x, y)| (y - a - b * x).powi(2)).sum();
    (a, b, if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 1.0 })
}

#[tokio::main]
async fn main() {
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!(
                "{e}\nusage: knn_bench --url http://127.0.0.1:16969 [--dim 32] [--n 10000,100000] [--queries 300] \
                 [--k 10] [--batch 500] [--concurrency 8] [--clusters 64] [--extrapolate 10000000] [--filtered] [--settle-secs 0] [--drain-secs 0] [--seed N] [--query-space <expert_space_id>]"
            );
            std::process::exit(2);
        }
    };
    let hub = HubClient::with_timeout(&args.url, Duration::from_secs(120));
    if let Some(space) = &args.query_space {
        query_only(&hub, &args, space).await;
        return;
    }
    hub.ensure_schema_ready(Duration::from_secs(60)).await.expect("schema on scratch helix");
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let run = format!("bench-{secs}-{}", std::process::id());
    let snap = hub
        .add_snapshot(&NewSnapshot {
            snapshot_id: run.clone(),
            timescale: "1m".into(),
            encoder_family: "bench".into(),
            generation: 0,
            garage_key: String::new(),
            status: "bench".into(),
        })
        .await
        .expect("snapshot");
    let space_id = format!("{run}-e0");
    hub.add_expert_space(
        snap.id,
        &NewExpertSpace { expert_space_id: space_id.clone(), snapshot_id: run.clone(), index: 0, dim: args.dim as u32 },
    )
    .await
    .expect("expert space");
    println!("expert space {space_id}");
    let mut rng = Rng(args.seed.max(1));
    let mix = Mixture::new(&mut rng, args.clusters, args.dim);
    let mut stored: Vec<Vec<f32>> = Vec::with_capacity(*args.sizes.last().unwrap_or(&0));
    let filter = if args.filtered { KnnFilter { kind: Some("observation".into()), ..Default::default() } } else { KnnFilter::default() };

    println!(
        "dim={} k={} queries={} batch={} concurrency={} clusters={} filtered={}",
        args.dim, args.k, args.queries, args.batch, args.concurrency, args.clusters, args.filtered
    );
    println!(
        "{:>9} {:>10} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "n", "insert/s", "drain s", "p50 ms", "p95 ms", "p99 ms", "max ms", "recall"
    );
    let mut rows: Vec<(f64, f64)> = Vec::new();
    for &target in &args.sizes {
        // grow to `target`
        let t0 = Instant::now();
        let added = target.saturating_sub(stored.len());
        let base = stored.len();
        let fresh: Vec<Vec<f32>> = (0..added).map(|_| mix.draw(&mut rng)).collect();
        let chunks: Vec<Vec<NewLatent>> = fresh
            .chunks(args.batch)
            .enumerate()
            .map(|(ci, c)| {
                c.iter()
                    .enumerate()
                    .map(|(i, v)| NewLatent {
                        vector: v.clone(),
                        expert_space_id: space_id.clone(),
                        kind: "observation".into(),
                        ref_id: format!("r{}", base + ci * args.batch + i),
                        instrument_id: String::new(),
                        ts_ms: 1_700_000_000_000 + (base + ci * args.batch + i) as i64,
                        uncertainty: 0.0,
                    })
                    .collect()
            })
            .collect();
        let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(args.concurrency.max(1)));
        let mut tasks = tokio::task::JoinSet::new();
        for chunk in chunks {
            let (hub, sem) = (hub.clone(), sem.clone());
            tasks.spawn(async move {
                let _p = sem.acquire_owned().await;
                let mut tries = 0;
                loop {
                    match hub.add_latents_checked(&chunk).await {
                        Ok(_) => return,
                        Err(e) if tries < 3 => {
                            tries += 1;
                            eprintln!("insert retry {tries}: {e}");
                            tokio::time::sleep(Duration::from_secs(1)).await;
                        }
                        Err(e) => panic!("insert failed: {e}"),
                    }
                }
            });
        }
        while let Some(r) = tasks.join_next().await {
            r.expect("insert task");
        }
        stored.extend(fresh);
        let rate = added as f64 / t0.elapsed().as_secs_f64().max(1e-9);

        // Helix indexes asynchronously: give the background index work time to drain.
        tokio::time::sleep(Duration::from_secs(args.settle_secs)).await;
        let drained = if args.drain_secs > 0 { drain(&hub, target, args.drain_secs).await } else { None };
        // warm up, then measure sequential queries (one in flight, like a decoder lookup)
        for _ in 0..10 {
            let q = mix.draw(&mut rng);
            let _ = hub.knn(&space_id, &q, args.k, &filter).await;
        }
        let mut lat = Vec::with_capacity(args.queries);
        let (mut hit, mut want) = (0usize, 0usize);
        let recall_every = (args.queries / 30).max(1);
        for qi in 0..args.queries {
            let q = mix.draw(&mut rng);
            let t = Instant::now();
            let hits = hub.knn(&space_id, &q, args.k, &filter).await.expect("knn");
            lat.push(t.elapsed());
            if qi % recall_every == 0 {
                let mut exact: Vec<(f32, usize)> =
                    stored.iter().enumerate().map(|(i, v)| (cosine_dist(&q, v), i)).collect();
                let kk = args.k.min(exact.len());
                exact.select_nth_unstable_by(kk - 1, |a, b| a.0.total_cmp(&b.0));
                let truth: std::collections::HashSet<String> =
                    exact[..kk].iter().map(|(_, i)| format!("r{i}")).collect();
                hit += hits.iter().filter(|h| truth.contains(&h.ref_id)).count();
                want += truth.len();
            }
        }
        lat.sort_unstable();
        let recall = hit as f64 / want.max(1) as f64;
        let p95 = pct(&lat, 0.95);
        println!(
            "{:>9} {:>10.0} {:>9} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.3}",
            target,
            rate,
            drained.map_or("-".to_owned(), |d: f64| format!("{d:.0}")),
            pct(&lat, 0.50),
            p95,
            pct(&lat, 0.99),
            lat.last().map_or(0.0, |d| d.as_secs_f64() * 1e3),
            recall
        );
        rows.push((target as f64, p95));
    }
    if rows.len() >= 2 {
        let xs_log: Vec<f64> = rows.iter().map(|r| r.0.ln()).collect();
        let xs_lin: Vec<f64> = rows.iter().map(|r| r.0).collect();
        let ys: Vec<f64> = rows.iter().map(|r| r.1).collect();
        let (a1, b1, r1) = fit(&xs_log, &ys);
        let (a2, b2, r2) = fit(&xs_lin, &ys);
        let n = args.extrapolate;
        println!(
            "p95 ~ {a1:.3} + {b1:.3}*ln(n)  r2={r1:.3}  -> {:.1} ms at n={n:.0}\n\
             p95 ~ {a2:.3} + {b2:.3e}*n    r2={r2:.3}  -> {:.1} ms at n={n:.0}",
            a1 + b1 * n.ln(),
            a2 + b2 * n
        );
    }
}

/// Latency of an existing space only (no inserts, no recall): re-measures a space after the
/// server finished its background index work.
async fn query_only(hub: &HubClient, args: &Args, space: &str) {
    let n = hub.count_space_latents(space).await.expect("count");
    let mut rng = Rng(args.seed.max(1));
    let mix = Mixture::new(&mut rng, args.clusters, args.dim);
    let filter = KnnFilter::default();
    for _ in 0..10 {
        let _ = hub.knn(space, &mix.draw(&mut rng), args.k, &filter).await;
    }
    let mut lat = Vec::with_capacity(args.queries);
    for _ in 0..args.queries {
        let q = mix.draw(&mut rng);
        let t = Instant::now();
        let hits = hub.knn(space, &q, args.k, &filter).await.expect("knn");
        lat.push(t.elapsed());
        assert!(hits.len() <= args.k);
    }
    lat.sort_unstable();
    println!(
        "{space}: n={n} p50={:.2} p95={:.2} p99={:.2} max={:.2} ms",
        pct(&lat, 0.5),
        pct(&lat, 0.95),
        pct(&lat, 0.99),
        lat.last().map_or(0.0, |d| d.as_secs_f64() * 1e3)
    );
}

/// Waits until Helix's index operation queue is drained: a fresh equality index is queued behind
/// every pending vector-index operation, so the time until it activates is the remaining
/// background index work. `None` when it did not drain within `max_secs`.
async fn drain(hub: &HubClient, step: usize, max_secs: u64) -> Option<f64> {
    let t = Instant::now();
    let receipts = hub.ensure_indexes(vec![IndexSpec::node_equality("BenchProbe", &format!("p{step}"), false)]).await.ok()?;
    hub.wait_for_receipts(&receipts, Duration::from_secs(max_secs)).await.ok()?;
    Some(t.elapsed().as_secs_f64())
}
