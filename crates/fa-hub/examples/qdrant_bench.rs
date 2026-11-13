//! Qdrant k-NN benchmark (ADR 0010): the same workload as `knn_bench` (Helix) and `knn_floor`
//! (in-process), against a scratch Qdrant over its REST API.
//!
//! Data: Gaussian mixture (64 clusters, σ 0.35), point `i` belongs to expert space `s<i % spaces>`
//! and has `ts = T0 + i` ms, payload `{space_id, kind: "observation", ts}`. Two layouts:
//! - `--layout shared`: one collection per dimension, `space_id` keyword index (`is_tenant`) and
//!   `ts` integer index — the Helix model (one `vec<d>` index, tenant-partitioned);
//! - `--layout per-space`: one collection per space (`ts` integer index only).
//!
//! Measured per size step (one query in flight, like a decoder lookup): upsert rate (wait=true
//! batches, `--concurrency` in flight), seconds until the collection is green (index built),
//! p50/p95/p99 of `search` with no filter, with the space filter, and with space + `ts <= as_of`
//! (as-of at `--asof-frac` of the time range), each with recall@k against an exact client scan.
//! ```sh
//! docker run -d --name fa-bench-qdrant --network fa-bench-net -v <scratch>:/qdrant/storage qdrant/qdrant:v1.20.0
//! qdrant_bench --url http://fa-bench-qdrant:6333 --dim 32 --n 50000,1000000 --layout shared
//! ```
use std::time::{Duration, Instant};

use rayon::prelude::*;
use serde_json::{Value, json};

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

fn draw(rng: &mut Rng, centers: &[Vec<f32>]) -> Vec<f32> {
    let c = &centers[(rng.u() * centers.len() as f64) as usize % centers.len()];
    let mut v: Vec<f32> = c.iter().map(|x| x + 0.35 * rng.normal() as f32).collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
    v
}

const T0: i64 = 1_700_000_000_000;

#[derive(Clone)]
struct Q {
    http: reqwest::Client,
    url: String,
}
impl Q {
    async fn call(&self, method: reqwest::Method, path: &str, body: Option<Value>) -> Value {
        let mut req = self.http.request(method, format!("{}{path}", self.url));
        if let Some(b) = body {
            req = req.json(&b);
        }
        let resp = req.send().await.expect("qdrant request");
        let status = resp.status();
        let v: Value = resp.json().await.unwrap_or(Value::Null);
        assert!(status.is_success(), "{path}: {status} {v}");
        v
    }
}

fn pct(sorted: &[Duration], p: f64) -> f64 {
    sorted[((sorted.len() as f64 - 1.0) * p).round() as usize].as_secs_f64() * 1e3
}

fn exact(data: &[f32], dim: usize, n: usize, q: &[f32], k: usize, keep: &(dyn Fn(usize) -> bool + Sync)) -> Vec<u64> {
    let mut all: Vec<(f32, usize)> = (0..n)
        .into_par_iter()
        .filter(|&i| keep(i))
        .map(|i| (1.0 - data[i * dim..(i + 1) * dim].iter().zip(q).map(|(a, b)| a * b).sum::<f32>(), i))
        .collect();
    let kk = k.min(all.len());
    if kk == 0 {
        return vec![];
    }
    all.select_nth_unstable_by(kk - 1, |a, b| a.0.total_cmp(&b.0));
    all[..kk].iter().map(|x| x.1 as u64).collect()
}

#[tokio::main]
async fn main() {
    let (mut url, mut dim, mut sizes, mut queries, mut k) =
        (String::new(), 32usize, vec![50_000usize], 300usize, 10usize);
    let (mut spaces, mut batch, mut conc, mut layout, mut asof_frac) =
        (8usize, 1000usize, 8usize, "shared".to_owned(), 0.7f64);
    let mut hnsw_ef: Option<usize> = None;
    let mut skip_load = false;
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        let mut v = || it.next().expect("flag value").replace('_', "");
        match f.as_str() {
            "--url" => url = v(),
            "--dim" => dim = v().parse().expect("dim"),
            "--n" => sizes = v().split(',').map(|s| s.parse().expect("n")).collect(),
            "--queries" => queries = v().parse().expect("queries"),
            "--k" => k = v().parse().expect("k"),
            "--spaces" => spaces = v().parse().expect("spaces"),
            "--batch" => batch = v().parse().expect("batch"),
            "--concurrency" => conc = v().parse().expect("concurrency"),
            "--layout" => layout = v(),
            "--asof-frac" => asof_frac = v().parse().expect("asof-frac"),
            "--hnsw-ef" => hnsw_ef = Some(v().parse().expect("hnsw-ef")),
            // query an existing collection again (no create, no upsert): warm re-measurement
            "--skip-load" => skip_load = true,
            other => panic!("unknown flag {other}"),
        }
    }
    assert!(!url.is_empty(), "--url required (scratch qdrant)");
    assert!(!url.contains(":6333") || url.contains("fa-bench"), "only a fa-bench-* scratch qdrant");
    sizes.sort_unstable();
    let per_space = layout == "per-space";
    let q = Q { http: reqwest::Client::builder().timeout(Duration::from_secs(600)).build().expect("client"), url };
    let coll = |s: usize| if per_space { format!("bench_d{dim}_s{s}") } else { format!("bench_d{dim}") };
    let colls: Vec<String> = if per_space { (0..spaces).map(coll).collect() } else { vec![coll(0)] };
    for c in colls.iter().filter(|_| !skip_load) {
        let _ = q.http.delete(format!("{}/collections/{c}", q.url)).send().await;
        q.call(
            reqwest::Method::PUT,
            &format!("/collections/{c}"),
            Some(
                json!({"vectors": {"size": dim, "distance": "Cosine"}, "hnsw_config": {"m": 16, "ef_construct": 100}}),
            ),
        )
        .await;
        if !per_space {
            q.call(
                reqwest::Method::PUT,
                &format!("/collections/{c}/index?wait=true"),
                Some(json!({"field_name": "space_id", "field_schema": {"type": "keyword", "is_tenant": true}})),
            )
            .await;
        }
        q.call(
            reqwest::Method::PUT,
            &format!("/collections/{c}/index?wait=true"),
            Some(json!({"field_name": "ts", "field_schema": {"type": "integer", "range": true, "lookup": false}})),
        )
        .await;
    }
    println!(
        "layout={layout} dim={dim} k={k} spaces={spaces} queries={queries} batch={batch} concurrency={conc} asof_frac={asof_frac} hnsw_ef={hnsw_ef:?}"
    );
    println!(
        "{:>9} {:>9} {:>7} | {:>7} {:>7} {:>6} | {:>7} {:>7} {:>6} | {:>7} {:>7} {:>7} {:>6}",
        "n", "upsert/s", "green s", "all p50", "p95", "rec", "spc p50", "p95", "rec", "asof p50", "p95", "p99", "rec"
    );
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let centers: Vec<Vec<f32>> = (0..64).map(|_| (0..dim).map(|_| rng.normal() as f32).collect()).collect();
    let mut data: Vec<f32> = Vec::new();
    let mut have = 0usize;
    for &n in &sizes {
        let t0 = Instant::now();
        let base = have;
        while data.len() < n * dim {
            data.extend(draw(&mut rng, &centers));
        }
        let data_ref = std::sync::Arc::new(data.clone());
        let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(conc));
        let mut tasks = tokio::task::JoinSet::new();
        // per-space layout: group each batch by space so every request goes to one collection
        let mut start = if skip_load { n } else { base };
        while start < n {
            let end = (start + batch * if per_space { spaces } else { 1 }).min(n);
            for s in 0..if per_space { spaces } else { 1 } {
                let (q, sem, d, c) = (q.clone(), sem.clone(), data_ref.clone(), coll(s));
                let ids: Vec<usize> = (start..end).filter(|i| !per_space || i % spaces == s).collect();
                tasks.spawn(async move {
                    let _p = sem.acquire_owned().await;
                    let points: Vec<Value> = ids
                        .iter()
                        .map(|&i| {
                            json!({"id": i as u64, "vector": &d[i * dim..(i + 1) * dim],
                                   "payload": {"space_id": format!("s{}", i % spaces), "kind": "observation", "ts": T0 + i as i64}})
                        })
                        .collect();
                    q.call(reqwest::Method::PUT, &format!("/collections/{c}/points?wait=true"), Some(json!({"points": points}))).await;
                });
            }
            start = end;
        }
        while let Some(r) = tasks.join_next().await {
            r.expect("upsert task");
        }
        let up_secs = t0.elapsed().as_secs_f64();
        have = n;
        // wait for green (optimizers + HNSW built)
        let tg = Instant::now();
        loop {
            let mut green = true;
            for c in &colls {
                let info = q.call(reqwest::Method::GET, &format!("/collections/{c}"), None).await;
                if info["result"]["status"] != "green" {
                    green = false;
                }
            }
            if green {
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        let green_secs = tg.elapsed().as_secs_f64();
        let as_of = T0 + (asof_frac * n as f64) as i64;
        // three query shapes; space 0 is queried
        let shapes: [(&str, Option<Value>); 3] = [
            ("all", None),
            (
                "space",
                if per_space { None } else { Some(json!({"must": [{"key": "space_id", "match": {"value": "s0"}}]})) },
            ),
            (
                "asof",
                Some(if per_space {
                    json!({"must": [{"key": "ts", "range": {"lte": as_of}}]})
                } else {
                    json!({"must": [{"key": "space_id", "match": {"value": "s0"}}, {"key": "ts", "range": {"lte": as_of}}]})
                }),
            ),
        ];
        let mut cols: Vec<String> = Vec::new();
        for (name, filter) in &shapes {
            let target = coll(0);
            let keep: Box<dyn Fn(usize) -> bool + Sync> = match (*name, per_space) {
                ("all", false) => Box::new(|_| true),
                ("asof", _) => Box::new(move |i| i % spaces == 0 && T0 + i as i64 <= as_of),
                _ => Box::new(move |i| i % spaces == 0),
            };
            let mut lat = Vec::with_capacity(queries);
            let (mut hit, mut want) = (0usize, 0usize);
            for qi in 0..queries + 10 {
                let v = draw(&mut rng, &centers);
                let mut body = json!({"vector": v, "limit": k, "with_payload": false});
                if let Some(f) = filter {
                    body["filter"] = f.clone();
                }
                if let Some(ef) = hnsw_ef {
                    body["params"] = json!({"hnsw_ef": ef});
                }
                let t = Instant::now();
                let r =
                    q.call(reqwest::Method::POST, &format!("/collections/{target}/points/search"), Some(body)).await;
                let el = t.elapsed();
                if qi < 10 {
                    continue; // warm-up
                }
                lat.push(el);
                if qi % 3 == 0 {
                    let truth = exact(&data, dim, n, &v, k, &*keep);
                    let got: Vec<u64> =
                        r["result"].as_array().map_or(vec![], |a| a.iter().filter_map(|h| h["id"].as_u64()).collect());
                    hit += got.iter().filter(|g| truth.contains(g)).count();
                    want += truth.len();
                }
            }
            lat.sort_unstable();
            let rec = hit as f64 / want.max(1) as f64;
            if *name == "asof" {
                cols.push(format!(
                    "{:>7.2} {:>7.2} {:>7.2} {:>6.3}",
                    pct(&lat, 0.5),
                    pct(&lat, 0.95),
                    pct(&lat, 0.99),
                    rec
                ));
            } else {
                cols.push(format!("{:>7.2} {:>7.2} {:>6.3}", pct(&lat, 0.5), pct(&lat, 0.95), rec));
            }
        }
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        println!(
            "{:>9} {:>9.0} {:>7.1} | {} | {} | {}   (unix {now})",
            n,
            (n - base) as f64 / up_secs,
            green_secs,
            cols[0],
            cols[1],
            cols[2]
        );
        let m =
            q.http.get(format!("{}/metrics", q.url)).send().await.expect("metrics").text().await.unwrap_or_default();
        for l in m.lines().filter(|l| l.starts_with("memory_resident_bytes") || l.starts_with("memory_allocated_bytes"))
        {
            println!("    {l}");
        }
    }
}
