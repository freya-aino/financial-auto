//! Hypervolume rung scores and NSGA-III selection (spec 34 §6.2 steps 4 and 6, D-085).
//!
//! Both work on **minimised** objective vectors (see [`crate::objectives::minimised`]) and are the
//! reference for `go/services/evo-orchestrator/pareto.go`; the shared fixture
//! `go/services/evo-orchestrator/testdata/nsga_fixture.json` pins the two to the same numbers.
//!
//! **Normalisation (spec 34 Q6).** Objectives of different units are min–max normalised over the
//! candidate set being ranked ([`normalise`]): finite values map to `[0, 1]`, a constant objective
//! to 0, a missing / non-finite value to the reference coordinate [`HV_REF`] (it adds no volume).
//!
//! **Hypervolume contribution** ([`hv_contributions`]): the volume of the box
//! `[point, HV_REF]^d` that no other member of the same front dominates — exactly for one or two
//! objectives (sweep), by a deterministic Monte-Carlo estimate with [`HV_SAMPLES`] points from a
//! SplitMix64 stream otherwise (the same stream in Go, so both estimate identically).
//!
//! **NSGA-III** ([`nsga3_select`], Deb & Jain 2014): fill by non-dominated fronts; the last front
//! that does not fit is split by reference-direction niching — Das–Dennis directions
//! ([`reference_directions`]) on the simplex, objectives translated by the ideal point and scaled
//! by the nadir of the fronts taken so far (the intercept step's documented fallback), each member
//! associated with the direction of smallest perpendicular distance, then repeatedly the least
//! crowded direction (lowest index on ties) admits its closest unselected member. Deterministic:
//! no random tie breaks.

use crate::{crowding_distance, non_dominated_sort};

/// Reference coordinate of the normalised hypervolume (`[0, 1]` objectives, 10 % margin).
pub const HV_REF: f64 = 1.1;
/// Monte-Carlo samples for three or more objectives.
pub const HV_SAMPLES: usize = 8192;
/// Seed of the Monte-Carlo stream.
pub const HV_SEED: u64 = 0x005e_ed0f_4b1d;
/// Largest number of Das–Dennis directions [`nsga3_select`] generates.
const MAX_DIRECTIONS: usize = 5000;

/// SplitMix64 (Steele et al.), ported verbatim to Go.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` (53 bits).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Min–max normalisation of every objective over `objs` (see the module docs).
pub fn normalise(objs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let d = objs.first().map_or(0, Vec::len);
    let mut lo = vec![f64::INFINITY; d];
    let mut hi = vec![f64::NEG_INFINITY; d];
    for o in objs {
        for k in 0..d {
            let x = o[k];
            if x.is_finite() {
                lo[k] = lo[k].min(x);
                hi[k] = hi[k].max(x);
            }
        }
    }
    objs.iter()
        .map(|o| {
            (0..d)
                .map(|k| {
                    let x = o[k];
                    if !x.is_finite() {
                        HV_REF
                    } else if hi[k] > lo[k] {
                        (x - lo[k]) / (hi[k] - lo[k])
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}

fn weakly_dominates(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y)
}

/// Exclusive hypervolume contribution of each point of `pts` (already normalised; one front)
/// w.r.t. the reference point `HV_REF^d`. Duplicates share their region, so each gets 0.
pub fn hv_contributions(pts: &[Vec<f64>]) -> Vec<f64> {
    let n = pts.len();
    if n == 0 {
        return Vec::new();
    }
    let d = pts[0].len();
    match d {
        0 => vec![0.0; n],
        1 => {
            // the best point owns [x, second best); everyone else nothing
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| pts[a][0].total_cmp(&pts[b][0]));
            let mut c = vec![0.0; n];
            let next = if n > 1 { pts[order[1]][0] } else { HV_REF };
            c[order[0]] = (next.min(HV_REF) - pts[order[0]][0]).max(0.0);
            c
        }
        2 => {
            // sweep by the first objective: the exclusive box of a point is bounded by its
            // neighbours on the (weakly) non-dominated staircase
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| pts[a][0].total_cmp(&pts[b][0]).then(pts[a][1].total_cmp(&pts[b][1])));
            let mut c = vec![0.0; n];
            for (pos, &i) in order.iter().enumerate() {
                let (x, y) = (pts[i][0], pts[i][1]);
                if x >= HV_REF || y >= HV_REF {
                    continue;
                }
                // dominated (weakly, by another point) → no exclusive volume
                if (0..n).any(|j| j != i && weakly_dominates(&pts[j], &pts[i])) {
                    continue;
                }
                let right = order[pos + 1..].iter().map(|&j| pts[j][0]).find(|&v| v > x).unwrap_or(HV_REF).min(HV_REF);
                let up = order[..pos].iter().rev().map(|&j| pts[j][1]).find(|&v| v > y).unwrap_or(HV_REF).min(HV_REF);
                c[i] = (right - x) * (up - y);
            }
            c
        }
        _ => {
            let mut rng = SplitMix64::new(HV_SEED);
            let mut hits = vec![0usize; n];
            let mut s = vec![0.0; d];
            for _ in 0..HV_SAMPLES {
                for v in &mut s {
                    *v = rng.uniform() * HV_REF;
                }
                let mut owner = usize::MAX;
                let mut count = 0;
                for (i, p) in pts.iter().enumerate() {
                    if weakly_dominates(p, &s) {
                        count += 1;
                        owner = i;
                        if count > 1 {
                            break;
                        }
                    }
                }
                if count == 1 {
                    hits[owner] += 1;
                }
            }
            let vol = HV_REF.powi(d as i32);
            hits.into_iter().map(|h| h as f64 / HV_SAMPLES as f64 * vol).collect()
        }
    }
}

/// Normalised points of one front over the objectives **every** member of the front defines
/// (finite raw value): a family without an objective (a reconstruction decoder has no `crps`)
/// would otherwise sit at the reference coordinate and own no volume at all (spec 34 Q6).
pub fn front_points(objs: &[Vec<f64>], norm: &[Vec<f64>], front: &[usize]) -> Vec<Vec<f64>> {
    let d = front.first().map_or(0, |&i| objs[i].len());
    let dims: Vec<usize> = (0..d).filter(|&k| front.iter().all(|&i| objs[i][k].is_finite())).collect();
    front.iter().map(|&i| dims.iter().map(|&k| norm[i][k]).collect()).collect()
}

/// Rung order (spec 34 §6.2 step 4): by non-dominated front, inside a front by hypervolume
/// contribution (descending; normalised over all of `objs`, over the objectives the whole front
/// defines — [`front_points`]), then crowding distance, then the
/// input order. Returns `(order, front rank, hv contribution)` with the last two indexed by
/// candidate.
pub fn hv_rank(objs: &[Vec<f64>]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let norm = normalise(objs);
    let mut rank = vec![0usize; objs.len()];
    let mut hv = vec![0.0; objs.len()];
    let mut order = Vec::with_capacity(objs.len());
    for (r, front) in non_dominated_sort(objs).into_iter().enumerate() {
        let c = hv_contributions(&front_points(objs, &norm, &front));
        let cd = crowding_distance(objs, &front);
        let mut idx: Vec<usize> = (0..front.len()).collect();
        idx.sort_by(|&a, &b| c[b].total_cmp(&c[a]).then(cd[b].total_cmp(&cd[a])));
        for (j, &i) in front.iter().enumerate() {
            rank[i] = r;
            hv[i] = c[j];
        }
        order.extend(idx.into_iter().map(|j| front[j]));
    }
    (order, rank, hv)
}

/// Das–Dennis directions with `h` divisions on the `d`-simplex, in lexicographic order of the
/// integer compositions (first coordinate descending).
pub fn reference_directions(d: usize, h: usize) -> Vec<Vec<f64>> {
    fn rec(d: usize, left: usize, h: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<f64>>) {
        if cur.len() + 1 == d {
            cur.push(left);
            out.push(cur.iter().map(|&x| x as f64 / h as f64).collect());
            cur.pop();
            return;
        }
        for x in (0..=left).rev() {
            cur.push(x);
            rec(d, left - x, h, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    if d == 0 || h == 0 {
        return out;
    }
    rec(d, h, h, &mut Vec::with_capacity(d), &mut out);
    out
}

fn binom(n: usize, k: usize) -> usize {
    let k = k.min(n - k.min(n));
    let mut r: usize = 1;
    for i in 0..k {
        r = r.saturating_mul(n - i) / (i + 1);
    }
    r
}

/// Divisions for `k` survivors in `d` objectives: the smallest `h ≥ 1` with at least `k`
/// directions, capped at [`MAX_DIRECTIONS`].
pub fn divisions_for(d: usize, k: usize) -> usize {
    let mut h = 1;
    while binom(h + d - 1, d - 1) < k && binom(h + 1 + d - 1, d - 1) <= MAX_DIRECTIONS {
        h += 1;
    }
    h
}

/// NSGA-III survivor selection of `k` indices (see the module docs). The fronts that fit come
/// first in front order; the split front's admitted members follow in admission order.
pub fn nsga3_select(objs: &[Vec<f64>], k: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(k);
    let fronts = non_dominated_sort(objs);
    let mut last: Option<Vec<usize>> = None;
    for front in fronts {
        if out.len() + front.len() <= k {
            out.extend(&front);
            if out.len() == k {
                return out;
            }
        } else {
            last = Some(front);
            break;
        }
    }
    let Some(last) = last else { return out };
    let d = objs[last[0]].len();
    if d == 0 {
        out.extend(last.into_iter().take(k - out.len()));
        return out;
    }
    // normalisation over S_t = selected ∪ last front (finite values only; ∞ → 1 beyond nadir)
    let pool: Vec<usize> = out.iter().chain(&last).copied().collect();
    let mut ideal = vec![f64::INFINITY; d];
    let mut nadir = vec![f64::NEG_INFINITY; d];
    for &i in &pool {
        for m in 0..d {
            let x = objs[i][m];
            if x.is_finite() {
                ideal[m] = ideal[m].min(x);
                nadir[m] = nadir[m].max(x);
            }
        }
    }
    let norm = |i: usize| -> Vec<f64> {
        (0..d)
            .map(|m| {
                let x = objs[i][m];
                if !x.is_finite() {
                    HV_REF
                } else if nadir[m] > ideal[m] {
                    (x - ideal[m]) / (nadir[m] - ideal[m])
                } else {
                    0.0
                }
            })
            .collect()
    };
    let dirs = reference_directions(d, divisions_for(d, k));
    let units: Vec<Vec<f64>> = dirs
        .iter()
        .map(|w| {
            let n = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            w.iter().map(|x| x / n).collect()
        })
        .collect();
    // association: (direction, perpendicular distance)
    let associate = |p: &[f64]| -> (usize, f64) {
        let mut best = (0usize, f64::INFINITY);
        for (j, u) in units.iter().enumerate() {
            let t: f64 = p.iter().zip(u).map(|(a, b)| a * b).sum();
            let dist2: f64 = p.iter().zip(u).map(|(a, b)| (a - t * b).powi(2)).sum();
            let dist = dist2.max(0.0).sqrt();
            if dist < best.1 {
                best = (j, dist);
            }
        }
        best
    };
    let mut niche = vec![0usize; units.len()];
    for &i in &out {
        niche[associate(&norm(i)).0] += 1;
    }
    let mut cand: Vec<(usize, usize, f64)> = last
        .iter()
        .map(|&i| {
            let (j, dist) = associate(&norm(i));
            (i, j, dist)
        })
        .collect();
    let mut excluded = vec![false; units.len()];
    while out.len() < k && !cand.is_empty() {
        // least crowded direction that still has candidates (lowest index on ties)
        let Some(j) = (0..units.len())
            .filter(|&j| !excluded[j])
            .filter(|&j| cand.iter().any(|c| c.1 == j))
            .min_by_key(|&j| (niche[j], j))
        else {
            break;
        };
        // closest member (input order on ties) — NSGA-III takes the closest when the niche is
        // empty and an arbitrary one otherwise; the closest keeps it deterministic
        let pos = cand
            .iter()
            .enumerate()
            .filter(|(_, c)| c.1 == j)
            .min_by(|a, b| a.1.2.total_cmp(&b.1.2).then(a.0.cmp(&b.0)))
            .map(|(p, _)| p);
        match pos {
            Some(p) => {
                out.push(cand.remove(p).0);
                niche[j] += 1;
            }
            None => excluded[j] = true,
        }
    }
    out
}

/// Survivor selection of spec 34 §6.2 step 6: NSGA-II for at most three objectives,
/// NSGA-III for four or more.
pub fn select(objs: &[Vec<f64>], k: usize) -> Vec<usize> {
    if objs.first().map_or(0, Vec::len) <= 3 { crate::nsga2_select(objs, k) } else { nsga3_select(objs, k) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_reference_values() {
        // published SplitMix64 test vector for seed 0 (first two outputs)
        let mut r = SplitMix64::new(0);
        assert_eq!(r.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(r.next_u64(), 0x6e78_9e6a_a1b9_65f4);
    }

    #[test]
    fn two_d_contributions_are_exact() {
        // staircase (0,1) (0.5,0.5) (1,0) with ref 1.1
        let pts = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let c = hv_contributions(&pts);
        assert!((c[0] - 0.5 * 0.1).abs() < 1e-12, "{c:?}");
        assert!((c[1] - 0.5 * 0.5).abs() < 1e-12, "{c:?}");
        assert!((c[2] - 0.1 * 0.5).abs() < 1e-12, "{c:?}");
        // a duplicate owns nothing exclusively
        let c = hv_contributions(&[vec![0.2, 0.2], vec![0.2, 0.2]]);
        assert_eq!(c, vec![0.0, 0.0]);
    }

    #[test]
    fn monte_carlo_matches_exact_box_in_3d() {
        // a single point at the origin owns the whole reference box
        let c = hv_contributions(&[vec![0.0, 0.0, 0.0]]);
        assert!((c[0] - HV_REF.powi(3)).abs() < 1e-12);
        // a knee point owns more than an extreme one
        let pts = vec![vec![0.0, 1.0, 1.0], vec![0.3, 0.3, 0.3], vec![1.0, 0.0, 1.0], vec![1.0, 1.0, 0.0]];
        let c = hv_contributions(&pts);
        assert!(c[1] > c[0] && c[1] > c[2] && c[1] > c[3], "{c:?}");
        assert!(c.iter().all(|x| *x > 0.0));
    }

    #[test]
    fn hv_rank_prefers_the_knee_inside_a_front() {
        let objs = vec![vec![0.0, 10.0], vec![4.0, 4.0], vec![10.0, 0.0], vec![9.0, 9.0], vec![1.0, 9.5]];
        let (order, rank, hv) = hv_rank(&objs);
        assert_eq!(rank, vec![0, 0, 0, 1, 0]);
        assert_eq!(order[0], 1, "{order:?} {hv:?}");
        assert_eq!(*order.last().unwrap(), 3);
    }

    #[test]
    fn das_dennis_counts() {
        assert_eq!(reference_directions(3, 4).len(), 15);
        assert_eq!(reference_directions(8, 2).len(), 36);
        assert!(reference_directions(3, 4).iter().all(|w| (w.iter().sum::<f64>() - 1.0).abs() < 1e-12));
        assert_eq!(divisions_for(8, 20), 2);
        assert_eq!(divisions_for(3, 15), 4);
    }

    #[test]
    fn nsga3_spreads_over_directions() {
        // one front of 4-objective points: three along the same reference direction, one along each other
        let objs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.99, 0.01, 0.0, 0.0],
            vec![0.98, 0.02, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let sel = nsga3_select(&objs, 4);
        assert_eq!(sel.len(), 4);
        for c in [3, 4, 5] {
            assert!(sel.contains(&c), "{sel:?}");
        }
        // dominated points are never preferred over the front
        let mut objs2 = objs.clone();
        objs2.push(vec![2.0, 2.0, 2.0, 2.0]);
        assert!(!nsga3_select(&objs2, 6).contains(&6));
        assert_eq!(select(&objs2, 7).len(), 7);
        assert_eq!(select(&[vec![1.0, 2.0], vec![2.0, 1.0]], 1), crate::nsga2_select(&[vec![1.0, 2.0], vec![2.0, 1.0]], 1));
    }
}

/// Shared fixture with `go/services/evo-orchestrator/pareto_test.go` (key `pareto` of
/// `nsga_fixture.json`). `FA_PRINT_PARETO_FIXTURE=1` prints the cases computed here.
#[cfg(test)]
mod fixture {
    use serde::{Deserialize, Serialize};

    use super::*;

    #[derive(Serialize, Deserialize)]
    struct Select {
        k: usize,
        expected: Vec<usize>,
    }

    #[derive(Serialize, Deserialize)]
    struct Case {
        name: String,
        objectives: Vec<Vec<serde_json::Value>>,
        hv_order: Vec<usize>,
        hv_rank: Vec<usize>,
        hv: Vec<f64>,
        nsga3: Vec<Select>,
    }

    fn num(v: &serde_json::Value) -> f64 {
        match v {
            serde_json::Value::String(s) if s == "inf" => f64::INFINITY,
            other => other.as_f64().expect("number"),
        }
    }

    fn inputs() -> Vec<(&'static str, Vec<Vec<f64>>)> {
        let inf = f64::INFINITY;
        vec![
            ("two_objectives", vec![vec![1.0, 4.0], vec![2.0, 2.0], vec![4.0, 1.0], vec![3.0, 3.0], vec![2.0, 2.0]]),
            (
                "three_objectives",
                vec![
                    vec![0.1, 0.9, 0.5],
                    vec![0.5, 0.5, 0.5],
                    vec![0.9, 0.1, 0.4],
                    vec![0.3, 0.3, 0.9],
                    vec![0.6, 0.6, 0.6],
                    vec![0.2, 0.8, 0.1],
                ],
            ),
            (
                "decoder_objectives_with_missing",
                // compute_cost, coverage_err, crps, -crps_skill, -hit_rate, -recon_skill, -trade_psr, -trade_sharpe
                vec![
                    vec![3.9, 0.05, 0.0021, -0.37, -0.52, 0.0, -0.55, -0.4],
                    vec![4.1, 0.02, 0.0023, -0.33, -0.51, 0.0, -0.6, -0.9],
                    vec![3.87, inf, inf, 0.0, -0.5, 0.002, -0.5, 0.0],
                    vec![4.0, 0.08, 0.0022, -0.34, -0.55, -0.3, -0.52, -0.2],
                    vec![3.95, inf, inf, 0.0, -0.5, -0.07, -0.5, 0.0],
                    vec![5.2, 0.1, 0.0030, -0.1, -0.49, 0.0, -0.45, 0.3],
                    vec![3.9, 0.05, 0.0021, -0.37, -0.52, 0.0, -0.55, -0.4],
                    vec![4.4, 0.03, 0.0025, -0.3, -0.53, -0.1, -0.58, -0.5],
                ],
            ),
            (
                "four_objectives_one_front",
                vec![
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![0.99, 0.01, 0.0, 0.0],
                    vec![0.98, 0.02, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0, 0.0],
                    vec![0.0, 0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 0.0, 1.0],
                    vec![0.5, 0.5, 0.5, 0.5],
                ],
            ),
        ]
    }

    fn compute(name: &str, objs: &[Vec<f64>]) -> Case {
        let (hv_order, hv_rank, hv) = hv_rank(objs);
        let nsga3 = (1..=objs.len()).map(|k| Select { k, expected: nsga3_select(objs, k) }).collect();
        Case {
            name: name.into(),
            objectives: objs
                .iter()
                .map(|r| r.iter().map(|x| if x.is_finite() { serde_json::json!(x) } else { serde_json::json!("inf") }).collect())
                .collect(),
            hv_order,
            hv_rank,
            hv,
            nsga3,
        }
    }

    #[test]
    fn matches_shared_pareto_fixture() {
        if std::env::var("FA_PRINT_PARETO_FIXTURE").is_ok() {
            let cases: Vec<Case> = inputs().iter().map(|(n, o)| compute(n, o)).collect();
            println!("{}", serde_json::to_string(&cases).unwrap());
        }
        #[derive(Deserialize)]
        struct Fixture {
            pareto: Vec<Case>,
        }
        let path =
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../go/services/evo-orchestrator/testdata/nsga_fixture.json");
        let fx: Fixture = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert!(fx.pareto.len() >= 4);
        for c in &fx.pareto {
            let objs: Vec<Vec<f64>> = c.objectives.iter().map(|r| r.iter().map(num).collect()).collect();
            let got = compute(&c.name, &objs);
            assert_eq!(got.hv_order, c.hv_order, "{}: hv order", c.name);
            assert_eq!(got.hv_rank, c.hv_rank, "{}: fronts", c.name);
            for (a, b) in got.hv.iter().zip(&c.hv) {
                assert!((a - b).abs() < 1e-12, "{}: hv {:?} != {:?}", c.name, got.hv, c.hv);
            }
            for (g, w) in got.nsga3.iter().zip(&c.nsga3) {
                assert_eq!(g.expected, w.expected, "{}: nsga3 k={}", c.name, w.k);
            }
        }
    }
}
