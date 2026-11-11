//! Batch refits of the richer kernels on `jobs.mcmc.refit` (spec 33 §6.2, §6.3 step 4).
//!
//! The job contract (owner: model-runtime enqueues, its `mcmc` worker task runs it; any pull
//! worker could, the computation is this pure function):
//!
//! 1. model-runtime keeps per expert space a bounded window of [`ObsRecord`]s (asset, previous
//!    region, from, to, dwell, covariates). At `batch_refit_interval` it uploads an [`ObsDump`]
//!    (window + serving Dirichlet graph) to Garage `fa-artifacts/mcmc/<kernel_id>/obs/<job>.json`
//!    and publishes `Job{kind: "mcmc.refit", spec: RefitSpec}` on `jobs.mcmc.refit`
//!    (work queue `JOBS_MCMC`).
//! 2. The worker calls [`run`]: **NUTS** ([`crate::nuts`]) for the logistic-normal
//!    covariate kernel (one independent posterior per source row, several chains) and **Gibbs**
//!    for the hierarchical Dirichlet across assets (Antoniak table counts, Dirichlet `β`,
//!    Escobar–West concentration). Divergence diagnostics (split R-hat, ESS, divergent
//!    transitions) decide per row / per fit whether the refit is accepted; rejected parts keep
//!    the previous posterior (spec 33 §8).
//! 3. The posterior samples go to Garage `fa-artifacts/mcmc/<kernel_id>/<ts>.bin`
//!    ([`encode`] / [`decode`]) and `JobResult{output_refs: [that key]}` is published on
//!    `jobs.result.mcmc.refit`; model-runtime loads the samples and hands them to the arena
//!    ([`crate::kernel::KernelSampler::apply_refit`]), where the refitted kernels compete with
//!    `conjugate` / `smc` on calibration and cost.

use std::collections::{BTreeMap, HashMap};

use fa_core::rng::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    MarkovGraph, RegionKey,
    nuts::{self, Chain, Diagnostics, LogDensity, NutsCfg},
};

/// Covariates of the logistic-normal kernel, in [`crate::rich::KernelCtx::x`] order.
pub const COVARIATES: [&str; 4] = ["ln_dwell", "vol_ratio", "novelty", "router_weight"];
pub const COV_DIM: usize = COVARIATES.len();
/// Pooled successor bucket of a fitted row (spread over the remaining successors at prediction).
pub const OTHER: &str = "~other";
/// Job kind and subject token.
pub const JOB_KIND: &str = "mcmc.refit";

/// One observed transition with its context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObsRecord {
    pub t_ms: i64,
    pub instrument: String,
    #[serde(default)]
    pub prev: Option<RegionKey>,
    pub from: RegionKey,
    pub to: RegionKey,
    pub dwell: u32,
    pub x: [f32; COV_DIM],
}

/// What the worker reads from Garage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObsDump {
    pub kernel_id: String,
    pub created_ms: i64,
    /// The serving Dirichlet evidence: prior means of the bias terms and the Gibbs base measure.
    pub graph: MarkovGraph,
    pub records: Vec<ObsRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RefitCfg {
    pub nuts: NutsCfg,
    pub chains: usize,
    /// Rows refitted per job (most observed first).
    pub max_rows: usize,
    /// Most recent observations per row used by NUTS.
    pub max_obs_per_row: usize,
    pub min_row_obs: usize,
    /// Named successors per row (the rest pool into [`OTHER`]).
    pub max_successors: usize,
    /// Posterior draws kept per row.
    pub keep_draws: usize,
    pub prior_sd_bias: f64,
    pub prior_sd_weight: f64,
    pub gibbs_burn: usize,
    pub gibbs_iters: usize,
    /// NUTS acceptance: R-hat ceiling, divergent share, minimum ESS.
    pub max_rhat: f64,
    pub max_div_share: f64,
    pub min_ess: f64,
    /// Gibbs acceptance R-hat ceiling (concentration and largest β of each row).
    pub gibbs_max_rhat: f64,
    /// Worker threads for the per-row NUTS fits.
    pub threads: usize,
}

impl Default for RefitCfg {
    fn default() -> Self {
        Self {
            nuts: NutsCfg { warmup: 100, samples: 100, max_depth: 5, ..NutsCfg::default() },
            chains: 2,
            max_rows: 48,
            max_obs_per_row: 400,
            min_row_obs: 30,
            max_successors: 8,
            keep_draws: 100,
            prior_sd_bias: 1.0,
            prior_sd_weight: 1.0,
            gibbs_burn: 200,
            gibbs_iters: 400,
            max_rhat: 1.05,
            max_div_share: 0.01,
            min_ess: 50.0,
            gibbs_max_rhat: 1.1,
            threads: 2,
        }
    }
}

/// `Job.spec` of `mcmc.refit`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefitSpec {
    pub kernel_id: String,
    pub snapshot_id: String,
    pub timescale: String,
    pub expert_index: u32,
    /// Garage key (bucket `fa-artifacts`) of the [`ObsDump`].
    pub obs_ref: String,
    /// `logit_cov`, `hier_asset`.
    pub kernels: Vec<String>,
    pub seed: u64,
    #[serde(default)]
    pub cfg: RefitCfg,
}

/// One fitted logistic row: `draws[s][k · (1 + COV_DIM) + d]`, `d = 0` the bias.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LogitRow {
    pub from: RegionKey,
    pub keys: Vec<RegionKey>,
    pub n_obs: usize,
    pub x_mean: [f64; COV_DIM],
    pub x_sd: [f64; COV_DIM],
    pub diag: Diagnostics,
    pub step_size: f64,
    pub accepted: bool,
    #[serde(skip)]
    pub draws: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HierRow {
    pub from: RegionKey,
    pub keys: Vec<RegionKey>,
    pub beta_mean: Vec<f64>,
    pub rhat: f64,
    #[serde(skip)]
    pub beta_draws: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HierFit {
    pub rows: Vec<HierRow>,
    pub assets: usize,
    pub conc_mean: f64,
    pub conc_draws: Vec<f64>,
    pub conc_rhat: f64,
    pub conc_ess: f64,
    pub max_rhat: f64,
    pub accepted: bool,
}

/// Everything a refit produced (dumped to Garage as `.bin`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RefitOutcome {
    pub kernel_id: String,
    pub snapshot_id: String,
    pub timescale: String,
    pub expert_index: u32,
    pub created_ms: i64,
    pub records: usize,
    pub logit: Vec<LogitRow>,
    pub hier: Option<HierFit>,
    pub grad_evals: u64,
    pub cost_ms: f64,
    /// Garage key of this outcome (set by the worker after upload).
    #[serde(default)]
    pub samples_ref: String,
}

impl RefitOutcome {
    /// Compact summary for `JobResult.result` and logs.
    pub fn summary(&self) -> serde_json::Value {
        serde_json::json!({
            "kernel_id": self.kernel_id, "records": self.records, "cost_ms": self.cost_ms,
            "grad_evals": self.grad_evals,
            "logit_rows": self.logit.len(),
            "logit_accepted": self.logit.iter().filter(|r| r.accepted).count(),
            "logit_max_rhat": self.logit.iter().map(|r| r.diag.max_rhat).fold(0.0, f64::max),
            "logit_divergences": self.logit.iter().map(|r| r.diag.divergences).sum::<usize>(),
            "hier_accepted": self.hier.as_ref().is_some_and(|h| h.accepted),
            "hier_conc": self.hier.as_ref().map(|h| h.conc_mean),
            "hier_max_rhat": self.hier.as_ref().map(|h| h.max_rhat),
            "samples_ref": self.samples_ref,
        })
    }
}

/// Successor keys of a row: the `max` most observed plus [`OTHER`].
fn row_keys(counts: &BTreeMap<&str, usize>, max: usize) -> Vec<RegionKey> {
    let mut v: Vec<(&str, usize)> = counts.iter().map(|(k, c)| (*k, *c)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    let mut keys: Vec<RegionKey> = v.into_iter().take(max.max(1)).map(|x| x.0.to_owned()).collect();
    keys.push(OTHER.to_owned());
    keys
}

fn key_index(keys: &[RegionKey], k: &str) -> usize {
    keys.iter().position(|x| x == k).unwrap_or(keys.len() - 1)
}

/// Base-measure probabilities of a row's keys from the Dirichlet graph (OTHER = the rest).
fn base_probs(g: &MarkovGraph, from: &str, keys: &[RegionKey]) -> Vec<f64> {
    let pred = g.predictive(from);
    let mut out = vec![0.0; keys.len()];
    for (k, p) in pred {
        out[key_index(keys, &k)] += p;
    }
    let s: f64 = out.iter().sum();
    out.iter().map(|p| (p / s.max(1e-12)).max(1e-4)).collect()
}

/// The logistic-normal posterior of one row.
struct LogitTarget {
    k: usize,
    xs: Vec<[f64; COV_DIM + 1]>,
    ys: Vec<usize>,
    mu_bias: Vec<f64>,
    sd_bias: f64,
    sd_w: f64,
}

const P: usize = COV_DIM + 1;

impl LogDensity for LogitTarget {
    fn dim(&self) -> usize {
        self.k * P
    }
    fn logp_grad(&self, th: &[f64], g: &mut [f64]) -> f64 {
        let mut lp = 0.0;
        for (k, mu) in self.mu_bias.iter().enumerate() {
            let b = th[k * P];
            lp -= 0.5 * ((b - mu) / self.sd_bias).powi(2);
            g[k * P] = -(b - mu) / (self.sd_bias * self.sd_bias);
            for d in 1..P {
                let w = th[k * P + d];
                lp -= 0.5 * (w / self.sd_w).powi(2);
                g[k * P + d] = -w / (self.sd_w * self.sd_w);
            }
        }
        let mut z = vec![0.0; self.k];
        for (x, &y) in self.xs.iter().zip(&self.ys) {
            let mut m = f64::NEG_INFINITY;
            for (k, zk) in z.iter_mut().enumerate() {
                *zk = (0..P).map(|d| th[k * P + d] * x[d]).sum();
                m = m.max(*zk);
            }
            let s: f64 = z.iter().map(|v| (v - m).exp()).sum();
            let lse = m + s.ln();
            lp += z[y] - lse;
            for (k, zk) in z.iter().enumerate() {
                let resid = f64::from(u8::from(k == y)) - (zk - lse).exp();
                for d in 0..P {
                    g[k * P + d] += resid * x[d];
                }
            }
        }
        lp
    }
}

/// NUTS fit of one row. `obs` are the row's records (most recent last).
fn fit_logit_row(g: &MarkovGraph, from: &str, obs: &[&ObsRecord], cfg: &RefitCfg, seed: u64) -> (LogitRow, u64) {
    let obs = &obs[obs.len().saturating_sub(cfg.max_obs_per_row)..];
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for o in obs {
        *counts.entry(o.to.as_str()).or_default() += 1;
    }
    let keys = row_keys(&counts, cfg.max_successors);
    let n = obs.len() as f64;
    let mut x_mean = [0.0; COV_DIM];
    let mut x_sd = [0.0; COV_DIM];
    for d in 0..COV_DIM {
        x_mean[d] = obs.iter().map(|o| f64::from(o.x[d])).sum::<f64>() / n;
        let v = obs.iter().map(|o| (f64::from(o.x[d]) - x_mean[d]).powi(2)).sum::<f64>() / n;
        x_sd[d] = if v > 1e-12 { v.sqrt() } else { 1.0 };
    }
    let xs: Vec<[f64; P]> = obs
        .iter()
        .map(|o| {
            let mut x = [1.0; P];
            for d in 0..COV_DIM {
                x[d + 1] = ((f64::from(o.x[d]) - x_mean[d]) / x_sd[d]).clamp(-6.0, 6.0);
            }
            x
        })
        .collect();
    let ys: Vec<usize> = obs.iter().map(|o| key_index(&keys, &o.to)).collect();
    let mu_bias: Vec<f64> = base_probs(g, from, &keys).into_iter().map(f64::ln).collect();
    let target = LogitTarget {
        k: keys.len(),
        xs,
        ys,
        mu_bias: mu_bias.clone(),
        sd_bias: cfg.prior_sd_bias,
        sd_w: cfg.prior_sd_weight,
    };
    let init: Vec<f64> = (0..target.dim()).map(|i| if i % P == 0 { mu_bias[i / P] } else { 0.0 }).collect();
    let chains: Vec<Chain> = (0..cfg.chains.max(2))
        .map(|c| {
            let mut rng = Rng::seed(seed ^ (c as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let jitter: Vec<f64> = init.iter().map(|v| v + 0.5 * rng.normal()).collect();
            nuts::sample(&target, &jitter, &cfg.nuts, &mut rng)
        })
        .collect();
    let diag = Diagnostics::of(&chains);
    let accepted = diag.acceptable(cfg.max_rhat, cfg.max_div_share, cfg.min_ess);
    let all: Vec<&Vec<f64>> = chains.iter().flat_map(|c| &c.draws).collect();
    let step = (all.len() / cfg.keep_draws.max(1)).max(1);
    let draws = all.iter().step_by(step).map(|d| d.iter().map(|v| *v as f32).collect()).collect();
    let evals = chains.iter().map(|c| c.grad_evals).sum();
    let step_size = chains.iter().map(|c| c.step_size).sum::<f64>() / chains.len() as f64;
    (LogitRow { from: from.to_owned(), keys, n_obs: obs.len(), x_mean, x_sd, diag, step_size, accepted, draws }, evals)
}

/// One Gibbs chain: the `c` trace and per row the `β` traces.
type GibbsTrace = (Vec<f64>, Vec<Vec<Vec<f64>>>);

/// `ln Γ(x)` (Lanczos, g = 7; |error| < 1e-13 for x > 0).
pub fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Dirichlet-multinomial log likelihood of every asset row given `c` and `β` (θ integrated out).
#[allow(clippy::type_complexity)]
fn conc_loglik(rows: &[(RegionKey, Vec<RegionKey>, Vec<f64>, Vec<Vec<usize>>)], beta: &[Vec<f64>], c: f64) -> f64 {
    let mut ll = 0.0;
    for ((_, _, _, assets), b) in rows.iter().zip(beta) {
        for n_a in assets {
            let n: usize = n_a.iter().sum();
            if n == 0 {
                continue;
            }
            ll += ln_gamma(c) - ln_gamma(c + n as f64);
            for (nj, bj) in n_a.iter().zip(b) {
                if *nj > 0 {
                    ll += ln_gamma(c * bj + *nj as f64) - ln_gamma(c * bj);
                }
            }
        }
    }
    ll
}

/// Gibbs for the hierarchical Dirichlet across assets: `θ_{a,i} ~ Dir(c·β_i)`,
/// `β_i ~ Dir(γ·p₀_i)`, `c ~ Gamma(1, 0.1)`. Per sweep: `c | β, n` by random-walk Metropolis on
/// `ln c` with θ integrated out (Dirichlet-multinomial likelihood; mixes far better than the
/// auxiliary-variable update when assets are homogeneous and `c` is large), then per row the
/// Antoniak table counts `m | c, β, n` and `β | m`. Returns the `c` trace and per row the `β`
/// traces.
#[allow(clippy::type_complexity)]
fn gibbs_chain(
    rows: &[(RegionKey, Vec<RegionKey>, Vec<f64>, Vec<Vec<usize>>)],
    cfg: &RefitCfg,
    c0: f64,
    rng: &mut Rng,
) -> GibbsTrace {
    let (a_c, b_c) = (1.0, 0.1);
    let log_post = |lc: f64, beta: &[Vec<f64>]| {
        let c = lc.exp();
        // Gamma(a, b) prior on c plus the Jacobian of c = e^{ln c}
        conc_loglik(rows, beta, c) + (a_c - 1.0) * lc - b_c * c + lc
    };
    let mut lc = c0.ln();
    let mut beta: Vec<Vec<f64>> = rows.iter().map(|r| r.2.clone()).collect();
    let mut lp = log_post(lc, &beta);
    let mut c_trace = Vec::with_capacity(cfg.gibbs_iters);
    let mut b_trace: Vec<Vec<Vec<f64>>> = vec![Vec::with_capacity(cfg.gibbs_iters); rows.len()];
    for it in 0..cfg.gibbs_burn + cfg.gibbs_iters {
        for _ in 0..8 {
            let prop = (lc + 0.4 * rng.normal()).clamp(-7.0, 12.0);
            let lp2 = log_post(prop, &beta);
            if (lp2 - lp).is_finite() && rng.uniform().max(f64::MIN_POSITIVE).ln() < lp2 - lp {
                lc = prop;
                lp = lp2;
            }
        }
        let c = lc.exp();
        for (ri, (_, keys, p0, assets)) in rows.iter().enumerate() {
            let gamma0 = keys.len() as f64;
            let mut m_tot = vec![0.0; keys.len()];
            for n_a in assets {
                for (j, &n) in n_a.iter().enumerate() {
                    let cb = c * beta[ri][j];
                    // Antoniak: tables for n customers at concentration cβ
                    let m: usize = (0..n).filter(|&l| rng.uniform() < cb / (cb + l as f64)).count();
                    m_tot[j] += m as f64;
                }
            }
            let alpha: Vec<f64> = p0.iter().zip(&m_tot).map(|(p, m)| gamma0 * p + m).collect();
            beta[ri] = rng.dirichlet(&alpha).into_iter().map(|v| v.max(1e-12)).collect();
            if it >= cfg.gibbs_burn {
                b_trace[ri].push(beta[ri].clone());
            }
        }
        lp = log_post(lc, &beta);
        if it >= cfg.gibbs_burn {
            c_trace.push(c);
        }
    }
    (c_trace, b_trace)
}

fn fit_hier(
    g: &MarkovGraph,
    by_row: &BTreeMap<&str, Vec<&ObsRecord>>,
    rows: &[&str],
    cfg: &RefitCfg,
    seed: u64,
) -> Option<HierFit> {
    let mut assets: BTreeMap<&str, usize> = BTreeMap::new();
    let mut data = Vec::new();
    for from in rows {
        let obs = &by_row[from];
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for o in obs {
            *counts.entry(o.to.as_str()).or_default() += 1;
        }
        let keys = row_keys(&counts, cfg.max_successors);
        let mut per_asset: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for o in obs {
            let n = assets.len();
            assets.entry(o.instrument.as_str()).or_insert(n);
            per_asset.entry(o.instrument.as_str()).or_insert_with(|| vec![0; keys.len()])[key_index(&keys, &o.to)] += 1;
        }
        let p0 = base_probs(g, from, &keys);
        data.push(((*from).to_owned(), keys, p0, per_asset.into_values().collect::<Vec<_>>()));
    }
    if data.is_empty() || assets.len() < 2 {
        return None; // one asset: nothing to share
    }
    let chains: Vec<GibbsTrace> = [1.0, 50.0]
        .iter()
        .enumerate()
        .map(|(i, c0)| gibbs_chain(&data, cfg, *c0, &mut Rng::seed(seed ^ (0xB1B5 + i as u64))))
        .collect();
    let c_chains: Vec<Vec<f64>> = chains.iter().map(|c| c.0.clone()).collect();
    // diagnostics on ln c (the concentration is scale-like and right-skewed)
    let log_c: Vec<Vec<f64>> = c_chains.iter().map(|c| c.iter().map(|v| v.ln()).collect()).collect();
    let conc_rhat = nuts::finite_rhat(nuts::split_rhat(&log_c));
    let conc_ess = nuts::ess(&log_c);
    let conc_draws: Vec<f64> = c_chains.concat();
    let conc_mean = conc_draws.iter().sum::<f64>() / conc_draws.len().max(1) as f64;
    let mut max_rhat = conc_rhat;
    let out_rows: Vec<HierRow> = data
        .iter()
        .enumerate()
        .map(|(ri, (from, keys, _, _))| {
            let draws: Vec<&Vec<f64>> = chains.iter().flat_map(|c| &c.1[ri]).collect();
            let n = draws.len().max(1) as f64;
            let beta_mean: Vec<f64> = (0..keys.len()).map(|j| draws.iter().map(|d| d[j]).sum::<f64>() / n).collect();
            let top = (0..keys.len()).max_by(|a, b| beta_mean[*a].total_cmp(&beta_mean[*b])).unwrap_or(0);
            let per_chain: Vec<Vec<f64>> = chains.iter().map(|c| c.1[ri].iter().map(|d| d[top]).collect()).collect();
            let rhat = nuts::finite_rhat(nuts::split_rhat(&per_chain));
            max_rhat = max_rhat.max(rhat);
            let step = (draws.len() / cfg.keep_draws.max(1)).max(1);
            HierRow {
                from: from.clone(),
                keys: keys.clone(),
                beta_mean,
                rhat,
                beta_draws: draws.iter().step_by(step).map(|d| d.iter().map(|v| *v as f32).collect()).collect(),
            }
        })
        .collect();
    let step = (conc_draws.len() / cfg.keep_draws.max(1)).max(1);
    Some(HierFit {
        rows: out_rows,
        assets: assets.len(),
        conc_mean,
        conc_draws: conc_draws.iter().step_by(step).copied().collect(),
        conc_rhat,
        conc_ess,
        max_rhat,
        accepted: max_rhat <= cfg.gibbs_max_rhat,
    })
}

/// Runs the refit described by `spec` on `dump` (pure; CPU bound).
pub fn run(dump: &ObsDump, spec: &RefitSpec) -> RefitOutcome {
    let t0 = std::time::Instant::now();
    let cfg = spec.cfg;
    let mut by_row: BTreeMap<&str, Vec<&ObsRecord>> = BTreeMap::new();
    for r in &dump.records {
        by_row.entry(r.from.as_str()).or_default().push(r);
    }
    for v in by_row.values_mut() {
        v.sort_by_key(|r| r.t_ms);
    }
    let mut rows: Vec<&str> = by_row.iter().filter(|(_, v)| v.len() >= cfg.min_row_obs).map(|(k, _)| *k).collect();
    rows.sort_by(|a, b| by_row[b].len().cmp(&by_row[a].len()).then_with(|| a.cmp(b)));
    rows.truncate(cfg.max_rows);
    let wants = |k: &str| spec.kernels.is_empty() || spec.kernels.iter().any(|x| x == k);

    let mut logit = Vec::new();
    let mut grad_evals = 0;
    if wants("logit_cov") {
        let threads = cfg.threads.max(1).min(rows.len().max(1));
        let chunks: Vec<Vec<(usize, &str)>> = (0..threads)
            .map(|t| rows.iter().copied().enumerate().filter(|(i, _)| i % threads == t).collect())
            .collect();
        let mut fitted: Vec<(usize, LogitRow, u64)> = std::thread::scope(|s| {
            let handles: Vec<_> = chunks
                .iter()
                .map(|chunk| {
                    let by_row = &by_row;
                    let g = &dump.graph;
                    s.spawn(move || {
                        chunk
                            .iter()
                            .map(|(i, from)| {
                                let (row, ev) =
                                    fit_logit_row(g, from, &by_row[from], &cfg, spec.seed ^ (*i as u64 * 7919));
                                (*i, row, ev)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
        });
        fitted.sort_by_key(|x| x.0);
        for (_, r, ev) in fitted {
            grad_evals += ev;
            logit.push(r);
        }
    }
    let hier = if wants("hier_asset") { fit_hier(&dump.graph, &by_row, &rows, &cfg, spec.seed) } else { None };
    RefitOutcome {
        kernel_id: spec.kernel_id.clone(),
        snapshot_id: spec.snapshot_id.clone(),
        timescale: spec.timescale.clone(),
        expert_index: spec.expert_index,
        created_ms: dump.created_ms,
        records: dump.records.len(),
        logit,
        hier,
        grad_evals,
        cost_ms: t0.elapsed().as_secs_f64() * 1000.0,
        samples_ref: String::new(),
    }
}

// ---------------------------------------------------------------------------------------------
// Garage `.bin` format: b"FAMCMC01", u32 LE header length, JSON header (the outcome without the
// draw arrays plus their shapes), then the draws as f32 LE in header order.

const MAGIC: &[u8; 8] = b"FAMCMC01";

#[derive(Serialize, Deserialize)]
struct BinHeader {
    outcome: RefitOutcome,
    /// (rows, cols) of each logit row's draws, then of each hier row's β draws.
    logit_shapes: Vec<(usize, usize)>,
    hier_shapes: Vec<(usize, usize)>,
}

pub fn encode(o: &RefitOutcome) -> Vec<u8> {
    let shape = |d: &Vec<Vec<f32>>| (d.len(), d.first().map_or(0, Vec::len));
    let header = BinHeader {
        outcome: o.clone(),
        logit_shapes: o.logit.iter().map(|r| shape(&r.draws)).collect(),
        hier_shapes: o.hier.iter().flat_map(|h| h.rows.iter().map(|r| shape(&r.beta_draws))).collect(),
    };
    let json = serde_json::to_vec(&header).unwrap_or_default();
    let mut out = Vec::with_capacity(16 + json.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&json);
    let all = o.logit.iter().map(|r| &r.draws).chain(o.hier.iter().flat_map(|h| h.rows.iter().map(|r| &r.beta_draws)));
    for d in all {
        for row in d {
            for v in row {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    out
}

pub fn decode(b: &[u8]) -> Result<RefitOutcome, String> {
    if b.len() < 12 || &b[..8] != MAGIC {
        return Err("not a FAMCMC01 posterior dump".into());
    }
    let n = u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize;
    let header: BinHeader =
        serde_json::from_slice(b.get(12..12 + n).ok_or("truncated header")?).map_err(|e| e.to_string())?;
    let mut pos = 12 + n;
    let mut take = |(r, c): (usize, usize)| -> Result<Vec<Vec<f32>>, String> {
        let mut out = Vec::with_capacity(r);
        for _ in 0..r {
            let bytes = b.get(pos..pos + 4 * c).ok_or("truncated draws")?;
            out.push(bytes.as_chunks::<4>().0.iter().map(|x| f32::from_le_bytes(*x)).collect());
            pos += 4 * c;
        }
        Ok(out)
    };
    let mut o = header.outcome;
    for (row, s) in o.logit.iter_mut().zip(header.logit_shapes) {
        row.draws = take(s)?;
    }
    if let Some(h) = o.hier.as_mut() {
        for (row, s) in h.rows.iter_mut().zip(header.hier_shapes) {
            row.beta_draws = take(s)?;
        }
    }
    Ok(o)
}

/// Garage key of a refit's posterior samples: `mcmc/<kernel_id>/<created_ms>.bin`.
pub fn samples_key(kernel_id: &str, created_ms: i64) -> String {
    format!("mcmc/{kernel_id}/{created_ms}.bin")
}

/// Garage key of a refit's observation window.
pub fn obs_key(kernel_id: &str, job_id: &str) -> String {
    format!("mcmc/{kernel_id}/obs/{job_id}.json")
}

/// Bounded observation window of one expert space (model-runtime side).
#[derive(Debug, Clone, Default)]
pub struct ObsWindow {
    cap: usize,
    pub records: std::collections::VecDeque<ObsRecord>,
    /// Records added since the last [`ObsWindow::take_dump`].
    pub fresh: usize,
}

impl ObsWindow {
    pub fn new(cap: usize) -> Self {
        Self { cap: cap.max(1), records: Default::default(), fresh: 0 }
    }
    pub fn push(&mut self, r: ObsRecord) {
        if self.records.len() == self.cap {
            self.records.pop_front();
        }
        self.records.push_back(r);
        self.fresh += 1;
    }
    pub fn len(&self) -> usize {
        self.records.len()
    }
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
    /// The dump for a refit; resets the fresh counter.
    pub fn take_dump(&mut self, kernel_id: &str, created_ms: i64, graph: &MarkovGraph) -> ObsDump {
        self.fresh = 0;
        ObsDump {
            kernel_id: kernel_id.to_owned(),
            created_ms,
            graph: graph.clone(),
            records: self.records.iter().cloned().collect(),
        }
    }
    /// Drops records mentioning regions that no longer exist (after a snapshot swap).
    pub fn clear(&mut self) {
        self.records.clear();
        self.fresh = 0;
    }
}

/// Distinct assets in a record set (diagnostic).
pub fn assets(records: &[ObsRecord]) -> usize {
    records.iter().map(|r| r.instrument.as_str()).collect::<std::collections::BTreeSet<_>>().len()
}

/// Per-row observation counts (diagnostic / scheduling).
pub fn row_counts(records: &[ObsRecord]) -> HashMap<&str, usize> {
    let mut m = HashMap::new();
    for r in records {
        *m.entry(r.from.as_str()).or_default() += 1;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        kernel::KernelSampler,
        rich::{HierAsset, KernelCtx, LogitCov},
    };

    /// Synthetic window: from `a`, the successor is `b` when covariate 1 (vol) is high, `c` when
    /// low; asset `x` prefers `a → b`, asset `y` `a → c` irrespective of vol for row `d`.
    fn window(n: usize, seed: u64) -> (MarkovGraph, Vec<ObsRecord>) {
        let mut rng = Rng::seed(seed);
        let mut g = MarkovGraph::new(1.0, 1.0);
        let mut recs = Vec::new();
        for t in 0..n {
            let vol = rng.normal();
            let p_b = 1.0 / (1.0 + (-3.0 * vol).exp());
            let to = if rng.uniform() < p_b { "b" } else { "c" };
            let x = [1.0, vol as f32, 0.5, 0.3];
            g.observe("a", to);
            recs.push(ObsRecord {
                t_ms: t as i64,
                instrument: if t % 2 == 0 { "x" } else { "y" }.into(),
                prev: None,
                from: "a".into(),
                to: to.into(),
                dwell: 1,
                x,
            });
            let inst = if t % 2 == 0 { "x" } else { "y" };
            let to = match (inst, rng.uniform() < 0.9) {
                ("x", true) | ("y", false) => "b",
                _ => "c",
            };
            g.observe("d", to);
            recs.push(ObsRecord {
                t_ms: t as i64,
                instrument: inst.into(),
                prev: None,
                from: "d".into(),
                to: to.into(),
                dwell: 1,
                x: [0.0; 4],
            });
        }
        (g, recs)
    }

    fn spec(cfg: RefitCfg) -> RefitSpec {
        RefitSpec {
            kernel_id: "s-e0".into(),
            snapshot_id: "s".into(),
            timescale: "1m".into(),
            expert_index: 0,
            obs_ref: String::new(),
            kernels: vec![],
            seed: 3,
            cfg,
        }
    }

    #[test]
    fn nuts_and_gibbs_refits_recover_covariate_and_asset_structure() {
        let (g, recs) = window(300, 5);
        let dump = ObsDump { kernel_id: "s-e0".into(), created_ms: 1, graph: g.clone(), records: recs };
        let cfg = RefitCfg {
            nuts: NutsCfg { warmup: 100, samples: 100, max_depth: 5, ..NutsCfg::default() },
            gibbs_burn: 100,
            gibbs_iters: 200,
            ..RefitCfg::default()
        };
        let o = run(&dump, &spec(cfg));
        assert_eq!(o.logit.len(), 2);
        let row = o.logit.iter().find(|r| r.from == "a").unwrap();
        assert!(row.accepted, "{:?}", row.diag);
        assert!(row.diag.max_rhat < 1.05 && row.diag.divergences == 0, "{:?}", row.diag);
        // the vol weight of `b` exceeds that of `c`
        let kb = row.keys.iter().position(|k| k == "b").unwrap();
        let kc = row.keys.iter().position(|k| k == "c").unwrap();
        let mean_w = |k: usize| row.draws.iter().map(|d| f64::from(d[k * P + 2])).sum::<f64>() / row.draws.len() as f64;
        assert!(mean_w(kb) - mean_w(kc) > 2.0, "{} {}", mean_w(kb), mean_w(kc));
        let h = o.hier.as_ref().unwrap();
        assert_eq!(h.assets, 2);
        assert!(h.accepted, "{h:?}");
        assert!(h.conc_mean > 0.0);

        // round trip through the Garage format
        let bytes = encode(&o);
        let back = decode(&bytes).unwrap();
        // draws are binary-exact; header floats go through JSON (≤ 1 ulp without float_roundtrip)
        assert_eq!(
            back.logit.iter().map(|r| &r.draws).collect::<Vec<_>>(),
            o.logit.iter().map(|r| &r.draws).collect::<Vec<_>>()
        );
        assert_eq!(
            back.logit.iter().map(|r| &r.keys).collect::<Vec<_>>(),
            o.logit.iter().map(|r| &r.keys).collect::<Vec<_>>()
        );
        let hb = back.hier.as_ref().unwrap();
        assert_eq!(hb.rows[0].beta_draws, h.rows[0].beta_draws);
        assert!((hb.conc_mean - h.conc_mean).abs() < 1e-9);
        assert!(decode(b"nope").is_err());

        // the arena kernels take their parts and condition on the context
        let mut lc = LogitCov::new(g.clone(), 32);
        let mut o2 = back;
        o2.samples_ref = samples_key("s-e0", 1);
        assert!(lc.apply_refit(&o2));
        let hi = KernelCtx { x: [1.0, 2.0, 0.5, 0.3], ..KernelCtx::default() };
        let lo = KernelCtx { x: [1.0, -2.0, 0.5, 0.3], ..KernelCtx::default() };
        let pb = |v: Vec<(RegionKey, f64)>| v.iter().find(|x| x.0 == "b").map_or(0.0, |x| x.1);
        assert!(pb(lc.predict_ctx("a", &hi)) > 0.9 && pb(lc.predict_ctx("a", &lo)) < 0.1);
        assert!((lc.predict_ctx("a", &hi).iter().map(|x| x.1).sum::<f64>() - 1.0).abs() < 1e-9);
        assert_eq!(lc.export_state().unwrap()["refit_ref"], "mcmc/s-e0/1.bin");
        let mut ha = HierAsset::new(g, 10.0, 32, 1);
        assert!(ha.apply_refit(&o2));
        assert!((ha.conc - h.conc_mean).abs() < 1e-12);
    }

    #[test]
    fn ln_gamma_matches_known_values() {
        assert!((ln_gamma(1.0)).abs() < 1e-12);
        assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-12);
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-12);
        assert!((ln_gamma(1e-3) - 6.907_178_885_383_854).abs() < 1e-9);
    }

    #[test]
    fn single_asset_windows_skip_the_hierarchy_and_sparse_rows_are_not_fitted() {
        let (g, recs) = window(20, 1);
        let recs: Vec<ObsRecord> = recs.into_iter().filter(|r| r.instrument == "x").collect();
        let dump = ObsDump { kernel_id: "k".into(), created_ms: 0, graph: g, records: recs };
        let o = run(&dump, &spec(RefitCfg::default()));
        assert!(o.logit.is_empty(), "rows below min_row_obs");
        assert!(o.hier.is_none());
        let mut w = ObsWindow::new(2);
        for r in &dump.records[..3] {
            w.push(r.clone());
        }
        assert_eq!((w.len(), w.fresh), (2, 3));
        let d = w.take_dump("k", 5, &dump.graph);
        assert_eq!((d.records.len(), w.fresh), (2, 0));
    }
}

/// Cost of one fitted `logit_cov` row online (32 posterior draws) and of a refit job.
/// Run: `cargo test -p fa-mcmc --release -- --ignored --nocapture bench_`.
#[cfg(test)]
mod bench {
    use std::time::Instant;

    use super::*;
    use crate::{
        kernel::KernelSampler,
        rich::{KernelCtx, LogitCov},
    };

    #[test]
    #[ignore = "benchmark: run with --release --ignored"]
    fn bench_refit_job_and_fitted_logit_prediction() {
        let mut rng = Rng::seed(4);
        let mut g = MarkovGraph::new(1.0, 1.0);
        let mut recs = Vec::new();
        for t in 0..20_000 {
            let from = format!("r{}", rng.below(24));
            let to = format!("r{}", rng.below(24));
            g.observe(&from, &to);
            recs.push(ObsRecord {
                t_ms: t,
                instrument: format!("i{}", t % 6),
                prev: None,
                from,
                to,
                dwell: 1 + rng.below(5) as u32,
                x: [rng.normal() as f32, rng.normal() as f32, rng.uniform() as f32, 0.3],
            });
        }
        let dump = ObsDump { kernel_id: "bench".into(), created_ms: 0, graph: g.clone(), records: recs };
        let spec = RefitSpec {
            kernel_id: "bench".into(),
            snapshot_id: "s".into(),
            timescale: "1m".into(),
            expert_index: 0,
            obs_ref: String::new(),
            kernels: vec![],
            seed: 1,
            cfg: RefitCfg::default(),
        };
        let t0 = Instant::now();
        let o = run(&dump, &spec);
        eprintln!(
            "refit job (20k obs, 24 rows): {:.1} s, {} grad evals, {:?}",
            t0.elapsed().as_secs_f64(),
            o.grad_evals,
            o.summary()
        );
        let mut lc = LogitCov::new(g, 32);
        lc.apply_refit(&o);
        let ctx = KernelCtx { x: [0.1, 0.2, 0.3, 0.3], ..KernelCtx::default() };
        let t1 = Instant::now();
        for i in 0..20_000 {
            let from = format!("r{}", i % 24);
            lc.predict_ctx(&from, &ctx);
            lc.observe_ctx(&from, "r1", &ctx);
        }
        let us = t1.elapsed().as_secs_f64() * 1e6 / 20_000.0;
        eprintln!("fitted logit_cov: {us:.1} µs per predict + observe");
        assert!(us < 1000.0);
    }
}
