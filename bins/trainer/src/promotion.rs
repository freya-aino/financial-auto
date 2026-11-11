//! Producers of promotion checklist items 4 and 5 (spec 36 §6.4), measured by the trainer at fit
//! time and written into `manifest.metrics`, where control-api's checklist reads them.
//!
//! * **Item 4, MCMC calibration (ECE).** Per expert, the probe windows of each instrument form a
//!   region sequence (nearest fitted centroid for the candidate; the parent's stored
//!   assignment of the same probes). A Dirichlet Markov graph ([`fa_mcmc::MarkovGraph`],
//!   symmetric prior 0.5, every region of the expert in its support) is warmed on the first
//!   60 % of each sequence and then scored **prequentially** on the rest with
//!   [`fa_mcmc::calibration::Calibration`] (predict the next region, score, then observe) —
//!   the same reliability-bin ECE the runtime arena reports. Metrics: `mcmc_ece` (candidate,
//!   all probes), and with a parent assignment `mcmc_ece_shared` / `parent_mcmc_ece` on the
//!   probes both assign and `mcmc_ece_regression = mcmc_ece_shared − parent_mcmc_ece` (the
//!   checklist compares it with `FA_PROMOTE_ECE_TOL`, 0.02). The transitions are between
//!   consecutive probes (the probe grid), identically for candidate and parent.
//! * **Item 5, elite transfer.** The frozen feedback decoders of a `generation` (the D-068 elite
//!   decoders, else the frozen linear-probe stand-in; `manifest.feedback.source` says which) are
//!   re-evaluated on the validation windows with the parent's latents and with the
//!   candidate's: `elite_transfer = (L_candidate − L_parent) / L_parent` (relative loss
//!   increase; ≤ `FA_PROMOTE_TRANSFER_TOL`, 0.10). Absent when there is no parent or no
//!   decoder reads both latents.

use std::collections::BTreeMap;

use fa_core::rng::Rng;
use fa_encoders::neural::{Dataset, FeedbackBatch, NeuralModel, NeuralTrainer};
use fa_mcmc::{MarkovGraph, calibration::Calibration};
use fa_patterns::FittedRegion;

use crate::fit::{ParentAssignment, ProbeKey};

/// Share of each sequence that only warms the kernel.
pub const WARMUP: f64 = 0.6;

/// Prequential ECE of one expert's region sequence(s): `keys[w]` = (instrument, ts) of probe
/// `w` (grouped by instrument, time-ordered), `assign[w]` its region label. `(ece, evals)`.
pub fn prequential_ece(keys: &[ProbeKey], assign: &[Option<String>]) -> Option<(f64, u64)> {
    let mut g = MarkovGraph::new(0.5, 1.0);
    for a in assign.iter().flatten() {
        g.register_region(a);
    }
    let mut cal = Calibration::new(1.0);
    let mut rng = Rng::seed(0);
    let mut i = 0;
    while i < keys.len() {
        let j = keys[i..].iter().position(|k| k.0 != keys[i].0).map_or(keys.len(), |d| i + d);
        let n = j - i;
        let warm = i + ((n as f64) * WARMUP).ceil() as usize;
        for w in i + 1..j {
            let (Some(a), Some(b)) = (&assign[w - 1], &assign[w]) else { continue };
            if w >= warm {
                cal.record(&g.posterior_mean(a), b, -1.0, &mut rng);
            }
            g.observe(a, b);
        }
        i = j;
    }
    (cal.evals > 0).then(|| (cal.scores().ece, cal.evals))
}

/// Nearest fitted centroid of every probe latent (region id).
pub fn assign_nearest(points: &[Vec<f32>], regions: &[FittedRegion]) -> Vec<Option<String>> {
    points
        .iter()
        .map(|p| {
            regions
                .iter()
                .map(|r| (r.centroid.iter().zip(p).map(|(a, b)| f64::from(a - b).powi(2)).sum::<f64>(), r))
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(_, r)| r.region_id.clone())
        })
        .collect()
}

fn pooled(per: &[(f64, u64)]) -> Option<(f64, u64)> {
    let n: u64 = per.iter().map(|x| x.1).sum();
    (n > 0).then(|| (per.iter().map(|(e, c)| e * *c as f64).sum::<f64>() / n as f64, n))
}

/// Item 4 metrics (see the module docs) into `metrics`.
pub fn ece_metrics(
    metrics: &mut BTreeMap<String, f64>,
    keys: &[ProbeKey],
    latents: &[Vec<Vec<f32>>],
    regions: &[Vec<FittedRegion>],
    parent: &ParentAssignment,
) {
    let mut cand = Vec::new();
    let mut cand_shared = Vec::new();
    let mut par = Vec::new();
    for (k, points) in latents.iter().enumerate() {
        let Some(rs) = regions.get(k).filter(|r| !r.is_empty()) else { continue };
        let a = assign_nearest(points, rs);
        if let Some(x) = prequential_ece(keys, &a) {
            cand.push(x);
        }
        let Some(pa) = parent.assign.get(k).filter(|p| p.len() == a.len()) else { continue };
        let p_lab: Vec<Option<String>> = pa.iter().map(|x| x.map(|i| format!("p{i}"))).collect();
        let c_masked: Vec<Option<String>> =
            a.iter().zip(&p_lab).map(|(c, p)| if p.is_some() { c.clone() } else { None }).collect();
        if let (Some(c), Some(p)) = (prequential_ece(keys, &c_masked), prequential_ece(keys, &p_lab)) {
            cand_shared.push(c);
            par.push(p);
        }
    }
    if let Some((e, n)) = pooled(&cand) {
        metrics.insert("mcmc_ece".into(), e);
        metrics.insert("mcmc_ece_n".into(), n as f64);
    }
    if let (Some((c, n)), Some((p, _))) = (pooled(&cand_shared), pooled(&par)) {
        metrics.insert("mcmc_ece_shared".into(), c);
        metrics.insert("parent_mcmc_ece".into(), p);
        metrics.insert("mcmc_ece_regression".into(), c - p);
        metrics.insert("mcmc_ece_shared_n".into(), n as f64);
    }
}

/// Item 5: mean feedback loss of `tr.feedback` on `windows` with the parent's latents and with
/// the candidate's. `None` when no decoder scores both.
pub fn elite_transfer(
    tr: &NeuralTrainer,
    parent: &NeuralModel,
    ds: &Dataset,
    windows: &[(usize, usize)],
) -> candle_core::Result<Option<(f64, f64, usize)>> {
    if tr.feedback.is_empty() || windows.is_empty() {
        return Ok(None);
    }
    let stride = (windows.len() / 1024).max(1);
    let idx: Vec<(usize, usize)> = windows.iter().step_by(stride).copied().collect();
    let pt = NeuralTrainer::from_model(parent.clone(), 0);
    let (mut lc, mut lp, mut n) = (0.0, 0.0, 0usize);
    for chunk in idx.chunks(256) {
        let b = tr.batch(ds, chunk)?;
        let f = tr.model.forward(&b.x, None)?;
        let pb = pt.batch(ds, chunk)?;
        let pf = parent.forward(&pb.x, None)?;
        let cb = FeedbackBatch {
            mu: &f.out.mu,
            probs: &f.out.probs,
            targets: &b.y,
            experts: &tr.model.moe.active,
            windows: &b.windows,
        };
        let pbatch = FeedbackBatch {
            mu: &pf.out.mu,
            probs: &pf.out.probs,
            targets: &pb.y,
            experts: &parent.moe.active,
            windows: &pb.windows,
        };
        for d in &tr.feedback {
            if let (Some(c), Some(p)) = (d.aux_loss(&cb)?, d.aux_loss(&pbatch)?) {
                let w = chunk.len() as f64;
                lc += fa_expert_space::nn::scalar(&c)? * w;
                lp += fa_expert_space::nn::scalar(&p)? * w;
                n += chunk.len();
            }
        }
    }
    Ok((n > 0).then(|| (lp / n as f64, lc / n as f64, tr.feedback.len())))
}

/// Item 5 metrics into `metrics` (logs and skips on error).
pub fn transfer_metrics(
    metrics: &mut BTreeMap<String, f64>,
    tr: &NeuralTrainer,
    parent: Option<&NeuralModel>,
    ds: &Dataset,
    windows: &[(usize, usize)],
) {
    let Some(p) = parent else { return };
    match tokio::task::block_in_place(|| elite_transfer(tr, p, ds, windows)) {
        Ok(Some((lp, lc, n))) => {
            metrics.insert("elite_loss_parent".into(), lp);
            metrics.insert("elite_loss_candidate".into(), lc);
            metrics.insert("elite_transfer".into(), if lp.abs() > 1e-12 { (lc - lp) / lp.abs() } else { lc - lp });
            metrics.insert("elite_transfer_n".into(), n as f64);
        }
        Ok(None) => {}
        Err(e) => tracing::warn!(error = %e, "elite transfer not measured"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ece_is_small_for_a_stationary_chain_and_large_for_a_shifted_one() {
        // deterministic alternation A,B,A,B…: a calibrated kernel learns it
        let keys: Vec<ProbeKey> = (0..400).map(|i| ("x".to_owned(), i)).collect();
        let alt: Vec<Option<String>> = (0..400).map(|i| Some(if i % 2 == 0 { "a" } else { "b" }.to_owned())).collect();
        let (e_alt, n) = prequential_ece(&keys, &alt).unwrap();
        assert!(n > 100);
        // regime change at the warm-up boundary: the kernel learnt "stay", then the chain alternates
        let shift: Vec<Option<String>> = (0..400)
            .map(|i| {
                Some(
                    if i < 240 {
                        if i < 120 { "a" } else { "b" }
                    } else if i % 2 == 0 {
                        "a"
                    } else {
                        "b"
                    }
                    .to_owned(),
                )
            })
            .collect();
        let (e_shift, _) = prequential_ece(&keys, &shift).unwrap();
        assert!(e_alt < 0.1, "{e_alt}");
        assert!(e_shift > e_alt + 0.1, "{e_shift} vs {e_alt}");
        // unassigned probes break transitions instead of inventing them
        let none = vec![None; 400];
        assert!(prequential_ece(&keys, &none).is_none());
    }

    #[test]
    fn ece_metrics_compare_candidate_and_parent_on_shared_probes() {
        let keys: Vec<ProbeKey> = (0..200).map(|i| ("x".to_owned(), i)).collect();
        let latents = vec![(0..200).map(|i| vec![if i % 2 == 0 { 0.0 } else { 1.0 }]).collect::<Vec<_>>()];
        let r = |id: &str, c: f32| FittedRegion {
            region_id: id.into(),
            lineage_id: id.into(),
            expert_index: 0,
            centroid: vec![c],
            radius: 0.3,
            mass: 0.5,
            parent_region_id: None,
            score: 0.0,
            ancestors: Vec::new(),
        };
        let regions = vec![vec![r("r0", 0.0), r("r1", 1.0)]];
        let parent = ParentAssignment {
            snapshot_id: "p".into(),
            ids: vec![vec![]],
            assign: vec![(0..200).map(|i| Some(i % 2)).collect()],
            covered: 200,
            note: None,
        };
        let mut m = BTreeMap::new();
        ece_metrics(&mut m, &keys, &latents, &regions, &parent);
        assert!(m["mcmc_ece"] < 0.1);
        assert!(m["mcmc_ece_regression"].abs() < 1e-9, "same partition, same calibration: {m:?}");
    }
}
