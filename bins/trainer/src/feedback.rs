//! Decoder feedback inputs for `generation` jobs (spec 34 §6.6, spec 36 §6.3), read from the
//! Postgres trial registry (D-052): the latest `validation.generation_stats` row (recorded as
//! `manifest.feedback.decoder_stats`) and the decoder skill reported for the parent snapshot
//! ([`parent_skill`]), which [`crate::search`] uses as the tie-breaker of its proxy.

use serde_json::{Value, json};
use sqlx::{Row, postgres::PgPoolOptions};

/// Latest generation statistics, or `None` when Postgres / the table is unavailable.
pub async fn latest_generation_stats(pg_url: &str) -> Option<Value> {
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .acquire_timeout(std::time::Duration::from_secs(3))
        .connect(pg_url)
        .await
        .map_err(|e| tracing::warn!(error = %e, "postgres unreachable; generation runs without decoder feedback"))
        .ok()?;
    let row = sqlx::query(
        "select generation, gen_index, kind, n_trials, sharpe_mean, sharpe_var, best_candidate, best_sharpe, \
         expected_max_sharpe, deflated_sharpe, pbo, pbo_trials, pbo_blocks, computed_at::text \
         from validation.generation_stats order by computed_at desc limit 1",
    )
    .fetch_optional(&pool)
    .await
    .map_err(|e| tracing::warn!(error = %e, "validation.generation_stats unreadable; no decoder feedback"))
    .ok()??;
    let f = |c: &str| row.try_get::<Option<f64>, _>(c).ok().flatten();
    Some(json!({
        "source": "postgres:validation.generation_stats",
        "generation": row.try_get::<String, _>("generation").ok(),
        "gen_index": row.try_get::<i32, _>("gen_index").ok(),
        "kind": row.try_get::<String, _>("kind").ok(),
        "n_trials": row.try_get::<i32, _>("n_trials").ok(),
        "sharpe_mean": f("sharpe_mean"),
        "sharpe_var": f("sharpe_var"),
        "best_candidate": row.try_get::<Option<String>, _>("best_candidate").ok().flatten(),
        "best_sharpe": f("best_sharpe"),
        "expected_max_sharpe": f("expected_max_sharpe"),
        "deflated_sharpe": f("deflated_sharpe"),
        "pbo": f("pbo"),
        "pbo_trials": row.try_get::<i32, _>("pbo_trials").ok(),
        "pbo_blocks": row.try_get::<i32, _>("pbo_blocks").ok(),
        "computed_at": row.try_get::<String, _>("computed_at").ok(),
    }))
}

/// Decoder skill reported for `snapshot_id` (the tie-breaker of [`crate::search`]):
/// 1. mean `objectives.crps_skill` of `validation.trial` rows whose `objectives.snapshot_id`
///    is the parent (decoders evaluated on its latents) — attributable, `keyed = true`;
/// 2. else the latest `validation.generation_stats.deflated_sharpe` — not attributable to a
///    snapshot today (trials do not record one), `keyed = false`;
/// 3. else none (the tie-breaker weight is then 0).
pub async fn parent_skill(pg_url: Option<&str>, snapshot_id: Option<&str>) -> crate::search::ParentSkill {
    use crate::search::ParentSkill;
    let none =
        |why: &str| ParentSkill { source: "none".into(), detail: json!({ "reason": why }), ..Default::default() };
    let Some(url) = pg_url else { return none("no postgres url") };
    let pool = match PgPoolOptions::new()
        .max_connections(1)
        .acquire_timeout(std::time::Duration::from_secs(3))
        .connect(url)
        .await
    {
        Ok(p) => p,
        Err(e) => return none(&format!("postgres unreachable: {e}")),
    };
    if let Some(id) = snapshot_id {
        let keyed = sqlx::query(
            "select avg((objectives->>'crps_skill')::float8) as skill, count(*) as n \
             from validation.trial where objectives->>'snapshot_id' = $1 \
             and objectives ? 'crps_skill' and holdout_id is null",
        )
        .bind(id)
        .fetch_optional(&pool)
        .await;
        if let Ok(Some(row)) = keyed {
            let n: i64 = row.try_get("n").unwrap_or(0);
            let v: Option<f64> = row.try_get("skill").ok().flatten();
            if n > 0 && v.is_some() {
                return ParentSkill {
                    value: v,
                    source: "validation.trial".into(),
                    keyed: true,
                    n_trials: n,
                    detail: json!({ "objective": "crps_skill", "snapshot_id": id }),
                };
            }
        }
    }
    let row = sqlx::query(
        "select generation, n_trials, deflated_sharpe, computed_at::text as at \
         from validation.generation_stats where deflated_sharpe is not null order by computed_at desc limit 1",
    )
    .fetch_optional(&pool)
    .await;
    match row {
        Ok(Some(r)) => ParentSkill {
            value: r.try_get::<Option<f64>, _>("deflated_sharpe").ok().flatten(),
            source: "validation.generation_stats".into(),
            keyed: false,
            n_trials: i64::from(r.try_get::<i32, _>("n_trials").unwrap_or(0)),
            detail: json!({
                "objective": "deflated_sharpe",
                "generation": r.try_get::<String, _>("generation").ok(),
                "computed_at": r.try_get::<String, _>("at").ok(),
                "note": "latest evolution generation, not attributable to the parent snapshot",
            }),
        },
        Ok(None) => none("no trials for the parent and no generation_stats rows"),
        Err(e) => none(&format!("validation tables unreadable: {e}")),
    }
}

// ── D-009 decoder feedback for neural fits ──────────────────────────────────

use candle_core::Tensor;
use fa_encoders::neural::{
    Dataset, DecoderFeedback, FeedbackBatch, N_FEATURES, N_TARGETS, NeuralEncoder, NeuralModel, data::targets_at,
};

/// Frozen "elite decoder" stand-in for the D-009 hook: one ridge-regression readout per expert of
/// the **parent** snapshot, fitted on the parent's own latents of training windows to predict the
/// next-step targets, then frozen. During generation training the child's expert latent at the
/// same position must stay decodable by the parent's readout (`aux = mean_k MSE(W_k·[z_k;1], y)`),
/// which is the D-009 form (backprop through frozen decoders into the encoder) with the simplest
/// decoder. Neural elite decoders from `fa-decoders` implement the same
/// [`fa_encoders::neural::DecoderFeedback`] trait and are pushed onto `NeuralTrainer::feedback`.
pub struct FrozenProbeFeedback {
    /// `[dim + 1, T]` per parent expert position.
    probes: Vec<Tensor>,
    r2: Vec<f64>,
    n_fit: usize,
    parent: String,
}

impl FrozenProbeFeedback {
    /// Fits the probes on up to `max_windows` training windows (evenly spread). Targets are
    /// normalised with the **child's** statistics so the loss is on the child's scale.
    pub fn fit(parent: &NeuralEncoder, child: &NeuralModel, ds: &Dataset, max_windows: usize) -> anyhow::Result<Self> {
        let idx: Vec<(usize, usize)> =
            crate::fit::evenly(ds.train.len(), max_windows).into_iter().map(|i| ds.train[i]).collect();
        if idx.len() < 8 {
            anyhow::bail!("too few windows ({}) to fit probe decoders", idx.len());
        }
        let k = parent.model().moe.n_active();
        let mut xs: Vec<Vec<Vec<f64>>> = vec![Vec::with_capacity(idx.len()); k];
        let mut ys: Vec<Vec<f64>> = Vec::with_capacity(idx.len());
        for (s, ends) in group(&idx) {
            let enc = parent.encode_windows(&ds.series[s].bars, &ends)?;
            for (e, end) in enc.into_iter().zip(&ends) {
                for (pos, mu) in e.mu.into_iter().enumerate() {
                    xs[pos].push(mu.into_iter().map(f64::from).chain([1.0]).collect());
                }
                let t = targets_at(&ds.series[s].feats, *end).unwrap_or([0.0; N_TARGETS]);
                ys.push(t.iter().enumerate().map(|(j, v)| f64::from(child.norm.target(j, *v))).collect());
            }
        }
        let mut probes = Vec::with_capacity(k);
        let mut r2 = Vec::with_capacity(k);
        for x in &xs {
            let w = ridge(x, &ys, 1e-2);
            let (mut sse, mut sst) = (0.0, 0.0);
            for (xi, yi) in x.iter().zip(&ys) {
                for (t, yt) in yi.iter().enumerate() {
                    let p: f64 = xi.iter().zip(&w).map(|(a, row)| a * row[t]).sum();
                    sse += (yt - p).powi(2);
                    sst += yt * yt;
                }
            }
            r2.push(if sst > 0.0 { 1.0 - sse / sst } else { 0.0 });
            let d = w.len();
            let flat: Vec<f32> = w.iter().flatten().map(|v| *v as f32).collect();
            probes.push(Tensor::from_vec(flat, (d, N_TARGETS), child.device())?);
        }
        let _ = N_FEATURES;
        Ok(Self { probes, r2, n_fit: idx.len(), parent: parent.snapshot_id_str().to_owned() })
    }
}

trait SnapshotIdStr {
    fn snapshot_id_str(&self) -> &str;
}

impl SnapshotIdStr for NeuralEncoder {
    fn snapshot_id_str(&self) -> &str {
        fa_encoders::EncoderFamily::snapshot_id(self)
    }
}

/// Windows grouped by series (in order).
fn group(idx: &[(usize, usize)]) -> Vec<(usize, Vec<usize>)> {
    let mut out: Vec<(usize, Vec<usize>)> = Vec::new();
    for &(s, e) in idx {
        match out.last_mut() {
            Some((ls, v)) if *ls == s => v.push(e),
            _ => out.push((s, vec![e])),
        }
    }
    out
}

/// Ridge regression `W = (XᵀX + λI)⁻¹ XᵀY` by Gauss–Jordan elimination (`X: [n][d]`, `Y: [n][t]`).
pub fn ridge(x: &[Vec<f64>], y: &[Vec<f64>], lambda: f64) -> Vec<Vec<f64>> {
    let d = x.first().map_or(0, Vec::len);
    let t = y.first().map_or(0, Vec::len);
    let mut a = vec![vec![0.0; d + t]; d];
    for (xi, yi) in x.iter().zip(y) {
        for i in 0..d {
            for j in 0..d {
                a[i][j] += xi[i] * xi[j];
            }
            for j in 0..t {
                a[i][d + j] += xi[i] * yi[j];
            }
        }
    }
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += lambda;
    }
    for c in 0..d {
        let piv = (c..d).max_by(|p, q| a[*p][c].abs().total_cmp(&a[*q][c].abs())).unwrap_or(c);
        a.swap(c, piv);
        let pv = a[c][c];
        if pv.abs() < 1e-12 {
            continue;
        }
        for v in a[c].iter_mut() {
            *v /= pv;
        }
        let pivot_row = a[c].clone();
        for (r, row) in a.iter_mut().enumerate() {
            if r != c {
                let f = row[c];
                if f != 0.0 {
                    for (v, p) in row.iter_mut().zip(&pivot_row) {
                        *v -= f * p;
                    }
                }
            }
        }
    }
    a.into_iter().map(|row| row[d..].to_vec()).collect()
}

impl DecoderFeedback for FrozenProbeFeedback {
    fn name(&self) -> &str {
        "frozen_linear_probe"
    }

    fn aux_loss(&self, b: &FeedbackBatch<'_>) -> candle_core::Result<Option<Tensor>> {
        let k = b.mu.dim(1)?.min(self.probes.len());
        if k == 0 {
            return Ok(None);
        }
        let bsz = b.mu.dim(0)?;
        let ones = Tensor::ones((bsz, 1), b.mu.dtype(), b.mu.device())?;
        let mut acc: Option<Tensor> = None;
        for (pos, w) in self.probes.iter().enumerate().take(k) {
            let z = Tensor::cat(&[&b.mu.narrow(1, pos, 1)?.squeeze(1)?, &ones], 1)?;
            let l = fa_expert_space::nn::mse(&z.matmul(w)?, b.targets)?;
            acc = Some(match acc {
                Some(a) => (a + l)?,
                None => l,
            });
        }
        acc.map(|a| a.affine(1.0 / k as f64, 0.0)).transpose()
    }

    fn summary(&self) -> Value {
        json!({
            "name": self.name(),
            "kind": "D-009 frozen elite decoders (stand-in: ridge readouts of the parent's experts)",
            "parent": self.parent,
            "experts": self.probes.len(),
            "fit_windows": self.n_fit,
            "probe_r2": self.r2,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ridge_recovers_a_linear_map() {
        let x: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64 / 10.0, ((i * 7) % 11) as f64, 1.0]).collect();
        let y: Vec<Vec<f64>> = x.iter().map(|r| vec![2.0 * r[0] - r[1] + 0.5, r[1]]).collect();
        let w = ridge(&x, &y, 1e-9);
        assert!((w[0][0] - 2.0).abs() < 1e-5 && (w[1][0] + 1.0).abs() < 1e-5 && (w[2][0] - 0.5).abs() < 1e-4, "{w:?}");
        assert!((w[1][1] - 1.0).abs() < 1e-6);
        assert_eq!(group(&[(0, 1), (0, 2), (1, 5), (0, 3)]), vec![(0, vec![1, 2]), (1, vec![5]), (0, vec![3])]);
    }
}
