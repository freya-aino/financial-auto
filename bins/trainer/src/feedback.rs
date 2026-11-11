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

// ── D-068 elite decoders as the D-009 feedback (replaces the probe stand-in) ──

use fa_decoders::neural::{
    data::TextTimeline,
    feedback::{self as elite, EliteCard, FeedbackSet},
};

/// Elites per output family taken from Garage.
pub const ELITES_PER_FAMILY: usize = 4;

/// **D-009 with the real elite decoders** (D-068, `fa_decoders::neural::feedback`): the top
/// [`ELITES_PER_FAMILY`] cards per output family whose decoders read a latent of exactly the
/// trained snapshot's width (`n_experts × expert_dim`, concatenated expert means in output order —
/// the `LatentState` layout they were trained on) and the fit's step, loaded frozen; per batch
/// their context and targets are built from the row's own price series and the loss is
/// back-propagated into the expert means only. News-reading elites sit out: the trainer has no
/// news timeline (rows without news are skipped by `FeedbackSet::prepare`).
pub struct EliteFeedback {
    set: FeedbackSet,
    /// Decoder price series per dataset series (bar ends, ln close).
    series: Vec<fa_decoders::baseline::Series>,
    text: TextTimeline,
    cards: Vec<Value>,
    latent_dim: usize,
}

impl EliteFeedback {
    pub fn n_elites(&self) -> usize {
        self.set.elites.len()
    }
}

/// Decoder price series of a fit's histories (same order as the dataset's series).
pub fn decoder_series(history: &[crate::data::BarHistory], step_ms: i64) -> Vec<fa_decoders::baseline::Series> {
    history
        .iter()
        .map(|h| fa_decoders::baseline::Series {
            instrument_id: h.instrument.clone(),
            ts_ms: h.ends_ms.clone(),
            log_price: h.bars.iter().map(|b| b.close.max(1e-12).ln()).collect(),
            step_ms,
        })
        .collect()
}

/// Cards eligible for a fit: same step, current normaliser, latent width `latent_dim`; then the
/// D-068 top-k per family and quota weights.
pub fn eligible(cards: &[EliteCard], step_ms: i64, latent_dim: usize) -> Vec<(EliteCard, f64)> {
    let same_step: Vec<EliteCard> = cards.iter().filter(|c| c.meta.step_ms == step_ms).cloned().collect();
    elite::quota_weights(&elite::select_top_k(&same_step, ELITES_PER_FAMILY, Some(latent_dim)))
}

/// Lists the decoder cards in Garage, loads the eligible elites frozen. `Ok(None)` when none fit.
pub async fn load_elites(
    blob: &fa_blob::S3BlobStore,
    history: &[crate::data::BarHistory],
    step_ms: i64,
    latent_dim: usize,
    dev: &candle_core::Device,
) -> anyhow::Result<Option<EliteFeedback>> {
    use fa_blob::BlobStore;
    let listed = blob.list(fa_blob::buckets::MODELS, elite::PREFIX).await?;
    let mut cards = Vec::new();
    for o in listed.iter().filter(|o| o.key.ends_with(elite::CARD_SUFFIX)) {
        let key = fa_blob::ObjectKey { bucket: fa_blob::buckets::MODELS, key: o.key.clone() };
        if let Ok(b) = blob.get(&key).await
            && let Ok(c) = serde_json::from_slice::<EliteCard>(&b)
        {
            cards.push(c);
        }
    }
    let chosen = eligible(&cards, step_ms, latent_dim);
    // what the cards offer: (step_ms, full latent width, current normaliser, beats naive) → count
    let mut shapes: std::collections::BTreeMap<(i64, usize, bool, bool), usize> = Default::default();
    for c in &cards {
        *shapes
            .entry((c.meta.step_ms, c.meta.full_latent_dim(), c.meta.is_current(), c.meta.score > 0.0))
            .or_default() += 1;
    }
    tracing::info!(
        cards = cards.len(),
        eligible = chosen.len(),
        latent_dim,
        step_ms,
        offered = ?shapes,
        "decoder elites for D-009 feedback ((step_ms, latent width, current normaliser, score > 0) → n)"
    );
    if chosen.is_empty() {
        return Ok(None);
    }
    let mut bundles = Vec::with_capacity(chosen.len());
    let mut summary = Vec::with_capacity(chosen.len());
    for (c, w) in &chosen {
        let key = fa_blob::ObjectKey { bucket: fa_blob::buckets::MODELS, key: c.key.clone() };
        match blob.get(&key).await {
            Ok(b) => {
                bundles.push((b, *w));
                summary.push(json!({
                    "key": c.key, "family": c.meta.family, "output_family": c.meta.output_family,
                    "score": c.meta.score, "weight": w, "generation": c.meta.generation,
                }));
            }
            Err(e) => tracing::warn!(key = %c.key, error = %e, "elite bundle unreadable"),
        }
    }
    let set = FeedbackSet::load(&bundles, latent_dim, dev).map_err(|e| anyhow::anyhow!("elite bundles: {e}"))?;
    if set.is_empty() {
        return Ok(None);
    }
    Ok(Some(EliteFeedback {
        series: decoder_series(history, step_ms),
        text: TextTimeline::new(),
        cards: summary,
        latent_dim,
        set,
    }))
}

#[cfg(test)]
impl EliteFeedback {
    /// For tests: a loaded set over given series.
    pub fn from_parts(set: FeedbackSet, series: Vec<fa_decoders::baseline::Series>) -> Self {
        let latent_dim = set.latent_dim();
        Self { set, series, text: TextTimeline::new(), cards: Vec::new(), latent_dim }
    }
}

impl DecoderFeedback for EliteFeedback {
    fn name(&self) -> &str {
        "elite_decoders"
    }

    fn aux_loss(&self, b: &FeedbackBatch<'_>) -> candle_core::Result<Option<Tensor>> {
        let (bsz, k, d) = b.mu.dims3()?;
        if k * d != self.latent_dim || b.windows.len() != bsz {
            return Ok(None);
        }
        let latents = b.mu.reshape((bsz, k * d))?;
        let steps: Vec<(&fa_decoders::baseline::Series, usize)> = b
            .windows
            .iter()
            .map(|&(s, end)| (&self.series[s.min(self.series.len().saturating_sub(1))], end.saturating_sub(1)))
            .collect();
        let msg = |e: fa_decoders::neural::NeuralError| candle_core::Error::Msg(format!("elite feedback: {e}"));
        let batch = self.set.prepare(&steps, &self.text).map_err(msg)?;
        let out = self.set.aux_loss(&latents, &batch).map_err(msg)?;
        Ok((!out.parts.is_empty()).then_some(out.total))
    }

    fn summary(&self) -> Value {
        json!({
            "name": self.name(),
            "kind": "D-009 frozen elite decoders (D-068 fa_decoders::neural::feedback)",
            "latent_dim": self.latent_dim,
            "elites": self.cards,
            "skipped": self.set.skipped,
            "text_timeline": "empty (news-reading elites sit out)",
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

    /// D-009 end to end on SYNTHETIC data: an `nn_forecast` decoder trained on an encoder's
    /// concatenated expert means (the `LatentState` layout) becomes a frozen elite; plugged into
    /// the trainer it adds a finite, non-zero auxiliary loss whose gradient reaches the encoder,
    /// and `eligible` keeps only elites of the fit's step and latent width.
    #[test]
    fn elite_decoders_feed_back_into_a_neural_fit() {
        use std::sync::Arc;

        use fa_decoders::{DecodeCtx, LatentPoint, genome::DecoderGenome, neural as dn};
        use fa_encoders::{
            MarketBar,
            neural::{NeuralConfig, NeuralFamily, NeuralTrainer, Series as ESeries},
        };

        let dev = candle_core::Device::Cpu;
        let mut rng = fa_core::rng::Rng::seed(4);
        let mut p = 100.0;
        let bars: Vec<MarketBar> = (0..900)
            .map(|i| {
                let vol = if (i / 60) % 2 == 0 { 0.001 } else { 0.004 };
                let o = p;
                p *= 1.0 + vol * rng.normal() + if (i / 60) % 2 == 0 { 0.0004 } else { -0.0004 };
                MarketBar {
                    open: o,
                    high: o.max(p) * 1.0005,
                    low: o.min(p) * 0.9995,
                    close: p,
                    volume: 10.0,
                    trade_count: 2,
                }
            })
            .collect();
        let hist = crate::data::BarHistory {
            instrument: "syn".into(),
            timescale: fa_core::Timescale::Minute,
            source: crate::data::Source::Bars,
            start_ms: 0,
            end_ms: 900 * 60_000,
            ends_ms: (1..=900).map(|i| i as i64 * 60_000).collect(),
            bars: bars.clone(),
        };
        let cfg = NeuralConfig {
            context_steps: 16,
            d_model: 8,
            layers: 2,
            steps: 40,
            batch: 32,
            lambda_feedback: 0.5,
            moe: fa_expert_space::moe::MoeConfig { max_experts: 2, expert_dim: 3, ..Default::default() },
            ..NeuralConfig::new(NeuralFamily::JepaVae)
        };
        let ds = fa_encoders::neural::Dataset::build(vec![ESeries::new("syn", bars.clone())], 16, cfg.horizon(), 0.1);
        let mut tr = NeuralTrainer::new(cfg, &ds, 1, &dev).unwrap();
        tr.train_steps(&ds, 20).unwrap();
        // the latents the decoder trains on: concatenated expert means as of every bar end
        let enc = fa_encoders::neural::NeuralEncoder::new(tr.model.clone(), fa_core::Timescale::Minute, "parent");
        let ends: Vec<usize> = (16..=900).collect();
        let lat = enc.encode_windows(&bars, &ends).unwrap();
        let mut ctx = DecodeCtx::default();
        for (e, l) in ends.iter().zip(&lat) {
            ctx.push("syn", LatentPoint { as_of_ms: hist.ends_ms[e - 1], vec: l.mu.concat() });
        }
        let series = decoder_series(std::slice::from_ref(&hist), 60_000);
        let g = DecoderGenome::parse("nn_forecast:h=3;lb=12;width=8;steps=40;lr=0.01").unwrap();
        let mut d =
            dn::train(&g, &series, Arc::new(ctx), Arc::new(TextTimeline::new()), &dn::TrainOptions::default()).unwrap();
        assert_eq!(d.meta.full_latent_dim(), 6);
        d.meta.score = 0.1;
        let card = EliteCard {
            meta: d.meta.clone(),
            bucket: elite::BUCKET.into(),
            key: "decoders/t/x.safetensors".into(),
            created_ms: 1,
        };
        assert_eq!(eligible(std::slice::from_ref(&card), 60_000, 6).len(), 1);
        assert!(eligible(std::slice::from_ref(&card), 1_000, 6).is_empty(), "other step");
        assert!(eligible(std::slice::from_ref(&card), 60_000, 9).is_empty(), "other latent width");
        let set = FeedbackSet::load(&[(d.to_bundle().unwrap(), 1.0)], 6, &dev).unwrap();
        let fb = EliteFeedback::from_parts(set, series);
        assert_eq!(fb.n_elites(), 1);
        tr.feedback.push(Box::new(fb));
        let log = tr.train_steps(&ds, 5).unwrap();
        assert!(log.feedback.is_finite() && log.feedback > 0.0, "{log:?}");
        assert_eq!(tr.feedback[0].summary()["name"], "elite_decoders");
    }
}
