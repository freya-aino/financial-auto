//! `evo.reeval.decoder` — elite re-evaluation (spec 34 §8 "decoder exploits encoder artefacts":
//! elites are re-evaluated on the next snapshot and their feedback weight decays when they fail
//! transfer; spec 36 §6.4 item 5).
//!
//! The job lists the decoder cards in Garage, takes the elites (`feedback::reeval_candidates`,
//! `k` per output family, optionally one timescale), loads each bundle **frozen** and scores it on
//! the job's data window — only after its own training cutoff, with the latents (recorded
//! `LATENT` stream) and news / frames of that window, i.e. of the encoder snapshot now serving —
//! with the same objectives as training (`crps_skill` for forecasting families, `recon_skill`
//! otherwise). Each elite's card gains a `meta.transfer[]` record `{snapshot_id, score,
//! base_score, ratio = score / base_score, n_eval, window, at_ms, job_id}`; the feedback path
//! multiplies the elite's weight by `clamp(ratio, 0, 1)` and drops elites whose ratio is ≤ 0
//! (`fa_decoders::neural::feedback::{quota_weights, select_top_k}`). `dry_run` scores without
//! writing. No GPU lease (inference only).
//!
//! Spec: `{"snapshot_id"?, "timescale"?, "k"?: 4, "data_window"?: {"hours": 2} | {"start_ms",
//! "end_ms"}, "dry_run"?: false}` (`snapshot_id` defaults to `Job.snapshot_id`, else `live`).
//! Result (`evo.reeval.decoder/v0`): `{elites: [{key, genome, output_family, base_score, score,
//! ratio, n_eval, passed, error?}], check: {elites, evaluated, passed, worst_ratio}, tol}`.

use std::{sync::Arc, time::Instant};

use fa_backtest::eval::{EvalSpec, evaluate, pool};
use fa_blob::{BlobStore, ObjectKey};
use fa_decoders::{
    baseline::Series,
    genome::DecoderGenome,
    neural::{
        NeuralDecoder, TRANSFER_TOL, Transfer,
        feedback::{self, EliteCard},
    },
};
use fa_proto::v1::{Job, JobResult, job_result::Status};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    data::{self, DataWindow},
    job::{Outcome, Worker},
    market::now_ms,
    run::JobRun,
};

pub const KIND: &str = "evo.reeval.decoder";
pub const PROTOCOL: &str = "evo.reeval.decoder/v0";
/// Default budget (ms) of a re-evaluation job.
pub const DEFAULT_BUDGET_MS: u64 = 600_000;
const DEFAULT_K: usize = 4;
const DEFAULT_HOURS: f64 = 2.0;
/// Rows scored for the reconstruction objective.
const RECON_ROWS: usize = 20_000;

#[derive(Debug, Default, Deserialize)]
pub struct ReevalSpec {
    #[serde(default)]
    pub snapshot_id: Option<String>,
    #[serde(default)]
    pub timescale: Option<String>,
    #[serde(default)]
    pub k: Option<usize>,
    #[serde(default)]
    pub data_window: Option<DataWindow>,
    #[serde(default)]
    pub dry_run: bool,
}

/// The transfer record of one re-evaluation (`ratio` = score / base score; a base ≤ 0 never
/// reaches here, elites beat naive).
pub fn transfer_of(card: &EliteCard, snapshot_id: &str, score: f64, n_eval: usize, window: (i64, i64), job_id: &str) -> Transfer {
    let base = card.meta.score;
    let ratio = if base > 0.0 && score.is_finite() { score / base } else { 0.0 };
    Transfer {
        snapshot_id: snapshot_id.to_owned(),
        score,
        base_score: base,
        ratio,
        n_eval,
        window_start_ms: window.0,
        window_end_ms: window.1,
        at_ms: now_ms(),
        job_id: job_id.to_owned(),
    }
}

impl Worker {
    /// `evo.reeval.decoder` (see the module docs).
    pub(crate) async fn reevaluate(&self, job: &Job, run: &JobRun) -> Outcome {
        let started = Instant::now();
        let spec: ReevalSpec = if job.spec.is_empty() {
            ReevalSpec::default()
        } else {
            match serde_json::from_slice(&job.spec) {
                Ok(s) => s,
                Err(e) => return self.failed(job, format!("reeval spec: {e}"), 0),
            }
        };
        let Some(blob) = &self.blob else {
            return self.failed(job, "no Garage credentials: decoder cards cannot be listed".into(), 0);
        };
        let snapshot_id = spec
            .snapshot_id
            .clone()
            .filter(|s| !s.is_empty())
            .or_else(|| Some(job.snapshot_id.clone()).filter(|s| !s.is_empty()))
            .unwrap_or_else(|| "live".into());
        let listed = match blob.list(fa_blob::buckets::MODELS, feedback::PREFIX).await {
            Ok(l) => l,
            Err(e) => return self.failed(job, format!("list cards: {e}"), started.elapsed().as_millis() as u64),
        };
        let mut cards = Vec::new();
        for o in listed.iter().filter(|o| o.key.ends_with(feedback::CARD_SUFFIX)) {
            let key = ObjectKey { bucket: fa_blob::buckets::MODELS, key: o.key.clone() };
            if let Ok(b) = blob.get(&key).await
                && let Ok(c) = serde_json::from_slice::<EliteCard>(&b)
            {
                cards.push(c);
            }
        }
        let mut elites = feedback::reeval_candidates(&cards, spec.k.unwrap_or(DEFAULT_K).max(1));
        if let Some(ts) = spec.timescale.as_deref().filter(|t| !t.is_empty()) {
            elites.retain(|c| DecoderGenome::parse(&c.meta.genome).is_ok_and(|g| g.timescale.to_string() == ts));
        }
        let window = spec.data_window.unwrap_or(DataWindow { hours: Some(DEFAULT_HOURS), ..Default::default() });
        let n = elites.len();
        run.progress(0.0, format!("re-evaluating {n} elite decoder(s) on snapshot {snapshot_id}")).await;
        let mut rows = Vec::new();
        let mut updated = Vec::new();
        for (i, card) in elites.into_iter().enumerate() {
            if run.should_stop() {
                break;
            }
            let row = self.reevaluate_one(job, &card, &window, &snapshot_id).await;
            if let Some(t) = row.1 {
                let mut c = card.clone();
                c.meta.transfer.push(t);
                updated.push(c);
            }
            rows.push(row.0);
            run.progress((i + 1) as f32 / n.max(1) as f32, format!("{} / {n}", i + 1)).await;
        }
        if let Some(out) = run.interrupted(&self.worker_id) {
            return out;
        }
        let mut written = 0;
        if !spec.dry_run {
            for c in &updated {
                let key = ObjectKey { bucket: fa_blob::buckets::MODELS, key: feedback::card_key(&c.key) };
                if let Ok(b) = serde_json::to_vec_pretty(c)
                    && blob.put(&key, b, fa_blob::JSON).await.is_ok()
                {
                    written += 1;
                }
            }
        }
        // the item-5 view over the cards as they are now
        let mut after = cards.clone();
        for u in &updated {
            if let Some(c) = after.iter_mut().find(|c| c.key == u.key) {
                *c = u.clone();
            }
        }
        let check = feedback::transfer_check(&after, spec.k.unwrap_or(DEFAULT_K).max(1), &snapshot_id, TRANSFER_TOL);
        let payload = json!({
            "protocol": PROTOCOL,
            "snapshot_id": snapshot_id,
            "cards": cards.len(),
            "elites": rows,
            "check": check,
            "tol": TRANSFER_TOL,
            "written": written,
            "dry_run": spec.dry_run,
            "worker_id": self.worker_id,
        });
        tracing::info!(job = %job.job_id, snapshot = %snapshot_id, elites = n, evaluated = check.evaluated,
                       passed = check.passed, "elite decoders re-evaluated");
        let elapsed_ms = started.elapsed().as_millis() as u64;
        Outcome {
            result: JobResult {
                job_id: job.job_id.clone(),
                status: Status::Ok as i32,
                result: serde_json::to_vec(&payload).unwrap_or_default(),
                used_gpu_ms: elapsed_ms,
                worker_id: self.worker_id.clone(),
                objectives: [
                    ("transfer_evaluated".to_owned(), check.evaluated as f64),
                    ("transfer_passed".to_owned(), check.passed as f64),
                ]
                .into_iter()
                .collect(),
                ..Default::default()
            },
            outputs: Vec::new(),
        }
    }

    /// One elite: `(row JSON, transfer record when scored)`.
    async fn reevaluate_one(&self, job: &Job, card: &EliteCard, window: &DataWindow, snapshot_id: &str) -> (Value, Option<Transfer>) {
        let mut row = json!({"key": card.key, "genome": card.meta.genome, "output_family": card.meta.output_family,
                             "base_score": card.meta.score});
        let fail = |mut row: Value, e: String| {
            row["error"] = json!(e);
            (row, None)
        };
        let genome = match DecoderGenome::parse(&card.meta.genome) {
            Ok(g) => g,
            Err(e) => return fail(row, e.to_string()),
        };
        let Some(blob) = &self.blob else { return fail(row, "no Garage".into()) };
        let bytes = match blob.get(&ObjectKey { bucket: fa_blob::buckets::MODELS, key: card.key.clone() }).await {
            Ok(b) => b,
            Err(e) => return fail(row, format!("bundle: {e}")),
        };
        let need = genome.lookback + genome.horizon + 64;
        let loaded = data::load(self.tsdb.as_ref(), &self.market, &genome, window, self.data_window_hours, need).await;
        if loaded.series.is_empty() {
            return fail(row, format!("insufficient market data ({} steps)", loaded.longest));
        }
        let (start, end) = (loaded.source.window_start_ms, loaded.source.window_end_ms);
        let from = start.max(card.meta.cutoff_ms + 1);
        let (ctx, _) = self.training_latents(window, start, end).await;
        let (text, _) = self.context_timeline(&genome, start, end).await;
        let series: Vec<Series> = loaded.series;
        let cost: Vec<f64> = series
            .iter()
            .map(|s| crate::news::trade_cost_bp(&self.cost, crate::news::venue_of(&loaded.venues, &s.instrument_id), s))
            .collect();
        let scored = tokio::task::spawn_blocking(move || -> Result<(f64, usize), String> {
            let d = NeuralDecoder::from_bundle(&bytes, &fa_decoders::neural::Device::Cpu)
                .map_err(|e| e.to_string())?
                .with_context(ctx, Arc::new(text))
                .with_eval_from(Some(from));
            if genome.family.forecasts() {
                let reports: Vec<_> = series
                    .iter()
                    .zip(&cost)
                    .map(|(s, c)| evaluate(&d, s, &EvalSpec { cost_bp: *c, ..EvalSpec::standard() }))
                    .collect();
                let p = pool(&reports).ok_or("no out-of-sample step after the cutoff")?;
                Ok((p.crps_skill, p.n_eval))
            } else {
                let r = d.recon_eval(&series, from, RECON_ROWS).map_err(|e| e.to_string())?.ok_or("no reconstruction")?;
                if r.n == 0 {
                    return Err("no reconstruction target after the cutoff".into());
                }
                Ok((r.skill, r.n))
            }
        })
        .await;
        match scored {
            Ok(Ok((score, n_eval))) => {
                let t = transfer_of(card, snapshot_id, score, n_eval, (from, end), &job.job_id);
                row["score"] = json!(score);
                row["ratio"] = json!(t.ratio);
                row["n_eval"] = json!(n_eval);
                row["passed"] = json!(t.ratio >= 1.0 - TRANSFER_TOL);
                (row, Some(t))
            }
            Ok(Err(e)) => fail(row, e),
            Err(e) => fail(row, e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use fa_decoders::neural::BundleMeta;

    use super::*;

    #[test]
    fn transfer_ratio_and_spec_defaults() {
        let meta: BundleMeta = serde_json::from_value(json!({
            "format": "fa-decoder-bundle/v1", "genome": "nn_forecast", "family": "nn_forecast", "output_family": "rnn_prior",
            "layout": {"seq": 1, "extra": 0, "latent": 0, "out": 1, "target": 1}, "norm": {"lat_mean": [], "lat_std": []},
            "norm_version": 2, "target_mean": [0.0], "step_ms": 1000, "cutoff_ms": 0,
            "train": {"train_rows": 1, "val_rows": 1, "steps": 1, "train_loss": 0.0, "initial_loss": 0.0, "val_loss": 0.0,
                      "elapsed_ms": 0.0, "latent_dim": 0, "latent_coverage": 0.0, "interrupted": false, "warm_started": 0, "device": "Cpu"},
            "score": 0.4
        }))
        .unwrap();
        assert!(meta.transfer.is_empty(), "cards written before D-085 have no transfer records");
        let card = EliteCard { meta, bucket: "fa-models".into(), key: "k".into(), created_ms: 0 };
        let t = transfer_of(&card, "s1", 0.3, 10, (1, 2), "j");
        assert!((t.ratio - 0.75).abs() < 1e-12 && t.base_score == 0.4);
        assert_eq!(transfer_of(&card, "s1", f64::NAN, 0, (1, 2), "j").ratio, 0.0);
        let s: ReevalSpec = serde_json::from_str(r#"{"k": 2, "data_window": {"hours": 1}}"#).unwrap();
        assert_eq!((s.k, s.dry_run, s.snapshot_id), (Some(2), false, None));
    }

    /// `FA_NATS_TEST=1 cargo test -p evo-worker -- --nocapture reeval_live_dry_run` (compose
    /// stack, Garage credentials in `config/secrets.toml`): re-evaluates the live elite cards on
    /// the last `FA_REEVAL_HOURS` (default 2) hours **without writing** (dry run).
    #[tokio::test]
    async fn reeval_live_dry_run() {
        if std::env::var("FA_NATS_TEST").is_err() {
            eprintln!("skipping (set FA_NATS_TEST=1)");
            return;
        }
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
        let cfg = fa_core::config::PlatformConfig::load(format!("{root}/config/platform.toml")).unwrap();
        let secrets = fa_core::Secrets::load_from(format!("{root}/config/secrets.toml")).unwrap();
        let mut worker = crate::holdout::tests::offline_worker().await;
        worker.blob = Some(fa_blob::S3BlobStore::from_config(&cfg, &secrets).unwrap());
        worker.tsdb = Some(fa_tsdb::Tsdb::new("http://127.0.0.1:8428".to_owned()));
        worker.recorded = crate::recorded::RecordedStore::default();
        let hours: f64 = std::env::var("FA_REEVAL_HOURS").ok().and_then(|h| h.parse().ok()).unwrap_or(2.0);
        let job = Job {
            job_id: format!("reeval-dry-{}", ulid::Ulid::generate()),
            kind: KIND.into(),
            spec: serde_json::to_vec(&json!({"dry_run": true, "k": 2, "data_window": {"hours": hours}})).unwrap(),
            budget_gpu_ms: 900_000,
            ..Default::default()
        };
        let out = worker.handle(&job).await;
        assert_eq!(out.result.status, Status::Ok as i32, "{}", out.result.error);
        let v: Value = serde_json::from_slice(&out.result.result).unwrap();
        eprintln!("REEVAL {}", serde_json::to_string_pretty(&v).unwrap());
        assert_eq!(v["written"], 0, "dry run writes nothing");
    }
}
