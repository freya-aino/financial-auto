//! `evo.eval.decoder` for the **neural** genome families (`nn_*`, spec 34 §6.1, `fa_decoders::neural`):
//! train on the first `train` share of the evaluation window (CPU; `cuda` feature → GPU), then
//! score out of sample on the rest.
//!
//! - Forecasting families (`nn_forecast`, `nn_retrieval`, `nn_future`) go through
//!   `fa_backtest::eval` exactly like the baselines (D-049 objectives, trade P&L into the trial
//!   registry); the decoder forecasts only after the train cutoff.
//! - Reconstruction families (`nn_chart`, `nn_future`, `nn_news`, `nn_narrative`) add `recon_skill`
//!   = `1 − loss / loss(training-mean target)` (max). Objectives a family cannot have get the naive
//!   value (`crps_skill` 0, `hit_rate` 0.5, `trade_sharpe` 0, `trade_psr` 0.5, `recon_skill` 0), so
//!   one generation can mix families.
//! - The text families read the `TEXT` stream (`news.item.>`) over `[start − tw, end]` by
//!   knowledge (receive) time.
//! - The trained bundle goes to Garage at `fa-models/decoders/<generation>/<family>-<hash>-<job>.safetensors`
//!   with a JSON card (`….card.json`: metadata + objectives) next to it — what model-runtime and the
//!   encoder trainer list to find elites (`fa_decoders::neural::feedback`).
//! - Envelope key `warm_start` (`"<bucket>/<key>"` of a parent bundle) warm-starts matching
//!   variables (spec 34 §6.2.3).
//!
//! Training gets at most [`TRAIN_BUDGET_SHARE`] of the job budget; it stops early (keeping the best
//! validation weights) when that share is spent, and on preemption / cancellation.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use fa_backtest::eval::{EvalReport, EvalSpec, Stop, evaluate, pool};
use fa_blob::{BlobStore, ObjectKey};
use fa_bus::replay::{ReadOptions, read_range};
use fa_decoders::{
    baseline::Series,
    genome::DecoderGenome,
    neural::{
        self, NeuralDecoder, ReconReport, TrainOptions,
        data::{TextTimeline, text_window_ms},
        feedback::{self, EliteCard},
    },
};
use fa_proto::v1::{self, DecoderOutput, Job, JobResult, job_result::Status};
use serde_json::{Value, json};

use crate::{
    data::{self, DataWindow},
    gpu::{self, Lease},
    job::{DIRECTIONS, Outcome, PROTOCOL, Worker, horizon_refs, ts_ms},
    market::now_ms,
    run::JobRun,
    trials::{self, TrialRow},
};

/// Share of the job budget training may use; the rest is for the out-of-sample evaluation.
pub const TRAIN_BUDGET_SHARE: f64 = 0.6;
/// Minimum series length (steps) beyond the genome's own needs.
const MIN_EXTRA_STEPS: usize = 64;
/// Rows scored for the reconstruction objective.
const RECON_ROWS: usize = 20_000;
/// Cap on news items read for a text genome.
const MAX_NEWS: usize = 200_000;

/// Objective directions of neural results: the baseline set plus `recon_skill`.
pub fn directions() -> Vec<(&'static str, &'static str)> {
    let mut d: Vec<(&str, &str)> = DIRECTIONS.to_vec();
    d.push(("recon_skill", "max"));
    d
}

/// Naive values of the forecast objectives for families without a forecast.
fn neutral_forecast_objectives() -> Vec<(&'static str, f64)> {
    vec![("crps_skill", 0.0), ("hit_rate", 0.5), ("trade_sharpe", 0.0), ("trade_psr", 0.5)]
}

/// The objective map of a neural evaluation (see the module docs).
pub fn objectives(
    genome: &DecoderGenome,
    pooled: Option<&EvalReport>,
    recon: Option<&ReconReport>,
    ops_per_forecast: f64,
) -> Vec<(String, f64)> {
    let mut o: Vec<(String, f64)> = match pooled {
        Some(p) => p.objectives().into_iter().map(|(k, v)| (k.to_owned(), v)).collect(),
        None => neutral_forecast_objectives().into_iter().map(|(k, v)| (k.to_owned(), v)).collect(),
    };
    o.retain(|(k, _)| k != "compute_cost" && k != "n_eval");
    o.push(("compute_cost".into(), ops_per_forecast.max(1.0).log10()));
    let recon_skill = recon.map_or(0.0, |r| if r.skill.is_finite() { r.skill } else { 0.0 });
    o.push(("recon_skill".into(), recon_skill));
    let n_eval = pooled.map_or(0, |p| p.n_eval).max(recon.map_or(0, |r| r.n));
    o.push(("n_eval".into(), n_eval as f64));
    let _ = genome;
    o
}

/// Family score: `crps_skill` when the family forecasts, else `recon_skill`.
pub fn family_score(genome: &DecoderGenome, obj: &[(String, f64)]) -> f64 {
    let key = if genome.family.forecasts() { "crps_skill" } else { "recon_skill" };
    obj.iter().find(|(k, _)| k == key).map_or(0.0, |(_, v)| *v)
}

/// `TextTimeline` from recorded `news.item.*` envelopes, items tagged with the instruments
/// `index` matches (see [`crate::news`]).
pub fn timeline_of(
    msgs: &[fa_bus::replay::RecordedMessage],
    index: &fa_decoders::text_baseline::InstrumentIndex,
) -> (TextTimeline, usize) {
    crate::news::timeline_of(msgs, index)
}

/// Successive-halving share of the genome's SGD steps for this rung (`train_steps_fraction` in
/// the eval envelope, `(0, 1]`; spec 34 §6.2 step 4: rung `r` of `R` trains `η^-(R−1−r)` of them).
pub fn steps_fraction(job: &Job) -> Option<f64> {
    let v: Value = serde_json::from_slice(&job.spec).ok()?;
    v.get("train_steps_fraction").and_then(Value::as_f64).filter(|f| *f > 0.0 && *f <= 1.0)
}

/// Optional `warm_start` (`"<bucket>/<key>"`) of an eval envelope.
pub fn warm_start_ref(job: &Job) -> Option<String> {
    let v: Value = serde_json::from_slice(&job.spec).ok()?;
    v.get("warm_start").and_then(Value::as_str).filter(|s| s.contains('/')).map(str::to_owned)
}

fn object_key(bucket_key: &str) -> Option<ObjectKey> {
    let (bucket, key) = bucket_key.split_once('/')?;
    let bucket: &'static str = match bucket {
        b if b == fa_blob::buckets::MODELS => fa_blob::buckets::MODELS,
        b if b == fa_blob::buckets::CKPT => fa_blob::buckets::CKPT,
        b if b == fa_blob::buckets::ARTIFACTS => fa_blob::buckets::ARTIFACTS,
        _ => return None,
    };
    Some(ObjectKey { bucket, key: key.to_owned() })
}

struct Trained {
    decoder: NeuralDecoder,
    reports: Vec<EvalReport>,
    recon: Option<ReconReport>,
    bundle: Result<Vec<u8>, String>,
    samples: Value,
}

impl Worker {
    /// `evo.eval.decoder` for an `nn_*` genome.
    pub(crate) async fn evaluate_neural(&self, job: &Job, run: &JobRun, genome: DecoderGenome, window: DataWindow) -> Outcome {
        let started = Instant::now();
        let budget_ms = run.budget_ms();
        run.progress(0.0, format!("parsed neural genome: {}", genome.summary())).await;

        // GPU lease (the arbiter accounts CPU training under the same class)
        let mut notices = self.notices.subscribe();
        let lease = if self.gpu_lease_mb > 0 {
            match gpu::acquire(&self.bus, &mut notices, &job.job_id, self.gpu_lease_mb, run.deadline).await {
                Ok(l) => l,
                Err(e) => return self.failed(job, e, started.elapsed().as_millis() as u64),
            }
        } else {
            Lease::Unavailable
        };
        drop(notices);
        let release = async {
            if lease == Lease::Granted {
                gpu::release(&self.bus, &job.job_id).await;
            }
        };

        // data
        let need = genome.lookback + genome.horizon + MIN_EXTRA_STEPS;
        let loaded = data::load(self.tsdb.as_ref(), &self.market, &genome, &window, self.data_window_hours, need).await;
        let src = loaded.source;
        let venues = loaded.venues;
        let series: Vec<Series> = loaded.series;
        if series.is_empty() {
            release.await;
            return self.failed(
                job,
                format!(
                    "insufficient market data: need ≥ {need} resampled {} steps for `{}`, longest series has {} (source: {}; vm: {})",
                    genome.timescale,
                    genome.summary(),
                    loaded.longest,
                    src.source,
                    src.vm_error.as_deref().unwrap_or("ok"),
                ),
                started.elapsed().as_millis() as u64,
            );
        }
        let (ctx, latent_block) = self.training_latents(&window, src.window_start_ms, src.window_end_ms).await;
        let (latent_inst, latent_points) = ctx.stats();
        let (text, text_block) = self.context_timeline(&genome, src.window_start_ms, src.window_end_ms).await;
        if genome.family.uses_frames() && text.is_empty() {
            release.await;
            return self.failed(
                job,
                format!(
                    "no frames: the MEDIA stream has no image / video chunk with a readable blob in the window                      (visual.frame needs a media producer, spec 10); {text_block}"
                ),
                started.elapsed().as_millis() as u64,
            );
        }
        let warm_ref = warm_start_ref(job);
        let train_steps = steps_fraction(job).map(|f| ((genome.nn.steps as f64 * f).ceil() as usize).max(1));
        let cost_bps: Vec<f64> =
            series.iter().map(|s| crate::news::trade_cost_bp(&self.cost, crate::news::venue_of(&venues, &s.instrument_id), s)).collect();
        let warm: Option<Vec<u8>> = match (&warm_ref, &self.blob) {
            (Some(r), Some(blob)) => match object_key(r) {
                Some(k) => match blob.get(&k).await {
                    Ok(b) => Some(b),
                    Err(e) => {
                        tracing::warn!(job = %job.job_id, warm = %r, error = %e, "warm-start bundle not loaded");
                        None
                    }
                },
                None => None,
            },
            _ => None,
        };

        let warm_loaded = warm.is_some();

        // train + evaluate (blocking)
        run.start_budget();
        run.progress(0.05, format!("training {} on {} series", genome.family.as_str(), series.len())).await;
        let task = {
            let genome = genome.clone();
            let run = run.clone();
            let ctx = ctx.clone();
            let train_budget = Duration::from_millis((budget_ms as f64 * TRAIN_BUDGET_SHARE) as u64);
            let cost_bps = cost_bps.clone();
            tokio::task::spawn_blocking(move || -> Result<Trained, String> {
                let t0 = Instant::now();
                let stop_run = run.clone();
                let opts = TrainOptions {
                    should_stop: Some(Box::new(move || {
                        stop_run.is_cancelled()
                            || stop_run.is_preempted()
                            || stop_run.past_deadline()
                            || t0.elapsed() > train_budget
                    })),
                    warm_start: warm,
                    steps: train_steps,
                    ..Default::default()
                };
                let text = Arc::new(text);
                let decoder =
                    neural::train(&genome, &series, ctx, text.clone(), &opts).map_err(|e| e.to_string())?;
                let cutoff = decoder.meta.cutoff_ms;
                let mut reports = Vec::new();
                if genome.family.forecasts() {
                    let n = series.len();
                    for (i, s) in series.iter().enumerate() {
                        let progress = run.on_progress();
                        let spec = EvalSpec {
                            should_stop: Some(run.stopper()),
                            on_progress: Some(Box::new(move |done, total| progress(done + total * i, total * n))),
                            progress_every: 64,
                            cost_bp: cost_bps[i],
                            ..EvalSpec::standard()
                        };
                        reports.push(evaluate(&decoder, s, &spec));
                        if run.should_stop() {
                            break;
                        }
                    }
                }
                let recon = decoder.recon_eval(&series, cutoff, RECON_ROWS).map_err(|e| e.to_string())?;
                let bundle = decoder.to_bundle().map_err(|e| e.to_string());
                let samples = inspection_samples(&decoder, &series, &text, SAMPLES);
                Ok(Trained { decoder, reports, recon, bundle, samples })
            })
        };
        let done = run.drive(task, &genome.summary()).await;
        release.await;
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let trained = match done {
            Ok(Ok(t)) => t,
            Ok(Err(e)) => return self.failed(job, format!("neural decoder: {e}"), elapsed_ms),
            Err(e) => return self.failed(job, format!("training task failed: {e}"), elapsed_ms),
        };
        if let Some(out) = run.interrupted(&self.worker_id) {
            return out;
        }
        let Trained { mut decoder, reports, recon, bundle, samples } = trained;
        let pooled = pool(&reports);
        if genome.family.forecasts() && pooled.is_none() && recon.is_none_or(|r| r.n == 0) {
            let status = if run.over_budget() { Status::BudgetExceeded } else { Status::Failed };
            return Outcome {
                result: JobResult {
                    job_id: job.job_id.clone(),
                    status: status as i32,
                    error: format!(
                        "no out-of-sample step evaluated after the train cutoff within the {budget_ms} ms budget"
                    ),
                    used_gpu_ms: elapsed_ms,
                    worker_id: self.worker_id.clone(),
                    ..Default::default()
                },
                outputs: Vec::new(),
            };
        }
        if !genome.family.forecasts() && recon.is_none_or(|r| r.n == 0) {
            return self.failed(job, "no out-of-sample row with a known reconstruction target".into(), elapsed_ms);
        }
        let ops = fa_decoders::baseline::SeriesDecoder::ops_per_forecast(&decoder, 0);
        let obj = objectives(&genome, pooled.as_ref(), recon.as_ref(), ops);
        let score = family_score(&genome, &obj);
        decoder.meta.objectives = obj.iter().cloned().collect();
        decoder.meta.score = score;
        decoder.meta.job_id = job.job_id.clone();
        decoder.meta.generation = job.parent_job_id.clone();
        let weights = self.persist_bundle(job, &decoder, bundle).await;
        let truncated = reports.iter().any(|r| r.stop == Stop::Interrupted);

        let (gen_index, rung) = Self::parse_position(job);
        let objectives_json: serde_json::Map<String, Value> = obj.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
        let generation_stats = match &pooled {
            Some(p) => {
                trials::register(
                    self.trials.as_ref(),
                    &TrialRow {
                        generation: if job.parent_job_id.is_empty() { "adhoc".into() } else { job.parent_job_id.clone() },
                        gen_index,
                        rung,
                        job_id: job.job_id.clone(),
                        candidate: genome.to_string(),
                        kind: crate::job::KIND.into(),
                        window_start_ms: src.window_start_ms,
                        window_end_ms: src.window_end_ms,
                        sharpe: p.trade_sharpe,
                        n_obs: p.n_eval,
                        objectives: Value::Object(objectives_json.clone()),
                        returns: p.pnl.clone(),
                        holdout_id: None,
                    },
                )
                .await
            }
            None => Value::Null,
        };
        let instruments: Vec<Value> = reports
            .iter()
            .map(|r| {
                let venue = crate::news::venue_of(&venues, &r.instrument_id);
                json!({"instrument_id": r.instrument_id, "venue": venue, "n_steps": r.n_series, "n_eval": r.n_eval,
                       "stop": r.stop, "crps_skill": r.crps_skill, "coverage": r.coverage, "last_ts_ms": r.last_ts_ms,
                       "cost_bp": crate::job::eval_inputs_cost(&cost_bps, &reports, &r.instrument_id)})
            })
            .collect();
        let report_key = fa_blob::eval_report(&job.parent_job_id, &job.job_id);
        let dirs = directions();
        let mut payload = json!({
            "protocol": PROTOCOL,
            "score": score,
            "objectives": objectives_json,
            "directions": dirs.iter().map(|(k, d)| ((*k).to_owned(), json!(d))).collect::<serde_json::Map<_, _>>(),
            "genome": genome,
            "neural": {
                "decoder_id": decoder.decoder_id(),
                "output_family": decoder.meta.output_family,
                "params": decoder.n_params(),
                "layout": decoder.meta.layout,
                "cutoff_ms": decoder.meta.cutoff_ms,
                "train": decoder.meta.train,
                "recon": recon,
                "weights": weights,
            },
            "data": {"source": src.source, "timescale": src.timescale, "step_ms": src.step_ms,
                     "window_start_ms": src.window_start_ms, "window_end_ms": src.window_end_ms,
                     "vm_url": src.vm_url, "vm_series": src.vm_series, "vm_error": src.vm_error,
                     "latents": {"instruments": latent_inst, "points": latent_points, "source": latent_block["source"], "recorded": latent_block},
                     "text": text_block,
                     "instruments": instruments},
            "generation_stats": generation_stats,
            "budget": {"budget_ms": budget_ms, "elapsed_ms": elapsed_ms, "truncated": truncated,
                       "train_ms": decoder.meta.train.elapsed_ms, "train_interrupted": decoder.meta.train.interrupted,
                       "train_steps": train_steps, "train_steps_fraction": steps_fraction(job)},
            "warm_start": {"ref": warm_ref, "loaded": warm_loaded, "variables": decoder.meta.train.warm_started},
            "trade_cost": {"model": self.cost.id, "notional_eur": crate::news::TRADE_NOTIONAL_EUR},
            "stop": pooled.as_ref().map_or(json!("complete"), |p| json!(p.stop)),
            "report_ref": {"planned": format!("{}/{}", report_key.bucket, report_key.key), "persisted": self.blob.is_some()},
            "worker_id": self.worker_id,
        });
        // EvaluatedOn / Reads (spec 34 §5) and inspection samples (decoder-out/), best effort
        payload["helix"] = self.record_evaluation(job, &genome, score, &Value::Object(objectives_json.clone())).await;
        payload["decoder_out"] = self.persist_samples(job, &decoder, samples).await;
        if !self.persist_report(&report_key, &payload).await {
            payload["report_ref"]["persisted"] = json!(false);
        }
        tracing::info!(job = %job.job_id, genome = %genome.summary(), score, train_ms = decoder.meta.train.elapsed_ms,
                       steps = decoder.meta.train.steps, "neural decoder evaluated");
        let outputs = if matches!(decoder.meta.output_family.as_str(), "rnn_prior" | "retrieval") {
            reports
                .iter()
                .filter(|r| r.last_forecast.is_some())
                .take(4)
                .map(|r| neural_output(job, &decoder, r))
                .collect()
        } else {
            // text.* / visual.* outputs follow the model-runtime score contract; they are served
            // there (elite bundles), not published from an evaluation.
            Vec::new()
        };
        Outcome {
            result: JobResult {
                job_id: job.job_id.clone(),
                status: Status::Ok as i32,
                result: serde_json::to_vec(&payload).unwrap_or_default(),
                used_gpu_ms: elapsed_ms,
                worker_id: self.worker_id.clone(),
                objectives: obj.into_iter().collect(),
                directions: dirs.iter().map(|(k, d)| ((*k).to_owned(), (*d).to_owned())).collect(),
                ..Default::default()
            },
            outputs,
        }
    }

    /// News timeline (`TEXT`, text families) or frame timeline (`MEDIA` + Garage blobs,
    /// `nn_frame`) over `[start − tw, end]` by receive time; empty for the other families.
    pub(crate) async fn context_timeline(&self, genome: &DecoderGenome, start_ms: i64, end_ms: i64) -> (TextTimeline, Value) {
        if genome.family.uses_text() {
            let from = start_ms - text_window_ms(genome);
            let opts = ReadOptions { max_messages: MAX_NEWS, ..Default::default() };
            let read = tokio::time::timeout(
                Duration::from_secs(120),
                read_range(&self.bus.jetstream(), "TEXT", "news.item.>", from, end_ms, &opts),
            )
            .await;
            match read {
                Ok(Ok(msgs)) => {
                    let index = self.news_refs.index();
                    let (t, bad) = timeline_of(&msgs, &index);
                    let block = json!({"source": "TEXT", "messages": msgs.len(), "items": t.len(), "matched": t.matched(),
                                       "reference_instruments": index.len(), "scope": if genome.nn.news_inst { "instrument" } else { "market" },
                                       "undecodable": bad, "from_ms": from});
                    (t, block)
                }
                Ok(Err(e)) => (TextTimeline::new(), json!({"source": "TEXT", "error": e.to_string()})),
                Err(_) => (TextTimeline::new(), json!({"source": "TEXT", "error": "read timeout"})),
            }
        } else if genome.family.uses_frames() {
            let from = start_ms - text_window_ms(genome);
            let opts = ReadOptions { max_messages: crate::frames::MAX_FRAMES, ..Default::default() };
            let read = tokio::time::timeout(
                Duration::from_secs(120),
                read_range(&self.bus.jetstream(), "MEDIA", "media.>", from, end_ms, &opts),
            )
            .await;
            match read {
                Ok(Ok(msgs)) => {
                    let (refs, skipped) = crate::frames::frame_refs(&msgs);
                    let (t, mut block) = crate::frames::timeline(self.blob.as_ref(), &refs).await;
                    block["messages"] = json!(msgs.len());
                    block["skipped"] = json!(skipped);
                    block["from_ms"] = json!(from);
                    (t, block)
                }
                Ok(Err(e)) => (TextTimeline::frames(), json!({"source": "MEDIA", "error": e.to_string()})),
                Err(_) => (TextTimeline::frames(), json!({"source": "MEDIA", "error": "read timeout"})),
            }
        } else {
            (TextTimeline::new(), Value::Null)
        }
    }

    /// Latents for training: the recorded `LATENT` stream over the whole window whenever the
    /// recorder is enabled (training needs the past, which the live buffer only has since the
    /// worker started), else the same choice as the baselines.
    pub(crate) async fn training_latents(
        &self,
        window: &DataWindow,
        start_ms: i64,
        end_ms: i64,
    ) -> (Arc<fa_decoders::DecodeCtx>, Value) {
        if self.recorded.enabled {
            let l = self.recorded.load_window(&self.bus.jetstream(), start_ms - self.recorded.preload_ms, end_ms).await;
            if let Some(rec) = &l.latents {
                let ctx = Arc::new(crate::recorded::decode_ctx(rec));
                let mut j = self.recorded.json(&l);
                j["source"] = json!("recorded");
                return (ctx, j);
            }
        }
        self.decoder_latents(window, start_ms, end_ms).await
    }

    /// Writes bundle + card to Garage; returns the `weights` JSON block of the payload.
    async fn persist_bundle(&self, job: &Job, decoder: &NeuralDecoder, bundle: Result<Vec<u8>, String>) -> Value {
        let key = feedback::bundle_key(&job.parent_job_id, &decoder.meta, &job.job_id);
        let bytes = match bundle {
            Ok(b) => b,
            Err(e) => return json!({"persisted": false, "error": e}),
        };
        let size = bytes.len();
        let Some(blob) = &self.blob else {
            return json!({"persisted": false, "reason": "no Garage credentials", "bytes": size});
        };
        let bundle_key = ObjectKey { bucket: fa_blob::buckets::MODELS, key: key.clone() };
        if let Err(e) = blob.put(&bundle_key, bytes, fa_blob::OCTET_STREAM).await {
            tracing::warn!(error = %e, key = %key, "decoder bundle persist failed");
            return json!({"persisted": false, "error": e.to_string()});
        }
        let card = EliteCard {
            meta: decoder.meta.clone(),
            bucket: fa_blob::buckets::MODELS.into(),
            key: key.clone(),
            created_ms: now_ms(),
        };
        let card_key = ObjectKey { bucket: fa_blob::buckets::MODELS, key: feedback::card_key(&key) };
        let card_ok = match serde_json::to_vec_pretty(&card) {
            Ok(b) => blob.put(&card_key, b, fa_blob::JSON).await.is_ok(),
            Err(_) => false,
        };
        json!({"persisted": true, "bucket": fa_blob::buckets::MODELS, "key": key, "card": card_key.key,
               "card_persisted": card_ok, "bytes": size})
    }
}

/// Inspection samples per evaluated decoder (spec 34 §5 "outputs for inspection").
const SAMPLES: usize = 6;

/// Up to `n` out-of-sample outputs of a trained decoder, evenly spread over the evaluation window
/// of its longest series: what it produced (quantiles / raster / embedding / summary), what was
/// realised, and the summary sentences for `text.summary` — the evolution lab's sample view.
pub fn inspection_samples(d: &NeuralDecoder, series: &[Series], text: &TextTimeline, n: usize) -> Value {
    let g = d.genome();
    let Some(s) = series.iter().max_by_key(|s| s.len()) else { return json!([]) };
    let first = s.ts_ms.partition_point(|t| *t < d.meta.cutoff_ms).max(g.lookback);
    let last = s.len().saturating_sub(g.horizon + 1);
    if first >= last || n == 0 {
        return json!([]);
    }
    let round = |v: &[f32]| -> Vec<f64> { v.iter().map(|x| (f64::from(*x) * 1e4).round() / 1e4).collect() };
    let step = ((last - first) / n).max(1);
    let mut out = Vec::new();
    for t in (first..last).step_by(step).take(n) {
        let lat = d.own_latent(s, t);
        let Some(o) = d.serve_at(s, t, lat.as_deref(), text) else { continue };
        let sigma = fa_decoders::neural::data::sigma_at(s, t, g.lookback).unwrap_or(1.0);
        let target = fa_decoders::neural::data::target_at(g, s, t, sigma, text).map(|(y, _)| y);
        let mut row = json!({"instrument_id": s.instrument_id, "ts_ms": s.ts_ms[t], "score": o.score,
                             "components": o.components});
        if !o.quantiles.is_empty() {
            row["quantiles"] = json!(o.quantiles);
            row["levels"] = json!(o.levels);
            row["realized"] = json!(s.forward_return(t, g.horizon));
        }
        if g.family.reconstructs() {
            row["recon"] = json!(round(&o.recon));
            row["target"] = json!(target.as_deref().map(round));
            if matches!(g.family, fa_decoders::genome::BaselineFamily::NnChart | fa_decoders::genome::BaselineFamily::NnFuture) {
                row["res"] = json!(g.nn.res);
            }
        }
        if !o.text.is_empty() {
            row["text"] = json!(o.text);
            row["target_text"] = json!(target.as_deref().map(fa_decoders::summary::summary_text));
        }
        out.push(row);
    }
    Value::Array(out)
}

impl Worker {
    /// Writes the inspection samples to `fa-artifacts/decoder-out/<decoder_id>/<job_id>.json`.
    async fn persist_samples(&self, job: &Job, d: &NeuralDecoder, samples: Value) -> Value {
        let n = samples.as_array().map_or(0, Vec::len);
        let key = ObjectKey {
            bucket: fa_blob::buckets::ARTIFACTS,
            key: format!("decoder-out/{}/{}.json", d.decoder_id(), job.job_id),
        };
        let Some(blob) = &self.blob else {
            return json!({"persisted": false, "reason": "no Garage credentials", "samples": n});
        };
        let doc = json!({"protocol": "decoder-out/v1", "decoder_id": d.decoder_id(), "genome": d.meta.genome,
                         "output_family": d.meta.output_family, "cutoff_ms": d.meta.cutoff_ms, "job_id": job.job_id,
                         "generation": job.parent_job_id, "score": d.meta.score, "samples": samples});
        match blob.put(&key, serde_json::to_vec(&doc).unwrap_or_default(), fa_blob::JSON).await {
            Ok(()) => json!({"persisted": true, "bucket": key.bucket, "key": key.key, "samples": n}),
            Err(e) => json!({"persisted": false, "error": e.to_string(), "samples": n}),
        }
    }
}

/// `DecoderOutput` of a neural forecast (same contract as the baselines: quantiles, analogs, `h:`).
fn neural_output(job: &Job, d: &NeuralDecoder, r: &EvalReport) -> DecoderOutput {
    let f = r.last_forecast.as_ref().expect("filtered");
    DecoderOutput {
        decoder_id: d.decoder_id(),
        genome_id: job.genome_id.clone(),
        family: d.meta.output_family.clone(),
        snapshot_id: job.snapshot_id.clone(),
        timescale: d.genome().timescale.to_string(),
        as_of: Some(ts_ms(r.last_ts_ms)),
        output_refs: horizon_refs(&r.instrument_id, &job.job_id, d.genome().horizon),
        quantiles: f.quantiles.clone(),
        analogs: f
            .analogs
            .iter()
            .map(|a| v1::Analog {
                window_id: a.window_id.clone(),
                window_end: Some(ts_ms(r.last_ts_ms - (a.time_distance_days * 86_400_000.0) as i64)),
                similarity: a.similarity,
                time_distance_days: a.time_distance_days,
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn objectives_cover_every_direction_for_every_family() {
        let keys: Vec<&str> = directions().iter().map(|(k, _)| *k).collect();
        for fam in ["nn_chart", "nn_news", "nn_narrative"] {
            let g = DecoderGenome::parse(fam).unwrap();
            let recon = ReconReport { n: 10, loss: 1.0, baseline: 2.0, skill: 0.5 };
            let o = objectives(&g, None, Some(&recon), 1000.0);
            for k in &keys {
                if *k == "crps" || *k == "coverage_err" {
                    continue; // not defined without a forecast (missing = worst, D-049)
                }
                assert!(o.iter().any(|(n, _)| n == k), "{fam}: {k} missing");
            }
            assert_eq!(family_score(&g, &o), 0.5);
            assert_eq!(o.iter().find(|(k, _)| k == "compute_cost").unwrap().1, 3.0);
        }
        let g = DecoderGenome::parse("nn_forecast").unwrap();
        assert_eq!(family_score(&g, &[("crps_skill".into(), 0.2), ("recon_skill".into(), 0.0)]), 0.2);
    }

    #[test]
    fn envelope_warm_start_and_object_keys() {
        let job = Job { spec: br#"{"genome":"nn_forecast","warm_start":"fa-models/decoders/g/x.safetensors"}"#.to_vec(), ..Default::default() };
        let r = warm_start_ref(&job).unwrap();
        let k = object_key(&r).unwrap();
        assert_eq!((k.bucket, k.key.as_str()), ("fa-models", "decoders/g/x.safetensors"));
        assert!(object_key("nope/x").is_none());
        assert!(warm_start_ref(&Job { spec: b"nn_forecast:h=3".to_vec(), ..Default::default() }).is_none());
    }

    #[test]
    fn news_envelopes_become_a_timeline() {
        use fa_bus::replay::RecordedMessage;
        use fa_proto::{
            Message,
            v1::{Envelope, envelope::Payload},
        };
        let env = |title: &str| Envelope {
            payload: Some(Payload::News(v1::NewsItem { title: title.into(), ..Default::default() })),
            ..Default::default()
        };
        let msgs = vec![
            RecordedMessage { subject: "news.item.x".into(), stream_seq: 2, published_ms: 2000, payload: env("Stocks rally").encode_to_vec() },
            RecordedMessage { subject: "news.item.x".into(), stream_seq: 1, published_ms: 1000, payload: env("Shares plunge").encode_to_vec() },
            RecordedMessage { subject: "news.item.x".into(), stream_seq: 3, published_ms: 3000, payload: vec![0xff, 0xff] },
        ];
        let (t, bad) = timeline_of(&msgs, &Default::default());
        assert_eq!((t.len(), bad), (2, 1));
        assert_eq!(t.span(), Some((1000, 2000)));
    }

    /// `FA_NATS_TEST=1 cargo test -p evo-worker -- --nocapture neural_eval_live` against the compose
    /// stack (NATS, VictoriaMetrics at 127.0.0.1:8428, the `LATENT` / `TEXT` streams): one forecasting
    /// and one text genome train and score out of sample on the last hours of real data.
    #[tokio::test]
    async fn neural_eval_live() {
        if std::env::var("FA_NATS_TEST").is_err() {
            eprintln!("skipping (set FA_NATS_TEST=1)");
            return;
        }
        let mut worker = crate::holdout::tests::offline_worker().await;
        worker.tsdb = Some(fa_tsdb::Tsdb::new(std::env::var("FA_VM_URL").unwrap_or_else(|_| "http://127.0.0.1:8428".into())));
        worker.recorded = crate::recorded::RecordedStore::default();
        let now = now_ms();
        for (genome, hours) in [
            ("nn_forecast:h=5;lb=32;width=16;steps=150;train=0.6", 1.0),
            ("nn_retrieval:h=5;lb=16;k=20;emb=8;steps=100;pool=512", 1.0),
            ("nn_news:lb=16;tdim=64;tw=10;steps=150", 0.4),
            ("nn_future:h=5;lb=16;res=8;steps=150", 1.0),
        ] {
            let spec = format!(
                r#"{{"genome":"{genome}","data_window":{{"start_ms":{},"end_ms":{}}}}}"#,
                now - (hours * 3_600_000.0) as i64 - 120_000,
                now - 120_000
            );
            let job = Job {
                job_id: format!("nn-live-{}", ulid::Ulid::generate()),
                kind: crate::job::KIND.into(),
                budget_gpu_ms: 240_000,
                spec: spec.into_bytes(),
                ..Default::default()
            };
            let out = worker.handle(&job).await;
            assert_eq!(out.result.status, Status::Ok as i32, "{genome}: {}", out.result.error);
            let v: Value = serde_json::from_slice(&out.result.result).unwrap();
            eprintln!(
                "{genome}: score {} objectives {} train {} text {} latents {}",
                v["score"], v["objectives"], v["neural"]["train"], v["data"]["text"], v["data"]["latents"]["points"]
            );
            assert!(out.result.objectives.contains_key("recon_skill"));
            assert!(v["neural"]["train"]["steps"].as_u64().unwrap() > 0);
        }
    }

    /// Read-only probe of arbitrary genomes on live data (no Garage, no registry, no outputs):
    /// `FA_NATS_TEST=1 FA_PROBE='nn_chart:ts=1m;lb=16@24|nn_news:tw=60@12' cargo test -p evo-worker
    /// -- --nocapture neural_probe_live` — `|`-separated `genome@hours` (window ending 2 min ago).
    #[tokio::test]
    async fn neural_probe_live() {
        let (Ok(_), Ok(probe)) = (std::env::var("FA_NATS_TEST"), std::env::var("FA_PROBE")) else {
            eprintln!("skipping (set FA_NATS_TEST=1 and FA_PROBE)");
            return;
        };
        let mut worker = crate::holdout::tests::offline_worker().await;
        worker.tsdb = Some(fa_tsdb::Tsdb::new(std::env::var("FA_VM_URL").unwrap_or_else(|_| "http://127.0.0.1:8428".into())));
        worker.recorded = crate::recorded::RecordedStore::default();
        let now = now_ms();
        for item in probe.split('|').filter(|s| !s.trim().is_empty()) {
            let (genome, hours) = item.rsplit_once('@').map_or((item, 1.0), |(g, h)| (g, h.parse().unwrap_or(1.0)));
            let spec = format!(
                r#"{{"genome":"{genome}","data_window":{{"start_ms":{},"end_ms":{}}}}}"#,
                now - (hours * 3_600_000.0) as i64 - 120_000,
                now - 120_000
            );
            let job = Job {
                job_id: format!("nn-probe-{}", ulid::Ulid::generate()),
                kind: crate::job::KIND.into(),
                budget_gpu_ms: 900_000,
                spec: spec.into_bytes(),
                ..Default::default()
            };
            let out = worker.handle(&job).await;
            if out.result.status != Status::Ok as i32 {
                eprintln!("PROBE {genome}@{hours}: FAILED {}", out.result.error);
                continue;
            }
            let v: Value = serde_json::from_slice(&out.result.result).unwrap();
            eprintln!(
                "PROBE {genome}@{hours}: score {} recon {} train {} text {} latents {}",
                v["score"], v["neural"]["recon"], v["neural"]["train"], v["data"]["text"], v["data"]["latents"]["points"]
            );
        }
    }

    /// `FA_NATS_TEST=1 cargo test -p evo-worker -- --nocapture feedback_from_live_elites` (compose
    /// stack, Garage credentials in `config/secrets.toml`): the D-009 path on real artifacts — list
    /// the decoder cards, pick the top elites for the live latent dimension, load the bundles
    /// frozen, build a batch from VictoriaMetrics prices + recorded latents and check that the
    /// auxiliary loss has a gradient on the latents.
    #[tokio::test]
    async fn feedback_from_live_elites() {
        if std::env::var("FA_NATS_TEST").is_err() {
            eprintln!("skipping (set FA_NATS_TEST=1)");
            return;
        }
        use fa_decoders::neural::{Device, Tensor, Var};
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
        let cfg = fa_core::config::PlatformConfig::load(format!("{root}/config/platform.toml")).unwrap();
        let secrets = fa_core::Secrets::load_from(format!("{root}/config/secrets.toml")).unwrap();
        let blob = fa_blob::S3BlobStore::from_config(&cfg, &secrets).unwrap();
        let listed = blob.list(fa_blob::buckets::MODELS, feedback::PREFIX).await.unwrap();
        let mut cards = Vec::new();
        for o in listed.iter().filter(|o| o.key.ends_with(feedback::CARD_SUFFIX)) {
            let b = blob.get(&ObjectKey { bucket: fa_blob::buckets::MODELS, key: o.key.clone() }).await.unwrap();
            cards.push(serde_json::from_slice::<EliteCard>(&b).unwrap());
        }
        let dim = cards.iter().map(|c| c.meta.layout.latent).max().unwrap_or(0);
        assert!(dim > 0, "no latent-conditioned decoder card in Garage ({} cards)", cards.len());
        let elites = feedback::select_top_k(&cards, 2, Some(dim));
        assert!(!elites.is_empty());
        let mut bundles = Vec::new();
        for (c, w) in feedback::quota_weights(&elites) {
            let b = blob.get(&ObjectKey { bucket: fa_blob::buckets::MODELS, key: c.key.clone() }).await.unwrap();
            bundles.push((b, w));
        }
        let set = feedback::FeedbackSet::load(&bundles, dim, &Device::Cpu).unwrap();
        // data: the last hour at 1 s + recorded latents
        let mut worker = crate::holdout::tests::offline_worker().await;
        worker.tsdb = Some(fa_tsdb::Tsdb::new("http://127.0.0.1:8428".to_owned()));
        worker.recorded = crate::recorded::RecordedStore::default();
        let g = DecoderGenome::parse("nn_forecast:lb=32").unwrap();
        let now = now_ms();
        let window = DataWindow { start_ms: Some(now - 3_600_000), end_ms: Some(now - 60_000), ..Default::default() };
        let loaded = data::load(worker.tsdb.as_ref(), &worker.market, &g, &window, 1.0, 200).await;
        let (ctx, _) = worker.training_latents(&window, loaded.source.window_start_ms, loaded.source.window_end_ms).await;
        let s = loaded.series.iter().max_by_key(|s| s.len()).expect("a series");
        let steps: Vec<(&Series, usize)> = (s.len() / 2..s.len())
            .step_by(7)
            .filter(|t| ctx.latent_as_of(&s.instrument_id, s.ts_ms[*t]).is_some_and(|l| l.vec.len() == dim))
            .take(128)
            .map(|t| (s, t))
            .collect();
        assert!(steps.len() > 16, "{} steps with latents", steps.len());
        let lat: Vec<f32> =
            steps.iter().flat_map(|(s, t)| ctx.latent_as_of(&s.instrument_id, s.ts_ms[*t]).unwrap().vec.clone()).collect();
        let latents = Var::from_tensor(&Tensor::from_vec(lat, (steps.len(), dim), &Device::Cpu).unwrap()).unwrap();
        let batch = set.prepare(&steps, &TextTimeline::new()).unwrap();
        let out = set.aux_loss(latents.as_tensor(), &batch).unwrap();
        let grads = out.total.backward().unwrap();
        let gnorm = grads.get(latents.as_tensor()).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        eprintln!(
            "feedback: {} cards, {} elites loaded (dim {dim}), skipped {:?}, rows {:?}, parts {:?}, total {}, |grad|² {gnorm}",
            cards.len(),
            set.elites.len(),
            set.skipped,
            batch.rows_per_elite(),
            out.parts,
            out.total.to_scalar::<f32>().unwrap()
        );
        assert!(gnorm > 0.0);
    }
}

