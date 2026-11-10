//! `evo.eval.decoder` job evaluation: genome → baseline decoder → sequential point-in-time
//! scoring on the evaluation window (VictoriaMetrics via `fa-tsdb`, buffer fallback; see
//! [`crate::data`]) → `JobResult` + `DecoderOutput`. `evo.vary.decoder` jobs are delegated to
//! [`crate::vary`].
//!
//! `Job.spec` for eval jobs is either the genome itself (kv / JSON, D-044) or an envelope
//! `{"genome": "<kv or bare id>", "data_window": {"hours": 6}}`; without a spec the genome is
//! `Job.genome_id`.
//!
//! Result payload (JSON, provisional protocol `evo.eval.decoder/v0`, D-025 compatible):
//! ```json
//! {
//!   "protocol": "evo.eval.decoder/v0",
//!   "score": 0.12,                       // = objectives.crps_skill, read by the orchestrator
//!   "objectives": {"crps": .., "crps_skill": .., "coverage": .., "coverage_err": ..,
//!                  "hit_rate": .., "trade_sharpe": .., "trade_psr": .., "compute_cost": .., "n_eval": ..},
//!   "directions": {"crps": "min", ...},  // also set as JobResult.objectives / .directions (proto maps)
//!   "genome": {...parsed genome...},
//!   "data":  {"source": "victoriametrics" | "md.trade.> in-memory buffer (fallback: ..)", "window_start_ms": ..,
//!             "latents": {"instruments": .., "points": ..}, "instruments": [...]},
//!   "budget": {"budget_ms": .., "elapsed_ms": .., "truncated": false},
//!   "stop": "complete",
//!   "report_ref": {"planned": "fa-artifacts/evals/<generation>/<job>.json", "persisted": true}
//! }
//! ```
//! The same JSON is written to `report_ref.planned` in Garage when the worker has an
//! `S3BlobStore` (credentials in `config/secrets.toml`); a failed write only logs a warning
//! and leaves `persisted: false`.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use fa_backtest::eval::{EvalReport, EvalSpec, Stop, evaluate, pool};
use fa_blob::{BlobStore, S3BlobStore};
use fa_bus::{Bus, subjects};
use fa_decoders::{
    baseline::{Series, build, build_with},
    genome::{DecoderGenome, stable_hash},
};
use fa_hub::HubClient;
use fa_proto::{
    Timestamp,
    v1::{self, DecoderOutput, Job, JobProgress, JobResult, job_result::Status},
};
use fa_tsdb::Tsdb;
use serde_json::{Value, json};
use tokio::sync::{broadcast, watch};

use crate::{
    data::{self, DataWindow},
    gpu::{self, Lease, Notice},
    latent::LatentBuffer,
    market::{MarketBuffer, now_ms},
    vary,
};

pub const KIND: &str = "evo.eval.decoder";
pub const PROTOCOL: &str = "evo.eval.decoder/v0";
const DEFAULT_BUDGET_MS: u64 = 60_000;
/// Objective directions of the `evo.eval.decoder/v0` payload (spec 34 §6.3).
pub const DIRECTIONS: [(&str, &str); 7] = [
    ("crps", "min"),
    ("crps_skill", "max"),
    ("coverage_err", "min"),
    ("hit_rate", "max"),
    ("trade_sharpe", "max"),
    ("trade_psr", "max"),
    ("compute_cost", "min"),
];
/// Decoder outputs published per job (one per instrument, most data first).
const MAX_OUTPUTS: usize = 4;

pub struct Worker {
    pub bus: Bus,
    pub worker_id: String,
    pub market: MarketBuffer,
    pub notices: broadcast::Sender<Notice>,
    pub gpu_lease_mb: u32,
    /// VictoriaMetrics reader (primary evaluation data); `None` = buffer only.
    pub tsdb: Option<Tsdb>,
    /// Default evaluation window when the job spec has none.
    pub data_window_hours: f64,
    /// Latest `latent.state.>` per instrument for decoder conditioning.
    pub latents: LatentBuffer,
    /// Helix client for genome lineage (`evo.vary.decoder`); `None` = not persisted.
    pub hub: Option<HubClient>,
    /// Garage store for eval reports; `None` without credentials.
    pub blob: Option<S3BlobStore>,
}

/// What `handle` produces; the caller publishes the result and acks.
pub struct Outcome {
    pub result: JobResult,
    pub outputs: Vec<DecoderOutput>,
}

pub fn ts_ms(ms: i64) -> Timestamp {
    Timestamp { seconds: ms.div_euclid(1000), nanos: (ms.rem_euclid(1000) * 1_000_000) as i32 }
}

fn directions_json() -> Value {
    Value::Object(DIRECTIONS.iter().map(|(k, d)| ((*k).to_owned(), json!(d))).collect())
}

fn deadline_of(job: &Job, budget: Duration) -> Instant {
    let fallback = Instant::now() + budget * 10 + Duration::from_secs(60);
    match &job.deadline {
        Some(t) => {
            let ms = t.seconds * 1000 + i64::from(t.nanos) / 1_000_000;
            let left = ms.saturating_sub(now_ms());
            if left <= 0 { Instant::now() } else { Instant::now() + Duration::from_millis(left as u64) }
        }
        None => fallback,
    }
}

impl Worker {
    fn failed(&self, job: &Job, error: String, used_ms: u64) -> Outcome {
        Outcome {
            result: JobResult {
                job_id: job.job_id.clone(),
                status: Status::Failed as i32,
                error,
                worker_id: self.worker_id.clone(),
                used_gpu_ms: used_ms,
                ..Default::default()
            },
            outputs: Vec::new(),
        }
    }

    async fn progress(&self, job_id: &str, fraction: f32, message: impl Into<String>) {
        let p = JobProgress { job_id: job_id.into(), fraction, message: message.into() };
        if let Err(e) = self.bus.publish(subjects::job_progress(job_id), &p).await {
            tracing::debug!(error = %e, "progress publish failed");
        }
    }

    /// Handles one job. Never panics on bad input; every path yields a `JobResult`.
    pub async fn handle(&self, job: &Job) -> Outcome {
        match job.kind.as_str() {
            KIND => self.evaluate(job).await,
            vary::KIND => self.vary(job).await,
            other => {
                self.failed(job, format!("job kind `{other}` not implemented (only `{KIND}`, `{}`)", vary::KIND), 0)
            }
        }
    }

    /// `evo.vary.decoder`: parents → children (see [`crate::vary`]). CPU only, no lease.
    async fn vary(&self, job: &Job) -> Outcome {
        let started = Instant::now();
        let req: vary::VaryRequest = match serde_json::from_slice(&job.spec) {
            Ok(r) => r,
            Err(e) => return self.failed(job, format!("vary spec: {e}"), 0),
        };
        let out = match vary::produce(&req) {
            Ok(o) => o,
            Err(e) => return self.failed(job, format!("vary: {e}"), started.elapsed().as_millis() as u64),
        };
        self.progress(&job.job_id, 0.5, format!("{} children from {} parents", out.children.len(), out.parents.len()))
            .await;
        let helix =
            match tokio::time::timeout(Duration::from_secs(20), vary::persist(self.hub.as_ref(), &req, &out)).await {
                Ok(v) => v,
                Err(_) => json!({"persisted": false, "reason": "helix timeout"}),
            };
        if helix["persisted"] != json!(true) {
            tracing::warn!(job = %job.job_id, helix = %helix, "vary lineage not (fully) persisted to Helix");
        }
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let payload = vary::payload(&req, &out, helix, &self.worker_id, elapsed_ms);
        tracing::info!(job = %job.job_id, parents = out.parents.len(), children = out.children.len(), "variation done");
        Outcome {
            result: JobResult {
                job_id: job.job_id.clone(),
                status: Status::Ok as i32,
                result: serde_json::to_vec(&payload).unwrap_or_default(),
                used_gpu_ms: elapsed_ms,
                worker_id: self.worker_id.clone(),
                ..Default::default()
            },
            outputs: Vec::new(),
        }
    }

    /// Splits an eval `Job.spec` into the genome string and the data window: an envelope
    /// `{"genome", "data_window"}`, a JSON genome (`family` key), a kv string, or nothing.
    pub fn parse_spec(job: &Job) -> (String, DataWindow) {
        let raw = std::str::from_utf8(&job.spec).map(str::trim).unwrap_or("");
        if raw.is_empty() {
            return (job.genome_id.clone(), DataWindow::default());
        }
        if raw.starts_with('{')
            && let Ok(v) = serde_json::from_str::<Value>(raw)
            && v.get("family").is_none()
        {
            let genome = v.get("genome").and_then(Value::as_str).filter(|g| !g.trim().is_empty()).map(str::to_owned);
            let window: DataWindow =
                v.get("data_window").cloned().and_then(|w| serde_json::from_value(w).ok()).unwrap_or_default();
            return (genome.unwrap_or_else(|| job.genome_id.clone()), window);
        }
        (raw.to_owned(), DataWindow::default())
    }

    /// `evo.eval.decoder`.
    async fn evaluate(&self, job: &Job) -> Outcome {
        let started = Instant::now();
        let budget_ms = if job.budget_gpu_ms == 0 { DEFAULT_BUDGET_MS } else { job.budget_gpu_ms };
        let budget = Duration::from_millis(budget_ms);
        let deadline = deadline_of(job, budget);
        if deadline <= Instant::now() {
            return self.failed(job, "job deadline already passed".into(), 0);
        }

        // 1. genome (Job.spec overrides the id string when present) + data window
        let (raw, window) = Self::parse_spec(job);
        let genome = match DecoderGenome::parse(&raw) {
            Ok(g) => g,
            Err(e) => return self.failed(job, format!("genome `{raw}`: {e}"), 0),
        };
        if !job.checkpoint_ref.is_empty() {
            tracing::info!(job = %job.job_id, checkpoint = %job.checkpoint_ref, "resuming preempted job (re-evaluating from the start; evaluation is cheap)");
        }
        self.progress(&job.job_id, 0.0, format!("parsed genome: {}", genome.summary())).await;

        // 2. GPU lease (optional)
        let mut notices = self.notices.subscribe();
        let lease = if self.gpu_lease_mb > 0 {
            match gpu::acquire(&self.bus, &mut notices, &job.job_id, self.gpu_lease_mb, deadline).await {
                Ok(l) => l,
                Err(e) => return self.failed(job, e, started.elapsed().as_millis() as u64),
            }
        } else {
            Lease::Unavailable
        };
        let release = async {
            if lease == Lease::Granted {
                gpu::release(&self.bus, &job.job_id).await;
            }
        };

        // 3. data window: VictoriaMetrics (fa-tsdb) first, in-memory buffer as fallback
        let need = build(&genome).warmup() + genome.horizon + 1;
        let loaded = data::load(self.tsdb.as_ref(), &self.market, &genome, &window, self.data_window_hours, need).await;
        let data_src = loaded.source;
        let venues = loaded.venues;
        let usable: Vec<Series> = loaded.series;
        if usable.is_empty() {
            release.await;
            return self.failed(
                job,
                format!(
                    "insufficient market data: need ≥ {need} resampled {} steps for `{}`, longest series has {} (source: {}; vm: {}; buffer: {} instruments, {} trades)",
                    genome.timescale,
                    genome.summary(),
                    loaded.longest,
                    data_src.source,
                    data_src.vm_error.as_deref().unwrap_or("ok"),
                    data_src.buffer_instruments,
                    data_src.buffer_trades,
                ),
                started.elapsed().as_millis() as u64,
            );
        }
        let ctx = self.latents.ctx();
        let (latent_inst, latent_points) = ctx.stats();

        // 4. evaluate within budget, preemptible
        let cancel = Arc::new(AtomicBool::new(false));
        let preempted = Arc::new(AtomicBool::new(false));
        let (ptx, mut prx) = watch::channel((0usize, 0usize, String::new()));
        let eval_started = Instant::now();
        let eval_inputs: Vec<Series> = usable;
        let eval_task = {
            let cancel = cancel.clone();
            let genome = genome.clone();
            let ctx = ctx.clone();
            tokio::task::spawn_blocking(move || {
                let decoder = build_with(&genome, ctx);
                let mut reports: Vec<EvalReport> = Vec::new();
                let n = eval_inputs.len();
                for (i, s) in eval_inputs.iter().enumerate() {
                    let inst = s.instrument_id.clone();
                    let ptx = ptx.clone();
                    let stop_flag = cancel.clone();
                    let spec = EvalSpec {
                        should_stop: Some(Box::new(move |_| {
                            stop_flag.load(Ordering::Relaxed) || eval_started.elapsed() > budget
                        })),
                        on_progress: Some(Box::new(move |done, total| {
                            let _ = ptx.send((done + total * i, total * n, format!("{inst}: step {done}/{total}")));
                        })),
                        progress_every: 16,
                        ..EvalSpec::standard()
                    };
                    reports.push(evaluate(decoder.as_ref(), s, &spec));
                    if cancel.load(Ordering::Relaxed) || eval_started.elapsed() > budget {
                        break;
                    }
                }
                reports
            })
        };

        let mut tick = tokio::time::interval(Duration::from_millis(500));
        tick.tick().await;
        let mut eval_task = eval_task;
        let reports = loop {
            tokio::select! {
                r = &mut eval_task => break r,
                _ = tick.tick() => {
                    let (done, total, msg) = prx.borrow_and_update().clone();
                    let frac = if total > 0 { done as f32 / total as f32 } else { 0.0 };
                    let budget_frac = eval_started.elapsed().as_secs_f32() / budget.as_secs_f32();
                    self.progress(&job.job_id, frac.max(budget_frac.min(0.99)), msg).await;
                }
                n = notices.recv() => {
                    if let Ok(Notice::Preempt(id)) = n && id == job.job_id {
                        tracing::warn!(job = %job.job_id, "preempted by the gpu arbiter");
                        preempted.store(true, Ordering::Relaxed);
                        cancel.store(true, Ordering::Relaxed);
                    }
                }
            }
        };
        release.await;
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let reports = match reports {
            Ok(r) => r,
            Err(e) => return self.failed(job, format!("evaluation task failed: {e}"), elapsed_ms),
        };

        // 5. results
        let (done, _, _) = prx.borrow().clone();
        if preempted.load(Ordering::Relaxed) {
            return Outcome {
                result: JobResult {
                    job_id: job.job_id.clone(),
                    status: Status::Preempted as i32,
                    checkpoint_ref: format!("{PROTOCOL}:step={done}"),
                    used_gpu_ms: elapsed_ms,
                    worker_id: self.worker_id.clone(),
                    ..Default::default()
                },
                outputs: Vec::new(),
            };
        }
        let Some(pooled) = pool(&reports) else {
            let status = if eval_started.elapsed() > budget { Status::BudgetExceeded } else { Status::Failed };
            return Outcome {
                result: JobResult {
                    job_id: job.job_id.clone(),
                    status: status as i32,
                    error: format!("no step evaluated within the {budget_ms} ms budget"),
                    used_gpu_ms: elapsed_ms,
                    worker_id: self.worker_id.clone(),
                    ..Default::default()
                },
                outputs: Vec::new(),
            };
        };
        let truncated = reports.iter().any(|r| r.stop == Stop::Interrupted);
        let report_key = fa_blob::eval_report(&job.parent_job_id, &job.job_id);
        let objectives: serde_json::Map<String, Value> =
            pooled.objectives().into_iter().map(|(k, v)| (k.to_owned(), json!(v))).collect();
        let instruments: Vec<Value> = reports
            .iter()
            .map(|r| {
                let venue = venues.iter().find(|(i, _)| *i == r.instrument_id).map(|(_, v)| v.as_str()).unwrap_or("");
                json!({
                    "instrument_id": r.instrument_id, "venue": venue, "n_steps": r.n_series, "n_eval": r.n_eval,
                    "stop": r.stop, "crps_skill": r.crps_skill, "coverage": r.coverage, "last_ts_ms": r.last_ts_ms,
                })
            })
            .collect();
        let mut payload = json!({
            "protocol": PROTOCOL,
            "score": pooled.score(),
            "objectives": objectives,
            "directions": directions_json(),
            "genome": genome,
            "data": {"source": data_src.source, "timescale": data_src.timescale, "step_ms": data_src.step_ms,
                     "window_start_ms": data_src.window_start_ms, "window_end_ms": data_src.window_end_ms,
                     "vm_url": data_src.vm_url, "vm_series": data_src.vm_series, "vm_error": data_src.vm_error,
                     "latents": {"instruments": latent_inst, "points": latent_points},
                     "instruments": instruments},
            "budget": {"budget_ms": budget_ms, "elapsed_ms": elapsed_ms, "truncated": truncated},
            "stop": pooled.stop,
            // Written below when a Garage store is configured; the stored document says
            // `persisted: true` about itself.
            "report_ref": {"planned": format!("{}/{}", report_key.bucket, report_key.key), "persisted": self.blob.is_some()},
            "worker_id": self.worker_id,
        });
        if !self.persist_report(&report_key, &payload).await {
            payload["report_ref"]["persisted"] = json!(false);
        }
        let outputs = reports
            .iter()
            .filter(|r| r.last_forecast.is_some())
            .take(MAX_OUTPUTS)
            .map(|r| self.decoder_output(job, &genome, r))
            .collect();
        Outcome {
            result: JobResult {
                job_id: job.job_id.clone(),
                status: Status::Ok as i32,
                result: serde_json::to_vec(&payload).unwrap_or_default(),
                used_gpu_ms: elapsed_ms,
                worker_id: self.worker_id.clone(),
                // typed mirror of the JSON for the orchestrator's NSGA-II ranking
                objectives: pooled.objectives().into_iter().map(|(k, v)| (k.to_owned(), v)).collect(),
                directions: DIRECTIONS.iter().map(|(k, d)| ((*k).to_owned(), (*d).to_owned())).collect(),
                ..Default::default()
            },
            outputs,
        }
    }

    /// Writes the report JSON to Garage. Returns whether it was persisted; failures only warn,
    /// the job result is published either way.
    async fn persist_report(&self, key: &fa_blob::ObjectKey, payload: &Value) -> bool {
        let Some(blob) = &self.blob else { return false };
        let bytes = match serde_json::to_vec_pretty(payload) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(error = %e, "eval report not serialisable; not persisted");
                return false;
            }
        };
        match blob.put(key, bytes, fa_blob::JSON).await {
            Ok(()) => {
                tracing::info!(bucket = key.bucket, key = %key.key, "eval report persisted");
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, bucket = key.bucket, key = %key.key, "eval report persist failed");
                false
            }
        }
    }

    fn decoder_output(&self, job: &Job, genome: &DecoderGenome, r: &EvalReport) -> DecoderOutput {
        let f = r.last_forecast.as_ref().expect("filtered");
        let analogs = f
            .analogs
            .iter()
            .map(|a| v1::Analog {
                window_id: a.window_id.clone(),
                window_end: Some(ts_ms(r.last_ts_ms - (a.time_distance_days * 86_400_000.0) as i64)),
                similarity: a.similarity,
                time_distance_days: a.time_distance_days,
            })
            .collect();
        DecoderOutput {
            decoder_id: format!("baseline-{}-{:08x}", genome.family.as_str(), stable_hash(&genome.id) as u32),
            genome_id: job.genome_id.clone(),
            family: genome.family.output_family().as_str().into(),
            snapshot_id: job.snapshot_id.clone(),
            timescale: genome.timescale.to_string(),
            as_of: Some(ts_ms(r.last_ts_ms)),
            // Decoder outputs are not stored as Garage artifacts yet; these refs locate the
            // forecast target (the eval report lives at the result's `report_ref`).
            output_refs: vec![format!("inst:{}", r.instrument_id), format!("job:{}", job.job_id)],
            quantiles: f.quantiles.clone(),
            analogs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_conversion() {
        let t = ts_ms(1_700_000_000_123);
        assert_eq!((t.seconds, t.nanos), (1_700_000_000, 123_000_000));
    }

    #[test]
    fn spec_parsing_accepts_envelope_genome_and_kv() {
        let job = |spec: &str| Job { genome_id: "g1".into(), spec: spec.as_bytes().to_vec(), ..Default::default() };
        assert_eq!(Worker::parse_spec(&job("")), ("g1".to_owned(), DataWindow::default()));
        let (g, w) = Worker::parse_spec(&job(r#"{"genome":"forecast:h=3;lb=8","data_window":{"hours":2}}"#));
        assert_eq!((g.as_str(), w.hours), ("forecast:h=3;lb=8", Some(2.0)));
        let (g, w) = Worker::parse_spec(&job(r#"{"data_window":{"start_ms":5,"end_ms":9}}"#));
        assert_eq!((g.as_str(), w.start_ms, w.end_ms), ("g1", Some(5), Some(9)));
        let (g, _) = Worker::parse_spec(&job(r#"{"id":"j","family":"retrieval","params":{"k":3}}"#));
        assert!(DecoderGenome::parse(&g).unwrap().encoding == "json");
        let (g, _) = Worker::parse_spec(&job("retrieval:k=4"));
        assert_eq!(g, "retrieval:k=4");
        assert_eq!(directions_json()["crps_skill"], "max");
    }
}
