//! `evo.eval.decoder` job evaluation: genome → baseline decoder → sequential point-in-time
//! scoring on the buffered market window → `JobResult` + `DecoderOutput`.
//!
//! Result payload (JSON, provisional protocol `evo.eval.decoder/v0`, D-025 compatible):
//! ```json
//! {
//!   "protocol": "evo.eval.decoder/v0",
//!   "score": 0.12,                       // = objectives.crps_skill, read by the orchestrator
//!   "objectives": {"crps": .., "crps_skill": .., "coverage": .., "coverage_err": ..,
//!                  "hit_rate": .., "trade_sharpe": .., "trade_psr": .., "compute_cost": .., "n_eval": ..},
//!   "directions": {"crps": "min", ...},  // for a future NSGA step
//!   "genome": {...parsed genome...},
//!   "data":  {"source": "md.trade.> in-memory buffer (stopgap)", "instruments": [...]},
//!   "budget": {"budget_ms": .., "elapsed_ms": .., "truncated": false},
//!   "stop": "complete"
//! }
//! ```

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use fa_backtest::eval::{EvalReport, EvalSpec, Stop, evaluate, pool};
use fa_bus::{Bus, subjects};
use fa_decoders::{
    baseline::{Series, build},
    genome::{DecoderGenome, stable_hash},
};
use fa_proto::{
    Timestamp,
    v1::{self, DecoderOutput, Job, JobProgress, JobResult, job_result::Status},
};
use serde_json::{Value, json};
use tokio::sync::{broadcast, watch};

use crate::{
    gpu::{self, Lease, Notice},
    market::{MarketBuffer, now_ms},
};

pub const KIND: &str = "evo.eval.decoder";
pub const PROTOCOL: &str = "evo.eval.decoder/v0";
const DEFAULT_BUDGET_MS: u64 = 60_000;
/// Decoder outputs published per job (one per instrument, most data first).
const MAX_OUTPUTS: usize = 4;

pub struct Worker {
    pub bus: Bus,
    pub worker_id: String,
    pub market: MarketBuffer,
    pub notices: broadcast::Sender<Notice>,
    pub gpu_lease_mb: u32,
}

/// What `handle` produces; the caller publishes the result and acks.
pub struct Outcome {
    pub result: JobResult,
    pub outputs: Vec<DecoderOutput>,
}

pub fn ts_ms(ms: i64) -> Timestamp {
    Timestamp { seconds: ms.div_euclid(1000), nanos: (ms.rem_euclid(1000) * 1_000_000) as i32 }
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

    /// Evaluates one job. Never panics on bad input; every path yields a `JobResult`.
    pub async fn handle(&self, job: &Job) -> Outcome {
        let started = Instant::now();
        if job.kind != KIND {
            return self.failed(job, format!("job kind `{}` not implemented (only `{KIND}`)", job.kind), 0);
        }
        let budget_ms = if job.budget_gpu_ms == 0 { DEFAULT_BUDGET_MS } else { job.budget_gpu_ms };
        let budget = Duration::from_millis(budget_ms);
        let deadline = deadline_of(job, budget);
        if deadline <= Instant::now() {
            return self.failed(job, "job deadline already passed".into(), 0);
        }

        // 1. genome (Job.spec JSON overrides the id string when present)
        let raw = match std::str::from_utf8(&job.spec) {
            Ok(s) if !s.trim().is_empty() => s.to_owned(),
            _ => job.genome_id.clone(),
        };
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

        // 3. data window
        let step_ms = i64::try_from(genome.timescale.duration().as_millis()).unwrap_or(1000);
        let snapshot = self.market.snapshot(genome.instrument.as_deref());
        let series: Vec<Series> = snapshot
            .iter()
            .map(|t| Series::resample(&t.instrument_id, &t.trades, step_ms))
            .filter(|s| !s.is_empty())
            .collect();
        let decoder = build(&genome);
        let need = decoder.warmup() + genome.horizon + 1;
        let usable: Vec<&Series> = series.iter().filter(|s| s.len() >= need).collect();
        if usable.is_empty() {
            let (n_inst, n_trades) = self.market.stats();
            let longest = series.iter().map(Series::len).max().unwrap_or(0);
            release.await;
            return self.failed(
                job,
                format!(
                    "insufficient market data: need ≥ {need} resampled {} steps for `{}`, longest series has {longest} (buffer: {n_inst} instruments, {n_trades} trades); the worker only buffers live md.trade.> since it started",
                    genome.timescale,
                    genome.summary()
                ),
                started.elapsed().as_millis() as u64,
            );
        }

        // 4. evaluate within budget, preemptible
        let cancel = Arc::new(AtomicBool::new(false));
        let preempted = Arc::new(AtomicBool::new(false));
        let (ptx, mut prx) = watch::channel((0usize, 0usize, String::new()));
        let eval_started = Instant::now();
        let eval_inputs: Vec<Series> = usable.iter().map(|s| (*s).clone()).collect();
        let eval_task = {
            let cancel = cancel.clone();
            let genome = genome.clone();
            tokio::task::spawn_blocking(move || {
                let decoder = build(&genome);
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
                let venue = snapshot
                    .iter()
                    .find(|t| t.instrument_id == r.instrument_id)
                    .map(|t| t.venue.as_str())
                    .unwrap_or("");
                json!({
                    "instrument_id": r.instrument_id, "venue": venue, "n_steps": r.n_series, "n_eval": r.n_eval,
                    "stop": r.stop, "crps_skill": r.crps_skill, "coverage": r.coverage, "last_ts_ms": r.last_ts_ms,
                })
            })
            .collect();
        let payload = json!({
            "protocol": PROTOCOL,
            "score": pooled.score(),
            "objectives": objectives,
            "directions": {"crps": "min", "crps_skill": "max", "coverage_err": "min", "hit_rate": "max",
                           "trade_sharpe": "max", "trade_psr": "max", "compute_cost": "min"},
            "genome": genome,
            "data": {"source": "md.trade.> in-memory buffer (stopgap; not VictoriaMetrics)", "timescale": genome.timescale.to_string(),
                     "instruments": instruments},
            "budget": {"budget_ms": budget_ms, "elapsed_ms": elapsed_ms, "truncated": truncated},
            "stop": pooled.stop,
            // Where the report would be persisted once fa-blob has an S3 client and Garage
            // credentials exist (config/secrets.toml); nothing is written today.
            "report_ref": {"planned": format!("{}/{}", report_key.bucket, report_key.key), "persisted": false},
            "worker_id": self.worker_id,
        });
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
                ..Default::default()
            },
            outputs,
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
            // No Garage artifacts yet (fa-blob has no S3 client); these refs locate the
            // forecast target instead.
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
}
