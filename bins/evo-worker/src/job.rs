//! `evo.eval.decoder` job evaluation: genome → baseline decoder (neural `nn_*` genomes:
//! [`crate::neural`], train then score out of sample) → sequential point-in-time
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
    sync::Arc,
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
use tokio::sync::broadcast;

use crate::{
    allocator,
    data::{self, DataWindow},
    gpu::{self, Lease, Notice},
    latent::LatentBuffer,
    market::{MarketBuffer, now_ms},
    run::{GRACE, JobRun, guard},
    strategy,
    trials::{self, TrialRow, TrialStore},
    vary, vary_strategy,
};

/// A decoder window ending at least this long ago is historical: its latents come from the
/// recorded `LATENT` stream instead of the live buffer (reproducible trials).
pub const LIVE_EDGE_MS: i64 = 60_000;

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
    /// Trial registry (`validation.trial`, `validation.generation_stats`, holdouts); `None` without Postgres.
    pub trials: Option<TrialStore>,
    /// Shared execution cost model (`go/services/broker-gateway/costs.yaml`).
    pub cost: Arc<fa_backtest::cost::CostModel>,
    /// Recorded `LATENT` traffic for strategy replays (bounded per-window cache, D-059).
    pub recorded: crate::recorded::RecordedStore,
    /// News → instrument matcher for the text genomes (`ninst`), refreshed from Postgres.
    pub news_refs: crate::news::NewsRefs,
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

impl Worker {
    pub(crate) fn failed(&self, job: &Job, error: String, used_ms: u64) -> Outcome {
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

    pub(crate) async fn progress(&self, job_id: &str, fraction: f32, message: impl Into<String>) {
        let p = JobProgress { job_id: job_id.into(), fraction, message: message.into() };
        if let Err(e) = self.bus.publish(subjects::job_progress(job_id), &p).await {
            tracing::debug!(error = %e, "progress publish failed");
        }
    }

    /// Every job kind this worker runs.
    pub const KINDS: [&'static str; 7] = [
        KIND,
        vary::KIND,
        strategy::KIND,
        vary_strategy::KIND,
        crate::holdout::KIND,
        allocator::KIND,
        allocator::VARY_KIND,
    ];

    /// Handles one job under a [`JobRun`] (deadline, budget, preempt / cancel watch, progress):
    /// an interruption always ends as `STATUS_PREEMPTED` / `STATUS_CANCELLED` (see
    /// [`crate::run::guard`]). Never panics on bad input; every path yields a `JobResult`.
    pub async fn handle(&self, job: &Job) -> Outcome {
        let default_budget = match job.kind.as_str() {
            KIND => DEFAULT_BUDGET_MS,
            _ => strategy::DEFAULT_BUDGET_MS,
        };
        let run = JobRun::new(job, default_budget, Some(self.bus.clone()));
        let watcher = run.watch(self.notices.subscribe());
        let out = guard(&run, &self.worker_id, GRACE, self.dispatch(job, &run)).await;
        watcher.abort();
        out
    }

    async fn dispatch(&self, job: &Job, run: &JobRun) -> Outcome {
        if run.past_deadline() {
            return self.failed(job, "job deadline already passed".into(), 0);
        }
        match job.kind.as_str() {
            KIND => self.evaluate(job, run).await,
            vary::KIND => self.vary(job).await,
            strategy::KIND => self.evaluate_strategy(job, run).await,
            vary_strategy::KIND => self.vary_strategy(job).await,
            crate::holdout::KIND => self.evaluate_holdout(job, run).await,
            allocator::KIND => self.evaluate_allocator(job, run).await,
            allocator::VARY_KIND => self.vary_allocator(job).await,
            other => self.failed(
                job,
                format!(
                    "job kind `{other}` not implemented (only {})",
                    Self::KINDS.map(|k| format!("`{k}`")).join(", ")
                ),
                0,
            ),
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

    /// `evo.vary.strategy`: parents → children (see [`crate::vary_strategy`]). CPU only.
    async fn vary_strategy(&self, job: &Job) -> Outcome {
        let started = Instant::now();
        let req: vary::VaryRequest = match serde_json::from_slice(&job.spec) {
            Ok(r) => r,
            Err(e) => return self.failed(job, format!("vary spec: {e}"), 0),
        };
        let out = match vary_strategy::produce(&req) {
            Ok(o) => o,
            Err(e) => return self.failed(job, format!("vary: {e}"), started.elapsed().as_millis() as u64),
        };
        self.progress(&job.job_id, 0.5, format!("{} children from {} parents", out.children.len(), out.parents.len()))
            .await;
        let helix =
            match tokio::time::timeout(Duration::from_secs(20), vary_strategy::persist(self.hub.as_ref(), &req, &out))
                .await
            {
                Ok(v) => v,
                Err(_) => json!({"persisted": false, "reason": "helix timeout"}),
            };
        if helix["persisted"] != json!(true) {
            tracing::warn!(job = %job.job_id, helix = %helix, "strategy lineage not (fully) persisted to Helix");
        }
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let payload = vary_strategy::payload(&req, &out, helix, &self.worker_id, elapsed_ms);
        tracing::info!(job = %job.job_id, parents = out.parents.len(), children = out.children.len(), immigrants = out.immigrants, "strategy variation done");
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

    /// `(generation loop index, rung)` from the eval envelope (0, 0 when absent).
    pub fn parse_position(job: &Job) -> (i32, i32) {
        let raw = std::str::from_utf8(&job.spec).unwrap_or("");
        if let Ok(v) = serde_json::from_str::<Value>(raw) {
            return (
                v.get("generation").and_then(Value::as_i64).unwrap_or(0) as i32,
                v.get("rung").and_then(Value::as_i64).unwrap_or(0) as i32,
            );
        }
        (0, 0)
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
    async fn evaluate(&self, job: &Job, run: &JobRun) -> Outcome {
        let started = Instant::now();
        let budget_ms = run.budget_ms();

        // 1. genome (Job.spec overrides the id string when present) + data window
        let (raw, window) = Self::parse_spec(job);
        let genome = match DecoderGenome::parse(&raw) {
            Ok(g) => g,
            Err(e) => return self.failed(job, format!("genome `{raw}`: {e}"), 0),
        };
        if !job.checkpoint_ref.is_empty() {
            tracing::info!(job = %job.job_id, checkpoint = %job.checkpoint_ref, "resuming preempted job (re-evaluating from the start; evaluation is cheap)");
        }
        if genome.family.is_neural() {
            return self.evaluate_neural(job, run, genome, window).await;
        }
        run.progress(0.0, format!("parsed genome: {}", genome.summary())).await;

        // 2. GPU lease (optional)
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
        // Latent conditioning: recorded `LATENT` traffic over [start − preload, end] for a
        // historical window (reproducible), the live buffer for windows ending at now.
        let (ctx, latent_block) = self.decoder_latents(&window, data_src.window_start_ms, data_src.window_end_ms).await;
        let (latent_inst, latent_points) = ctx.stats();

        // 4. evaluate within budget, preemptible / cancellable; the naive policy pays the venue
        //    cost model (spec 35, costs.yaml) per unit of position change
        run.start_budget();
        let eval_inputs: Vec<Series> = usable;
        let cost_bps: Vec<f64> = eval_inputs
            .iter()
            .map(|s| crate::news::trade_cost_bp(&self.cost, crate::news::venue_of(&venues, &s.instrument_id), s))
            .collect();
        let eval_task = {
            let cost_bps = cost_bps.clone();
            let genome = genome.clone();
            let ctx = ctx.clone();
            let run = run.clone();
            tokio::task::spawn_blocking(move || {
                let decoder = build_with(&genome, ctx);
                let mut reports: Vec<EvalReport> = Vec::new();
                let n = eval_inputs.len();
                for (i, s) in eval_inputs.iter().enumerate() {
                    let progress = run.on_progress();
                    let spec = EvalSpec {
                        should_stop: Some(run.stopper()),
                        on_progress: Some(Box::new(move |done, total| progress(done + total * i, total * n))),
                        progress_every: 16,
                        cost_bp: cost_bps[i],
                        ..EvalSpec::standard()
                    };
                    reports.push(evaluate(decoder.as_ref(), s, &spec));
                    if run.should_stop() {
                        break;
                    }
                }
                reports
            })
        };
        let reports = run.drive(eval_task, &genome.summary()).await;
        release.await;
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let reports = match reports {
            Ok(r) => r,
            Err(e) => return self.failed(job, format!("evaluation task failed: {e}"), elapsed_ms),
        };

        // 5. results
        if let Some(out) = run.interrupted(&self.worker_id) {
            return out;
        }
        let Some(pooled) = pool(&reports) else {
            let status = if run.over_budget() { Status::BudgetExceeded } else { Status::Failed };
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
        // Trial registry (spec 41 §6.5): the naive sign policy's per-step P&L is the trial's return series.
        let (gen_index, rung) = Self::parse_position(job);
        let generation_stats = trials::register(
            self.trials.as_ref(),
            &TrialRow {
                generation: if job.parent_job_id.is_empty() { "adhoc".into() } else { job.parent_job_id.clone() },
                gen_index,
                rung,
                job_id: job.job_id.clone(),
                candidate: genome.to_string(),
                kind: KIND.into(),
                window_start_ms: data_src.window_start_ms,
                window_end_ms: data_src.window_end_ms,
                sharpe: pooled.trade_sharpe,
                n_obs: pooled.n_eval,
                objectives: Value::Object(objectives.clone()),
                returns: pooled.pnl.clone(),
                holdout_id: None,
            },
        )
        .await;
        let instruments: Vec<Value> = reports
            .iter()
            .map(|r| {
                let venue = crate::news::venue_of(&venues, &r.instrument_id);
                let cost_bp = eval_inputs_cost(&cost_bps, &reports, &r.instrument_id);
                json!({
                    "instrument_id": r.instrument_id, "venue": venue, "n_steps": r.n_series, "n_eval": r.n_eval,
                    "stop": r.stop, "crps_skill": r.crps_skill, "coverage": r.coverage, "last_ts_ms": r.last_ts_ms,
                    "cost_bp": cost_bp,
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
                     "latents": {"instruments": latent_inst, "points": latent_points, "source": latent_block["source"], "recorded": latent_block},
                     "instruments": instruments},
            "generation_stats": generation_stats,
            "budget": {"budget_ms": budget_ms, "elapsed_ms": elapsed_ms, "truncated": truncated},
            "trade_cost": {"model": self.cost.id, "notional_eur": crate::news::TRADE_NOTIONAL_EUR,
                           "basis": "costs.yaml taker fill per unit of position change (half spread + impact + commission)"},
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

    /// Decoder conditioning for an evaluation window: `recorded` (the `LATENT` stream over
    /// `[start − preload, end]`, through the incremental cache) when the job's window ends
    /// [`LIVE_EDGE_MS`] or more in the past, else `live` (the `latent.state.>` buffer). A failed
    /// recorded read falls back to the live buffer and says so.
    pub(crate) async fn decoder_latents(
        &self,
        window: &DataWindow,
        start_ms: i64,
        end_ms: i64,
    ) -> (Arc<fa_decoders::DecodeCtx>, Value) {
        if window.is_historical(now_ms(), LIVE_EDGE_MS) && self.recorded.enabled {
            let l = self.recorded.load_window(&self.bus.jetstream(), start_ms - self.recorded.preload_ms, end_ms).await;
            if let Some(rec) = &l.latents {
                let ctx = Arc::new(crate::recorded::decode_ctx(rec));
                let mut j = self.recorded.json(&l);
                j["source"] = json!("recorded");
                return (ctx, j);
            }
            let ctx = self.latents.ctx();
            return (
                ctx,
                json!({"source": "live", "reason": format!("recorded read failed: {}", l.error.unwrap_or_default())}),
            );
        }
        (self.latents.ctx(), json!({"source": "live"}))
    }

    /// Writes the report JSON to Garage. Returns whether it was persisted; failures only warn,
    /// the job result is published either way.
    pub(crate) async fn persist_report(&self, key: &fa_blob::ObjectKey, payload: &Value) -> bool {
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
            // `h:<bars>` = forecast horizon in bars of the genome's timescale: strategy-runtime's
            // `fc_quantile(q, k)` / `fc_dist(k)` pick the output whose horizon is closest to `k`.
            output_refs: horizon_refs(&r.instrument_id, &job.job_id, genome.horizon),
            quantiles: f.quantiles.clone(),
            analogs,
        }
    }
}

/// `output_refs` of a published `DecoderOutput`: `inst:<id>`, `job:<id>`, `h:<horizon bars>`.
/// Cost (bp) the report's series was charged: reports are in input order, so position `i`.
pub(crate) fn eval_inputs_cost(cost_bps: &[f64], reports: &[EvalReport], instrument: &str) -> Option<f64> {
    reports.iter().position(|r| r.instrument_id == instrument).and_then(|i| cost_bps.get(i).copied())
}

pub fn horizon_refs(instrument_id: &str, job_id: &str, horizon: usize) -> Vec<String> {
    vec![format!("inst:{instrument_id}"), format!("job:{job_id}"), format!("h:{horizon}")]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_output_refs_carry_the_horizon() {
        let g = DecoderGenome::parse("forecast:h=7;lb=16").unwrap();
        assert_eq!(g.horizon, 7);
        assert_eq!(horizon_refs("0190ab", "j1", g.horizon), ["inst:0190ab", "job:j1", "h:7"]);
    }

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
        assert_eq!(Worker::parse_position(&job(r#"{"genome":"g1","generation":2,"rung":1}"#)), (2, 1));
        assert_eq!(Worker::parse_position(&job("retrieval:k=4")), (0, 0));
    }

    /// `FA_NATS_TEST=1 cargo test -p evo-worker -- --nocapture cancel_while_running`: a running
    /// decoder evaluation (SYNTHETIC buffered trades, a slow retrieval genome) receives
    /// `ctl.evo.cancel {job_id}` on the bus and ends with `STATUS_CANCELLED` well before it
    /// would have finished.
    #[tokio::test]
    async fn cancel_while_running() {
        if std::env::var("FA_NATS_TEST").is_err() {
            eprintln!("skipping (set FA_NATS_TEST=1)");
            return;
        }
        use fa_proto::{Message, v1::CtlRequest};
        let worker = crate::holdout::tests::offline_worker().await;
        tokio::spawn(gpu::run_notices(worker.bus.clone(), worker.notices.clone()));
        tokio::time::sleep(Duration::from_millis(300)).await; // subscriptions in place
        let mut p = 100.0;
        for i in 0..40_000i64 {
            p *= 1.0 + 0.001 * ((i * 7919 % 13) as f64 - 6.0) / 6.0;
            worker.market.push("01900000-0000-7000-8000-00000000c0de", "sim", i * 1000, p);
        }
        let job_id = format!("cancel-{}", ulid::Ulid::generate());
        let job = Job {
            job_id: job_id.clone(),
            kind: KIND.into(),
            budget_gpu_ms: 600_000,
            spec: b"retrieval:h=3;lb=64;k=50;gap=3".to_vec(),
            ..Default::default()
        };
        let canceller = {
            let bus = worker.bus.clone();
            let job_id = job_id.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(1500)).await;
                let req = CtlRequest { verb: "cancel".into(), args: [("job_id".to_owned(), job_id)].into() };
                bus.client().publish(subjects::ctl("evo", "cancel"), req.encode_to_vec().into()).await.unwrap();
                bus.client().flush().await.unwrap();
            })
        };
        let started = Instant::now();
        let out = worker.handle(&job).await;
        canceller.await.unwrap();
        eprintln!(
            "cancel_while_running: status {:?} error {:?} after {:?}",
            Status::try_from(out.result.status),
            out.result.error,
            started.elapsed()
        );
        assert_eq!(out.result.status, Status::Cancelled as i32, "{:?}", out.result);
        assert!(started.elapsed() < Duration::from_secs(15), "{:?}", started.elapsed());
        assert!(out.result.error.contains("ctl.evo.cancel"));
    }

    /// `FA_NATS_TEST=1` against the compose stack (NATS + VictoriaMetrics at 127.0.0.1:8428):
    /// a decoder evaluation over a historical window takes its latents from the recorded
    /// `LATENT` stream (`data.latents.source = recorded`), a now-relative one from the live
    /// buffer (`live`).
    #[tokio::test]
    async fn decoder_latents_recorded_for_historical_windows() {
        if std::env::var("FA_NATS_TEST").is_err() {
            eprintln!("skipping (set FA_NATS_TEST=1)");
            return;
        }
        let mut worker = crate::holdout::tests::offline_worker().await;
        worker.tsdb = Some(Tsdb::new(std::env::var("FA_VM_URL").unwrap_or_else(|_| "http://127.0.0.1:8428".into())));
        worker.recorded = crate::recorded::RecordedStore::default();
        worker.recorded.preload_ms = 600_000;
        let now = now_ms();
        let job = |spec: String| Job {
            job_id: format!("lat-{}", ulid::Ulid::generate()),
            kind: KIND.into(),
            budget_gpu_ms: 30_000,
            spec: spec.into_bytes(),
            ..Default::default()
        };
        let hist = format!(
            r#"{{"genome":"retrieval:h=3;lb=8;k=10","data_window":{{"start_ms":{},"end_ms":{}}}}}"#,
            now - 25 * 60_000,
            now - 5 * 60_000
        );
        let out = worker.handle(&job(hist)).await;
        assert_eq!(out.result.status, Status::Ok as i32, "{}", out.result.error);
        let v: Value = serde_json::from_slice(&out.result.result).unwrap();
        let l = &v["data"]["latents"];
        eprintln!(
            "historical: source {} points {} plan {} read {}",
            l["source"], l["points"], l["recorded"]["plan"], l["recorded"]["messages_read"]
        );
        assert_eq!(l["source"], "recorded");
        assert!(l["points"].as_u64().unwrap() > 0, "{l}");
        let live = out_of(
            &worker.handle(&job(r#"{"genome":"retrieval:h=3;lb=8;k=10","data_window":{"hours":0.3}}"#.into())).await,
        );
        eprintln!("now-relative: source {}", live["data"]["latents"]["source"]);
        assert_eq!(live["data"]["latents"]["source"], "live");
    }

    fn out_of(o: &Outcome) -> Value {
        assert_eq!(o.result.status, Status::Ok as i32, "{}", o.result.error);
        serde_json::from_slice(&o.result.result).unwrap()
    }
}
