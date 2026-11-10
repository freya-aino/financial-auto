//! `evo.eval.strategy` (spec 35 §6.1–6.2, spec 41 §4): replay one GP strategy genome through
//! `fa_backtest::replay` over VictoriaMetrics bars at the strategy's timescale, net of the
//! shared cost model, with the **recorded latent graph and decoder outputs** of the window
//! ([`crate::recorded`], D-053 feeds replayed point-in-time), and report the objective set.
//! **Evaluation only**: nothing is published on `signal.*`; the live `strategy-runtime` keeps
//! its own population.
//!
//! `Job.spec` is the genome text (`fa_strategies::genome::StrategyGenome` inline TOML) or the
//! envelope `{"genome": "<text>", "data_window": {...}, "generation": n, "rung": r,
//! "purpose": "selection" | "diagnostic"}`; without a spec, `Job.genome_id`. A `promotion`
//! purpose is **refused** here: promotion replays run only as `evo.eval.strategy.holdout`
//! ([`crate::holdout`]) inside a locked holdout.
//!
//! Result payload (protocol `evo.eval.strategy/v0`):
//! ```json
//! {"protocol":"evo.eval.strategy/v0","score":<deflated_sharpe>,
//!  "objectives":{"deflated_sharpe","sharpe","max_drawdown","turnover","cost_sensitivity","compute_cost","calibration","cost_share","psr","n_trades","n_obs","sources.graph","sources.decoder","sources.fallback"},
//!  "directions":{"deflated_sharpe":"max","sharpe":"max","max_drawdown":"min","turnover":"min","cost_sensitivity":"min","compute_cost":"min"},
//!  "genome":{"text":"{ id = … }","id":"gp-…","timescale":"1s","steps":5,"lookback_bars":32,"signal":"…","uncertainty":"…"},
//!  "replay":{"purpose":"selection","window_start_ms":…,"window_end_ms":…,"cost_model":"costs.yaml","capital":…,"n_trials":…,"trial_sharpe_var":…,
//!            "instruments":[…],"n_steps":…,"n_obs":…,"n_trades":…,"gross_pnl":…,"total_cost":…,"net_pnl":…,"stop":"complete",
//!            "snapshot":"stat-v1-1s","sources":{"graph":…,"decoder":…,"fallback":…},"latents":{"available":true,"fed":{…},"snapshot":{…}}},
//!  "latents":{"source":"jetstream LATENT …","cached":false,"summary":{…},"cache":{…}},
//!  "data":{"source":"victoriametrics",…},"generation_stats":{…}|null,"helix":{…},
//!  "budget":{…},"report_ref":{"planned":"fa-artifacts/evals/<generation>/<job>.json","persisted":true}}
//! ```
//! The equity curve and the per-trade log go to the Garage report only (not on the bus).

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use fa_backtest::{
    ReplayError,
    cost::CostModel,
    eval::Stop,
    replay::{
        self, DIRECTIONS, EvalReport, Evaluator, Holdout, PipelineReplay, ReplayControls, ReplayPurpose, ReplaySpec,
    },
};
use fa_hub::queries::{NewFitness, NewGenome};
use fa_proto::v1::{Job, JobResult, job_result::Status};
use fa_strategies::genome::StrategyGenome;
use serde_json::{Value, json};
use tokio::sync::watch;

use crate::{
    data::{self, DataWindow, LoadedBars},
    job::{Outcome, Worker},
    recorded::LoadedLatents,
    trials::{self, TrialRow},
};

pub const KIND: &str = "evo.eval.strategy";
pub const PROTOCOL: &str = "evo.eval.strategy/v0";
pub(crate) const DEFAULT_BUDGET_MS: u64 = 60_000;
/// Strategy capital per replay (EUR): large enough that the commission floor does not dominate.
pub const CAPITAL: f64 = 100_000.0;

/// Parsed eval envelope.
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyJobSpec {
    pub genome_text: String,
    pub window: DataWindow,
    pub gen_index: i32,
    pub rung: i32,
    pub purpose: ReplayPurpose,
}

/// Splits `Job.spec` into the genome text and the envelope fields (see module docs).
pub fn parse_spec(job: &Job) -> StrategyJobSpec {
    let raw = std::str::from_utf8(&job.spec).map(str::trim).unwrap_or("");
    let mut out = StrategyJobSpec {
        genome_text: job.genome_id.clone(),
        window: DataWindow::default(),
        gen_index: 0,
        rung: 0,
        purpose: ReplayPurpose::Selection,
    };
    if raw.is_empty() {
        return out;
    }
    if raw.starts_with('{')
        && let Ok(v) = serde_json::from_str::<Value>(raw)
        && (v.get("genome").is_some() || v.get("data_window").is_some())
    {
        if let Some(g) = v.get("genome").and_then(Value::as_str).filter(|g| !g.trim().is_empty()) {
            out.genome_text = g.to_owned();
        }
        out.window = v.get("data_window").cloned().and_then(|w| serde_json::from_value(w).ok()).unwrap_or_default();
        out.gen_index = v.get("generation").and_then(Value::as_i64).unwrap_or(0) as i32;
        out.rung = v.get("rung").and_then(Value::as_i64).unwrap_or(0) as i32;
        out.purpose = v.get("purpose").cloned().and_then(|p| serde_json::from_value(p).ok()).unwrap_or_default();
        return out;
    }
    out.genome_text = raw.to_owned();
    out
}

/// Why an `evo.eval.strategy` job may not run with `purpose = promotion` (spec 41 §6.4, D-059).
pub const PROMOTION_REFUSED: &str =
    "promotion replays run only as `evo.eval.strategy.holdout` jobs inside a locked holdout (spec 41 §6.4)";

/// Admission of an `evo.eval.strategy` job: only `selection` and `diagnostic` purposes.
pub fn admit(spec: &StrategyJobSpec) -> Result<(), &'static str> {
    if spec.purpose == ReplayPurpose::Promotion { Err(PROMOTION_REFUSED) } else { Ok(()) }
}

pub fn directions_json() -> Value {
    Value::Object(DIRECTIONS.iter().map(|(k, d)| ((*k).to_owned(), json!(d))).collect())
}

/// Objective map of a report as JSON.
pub fn objectives_json(report: &EvalReport) -> serde_json::Map<String, Value> {
    report.objectives().into_iter().map(|(k, v)| (k.to_owned(), json!(v))).collect()
}

/// `JobResult` carrying the typed objectives of a report.
pub fn ok_result(worker_id: &str, job: &Job, payload: &Value, report: &EvalReport, elapsed_ms: u64) -> JobResult {
    JobResult {
        job_id: job.job_id.clone(),
        status: Status::Ok as i32,
        result: serde_json::to_vec(payload).unwrap_or_default(),
        used_gpu_ms: elapsed_ms,
        worker_id: worker_id.to_owned(),
        objectives: report.objectives().into_iter().map(|(k, v)| (k.to_owned(), v)).collect(),
        directions: DIRECTIONS.iter().map(|(k, d)| ((*k).to_owned(), (*d).to_owned())).collect(),
        ..Default::default()
    }
}

/// The `replay` block of a result payload.
pub fn replay_json(rspec: &ReplaySpec, report: &EvalReport, n_holdouts: usize, registry: bool) -> Value {
    json!({
        "purpose": rspec.purpose, "window_start_ms": rspec.window_ms.0, "window_end_ms": rspec.window_ms.1,
        "cost_model": report.cost_model, "capital": report.capital, "n_trials": rspec.n_trials,
        "trial_sharpe_var": rspec.trial_sharpe_var, "instruments": report.instruments,
        "n_steps": report.n_steps, "n_obs": report.n_obs, "n_trades": report.n_trades,
        "gross_pnl": report.gross_pnl, "total_cost": report.total_cost, "net_pnl": report.net_pnl,
        "final_equity": report.final_equity, "first_ts_ms": report.first_ts_ms, "last_ts_ms": report.last_ts_ms,
        "stop": report.stop, "holdouts": n_holdouts, "registry": registry,
        "snapshot": report.latents.snapshot.declared, "sources": report.sources, "latents": report.latents,
    })
}

/// The `data` block of a result payload.
pub fn data_json(loaded: &LoadedBars) -> Value {
    json!({
        "source": loaded.source.source, "timescale": loaded.source.timescale, "step_ms": loaded.source.step_ms,
        "window_start_ms": loaded.source.window_start_ms, "window_end_ms": loaded.source.window_end_ms,
        "vm_url": loaded.source.vm_url, "vm_series": loaded.source.vm_series, "vm_error": loaded.source.vm_error,
        "bars": loaded.n_bars,
    })
}

/// A finished replay.
pub struct Replayed {
    pub report: EvalReport,
    pub latents: LoadedLatents,
    pub elapsed_ms: u64,
    pub budget_ms: u64,
    pub truncated: bool,
}

impl Worker {
    /// Loads the recorded latents for `rspec` and runs the replay within the job budget
    /// (blocking task, cancellable, progress every 500 ms). `Err` is the failure `Outcome` (boxed).
    pub(crate) async fn replay_within_budget(
        &self,
        job: &Job,
        def: &fa_strategies::population::StrategyDef,
        rspec: &ReplaySpec,
        bars: Vec<replay::BarSeries>,
        holdouts: Vec<Holdout>,
        started: Instant,
    ) -> Result<Replayed, Box<Outcome>> {
        let budget_ms = if job.budget_gpu_ms == 0 { DEFAULT_BUDGET_MS } else { job.budget_gpu_ms };
        let budget = Duration::from_millis(budget_ms);
        let latents = self.recorded.load(&self.bus.jetstream(), rspec).await;
        self.progress(
            &job.job_id,
            0.0,
            match &latents.latents {
                Some(r) => format!(
                    "recorded latents: {} messages ({})",
                    r.len(),
                    if latents.cached { "cached" } else { "read" }
                ),
                None => format!("no recorded latents ({})", latents.error.as_deref().unwrap_or("none")),
            },
        )
        .await;
        let cancel = Arc::new(AtomicBool::new(false));
        let (ptx, mut prx) = watch::channel((0usize, 0usize));
        let eval_started = Instant::now();
        let cost: Arc<CostModel> = self.cost.clone();
        let rec = latents.latents.clone();
        let task = {
            let cancel = cancel.clone();
            let def = def.clone();
            let rspec = rspec.clone();
            tokio::task::spawn_blocking(move || {
                let controls = ReplayControls {
                    max_steps: None,
                    should_stop: Some(Box::new(move |_| {
                        cancel.load(Ordering::Relaxed) || eval_started.elapsed() > budget
                    })),
                    on_progress: Some(Box::new(move |done, total| {
                        let _ = ptx.send((done, total));
                    })),
                    progress_every: 32,
                };
                PipelineReplay { data: &bars, cost: &cost, holdouts: &holdouts, controls, latents: rec.as_deref() }
                    .evaluate(&rspec, &def)
            })
        };
        let mut tick = tokio::time::interval(Duration::from_millis(500));
        tick.tick().await;
        let mut task = task;
        let result = loop {
            tokio::select! {
                r = &mut task => break r,
                _ = tick.tick() => {
                    let (done, total) = *prx.borrow_and_update();
                    let frac = if total > 0 { done as f32 / total as f32 } else { 0.0 };
                    self.progress(&job.job_id, frac.min(0.99), format!("step {done}/{total}")).await;
                }
            }
        };
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let report: EvalReport = match result {
            Ok(Ok(r)) => r,
            Ok(Err(ReplayError::NoData)) => {
                return Err(Box::new(self.failed(job, "no bars in the replay window".into(), elapsed_ms)));
            }
            Ok(Err(e @ (ReplayError::HoldoutIntersect { .. } | ReplayError::SnapshotMismatch { .. }))) => {
                tracing::warn!(job = %job.job_id, error = %e, "replay rejected");
                return Err(Box::new(self.failed(job, format!("rejected: {e}"), elapsed_ms)));
            }
            Ok(Err(e)) => return Err(Box::new(self.failed(job, format!("replay: {e}"), elapsed_ms))),
            Err(e) => return Err(Box::new(self.failed(job, format!("replay task failed: {e}"), elapsed_ms))),
        };
        if report.n_steps == 0 {
            let status = if eval_started.elapsed() > budget { Status::BudgetExceeded } else { Status::Failed };
            return Err(Box::new(Outcome {
                result: JobResult {
                    job_id: job.job_id.clone(),
                    status: status as i32,
                    error: format!("no decision step completed within the {budget_ms} ms budget"),
                    used_gpu_ms: elapsed_ms,
                    worker_id: self.worker_id.clone(),
                    ..Default::default()
                },
                outputs: Vec::new(),
            }));
        }
        let truncated = report.stop == Stop::Interrupted;
        Ok(Replayed { report, latents, elapsed_ms, budget_ms, truncated })
    }

    /// `evo.eval.strategy`.
    pub async fn evaluate_strategy(&self, job: &Job) -> Outcome {
        let started = Instant::now();

        // 1. genome + envelope
        let spec = parse_spec(job);
        if let Err(e) = admit(&spec) {
            return self.failed(job, format!("rejected: {e}"), 0);
        }
        let genome = match StrategyGenome::parse(&spec.genome_text) {
            Ok(g) => g,
            Err(e) => return self.failed(job, format!("strategy genome: {e}"), 0),
        };
        let def = genome.def();
        self.progress(&job.job_id, 0.0, format!("parsed strategy {} ({} nodes)", def.id, genome.size())).await;

        // 2. bars at the strategy's timescale (VictoriaMetrics, buffer fallback)
        let need = genome.lookback_bars + genome.steps as usize + 2;
        let loaded = data::load_bars(
            self.tsdb.as_ref(),
            &self.market,
            genome.timescale,
            None,
            &spec.window,
            self.data_window_hours,
            need,
        )
        .await;
        if loaded.bars.is_empty() {
            return self.failed(
                job,
                format!(
                    "insufficient market data: need ≥ {need} {} bars for `{}`, longest series has {} (source: {}; vm: {})",
                    genome.timescale,
                    def.id,
                    loaded.longest,
                    loaded.source.source,
                    loaded.source.vm_error.as_deref().unwrap_or("ok"),
                ),
                started.elapsed().as_millis() as u64,
            );
        }

        // 3. holdout + deflation inputs from the registry
        let holdouts = match &self.trials {
            Some(t) => t.holdouts(genome.timescale).await.unwrap_or_else(|e| {
                tracing::warn!(error = %e, "holdout lookup failed; assuming none");
                Vec::new()
            }),
            None => Vec::new(),
        };
        let (n_trials, sharpe_var) = match &self.trials {
            Some(t) if !job.parent_job_id.is_empty() => t.deflation_inputs(&job.parent_job_id).await,
            _ => (0, 0.0),
        };
        let rspec = ReplaySpec {
            window_ms: (loaded.source.window_start_ms, loaded.source.window_end_ms),
            purpose: spec.purpose,
            instruments: Vec::new(),
            timescale: genome.timescale,
            snapshot: job.snapshot_id.clone(),
            cost_model: self.cost.id.clone(),
            seed: job.seed,
            capital: CAPITAL,
            // this trial counts too
            n_trials: n_trials + 1,
            trial_sharpe_var: sharpe_var,
            ..ReplaySpec::selection(genome.timescale, 0, 0)
        };
        if let Err(e) = replay::check_holdout(&rspec, &holdouts) {
            tracing::warn!(job = %job.job_id, error = %e, "selection replay rejected (holdout)");
            return self.failed(job, format!("rejected: {e}"), started.elapsed().as_millis() as u64);
        }

        // 4. recorded latents + replay within budget
        let n_holdouts = holdouts.len();
        let data_block = data_json(&loaded);
        let run = match self.replay_within_budget(job, &def, &rspec, loaded.bars, holdouts, started).await {
            Ok(r) => r,
            Err(outcome) => return *outcome,
        };
        let report = &run.report;

        // 5. registry + stats, Helix, Garage, result
        let objectives = objectives_json(report);
        let generation_stats = trials::register(
            self.trials.as_ref(),
            &TrialRow {
                generation: if job.parent_job_id.is_empty() { "adhoc".into() } else { job.parent_job_id.clone() },
                gen_index: spec.gen_index,
                rung: spec.rung,
                job_id: job.job_id.clone(),
                candidate: genome.to_string(),
                kind: KIND.into(),
                window_start_ms: rspec.window_ms.0,
                window_end_ms: rspec.window_ms.1,
                sharpe: report.sharpe,
                n_obs: report.n_obs,
                objectives: Value::Object(objectives.clone()),
                returns: report.returns.clone(),
                holdout_id: None,
            },
        )
        .await;
        let helix = match tokio::time::timeout(
            Duration::from_secs(10),
            persist_helix(self.hub.as_ref(), &genome, spec.gen_index as u32, &objectives),
        )
        .await
        {
            Ok(v) => v,
            Err(_) => json!({"persisted": false, "reason": "helix timeout"}),
        };
        let report_key = fa_blob::eval_report(&job.parent_job_id, &job.job_id);
        let mut payload = json!({
            "protocol": PROTOCOL,
            "score": report.score(),
            "objectives": objectives,
            "directions": directions_json(),
            "genome": {"text": genome.to_string(), "id": def.id, "timescale": def.timescale, "steps": def.steps,
                       "lookback_bars": def.lookback_bars, "signal": def.signal, "uncertainty": def.uncertainty,
                       "nodes": genome.size()},
            "replay": replay_json(&rspec, report, n_holdouts, self.trials.is_some()),
            "latents": self.recorded.json(&run.latents),
            "data": data_block,
            "generation_stats": generation_stats,
            "helix": helix,
            "budget": {"budget_ms": run.budget_ms, "elapsed_ms": run.elapsed_ms, "truncated": run.truncated},
            "stop": report.stop,
            "report_ref": {"planned": format!("{}/{}", report_key.bucket, report_key.key), "persisted": self.blob.is_some()},
            "worker_id": self.worker_id,
        });
        // The stored document carries the curves; the bus payload does not.
        let mut stored = payload.clone();
        stored["equity"] = json!(report.equity);
        stored["trades"] = json!(report.trades);
        stored["returns"] = json!(report.returns);
        if !self.persist_report(&report_key, &stored).await {
            payload["report_ref"]["persisted"] = json!(false);
        }
        tracing::info!(job = %job.job_id, strategy = %def.id, sharpe = report.sharpe, dsr = report.deflated_sharpe,
            trades = report.n_trades, steps = report.n_steps, sources = ?report.sources, "strategy evaluated");
        Outcome { result: ok_result(&self.worker_id, job, &payload, report, run.elapsed_ms), outputs: Vec::new() }
    }
}

/// Best-effort Helix write: the strategy as a `Genome` node (kind `strategy`, family `gp`,
/// params = text form) plus a `FitnessRecord` with the objectives (spec 35 §6.4 elites are a
/// subset of these; the vary job re-scores the parents).
pub async fn persist_helix(
    hub: Option<&fa_hub::HubClient>,
    genome: &StrategyGenome,
    generation: u32,
    objectives: &serde_json::Map<String, Value>,
) -> Value {
    let Some(hub) = hub else { return json!({"persisted": false, "reason": "no helix client"}) };
    let g =
        NewGenome { genome_id: genome.id(), kind: "strategy".into(), family: "gp".into(), params: genome.to_string() };
    if let Err(e) = hub.upsert_genome(&g).await {
        return json!({"persisted": false, "reason": e.to_string()});
    }
    let f = NewFitness {
        genome_id: genome.id(),
        generation,
        objectives: serde_json::to_string(objectives).unwrap_or_default(),
        pareto_rank: 0,
    };
    match hub.add_fitness_record(&f).await {
        Ok(_) => json!({"persisted": true, "genome": genome.id()}),
        Err(e) => json!({"persisted": false, "reason": e.to_string()}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_parsing_accepts_envelope_and_inline_genome() {
        let text = r#"{ id = "gp-1", timescale = "1s", signal = "(mul 2 analog_forecast)" }"#;
        let job =
            |spec: &str| Job { genome_id: "fallback".into(), spec: spec.as_bytes().to_vec(), ..Default::default() };
        let s = parse_spec(&job(text));
        assert_eq!(s.genome_text, text, "a TOML inline table is not a JSON envelope");
        assert_eq!((s.gen_index, s.rung, s.purpose), (0, 0, ReplayPurpose::Selection));
        let env = format!(
            r#"{{"genome": {}, "data_window": {{"hours": 2}}, "generation": 3, "rung": 1, "purpose": "promotion"}}"#,
            serde_json::to_string(text).unwrap()
        );
        let s = parse_spec(&job(&env));
        assert_eq!(s.genome_text, text);
        assert_eq!((s.window.hours, s.gen_index, s.rung, s.purpose), (Some(2.0), 3, 1, ReplayPurpose::Promotion));
        assert!(StrategyGenome::parse(&s.genome_text).is_ok());
        assert_eq!(parse_spec(&job("")).genome_text, "fallback");
        assert_eq!(directions_json()["deflated_sharpe"], "max");
        assert_eq!(parse_spec(&job(r#"{"data_window": {"hours": 1}}"#)).genome_text, "fallback");
    }

    #[test]
    fn promotion_purpose_is_refused_outside_the_holdout_job() {
        let text = r#"{ id = "gp-1", timescale = "1s", signal = "1" }"#;
        let job = |spec: String| Job { spec: spec.into_bytes(), ..Default::default() };
        let env = format!(r#"{{"genome": {}, "purpose": "promotion"}}"#, serde_json::to_string(text).unwrap());
        let refused = admit(&parse_spec(&job(env))).unwrap_err();
        assert!(refused.contains("evo.eval.strategy.holdout"), "{refused}");
        assert!(admit(&parse_spec(&job(text.to_owned()))).is_ok());
        let diag = format!(r#"{{"genome": {}, "purpose": "diagnostic"}}"#, serde_json::to_string(text).unwrap());
        assert!(admit(&parse_spec(&job(diag))).is_ok());
    }

    /// `FA_NATS_TEST=1`: the worker refuses the job before touching any data.
    #[tokio::test]
    async fn worker_refuses_promotion_purpose() {
        if std::env::var("FA_NATS_TEST").is_err() {
            return;
        }
        let worker = crate::holdout::tests::offline_worker().await;
        let text = r#"{ id = "gp-1", timescale = "1s", signal = "1" }"#;
        let env = format!(r#"{{"genome": {}, "purpose": "promotion"}}"#, serde_json::to_string(text).unwrap());
        let job = Job { job_id: "j1".into(), kind: KIND.into(), spec: env.into_bytes(), ..Default::default() };
        let out = worker.evaluate_strategy(&job).await;
        assert_eq!(out.result.status, Status::Failed as i32);
        assert!(out.result.error.contains("evo.eval.strategy.holdout"), "{}", out.result.error);
        eprintln!("promotion purpose → {}", out.result.error);
    }
}
