//! Allocator tournament jobs (spec 40 §6.2, §3 "pluggable meta-allocator"):
//!
//! * `evo.eval.allocator` replays the recorded strategy signals of the window (`SIGNAL`
//!   stream, `signal.exposure.>`, read by time through the incremental cache of
//!   [`crate::recorded`]) and the VictoriaMetrics marks through one
//!   [`AllocatorGenome`] → netting → the shared cost model (`fa_backtest::allocator_replay`),
//!   scores the portfolio and writes a `validation.trial` row with `kind = allocator`
//!   (generation statistics as for strategies).
//! * `evo.vary.allocator` breeds the next population with `fa_evolution::vary::vary_with` and
//!   `fa_allocator::genome::AllocatorOps` plus random immigrants.
//!
//! `Job.spec` of an eval is the genome text (`alloc:name=risk_parity;cap=0.4;…`) or the
//! orchestrator envelope `{"genome", "generation", "rung", "data_window"?}`; now-relative windows
//! are pinned to the minute like strategy replays, so every candidate of a rung sees identical
//! signals and marks (comparable portfolio P&L, spec 40 §9).
//!
//! Result (protocol `evo.eval.allocator/v0`): `score` = `deflated_sharpe`; `objectives` /
//! `directions` = `deflated_sharpe` max, `sharpe` max, `max_drawdown` min, `turnover` min,
//! `concentration` min, `cost_share` min (+ informational `psr`, `n_trades`, `n_obs`,
//! `n_strategies`, `mean_gross`, `max_weight`); `genome {text, id, name, params}`,
//! `selection {name, params}` (the `ctl.risk.limits` payload), `replay {…}`, `signals {…}`,
//! `data`, `generation_stats`, `helix`, `report_ref`. The Garage report adds `equity`, `returns`.

use std::{collections::BTreeSet, time::Duration};

use fa_allocator::genome::{AllocatorGenome, AllocatorOps};
use fa_backtest::{
    ReplayError,
    allocator_replay::{AllocatorReplaySpec, DIRECTIONS, replay_allocator},
    eval::Stop,
    replay::ReplayControls,
};
use fa_core::{Timescale, rng::Rng};
use fa_evolution::{plan_halving, vary::vary_with};
use fa_hub::{
    HubClient,
    queries::{NewFitness, NewGenome},
};
use fa_proto::v1::{Job, JobResult, job_result::Status};
use serde_json::{Value, json};

use crate::{
    data::{self, PIN_MS},
    job::{Outcome, Worker},
    market::now_ms,
    run::JobRun,
    strategy::{self, CAPITAL},
    trials::{self, TrialRow},
    vary::{Lineage, MAX_CHILDREN, VaryRequest},
};

pub const KIND: &str = "evo.eval.allocator";
pub const VARY_KIND: &str = "evo.vary.allocator";
pub const PROTOCOL: &str = "evo.eval.allocator/v0";
pub const VARY_PROTOCOL: &str = "evo.vary.allocator/v0";
/// `validation.trial.kind` of allocator evaluations.
pub const TRIAL_KIND: &str = "allocator";
/// Decision grid of the portfolio replay.
pub const STEP_MS: i64 = 1_000;
/// A strategy's latest signal counts this long (risk-engine `signal_ttl_secs`).
pub const SIGNAL_TTL_MS: i64 = 30_000;
/// Share of random immigrants among the children.
pub const DEFAULT_IMMIGRANT_FRAC: f64 = 0.1;

pub fn directions_json() -> Value {
    Value::Object(DIRECTIONS.iter().map(|(k, d)| ((*k).to_owned(), json!(d))).collect())
}

impl Worker {
    /// `evo.eval.allocator`.
    pub async fn evaluate_allocator(&self, job: &Job, run: &JobRun) -> Outcome {
        let started = std::time::Instant::now();
        let mut spec = strategy::parse_spec(job);
        spec.window = spec.window.pinned(now_ms(), PIN_MS);
        let genome = match AllocatorGenome::parse(&spec.genome_text) {
            Ok(g) => g,
            Err(e) => return self.failed(job, format!("allocator genome: {e}"), 0),
        };
        run.progress(0.0, format!("allocator {}", genome)).await;

        // 1. marks: 1 s closes of every instrument (VictoriaMetrics, buffer fallback)
        let loaded = data::load_bars(
            self.tsdb.as_ref(),
            &self.market,
            Timescale::Second,
            None,
            &spec.window,
            self.data_window_hours,
            2,
        )
        .await;
        if loaded.bars.is_empty() {
            return self.failed(
                job,
                format!(
                    "no marks in the window (source: {}; vm: {})",
                    loaded.source.source,
                    loaded.source.vm_error.as_deref().unwrap_or("ok")
                ),
                started.elapsed().as_millis() as u64,
            );
        }
        let (w0, w1) = (loaded.source.window_start_ms, loaded.source.window_end_ms);

        // 2. recorded signals (TTL preload before the window)
        let sig = self.recorded.load_signals(&self.bus.jetstream(), w0 - SIGNAL_TTL_MS, w1).await;
        if let Some(e) = &sig.error {
            return self.failed(
                job,
                format!("recorded signals unavailable: {e}"),
                started.elapsed().as_millis() as u64,
            );
        }
        if sig.signals.is_empty() {
            return self.failed(
                job,
                format!("no recorded signal.exposure.* in [{w0}, {w1}] (SIGNAL stream)"),
                started.elapsed().as_millis() as u64,
            );
        }
        run.progress(0.0, format!("{} signals (cache {}, {} read)", sig.signals.len(), sig.plan, sig.messages_read))
            .await;

        // 3. deflation inputs, replay within budget
        let (n_trials, sharpe_var) = match &self.trials {
            Some(t) if !job.parent_job_id.is_empty() => t.deflation_inputs(&job.parent_job_id).await,
            _ => (0, 0.0),
        };
        let rspec = AllocatorReplaySpec {
            window_ms: (w0, w1),
            step_ms: STEP_MS,
            capital: CAPITAL,
            signal_ttl_ms: SIGNAL_TTL_MS,
            n_trials: n_trials + 1,
            trial_sharpe_var: sharpe_var,
            // the live risk-engine's rebalance rule (crate::live_rule), unless the job overrides it
            trade_rule: spec.trade_rule.clone().unwrap_or_else(|| self.trade_rule.clone()),
            ..AllocatorReplaySpec::new(w0, w1)
        };
        run.start_budget();
        let task = {
            let (g, signals, bars, rspec, cost, run) = (
                genome.clone(),
                sig.signals.clone(),
                loaded.bars.clone(),
                rspec.clone(),
                self.cost.clone(),
                run.clone(),
            );
            tokio::task::spawn_blocking(move || {
                let controls = ReplayControls {
                    max_steps: None,
                    should_stop: Some(run.stopper()),
                    on_progress: Some(run.on_progress()),
                    progress_every: 64,
                };
                replay_allocator(&g, &signals, &bars, &rspec, &cost, &controls)
            })
        };
        let result = run.drive(task, genome.name.as_str()).await;
        if let Some(out) = run.interrupted(&self.worker_id) {
            return out;
        }
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let report = match result {
            Ok(Ok(r)) => r,
            Ok(Err(ReplayError::NoData)) => {
                return self.failed(job, "no signals or marks in the replay window".into(), elapsed_ms);
            }
            Ok(Err(e)) => return self.failed(job, format!("allocator replay: {e}"), elapsed_ms),
            Err(e) => return self.failed(job, format!("allocator replay task failed: {e}"), elapsed_ms),
        };
        if report.n_obs == 0 {
            let status = if run.over_budget() { Status::BudgetExceeded } else { Status::Failed };
            return Outcome {
                result: JobResult {
                    job_id: job.job_id.clone(),
                    status: status as i32,
                    error: format!("no portfolio step completed within the {} ms budget", run.budget_ms()),
                    used_gpu_ms: elapsed_ms,
                    worker_id: self.worker_id.clone(),
                    ..Default::default()
                },
                outputs: Vec::new(),
            };
        }

        // 4. registry, Helix, Garage, result
        let objectives: serde_json::Map<String, Value> =
            report.objectives().into_iter().map(|(k, v)| (k.to_owned(), json!(v))).collect();
        let generation_stats = trials::register(
            self.trials.as_ref(),
            &TrialRow {
                generation: if job.parent_job_id.is_empty() { "adhoc".into() } else { job.parent_job_id.clone() },
                gen_index: spec.gen_index,
                rung: spec.rung,
                job_id: job.job_id.clone(),
                candidate: genome.to_string(),
                kind: TRIAL_KIND.into(),
                window_start_ms: w0,
                window_end_ms: w1,
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
            "genome": {"text": genome.to_string(), "id": genome.id(), "name": genome.name, "params": genome.params()},
            "selection": genome.spec(),
            "replay": {
                "window_start_ms": w0, "window_end_ms": w1, "step_ms": rspec.step_ms, "capital": rspec.capital,
                "signal_ttl_ms": rspec.signal_ttl_ms, "cost_model": report.cost_model, "n_trials": rspec.n_trials,
                "trial_sharpe_var": rspec.trial_sharpe_var, "n_steps": report.n_steps, "n_obs": report.n_obs,
                "n_active_steps": report.n_active_steps, "n_signals": report.n_signals, "strategies": report.strategies,
                "instruments": report.instruments.len(), "n_trades": report.n_trades, "gross_pnl": report.gross_pnl,
                "total_cost": report.total_cost, "net_pnl": report.net_pnl, "final_equity": report.final_equity,
                "mean_weights": report.mean_weights, "stop": report.stop,
                "trade_rule": rspec.trade_rule, "total_fees": report.total_fees, "rebalance": report.rebalance,
            },
            "signals": {"source": "jetstream SIGNAL (signal.exposure.>, by_start_time ordered consumer)", "plan": sig.plan,
                        "messages_read": sig.messages_read, "decoded": sig.signals.len(), "undecodable": sig.undecodable,
                        "read_start_ms": sig.read_start_ms, "read_end_ms": sig.read_end_ms, "elapsed_ms": sig.elapsed_ms},
            "data": strategy::data_json(&loaded),
            "generation_stats": generation_stats,
            "helix": helix,
            "budget": {"budget_ms": run.budget_ms(), "elapsed_ms": elapsed_ms, "truncated": report.stop == Stop::Interrupted},
            "stop": report.stop,
            "report_ref": {"planned": format!("{}/{}", report_key.bucket, report_key.key), "persisted": self.blob.is_some()},
            "worker_id": self.worker_id,
        });
        let mut stored = payload.clone();
        stored["equity"] = json!(report.equity);
        stored["returns"] = json!(report.returns);
        if !self.persist_report(&report_key, &stored).await {
            payload["report_ref"]["persisted"] = json!(false);
        }
        tracing::info!(job = %job.job_id, allocator = %genome, sharpe = report.sharpe, dsr = report.deflated_sharpe,
            trades = report.n_trades, strategies = report.strategies.len(), signals = report.n_signals, "allocator evaluated");
        Outcome {
            result: JobResult {
                job_id: job.job_id.clone(),
                status: Status::Ok as i32,
                result: serde_json::to_vec(&payload).unwrap_or_default(),
                used_gpu_ms: elapsed_ms,
                worker_id: self.worker_id.clone(),
                objectives: report.objectives().into_iter().map(|(k, v)| (k.to_owned(), v)).collect(),
                directions: DIRECTIONS.iter().map(|(k, d)| ((*k).to_owned(), (*d).to_owned())).collect(),
                ..Default::default()
            },
            outputs: Vec::new(),
        }
    }

    /// `evo.vary.allocator`.
    pub async fn vary_allocator(&self, job: &Job) -> Outcome {
        let started = std::time::Instant::now();
        let req: VaryRequest = match serde_json::from_slice(&job.spec) {
            Ok(r) => r,
            Err(e) => return self.failed(job, format!("vary spec: {e}"), 0),
        };
        let out = match produce(&req) {
            Ok(o) => o,
            Err(e) => return self.failed(job, format!("vary: {e}"), started.elapsed().as_millis() as u64),
        };
        let helix =
            match tokio::time::timeout(Duration::from_secs(20), persist_lineage(self.hub.as_ref(), &req, &out)).await {
                Ok(v) => v,
                Err(_) => json!({"persisted": false, "reason": "helix timeout"}),
            };
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let payload = json!({
            "protocol": VARY_PROTOCOL,
            "children": out.children.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "child_ids": out.children.iter().map(AllocatorGenome::id).collect::<Vec<_>>(),
            "n_parents": out.parents.len(), "n_children": out.children.len(), "immigrants": out.immigrants,
            "requested_children": req.n_children, "generation": req.generation, "seed": req.seed, "plan": out.plan,
            "lineage": out.lineage, "helix": helix, "worker_id": self.worker_id, "elapsed_ms": elapsed_ms,
        });
        tracing::info!(job = %job.job_id, parents = out.parents.len(), children = out.children.len(), "allocator variation done");
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
}

#[derive(Debug)]
pub struct VaryOutcome {
    pub parents: Vec<AllocatorGenome>,
    pub children: Vec<AllocatorGenome>,
    pub lineage: Vec<Lineage>,
    pub immigrants: usize,
    pub plan: Option<Value>,
}

fn op_name(op: &str) -> &'static str {
    const KNOWN: [&str; 7] = [
        "gaussian_gene",
        "gaussian_all",
        "name_switch",
        "crossover+gaussian_gene",
        "crossover+gaussian_all",
        "crossover+name_switch",
        "immigrant",
    ];
    KNOWN.iter().find(|k| **k == op).copied().unwrap_or("variation")
}

/// Pure variation step: tournament + uniform crossover + Gaussian / name mutation through the
/// shared loop, random immigrants, de-duplicated by genome id against the parents.
pub fn produce(req: &VaryRequest) -> Result<VaryOutcome, String> {
    if req.parents.is_empty() {
        return Err("no parents".into());
    }
    let parents: Vec<AllocatorGenome> = req
        .parents
        .iter()
        .map(|p| AllocatorGenome::parse(p.genome()).map_err(|e| format!("parent `{}`: {e}", p.genome())))
        .collect::<Result<_, _>>()?;
    let (n_children, plan) = match (req.n_children, req.budget) {
        (0, Some(b)) => {
            let plan = plan_halving(b.total_gpu_ms, b.min_budget_gpu_ms, b.eta, b.rungs);
            (
                plan.initial_candidates.saturating_sub(parents.len()),
                Some(json!({"initial_candidates": plan.initial_candidates, "rungs": plan.rungs, "eta": plan.eta,
                            "min_budget_gpu_ms": plan.min_budget_gpu_ms, "total_gpu_ms": b.total_gpu_ms})),
            )
        }
        (n, _) => (n, None),
    };
    let n_children = n_children.min(MAX_CHILDREN);
    let n_immigrants =
        ((n_children as f64) * req.immigrant_frac.unwrap_or(DEFAULT_IMMIGRANT_FRAC).clamp(0.0, 1.0)).round() as usize;
    let n_bred = n_children.saturating_sub(n_immigrants);
    let ops = AllocatorOps { sigma: req.sigma.unwrap_or(AllocatorOps::default().sigma), ..AllocatorOps::default() };
    let crossover_prob = req.crossover_prob.unwrap_or(0.5);
    let mut seen: BTreeSet<String> = parents.iter().map(AllocatorGenome::id).collect();
    let (mut children, mut lineage) = (Vec::new(), Vec::new());
    let mut round = 0u64;
    while children.len() < n_bred && round < 8 {
        let want = (n_bred - children.len()) * 2;
        for c in vary_with(&parents, want, req.seed.wrapping_add(round.wrapping_mul(0x9E37_79B9)), crossover_prob, &ops)
        {
            if children.len() >= n_bred {
                break;
            }
            if !seen.insert(c.genome.id()) {
                continue;
            }
            lineage.push(Lineage {
                child: c.genome.to_string(),
                parents: c.parents.clone(),
                op: op_name(&c.op),
                parents_as_sent: Vec::new(),
            });
            children.push(c.genome);
        }
        round += 1;
    }
    let mut rng = Rng::seed(req.seed ^ 0xA110_C470);
    let (mut immigrants, mut tries) = (0usize, 0usize);
    while immigrants < n_immigrants && tries < n_immigrants * 8 + 8 {
        tries += 1;
        let g = AllocatorGenome::random(&mut rng);
        if !seen.insert(g.id()) {
            continue;
        }
        lineage.push(Lineage {
            child: g.to_string(),
            parents: Vec::new(),
            op: "immigrant",
            parents_as_sent: Vec::new(),
        });
        children.push(g);
        immigrants += 1;
    }
    Ok(VaryOutcome { parents, children, lineage, immigrants, plan })
}

fn new_genome(g: &AllocatorGenome) -> NewGenome {
    NewGenome { genome_id: g.id(), kind: "allocator".into(), family: g.name.to_string(), params: g.to_string() }
}

/// Best-effort Helix write of an evaluated allocator (`Genome{kind: allocator}` + `FitnessRecord`).
pub async fn persist_helix(
    hub: Option<&HubClient>,
    g: &AllocatorGenome,
    generation: u32,
    objectives: &serde_json::Map<String, Value>,
) -> Value {
    let Some(hub) = hub else { return json!({"persisted": false, "reason": "no helix client"}) };
    if let Err(e) = hub.upsert_genome(&new_genome(g)).await {
        return json!({"persisted": false, "reason": e.to_string()});
    }
    let f = NewFitness {
        genome_id: g.id(),
        generation,
        objectives: serde_json::to_string(objectives).unwrap_or_default(),
        pareto_rank: 0,
    };
    match hub.add_fitness_record(&f).await {
        Ok(_) => json!({"persisted": true, "genome": g.id()}),
        Err(e) => json!({"persisted": false, "reason": e.to_string()}),
    }
}

/// Best-effort lineage write (parents, children, `link_parent` edges by genome id).
pub async fn persist_lineage(hub: Option<&HubClient>, req: &VaryRequest, out: &VaryOutcome) -> Value {
    let Some(hub) = hub else { return json!({"persisted": false, "reason": "no helix client"}) };
    let (mut genomes, mut edges, mut errors) = (0usize, 0usize, Vec::<String>::new());
    for g in out.parents.iter().chain(&out.children) {
        match hub.upsert_genome(&new_genome(g)).await {
            Ok(_) => genomes += 1,
            Err(e) if errors.len() < 5 => errors.push(e.to_string()),
            Err(_) => {}
        }
    }
    for (c, l) in out.children.iter().zip(&out.lineage) {
        for p in &l.parents {
            match hub.link_parent(p, &c.id(), l.op).await {
                Ok(_) => edges += 1,
                Err(e) if errors.len() < 5 => errors.push(e.to_string()),
                Err(_) => {}
            }
        }
    }
    let _ = req;
    json!({"persisted": errors.is_empty(), "genomes": genomes, "lineage_edges": edges, "errors": errors})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(json: &str) -> VaryRequest {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn children_are_valid_distinct_deterministic() {
        let r = req(r#"{"parents":[{"genome":"alloc:name=risk_parity;cap=0.4","objectives":{"sharpe":0.1}},
                       "alloc:name=fractional_kelly;kelly=0.3","alloc:name=weighted_blend"],"n_children":10,"seed":4,"generation":1}"#);
        let a = produce(&r).unwrap();
        let b = produce(&r).unwrap();
        assert_eq!(a.children, b.children, "deterministic in seed");
        assert_eq!((a.children.len(), a.immigrants), (10, 1));
        let parent_ids: BTreeSet<String> = a.parents.iter().map(AllocatorGenome::id).collect();
        let mut seen = BTreeSet::new();
        for (c, l) in a.children.iter().zip(&a.lineage) {
            assert!(!parent_ids.contains(&c.id()) && seen.insert(c.id()));
            assert_eq!(AllocatorGenome::parse(&c.to_string()).unwrap(), *c);
            assert_eq!(l.child, c.to_string());
            assert!(l.op == "immigrant" || l.parents.iter().all(|p| parent_ids.contains(p)), "{l:?}");
        }
        assert!(a.lineage.iter().any(|l| l.op.starts_with("crossover+")));
        assert!(produce(&req(r#"{"parents":[],"n_children":1}"#)).is_err());
        assert!(produce(&req(r#"{"parents":["forecast:h=3"],"n_children":1}"#)).unwrap_err().contains("parent"));
        let sized = produce(&req(r#"{"parents":["alloc:name=risk_parity","alloc:"],"n_children":0,"seed":1,
            "budget":{"total_gpu_ms":3600000,"min_budget_gpu_ms":10000,"eta":3,"rungs":4}}"#))
        .unwrap();
        assert_eq!(sized.children.len(), 88);
        assert_eq!(directions_json()["concentration"], "min");
    }
}
