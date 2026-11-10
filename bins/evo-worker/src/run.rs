//! Uniform job lifecycle for every job kind (spec 60 §6.3): a [`JobRun`] carries the job's
//! deadline and budget clock, the preempt and cancel flags (set by [`JobRun::watch`] from
//! `ctl.gpu.preempt` / `ctl.evo.cancel` notices naming the job id), the step counter that
//! becomes the `checkpoint_ref`, and the `JobProgress` publisher. Kind handlers hand
//! [`JobRun::stopper`] / [`JobRun::on_progress`] to their blocking evaluation, await it through
//! [`JobRun::drive`] (progress every 500 ms) and check [`JobRun::interrupted`] afterwards;
//! [`guard`] wraps the whole handler so a cancelled or preempted job always ends with
//! `STATUS_CANCELLED` / `STATUS_PREEMPTED`, even when the handler does not return within
//! [`GRACE`] of the interruption.

use std::{
    future::Future,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use fa_bus::{Bus, subjects};
use fa_proto::v1::{Job, JobProgress, JobResult, job_result::Status};
use tokio::sync::broadcast;

use crate::{gpu::Notice, job::Outcome, market::now_ms};

/// How long [`guard`] waits for a handler to wind down after a cancel / preempt.
pub const GRACE: Duration = Duration::from_secs(10);
/// Progress publish period of [`JobRun::drive`].
pub const PROGRESS_EVERY: Duration = Duration::from_millis(500);

/// One running job.
#[derive(Clone)]
pub struct JobRun {
    pub job_id: String,
    pub kind: String,
    pub started: Instant,
    pub deadline: Instant,
    pub budget: Duration,
    budget_from: Arc<OnceLock<Instant>>,
    preempted: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
    done: Arc<AtomicUsize>,
    total: Arc<AtomicUsize>,
    bus: Option<Bus>,
}

/// `Job.deadline`, else `budget · 10 + 60 s` from now.
pub fn deadline_of(job: &Job, budget: Duration) -> Instant {
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

impl JobRun {
    /// Run state for `job`; `default_budget_ms` applies when `Job.budget_gpu_ms` is 0.
    pub fn new(job: &Job, default_budget_ms: u64, bus: Option<Bus>) -> Self {
        let budget = Duration::from_millis(if job.budget_gpu_ms == 0 { default_budget_ms } else { job.budget_gpu_ms });
        Self {
            job_id: job.job_id.clone(),
            kind: job.kind.clone(),
            started: Instant::now(),
            deadline: deadline_of(job, budget),
            budget,
            budget_from: Arc::new(OnceLock::new()),
            preempted: Arc::new(AtomicBool::new(false)),
            cancelled: Arc::new(AtomicBool::new(false)),
            done: Arc::new(AtomicUsize::new(0)),
            total: Arc::new(AtomicUsize::new(0)),
            bus,
        }
    }

    pub fn budget_ms(&self) -> u64 {
        self.budget.as_millis() as u64
    }

    /// Starts the budget clock (first call wins); before it the budget is not spent.
    pub fn start_budget(&self) {
        let _ = self.budget_from.set(Instant::now());
    }

    pub fn over_budget(&self) -> bool {
        self.budget_from.get().is_some_and(|t| t.elapsed() > self.budget)
    }

    pub fn past_deadline(&self) -> bool {
        Instant::now() >= self.deadline
    }

    pub fn preempt(&self) {
        self.preempted.store(true, Ordering::Relaxed);
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    pub fn is_preempted(&self) -> bool {
        self.preempted.load(Ordering::Relaxed)
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Cancelled, preempted, over budget or past the deadline.
    pub fn should_stop(&self) -> bool {
        self.is_cancelled() || self.is_preempted() || self.over_budget() || self.past_deadline()
    }

    /// `should_stop` hook for `EvalSpec` / `ReplayControls` (records the step reached).
    pub fn stopper(&self) -> Box<dyn Fn(usize) -> bool + Send + Sync> {
        let run = self.clone();
        Box::new(move |step| {
            run.done.fetch_max(step, Ordering::Relaxed);
            run.should_stop()
        })
    }

    /// `on_progress` hook: records `(done, total)` for [`drive`](Self::drive) and the checkpoint.
    pub fn on_progress(&self) -> Box<dyn Fn(usize, usize) + Send + Sync> {
        let (done, total) = (self.done.clone(), self.total.clone());
        Box::new(move |d, t| {
            done.store(d, Ordering::Relaxed);
            total.store(t, Ordering::Relaxed);
        })
    }

    /// `(done, total)` steps reported so far.
    pub fn steps(&self) -> (usize, usize) {
        (self.done.load(Ordering::Relaxed), self.total.load(Ordering::Relaxed))
    }

    /// Resume point of an interrupted job: `<kind>/v0:step=<n>`.
    pub fn checkpoint_ref(&self) -> String {
        format!("{}/v0:step={}", self.kind, self.steps().0)
    }

    /// Publishes `JobProgress` (best effort; no-op without a bus).
    pub async fn progress(&self, fraction: f32, message: impl Into<String>) {
        let Some(bus) = &self.bus else { return };
        let p = JobProgress { job_id: self.job_id.clone(), fraction, message: message.into() };
        if let Err(e) = bus.publish(subjects::job_progress(&self.job_id), &p).await {
            tracing::debug!(error = %e, "progress publish failed");
        }
    }

    /// Awaits `fut`, publishing progress from the recorded steps (or the budget share) every
    /// [`PROGRESS_EVERY`].
    pub async fn drive<T>(&self, fut: impl Future<Output = T>, label: &str) -> T {
        tokio::pin!(fut);
        let mut tick = tokio::time::interval(PROGRESS_EVERY);
        tick.tick().await;
        loop {
            tokio::select! {
                r = &mut fut => return r,
                _ = tick.tick() => {
                    let (d, t) = self.steps();
                    let frac = if t > 0 { d as f32 / t as f32 } else { 0.0 };
                    let budget_frac = self.budget_from.get()
                        .map_or(0.0, |s| s.elapsed().as_secs_f32() / self.budget.as_secs_f32().max(1e-3));
                    self.progress(frac.max(budget_frac.min(0.99)).min(0.99), format!("{label}: step {d}/{t}")).await;
                }
            }
        }
    }

    /// Sets the flags from `ctl.gpu.preempt` / `ctl.evo.cancel` notices naming this job until
    /// the returned handle is aborted.
    pub fn watch(&self, mut notices: broadcast::Receiver<Notice>) -> tokio::task::JoinHandle<()> {
        let run = self.clone();
        tokio::spawn(async move {
            loop {
                match notices.recv().await {
                    Ok(Notice::Preempt(id)) if id == run.job_id => {
                        tracing::warn!(job = %run.job_id, "preempted by the gpu arbiter");
                        run.preempt();
                    }
                    Ok(Notice::Cancel(id)) if id == run.job_id => {
                        tracing::warn!(job = %run.job_id, "cancelled (ctl.evo.cancel) while running");
                        run.cancel();
                    }
                    Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(broadcast::error::RecvError::Closed) => return,
                }
            }
        })
    }

    /// The interruption outcome when the job was cancelled (wins) or preempted.
    pub fn interrupted(&self, worker_id: &str) -> Option<Outcome> {
        let (status, error, checkpoint) = if self.is_cancelled() {
            (Status::Cancelled, "cancelled by operator (ctl.evo.cancel while running)".to_owned(), String::new())
        } else if self.is_preempted() {
            (Status::Preempted, String::new(), self.checkpoint_ref())
        } else {
            return None;
        };
        Some(Outcome {
            result: JobResult {
                job_id: self.job_id.clone(),
                status: status as i32,
                error,
                checkpoint_ref: checkpoint,
                used_gpu_ms: self.started.elapsed().as_millis() as u64,
                worker_id: worker_id.to_owned(),
                ..Default::default()
            },
            outputs: Vec::new(),
        })
    }
}

/// Runs a handler under `run`: an interruption overrides whatever the handler produced, and a
/// handler that has not returned [`GRACE`] after the interruption is dropped (its blocking work
/// sees `should_stop` and winds down on its own).
pub async fn guard(run: &JobRun, worker_id: &str, grace: Duration, handler: impl Future<Output = Outcome>) -> Outcome {
    tokio::pin!(handler);
    let interrupted = async {
        loop {
            if run.is_cancelled() || run.is_preempted() {
                tokio::time::sleep(grace).await;
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    };
    let out = tokio::select! {
        o = &mut handler => o,
        _ = interrupted => {
            tracing::warn!(job = %run.job_id, "handler did not stop within the grace period; dropped");
            Outcome { result: JobResult::default(), outputs: Vec::new() }
        }
    };
    run.interrupted(worker_id).unwrap_or(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(budget_ms: u64, deadline_in_ms: Option<i64>) -> Job {
        Job {
            job_id: "j1".into(),
            kind: "evo.eval.test".into(),
            budget_gpu_ms: budget_ms,
            deadline: deadline_in_ms.map(|d| crate::job::ts_ms(now_ms() + d)),
            ..Default::default()
        }
    }

    fn ok(job_id: &str) -> Outcome {
        Outcome { result: JobResult { job_id: job_id.into(), ..Default::default() }, outputs: Vec::new() }
    }

    #[test]
    fn budget_deadline_and_flags() {
        let run = JobRun::new(&job(0, None), 60_000, None);
        assert_eq!(run.budget_ms(), 60_000, "default budget");
        assert!(!run.should_stop() && !run.over_budget(), "budget clock not started");
        let short = JobRun::new(&job(30, None), 60_000, None);
        short.start_budget();
        assert!(!short.over_budget());
        std::thread::sleep(Duration::from_millis(40));
        assert!(short.over_budget() && short.should_stop());
        let late = JobRun::new(&job(10_000, Some(-5)), 60_000, None);
        assert!(late.past_deadline() && late.should_stop(), "deadline in the past");
        let soon = JobRun::new(&job(10_000, Some(60_000)), 60_000, None);
        assert!(!soon.past_deadline());
        // stopper records the step; preempt → checkpoint_ref; cancel wins
        let stop = run.stopper();
        assert!(!stop(7));
        run.on_progress()(9, 20);
        assert_eq!(run.steps(), (9, 20));
        assert!(run.interrupted("w").is_none());
        run.preempt();
        assert!(stop(3), "a preempted run stops");
        let p = run.interrupted("w").unwrap().result;
        assert_eq!((p.status, p.checkpoint_ref.as_str()), (Status::Preempted as i32, "evo.eval.test/v0:step=9"));
        run.cancel();
        let c = run.interrupted("w").unwrap().result;
        assert_eq!(c.status, Status::Cancelled as i32);
        assert!(c.error.contains("cancel"));
    }

    #[tokio::test]
    async fn guard_overrides_and_drops_hung_handlers() {
        // a cooperative handler stops at the flag; the outcome becomes CANCELLED
        let run = JobRun::new(&job(60_000, None), 60_000, None);
        run.start_budget();
        let r2 = run.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            r2.cancel();
        });
        let r3 = run.clone();
        let started = Instant::now();
        let out = guard(&run, "w", GRACE, async move {
            let stop = r3.stopper();
            let mut n = 0;
            while !stop(n) {
                n += 1;
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            ok("j1")
        })
        .await;
        assert_eq!(out.result.status, Status::Cancelled as i32);
        assert!(started.elapsed() < Duration::from_secs(2));
        // a handler that never looks at the flags is dropped after the grace period
        let run = JobRun::new(&job(60_000, None), 60_000, None);
        run.preempt();
        let out = guard(&run, "w", Duration::from_millis(100), std::future::pending()).await;
        assert_eq!(out.result.status, Status::Preempted as i32);
        assert!(out.result.checkpoint_ref.starts_with("evo.eval.test/v0:step="));
        // an uninterrupted handler's outcome passes through
        let run = JobRun::new(&job(60_000, None), 60_000, None);
        let out = guard(&run, "w", GRACE, async { ok("j1") }).await;
        assert_eq!(out.result.status, Status::Ok as i32);
        // watch() maps notices for this job id only
        let (tx, _) = broadcast::channel(8);
        let h = run.watch(tx.subscribe());
        tx.send(Notice::Cancel("other".into())).unwrap();
        tx.send(Notice::Preempt("j1".into())).unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(run.is_preempted() && !run.is_cancelled());
        tx.send(Notice::Cancel("j1".into())).unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(run.is_cancelled());
        h.abort();
    }
}
