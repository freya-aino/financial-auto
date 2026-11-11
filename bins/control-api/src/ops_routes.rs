//! The `ops` schema writers and reads (spec 20 §5.4, spec 60; `fa_warehouse::ops`):
//! - **job history**: a durable JetStream consumer `ops-job-history` on `JOB_RESULTS`
//!   (`jobs.result.>`) stores every `fa.v1.JobResult` in `ops.job_run` (kind from the subject,
//!   `result` as JSON when it parses). Durable and explicitly acked, so results published while
//!   control-api was down are stored when it returns, and the stream's retained history is
//!   backfilled on first start;
//! - **config snapshots**: every 5 minutes (and at start) each platform config section and the
//!   effective `[runtime]` overlay, redacted like `GET /api/config/{section}`, are stored in
//!   `ops.config_snapshot` when their digest changed (`actor = 'system:control-api'`);
//! - reads: `GET /api/ops/jobs?kind=&limit=`, `GET /api/ops/jobs/summary?hours=`,
//!   `GET /api/ops/config-snapshots?service=&limit=&config=true`.
//!
//! Alert history (`ops.alert_event`) is written by `alerts_routes`.
use std::time::Duration;

use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use chrono::{DateTime, Utc};
use fa_bus::async_nats::jetstream::{
    self,
    consumer::{AckPolicy, DeliverPolicy, pull},
};
use fa_core::config::PlatformConfig;
use fa_proto::v1::{JobResult, job_result::Status};
use fa_warehouse::ops::{self, JobRun};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::json;

use crate::state::AppState;

/// Durable consumer name on `JOB_RESULTS`.
pub const CONSUMER: &str = "ops-job-history";
const ACTOR: &str = "system:control-api";

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/ops/jobs", get(jobs))
        .route("/api/ops/jobs/summary", get(summary))
        .route("/api/ops/config-snapshots", get(snapshots))
}

/// Starts the job-history consumer and the config snapshotter (both wait for Postgres).
pub fn spawn(state: &AppState) {
    spawn_job_history(state.clone());
    spawn_config_snapshots(state.clone());
}

fn status_name(s: i32) -> &'static str {
    match Status::try_from(s).unwrap_or(Status::Ok) {
        Status::Ok => "ok",
        Status::Failed => "failed",
        Status::Preempted => "preempted",
        Status::BudgetExceeded => "budget_exceeded",
        Status::Cancelled => "cancelled",
    }
}

/// The `ops.job_run` row of one `jobs.result.<kind>` message (pure).
pub fn job_run(subject: &str, r: &JobResult, stream_seq: Option<u64>, at: DateTime<Utc>) -> JobRun {
    let opt = |s: &str| Some(s.to_owned()).filter(|s| !s.is_empty());
    JobRun {
        job_id: r.job_id.clone(),
        kind: subject.strip_prefix("jobs.result.").unwrap_or(subject).to_owned(),
        status: status_name(r.status).to_owned(),
        worker_id: opt(&r.worker_id),
        error: opt(&r.error),
        used_gpu_ms: i64::try_from(r.used_gpu_ms).unwrap_or(i64::MAX),
        output_refs: r.output_refs.clone(),
        checkpoint_ref: opt(&r.checkpoint_ref),
        result: ops::result_json(&r.result),
        stream_seq: stream_seq.and_then(|s| i64::try_from(s).ok()),
        finished_at: at,
    }
}

async fn wait_for_pool(state: &AppState) -> sqlx::PgPool {
    loop {
        if let Some(p) = state.pg.get() {
            return p.clone();
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

fn spawn_job_history(state: AppState) {
    let Some(nc) = state.nats.clone() else { return };
    tokio::spawn(async move {
        let pool = wait_for_pool(&state).await;
        let js = jetstream::new(nc);
        let stored = fa_bus::metrics::counter("fa_ops_job_runs_stored_total", "Job results stored in ops.job_run");
        loop {
            let run = async {
                let stream = js.get_stream("JOB_RESULTS").await?;
                let consumer = stream
                    .get_or_create_consumer(
                        CONSUMER,
                        pull::Config {
                            durable_name: Some(CONSUMER.to_owned()),
                            filter_subject: "jobs.result.>".to_owned(),
                            deliver_policy: DeliverPolicy::All,
                            ack_policy: AckPolicy::Explicit,
                            max_deliver: 20,
                            ack_wait: Duration::from_secs(30),
                            ..Default::default()
                        },
                    )
                    .await?;
                let mut msgs = consumer.messages().await?;
                tracing::info!(consumer = CONSUMER, "job history consumer on JOB_RESULTS");
                while let Some(m) = msgs.next().await {
                    let m = m?;
                    let (seq, at) = m.info().map_or((None, Utc::now()), |i| {
                        let t = DateTime::from_timestamp_nanos(i.published.unix_timestamp_nanos() as i64);
                        (Some(i.stream_sequence), t)
                    });
                    match <JobResult as fa_proto::Message>::decode(m.payload.as_ref()) {
                        Ok(r) if !r.job_id.is_empty() => {
                            match ops::record_job_run(&pool, &job_run(m.subject.as_str(), &r, seq, at)).await {
                                Ok(_) => {
                                    stored.inc();
                                    let _ = m.ack().await;
                                }
                                // Postgres hiccup: redelivered after ack_wait
                                Err(e) => tracing::warn!(job = %r.job_id, error = %e, "job run not stored"),
                            }
                        }
                        _ => {
                            tracing::warn!(subject = %m.subject, "undecodable job result skipped");
                            let _ = m.ack_with(jetstream::AckKind::Term).await;
                        }
                    }
                }
                anyhow::Ok(())
            }
            .await;
            if let Err(e) = run {
                tracing::warn!(error = %e, "job history consumer stopped; retrying in 30 s");
            }
            tokio::time::sleep(Duration::from_secs(30)).await;
        }
    });
}

fn spawn_config_snapshots(state: AppState) {
    tokio::spawn(async move {
        let pool = wait_for_pool(&state).await;
        let mut t = tokio::time::interval(Duration::from_secs(300));
        loop {
            t.tick().await;
            let runtime = state.live().runtime().clone();
            let sections: Vec<(&str, serde_json::Value)> = PlatformConfig::section_names()
                .iter()
                .copied()
                .chain(["runtime"])
                .filter_map(|s| crate::config_routes::redacted_section(&state.cfg, &runtime, s).map(|v| (s, v)))
                .collect();
            for (section, v) in sections {
                match ops::snapshot_config(&pool, "control-api", section, &v, ACTOR).await {
                    Ok(true) => tracing::info!(section, "config snapshot stored in ops.config_snapshot"),
                    Ok(false) => {}
                    Err(e) => tracing::warn!(section, error = %e, "config snapshot not stored"),
                }
            }
        }
    });
}

fn unavailable() -> Response {
    (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "postgres_unavailable"}))).into_response()
}

fn db_err(e: fa_warehouse::WarehouseError) -> Response {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response()
}

#[derive(Debug, Deserialize)]
struct JobsQuery {
    kind: Option<String>,
    limit: Option<i64>,
}

async fn jobs(State(s): State<AppState>, Query(q): Query<JobsQuery>) -> Response {
    let Some(pool) = s.pg.get() else { return unavailable() };
    match ops::job_runs(pool, q.kind.as_deref(), q.limit.unwrap_or(100).clamp(1, 2_000)).await {
        Ok(runs) => Json(json!({"runs": runs})).into_response(),
        Err(e) => db_err(e),
    }
}

#[derive(Debug, Deserialize)]
struct SummaryQuery {
    hours: Option<i64>,
}

async fn summary(State(s): State<AppState>, Query(q): Query<SummaryQuery>) -> Response {
    let Some(pool) = s.pg.get() else { return unavailable() };
    let since = Utc::now() - chrono::Duration::hours(q.hours.unwrap_or(24).clamp(1, 24 * 90));
    match ops::job_summary(pool, since).await {
        Ok(rows) => Json(json!({
            "since": since,
            "counts": rows.into_iter().map(|(k, st, n)| json!({"kind": k, "status": st, "count": n})).collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(e) => db_err(e),
    }
}

#[derive(Debug, Deserialize)]
struct SnapshotsQuery {
    service: Option<String>,
    limit: Option<i64>,
    config: Option<bool>,
}

async fn snapshots(State(s): State<AppState>, Query(q): Query<SnapshotsQuery>) -> Response {
    let Some(pool) = s.pg.get() else { return unavailable() };
    let limit = q.limit.unwrap_or(50).clamp(1, 1_000);
    match ops::config_snapshots(pool, q.service.as_deref(), limit, q.config.unwrap_or(false)).await {
        Ok(rows) => Json(json!({"snapshots": rows})).into_response(),
        Err(e) => db_err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_results_map_to_rows() {
        let r = JobResult {
            job_id: "j1".into(),
            status: Status::Failed as i32,
            result: br#"{"score": 1.5}"#.to_vec(),
            output_refs: vec!["fa-artifacts/x".into()],
            worker_id: "w1".into(),
            error: "oom".into(),
            ..Default::default()
        };
        let at = Utc::now();
        let j = job_run("jobs.result.evo.eval.decoder", &r, Some(42), at);
        assert_eq!((j.kind.as_str(), j.status.as_str(), j.stream_seq), ("evo.eval.decoder", "failed", Some(42)));
        assert_eq!(j.result, Some(json!({"score": 1.5})));
        assert_eq!((j.worker_id.as_deref(), j.error.as_deref(), j.checkpoint_ref), (Some("w1"), Some("oom"), None));
        let ok = job_run("jobs.result.mcmc.refit", &JobResult { job_id: "j2".into(), ..Default::default() }, None, at);
        assert_eq!((ok.status.as_str(), ok.result), ("ok", None));
    }
}
