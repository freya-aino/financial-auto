//! The `ops` schema (spec 20 §5.4, spec 60 §6.5; migration `20261113000013`): alert history,
//! the job registry's run history and configuration snapshots. control-api is the writer.
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::{PgPool, Row};

use crate::WarehouseError;

/// One alert transition (`ops.alert` message, spec 60 §4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlertRecord {
    pub status: String,
    pub alertname: String,
    pub severity: String,
    pub summary: String,
    pub fingerprint: String,
    pub labels: Value,
    pub annotations: Value,
    pub starts_at: Option<DateTime<Utc>>,
    pub ends_at: Option<DateTime<Utc>>,
    pub generator_url: Option<String>,
    pub at: DateTime<Utc>,
    #[serde(default)]
    pub expired: bool,
}

/// Stores a transition; a repeat of the same `(fingerprint, status, at)` is a no-op (`false`).
pub async fn record_alert(pool: &PgPool, a: &AlertRecord) -> Result<bool, WarehouseError> {
    let n = sqlx::query(
        "INSERT INTO ops.alert_event (fingerprint, alertname, status, severity, summary, labels, annotations, \
         starts_at, ends_at, generator_url, expired, at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12) \
         ON CONFLICT DO NOTHING",
    )
    .bind(&a.fingerprint)
    .bind(&a.alertname)
    .bind(&a.status)
    .bind(&a.severity)
    .bind(&a.summary)
    .bind(&a.labels)
    .bind(&a.annotations)
    .bind(a.starts_at)
    .bind(a.ends_at)
    .bind(&a.generator_url)
    .bind(a.expired)
    .bind(a.at)
    .execute(pool)
    .await?
    .rows_affected();
    Ok(n > 0)
}

/// Transitions, newest first, optionally since `since` and for one `alertname`.
pub async fn alerts(
    pool: &PgPool,
    since: Option<DateTime<Utc>>,
    alertname: Option<&str>,
    limit: i64,
) -> Result<Vec<AlertRecord>, WarehouseError> {
    let rows = sqlx::query(
        "SELECT fingerprint, alertname, status, severity, summary, labels, annotations, starts_at, ends_at, \
         generator_url, expired, at FROM ops.alert_event \
         WHERE ($1::timestamptz IS NULL OR at >= $1) AND ($2::text IS NULL OR alertname = $2) \
         ORDER BY at DESC, id DESC LIMIT $3",
    )
    .bind(since)
    .bind(alertname)
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(rows
        .iter()
        .map(|r| AlertRecord {
            status: r.get("status"),
            alertname: r.get("alertname"),
            severity: r.get("severity"),
            summary: r.get("summary"),
            fingerprint: r.get("fingerprint"),
            labels: r.get("labels"),
            annotations: r.get("annotations"),
            starts_at: r.get("starts_at"),
            ends_at: r.get("ends_at"),
            generator_url: r.get("generator_url"),
            at: r.get("at"),
            expired: r.get("expired"),
        })
        .collect())
}

/// One finished job (`fa.v1.JobResult` on `jobs.result.<kind>`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobRun {
    pub job_id: String,
    pub kind: String,
    /// `ok` | `failed` | `preempted` | `budget_exceeded` | `cancelled`
    pub status: String,
    pub worker_id: Option<String>,
    pub error: Option<String>,
    pub used_gpu_ms: i64,
    pub output_refs: Vec<String>,
    pub checkpoint_ref: Option<String>,
    pub result: Option<Value>,
    /// JetStream sequence in `JOB_RESULTS` (ordering, replay position).
    pub stream_seq: Option<i64>,
    pub finished_at: DateTime<Utc>,
}

/// The `result` bytes as JSON when they parse, else their size.
pub fn result_json(bytes: &[u8]) -> Option<Value> {
    if bytes.is_empty() {
        return None;
    }
    Some(serde_json::from_slice(bytes).unwrap_or_else(|_| serde_json::json!({ "bytes": bytes.len() })))
}

/// Upserts a job's latest result (a redelivered or re-run result replaces the row only when its
/// stream position is newer). Returns whether the row changed.
pub async fn record_job_run(pool: &PgPool, j: &JobRun) -> Result<bool, WarehouseError> {
    let n = sqlx::query(
        "INSERT INTO ops.job_run (job_id, kind, status, worker_id, error, used_gpu_ms, output_refs, checkpoint_ref, \
         result, stream_seq, finished_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) \
         ON CONFLICT (job_id) DO UPDATE SET kind = excluded.kind, status = excluded.status, \
           worker_id = excluded.worker_id, error = excluded.error, used_gpu_ms = excluded.used_gpu_ms, \
           output_refs = excluded.output_refs, checkpoint_ref = excluded.checkpoint_ref, result = excluded.result, \
           stream_seq = excluded.stream_seq, finished_at = excluded.finished_at, recorded_at = now() \
         WHERE coalesce(ops.job_run.stream_seq, -1) < coalesce(excluded.stream_seq, 0)",
    )
    .bind(&j.job_id)
    .bind(&j.kind)
    .bind(&j.status)
    .bind(&j.worker_id)
    .bind(&j.error)
    .bind(j.used_gpu_ms)
    .bind(serde_json::json!(j.output_refs))
    .bind(&j.checkpoint_ref)
    .bind(&j.result)
    .bind(j.stream_seq)
    .bind(j.finished_at)
    .execute(pool)
    .await?
    .rows_affected();
    Ok(n > 0)
}

/// Job runs, newest first, optionally of one kind (prefix match: `evo` matches `evo.eval.decoder`).
pub async fn job_runs(pool: &PgPool, kind: Option<&str>, limit: i64) -> Result<Vec<JobRun>, WarehouseError> {
    let rows = sqlx::query(
        "SELECT job_id, kind, status, worker_id, error, used_gpu_ms, output_refs, checkpoint_ref, result, stream_seq, \
         finished_at FROM ops.job_run WHERE ($1::text IS NULL OR kind = $1 OR kind LIKE $1 || '.%') \
         ORDER BY finished_at DESC, job_id LIMIT $2",
    )
    .bind(kind)
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(rows
        .iter()
        .map(|r| JobRun {
            job_id: r.get("job_id"),
            kind: r.get("kind"),
            status: r.get("status"),
            worker_id: r.get("worker_id"),
            error: r.get("error"),
            used_gpu_ms: r.get("used_gpu_ms"),
            output_refs: serde_json::from_value(r.get("output_refs")).unwrap_or_default(),
            checkpoint_ref: r.get("checkpoint_ref"),
            result: r.get("result"),
            stream_seq: r.get("stream_seq"),
            finished_at: r.get("finished_at"),
        })
        .collect())
}

/// Counts per `(kind, status)` since `since` (the registry overview).
pub async fn job_summary(pool: &PgPool, since: DateTime<Utc>) -> Result<Vec<(String, String, i64)>, WarehouseError> {
    Ok(sqlx::query_as(
        "SELECT kind, status, count(*) FROM ops.job_run WHERE finished_at >= $1 GROUP BY 1, 2 ORDER BY 1, 2",
    )
    .bind(since)
    .fetch_all(pool)
    .await?)
}

/// Stores `config` for `(service, section)` when it differs from every earlier snapshot of it
/// (digest = md5 of the JSON text in Postgres). Returns whether a new snapshot was taken.
pub async fn snapshot_config(
    pool: &PgPool,
    service: &str,
    section: &str,
    config: &Value,
    actor: &str,
) -> Result<bool, WarehouseError> {
    let latest: Option<(String,)> = sqlx::query_as(
        "SELECT digest FROM ops.config_snapshot WHERE service = $1 AND section = $2 ORDER BY taken_at DESC, id DESC \
         LIMIT 1",
    )
    .bind(service)
    .bind(section)
    .fetch_optional(pool)
    .await?;
    let (digest,): (String,) = sqlx::query_as("SELECT md5($1::jsonb::text)").bind(config).fetch_one(pool).await?;
    if latest.is_some_and(|(d,)| d == digest) {
        return Ok(false);
    }
    // a change back to an earlier config is a new snapshot: drop the old row of that digest first
    let mut tx = pool.begin().await?;
    sqlx::query("DELETE FROM ops.config_snapshot WHERE service = $1 AND section = $2 AND digest = $3")
        .bind(service)
        .bind(section)
        .bind(&digest)
        .execute(&mut *tx)
        .await?;
    sqlx::query(
        "INSERT INTO ops.config_snapshot (service, section, digest, config, actor) VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(service)
    .bind(section)
    .bind(&digest)
    .bind(config)
    .bind(actor)
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(true)
}

/// Snapshot rows (without the config body unless `with_config`), newest first.
pub async fn config_snapshots(
    pool: &PgPool,
    service: Option<&str>,
    limit: i64,
    with_config: bool,
) -> Result<Vec<Value>, WarehouseError> {
    let rows = sqlx::query(
        "SELECT id, service, section, digest, actor, taken_at, config FROM ops.config_snapshot \
         WHERE ($1::text IS NULL OR service = $1) ORDER BY taken_at DESC, id DESC LIMIT $2",
    )
    .bind(service)
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(rows
        .iter()
        .map(|r| {
            let mut v = serde_json::json!({
                "id": r.get::<i64, _>("id"), "service": r.get::<String, _>("service"),
                "section": r.get::<String, _>("section"), "digest": r.get::<String, _>("digest"),
                "actor": r.get::<String, _>("actor"), "taken_at": r.get::<DateTime<Utc>, _>("taken_at"),
            });
            if with_config {
                v["config"] = r.get("config");
            }
            v
        })
        .collect())
}

/// Live tests against the compose Postgres (`DATABASE_URL`, migration `20261113000013` applied).
/// They use unique names and remove their rows.
#[cfg(test)]
mod tests {
    use super::*;

    async fn pool() -> Option<PgPool> {
        let url = std::env::var("DATABASE_URL").ok()?;
        Some(PgPool::connect(&url).await.expect("connect"))
    }

    #[tokio::test]
    async fn ops_round_trip() {
        let Some(pool) = pool().await else { return };
        let tag = format!("test-{}", uuid::Uuid::now_v7());
        let at = chrono::DateTime::from_timestamp(Utc::now().timestamp(), 0).unwrap();
        let a = AlertRecord {
            status: "firing".into(),
            alertname: tag.clone(),
            severity: "warning".into(),
            summary: "s".into(),
            fingerprint: tag.clone(),
            labels: serde_json::json!({"alertname": tag}),
            annotations: serde_json::json!({}),
            starts_at: Some(at),
            ends_at: None,
            generator_url: None,
            at,
            expired: false,
        };
        assert!(record_alert(&pool, &a).await.unwrap());
        assert!(!record_alert(&pool, &a).await.unwrap(), "idempotent");
        let r = AlertRecord {
            status: "resolved".into(),
            expired: true,
            at: at + chrono::Duration::seconds(5),
            ..a.clone()
        };
        assert!(record_alert(&pool, &r).await.unwrap());
        let got = alerts(&pool, None, Some(&tag), 10).await.unwrap();
        assert_eq!(got, vec![r, a]);

        let mut j = JobRun {
            job_id: tag.clone(),
            kind: "test.kind".into(),
            status: "failed".into(),
            worker_id: Some("w".into()),
            error: Some("boom".into()),
            used_gpu_ms: 0,
            output_refs: vec!["x".into()],
            checkpoint_ref: None,
            result: result_json(b"{\"a\":1}"),
            stream_seq: Some(10),
            finished_at: at,
        };
        assert!(record_job_run(&pool, &j).await.unwrap());
        j.status = "ok".into();
        j.stream_seq = Some(9);
        assert!(!record_job_run(&pool, &j).await.unwrap(), "an older delivery does not overwrite");
        j.stream_seq = Some(11);
        assert!(record_job_run(&pool, &j).await.unwrap());
        let runs = job_runs(&pool, Some("test"), 10).await.unwrap();
        let got = runs.iter().find(|x| x.job_id == tag).unwrap();
        assert_eq!((got.status.as_str(), got.result.clone()), ("ok", Some(serde_json::json!({"a": 1}))));
        assert_eq!(result_json(&[0xff, 0x00]), Some(serde_json::json!({"bytes": 2})));
        assert!(job_summary(&pool, at).await.unwrap().iter().any(|(k, s, n)| k == "test.kind" && s == "ok" && *n >= 1));

        let c1 = serde_json::json!({"a": 1});
        assert!(snapshot_config(&pool, &tag, "runtime", &c1, "test").await.unwrap());
        assert!(!snapshot_config(&pool, &tag, "runtime", &c1, "test").await.unwrap(), "unchanged");
        assert!(snapshot_config(&pool, &tag, "runtime", &serde_json::json!({"a": 2}), "test").await.unwrap());
        assert!(snapshot_config(&pool, &tag, "runtime", &c1, "test").await.unwrap(), "changed back");
        let snaps = config_snapshots(&pool, Some(&tag), 10, true).await.unwrap();
        assert_eq!(snaps.len(), 2);
        assert_eq!(snaps[0]["config"], c1);

        for (sql, v) in [
            ("DELETE FROM ops.alert_event WHERE alertname = $1", &tag),
            ("DELETE FROM ops.job_run WHERE job_id = $1", &tag),
            ("DELETE FROM ops.config_snapshot WHERE service = $1", &tag),
        ] {
            sqlx::query(sql).bind(v).execute(&pool).await.unwrap();
        }
    }
}
