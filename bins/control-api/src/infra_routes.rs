//! Infra observability (spec 51 §6 round 9): real usage per store instead of TCP badges, the
//! latest maintenance run per job and the backup / WAL-archive / restore-drill status from the
//! series the backup containers push to VictoriaMetrics.
//!
//! | Route | Answer |
//! |---|---|
//! | `GET /api/infra/stores` | per-store usage (postgres size + connections, VictoriaMetrics bytes/series vs `FA_VM_DISK_BUDGET_GIB`, Garage buckets, Qdrant collections, Helix node/edge counts), each probed concurrently with a 5 s timeout; a down store reports `{"error": …}` |
//! | `GET /api/infra/maintenance` | latest `audit.log` row per maintenance action (`ops.vm_retention`, `ops.genome_gc`, `ops.media_downsample`, `ops.accounting_close`) |
//! | `GET /api/infra/backup` | current `fa_backup_*` / `fa_wal_archive_*` / `fa_restore_drill_*` values from VictoriaMetrics, shaped as backup / WAL / drill cards with ages |
use std::{collections::BTreeMap, time::Duration};

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use fa_blob::{BlobStore, S3BlobStore, buckets};
use fa_hub::{Batch, EdgeRef, NodeRef, Op};
use serde_json::{Value, json};

use crate::state::AppState;

/// Per-store probe budget: a down store answers `{"error": …}` instead of stalling the page.
const STORE_TIMEOUT: Duration = Duration::from_secs(5);
/// Lookback for the backup series (pushed daily / weekly, far apart from VM's staleness window).
const BACKUP_LOOKBACK: &str = "14d";
/// The maintenance jobs of `maintenance.rs` + the daily accounting close, in display order.
pub const MAINT_ACTIONS: [&str; 4] =
    ["ops.vm_retention", "ops.genome_gc", "ops.media_downsample", "ops.accounting_close"];

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/infra/stores", get(stores))
        .route("/api/infra/maintenance", get(maintenance))
        .route("/api/infra/backup", get(backup))
}

fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(json!({ "error": msg.into() }))).into_response()
}

/// `FA_VM_DISK_BUDGET_GIB` (default 50, spec 20 §8) — the same budget `maintenance.rs` tightens
/// the retention policy against.
pub fn vm_disk_budget_gib() -> f64 {
    std::env::var("FA_VM_DISK_BUDGET_GIB").ok().and_then(|v| v.parse().ok()).unwrap_or(50.0)
}

/// Caps a probe at [`STORE_TIMEOUT`] and turns any failure into `{"error": …}` so one down
/// store never fails the whole response.
async fn guard(fut: impl Future<Output = Result<Value, String>>) -> Value {
    match tokio::time::timeout(STORE_TIMEOUT, fut).await {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => json!({ "error": e }),
        Err(_) => json!({ "error": format!("no answer within {} s", STORE_TIMEOUT.as_secs()) }),
    }
}

async fn stores(State(s): State<AppState>) -> Response {
    let (postgres, victoriametrics, garage, qdrant, helix) = tokio::join!(
        guard(pg_usage(&s)),
        guard(vm_usage(&s)),
        guard(garage_usage(&s)),
        guard(qdrant_usage(&s)),
        guard(helix_usage(&s)),
    );
    Json(json!({
        "stores": {
            "postgres": postgres,
            "victoriametrics": victoriametrics,
            "garage": garage,
            "qdrant": qdrant,
            "helix": helix,
        },
        "vm_disk_budget_gib": vm_disk_budget_gib(),
    }))
    .into_response()
}

/// Database size and connection usage from the state's pool.
async fn pg_usage(s: &AppState) -> Result<Value, String> {
    let pool = s.pg.get().ok_or("postgres not attached")?;
    let (size, connections, max_connections): (i64, i64, i32) = sqlx::query_as(
        "SELECT pg_database_size(current_database()), \
                (SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()), \
                current_setting('max_connections')::int",
    )
    .fetch_one(pool)
    .await
    .map_err(|e| e.to_string())?;
    Ok(json!({ "size_bytes": size, "connections": connections, "max_connections": max_connections }))
}

/// Disk bytes and active series from VM's own `/metrics` (the `fa_tsdb` helpers `maintenance.rs`
/// uses) plus the total series count (`/api/v1/series/count`), against the disk budget.
async fn vm_usage(s: &AppState) -> Result<Value, String> {
    let vm = fa_tsdb::Tsdb::new(s.cfg.victoriametrics.url.clone());
    let (data, active) = tokio::join!(vm.data_size_bytes(), vm.active_series());
    let data = data.map_err(|e| e.to_string())?;
    let active = active.map_err(|e| e.to_string())?;
    let series = series_count(&s.cfg.victoriametrics.url).await?;
    let budget_bytes = vm_disk_budget_gib() * (1u64 << 30) as f64;
    Ok(json!({
        "data_size_bytes": data,
        "active_series": active,
        "series": series,
        "disk_budget_gib": vm_disk_budget_gib(),
        "used_ratio": data as f64 / budget_bytes,
    }))
}

/// Total series count (VictoriaMetrics `/api/v1/series/count`, `{"data": [n]}`).
async fn series_count(base: &str) -> Result<u64, String> {
    let url = format!("{}/api/v1/series/count", base.trim_end_matches('/'));
    let v: Value = reqwest::Client::new()
        .get(&url)
        .timeout(STORE_TIMEOUT)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;
    v["data"][0].as_u64().ok_or_else(|| format!("series/count: unexpected answer {v}"))
}

/// Object count and bytes per configured bucket via the S3 API (the credentials the repo has;
/// there is no Garage admin token). A missing bucket reports its own error.
async fn garage_usage(s: &AppState) -> Result<Value, String> {
    let secrets = fa_core::Secrets::load().map_err(|e| format!("garage secrets: {e}"))?;
    let store = S3BlobStore::from_config(&s.cfg, &secrets).map_err(|e| format!("garage: {e}"))?;
    const BUCKETS: [&str; 5] = [buckets::RAW, buckets::MEDIA, buckets::MODELS, buckets::CKPT, buckets::ARTIFACTS];
    let listings = futures::future::join_all(BUCKETS.iter().map(|b| {
        let store = store.clone();
        async move { (*b, store.list(b, "").await) }
    }))
    .await;
    let rows: Vec<Value> = listings
        .into_iter()
        .map(|(bucket, r)| match r {
            Ok(objs) => json!({
                "bucket": bucket,
                "objects": objs.len(),
                "bytes": objs.iter().map(|o| o.size).sum::<u64>(),
            }),
            Err(e) => json!({ "bucket": bucket, "error": e.to_string() }),
        })
        .collect();
    Ok(json!({ "buckets": rows }))
}

/// `GET /collections` then per-collection points from the configured Qdrant.
async fn qdrant_usage(s: &AppState) -> Result<Value, String> {
    let base = s.cfg.qdrant.url.trim_end_matches('/').to_owned();
    let client = reqwest::Client::builder().timeout(STORE_TIMEOUT).build().map_err(|e| e.to_string())?;
    let v: Value = client
        .get(format!("{base}/collections"))
        .send()
        .await
        .map_err(|e| format!("qdrant unreachable: {e}"))?
        .json()
        .await
        .map_err(|e| e.to_string())?;
    let names: Vec<String> = v["result"]["collections"]
        .as_array()
        .ok_or_else(|| format!("collections: unexpected answer {v}"))?
        .iter()
        .filter_map(|c| c["name"].as_str().map(str::to_owned))
        .collect();
    let mut collections = Vec::new();
    for name in names {
        let info: Value = match client.get(format!("{base}/collections/{name}")).send().await.map_err(|e| e.to_string())
        {
            Ok(r) => r.json().await.unwrap_or_default(),
            Err(e) => {
                collections.push(json!({ "name": name, "error": e }));
                continue;
            }
        };
        collections.push(json!({
            "name": name,
            "points": info["result"]["points_count"],
            "status": info["result"]["status"],
        }));
    }
    Ok(json!({ "collections": collections }))
}

/// Node and edge counts in one Helix read batch (metadata counts, measured ~50 ms live).
async fn helix_usage(s: &AppState) -> Result<Value, String> {
    let batch =
        Batch::read().ret("nodes", Op::nodes(NodeRef::All).count()).ret("edges", Op::edges(EdgeRef::All).count());
    let mut resp = s.hub.batch(&batch).await.map_err(|e| e.to_string())?;
    let nodes: u64 = resp.take("nodes").map_err(|e| e.to_string())?;
    let edges: u64 = resp.take("edges").map_err(|e| e.to_string())?;
    Ok(json!({ "nodes": nodes, "edges": edges }))
}

/// One `GET /api/infra/maintenance` row: the action with its latest audit row, or nulls when
/// the job never ran. The `maintenance.rs` payload is `{ok, report}` or `{ok: false, error}`.
pub fn maintenance_row(action: &str, latest: Option<(chrono::DateTime<chrono::Utc>, Value)>) -> Value {
    match latest {
        None => json!({ "action": action, "at": null, "ok": null, "report": null, "error": null }),
        Some((at, payload)) => json!({
            "action": action,
            "at": at,
            "ok": payload["ok"].as_bool(),
            "report": payload.get("report").cloned().unwrap_or(Value::Null),
            "error": payload.get("error").cloned().unwrap_or(Value::Null),
        }),
    }
}

async fn maintenance(State(s): State<AppState>) -> Response {
    let Some(pool) = s.pg.get() else {
        return err(StatusCode::SERVICE_UNAVAILABLE, "postgres not ready");
    };
    let rows: Vec<(String, chrono::DateTime<chrono::Utc>, Value)> = match sqlx::query_as(
        "SELECT DISTINCT ON (action) action, at, payload FROM audit.log \
         WHERE action = ANY($1) ORDER BY action, at DESC, id DESC",
    )
    .bind(&MAINT_ACTIONS[..])
    .fetch_all(pool)
    .await
    {
        Ok(r) => r,
        Err(e) => return err(StatusCode::BAD_GATEWAY, e.to_string()),
    };
    let mut by_action: BTreeMap<String, (chrono::DateTime<chrono::Utc>, Value)> =
        rows.into_iter().map(|(action, at, payload)| (action, (at, payload))).collect();
    let jobs: Vec<Value> = MAINT_ACTIONS.iter().map(|a| maintenance_row(a, by_action.remove(*a))).collect();
    Json(json!({ "jobs": jobs })).into_response()
}

/// Latest value per backup metric name over [`BACKUP_LOOKBACK`]. Series of the same name are
/// folded conservatively: timestamps keep the newest, `*_status` the worst, counters the sum.
pub fn fold_metrics(result: &[Value]) -> BTreeMap<String, f64> {
    let mut out: BTreeMap<String, f64> = BTreeMap::new();
    for series in result {
        let Some(name) = series["metric"]["__name__"].as_str() else { continue };
        let Some(v) = series["value"][1].as_str().and_then(|s| s.parse::<f64>().ok()) else { continue };
        out.entry(name.to_owned())
            .and_modify(|cur| {
                if name.ends_with("_status") || name.ends_with("_on") || name.ends_with("_up") {
                    *cur = cur.min(v);
                } else if name.ends_with("_timestamp_seconds") {
                    *cur = cur.max(v);
                } else {
                    *cur += v;
                }
            })
            .or_insert(v);
    }
    out
}

/// Shapes the folded metrics into the UI's backup / WAL / drill cards (ages from `now_s`).
pub fn shape_backup(m: &BTreeMap<String, f64>, now_s: f64) -> Value {
    let get = |k: &str| m.get(k).copied();
    let age = |k: &str| get(k).filter(|t| *t > 0.0).map(|t| (now_s - t).max(0.0));
    json!({
        "backup": {
            "last_success_ts": get("fa_backup_last_success_timestamp_seconds"),
            "age_s": age("fa_backup_last_success_timestamp_seconds"),
            "last_run_ts": get("fa_backup_last_run_timestamp_seconds"),
            "ok": get("fa_backup_last_status").map(|v| v >= 1.0),
            "last_bytes": get("fa_backup_last_bytes"),
            "last_duration_s": get("fa_backup_last_duration_seconds"),
            "kept": get("fa_backup_scheduled_kept"),
        },
        "wal": {
            "up": get("fa_wal_archive_up").map(|v| v >= 1.0),
            "mode_on": get("fa_wal_archive_mode_on").map(|v| v >= 1.0),
            "files": get("fa_wal_archive_files"),
            "bytes": get("fa_wal_archive_bytes"),
            "archived_total": get("fa_wal_archive_archived_total"),
            "failed_total": get("fa_wal_archive_failed_total"),
            "last_archived_ts": get("fa_wal_archive_last_archived_timestamp_seconds"),
            "last_archived_age_s": age("fa_wal_archive_last_archived_timestamp_seconds"),
            "last_failed_ts": get("fa_wal_archive_last_failed_timestamp_seconds"),
        },
        "drill": {
            "ok": get("fa_restore_drill_last_status").map(|v| v >= 1.0),
            "last_success_ts": get("fa_restore_drill_last_success_timestamp_seconds"),
            "age_s": age("fa_restore_drill_last_success_timestamp_seconds"),
        },
    })
}

async fn backup(State(s): State<AppState>) -> Response {
    let base = s.cfg.victoriametrics.url.trim_end_matches('/');
    let query = format!(
        "last_over_time({{__name__=~\"fa_backup_.*|fa_wal_archive_.*|fa_restore_drill_.*\"}}[{BACKUP_LOOKBACK}])"
    );
    let client = match reqwest::Client::builder().timeout(STORE_TIMEOUT).build() {
        Ok(c) => c,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let resp = match client.get(format!("{base}/api/v1/query")).query(&[("query", query.as_str())]).send().await {
        Ok(r) => r,
        Err(e) => return err(StatusCode::SERVICE_UNAVAILABLE, format!("victoriametrics unreachable: {e}")),
    };
    let v: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_GATEWAY, format!("victoriametrics: {e}")),
    };
    if v["status"] != "success" {
        return err(StatusCode::BAD_GATEWAY, format!("victoriametrics: {v}"));
    }
    let folded = fold_metrics(v["data"]["result"].as_array().map(Vec::as_slice).unwrap_or_default());
    let now_s = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;
    let mut out = shape_backup(&folded, now_s);
    out["metrics"] = json!(folded);
    Json(out).into_response()
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use fa_core::config::PlatformConfig;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    /// Unreachable stores, no Postgres: nothing outside the test can change.
    fn app() -> Router {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let mut cfg = PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        cfg.postgres.url = String::new();
        cfg.helix.url = "http://127.0.0.1:9".into();
        cfg.qdrant.url = "http://127.0.0.1:9".into();
        cfg.victoriametrics.url = "http://127.0.0.1:9".into();
        cfg.garage.endpoint = "http://127.0.0.1:9".into();
        let state = AppState::new(cfg, format!("{root}/sources.yaml").into(), None);
        router().with_state(state)
    }

    async fn get_json(uri: &str) -> (StatusCode, Value) {
        let res = app().oneshot(Request::get(uri).body(Body::empty()).unwrap()).await.unwrap();
        let status = res.status();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&body).unwrap())
    }

    /// Spec 51 round 9: a down store reports `{"error": …}`, never fails the whole response.
    #[tokio::test]
    async fn stores_isolate_per_store_errors() {
        let (status, v) = get_json("/api/infra/stores").await;
        assert_eq!(status, StatusCode::OK);
        for store in ["postgres", "victoriametrics", "garage", "qdrant", "helix"] {
            assert!(v["stores"][store]["error"].is_string(), "{store}: {}", v["stores"][store]);
        }
        assert_eq!(v["stores"]["postgres"]["error"], "postgres not attached");
        assert_eq!(v["vm_disk_budget_gib"], vm_disk_budget_gib());
    }

    #[tokio::test]
    async fn maintenance_needs_postgres_and_backup_needs_vm() {
        let (status, _) = get_json("/api/infra/maintenance").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        let (status, v) = get_json("/api/infra/backup").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(v["error"].as_str().unwrap().contains("victoriametrics"));
    }

    #[test]
    fn maintenance_rows_cover_never_ran_ok_and_failed() {
        let never = maintenance_row("ops.genome_gc", None);
        assert_eq!(never, json!({ "action": "ops.genome_gc", "at": null, "ok": null, "report": null, "error": null }));
        let at = chrono::Utc::now();
        let ok = maintenance_row("ops.vm_retention", Some((at, json!({ "ok": true, "report": { "deleted": 3 } }))));
        assert_eq!(ok["ok"], true);
        assert_eq!(ok["report"]["deleted"], 3);
        let failed = maintenance_row("ops.genome_gc", Some((at, json!({ "ok": false, "error": "garage down" }))));
        assert_eq!(failed["ok"], false);
        assert_eq!(failed["error"], "garage down");
        assert!(MAINT_ACTIONS.contains(&"ops.accounting_close"));
    }

    #[test]
    fn backup_metrics_fold_and_shape() {
        let result = vec![
            json!({ "metric": { "__name__": "fa_backup_last_success_timestamp_seconds", "kind": "scheduled" }, "value": [0, "1000"] }),
            json!({ "metric": { "__name__": "fa_backup_last_success_timestamp_seconds", "kind": "manual" }, "value": [0, "900"] }),
            json!({ "metric": { "__name__": "fa_backup_last_status", "kind": "scheduled" }, "value": [0, "1"] }),
            json!({ "metric": { "__name__": "fa_backup_last_status", "kind": "manual" }, "value": [0, "0"] }),
            json!({ "metric": { "__name__": "fa_wal_archive_bytes" }, "value": [0, "2048"] }),
            json!({ "metric": { "__name__": "fa_wal_archive_failed_total", "a": "x" }, "value": [0, "1"] }),
            json!({ "metric": { "__name__": "fa_wal_archive_failed_total", "a": "y" }, "value": [0, "2"] }),
            json!({ "metric": { "__name__": "bad" } }),
        ];
        let m = fold_metrics(&result);
        assert_eq!(m["fa_backup_last_success_timestamp_seconds"], 1000.0, "newest success wins");
        assert_eq!(m["fa_backup_last_status"], 0.0, "worst status wins");
        assert_eq!(m["fa_wal_archive_failed_total"], 3.0, "counters sum");
        assert!(!m.contains_key("bad"));

        let v = shape_backup(&m, 1300.0);
        assert_eq!(v["backup"]["age_s"], 300.0);
        assert_eq!(v["backup"]["ok"], false);
        assert_eq!(v["wal"]["bytes"], 2048.0);
        assert_eq!(v["drill"]["ok"], Value::Null, "no drill series yet");
        assert_eq!(v["drill"]["age_s"], Value::Null);
    }
}
