//! Evolution routes of spec 50 §4.6 that were still 501, plus `/ws/evolution`.
//!
//! | Route | Source |
//! |---|---|
//! | `GET /api/evolution/generations` | `generation.summary` messages aggregated in `live` (D-049) |
//! | `GET /api/evolution/genomes/{id}` | Helix `Genome` + `FitnessRecord`s + parents (`fa_hub::genome_detail`) and the in-memory jobs / generations naming the genome |
//! | `GET /api/evolution/genomes/{id}/outputs` | recent `DecoderOutput`s of the genome from `live` + presigned eval-report URLs (`fa-artifacts/evals/<generation>/<job>.json`, D-045) on `garage.public_endpoint` |
//! | `POST /api/evolution/runs` | alias of `POST /api/evolution/generations` (`ctl.evo.start`) |
//! | `POST /api/evolution/runs/{id}/pause\|resume\|stop` | `ctl.evo.pause\|resume\|stop` with arg `generation_id` |
//! | `WS /ws/evolution` | generation summaries (survivors per rung) and job progress from `live`, latest value per topic |
use std::time::Duration;

use axum::{
    Json, Router,
    extract::{Path, Query, State, ws::WebSocketUpgrade},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use bytes::Bytes;
use fa_blob::{ObjectKey, S3BlobStore, buckets};
use fa_bus::subjects;
use fa_hub::HubError;
use fa_proto::{
    Message,
    v1::{CtlReply, CtlRequest},
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    config_routes::audit,
    live_routes,
    state::AppState,
    ws_routes::{self, Coalescer},
};

const CTL_TIMEOUT: Duration = Duration::from_secs(5);
/// Lifetime of presigned URLs handed to the UI.
const PRESIGN_TTL: Duration = Duration::from_secs(3600);

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/evolution/genomes/{id}", get(genome))
        .route("/api/evolution/genomes/{id}/outputs", get(genome_outputs))
        .route("/api/evolution/runs", post(live_routes::start_generation))
        .route("/api/evolution/runs/{id}/{action}", post(run_action))
        .route("/ws/evolution", get(ws_evolution))
}

/// `GET /api/evolution/generations` (registered in `routes.rs` next to the POST).
pub async fn generations(State(s): State<AppState>) -> impl IntoResponse {
    Json(s.live().generations_view())
}

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({ "error": msg.into() }))).into_response()
}

/// `CtlReply` → HTTP: `not_found:` 404, `conflict:` 409, `invalid:` 400, other failures 422.
pub fn ctl_status(r: &CtlReply) -> StatusCode {
    if r.ok {
        return StatusCode::OK;
    }
    match r.message.split(':').next().unwrap_or_default() {
        "not_found" => StatusCode::NOT_FOUND,
        "conflict" => StatusCode::CONFLICT,
        "invalid" => StatusCode::BAD_REQUEST,
        _ => StatusCode::UNPROCESSABLE_ENTITY,
    }
}

async fn ctl_request(s: &AppState, subject: String, req: &CtlRequest) -> Result<CtlReply, Box<Response>> {
    let Some(client) = s.nats.as_ref().filter(|_| s.nats_connected()) else {
        return Err(Box::new(err(StatusCode::SERVICE_UNAVAILABLE, "NATS not connected")));
    };
    let fut = client.request(subject.clone(), Bytes::from(req.encode_to_vec()));
    match tokio::time::timeout(CTL_TIMEOUT, fut).await {
        Ok(Ok(msg)) => CtlReply::decode(msg.payload.as_ref())
            .map_err(|e| Box::new(err(StatusCode::BAD_GATEWAY, format!("bad reply on {subject}: {e}")))),
        Ok(Err(e)) => Err(Box::new(err(StatusCode::BAD_GATEWAY, format!("{subject}: {e} (is the service running?)")))),
        Err(_) => {
            Err(Box::new(err(StatusCode::GATEWAY_TIMEOUT, format!("{subject}: no reply within {CTL_TIMEOUT:?}"))))
        }
    }
}

/// Status a successful `pause | resume | stop` leaves the generation in.
pub fn status_after(action: &str) -> Option<&'static str> {
    match action {
        "pause" => Some("paused"),
        "resume" => Some("running"),
        "stop" => Some("stopped"),
        _ => None,
    }
}

async fn run_action(State(s): State<AppState>, Path((id, action)): Path<(String, String)>) -> Response {
    let Some(status) = status_after(&action) else {
        return err(StatusCode::NOT_FOUND, format!("unknown action {action:?}; use pause|resume|stop"));
    };
    let req =
        CtlRequest { verb: action.clone(), args: [("generation_id".to_owned(), id.clone())].into_iter().collect() };
    let reply = match ctl_request(&s, subjects::ctl("evo", &action), &req).await {
        Ok(r) => r,
        Err(res) => return *res,
    };
    let http = ctl_status(&reply);
    if !reply.ok {
        return err(http, reply.message);
    }
    s.live_mut().set_generation_status(&id, status);
    let audited = audit(&s, &format!("evolution.run.{action}"), &id, json!({ "message": reply.message })).await;
    Json(json!({ "ok": true, "generation_id": id, "action": action, "status": status, "message": reply.message, "audited": audited }))
        .into_response()
}

// ------------------------------------------------------------- genomes

fn hub_json(e: &HubError) -> Value {
    if e.is_unreachable() {
        json!({ "error": "helix_unavailable", "detail": e.to_string() })
    } else {
        json!({ "error": "helix_error", "detail": e.to_string() })
    }
}

async fn genome(State(s): State<AppState>, Path(id): Path<String>) -> Response {
    let (jobs, generations) = {
        let live = s.live();
        let jobs: Vec<Value> =
            live.jobs_for_genome(&id).iter().map(|j| serde_json::to_value(j).unwrap_or_default()).collect();
        let gens: Vec<Value> = live.generations_view()["generations"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|g| {
                let gid = g["id"].as_str()?;
                let gv = live.generation(gid)?;
                let rank = gv
                    .pareto_front
                    .iter()
                    .find(|p| p["genome"].as_str() == Some(id.as_str()))
                    .map(|p| p["rank"].clone());
                let survivor = gv.survivors.contains(&id);
                (rank.is_some() || survivor).then(|| {
                    json!({ "generation_id": gid, "generation": gv.generation, "rung": gv.rung, "status": gv.status,
                            "pareto_rank": rank, "survivor": survivor })
                })
            })
            .collect();
        (jobs, gens)
    };
    let latest_objectives = jobs.iter().find(|j| j["objectives"].as_object().is_some_and(|o| !o.is_empty())).map(|j| {
        json!({ "job_id": j["job_id"], "status": j["status"], "objectives": j["objectives"], "score": j["score"], "updated_at": j["updated_at"] })
    });
    let (helix, detail) = match s.hub.genome_detail(&id).await {
        Ok(Some(d)) => (Value::Null, Some(d)),
        Ok(None) => (Value::Null, None),
        Err(e) => (hub_json(&e), None),
    };
    if detail.is_none() && jobs.is_empty() && generations.is_empty() {
        return if helix.is_null() {
            (StatusCode::NOT_FOUND, Json(json!({ "error": "not_found", "label": "Genome", "id": id }))).into_response()
        } else {
            let status = if helix["error"] == "helix_unavailable" {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::BAD_GATEWAY
            };
            (status, Json(helix)).into_response()
        };
    }
    let fitness: Vec<Value> = detail
        .as_ref()
        .map(|d| {
            d.fitness
                .iter()
                .map(|f| {
                    let objectives: Value = serde_json::from_str(&f.objectives).unwrap_or(Value::String(f.objectives.clone()));
                    json!({ "$id": f.id, "generation": f.generation, "pareto_rank": f.pareto_rank, "objectives": objectives })
                })
                .collect()
        })
        .unwrap_or_default();
    let params: Option<Value> = detail
        .as_ref()
        .map(|d| serde_json::from_str(&d.genome.params).unwrap_or(Value::String(d.genome.params.clone())));
    Json(json!({
        "genome_id": id,
        "genome": detail.as_ref().map(|d| &d.genome),
        "params": params,
        "parents": detail.as_ref().map(|d| &d.parents),
        "fitness": fitness,
        "latest_objectives": latest_objectives,
        "jobs": jobs,
        "generations": generations,
        "helix": helix,
    }))
    .into_response()
}

#[derive(Debug, Deserialize)]
pub struct OutputsQuery {
    pub limit: Option<usize>,
}

/// Garage client on the host-reachable endpoint, or why there is none.
fn presigner(s: &AppState) -> Result<S3BlobStore, String> {
    let secrets = fa_core::Secrets::load().map_err(|e| e.to_string())?;
    S3BlobStore::new(
        &s.cfg.garage.public_endpoint,
        &s.cfg.garage.region,
        &secrets.garage.access_key_id,
        &secrets.garage.secret_access_key,
    )
    .map_err(|e| e.to_string())
}

/// `<bucket>/<key>` → `ObjectKey` for the known buckets.
pub fn object_key(reference: &str) -> Option<ObjectKey> {
    let (bucket, key) = reference.split_once('/')?;
    let bucket = [buckets::RAW, buckets::MEDIA, buckets::MODELS, buckets::CKPT, buckets::ARTIFACTS]
        .into_iter()
        .find(|b| *b == bucket)?;
    (!key.is_empty()).then(|| ObjectKey { bucket, key: key.to_owned() })
}

async fn genome_outputs(State(s): State<AppState>, Path(id): Path<String>, Query(q): Query<OutputsQuery>) -> Response {
    let limit = q.limit.unwrap_or(50).clamp(1, 200);
    let (outputs, jobs) = {
        let live = s.live();
        let outputs: Vec<Value> = live
            .decoder_outputs_for_genome(&id, limit)
            .iter()
            .map(|o| serde_json::to_value(o).unwrap_or_default())
            .collect();
        let jobs: Vec<(String, String, String)> = live
            .jobs_for_genome(&id)
            .iter()
            .map(|j| (j.job_id.clone(), j.parent_job_id.clone(), j.status.clone()))
            .collect();
        (outputs, jobs)
    };
    let store = presigner(&s);
    let presign = |key: &ObjectKey| -> Option<String> {
        store.as_ref().ok().and_then(|st| st.presigned_get(key, PRESIGN_TTL).ok()).map(|u| u.to_string())
    };
    let eval_reports: Vec<Value> = jobs
        .iter()
        .map(|(job_id, generation_id, status)| {
            let key = fa_blob::eval_report(generation_id, job_id);
            json!({
                "job_id": job_id, "generation_id": generation_id, "status": status,
                "key": format!("{}/{}", key.bucket, key.key),
                "url": if status == "ok" { presign(&key) } else { None },
            })
        })
        .collect();
    let outputs: Vec<Value> = outputs
        .into_iter()
        .map(|mut o| {
            let urls: Vec<Value> = o["output_refs"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .iter()
                .map(|r| {
                    let r = r.as_str().unwrap_or_default();
                    json!({ "ref": r, "url": object_key(r).and_then(|k| presign(&k)) })
                })
                .collect();
            o["refs"] = Value::Array(urls);
            o
        })
        .collect();
    Json(json!({
        "genome_id": id,
        "outputs": outputs,
        "eval_reports": eval_reports,
        "presign": {
            "configured": store.is_ok(),
            "endpoint": s.cfg.garage.public_endpoint,
            "ttl_secs": PRESIGN_TTL.as_secs(),
            "error": store.as_ref().err(),
        },
    }))
    .into_response()
}

// -------------------------------------------------------- ws/evolution

async fn ws_evolution(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| async move {
        let (initial, events, hz) = {
            let live = state.live();
            (live.evolution_frames(), live.subscribe(), live.runtime().ws.latent_hz)
        };
        let source = ws_routes::spawn_live_source(events, "evolution", |_| true);
        ws_routes::serve(
            socket,
            "evolution",
            initial,
            source,
            Coalescer::latest_value(),
            ws_routes::flush_interval(hz),
        )
        .await;
    })
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    fn state() -> AppState {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let cfg = fa_core::config::PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        let state = AppState::new(cfg, format!("{root}/sources.yaml").into(), None);
        AppState { hub: fa_hub::HubClient::new("http://127.0.0.1:1"), ..state }
    }

    async fn send(app: Router, req: Request<Body>) -> (StatusCode, Value) {
        let res = app.oneshot(req).await.unwrap();
        let status = res.status();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
    }

    #[test]
    fn ctl_reply_prefixes_and_actions() {
        let r = |ok, m: &str| CtlReply { ok, message: m.into() };
        assert_eq!(ctl_status(&r(true, "paused")), StatusCode::OK);
        assert_eq!(ctl_status(&r(false, "not_found: generation x")), StatusCode::NOT_FOUND);
        assert_eq!(ctl_status(&r(false, "conflict: already paused")), StatusCode::CONFLICT);
        assert_eq!(ctl_status(&r(false, "invalid: generation_id required")), StatusCode::BAD_REQUEST);
        assert_eq!(ctl_status(&r(false, "boom")), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(status_after("pause"), Some("paused"));
        assert_eq!(status_after("stop"), Some("stopped"));
        assert_eq!(status_after("dance"), None);
        assert_eq!(object_key("fa-artifacts/evals/g/j.json").unwrap().key, "evals/g/j.json");
        assert!(object_key("inst:abc").is_none());
        assert!(object_key("fa-artifacts/").is_none());
        assert!(object_key("other-bucket/x").is_none());
    }

    #[tokio::test]
    async fn routes_without_nats_or_helix() {
        let s = state();
        let app = router().with_state(s.clone());
        let (st, v) = send(app.clone(), Request::get("/api/evolution/genomes/g1").body(Body::empty()).unwrap()).await;
        assert_eq!(st, StatusCode::SERVICE_UNAVAILABLE, "{v}");
        assert_eq!(v["error"], "helix_unavailable");
        // a genome known only from the bus still answers 200 with the helix error attached
        {
            use fa_proto::v1::Job;
            let job = Job {
                job_id: "j1".into(),
                kind: "evo.eval.decoder".into(),
                genome_id: "g1".into(),
                parent_job_id: "G".into(),
                ..Default::default()
            };
            s.live_mut().ingest("jobs.evo.eval.decoder", &job.encode_to_vec());
        }
        let (st, v) = send(app.clone(), Request::get("/api/evolution/genomes/g1").body(Body::empty()).unwrap()).await;
        assert_eq!(st, StatusCode::OK, "{v}");
        assert_eq!(v["jobs"][0]["job_id"], "j1");
        assert_eq!(v["helix"]["error"], "helix_unavailable");
        let (st, v) =
            send(app.clone(), Request::get("/api/evolution/genomes/g1/outputs").body(Body::empty()).unwrap()).await;
        assert_eq!(st, StatusCode::OK, "{v}");
        assert_eq!(v["eval_reports"][0]["key"], "fa-artifacts/evals/G/j1.json");
        assert_eq!(v["eval_reports"][0]["url"], Value::Null, "job not ok yet");
        assert!(v["presign"]["endpoint"].as_str().unwrap().starts_with("http://"));
        let (st, _) =
            send(app.clone(), Request::post("/api/evolution/runs/G/pause").body(Body::empty()).unwrap()).await;
        assert_eq!(st, StatusCode::SERVICE_UNAVAILABLE);
        let (st, _) =
            send(app.clone(), Request::post("/api/evolution/runs/G/explode").body(Body::empty()).unwrap()).await;
        assert_eq!(st, StatusCode::NOT_FOUND);
        let req = Request::post("/api/evolution/runs")
            .header("content-type", "application/json")
            .body(Body::from(json!({"kind": "", "genomes": []}).to_string()))
            .unwrap();
        let (st, _) = send(app, req).await;
        assert_eq!(st, StatusCode::BAD_REQUEST);
    }
}
