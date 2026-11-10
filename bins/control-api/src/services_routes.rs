//! Service supervision over the Docker Engine API (spec 50 §4.1, open question 1 → D-060).
//!
//! `POST /api/services/{name}/start|stop|restart` acts on the containers of compose service
//! `{name}` in the project `FA_COMPOSE_PROJECT` (default `financial-auto`), discovered through
//! the `com.docker.compose.*` labels, never on arbitrary container names. `control-api` itself
//! and the infrastructure services (`nats`, `postgres`, `helix`, `victoriametrics`, `garage`) are
//! refused with 409: stopping them from the API takes the API or the whole stack with it.
//! Stopping or restarting `risk-engine` halts trading (spec 40), so it needs `{"confirm": true}`
//! in the body (428 otherwise). Without a reachable Docker socket every route answers
//! `501 {"error": "no_supervisor"}` so host-run dev instances keep working; daemon errors are 502.
//! Every action that reaches the supervisor, and every refusal (409/428), is written to `audit.log`.
//!
//! `GET /api/services/{name}/container` reports the container state, and this module shadows
//! `GET /api/services` to fold `container` into each heartbeat row (plus a `supervisor` object).
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::{Value, json};

use crate::{
    config_routes::audit,
    docker::{Applied, ContainerDetail, ContainerSummary, Docker, DockerError, STOP_TIMEOUT_SECS},
    state::AppState,
};

/// Services the API refuses to start/stop/restart (409).
pub const REFUSED: [&str; 6] = ["control-api", "nats", "postgres", "helix", "victoriametrics", "garage"];
/// Services whose `stop` / `restart` need `{"confirm": true}` (428 otherwise).
pub const CONFIRM: [&str; 1] = ["risk-engine"];
pub const ACTIONS: [&str; 3] = ["start", "stop", "restart"];

#[derive(Clone)]
pub struct SvcState {
    pub app: AppState,
    pub docker: Arc<Docker>,
}

/// Returns `app` with the service routes layered in front (shadows `GET /api/services`).
pub fn install(state: AppState, app: Router) -> Router {
    router(state, Docker::from_env()).fallback_service(app)
}

pub fn router(app: AppState, docker: Docker) -> Router {
    Router::new()
        .route("/api/services", get(services))
        .route("/api/services/{name}/container", get(container))
        .route("/api/services/{name}/{action}", post(action))
        .with_state(SvcState { app, docker: Arc::new(docker) })
}

fn err(status: StatusCode, error: &str, extra: Value) -> Response {
    let mut body = json!({ "error": error });
    if let Some(obj) = extra.as_object() {
        body.as_object_mut().expect("object").extend(obj.clone());
    }
    (status, Json(body)).into_response()
}

fn docker_err(e: &DockerError) -> Response {
    match e {
        DockerError::Unreachable(m) => err(
            StatusCode::NOT_IMPLEMENTED,
            "no_supervisor",
            json!({ "detail": m, "hint": "mount the Docker socket into control-api (FA_DOCKER_SOCK); see docker/README.md" }),
        ),
        other => err(StatusCode::BAD_GATEWAY, "docker_error", json!({ "detail": other.to_string() })),
    }
}

/// Why a service may be refused.
pub fn refusal(name: &str) -> Option<&'static str> {
    if name == "control-api" {
        Some("control-api cannot supervise itself; use `docker compose restart control-api`")
    } else if REFUSED.contains(&name) {
        Some("infrastructure services are not controlled from the API; use `docker compose`")
    } else {
        None
    }
}

/// `stop`/`restart` of a confirm-gated service needs `{"confirm": true}`.
pub fn needs_confirm(name: &str, action: &str) -> bool {
    CONFIRM.contains(&name) && action != "start"
}

/// Containers of `name` in the project; `None` when the service is unknown.
async fn containers_of(docker: &Docker, name: &str) -> Result<Option<Vec<ContainerSummary>>, DockerError> {
    let mut all = docker.list_containers().await?;
    let known = all.iter().any(|c| c.service() == Some(name));
    if !known {
        return Ok(None);
    }
    all.retain(|c| c.service() == Some(name));
    all.sort_by(|a, b| a.name().cmp(b.name()));
    Ok(Some(all))
}

fn known_services(list: &[ContainerSummary]) -> Vec<String> {
    let mut v: Vec<String> = list.iter().filter_map(|c| c.service().map(str::to_owned)).collect();
    v.sort();
    v.dedup();
    v
}

async fn unknown(docker: &Docker, name: &str) -> Response {
    let known = docker.list_containers().await.map(|l| known_services(&l)).unwrap_or_default();
    err(StatusCode::NOT_FOUND, "unknown_service", json!({ "service": name, "known": known }))
}

fn summary_json(c: &ContainerSummary) -> Value {
    json!({ "name": c.name(), "id": c.id, "image": c.image, "state": c.state, "status": c.status })
}

fn detail_json(d: &ContainerDetail) -> Value {
    serde_json::to_value(d).unwrap_or(Value::Null)
}

/// `GET /api/services`: heartbeat rows with `container` folded in when the supervisor answers.
async fn services(State(s): State<SvcState>) -> Response {
    let list: Vec<_> = s.app.services.read().expect("services lock").values().cloned().collect();
    let (containers, supervisor) = match s.docker.list_containers().await {
        Ok(cs) => {
            let sup = json!({ "available": true, "project": s.docker.project(), "socket": s.docker.socket(), "services": known_services(&cs) });
            (Some(cs), sup)
        }
        Err(e) => (
            None,
            json!({ "available": false, "project": s.docker.project(), "socket": s.docker.socket(), "detail": e.to_string() }),
        ),
    };
    let rows: Vec<Value> = list
        .iter()
        .map(|svc| {
            let mut row = serde_json::to_value(svc).unwrap_or(Value::Null);
            let container = containers.as_ref().and_then(|cs| {
                let mut mine: Vec<&ContainerSummary> =
                    cs.iter().filter(|c| c.service() == Some(&svc.service)).collect();
                mine.sort_by_key(|c| c.name().to_owned());
                mine.first().map(|c| summary_json(c))
            });
            row["container"] = container.unwrap_or(Value::Null);
            row
        })
        .collect();
    Json(json!({ "services": rows, "nats_connected": s.app.nats_connected(), "supervisor": supervisor }))
        .into_response()
}

/// `GET /api/services/{name}/container`.
async fn container(State(s): State<SvcState>, Path(name): Path<String>) -> Response {
    let found = match containers_of(&s.docker, &name).await {
        Ok(Some(cs)) => cs,
        Ok(None) => return unknown(&s.docker, &name).await,
        Err(e) => return docker_err(&e),
    };
    let mut details = Vec::with_capacity(found.len());
    for c in &found {
        match s.docker.inspect(&c.id).await {
            Ok(d) => details.push(d),
            Err(e) => return docker_err(&e),
        }
    }
    let first = details.first().map(detail_json).unwrap_or(Value::Null);
    let mut body = json!({ "service": name, "replicas": details.len(), "containers": details.iter().map(detail_json).collect::<Vec<_>>() });
    if let Some(obj) = first.as_object() {
        body.as_object_mut().expect("object").extend(obj.clone());
    }
    Json(body).into_response()
}

/// `POST /api/services/{name}/start|stop|restart`.
async fn action(
    State(s): State<SvcState>,
    Path((name, action)): Path<(String, String)>,
    body: Option<Json<Value>>,
) -> Response {
    if !ACTIONS.contains(&action.as_str()) {
        return err(StatusCode::NOT_FOUND, "unknown_action", json!({ "action": action, "actions": ACTIONS }));
    }
    let subject = format!("{name}:{action}");
    let audit_action = format!("service.{action}");
    if let Some(reason) = refusal(&name) {
        audit(&s.app, &audit_action, &subject, json!({ "service": name, "refused": reason })).await;
        return err(StatusCode::CONFLICT, "refused", json!({ "service": name, "action": action, "message": reason }));
    }
    let confirmed = body.as_ref().is_some_and(|b| b["confirm"] == json!(true));
    if needs_confirm(&name, &action) && !confirmed {
        audit(&s.app, &audit_action, &subject, json!({ "service": name, "refused": "confirm_required" })).await;
        return err(
            StatusCode::PRECONDITION_REQUIRED,
            "confirm_required",
            json!({ "service": name, "action": action, "message": format!("{action} of {name} halts trading (spec 40); resend with body {{\"confirm\": true}}") }),
        );
    }
    let found = match containers_of(&s.docker, &name).await {
        Ok(Some(cs)) => cs,
        Ok(None) => return unknown(&s.docker, &name).await,
        Err(e) => return docker_err(&e),
    };
    let running = found.iter().filter(|c| c.state == "running").count();
    let already = match action.as_str() {
        "start" if running == found.len() => Some("already_running"),
        "stop" if running == 0 => Some("already_stopped"),
        _ => None,
    };
    if let Some(e) = already {
        return err(
            StatusCode::CONFLICT,
            e,
            json!({ "service": name, "action": action, "containers": found.iter().map(summary_json).collect::<Vec<_>>() }),
        );
    }
    let mut applied = Vec::with_capacity(found.len());
    let mut failure: Option<DockerError> = None;
    for c in &found {
        let r = match action.as_str() {
            "start" => s.docker.start(&c.id).await,
            "stop" => s.docker.stop(&c.id, STOP_TIMEOUT_SECS).await,
            _ => s.docker.restart(&c.id, STOP_TIMEOUT_SECS).await,
        };
        match r {
            Ok(a) => applied.push((c.name().to_owned(), a)),
            Err(e) => {
                failure = Some(e);
                break;
            }
        }
    }
    let names: Vec<&str> = found.iter().map(ContainerSummary::name).collect();
    if let Some(e) = failure {
        let audited = audit(
            &s.app,
            &audit_action,
            &subject,
            json!({ "service": name, "containers": names, "ok": false, "error": e.to_string() }),
        )
        .await;
        tracing::warn!(service = %name, action = %action, error = %e, audited, "service action failed");
        return docker_err(&e);
    }
    let mut states = Vec::with_capacity(found.len());
    for c in &found {
        states.push(s.docker.inspect(&c.id).await.map(|d| d.state).unwrap_or_else(|_| "unknown".to_owned()));
    }
    let state = states.first().cloned().unwrap_or_default();
    let noop = applied.iter().all(|(_, a)| *a == Applied::AlreadyInState);
    let audited = audit(
        &s.app,
        &audit_action,
        &subject,
        json!({ "service": name, "containers": names, "ok": true, "state": state, "confirm": confirmed, "noop": noop }),
    )
    .await;
    tracing::info!(service = %name, action = %action, %state, audited, "service action applied");
    Json(json!({
        "ok": true,
        "service": name,
        "action": action,
        "container": names.first().copied().unwrap_or_default(),
        "containers": names.iter().zip(&states).map(|(n, st)| json!({ "name": n, "state": st })).collect::<Vec<_>>(),
        "state": state,
        "noop": noop,
        "audited": audited,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use fa_core::config::PlatformConfig;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;
    use crate::docker::fake::{FakeDaemon, Route, inspect_doc, summary};

    const P: &str = "financial-auto";

    fn app_state() -> AppState {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let cfg = PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        AppState::new(cfg, format!("{root}/sources.yaml").into(), None)
    }

    fn daemon() -> FakeDaemon {
        let rows = json!([
            summary(P, "nats", "running"),
            summary(P, "control-api", "running"),
            summary(P, "broker-gateway", "running"),
            summary(P, "risk-engine", "running"),
            summary(P, "evo-worker", "exited"),
        ]);
        let bg = "broker-gateway".to_owned() + &"0".repeat(60 - "broker-gateway".len());
        let ew = "evo-worker".to_owned() + &"0".repeat(60 - "evo-worker".len());
        let re = "risk-engine".to_owned() + &"0".repeat(60 - "risk-engine".len());
        let routes: Vec<Route> = vec![
            ("GET", "/containers/json?".into(), 200, rows.to_string()),
            ("GET", format!("/containers/{bg}/json"), 200, inspect_doc(P, "broker-gateway", "running").to_string()),
            ("GET", format!("/containers/{ew}/json"), 200, inspect_doc(P, "evo-worker", "exited").to_string()),
            ("GET", format!("/containers/{re}/json"), 200, inspect_doc(P, "risk-engine", "running").to_string()),
            ("POST", format!("/containers/{bg}/restart?t=10"), 204, String::new()),
            ("POST", format!("/containers/{re}/stop?t=10"), 204, String::new()),
            ("POST", format!("/containers/{ew}/start"), 204, String::new()),
        ];
        FakeDaemon::spawn(routes, true)
    }

    async fn send(app: Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
        let mut req = Request::builder().method(method).uri(uri);
        let req = match body {
            Some(b) => {
                req = req.header("content-type", "application/json");
                req.body(Body::from(b.to_string())).unwrap()
            }
            None => req.body(Body::empty()).unwrap(),
        };
        let res = app.oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    #[test]
    fn refusal_list_and_confirm_rules() {
        for s in REFUSED {
            assert!(refusal(s).is_some(), "{s}");
        }
        assert!(refusal("control-api").unwrap().contains("itself"));
        assert!(refusal("broker-gateway").is_none());
        assert!(refusal("ui").is_none());
        assert!(needs_confirm("risk-engine", "stop"));
        assert!(needs_confirm("risk-engine", "restart"));
        assert!(!needs_confirm("risk-engine", "start"));
        assert!(!needs_confirm("broker-gateway", "stop"));
    }

    #[tokio::test]
    async fn without_a_socket_everything_is_501_except_refusals_and_confirm() {
        let app = || router(app_state(), Docker::at("/nonexistent/docker.sock", P));
        let (s, v) = send(app(), "POST", "/api/services/broker-gateway/restart", None).await;
        assert_eq!(s, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(v["error"], "no_supervisor");
        let (s, v) = send(app(), "GET", "/api/services/broker-gateway/container", None).await;
        assert_eq!((s, v["error"].as_str()), (StatusCode::NOT_IMPLEMENTED, Some("no_supervisor")));
        // refusal and confirm are decided before the daemon is asked
        let (s, v) = send(app(), "POST", "/api/services/control-api/stop", None).await;
        assert_eq!((s, v["error"].as_str()), (StatusCode::CONFLICT, Some("refused")));
        let (s, v) = send(app(), "POST", "/api/services/risk-engine/stop", None).await;
        assert_eq!((s, v["error"].as_str()), (StatusCode::PRECONDITION_REQUIRED, Some("confirm_required")));
        let (s, v) = send(app(), "POST", "/api/services/broker-gateway/kick", None).await;
        assert_eq!((s, v["error"].as_str()), (StatusCode::NOT_FOUND, Some("unknown_action")));
        // the heartbeat list still answers, with the supervisor marked unavailable
        let (s, v) = send(app(), "GET", "/api/services", None).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["supervisor"]["available"], false);
        assert_eq!(v["nats_connected"], false);
    }

    #[tokio::test]
    async fn actions_against_the_fake_daemon() {
        let d = daemon();
        let app = || router(app_state(), Docker::at(&d.socket, P));

        let (s, v) = send(app(), "POST", "/api/services/broker-gateway/restart", None).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["ok"], true);
        assert_eq!(v["container"], "financial-auto-broker-gateway-1");
        assert_eq!(v["state"], "running");
        assert_eq!(v["audited"], false, "no Postgres pool in tests");

        let (s, v) = send(app(), "POST", "/api/services/nope/start", None).await;
        assert_eq!((s, v["error"].as_str()), (StatusCode::NOT_FOUND, Some("unknown_service")));
        assert!(v["known"].as_array().unwrap().iter().any(|k| k == "broker-gateway"));

        for infra in ["nats", "postgres", "helix", "victoriametrics", "garage"] {
            let (s, v) = send(app(), "POST", &format!("/api/services/{infra}/restart"), None).await;
            assert_eq!((s, v["error"].as_str()), (StatusCode::CONFLICT, Some("refused")), "{infra}");
        }

        let (s, v) = send(app(), "POST", "/api/services/broker-gateway/start", None).await;
        assert_eq!((s, v["error"].as_str()), (StatusCode::CONFLICT, Some("already_running")));
        let (s, v) = send(app(), "POST", "/api/services/evo-worker/stop", None).await;
        assert_eq!((s, v["error"].as_str()), (StatusCode::CONFLICT, Some("already_stopped")));
        let (s, v) = send(app(), "POST", "/api/services/evo-worker/start", None).await;
        assert_eq!(s, StatusCode::OK, "{v}");

        let (s, _) = send(app(), "POST", "/api/services/risk-engine/stop", Some(json!({"confirm": false}))).await;
        assert_eq!(s, StatusCode::PRECONDITION_REQUIRED);
        let (s, v) = send(app(), "POST", "/api/services/risk-engine/stop", Some(json!({"confirm": true}))).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["action"], "stop");

        // a container the daemon refuses to act on → 502
        let (s, v) = send(app(), "POST", "/api/services/risk-engine/restart", Some(json!({"confirm": true}))).await;
        assert_eq!((s, v["error"].as_str()), (StatusCode::BAD_GATEWAY, Some("docker_error")), "{v}");

        let (s, v) = send(app(), "GET", "/api/services/broker-gateway/container", None).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["state"], "running");
        assert_eq!(v["health"], "healthy");
        assert_eq!(v["restart_count"], 2);
        assert_eq!(v["image"], "financial-auto/rust:dev");
        assert_eq!(v["replicas"], 1);

        let requests = d.requests.lock().await.clone();
        assert!(requests.iter().all(|r| r.contains("/containers/")), "{requests:?}");
        assert!(requests.iter().any(|r| r.contains("/restart?t=10")));
    }

    #[tokio::test]
    async fn services_rows_gain_container_state() {
        let d = daemon();
        let state = app_state();
        state.record_heartbeat(fa_proto::v1::Heartbeat {
            service: "broker-gateway".into(),
            instance_id: "i1".into(),
            version: "0".into(),
            ..Default::default()
        });
        state.record_heartbeat(fa_proto::v1::Heartbeat { service: "host-only".into(), ..Default::default() });
        let app = router(state, Docker::at(&d.socket, P));
        let (s, v) = send(app, "GET", "/api/services", None).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["supervisor"]["available"], true);
        assert_eq!(v["supervisor"]["project"], P);
        let rows = v["services"].as_array().unwrap();
        let bg = rows.iter().find(|r| r["service"] == "broker-gateway").unwrap();
        assert_eq!(bg["container"]["name"], "financial-auto-broker-gateway-1");
        assert_eq!(bg["container"]["state"], "running");
        let host = rows.iter().find(|r| r["service"] == "host-only").unwrap();
        assert!(host["container"].is_null());
        assert!(bg["last_seen"].is_string(), "heartbeat fields kept");
    }
}
