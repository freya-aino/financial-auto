//! `GET /api/bus/streams` (spec 50 §4.3): JetStream streams, consumers and lag, straight from
//! `fa_bus::streams::list_streams`, each with its applied `max_bytes` cap (−1 = unlimited), the
//! spec 21 §5 `planned_max_bytes` from `fa_bus::streams::table()` and `cap_applied` (false when
//! `ensure_streams` kept a larger cap because the stream already held more than the plan). Also provisions the spec 21 §5 streams once NATS is up and
//! then runs the dead-letter forwarder (`fa_bus::dlq`, spec 21 §8).
use std::time::Duration;

use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::get};
use fa_bus::{async_nats::jetstream, streams};
use serde_json::json;

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/api/bus/streams", get(list))
}

/// Runs `ensure_streams` once the client is connected (control-api may start with NATS down and
/// the client retries in the background), then again after every reconnect-worthy failure.
pub fn spawn_ensure_streams(state: &AppState) {
    let Some(client) = state.nats.clone() else {
        return;
    };
    tokio::spawn(async move {
        let js = jetstream::new(client.clone());
        let mut backoff = Duration::from_secs(1);
        loop {
            if client.connection_state() == fa_bus::async_nats::connection::State::Connected {
                match streams::ensure_streams(&js).await {
                    Ok(changed) => {
                        tracing::info!(?changed, "JetStream streams ensured (spec 21 §5)");
                        // poison messages → `dlq.<stream>.<subject>` + `ops.alert` (spec 21 §8)
                        fa_bus::dlq::spawn_forwarder(client.clone());
                        return;
                    }
                    Err(e) => tracing::warn!(error = %e, "ensuring JetStream streams failed; retrying"),
                }
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(30));
        }
    });
}

async fn list(State(state): State<AppState>) -> impl IntoResponse {
    let Some(client) = state.nats.clone().filter(|_| state.nats_connected()) else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({ "error": "NATS not connected" })));
    };
    match streams::list_streams(&jetstream::new(client)).await {
        Ok(list) => {
            (StatusCode::OK, Json(json!({ "streams": with_caps(serde_json::to_value(list).unwrap_or_default()) })))
        }
        Err(e) => (StatusCode::BAD_GATEWAY, Json(json!({ "error": e.to_string() }))),
    }
}

/// Adds `planned_max_bytes` / `cap_applied` / `used_ratio` to each listed stream.
fn with_caps(mut list: serde_json::Value) -> serde_json::Value {
    let table = streams::table();
    if let Some(arr) = list.as_array_mut() {
        for s in arr.iter_mut() {
            let name = s["name"].as_str().unwrap_or_default().to_owned();
            let applied = s["max_bytes"].as_i64().unwrap_or(-1);
            let planned = table.streams.iter().find(|t| t.name == name).map(|t| t.max_bytes());
            let used = s["bytes"].as_f64().unwrap_or(0.0);
            if let Some(o) = s.as_object_mut() {
                o.insert("planned_max_bytes".into(), json!(planned));
                o.insert("cap_applied".into(), json!(planned.is_none_or(|p| p == applied)));
                o.insert("used_ratio".into(), json!((applied > 0).then(|| used / applied as f64)));
            }
        }
    }
    list
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use fa_core::config::PlatformConfig;
    use tower::ServiceExt;

    use super::*;

    #[tokio::test]
    async fn streams_need_nats() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let cfg = PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        let state = AppState::new(cfg, format!("{root}/sources.yaml").into(), None);
        let app = router().with_state(state);
        let res = app.oneshot(Request::get("/api/bus/streams").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn caps_are_annotated() {
        let md = streams::table().streams.into_iter().find(|t| t.name == "MD").unwrap();
        let v = with_caps(json!([
            { "name": "MD", "max_bytes": md.max_bytes(), "bytes": md.max_bytes() / 4 },
            { "name": "MD", "max_bytes": md.max_bytes() * 2, "bytes": 1 },
            { "name": "ADHOC", "max_bytes": -1, "bytes": 5 },
        ]));
        assert_eq!(v[0]["planned_max_bytes"], md.max_bytes());
        assert_eq!(v[0]["cap_applied"], true);
        assert!((v[0]["used_ratio"].as_f64().unwrap() - 0.25).abs() < 1e-9);
        assert_eq!(v[1]["cap_applied"], false, "a kept larger cap is not the plan");
        assert!(v[2]["planned_max_bytes"].is_null() && v[2]["used_ratio"].is_null());
        assert_eq!(v[2]["cap_applied"], true);
    }
}
