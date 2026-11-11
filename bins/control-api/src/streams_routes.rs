//! `GET /api/bus/streams` (spec 50 §4.3): JetStream streams, consumers and lag, straight from
//! `fa_bus::streams::list_streams`. Also provisions the spec 21 §5 streams once NATS is up and
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
        Ok(list) => (StatusCode::OK, Json(json!({ "streams": list }))),
        Err(e) => (StatusCode::BAD_GATEWAY, Json(json!({ "error": e.to_string() }))),
    }
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
}
