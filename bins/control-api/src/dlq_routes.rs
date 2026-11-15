//! DLQ browser (spec 51 §6 round 9) over the `DLQ` JetStream stream `fa_bus::dlq` forwards
//! poison messages into (`dlq.<STREAM>.<original subject>`, spec 21 §8).
//!
//! | Route | Answer |
//! |---|---|
//! | `GET /api/dlq?limit=` | recent dead letters, newest first: stream seq, original stream/subject, reason, age, size, payload preview (first 256 bytes, lossy flag) |
//! | `POST /api/dlq/replay` | `{"seq": n}` → republish that payload to its original subject and delete it from the DLQ stream; audited `dlq.replay` |
//! | `POST /api/dlq/purge` | `{"confirm": true, "subject"?}` → purge the DLQ stream (optionally filtered); 428 without confirm; audited `dlq.purge` |
use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use fa_bus::{
    async_nats::jetstream::{self, message::StreamMessage},
    dlq::DLQ_STREAM,
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::state::AppState;

/// Payload preview bytes per row.
pub const PREVIEW_BYTES: usize = 256;
/// Deleted sequences are skipped one by one, so a sparse stream is capped at this many lookups.
const SCAN_CAP: usize = 2_000;
const MAX_LIMIT: usize = 500;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/dlq", get(list))
        .route("/api/dlq/replay", post(replay))
        .route("/api/dlq/purge", post(purge))
}

fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(json!({ "error": msg.into() }))).into_response()
}

fn jetstream(s: &AppState) -> Option<jetstream::Context> {
    s.nats.clone().filter(|_| s.nats_connected()).map(jetstream::new)
}

fn header<'m>(m: &'m StreamMessage, name: &str) -> Option<&'m str> {
    m.headers.get(name).map(|v| v.as_str())
}

/// The subject a dead letter came from: the forwarder's `Fa-Dlq-Subject` header, else the
/// `dlq.<STREAM>.` prefix stripped from the DLQ subject.
pub fn original_subject(dlq_subject: &str, header: Option<&str>) -> String {
    match header {
        Some(h) if !h.is_empty() => h.to_owned(),
        _ => dlq_subject.splitn(3, '.').nth(2).unwrap_or(dlq_subject).to_owned(),
    }
}

/// First [`PREVIEW_BYTES`] of a payload as text: `(preview, lossy, truncated)`. `lossy` flags a
/// preview whose bytes were not valid UTF-8 (replacement characters stand in).
pub fn preview(payload: &[u8]) -> (String, bool, bool) {
    let cut = &payload[..payload.len().min(PREVIEW_BYTES)];
    (String::from_utf8_lossy(cut).into_owned(), std::str::from_utf8(cut).is_err(), payload.len() > cut.len())
}

/// One `GET /api/dlq` row from a raw DLQ message.
pub fn row(m: &StreamMessage, now_s: i64) -> Value {
    let (preview, lossy, truncated) = preview(&m.payload);
    let at_s = (m.time.unix_timestamp_nanos() / 1_000_000_000) as i64;
    json!({
        "seq": m.sequence,
        "dlq_subject": m.subject.as_str(),
        "original_stream": header(m, "Fa-Dlq-Stream"),
        "original_subject": original_subject(m.subject.as_str(), header(m, "Fa-Dlq-Subject")),
        "original_seq": header(m, "Fa-Dlq-Seq"),
        "consumer": header(m, "Fa-Dlq-Consumer"),
        "reason": header(m, "Fa-Dlq-Reason"),
        "deliveries": header(m, "Fa-Dlq-Deliveries"),
        "at": at_s,
        "age_s": (now_s - at_s).max(0),
        "size": m.payload.len(),
        "preview": preview,
        "lossy": lossy,
        "truncated": truncated,
    })
}

#[derive(Debug, Default, Deserialize)]
struct ListQuery {
    limit: Option<usize>,
}

async fn list(State(s): State<AppState>, Query(q): Query<ListQuery>) -> Response {
    let Some(js) = jetstream(&s) else {
        return err(StatusCode::SERVICE_UNAVAILABLE, "NATS not connected");
    };
    let limit = q.limit.unwrap_or(50).clamp(1, MAX_LIMIT);
    let mut stream = match js.get_stream(DLQ_STREAM).await {
        Ok(st) => st,
        Err(e) => return err(StatusCode::BAD_GATEWAY, format!("DLQ stream: {e}")),
    };
    let state = match stream.info().await {
        Ok(i) => i.state.clone(),
        Err(e) => return err(StatusCode::BAD_GATEWAY, format!("DLQ stream info: {e}")),
    };
    let now_s = chrono::Utc::now().timestamp();
    let mut rows = Vec::new();
    if state.messages > 0 {
        let mut seq = state.last_sequence;
        let mut lookups = 0;
        while seq >= state.first_sequence && rows.len() < limit && lookups < SCAN_CAP {
            if let Ok(m) = stream.get_raw_message(seq).await {
                rows.push(row(&m, now_s));
            }
            lookups += 1;
            if seq == 0 {
                break;
            }
            seq -= 1;
        }
    }
    Json(json!({
        "stream": {
            "messages": state.messages,
            "bytes": state.bytes,
            "first_seq": state.first_sequence,
            "last_seq": state.last_sequence,
        },
        "messages": rows,
    }))
    .into_response()
}

#[derive(Debug, Default, Deserialize)]
struct ReplayBody {
    seq: Option<u64>,
}

async fn replay(State(s): State<AppState>, body: Option<Json<ReplayBody>>) -> Response {
    let Some(seq) = body.and_then(|Json(b)| b.seq) else {
        return err(StatusCode::BAD_REQUEST, "replay needs {\"seq\": n} (a DLQ stream sequence)");
    };
    let Some(js) = jetstream(&s) else {
        return err(StatusCode::SERVICE_UNAVAILABLE, "NATS not connected");
    };
    let stream = match js.get_stream(DLQ_STREAM).await {
        Ok(st) => st,
        Err(e) => return err(StatusCode::BAD_GATEWAY, format!("DLQ stream: {e}")),
    };
    let msg = match stream.get_raw_message(seq).await {
        Ok(m) => m,
        Err(_) => return err(StatusCode::NOT_FOUND, format!("seq {seq} is not in the DLQ stream")),
    };
    let subject = original_subject(msg.subject.as_str(), header(&msg, "Fa-Dlq-Subject"));
    // the original headers, without the forwarder's bookkeeping and its dedupe id
    let mut headers = fa_bus::async_nats::HeaderMap::new();
    for (k, vs) in msg.headers.iter() {
        let name: &str = k.as_ref();
        if name.starts_with("Fa-Dlq-") || name == "Nats-Msg-Id" {
            continue;
        }
        for v in vs {
            headers.append(k.clone(), v.clone());
        }
    }
    let publish = async {
        js.publish_with_headers(subject.clone(), headers, msg.payload.clone())
            .await
            .map_err(|e| e.to_string())?
            .await
            .map_err(|e| e.to_string())
    };
    if let Err(e) = publish.await {
        return err(StatusCode::BAD_GATEWAY, format!("replay to {subject}: {e}"));
    }
    let deleted = stream.delete_message(seq).await.unwrap_or(false);
    crate::audit::record(
        &s,
        "dlq.replay",
        Some(&subject),
        json!({ "seq": seq, "dlq_subject": msg.subject.as_str(), "size": msg.payload.len(), "deleted": deleted }),
    )
    .await;
    Json(json!({ "ok": true, "seq": seq, "subject": subject, "deleted": deleted })).into_response()
}

#[derive(Debug, Default, Deserialize)]
struct PurgeBody {
    #[serde(default)]
    confirm: bool,
    subject: Option<String>,
}

async fn purge(State(s): State<AppState>, body: Option<Json<PurgeBody>>) -> Response {
    let b = body.map(|Json(b)| b).unwrap_or_default();
    if !b.confirm {
        crate::audit::record(&s, "dlq.purge.refused", b.subject.as_deref(), json!({ "why": "confirm required" })).await;
        return (
            StatusCode::PRECONDITION_REQUIRED,
            Json(json!({ "error": "confirm_required", "detail": "purging the DLQ stream needs {\"confirm\": true}" })),
        )
            .into_response();
    }
    let Some(js) = jetstream(&s) else {
        return err(StatusCode::SERVICE_UNAVAILABLE, "NATS not connected");
    };
    let stream = match js.get_stream(DLQ_STREAM).await {
        Ok(st) => st,
        Err(e) => return err(StatusCode::BAD_GATEWAY, format!("DLQ stream: {e}")),
    };
    let purged = match &b.subject {
        Some(f) if !f.trim().is_empty() => stream.purge().filter(f.trim()).await,
        _ => stream.purge().await,
    };
    match purged {
        Ok(r) => {
            crate::audit::record(
                &s,
                "dlq.purge",
                b.subject.as_deref(),
                json!({ "purged": r.purged, "filter": b.subject }),
            )
            .await;
            Json(json!({ "ok": r.success, "purged": r.purged })).into_response()
        }
        Err(e) => err(StatusCode::BAD_GATEWAY, format!("purge: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use fa_bus::async_nats::HeaderMap;
    use fa_core::config::PlatformConfig;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    fn app() -> Router {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let cfg = PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        let state = AppState::new(cfg, format!("{root}/sources.yaml").into(), None);
        router().with_state(state)
    }

    async fn call(method: &str, uri: &str, body: &str) -> (StatusCode, Value) {
        let req = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_owned()))
            .unwrap();
        let res = app().oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn dlq_routes_validate_before_nats() {
        let (status, _) = call("GET", "/api/dlq?limit=10", "").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "list needs NATS");
        let (status, v) = call("POST", "/api/dlq/replay", "{}").await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "replay without seq: {v}");
        let (status, _) = call("POST", "/api/dlq/replay", r#"{"seq": 7}"#).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "replay needs NATS");
        let (status, v) = call("POST", "/api/dlq/purge", "{}").await;
        assert_eq!(status, StatusCode::PRECONDITION_REQUIRED, "purge needs confirm");
        assert_eq!(v["error"], "confirm_required");
        let (status, _) = call("POST", "/api/dlq/purge", r#"{"confirm": true}"#).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "confirmed purge needs NATS");
    }

    #[test]
    fn rows_carry_headers_preview_and_age() {
        let mut h = HeaderMap::new();
        h.insert("Fa-Dlq-Stream", "LATENT");
        h.insert("Fa-Dlq-Subject", "latent.state.1m.x");
        h.insert("Fa-Dlq-Seq", "41");
        h.insert("Fa-Dlq-Consumer", "model-runtime");
        h.insert("Fa-Dlq-Reason", "max_deliveries");
        h.insert("Fa-Dlq-Deliveries", "5");
        let m = StreamMessage {
            subject: "dlq.LATENT.latent.state.1m.x".into(),
            sequence: 3,
            headers: h,
            payload: b"hello dead letter".to_vec().into(),
            time: fa_bus::async_nats::datetime::DateTime::from_unix_timestamp_nanos(1_000i128 * 1_000_000_000).unwrap(),
        };
        let v = row(&m, 1_060);
        assert_eq!(v["seq"], 3);
        assert_eq!(v["original_stream"], "LATENT");
        assert_eq!(v["original_subject"], "latent.state.1m.x");
        assert_eq!(v["reason"], "max_deliveries");
        assert_eq!(v["age_s"], 60);
        assert_eq!(v["preview"], "hello dead letter");
        assert_eq!((v["lossy"].as_bool(), v["truncated"].as_bool()), (Some(false), Some(false)));
    }

    #[test]
    fn subjects_fall_back_and_previews_flag_binary() {
        assert_eq!(original_subject("dlq.MD.md.trade.kraken", None), "md.trade.kraken");
        assert_eq!(original_subject("dlq.MD.md.trade.kraken", Some("")), "md.trade.kraken");
        assert_eq!(original_subject("dlq.MD.md.trade.kraken", Some("md.other")), "md.other");
        assert_eq!(original_subject("odd", None), "odd");

        let (text, lossy, truncated) = preview(b"plain");
        assert_eq!((text.as_str(), lossy, truncated), ("plain", false, false));
        let (_, lossy, _) = preview(&[0xff, 0xfe, 0x00]);
        assert!(lossy, "invalid UTF-8 is flagged");
        let long = vec![b'a'; PREVIEW_BYTES + 10];
        let (text, _, truncated) = preview(&long);
        assert_eq!((text.len(), truncated), (PREVIEW_BYTES, true));
    }
}
