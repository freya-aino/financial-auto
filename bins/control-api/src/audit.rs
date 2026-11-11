//! Request audit, the `/api/v1` alias and per-request trace ids (spec 50 §3, §4, Q2; spec 60 §6.5).
//!
//! One middleware wraps the whole router ([`wrap`]) and runs before routing:
//!
//! * **`/api/v1` alias (Q2, provisional):** `/api/v1/<rest>` is rewritten to `/api/<rest>`, so
//!   every route answers under both prefixes; `/api/…` stays canonical (frontend, scripts and the
//!   OpenAPI document use it). A breaking change would get `/api/v2` routed separately.
//! * **Trace id:** a `traceparent` request header is continued, otherwise a new trace starts; the
//!   handler runs under [`fa_bus::trace::scope`], so the NATS commands it sends carry the same
//!   trace id, which is also stored in its audit rows (`payload.trace_id`) and returned in the
//!   `traceparent` response header.
//! * **Audit (spec 50 §3 "every mutating endpoint writes an audit entry"):** every `POST`,
//!   `PUT`, `PATCH` and `DELETE` on a catalogued `/api/…` route writes at least one `audit.log`
//!   row from control-api: the handler's own domain row(s) (written through [`append`] /
//!   [`record`], which note the action for this request), or — when the handler wrote none, e.g.
//!   it failed validation, the owning service refused, or it forwards a command whose effect the
//!   service audits — one generic row `api.request` (actor `ui`, subject = route template,
//!   payload `{method, path, query, status, body, trace_id}`). The response header `x-fa-audit`
//!   names the actions written. Not audited per request ([`NOT_AUDITED`]): `POST
//!   /api/graph/query` and `POST /api/trading/pretrade` (reads) and the two vmalert / Alertmanager receivers (machine traffic
//!   every 30 s; they audit each alert transition as `system:vmalert`).
//!
//! Rows from owning services are separate events, not duplicates: control-api records the
//! operator's *request* (actor `ui`, e.g. `halt.request`), the risk-engine records the *effect*
//! it applied (actor `risk-engine`, e.g. `halt`); both carry the trace id where the service logs
//! it. This way a kill switch or paper order is on record even when the risk-engine is down.
use std::sync::{Arc, Mutex};

use axum::{
    Json, Router,
    body::Body,
    extract::{Request, State},
    http::{HeaderValue, Method, StatusCode, Uri},
    middleware::{self, Next},
    response::{IntoResponse, Response},
};
use fa_bus::trace::{self, TraceContext};
use serde_json::{Value, json};
use sqlx::PgPool;

use crate::state::AppState;

/// Actor of rows written for operator requests (spec 42 `audit.log`).
pub const ACTOR: &str = "ui";
/// Action of the generic per-request row.
pub const GENERIC_ACTION: &str = "api.request";
/// Response header naming the audit actions a request wrote.
pub const AUDIT_HEADER: &str = "x-fa-audit";
/// Mutating-method routes that are not audited per request (see module docs).
pub const NOT_AUDITED: [&str; 4] =
    ["/api/graph/query", "/api/trading/pretrade", "/api/alerts/vmalert/api/v2/alerts", "/api/alerts/webhook"];
/// Request bodies above this are refused (axum's own JSON limit is 2 MiB too).
const MAX_BODY: usize = 2 * 1024 * 1024;
/// Bodies above this are stored as `{truncated, bytes}` in the generic row.
const MAX_BODY_IN_ROW: usize = 16 * 1024;

tokio::task_local! {
    static WRITTEN: Arc<Mutex<Vec<String>>>;
}

/// Notes that the running request wrote (or attempted) an audit row with `action`.
pub fn note(action: &str) {
    let _ = WRITTEN.try_with(|w| w.lock().expect("audit note lock").push(action.to_owned()));
}

/// Adds the current trace id to an object payload.
fn with_trace(mut payload: Value) -> Value {
    if let (Some(t), Value::Object(m)) = (trace::current(), &mut payload) {
        m.entry("trace_id").or_insert_with(|| json!(t.trace_id_hex()));
    }
    payload
}

/// `fa_warehouse::audit::append` that also notes the action for the request audit and adds the
/// trace id. Every control-api audit write goes through here or [`record`].
pub async fn append(
    pool: &PgPool,
    actor: &str,
    action: &str,
    subject: Option<&str>,
    payload: Value,
) -> Result<i64, fa_warehouse::WarehouseError> {
    note(action);
    fa_warehouse::audit::append(pool, actor, action, subject, with_trace(payload)).await
}

/// Writes a row as `ui` when Postgres is attached; returns whether it did. The action is noted
/// either way (without Postgres no row can be written, the generic one included).
pub async fn record(s: &AppState, action: &str, subject: Option<&str>, payload: Value) -> bool {
    note(action);
    let Some(pool) = s.pg.get() else {
        return false;
    };
    match fa_warehouse::audit::append(pool, ACTOR, action, subject, with_trace(payload)).await {
        Ok(_) => true,
        Err(e) => {
            tracing::warn!(action, error = %e, "audit row not written");
            false
        }
    }
}

/// `/api/v1` and `/api/v1/<rest>` → `/api` and `/api/<rest>`.
pub fn strip_version(path: &str) -> Option<String> {
    let rest = path.strip_prefix("/api/v1")?;
    (rest.is_empty() || rest.starts_with('/')).then(|| format!("/api{rest}"))
}

pub fn is_mutating(m: &Method) -> bool {
    matches!(*m, Method::POST | Method::PUT | Method::PATCH | Method::DELETE)
}

/// The catalogued route template a request hits (`/api/jobs/{id}/cancel`), most literal first;
/// `None` for paths without a mutating route of that method.
pub fn template_for(method: &Method, path: &str) -> Option<String> {
    let m = method.as_str().to_ascii_lowercase();
    let segs: Vec<&str> = path.trim_end_matches('/').split('/').collect();
    crate::openapi::all_routes()
        .into_iter()
        .filter(|r| r.method == m)
        .filter_map(|r| {
            let tpl: Vec<&str> = r.path.split('/').collect();
            let mut literal = 0;
            for (i, t) in tpl.iter().enumerate() {
                if t.starts_with("{*") {
                    return Some((literal, r.path.clone()));
                }
                let s = segs.get(i)?;
                if t.starts_with('{') {
                    if s.is_empty() {
                        return None;
                    }
                } else if t == s {
                    literal += 1;
                } else {
                    return None;
                }
            }
            (tpl.len() == segs.len()).then_some((literal, r.path))
        })
        .max_by_key(|(literal, _)| *literal)
        .map(|(_, p)| p)
}

fn body_value(bytes: &[u8]) -> Value {
    if bytes.is_empty() {
        return Value::Null;
    }
    if bytes.len() > MAX_BODY_IN_ROW {
        return json!({ "truncated": true, "bytes": bytes.len() });
    }
    serde_json::from_slice(bytes).unwrap_or_else(|_| json!(String::from_utf8_lossy(bytes)))
}

/// The middleware (see module docs).
pub async fn middleware(State(s): State<AppState>, mut req: Request, next: Next) -> Response {
    if let Some(p) = strip_version(req.uri().path()) {
        let pq = match req.uri().query() {
            Some(q) => format!("{p}?{q}"),
            None => p,
        };
        if let Ok(uri) = pq.parse::<Uri>() {
            *req.uri_mut() = uri;
        }
    }
    let ctx = req
        .headers()
        .get(trace::TRACEPARENT)
        .and_then(|v| v.to_str().ok())
        .and_then(TraceContext::parse)
        .map_or_else(TraceContext::new, |c| c.child());
    let method = req.method().clone();
    let path = req.uri().path().to_owned();
    let template = (is_mutating(&method) && path.starts_with("/api/") && !NOT_AUDITED.contains(&path.as_str()))
        .then(|| template_for(&method, &path))
        .flatten();

    let mut resp = match template {
        None => trace::scope(ctx, next.run(req)).await,
        Some(template) => {
            let query = req.uri().query().map(str::to_owned);
            let (parts, body) = req.into_parts();
            let bytes = match axum::body::to_bytes(body, MAX_BODY).await {
                Ok(b) => b,
                Err(e) => {
                    return (StatusCode::PAYLOAD_TOO_LARGE, Json(json!({ "error": format!("request body: {e}") })))
                        .into_response();
                }
            };
            let req = Request::from_parts(parts, Body::from(bytes.clone()));
            let written = Arc::new(Mutex::new(Vec::<String>::new()));
            let mut resp = trace::scope(ctx, WRITTEN.scope(written.clone(), next.run(req))).await;
            let mut actions = std::mem::take(&mut *written.lock().expect("audit note lock"));
            if actions.is_empty() {
                let payload = json!({
                    "method": method.as_str(),
                    "path": path,
                    "query": query,
                    "status": resp.status().as_u16(),
                    "body": body_value(&bytes),
                });
                trace::scope(ctx, record(&s, GENERIC_ACTION, Some(&template), payload)).await;
                actions.push(GENERIC_ACTION.to_owned());
            }
            actions.dedup();
            if let Ok(v) = HeaderValue::from_str(&actions.join(",")) {
                resp.headers_mut().insert(AUDIT_HEADER, v);
            }
            resp
        }
    };
    if let Ok(v) = HeaderValue::from_str(&ctx.traceparent()) {
        resp.headers_mut().insert(trace::TRACEPARENT, v);
    }
    resp
}

/// `app` behind the middleware; the outer router only has `app` as its fallback, so the
/// middleware sees (and may rewrite) the request before `app` routes it.
pub fn wrap(state: AppState, app: Router) -> Router {
    Router::new().fallback_service(app).layer(middleware::from_fn_with_state(state, middleware))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_prefix_is_stripped() {
        assert_eq!(strip_version("/api/v1/health").as_deref(), Some("/api/health"));
        assert_eq!(strip_version("/api/v1").as_deref(), Some("/api"));
        assert_eq!(strip_version("/api/v10/x"), None);
        assert_eq!(strip_version("/api/health"), None);
        assert_eq!(strip_version("/ws/bus"), None);
    }

    #[test]
    fn templates_prefer_literal_segments() {
        assert_eq!(template_for(&Method::POST, "/api/jobs/abc/cancel").as_deref(), Some("/api/jobs/{id}/cancel"));
        assert_eq!(
            template_for(&Method::POST, "/api/latent/snapshots/s1/shadow").as_deref(),
            Some("/api/latent/snapshots/{id}/shadow")
        );
        assert_eq!(
            template_for(&Method::POST, "/api/latent/snapshots/s1/promote").as_deref(),
            Some("/api/latent/snapshots/{id}/{action}")
        );
        assert_eq!(template_for(&Method::POST, "/api/no/such/route"), None);
        assert_eq!(template_for(&Method::DELETE, "/api/jobs/abc/cancel"), None);
    }

    #[test]
    fn bodies_are_kept_small() {
        assert_eq!(body_value(b""), Value::Null);
        assert_eq!(body_value(br#"{"a":1}"#), json!({"a": 1}));
        assert_eq!(body_value(b"plain"), json!("plain"));
        assert_eq!(body_value(&vec![b'x'; MAX_BODY_IN_ROW + 1])["truncated"], true);
    }
}
