//! `GET /api/config/{section}` and `PUT /api/config/{section}` (spec 50 §4.1).
//!
//! GET returns any section of `PlatformConfig` (`timescales`, `nats`, `helix`, `victoriametrics`,
//! `postgres`, `garage`, `control_api`, `runtime`) passed through `fa_core::secrets::redact`, so
//! a `*key*|*secret*|*token*` value can never leave the process; `postgres.url` additionally has
//! its password masked. PUT is accepted only for the whitelisted runtime-tunable section
//! `runtime`: the body (JSON, complete or partial) is merged over the current values, validated
//! against `RUNTIME_BOUNDS`, applied in-process (`Live::set_runtime`), persisted to
//! `config/runtime.toml` next to the platform config (git-ignored, merged on the next start) and
//! audited in Postgres `audit.log` when a pool is attached. PUT on any other section → 405 with
//! the list of tunable sections. Also hosts the [`audit`] helper the other mutating routes use.
use std::{
    path::PathBuf,
    sync::{Mutex, OnceLock},
};

use axum::{
    Json, Router,
    extract::{Path, State, rejection::JsonRejection},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use fa_core::config::{PlatformConfig, RUNTIME_BOUNDS, RuntimeConfig};
use serde_json::{Value, json};

use crate::state::AppState;

/// Sections `PUT` accepts.
pub const TUNABLE_SECTIONS: [&str; 1] = ["runtime"];

pub fn router() -> Router<AppState> {
    Router::new().route("/api/config/{section}", get(get_section).put(put_section))
}

fn runtime_path_slot() -> &'static Mutex<Option<PathBuf>> {
    static SLOT: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
    SLOT.get_or_init(Mutex::default)
}

/// Where `PUT /api/config/runtime` persists: `runtime.toml` next to the platform config `main`
/// was started with (set once at startup; defaults to `config/runtime.toml` in the cwd).
pub fn set_runtime_path(platform_config: impl AsRef<std::path::Path>) {
    *runtime_path_slot().lock().expect("runtime path") = Some(fa_core::config::runtime_path(platform_config));
}

pub fn runtime_path() -> PathBuf {
    runtime_path_slot()
        .lock()
        .expect("runtime path")
        .clone()
        .unwrap_or_else(|| fa_core::config::runtime_path("config/platform.toml"))
}

/// One section as redacted JSON. `None` for an unknown section.
pub fn redacted_section(cfg: &PlatformConfig, runtime: &RuntimeConfig, section: &str) -> Option<Value> {
    let mut v = if section == "runtime" { toml::Value::try_from(runtime).ok()? } else { cfg.section(section)? };
    fa_core::secrets::redact(&mut v);
    if let toml::Value::Table(t) = &mut v {
        for (k, val) in t.iter_mut() {
            if k.ends_with("url")
                && let toml::Value::String(s) = val
            {
                *s = mask_url_password(s);
            }
        }
    }
    Some(toml_to_json(v))
}

/// `scheme://user:password@host` → `scheme://user:***@host`.
pub fn mask_url_password(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return url.to_owned();
    };
    let Some((userinfo, host)) = rest.split_once('@') else {
        return url.to_owned();
    };
    match userinfo.split_once(':') {
        Some((user, pw)) if !pw.is_empty() => format!("{scheme}://{user}:{}@{host}", fa_core::secrets::REDACTED),
        _ => url.to_owned(),
    }
}

fn toml_to_json(v: toml::Value) -> Value {
    match v {
        toml::Value::String(s) => Value::String(s),
        toml::Value::Integer(i) => json!(i),
        toml::Value::Float(f) => json!(f),
        toml::Value::Boolean(b) => Value::Bool(b),
        toml::Value::Datetime(d) => Value::String(d.to_string()),
        toml::Value::Array(a) => Value::Array(a.into_iter().map(toml_to_json).collect()),
        toml::Value::Table(t) => Value::Object(t.into_iter().map(|(k, v)| (k, toml_to_json(v))).collect()),
    }
}

fn json_to_toml(v: &Value) -> Result<toml::Value, String> {
    Ok(match v {
        Value::Null => return Err("null is not a config value".into()),
        Value::Bool(b) => toml::Value::Boolean(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                toml::Value::Integer(i)
            } else if let Some(f) = n.as_f64() {
                toml::Value::Float(f)
            } else {
                return Err(format!("{n} is not representable"));
            }
        }
        Value::String(s) => toml::Value::String(s.clone()),
        Value::Array(a) => toml::Value::Array(a.iter().map(json_to_toml).collect::<Result<_, _>>()?),
        Value::Object(o) => toml::Value::Table(
            o.iter().map(|(k, v)| json_to_toml(v).map(|t| (k.clone(), t))).collect::<Result<_, _>>()?,
        ),
    })
}

async fn get_section(State(s): State<AppState>, Path(section): Path<String>) -> Response {
    let runtime = s.live().runtime().clone();
    match redacted_section(&s.cfg, &runtime, &section) {
        Some(values) => Json(json!({
            "section": section,
            "tunable": TUNABLE_SECTIONS.contains(&section.as_str()),
            "values": values,
            "bounds": if section == "runtime" { json!(RUNTIME_BOUNDS.iter().map(|(k, lo, hi)| json!({"key": k, "min": lo, "max": hi})).collect::<Vec<_>>()) } else { Value::Null },
        }))
        .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "unknown_section", "section": section, "sections": PlatformConfig::section_names() })),
        )
            .into_response(),
    }
}

/// Merges `body` (JSON, complete or partial) over `current` and validates (pure).
pub fn apply_update(current: &RuntimeConfig, body: &Value) -> Result<RuntimeConfig, String> {
    if !body.is_object() {
        return Err("body must be a JSON object with runtime keys".into());
    }
    let partial = json_to_toml(body)?;
    current.merged_with(&partial)
}

async fn put_section(
    State(s): State<AppState>,
    Path(section): Path<String>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    if !TUNABLE_SECTIONS.contains(&section.as_str()) {
        let status = if PlatformConfig::section_names().contains(&section.as_str()) {
            StatusCode::METHOD_NOT_ALLOWED
        } else {
            StatusCode::NOT_FOUND
        };
        let mut res = (
            status,
            Json(json!({
                "error": if status == StatusCode::NOT_FOUND { "unknown_section" } else { "not_tunable" },
                "section": section,
                "tunable": TUNABLE_SECTIONS,
                "detail": "only runtime-tunable sections accept PUT (spec 50 §4.1)",
            })),
        )
            .into_response();
        res.headers_mut().insert(header::ALLOW, HeaderValue::from_static("GET"));
        return res;
    }
    let body = match body {
        Ok(Json(b)) => b,
        Err(rej) => {
            return (StatusCode::BAD_REQUEST, Json(json!({ "error": "bad_request", "detail": rej.body_text() })))
                .into_response();
        }
    };
    let current = s.live().runtime().clone();
    let next = match apply_update(&current, &body) {
        Ok(n) => n,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({ "error": "invalid", "detail": e }))).into_response(),
    };
    let changed = next != current;
    s.live_mut().set_runtime(next.clone());
    let path = runtime_path();
    let persisted = match next.save(&path) {
        Ok(()) => json!({ "ok": true, "path": path.display().to_string() }),
        Err(e) => {
            tracing::warn!(error = %e, "runtime config applied but not persisted");
            json!({ "ok": false, "path": path.display().to_string(), "error": e.to_string() })
        }
    };
    let audited = audit(
        &s,
        "config.put",
        "runtime",
        json!({ "before": current, "after": next, "changed": changed, "persisted": persisted["ok"] }),
    )
    .await;
    Json(json!({
        "ok": true,
        "section": "runtime",
        "changed": changed,
        "values": serde_json::to_value(&next).unwrap_or_default(),
        "persisted": persisted,
        "audited": audited,
    }))
    .into_response()
}

/// Writes an `audit.log` row (`actor = 'ui'`) when Postgres is attached; returns whether it did.
/// Same insert as `ledger_routes` uses (control-api has no `fa-warehouse` dependency).
pub async fn audit(s: &AppState, action: &str, subject: &str, payload: Value) -> bool {
    let Some(pool) = s.pg.get() else {
        return false;
    };
    match sqlx::query("INSERT INTO audit.log (actor, action, subject, payload) VALUES ('ui', $1, $2, $3)")
        .bind(action)
        .bind(subject)
        .bind(payload)
        .execute(pool)
        .await
    {
        Ok(_) => true,
        Err(e) => {
            tracing::warn!(action, error = %e, "audit row not written");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    fn app() -> (AppState, Router) {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let mut cfg = PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        cfg.postgres.url = "postgres://fa:hunter2@127.0.0.1:5432/fa".into();
        let state = AppState::new(cfg, format!("{root}/sources.yaml").into(), None);
        (state.clone(), router().with_state(state))
    }

    async fn send(app: Router, req: Request<Body>) -> (StatusCode, Value) {
        let res = app.oneshot(req).await.unwrap();
        let status = res.status();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
    }

    #[test]
    fn redaction_hides_secrets_and_url_passwords() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let mut cfg = PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        cfg.postgres.url = "postgres://fa:hunter2@127.0.0.1:5432/fa".into();
        let r = RuntimeConfig::default();
        let pg = redacted_section(&cfg, &r, "postgres").unwrap();
        assert_eq!(pg["url"], "postgres://fa:***@127.0.0.1:5432/fa");
        let garage = redacted_section(&cfg, &r, "garage").unwrap();
        assert_eq!(garage["endpoint"], "http://127.0.0.1:3900");
        assert_eq!(garage["public_endpoint"], "http://127.0.0.1:3900");
        assert!(garage.get("access_key_id").is_none(), "secrets are not part of the platform config at all");
        assert_eq!(redacted_section(&cfg, &r, "runtime").unwrap()["ui"]["poll_ms"], 2000);
        assert_eq!(redacted_section(&cfg, &r, "timescales").unwrap()[0], "1s");
        assert!(redacted_section(&cfg, &r, "secrets").is_none());
        assert_eq!(mask_url_password("nats://127.0.0.1:4222"), "nats://127.0.0.1:4222");
        assert_eq!(mask_url_password("http://u@h"), "http://u@h");
        // belt and braces: a sensitive key inside a table is still redacted by the generic pass
        let mut v: toml::Value = toml::from_str("api_key = \"k\"\nname = \"n\"\n").unwrap();
        fa_core::secrets::redact(&mut v);
        assert_eq!(toml_to_json(v)["api_key"], "***");
    }

    #[test]
    fn whitelist_and_validation_are_pure() {
        assert_eq!(TUNABLE_SECTIONS, ["runtime"]);
        let cur = RuntimeConfig::default();
        let next = apply_update(&cur, &json!({ "ui": { "poll_ms": 1000 } })).unwrap();
        assert_eq!(next.ui.poll_ms, 1000);
        assert_eq!(next.bus_tap, cur.bus_tap, "unmentioned keys stay");
        assert!(apply_update(&cur, &json!({ "ui": { "poll_ms": 10 } })).unwrap_err().contains("ui.poll_ms"));
        assert!(apply_update(&cur, &json!({ "nats": { "url": "x" } })).is_err(), "unknown key");
        assert!(apply_update(&cur, &json!([1])).is_err());
        assert!(apply_update(&cur, &json!({ "ui": { "poll_ms": null } })).is_err());
    }

    #[tokio::test]
    async fn get_any_section_put_only_runtime() {
        let (_, app) = app();
        let (s, v) = send(app.clone(), Request::get("/api/config/postgres").body(Body::empty()).unwrap()).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["tunable"], false);
        assert_eq!(v["values"]["url"], "postgres://fa:***@127.0.0.1:5432/fa");
        let (s, v) = send(app.clone(), Request::get("/api/config/runtime").body(Body::empty()).unwrap()).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["tunable"], true);
        assert!(v["bounds"].as_array().unwrap().len() >= 8);
        let (s, v) = send(app.clone(), Request::get("/api/config/nope").body(Body::empty()).unwrap()).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert_eq!(v["error"], "unknown_section");

        let put = |path: &str, body: Value| {
            Request::put(path).header("content-type", "application/json").body(Body::from(body.to_string())).unwrap()
        };
        let res = app.clone().oneshot(put("/api/config/garage", json!({}))).await.unwrap();
        assert_eq!(res.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert_eq!(res.headers().get(header::ALLOW).unwrap(), "GET");
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["tunable"], json!(["runtime"]));
        let (s, _) = send(app.clone(), put("/api/config/nope", json!({}))).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        let (s, v) = send(app.clone(), put("/api/config/runtime", json!({ "ui": { "poll_ms": 1 } }))).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert_eq!(v["error"], "invalid");
    }

    #[tokio::test]
    async fn put_runtime_applies_in_process_and_persists() {
        let dir = std::env::temp_dir().join(format!("control-api-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        set_runtime_path(dir.join("platform.toml"));
        assert_eq!(runtime_path(), dir.join("runtime.toml"));
        let (state, app) = app();
        let req = Request::put("/api/config/runtime")
            .header("content-type", "application/json")
            .body(Body::from(json!({ "bus_tap": { "max_msgs_per_sec": 50 }, "live": { "max_jobs": 20 } }).to_string()))
            .unwrap();
        let (s, v) = send(app, req).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["changed"], true);
        assert_eq!(v["persisted"]["ok"], true, "{v}");
        assert_eq!(v["audited"], false, "no postgres attached");
        assert_eq!(state.live().runtime().bus_tap.max_msgs_per_sec, 50);
        let saved = RuntimeConfig::load_overlay(dir.join("runtime.toml")).unwrap().unwrap();
        assert_eq!(saved.live.max_jobs, 20);
        assert_eq!(saved.ui.poll_ms, 2000);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
