//! The LLM suite's surface, proxied (D-117), plus the endpoint configuration the UI writes.
//!
//! | Route | What |
//! |---|---|
//! | `GET /api/llm/status` | endpoint, model, tool catalogue, whether mutating tools are available |
//! | `POST /api/llm/chat` | one agent turn (`{message, conversation_id?, approvals[]}`) |
//! | `GET /api/llm/conversations` | conversation list, newest first |
//! | `GET /api/llm/conversations/{id}` | one conversation with its messages and tool calls |
//! | `GET /api/llm/embeddings` | news-embedding health: counts per provider, newest item |
//! | `PUT /api/llm/endpoint` | set base URL / models / key; the key goes to `secrets.toml` |
//! | `GET /api/llm/mcp` | registered external MCP servers, their status and tools (D-120) |
//! | `PUT /api/llm/mcp/{id}` | register or update one; probes it and returns the outcome |
//! | `DELETE /api/llm/mcp/{id}` | remove one |
//! | `POST /api/llm/mcp/{id}/probe` | re-handshake and re-read its tool list |
//!
//! Why proxy rather than let the UI call the suite directly: spec 51 keeps the frontend on exactly
//! one API, so everything the operator sees goes through control-api's audit middleware, its CORS
//! and its one port. `llm-suite` therefore binds a private address inside compose.
//!
//! **The key is write-only.** `PUT /api/llm/endpoint` takes `api_key` and writes it to the
//! git-ignored `config/secrets.toml`; no route ever returns it, `GET /api/llm/status` reports only
//! `key_set`, and the audit row records which fields changed, never their values. A key that
//! arrives here is also never logged: the handler does not log its body.
use std::path::PathBuf;

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post, put},
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::state::AppState;

/// Where the suite listens inside compose (`FA_LLM_URL` overrides).
fn suite_url() -> String {
    std::env::var("FA_LLM_URL").ok().filter(|u| !u.is_empty()).unwrap_or_else(|| "http://llm-suite:7710".into())
}

fn secrets_path() -> PathBuf {
    fa_core::secrets::default_path()
}

/// The platform config this process was started with, so the runtime overlay lands next to it.
fn config_path() -> PathBuf {
    std::env::var_os("FA_CONFIG").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("config/platform.toml"))
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/llm/status", get(status))
        .route("/api/llm/chat", post(chat))
        .route("/api/llm/conversations", get(conversations))
        .route("/api/llm/conversations/{id}", get(conversation))
        .route("/api/llm/embeddings", get(embeddings))
        .route("/api/llm/endpoint", put(set_endpoint))
        .route("/api/llm/mcp", get(mcp_list))
        .route("/api/llm/mcp/{id}", put(mcp_put).delete(mcp_delete))
        .route("/api/llm/mcp/{id}/probe", post(mcp_probe))
}

/// GETs `path` from the suite and relays its answer.
async fn relay(path: &str) -> Response {
    let url = format!("{}{path}", suite_url());
    let http = reqwest::Client::new();
    match http.get(&url).timeout(std::time::Duration::from_secs(20)).send().await {
        Ok(r) => {
            let code = StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let body: Value = r.json().await.unwrap_or_else(|e| json!({"error": e.to_string()}));
            (code, Json(body)).into_response()
        }
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": format!("llm-suite unreachable: {e}"), "configured": false})),
        )
            .into_response(),
    }
}

/// Sends `body` to the suite with `method` and relays its answer.
async fn forward(method: reqwest::Method, path: &str, body: Option<Value>) -> Response {
    let url = format!("{}{path}", suite_url());
    let mut req = reqwest::Client::new().request(method, &url).timeout(std::time::Duration::from_secs(45));
    if let Some(b) = body {
        req = req.json(&b);
    }
    match req.send().await {
        Ok(r) => {
            let code = StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let out: Value = r.json().await.unwrap_or_else(|e| json!({"error": e.to_string()}));
            (code, Json(out)).into_response()
        }
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": format!("llm-suite unreachable: {e}")})),
        )
            .into_response(),
    }
}

/// A server id as both a URL path segment and part of a tool name: a short slug, nothing else.
fn valid_server_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 32
        && id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
}

async fn mcp_list(State(_s): State<AppState>) -> Response {
    relay("/api/llm/mcp").await
}

/// Registers or updates one MCP server.
///
/// The body passes through unchanged — including an optional `token`, which the suite stores
/// write-only and no route returns. What this handler adds is the audit row, and that row names
/// the server, its URL and the access level the operator chose, never the token: granting the
/// agent a door out of the platform is exactly the kind of change the log exists for.
async fn mcp_put(State(s): State<AppState>, Path(id): Path<String>, Json(body): Json<Value>) -> Response {
    if !valid_server_id(&id) {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "bad server id"}))).into_response();
    }
    let out = forward(reqwest::Method::PUT, &format!("/api/llm/mcp/{id}"), Some(body.clone())).await;
    crate::audit::record(
        &s,
        "llm.mcp.set",
        Some(&id),
        json!({
            "url": body.get("url").and_then(Value::as_str).unwrap_or_default(),
            "access": body.get("access").and_then(Value::as_str).unwrap_or("mutating"),
            "enabled": body.get("enabled").and_then(Value::as_bool).unwrap_or(true),
            "token_given": body.get("token").and_then(Value::as_str).is_some_and(|t| !t.is_empty()),
        }),
    )
    .await;
    out
}

async fn mcp_delete(State(s): State<AppState>, Path(id): Path<String>) -> Response {
    if !valid_server_id(&id) {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "bad server id"}))).into_response();
    }
    let out = forward(reqwest::Method::DELETE, &format!("/api/llm/mcp/{id}"), None).await;
    crate::audit::record(&s, "llm.mcp.remove", Some(&id), json!({})).await;
    out
}

/// Re-probing changes nothing, so it is not audited.
async fn mcp_probe(State(_s): State<AppState>, Path(id): Path<String>) -> Response {
    if !valid_server_id(&id) {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "bad server id"}))).into_response();
    }
    forward(reqwest::Method::POST, &format!("/api/llm/mcp/{id}/probe"), Some(json!({}))).await
}

async fn status(State(_s): State<AppState>) -> Response {
    relay("/api/llm/status").await
}

async fn conversations(State(_s): State<AppState>) -> Response {
    relay("/api/llm/conversations").await
}

async fn conversation(State(_s): State<AppState>, Path(id): Path<String>) -> Response {
    if id.contains('/') || id.len() > 64 {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "bad conversation id"}))).into_response();
    }
    relay(&format!("/api/llm/conversations/{id}")).await
}

/// One agent turn. The body is passed through unchanged; the suite owns the loop.
///
/// A long timeout on purpose: a turn can take several tool rounds, each a model call.
async fn chat(State(_s): State<AppState>, Json(body): Json<Value>) -> Response {
    let url = format!("{}/api/llm/chat", suite_url());
    let http = reqwest::Client::new();
    match http.post(&url).json(&body).timeout(std::time::Duration::from_secs(300)).send().await {
        Ok(r) => {
            let code = StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let out: Value = r.json().await.unwrap_or_else(|e| json!({"error": e.to_string()}));
            (code, Json(out)).into_response()
        }
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": format!("llm-suite unreachable: {e}")})),
        )
            .into_response(),
    }
}

/// News-embedding health: how much of the text expert's input came from the endpoint and how much
/// from the hashed fallback (D-117). Read from Postgres directly — it is one aggregate.
async fn embeddings(State(s): State<AppState>) -> Response {
    let Some(pg) = s.pg.get() else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "postgres unavailable"}))).into_response();
    };
    let rows = sqlx::query_as::<_, (String, i64, Option<chrono::DateTime<chrono::Utc>>)>(
        "SELECT provider, count(*), max(ts) FROM llm.news_embedding GROUP BY provider",
    )
    .fetch_all(pg)
    .await;
    match rows {
        Ok(rows) => {
            let total: i64 = rows.iter().map(|(_, n, _)| *n).sum();
            let by: Vec<Value> = rows
                .iter()
                .map(|(p, n, newest)| {
                    json!({"provider": p, "count": n, "newest": newest.map(|t| t.to_rfc3339())})
                })
                .collect();
            let hashed = rows.iter().find(|(p, _, _)| p == "hashed").map(|(_, n, _)| *n).unwrap_or(0);
            Json(json!({
                "total": total,
                "by_provider": by,
                // The share that is fallback: a latent space built mostly on hashed vectors is a
                // different space, and this is the number that says so.
                "hashed_share": if total > 0 { hashed as f64 / total as f64 } else { 0.0 },
            }))
            .into_response()
        }
        Err(e) => (StatusCode::BAD_GATEWAY, Json(json!({"error": e.to_string()}))).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct EndpointBody {
    /// Base URL of an OpenAI-compatible API. Empty string clears it (the suite then runs on the
    /// hashed fallback and the agent says it has no endpoint).
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub embedding_model: Option<String>,
    #[serde(default)]
    pub embedding_dim: Option<usize>,
    /// Written to `config/secrets.toml` and never returned. Omit to leave the stored key alone;
    /// send `""` to remove it.
    #[serde(default)]
    pub api_key: Option<String>,
}

/// `PUT /api/llm/endpoint`: writes `[llm]` to the runtime config and the key to the secrets file.
///
/// Two files because they have different lifetimes: the endpoint and model names belong with the
/// other operator-tunable settings (`config/runtime.toml`, which the platform already merges over
/// the platform file), while the key belongs only in the git-ignored secrets file. Both are
/// rewritten atomically — written next to the target and renamed — so a crash mid-write cannot
/// leave the platform with half a config.
async fn set_endpoint(State(s): State<AppState>, Json(body): Json<EndpointBody>) -> Response {
    if let Some(u) = &body.base_url
        && !u.trim().is_empty()
        && !(u.starts_with("http://") || u.starts_with("https://"))
    {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "base_url must be http(s)"}))).into_response();
    }
    if let Some(d) = body.embedding_dim
        && (d == 0 || d > 16_384)
    {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "embedding_dim out of range"}))).into_response();
    }

    let mut changed: Vec<&str> = Vec::new();
    if body.base_url.is_some() {
        changed.push("base_url");
    }
    if body.model.is_some() {
        changed.push("model");
    }
    if body.embedding_model.is_some() {
        changed.push("embedding_model");
    }
    if body.embedding_dim.is_some() {
        changed.push("embedding_dim");
    }
    if body.api_key.is_some() {
        changed.push("api_key");
    }
    if changed.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "nothing to set"}))).into_response();
    }

    if let Err(e) = write_runtime_llm(&fa_core::config::runtime_path(config_path()), &body) {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e}))).into_response();
    }
    if let Some(key) = &body.api_key
        && let Err(e) = write_secret_key(&secrets_path(), key)
    {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e}))).into_response();
    }

    // The audit row names the fields, never their values (spec 50 §3 / D-086).
    crate::audit::record(&s, "llm.endpoint.set", Some("llm"), json!({"fields": changed})).await;
    Json(json!({
        "ok": true,
        "fields": changed,
        // The suite reads its config at start, so a change needs a restart to take effect. Saying
        // so is better than a UI that looks applied and is not.
        "restart_required": true,
        "note": "restart llm-suite to pick this up (POST /api/services/llm-suite/restart)",
    }))
    .into_response()
}

/// Merges `[llm]` into `config/runtime.toml` next to the platform config.
fn write_runtime_llm(path: &std::path::Path, body: &EndpointBody) -> Result<(), String> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut doc: toml::Table = toml::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))?;
    let entry = doc.entry("llm".to_owned()).or_insert_with(|| toml::Value::Table(Default::default()));
    let table = entry.as_table_mut().ok_or_else(|| "[llm] is not a table".to_owned())?;
    if let Some(v) = &body.base_url {
        table.insert("base_url".into(), toml::Value::String(v.trim().to_owned()));
    }
    if let Some(v) = &body.model {
        table.insert("model".into(), toml::Value::String(v.trim().to_owned()));
    }
    if let Some(v) = &body.embedding_model {
        table.insert("embedding_model".into(), toml::Value::String(v.trim().to_owned()));
    }
    if let Some(v) = body.embedding_dim {
        table.insert("embedding_dim".into(), toml::Value::Integer(v as i64));
    }
    let out = toml::to_string_pretty(&doc).map_err(|e| e.to_string())?;
    atomic_write(path, &out)
}

/// Merges `[llm] api_key` into the secrets file.
fn write_secret_key(path: &std::path::Path, key: &str) -> Result<(), String> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut doc: toml::Table = toml::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))?;
    let entry = doc.entry("llm".to_owned()).or_insert_with(|| toml::Value::Table(Default::default()));
    let table = entry.as_table_mut().ok_or_else(|| "[llm] is not a table".to_owned())?;
    if key.trim().is_empty() {
        table.remove("api_key");
    } else {
        table.insert("api_key".into(), toml::Value::String(key.trim().to_owned()));
    }
    let out = toml::to_string_pretty(&doc).map_err(|e| e.to_string())?;
    atomic_write(path, &out)
}

/// Write-then-rename, so a reader never sees a partial file and a crash cannot truncate the
/// existing one. The temporary file is created next to the target so the rename stays on one
/// filesystem, and it is removed if the rename fails.
fn atomic_write(path: &std::path::Path, contents: &str) -> Result<(), String> {
    if let Some(dir) = path.parent()
        && !dir.as_os_str().is_empty()
    {
        std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
    }
    let tmp = path.with_extension("tmp-write");
    std::fs::write(&tmp, contents).map_err(|e| format!("writing {}: {e}", tmp.display()))?;
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("replacing {}: {e}", path.display()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fa-llm-routes-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_key_is_written_to_the_secrets_file_and_an_empty_one_removes_it() {
        let dir = tmp("key");
        let secrets = dir.join("secrets.toml");

        write_secret_key(&secrets, "sk-written-here").unwrap();
        let text = std::fs::read_to_string(&secrets).unwrap();
        let parsed = fa_core::secrets::Secrets::parse(&text).unwrap();
        assert_eq!(parsed.llm.api_key, "sk-written-here");
        assert!(!format!("{parsed:?}").contains("sk-written-here"), "debug output stays redacted");

        write_secret_key(&secrets, "").unwrap();
        let parsed = fa_core::secrets::Secrets::parse(&std::fs::read_to_string(&secrets).unwrap()).unwrap();
        assert!(!parsed.llm.is_configured(), "an empty key is removed, not stored blank");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_existing_secret_survives_an_unrelated_write() {
        let dir = tmp("keep");
        let secrets = dir.join("secrets.toml");
        std::fs::write(&secrets, "[garage]\naccess_key_id = \"gk\"\nsecret_access_key = \"gs\"\n").unwrap();

        write_secret_key(&secrets, "sk-new").unwrap();
        let parsed = fa_core::secrets::Secrets::parse(&std::fs::read_to_string(&secrets).unwrap()).unwrap();
        assert_eq!(parsed.garage.access_key_id, "gk", "the Garage credentials must not be dropped");
        assert_eq!(parsed.llm.api_key, "sk-new");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_endpoint_goes_to_the_runtime_overlay_and_the_key_never_does() {
        let dir = tmp("runtime");
        let runtime = dir.join("runtime.toml");
        let body = EndpointBody {
            base_url: Some("https://api.example.com/v1".into()),
            model: Some("gpt-4o-mini".into()),
            embedding_model: None,
            embedding_dim: Some(1536),
            api_key: Some("sk-must-not-appear".into()),
        };
        write_runtime_llm(&runtime, &body).unwrap();
        let text = std::fs::read_to_string(&runtime).unwrap();
        assert!(text.contains("https://api.example.com/v1"));
        assert!(text.contains("1536"));
        assert!(!text.contains("sk-must-not-appear"), "the key belongs only in the secrets file: {text}");

        // A second write keeps what it does not set.
        write_runtime_llm(&runtime, &EndpointBody {
            base_url: None,
            model: Some("other-model".into()),
            embedding_model: None,
            embedding_dim: None,
            api_key: None,
        })
        .unwrap();
        let text = std::fs::read_to_string(&runtime).unwrap();
        assert!(text.contains("https://api.example.com/v1"), "base_url survived: {text}");
        assert!(text.contains("other-model"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_mcp_server_id_is_a_slug_so_it_cannot_reshape_the_proxied_path() {
        for ok in ["docs", "my-server", "a_b", "s3"] {
            assert!(valid_server_id(ok), "{ok}");
        }
        for bad in ["", "Docs", "has space", "../admin", "a/b", "a?b", &"x".repeat(33)] {
            assert!(!valid_server_id(bad), "{bad:?} must be refused before it reaches the suite");
        }
    }

    #[test]
    fn atomic_write_replaces_and_leaves_no_temporary_behind() {
        let dir = tmp("atomic");
        let p = dir.join("x.toml");
        atomic_write(&p, "a = 1\n").unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "a = 1\n");
        assert!(!p.with_extension("tmp-write").exists());
        atomic_write(&p, "a = 2\n").unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "a = 2\n", "a rewrite replaces, not appends");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
