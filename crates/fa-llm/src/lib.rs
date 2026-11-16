//! Client for an OpenAI-compatible endpoint: chat completions for the agent suite, embeddings for
//! the encoder's text expert (D-117).
//!
//! One crate for both because they are one endpoint with one key, and because the two callers want
//! the same things from it: a configured-or-not answer rather than a panic, bounded retries, and
//! metrics that say what was spent.
//!
//! - [`LlmClient::from_env`] returns `None` when no endpoint is configured. That is not an error:
//!   the agent then says it has no endpoint, and the embedder falls back to hashed embeddings
//!   instead of stalling the model path. Nothing in the platform requires an external API to run.
//! - The key comes from `config/secrets.toml` (`[llm] api_key`, `FA_LLM_API_KEY`) and is never
//!   logged, never returned by a getter and never put in an error message — [`LlmError::Status`]
//!   carries the response body, which providers do not echo credentials into, and
//!   [`redact_body`] trims it to a bounded prefix anyway.
//! - Embeddings are validated against [`fa_core::config::LlmConfig::embedding_dim`]. A collection's
//!   dimension is fixed when it is created, so a model that silently returns a different width
//!   would corrupt the store; a mismatch is refused instead ([`LlmError::DimMismatch`]).
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

pub use fa_core::config::LlmConfig;
pub use fa_core::secrets::LlmSecrets;

/// How many times a request is retried on a transport error or a retryable status (429, 5xx).
pub const RETRIES: u32 = 2;

static REQUESTS: LazyLock<fa_metrics::Counter> =
    LazyLock::new(|| fa_metrics::counter("fa_llm_requests_total", "LLM endpoint requests by kind and outcome"));
static TOKENS: LazyLock<fa_metrics::Counter> =
    LazyLock::new(|| fa_metrics::counter("fa_llm_tokens_total", "LLM tokens billed by kind"));
static LATENCY: LazyLock<fa_metrics::Histogram> = LazyLock::new(|| {
    fa_metrics::histogram(
        "fa_llm_request_seconds",
        "LLM endpoint request latency",
        &[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    )
});

/// Longest error body kept in a [`LlmError::Status`].
const BODY_LIMIT: usize = 400;

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("llm transport: {0}")]
    Http(String),
    #[error("llm {code}: {body}")]
    Status { code: u16, body: String },
    #[error("llm response shape: {0}")]
    Shape(String),
    #[error("embedding width {got} from model `{model}`, but [llm] embedding_dim is {expected}")]
    DimMismatch { expected: usize, got: usize, model: String },
}

type Result<T> = std::result::Result<T, LlmError>;

/// One message of a chat exchange.
///
/// `role` is `system` | `user` | `assistant` | `tool`. An assistant turn that calls tools carries
/// `tool_calls` and usually an empty `content`; the caller answers each one with a `tool` message
/// quoting its `tool_call_id`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self::of("system", content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::of("user", content)
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::of("assistant", content)
    }

    /// The answer to one tool call, quoting its id so the provider can pair them.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    fn of(role: &str, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into(), tool_calls: Vec::new(), tool_call_id: None }
    }
}

/// A tool call the model asked for.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(default = "function_type")]
    pub r#type: String,
    pub function: FunctionCall,
}

fn function_type() -> String {
    "function".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    /// JSON object as a string — that is how the wire format carries arguments, and a model can
    /// emit text that is not valid JSON, so parsing is the caller's decision.
    #[serde(default)]
    pub arguments: String,
}

impl FunctionCall {
    /// The arguments as JSON, or the parse error. A model that emits malformed arguments is an
    /// ordinary event, not a bug: the caller answers with a `tool` message saying so.
    pub fn parse_arguments(&self) -> std::result::Result<serde_json::Value, String> {
        if self.arguments.trim().is_empty() {
            return Ok(serde_json::json!({}));
        }
        serde_json::from_str(&self.arguments).map_err(|e| e.to_string())
    }
}

/// A tool offered to the model (JSON Schema in `parameters`).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Default)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ToolSpec>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
}

#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub message: ChatMessage,
    pub finish_reason: String,
    pub usage: Usage,
    pub model: String,
}

/// Client for one endpoint. Cheap to clone (the inner `reqwest::Client` is shared).
#[derive(Clone)]
pub struct LlmClient {
    http: reqwest::Client,
    base: String,
    key: String,
    model: String,
    embedding_model: String,
    embedding_dim: usize,
    embed_batch: usize,
}

impl std::fmt::Debug for LlmClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmClient")
            .field("base", &self.base)
            .field("model", &self.model)
            .field("embedding_model", &self.embedding_model)
            .field("embedding_dim", &self.embedding_dim)
            .field("key", &if self.key.is_empty() { "unset" } else { "set" })
            .finish()
    }
}

impl LlmClient {
    /// Builds a client from `[llm]` and `[llm] api_key`, with `FA_LLM_BASE_URL`, `FA_LLM_MODEL`,
    /// `FA_LLM_EMBEDDING_MODEL` and `FA_LLM_EMBEDDING_DIM` overriding the file.
    ///
    /// `None` when no `base_url` is configured. A base URL without a key is allowed on purpose:
    /// local servers (llama.cpp, vLLM, Ollama's OpenAI shim) take no key, and refusing them would
    /// make the suite dependent on a paid provider.
    pub fn from_env(cfg: &LlmConfig, secrets: &LlmSecrets) -> Option<Self> {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        let base = env("FA_LLM_BASE_URL").unwrap_or_else(|| cfg.base_url.clone());
        let base = base.trim().trim_end_matches('/').to_owned();
        if base.is_empty() {
            return None;
        }
        let dim = env("FA_LLM_EMBEDDING_DIM")
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|d| *d > 0)
            .unwrap_or(cfg.embedding_dim);
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_secs.max(1)))
            .build()
            .unwrap_or_default();
        Some(Self {
            http,
            base,
            key: env("FA_LLM_API_KEY").unwrap_or_else(|| secrets.api_key.clone()),
            model: env("FA_LLM_MODEL").unwrap_or_else(|| cfg.model.clone()),
            embedding_model: env("FA_LLM_EMBEDDING_MODEL").unwrap_or_else(|| cfg.embedding_model.clone()),
            embedding_dim: dim,
            embed_batch: cfg.embed_batch.max(1),
        })
    }

    /// A client against `base_url` with no key, for tests and local servers.
    pub fn new(base_url: &str, model: &str, embedding_model: &str, embedding_dim: usize) -> Self {
        Self {
            http: reqwest::Client::builder().timeout(Duration::from_secs(60)).build().unwrap_or_default(),
            base: base_url.trim().trim_end_matches('/').to_owned(),
            key: String::new(),
            model: model.into(),
            embedding_model: embedding_model.into(),
            embedding_dim,
            embed_batch: 64,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base
    }
    pub fn model(&self) -> &str {
        &self.model
    }
    pub fn embedding_model(&self) -> &str {
        &self.embedding_model
    }
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    /// Whether a key is set (never the key itself).
    pub fn has_key(&self) -> bool {
        !self.key.is_empty()
    }

    /// One chat completion. Non-streaming: the agent loop needs the whole message, tool calls
    /// included, before it can act on it.
    pub async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": req.messages,
            "stream": false,
        });
        if !req.tools.is_empty() {
            let tools: Vec<serde_json::Value> = req
                .tools
                .iter()
                .map(|t| serde_json::json!({"type": "function", "function": {
                    "name": t.name, "description": t.description, "parameters": t.parameters }}))
                .collect();
            body["tools"] = serde_json::Value::Array(tools);
        }
        if let Some(t) = req.temperature {
            body["temperature"] = serde_json::json!(t);
        }
        if let Some(m) = req.max_tokens {
            body["max_tokens"] = serde_json::json!(m);
        }

        let v = self.post("/chat/completions", &body, "chat").await?;
        let choice = v
            .pointer("/choices/0")
            .ok_or_else(|| LlmError::Shape("no choices[0] in the response".into()))?;
        let message: ChatMessage = serde_json::from_value(
            choice.get("message").cloned().ok_or_else(|| LlmError::Shape("no choices[0].message".into()))?,
        )
        .map_err(|e| LlmError::Shape(e.to_string()))?;
        let usage: Usage = v.get("usage").cloned().and_then(|u| serde_json::from_value(u).ok()).unwrap_or_default();
        TOKENS.add_with(&[("kind", "prompt")], usage.prompt_tokens as f64);
        TOKENS.add_with(&[("kind", "completion")], usage.completion_tokens as f64);
        Ok(ChatResponse {
            message,
            finish_reason: choice.get("finish_reason").and_then(|f| f.as_str()).unwrap_or_default().to_owned(),
            usage,
            model: v.get("model").and_then(|m| m.as_str()).unwrap_or(&self.model).to_owned(),
        })
    }

    /// Embeddings for `inputs`, in the same order, batched at `[llm] embed_batch`.
    ///
    /// Every vector is checked against `embedding_dim`: see [`LlmError::DimMismatch`].
    pub async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(inputs.len());
        for chunk in inputs.chunks(self.embed_batch) {
            let body = serde_json::json!({ "model": self.embedding_model, "input": chunk });
            let v = self.post("/embeddings", &body, "embed").await?;
            let data = v
                .get("data")
                .and_then(|d| d.as_array())
                .ok_or_else(|| LlmError::Shape("no data[] in the embeddings response".into()))?;
            if data.len() != chunk.len() {
                return Err(LlmError::Shape(format!("asked for {} embeddings, got {}", chunk.len(), data.len())));
            }
            // `index` orders the response; providers may return them out of order.
            let mut rows: Vec<(usize, Vec<f32>)> = Vec::with_capacity(data.len());
            for (i, d) in data.iter().enumerate() {
                let idx = d.get("index").and_then(|i| i.as_u64()).map(|i| i as usize).unwrap_or(i);
                let vec: Vec<f32> = d
                    .get("embedding")
                    .and_then(|e| e.as_array())
                    .ok_or_else(|| LlmError::Shape("no embedding[] in data[]".into()))?
                    .iter()
                    .map(|x| x.as_f64().unwrap_or(f64::NAN) as f32)
                    .collect();
                if vec.len() != self.embedding_dim {
                    return Err(LlmError::DimMismatch {
                        expected: self.embedding_dim,
                        got: vec.len(),
                        model: self.embedding_model.clone(),
                    });
                }
                if vec.iter().any(|x| !x.is_finite()) {
                    return Err(LlmError::Shape("embedding contains a non-finite value".into()));
                }
                rows.push((idx, vec));
            }
            rows.sort_by_key(|(i, _)| *i);
            out.extend(rows.into_iter().map(|(_, v)| v));
        }
        Ok(out)
    }

    /// POST with bounded retries on transport errors and retryable statuses.
    async fn post(&self, path: &str, body: &serde_json::Value, kind: &'static str) -> Result<serde_json::Value> {
        let url = format!("{}{path}", self.base);
        let mut attempt = 0;
        loop {
            let started = Instant::now();
            let mut req = self.http.post(&url).json(body);
            if !self.key.is_empty() {
                req = req.bearer_auth(&self.key);
            }
            let result = req.send().await;
            let outcome: Result<serde_json::Value> = match result {
                Ok(resp) => {
                    let code = resp.status().as_u16();
                    if resp.status().is_success() {
                        LATENCY.observe_with(&[("kind", kind)], started.elapsed().as_secs_f64());
                        REQUESTS.inc_with(&[("kind", kind), ("outcome", "ok")]);
                        return resp.json::<serde_json::Value>().await.map_err(|e| LlmError::Shape(e.to_string()));
                    }
                    let body = resp.text().await.unwrap_or_default();
                    Err(LlmError::Status { code, body: redact_body(&body) })
                }
                Err(e) => Err(LlmError::Http(e.to_string())),
            };
            let err = outcome.unwrap_err();
            let retryable = match &err {
                LlmError::Http(_) => true,
                LlmError::Status { code, .. } => *code == 429 || *code >= 500,
                _ => false,
            };
            REQUESTS.inc_with(&[("kind", kind), ("outcome", if retryable { "retryable" } else { "failed" })]);
            if !retryable || attempt >= RETRIES {
                return Err(err);
            }
            attempt += 1;
            let backoff = Duration::from_millis(250 * 2u64.pow(attempt));
            tracing::warn!(kind, attempt, error = %err, "llm request failed; retrying");
            tokio::time::sleep(backoff).await;
        }
    }
}

/// An error body trimmed to [`BODY_LIMIT`] with newlines flattened, so one provider error cannot
/// fill the log or an audit row.
pub fn redact_body(body: &str) -> String {
    let flat: String = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.len() <= BODY_LIMIT { flat } else { format!("{}…", &flat[..BODY_LIMIT]) }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::{Json, Router, extract::State, routing::post};

    use super::*;

    /// Requests the stub saw, so a test can assert what went out as well as what came back.
    type Seen = Arc<Mutex<Vec<serde_json::Value>>>;

    async fn stub(chat: serde_json::Value, embed: serde_json::Value) -> (String, Seen) {
        let seen: Seen = Arc::new(Mutex::new(Vec::new()));
        let state = (seen.clone(), chat, embed);
        let app = Router::new()
            .route(
                "/chat/completions",
                post(|State((seen, chat, _)): State<(Seen, serde_json::Value, serde_json::Value)>, Json(b): Json<serde_json::Value>| async move {
                    seen.lock().unwrap().push(b);
                    Json(chat)
                }),
            )
            .route(
                "/embeddings",
                post(|State((seen, _, embed)): State<(Seen, serde_json::Value, serde_json::Value)>, Json(b): Json<serde_json::Value>| async move {
                    seen.lock().unwrap().push(b);
                    Json(embed)
                }),
            )
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}"), seen)
    }

    fn embeddings(vectors: &[&[f32]]) -> serde_json::Value {
        serde_json::json!({
            "data": vectors.iter().enumerate().map(|(i, v)| serde_json::json!({"index": i, "embedding": v}))
                .collect::<Vec<_>>(),
            "model": "stub-embed",
        })
    }

    #[tokio::test]
    async fn chat_sends_tools_and_parses_tool_calls() {
        let reply = serde_json::json!({
            "model": "stub",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "trading_risk", "arguments": "{\"book\":\"paper\"}"}
                    }]
                }
            }],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        });
        let (base, seen) = stub(reply, serde_json::json!({})).await;
        let c = LlmClient::new(&base, "m", "e", 4);
        let out = c
            .chat(&ChatRequest {
                messages: vec![ChatMessage::system("be useful"), ChatMessage::user("what is the risk?")],
                tools: vec![ToolSpec {
                    name: "trading_risk".into(),
                    description: "read the risk state".into(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                }],
                temperature: Some(0.0),
                max_tokens: Some(256),
            })
            .await
            .expect("chat");

        assert_eq!(out.finish_reason, "tool_calls");
        assert_eq!(out.usage, Usage { prompt_tokens: 11, completion_tokens: 3 });
        let call = &out.message.tool_calls[0];
        assert_eq!(call.function.name, "trading_risk");
        assert_eq!(call.function.parse_arguments().unwrap()["book"], "paper");

        let sent = &seen.lock().unwrap()[0];
        assert_eq!(sent["model"], "m");
        assert_eq!(sent["stream"], false, "the agent loop needs whole messages");
        assert_eq!(sent["tools"][0]["function"]["name"], "trading_risk");
        assert_eq!(sent["tools"][0]["type"], "function");
        assert_eq!(sent["messages"][1]["content"], "what is the risk?");
        assert!(sent["messages"][0].get("tool_calls").is_none(), "empty fields stay off the wire");
    }

    #[tokio::test]
    async fn embeddings_are_batched_ordered_and_width_checked() {
        // Two inputs per request, returned out of order on purpose.
        let reply = serde_json::json!({
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ],
            "model": "stub-embed",
        });
        let (base, seen) = stub(serde_json::json!({}), reply).await;
        let mut c = LlmClient::new(&base, "m", "e", 2);
        c.embed_batch = 2;
        let out = c.embed(&["a".into(), "b".into(), "c".into(), "d".into()]).await.expect("embed");
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], vec![1.0, 0.0], "ordered by `index`, not arrival");
        assert_eq!(out[1], vec![0.0, 1.0]);
        assert_eq!(seen.lock().unwrap().len(), 2, "4 inputs at batch 2 = 2 requests");
        assert_eq!(seen.lock().unwrap()[0]["input"], serde_json::json!(["a", "b"]));
    }

    #[tokio::test]
    async fn a_different_embedding_width_is_refused() {
        let (base, _) = stub(serde_json::json!({}), embeddings(&[&[1.0, 2.0, 3.0]])).await;
        let c = LlmClient::new(&base, "m", "text-embedding-3-small", 1536);
        let err = c.embed(&["a".into()]).await.expect_err("must refuse");
        assert!(
            matches!(err, LlmError::DimMismatch { expected: 1536, got: 3, .. }),
            "a collection's width is fixed at creation, so a silent change must not be written: {err}"
        );
    }

    #[tokio::test]
    async fn a_four_hundred_is_not_retried() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(Mutex::new(0usize));
        let h = hits.clone();
        let app = Router::new().route(
            "/embeddings",
            post(move || {
                let h = h.clone();
                async move {
                    *h.lock().unwrap() += 1;
                    (axum::http::StatusCode::BAD_REQUEST, "model not found")
                }
            }),
        );
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        let c = LlmClient::new(&format!("http://{addr}"), "m", "e", 2);
        let err = c.embed(&["a".into()]).await.expect_err("must fail");
        assert!(matches!(err, LlmError::Status { code: 400, .. }), "{err}");
        assert_eq!(*hits.lock().unwrap(), 1, "a client error is not retried");
    }

    #[test]
    fn unconfigured_is_none_not_an_error() {
        let cfg = LlmConfig::default();
        assert!(!cfg.is_configured());
        assert!(
            LlmClient::from_env(&cfg, &LlmSecrets::default()).is_none(),
            "no endpoint is a state the platform runs in, not a failure"
        );
    }

    #[test]
    fn a_base_url_without_a_key_is_allowed() {
        let cfg = LlmConfig { base_url: "http://127.0.0.1:11434/v1/".into(), ..Default::default() };
        let c = LlmClient::from_env(&cfg, &LlmSecrets::default()).expect("configured");
        assert_eq!(c.base_url(), "http://127.0.0.1:11434/v1", "trailing slash trimmed once");
        assert!(!c.has_key(), "local servers take no key");
        assert!(!format!("{c:?}").contains("api_key"), "debug never prints a key field value");
    }

    #[test]
    fn the_key_never_appears_in_debug() {
        let cfg = LlmConfig { base_url: "https://example.invalid/v1".into(), ..Default::default() };
        let c = LlmClient::from_env(&cfg, &LlmSecrets { api_key: "sk-secret-value".into() }).expect("configured");
        let shown = format!("{c:?}");
        assert!(c.has_key());
        assert!(!shown.contains("sk-secret-value"), "the key must not reach any log line: {shown}");
        assert!(shown.contains("key: \"set\""));
    }

    #[test]
    fn error_bodies_are_bounded_and_flat() {
        let long = "x".repeat(5_000);
        let out = redact_body(&format!("line one\nline two {long}"));
        assert!(out.len() <= BODY_LIMIT + 4);
        assert!(!out.contains('\n'));
    }
}
