//! External MCP servers: a JSON-RPC client over the streamable-HTTP transport, and the registry
//! that merges their tools into the agent's catalogue (D-120).
//!
//! This is the one place the agent reaches outside the platform. Three properties make that
//! acceptable:
//!
//! - **An MCP tool is a tool like any other.** It arrives in [`crate::tools::Caller::call`]
//!   through the same path as a control-api tool, so the approval rule, the live-mode interlock
//!   and the `llm.tool_call` audit row apply unchanged. A server cannot hand the agent authority
//!   the operator did not give it.
//! - **The operator classifies the server, not the server itself.** MCP lets a tool advertise
//!   `annotations.readOnlyHint`, and that hint is a *self-description*: a server that wants to be
//!   trusted will set it. So the registration carries the access level and the hint is only
//!   reported. A server registered `mutating` (the default) needs per-call approval for every one
//!   of its tools.
//! - **Everything a server returns is data.** Results are flattened to text, truncated at
//!   [`MAX_RESULT_BYTES`], and handed to the model as a tool result — never spliced into the
//!   system prompt. The prompt says so too, because a remote tool result is the obvious place for
//!   an injected instruction to arrive.
//!
//! Only the streamable-HTTP transport is implemented. stdio would mean the suite spawning
//! processes inside its own container, which is a much larger trust decision than calling a URL,
//! and the operator can put a local stdio server behind one of the many HTTP bridges.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};
use sqlx::{PgPool, Row};

use crate::tools::Access;

/// The protocol revision this client announces. A server that answers with another revision is
/// still used: the three methods used here (`initialize`, `tools/list`, `tools/call`) have been
/// stable across revisions, and refusing would turn a working server into a dead one.
pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// Most tools taken from one server. A server with hundreds of tools would otherwise fill the
/// model's context with a catalogue and leave no room for the platform's own.
pub const MAX_TOOLS: usize = 48;

/// Most bytes of one tool result handed to the model.
pub const MAX_RESULT_BYTES: usize = 16 * 1024;

const PROBE_TIMEOUT: Duration = Duration::from_secs(20);
const CALL_TIMEOUT: Duration = Duration::from_secs(60);

/// How often the registry re-probes its servers (a server restarts, its tool list changes).
pub const REPROBE: Duration = Duration::from_secs(300);

/// One registered server, as the operator entered it. The token is never part of this struct's
/// `Debug` output and never leaves the suite.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct Server {
    pub id: String,
    pub url: String,
    pub label: String,
    pub access: Access,
    pub enabled: bool,
    /// Bearer token, if the server needs one. Held in Postgres, never returned by any route.
    pub token: Option<String>,
}

impl std::fmt::Debug for Server {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Server")
            .field("id", &self.id)
            .field("url", &self.url)
            .field("access", &self.access)
            .field("enabled", &self.enabled)
            .field("token", &if self.token.is_some() { "***" } else { "" })
            .finish()
    }
}

/// A tool a server offers, under the name the model sees.
#[derive(Debug, Clone, PartialEq)]
pub struct RemoteTool {
    /// `mcp__<server>__<tool>`: what the model calls.
    pub exposed: String,
    /// What the server calls it.
    pub remote: String,
    pub description: String,
    pub schema: Value,
    /// The server's own `annotations.readOnlyHint`. Reported, never believed (see the module docs).
    pub read_only_hint: bool,
}

/// What the last probe found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// Registered but not probed yet.
    Pending,
    Disabled,
    Ok,
    Error(String),
}

impl Status {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Pending => "pending",
            Self::Disabled => "disabled",
            Self::Ok => "ok",
            Self::Error(_) => "error",
        }
    }
}

/// A registration plus what probing it found.
#[derive(Debug, Clone)]
pub struct Entry {
    pub server: Server,
    pub status: Status,
    pub tools: Vec<RemoteTool>,
    pub probed_at: Option<chrono::DateTime<chrono::Utc>>,
    /// `serverInfo.name serverInfo.version` as reported by `initialize`.
    pub info: Option<String>,
    pub protocol: Option<String>,
}

impl Entry {
    fn new(server: Server) -> Self {
        let status = if server.enabled { Status::Pending } else { Status::Disabled };
        Self { server, status, tools: Vec::new(), probed_at: None, info: None, protocol: None }
    }

    /// What `GET /api/llm/mcp` reports. The token is reduced to a boolean here, which is the only
    /// form it ever leaves the process in.
    pub fn as_json(&self) -> Value {
        json!({
            "id": self.server.id,
            "url": self.server.url,
            "label": self.server.label,
            "access": self.server.access.as_str(),
            "enabled": self.server.enabled,
            "token_set": self.server.token.is_some(),
            "status": self.status.as_str(),
            "error": match &self.status { Status::Error(e) => Some(e.clone()), _ => None },
            "probed_at": self.probed_at.map(|t| t.to_rfc3339()),
            "info": self.info,
            "protocol": self.protocol,
            "tools": self.tools.iter().map(|t| json!({
                "name": t.exposed,
                "remote": t.remote,
                "description": t.description,
                "read_only_hint": t.read_only_hint,
            })).collect::<Vec<_>>(),
        })
    }
}

/// Every registered server and its discovered tools.
///
/// Cloning shares the state: the API handlers, the agent loop and the re-probe task all hold the
/// same registry.
#[derive(Clone, Default)]
pub struct Registry {
    entries: Arc<Mutex<Vec<Entry>>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Entry>> {
        self.entries.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Replaces the registrations, keeping the probe results of servers that did not change.
    ///
    /// Keeping them matters: the registry is reloaded from Postgres on a timer, and dropping every
    /// tool list on each reload would make the catalogue flicker between reload and re-probe.
    pub fn set_servers(&self, servers: Vec<Server>) {
        let mut entries = self.lock();
        let mut next: Vec<Entry> = Vec::with_capacity(servers.len());
        for s in servers {
            match entries.iter().find(|e| e.server.id == s.id && e.server == s) {
                Some(old) => next.push(old.clone()),
                None => next.push(Entry::new(s)),
            }
        }
        *entries = next;
    }

    pub fn entries(&self) -> Vec<Entry> {
        self.lock().clone()
    }

    /// The tools of every enabled, successfully probed server, as the chat API wants them.
    pub fn specs(&self) -> Vec<fa_llm::ToolSpec> {
        self.lock()
            .iter()
            .filter(|e| e.server.enabled && e.status == Status::Ok)
            .flat_map(|e| {
                let server = e.server.clone();
                e.tools.iter().map(move |t| fa_llm::ToolSpec {
                    name: t.exposed.clone(),
                    description: describe(&server, t),
                    parameters: t.schema.clone(),
                })
            })
            .collect()
    }

    /// The entry and tool behind an exposed name, if an enabled server offers it.
    pub fn find(&self, exposed: &str) -> Option<(Server, RemoteTool)> {
        self.lock().iter().filter(|e| e.server.enabled).find_map(|e| {
            e.tools.iter().find(|t| t.exposed == exposed).map(|t| (e.server.clone(), t.clone()))
        })
    }

    /// Probes one server and stores the outcome. Returns the status it ended in.
    pub async fn probe(&self, id: &str) -> Status {
        let Some(entry) = self.lock().iter().find(|e| e.server.id == id).cloned() else {
            return Status::Error("not registered".into());
        };
        if !entry.server.enabled {
            self.store(id, Status::Disabled, Vec::new(), None, None);
            return Status::Disabled;
        }
        let client = Client::new(&entry.server.url, entry.server.token.clone());
        let (status, tools, info, protocol) = match client.initialize().await {
            Err(e) => (Status::Error(e), Vec::new(), None, None),
            Ok(hello) => match client.list_tools(&entry.server.id).await {
                Ok(tools) => (Status::Ok, tools, Some(hello.info), Some(hello.protocol)),
                Err(e) => (Status::Error(e), Vec::new(), Some(hello.info), Some(hello.protocol)),
            },
        };
        self.store(id, status.clone(), tools, info, protocol);
        status
    }

    /// Probes every registration, one after another (a probe is two round trips; there are few
    /// servers, and serialising keeps a slow server from being hidden behind a fast one).
    pub async fn probe_all(&self) {
        let ids: Vec<String> = self.lock().iter().map(|e| e.server.id.clone()).collect();
        for id in ids {
            let status = self.probe(&id).await;
            match &status {
                Status::Error(e) => tracing::warn!(server = %id, error = %e, "probing MCP server failed"),
                Status::Ok => {
                    let n = self.lock().iter().find(|x| x.server.id == id).map_or(0, |x| x.tools.len());
                    tracing::info!(server = %id, tools = n, "MCP server ready");
                }
                _ => {}
            }
        }
    }

    /// Fills a server's tool list without a server, for the tests of the calling path.
    #[cfg(test)]
    pub fn set_tools_for_test(&self, id: &str, names: &[&str]) {
        let tools = names
            .iter()
            .map(|n| RemoteTool {
                exposed: exposed_name(id, n),
                remote: (*n).to_owned(),
                description: String::new(),
                schema: json!({"type": "object", "properties": {}}),
                read_only_hint: false,
            })
            .collect();
        self.store(id, Status::Ok, tools, None, None);
    }

    fn store(&self, id: &str, status: Status, tools: Vec<RemoteTool>, info: Option<String>, protocol: Option<String>) {
        let mut entries = self.lock();
        if let Some(e) = entries.iter_mut().find(|e| e.server.id == id) {
            e.status = status;
            e.tools = tools;
            e.probed_at = Some(chrono::Utc::now());
            if info.is_some() {
                e.info = info;
            }
            if protocol.is_some() {
                e.protocol = protocol;
            }
        }
    }

    /// Calls one MCP tool. The caller has already applied the approval and live-mode rules.
    pub async fn call(&self, exposed: &str, args: &Value) -> Result<Value, String> {
        let (server, tool) = self.find(exposed).ok_or_else(|| format!("no MCP tool named `{exposed}`"))?;
        let client = Client::new(&server.url, server.token.clone());
        // Each call initialises its own session. That costs one extra round trip and buys
        // statelessness: a server that was restarted, or that expired the session, works anyway,
        // and nothing in the suite has to track session lifetimes.
        client.initialize().await?;
        client.call_tool(&tool.remote, args).await
    }
}

/// What the model is told about an MCP tool: the server's description, plus where it came from and
/// whether running it needs the operator. The provenance is part of the description because the
/// model should be able to say "that number came from an external server", and because a tool
/// whose description claims otherwise is then visibly contradicted.
fn describe(server: &Server, tool: &RemoteTool) -> String {
    let what = if tool.description.is_empty() { "no description given" } else { tool.description.as_str() };
    let mutating = if server.access == Access::Mutating { " Mutating: needs operator approval." } else { "" };
    format!("[external MCP server `{}`] {what}{mutating}", server.id)
}

/// A valid server id: a short slug, because it becomes part of a tool name the chat API must
/// accept (`^[a-zA-Z0-9_-]{1,64}$`) and part of a URL path.
pub fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 32
        && id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
        && id.starts_with(|c: char| c.is_ascii_lowercase() || c.is_ascii_digit())
}

/// `mcp__<server>__<tool>`, with everything the chat API would reject replaced and the whole kept
/// inside the 64-character limit.
pub fn exposed_name(server: &str, remote: &str) -> String {
    let safe: String = remote
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    let prefix = format!("mcp__{server}__");
    let room = 64usize.saturating_sub(prefix.len()).max(1);
    let mut out = prefix;
    out.extend(safe.chars().take(room));
    out
}

/// Server info from `initialize`.
#[derive(Debug, Clone)]
pub struct Hello {
    pub info: String,
    pub protocol: String,
}

/// One MCP server over streamable HTTP.
pub struct Client {
    http: reqwest::Client,
    url: String,
    token: Option<String>,
    session: Mutex<Option<String>>,
    next_id: AtomicU64,
}

impl Client {
    pub fn new(url: &str, token: Option<String>) -> Self {
        Self {
            http: reqwest::Client::builder().build().unwrap_or_default(),
            url: url.to_owned(),
            token,
            session: Mutex::new(None),
            next_id: AtomicU64::new(1),
        }
    }

    fn request(&self, timeout: Duration) -> reqwest::RequestBuilder {
        let mut req = self
            .http
            .post(&self.url)
            .timeout(timeout)
            // Both, because a streamable-HTTP server may answer either way for the same request.
            .header("accept", "application/json, text/event-stream")
            .header("mcp-protocol-version", PROTOCOL_VERSION);
        if let Some(t) = &self.token {
            req = req.header("authorization", format!("Bearer {t}"));
        }
        if let Some(s) = self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).as_deref() {
            req = req.header("mcp-session-id", s);
        }
        req
    }

    /// One JSON-RPC call. Returns the `result` member, or the error as text.
    async fn send(&self, method: &str, params: Value, timeout: Duration) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let body = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let resp = self
            .request(timeout)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("{method}: {}", redact_url(&e.to_string())))?;
        if let Some(sid) = resp.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()) {
            *self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(sid.to_owned());
        }
        let code = resp.status();
        let ctype =
            resp.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or_default().to_owned();
        let text = resp.text().await.unwrap_or_default();
        if !code.is_success() {
            return Err(format!("{method}: HTTP {} {}", code.as_u16(), first_line(&text)));
        }
        let msg = parse_rpc_body(&ctype, &text, id)
            .ok_or_else(|| format!("{method}: no JSON-RPC response in the body ({})", first_line(&text)))?;
        if let Some(err) = msg.get("error") {
            let m = err.get("message").and_then(Value::as_str).unwrap_or("unspecified");
            let c = err.get("code").and_then(Value::as_i64).unwrap_or(0);
            return Err(format!("{method}: server error {c}: {m}"));
        }
        Ok(msg.get("result").cloned().unwrap_or(Value::Null))
    }

    /// A notification: no id, and the server answers with an empty 202.
    async fn notify(&self, method: &str) -> Result<(), String> {
        let body = json!({"jsonrpc": "2.0", "method": method});
        self.request(PROBE_TIMEOUT)
            .json(&body)
            .send()
            .await
            .map(|_| ())
            .map_err(|e| format!("{method}: {}", redact_url(&e.to_string())))
    }

    /// The handshake. Also sends `notifications/initialized`, which servers that follow the spec
    /// wait for before accepting anything else.
    pub async fn initialize(&self) -> Result<Hello, String> {
        let result = self
            .send(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "financial-auto llm-suite", "version": env!("CARGO_PKG_VERSION")},
                }),
                PROBE_TIMEOUT,
            )
            .await?;
        let name = result.pointer("/serverInfo/name").and_then(Value::as_str).unwrap_or("unnamed");
        let version = result.pointer("/serverInfo/version").and_then(Value::as_str).unwrap_or("");
        let protocol = result.get("protocolVersion").and_then(Value::as_str).unwrap_or("").to_owned();
        // A server that rejects the notification is not broken for our purposes, so this is logged
        // rather than returned: the next call will fail loudly if it really does need it.
        if let Err(e) = self.notify("notifications/initialized").await {
            tracing::debug!(error = %e, "MCP initialized notification was not accepted");
        }
        Ok(Hello { info: format!("{name} {version}").trim().to_owned(), protocol })
    }

    /// Every tool the server offers, following `nextCursor` and stopping at [`MAX_TOOLS`].
    pub async fn list_tools(&self, server_id: &str) -> Result<Vec<RemoteTool>, String> {
        let mut out: Vec<RemoteTool> = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..8 {
            let params = match &cursor {
                Some(c) => json!({"cursor": c}),
                None => json!({}),
            };
            let result = self.send("tools/list", params, PROBE_TIMEOUT).await?;
            let tools = result.get("tools").and_then(Value::as_array).cloned().unwrap_or_default();
            for t in tools {
                let Some(remote) = t.get("name").and_then(Value::as_str) else { continue };
                let exposed = exposed_name(server_id, remote);
                // A name collision after sanitising would silently route one tool to another, so
                // the second one is dropped and the first keeps the name.
                if out.iter().any(|x: &RemoteTool| x.exposed == exposed) {
                    tracing::warn!(server = server_id, remote, exposed, "MCP tool name collides; skipped");
                    continue;
                }
                let schema = match t.get("inputSchema") {
                    Some(s) if s.is_object() => s.clone(),
                    // The chat API needs an object schema; a server that omits it gets an empty one.
                    _ => json!({"type": "object", "properties": {}}),
                };
                out.push(RemoteTool {
                    exposed,
                    remote: remote.to_owned(),
                    description: t
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .chars()
                        .take(400)
                        .collect(),
                    schema,
                    read_only_hint: t
                        .pointer("/annotations/readOnlyHint")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                });
                if out.len() >= MAX_TOOLS {
                    tracing::warn!(server = server_id, cap = MAX_TOOLS, "MCP tool list truncated");
                    return Ok(out);
                }
            }
            match result.get("nextCursor").and_then(Value::as_str) {
                Some(c) if !c.is_empty() => cursor = Some(c.to_owned()),
                _ => break,
            }
        }
        Ok(out)
    }

    /// Calls one tool by its remote name.
    pub async fn call_tool(&self, remote: &str, args: &Value) -> Result<Value, String> {
        let result = self
            .send("tools/call", json!({"name": remote, "arguments": args}), CALL_TIMEOUT)
            .await?;
        let is_error = result.get("isError").and_then(Value::as_bool).unwrap_or(false);
        let text = flatten_content(&result);
        Ok(json!({
            "status": if is_error { "tool_error" } else { "ok" },
            // Named `external` so the model's own transcript records that this text came from
            // outside the platform and is not an instruction (see the system prompt).
            "external": truncate(&text, MAX_RESULT_BYTES),
            "structured": result.get("structuredContent").cloned(),
        }))
    }
}

/// The JSON-RPC message with `id` out of a body that is either JSON or an SSE stream.
pub fn parse_rpc_body(content_type: &str, body: &str, id: u64) -> Option<Value> {
    if content_type.contains("text/event-stream") || body.starts_with("event:") || body.starts_with("data:") {
        // One `data:` line per event here; multi-line data payloads are concatenated per the SSE
        // rules before parsing.
        let mut buf = String::new();
        for line in body.lines() {
            if let Some(rest) = line.strip_prefix("data:") {
                buf.push_str(rest.trim_start());
            } else if line.trim().is_empty() && !buf.is_empty() {
                if let Some(v) = matching(&buf, id) {
                    return Some(v);
                }
                buf.clear();
            }
        }
        if !buf.is_empty() {
            return matching(&buf, id);
        }
        return None;
    }
    matching(body, id)
}

fn matching(text: &str, id: u64) -> Option<Value> {
    let v: Value = serde_json::from_str(text.trim()).ok()?;
    // Accept the id as a number or a string, and accept a lone response with no id at all —
    // some servers answer a single request without echoing it.
    match v.get("id") {
        None => Some(v),
        Some(Value::Number(n)) if n.as_u64() == Some(id) => Some(v),
        Some(Value::String(s)) if s == &id.to_string() => Some(v),
        _ => None,
    }
}

/// An MCP result's `content` blocks as one piece of text. Non-text blocks are named rather than
/// embedded: an image the model cannot see is better described than dropped.
pub fn flatten_content(result: &Value) -> String {
    let Some(blocks) = result.get("content").and_then(Value::as_array) else {
        return result.get("structuredContent").map(|v| v.to_string()).unwrap_or_default();
    };
    let mut out = String::new();
    for b in blocks {
        match b.get("type").and_then(Value::as_str) {
            Some("text") => out.push_str(b.get("text").and_then(Value::as_str).unwrap_or_default()),
            Some("resource") => {
                let uri = b.pointer("/resource/uri").and_then(Value::as_str).unwrap_or("?");
                let text = b.pointer("/resource/text").and_then(Value::as_str).unwrap_or_default();
                out.push_str(&format!("[resource {uri}] {text}"));
            }
            Some(other) => out.push_str(&format!("[{other} block omitted]")),
            None => {}
        }
        out.push('\n');
    }
    out.trim_end().to_owned()
}

/// Truncates on a character boundary and says it did.
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_owned();
    }
    let mut cut = max;
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}\n[truncated at {max} bytes]", &s[..cut])
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or_default().chars().take(200).collect()
}

/// reqwest puts the URL in its error text, and an MCP URL can carry a token in a query string.
fn redact_url(msg: &str) -> String {
    match (msg.find("://"), msg.find('?')) {
        (Some(_), Some(q)) => format!("{}?***", &msg[..q]),
        _ => msg.to_owned(),
    }
}

// --- Persistence ------------------------------------------------------------------------------

/// Every registration, oldest first.
pub async fn load(pg: &PgPool) -> Result<Vec<Server>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT id, url, label, access, enabled, token FROM llm.mcp_server ORDER BY created_at, id",
    )
    .fetch_all(pg)
    .await?;
    Ok(rows
        .iter()
        .map(|r| Server {
            id: r.get("id"),
            url: r.get("url"),
            label: r.get("label"),
            access: if r.get::<String, _>("access") == "read" { Access::Read } else { Access::Mutating },
            enabled: r.get("enabled"),
            token: r.get::<Option<String>, _>("token").filter(|t| !t.is_empty()),
        })
        .collect())
}

/// Inserts or updates one registration. `token: None` leaves a stored token alone — the UI cannot
/// read a token back, so an empty field there means "unchanged", and clearing is explicit.
pub async fn upsert(pg: &PgPool, s: &Server, clear_token: bool) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO llm.mcp_server (id, url, label, access, enabled, token) \
         VALUES ($1, $2, $3, $4, $5, $6) \
         ON CONFLICT (id) DO UPDATE SET url = $2, label = $3, access = $4, enabled = $5, \
             token = CASE WHEN $7 THEN NULL WHEN $6 IS NOT NULL THEN $6 ELSE llm.mcp_server.token END, \
             updated_at = now()",
    )
    .bind(&s.id)
    .bind(&s.url)
    .bind(&s.label)
    .bind(s.access.as_str())
    .bind(s.enabled)
    .bind(&s.token)
    .bind(clear_token)
    .execute(pg)
    .await
    .map(|_| ())
}

pub async fn delete(pg: &PgPool, id: &str) -> Result<u64, sqlx::Error> {
    sqlx::query("DELETE FROM llm.mcp_server WHERE id = $1").bind(id).execute(pg).await.map(|r| r.rows_affected())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_server_id_is_a_slug_because_it_becomes_a_tool_name() {
        for ok in ["docs", "my-server", "s3", "a_b"] {
            assert!(valid_id(ok), "{ok}");
        }
        for bad in ["", "Docs", "-x", "_x", "has space", "sl/ash", &"x".repeat(33)] {
            assert!(!valid_id(bad), "{bad:?} must be refused");
        }
    }

    #[test]
    fn exposed_names_are_namespaced_sanitised_and_bounded() {
        assert_eq!(exposed_name("docs", "search"), "mcp__docs__search");
        assert_eq!(exposed_name("docs", "search.pages"), "mcp__docs__search_pages");
        assert_eq!(exposed_name("docs", "a b/c"), "mcp__docs__a_b_c");
        let long = exposed_name("docs", &"t".repeat(100));
        assert_eq!(long.len(), 64, "the chat API's limit: {long}");
        assert!(long.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'), "{long}");
        // Two servers may offer the same tool without colliding.
        assert_ne!(exposed_name("a", "search"), exposed_name("b", "search"));
    }

    #[test]
    fn a_response_is_read_from_json_or_from_an_event_stream() {
        let json_body = r#"{"jsonrpc":"2.0","id":7,"result":{"ok":true}}"#;
        assert_eq!(parse_rpc_body("application/json", json_body, 7).unwrap()["result"]["ok"], true);
        assert!(parse_rpc_body("application/json", json_body, 8).is_none(), "another id is not our answer");

        let sse = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"ok\":true}}\n\n";
        assert_eq!(parse_rpc_body("text/event-stream", sse, 7).unwrap()["result"]["ok"], true);

        // The answer may arrive after other events.
        let sse = "event: ping\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"ping\",\"id\":99}\n\n\
                   event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":1}\n\n";
        assert_eq!(parse_rpc_body("text/event-stream", sse, 7).unwrap()["result"], 1);

        // A string id is accepted, and so is a lone response with none.
        let s = r#"{"jsonrpc":"2.0","id":"7","result":2}"#;
        assert_eq!(parse_rpc_body("application/json", s, 7).unwrap()["result"], 2);
        let s = r#"{"jsonrpc":"2.0","result":3}"#;
        assert_eq!(parse_rpc_body("application/json", s, 7).unwrap()["result"], 3);

        assert!(parse_rpc_body("application/json", "not json", 1).is_none());
    }

    #[test]
    fn content_blocks_flatten_and_long_results_are_cut() {
        let r = json!({"content": [
            {"type": "text", "text": "first"},
            {"type": "image", "data": "…"},
            {"type": "resource", "resource": {"uri": "file:///x", "text": "body"}},
        ]});
        let out = flatten_content(&r);
        assert!(out.contains("first"));
        assert!(out.contains("[image block omitted]"), "{out}");
        assert!(out.contains("[resource file:///x] body"), "{out}");

        // No content, but structured output: that is the result.
        assert_eq!(flatten_content(&json!({"structuredContent": {"a": 1}})), r#"{"a":1}"#);

        let long = "x".repeat(MAX_RESULT_BYTES + 100);
        let cut = truncate(&long, MAX_RESULT_BYTES);
        assert!(cut.len() < long.len() && cut.contains("[truncated"));
        assert_eq!(truncate("short", MAX_RESULT_BYTES), "short");
        // A cut that lands inside a multi-byte character steps back to a boundary.
        let utf8 = "é".repeat(100);
        let cut = truncate(&utf8, 11);
        assert!(cut.starts_with(&"é".repeat(5)) && cut.contains("[truncated"), "{cut}");
        assert_eq!(cut.chars().take_while(|c| *c == 'é').count(), 5, "11 bytes holds five 2-byte chars");
    }

    #[test]
    fn a_token_never_reaches_a_log_line_or_the_api() {
        let s = Server {
            id: "docs".into(),
            url: "https://example.com/mcp".into(),
            label: "Docs".into(),
            access: Access::Read,
            enabled: true,
            token: Some("mcp-secret-token".into()),
        };
        assert!(!format!("{s:?}").contains("mcp-secret-token"), "{s:?}");
        let e = Entry::new(s);
        let json = e.as_json().to_string();
        assert!(!json.contains("mcp-secret-token"), "{json}");
        assert_eq!(e.as_json()["token_set"], true);
        assert_eq!(e.as_json()["status"], "pending");
        assert_eq!(redact_url("error for url (https://x/mcp?token=abc)"), "error for url (https://x/mcp?***");
    }

    #[test]
    fn the_registry_keeps_probe_results_across_an_unchanged_reload() {
        let reg = Registry::new();
        let s = Server {
            id: "docs".into(),
            url: "http://x/mcp".into(),
            label: String::new(),
            access: Access::Read,
            enabled: true,
            token: None,
        };
        reg.set_servers(vec![s.clone()]);
        reg.store(
            "docs",
            Status::Ok,
            vec![RemoteTool {
                exposed: "mcp__docs__search".into(),
                remote: "search".into(),
                description: "Search the docs".into(),
                schema: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
                read_only_hint: true,
            }],
            Some("docs 1.0".into()),
            Some(PROTOCOL_VERSION.into()),
        );
        assert_eq!(reg.specs().len(), 1);
        let spec = &reg.specs()[0];
        assert_eq!(spec.name, "mcp__docs__search");
        assert!(spec.description.contains("external MCP server `docs`"), "{}", spec.description);
        assert!(!spec.description.contains("needs operator approval"), "a read server's tools are free");

        reg.set_servers(vec![s.clone()]);
        assert_eq!(reg.specs().len(), 1, "an unchanged reload keeps the tool list");

        // A changed URL is a different server: its old tools must not answer for the new one.
        reg.set_servers(vec![Server { url: "http://y/mcp".into(), ..s.clone() }]);
        assert!(reg.specs().is_empty(), "a changed registration is unprobed");

        // Disabling hides the tools even while the entry stays.
        reg.set_servers(vec![s.clone()]);
        reg.store("docs", Status::Ok, reg.entries()[0].tools.clone(), None, None);
        reg.set_servers(vec![Server { enabled: false, ..s.clone() }]);
        assert!(reg.specs().is_empty());
        assert!(reg.find("mcp__docs__search").is_none());
    }

    /// A stub MCP server: `json` answers with `application/json`, `sse` with an event stream, so
    /// both transports of the same protocol are exercised against the real client.
    async fn stub(sse: bool) -> (String, tokio::task::JoinHandle<()>) {
        use axum::{Json as AxJson, Router, response::IntoResponse, routing::post};

        let handler = move |AxJson(body): AxJson<Value>| async move {
            let id = body.get("id").cloned();
            let method = body.get("method").and_then(Value::as_str).unwrap_or("").to_owned();
            if id.is_none() {
                // A notification.
                return (axum::http::StatusCode::ACCEPTED, String::new()).into_response();
            }
            let result = match method.as_str() {
                "initialize" => json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "stub", "version": "0.1"},
                }),
                "tools/list" => json!({"tools": [
                    {
                        "name": "search.docs",
                        "description": "Search the docs",
                        "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
                        "annotations": {"readOnlyHint": true},
                    },
                    {"name": "no_schema"},
                ]}),
                "tools/call" => {
                    let name = body.pointer("/params/name").and_then(Value::as_str).unwrap_or("");
                    let q = body.pointer("/params/arguments/q").and_then(Value::as_str).unwrap_or("");
                    json!({"content": [{"type": "text", "text": format!("{name} saw {q}")}]})
                }
                _ => json!({}),
            };
            let msg = json!({"jsonrpc": "2.0", "id": id, "result": result});
            if sse {
                (
                    [("content-type", "text/event-stream"), ("mcp-session-id", "stub-session")],
                    format!("event: message\ndata: {msg}\n\n"),
                )
                    .into_response()
            } else {
                ([("content-type", "application/json")], msg.to_string()).into_response()
            }
        };
        let app = Router::new().route("/mcp", post(handler));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}/mcp"), task)
    }

    #[tokio::test]
    async fn the_client_handshakes_lists_and_calls_over_json_and_over_sse() {
        for sse in [false, true] {
            let (url, task) = stub(sse).await;
            let c = Client::new(&url, Some("tok".into()));
            let hello = c.initialize().await.expect("initialize");
            assert_eq!(hello.info, "stub 0.1");
            assert_eq!(hello.protocol, PROTOCOL_VERSION);

            let tools = c.list_tools("docs").await.expect("tools/list");
            assert_eq!(tools.len(), 2, "sse={sse}");
            assert_eq!(tools[0].exposed, "mcp__docs__search_docs", "the `.` is not allowed in a tool name");
            assert_eq!(tools[0].remote, "search.docs", "the remote name is kept for the call");
            assert!(tools[0].read_only_hint);
            assert_eq!(tools[1].schema["type"], "object", "a missing inputSchema becomes an empty object");

            let out = c.call_tool("search.docs", &json!({"q": "halts"})).await.expect("tools/call");
            assert_eq!(out["status"], "ok");
            assert_eq!(out["external"], "search.docs saw halts");
            task.abort();
        }
    }

    #[tokio::test]
    async fn probing_a_server_fills_the_catalogue_and_a_dead_one_is_an_error_not_a_panic() {
        let (url, task) = stub(false).await;
        let reg = Registry::new();
        reg.set_servers(vec![
            Server {
                id: "docs".into(),
                url,
                label: "Docs".into(),
                access: Access::Read,
                enabled: true,
                token: None,
            },
            Server {
                id: "dead".into(),
                url: "http://127.0.0.1:1/mcp".into(),
                label: String::new(),
                access: Access::Mutating,
                enabled: true,
                token: None,
            },
        ]);
        reg.probe_all().await;
        let entries = reg.entries();
        assert_eq!(entries[0].status, Status::Ok);
        assert_eq!(entries[0].info.as_deref(), Some("stub 0.1"));
        assert!(matches!(entries[1].status, Status::Error(_)), "{:?}", entries[1].status);

        // Only the working server's tools are offered.
        let specs = reg.specs();
        assert_eq!(specs.len(), 2);
        assert!(specs.iter().all(|s| s.name.starts_with("mcp__docs__")));

        let out = reg.call("mcp__docs__search_docs", &json!({"q": "x"})).await.expect("call");
        assert_eq!(out["external"], "search.docs saw x");
        assert!(reg.call("mcp__docs__gone", &json!({})).await.is_err());
        task.abort();
    }

    #[test]
    fn a_mutating_servers_tools_say_so_in_the_description_the_model_sees() {
        let s = Server {
            id: "ops".into(),
            url: "http://x".into(),
            label: String::new(),
            access: Access::Mutating,
            enabled: true,
            token: None,
        };
        let t = RemoteTool {
            exposed: "mcp__ops__deploy".into(),
            remote: "deploy".into(),
            description: String::new(),
            schema: json!({}),
            read_only_hint: true,
        };
        let d = describe(&s, &t);
        assert!(d.contains("needs operator approval"), "{d}");
        assert!(d.contains("no description given"), "{d}");
    }
}
