//! The agent loop and the API control-api proxies to it (D-117).
//!
//! One turn is: send the conversation plus the tool catalogue to the endpoint, run whatever tools
//! the model asks for, feed the results back, and repeat until it answers in prose or the round
//! budget runs out. Nothing streams: the loop needs a whole message, tool calls included, before
//! it can act on it, and the UI gets the turn's transcript when it completes.
//!
//! Three properties are deliberate:
//!
//! - **Mutating calls stop the turn.** When the model asks for a mutating tool without approval,
//!   the turn ends and returns the pending call for the operator to approve; it is not refused to
//!   the model and retried in a loop. The operator then re-sends the turn with that approval, and
//!   the approval is tied to the exact arguments.
//! - **The round budget is hard.** [`MAX_ROUNDS`] tool rounds per turn, after which the turn ends
//!   with what it has. A model that loops on a failing tool costs a bounded number of calls.
//! - **Everything is persisted before it is returned.** The conversation, every message and every
//!   tool call with its arguments and outcome go to Postgres, so what the agent did is auditable
//!   next to `audit.log` rather than only in a browser tab.
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use fa_core::config::PlatformConfig;
use fa_llm::{ChatMessage, ChatRequest, LlmClient};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqlx::{PgPool, Row};

use crate::tools;

/// Tool rounds per turn.
pub const MAX_ROUNDS: usize = 6;

/// The system prompt. Deliberately short on persona and long on boundaries.
pub const SYSTEM: &str = "\
You are the operator agent of `financial-auto`, an automated multi-asset trading platform that \
trades paper by default. You answer questions about the running system and, when asked, take the \
operator actions your tools expose.

Ground rules:
- Read the platform through your tools rather than guessing. If a tool fails, say what failed.
- Numbers come from tools, never from memory. Quote the figures you read, with their units (EUR) \
  and the window they cover.
- You cannot promote or roll back a snapshot, change trading mode, re-arm a halt, move money or \
  flatten the book. Those are the owner's. If asked, say so plainly and stop.
- A mutating tool call ends your turn pending the operator's approval. Explain what you want to \
  do and why before asking for it.
- The platform trades paper money and has no live broker. Do not imply otherwise.
- Be concise. The reader is the person running this system.";

#[derive(Clone)]
pub struct AgentState {
    llm: Option<LlmClient>,
    caller: tools::Caller,
    pg: Option<PgPool>,
    live_mode: Arc<std::sync::atomic::AtomicBool>,
}

/// One turn's request.
#[derive(Debug, Deserialize)]
pub struct TurnRequest {
    /// Conversation to continue; a new one is created when absent.
    #[serde(default)]
    pub conversation_id: Option<String>,
    pub message: String,
    /// Tool calls the operator approved, by `tool` name and exact arguments.
    #[serde(default)]
    pub approvals: Vec<Approval>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct Approval {
    pub tool: String,
    #[serde(default)]
    pub arguments: Value,
}

/// What a turn did.
#[derive(Debug, Serialize)]
pub struct TurnResponse {
    pub conversation_id: String,
    pub reply: String,
    pub tool_calls: Vec<ToolRecord>,
    /// Calls waiting for the operator: approve one and send the turn again.
    pub pending: Vec<Approval>,
    pub rounds: usize,
    pub usage: fa_llm::Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolRecord {
    pub tool: String,
    pub arguments: Value,
    pub outcome: String,
    pub detail: Value,
}

/// Whether an approval covers this exact call. Arguments must match, not just the tool name.
fn approved(approvals: &[Approval], tool: &str, args: &Value) -> bool {
    approvals.iter().any(|a| a.tool == tool && &a.arguments == args)
}

/// Runs one turn.
pub async fn turn(state: &AgentState, req: TurnRequest) -> Result<TurnResponse, String> {
    let Some(llm) = state.llm.as_ref() else {
        return Err("no LLM endpoint is configured: enter one in the agent tab (it is stored in \
                    config/secrets.toml, never returned by the API)"
            .into());
    };
    let live = state.live_mode.load(std::sync::atomic::Ordering::Relaxed);
    let conversation_id = req.conversation_id.clone().unwrap_or_else(|| ulid::Ulid::generate().to_string().to_lowercase());

    let mut messages = match (&state.pg, &req.conversation_id) {
        (Some(pg), Some(id)) => history(pg, id).await.unwrap_or_default(),
        _ => Vec::new(),
    };
    if messages.is_empty() {
        messages.push(ChatMessage::system(SYSTEM));
    }
    messages.push(ChatMessage::user(&req.message));

    let specs = tools::specs();
    let mut records: Vec<ToolRecord> = Vec::new();
    let mut pending: Vec<Approval> = Vec::new();
    let mut usage = fa_llm::Usage::default();
    let mut reply = String::new();
    let mut rounds = 0;

    while rounds < MAX_ROUNDS {
        let out = llm
            .chat(&ChatRequest {
                messages: messages.clone(),
                tools: specs.clone(),
                temperature: Some(0.2),
                max_tokens: Some(1200),
            })
            .await
            .map_err(|e| e.to_string())?;
        usage.prompt_tokens += out.usage.prompt_tokens;
        usage.completion_tokens += out.usage.completion_tokens;
        messages.push(out.message.clone());

        if out.message.tool_calls.is_empty() {
            reply = out.message.content;
            break;
        }
        rounds += 1;
        let mut stop = false;
        for call in &out.message.tool_calls {
            let args = match call.function.parse_arguments() {
                Ok(v) => v,
                Err(e) => {
                    messages.push(ChatMessage::tool_result(&call.id, format!("arguments are not JSON: {e}")));
                    records.push(ToolRecord {
                        tool: call.function.name.clone(),
                        arguments: json!({"raw": call.function.arguments}),
                        outcome: "bad_arguments".into(),
                        detail: json!({"error": e}),
                    });
                    continue;
                }
            };
            let ok = approved(&req.approvals, &call.function.name, &args);
            match state.caller.call(&call.function.name, &args, ok, live).await {
                Ok(result) => {
                    messages.push(ChatMessage::tool_result(&call.id, result.to_string()));
                    records.push(ToolRecord {
                        tool: call.function.name.clone(),
                        arguments: args,
                        outcome: "ok".into(),
                        detail: result,
                    });
                }
                Err(refusal) => {
                    let msg = refusal.message();
                    messages.push(ChatMessage::tool_result(&call.id, &msg));
                    let outcome = match &refusal {
                        tools::Refusal::NeedsApproval(_) => {
                            pending.push(Approval { tool: call.function.name.clone(), arguments: args.clone() });
                            stop = true;
                            "needs_approval"
                        }
                        tools::Refusal::LiveMode(_) => "refused_live_mode",
                        tools::Refusal::UnknownTool(_) => "unknown_tool",
                        tools::Refusal::BadArguments(_) => "bad_arguments",
                    };
                    records.push(ToolRecord {
                        tool: call.function.name.clone(),
                        arguments: args,
                        outcome: outcome.into(),
                        detail: json!({"refused": msg}),
                    });
                }
            }
        }
        if stop {
            reply = if out.message.content.is_empty() {
                "This needs your approval before I can run it.".into()
            } else {
                out.message.content
            };
            break;
        }
    }
    if reply.is_empty() {
        reply = format!(
            "I stopped after {rounds} tool rounds without reaching an answer. What I called is in the \
             transcript."
        );
    }

    if let Some(pg) = &state.pg
        && let Err(e) = persist(pg, &conversation_id, &req, &reply, &records, &usage).await
    {
        tracing::error!(error = %e, "persisting the agent turn failed");
    }
    Ok(TurnResponse { conversation_id, reply, tool_calls: records, pending, rounds, usage })
}

/// The stored messages of a conversation, oldest first, with the system prompt in front.
async fn history(pg: &PgPool, conversation_id: &str) -> Result<Vec<ChatMessage>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT role, content FROM llm.message WHERE conversation_id = $1 AND role IN ('user', 'assistant') \
         ORDER BY seq",
    )
    .bind(conversation_id)
    .fetch_all(pg)
    .await?;
    let mut out = vec![ChatMessage::system(SYSTEM)];
    for r in &rows {
        let role: String = r.get("role");
        let content: String = r.get("content");
        out.push(ChatMessage { role, content, tool_calls: Vec::new(), tool_call_id: None });
    }
    Ok(out)
}

async fn persist(
    pg: &PgPool,
    conversation_id: &str,
    req: &TurnRequest,
    reply: &str,
    records: &[ToolRecord],
    usage: &fa_llm::Usage,
) -> Result<(), sqlx::Error> {
    let mut tx = pg.begin().await?;
    sqlx::query(
        "INSERT INTO llm.conversation (conversation_id, title) VALUES ($1, left($2, 120)) \
         ON CONFLICT (conversation_id) DO UPDATE SET updated_at = now()",
    )
    .bind(conversation_id)
    .bind(&req.message)
    .execute(&mut *tx)
    .await?;
    for (role, content) in [("user", req.message.as_str()), ("assistant", reply)] {
        sqlx::query(
            "INSERT INTO llm.message (conversation_id, role, content, prompt_tokens, completion_tokens) \
             VALUES ($1, $2, $3, $4, $5)",
        )
        .bind(conversation_id)
        .bind(role)
        .bind(content)
        .bind(if role == "assistant" { usage.prompt_tokens as i64 } else { 0 })
        .bind(if role == "assistant" { usage.completion_tokens as i64 } else { 0 })
        .execute(&mut *tx)
        .await?;
    }
    for r in records {
        sqlx::query(
            "INSERT INTO llm.tool_call (conversation_id, tool, arguments, outcome, detail) \
             VALUES ($1, $2, $3, $4, $5)",
        )
        .bind(conversation_id)
        .bind(&r.tool)
        .bind(&r.arguments)
        .bind(&r.outcome)
        .bind(&r.detail)
        .execute(&mut *tx)
        .await?;
    }
    tx.commit().await
}

/// `GET /api/llm/status`: what the suite can do right now. Never the key.
async fn status(State(s): State<AgentState>) -> impl IntoResponse {
    let live = s.live_mode.load(std::sync::atomic::Ordering::Relaxed);
    Json(json!({
        "configured": s.llm.is_some(),
        "model": s.llm.as_ref().map(|c| c.model().to_owned()),
        "embedding_model": s.llm.as_ref().map(|c| c.embedding_model().to_owned()),
        "embedding_dim": s.llm.as_ref().map(|c| c.embedding_dim()),
        "endpoint": s.llm.as_ref().map(|c| c.base_url().to_owned()),
        "key_set": s.llm.as_ref().is_some_and(|c| c.has_key()),
        "trading_mode": if live { "live" } else { "paper" },
        "mutating_tools_available": !live,
        "max_rounds": MAX_ROUNDS,
        "tools": tools::TOOLS.iter().map(|t| json!({
            "name": t.name,
            "access": if t.access == tools::Access::Mutating { "mutating" } else { "read" },
            "description": t.description,
        })).collect::<Vec<_>>(),
    }))
}

async fn chat(State(s): State<AgentState>, Json(req): Json<TurnRequest>) -> impl IntoResponse {
    match turn(&s, req).await {
        Ok(r) => (StatusCode::OK, Json(serde_json::to_value(r).unwrap_or_default())).into_response(),
        Err(e) => (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": e}))).into_response(),
    }
}

async fn conversations(State(s): State<AgentState>) -> impl IntoResponse {
    let Some(pg) = s.pg.as_ref() else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "postgres unavailable"}))).into_response();
    };
    let rows = sqlx::query(
        "SELECT c.conversation_id, c.title, c.created_at, c.updated_at, \
                (SELECT count(*) FROM llm.message m WHERE m.conversation_id = c.conversation_id) AS messages \
         FROM llm.conversation c ORDER BY c.updated_at DESC LIMIT 100",
    )
    .fetch_all(pg)
    .await;
    match rows {
        Ok(rows) => Json(json!({"conversations": rows.iter().map(|r| json!({
            "conversation_id": r.get::<String, _>("conversation_id"),
            "title": r.get::<String, _>("title"),
            "created_at": r.get::<chrono::DateTime<chrono::Utc>, _>("created_at").to_rfc3339(),
            "updated_at": r.get::<chrono::DateTime<chrono::Utc>, _>("updated_at").to_rfc3339(),
            "messages": r.get::<i64, _>("messages"),
        })).collect::<Vec<_>>()}))
        .into_response(),
        Err(e) => (StatusCode::BAD_GATEWAY, Json(json!({"error": e.to_string()}))).into_response(),
    }
}

async fn conversation(State(s): State<AgentState>, Path(id): Path<String>) -> impl IntoResponse {
    let Some(pg) = s.pg.as_ref() else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "postgres unavailable"}))).into_response();
    };
    let msgs = sqlx::query("SELECT seq, role, content, created_at FROM llm.message WHERE conversation_id = $1 ORDER BY seq")
        .bind(&id)
        .fetch_all(pg)
        .await
        .unwrap_or_default();
    let calls = sqlx::query(
        "SELECT tool, arguments, outcome, detail, created_at FROM llm.tool_call WHERE conversation_id = $1 ORDER BY id",
    )
    .bind(&id)
    .fetch_all(pg)
    .await
    .unwrap_or_default();
    Json(json!({
        "conversation_id": id,
        "messages": msgs.iter().map(|r| json!({
            "seq": r.get::<i64, _>("seq"),
            "role": r.get::<String, _>("role"),
            "content": r.get::<String, _>("content"),
            "created_at": r.get::<chrono::DateTime<chrono::Utc>, _>("created_at").to_rfc3339(),
        })).collect::<Vec<_>>(),
        "tool_calls": calls.iter().map(|r| json!({
            "tool": r.get::<String, _>("tool"),
            "arguments": r.get::<Value, _>("arguments"),
            "outcome": r.get::<String, _>("outcome"),
            "detail": r.get::<Value, _>("detail"),
            "created_at": r.get::<chrono::DateTime<chrono::Utc>, _>("created_at").to_rfc3339(),
        })).collect::<Vec<_>>(),
    }))
    .into_response()
}

/// Serves the suite's API and keeps `trading_mode` current from the bus.
pub async fn serve(
    bind: String,
    cfg: PlatformConfig,
    llm: Option<LlmClient>,
    _bus: fa_bus::Bus,
) -> anyhow::Result<()> {
    let control = std::env::var("FA_CONTROL_API")
        .ok()
        .filter(|u| !u.is_empty())
        .unwrap_or_else(|| "http://control-api:7700".into());
    let pg = match sqlx::postgres::PgPoolOptions::new()
        .max_connections(4)
        .acquire_timeout(std::time::Duration::from_secs(5))
        .connect(&cfg.postgres.url)
        .await
    {
        Ok(p) => Some(p),
        Err(e) => {
            tracing::warn!(error = %e, "postgres unavailable: conversations will not be persisted");
            None
        }
    };
    let state = AgentState {
        llm,
        caller: tools::Caller::new(&control),
        pg,
        live_mode: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    // Trading mode decides whether mutating tools exist at all, so it is read at start and then
    // refreshed; a failure to read it leaves the safer value (paper reads, mutations allowed) only
    // because there is no live mode yet — when there is, this must fail closed instead.
    let mode_state = state.clone();
    let mode_url = format!("{control}/api/trading/risk");
    tokio::spawn(async move {
        let http = reqwest::Client::new();
        loop {
            if let Ok(r) = http.get(&mode_url).timeout(std::time::Duration::from_secs(5)).send().await
                && let Ok(v) = r.json::<Value>().await
            {
                let live = v.get("mode").and_then(Value::as_str) == Some("live");
                mode_state.live_mode.store(live, std::sync::atomic::Ordering::Relaxed);
            }
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        }
    });

    let app = Router::new()
        .route("/health", get(|| async { Json(json!({"service": "llm-suite", "ok": true})) }))
        .route("/api/llm/status", get(status))
        .route("/api/llm/chat", post(chat))
        .route("/api/llm/conversations", get(conversations))
        .route("/api/llm/conversations/{id}", get(conversation))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(%bind, control, "llm-suite api listening");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_approval_covers_one_call_with_exactly_those_arguments() {
        let a = vec![Approval { tool: "ingest_control".into(), arguments: json!({"source": "kraken", "action": "stop"}) }];
        assert!(approved(&a, "ingest_control", &json!({"source": "kraken", "action": "stop"})));
        assert!(
            !approved(&a, "ingest_control", &json!({"source": "yahoo", "action": "stop"})),
            "approving one connector must not approve another"
        );
        assert!(
            !approved(&a, "ingest_control", &json!({"source": "kraken", "action": "start"})),
            "approving a stop must not approve a start"
        );
        assert!(!approved(&a, "cancel_job", &json!({"source": "kraken", "action": "stop"})));
        assert!(!approved(&[], "ingest_control", &json!({})));
    }

    #[test]
    fn the_system_prompt_states_the_boundaries_the_tools_enforce() {
        for must in ["cannot promote", "trading mode", "paper", "owner"] {
            assert!(SYSTEM.contains(must), "the prompt must say `{must}`");
        }
    }

    #[tokio::test]
    async fn a_turn_without_an_endpoint_is_an_explicit_answer() {
        let state = AgentState {
            llm: None,
            caller: tools::Caller::new("http://127.0.0.1:1"),
            pg: None,
            live_mode: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };
        let err = turn(&state, TurnRequest { conversation_id: None, message: "hi".into(), approvals: vec![] })
            .await
            .expect_err("no endpoint");
        assert!(err.contains("no LLM endpoint is configured"), "{err}");
    }
}
