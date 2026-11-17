//! The tools the agent may call (D-117).
//!
//! Every tool is a call to the platform's own control API, never a direct store access. That is
//! the same boundary the frontend respects (spec 51), and it means the agent inherits what
//! control-api already enforces: the request audit middleware writes a row for every mutating
//! catalogued `/api/…` call, the risk engine stays the only writer of the book, and a route that
//! refuses an operator refuses the agent identically.
//!
//! Authority is fixed here, not negotiated with the model:
//!
//! - [`Access::Read`] tools answer freely.
//! - [`Access::Mutating`] tools are the audited operator actions — start/stop a connector, start a
//!   generation, cancel a job, act on a strategy — and each one needs the operator's approval for
//!   that specific call before it runs.
//! - Flipping trading mode, promoting or rolling back a snapshot, re-arming a halt and anything
//!   that moves money are **not tools at all**. They are the owner's, and a model that can ask for
//!   them is a model that can be talked into asking for them.
//!
//! An external MCP server's tools ([`crate::mcp`]) join this catalogue under
//! `mcp__<server>__<tool>` and are resolved by the same [`Caller::call`], so they inherit the
//! approval rule, the live-mode interlock and the `llm.tool_call` row. Their access level comes
//! from the operator's registration, never from the server's own description of itself (D-120).
use std::time::Duration;

use serde_json::{Value, json};

/// What a tool is allowed to do. [`Access::Mutating`] is the default, so anything whose authority
/// is not stated — an external MCP server, a registration row with a bad value — needs approval.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Access {
    Read,
    #[default]
    Mutating,
}

impl Access {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Mutating => "mutating",
        }
    }

    /// Parses a stored value; anything unrecognised is [`Access::Mutating`].
    pub fn parse(s: &str) -> Self {
        if s == "read" { Self::Read } else { Self::Mutating }
    }
}

/// One tool: a name the model sees, a description, its arguments, and the control-api call.
#[derive(Debug, Clone)]
pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    pub access: Access,
    method: &'static str,
    /// Path with `{arg}` placeholders filled from the call's arguments.
    path: &'static str,
    /// Argument names that go into the path (the rest become a JSON body, or a query for GETs).
    path_args: &'static [&'static str],
    /// JSON Schema of the arguments.
    schema_fn: fn() -> Value,
}

fn empty_schema() -> Value {
    json!({"type": "object", "properties": {}, "additionalProperties": false})
}

fn range_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "range": {"type": "string", "description": "window, e.g. `24h`, `7d`, an RFC 3339 instant, or `all`"}
        },
        "additionalProperties": false
    })
}

fn timescale_schema() -> Value {
    json!({
        "type": "object",
        "properties": {"ts": {"type": "string", "enum": ["1s", "1m", "1h", "1d", "1w"]}},
        "required": ["ts"],
        "additionalProperties": false
    })
}

fn source_action_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "connector id as /api/sources lists it"},
            "action": {"type": "string", "enum": ["start", "stop", "pause", "resume"]}
        },
        "required": ["source", "action"],
        "additionalProperties": false
    })
}

fn generation_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "timescale": {"type": "string", "enum": ["1s", "1m", "1h", "1d", "1w"]},
            "window_hours": {"type": "number", "minimum": 1, "maximum": 720},
            "instruments": {"type": "string", "description": "`all` or a comma-separated list"}
        },
        "required": ["timescale"],
        "additionalProperties": false
    })
}

fn job_schema() -> Value {
    json!({
        "type": "object",
        "properties": {"id": {"type": "string", "description": "job id as /api/jobs lists it"}},
        "required": ["id"],
        "additionalProperties": false
    })
}

fn strategy_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "action": {"type": "string", "enum": ["pause", "resume", "shadow", "retire", "flatten"]}
        },
        "required": ["id", "action"],
        "additionalProperties": false
    })
}

/// The catalogue. Read tools first; the four mutating ones are last and marked.
pub const TOOLS: &[Tool] = &[
    Tool {
        name: "platform_health",
        description: "Health of control-api and every dependency (NATS, Postgres, Qdrant, graph, VictoriaMetrics, Garage).",
        access: Access::Read,
        method: "GET",
        path: "/api/health",
        path_args: &[],
        schema_fn: empty_schema,
    },
    Tool {
        name: "services",
        description: "Every platform service with its container state and heartbeat counters.",
        access: Access::Read,
        method: "GET",
        path: "/api/services",
        path_args: &[],
        schema_fn: empty_schema,
    },
    Tool {
        name: "sources",
        description: "The source catalogue with live health: which connectors run, their status and last frames.",
        access: Access::Read,
        method: "GET",
        path: "/api/sources",
        path_args: &[],
        schema_fn: empty_schema,
    },
    Tool {
        name: "trading_risk",
        description: "Risk state: mode, equity, deployable capital, drawdown, gross exposure and active halts, both as the risk engine reports it and as the paper book stands.",
        access: Access::Read,
        method: "GET",
        path: "/api/trading/risk",
        path_args: &[],
        schema_fn: empty_schema,
    },
    Tool {
        name: "trading_positions",
        description: "Open positions with marks, average cost and realised/unrealised P&L.",
        access: Access::Read,
        method: "GET",
        path: "/api/trading/positions",
        path_args: &[],
        schema_fn: empty_schema,
    },
    Tool {
        name: "trading_pnl",
        description: "Realised P&L, fees and turnover over a window, per instrument.",
        access: Access::Read,
        method: "GET",
        path: "/api/trading/pnl",
        path_args: &[],
        schema_fn: range_schema,
    },
    Tool {
        name: "trading_strategies",
        description: "Every strategy with its lifecycle status, sleeve, sub-book, drawdown and promotion gate.",
        access: Access::Read,
        method: "GET",
        path: "/api/trading/strategies",
        path_args: &[],
        schema_fn: empty_schema,
    },
    Tool {
        name: "trading_orders",
        description: "Recent order intents and their fills.",
        access: Access::Read,
        method: "GET",
        path: "/api/trading/orders",
        path_args: &[],
        schema_fn: empty_schema,
    },
    Tool {
        name: "latent_snapshots",
        description: "Encoder snapshots per timescale: family, generation, status, manifest.",
        access: Access::Read,
        method: "GET",
        path: "/api/latent/snapshots",
        path_args: &[],
        schema_fn: empty_schema,
    },
    Tool {
        name: "latent_regions",
        description: "Regions of one timescale's active snapshot with occupancy and lineage.",
        access: Access::Read,
        method: "GET",
        path: "/api/latent/timescales/{ts}/regions",
        path_args: &["ts"],
        schema_fn: timescale_schema,
    },
    Tool {
        name: "latent_transitions",
        description: "The Markov transition posteriors of one timescale.",
        access: Access::Read,
        method: "GET",
        path: "/api/latent/timescales/{ts}/transitions",
        path_args: &["ts"],
        schema_fn: timescale_schema,
    },
    Tool {
        name: "generations",
        description: "Encoder generation records: phase, candidates, Pareto selection, validation outcome.",
        access: Access::Read,
        method: "GET",
        path: "/api/evolution/generations",
        path_args: &[],
        schema_fn: empty_schema,
    },
    Tool {
        name: "jobs",
        description: "Job queue: training, evolution and MCMC jobs with status, progress and scores.",
        access: Access::Read,
        method: "GET",
        path: "/api/jobs",
        path_args: &[],
        schema_fn: empty_schema,
    },
    Tool {
        name: "decoder_outputs",
        description: "Recent decoder outputs by family (forecast quantiles, text and visual scores).",
        access: Access::Read,
        method: "GET",
        path: "/api/decoders/outputs",
        path_args: &[],
        schema_fn: empty_schema,
    },
    Tool {
        name: "audit",
        description: "The audit log: every mutating request and domain action, newest first.",
        access: Access::Read,
        method: "GET",
        path: "/api/audit",
        path_args: &[],
        schema_fn: range_schema,
    },
    // --- Mutating: each call needs the operator's approval, and none is available in live mode.
    Tool {
        name: "ingest_control",
        description: "Start or stop one connector. Mutating: needs operator approval.",
        access: Access::Mutating,
        method: "POST",
        path: "/api/ingest/{source}/{action}",
        path_args: &["source", "action"],
        schema_fn: source_action_schema,
    },
    Tool {
        name: "start_generation",
        description: "Enqueue one encoder generation (`train.cycle`) for a timescale. Never promotes anything. Mutating: needs operator approval.",
        access: Access::Mutating,
        method: "POST",
        path: "/api/generations",
        path_args: &[],
        schema_fn: generation_schema,
    },
    Tool {
        name: "cancel_job",
        description: "Cancel one queued or running job. Mutating: needs operator approval.",
        access: Access::Mutating,
        method: "POST",
        path: "/api/jobs/{id}/cancel",
        path_args: &["id"],
        schema_fn: job_schema,
    },
    Tool {
        name: "strategy_action",
        description: "Pause, resume, shadow, retire or flatten one strategy. Mutating: needs operator approval. Promotion is not available to the agent.",
        access: Access::Mutating,
        method: "POST",
        path: "/api/trading/strategies/{id}/{action}",
        path_args: &["id", "action"],
        schema_fn: strategy_schema,
    },
];

pub fn find(name: &str) -> Option<&'static Tool> {
    TOOLS.iter().find(|t| t.name == name)
}

/// The tool list as the chat API wants it.
pub fn specs() -> Vec<fa_llm::ToolSpec> {
    TOOLS
        .iter()
        .map(|t| fa_llm::ToolSpec {
            name: t.name.to_owned(),
            description: t.description.to_owned(),
            parameters: (t.schema_fn)(),
        })
        .collect()
}

/// Why a call was refused before it was made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    UnknownTool(String),
    /// A mutating tool without the operator's approval for this call.
    NeedsApproval(String),
    /// Real-money mode is on: reads stay, mutations stop (D-117).
    LiveMode(String),
    /// A required argument was missing or the wrong shape.
    BadArguments(String),
}

impl Refusal {
    pub fn message(&self) -> String {
        match self {
            Self::UnknownTool(n) => format!("no tool named `{n}`"),
            Self::NeedsApproval(n) => {
                format!("`{n}` changes the platform and needs the operator's approval for this exact call")
            }
            Self::LiveMode(n) => format!(
                "`{n}` is refused while trading mode is `live`: with real money at stake the agent reads \
                 but does not act"
            ),
            Self::BadArguments(m) => format!("bad arguments: {m}"),
        }
    }
}

/// Calls control-api — or a registered MCP server — on the agent's behalf.
#[derive(Clone)]
pub struct Caller {
    http: reqwest::Client,
    base: String,
    mcp: crate::mcp::Registry,
}

impl Caller {
    pub fn new(base: &str) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
            base: base.trim_end_matches('/').to_owned(),
            mcp: crate::mcp::Registry::new(),
        }
    }

    /// Shares an MCP registry with this caller (the registry is cheap to clone and shared state).
    pub fn with_mcp(mut self, mcp: crate::mcp::Registry) -> Self {
        self.mcp = mcp;
        self
    }

    /// The whole catalogue the model sees: the platform's tools, then every enabled MCP server's.
    pub fn specs(&self) -> Vec<fa_llm::ToolSpec> {
        let mut out = specs();
        out.extend(self.mcp.specs());
        out
    }

    /// Resolves and performs one tool call.
    ///
    /// `approved` is for this call only — the operator approves a tool *invocation*, not a tool,
    /// so an approval cannot be reused by a later turn with different arguments. `live_mode`
    /// refuses every mutation outright.
    pub async fn call(
        &self,
        name: &str,
        args: &Value,
        approved: bool,
        live_mode: bool,
    ) -> std::result::Result<Value, Refusal> {
        let Some(tool) = find(name) else {
            return self.call_mcp(name, args, approved, live_mode).await;
        };
        gate(tool.access, name, approved, live_mode)?;
        let obj = args.as_object().cloned().unwrap_or_default();
        let mut path = tool.path.to_owned();
        for key in tool.path_args {
            let v = obj
                .get(*key)
                .and_then(Value::as_str)
                .ok_or_else(|| Refusal::BadArguments(format!("`{key}` is required and must be a string")))?;
            if v.is_empty() || v.contains('/') || v.contains('?') || v.contains('#') {
                return Err(Refusal::BadArguments(format!("`{key}` = {v:?} is not a single path segment")));
            }
            path = path.replace(&format!("{{{key}}}"), v);
        }
        let rest: serde_json::Map<String, Value> =
            obj.into_iter().filter(|(k, _)| !tool.path_args.contains(&k.as_str())).collect();
        let url = format!("{}{path}", self.base);
        let req = match tool.method {
            "GET" => {
                let q: Vec<(String, String)> = rest
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string())))
                    .collect();
                self.http.get(&url).query(&q)
            }
            _ => self.http.post(&url).json(&Value::Object(rest)),
        };
        match req.send().await {
            Ok(resp) => {
                let code = resp.status().as_u16();
                let body = resp.text().await.unwrap_or_default();
                let parsed: Value = serde_json::from_str(&body).unwrap_or_else(|_| json!({"body": body}));
                // A non-2xx is data for the model, not a failure of the loop: control-api's
                // refusals say why, and the agent should relay that rather than retry blindly.
                Ok(json!({"status": code, "result": parsed}))
            }
            Err(e) => Ok(json!({"status": 0, "error": format!("control-api unreachable: {e}")})),
        }
    }

    /// One MCP tool call, under the access level of the server the operator registered.
    ///
    /// A server being unreachable is reported to the model as a result, not raised: the same
    /// reasoning as for control-api above, and a remote server is the more likely of the two to
    /// be down.
    async fn call_mcp(
        &self,
        name: &str,
        args: &Value,
        approved: bool,
        live_mode: bool,
    ) -> std::result::Result<Value, Refusal> {
        let Some((server, _)) = self.mcp.find(name) else {
            return Err(Refusal::UnknownTool(name.to_owned()));
        };
        gate(server.access, name, approved, live_mode)?;
        match self.mcp.call(name, args).await {
            Ok(v) => Ok(json!({"status": 200, "server": server.id, "result": v})),
            Err(e) => Ok(json!({"status": 0, "server": server.id, "error": e})),
        }
    }
}

/// The two rules that stand between a model and a change: the operator approved *this* call, and
/// the platform is not trading real money.
fn gate(access: Access, name: &str, approved: bool, live_mode: bool) -> std::result::Result<(), Refusal> {
    if access == Access::Mutating {
        if live_mode {
            return Err(Refusal::LiveMode(name.to_owned()));
        }
        if !approved {
            return Err(Refusal::NeedsApproval(name.to_owned()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_has_a_distinct_name_and_a_schema_that_matches_its_path() {
        let mut names: Vec<&str> = TOOLS.iter().map(|t| t.name).collect();
        names.sort_unstable();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n, "tool names must be distinct");
        for t in TOOLS {
            let schema = (t.schema_fn)();
            let props = schema["properties"].as_object().expect("object schema");
            for key in t.path_args {
                assert!(props.contains_key(*key), "{}: `{key}` is in the path but not in the schema", t.name);
            }
            let placeholders = t.path.matches('{').count();
            assert_eq!(placeholders, t.path_args.len(), "{}: path placeholders and path_args disagree", t.name);
            assert!(!t.description.is_empty());
            if t.access == Access::Mutating {
                assert_eq!(t.method, "POST", "{}: a mutating tool is a POST", t.name);
                assert!(
                    t.description.contains("needs operator approval"),
                    "{}: a mutating tool says so in its description, which the model sees",
                    t.name
                );
            } else {
                assert_eq!(t.method, "GET", "{}: a read tool is a GET", t.name);
            }
        }
    }

    #[test]
    fn the_owners_actions_are_not_tools() {
        // Promotion, mode, money and re-arming halts are deliberately absent: see the module docs.
        for forbidden in ["promote", "rollback", "mode", "payout", "rearm", "re-arm", "kill", "flatten_all", "resume_halt"] {
            assert!(
                !TOOLS.iter().any(|t| t.name.contains(forbidden)),
                "`{forbidden}` must not be reachable as a tool"
            );
        }
        // And no tool may address those routes even under another name.
        for t in TOOLS {
            for route in ["/promote", "/rollback", "/trading/mode", "/kill-switch", "/rearm", "/payout", "/halt", "/resume"] {
                assert!(!t.path.contains(route), "{} addresses {route}", t.name);
            }
        }
    }

    #[tokio::test]
    async fn mutating_tools_need_approval_and_are_off_in_live_mode() {
        let c = Caller::new("http://127.0.0.1:1");
        let args = json!({"source": "kraken", "action": "stop"});

        let err = c.call("ingest_control", &args, false, false).await.expect_err("unapproved");
        assert_eq!(err, Refusal::NeedsApproval("ingest_control".into()));
        assert!(err.message().contains("operator's approval"));

        let err = c.call("ingest_control", &args, true, true).await.expect_err("live mode");
        assert_eq!(err, Refusal::LiveMode("ingest_control".into()));

        let err = c.call("nope", &json!({}), true, false).await.expect_err("unknown");
        assert_eq!(err, Refusal::UnknownTool("nope".into()));

        // Reads need neither approval nor paper mode; this one only fails to connect.
        let out = c.call("trading_risk", &json!({}), false, true).await.expect("reads stay open");
        assert_eq!(out["status"], 0, "unreachable control-api is reported, not a refusal: {out}");
    }

    #[tokio::test]
    async fn path_arguments_cannot_escape_their_segment() {
        let c = Caller::new("http://127.0.0.1:1");
        for bad in ["../../api/trading/kill-switch", "kraken/stop", "a?b", "x#y", ""] {
            let err = c
                .call("ingest_control", &json!({"source": bad, "action": "stop"}), true, false)
                .await
                .expect_err("must refuse");
            assert!(matches!(err, Refusal::BadArguments(_)), "{bad:?} was accepted: {err:?}");
        }
        let err = c.call("latent_regions", &json!({}), false, false).await.expect_err("missing ts");
        assert!(matches!(err, Refusal::BadArguments(_)), "{err:?}");
    }

    #[tokio::test]
    async fn an_mcp_tool_passes_through_the_same_gate_as_a_platform_tool() {
        use crate::mcp;
        let reg = mcp::Registry::new();
        let read = mcp::Server {
            id: "docs".into(),
            url: "http://127.0.0.1:1/mcp".into(),
            label: String::new(),
            access: Access::Read,
            enabled: true,
            token: None,
        };
        let ops = mcp::Server { id: "ops".into(), access: Access::Mutating, ..read.clone() };
        reg.set_servers(vec![read, ops]);
        reg.set_tools_for_test("docs", &["search"]);
        reg.set_tools_for_test("ops", &["deploy"]);
        let c = Caller::new("http://127.0.0.1:1").with_mcp(reg);

        assert_eq!(c.specs().len(), TOOLS.len() + 2, "the catalogue is platform tools plus MCP tools");
        assert!(c.specs().iter().any(|s| s.name == "mcp__ops__deploy"));

        // A mutating server's tool needs approval and is off in live mode, exactly like a local one.
        let err = c.call("mcp__ops__deploy", &json!({}), false, false).await.expect_err("unapproved");
        assert_eq!(err, Refusal::NeedsApproval("mcp__ops__deploy".into()));
        let err = c.call("mcp__ops__deploy", &json!({}), true, true).await.expect_err("live");
        assert_eq!(err, Refusal::LiveMode("mcp__ops__deploy".into()));

        // A read server's tool runs without approval even in live mode; this one only fails to
        // connect, which is a result rather than a refusal.
        let out = c.call("mcp__docs__search", &json!({"q": "x"}), false, true).await.expect("reads stay open");
        assert_eq!(out["status"], 0, "{out}");
        assert_eq!(out["server"], "docs");

        // An unregistered MCP name is an unknown tool, not an attempt at a server.
        let err = c.call("mcp__ghost__x", &json!({}), true, false).await.expect_err("unknown");
        assert_eq!(err, Refusal::UnknownTool("mcp__ghost__x".into()));
    }

    #[test]
    fn the_spec_list_is_what_the_model_sees() {
        let specs = specs();
        assert_eq!(specs.len(), TOOLS.len());
        let risk = specs.iter().find(|s| s.name == "trading_risk").expect("trading_risk");
        assert_eq!(risk.parameters["type"], "object");
        assert!(specs.iter().all(|s| !s.description.is_empty()));
    }
}
