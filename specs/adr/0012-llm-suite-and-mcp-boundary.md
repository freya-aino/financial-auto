# ADR 0012 — The LLM suite, and what an agent inside this platform may do

**Status:** accepted (owner 2026-11-17: "a custom agentic suite with the ability to connect external MCPs and utilize llms to manage this project as well") · **Date:** 2026-11-17 · **Decisions:** D-117, D-119, D-120

## Decision

One service, `bins/llm-suite`, holds everything that talks to a language model: the news embedder
that feeds the encoder's text expert, and the operator agent the UI reaches at `/api/llm/*`. Both
go through one OpenAI-compatible endpoint (`crates/fa-llm`), configured from the UI and stored in
`config/runtime.toml` with the key in the git-ignored `config/secrets.toml`.

The agent's authority is fixed in code, in three layers:

1. **It acts only through control-api.** Every platform tool is an HTTP call to a catalogued
   `/api/…` route, never a store access. The agent therefore inherits what the platform already
   enforces: the request audit middleware writes a row per mutating call, the risk engine stays the
   single writer of the book, and a route that refuses an operator refuses the agent identically.
2. **Mutations need the operator, per call.** A mutating tool call ends the turn; the response
   carries the pending call with its exact arguments, and an approval covers *those* arguments. It
   is not refused to the model and retried in a loop.
3. **Some actions are not tools at all.** Promotion and rollback, trading mode, halt re-arming, the
   kill switch, flattening, money movement and payouts have no tool and no reachable path. A test
   asserts that no tool name or path can reach those routes. A model that can ask for them is a
   model that can be talked into asking for them.

While `trading.mode` is `live`, every mutating tool is refused outright and reads stay open.

## The MCP boundary (D-120)

External MCP servers are the one place the agent reaches outside the platform. The boundary is
drawn so that connecting one grants the agent **no new authority**:

- An MCP tool resolves through the same `tools::Caller::call` as a control-api tool, under the same
  approval rule, the same live-mode interlock and the same `llm.tool_call` row.
- **The operator classifies the server, not the server itself.** A registration carries `read` or
  `mutating`, and `mutating` is the default — including for an unrecognised stored value. MCP lets
  a tool advertise `annotations.readOnlyHint`; a server that wants to be trusted will set it, so
  the hint is displayed as a claim and grants nothing.
- **What a server returns is data.** Results are flattened to text, cut at 16 KiB and handed to the
  model as a tool result under an `external` key — never spliced into the system prompt. The prompt
  states that MCP output is data and never an instruction, because a remote tool result is the
  obvious arrival point for an injected one.
- **Streamable HTTP only.** stdio would mean the suite spawning processes inside its own container,
  a much larger trust decision than calling a URL; a local stdio server can sit behind an HTTP
  bridge.

## Context

The owner asked for an LLM suite that both supplies the text expert the latent space lacked and
can help manage the project, with external MCP servers and the connector configured from the UI.
Two questions had to be answered before writing it.

**Where does an agent sit in a system that trades money?** The platform's safety properties are
route-level: single-writer, audited mutations, halts, the promotion gate. An agent with its own
database handle or its own order path would sit *beside* those properties rather than inside them,
and every one of them would have to be restated. Routing the agent through control-api means there
is nothing to restate — and it makes the agent weaker than an operator with a browser, never
stronger.

**What is an MCP server, from this platform's point of view?** It is a third party whose
description of itself arrives in the same channel as its data. Nothing it says can be evidence
about what it is allowed to do here. That is why the access level is a column in
`llm.mcp_server` written by the operator, not a field read from `tools/list`.

## Consequences

- An agent turn costs one model call per tool round, bounded at `MAX_ROUNDS` = 6, so a model
  looping on a failing tool costs a bounded number of calls.
- Conversations, messages and every tool call — refusals included — are durable in schema `llm`.
  The refusals are the more interesting rows: they record what the agent wanted to do and was not
  allowed to.
- The text expert depends on the embedder, not on the agent: without a configured endpoint the
  embedder falls back to a deterministic hashed embedding and marks the provenance, so the expert
  exists before anyone enters a key and a latent built on fallbacks is visible rather than quietly
  different (`GET /api/llm/embeddings` reports the share).
- A server registered `read` is a real grant: its tools run without approval, including in live
  mode. That is the operator's assertion to make, and the UI words it that way.
- Not built: a resource/prompt surface for MCP servers (only tools are used), and OAuth against an
  MCP server — a bearer token is the only credential the registry holds.
