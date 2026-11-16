# fa-llm

Client for an OpenAI-compatible endpoint (D-117): chat completions for the agent in
[`llm-suite`](../../bins/llm-suite), embeddings for the encoder's text expert. One crate for both
because they are one endpoint with one key, and both callers want the same things from it.

**Status:** implemented and unit-tested against a stub server (tool calls, batching and ordering,
width validation, retry policy, redaction).

- `LlmClient::from_env(&cfg.llm, &secrets.llm)` → `Option<Self>`: **`None` when no endpoint is
  configured, which is not an error.** The agent then says it has no endpoint and the embedder falls
  back to hashed embeddings; nothing in the platform requires an external API to run.
- A base URL **without** a key is allowed — local servers (llama.cpp, vLLM, Ollama's OpenAI shim)
  take none, and refusing them would tie the platform to a paid provider.
- `chat(&ChatRequest)`: non-streaming on purpose. The agent loop needs the whole message, tool calls
  included, before it can act on it. Tool arguments arrive as a JSON *string*, as the wire format
  has them, and `FunctionCall::parse_arguments` makes malformed arguments an ordinary event the
  caller answers with a `tool` message.
- `embed(&[String])`: batched at `[llm] embed_batch`, reordered by the response's `index` (providers
  may answer out of order), and **every vector is checked against `embedding_dim`**. A width that
  differs is `LlmError::DimMismatch` rather than a write: a collection's dimension is fixed when it
  is created, so silently mixing widths is unrecoverable. Non-finite values are refused too.
- Retries: `RETRIES` = 2 on transport errors, 429 and 5xx, with exponential backoff. A 4xx is not
  retried — a bad model name does not get better by asking again.
- The key never appears in `Debug` (`key: "set" | "unset"`), in a log line, or in an error: a status
  error carries the response body, trimmed to 400 characters and flattened by `redact_body`.

Metrics (registered on first use): `fa_llm_requests_total{kind,outcome}`,
`fa_llm_tokens_total{kind}`, `fa_llm_request_seconds{kind}`.
