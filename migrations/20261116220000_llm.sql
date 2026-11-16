-- The LLM suite's own state (D-117): agent conversations, the tool calls they made, and which
-- path embedded each news item.
--
-- Conversations live here rather than in a browser tab because what the agent did is operational
-- history: it sits next to `audit.log`, which already holds a row per mutating request the agent
-- caused, and the two are read together when answering "why did this change?".
CREATE SCHEMA IF NOT EXISTS llm;

CREATE TABLE IF NOT EXISTS llm.conversation (
    conversation_id text PRIMARY KEY,
    title           text NOT NULL DEFAULT '',
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS llm.message (
    seq               bigserial PRIMARY KEY,
    conversation_id   text NOT NULL REFERENCES llm.conversation(conversation_id) ON DELETE CASCADE,
    role              text NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content           text NOT NULL,
    -- Tokens are recorded per assistant message so spend is attributable to a turn, not just a day.
    prompt_tokens     bigint NOT NULL DEFAULT 0,
    completion_tokens bigint NOT NULL DEFAULT 0,
    created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS message_conversation_idx ON llm.message (conversation_id, seq);

-- Every tool call, including the refused ones: a refusal is the more interesting row of the two,
-- because it records what the agent wanted to do and was not allowed to.
CREATE TABLE IF NOT EXISTS llm.tool_call (
    id              bigserial PRIMARY KEY,
    conversation_id text NOT NULL REFERENCES llm.conversation(conversation_id) ON DELETE CASCADE,
    tool            text NOT NULL,
    arguments       jsonb NOT NULL DEFAULT '{}'::jsonb,
    outcome         text NOT NULL,
    detail          jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS tool_call_conversation_idx ON llm.tool_call (conversation_id, id);
CREATE INDEX IF NOT EXISTS tool_call_outcome_idx ON llm.tool_call (outcome, created_at DESC);

-- One row per embedded news item. Three jobs:
--   * provenance — `provider` says whether the vector came from the endpoint or the deterministic
--     hashed fallback, so a latent space built while the endpoint was down is visible rather than
--     quietly different (D-117);
--   * idempotence across restarts — the embedder's in-process dedupe set does not survive one, and
--     re-embedding the same item costs money and writes a second vector;
--   * a cheap answer to "how much of the text expert's input is fallback?".
CREATE TABLE IF NOT EXISTS llm.news_embedding (
    doc_id     text PRIMARY KEY,
    source     text NOT NULL DEFAULT '',
    ts         timestamptz NOT NULL,
    provider   text NOT NULL CHECK (provider IN ('endpoint', 'hashed')),
    model      text NOT NULL DEFAULT '',
    dim        integer NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS news_embedding_ts_idx ON llm.news_embedding (ts DESC);
CREATE INDEX IF NOT EXISTS news_embedding_provider_idx ON llm.news_embedding (provider, ts DESC);
