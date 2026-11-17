-- Registered external MCP servers (D-120).
--
-- The registration is the agent's only door out of the platform, so what is stored here is a
-- policy decision, not just a connection string:
--
--   * `access` is the operator's classification of the server, and the default is the strict one.
--     MCP lets a tool advertise `annotations.readOnlyHint`, but a server describing itself as
--     harmless is not evidence; that hint is reported in the UI and never used to grant authority.
--   * `token` is write-only through the API: `PUT /api/llm/mcp/{id}` accepts it, no route returns
--     it, and `GET` reports `token_set` only. It lives here rather than in `config/secrets.toml`
--     because that file is bind-mounted into the service read-only and per inode, so a write to it
--     needs a service restart (which is correct for the endpoint key, entered once, and wrong for
--     a server the operator registers and expects to work immediately).
--   * `enabled` keeps a registration while removing its tools from the catalogue, which is the
--     cheap way to answer "was it that server?" without losing the URL.
CREATE TABLE IF NOT EXISTS llm.mcp_server (
    id         text PRIMARY KEY,
    url        text NOT NULL,
    label      text NOT NULL DEFAULT '',
    access     text NOT NULL DEFAULT 'mutating' CHECK (access IN ('read', 'mutating')),
    enabled    boolean NOT NULL DEFAULT true,
    token      text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
