-- Strategy lifecycle (spec 40 §3, §6.6; D-051): one row per strategy the risk-engine has seen.
-- status: active | paused | flattening | shadow. prev_status is what `resume` returns to.
-- Shadow progress (spec 41 §6.6) feeds the promotion gate; `gate` is the last checklist computed.
-- Idempotent, so a manually applied copy does not break sqlx::migrate on the next control-api start.
create table if not exists ledger.strategy_state (
    strategy_id        text primary key,
    status             text not null check (status in ('active', 'paused', 'flattening', 'shadow')),
    prev_status        text,
    since              timestamptz not null default now(),
    first_seen         timestamptz not null default now(),
    shadow_since       timestamptz,
    horizons_completed bigint not null default 0,
    sleeve_eur         numeric(38, 18) not null default 0,
    pnl_eur            numeric(38, 18) not null default 0,
    peak_pnl_eur       numeric(38, 18) not null default 0,
    max_drawdown       numeric(38, 18) not null default 0, -- worst observed, fraction of sleeve
    gate               jsonb,
    updated_at         timestamptz not null default now()
);
create index if not exists strategy_state_status_idx on ledger.strategy_state (status);
