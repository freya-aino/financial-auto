-- Risk-chain history (spec 40 §5): allocation snapshots, netted portfolio targets before and
-- after the risk chain, and promotion-gate reviews. Written by the risk-engine
-- (bins/risk-engine/src/history.rs); allocation snapshots and targets on change and at least
-- every `[history] snapshot_secs`, pruned after `[history] retention_days`; promotion reviews
-- are kept (an audit trail of every gate decision). Idempotent.

-- One row per allocator run worth keeping: the active allocator's weights per strategy.
create table if not exists ledger.allocation_snapshot (
    book          text        not null,
    at            timestamptz not null,
    allocator     text        not null,
    spec          jsonb       not null default '{}'::jsonb, -- name + params of the active allocator
    weights       jsonb       not null,                     -- {strategy_id: weight}
    sleeves_eur   jsonb       not null default '{}'::jsonb, -- {strategy_id: weight × deployable}
    equity_eur    numeric     not null,
    deployable_eur numeric    not null,
    fallback      boolean     not null default false,       -- allocator output rejected (spec 40 §8)
    primary key (book, at)
);

-- Netted portfolio target per instrument at one decision tick: the allocator's netted target
-- (`raw`), after the caps (`capped`), after halts / feed guard / stale FX (`gated`), the
-- position it was compared with and what the tick did with it.
create table if not exists ledger.portfolio_target (
    book          text        not null,
    at            timestamptz not null,
    instrument_id uuid        not null,
    raw_eur       numeric     not null,
    capped_eur    numeric     not null,
    gated_eur     numeric     not null,
    current_eur   numeric     not null,
    contributors  jsonb       not null default '{}'::jsonb, -- {strategy_id: signed EUR contribution}
    decision      text        not null,                     -- trade | skip:<reason> | none
    intent_id     text,
    primary key (book, at, instrument_id)
);
create index if not exists portfolio_target_instrument_at on ledger.portfolio_target (instrument_id, at desc);

-- Every promotion-gate evaluation that decided something (spec 40 §6.6): an operator or
-- automatic promote attempt (pass or refusal), and every change of a shadow strategy's gate
-- outcome (plus an hourly record while it is unchanged).
create table if not exists ledger.promotion_review (
    id            uuid        primary key,
    book          text        not null,
    strategy_id   text        not null,
    at            timestamptz not null,
    trigger       text        not null, -- promote | pipeline | gate_change | gate_hourly
    actor         text        not null,
    status_before text,
    status_after  text,
    passed        boolean     not null, -- gate green
    missing       jsonb       not null default '[]'::jsonb,
    checklist     jsonb       not null, -- fa_risk::strategy::Checklist (items with value / threshold / detail)
    evidence      jsonb       not null default '{}'::jsonb
);
create index if not exists promotion_review_strategy_at on ledger.promotion_review (strategy_id, at desc);
