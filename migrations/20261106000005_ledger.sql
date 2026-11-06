-- Ledger (spec 42). Postgres is the source of truth for orders, fills, cash and positions.
create table ledger.order_intent (
    id                  uuid primary key,
    book                text not null, -- paper | shadow | live
    portfolio_target_id uuid,
    instrument_id       uuid not null,
    side                text not null check (side in ('buy', 'sell')),
    qty                 numeric(38, 18) not null,
    exec_algo           text not null,
    params              jsonb,
    idempotency_key     text unique not null,
    created_at          timestamptz not null default now()
);

create table ledger.venue_order (
    client_order_id text primary key, -- deterministic: intent id + seq
    intent_id       uuid not null references ledger.order_intent (id),
    venue_id        uuid not null,
    venue_order_id  text,
    status          text not null,
    qty             numeric(38, 18),
    limit_price     numeric(38, 18),
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

create table ledger.fill (
    id              uuid primary key,
    client_order_id text not null references ledger.venue_order (client_order_id),
    venue_fill_id   text not null,
    qty             numeric(38, 18) not null,
    price           numeric(38, 18) not null,
    ccy             text not null,
    fee             numeric(38, 18) not null,
    fee_ccy         text not null,
    fx_rate_eur     numeric(38, 18) not null,
    fx_source       text not null,
    arrival_price   numeric(38, 18),
    filled_at       timestamptz not null,
    unique (client_order_id, venue_fill_id)
);

create table ledger.fill_attribution (
    fill_id     uuid not null references ledger.fill (id),
    strategy_id uuid not null,
    qty         numeric(38, 18) not null,
    fee_share   numeric(38, 18) not null,
    primary key (fill_id, strategy_id)
);

-- deposits, withdrawals, payouts, funding, borrow, dividends
create table ledger.cash_movement (
    id          uuid primary key,
    book        text not null,
    venue_id    uuid,
    kind        text not null,
    amount      numeric(38, 18) not null,
    ccy         text not null,
    fx_rate_eur numeric(38, 18) not null,
    ref         text not null,
    occurred_at timestamptz not null,
    unique (venue_id, kind, ref)
);

create table ledger.position_snapshot (
    book          text not null,
    venue_id      uuid not null,
    instrument_id uuid not null,
    qty           numeric(38, 18) not null,
    avg_price     numeric(38, 18),
    as_of         timestamptz not null,
    primary key (book, venue_id, instrument_id, as_of)
);

create table ledger.payout_quota (
    day          date primary key,
    quota_eur    numeric(38, 18) not null,
    reserved_eur numeric(38, 18) not null default 0,
    paid_eur     numeric(38, 18) not null default 0
);

create table ledger.reconciliation_run (
    id         uuid primary key,
    venue_id   uuid not null,
    started_at timestamptz not null default now(),
    status     text not null,
    diffs      jsonb
);
