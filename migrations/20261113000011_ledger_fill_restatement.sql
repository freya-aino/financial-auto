-- FX restatements of booked fills (spec 42 §6, round 3): `fa_ledger::restate` rewrites the
-- currency / rate / rate source of fills that were booked before their instrument's quote
-- currency was known, and keeps every old value here (append-only; one batch per apply), so a
-- restatement is auditable and reversible. Quantities, prices and fees are never restated.
-- Idempotent.
create table if not exists ledger.fill_restatement (
    id              bigserial primary key,
    batch_id        uuid not null,
    fill_id         uuid not null references ledger.fill (id) on delete cascade,
    old_ccy         text not null,
    old_fx_rate_eur numeric(38, 18) not null,
    old_fx_source   text not null,
    new_ccy         text not null,
    new_fx_rate_eur numeric(38, 18) not null,
    new_fx_source   text not null,
    -- validity time of the rate applied (ref.fx_rate.ts)
    rate_ts         timestamptz,
    reason          text not null,
    actor           text not null,
    restated_at     timestamptz not null default now(),
    unique (batch_id, fill_id)
);
create index if not exists fill_restatement_fill_idx on ledger.fill_restatement (fill_id);
