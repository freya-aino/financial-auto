-- Spec 42 §10 Q1/Q5 (2026-11-11): settlement currency per venue and the payout rule that
-- produced a day's quota. Idempotent: it may be applied by hand before control-api records it.

-- How a venue settles fills: 'eur' (default: fill cash converts to EUR at the fill's rate) or
-- 'native' (fill cash stays in the fill currency as a revalued foreign balance).
-- Venues without a row settle in EUR.
create table if not exists ledger.venue_settlement (
    venue_id     uuid primary key,
    settlement   text not null default 'eur' check (settlement in ('eur', 'native')),
    margin_model text not null default 'cash' check (margin_model in ('cash', 'cross', 'isolated')),
    note         text,
    updated_at   timestamptz not null default now()
);

-- the paper venue (fa_ledger::pg::PAPER_VENUE) settles in EUR
insert into ledger.venue_settlement (venue_id, settlement, note)
values ('70617065-7200-0000-0000-000000000001', 'eur', 'paper venue: EUR settlement at the fill rate')
on conflict (venue_id) do nothing;

-- the rule (and its inputs) that set ledger.payout_quota.quota_eur (fa_ledger::payout)
alter table ledger.payout_quota add column if not exists rule jsonb;
