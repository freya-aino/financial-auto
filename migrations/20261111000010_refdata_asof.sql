-- Reference data completion (spec 12 §5, §6, §8; spec 20 §5.4, §6.3). Idempotent, so a manually
-- applied copy does not break sqlx::migrate on the next control-api start.

-- 1. Bitemporal-lite everywhere (spec 12 §5): every ref table carries valid_from/valid_to
--    (real-world validity) and recorded_at (when we learned it).
alter table ref.entity add column if not exists valid_from timestamptz not null default '-infinity';
alter table ref.entity add column if not exists valid_to timestamptz not null default 'infinity';
alter table ref.venue add column if not exists recorded_at timestamptz not null default now();
alter table ref.venue add column if not exists valid_from timestamptz not null default '-infinity';
alter table ref.venue add column if not exists valid_to timestamptz not null default 'infinity';
alter table ref.calendar_session add column if not exists recorded_at timestamptz not null default now();
alter table ref.calendar_session add column if not exists source_id text references catalog.source (id);
alter table ref.fx_rate add column if not exists recorded_at timestamptz not null default now();

-- 2. Corporate actions (spec 12 §3, §6, §8): lifecycle, symbol changes and an idempotency key.
--    `status`: announced | confirmed | cancelled (a cancellation is a new fact with its own
--    recorded_at; as-of queries before it still see the action). `ratio` for splits is
--    new shares per old share (2-for-1 → 2; 1-for-10 reverse split → 0.1).
alter table ref.corporate_action add column if not exists status text not null default 'confirmed';
alter table ref.corporate_action add column if not exists cancelled_at timestamptz;
alter table ref.corporate_action add column if not exists venue_id uuid references ref.venue (id);
alter table ref.corporate_action add column if not exists old_symbol text;
alter table ref.corporate_action add column if not exists new_symbol text;
alter table ref.corporate_action add column if not exists notes text;
do $$ begin
  alter table ref.corporate_action add constraint corporate_action_status_chk
    check (status in ('announced', 'confirmed', 'cancelled'));
exception when duplicate_object then null; end $$;
create unique index if not exists corporate_action_natural_key
    on ref.corporate_action (instrument_id, kind, ex_date, coalesce(source_id, ''));

-- 3. Instrument relations mirrored to Helix as SAME_ECONOMIC_EXPOSURE etc. (spec 12 §5).
create table if not exists ref.instrument_relation (
    from_id     uuid not null references ref.instrument (id),
    to_id       uuid not null references ref.instrument (id),
    kind        text not null, -- same_economic_exposure | underlying | successor
    valid_from  timestamptz not null default '-infinity',
    valid_to    timestamptz not null default 'infinity',
    recorded_at timestamptz not null default now(),
    primary key (from_id, to_id, kind, valid_from)
);

-- 4. As-of read paths: latest validity per key at or before t.
create index if not exists listing_asof_idx on ref.listing (venue_id, venue_symbol, valid_from desc, recorded_at);
create index if not exists listing_source_symbol_idx on ref.listing (source_id, venue_symbol, valid_from desc);
create index if not exists contract_spec_asof_idx on ref.contract_spec (instrument_id, venue_id, valid_from desc);
create index if not exists identifier_asof_idx on ref.identifier (instrument_id, scheme, valid_from desc);
create index if not exists corporate_action_recorded_idx on ref.corporate_action (instrument_id, recorded_at);
create index if not exists calendar_session_date_idx on ref.calendar_session (date, venue_id);

-- 5. Reference venues with MIC and timezone (spec 12 §5; trading calendars per venue). The
--    provisioned per-source venues (`yahoo`, `lemon`, ...) stay as they are.
insert into ref.venue (id, code, mic, name, timezone, kind, always_open) values
  ('6f1c0a52-8a1f-5b0e-9d6e-000000000001', 'xnys', 'XNYS', 'New York Stock Exchange', 'America/New_York', 'exchange', false),
  ('6f1c0a52-8a1f-5b0e-9d6e-000000000002', 'xnas', 'XNAS', 'Nasdaq', 'America/New_York', 'exchange', false),
  ('6f1c0a52-8a1f-5b0e-9d6e-000000000003', 'xetr', 'XETR', 'Xetra', 'Europe/Berlin', 'exchange', false),
  ('6f1c0a52-8a1f-5b0e-9d6e-000000000004', 'fx_otc', null, 'FX OTC (24/5)', 'UTC', 'otc', false),
  ('6f1c0a52-8a1f-5b0e-9d6e-000000000005', 'crypto', null, 'Crypto (24/7, aggregated)', 'UTC', 'otc', true),
  ('6f1c0a52-8a1f-5b0e-9d6e-000000000006', 'kraken', null, 'Kraken', 'UTC', 'exchange', true)
on conflict do nothing;

-- 6. Repair the first provisioning pass, which classified every Yahoo symbol as an equity
--    (D-050 hint). Yahoo symbol syntax: `XXXYYY=X` fx pair, `XXX-YYY` crypto, `^...` index,
--    `...=F` future. Only provisional rows are touched; an operator-activated instrument wins.
update ref.instrument i set asset_class = 'fx', kind = 'fx_pair',
       base_ccy = substr(l.venue_symbol, 1, 3), quote_ccy = substr(l.venue_symbol, 4, 3)
  from ref.listing l
 where l.instrument_id = i.id and l.source_id = 'yahoo' and i.status = 'provisional'
   and l.venue_symbol ~ '^[A-Z]{6}=X$' and i.kind <> 'fx_pair';
update ref.instrument i set asset_class = 'crypto', kind = 'crypto_spot',
       base_ccy = split_part(l.venue_symbol, '-', 1), quote_ccy = split_part(l.venue_symbol, '-', 2)
  from ref.listing l
 where l.instrument_id = i.id and l.source_id = 'yahoo' and i.status = 'provisional'
   and l.venue_symbol ~ '^[A-Z0-9]{2,10}-(USD|EUR|USDT|USDC|GBP|BTC|ETH)$' and i.kind <> 'crypto_spot';
update ref.instrument i set asset_class = 'index', kind = 'index'
  from ref.listing l
 where l.instrument_id = i.id and l.source_id = 'yahoo' and i.status = 'provisional'
   and l.venue_symbol like '^%' and i.kind <> 'index';
update ref.instrument i set quote_ccy = 'USD'
  from ref.listing l
 where l.instrument_id = i.id and l.source_id = 'yahoo' and i.status = 'provisional'
   and i.kind in ('equity', 'etf') and i.quote_ccy is null and l.venue_symbol ~ '^[A-Z.]{1,6}$';

-- 7. Fundamentals with vintages (spec 20 §5.1: (entity, field, period, vintage)); macro
--    vintages already live in wh.macro_observation.
create table if not exists wh.fundamental (
    instrument_id uuid references ref.instrument (id),
    entity_id     uuid references ref.entity (id),
    field         text not null, -- revenue, eps_diluted, shares_outstanding, ...
    period        text not null, -- 2026Q3, FY2025, ...
    vintage       timestamptz not null, -- when this value became public
    value         double precision not null,
    unit          text,
    source_id     text not null references catalog.source (id),
    recorded_at   timestamptz not null default now(),
    check (instrument_id is not null or entity_id is not null)
);
create unique index if not exists fundamental_natural_key
    on wh.fundamental (coalesce(instrument_id, entity_id), field, period, vintage, source_id);

-- 8. Macro as-of reads (spec 20 §6.3): latest vintage <= t per series and period.
create index if not exists macro_observation_asof_idx on wh.macro_observation (source_id, series_id, period, vintage desc);
