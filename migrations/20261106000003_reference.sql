-- Reference data (spec 12). Bitemporal-lite: valid_from/valid_to + recorded_at.
create table ref.entity (
    id          uuid primary key,
    lei         text unique,
    name        text not null,
    country     text,
    sector      text,
    parent_id   uuid references ref.entity (id),
    recorded_at timestamptz not null default now()
);

create table ref.instrument (
    id          uuid primary key,
    asset_class text not null,
    kind        text not null, -- equity, etf, fx_pair, crypto_spot, perp, future, option, bond, prediction_contract, index, macro_series_proxy
    name        text not null,
    base_ccy    text,
    quote_ccy   text,
    issuer_id   uuid references ref.entity (id),
    status      text not null default 'active',
    valid_from  timestamptz not null default '-infinity',
    valid_to    timestamptz not null default 'infinity',
    recorded_at timestamptz not null default now()
);

create table ref.identifier (
    instrument_id uuid not null references ref.instrument (id),
    scheme        text not null, -- figi, composite_figi, share_class_figi, isin, cusip, cik, lei, coingecko_id, contract_address, ...
    value         text not null,
    valid_from    timestamptz not null default '-infinity',
    valid_to      timestamptz not null default 'infinity',
    recorded_at   timestamptz not null default now(),
    primary key (scheme, value, valid_from)
);
create index identifier_instrument_idx on ref.identifier (instrument_id);

create table ref.venue (
    id       uuid primary key,
    code     text not null unique, -- short token used in NATS subjects, e.g. 'lemon', 'xetr', 'sim'
    mic      text unique,
    name     text not null,
    timezone text not null,
    kind     text not null check (kind in ('exchange', 'broker', 'dex', 'otc')),
    always_open boolean not null default false -- crypto venues
);

create table ref.listing (
    instrument_id uuid not null references ref.instrument (id),
    venue_id      uuid not null references ref.venue (id),
    venue_symbol  text not null,
    source_id     text references catalog.source (id),
    valid_from    timestamptz not null default '-infinity',
    valid_to      timestamptz not null default 'infinity',
    recorded_at   timestamptz not null default now(),
    primary key (venue_id, venue_symbol, valid_from)
);
create index listing_instrument_idx on ref.listing (instrument_id);

create table ref.contract_spec (
    instrument_id    uuid not null references ref.instrument (id),
    venue_id         uuid not null references ref.venue (id),
    tick_size        numeric(38, 18),
    lot_size         numeric(38, 18),
    min_notional     numeric(38, 18),
    multiplier       numeric(38, 18) not null default 1,
    expiry           timestamptz,
    settlement_ccy   text,
    shortable        boolean,
    max_leverage     numeric(38, 18),
    fee_schedule_ref text,
    valid_from       timestamptz not null default '-infinity',
    valid_to         timestamptz not null default 'infinity',
    recorded_at      timestamptz not null default now(),
    primary key (instrument_id, venue_id, valid_from)
);

create table ref.calendar_session (
    venue_id   uuid not null references ref.venue (id),
    date       date not null,
    open       timestamptz,
    close      timestamptz,
    segments   jsonb not null default '[]', -- pre / regular / post / auction
    is_holiday boolean not null default false,
    primary key (venue_id, date)
);

create table ref.corporate_action (
    id            uuid primary key,
    instrument_id uuid not null references ref.instrument (id),
    kind          text not null, -- split, reverse_split, cash_dividend, stock_dividend, spinoff, merger, symbol_change
    ex_date       date not null,
    record_date   date,
    pay_date      date,
    ratio         numeric(38, 18),
    amount        numeric(38, 18),
    ccy           text,
    source_id     text references catalog.source (id),
    announced_at  timestamptz, -- point-in-time availability
    recorded_at   timestamptz not null default now()
);
create index corporate_action_instrument_idx on ref.corporate_action (instrument_id, ex_date);

create table ref.fx_rate (
    ccy         text not null,
    ts          timestamptz not null,
    rate_to_eur numeric(38, 18) not null,
    source_id   text not null references catalog.source (id),
    primary key (ccy, ts, source_id)
);
