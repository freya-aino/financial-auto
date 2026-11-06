-- Ingestion bookkeeping (spec 10).
create table ingest.quarantine (
    source_id      text not null references catalog.source (id),
    venue_symbol   text not null,
    first_seen     timestamptz not null default now(),
    last_seen      timestamptz not null default now(),
    count          bigint not null default 1,
    sample_raw_ref text,
    status         text not null default 'open' check (status in ('open', 'resolved', 'ignored')),
    primary key (source_id, venue_symbol)
);

create table ingest.backfill_cursor (
    id            uuid primary key,
    source_id     text not null references catalog.source (id),
    instrument_id uuid, -- null for non-instrument series (macro, news)
    timescale     text not null,
    range_start   timestamptz not null,
    range_end     timestamptz not null,
    cursor        timestamptz,
    status        text not null default 'pending' check (status in ('pending', 'running', 'done', 'failed')),
    updated_at    timestamptz not null default now(),
    unique nulls not distinct (source_id, instrument_id, timescale, range_start)
);

create table ingest.connector_run (
    id         uuid primary key,
    source_id  text not null references catalog.source (id),
    started_at timestamptz not null default now(),
    ended_at   timestamptz,
    status     text not null,
    events     bigint not null default 0,
    errors     bigint not null default 0,
    last_error text
);

-- Macro observations with vintages, for point-in-time backtests (specs 20, 41).
create table wh.macro_observation (
    source_id text not null references catalog.source (id),
    series_id text not null,
    period    text not null,
    vintage   timestamptz not null, -- release time of this value
    value     double precision not null,
    revision  boolean not null default false,
    primary key (source_id, series_id, period, vintage)
);
