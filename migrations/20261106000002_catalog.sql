-- Source catalog (spec 11). Seeded from config/sources.yaml; Postgres is authoritative at runtime.
create table catalog.source (
    id            text primary key,
    name          text        not null,
    category      text        not null,
    capabilities  text[]      not null default '{}',   -- data | execution
    asset_classes text[]      not null default '{}',
    transport     text[]      not null default '{}',
    auth          text        not null,
    status        text        not null default 'planned'
                  check (status in ('planned', 'stub', 'live', 'disabled')),
    tos_notes     text        not null default '',
    notes         text        not null default '',
    seeded_at     timestamptz not null default now(),
    updated_at    timestamptz not null default now()
);
create index source_category_idx on catalog.source (category);
create index source_status_idx on catalog.source (status);
