-- Filing metadata (spec 20 §5.1 "filings metadata" → Postgres wh; spec 10 §5.2 FilingEvent).
-- Written by the ingestor's warehouse writer (go/pkg/warehouse) for every FilingEvent envelope,
-- idempotent on (source_id, accession). Documents stay at the source / in Garage (document_refs).
create table if not exists wh.filing (
    source_id     text not null references catalog.source (id),
    accession     text not null,
    issuer_cik    text,
    issuer_lei    text,
    form_type     text not null,
    filing_time   timestamptz not null, -- acceptance time at the source (event time)
    publish_time  timestamptz,          -- when the source made it available
    document_refs text[] not null default '{}',
    event_id      text,                 -- canonical envelope id (stable, spec 10 §9)
    recorded_at   timestamptz not null default now(),
    primary key (source_id, accession)
);
create index if not exists filing_issuer_time_idx on wh.filing (issuer_cik, filing_time desc);
create index if not exists filing_form_time_idx on wh.filing (form_type, filing_time desc);
