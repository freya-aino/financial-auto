-- Audit log (spec 42): every manual action and every risk/state transition.
create table audit.log (
    id      bigserial primary key,
    at      timestamptz not null default now(),
    actor   text not null, -- 'system:<service>' | 'ui'
    action  text not null,
    subject text,
    payload jsonb
);
create index audit_log_at_idx on audit.log (at);
