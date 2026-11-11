-- The `ops` schema of spec 20 §5.4 (job registry, config snapshots) plus the alert history of
-- spec 60 §6.5. Written by control-api through `fa_warehouse::ops`:
--   ops.alert_event     every alert transition control-api publishes on `ops.alert` (vmalert
--                       receiver, expiries), so `GET /api/alerts` history survives restarts
--   ops.job_run         one row per finished job (`jobs.result.<kind>` from the JOB_RESULTS
--                       stream, durable consumer `ops-job-history`), the job registry's history
--   ops.config_snapshot the effective (redacted) configuration whenever it changes
-- Append-only except job_run, which keeps the latest result per job id. Idempotent.
create schema if not exists ops;

create table if not exists ops.alert_event (
    id            bigserial primary key,
    fingerprint   text not null,
    alertname     text not null,
    status        text not null check (status in ('firing', 'resolved')),
    severity      text not null default '',
    summary       text not null default '',
    labels        jsonb not null default '{}'::jsonb,
    annotations   jsonb not null default '{}'::jsonb,
    starts_at     timestamptz,
    ends_at       timestamptz,
    generator_url text,
    expired       boolean not null default false,
    at            timestamptz not null,
    recorded_at   timestamptz not null default now(),
    unique (fingerprint, status, at)
);
create index if not exists alert_event_at_idx on ops.alert_event (at desc);
create index if not exists alert_event_name_at_idx on ops.alert_event (alertname, at desc);

create table if not exists ops.job_run (
    job_id         text primary key,
    kind           text not null,
    status         text not null,
    worker_id      text,
    error          text,
    used_gpu_ms    bigint not null default 0,
    output_refs    jsonb not null default '[]'::jsonb,
    checkpoint_ref text,
    -- the JobResult.result payload: JSON when it parses as JSON, else {"bytes": n}
    result         jsonb,
    stream_seq     bigint,
    finished_at    timestamptz not null,
    recorded_at    timestamptz not null default now()
);
create index if not exists job_run_kind_finished_idx on ops.job_run (kind, finished_at desc);
create index if not exists job_run_finished_idx on ops.job_run (finished_at desc);

create table if not exists ops.config_snapshot (
    id       bigserial primary key,
    service  text not null,
    section  text not null,
    digest   text not null,
    config   jsonb not null,
    actor    text not null,
    taken_at timestamptz not null default now(),
    unique (service, section, digest)
);
create index if not exists config_snapshot_taken_idx on ops.config_snapshot (service, section, taken_at desc);
