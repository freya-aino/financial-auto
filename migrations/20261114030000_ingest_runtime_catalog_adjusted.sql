-- Spec 11 §4/§6.1/§7/§8, spec 12 §6/§7 (2026-11-14). Idempotent.
--
-- 1. Runtime catalog edits (ctl.ingest.configure, PATCH /api/sources/{id}): per-source overrides of
--    the YAML `rate_limit` / `config` blocks. NULL = the YAML value applies. Like `status`, a
--    non-NULL override wins over the YAML seed on every ingestor start and reload.
-- 2. Status provenance: why and when the status last changed (auto-disable after persistent
--    failure records its reason here, spec 11 §6.1/§8).
alter table catalog.source add column if not exists rate_limit jsonb;
alter table catalog.source add column if not exists config jsonb;
alter table catalog.source add column if not exists status_note text not null default '';
alter table catalog.source add column if not exists status_changed_at timestamptz;

-- 3. Materialised adjusted daily bars (spec 12 §6/§7 `adjustment.materialize_daily`): back-adjusted
--    to the latest bar for instruments with effective corporate actions, written by
--    fa_warehouse::adjusted. `actions_digest` identifies the corporate-action set the rows were
--    computed from; a new or cancelled action changes it and the rows are recomputed. The raw
--    series stays in VictoriaMetrics, unchanged.
create table if not exists wh.adjusted_bar_1d (
    instrument_id  uuid not null references ref.instrument (id),
    mode           text not null check (mode in ('splits_only', 'total_return')),
    day            date not null,
    open           double precision not null,
    high           double precision not null,
    low            double precision not null,
    close          double precision not null,
    volume         double precision not null,
    raw_close      double precision not null,
    factor         double precision not null,
    actions_digest text not null,
    computed_at    timestamptz not null default now(),
    primary key (instrument_id, mode, day)
);
