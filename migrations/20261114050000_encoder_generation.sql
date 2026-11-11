-- Encoder generation state machine (spec 36 §4.1, §5): one row per generation run of the
-- trainer (`trainer cycle` / `trainer bootstrap` / `train.cycle` jobs), its current `GenPhase`,
-- status and the full record (phase events with attempts, frozen data window, candidates, Pareto
-- selection, artefacts, validation). `data_window` keeps the train / validation boundaries and
-- the excluded locked holdouts of each generation. Written by the trainer, read by control-api
-- (`/api/generations`). Additive and idempotent.
create table if not exists validation.encoder_generation (
    generation_id     text primary key,
    timescale         text not null,
    kind              text not null,               -- bootstrap | cycle
    phase             text not null,               -- planned … validate | promoted | rejected
    status            text not null,               -- running | awaiting_promotion | promoted | rejected | failed | aborting | aborted
    job_id            text,
    selected_snapshot text,
    record            jsonb not null default '{}'::jsonb,
    started_at        timestamptz not null default now(),
    updated_at        timestamptz not null default now()
);
create index if not exists encoder_generation_ts_idx on validation.encoder_generation (timescale, started_at desc);
create index if not exists encoder_generation_selected_idx on validation.encoder_generation (selected_snapshot);

create table if not exists validation.data_window (
    generation_id text not null references validation.encoder_generation (generation_id) on delete cascade,
    kind          text not null,                   -- train | validation | holdout_excluded
    timescale     text not null,
    start_at      timestamptz not null,
    end_at        timestamptz not null,
    detail        jsonb,
    primary key (generation_id, kind, start_at)
);
