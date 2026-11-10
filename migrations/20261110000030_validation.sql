-- Validation registry (spec 41 §5, §6.4, §6.5; D-052). Idempotent so a manually applied copy
-- does not break sqlx::migrate on the next control-api start.
create schema if not exists validation;

-- Locked holdout periods per timescale: used only by Promotion replays; Selection replays whose
-- window intersects a row are rejected by the evaluator (fa_backtest::replay::check_holdout).
-- Immutable once locked: updates and deletes are refused; a re-lock is a new row (audit log).
create table if not exists validation.holdout (
    timescale  text        not null,
    start_at   timestamptz not null,
    end_at     timestamptz not null,
    locked_at  timestamptz not null default now(),
    note       text,
    primary key (timescale, start_at),
    check (end_at > start_at)
);

create or replace function validation.holdout_immutable() returns trigger
language plpgsql as $$
begin
    raise exception 'validation.holdout is immutable once locked (insert a new row to re-lock)';
end;
$$;

drop trigger if exists holdout_immutable on validation.holdout;
create trigger holdout_immutable
    before update or delete on validation.holdout
    for each row execute function validation.holdout_immutable();

-- Trial registry: every evaluated candidate (decoder or strategy) with its selection-window
-- statistics; the input of the deflated Sharpe and PBO corrections (spec 41 §6.5).
-- `returns` holds the per-step net returns compressed into fa_backtest::validation::N_BLOCKS
-- block sums so trials are comparable in CSCV; `objectives` is the worker's objective map.
create table if not exists validation.trial (
    id           uuid        primary key default gen_random_uuid(),
    generation   text        not null,              -- generation id (Job.parent_job_id)
    gen_index    integer     not null default 0,    -- loop index inside the generation
    rung         integer     not null default 0,
    job_id       text        not null unique,
    candidate    text        not null,              -- genome text form
    kind         text        not null,              -- evo.eval.decoder | evo.eval.strategy
    window_start timestamptz,
    window_end   timestamptz,
    sharpe       double precision,
    n_obs        integer     not null default 0,
    objectives   jsonb       not null default '{}'::jsonb,
    returns      double precision[],
    created_at   timestamptz not null default now()
);
create index if not exists trial_generation_idx on validation.trial (generation, gen_index, created_at);
create index if not exists trial_candidate_idx on validation.trial (candidate, created_at desc);

-- Per-generation statistics over the trial set, recomputed by the worker after every trial.
create table if not exists validation.generation_stats (
    generation          text        not null,
    gen_index           integer     not null default 0,
    kind                text        not null,
    n_trials            integer     not null,
    sharpe_mean         double precision,
    sharpe_var          double precision,
    best_candidate      text,
    best_sharpe         double precision,
    expected_max_sharpe double precision,
    deflated_sharpe     double precision,
    pbo                 double precision,
    pbo_trials          integer     not null default 0,
    pbo_blocks          integer     not null default 16,
    computed_at         timestamptz not null default now(),
    primary key (generation, gen_index)
);
