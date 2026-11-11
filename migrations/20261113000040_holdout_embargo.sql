-- Holdout embargo on the lock record (spec 41 §6.4/§7). The locker (risk-engine auto-lock,
-- `[holdout.embargo_secs]` → `fa_backtest::holdout::LockPlan::Lock.embargo_ms`) records the
-- embargo.<timescale> it locked with; evo-worker keeps selection windows
-- max(embargo_ms, 2 × strategy horizon, 60 s) before the holdout
-- (`fa_backtest::holdout::selection_embargo_ms`). NULL = locked without one (manual API, rows
-- before this migration): readers use the timescale default. Additive and idempotent; ALTER
-- TABLE does not fire the immutability row trigger.
alter table validation.holdout add column if not exists embargo_ms bigint;
alter table validation.holdout drop constraint if exists holdout_embargo_nonneg;
alter table validation.holdout add constraint holdout_embargo_nonneg check (embargo_ms is null or embargo_ms >= 0);
comment on column validation.holdout.embargo_ms is
    'embargo.<timescale> the holdout was locked with (ms); NULL = timescale default (spec 41 §6.4)';
