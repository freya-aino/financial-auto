-- Holdout lock workflow (spec 41 §6.4, spec 40 §6.6; D-059). Additive and idempotent: the
-- validation.holdout table (migration 20261110000030) stays immutable by trigger; rows gain a
-- stable id the holdout evaluation job (`evo.eval.strategy.holdout`) and the control-api
-- `POST /api/validation/holdout` route refer to. ALTER TABLE does not fire the row trigger.
alter table validation.holdout add column if not exists id uuid not null default gen_random_uuid();
create unique index if not exists holdout_id_idx on validation.holdout (id);
comment on column validation.holdout.id is 'stable handle for promotion evaluations and the API (D-059)';

-- Holdout evaluations are trial rows with kind = ''holdout'' and the holdout they ran on; they are
-- excluded from every selection statistic (evo-worker TrialStore::trials, control-api pareto).
alter table validation.trial add column if not exists holdout_id uuid;
create index if not exists trial_kind_idx on validation.trial (kind, created_at desc);
create index if not exists trial_holdout_idx on validation.trial (holdout_id) where holdout_id is not null;
