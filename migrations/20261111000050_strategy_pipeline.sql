-- Automatic evolved-strategy lifecycle (spec 35 §6.4, spec 40 §6.6): the pipeline statuses
-- candidate | holdout_passed | retired join active | paused | flattening | shadow; `pipeline`
-- holds the genome text, generation, selection / holdout evidence and attempt bookkeeping
-- (fa_risk::pipeline::PipelineRecord), `last_reason` why the last automatic transition happened
-- (survives restarts so a drawdown pause can be told from an operator pause). Idempotent, so a
-- manually applied copy does not break sqlx::migrate on the next control-api start.
alter table ledger.strategy_state drop constraint if exists strategy_state_status_check;
alter table ledger.strategy_state add constraint strategy_state_status_check
    check (status in ('active', 'paused', 'flattening', 'shadow', 'candidate', 'holdout_passed', 'retired'));
alter table ledger.strategy_state add column if not exists pipeline jsonb;
alter table ledger.strategy_state add column if not exists last_reason text;
create index if not exists strategy_state_evolved_idx on ledger.strategy_state (status) where pipeline is not null;
comment on column ledger.strategy_state.pipeline is
    'evolved strategies: genome, generation, selection/holdout evidence, attempts, demotions (fa_risk::pipeline::PipelineRecord); NULL for hand-written ones';
