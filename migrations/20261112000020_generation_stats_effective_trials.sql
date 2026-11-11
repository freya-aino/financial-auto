-- Effective number of independent trials behind a generation's deflated Sharpe (spec 41 §6.5):
-- N_eff = N / (1 + (N − 1)·ρ̄) from the mean pairwise correlation of the trials' block returns,
-- plus trials without a finite Sharpe. NULL for rows computed before this column existed.
alter table validation.generation_stats add column if not exists effective_trials double precision;
