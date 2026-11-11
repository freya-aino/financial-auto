-- Restart-safe hard-limit state (spec 40 §8: "kill switch triggered → persistent until manual
-- re-arm (survives restart)"): one row per book with the risk-engine's halt state machine
-- (peak / day-start equity, every active halt with reason and since, the UTC day), the hourly
-- equity history behind the limit calibration rules and the allocator tournament records
-- (bins/risk-engine engine::RiskRecord). Limits are never stored here: they come from risk.toml.
-- Idempotent.
create table if not exists ledger.risk_state (
    book       text primary key,
    state      jsonb not null,
    updated_at timestamptz not null default now()
);
