# fa-risk

Non-evolvable hard risk limits, kill switch, health halts.

**Status:** implemented (v1) — position/liquidity/leverage caps (`cap_targets`), payout-quota netting, drawdown/daily-loss halt state machine with reduce-only gating (`gate`), manual re-arm (`rearm`), bus scope mapping (`HaltScope::parse/as_str`) (tested). Used live by `bins/risk-engine`. Per-strategy drawdown, concentration per class/venue and health halts from `ingest.health.*` pending.

## Owns
Pre-trade checks and halt state machine.

## Governing specs
- [40-execution-and-risk](../../specs/40-execution-and-risk.md)
