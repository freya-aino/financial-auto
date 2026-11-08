# fa-ledger

Orders/fills/positions/cash ledger, EUR conversion, reconciliation, payout quota.

**Status:** implemented (in-memory) — EUR position/cash/fee accounting, idempotent intents and fills, reconciliation (tested); used as the paper book by `control-api` (D-028) and `bins/risk-engine`. Postgres persistence pending.

## Owns
Ledger repositories and reconciliation logic.

## Governing specs
- [42-ledger-and-accounting](../../specs/42-ledger-and-accounting.md)
