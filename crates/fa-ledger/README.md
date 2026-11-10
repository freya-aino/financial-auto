# fa-ledger

Orders/fills/positions/cash ledger, EUR conversion, reconciliation, payout quota.

**Status:** implemented — in-memory `Ledger` (EUR position/cash/fee accounting, idempotent intents and fills, `reconcile`/`diff_books`) plus `pg::PgLedger` persisting to the `ledger.*` tables (intents + paper venue orders, fills with strategy attribution, cash movements, position snapshots, reconciliation runs, payout quota; idempotency enforced by the DB constraints via `ON CONFLICT DO NOTHING`) and `load_book(book)` rebuilding a `Ledger` from cash movements + fills (D-047). Tested in memory and against the live DB (`DATABASE_URL`). Used as the paper book by `control-api` (seeded from Postgres at startup) and `bins/risk-engine` (persists). Money is `NUMERIC` in Postgres, `f64` in Rust. Since D-061 `PgLedger::record_fill_with(fill, at, strategies, &FillDetails {ccy, fx_source, arrival_price, slippage_bps})` stores the fill currency, FX source and execution quality. `ledger.fill.slippage_bps` comes from migration `20261110000060`, and `record_fill` remains the EUR/`fixed` shorthand. `FillRow` reads these fields back, and `slippage_by_strategy` gives the mean realised slippage per attributed strategy. `Fill.fx_to_eur` is set by risk-engine from the ECB as-of table (1.0 for EUR and without Postgres).

## Owns
Ledger repositories and reconciliation logic.

## Governing specs
- [42-ledger-and-accounting](../../specs/42-ledger-and-accounting.md)
