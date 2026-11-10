# fa-ledger

Orders/fills/positions/cash ledger, EUR conversion, reconciliation, payout quota.

**Status:** implemented — in-memory `Ledger` (EUR position/cash/fee accounting, idempotent intents and fills, `reconcile`/`diff_books`) plus `pg::PgLedger` persisting to the `ledger.*` tables (intents + paper venue orders, fills with strategy attribution, cash movements, position snapshots, reconciliation runs, payout quota; idempotency enforced by the DB constraints via `ON CONFLICT DO NOTHING`) and `load_book(book)` rebuilding a `Ledger` from cash movements + fills (D-047). Tested in memory and against the live DB (`DATABASE_URL`). Used as the paper book by `control-api` (seeded from Postgres at startup) and `bins/risk-engine` (persists). Money is `NUMERIC` in Postgres, `f64` in Rust; FX fixed at 1.0 (D-028).

## Owns
Ledger repositories and reconciliation logic.

## Governing specs
- [42-ledger-and-accounting](../../specs/42-ledger-and-accounting.md)
