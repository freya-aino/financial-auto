# migrations

PostgreSQL warehouse + ledger schema, in sqlx format (`<timestamp>_<name>.sql`, D-013).

**Status:** initial schema written. Verified 2026-11-06 by applying all files in order to a scratch database on `postgres:17-alpine`: 23 tables created, no errors.

| File | Schema | Spec |
|---|---|---|
| `…01_schemas.sql` | `catalog`, `ref`, `ingest`, `wh`, `ledger`, `audit` | 20 §5.4 |
| `…02_catalog.sql` | `catalog.source` | 11 |
| `…03_reference.sql` | `ref.entity/instrument/identifier/venue/listing/contract_spec/calendar_session/corporate_action/fx_rate` | 12 |
| `…04_ingest.sql` | `ingest.quarantine/backfill_cursor/connector_run`, `wh.macro_observation` | 10, 20 |
| `…05_ledger.sql` | `ledger.order_intent/venue_order/fill/fill_attribution/cash_movement/position_snapshot/payout_quota/reconciliation_run` | 42 |
| `…06_audit.sql` | `audit.log` | 42 |

Apply with `just migrate`, which needs `sqlx-cli`: `cargo install sqlx-cli --no-default-features --features rustls,postgres`.
