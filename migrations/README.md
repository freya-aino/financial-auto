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
| `20261110000010_ledger_indexes.sql`, `…11_cash_movement_book_key.sql` | ledger read-path indexes, cash movement idempotency key | 42, D-047 |
| `20261110000040_quarantine_review.sql` | `ingest.quarantine.instrument_id/resolved_at/resolution` (+ backfill from `ref.listing`), search indexes on `ref.instrument.name`, `ref.identifier.value`, `ref.listing.venue_symbol` | 10 §5.3, 12 §6, D-054 |
| `20261111000030_ledger_lots_orphans.sql` | `ledger.orphan_fill`, tax-lot cache `ledger.lot/lot_closing/lot_rebuild`, view `ledger.fill_valued` (native + EUR + FX source/time per fill), as-of indexes | 42 |
| `20261111000050_strategy_pipeline.sql` | `ledger.strategy_state`: statuses `candidate/holdout_passed/retired`, `pipeline` JSONB, `last_reason` | 35 §6.4, 40 §6.6 |
| `20261111000051_risk_state.sql` | `ledger.risk_state` (restart-safe halt state machine, calibration history, tournament records) | 40 §8 |
| `20261111000010_refdata_asof.sql` | bitemporal-lite columns on `ref.entity/venue/calendar_session/fx_rate`, corporate-action lifecycle + symbol changes + natural key, `ref.instrument_relation`, as-of indexes, reference venues, repair of Yahoo fx/crypto classification, `wh.fundamental`, macro as-of index | 12, 20 |
| `20261111000045_wh_filings.sql` | `wh.filing` (FilingEvent metadata, idempotent on source + accession) | 10, 20 |
| `20261112000010_ledger_settlement_payout.sql` | `ledger.venue_settlement` (settlement `eur`/`native` + margin model per venue; paper venue seeded `eur`), `ledger.payout_quota.rule` (rule + inputs that set the quota); idempotent | 42 §10 Q1/Q5 |

Apply with `just migrate`, which needs `sqlx-cli`: `cargo install sqlx-cli --no-default-features --features rustls,postgres`.
