# ADR 0008 — PostgreSQL warehouse for static records and the ledger

**Status:** accepted · **Date:** 2026-11-06

## Context
The user wants a data-warehouse-type structure on Postgres for all static records. The system also needs an ACID store for orders, fills, positions and cash, and bitemporal reference data for point-in-time backtests.

## Decision
PostgreSQL holds schemas: `ref` (instruments, identifiers, venues, calendars, corporate actions, contract specs, FX), `catalog` (sources), `ingest` (cursors, quarantine), `wh` (fundamentals, macro vintages, filings metadata), `ledger` (intents/orders/fills/positions/cash), `audit` (append-only control log), `ops` (jobs, config snapshots). Migrations via sqlx in `migrations/`. Rust access via `fa-warehouse`/`fa-ledger`, Go via `pgx`.

## Consequences
- Mature, transactional, easy backups; exact decimals for money.
- Reference data mirrored into Helix as graph edges → sync process needed (`ref.changed.*`).
- **Trading halts if Postgres is unavailable** (ledger is mandatory).
- Not used for high-frequency series (VictoriaMetrics) or blobs (Garage).

## Alternatives considered
- **Helix for everything:** poor fit for ledger/decimal/bitemporal needs.
- **DuckDB warehouse:** great analytics, weak concurrent writes; may be added later for offline analysis over Parquet exports.
