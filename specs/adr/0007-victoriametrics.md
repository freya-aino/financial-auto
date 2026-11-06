# ADR 0007 — VictoriaMetrics for time series

**Status:** accepted · **Date:** 2026-11-06

## Context
Ticks, quotes, bars, book snapshots, derived features and system metrics need a high-ingest, compact, queryable time-series store. HelixDB is not a time-series store. The user first asked for a Rust TSDB, then chose **VictoriaMetrics** (Go), noting Go is a good option here.

## Decision
VictoriaMetrics (single-node now) stores market series, features and ops metrics, following the modeling rules in spec 20 §5.3 (bounded labels, top-N book snapshots at fixed cadence, per-ms trade aggregation). Ingest via remote-write/import API from Go; query via MetricsQL/export API from Rust (`fa-tsdb`). Grafana-free: the Svelte frontend queries through the control API.

## Consequences
- Excellent compression and ingest rate; single binary; cluster version available for scaling.
- Data model is float64 + labels + **millisecond** timestamps: no exact decimals (ledger uses Postgres), no per-trade identity, sub-ms ordering lost (exact sequence only in Garage raw archive).
- Label cardinality must be governed (budget, demotion to bars).
- No native bitemporal/as-of semantics → availability lag applied in `fa-tsdb` (spec 20 §6.3).

## Alternatives considered
- **GreptimeDB / InfluxDB 3 (Rust):** closer to the original "Rust TSDB" request, richer SQL; user preferred VictoriaMetrics.
- **QuestDB / ClickHouse:** strong for tick data, heavier ops, not chosen.
- **Embedded Parquet + DataFusion:** max control, more engineering.
