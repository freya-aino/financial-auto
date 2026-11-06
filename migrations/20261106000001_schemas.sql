-- Schemas (specs 11, 12, 20, 42). Money as NUMERIC(38,18), times as timestamptz (UTC).
create schema if not exists catalog;   -- source catalog (spec 11)
create schema if not exists ref;       -- reference data (spec 12)
create schema if not exists ingest;    -- ingestion bookkeeping (spec 10)
create schema if not exists wh;        -- static/slow records: macro vintages, fundamentals, filings (spec 20)
create schema if not exists ledger;    -- orders, fills, cash, positions (spec 42)
create schema if not exists audit;     -- audit log (spec 42)
