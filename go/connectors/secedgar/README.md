# secedgar

SEC EDGAR filings via the submissions API, catalog id `sec_edgar`. No key; a declared User-Agent is required.

**Status:** implemented; verified 2026-11-11 (`data.sec.gov/submissions/CIK0000320193.json` → 200 with UA `financial-auto ingestor`, 403 "Undeclared Automated Tool" with the ingestor's generic UA).

## Behaviour
- **Config:** `config.extra.ciks` (comma list, e.g. `320193,789019`; not `config.symbols`, which backfill would resolve as instruments), `config.extra.user_agent` or secret `[sources.sec_edgar] user_agent` / `FA_SEC_EDGAR_USER_AGENT` (default `financial-auto ingestor`), `config.extra.lookback` (default `168h`).
- **Poll** (every 10 minutes): one submissions document per CIK.
- **Normalize:** each `filings.recent` entry → `FilingEvent` on `filing.event.sec_edgar`: `issuer_cik` (10 digits), `issuer_lei` (when EDGAR has one), `form_type`, `accession`, `filing_time = acceptanceDateTime` (UTC), `document_refs = [https://www.sec.gov/Archives/edgar/data/<cik>/<accession without dashes>/<primaryDocument>]` (the `-index.htm` page when no primary document). `event_time = publish_time` = acceptance time; without one, event = filingDate 00:00 ET and publish = filingDate 22:00 ET (conservative, EDGAR's last acceptance for that date). No instrument id.
- **De-dup:** accession set in memory plus a per-CIK cursor (`ingest.backfill_cursor`, scope `cik:<cik>`) = newest acceptance published. A first run without a cursor only emits filings accepted within `lookback` so the ~1000 historical entries are not replayed live.
- **Backfill:** the same document filtered to `[From, To)` on acceptance time, for the job's CIK or all configured CIKs; older pages (`filings.files[]`) are fetched when their date span overlaps the range.

## Tests
`secedgar_test.go` on `testdata/submissions_aapl.json` (live, trimmed to 5 filings) and `testdata/submissions_aapl_001.json` (older page, trimmed to 3): mapping, document URLs, dedup, cursor/restart, first-run lookback, User-Agent handling (403), deterministic payloads, backfill range incl. older pages, estimated times, CIK parsing.

## Governing specs
- [10-ingestion](../../../specs/10-ingestion.md), [11-source-catalog](../../../specs/11-source-catalog.md), [20-storage](../../../specs/20-storage.md) (filings metadata in `wh`)
