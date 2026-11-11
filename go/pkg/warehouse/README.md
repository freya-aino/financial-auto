# warehouse

The ingestor's Postgres write path for slow canonical records (spec 20 §6.1 (b)). Driver: pgx v5.

**Status:** implemented. The ingestor uses it with `-postgres` (`Pipeline.Warehouse`). Live tests run against the compose Postgres and are gated by `FA_TEST_POSTGRES_URL`.

- `MacroRows` / `FilingRows` / `FundamentalRows` pick the matching envelopes out of everything a connector produced and count the invalid ones. Envelopes of any other kind are ignored, so the pipeline can pass every batch.
- `Writer.Write` sends one pgx batch per call (10 s timeout). Every insert is `on conflict do nothing`, so live polls, backfill jobs and archive replays may all write the same record (spec 10 §6.2: revisions are new vintages, never overwrites):

| Envelope | Table | Idempotency key |
|---|---|---|
| `MacroObservation` (frankfurter, ecb) | `wh.macro_observation` | `(source_id, series_id, period, vintage)` |
| `FilingEvent` (sec_edgar) | `wh.filing` | `(source_id, accession)` |
| `FundamentalFact` (sec_edgar XBRL, D-078) | `wh.fundamental` | CIK, field, unit, period, vintage, source; `instrument_id` is set only when exactly one instrument carries the CIK (`ref.identifier` scheme `cik`) |

- `Stats{Inserted, Duplicates, Invalid, Errors}` are cumulative. The ingestor reports a failed batch as a connector error.

## Governing specs
- [20-storage](../../../specs/20-storage.md) §6.1, [10-ingestion](../../../specs/10-ingestion.md) §6.2, §6.4
