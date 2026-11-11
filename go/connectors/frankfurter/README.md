# frankfurter

Keyless ECB reference-rate mirror (`https://api.frankfurter.dev/v1`), catalog id `frankfurter`.

- **Poll** hourly: `/<cursor − 7 d>..?base=EUR&symbols=…` → one `MacroObservation` per currency and business day on `macro.obs.frankfurter.<BASE>_<CCY>` (value = CCY per 1 BASE).
- **Vintage** = estimated release, `<date> 16:00 Europe/Berlin`; `publish_time_estimated = true`; publish_time = ingest time live, the estimate in backfill. No revisions.
- **De-dup**: in-memory `(ccy, date)` set + cursor `rates` (newest reference date) in `ingest.backfill_cursor`.
- **Backfill**: yearly range windows over `[From, To)`, clamped to 1999-01-04.
- **Config**: `extra.base` (default EUR), `extra.currencies` (comma list; default USD,GBP,JPY,CHF, the catalog lists 8). Catalog `rate_limit` 0.5/s.
- With `-postgres` the ingestor writes the observations to `wh.macro_observation` (`go/pkg/warehouse`, idempotent per vintage).
- Why not `Quote`: a daily fixing is not a venue bid/ask; quotes would provision fake FX instruments.
- Fixtures in `testdata/` were recorded 2026-11-11.
