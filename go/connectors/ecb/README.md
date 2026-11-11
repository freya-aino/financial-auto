# ecb

Keyless ECB Data Portal SDMX 2.1 REST (`https://data-api.ecb.europa.eu/service/data`), catalog id `ecb`.

- **Queries**: `config.queries` = `<FLOW>/<KEY>` (default HICP euro-area annual rate, MRO, DFR, EXR USD).
- **Poll** hourly with `includeHistory=true&lastNObservations=3`; every published version (`validFrom`) of an observation becomes its own `MacroObservation` on `macro.obs.ecb.<FLOW>:<dims>` with vintage = publish_time = `validFrom`; `revision = true` when an earlier vintage of the same period is in the payload. `Delete` datasets and null values are skipped.
- **History-less flows** (EXR: `validFrom == prepared`): deterministic estimated vintage (daily → 16:00 Europe/Berlin, else period end) and `publish_time_estimated = true`.
- **event_time** = end of the reference period.
- **De-dup**: `(series, period, vintage)` set + cursor `q:<FLOW>/<KEY>` (newest vintage).
- **Backfill**: 5-year `startPeriod`/`endPeriod` windows per query (or the job's instrument, if it is a query). The source has no venue, so the ingestor does not resolve job instruments into `ref` instruments; each query gets its own range cursor (scope `<timescale>|<query>`).
- **WAF**: the portal answers HTML block pages (400/504) to bursts and some user agents; they are reported as errors. Keep ≥ 10 s between requests (catalog `rate_limit` 0.1/s, burst 1).
- With `-postgres` the ingestor writes every observation to `wh.macro_observation` (`go/pkg/warehouse`, idempotent per `(source, series, period, vintage)`).
- `testdata/hicp_revision_synthetic.json` is hand-made (a flash/final pair); the others were recorded 2026-11-11.
