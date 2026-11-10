# archive

Garage (S3) raw-payload writer — the ingestor's `RawArchive` (spec 10 §4.2, spec 20 §5.5).

**Status:** implemented with `github.com/aws/aws-sdk-go-v2/service/s3` (SigV4, path-style, static credentials, checksums only when required — Garage compatible), live-tested against compose Garage (`FA_GARAGE_URL` + `FA_GARAGE_ACCESS_KEY_ID` + `FA_GARAGE_SECRET_ACCESS_KEY` gate the test).

- Bucket `fa-raw`, key `raw/<source>/<yyyy>/<mm>/<dd>/<hh>/<ulid>.raw.gz`: one object per `(source, UTC minute)`, gzip of the concatenated payloads each terminated by `\n` (gzip from the standard library instead of the spec's zstd — no extra dependency; D-050).
- `Put` returns a `RawRef{key, offset, length}` immediately; `offset`/`length` index the **decompressed** stream. Uploads happen when the minute rolls over, when a batch exceeds 8 MiB, and on shutdown (`FlushSync`); at-least-once, 3 attempts.
- `Stats{Messages, Objects, Bytes, Errors}` are mirrored into the ingestor heartbeat counters `archive.*`.

## Governing specs
- [10-ingestion](../../../specs/10-ingestion.md), [20-storage](../../../specs/20-storage.md) §5.5, [41-backtest-and-validation](../../../specs/41-backtest-and-validation.md) §5
