# archive

Garage (S3) raw-payload writer — the ingestor's `RawArchive` (spec 10 §4.2, spec 20 §5.5).

**Status:** implemented with `github.com/aws/aws-sdk-go-v2/service/s3` (SigV4, path-style, static credentials, checksums only when required — Garage compatible), live-tested against compose Garage (`FA_GARAGE_URL` + `FA_GARAGE_ACCESS_KEY_ID` + `FA_GARAGE_SECRET_ACCESS_KEY` gate the test).

- Bucket `fa-raw`, key `raw/<source>/<yyyy>/<mm>/<dd>/<hh>/<ulid>.raw.gz`: one object per `(source, UTC minute)`, gzip of the concatenated payloads each terminated by `\n` (gzip from the standard library instead of the spec's zstd — no extra dependency; D-050).
- `Put` returns a `RawRef{key, offset, length}` immediately; `offset`/`length` index the **decompressed** stream. Uploads happen when the minute rolls over, when a batch exceeds 8 MiB, and on shutdown (`FlushSync`); at-least-once, 3 attempts.
- **Index sidecar:** next to every data object an index object `<ulid>.idx.jsonl.gz` (one `IndexEntry` per payload: offset, length, `received_at`, content type, transport meta) is written, so archived payloads can be rebuilt exactly as received (D-067).
- **Spool (spec 10 §8):** with `WriterOptions.SpoolDir` (ingestor `-spool-dir` / `FA_ARCHIVE_SPOOL`) a batch whose upload failed is kept on local disk (bounded, default 256 MiB) and retried every 30 s; while the spool is full `Put` returns `ErrSpoolFull` and the pipeline flags the envelopes `raw_missing`.
- **Replay** (`replay.go`): `Replay(ctx, store, bucket, source, from, to)` lists the hour prefixes covering `[from, to)` (ListObjectsV2, paginated), reads data + index objects and returns `ReplayItem{Raw, Ref}` in `(received_at, key, offset)` order with the same `RawRef` the live pipeline handed out; objects without an index are counted (`MissingIndex`), not read. Used by the ingestor's `ctl.ingest.replay`.
- `Stats{Messages, Objects, Bytes, Errors, Spooled, Respooled, Dropped, SpoolBytes}` are mirrored into the ingestor heartbeat counters `archive.messages|objects|errors|spooled|respooled|raw_missing|spool_bytes`.
- The evo-orchestrator reuses `S3Store` to write promotion artifacts to `fa-artifacts`.

## Governing specs
- [10-ingestion](../../../specs/10-ingestion.md), [20-storage](../../../specs/20-storage.md) §5.5, [41-backtest-and-validation](../../../specs/41-backtest-and-validation.md) §5
