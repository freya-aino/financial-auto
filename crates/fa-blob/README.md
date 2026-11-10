# fa-blob

Garage S3 access: raw payload archive, media, model snapshots, checkpoints.

**Status:** implemented — Garage key layout (raw, media, snapshots, adapters, genome weights, checkpoints, decoder artifacts, evo eval reports `fa-artifacts/evals/<generation>/<job>.json`); async `BlobStore` trait (put with content type, get, exists, delete, list prefix) with `MemoryStore` and `S3BlobStore`: a path-style S3 client over plain `reqwest` with a hand-rolled AWS SigV4 signer (`sigv4.rs`, tested against the aws-sig-v4-test-suite `get-vanilla*`/`get-utf8` vectors and the S3 docs GET/presigned examples), ListObjectsV2 parsing via `quick-xml` with pagination, and presigned GET URLs for the UI. `S3BlobStore::from_config(&PlatformConfig, &Secrets)` reads `[garage] endpoint/region` from `config/platform.toml` and the key from `config/secrets.toml` (D-020); it returns `BlobError::NotConfigured` without credentials. The live round-trip test `tests/garage.rs` runs only with `FA_GARAGE_TEST=1` against the compose Garage (`docker/garage-init.sh` first) and was green on 2026-11-10. `evo-worker` uses it to persist eval reports.

```sh
FA_GARAGE_TEST=1 cargo test -p fa-blob --test garage
```

## Owns
Content-addressed blob keys and snapshot layout.

## Governing specs
- [20-storage](../../specs/20-storage.md)
- [36-training-lifecycle](../../specs/36-training-lifecycle.md)
