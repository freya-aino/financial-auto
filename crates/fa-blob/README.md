# fa-blob

Garage S3 access: raw payload archive, media, model snapshots, checkpoints.

**Status:** implemented — Garage key layout (raw, media, snapshots by timescale/family/generation and by id `fa-models/snapshots/<snapshot_id>/<name>`, adapter states, genome weights, checkpoints and resumable job state `fa-ckpt/<job>/state.json`, decoder artifacts, evo eval reports `fa-artifacts/evals/<generation>/<job>.json`); async `BlobStore` trait (put with content type, get, exists, delete, list prefix) with `MemoryStore` and `S3BlobStore`: a path-style S3 client over plain `reqwest` with a hand-rolled AWS SigV4 signer (`sigv4.rs`, tested against the aws-sig-v4-test-suite `get-vanilla*`/`get-utf8` vectors and the S3 docs GET/presigned examples), ListObjectsV2 parsing via `quick-xml` with pagination, and presigned GET URLs for the UI. `S3BlobStore::from_config(&PlatformConfig, &Secrets)` reads `[garage] endpoint/region` from `config/platform.toml` and the key from `config/secrets.toml` (D-020); it returns `BlobError::NotConfigured` without credentials. The live round-trip test `tests/garage.rs` runs only with `FA_GARAGE_TEST=1` against the compose Garage (`docker/garage-init.sh` first) and was green on 2026-11-10. Users: evo-worker (eval reports, strategy/holdout/allocator artifacts, neural decoder bundles), trainer (snapshot artifacts, checkpoints, decoder feedback), model-runtime (snapshots, neural serving, `mcmc.refit` windows and draws), risk-engine (strategy pipeline files), strategy-runtime and control-api (latent, evolution and console reads); callers that write other `fa-artifacts`/`fa-models` paths (e.g. `fa-artifacts/mcmc/…`, `fa-artifacts/live/strategies.toml`, `fa-models/decoders/<gen>/…`) build those keys themselves.

Key helpers in use: `snapshot_file`/`snapshot_prefix`, `adapter_state`, `job_state`, `eval_report` and the bucket constants. The others (`raw_payload`, `media`, `snapshot` by timescale/family/generation, `adapter`, `genome_weights`, `checkpoint`, `decoder_artifact`) encode the spec 20 §5.5 proposal and have no caller. In particular the raw archive is written by the Go ingestor (`go/pkg/archive`), not through this crate, as `fa-raw/raw/<source>/<yyyy>/<mm>/<dd>/<hh>/<ulid>.raw.gz` plus an `<ulid>.idx.jsonl.gz` index, while `raw_payload` still produces the unused `<source>/…/<ulid>.zst` layout. The `fa-backup` bucket (created by `docker/garage-init.sh`, copied by the backup sidecar) has no constant here.

```sh
FA_GARAGE_TEST=1 cargo test -p fa-blob --test garage
```

## Owns
Content-addressed blob keys and snapshot layout.

## Governing specs
- [20-storage](../../specs/20-storage.md)
- [36-training-lifecycle](../../specs/36-training-lifecycle.md)
