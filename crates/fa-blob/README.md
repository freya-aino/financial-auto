# fa-blob

Garage S3 access: raw payload archive, media, model snapshots, checkpoints.

**Status:** partial — Garage key layout implemented + tested (raw, media, snapshots, adapters, genome weights, checkpoints, decoder artifacts, evo eval reports `fa-artifacts/evals/<generation>/<job>.json`); `BlobStore` trait with in-memory impl. **S3 impl pending** (needs a SigV4 client — no S3 crate in the workspace — and `config/secrets.toml` credentials); until then `evo-worker` only records the planned key.

## Owns
Content-addressed blob keys and snapshot layout.

## Governing specs
- [20-storage](../../specs/20-storage.md)
- [36-training-lifecycle](../../specs/36-training-lifecycle.md)
