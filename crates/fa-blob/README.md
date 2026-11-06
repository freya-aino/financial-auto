# fa-blob

Garage S3 access: raw payload archive, media, model snapshots, checkpoints.

**Status:** partial — Garage key layout implemented + tested; `BlobStore` trait with in-memory impl, S3 impl pending.

## Owns
Content-addressed blob keys and snapshot layout.

## Governing specs
- [20-storage](../../specs/20-storage.md)
- [36-training-lifecycle](../../specs/36-training-lifecycle.md)
