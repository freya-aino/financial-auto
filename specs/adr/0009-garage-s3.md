# ADR 0009 — Garage as S3 object store

**Status:** accepted · **Date:** 2026-11-06

## Context
The platform produces and consumes large blobs: raw API payloads (exact replay), video/audio/images from broadcasts, model weights, encoder snapshots, adapters, checkpoints of preemptible jobs, decoder outputs (rendered charts, generated text). The user chose **Garage** (Rust, lightweight S3-compatible store).

## Decision
Garage provides buckets `fa-raw`, `fa-media`, `fa-models`, `fa-ckpt`, `fa-artifacts`, `fa-backup` (layout in spec 20 §5.5). All other stores reference blobs by key (content hash where practical). Access via S3 SDKs (Rust `fa-blob`, Go AWS SDK).

## Consequences
- S3 API means tools and SDKs work unchanged; scaling to more nodes is native to Garage (geo-distributed replication).
- Single-node Garage has no redundancy → backups/replication matter once real money is involved.
- Raw archive is the **source of truth for replay** (spec 41); its retention is "forever".
- Garage lacks some S3 features (e.g., certain lifecycle/versioning APIs) — retention jobs implemented by us.

## Alternatives considered
- **MinIO:** more features, heavier, licensing changes.
- **Plain filesystem (content-addressed):** simplest, but no S3 API and harder to scale.
