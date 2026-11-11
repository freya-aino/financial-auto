# ADR 0002 — HelixDB as central graph + vector hub

**Status:** accepted · **Date:** 2026-11-06

## Context
The core model is a graph over latent regions with vector embeddings (expert spaces, news, media, historical windows) plus deep lineage of evolved genomes. Queries combine traversal (transitions, lineage, instrument relations) with vector similarity (retrieval decoders, region lookup). The user designated HelixDB (Rust, graph-vector database) as the central data hub.

## Decision
HelixDB stores: instrument/entity/venue graph (mirrored from Postgres `ref` by control-api: full sync at start and periodically, incremental on `ref.changed.*`), expert-space registry, latent regions, transitions with kernel summaries, state observations at decision cadence, retrieval windows, text/media embeddings, genomes, fitness records, lineage, snapshot metadata. Schema in `helix/schema.hx`, queries in `helix/queries.hx`, Rust access via `fa-hub`.

## Consequences
- Graph + vector queries in one engine (e.g., "k-NN windows in expert space E, then follow TRANSITION edges").
- HelixDB is **not** used for raw time series, blobs, or the financial ledger (see ADR 0007/0008/0009).
- HelixDB is a young project: scale limits (vector count, write throughput, transactional guarantees) are unverified → **benchmark spike required** before 30–34 depend on Helix-only queries (spec 20 open question 2).
- Live runtime must tolerate Helix outages via in-memory graph + buffered writes.

## Alternatives considered
- **Neo4j/Memgraph + separate vector DB (Qdrant):** mature, but two systems and JVM/C++ ops; rejected per user choice.
- **Postgres + pgvector + recursive CTEs:** simpler ops, weaker graph traversal ergonomics at scale.
- **SurrealDB:** multi-model, less vector focus.
