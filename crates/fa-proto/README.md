# fa-proto

Prost-generated Rust types from `proto/` (build.rs).

**Status:** implemented — prost codegen via protox (no protoc needed) over every `proto/fa/v1/*.proto` (`build.rs` compiles the directory, so a new file needs no registration); re-exports `prost::Message` and `Timestamp` plus `ts`/`now` helpers. Files: `envelope`, `market`, `text` (incl. `FundamentalFact`, D-078), `macro`, `media`, `latent` (incl. the D-069 membership/lifecycle/MCMC/hierarchy/cross-prior messages and D-073 `ExpertView`/`ExpertHealth`), `decoder`, `strategy`, `order` (`OrderCancel`, D-076; `OrderEvent.venue_fill_id`/`instrument_id`/`venue_symbol`, D-078), `risk` (`Resume`, `RiskEvent`, D-067), `ingest` (`IngestHealth`/`IngestGap`/`IngestUnresolved`/`RefChanged`, D-067), `jobs`, `control`. `buf lint`/`buf breaking` configuration lives in `proto/`.

## Owns
Generated protobuf types; no hand-written logic.

## Governing specs
- [21-bus-and-contracts](../../specs/21-bus-and-contracts.md)
