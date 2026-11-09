# proto

Protobuf contracts (`fa.v1.*`) shared by Rust and Go; managed with buf.

**Status:** implemented — 12 files under `fa/v1/` (envelope, market, latent, decoder, strategy, order, risk, jobs, control, macro, media, text). Rust codegen in `crates/fa-proto/build.rs` (protox, no protoc); Go codegen committed under `go/pkg/bus/pb` via `just proto` (buf, D-002). Messages without a producer yet: `BookSnapshot`, `Quote`, `Bar`, `MacroObservation`, `OnchainEvent`, `FilingEvent`, `MediaChunk`.

## Owns
Canonical events, latent, signal, order, job messages.

## Governing specs
- [21-bus-and-contracts](../specs/21-bus-and-contracts.md)
