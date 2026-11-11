# proto

Protobuf contracts (`fa.v1.*`) shared by Rust and Go; managed with buf.

**Status:** implemented — 13 files under `fa/v1/` (envelope, market, latent, decoder, strategy, order, risk, jobs, control, ingest, macro, media, text). Recent additions: `IngestHealth`/`IngestGap`/`IngestUnresolved`/`RefChanged` (ingest.proto) and `Resume`/`RiskEvent` (risk.proto; `Resume` keeps `Halt`'s field numbers 1–4), `Envelope` fields 9–12 (D-067); the D-069 latent messages (`RegionMembership`, `RegionLifecycleEvent`, `KernelUpdate`, `RegionHierarchy`, `CrossAssetPriors`, …) and `ExpertView`/`ExpertHealth` (D-073); `OrderCancel` (D-076); `OrderEvent` fields 12–14 `venue_fill_id`, `instrument_id`, `venue_symbol` (D-078). Rust codegen in `crates/fa-proto/build.rs` (protox, no protoc); Go codegen committed under `go/pkg/bus/pb` via `just proto` (buf lint + generate, D-002); `buf.yaml` also configures `buf breaking` (FILE). Messages without a producer yet: `BookSnapshot`, `OnchainEvent`, `MediaChunk` (`Quote`/`Bar` come from Kraken, Yahoo, Lemon, OANDA; `MacroObservation` from ECB and Frankfurter; `FilingEvent`/`FundamentalFact` from SEC EDGAR).

## Owns
Canonical events, ingest health, latent, decoder, signal, order, risk, job and control messages.

## Governing specs
- [21-bus-and-contracts](../specs/21-bus-and-contracts.md)
