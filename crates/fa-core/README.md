# fa-core

Shared domain types: Instrument, Venue, Timescale, Horizon, canonical event structs, ids.

**Status:** implemented — ids, timescales/horizons, platform config (`config.rs`: `PlatformConfig::load` fills `[garage] public_endpoint` (defaults to `endpoint`; the host-reachable S3 endpoint presigned URLs are built on) and merges the git-ignored `config/runtime.toml` next to the platform file over the operator-tunable `[runtime]` section — `ui.poll_ms`, `bus_tap.max_msgs_per_sec`, `live.max_*`, `ws.*_hz`, each bounded by `RUNTIME_BOUNDS` and checked by `RuntimeConfig::validate`; `merged_with` applies a partial TOML update, `save`/`load_overlay` persist it; every config struct is `Serialize` so `PlatformConfig::section(name)` can be dumped through `secrets::redact`), secrets (`secrets.rs`: `config/secrets.toml` with `FA_<SECTION>_<KEY>` env overrides per D-020, `$FA_SECRETS` path, missing file = empty; `Debug` and `secrets::redact*` hide every `*key*|*secret*|*token*` value per spec 60 §6.7), deterministic RNG + samplers (tested).

## Owns
Core Rust types used by every crate.

## Governing specs
- [01-architecture](../../specs/01-architecture.md)
- [Ingestion reference (ex spec 10)](../../go/services/ingestor/README.md#ingestion-reference)
- [Reference data reference (ex spec 12)](../fa-warehouse/README.md#reference-data-reference)
