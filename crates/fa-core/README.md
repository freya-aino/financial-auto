# fa-core

Shared domain types: Instrument, Venue, Timescale, Horizon, canonical event structs, ids.

**Status:** implemented — ids, timescales/horizons, platform config (`config.rs`: `PlatformConfig::load` fills `[garage] public_endpoint` (defaults to `endpoint`; the host-reachable S3 endpoint presigned URLs are built on) and merges the git-ignored `config/runtime.toml` next to the platform file over the operator-tunable `[runtime]` section — `ui.poll_ms`, `bus_tap.max_msgs_per_sec`, `live.max_*`, `ws.*_hz`, each bounded by `RUNTIME_BOUNDS` and checked by `RuntimeConfig::validate`; `merged_with` applies a partial TOML update, `save`/`load_overlay` persist it; every config struct is `Serialize` so `PlatformConfig::section(name)` can be dumped through `secrets::redact`; `[qdrant] vector_backend` (D-101) and `[graph] backend` (D-108) are validated against their allowed values), secrets (`secrets.rs`: `config/secrets.toml` with `FA_<SECTION>_<KEY>` env overrides per D-020, `$FA_SECRETS` path, missing file = empty; `Debug` and `secrets::redact*` hide every `*key*|*secret*|*token*` value per spec 60 §6.7), logging (`logging.rs`, D-086: `logging::init(service)` installs the global subscriber; `FA_LOG_FORMAT=json` switches to JSON lines, default human-readable text; `RUST_LOG` filters, default `info`), market-data mode (`market_data.rs`, D-103: `MarketDataMode` = `real` (default) | `synthetic` | `all` from `FA_MARKET_DATA`, unknown value = error so a service fails at start; `admits_venue` / `admits_subject` drop the synthetic venues (`SYNTHETIC_VENUES` = `sim`, the `heartbeat` and `synthetic` connectors) from the paper-trading path in `real` mode and the real ones in `synthetic`; Go mirror `go/pkg/bus/marketdata.go`), deterministic RNG + samplers (tested).

## Owns
Core Rust types used by every crate.

## Governing specs
- [01-architecture](../../specs/01-architecture.md)
- [Ingestion reference (ex spec 10)](../../go/services/ingestor/README.md#ingestion-reference)
- [Reference data reference (ex spec 12)](../fa-warehouse/README.md#reference-data-reference)

## `[llm]` and `[llm] api_key` (D-117)

`LlmConfig` (`config.rs`) holds the OpenAI-compatible endpoint the LLM suite calls — `base_url`,
`model`, `embedding_model`, `embedding_dim`, `timeout_secs`, `embed_batch` — and `LlmSecrets`
(`secrets.rs`) holds its key, separately and git-ignored, so a platform config can be committed and
shown in the UI without redaction. `FA_LLM_API_KEY` overrides the file; `LlmSecrets`'s `Debug` is
redacted like the Garage credentials. An empty `base_url` means "not configured", which the suite
treats as a state to run in rather than an error.
