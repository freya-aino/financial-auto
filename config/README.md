# config

Runtime configuration: source catalog, platform config, timescale ladder. Secrets live in git-ignored `secrets.toml`.

**Status:** in use — `platform.toml` (host) and `platform.docker.toml` (compose hostnames) are read by every Rust service via `fa-core` (`FA_CONFIG`); `sources.yaml` is loaded by `go/pkg/catalog` and served by control-api `/api/sources`; `secrets.example.toml` documents the git-ignored `secrets.toml` (no code reads it yet).

## Owns
`sources.yaml`, `platform.toml`, `platform.docker.toml`, `secrets.example.toml`.

## Governing specs
- [11-source-catalog](../specs/11-source-catalog.md)
- [60-ops](../specs/60-ops.md)
