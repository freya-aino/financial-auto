# ADR 0006 — Local-only deployment, no authentication

**Status:** accepted · **Date:** 2026-11-06

## Context
The platform and its management frontend run on the user's workstation only. The user explicitly does not want security features at first. The previous repo committed an API key file (`APIs/lemon/.env`), so credential hygiene still matters.

## Decision
- No authentication/authorization on the control API, frontend, NATS, Helix, VictoriaMetrics, Postgres (local credentials only) or Garage (local keys).
- All services **bind to localhost** (or the compose network) by default.
- **Secrets** (broker/API keys, wallet keys) live only in git-ignored `config/secrets.toml` or environment variables; never in `sources.yaml`, specs, logs, or the bus. `.gitignore` covers `config/secrets.toml`, `.env*`.
- Every control action is still recorded in the Postgres audit log (traceability, not security).

## Consequences
- Fast iteration; zero auth plumbing.
- Exposing any port beyond localhost (e.g., second machine for scaling) requires revisiting this ADR (TLS, NATS accounts, API auth).
- Wallet keys for DEX/prediction markets (Hyperliquid, dYdX, Polymarket) are high-value: **(proposal)** keep them in a separate file with restrictive permissions.

## Alternatives considered
- **Basic auth / tokens from day one:** low cost but explicitly not wanted now.
