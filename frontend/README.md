# frontend

SvelteKit 3 (Svelte 5, runes) + TypeScript operator UI, built as a static SPA with `adapter-static` and served by `control-api` (D-015).

**Status:** all four v1 views implemented against live control-api data (polled every 2 s). Paper trading only: the LIVE control is permanently disabled (D-029). `pnpm check` (0 errors, 0 warnings) and `pnpm build` verified 2026-11-07. Without producers, each panel shows which bus subject would fill it; `cargo run -p control-api --example demo_publisher` fills every view with synthetic data.

## Views
| Route | View | State |
|---|---|---|
| `/` | Ops: services (heartbeats, polled every 2 s), live bus tap with msg/s and dropped counters, source catalog with search + category/status/capability filters | live |
| `/latent/` | Latent & Graph: timescale selector; region transition graph (SVG, circular layout, edge width = posterior mean, 90% interval in tooltip, click to pin a region); region table with lineage/status/occupancy; expert states per instrument (router weights, region per expert, ‖μ‖, σ̄); decoder-output feed with family filter, forecast quantile strips, retrieval analogs with similarity and time distance | live |
| `/evolution/` | Evolution Lab: job counts per kind × status; GPU arbiter (from orchestrator heartbeat); start-generation form (`ctl.evo.start`); Pareto scatter with objective pickers and non-dominated points highlighted (falls back to a score histogram); recent jobs with progress bars | live |
| `/trading/` | Trading Desk: PAPER badge, LIVE disabled; working kill switch (prompt for reason) + resume; equity / deployable / exposure / drawdown / P&L tiles (risk service if reporting, else paper book); positions; manual paper order ticket; active halts; orders & fills; strategy signals as exposure mean ± σ bars | live |

Top bar: NATS state and dependency reachability from `/api/health`, polled every 5 s.

Deviation from spec 51's route list: Ops lives at `/` rather than `/ops`, and sources/bus are panels on it rather than sub-routes. Splitting them later is cheap.

## Develop
```sh
just fe-install
just control-api      # :7700
just fe-dev           # :5173, proxies /api and /ws to :7700
just fake-publisher   # optional: fake heartbeats + trades
cargo run -p control-api --example demo_publisher   # optional: synthetic data for every view
just fe-build         # svelte-check + static build into frontend/build
```
Without local Node, the `just pnpm …` recipe runs pnpm in `node:22-alpine`.

Notes: SvelteKit 3 has no `svelte.config.js`, so kit options go in `sveltekit({...})` in `vite.config.ts`. TypeScript is pinned to 6.x because kit 3.1 and svelte-check don't accept TS 7 yet.

## Governing specs
- [51-frontend](../specs/51-frontend.md)
- [50-control-api](../specs/50-control-api.md)
