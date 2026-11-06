# frontend

SvelteKit 3 (Svelte 5, runes) + TypeScript operator UI, built as a static SPA with `adapter-static` and served by `control-api` (D-015).

**Status:** v1 shell implemented. The Ops view is live; Latent & Graph, Evolution Lab and Trading Desk are structured placeholders listing what each will show (spec 51) and which endpoints feed it. `pnpm check` (0 errors, 0 warnings) and `pnpm build` verified 2026-11-06.

## Views
| Route | View | State |
|---|---|---|
| `/` | Ops: services (heartbeats, polled every 2 s), live bus tap with msg/s and dropped counters, source catalog with search + category/status/capability filters | live |
| `/latent/` | Latent & Graph | placeholder |
| `/evolution/` | Evolution Lab | placeholder |
| `/trading/` | Trading Desk (kill switch shown, disabled) | placeholder |

Top bar: NATS state and dependency reachability from `/api/health`, polled every 5 s.

Deviation from spec 51's route list: Ops lives at `/` rather than `/ops`, and sources/bus are panels on it rather than sub-routes. Splitting them later is cheap.

## Develop
```sh
just fe-install
just control-api      # :7700
just fe-dev           # :5173, proxies /api and /ws to :7700
just fake-publisher   # optional: fake heartbeats + trades
just fe-build         # svelte-check + static build into frontend/build
```
Without local Node, the `just pnpm …` recipe runs pnpm in `node:22-alpine`.

Notes: SvelteKit 3 has no `svelte.config.js`, so kit options go in `sveltekit({...})` in `vite.config.ts`. TypeScript is pinned to 6.x because kit 3.1 and svelte-check don't accept TS 7 yet.

## Governing specs
- [51-frontend](../specs/51-frontend.md)
- [50-control-api](../specs/50-control-api.md)
