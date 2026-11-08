# frontend

SvelteKit 3 (Svelte 5, runes) + TypeScript operator UI, built as a static SPA with `adapter-static`. In compose it is the `ui` service (nginx, `docker/ui.Dockerfile`, http://127.0.0.1:8080) proxying `/api` and `/ws` to control-api (D-035); on the host, `control-api` serves `frontend/build` directly (D-015).

**Status:** all four v1 views and the global chrome (spec 51 §6.1) implemented against live control-api data. Views poll REST every 2 s and additionally refetch (debounced 250 ms) when the shared bus WebSocket (`src/lib/ws.svelte.ts`, spec 51 §4/§6.6) sees a relevant frame, so updates land within a few hundred ms while polling remains the fallback. Paper trading only: the LIVE control is permanently disabled (D-029). The kill switch is reachable from every page (top bar, confirm dialog, result taken from the API response). View preferences (timescale, filters, objective pickers, tap subjects) and the last 300 risk samples for the Trading Desk charts persist in `localStorage` behind try/catch. `pnpm check` (0 errors, 0 warnings), `pnpm test` (7 files, 39 tests) and `pnpm build` verified 2026-11-08. Without producers, each panel shows which bus subject would fill it; `cargo run -p control-api --example demo_publisher` fills every view with synthetic data.

## Views
| Route | View | State |
|---|---|---|
| `/` | Ops: services (heartbeats, polled every 2 s); Jobs & GPU (queue by priority class, running jobs with cancel, arbiter state); live bus tap on the shared socket (rows appended once per animation frame, msg/s, server-dropped counter, pattern count); source catalog with search + category/status/capability filters and per-source ingest start/stop (501 from the API until the ingestor handles `ctl.ingest.*`, shown inline) | live |
| `/latent/` | Latent & Graph (refetches on `latent.>` / `decoder.output.>` frames): timescale selector; region transition graph (SVG, circular layout, edge width = posterior mean, 90% interval in tooltip, click to pin a region); region table with lineage/status/occupancy; expert states per instrument (router weights, region per expert, ‖μ‖, σ̄); decoder-output feed with family filter, forecast quantile strips, retrieval analogs with similarity and time distance | live |
| `/evolution/` | Evolution Lab (refetches on `jobs.>`): job counts per kind × status; GPU arbiter (from orchestrator heartbeat); start-generation form (`ctl.evo.start`); Pareto scatter with objective pickers and non-dominated points highlighted (`src/lib/pareto.ts`; falls back to a score histogram, `src/lib/histogram.ts`); recent jobs with progress bars | live |
| `/trading/` | Trading Desk (refetches on `order.>`, `risk.>`, `signal.exposure.>`): PAPER badge, LIVE disabled; working kill switch (prompt for reason) + resume; equity / deployable / exposure / drawdown / P&L tiles (risk service if reporting, else paper book); equity, P&L (zero baseline, sign-coloured) and drawdown sparklines built client-side from the polled risk figures (ring buffer of 900 samples ≈ 30 min, newest 300 persisted; hover crosshair) since `/api/trading/pnl` is 501; exposure gauges drawn against equity because the API reports no caps (`/api/trading/limits` is 501); Strategies table derived from signals + orders (`src/lib/strategies.ts`: genome, horizon, last-signal age, instruments, mean and largest \|exposure\|, orders/filled/rejected attributed via `strategy_ids`) with pause/resume/flatten/promote buttons disabled and labelled with the 501 endpoint; positions; manual paper order ticket; active halts; orders & fills; strategy signals as exposure mean ± σ bars | live |

Top bar on every page: PAPER mode badge, equity and P&L (risk service if reporting, else paper book), active-halt count, NATS state and dependency dots from `/api/health` (5 s), and the kill switch / resume (`/api/trading/risk` every 2 s, plus an immediate refetch on `risk.>` frames). The bar turns red while the portfolio is halted. Below it, a stale banner (spec 51 §8) appears while control-api is unreachable or the bus socket is not open, showing the reconnect countdown and attempt number, a "Retry now" button, and panels get a dashed border while live updates are paused.

## Realtime (`src/lib/ws.svelte.ts`)
One `BusManager` instance (`bus`) owns the single `/ws/bus` connection for the whole app. Components call `bus.subscribe(patterns, listener)` (NATS wildcard matching client-side, so several subscribers share one socket), `bus.topic(pattern)` for a reactive latest-value store coalesced to one update per animation frame, or `onBus(patterns, refetch)` for the debounced "refetch on activity" pattern the views use. Reconnect backoff is 0.5 s doubling to 10 s (`ui.ws_reconnect_backoff`), reset on open; the wanted pattern set is resubscribed on every connect. Because control-api only takes the subject list in the URL (`?subjects=a,b`; in-band `subscribe`/`unsubscribe` ops of spec 50 §4.8 are not implemented yet), a change in the wanted set closes and reopens the socket with the merged list; changes within one tick are batched into one reconnect. Server `{op:"lagged",dropped}` frames accumulate in `bus.dropped`, `{op:"error"}` in `bus.error`.

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

## Testing
```sh
just pnpm test          # vitest run (node environment, src/**/*.test.ts)
just pnpm test:watch
```
Unit tests cover `api.ts` (formatters, query building, `ApiError` from `postJson` with a mocked `fetch`), `prefs.ts` (prefix, corrupt values, unavailable storage), `ws.svelte.ts` (backoff schedule, subject matching, one socket per pattern set with reconnect on change, fan-out, lag/error ops, resubscribe, animation-frame coalescing and debounce, all against a fake `WebSocket` with fake timers), `pareto.ts`, `histogram.ts`, `series.ts` (ring buffer, persistence, drawdown, SVG path scaling) and `strategies.ts`. Component rendering is not under test (no DOM environment is installed; `pnpm check` type-checks the components).

Notes: SvelteKit 3 has no `svelte.config.js`, so kit options go in `sveltekit({...})` in `vite.config.ts`. TypeScript is pinned to 6.x because kit 3.1 and svelte-check don't accept TS 7 yet.

## Governing specs
- [51-frontend](../specs/51-frontend.md)
- [50-control-api](../specs/50-control-api.md)
