# frontend

SvelteKit 3 (Svelte 5, runes) + TypeScript operator UI, built as a static SPA with `adapter-static`. In compose it is the `ui` service (nginx, `docker/ui.Dockerfile`, http://127.0.0.1:8080) proxying `/api` and `/ws` to control-api (D-035); on the host, `control-api` serves `frontend/build` directly (D-015).

**Status:** all four spec 51 v1 views, the global chrome (§6.1) and the round-2 API surfaces (D-045..D-055: streams, source health + backfill, instruments/quarantine, audit, runtime config, persisted snapshots/experts/projection/MCMC graph, graph explorer, generations + run control, populations/Pareto from the trial registry, genome drawer, validation, strategy lifecycle, limits, flatten, P&L, reconciliation) implemented against live control-api data from the compose stack. Live state comes from the four WebSockets (`/ws/bus`, `/ws/latent`, `/ws/evolution`, `/ws/trading`, spec 50 §4.8 / D-055); REST polling at `ui.poll_ms` from `GET /api/config/runtime` (2 s default) is the fallback while a socket is down and a slow safety refetch (10–15 × `poll_ms`) while it is up. Paper trading only: the LIVE control is permanently disabled (D-029). The kill switch is reachable from every page (top bar, confirm dialog, result taken from the API response). View preferences and the last 300 risk samples persist in `localStorage` behind try/catch. `pnpm check` (0 errors, 0 warnings), `pnpm test` (13 files, 76 tests) and `pnpm build` verified 2026-11-10; every route and compiled module served 200 through `pnpm dev`. Without producers, each panel names the bus subject or route that would fill it.

## Views
| Route | View | State |
|---|---|---|
| `/` | Ops: services with heartbeat gauge groups (`primitives.graph\|decoder\|fallback` share bar, `validation.*`, `pareto.*`, `hub.*`, `gpu.*`; per-strategy / per-subject counters folded, click to expand); Jobs & GPU (queue by priority class, running + queued jobs with cancel, recent ops jobs such as `ingest.backfill`, live `job.<id>` frames); bus tap; Streams (JetStream messages/bytes/seq, consumer `ack_pending`/`num_pending`, click for the consumer table); Sources with health `ok\|degraded\|down`, limiter saturation bar, last event age, errors, enable/disable, start/stop, Backfill dialog (instruments, timescale, `range` or start/end), row detail with cursors / connector runs / recent errors; Instruments search + Quarantine review with the Resolve dialog (activate with ISIN/FIGI/asset class/kind, merge into an instrument id with a search helper, or ignore; client-side validation incl. ISIN check digit); Audit log (`since` picker, filter, click to expand payload); Runtime Config editor with server bounds, `changed`/`persisted`/`audited` from the reply | live |
| `/latent/` | Latent & Graph: timescale selector + persisted Snapshots selector (status badge, promote / rollback with confirm); region transition graph switchable between the live in-memory view and the persisted `/api/mcmc/{snapshot}/graph`; Experts panel (gate mean, top share, regions, ‖μ‖, σ̄; correlation / covariance heatmap); Projection scatter (PCA 2-D per expert, coloured by region id with legend, explained variance, click a point or legend swatch to pin a region, `range`/`n` controls); Graph Explorer (neighbourhood by any business id with clickable edges, named-query catalog with a parameter form); region table; expert states per instrument (live from `/ws/latent` `state.<snapshot>.<instrument>`, region lifecycle frames trigger a refetch); decoder-output feed | live |
| `/evolution/` | Evolution Lab: Generations table (REST list merged with `/ws/evolution` `generation.<id>` frames and registry stats: status, loop, rung, candidates, front sizes, survivors per rung, deflated Sharpe / PBO, artifact key; pause / resume / stop; row expands to the Pareto front and survivors); Start Generation form for `evo.eval.decoder` (kv genomes) and `evo.eval.strategy` (one inline-TOML table per line) with `eta`, `max_generations`, min/max/total budgets, `data_window_hours`; GPU arbiter + jobs by kind; Populations + Pareto front from `validation.trial` (population / loop pickers, objective pickers with directions, rank-0 front highlighted, click → genome drawer); Recent jobs; Validation card (locked holdouts, candidate trial reports); Genome drawer (definition pretty-printed, fitness records + histograms, presigned eval reports opening in a new tab, decoder outputs via QuantileStrip / Histogram) | live |
| `/trading/` | Trading Desk: PAPER badge, LIVE disabled, kill switch + resume, FLATTEN ALL (confirm); equity / deployable / exposure / drawdown / P&L tiles; equity, P&L and drawdown sparklines sampled from `risk` frames (polling fallback); exposure gauges against the caps from `/api/trading/limits` (gross cap, drawdown, daily loss, position fraction); Strategies table from `/api/trading/strategies` (status, sleeve, weight, P&L, drawdown, shadow progress, gate checklist expandable with pass/fail/unknown + advisory, latest signal; pause / resume / flatten / promote / shadow with confirm; promote disabled with a tooltip listing the missing gate items) with a derived fallback table while the route is unavailable; Risk limits editor (tighten-only check before sending, 422 `loosen:` shown inline); P&L panel (`range`, strategy filter, by instrument); Reconciliation runs with cash / position diffs; positions; manual paper order ticket (goes through `order.intent.paper`); halts with `portfolio` / `daily_loss` / `strategy:<id>` / `source:<id>` / `broker` scopes; orders & fills with `ack` resting limit orders highlighted and arrival price / slippage bps columns (shown when the order view carries them); strategy signals | live |

| `/alerts/` | Alerts (D-066): alert-pipeline health (Watchdog age, posts; `pipeline.ok = false` shown as an alert), active alerts critical-first with labels / annotations / rule link, recent transitions, live `ops.alert` tail | live |
| `/refdata/` | Reference data (D-067): `at` / `known_at` bar; symbol resolve (none / one / ambiguous), instrument as of (listings, identifiers, contract specs), FX → EUR as of; venue calendars (table vs built-in rule, holidays, early closes, segments) + materialise; corporate actions (as-of per instrument or newest), entry, cancel, symbol change — each confirmed and audited | live |
| `/latent/mcmc/` | MCMC & regions (D-069): kernels with prequential sampler scores and the serving sampler; the seven spec 33 §4.2 queries as posterior intervals (neighbors, transition, k-step, paths, path-prob, hitting with pmf, stationary + relaxation); region lineage (ancestors / descendants / members); REFINES hierarchy; cross-asset priors; live soft memberships (`latent.membership.*`) and lifecycle events (`latent.lifecycle.*`) | live |
| `/latent/snapshots/` | Encoder snapshots (D-070): all snapshots with family (neural `fsq_hybrid` / `jepa_vae`), expert health (K_eff, residual-correlation heatmap, per-expert skill and usage, mixture skill), full promotion checklist incl. the neural items and expected-but-missing ones, not-evaluated items, shadow gauges, training metrics; shadow start / stop, promote, rollback (confirm, audited) | live |
| `/evolution/decoders/` | Neural decoders (D-068): slots per (family, timescale) with serving vs shadow, online n / loss and progress to the 200-output / 2 % rule, bus output counts per role; decoder cards with skills; reload / promote / pin / unpin / off / on via `ctl.decoders.*` (confirm, audited) | live |
| `/trading/pipeline/` | Strategy pipeline (D-071): stage counts candidate → holdout_passed → shadow → active → retired; evolved strategies with last_reason, selection PBO / DSR against the policy, holdout outcome, gate; last round (holdouts, actions, waiting, live set, policy) and run-now; rejected candidates; allocator tournament ranking; Validation card linked to the selected row | live |
| `/trading/risk/` | Risk & halts (D-071): halts persisted in `ledger.risk_state` merged with bus halts; re-arm per scope for every re-armable family (confirm + reason, audited); risk-event history from `audit.log` + live `risk.event / halt / resume` tail with kind filter | live |
| `/trading/ledger/` | Ledger (D-067): P&L breakdown with the FX split (book and per strategy), FIFO lots, statement summary + CSV download, orphan fills | live |

Top bar on every page: PAPER mode badge, a second row with the section's sub-pages, a global alert banner (active alerts critical-first, dead alert pipeline), equity and P&L (from the `/ws/trading` `risk` topic while open, else `/api/trading/risk`), active-halt count, NATS state and dependency dots from `/api/health` (5 s), a `ws` dot for `/ws/trading`, and the kill switch / resume. The bar turns red while the portfolio is halted. Below it, a stale banner (spec 51 §8) lists every wanted socket that is not open with its reconnect countdown, attempt number and a "Retry now" button; panels get a dashed border while live updates are paused.

## Panel → route / subject
| Panel | REST | Live |
|---|---|---|
| Services | `GET /api/services` | `svc.heartbeat.>` (bus) |
| Jobs & GPU | `GET /api/evolution/jobs`, `GET /api/evolution/gpu`, `POST /api/jobs/{id}/cancel` | `/ws/evolution` `job.<id>`, `jobs.>` (bus) |
| Bus tap | — | `/ws/bus?subjects=` |
| Streams | `GET /api/bus/streams` | poll 5 × `poll_ms` |
| Sources | `GET /api/sources`, `GET /api/sources/{id}`, `POST /api/sources/{id}/enable\|disable\|backfill`, `POST /api/ingest/{id}/start\|stop` | `ingest.health.>`, `risk.halt.>`, `risk.resume.>` (bus) |
| Instruments / Quarantine | `GET /api/instruments?q=`, `GET /api/instruments/quarantine`, `POST /api/instruments/quarantine/{id}/resolve` | poll 10 × `poll_ms` |
| Audit log | `GET /api/audit?since=` | poll 5 × `poll_ms` |
| Runtime config | `GET/PUT /api/config/runtime` | — |
| Timescale / snapshot selectors | `GET /api/latent/timescales`, `GET /api/latent/snapshots`, `POST /api/latent/snapshots/{id}/promote\|rollback` | — |
| Region graph | `GET /api/latent/timescales/{ts}/transitions` or `GET /api/mcmc/{snapshot}/graph?timescale=` | `/ws/latent` `region.<snapshot>.<id>` |
| Regions / expert states | `GET /api/latent/timescales/{ts}/regions\|states` | `/ws/latent?snapshot=` `state.<snapshot>.<instrument>` |
| Experts | `GET /api/latent/{snapshot}/experts` | poll 15 × `poll_ms` |
| Projection | `GET /api/latent/{snapshot}/projection?expert=&range=&n=&dims=2` | on control change |
| Graph explorer | `GET /api/graph/neighborhood?node=&depth=`, `GET/POST /api/graph/query` | — |
| Decoder outputs | `GET /api/decoders/outputs` | `decoder.output.>` (bus) |
| Generations | `GET /api/evolution/generations`, `POST /api/evolution/runs/{id}/pause\|resume\|stop` | `/ws/evolution` `generation.<id>` |
| Start generation | `POST /api/evolution/generations` | — |
| Populations / Pareto | `GET /api/evolution/populations`, `GET /api/evolution/populations/{id}/pareto?gen=` | refetch on generation frames |
| Genome drawer | `GET /api/evolution/genomes/{id}`, `GET /api/evolution/genomes/{id}/outputs` | — |
| Validation | `GET /api/validation/holdout`, `GET /api/validation/reports/{candidate}` | — |
| Tiles / charts / halts | `GET /api/trading/risk` | `/ws/trading` `risk`, `halts` |
| Positions | `GET /api/trading/positions` | `/ws/trading` `positions` |
| Orders & fills | `GET /api/trading/orders` | `/ws/trading` `order.<intent>`, `fill.<intent>` |
| Strategies | `GET /api/trading/strategies`, `POST /api/trading/strategies/{id}/{action}` | `signal.exposure.>` (bus), poll 3 × `poll_ms` |
| Limits / gauges | `GET/PUT /api/trading/limits` | — |
| Flatten all / kill switch | `POST /api/trading/flatten`, `POST /api/trading/halt\|resume` | — |
| P&L | `GET /api/trading/pnl?range=&strategy=` | refetch on fill frames |
| Reconciliation | `GET /api/trading/reconciliation` | refetch on fill frames |
| Manual order | `POST /api/trading/paper-order` | — |

## Realtime (`src/lib/ws.svelte.ts`)
`BusManager` (`bus`) owns the single `/ws/bus` connection: components call `bus.subscribe(patterns, listener)` (NATS wildcard matching client-side), `bus.topic(pattern)` for a reactive latest value coalesced per animation frame, or `onBus(patterns, refetch)` for the debounced "refetch on activity" pattern. Because the tap only takes its NATS subject list in the URL (`?subjects=`; the in-band ops filter what the tap already receives), a change of the wanted set reopens the socket with the merged list (batched per tick).

`StreamSocket` (`tradingWs`, `evolutionWs`, `latentWs`) serves the state streams of D-055: it connects while anything listens, sends `{op: "subscribe", topics}` for the whole wanted filter set on every open (so reconnects resubscribe) and in-band `subscribe`/`unsubscribe` ops as listeners come and go, handles `{op: "hello", flush_ms}`, `{op: "subscribed"}`, `{op: "lagged", dropped}` (→ `.lagged`) and `{op: "error"}`, and reconnects with the bus backoff (0.5 s doubling to 10 s). `socket.topic(filter)` is the newest frame of any matching topic, `socket.topics(filter)` a reactive `Map<topic, frame>` (e.g. `order.>` → one entry per intent), both replaced at most once per animation frame; `setQuery({snapshot})` reconnects `/ws/latent` when the snapshot changes. `FrameRouter` (parsing, op dispatch, routing with the server's `topic_matches` semantics) and `TopicCache` are the pure parts under unit test. Views keep `poll(fn, () => socket.live ? 10 × poll_ms : poll_ms)` as the fallback; `src/lib/runtime.svelte.ts` holds `ui.poll_ms`.

Deviation from spec 51's route list: Ops lives at `/` rather than `/ops`, and sources / bus / audit / config are panels on it rather than sub-routes (the D-066..D-071 pages above are sub-routes); the genome inspector is a drawer on `/evolution/` rather than `/evolution/[genome]`. Splitting them later is cheap.

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
Unit tests cover `api.ts` (formatters, query building, `ApiError`), `prefs.ts`, `ws.svelte.ts` (bus backoff / matching / reconnect / fan-out / coalescing, and for the state streams `topicMatches`, `FrameRouter` ops + routing, `TopicCache`, `StreamSocket` connect / in-band ops / reconnect-with-resubscribe / `setQuery` / `topics()` against a fake `WebSocket` with fake timers), `pareto.ts` (non-dominated set, `directionFor` with the D-049/D-052 keys, diagnostics filtering, candidate labels, ranked vs unranked fronts), `generations.ts` (survivors-per-rung keys, REST/WS/registry merge, run actions), `limits.ts` (tighten / loosen classification, PUT body, `loosen:` parsing), `quarantine.ts` (ISIN check digit, activate / merge / ignore validation), `projection.ts` (stable region colours, diverging heat colour, extents, explained-variance label), `services.ts` (gauge grouping, primitive shares), `histogram.ts`, `series.ts`, `strategies.ts`, plus `render.test.ts`, which SSR-renders all four pages and the data-bearing panels with live-shaped props to catch template errors (no DOM environment is installed; `pnpm check` type-checks the components).

Notes: SvelteKit 3 has no `svelte.config.js`, so kit options go in `sveltekit({...})` in `vite.config.ts`. TypeScript is pinned to 6.x because kit 3.1 and svelte-check don't accept TS 7 yet.

## Governing specs
- [51-frontend](../specs/51-frontend.md)
- [50-control-api](../specs/50-control-api.md)
