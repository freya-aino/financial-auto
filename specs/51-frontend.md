# 51 — Frontend

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium |
| **Owners (code)** | `frontend/` (SvelteKit) |
| **Depends on** | 50 Control API |

## 1. Purpose
A locally deployed operator console for managing the backend's complexity by hand: watching data flow, inspecting the latent/graph landscape, steering evolution, and supervising trading — including emergency actions.

## 2. Scope
**In scope:** the four v1 views, global status bar, realtime updates, emergency controls.
**Out of scope:** auth, mobile layout polish, public hosting.

## 3. Decisions
- **SvelteKit, Svelte 5 (runes), TypeScript, pnpm**.
- Talks **only** to the control API (50): REST + WebSocket; dev server proxies `/api` and `/ws`.
- Local-only, no auth.
- v1 views: **Pipeline/ops control**, **Latent/graph explorer**, **Evolution lab**, **Trading desk**.

## 4. Interfaces
- REST client generated from `/api/openapi.json` **(proposal)** into `frontend/src/lib/api/`.
- One shared WebSocket connection manager (`src/lib/ws.svelte.ts`) multiplexing topics, exposing reactive stores per topic, auto-reconnect with resubscribe.
- Routes:
```
/                    → overview (status bar + key tiles)
/ops                 → services, jobs, GPU queue, config
/ops/sources         → catalog table, source detail, backfill
/ops/bus             → bus tap
/latent              → snapshot picker, expert spaces, projections
/latent/graph        → MCMC region graph
/evolution           → populations, runs
/evolution/[genome]  → genome inspector
/trading             → desk
/trading/strategies/[id]
/audit
```

## 5. Data model
No persistent state; UI preferences (selected snapshot, panel layout, filters) in `localStorage` (wrapped in try/catch).

## 6. Behaviour & algorithms

**Implemented today (D-057):** the four v1 views render every route of spec 50 §4 and consume the four WebSocket feeds (`/ws/bus`, `/ws/latent`, `/ws/evolution`, `/ws/trading`) with REST polling at `ui.poll_ms` only as the fallback. Deviations kept: Ops lives at `/`, panels are not sub-routes, the genome inspector is a drawer. The LIVE control stays disabled (D-029).

**Implemented for D-066..D-071:** each v1 view has sub-pages in a second navigation row: Ops → Overview, **Alerts** (`/alerts/`: alert-pipeline health, active alerts by severity with labels/annotations, recent transitions, live `ops.alert` tail), **Reference data** (`/refdata/`: bitemporal `at` / `known_at` bar driving symbol resolve (none/one/ambiguous), instrument as-of, FX as-of, venue calendars with table-vs-rule basis and materialise, corporate actions list / entry / cancel and symbol changes); Latent & Graph → Explorer, **MCMC** (`/latent/mcmc/`: kernels with prequential sampler scores and the serving sampler, all seven spec 33 §4.2 queries rendered as posterior intervals, region lineage, REFINES hierarchy, cross-asset priors, live soft memberships and lifecycle events from the bus tap), **Encoder snapshots** (`/latent/snapshots/`: family incl. neural `fsq_hybrid`/`jepa_vae`, expert health — K_eff, residual-correlation heatmap, per-expert skill and usage — the full promotion checklist with the D-070 items and any expected-but-missing neural item, not-evaluated items, shadow gauges, shadow start/stop and promote/rollback); Evolution Lab → Lab, **Neural decoders** (`/evolution/decoders/`: serving vs shadow per slot with online n/loss and progress to the 200-output / 2 % promotion rule, decoder cards with skills, `ctl.decoders.*` reload/promote/pin/unpin/off/on); Trading Desk → Desk, **Strategy pipeline** (`/trading/pipeline/`: stage counts candidate → holdout_passed → shadow → active → retired, per evolved strategy last_reason, selection PBO/DSR against the policy and the holdout outcome, last round's holdouts/actions/rejections/live set/policy, run-round, allocator tournament ranking, Validation card), **Risk & halts** (`/trading/risk/`: persisted + bus halts with re-arm for every re-armable scope, RiskEvent history from `audit.log` merged with the live `risk.event/halt/resume` tail), **Ledger** (`/trading/ledger/`: P&L breakdown with the FX split per book and strategy, FIFO lots, statement summary + CSV, orphan fills). A global alert banner under the top bar shows active alerts (critical first) and a dead alert pipeline. Every mutation keeps the confirm-then-API-result pattern of the kill switch (never optimistic) and is audited server-side. Pure logic lives in `alerts.ts`, `riskdesk.ts`, `models.ts` (tested in `console.test.ts`); all pages are SSR-smoke-rendered in `render.test.ts`.

### 6.1 Global chrome
Persistent top bar on every page: trading mode (paper/live), kill-switch state + button (confirm dialog), portfolio equity & day P&L, dependency health dots, active halts count.

### 6.2 Pipeline / ops control
- Services table: heartbeat, version, status, start/stop.
- Sources: full catalog with filters (category, status, capability), rate-limit usage bars, last event age, error log, enable/disable, backfill form.
- Bus monitor: subject filter, live event list (decoded), rate sparkline per subject, JetStream consumer lag.
- Jobs & GPU: queue by priority class, running job, preemptions, cancel.
- Config viewer/editor for runtime-tunable keys.

### 6.3 Latent / graph explorer
- Snapshot picker (timescale × snapshot), expert list with routing share and **residual covariance heatmap** between experts.
- Expert space projection: 2D/3D scatter of latent points over a time range, coloured by region / instrument / time, with the live position of selected instruments animated.
- Region detail: occupancy over time, typical decoder reconstructions (chart renderings, retrieved analogs), outgoing transitions.
- MCMC graph: regions as nodes, transition posteriors as edges (width = mean, opacity = certainty), cross-timescale hierarchy toggle, filter by probability threshold.
- Rendering **(proposal)**: WebGL for large point clouds and graphs (e.g. deck.gl or regl for scatter; sigma.js/graphology for graphs) with server-side downsampling (50 §6); SVG only for small views.

### 6.4 Evolution lab
- Populations overview (decoders by family, strategies GP vs neural, allocators), generation, budget used.
- Pareto front plot per population (pick 2–3 objectives, others as colour/size), selectable points.
- Successive-halving progress per run (survivors per round).
- Lineage tree for a genome.
- Genome inspector: genome definition (GP program pretty-printed, or neural architecture), fitness breakdown, eval report (equity curve, DSR, PBO, cost breakdown), decoder outputs (images, text, forecast fan charts, retrieved analog windows with time distance).
- Start/pause/stop runs with GPU-hour budget.

### 6.5 Trading desk
- Positions table (live and paper books toggle), per-strategy attribution.
- P&L chart, drawdown chart, exposure by asset class / venue vs caps (gauges showing limit headroom).
- Strategies table: status, sleeve, shadow progress (days, horizons), gate checklist; actions pause/resume/flatten/promote.
- Orders & fills blotter with slippage.
- Risk events feed, reconciliation status, manual order ticket (shows risk-check result before submit).

### 6.6 Realtime
Topics: `bus`, `latent`, `evolution`, `trading`, `ops`. Stores update at most at animation-frame rate; high-rate topics coalesced to latest value.

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `ui.max_bus_rows` | 2 000 | ring buffer |
| `ui.projection_point_budget` | 50 000 | matches API |
| `ui.ws_reconnect_backoff` | 0.5 s → 10 s | |

## 8. Failure modes
| Failure | Response |
|---|---|
| WS disconnected | banner, data marked stale, auto-reconnect |
| API 503 | panel shows dependency down, rest of UI usable |
| Kill switch click | confirm dialog; result shown from API response, not optimistic |

## 9. Acceptance criteria
- All four views reachable and populated from the API (mock data acceptable before backend exists).
- Kill switch reachable from every page within one click + confirm.
- 50k-point projection renders interactively (pan/zoom) on the workstation.
- `pnpm check` and `pnpm build` pass.
- UI never calls anything except the control API.

## 10. Open questions
1. Visualisation libraries (deck.gl / regl / sigma.js / ECharts) — choose during scaffold spike.
2. Generated TS client vs hand-written fetch layer (shared with 50 Q3).
3. Dark-only operator theme or light/dark.

## 11. Deferred
- Auth and multi-user.
- Notebook-like ad-hoc query view.
- Mobile layout.
