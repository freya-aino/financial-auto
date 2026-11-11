# 51 — Frontend

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium-high |
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
- REST client generated from `/api/openapi.json` **(proposal)** into `frontend/src/lib/api/`. *Superseded (round 3, Q2): hand-written fetch layer `src/lib/api.ts` + `src/lib/console.ts`, contract-checked by `openapi.test.ts` against the committed generated document `src/lib/openapi.json`.*
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
As built (`src/routes/`): `/` (Ops: services, jobs & GPU, bus tap, streams, sources, instruments, audit, runtime config as panels), `/alerts`, `/refdata`, `/latent`, `/latent/mcmc`, `/latent/snapshots`, `/evolution` (genome inspector as a drawer), `/evolution/decoders`, `/trading` (strategies as a table, no per-strategy route), `/trading/pipeline`, `/trading/risk`, `/trading/validation` (round 8), `/trading/ledger`, `/history` (round 8); a second navigation row lists each section's sub-pages.

## 5. Data model
No persistent state; UI preferences (selected snapshot, panel layout, filters) in `localStorage` (wrapped in try/catch).

## 6. Behaviour & algorithms

**Implemented today (D-057):** the four v1 views render every route of spec 50 §4 and consume the four WebSocket feeds (`/ws/bus`, `/ws/latent`, `/ws/evolution`, `/ws/trading`) with REST polling at `ui.poll_ms` only as the fallback. Deviations kept: Ops lives at `/`, panels are not sub-routes, the genome inspector is a drawer. The LIVE control stays disabled (D-029).

**Implemented for D-066..D-071:** each v1 view has sub-pages in a second navigation row: Ops → Overview, **Alerts** (`/alerts/`: alert-pipeline health, active alerts by severity with labels/annotations, recent transitions, live `ops.alert` tail), **Reference data** (`/refdata/`: bitemporal `at` / `known_at` bar driving symbol resolve (none/one/ambiguous), instrument as-of, FX as-of, venue calendars with table-vs-rule basis and materialise, corporate actions list / entry / cancel and symbol changes); Latent & Graph → Explorer, **MCMC** (`/latent/mcmc/`: kernels with prequential sampler scores and the serving sampler, all seven spec 33 §4.2 queries rendered as posterior intervals, region lineage, REFINES hierarchy, cross-asset priors, live soft memberships and lifecycle events from the bus tap), **Encoder snapshots** (`/latent/snapshots/`: family incl. neural `fsq_hybrid`/`jepa_vae`, expert health — K_eff, residual-correlation heatmap, per-expert skill and usage — the full promotion checklist with the D-070 items and any expected-but-missing neural item, not-evaluated items, shadow gauges, shadow start/stop and promote/rollback); Evolution Lab → Lab, **Neural decoders** (`/evolution/decoders/`: serving vs shadow per slot with online n/loss and progress to the 200-output / 2 % promotion rule, decoder cards with skills, `ctl.decoders.*` reload/promote/pin/unpin/off/on); Trading Desk → Desk, **Strategy pipeline** (`/trading/pipeline/`: stage counts candidate → holdout_passed → shadow → active → retired, per evolved strategy last_reason, selection PBO/DSR against the policy and the holdout outcome, last round's holdouts/actions/rejections/live set/policy, run-round, allocator tournament ranking, Validation card), **Risk & halts** (`/trading/risk/`: persisted + bus halts with re-arm for every re-armable scope, RiskEvent history from `audit.log` merged with the live `risk.event/halt/resume` tail), **Ledger** (`/trading/ledger/`: P&L breakdown with the FX split per book and strategy, FIFO lots, statement summary + CSV, orphan fills). A global alert banner under the top bar shows active alerts (critical first) and a dead alert pipeline. Every mutation keeps the confirm-then-API-result pattern of the kill switch (never optimistic) and is audited server-side. Pure logic lives in `alerts.ts`, `riskdesk.ts`, `models.ts` (tested in `console.test.ts`); all pages are SSR-smoke-rendered in `render.test.ts`.

**Round 3:** *Encoder snapshots* adds **live expert health** from `latent.expert.health` (via `GET /api/latent/{snapshot}/experts` → `health`: K_eff, max |residual corr|, min skill, per-expert usage/skill by expert id, live residual-correlation matrix; `models.ts liveExpertHealth`) next to the validation-split health, and recognises all four neural families. Residual-correlation heatmaps use a **sequential one-hue palette on |corr|** (`Heatmap scale="sequential"`, `projection.ts seqColor`, tokens `--seq-0..4` per theme, breaks 0.05 / 0.1 / 0.2 / 0.4 so the 0.1 target is a step boundary, diagonal blank, step legend; ramps validated with the dataviz ordinal check in both themes: monotone lightness, single hue, light end ≥ 2:1 against the surface) instead of the diverging red/green used where the sign matters. *Neural decoders* shows each shadow's **age against the promotion minimum** (`shadow for 12 min / 30 min`, from `shadow_age_s` / `promote_min_s` in `ctl.decoders.status`; `models.ts shadowAge`). *Ledger* gets operator forms: **orphan resolution** per open orphan (book — enabled only once the instrument is resolved — or dismiss with a required reason) and **payout booking** (day ≤ today, positive EUR amount); both validate in `ledgerForms.ts`, ask `confirm()` with a text naming the fill / amount, book and consequence, then show the API's answer (tested in `ledgerForms.test.ts` against no live book). *Streams* shows usage against each stream's cap and flags a planned cap that was not applied. `openapi.test.ts` checks every `/api/…` call of the SPA against control-api's generated OpenAPI document (spec 50 Q3). `pnpm test`: 16 files, 108 tests, green 2026-11-11.

**Round 8 (2026-11-11, D-085 proposal):** every route of spec 50 §4 has a UI again. *Services* gets start / stop / restart per compose service (`POST /api/services/{name}/{action}`, D-060): buttons gated by `services.ts actionGate` (infrastructure and control-api refused, no Docker socket → off, only state-changing actions enabled), a `confirm()` naming the consequence (risk-engine stop/restart halts trading and is sent with `{confirm: true}`), the API's answer shown (`state`, `noop`, `audited`, or the 409/428/501 message); compose services without a heartbeat (stopped ones) get a row from `supervisor.services` with `GET /api/services/{name}/container` (state, exit code, restarts, health; at most every 10 s). *Bus tap* shows a 60 s msg/s sparkline per subject, folded to 1–4 tokens or the full subject (`busrates.ts`). *Genome drawer* draws the **lineage tree** from `GET /api/graph/lineage/{id}` (columns by ancestry depth, nodes ordered by the barycentre of their children; edges below depth 1 from the parents lists of `GET /api/evolution/genomes/{id}`, at most 40 lookups; click a node to open that genome; `lineage.ts`). *Manual order ticket*: **Check risk** sends the ticket to `POST /api/trading/pretrade` (`{ok, reasons}`, a dry run of `fa_risk::pretrade::check_order`); Submit is enabled only once the *current* ticket (any edit invalidates the result) passed, or the route answered 404/405/501/502/503 (shown as "unavailable": the risk-engine still runs the same check on the intent); `ticket.ts`. New pages: **Ops → History** (`/history/`: job summary kind × status over 1 h–30 d, stored job runs with result / outputs, config snapshots per service/section with the previous digest and the redacted section on demand), **Alerts → History** panel (`GET /api/alerts/history`, window + alertname filter), **Trading → Holdouts & shadow** (`/trading/validation/`: locked holdouts, lock form — completed segment, known timescale, confirm naming the immutability; evaluate form — holdout, candidate, optional snapshot / budget / seed, confirm naming that the holdout is spent, then the candidate's Validation card; shadow-vs-backtest consistency per strategy with the ratio and the raw `ledger.strategy_state` row), **Reference data → Overrides** (identifier with scheme / value / valid_from; contract-spec version with tick, lot, min notional, multiplier, max leverage, settlement ccy, shortable, fee schedule, expiry, valid_from; empty fields omitted), **Ledger → FX restatement** (dry run of `POST /api/trading/restatements/fx` first: fills per old source, missing rates, cash delta, before / after cash, equity, P&L and strategy gap, the per-fill lines; *apply* only for the plan on screen — editing the form disables it — with a required reason and a `confirm()` naming the rewrite and the risk-engine restart; the UI never sends `apply` without that). Form logic in `operator.ts` (tested in `operator.test.ts`).

### 6.1 Global chrome
Persistent top bar on every page: trading mode (paper/live), kill-switch state + button (confirm dialog), portfolio equity & day P&L, dependency health dots, active halts count.

### 6.2 Pipeline / ops control
- Services table: heartbeat, version, status, start/stop. *(As built: heartbeat, version, gauge groups, container state and start / stop / restart with confirm, round 8.)*
- Sources: full catalog with filters (category, status, capability), rate-limit usage bars, last event age, error log, enable/disable, backfill form.
- Bus monitor: subject filter, live event list (decoded), rate sparkline per subject, JetStream consumer lag. *(As built: per-subject 60 s sparklines with token folding, round 8; consumer lag in the Streams panel.)*
- Jobs & GPU: queue by priority class, running job, preemptions, cancel.
- Config viewer/editor for runtime-tunable keys.

### 6.3 Latent / graph explorer
- Snapshot picker (timescale × snapshot), expert list with routing share and **residual covariance heatmap** between experts.
- Expert space projection: 2D/3D scatter of latent points over a time range, coloured by region / instrument / time, with the live position of selected instruments animated.
- Region detail: occupancy over time, typical decoder reconstructions (chart renderings, retrieved analogs), outgoing transitions.
- MCMC graph: regions as nodes, transition posteriors as edges (width = mean, opacity = certainty), cross-timescale hierarchy toggle, filter by probability threshold.
- Rendering **(proposal)**: WebGL for large point clouds and graphs (e.g. deck.gl or regl for scatter; sigma.js/graphology for graphs) with server-side downsampling (50 §6); SVG only for small views. *As built (round 8, Q1): the projection is a dependency-free **Canvas 2D point cloud** (`PointCloud.svelte` + `pointcloud.ts`): points are written into an ImageData buffer grouped by colour (no per-point canvas call), axes / live markers / hover are vector overlays, redraws coalesced per animation frame; 2-D drag-pan, 3-D orbit (shift- or right-drag pans), wheel / `+` `−` zoom about the cursor, arrows, `0` / double-click reset, nearest-point hover and click-to-pin. 50 000 synthetic points measured at 4–25 ms per frame in headless software-rendered Chromium (dev build, no GPU). Colour by region, instrument (12 hues, legend of the top 24) or time (the `--seq-*` ramp, oldest → newest). The point picker offers 500 … 50 000; control-api clamps `n` to its 20 000 cap and the panel says so. Small views (Pareto, region graph, heatmaps, lineage) stay SVG.*
- *As built (round 8):* **live positions** — tracked instruments (up to 8, default the first three live ones) are drawn at the centroid of their current region for the selected expert (`state.<snapshot>.<instrument>` on `/ws/latent`; an explicit `centroid` row when fetched, else the mean of the region's sample points) and tween there when the region changes, with the previous regions as a trail: the state frames carry the region, not the latent vector, so the exact position is not shown. **Occupancy over time** per region: share of the fetched sample per time bucket (30 buckets), or the `fa_region_weight` series through `GET /api/metrics/query_range` (`sum by (region)` ÷ total for the expert space); six largest regions, or the pinned one. **Decoder reconstructions per region** are not shown: no route links decoder outputs or analogs to a region.

### 6.4 Evolution lab
- Populations overview (decoders by family, strategies GP vs neural, allocators), generation, budget used.
- Pareto front plot per population (pick 2–3 objectives, others as colour/size), selectable points.
- Successive-halving progress per run (survivors per round).
- Lineage tree for a genome. *(As built: in the genome drawer, round 8.)*
- Genome inspector: genome definition (GP program pretty-printed, or neural architecture), fitness breakdown, eval report (equity curve, DSR, PBO, cost breakdown), decoder outputs (images, text, forecast fan charts, retrieved analog windows with time distance).
- Start/pause/stop runs with GPU-hour budget.

### 6.5 Trading desk
- Positions table (live and paper books toggle), per-strategy attribution.
- P&L chart, drawdown chart, exposure by asset class / venue vs caps (gauges showing limit headroom).
- Strategies table: status, sleeve, shadow progress (days, horizons), gate checklist; actions pause/resume/flatten/promote.
- Orders & fills blotter with slippage.
- Risk events feed, reconciliation status, manual order ticket (shows risk-check result before submit). *(As built: Check risk → `POST /api/trading/pretrade`, Submit gated on a result for the current ticket, round 8; the route is not served by control-api yet.)*

### 6.6 Realtime
Topics: `bus`, `latent`, `evolution`, `trading`, `ops`. Stores update at most at animation-frame rate; high-rate topics coalesced to latest value. *As built: four sockets (`/ws/bus`, `/ws/latent`, `/ws/evolution`, `/ws/trading`); there is no `ops` socket — the alert banner polls `/api/alerts` every 15 s and refetches on `ops.alert` frames from the bus tap.*

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `ui.max_bus_rows` | 2 000 | ring buffer; as built: 200 rows (`BusTap.svelte`, constant); per-subject rate window 60 s, 200 keys, top 12 shown |
| `ui.projection_point_budget` | 50 000 | picker up to 50 000 (round 8), 2 000 default; control-api still clamps to 20 000 (`latent_routes::MAX_POINTS`) |
| `ui.ws_reconnect_backoff` | 0.5 s → 10 s | as built (`ws.svelte.ts`) |
| `ui.poll_ms` | 2 000 ms | fallback polling, runtime-tunable via `PUT /api/config/runtime` |

## 8. Failure modes
| Failure | Response |
|---|---|
| WS disconnected | banner, data marked stale, auto-reconnect |
| API 503 | panel shows dependency down, rest of UI usable |
| Kill switch click | confirm dialog; result shown from API response, not optimistic |

## 9. Acceptance criteria
**Remaining (checked against the working tree on 2026-11-11, round 8):** the frontend side of every item is built; what is left needs control-api / model-runtime:
1. **Pre-trade check route** (criterion: ticket shows the risk-check result): `POST /api/trading/pretrade` (body `{instrument_id, side, quantity, limit_price?, mode}` → `{ok, reasons: string[]}`, nothing published; 400/422 with `reasons` also accepted) is not served — the ticket shows "unavailable" and lets the risk-engine decide. Listed in `openapi.test.ts PENDING_ROUTES`.
2. **50 000-point projection end to end** (criterion 3): the renderer handles 50 000 points (measured headless, not yet on the workstation's browser); `GET /api/latent/{snapshot}/projection` clamps `n` to 20 000 (`MAX_POINTS`).
3. **Latent explorer data** (§6.3): decoder reconstructions / retrieved analogs per region need a route that ties decoder outputs to a region (e.g. `GET /api/latent/regions/{id}/reconstructions`); exact animated live positions need the latent vector (or its PCA coordinates) in the `state.*` frames. Server-side occupancy uses `GET /api/metrics/query_range`, which is in the control-api working tree but not yet in `openapi.json` (`PENDING_ROUTES`).

Met: four views reachable and populated, kill switch on every page with confirm, `pnpm check` (0 errors, 0 warnings) / `pnpm test` (21 files, 146 tests) / `pnpm build` green 2026-11-11, only the control API is called (`openapi.test.ts`), every spec 50 §4 route has a UI (round 8), Q1 decided.

- All four views reachable and populated from the API (mock data acceptable before backend exists).
- Kill switch reachable from every page within one click + confirm.
- 50k-point projection renders interactively (pan/zoom) on the workstation.
- `pnpm check` and `pnpm build` pass.
- UI never calls anything except the control API.

## 10. Open questions
1. Visualisation libraries (deck.gl / regl / sigma.js / ECharts) — choose during scaffold spike. *Decided (round 8, D-085 proposal): no library. The point cloud is a hand-written Canvas 2D renderer writing pixels into an ImageData buffer (50 000 points in 4–25 ms per frame without GPU acceleration, 2-D and 3-D, pan / zoom / orbit); everything smaller stays SVG. Revisit (regl / WebGL) only if the point budget grows past ~200 000 or per-point size / opacity encodings are needed.*
2. Generated TS client vs hand-written fetch layer (shared with 50 Q3). *Decided (round 3): hand-written, contract-checked against the generated `/api/openapi.json` copy in `frontend/src/lib/openapi.json` (spec 50 §9).*
3. Dark-only operator theme or light/dark. *As built: both — dark tokens by default in `src/app.css`, light overrides under `prefers-color-scheme: light` (incl. the `--seq-*` ramp); no manual toggle.*

## 11. Deferred
- Auth and multi-user.
- Notebook-like ad-hoc query view.
- Mobile layout.
