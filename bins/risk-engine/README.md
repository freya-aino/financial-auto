# risk-engine

Live service: meta-allocation across strategy signals, non-evolvable hard risk limits, the halt state machine and the paper book (specs 40, 42; D-028, D-029, D-030).

**Status:** running (paper only) — `signal.exposure.>` → strategy lifecycle filter (only `active` strategies) → allocator (`fa-allocator`) → EUR targets → caps + halt gate (`fa-risk`) → diff against the paper book (`fa-ledger`) → `order.intent.paper` + `order.approved.paper`; books fills from `order.event.>` and attributes them to strategy sub-books, marks from `md.trade.>`; publishes `risk.state` (equity, deployable, gross exposure, drawdown, daily P&L, active halts) and raises `risk.halt.portfolio` / `risk.halt.daily_loss` / `risk.halt.broker` itself; honours `risk.halt.>` (incl. the UI kill switch and the ingestor's `risk.halt.source:<id>`), `risk.resume.<scope>` and `ingest.health.<source>`. Operator control is request/reply on `ctl.risk.strategy|limits|flatten` (D-051, below). With `[postgres]` reachable the paper book is persisted (D-047): the 100 000 EUR opening deposit is a `cash_movement` booked once, every own and foreign intent and every fill is written to `ledger.*` (idempotent), a `position_snapshot` and every `ledger.strategy_state` row are written every `snapshot_secs` (60; changed strategy rows within 1 s, snapshots also on each halt), halts/resumes/limit loads/lifecycle actions/limit changes go to `audit.log`, and every `reconcile_secs` (300) the in-memory book is compared with the Postgres-derived one into `reconciliation_run` (`ok`/`diff`); **a diff halts scope `broker`** (flatten, no orders) until `risk.resume.broker` (spec 42 §6). Restarting reloads the book and the strategy states from Postgres (drawdown peak and day-start equity restart from the loaded equity; intents pending at shutdown and the strategies' virtual sub-books are not restored). Without Postgres it logs once and stays in memory.

## Pipeline per tick (spec 40 §6.1)
1. Strategy books: every active strategy's attributed sub-book (fills split evenly across the intent's contributors, D-047) and every shadow strategy's virtual book (its own signal on `shadow_sleeve_eur`) are marked to market; a drawdown ≥ `limits.strategy_max_drawdown` of the sleeve pauses the strategy (its position is flattened by the diff below) and audits `strategy.paused`.
2. Latest signal per **active** strategy (stale after `signal_ttl_secs`); paused / flattening / shadow strategies contribute no exposure.
3. Allocator weights (`equal_weight` control allocator by default; `risk_parity` / `fractional_kelly` fed by an online per-strategy performance posterior); `weight × deployable` is the strategy's sleeve.
4. `net_targets` → signed EUR target per instrument from `deployable = equity − payout quota`.
5. `fa_risk::RiskEngine::cap_targets`: per-instrument position cap, liquidity cap (`k ×` rolling traded volume, extrapolated while the window fills), per-asset-class leverage **and** concentration cap, per-venue cap (one bucket until reference data reaches the engine), portfolio gross cap.
6. `gate`: kill switch / health halts → flatten everything; daily-loss halt → reduce-only.
7. Diff against book + pending intents, no-trade band (`max(min_order_eur, band × |target|)`) → market `OrderIntent`s in `TradingMode::Paper`, account `paper`, contributors in `strategy_ids`. Each is published on `order.intent.paper` and immediately on `order.approved.paper` (the broker-gateway paper engine fills it). `flattening` strategies become `paused` once their closing intents are out (`strategy.flattened` audit).

Foreign intents seen on `order.intent.<account>` (control-api's manual orders since D-051) are gated the same way: non-paper mode or an active halt → `OrderEvent{REJECTED}` on `order.event.<account>`; otherwise republished on `order.approved.<account>`.

## Strategy lifecycle (spec 40 §3, §6.6; D-051)
| Status | Exposure | Enter via |
|---|---|---|
| `active` | allocated | `new_strategy_status = "active"` (default) on first signal; `resume`; `promote` |
| `paused` | none (position flattened) | `pause`; automatic per-strategy drawdown breach; `flatten` after the closing tick |
| `flattening` | none | `flatten` (one tick) |
| `shadow` | none; virtual book + horizon counter | `new_strategy_status = "shadow"`; `shadow` (demotion, extension beyond spec §4.4) |

`resume` returns to the status before the pause (a paused shadow strategy never skips the gate). `promote` (shadow → active, there is no live sleeve, D-029) succeeds only when the gate is green: `promotion.min_shadow_days`, `promotion.min_horizons` (signals whose `as_of` advanced by ≥ one horizon while in shadow), deflated Sharpe > `promotion.min_deflated_sharpe` read from `validation.trial(strategy_id, deflated_sharpe)` if the validation workstream adds per-strategy rows, else from `validation.generation_stats(best_candidate, deflated_sharpe)` (migration `20261110000030`) where the strategy id or its genome id is the generation's best candidate — the DSR corrects the selected candidate, so it applies to no other strategy (no match → "unknown", which blocks), and no active source/model/broker halt; PBO and shadow-vs-backtest consistency are reported as advisory. Refusals name the missing criteria.

## Control (request/reply, `CtlRequest` → `CtlReply`, message = JSON)
| Subject | Args | Reply |
|---|---|---|
| `ctl.risk.strategy` | `strategy_id`, `action` ∈ `pause\|resume\|flatten\|promote\|shadow`; or `action=list` | the strategy row (status, sleeve, weight, P&L, drawdown, shadow progress, gate checklist) / `{strategies: [...]}` |
| `ctl.risk.limits` | none → current limits; `set=<json>` partial limits object (`leverage_cap` keys merge) | the limits; `loosen: <fields>` when any field would loosen (spec 40 §6.5) |
| `ctl.risk.flatten` | none | `{flattening: [ids]}` — every active strategy is flattened and paused, no halt |

Errors use the control-api prefix convention (`not_found:` / `conflict:` / `invalid:` / `loosen:`). An `actor` arg (`ui`) is recorded in `audit.log` (`strategy.<action>`, `limits.tightened`, `flatten.portfolio`).

## Halts
| Scope | Raised by | Effect | Cleared by |
|---|---|---|---|
| `portfolio` | drawdown ≥ `limits.max_drawdown`, or any `risk.halt.portfolio` (UI kill switch) | flatten, no orders | `risk.resume.portfolio` (re-arms the drawdown peak) |
| `daily_loss` | loss since UTC day start ≥ `limits.max_daily_loss` | reduce-only | next UTC day, or `risk.resume.daily_loss` |
| `source:<id>` | ingestor (`risk.halt.source:<id>`) or `ingest.health.<id>` frame with `status: down` | flatten, no orders | `ingest.health.<id>` `status: ok` (health-raised halts only), `risk.resume.source:<id>`, or bare `risk.resume.source` for all |
| `broker` | reconciliation diff (spec 42), or `risk.halt.broker` / `venue:*` from other services | flatten, no orders | `risk.resume.broker` |
| `model` | other services | flatten, no orders | `risk.resume.model` |
| per strategy | attributed drawdown ≥ `strategy_max_drawdown` of the sleeve | that strategy paused + flattened | `ctl.risk.strategy resume` |

## Settings
[`risk.toml`](risk.toml) (`--risk`, `FA_RISK`); the built-in copy is used when the file is missing. `mode` must be `paper`; anything else fails startup. Limits follow the spec 40 §7 placeholders; at runtime they can only be tightened (`ctl.risk.limits`), loosening needs an edit + restart. `new_strategy_status`, `shadow_sleeve_eur` and `[promotion]` configure the lifecycle.

## Run
```sh
cargo run -p risk-engine                                   # config/platform.toml, bins/risk-engine/risk.toml
FA_CONFIG=config/platform.docker.toml FA_RISK=/app/bins/risk-engine/risk.toml risk-engine   # in the Rust image
```
Then `GET /api/trading/{signals,orders,positions,risk,strategies,limits}` on control-api shows the flow; `POST /api/trading/halt` flattens and stops orders, `POST /api/trading/resume` re-arms; `POST /api/trading/strategies/{id}/pause|resume|flatten|promote|shadow`, `PUT /api/trading/limits` and `POST /api/trading/flatten` drive the control subjects.

## Layout
- `src/engine.rs` — pure, deterministic engine (`on_signal`, `on_trade`, `on_intent_seen`, `on_order_event`, `on_halt`, `on_resume`, `on_source_health`, `strategy_action`, `set_limits`, `flatten_all`, `tick`, `state`, `strategies_view`), tested end-to-end with synthetic signals/fills (lifecycle, shadow books, per-source halts, tighten-only limits).
- `src/settings.rs` — `risk.toml` schema and validation.
- `src/persist.rs` — Postgres side effects (`fa-ledger::pg::PgLedger`, `fa-warehouse::audit`, `ledger.strategy_state`, `validation.trial` lookup): intents, fills, snapshots, strategy rows, reconciliation runs, audit entries; best effort, idempotent (DB test gated on `DATABASE_URL`).
- `src/main.rs` — NATS wiring, control subjects, heartbeat `svc.heartbeat.risk-engine` (counters: signals, trades, intents.ours, intents.foreign, approved.foreign, fills, halts.*, resumes, strategies.seen, strategy.actions, strategy.drawdown_pauses, limits.tightened, flatten.portfolio, equity_eur, cash_eur, gross_exposure_eur, positions, halted).

## Governing specs
- [40-execution-and-risk](../../specs/40-execution-and-risk.md)
- [42-ledger-and-accounting](../../specs/42-ledger-and-accounting.md)
- [21-bus-and-contracts](../../specs/21-bus-and-contracts.md)
