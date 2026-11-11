# risk-engine

Live service: meta-allocation across strategy signals, non-evolvable hard risk limits, the halt state machine and the paper book (specs 40, 42; D-028, D-029, D-030).

**Status:** running (paper only) — `signal.exposure.>` → strategy lifecycle filter (only `active` strategies) → allocator (`fa-allocator`) → EUR targets → caps + halt gate (`fa-risk`) → diff against the paper book (`fa-ledger`) → `order.intent.paper` + `order.approved.paper`; books fills from `order.event.>` and attributes them to strategy sub-books, marks from `md.trade.>`; publishes `risk.state` (equity, deployable, gross exposure, drawdown, daily P&L, active halts) and raises `risk.halt.portfolio` / `risk.halt.daily_loss` / `risk.halt.broker` itself; honours `risk.halt.>` (incl. the UI kill switch and the ingestor's `risk.halt.source:<id>`), `risk.resume.<scope>` and `ingest.health.<source>`. Operator control is request/reply on `ctl.risk.strategy|limits|flatten` (D-051, below). With `[postgres]` reachable the paper book is persisted (D-047): the 100 000 EUR opening deposit is a `cash_movement` booked once, every own and foreign intent and every fill is written to `ledger.*` (idempotent), a `position_snapshot` and every `ledger.strategy_state` row are written every `snapshot_secs` (60; changed strategy rows within 1 s, snapshots also on each halt), halts/resumes/limit loads/lifecycle actions/limit changes go to `audit.log`, and every `reconcile_secs` (300) the in-memory book is compared with the Postgres-derived one into `reconciliation_run` (`ok`/`diff`); **a diff halts scope `broker`** (flatten, no orders) until `risk.resume.broker` (spec 42 §6). Restarting reloads the book, the strategy states **and each strategy's sub-book** from Postgres (drawdown peak and day-start equity restart from the loaded equity; intents pending at shutdown are not restored). SIGTERM / Ctrl-C flush a final position snapshot and every strategy row before exit. Without Postgres it logs once and stays in memory, with FX 1.0 and no reference data (every instrument `default_asset_class`, one venue bucket).

## Reference data, FX, sub-books (D-061)
- **Reference data:** `fa_warehouse::venues` (`ref.instrument` + current `ref.listing` + `ref.venue`) gives per instrument asset class, kind, venue code, quote currency, status and the listing source(s). Loaded at start, every `refdata_refresh_secs` (300) and, debounced by 2 s, on any `ref.changed.>` or `ingest.unresolved.>` message. Instruments without a row use `default_asset_class` and the unnamed venue bucket; they are counted in the gauge `refdata.unknown`. The class and venue caps (D-051) apply per real class and venue. `provisional` instruments (D-050) get `limits.provisional_position_frac`, which defaults to `max_position_frac × 0.5` and is never looser than it.
- **FX (spec 42 §6, spec 12 §6):** a task fetches `https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml` every `fx.ecb_refresh_secs` (86 400, and once at start). It upserts EUR reference rates into `ref.fx_rate` (`source_id = ecb_daily`, idempotent on `(ccy, ts, source_id)`, `rate_to_eur = 1 / published rate`, `ts` = the rate day at 00:00 UTC) and reloads the in-memory as-of table (`fx.history_days`). Fetch errors are logged and counted (`fx.ecb_errors`), and the last table stays in use.
  - Every mark (`md.trade`) and fill is converted from the instrument's `quote_ccy` (unknown = EUR) with the latest rate dated at or before the event time. The fee is converted from `fee_currency`.
  - `ledger.fill` records `ccy`, `fx_rate_eur`, `fx_source` (`eur` / `ecb_daily` / `missing`), `arrival_price` and `slippage_bps`.
  - A rate older than `fx.max_age_hours` (72) is still used but flagged stale: counter `fx.stale` per conversion, `stale` list in the portfolio view. With `fx.halt_on_stale = true` the engine raises the reduce-only halt `fx` (`risk.halt.fx`) and clears it itself (`risk.resume.fx`) once a fresh table arrives. The default is warning only.
- **Sub-books:** each strategy's attributed sub-book is persisted in `ledger.strategy_state.book` (JSONB, migration `20261110000060`) with every dirty row (≤ 1 s), every `snapshot_secs` and at shutdown. The book holds positions with average cost and realised P&L, cash, fees, the last marked value, the realised-slippage sum and count, and hourly P&L points (bounded to 60 days). Shadow books also keep their virtual exposure and horizon anchor. A restored book books no P&L until every held instrument has a mark again, so a restart never shows a jump between the persisted value and a cost-marked one.
- **Payout quota (spec 40 §6.5, spec 42 §6):** the reserve is the larger of `limits.daily_payout_quota_eur` and today's `ledger.payout_quota` row (`max(quota − paid, reserved)`, polled every 30 s), and deployable = equity − reserve. `ctl.risk.limits {set: {daily_payout_quota_eur}}` raises both the limit and today's row. The quota may only rise: a lower value gets `loosen:`. Payout booking itself (`cash_movement kind=payout`) stays manual; the rule is still open (spec 42 Q1).

## Pipeline per tick (spec 40 §6.1)
1. Strategy books: every active strategy's attributed sub-book (fills split evenly across the intent's contributors, D-047) and every shadow strategy's virtual book (its own signal on `shadow_sleeve_eur`) are marked to market; a drawdown ≥ `limits.strategy_max_drawdown` of the sleeve pauses the strategy (its position is flattened by the diff below) and audits `strategy.paused`.
2. Latest signal per **active** strategy (stale after `signal_ttl_secs`); paused / flattening / shadow strategies contribute no exposure.
3. Allocator weights (`equal_weight` control allocator by default; `risk_parity` / `fractional_kelly` fed by an online per-strategy performance posterior); `weight × deployable` is the strategy's sleeve.
4. `net_targets` → signed EUR target per instrument from `deployable = equity − payout quota`.
5. `fa_risk::RiskEngine::cap_targets` applies these caps:
   - per-instrument position cap, stricter for `provisional` instruments;
   - liquidity cap (`k ×` rolling traded volume, extrapolated while the window fills);
   - per-asset-class leverage **and** concentration cap, and the per-venue cap, both from reference data;
   - portfolio gross cap;
   - shorts: no short without borrow (class absent from `limits.short_margin`, or `ref.contract_spec.shortable = false`), all shorts scaled down when Σ |short| × class margin exceeds `short_margin_budget_frac` × deployable.
6. `gate`: kill switch and model/broker halts flatten everything; daily-loss and `fx` halts are reduce-only. A `source:<id>` halt flattens only the instruments that source feeds when every instrument in play has exactly one known listing source; otherwise it flattens everything. The bare `source` scope always flattens everything. Foreign intents follow the same per-instrument rule.
7. Diff against book + pending intents, no-trade band (`max(min_order_eur, band × |target|)`) → `OrderIntent`s whose `exec_algo` comes from `fa_risk::exec` (`[exec]`: urgent or horizon < 60 s → `market`; else `pov` / `twap` / `vwap` / `smart_limit` by participation in the horizon volume, spec 40 §6.4); a working order stays pending for its working time and partial fills shrink the pending remainder; in `TradingMode::Paper`, account `paper`, contributors in `strategy_ids`. Each is published on `order.intent.paper` and immediately on `order.approved.paper` (the broker-gateway paper engine fills it). `flattening` strategies become `paused` once their closing intents are out (`strategy.flattened` audit).

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
| `ctl.risk.strategy` | `strategy_id`, `action` ∈ `pause\|resume\|flatten\|promote\|shadow`; or `action=list` | the strategy row (status, sleeve, weight, P&L, drawdown, `slippage_bps_mean`, `book` {cash, value, realised, fees, positions, history points}, shadow progress, gate checklist) / `{strategies: [...], portfolio: {by_venue, by_class, target_by_venue, target_by_class, provisional_eur, payout_reserve_eur, fx, refdata, allocator, halts}}` |
| `ctl.risk.limits` | none → current limits (+ `allocator`, `payout_reserve_eur`); `set=<json>` partial limits object (`leverage_cap` keys merge); `set.allocator = {name, params}` or `{genome: "alloc:…"}` switches the allocator at runtime (`fa_allocator::genome`: `equal_weight\|risk_parity\|fractional_kelly\|weighted_blend`, genes `cap, kelly, w_eq, w_rp, w_k, hl, pv`, cap ≤ `max_strategy_weight`). Not a limit: allowed either way, audited `allocator.changed`, not persisted (a restart returns to `risk.toml`) | the limits; `loosen: <fields>` when any field would loosen (spec 40 §6.5) |
| `ctl.risk.flatten` | none | `{flattening: [ids]}` — every active strategy is flattened and paused, no halt |

Errors use the control-api prefix convention (`not_found:` / `conflict:` / `invalid:` / `loosen:`). An `actor` arg (`ui`) is recorded in `audit.log` (`strategy.<action>`, `limits.tightened`, `flatten.portfolio`).

## Halts
| Scope | Raised by | Effect | Cleared by |
|---|---|---|---|
| `portfolio` | drawdown ≥ `limits.max_drawdown`, or any `risk.halt.portfolio` (UI kill switch) | flatten, no orders | `risk.resume.portfolio` (re-arms the drawdown peak) |
| `daily_loss` | loss since UTC day start ≥ `limits.max_daily_loss` | reduce-only | next UTC day, or `risk.resume.daily_loss` |
| `source:<id>` | ingestor (`risk.halt.source:<id>`) or `ingest.health.<id>` frame with `status: down`; one entry per source id, each listed in `risk.state` | flatten that source's instruments (rule in step 6), no new orders on them | `ingest.health.<id>` `status: ok` (health-raised halts only), `risk.resume.source:<id>`, or bare `risk.resume.source`, which clears **every** source halt |
| `source` (bare) | `risk.halt.source` without an id | flatten everything | bare `risk.resume.source` |
| `fx` | accounting FX older than `fx.max_age_hours` with `fx.halt_on_stale = true` (spec 42 §8) | reduce-only | a fresh rate (engine publishes `risk.resume.fx`), or `risk.resume.fx` |
| `broker` | reconciliation diff (spec 42), or `risk.halt.broker` / `venue:*` from other services | flatten, no orders | `risk.resume.broker` |
| `model` (bare) | other services | flatten, no orders | `risk.resume.model` (also clears every snapshot halt) |
| `model:<snapshot>` | `risk.halt.model:<snapshot>`, or the engine's drift alarm on `latent.mcmc.*` (`drift_kl ≥ [model_drift] halt_kl` for `sustain` updates) | strategies of that snapshot's timescale leave allocation (their sleeves net away); others trade on | `risk.resume.model:<snapshot>`; a halt the alarm raised also clears itself after `clear_sustain` updates ≤ `clear_kl` |
| per strategy | attributed drawdown ≥ `strategy_max_drawdown` of the sleeve | that strategy paused + flattened | `ctl.risk.strategy resume` |

## Settings
[`risk.toml`](risk.toml) (`--risk`, `FA_RISK`); the built-in copy is used when the file is missing. `mode` must be `paper`; anything else fails startup. Limits follow the spec 40 §7 placeholders; at runtime they can only be tightened (`ctl.risk.limits`), loosening needs an edit + restart. `new_strategy_status`, `shadow_sleeve_eur` and `[promotion]` configure the lifecycle; `refdata_refresh_secs` and `[fx]` (`ecb_refresh_secs`, `max_age_hours`, `halt_on_stale`, `source`, `history_days`) configure reference data and FX; `[exec]` the algorithm choice, `[model_drift]` the drift alarm, `[limits.short_margin]` / `short_margin_budget_frac` the paper short rules. Heartbeat gauges `headroom.*` give the distance to every limit (spec 60 §6.5).

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
- `src/main.rs` — NATS wiring, control subjects, heartbeat `svc.heartbeat.risk-engine` (counters: signals, trades, intents.ours, intents.foreign, approved.foreign, fills, halts.*, resumes, strategies.seen, strategy.actions, strategy.drawdown_pauses, limits.tightened, flatten.portfolio, allocator.switches, strategies.restored, refdata.loads, refdata.instruments, refdata.unknown, fx.rates, fx.stale, fx.stale_now, fx.ecb_fetches, fx.ecb_errors, payout_reserve_eur, equity_eur, cash_eur, gross_exposure_eur, `exposure_eur.venue.<code>`, positions, halted).

## Governing specs
- [40-execution-and-risk](../../specs/40-execution-and-risk.md)
- [42-ledger-and-accounting](../../specs/42-ledger-and-accounting.md)
- [21-bus-and-contracts](../../specs/21-bus-and-contracts.md)
