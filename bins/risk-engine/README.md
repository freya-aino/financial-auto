# risk-engine

Live service: meta-allocation across strategy signals, non-evolvable hard risk limits, the halt state machine and the paper book (retired spec 40 — its lasting content is [below](#execution-and-risk-lasting-content-of-the-retired-spec-40) —, spec 42; D-028, D-029, D-030).

**Status:** running (paper only) — `signal.exposure.>` → strategy lifecycle filter (only `active` strategies) → allocator (`fa-allocator`) → EUR targets → caps + halt gate (`fa-risk`) → diff against the paper book (`fa-ledger`) → cost-aware rebalance rule → `order.intent.paper` + `order.approved.paper` with a chosen execution algorithm; books fills from `order.event.paper` only (its own account, D-072) and attributes them to strategy sub-books, marks from `md.trade.>` (through a per-instrument feed guard); publishes `risk.state` (equity, deployable, gross exposure, drawdown, daily P&L, active halts) and raises `risk.halt.portfolio` / `risk.halt.daily_loss` / `risk.halt.broker` itself; honours `risk.halt.>` (incl. the UI kill switch and the ingestor's `risk.halt.source:<id>`), `risk.resume.<scope>` and `ingest.health.<source>`. Operator control is request/reply on `ctl.risk.strategy|limits|flatten|pipeline|allocators` (D-051, D-071, below). With Postgres it also runs the automatic evolved-strategy lifecycle and keeps a locked holdout per timescale (D-071, below). With `[postgres]` reachable the paper book is persisted (D-047): the 100 000 EUR opening deposit is a `cash_movement` booked once, every own and foreign intent and every fill is written to `ledger.*` (idempotent), a `position_snapshot` and every `ledger.strategy_state` row are written every `snapshot_secs` (60; changed strategy rows within 1 s, snapshots also on each halt), halts/resumes/limit loads/lifecycle actions/limit changes go to `audit.log`, and every `reconcile_secs` (300) the in-memory book is compared with the Postgres-derived one at one cut (`PgLedger::reconcile_cut`: exact decimals, fills younger than 60 s excluded, rows only Postgres has applied to memory) into `reconciliation_run` (`ok`/`diff`); **a diff halts scope `broker`** (flatten, no orders) until `risk.resume.broker` (spec 42 §6). Fill writes are confirmed per fill and retried every 5 s; a fill for an unknown intent is recovered from `ledger.order_intent` or parked in `ledger.orphan_fill`, and an open paper orphan also raises `broker` (`src/recon.rs`, D-072). Restarting reloads the book, the strategy states, **each strategy's sub-book** and the halt state machine (`ledger.risk_state`: active halts incl. the kill switch, drawdown peak, day-start equity, hourly equity, tournament records) from Postgres; limits always come from `risk.toml`. SIGTERM / Ctrl-C flush a final position snapshot, the risk state and every strategy row before exit. Without Postgres it logs once and stays in memory, with FX 1.0 and no reference data (every instrument `default_asset_class`, one venue bucket).

## Reference data, FX, sub-books (D-061)
- **Reference data:** `fa_warehouse::venues` (`ref.instrument` + current `ref.listing` + `ref.venue`) gives per instrument asset class, kind, venue code, quote currency, status and the listing source(s). Loaded at start, every `refdata_refresh_secs` (300) and, debounced by 2 s, on any `ref.changed.>` or `ingest.unresolved.>` message. Instruments without a row use `default_asset_class` and the unnamed venue bucket; they are counted in the gauge `refdata.unknown`. The class and venue caps (D-051) apply per real class and venue. `provisional` instruments (D-050) get `limits.provisional_position_frac`, which defaults to `max_position_frac × 0.5` and is never looser than it.
- **FX (spec 42 §6, spec 12 §6):** a task fetches `https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml` every `fx.ecb_refresh_secs` (86 400, and once at start). It upserts EUR reference rates into `ref.fx_rate` (`source_id = ecb_daily`, idempotent on `(ccy, ts, source_id)`, `rate_to_eur = 1 / published rate`, `ts` = the rate day at 00:00 UTC) and reloads the in-memory as-of table (`fx.history_days`). Fetch errors are logged and counted (`fx.ecb_errors`), and the last table stays in use.
  - Every mark (`md.trade`) and fill is converted from the instrument's `quote_ccy` (unknown = EUR) with the latest rate dated at or before the event time. The fee is converted from `fee_currency`.
  - `ledger.fill` records `ccy`, `fx_rate_eur`, `fx_source` (`eur` / `ecb_daily` / `missing`), `arrival_price` and `slippage_bps`.
  - A rate older than `fx.max_age_hours` (72) is still used but flagged stale: counter `fx.stale` per conversion, `stale` list in the portfolio view. A stale currency is reduce-only by default (`fx.block_stale_ccy`, spec 42 §8): targets on instruments quoted in it are held toward zero (`risk.event fx.stale_block`, gauge `fx.blocked`) and foreign orders adding risk in it are rejected pre-trade, while other currencies trade on (`blocked` in the portfolio view). With `fx.halt_on_stale = true` the engine additionally raises the book-wide reduce-only halt `fx` (`risk.halt.fx`) and clears it itself (`risk.resume.fx`) once a fresh table arrives.
  - Intraday FX (`fx.intraday`, default on, spec 42 §6): trades of EUR FX pairs in reference data (`fa_warehouse::fx::load_eur_crosses`, today Yahoo `EURUSD=X`) become rates under the feed's source id (`yahoo`) unless more than `fx.intraday_max_dev` (5 %) off the latest daily rate (`fx.intraday_rejected`). They survive the daily reloads, are pruned after 48 h in memory and are written to `ref.fx_rate` every 60 s (`fx.intraday_rates`, `fx.intraday_store_errors`; `intraday` in the portfolio view).
- **Sub-books:** each strategy's attributed sub-book is persisted in `ledger.strategy_state.book` (JSONB, migration `20261110000060`) with every dirty row (≤ 1 s), every `snapshot_secs` and at shutdown. The book holds positions with average cost and realised P&L, cash, fees, the last marked value, the realised-slippage sum and count, and hourly P&L points (bounded to 60 days). Shadow books also keep their virtual exposure and horizon anchor. A restored book books no P&L until every held instrument has a mark again, so a restart never shows a jump between the persisted value and a cost-marked one.
- **Payout quota (spec 40 §6.5, spec 42 §6):** the reserve is the larger of `limits.daily_payout_quota_eur` and today's `ledger.payout_quota` row (`max(quota − paid, reserved)`, polled every 30 s), and deployable = equity − reserve. `ctl.risk.limits {set: {daily_payout_quota_eur}}` raises both the limit and today's row. The quota may only rise: a lower value gets `loosen:`. The `[payout]` rule (`fa_ledger::payout::PayoutRule`: `fixed_eur` | `pct_equity` | `pct_profit`, `rollover`, capped at `max_equity_frac` 0.05; default `fixed_eur` 0) raises today's row at the first 30 s check of each UTC day (raise-only, idempotent). Booking the payout itself (`cash_movement kind=payout`, `POST /api/trading/payouts`) stays an operator action.

## Pipeline per tick (spec 40 §6.1)
0. Marks pass the per-instrument feed guard (`fa_risk::health`, `[feed]`): an instrument without a trade for `stale_freeze_bars` × max(fastest strategy bar, its own trade gap) is frozen reduce-only; a trade more than `jump_max` from the mark is held until `jump_confirm` trades confirm it.
1. Strategy books: every active strategy's attributed sub-book (fills split across the intent's contributors by the causal pro-rata rule, §10 Q3 below; evenly for foreign intents, D-047) and every shadow strategy's virtual book (its own signal on `shadow_sleeve_eur`) are marked to market; a drawdown ≥ `limits.strategy_max_drawdown` of the sleeve pauses the strategy (its position is flattened by the diff below) and audits `strategy.paused`.
2. Latest signal per **active** strategy (stale after `signal_ttl_secs`); paused / flattening / shadow / pre-live / retired strategies, and strategies of a timescale under a `model:<snapshot>` halt, contribute no exposure; a strategy over its turnover budget contributes its last (frozen) distribution.
3. Allocator weights (`equal_weight` control allocator by default; `risk_parity` / `fractional_kelly` / `thompson_rp` / a `weighted_blend` genome fed by an online per-strategy performance posterior); NaN, negative or over-unit weights fall back to the previous allocation and raise a risk event; `weight × deployable` is the strategy's sleeve. The live allocator tournament (`fa_allocator::tournament`: the four fixed allocators + the active one, `tournament_cost_bps`) runs on the same posteriors.
4. `net_targets` → signed EUR target per instrument from `deployable = equity − payout quota`.
5. `fa_risk::RiskEngine::cap_targets` applies these caps:
   - per-instrument position cap, stricter for `provisional` instruments;
   - liquidity cap (`k ×` rolling traded volume, extrapolated while the window fills);
   - per-asset-class leverage **and** concentration cap, and the per-venue cap, both from reference data;
   - portfolio gross cap;
   - shorts: no short without borrow (class absent from `limits.short_margin`, or `ref.contract_spec.shortable = false`), all shorts scaled down when Σ |short| × class margin exceeds `short_margin_budget_frac` × deployable.
6. `gate`: kill switch and model/broker halts flatten everything; daily-loss and `fx` halts are reduce-only. A `source:<id>` halt flattens only the instruments that source feeds when every instrument in play has exactly one known listing source; otherwise it flattens everything. The bare `source` scope always flattens everything. Foreign intents follow the same per-instrument rule.
7. Diff against book + pending intents, no-trade band (`max(min_order_eur, band × |target|)`), then the **cost-aware rebalance rule** (`fa_risk::rebalance`, `[rebalance]`, spec 40 §6.1: expected benefit ≥ full expected cost from `costs.yaml` (`FA_COSTS`) with impact `k` from `exec.costs.calibrated`, min order notional from the fee minimum, `confirm_secs` batching, per-strategy turnover budget; urgent and limit-bound orders bypass it; counters `rebalance.*`) → `OrderIntent`s whose `exec_algo` comes from `fa_risk::exec` (`[exec]`: urgent or horizon < 60 s → `market`; else `pov` / `twap` / `vwap` / `smart_limit` by participation in the horizon volume, spec 40 §6.4); a working order stays pending for its working time and partial fills shrink the pending remainder; in `TradingMode::Paper`, account `paper`, contributors in `strategy_ids`. Each is published on `order.intent.paper` and immediately on `order.approved.paper` (the broker-gateway paper engine fills it). `flattening` strategies become `paused` once their closing intents are out (`strategy.flattened` audit).

Foreign intents seen on `order.intent.<account>` (control-api's manual orders since D-051) are gated the same way: non-paper mode, an active halt or a failed pre-trade check (`fa_risk::pretrade`: sanity, `max_order_frac`, `limit_price_band`, post-trade caps, short borrow / margin, frozen feed reduce-only; reductions always pass) → `OrderEvent{REJECTED}` on `order.event.<account>` plus a `RiskEvent` on `risk.event.<account>`; otherwise republished on `order.approved.<account>`.

Limit clips (target cut by more than `min_order_eur`), pre-trade rejections, feed freezes / held jumps / stale flattens and allocator fallbacks go out as `fa.v1.RiskEvent{LIMIT_BREACH}` on `risk.event.paper` (at most once per minute per kind and instrument) and into `audit.log` as `risk.event.<kind>`; halts, resumes and rejections are mirrored as `HALT_RAISED|HALT_CLEARED|INTENT_REJECTED`.

## Strategy lifecycle (spec 40 §3, §6.6; D-051, D-071)
| Status | Exposure | Enter via |
|---|---|---|
| `candidate` | none (not running) | pipeline selection from a strategy generation's front |
| `holdout_passed` | none (not running) | pipeline holdout verdict |
| `active` | allocated | `new_strategy_status = "active"` (default) on first signal; `resume`; `promote` |
| `paused` | none (position flattened) | `pause`; automatic per-strategy drawdown breach; `flatten` after the closing tick |
| `flattening` | none | `flatten` (one tick) |
| `shadow` | none; virtual book + horizon counter | `new_strategy_status = "shadow"`; `admit` (pipeline admission into the managed live set); `shadow` (demotion, extension beyond spec §4.4) |
| `retired` | none (terminal, removed from the live set) | `retire`; pipeline (holdout failure, shadow drawdown, `max_shadow_days`, `max_demotions`, runtime auto-pause) |

`resume` returns to the status before the pause (a paused shadow strategy never skips the gate). `promote` (shadow → active, there is no live sleeve, D-029) succeeds only when the gate is green; every item blocks and unknown blocks: `promotion.min_shadow_days`, `promotion.min_horizons` (signals whose `as_of` advanced by ≥ one horizon while in shadow), deflated Sharpe > `promotion.min_deflated_sharpe` (the strategy's holdout trial, else `validation.generation_stats` of a generation it or its genome won — the DSR corrects the selected candidate only), PBO of its generation < `promotion.max_pbo`, shadow-vs-backtest consistency (`fa_risk::strategy::consistency`, `promotion.consistency_z`, ≥ `decay_min_hours` hourly points) and no active source/model/broker halt. PBO and consistency are `n/a` for hand-written strategies. Refusals name the missing criteria.

## Strategy pipeline and holdouts (spec 35 §6.4.1, spec 41 §6.4; D-071)
With Postgres, `src/pipeline.rs` runs a round at start, every `[pipeline] interval_secs` (60) and on `ctl.risk.pipeline {action: run}` (decisions in `fa_risk::pipeline`):
1. **Holdouts:** every `[holdout] timescales` entry (and every waiting candidate's timescale) gets a locked holdout `[end − length_secs.<ts>, end)` once VictoriaMetrics holds ≥ `min_bars` bars in it (advisory-locked, overlap-checked insert into `validation.holdout`, audited `validation.holdout.lock`); missing ones are reported (`holdout.missing`, status `holdouts.<ts>.reason`).
2. **Discovery / selection:** the newest `generations_scanned` `fa-artifacts/strategies/<generation>.toml` fronts; members with known generation PBO ≤ `pbo_max` and selection DSR ≥ `dsr_min_candidate` (best `max_candidates_per_generation`) become `candidate`.
3. **Holdout:** one `evo.eval.strategy.holdout` job per candidate on a holdout its selection windows never touched (retried up to `holdout_max_attempts` when no trial row arrives within `holdout_timeout_secs`); pass = DSR ≥ `holdout_min_dsr`, Sharpe > 0, ≥ `holdout_min_obs` → `holdout_passed`, else `retired`.
4. **Admission:** while fewer than `max_live_evolved` evolved strategies are live, rewrite `fa-artifacts/live/strategies.toml` and send `ctl.strategy.reload {artifact_key: managed}` → `shadow`.
5. **Promotion / demotion / retirement:** shadow → active on a green gate with `auto_promote`; active → shadow on drawdown, gate decay or a strategy-runtime auto-pause, or after a snapshot remap / suspension (`strategy.snapshot_shadow`); retired after `max_demotions`, on `max_shadow_days`, shadow drawdown or definitive gate failure.
6. **Helix mirror:** each evolved strategy's status onto its `Genome` node (best effort, `helix.mirror_written|failed`, `helix.mirrored`).

Transitions are persisted in `ledger.strategy_state` (`pipeline` JSONB, `last_reason`) and audited with actor `system:pipeline`. Hand-written baselines are never touched.

## Control (request/reply, `CtlRequest` → `CtlReply`, message = JSON)
| Subject | Args | Reply |
|---|---|---|
| `ctl.risk.strategy` | `strategy_id`, `action` ∈ `pause\|resume\|flatten\|promote\|shadow\|retire\|admit`; or `action=list` | the strategy row (status, sleeve, weight, P&L, drawdown, `slippage_bps_mean`, `book` {cash, value, realised, fees, positions, history points}, shadow progress, gate checklist) / `{strategies: [...], portfolio: {by_venue, by_class, target_by_venue, target_by_class, provisional_eur, payout_reserve_eur, fx, refdata, allocator, halts}}` |
| `ctl.risk.limits` | none → current limits (+ `allocator`, `payout_reserve_eur`); `set=<json>` partial limits object (`leverage_cap` keys merge); `set.allocator = {name, params}` or `{genome: "alloc:…"}` switches the allocator at runtime (`fa_allocator::genome`: `equal_weight\|risk_parity\|fractional_kelly\|weighted_blend\|thompson_rp`, genes `cap, shrink, kelly, w_eq, w_rp, w_k, hl, pv`, cap ≤ `max_strategy_weight`). Not a limit: allowed either way, audited `allocator.changed`, not persisted (a restart returns to `risk.toml`) | the limits; `loosen: <fields>` when any field would loosen (spec 40 §6.5) |
| `ctl.risk.flatten` | none | `{flattening: [ids]}` — every active strategy is flattened and paused, no halt |
| `ctl.risk.limits` | `calibrate = "{}"` or `"{\"apply\": true}"` | `fa_risk::calibrate` suggestion from hourly equity and instrument marks plus the tighten-only set; `apply` applies only stricter values (audited `limits.calibrated`) |
| `ctl.risk.pipeline` | `action` ∈ `status` (default) \| `run` | pipeline status (holdouts, candidates, live set, `helix_mirror`) / the result of an immediate round (audited `pipeline.run`); needs Postgres |
| `ctl.risk.allocators` | none | live allocator tournament: per entry cumulative return, Sharpe, max drawdown, turnover, costs, weights |
| `ctl.risk.pretrade` | a JSON request `{instrument_id, side: buy\|sell, quantity, limit_price?, mode?}` (reply: JSON `{ok, reasons: [..]}`), or a `CtlRequest` with the same fields as `args` (reply: `CtlReply{ok = verdict, message = that JSON}`) | the checks a manual order would meet (mode, halts, `fa_risk::pretrade`, feed guard, stale FX) against the in-memory book, **without side effects** (no intent, no event, no counter); behind `POST /api/trading/pretrade` |

Errors use the control-api prefix convention (`not_found:` / `conflict:` / `invalid:` / `loosen:`). An `actor` arg (`ui`) is recorded in `audit.log` (`strategy.<action>`, `limits.tightened`, `flatten.portfolio`).

## Halts
| Scope | Raised by | Effect | Cleared by |
|---|---|---|---|
| `portfolio` | drawdown ≥ `limits.max_drawdown`, or any `risk.halt.portfolio` (UI kill switch) | flatten, no orders | `risk.resume.portfolio` (re-arms the drawdown peak) |
| `daily_loss` | loss since the day start ≥ `limits.max_daily_loss`; the day is per calendar group (UTC for 24/7 venues, the exchange day for session venues, below) | reduce-only | the next roll that leaves the loss below the limit, or `risk.resume.daily_loss` |
| `source:<id>` | ingestor (`risk.halt.source:<id>`) or `ingest.health.<id>` frame with `status: down`; one entry per source id, each listed in `risk.state` | flatten that source's instruments (rule in step 6), no new orders on them | `ingest.health.<id>` `status: ok` (health-raised halts only), `risk.resume.source:<id>`, or bare `risk.resume.source`, which clears **every** source halt |
| `source` (bare) | `risk.halt.source` without an id | flatten everything | bare `risk.resume.source` |
| `fx` | accounting FX older than `fx.max_age_hours` with `fx.halt_on_stale = true` (spec 42 §8) | reduce-only | a fresh rate (engine publishes `risk.resume.fx`), or `risk.resume.fx` |
| `broker` | reconciliation diff or an open paper orphan fill (spec 42), or `risk.halt.broker` / `venue:*` from other services | flatten, no orders | `risk.resume.broker` |
| `model` (bare) | other services | flatten, no orders | `risk.resume.model` (also clears every snapshot halt) |
| `model:<snapshot>` | `risk.halt.model:<snapshot>`, or the engine's drift alarm on `latent.mcmc.*` (`drift_kl ≥ [model_drift] halt_kl` for `sustain` updates) | strategies of that snapshot's timescale leave allocation (their sleeves net away); others trade on | `risk.resume.model:<snapshot>`; a halt the alarm raised also clears itself after `clear_sustain` updates ≤ `clear_kl` |
| per strategy | attributed drawdown ≥ `strategy_max_drawdown` of the sleeve | that strategy paused + flattened | `ctl.risk.strategy resume` |

Every halt above except the per-strategy pause (which lives in `ledger.strategy_state`), including `source:<id>` and `model:<snapshot>` halts, is persisted in `ledger.risk_state` and survives a restart until cleared.

## Settings
[`risk.toml`](risk.toml) (`--risk`, `FA_RISK`); the built-in copy is used when the file is missing. `mode` must be `paper`; anything else fails startup. Limits follow the spec 40 §7 placeholders; at runtime they can only be tightened (`ctl.risk.limits`), loosening needs an edit + restart. `new_strategy_status`, `shadow_sleeve_eur` and `[promotion]` configure the lifecycle; `refdata_refresh_secs` and `[fx]` (`ecb_refresh_secs`, `max_age_hours`, `halt_on_stale`, `block_stale_ccy`, `intraday`, `intraday_max_dev`, `source`, `history_days`) configure reference data and FX; `[exec]` the algorithm choice, `[rebalance]` the cost-aware rule (with `FA_COSTS` = the gateway's `costs.yaml`; without it a built-in default schedule), `[model_drift]` the drift alarm, `[limits.short_margin]` / `short_margin_budget_frac` the paper short rules, `[feed]` the feed guard, `max_order_frac` / `limit_price_band` the pre-trade checks, `[payout]` the daily payout quota rule, `tournament_cost_bps` the live tournament, `[history]` the risk-chain history (`snapshot_secs` 60, `retention_days` 30, `review_unchanged_secs` 3600), `[pipeline]` and `[holdout]` (`length_secs`, `embargo_secs`, `min_bars`, `relock_after_days`) the strategy pipeline and holdout locking — `embargo_secs` is stored with every auto-locked row (`validation.holdout.embargo_ms`, migration `20261113000040`) and evo-worker keeps selection windows `max(it, 2 × horizon, 60 s)` before that holdout. evo-worker reads the same file (`FA_RISK`) for its fitness replays' rebalance rule (`no_trade_band_frac`, `min_order_eur`, `performance_decay`, `[rebalance]`), so selection is scored under the rule this engine trades with. Heartbeat gauges `headroom.*` give the distance to every limit (spec 60 §6.5).

## Execution and risk (lasting content of the retired spec 40)
Spec 40 was retired on 2026-11-11 once every in-scope acceptance criterion was met and its open questions decided; the live-venue items stay out of scope until a live adapter exists. Its decisions, data model, rules, tunables and failure modes live here; the D-entries in [`specs/DECISIONS.md`](../../specs/DECISIONS.md) (D-028..D-030, D-041, D-047, D-051, D-061, D-071, D-072, D-076, D-079 and later) record why.

### Decisions
- **Paper first, gated promotion.** Every strategy starts in paper; promotion needs a minimum shadow time plus statistics (gate below). Only paper exists (D-029): any non-paper mode is refused at startup, by this engine and by the gateway.
- **Pluggable meta-allocator** (`fa_allocator`): `equal_weight` (control, configured default), `risk_parity`, `fractional_kelly`, `weighted_blend`, the Bayesian bandit `thompson_rp`, and evolved `AllocatorGenome`s; they compete in a live paper tournament and offline in `evo.eval.allocator`; the active one is switched at runtime (`ctl.risk.limits {set: {allocator}}`).
- **Cross-strategy netting** after allocation and before the limits: exposures (fraction of the sleeve, `mean / (1 + stddev)`) become signed EUR per instrument and are summed; every portfolio position keeps per-strategy virtual sub-books.
- **Pluggable execution algorithms** chosen per intent by horizon and size vs volume (`fa_risk::exec`; the gateway implements market, limit, smart-limit, TWAP, VWAP, POV); realised slippage vs arrival is recorded per fill and recalibrates the impact model.
- **Long and short**, gross leverage capped per asset class; shorts need borrow and margin.
- **Mandatory, non-evolvable limits** from `risk.toml` only (runtime changes may only tighten): drawdown kill switch (portfolio and per strategy), daily loss, position / class / venue / portfolio-gross caps, liquidity cap net of the payout reserve, data / model / broker health halts.
- **Manual overrides**: pause / resume / flatten a strategy, flatten the portfolio, manual orders (through the full pre-trade chain), kill switch and re-arm.
- Base currency for limits and P&L is **EUR**.

### Data model
| Entity | Store |
|---|---|
| Strategy status, sleeve, shadow progress, sub-book, latest gate, pipeline evidence | `ledger.strategy_state` (one row per strategy; transitions in `audit.log`) |
| Halt state machine (halts incl. kill switch, peak and day-start equity, session days, hourly equity, tournament records) | `ledger.risk_state` (restored at start; limits never) |
| Allocation snapshot (allocator + spec, weights, sleeves, equity, deployable, fallback) | `ledger.allocation_snapshot` (migration `20261114070000`) |
| Netted portfolio target per instrument: raw (allocator), capped, gated (halts / feed / FX), the position compared, contributions per strategy, decision (`trade` / `skip:<reason>` / `none`), intent id | `ledger.portfolio_target` (same migration) |
| Promotion reviews (every promote attempt, every gate-outcome change of a shadow strategy, hourly otherwise): checklist with values and thresholds, evidence, statuses before / after, actor | `ledger.promotion_review` (same migration; kept, never pruned) |
| Intents, fills (arrival price, slippage), attribution shares, cash movements (deposit, payout, **funding, borrow**) | `ledger.*` (spec 42, [fa-ledger README](../../crates/fa-ledger/README.md)) |
| Risk events | `audit.log` + `risk.event.<account>` (never deleted) |
| Slippage | per fill in `ledger.fill`; gateway gauges `exec_slippage_bps.<algo>`, `slippage.<venue>.<class>.*` |

Allocation snapshots and targets are kept on change (an order, a weight move > 1e-4, a gated target move > `min_order_eur`) and at least every `[history] snapshot_secs` (60), written every 5 s and pruned after `retention_days` (30); counters `history.kept|dropped|write_errors`.

### Netting cost attribution (§10 Q3, decided: causal pro-rata)
A netted order's fill — quantity, fee, slippage — is split across the intent's contributors by **how far each one's own sub-target is from what its sub-book already holds, in the order's direction**: weight `max(0, sign(Δ)·(c_s − h_s))` with `c_s` the strategy's signed EUR contribution to the instrument's target this tick and `h_s` its sub-book's position value (`fa_allocator::attribution_weights`). A strategy whose target did not move (or moved against the order) gets none of it; sub-books converge to each strategy's own contribution. Fallbacks: no strategy wants the order's direction → contribution shares `|c_s|`; nothing at all → even split (D-047). Weights travel with the intent (in memory) and the booked fill (`FillEvent.strategy_weights` → `PgLedger::record_fill_weighted`, shares summing to the fill exactly, the last contributor taking the rounding remainder). Foreign / recovered intents keep the even split. The turnover budget of the rebalance rule still charges contributors evenly. Why not plain pro-rata to |contribution|: a strategy holding a steady target would pay for another strategy's rebalance, which biases the per-strategy drawdown tracker and the gate's consistency test.

### Daily-loss day (§10 Q4, decided: per calendar group, one portfolio measure)
`fa_risk::session` + `src/engine/days.rs`. Each instrument's day group comes from its exchange calendar (`fa_warehouse::calendar` rule of its current listing's venue, else class + symbol suffix for aggregator venues; loaded at start and with every reference-data reload): 24/7 venues and instruments without a calendar are on the **UTC day**; `us`, `xetr` and `fx` days end at the session close (key = the date of the next close, so overnight and weekends belong to the next session; holidays and early closes from the rule). One loss measure against one limit: `1 − equity / day_start`. A session group's instruments track their day P&L (value at the group's roll + fill cash flows since); when a **session group rolls** its finished day is banked into `day_start`, when the **UTC day rolls** `day_start = equity − Σ running session-day P&L` (24/7 part and book-level cash restart, open exchange days keep counting). The `daily_loss` halt clears after a roll only when the remaining loss is below the limit — a US-session loss halts until the US close, not until UTC midnight. With only 24/7 instruments (today's book) this is exactly the former UTC-day rule. The session days persist in `ledger.risk_state`; gauges `daily_loss.session_instruments`, `daily_loss.rolls`. The `ref.calendar_session` table overrides are not read (rules only).

### Paper funding / borrow (spec 40 §6.5 shorts)
The broker-gateway accrues the `costs.yaml` carry of every open position (`funding_bps_day` on any side, `borrow_bps_day` on shorts — the backtest's rule) and publishes one charge per (instrument, kind, hour) on `exec.carry.<account>` ([gateway README](../../go/services/broker-gateway/README.md#paper-carry-carrygo-spec-40-65-shorts)); this engine books each reference once — Postgres first (`ledger.cash_movement` kind `funding` / `borrow`, converted to EUR at the period end), then the paper book with the identical exact amount; a failed write is not applied (the gateway re-announces 6 periods). Carry is book-level cash (unattributed in the accounting), lowers equity and so counts toward drawdown and daily loss. Counters `carry.booked|duplicates|ignored|persist_errors`, `carry.funding_eur`, `carry.borrow_eur`.

### Hard limits — rules and basis
Rules: §"Pipeline per tick" step 5–6 above; drawdown kill switch = equity vs running peak, manual re-arm; per-strategy drawdown = attributed sub-book vs `strategy_max_drawdown` of the sleeve → flatten + pause; leverage per class `min(leverage_cap.<class>, max_class_frac)`; venue cap (counterparty risk); liquidity `|position| ≤ liquidity_k × horizon volume`; deployable = equity − payout reserve; shorts: class in `limits.short_margin` and `ref.contract_spec.shortable` not false, Σ |short| × margin ≤ `short_margin_budget_frac` × deployable; non-finite targets are no position. **Limit values and their basis (2026-11-11 calibration):** `ctl.risk.limits {calibrate}` suggested `max_position_frac` 8.9 % from the median instrument σ_day 2.4 % — the configured 5 % is tighter and stays (one-day 99 % VaR 0.28 % of equity); the portfolio rules had no basis yet (< 24 hourly returns since the 2026-11-10 reset; the 1 s baselines' σ_day ≈ 7 % would suggest the clamped 5 % / 15 %), so the placeholders 2 % daily loss / 10 % drawdown bind. Every rule-derived value is looser than its placeholder; revisit with ≥ 24 h of portfolio history and ≥ 3 instruments with full hourly history. Calibration rules: `max_daily_loss = 1.5 · z99 · σ_day` (0.5–5 %), `max_drawdown = 3 ×` daily loss (3–25 %), `max_position_frac = 0.5 % / (z99 · σ_day,inst)` (1–20 %).

### Promotion gate (paper shadow → active)
All of: min shadow calendar days (`promotion.min_shadow_days`, 30) and completed horizons (20); deflated Sharpe on the locked holdout ≥ threshold; generation PBO < `max_pbo` (0.2); shadow consistent with the backtest (`consistency_z` 2.0 standard errors, ≥ `decay_min_hours`); no health halts in the window. Unknown blocks; PBO and consistency are `n/a` for hand-written strategies. Manual promotion needs a green gate; evolved strategies are promoted automatically with `auto_promote`; demotion on drawdown, gate decay or runtime auto-pause. Every decision lands in `ledger.promotion_review`.

### Tunables (placeholders unless noted; running values in `risk.toml`)
| Name | Default | Notes |
|---|---|---|
| `max_drawdown` / `strategy_max_drawdown` / `max_daily_loss` | 10 % / 5 % of sleeve / 2 % | kill switch (manual re-arm) / pause + flatten / reduce-only until the day ends |
| `leverage_cap.<class>` / `portfolio_gross_cap` | equity / etf 1.5, fx 3.0, others 1.0 / 2.0 | |
| `max_position_frac` / `max_class_frac` / `max_venue_frac` | 5 % / 40 % / 50 % | provisional instruments: × 0.5 |
| `liquidity_k` | 0.01 | position ≤ k × horizon volume |
| `short_margin.<class>` / `short_margin_budget_frac` | equity / etf 0.5, crypto 1.0, fx 0.05, index 0.1, commodity / derivative 0.2 / 0.5 | |
| `max_order_frac` / `limit_price_band` | 10 % / 10 % | pre-trade checks |
| `feed.stale_freeze_bars` / `jump_max` / `jump_confirm` | 3 / 0.15 / 3 | feed guard |
| `no_trade_band_frac` / `min_order_eur` | 0.5 % / 25 EUR | applied before the cost-aware rule |
| `rebalance.*` | ratio 1.0, prior 20 bp / 600 obs, max edge 200 bp, holding 1 horizon / ≥ 60 s, risk aversion 1.0, max fee 10 bp, turnover 20 × sleeve / day, confirm 3 s | cost-aware rule |
| `promotion.min_shadow_days` / `min_horizons` / `max_pbo` / `consistency_z` | 30 / 20 / 0.2 / 2.0 | gate |
| `promotion.initial_live_sleeve` | 1 % equity | live only (out of scope) |
| `allocator.kelly_fraction` / `tournament_cost_bps` | 0.25 / 5 | |
| `model_drift.halt_kl` / `sustain` / `clear_kl` / `clear_sustain` | 1.0 / 5 / 0.25 / 10 | snapshot-scoped drift halt |
| `history.snapshot_secs` / `retention_days` / `review_unchanged_secs` | 60 / 30 / 3600 | risk-chain history |
| gateway `-risk-heartbeat-timeout` / `-carry-period` | 15 s / 1 h | fail-closed / carry charging |

### Failure modes
| Failure | Response |
|---|---|
| Broker disconnect / gateway restart mid-execution | gateway persists working / resting orders, intent ids and positions (`orders.json`) and resumes; reconciliation (spec 42) before trusting the book |
| Duplicate intent delivery | idempotent by intent id in the gateway (last 50 000) and the ledger |
| Fill for an unknown order | recovered from `ledger.order_intent`, else parked in `ledger.orphan_fill`; an open paper orphan halts `broker` |
| Risk service crash | gateway fail-closed: risk-increasing intents rejected while the heartbeat is older than 15 s |
| Price spike / bad tick | feed guard holds the jump, instrument reduce-only until confirmed |
| Allocator NaN / negative / > 1 weights | previous allocation + risk event (`allocator.fallbacks`, `fallback` in the allocation snapshot) |
| Kill switch | flatten all, halt all, survives restart until manual re-arm |
| Carry charge lost in transit | re-announced for 6 periods; booked once by reference |

### Open questions, answered
1. Payout quota mechanics — `fa_ledger::payout::PayoutRule` (`[payout]`, see above; amount is the user's call).
2. Manual orders bypassing limits — no: they pass the pre-trade checks; reductions always pass.
3. Netting cost attribution — causal pro-rata (above).
4. Daily-loss day — per calendar group, one portfolio measure (above).
5. Shorts — class margin schedule + `ref.contract_spec.shortable`; per-venue borrow rates and locate data need a live adapter.
6. First live allocator — live only, out of scope with the live adapter.

### Acceptance status at retirement
- Randomised risk chain: `crates/fa-risk/tests/risk_chain_props.rs` (proptest, 2 000 cases each: caps hold on random limits / markets / targets incl. NaN and ±inf, halts only reduce, the trade rule never overrides the chain) and `random_runs_never_approve_an_order_beyond_the_limits` (8 seeded 600-tick runs of the whole engine). It found that a NaN target passed `cap_targets` as NaN (fixed: non-finite targets are no position).
- Kill switch survives restarts; tighten-only limits; per-fill attribution and slippage; promotion refuses a red gate; allocator tournament — met (tests named in the sections above).
- **Out of scope (live adapter):** paper/live shared `Venue` integration tests, `trading.mode.live_enabled`, the initial live sleeve, the first live allocator.

### Deferred
All numeric limit values beyond the basis above; venue-specific order types (iceberg, post-only, reduce-only); options / derivatives risk (greeks — derivatives are delta-1 notionals).

## Run
```sh
cargo run -p risk-engine                                   # config/platform.toml, bins/risk-engine/risk.toml
FA_CONFIG=config/platform.docker.toml FA_RISK=/app/bins/risk-engine/risk.toml risk-engine   # in the Rust image
```
Then `GET /api/trading/{signals,orders,positions,risk,strategies,limits,pipeline,allocators,halts,risk-events}` on control-api shows the flow; `POST /api/trading/halt` flattens and stops orders, `POST /api/trading/resume` re-arms; `POST /api/trading/strategies/{id}/pause|resume|flatten|promote|shadow|retire|admit`, `PUT /api/trading/limits`, `POST /api/trading/flatten`, `POST /api/trading/pipeline/run` and `POST /api/trading/halts/{scope}/rearm` drive the control subjects.

## Layout
- `src/engine.rs` — pure, deterministic engine (`on_signal`, `on_trade`, `on_intent_seen`, `on_order_event`, `on_halt`, `on_resume`, `on_source_health`, `strategy_action`, `set_limits`, `flatten_all`, `tick`, `state`, `strategies_view`), tested end-to-end with synthetic signals/fills (lifecycle, shadow books, per-source halts, tighten-only limits).
- `src/engine/` — engine parts with their tests: `days.rs` (daily-loss day per exchange calendar), `carry.rs` (funding / borrow booking), `history.rs` (allocation snapshots, netted targets, promotion reviews), `dryrun.rs` (`ctl.risk.pretrade`), `spec40_tests.rs` (randomised whole-chain run and the tests of these parts).
- `src/risk_io.rs` — their I/O: `exec.carry.<account>` booking, the history writer / pruner, promotion reviews, calendar loading, the `ctl.risk.pretrade` responder.
- `src/settings.rs` — `risk.toml` schema and validation.
- `src/pipeline.rs` — the I/O half of the strategy pipeline and holdout locking (Garage, Postgres, VictoriaMetrics, `JOBS_EVO`, Helix).
- `src/recon.rs` — ledger interface: confirmed / retried fill writes, unknown-intent recovery, orphan halt, reconciliation at one cut.
- `src/decision_replay.rs` (tests) — decision-level replay: `testdata/decision_fixture.jsonl` (a recorded live window) must reproduce the live intents byte for byte (spec 41 §9); `--ignored` recorders for a live window and for the retained `SIGNAL` / `MD` streams.
- `src/persist.rs` — Postgres side effects (`fa-ledger::pg::PgLedger`, `fa-warehouse::audit`, `ledger.strategy_state`, `validation.trial` lookup): intents, fills, snapshots, strategy rows, reconciliation runs, audit entries; best effort, idempotent (DB test gated on `DATABASE_URL`).
- `src/main.rs` — NATS wiring, control subjects, heartbeat `svc.heartbeat.risk-engine` (counters: signals, trades, intents.ours, intents.foreign, approved.foreign, fills, halts.*, resumes, strategies.seen, strategy.actions, strategy.drawdown_pauses, limits.tightened, limits.clipped_targets, flatten.portfolio, allocator.switches, allocator.fallbacks, `allocator.<name>.return`, rejected.halted|mode|pretrade, risk_events, feed.frozen, feed.jump_*, `rebalance.trades|bypass|skip.*|expected_cost_eur|avoided_cost_eur|frozen_strategies`, `exec.algo.<name>`, halts.model_snapshots, model_halt.paused_strategies, pipeline.candidates, pipeline.transitions, holdout.missing, helix.mirror_*, risk_state.restored, ledger.open_orphans, fills.unconfirmed, `headroom.*`, strategies.restored, refdata.loads, refdata.instruments, refdata.unknown, fx.rates, fx.stale, fx.stale_now, fx.ecb_fetches, fx.ecb_errors, payout_reserve_eur, equity_eur, cash_eur, gross_exposure_eur, `exposure_eur.venue.<code>`, positions, halted).

## Governing specs
- Retired spec 40 (Execution & risk): [lasting content below](#execution-and-risk-lasting-content-of-the-retired-spec-40), decisions in [`specs/DECISIONS.md`](../../specs/DECISIONS.md)
- [fa-ledger README](../../crates/fa-ledger/README.md) (retired spec 42, Ledger & accounting)
- [21-bus-and-contracts](../../specs/21-bus-and-contracts.md)
