# 40 — Execution & Risk

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium-high |
| **Owners (code)** | `crates/fa-allocator`, `crates/fa-risk`, `crates/fa-ledger` (paper book), `bins/risk-engine` (allocation, limits, halts, intents; D-040..D-042), `go/services/broker-gateway`, `bins/strategy-runtime` (signal emission) |
| **Depends on** | 21 Bus & contracts, 35 Strategies, 41 Backtest & validation, 42 Ledger & accounting, 12 Reference data |

## 1. Purpose
Turn the exposure distributions produced by promoted strategies into real (or paper) orders, without ever letting the evolutionary parts of the system breach a fixed set of hard risk limits. This layer is the boundary between "research that may be wrong" and "capital that can be lost": it is deliberately **not evolvable** at the limit level, while allocation and execution algorithms are pluggable and compared in paper mode.

## 2. Scope
**In scope:** meta-allocation across strategies, cross-strategy netting, hard risk limits, kill switch, order intent → venue order routing, execution algorithms, paper vs live modes, promotion gate (paper → live), manual overrides from the UI.
**Out of scope:** strategy generation and fitness (35), backtest mechanics and statistical tests (41), ledger persistence and reconciliation (42), broker connectivity details per venue (11 / connector specs).

## 3. Decisions
- **Paper first, gated promotion.** Every strategy starts in paper mode; live allocation only after passing the promotion gate.
- **Meta-allocator is pluggable.** An `Allocator` trait with several implementations (Bayesian bandit + risk parity, fractional Kelly, evolved allocator, …) is compared in paper mode; one is selected as *active* for live capital.
- **Cross-strategy netting** happens after allocation and before risk checks: strategies' target exposures on the same instrument are summed into one portfolio target.
- **Execution algorithms are pluggable**: market, limit, smart-limit, TWAP, VWAP, selected by strategy horizon and instrument liquidity. Realised slippage is tracked and fed back into fitness (35/41).
- **Long + short allowed**, with a **capped gross leverage per asset class** (config).
- **Mandatory, non-evolvable limits:**
  1. Max-drawdown kill switch — portfolio level and per strategy.
  2. Daily loss limit — portfolio stops opening risk for the rest of the trading day.
  3. Position / concentration caps — per asset, per asset class, per venue.
  4. Data / model / broker health halts — stale feeds, model drift alarms, broker errors halt affected strategies.
  5. **Liquidity cap net of payouts** — position ≤ k × traded volume over the strategy's horizon, and deployable capital = equity − reserved daily payout quota.
- **Promotion gate = minimum shadow calendar time + statistical criteria** (deflated Sharpe on holdout, PBO threshold, shadow-vs-backtest consistency; see 41). Defaults are tunable.
- **Manual overrides** from the UI: pause/resume strategy, flatten strategy / instrument / portfolio, manual order, trigger kill switch.
- Base currency for all limits and P&L is **EUR**.

## 4. Interfaces

### 4.1 Inputs
- `signal.exposure.<strategy_id>` (NATS, proto `fa.v1.ExposureDistribution`): per instrument, mean target exposure in [-1, 1] of the strategy's sleeve plus uncertainty (std or quantiles), timescale, horizon, snapshot id.
- Strategy registry (Helix + Postgres): status (`paper`, `shadow`, `live`, `paused`, `retired`), sleeve, asset set, horizon.
- Market state (VictoriaMetrics / bus): last prices, rolling traded volume per horizon, spreads.
- Ledger (42): positions, cash, equity, realised/unrealised P&L, payout quota.
- Health signals: `ops.health.*` (feed staleness, model drift, broker status).

**Implemented inputs (risk-engine):** `signal.exposure.>`; lifecycle status from `ledger.strategy_state` (`active|paused|flattening|shadow|candidate|holdout_passed|retired`, §6.6); `md.trade.>` (marks, traded volume, σ, per-instrument feed guard); `order.*.<account>` of its own account only (D-072); reference data and FX from Postgres (D-061); `exec.costs.calibrated` (impact `k` for the rebalance rule); health from `ingest.health.<source>`, `risk.halt.>` / `risk.resume.>` and the drift of `latent.mcmc.*` kernel updates.

### 4.2 Outputs
- `order.intent.<venue>` (proto `fa.v1.OrderIntent`): instrument, side, quantity, exec algo + params, parent portfolio target id, idempotency key.
- `order.event.<venue>` (from broker-gateway): acknowledgements, fills, cancels, rejects → ledger (42).
- `risk.event` (proto `fa.v1.RiskEvent`): limit breaches, halts, kill-switch activations.

**Implemented outputs:** `order.intent.<account>` + `order.approved.<account>` (account `paper`; venue routing is the gateway's, by instrument), `order.event.<account>` from the gateway, `risk.event.<account>` (`fa.v1.RiskEvent`, §6.5), `risk.halt.<scope>` / `risk.resume.<scope>`, `risk.state`.

### 4.3 Traits (Rust, sketch)
```rust
pub trait Allocator: Send + Sync {
    fn id(&self) -> AllocatorId;
    /// Capital weights per strategy given their posterior performance and exposure distributions.
    fn allocate(&mut self, ctx: &AllocationContext) -> Allocation;
    fn on_outcome(&mut self, outcome: &StrategyOutcome); // update beliefs
}

pub trait RiskLimit: Send + Sync {
    fn id(&self) -> &'static str;
    /// Clip or reject a proposed portfolio target. Never evolvable; configured from config only.
    fn check(&self, target: &PortfolioTarget, state: &RiskState) -> LimitVerdict;
}

pub enum LimitVerdict { Pass, Clip(PortfolioTarget), Halt { scope: HaltScope, reason: String } }
```
```go
// go/services/broker-gateway
type ExecAlgo interface {
    Name() string
    // Plan turns an intent into child orders over time; Next is driven by market events.
    Start(ctx context.Context, intent *pb.OrderIntent, venue Venue) (Execution, error)
}
type Venue interface {
    Submit(ctx context.Context, o ChildOrder) (VenueOrderID, error)
    Cancel(ctx context.Context, id VenueOrderID) error
    Events() <-chan VenueEvent
    Mode() Mode // Paper | Live
}
```

### 4.4 Control (via 50 Control API)
`POST /api/trading/strategies/{id}/pause|resume|flatten`, `POST /api/trading/flatten`, `POST /api/trading/kill-switch`, `POST /api/trading/orders` (manual), `POST /api/trading/strategies/{id}/promote` (only succeeds if gate passes, see §6.6).

## 5. Data model
| Entity | Store | Notes |
|---|---|---|
| Strategy sleeve & status | Postgres (`strategy_allocation`), Helix (lineage) | status transitions are audited |
| Allocation snapshot | Postgres | per allocator run: weights per strategy, allocator id |
| Portfolio target | Postgres + NATS | netted target per instrument, pre/post-risk |
| Order intents, child orders, fills | Postgres ledger (42) | idempotency key = `(portfolio_target_id, instrument_id, seq)` |
| Risk state | in-memory in `fa-risk`, persisted to Postgres on change | drawdown peaks, daily P&L, halts |
| Risk events | Postgres audit log + `risk.event` | never deleted |
| Slippage measurements | VictoriaMetrics (`exec_slippage_bps{venue,algo,instrument}`) + Postgres per fill | feeds fitness |
| Promotion gate results | Postgres (`promotion_review`) | inputs, statistics, pass/fail |

**Implemented stores:** strategy status, sleeve, shadow progress, sub-book, gate checklist and pipeline evidence in `ledger.strategy_state` (one row per strategy; `gate` JSONB instead of a `promotion_review` history, transitions in `audit.log`); halts, drawdown peak, day-start equity, hourly equity and the live allocator tournament in `ledger.risk_state`; intents / fills / attribution in `ledger.*` (42) with `arrival_price` and `slippage_bps` per fill; slippage gauges in the gateway heartbeat (`exec_slippage_bps.<algo>`, `slippage.<venue>.<class>.*`); risk events in `audit.log` + `risk.event.paper`. Allocation snapshots and netted portfolio targets are not persisted (the intent carries its contributing `strategy_ids`).

## 6. Behaviour & algorithms

### 6.1 Pipeline per decision tick
1. Collect latest `ExposureSignal` per active strategy (paper + live sleeves processed separately).
2. **Allocate**: active allocator produces capital weights per strategy within the sleeve's mode.
3. **Scale**: strategy exposure (fraction of its sleeve) × allocated capital → EUR target per instrument.
4. **Net** across strategies → portfolio target per instrument.
5. **Risk chain**: apply every `RiskLimit` in fixed order (health → kill switch → daily loss → leverage → concentration → liquidity/payout). Any `Halt` stops the chain for its scope.
6. **Diff** target vs current position (ledger) → order intents, with a minimum trade size / no-trade band to avoid churn.
7. Choose exec algo from horizon + liquidity (§6.4) → publish `order.intent.<venue>`.

Decision ticks are driven per timescale: a strategy on the 1h scale only re-targets on its own cadence; netting is recomputed whenever any constituent changes.

**Cost-aware rebalancing (implemented, step 6, `fa_risk::rebalance`, `risk.toml [rebalance]`).** The D-041 band (0.5 % / 25 EUR) sat far below the paper venue's 1 EUR fee minimum: between 2026-11-10 19:00 and 11-11 02:00 the 1 s baselines rebalanced ~7 000 times at ~300 EUR (30–50 bp each, ~7 000 EUR of fees; `ledger.fill`). A non-urgent order is now sent only when
1. its **expected benefit** ≥ `min_benefit_cost_ratio` (1.0) × its **full expected cost**. Cost = `max(commission_min, |Δ|·commission_bps) + |Δ|·half_spread + |Δ|·k·σ_day·sqrt(|Δ|/ADV)` from the paper venue's schedule for the instrument's (md venue, `costs.yaml` class) — risk-engine loads `costs.yaml` (`FA_COSTS`), takes `k` from `exec.costs.calibrated`, σ_day from its own EWMA of the tape (the venue's estimator) and ADV from the volume window. Benefit = edge × |Δ| for a trade with the strategies' view (opening, adding, or flipping a position that is against it); for a reduction toward a smaller target on the same side, `(risk_aversion · σ_hold − edge) × |Δ|` with `σ_hold = σ_day·sqrt(max(holding, min_holding_secs)/1 d)`; a position nobody targets any more (orphan) is held for a day of σ. The edge per instrument is the contribution-weighted (signed) mean of the contributors' edges; a strategy's edge is its tracked step return per unit exposure scaled to `holding_horizons` × its horizon, shrunk toward `prior_edge_bps` (20) with `prior_obs` (600) pseudo-observations against the EWMA's effective sample (≤ 39 at decay 0.95) and clamped to ±`max_edge_bps` — a losing strategy opens nothing it has to pay for;
2. `|Δ|` ≥ the **min order notional** `max(min_order_eur, commission_min · 1e4 / max_fee_bps)` (1 000 EUR at a 1 EUR minimum and 10 bp; 25 EUR on fee-minimum-free venues such as Kraken or `sim`);
3. it stayed worthwhile in the same direction for `confirm_secs` (3) — **batching**: strategies whose targets swing against each other within seconds net out in the portfolio target (§6.3) before anything is sent;
4. not every contributing strategy has spent its **turnover budget**: each strategy may cause `turnover_budget_per_day` (20) × its sleeve of traded notional per UTC day (orders attributed evenly to contributors, as fills are); a strategy over budget is netted with its last distribution (frozen sub-target) until the day rolls.
Urgent orders (any halt, strategy flatten, stale-feed flatten) and **limit-bound** reductions (the cap or gate chain cut the target below the position) bypass the cost test — limits are never traded off against fees. Heartbeat counters `rebalance.trades|bypass|skip.{band,min_notional,cost,confirming,budget}|expected_cost_eur|avoided_cost_eur|frozen_strategies`. The same `Rebalancer` runs in `fa_backtest::allocator_replay` and `fa_backtest::replay` (`TradeRule::CostAware`) so replays measure the live rule; evo-worker's fitness jobs (`evo.eval.strategy`, `.holdout`, `evo.eval.allocator`) parse it from this `risk.toml` (`FA_RISK`, built-in copy as fallback; `FA_EVO_TRADE_RULE=fraction` reproduces the old fitness rule), so selection is scored on the trades this engine would send. **Evidence:** `cost_aware_rebalancing_suppresses_the_recorded_churn_pattern` (engine, synthetic replay of the pattern: band 1 718 orders / 1 727 EUR fees at 346 EUR average → 3 orders / 9 EUR, a halt still flattens on the next tick) and `the_churn_pattern_is_suppressed` (`fa_risk`); replaying the recorded `SIGNAL` stream with VictoriaMetrics marks through allocation → netting → rule → the then-current `costs.yaml` (`cargo run -p fa-backtest --example replay_rebalance -- <start> <end> <costs>`), 2026-11-10 19:48–22:00: band 26 294 orders, 20 643 EUR fees (243 EUR average), net −21 325 EUR; cost-aware 837 orders, 17 EUR fees (953 EUR average), net +165 EUR; 11-11 01:00–02:00: band 2 554 orders, 1 640 EUR fees; cost-aware 127 orders, 3.2 EUR (with today's crypto pricing: band 1 008 EUR, cost-aware 2.0 EUR). The allocator replay applies no position / liquidity caps or halts, so it over-trades the live book (~6 000 fills, ~6 100 EUR in those hours) about 4×; the relative reduction (> 99 % of fees) is the estimate for the live book. The ticks of the deployed engine sit on the 1 s wall-clock grid so decisions can be replayed (spec 41 §9).

### 6.2 Allocators (initial set)
- **Bayesian bandit + risk parity (proposal default for paper)**: Thompson sampling over each strategy's posterior return distribution; resulting weights capped by a risk-parity budget (inverse volatility, covariance-aware). Can consume MCMC posteriors directly.
- **Fractional Kelly**: covariance-aware Kelly weights from shrunk return estimates, multiplied by a fraction (placeholder 0.25).
- **Evolved allocator**: genome-defined allocation policy evaluated under the same Pareto fitness as strategies (35), in paper only until promoted.
- **Equal-risk baseline**: always running in paper as a control.
All allocators run in parallel in paper ("allocator tournament"); their portfolio P&L is compared on the same strategy signals.

**Implemented today (D-061, D-064, D-071):** `equal_weight` (the configured default in `risk.toml`), `risk_parity`, `fractional_kelly`, the parametric `weighted_blend` and the Bayesian bandit `thompson_rp` (`fa_allocator::ThompsonRiskParity`: one Thompson draw per strategy from `N(mean, vol²/n)`, positive draws share capital by inverse volatility, capped; deterministic per seed and call sequence), all expressible as an `AllocatorGenome` (`alloc:name=…;cap=…;…`). Allocators compete in a tournament (`evo.eval.allocator` replays recorded signals through allocation, netting and the cost model); the winning front is written to `fa-artifacts/allocators/<generation>.toml`, and risk-engine switches allocator at runtime via `ctl.risk.limits {set: {allocator: {name, params}}}` (audited, not persisted). Caps and concentration limits use per-instrument reference data (class, venue, provisional status).

**Live paper tournament (implemented):** risk-engine runs `equal_weight` (control), `risk_parity`, `fractional_kelly`, `thompson_rp` and the active allocator side by side every tick (`fa_allocator::tournament`): each allocates on the same strategy posteriors, its weights earn the same per-tick strategy sleeve returns, weight turnover is charged `tournament_cost_bps` (5). Reported per entry (cumulative return, Sharpe, max drawdown, turnover, costs, weights) by `ctl.risk.allocators`, in `GET /api/trading/strategies` (`allocator_tournament`) and as heartbeat counters `allocator.<name>.return`; records persist with the risk state. The offline tournament (`evo.eval.allocator`) remains the exact one (netting and the venue cost model). **Allocator failure:** NaN / negative weights or a sum > 1 fall back to the previous allocation and raise a risk event (§8).

### 6.3 Netting
Exposures are converted to signed EUR notionals and summed per instrument. Attribution is kept: every portfolio position carries per-strategy virtual sub-positions so strategy P&L and fitness remain computable even though orders are netted. Fees/slippage of netted orders are attributed pro-rata to the strategies whose deltas caused the trade **(proposal)**. **Implemented:** every fill is split **evenly** across the intent's contributing strategies (`ledger.fill_attribution`, D-047; exact decimal shares, D-078) and booked into their sub-books; a pro-rata or causal rule is still open (§10 Q3).

### 6.4 Execution algorithms
| Algo | Use when (proposal) |
|---|---|
| Market | urgent (kill switch, flatten), very liquid, small vs volume |
| Limit | patient, long horizon, spread wide relative to expected edge |
| Smart-limit | default: passive limit with timeout, then reprice/cross |
| TWAP | order large vs per-interval volume, horizon ≥ hours |
| VWAP | as TWAP but volume-profile weighted (needs intraday volume curve) |
Selection inputs: strategy horizon, order notional / horizon volume, spread, venue capabilities (from catalog 11). Every child fill records arrival price for slippage measurement.

**Crypto pricing:** `costs.yaml` maps the Kraken XBTEUR / ETHEUR / SOLEUR and Yahoo BTC-USD instruments to class `crypto` (they fell back to `equity` and its 1 EUR minimum before) and has a `kraken` venue (Kraken Pro lowest tier: 40 bp taker, no minimum, 2 bp modelled spread, `default` = crypto for unmapped pairs).

**Implemented today (D-051):** the paper venue (`go/services/broker-gateway`) prices every fill with `costs.yaml` (commission bps + minimum, half spread from `md.quote.>` or a modelled `spread_bps`, square-root impact `k·σ_day·sqrt(notional/ADV)` estimated from the tape); `OrderEvent` carries `arrival_price` and signed `slippage_bps`. Limit orders rest as `ACK` until crossed and fill passively at the limit; cancellation used a provisional `order.cancel.<account>` `OrderIntent` payload (now `fa.v1.OrderCancel`, below). Since then every §6.4 algorithm exists (below).

**Execution algorithms (implemented, paper venue):** `OrderIntent.exec_algo` is `name` or `name:k=v,…` — `market`, `limit`, `smart_limit` (`timeout_s` 30, `offset_bps` 0: passive at the touch, crossed at the timeout), `twap` (`dur_s` 300, `slices` 10), `vwap` (same, child sizes from the instrument's time-of-day volume profile learned from the tape, uniform when unknown), `pov` (`rate` 0.1 of observed traded size, remainder crossed at `max_dur_s` 900) and `auto` (gateway fallback: participation = notional / ADV ≤ 0.1 % and spread ≤ 10 bps → market, wider spread → smart_limit, ≤ 1 % → twap, else pov); unknown names or parameters are `REJECTED`. Algo orders are `ACK`ed, children go out as `PARTIAL_FILL` (`venue_order_id = paper-<intent>-<n>`), the last as `FILL`; every child carries the **parent** arrival price and its implementation shortfall in `slippage_bps`. Cancellation is the proto `fa.v1.OrderCancel{intent_id, account_id, reason, at, actor}` on `order.cancel.<account>` (a legacy `OrderIntent` payload still decodes by intent id) and cancels a resting limit or the remainder of a working algo order. **Live slippage recalibration:** per (venue, class) the gateway regresses realised-minus-half-spread bps on `1e4·σ·sqrt(N/ADV)` over a rolling window of informative taker fills, shrinks `k̂` toward the `costs.yaml` prior (`n/(n+50)`), clamps to `[0.5, 2] ×` prior, and every 15 min (≥ 30 samples) writes `costs.calibrated.yaml` + report (state volume), applies it in memory and publishes the report on `exec.costs.calibrated`. In paper, fills priced at the arrival itself are the model echoing itself and are excluded; only algo children (market drift over the schedule) inform the fit, so paper calibration measures drift, not true impact — live fills would use the same path. Gauges `slippage.<venue>.<class>.bps_mean|model_bps_mean|n`, `exec_slippage_bps.<algo>`, `costs.k.<venue>.<class>`, `algo.<name>.working|done|cancelled` (spec 60 §6.5). Details: `go/services/broker-gateway/README.md`.

**Algorithm choice in the risk chain (implemented, `fa_risk::exec`, `risk.toml [exec]`):** every intent the risk-engine emits names its algorithm (§6.1 step 7), first match wins: urgent — any halt in force, an instrument of a strategy being flattened, a stale-feed flatten — → `market`; shortest contributing horizon < `market_below_horizon_secs` (60) → `market` (the edge decays faster than a passive order waits); participation = order notional / (window volume scaled to the horizon) ≥ `pov_participation` (5 %) → `pov:rate=0.1,max_dur_s=min(horizon, 3600)`; ≥ `twap_participation` (1 %) with horizon ≥ `sliced_min_horizon_secs` (600) → `vwap` (instrument seen ≥ 1 day) else `twap`, `dur_s = min(horizon/2, 3600)`, 30 s slices (2–20); notional < `market_below_eur` (250) → `market`; otherwise `smart_limit:timeout_s=clamp(horizon/4, 5, 30)`. Unknown volume never slices. The intent stays pending for `pending_ttl_secs` beyond the algorithm's working time; `PARTIAL_FILL`s shrink the pending remainder instead of clearing it, so a working order is never re-issued. Heartbeat `exec.algo.<name>` counts the choices. The 1 s baselines (horizons of seconds) keep trading `market`; slower strategies use the working algorithms.

### 6.5 Hard limits — details
- **Drawdown kill switch**: portfolio equity vs running peak (EUR); breach → flatten all live positions with market/smart-limit-aggressive, halt all live strategies, require manual re-arm. Per-strategy drawdown → flatten + pause that strategy.
- **Daily loss limit**: realised + unrealised P&L since session start (UTC day for 24/7 venues, exchange day otherwise) ≤ −X% equity → no new risk-increasing orders until next day; reducing orders allowed.
- **Leverage**: gross exposure / equity ≤ cap per asset class; also a portfolio-wide gross cap.
- **Concentration**: |position| / equity ≤ cap per instrument, per asset class, per venue (venue cap limits counterparty risk).
- **Health halts**: feed stale > threshold for an instrument → strategies using it frozen (no new orders, existing positions kept unless stale exceeds second threshold); model drift alarm on a snapshot → strategies depending on that snapshot paused; broker gateway errors / disconnected → venue halted, intents queued then dropped after TTL.
- **Shorts** (paper answer to §10 Q5): a short needs borrow — the instrument's class must be in `limits.short_margin` and `ref.contract_spec.shortable` must not be false — and margin room: Σ |short| × class margin ≤ `short_margin_budget_frac` × deployable.
- **Liquidity cap net of payouts**: `|position_i| ≤ k × volume_i(horizon)`, and `Σ capital deployed ≤ equity − payout_quota_reserved_today` (payout quota from ledger 42).
Limits are loaded from `bins/risk-engine/risk.toml` at startup and can only be tightened at runtime from the UI; loosening requires config change + restart (implemented, `RiskConfig::check_tightens`).

**Implemented today (D-041, D-051):** position, liquidity and leverage caps, concentration caps per asset class and venue, a portfolio gross cap, per-strategy drawdown (attributed sub-books, auto-pause at `strategy_max_drawdown` of the sleeve), payout-quota netting, and the halt state machine with scopes `portfolio`, `daily_loss`, `source:<id>`, `model` (and `model:<snapshot>`, below), `broker` (raised on a reconciliation diff or an open paper orphan fill, D-072) and the reduce-only `fx` (stale accounting FX, D-061). Runtime limit changes go through `ctl.risk.limits` and may only tighten (`RiskConfig::check_tightens`).

**Added (implemented):**
- **Restart-safe halts:** the halt state machine (peak and day-start equity, every active halt with reason / since / manual, the UTC day) is persisted in `ledger.risk_state` (migration `20261111000051`) on every change, every `snapshot_secs` and at shutdown, and restored after the book at start — the kill switch survives a restart until manually re-armed (§8). Limits are never restored from it; they always come from `risk.toml`.
- **Per-instrument feed guard** (`fa_risk::health`, `[feed]`): an instrument without a trade for `stale_freeze_bars` (3) × max(fastest strategy bar trading it, its own EWMA trade gap) — at least `stale_freeze_min_secs` — is frozen **reduce-only** (no new or flipped risk; halts still flatten through it); `stale_flatten_secs` (default 0 = never) closes it. A trade more than `jump_max` (0.15 log return) from the mark is held back as a possible bad tick (instrument frozen) until `jump_confirm` (3) trades confirm the level; a trade back near the mark discards it.
- **Pre-trade checks** for orders outside the allocator's chain (manual UI orders, any foreign `order.intent.*`; `fa_risk::pretrade`): finite quantity, known mark, limit price within `limit_price_band` (10 %) of the mark, notional ≤ `max_order_frac` (10 %) × deployable, post-trade position ≤ position cap (provisional cap) and liquidity cap, post-trade class / venue / portfolio gross ≤ their caps, frozen feeds reduce-only. Reductions always pass. This answers §10 Q2 for paper: manual orders do **not** bypass limits (an emergency hedge must fit them).
- **Risk events:** limit clips (target cut by more than `min_order_eur`), pre-trade rejections, feed freezes / held jumps / stale flattens and allocator fallbacks are published as `fa.v1.RiskEvent{LIMIT_BREACH}` on `risk.event.paper` (`scope` = event kind, `reason` = kind + JSON detail, numeric detail as `metrics`), at most once per minute per (kind, instrument), and appended to `audit.log` as `risk.event.<kind>`. Halts, resumes and rejections are mirrored as `HALT_RAISED|HALT_CLEARED|INTENT_REJECTED`.
- **Snapshot-scoped model-drift halts (implemented):** `risk.halt.model:<snapshot_id>` no longer stops everything: only strategies whose timescale that snapshot serves (learned from `latent.mcmc.*` kernel updates, else the id's timescale suffix; an unknown snapshot pauses every strategy, conservatively) are left out of allocation while it is active, so their sleeves are netted away through the normal chain; other strategies keep trading. A bare `model` halt stays global. The risk-engine raises the scoped halt itself when a snapshot's region occupancy drift (`KernelUpdate.drift_kl`) stays ≥ `[model_drift] halt_kl` (1.0, ten times the adapter-refit trigger of spec 33) for `sustain` (5) kernel updates, and clears **only that self-raised** halt after `clear_sustain` (10) updates of every expert at ≤ `clear_kl` (0.25); operator or bus halts need `risk.resume.model:<snapshot>`. Snapshot halts persist in `ledger.risk_state`, appear in `risk.state` as `model:<snapshot>`, count as health halts for the promotion gate (item 5) and as heartbeat `halts.model_snapshots` / `model_halt.paused_strategies`.
- **Margin / borrow for paper shorts (implemented):** `fa_risk::RiskConfig.short_margin` (initial margin per class: equity / ETF 50 % Reg-T style, crypto 100 % fully collateralised, FX 5 %, index 10 %, commodity / derivative 20 %; rates and prediction markets absent = not shortable) and `short_margin_budget_frac` (0.5). The cap chain clips a short target to 0 when the instrument cannot be borrowed (class absent, or `ref.contract_spec.shortable = false`, reloaded with the reference data) and scales all shorts down together when their margin exceeds the budget; pre-trade checks reject a foreign / manual order that opens or adds to such a short (covering always passes). Tighten-only covers it: a lower margin, a newly shortable class or a larger budget is a loosening. Borrow *cost* is charged by the backtest (`borrow_bps_day`), not yet accrued on the paper book.
- **Limit headroom gauges (implemented, spec 60 §6.5):** heartbeat `headroom.drawdown`, `headroom.daily_loss` (fractions), `headroom.gross_eur`, `headroom.short_margin_eur`, `headroom.position_min_eur` (tightest instrument against min(position, liquidity cap)), `headroom.class.<class>_eur`, `headroom.venue.<venue>_eur`; negative = the clip or halt is acting.
- **Limit calibration rules** (`fa_risk::calibrate`, §11): from hourly equity and instrument marks the engine suggests `max_daily_loss = 1.5 · z99 · σ_day` (0.5–5 %), `max_drawdown = 3 × daily loss` (3–25 %), `max_position_frac = 0.5 % / (z99 · σ_day,inst)` at the median instrument (1–20 %), with ≥ 24 hourly points. `ctl.risk.limits {calibrate: "{}"}` reports suggestion and the tighten-only set; `{calibrate: "{\"apply\": true}"}` applies only the stricter values (audited `limits.calibrated`). Loosening still needs `risk.toml` + restart. The per-instrument history is seeded at start from VictoriaMetrics hourly closes of the last 30 days, converted to EUR as of each close (a USD series meeting live EUR marks would fake an FX jump).

**Limit values and their basis (2026-11-11 calibration on the stack's data):** `ctl.risk.limits {calibrate}` on the running engine: instruments with ≥ 24 hourly closes (28 h of VictoriaMetrics history: two Yahoo instruments at σ_day 2.4 % and 0.4 %) give a median σ_day of 2.4 % → `max_position_frac` suggestion 0.005 / (2.326 × 0.024) = **8.9 %**; the configured **5 %** is tighter and stays (its one-day 99 % VaR is 0.28 % of equity). The portfolio rules have no basis yet: the paper book holds 8 h of equity since the 2026-11-10 19:48 reset (< 24 hourly returns); its 5-minute equity returns scale to σ_day ≈ 7 % (the 1 s baselines lost 7 % in hours), which would suggest the clamped maxima (daily loss 5 %, drawdown 15 %) — looser than the configured **2 % / 10 %**, so those stay. Every rule-derived value is therefore looser than the placeholder it would replace and the placeholders are the binding, defensible defaults (tighten-only: runtime may only lower them, `apply` changes nothing today); revisit when ≥ 24 h of portfolio history and ≥ 3 instruments with full hourly history exist. A calibration with the FX bug above had suggested 1.9 % from a spurious 11 % σ — the conversion fix matters.

### 6.6 Promotion gate (paper → live)
A strategy is promotable when **all** hold:
1. Minimum shadow calendar time in live paper (placeholder 30 days) **and** a minimum number of completed horizons (placeholder 20).
2. Deflated Sharpe on locked holdout above threshold (41).
3. PBO below threshold (41).
4. Shadow performance statistically consistent with backtest (e.g. shadow Sharpe within the backtest's confidence band; slippage within modelled band).
5. No unresolved health halts in the shadow window.
Promotion is executed manually from the UI (gate must be green); initial live sleeve is small (placeholder 1% of equity) and grows only via the allocator. Demotion is automatic on per-strategy drawdown breach or gate-statistics decay.

**Implemented today (D-051):** lifecycle `active|paused|flattening|shadow` per strategy in `ledger.strategy_state`, controlled by `ctl.risk.strategy` (`pause|resume|flatten|promote|shadow`) and `POST /api/trading/strategies/{id}/{action}`. Promotion is shadow → active, since no live sleeve exists (D-029). `new_strategy_status` defaults to `active` so the baseline keeps trading.

**Gate since the strategy pipeline:** every item is blocking and unknown blocks: min shadow days, min horizons, deflated Sharpe (the strategy's holdout trial, else the registry DSR of a generation it won), PBO of its generation < `promotion.max_pbo` (0.2), shadow-vs-backtest consistency (spec 41 §6.6, `promotion.consistency_z`), no health halts. For hand-written strategies (no search, no backtest) PBO and consistency are `n/a` and pass. Evolved strategies arrive as `candidate → holdout_passed → shadow` and are promoted automatically on a green gate (`[pipeline] auto_promote`), demoted on drawdown / gate decay / runtime auto-pause and retired after `max_demotions` (spec 35 §6.4.1). `ctl.risk.strategy` also accepts `retire` and `admit`; statuses `candidate|holdout_passed|retired` are persisted (migration `20261111000050`). `POST /api/trading/strategies/{id}/retire|admit` pass them through (control-api whitelist).

### 6.7 Modes
- **Paper**: broker-gateway `Venue` implementation simulates fills against live quotes using the same cost/slippage model as backtest (41). Shares code paths with live except the final `Submit`.
- **Live**: real venue adapter. A global `trading.mode.live_enabled` switch (config + UI) must be on, otherwise all live intents are rejected.
- **Shadow** is a status, not a mode: a strategy in paper whose clock counts toward promotion.

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `risk.portfolio_max_drawdown` | 10% | kill switch, manual re-arm |
| `risk.strategy_max_drawdown` | 5% of sleeve | pause + flatten strategy |
| `risk.daily_loss_limit` | 2% equity | |
| `risk.gross_leverage_cap.<class>` | equities 1.5, fx 3.0, crypto 1.0, other 1.0 | |
| `risk.portfolio_gross_cap` | 2.0 | |
| `risk.max_position_pct` | 5% equity | per instrument |
| `risk.max_class_pct` / `max_venue_pct` | 40% / 50% | |
| `risk.liquidity_k` | 0.01 | position ≤ k × horizon volume |
| `risk.feed_stale_freeze` | 3 × bar interval | per timescale |
| `exec.no_trade_band` | 0.5% of target | churn control (still applied first; the cost-aware rule decides above it) |
| `rebalance.min_benefit_cost_ratio` / `prior_edge_bps` / `prior_obs` / `max_edge_bps` | 1.0 / 20 / 600 / 200 | cost-aware rebalance (§6.1) |
| `rebalance.holding_horizons` / `min_holding_secs` / `risk_aversion` | 1 / 60 / 1.0 | benefit of openings / reductions |
| `rebalance.max_fee_bps` / `turnover_budget_per_day` / `confirm_secs` | 10 / 20 × sleeve / 3 s | min order notional, budget, batching |
| `gateway.risk_heartbeat_timeout` | 15 s | fail-closed (§8) |
| `exec.smart_limit_timeout` | 30 s (scaled by timescale) | |
| `allocator.kelly_fraction` | 0.25 | |
| `risk.max_order_frac` / `risk.limit_price_band` | 10 % / 10 % | pre-trade checks for manual / foreign orders |
| `feed.stale_freeze_bars` / `feed.jump_max` / `feed.jump_confirm` | 3 / 0.15 / 3 | feed guard |
| `promotion.max_pbo` / `promotion.consistency_z` | 0.2 / 2.0 | gate |
| `tournament_cost_bps` | 5 | live allocator tournament |
| `exec.*` | see §6.4 | algorithm choice (`[exec]`) |
| `model_drift.halt_kl` / `sustain` / `clear_kl` / `clear_sustain` | 1.0 / 5 / 0.25 / 10 | snapshot-scoped drift halt |
| `limits.short_margin.<class>` / `short_margin_budget_frac` | equity 0.5, etf 0.5, crypto 1.0, fx 0.05, index 0.1, commodity 0.2, derivative 0.2 / 0.5 | shorts |
| `promotion.min_shadow_days` | 30 | calendar days |
| `promotion.min_horizons` | 20 | |
| `promotion.initial_live_sleeve` | 1% equity | |

## 8. Failure modes
| Failure | Response |
|---|---|
| Broker disconnect mid-execution | cancel-on-disconnect where supported; on reconnect reconcile (42) before any new intent |
| Duplicate intent delivery (NATS redelivery) | idempotency key dedup in gateway and ledger. **Implemented:** the gateway remembers the last 50 000 intent ids across restarts; ledger writes are idempotent on intent / fill id |
| Fill for unknown order | ledger flags anomaly, venue halted, manual review. **Implemented (D-072):** risk-engine recovers the intent from `ledger.order_intent`, else parks the fill in `ledger.orphan_fill`; an open paper orphan raises `broker` until resolved (`POST /api/trading/orphans/{id}/resolve`) and resumed |
| Risk service crash | broker-gateway rejects all new risk-increasing intents if no risk heartbeat within T (fail-closed). **Implemented:** the gateway tracks `svc.heartbeat.risk-engine`; while the last one is older than `-risk-heartbeat-timeout` (15 s, risk-engine beats every 2 s) or none was seen since the gateway started, an intent that moves the account's projected position in the instrument (filled + working + resting remainders, per account, persisted) away from zero or through it is `REJECTED` (`fail-closed: …`, counter `failclosed.rejected`, gauges `risk_heartbeat_age_s` / `risk_heartbeat_alive`); reductions and closes execute, working algo children continue. |
| Gateway restart mid-execution | **Implemented:** working algo orders (schedule, filled quantity, children, parent arrival), resting limits, the last 50 000 intent ids and per-account positions persist in `$FA_STATE_DIR/orders.json` (atomic, ≤ 1/s on change, at shutdown) and are restored before any subscription; a TWAP resumes its schedule (slices due during the downtime go out with the first trade after restart, never blind), a resting limit keeps resting, a redelivered intent stays a duplicate; an unreadable file stops the gateway instead of forgetting orders. A calibrated cost file contributes only its fitted `impact_k`s (`MergeCalibrated`); schedules and classes come from the current `costs.yaml`. |
| Price feed spike / bad tick | health check: price-jump sanity filter freezes instrument (implemented, §6.5 feed guard) |
| Allocator produces NaN / empty weights | fall back to previous allocation, raise risk event (implemented, `allocator.fallbacks`) |
| Kill switch triggered | flatten all live, halt all, persistent until manual re-arm (survives restart; implemented, `ledger.risk_state`) |

## 9. Acceptance criteria
- No order can reach a live venue without passing the full risk chain; tested by property tests injecting random targets.
- Kill switch flattens all live positions and state survives a restart of every service. **Met for the risk-engine** (`ledger.risk_state`).
- Loosening a limit at runtime via API is rejected; tightening succeeds and is audited.
- Paper and live code paths differ only in the `Venue` implementation (verified by shared integration tests).
- Every fill has attribution to originating strategies and a recorded slippage vs arrival price. **Met in paper** (`ledger.fill_attribution`; `ledger.fill.arrival_price` / `slippage_bps`).
- Promotion endpoint refuses a strategy whose gate is not green. **Met.**
- Allocator tournament reports comparable paper P&L for all allocators on identical signals. **Met** (live: `ctl.risk.allocators`; offline: `evo.eval.allocator`).

## 10. Open questions
1. Exact payout quota mechanics: fixed EUR/day, % of equity, or % of realised profit? Rollover if not withdrawn? — blocks liquidity/payout limit implementation.
2. Can manual orders bypass limits (e.g. emergency hedge)? — blocks override API semantics. *(Paper answer implemented: no, they pass the pre-trade checks; reductions always pass.)*
3. Netting cost attribution rule (pro-rata vs causal) — affects strategy fitness.
4. Daily-loss "day" definition for mixed 24/7 and session venues — blocks daily loss limit.
5. Margin/borrow availability checks for shorts per venue — needs catalog capability data. *(Paper answer implemented: class margin schedule + `ref.contract_spec.shortable`; per-venue borrow rates and locate data remain open.)*
6. Which allocator is active for the first live capital.

## 11. Deferred
- All numeric limit values (placeholders above). *(Basis recorded in §6.5 "Limit values and their basis"; the placeholders bind because every calibrated value is looser.)*
- Venue-specific order types (iceberg, post-only, reduce-only) beyond the generic algos.
- Options/derivatives-specific risk (greeks); v1 treats derivatives as delta-1 notionals.
