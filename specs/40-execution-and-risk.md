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
- `signal.exposure.<strategy_id>` (NATS, proto `fa.v1.ExposureSignal`): per instrument, mean target exposure in [-1, 1] of the strategy's sleeve plus uncertainty (std or quantiles), timescale, horizon, snapshot id.
- Strategy registry (Helix + Postgres): status (`paper`, `shadow`, `live`, `paused`, `retired`), sleeve, asset set, horizon.
- Market state (VictoriaMetrics / bus): last prices, rolling traded volume per horizon, spreads.
- Ledger (42): positions, cash, equity, realised/unrealised P&L, payout quota.
- Health signals: `ops.health.*` (feed staleness, model drift, broker status).

### 4.2 Outputs
- `order.intent.<venue>` (proto `fa.v1.OrderIntent`): instrument, side, quantity, exec algo + params, parent portfolio target id, idempotency key.
- `order.event.<venue>` (from broker-gateway): acknowledgements, fills, cancels, rejects → ledger (42).
- `risk.event` (proto `fa.v1.RiskEvent`): limit breaches, halts, kill-switch activations.

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

### 6.2 Allocators (initial set)
- **Bayesian bandit + risk parity (proposal default for paper)**: Thompson sampling over each strategy's posterior return distribution; resulting weights capped by a risk-parity budget (inverse volatility, covariance-aware). Can consume MCMC posteriors directly.
- **Fractional Kelly**: covariance-aware Kelly weights from shrunk return estimates, multiplied by a fraction (placeholder 0.25).
- **Evolved allocator**: genome-defined allocation policy evaluated under the same Pareto fitness as strategies (35), in paper only until promoted.
- **Equal-risk baseline**: always running in paper as a control.
All allocators run in parallel in paper ("allocator tournament"); their portfolio P&L is compared on the same strategy signals.

### 6.3 Netting
Exposures are converted to signed EUR notionals and summed per instrument. Attribution is kept: every portfolio position carries per-strategy virtual sub-positions so strategy P&L and fitness remain computable even though orders are netted. Fees/slippage of netted orders are attributed pro-rata to the strategies whose deltas caused the trade **(proposal)**.

### 6.4 Execution algorithms
| Algo | Use when (proposal) |
|---|---|
| Market | urgent (kill switch, flatten), very liquid, small vs volume |
| Limit | patient, long horizon, spread wide relative to expected edge |
| Smart-limit | default: passive limit with timeout, then reprice/cross |
| TWAP | order large vs per-interval volume, horizon ≥ hours |
| VWAP | as TWAP but volume-profile weighted (needs intraday volume curve) |
Selection inputs: strategy horizon, order notional / horizon volume, spread, venue capabilities (from catalog 11). Every child fill records arrival price for slippage measurement.

### 6.5 Hard limits — details
- **Drawdown kill switch**: portfolio equity vs running peak (EUR); breach → flatten all live positions with market/smart-limit-aggressive, halt all live strategies, require manual re-arm. Per-strategy drawdown → flatten + pause that strategy.
- **Daily loss limit**: realised + unrealised P&L since session start (UTC day for 24/7 venues, exchange day otherwise) ≤ −X% equity → no new risk-increasing orders until next day; reducing orders allowed.
- **Leverage**: gross exposure / equity ≤ cap per asset class; also a portfolio-wide gross cap.
- **Concentration**: |position| / equity ≤ cap per instrument, per asset class, per venue (venue cap limits counterparty risk).
- **Health halts**: feed stale > threshold for an instrument → strategies using it frozen (no new orders, existing positions kept unless stale exceeds second threshold); model drift alarm on a snapshot → strategies depending on that snapshot paused; broker gateway errors / disconnected → venue halted, intents queued then dropped after TTL.
- **Liquidity cap net of payouts**: `|position_i| ≤ k × volume_i(horizon)`, and `Σ capital deployed ≤ equity − payout_quota_reserved_today` (payout quota from ledger 42).
Limits are loaded from `config/risk.toml` at startup and can only be tightened at runtime from the UI; loosening requires config change + restart **(proposal)**.

### 6.6 Promotion gate (paper → live)
A strategy is promotable when **all** hold:
1. Minimum shadow calendar time in live paper (placeholder 30 days) **and** a minimum number of completed horizons (placeholder 20).
2. Deflated Sharpe on locked holdout above threshold (41).
3. PBO below threshold (41).
4. Shadow performance statistically consistent with backtest (e.g. shadow Sharpe within the backtest's confidence band; slippage within modelled band).
5. No unresolved health halts in the shadow window.
Promotion is executed manually from the UI (gate must be green); initial live sleeve is small (placeholder 1% of equity) and grows only via the allocator. Demotion is automatic on per-strategy drawdown breach or gate-statistics decay.

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
| `exec.no_trade_band` | 0.5% of target | churn control |
| `exec.smart_limit_timeout` | 30 s (scaled by timescale) | |
| `allocator.kelly_fraction` | 0.25 | |
| `promotion.min_shadow_days` | 30 | calendar days |
| `promotion.min_horizons` | 20 | |
| `promotion.initial_live_sleeve` | 1% equity | |

## 8. Failure modes
| Failure | Response |
|---|---|
| Broker disconnect mid-execution | cancel-on-disconnect where supported; on reconnect reconcile (42) before any new intent |
| Duplicate intent delivery (NATS redelivery) | idempotency key dedup in gateway and ledger |
| Fill for unknown order | ledger flags anomaly, venue halted, manual review |
| Risk service crash | broker-gateway rejects all new risk-increasing intents if no risk heartbeat within T (fail-closed) |
| Price feed spike / bad tick | health check: price-jump sanity filter freezes instrument |
| Allocator produces NaN / empty weights | fall back to previous allocation, raise risk event |
| Kill switch triggered | flatten all live, halt all, persistent until manual re-arm (survives restart) |

## 9. Acceptance criteria
- No order can reach a live venue without passing the full risk chain; tested by property tests injecting random targets.
- Kill switch flattens all live positions and state survives a restart of every service.
- Loosening a limit at runtime via API is rejected; tightening succeeds and is audited.
- Paper and live code paths differ only in the `Venue` implementation (verified by shared integration tests).
- Every fill has attribution to originating strategies and a recorded slippage vs arrival price.
- Promotion endpoint refuses a strategy whose gate is not green.
- Allocator tournament reports comparable paper P&L for all allocators on identical signals.

## 10. Open questions
1. Exact payout quota mechanics: fixed EUR/day, % of equity, or % of realised profit? Rollover if not withdrawn? — blocks liquidity/payout limit implementation.
2. Can manual orders bypass limits (e.g. emergency hedge)? — blocks override API semantics.
3. Netting cost attribution rule (pro-rata vs causal) — affects strategy fitness.
4. Daily-loss "day" definition for mixed 24/7 and session venues — blocks daily loss limit.
5. Margin/borrow availability checks for shorts per venue — needs catalog capability data.
6. Which allocator is active for the first live capital.

## 11. Deferred
- All numeric limit values (placeholders above).
- Venue-specific order types (iceberg, post-only, reduce-only) beyond the generic algos.
- Options/derivatives-specific risk (greeks); v1 treats derivatives as delta-1 notionals.
