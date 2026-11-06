# 41 — Backtest & Validation

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium-high |
| **Owners (code)** | `crates/fa-backtest`, `crates/fa-evolution` (fitness corrections), `bins/evo-worker` |
| **Depends on** | 10 Ingestion, 12 Reference data, 20 Storage, 34 Decoder population, 35 Strategies, 40 Execution & risk |

## 1. Purpose
Provide a replay engine that evaluates decoders, strategies and allocators on historical data through the **same code path as live**, and a validation protocol that protects a system evolving very large populations from selecting overfit noise. Its outputs are fitness inputs (34/35) and promotion-gate statistics (40).

## 2. Scope
**In scope:** event replay, point-in-time correctness, cost/slippage model, locked holdout management, multiple-testing corrections (deflated Sharpe, PBO), live paper shadow bookkeeping, evaluation APIs used by evo-workers.
**Out of scope:** fitness objective definitions per decoder family (34), risk limits (40), ledger (42).

## 3. Decisions
- **Replay through the same code path as live**: events are re-emitted from the Garage raw archive (normalised by the same connector code) and VictoriaMetrics series into the same consumers.
- **Point-in-time as-of joins**: every lookup uses the knowledge time (`publish_time`/`ingest_time`), never `event_time` alone; macro data uses **vintages**.
- Overfitting control:
  - **Locked holdout period** — used only for promotion gates, never in selection.
  - **Deflated Sharpe ratio / Probability of Backtest Overfitting (PBO)** — applied as corrections inside fitness.
  - **Live paper shadow** — candidates must survive a shadow period on live data (40 defines the gate).
- **No walk-forward inside evolution** (single-GPU budget). Evolution evaluates on a fixed selection window; holdout + shadow provide out-of-sample control.
- Fitness is **net of costs**: fees, spread, slippage, funding, borrow.

## 4. Interfaces
```rust
pub trait EventSource {
    /// Ordered by knowledge time; ties broken deterministically by (source, seq).
    fn next(&mut self) -> Option<TimedEvent>;
}

pub struct ReplaySpec {
    pub window: TimeRange,            // must not intersect the locked holdout unless purpose == Promotion
    pub purpose: ReplayPurpose,       // Selection | Promotion | Diagnostic
    pub instruments: AssetSet,
    pub timescales: Vec<Timescale>,
    pub snapshot: SnapshotId,         // encoder snapshot to use
    pub cost_model: CostModelId,
    pub seed: u64,
}

pub trait Evaluator {
    fn evaluate(&self, spec: &ReplaySpec, candidate: &CandidateRef) -> EvalReport;
}
```
- Work queue: `jobs.eval.request` / `jobs.eval.result` (21) with `EvalJob { candidate, replay_spec }` → `EvalReport`.
- Control API: `GET /api/validation/holdout`, `GET /api/validation/reports/{candidate}`, `GET /api/validation/shadow/{strategy}`.

## 5. Data model
| Entity | Store |
|---|---|
| Raw payload archive (replay source of truth) | Garage `raw/<source>/<date>/…` |
| Bars / features | VictoriaMetrics |
| Macro vintages | Postgres (`macro_observation(series_id, ref_period, value, published_at, vintage)`) |
| Corporate actions, calendars | Postgres (12) |
| Holdout definition | Postgres `validation_holdout(timescale, start, end, locked_at)` — immutable once locked |
| EvalReport | Postgres (summary) + Helix (link to genome) + Garage (equity curves, per-trade logs) |
| Trial registry (all tested candidates) | Postgres `trial(id, generation, candidate, window, sharpe, n_obs, …)` — required for deflated Sharpe & PBO |
| Shadow tracking | Postgres `shadow_period(strategy_id, started_at, horizons_completed, stats…)` |

## 6. Behaviour & algorithms

### 6.1 Replay engine
- Merges per-source event streams into one stream ordered by **knowledge time**.
- Deterministic: same `ReplaySpec` + seed → bit-identical report (CI test).
- Runs encoders (frozen snapshot), MCMC, decoders, strategies, allocator and **paper venue** exactly as live; online adapters are either frozen or replayed online per spec flag **(proposal)**.
- Speed: fast-forward mode skips wall-clock waits; heavy encoder inference is cached per (snapshot, window) in Garage to make many strategy evaluations cheap **(proposal)**.

### 6.2 Point-in-time rules
- An event is visible at time t iff `knowledge_time ≤ t`, where knowledge_time = `publish_time` if known else `ingest_time` (live) or a conservative per-source delay (backfilled data, catalog field).
- Bars become visible at bar close + source latency.
- Macro: use the vintage available at t, not the revised value.
- Corporate actions: price series adjusted only with actions announced before t; raw prices used for execution simulation.
- Instrument universe: survivorship-free; delisted instruments remain in reference data with validity ranges.

### 6.3 Cost model
Per venue & instrument class: commission schedule, spread (from quotes, or modelled from bar range when quotes unavailable), slippage as function of order notional / volume and volatility (square-root impact **(proposal)**), funding (perps/FX swaps), borrow (shorts). Paper venue (40) uses the same model; realised live slippage recalibrates it periodically.

### 6.4 Data windows
Per timescale: `[selection window] [embargo gap] [locked holdout] [live]`. The holdout is the most recent completed segment at lock time, then fixed; it rolls forward only by an explicit re-lock action recorded in the audit log (re-lock invalidates gate results computed on the old holdout). Selection evaluations requesting any timestamp in the holdout are rejected.

### 6.5 Multiple-testing corrections
- **Trial registry**: every evaluated candidate is recorded with its selection-window statistics.
- **Deflated Sharpe Ratio** (Bailey & López de Prado): adjusts observed Sharpe for the number of trials, variance of trial Sharpes, skew and kurtosis; fitness uses DSR (or its probability) instead of raw Sharpe.
- **PBO** via combinatorially symmetric cross-validation over the selection window's sub-blocks, computed per generation for the elite set; reported and used in the promotion gate. (CSCV is not walk-forward training; it reuses existing evaluations.)
- Effective number of trials estimated from clustered correlation of trial returns **(proposal)**.

### 6.6 Live paper shadow
Strategies entering shadow run on live data with the paper venue. Tracked: calendar days, completed horizons, Sharpe / drawdown / hit rate, slippage vs model, consistency vs backtest distribution. Feeds the promotion gate in 40.

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `holdout.length.<timescale>` | seconds: 3d, minutes: 1m, hours: 3m, days: 1y | |
| `embargo.<timescale>` | 2 × horizon | |
| `pbo.n_blocks` | 16 | CSCV partitions |
| `pbo.max_for_promotion` | 0.2 | |
| `dsr.min_for_promotion` | 0.95 | probability |
| `cost.impact_coeff` | 0.1 | sqrt-impact |
| `backfill.knowledge_delay.<source>` | catalog | conservative default 1 bar |

## 8. Failure modes
| Failure | Response |
|---|---|
| Missing data in window | report marks coverage; candidates with < min coverage get no fitness |
| Look-ahead detected (assertion: event used before knowledge time) | evaluation fails hard; bug, not a fitness penalty |
| Holdout accessed by selection job | job rejected, audit event |
| Non-deterministic replay | CI failure |
| Cost model stale vs live slippage | alert, recalibration job |

## 9. Acceptance criteria
- Replaying a recorded live day reproduces the live paper decisions exactly (same orders, same timestamps).
- Determinism test: identical spec+seed → identical report hash.
- Look-ahead guard: synthetic test injecting a future-dated event is caught.
- Holdout is immutable via API; selection jobs touching it are rejected.
- Every evaluation appears in the trial registry; DSR and PBO are computable for any generation.
- Fitness values reported to evolution are net of the cost model.

## 10. Open questions
1. Holdout length per timescale (placeholders) — blocks gate calibration.
2. Should online adapters be replayed online or frozen during selection replay? — affects evaluation cost and realism.
3. Latency simulation: model order latency per venue, or assume next-event execution? — affects short timescales.
4. Effective-trials estimation method for DSR with huge correlated populations.
5. Re-lock cadence for the holdout as the system ages.

## 11. Deferred
- Walk-forward / purged CV inside evolution (explicitly rejected for now; revisit when compute scales).
- Market-impact models beyond square-root.
- Order-book-level (L2/L3) simulation.
