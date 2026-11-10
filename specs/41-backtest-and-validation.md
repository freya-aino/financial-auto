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
- **Implemented today (D-044):** `fa_backtest::eval::evaluate` scores a `fa_decoders` series decoder **sequentially point-in-time** (forecast at every step using data `≤ t`, scored against the realised `h`-step log return) and returns the objective set of spec 34 §6.3 (`crps`, `crps_skill`, `coverage`, `coverage_err`, `hit_rate`, `trade_sharpe`, `trade_psr`, `compute_cost`, `n_eval`) with budget/preemption stop hooks and multi-instrument pooling. It is scoring, not walk-forward retraining. The job surface is spec 21's generic `Job` / `JobResult` on `jobs.evo.eval.decoder`, not the dedicated messages above. Evaluation data comes from VictoriaMetrics since D-049 (`fa_tsdb::trade_prices`, in-memory buffer as fallback). Deflated Sharpe and PBO (CSCV) are implemented in the crate but not yet applied across the generation's trial set; the ordered replay driver exists, the full pipeline replay (encoders → strategies → paper venue) and the per-venue cost model do not.
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

**Implemented today (D-052):** `fa_backtest::replay` replays GP strategy genomes step by step over VictoriaMetrics bars (visible at close, ordered by `(close, instrument)` through the look-ahead guard) with a virtual book per sleeve and the cost model of §6.3; identical spec + seed + data give a byte-identical report. Encoders, MCMC and decoders are not re-run inside the replay yet: the strategy sees the price-only fallback context (feeding recorded latents is the next step).

**Since D-059** replays feed the recorded `latent.*` / `decoder.output.*` messages (read from the `LATENT` JetStream stream by time range) into the same `LatentGraph` / `DecoderOutputs` the live runtime uses, point-in-time, and report `sources{graph, decoder, fallback}`; a replay declares its snapshot and refuses latents of another one unless Diagnostic.

### 6.2 Point-in-time rules
- An event is visible at time t iff `knowledge_time ≤ t`, where knowledge_time = `publish_time` if known else `ingest_time` (live) or a conservative per-source delay (backfilled data, catalog field).
- Bars become visible at bar close + source latency.
- Macro: use the vintage available at t, not the revised value.
- Corporate actions: price series adjusted only with actions announced before t; raw prices used for execution simulation.
- Instrument universe: survivorship-free; delisted instruments remain in reference data with validity ranges.

### 6.3 Cost model
Per venue & instrument class: commission schedule, spread (from quotes, or modelled from bar range when quotes unavailable), slippage as function of order notional / volume and volatility (square-root impact **(proposal)**), funding (perps/FX swaps), borrow (shorts). Paper venue (40) uses the same model; realised live slippage recalibrates it periodically.

**Implemented today (D-051, D-052):** one model shared by the paper venue (`go/services/broker-gateway/costs.yaml`) and the backtest (`fa_backtest::cost`, pinned to the venue's `testdata/cost_fixture.json`): lookup `venues.<v>.classes.<c>` → `venues.<v>.default` → `default.classes.<c>` → `default.default`; fill = price ± (half spread + `k·σ_day·sqrt(N/ADV)`), commission with floor, funding/borrow carry per day as placeholders. Live slippage recalibration is pending.

### 6.4 Data windows
Per timescale: `[selection window] [embargo gap] [locked holdout] [live]`. The holdout is the most recent completed segment at lock time, then fixed; it rolls forward only by an explicit re-lock action recorded in the audit log (re-lock invalidates gate results computed on the old holdout). Selection evaluations requesting any timestamp in the holdout are rejected.

**Implemented today (D-059):** `POST /api/validation/holdout` locks a completed segment per timescale (immutable by trigger, 409 on overlap); Promotion-purpose replays run only as `evo.eval.strategy.holdout` jobs inside a locked holdout and their trial rows never enter selection statistics.

### 6.5 Multiple-testing corrections
- **Trial registry**: every evaluated candidate is recorded with its selection-window statistics.
- **Deflated Sharpe Ratio** (Bailey & López de Prado): adjusts observed Sharpe for the number of trials, variance of trial Sharpes, skew and kurtosis; fitness uses DSR (or its probability) instead of raw Sharpe.
- **PBO** via combinatorially symmetric cross-validation over the selection window's sub-blocks, computed per generation for the elite set; reported and used in the promotion gate. (CSCV is not walk-forward training; it reuses existing evaluations.)
- Effective number of trials estimated from clustered correlation of trial returns **(proposal)**.

**Implemented today (D-052):** every decoder or strategy evaluation writes a `validation.trial` row (generation, loop, rung, candidate, Sharpe, n_obs, objectives, 64 return block sums). After each trial the worker recomputes `validation.generation_stats` (deflated Sharpe of the best trial against the expected maximum of n trials, CSCV PBO with 16 blocks), which the orchestrator publishes in every `generation.summary`. Selection replays intersecting a locked `validation.holdout` row are rejected; the holdout lock workflow itself is pending.

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
