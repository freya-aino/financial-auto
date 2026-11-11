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
- **Implemented today (D-044):** `fa_backtest::eval::evaluate` scores a `fa_decoders` series decoder **sequentially point-in-time** (forecast at every step using data `≤ t`, scored against the realised `h`-step log return) and returns the objective set of spec 34 §6.3 (`crps`, `crps_skill`, `coverage`, `coverage_err`, `hit_rate`, `trade_sharpe`, `trade_psr`, `compute_cost`, `n_eval`) with budget/preemption stop hooks and multi-instrument pooling. It is scoring, not walk-forward retraining. The job surface is spec 21's generic `Job` / `JobResult` on `jobs.evo.eval.decoder`, not the dedicated messages above. Evaluation data comes from VictoriaMetrics since D-049 (`fa_tsdb::trade_prices`, in-memory buffer as fallback). Deflated Sharpe and PBO (CSCV) are applied across every generation's trial set since D-052 (§6.5); the full pipeline replay re-runs strategies on recorded latents (§6.1), not the encoders themselves.
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

**Implemented today (D-051, D-052):** one model shared by the paper venue (`go/services/broker-gateway/costs.yaml`) and the backtest (`fa_backtest::cost`, pinned to the venue's `testdata/cost_fixture.json`): lookup `venues.<v>.classes.<c>` → `venues.<v>.default` → `default.classes.<c>` → `default.default`; fill = price ± (half spread + `k·σ_day·sqrt(N/ADV)`), commission with floor, funding/borrow carry per day as placeholders.

**Live slippage recalibration (implemented in the paper venue, spec 40 §6.4):** the broker-gateway fits the impact coefficient `k` per (venue, class) from taker fills vs their parent arrival price (shrunk to the `costs.yaml` prior, clamped to [0.5, 2]×, ≥ 30 informative fills), writes `costs.calibrated.yaml` every 15 min, applies it in memory and announces it on `exec.costs.calibrated`. In paper only algo children are informative (a market fill at the arrival price is the model itself), so the fit measures market drift over the schedule; the backtest keeps pricing with the repo `costs.yaml` / fixture until fills from a real venue exist to calibrate against (§8 "cost model stale vs live slippage").

**Replay-vs-live identity (implemented, `fa_backtest::venue`):** a port of the paper venue's market state (last trade, quotes, EWMA σ of trade-to-trade returns, 1 h ADV window extrapolated to a day — `paper.go`) plus `fa_backtest::cost` reproduces the gateway's fills from a recorded window. `cargo run -p fa-backtest --example record_venue_window -- live …` captures the bus with every message stamped at its receive (knowledge) time and each trade's event time kept separately, marks a broker-gateway (re)start as `reset` and publishes small market probe intents on a throwaway account; `export` reads a past window from VictoriaMetrics and `ORDERS`. `tests/venue_identity.rs` replays `testdata/venue_fixture.jsonl` and requires every recorded market fill to be reproduced (live capture 2026-11-11 03:54–04:19 UTC across a gateway restart, Kraken BTC / `sim` / Yahoo market probes on `paper-test`: 289 of 289 fills reproduced, arrival price identical for all, fill price within 0.007 bps, fees exact; a 2× cost model is detected). A past window exported from VictoriaMetrics does **not** reproduce: stored trade samples carry the event time, and Kraken trades reach the bus 20–90 s after it (3–4 trades behind), so an event-time replay sees prices the live venue did not yet know — a measured look-ahead of the event-time bar replays for late-publishing feeds (spec 41 §6.2 knowledge time; `backfill.knowledge_delay.<source>` should carry it for `kraken`).

### 6.4 Data windows
Per timescale: `[selection window] [embargo gap] [locked holdout] [live]`. The holdout is the most recent completed segment at lock time, then fixed; it rolls forward only by an explicit re-lock action recorded in the audit log (re-lock invalidates gate results computed on the old holdout). Selection evaluations requesting any timestamp in the holdout are rejected.

**Implemented today (D-059):** `POST /api/validation/holdout` locks a completed segment per timescale (immutable by trigger, 409 on overlap); Promotion-purpose replays run only as `evo.eval.strategy.holdout` jobs inside a locked holdout and their trial rows never enter selection statistics.

**Automatic locking (implemented):** the holdout can no longer be silently absent. risk-engine plans it per timescale at start and every `[holdout] check_secs` (300) with `fa_backtest::holdout::plan_lock`: a timescale in `[holdout] timescales` (`1s`, `1m`) or of a waiting candidate without a holdout gets `[end − holdout.length.<ts>, end)`, `end` = now on the bar grid; it is inserted only when VictoriaMetrics holds ≥ `min_bars` (300) bars of the timescale in it (no empty holdouts), in a transaction holding `pg_advisory_xact_lock` and refusing overlaps (idempotent across restarts and concurrent lockers), and audited as `validation.holdout.lock` (actor `system:pipeline`). An existing row means nothing to do; `relock_after_days` (default 0 = never) adds a new non-overlapping segment once the latest one has aged. Timescales that cannot be locked are reported in `ctl.risk.pipeline` status (`holdouts.<ts>.reason`) and the heartbeat gauge `holdout.missing`. Lengths are `risk.toml [holdout.length_secs]` (1s: 2 h, 1m: 12 h, 1h: 90 d, 1d: 365 d — scaled to the history the stack holds; §7 placeholders stay the target).

**Selection windows avoid holdouts:** a now-relative selection window of `evo.eval.strategy` that would intersect a locked holdout is moved to end `2 × horizon` (≥ 60 s) before the holdout starts (`fa_backtest::holdout::avoid_holdouts`, the §6.4 layout), instead of being rejected until the window has passed it; explicit windows are still rejected. The pipeline (spec 35 §6.4.1) evaluates a candidate only on a holdout none of its selection windows touched.

### 6.5 Multiple-testing corrections
- **Trial registry**: every evaluated candidate is recorded with its selection-window statistics.
- **Deflated Sharpe Ratio** (Bailey & López de Prado): adjusts observed Sharpe for the number of trials, variance of trial Sharpes, skew and kurtosis; fitness uses DSR (or its probability) instead of raw Sharpe.
- **PBO** via combinatorially symmetric cross-validation over the selection window's sub-blocks, computed per generation for the elite set; reported and used in the promotion gate. (CSCV is not walk-forward training; it reuses existing evaluations.)
- Effective number of trials estimated from clustered correlation of trial returns **(proposal)**.

**Implemented today (D-052):** every decoder or strategy evaluation writes a `validation.trial` row (generation, loop, rung, candidate, Sharpe, n_obs, objectives, 64 return block sums). After each trial the worker recomputes `validation.generation_stats` (deflated Sharpe of the best trial against the expected maximum of n trials, CSCV PBO with 16 blocks), which the orchestrator publishes in every `generation.summary`. Selection replays intersecting a locked `validation.holdout` row are rejected; holdouts are locked automatically (§6.4).

**Effective number of trials (implemented):** `fa_backtest::validation::effective_trials` = `N / (1 + (N − 1)·ρ̄)` with ρ̄ the mean pairwise correlation of the trials' block returns (negative → 0); `generation_stats` deflates the best trial against the expected maximum of `N_eff` (plus trials without a finite Sharpe) instead of `N`, and reports `effective_trials`. A population of near-copies no longer counts as many independent tries. The evo-worker writes it to the result JSON and to the column `validation.generation_stats.effective_trials` (migration `20261112000020`; NULL on rows computed before it).

**Use in promotion:** PBO and the holdout deflated Sharpe are blocking items of the spec 40 §6.6 gate; candidate selection (spec 35 §6.4.1) applies `pbo_max` per generation and `dsr_min_candidate` per member; the single holdout trial decides with `dsr.min_for_promotion`.

### 6.6 Live paper shadow
Strategies entering shadow run on live data with the paper venue. Tracked: calendar days, completed horizons, Sharpe / drawdown / hit rate, slippage vs model, consistency vs backtest distribution. Feeds the promotion gate in 40.

**Implemented:** risk-engine keeps the shadow book (virtual exposure on `shadow_sleeve_eur`, hourly P&L points) and computes the consistency itself (`fa_risk::strategy::consistency`): the hourly Sharpe of the book since admission against the selection trial's per-step Sharpe scaled to an hour (`× sqrt(3600 / step_secs)`), consistent unless the shadow falls more than `promotion.consistency_z` (2) standard errors (`1/sqrt(n_hours)`) below; at least `decay_min_hours` (24) points. The same test on an active strategy's own book is the gate-decay demotion signal.

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `holdout.length.<timescale>` | seconds: 3d, minutes: 1m, hours: 3m, days: 1y | running values `risk.toml [holdout.length_secs]`: 2 h / 12 h / 90 d / 365 d until the history is longer |
| `embargo.<timescale>` | 2 × horizon | selection-window shift uses 2 × the strategy horizon (≥ 60 s) |
| `holdout.min_bars` | 300 | a segment with fewer bars is not locked (reported instead) |
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
- Replaying a recorded live day reproduces the live paper decisions exactly (same orders, same timestamps). **Partly met:** the execution half is pinned — a recorded live venue window reproduces every paper fill (`tests/venue_identity.rs`, §6.3); replaying recorded latents + tape through strategy-runtime → risk-engine to reproduce the intents themselves is not built (the `MD` stream keeps only minutes of trades, so a full day needs a longer market-data recording first).
- Determinism test: identical spec+seed → identical report hash. **Met** (`replay_is_deterministic_and_pays_costs`, `evaluation_is_deterministic_and_reasonable`, the allocator replay fixture).
- Look-ahead guard: synthetic test injecting a future-dated event is caught. **Met** (`look_ahead_guard_catches_future_event`, `replay_in_order_and_rejects_lookahead`).
- Holdout is immutable via API; selection jobs touching it are rejected. **Met**, and a holdout per configured timescale is locked automatically (§6.4).
- Every evaluation appears in the trial registry; DSR and PBO are computable for any generation. **Met** (with the effective-trials correction).
- Fitness values reported to evolution are net of the cost model. **Met** (`fa_backtest::cost` in every strategy / allocator replay, `cost_sensitivity` at 2×).

## 10. Open questions
1. Holdout length per timescale (placeholders) — blocks gate calibration.
2. Should online adapters be replayed online or frozen during selection replay? — affects evaluation cost and realism.
3. Latency simulation: model order latency per venue, or assume next-event execution? — affects short timescales.
4. Effective-trials estimation method for DSR with huge correlated populations. *(Mean-correlation estimate implemented; eigenvalue / clustering methods remain options.)*
5. Re-lock cadence for the holdout as the system ages. *(Mechanism exists: `[holdout] relock_after_days`; the cadence is still open, default never.)*

## 11. Deferred
- Walk-forward / purged CV inside evolution (explicitly rejected for now; revisit when compute scales).
- Market-impact models beyond square-root.
- Order-book-level (L2/L3) simulation.
