# fa-backtest

Replay engine over archived streams through the live code path; validation statistics.

**Status:** implemented (v1) — deflated Sharpe, PBO (CSCV), ordered replay driver, `eval` (sequential point-in-time scoring of `fa-decoders` series decoders), **`replay`** (pipeline replay of `fa-strategies` programs with a virtual book and execution simulation), **`cost`** (the paper venue's cost model, pinned to the broker-gateway fixture) and **`validation`** (per-generation deflated Sharpe / PBO over a trial set). All tested. Encoders/MCMC inside the replay (spec 41 §6.1) are pending: strategy programs see `features::FallbackContext` (price-only) during replay.

## Modules
| Module | What |
|---|---|
| `lib` | `ReplaySource` / `ReplaySink` / `run_replay`: knowledge-time ordered delivery, out-of-order = `ReplayError::OutOfOrder` (look-ahead guard, spec 41 §9) |
| `stats` | `sharpe`, `probabilistic_sharpe`, `expected_max_sharpe`, `deflated_sharpe`, `pbo_cscv` |
| `eval` | `evaluate(decoder, series, spec) -> EvalReport` for decoders (spec 34 §6.3 objectives); `pnl` = per-step P&L of the naive sign policy (trial registry input); `pool` |
| `cost` | `CostModel` = port of `go/services/broker-gateway/costs.go`: `from_yaml(costs.yaml)`, `schedule(venue, class)` (lookup `venues.<v>.classes.<c>` → `venues.<v>.default` → `default.classes.<c>` → `default.default`), `price(CostInputs) -> CostResult` (half spread + square-root impact `price·k·σ_day·sqrt(notional/ADV)` on the fill price, commission with a floor), `ClassCosts::carry` (funding / borrow per day), `scaled(2.0)`. `tests::matches_broker_gateway_fixture` replays `go/services/broker-gateway/testdata/cost_fixture.json` |
| `replay` | `ReplaySpec` (window, `ReplayPurpose::{Selection, Promotion, Diagnostic}`, instruments, timescale, snapshot, cost model, seed, capital, `n_trials` / `trial_sharpe_var` for the deflated Sharpe, `min_trade_fraction`), `Holdout` + `check_holdout` (Selection windows intersecting a locked holdout are rejected), `BarSeries::from_points` (bars stamped at close), `Evaluator` trait, `PipelineReplay` and `replay(...) -> EvalReport` |
| `validation` | `Trial`, `block_sums` (`N_BLOCKS` = 64), `generation_stats(trials, PBO_BLOCKS)` → `GenerationStats { n_trials, sharpe_var, best_candidate, expected_max_sharpe, deflated_sharpe, pbo, … }` |

## Strategy replay (`replay`)
Events are bar closes merged across instruments and ordered by `(close time, instrument)`; a bar is visible at its close. At every step: mark to market (equity point, per-step net return, funding/borrow carry), then the program decides for every instrument with `lookback_bars + 1` closes using only closes `≤ t` and its own virtual book (`position`, `unrealised_pnl`, `time_in_trade`), and each sleeve (`capital / n_instruments`) is filled at the close through `CostModel::price` (σ_day = lookback return std scaled to a day). Accounting is cash + mark-to-market, so `Δequity` is the net return and `Δequity + costs` the gross one. Identical spec + seed + data → identical report (test).

Objectives (`EvalReport::objectives`, directions in `replay::DIRECTIONS`): `deflated_sharpe` (max; PSR vs the expected max Sharpe of `n_trials` trials — with one trial, PSR vs 0), `sharpe` (max), `max_drawdown` (min), `turnover` (min), `cost_sensitivity` = `sharpe − sharpe under 2× costs` (min), `compute_cost` = log10 program nodes per step (min); informational `calibration` (corr of emitted `std` with the next |return|), `cost_share`, `psr`, `n_trades`, `n_obs`. The report also carries the equity curve, per-step returns and the per-trade log (fill price, fee, slippage bps).

## Owns
Replay driver, cost model, deflated Sharpe/PBO, decoder evaluation, strategy replay, generation statistics.

## Governing specs
- [41-backtest-and-validation](../../specs/41-backtest-and-validation.md), [35-strategies](../../specs/35-strategies.md) §6.1–6.2, [40-execution-and-risk](../../specs/40-execution-and-risk.md) §6.7 (cost model shared with the paper venue)
