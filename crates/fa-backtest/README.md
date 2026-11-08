# fa-backtest

Replay engine over archived streams through the live code path; validation statistics.

**Status:** partial — deflated Sharpe, PBO (CSCV), ordered replay driver, and `eval`: sequential point-in-time scoring of `fa-decoders` series decoders (CRPS + skill vs naive, interval coverage, hit rate, naive-policy Sharpe/PSR net of a bp cost, deterministic compute-cost proxy; budget/preemption stop hooks; multi-instrument pooling), all tested. Cost model per venue and pipeline replay pending.

`eval::evaluate(decoder, series, spec) -> EvalReport`; `EvalReport::objectives()` is the objective vector `evo-worker` reports, `score()` the provisional D-025 scalar (`crps_skill`). This is scoring without retraining, consistent with spec 41's "no walk-forward inside evolution".

## Owns
Replay driver, cost model, deflated Sharpe/PBO, decoder evaluation.

## Governing specs
- [41-backtest-and-validation](../../specs/41-backtest-and-validation.md)
