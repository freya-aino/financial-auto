# fa-risk

Non-evolvable hard risk limits, kill switch, health halts, strategy lifecycle and promotion gate.

**Status:** implemented (v1, D-051) — position / liquidity / leverage caps plus concentration caps per asset class, per venue and portfolio-wide (`cap_targets`), payout-quota netting, drawdown / daily-loss halt state machine with reduce-only gating (`gate`), manual re-arm (`rearm`), bus scope mapping (`HaltScope::parse/as_str`), tighten-only runtime limit changes (`RiskConfig::check_tightens`), and in `strategy`: the `active|paused|flattening|shadow` lifecycle (`StrategyState::apply`), the per-strategy drawdown tracker (`observe_pnl` / `breach`) and the promotion gate checklist (`PromotionGate::check`: min shadow days, min horizons, deflated Sharpe, no health halts are blocking; PBO and shadow-vs-backtest consistency are listed as advisory until a service computes them). All pure and tested; used live by `bins/risk-engine`.

## Owns
Pre-trade checks, halt state machine, lifecycle rules.

## Governing specs
- [40-execution-and-risk](../../specs/40-execution-and-risk.md)
