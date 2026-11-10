# risk-engine

Live service: meta-allocation across strategy signals, non-evolvable hard risk limits, the halt state machine and the paper book (specs 40, 42; D-028, D-029, D-030).

**Status:** running (paper only) — `signal.exposure.>` → allocator (`fa-allocator`) → EUR targets → caps + halt gate (`fa-risk`) → diff against the paper book (`fa-ledger`) → `order.intent.paper` + `order.approved.paper`; books fills from `order.event.>`, marks from `md.trade.>`; publishes `risk.state` (equity, deployable, gross exposure, drawdown, daily P&L, active halts) and raises `risk.halt.portfolio` / `risk.halt.daily_loss` itself; honours `risk.halt.>` (incl. the UI kill switch) and `risk.resume.<scope>`. With `[postgres]` reachable the paper book is persisted (D-047): the 100 000 EUR opening deposit is a `cash_movement` booked once, every own and foreign intent and every fill is written to `ledger.*` (idempotent), a `position_snapshot` is taken every `snapshot_secs` (60) and on each halt, halts/resumes/limit loads go to `audit.log`, and every `reconcile_secs` (300) the in-memory book is compared with the Postgres-derived one into `reconciliation_run` (`ok`/`diff`; a diff is logged as an error but does not halt yet — spec 42 follow-up). Restarting reloads the book from Postgres (drawdown peak and day-start equity restart from the loaded equity; intents pending at shutdown are not restored). Without Postgres it logs once and stays in memory.

## Pipeline per tick (spec 40 §6.1)
1. Latest signal per strategy (stale after `signal_ttl_secs`).
2. Allocator weights (`equal_weight` control allocator by default; `risk_parity` / `fractional_kelly` fed by an online per-strategy performance posterior).
3. `net_targets` → signed EUR target per instrument from `deployable = equity − payout quota`.
4. `fa_risk::RiskEngine::cap_targets`: per-instrument position cap, liquidity cap (`k ×` rolling traded volume, extrapolated while the window fills), per-asset-class leverage cap.
5. `gate`: kill switch / health halts → flatten everything; daily-loss halt → reduce-only.
6. Diff against book + pending intents, no-trade band (`max(min_order_eur, band × |target|)`) → market `OrderIntent`s in `TradingMode::Paper`, account `paper`, contributors in `strategy_ids`. Each is published on `order.intent.paper` and immediately on `order.approved.paper` (the broker-gateway paper engine fills it).

Foreign intents seen on `order.intent.<account>` (none are published today; control-api sends manual orders straight to `order.approved.paper`, D-029) are gated the same way: non-paper mode or an active halt → `OrderEvent{REJECTED}` on `order.event.<account>`.

## Halts
| Scope | Raised by | Effect | Cleared by |
|---|---|---|---|
| `portfolio` | drawdown ≥ `limits.max_drawdown`, or any `risk.halt.portfolio` (UI kill switch) | flatten, no orders | `risk.resume.portfolio` (re-arms the drawdown peak) |
| `daily_loss` | loss since UTC day start ≥ `limits.max_daily_loss` | reduce-only | next UTC day, or `risk.resume.daily_loss` |
| `source`, `model`, `broker` (`venue:*` etc.) | other services | flatten, no orders | `risk.resume.<scope>` |

## Settings
[`risk.toml`](risk.toml) (`--risk`, `FA_RISK`); the built-in copy is used when the file is missing. `mode` must be `paper`; anything else fails startup. Limits follow the spec 40 §7 placeholders.

## Run
```sh
cargo run -p risk-engine                                   # config/platform.toml, bins/risk-engine/risk.toml
FA_CONFIG=config/platform.docker.toml FA_RISK=/app/bins/risk-engine/risk.toml risk-engine   # in the Rust image
```
Then `GET /api/trading/{signals,orders,positions,risk}` on control-api shows the flow; `POST /api/trading/halt` flattens and stops orders, `POST /api/trading/resume` re-arms.

## Layout
- `src/engine.rs` — pure, deterministic engine (`on_signal`, `on_trade`, `on_intent_seen`, `on_order_event`, `on_halt`, `on_resume`, `tick`, `state`), tested end-to-end with synthetic signals/fills.
- `src/settings.rs` — `risk.toml` schema and validation.
- `src/persist.rs` — Postgres side effects (`fa-ledger::pg::PgLedger`, `fa-warehouse::audit`): intents, fills, snapshots, reconciliation runs, audit entries; best effort, idempotent.
- `src/main.rs` — NATS wiring, heartbeat `svc.heartbeat.risk-engine` (counters: signals, trades, intents.ours, fills, halts.*, equity_eur, cash_eur, gross_exposure_eur, positions, halted).

## Governing specs
- [40-execution-and-risk](../../specs/40-execution-and-risk.md)
- [42-ledger-and-accounting](../../specs/42-ledger-and-accounting.md)
- [21-bus-and-contracts](../../specs/21-bus-and-contracts.md)
