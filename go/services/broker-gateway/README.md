# broker-gateway

Execution adapters: order intents → venues; paper mode first.

**Status:** implemented (paper only, D-051 + execution algorithms) — executes `order.approved.>` intents against the live tape with the cost + slippage model of spec 41 §6.3 (`costs.yaml`), publishes `OrderEvent` on `order.event.<account>` with `fee`, `arrival_price`, the realised `slippage_bps` and, per fill, a unique `venue_fill_id` (`pf-<unix ns>-<seq>`) plus `instrument_id` / `venue_symbol` (D-078; the ledger keys fills `intent:venue_order:vf:<id>` and books orphans from them), idempotent by `intent_id`. Execution algorithms (`algo.go`): market, limit, smart-limit, TWAP, VWAP, POV and an `auto` choice; working orders can be cancelled with `OrderCancel`. Live fills recalibrate the impact coefficient (`calib.go`). Working orders, resting limits, intent ids and positions persist across restarts, and risk-increasing intents are rejected while the risk-engine heartbeat is missing (`state.go`, D-079). Open positions pay the schedule's funding / borrow carry (`carry.go`, charged hourly on `exec.carry.<account>`). Live mode is rejected; `VenueAdapter` is the interface the paper engine implements and a live adapter will share.

## Execution algorithms (`exec_algo`, spec 40 §6.4)
Grammar: `name` or `name:k=v,k=v`; empty = `market`. Unknown names, unknown parameters or out-of-range values → `REJECTED` with the reason.

| Algo | Parameters (default) | Behaviour |
|---|---|---|
| `market` | — | one taker fill at last ± costs (`paper-<intent>`) |
| `limit` | `limit_price` | marketable → taker fill; else rests as `ACK` until a trade crosses it (passive fill at the limit, commission only). Any intent carrying `limit_price` with `market`/empty behaves the same (back-compat) |
| `smart_limit` | `timeout_s` 30, `offset_bps` 0 | `ACK`, then rests passively at the touch (bid for buys, ask for sells, last without a quote) improved by `offset_bps`; a crossing trade fills it passively, otherwise it crosses as a taker at the timeout. Single fill `paper-<intent>` |
| `twap` | `dur_s` 300, `slices` 10 | `ACK`; equal child slices, the first at acceptance, then every `dur_s/slices` |
| `vwap` | `dur_s` 300, `slices` 10 | as TWAP, child sizes weighted by the learned time-of-day volume profile of the instrument (traded notional per 5-min UTC bucket observed on the tape since start); uniform when nothing is known |
| `pov` | `rate` 0.1, `max_dur_s` 900 | `ACK`; children execute `rate` × traded size observed on the tape since acceptance; the remainder crosses at `max_dur_s` |
| `auto` | — | gateway fallback choice from participation = notional / learned ADV and spread: ≤ 0.1 % and spread ≤ 10 bps → market; ≤ 0.1 % with a wider spread → smart_limit; ≤ 1 % → twap; above → pov (defaults). Counted as `algo.auto.<chosen>` |

Children are driven by a 1 s ticker (`-algo-tick`) and by trades of the instrument; a slice waits while no price has been seen. Each child is priced by the cost model at the then-last price with the child quantity (σ / ADV from the tape) and published as `PARTIAL_FILL` with `venue_order_id = paper-<intent>-<n>` (n from 1); the last child is `FILL`. `arrival_price` is the **parent** arrival (last trade when the intent was accepted) and `slippage_bps` the implementation shortfall against it. A cancel of a working algo order cancels the remainder: `CANCELED`, `fill_quantity` 0, reason with the quantity filled so far. Working orders survive a gateway restart (see restart safety below).

## Live slippage recalibration (`calib.go`, spec 41 §6.3)
Each taker fill adds a sample per (venue, asset class): `x = 1e4 · σ_day · sqrt(notional/ADV)`, `y = realised_bps − half_spread_bps` (realised vs the parent arrival). Every `-recalibrate-every` (15 min) a key with ≥ `-recalibrate-min-samples` (30) informative samples in its rolling window (500) gets `k̂ = Σxy/Σx²`, shrunk toward the prior `k` of the base costs file with weight `n/(n+50)` and clamped to `[0.5, 2] ×` prior; a prior of 0 (costless `sim`) stays 0. The calibrated model (base + `venues.<v>.classes.<c>.impact_k`) is written to `-costs-calibrated` (default `$FA_STATE_DIR/costs.calibrated.yaml`, compose: named volume `broker-gateway-state` at `/var/lib/fa`) with `<file>.report.json`, applied in memory when `-recalibrate-apply` (default true), loaded at start when present (the base file always stays the prior), and the report JSON is published on `exec.costs.calibrated`.

**Honest limit in paper:** a fill priced at the arrival price itself (market orders, single crosses) is the model echoing itself (`y = k_in_use · x`), so such samples are counted (`echo`) and excluded from the fit; only algo children, whose price drifted from the parent arrival, carry information — and in paper that information is market drift over the algo's horizon, not real impact. With a live venue adapter every fill is informative through the same code path. The Rust backtester (evo-worker, `FA_COSTS`) prices with the base `costs.yaml`, not the calibrated overlay; `testdata/cost_fixture.json` only pins the Rust model in tests.

## Restart safety and fail-closed (`state.go`, spec 40 §8)
- **Orders state:** working algo orders (schedule, filled quantity, child count, parent arrival), resting limits, the last 50 000 intent ids and the net position per `account|instrument` are written to `-orders-state` (default `$FA_STATE_DIR/orders.json`, the `broker-gateway-state` volume) atomically at most once per second when they changed, and at shutdown; at start the file is restored before any subscription. Slices that fell due while the gateway was down go out with the first trade seen after the restart (never priced blind). An unreadable file stops the gateway (exit 1) rather than forgetting working orders; a missing file is a clean start. Verified live 2026-11-11 05:57 (throwaway account `paper-econ`, TWAP 6 × 1 on `sim`, gateway restarted after 3 children: `orders state restored working=1`, children 4–6 continued with the parent arrival). A calibrated costs file only contributes its fitted `impact_k`s (`MergeCalibrated`).
- **Fail-closed:** `svc.heartbeat.risk-engine` feeds a guard; while the last heartbeat is older than `-risk-heartbeat-timeout` (15 s; 0 disables) or none was seen since start, an intent that moves the account's projected position (filled + working + resting) in the instrument away from zero or through it is `REJECTED` with `fail-closed: …`; reductions and closes still fill, working children continue. Gauges `risk_heartbeat_age_s` (−1 = never), `risk_heartbeat_alive`, counter `failclosed.rejected`. Verified live 2026-11-11 05:59 with the risk-engine stopped: a buy on `paper-econ` rejected after 22 s, the closing sell filled.

## Cost model (`costs.yaml`, `-costs` / `FA_COSTS`)
Per venue / asset class fee schedule, looked up in this order: `venues.<venue>.classes.<class>`, `venues.<venue>.default`, `default.classes.<class>`, `default.default` (first table found wins). The venue is the one of the latest `md.trade.<venue>.<token>` for the instrument; the class comes from the `instruments:` map (`default_class` otherwise — the gateway has no reference data).

| Term | Rule |
|---|---|
| half spread | `(ask − bid)/2` from the latest `md.quote.>` for the instrument, else `price · spread_bps / 2e4` |
| impact | square-root impact `price · k · σ_day · sqrt(notional / ADV)`; `k` = `impact_k` of the schedule or `default_k`; σ_day = EWMA of trade-to-trade log returns per second × √86400 (`default_sigma` until 5 returns are seen); ADV = last hour of traded notional extrapolated to 24 h (`default_adv_eur` when nothing observed) |
| fill | `last ± (half spread + impact)` (+ buy, − sell); passive limit fills: at the limit price, no spread, no impact |
| fee | `max(commission_min_eur, |qty · fill| · commission_bps / 1e4)`, EUR |
| `slippage_bps` | signed cost vs the arrival price (last trade when the intent was accepted), positive = paid |
| funding / borrow | charged on open positions per carry period (below): `funding_bps_day` on any position, `borrow_bps_day` on shorts — the backtest's rule (`fa_backtest::cost::ClassCosts::carry`) |

Without a costs file the built-in default applies (5 bps commission, 1 EUR minimum, 5 bps spread, k 0.1, σ 2 %/day, ADV 10 M EUR). The repo `costs.yaml` makes the synthetic `sim` venue costless so the baseline paper P&L stays readable, maps the Kraken pairs (`XBTEUR`, `ETHEUR`, `SOLEUR`) and Yahoo `BTC-USD` to class `crypto`, and prices venue `kraken` with Kraken Pro's lowest-tier 40 bp taker fee (assumed, D-079); a minimum fee per order is why the risk-engine's cost-aware rebalancing (`fa_risk::rebalance`) skips small rebalances.

`testdata/cost_fixture.json` is exported from `costs.yaml` by `TestFixtureMatchesModel` (`UPDATE_FIXTURE=1 go test ./services/broker-gateway` regenerates it): the model as JSON plus input → expected `fee`, `fill_price`, `half_spread`, `impact`, `slippage_bps` cases, so `crates/fa-backtest` can assert it prices fills identically.

## Paper carry (`carry.go`, spec 40 §6.5 shorts)
Every scheduler tick integrates each open `account|instrument` position at the then-last mark: `|qty · last| · dt / 1 d` notional-days, times the schedule's `funding_bps_day` (any side) and, for shorts, `borrow_bps_day` (schedule of the instrument's md venue and `costs.yaml` class, as for fees). At the end of every `-carry-period` (1 h, UTC grid) each non-zero accrual becomes one charge on `exec.carry.<account>` (JSON `{account, instrument_id, kind: funding|borrow, amount (negative = paid, in the instrument's price currency), ccy ("" = quote currency), bps_day, notional_days, venue, class, period_start, period_end, reference}`; `reference = carry:<instrument>:<kind>:<period end unix>`). The last 6 periods are re-announced with every new one, so a consumer that was down books them late; the risk-engine books each reference once as `ledger.cash_movement` of that kind (converted to EUR at `period_end`). Accruals and the recent periods persist in `orders.json` (`carry`), so a restart neither repeats nor drops a period; time the gateway was down is accrued on the next tick (≤ 7 days) at the mark then known — a position whose instrument has no mark yet accrues nothing for that interval. Counters `carry.charges`, `carry.funding_paid`, `carry.borrow_paid`, `carry.publish_errors`. The repo schedule charges equity borrow 0.3 bp/day, FX funding 0.5 bp/day and default crypto funding 1 bp/day (Kraken spot 0).

## Subjects
| Subject | Direction | Payload |
|---|---|---|
| `order.approved.>` | in | `OrderIntent`; non-paper → `REJECTED`, no price seen → `REJECTED`, unmarketable limit → `ACK` (resting) |
| `md.trade.>` | in | `Envelope{Trade}`: mark, σ, volume; fills resting limits that became marketable |
| `md.quote.>` | in | `Envelope{Quote}`: spread for the next fills |
| `order.cancel.<account>` | in | `OrderCancel{intent_id, account_id, reason, at, actor}` (a legacy `OrderIntent` payload still decodes: field 1 is the intent id in both); cancels a resting limit or a working algo order → `CANCELED`, ignored (`cancels.unknown`) when nothing is working |
| `order.event.<account>` | out | `OrderEvent` (`ACK` / `PARTIAL_FILL` / `FILL` / `CANCELED` / `REJECTED`) |
| `exec.carry.<account>` | out | JSON carry charge per (instrument, kind, period), see paper carry |
| `exec.costs.calibrated` | out | JSON calibration report `{at, applied, path, error?, entries[{venue, class, fills, informative, echo, k_prior, k_fit, weight, k_new, clamped, realised_bps_ewma, model_bps_ewma, status}]}` |

Heartbeat counters (also `fa_heartbeat_counter{service="broker-gateway",key=…}` on `/metrics`, spec 60 §6.5): `events.<KIND>`, `resting`, `working`, `quotes`, `intents.duplicate`, `cancels.unknown`, `errors.decode`, `algo.<name>.working|done|cancelled`, `algo.auto.<chosen>`, `slippage.<venue>.<class>.bps_mean` (realised vs arrival, EWMA α 0.05), `slippage.<venue>.<class>.model_bps_mean` (modelled half spread + impact), `slippage.<venue>.<class>.n`, `exec_slippage_bps.<algo>` (EWMA), `costs.k.<venue>.<class>` (impact coefficient in use), `calibrations`, `risk_heartbeat_age_s`, `risk_heartbeat_alive`, `failclosed.rejected`, `state.saves`, `state.save_errors`, `state.restored_working`.

In compose the gateway loads the repo `costs.yaml` baked into the image (`FA_COSTS=/app/config/costs.yaml`; before this it ran on the built-in default model), keeps `FA_STATE_DIR=/var/lib/fa` on the `broker-gateway-state` volume (calibrated costs + report, `orders.json`), serves `/metrics` on `0.0.0.0:9100` (published as `127.0.0.1:9106`) and is health-checked by `healthprobe` (`go/services/healthprobe`).

## Run
```sh
go/bin/broker-gateway -nats nats://127.0.0.1:4222 -costs go/services/broker-gateway/costs.yaml
go test ./services/broker-gateway        # cost model, fixture, limit/cancel paths, algos, cancel decode, recalibration, carry
```
| Flag | Env | Default |
|---|---|---|
| `-nats` | `FA_NATS_URL` | `nats://127.0.0.1:4222` |
| `-costs` | `FA_COSTS` | `""` (built-in default model) |
| `-costs-calibrated` | `FA_COSTS_CALIBRATED` | `$FA_STATE_DIR/costs.calibrated.yaml`, else next to `FA_COSTS` (`""`: not persisted) |
| `-orders-state` | `FA_ORDERS_STATE` | `$FA_STATE_DIR/orders.json`, empty without `FA_STATE_DIR` (`""`: in memory only) |
| `-recalibrate-every` / `-recalibrate-apply` / `-recalibrate-min-samples` | | `15m` / `true` / `30` |
| `-algo-tick` | | `1s` |
| `-risk-heartbeat-timeout` | | `15s` (0 disables fail-closed) |
| `-carry-period` | | `1h` (0 disables funding / borrow charges) |
| `-metrics` | `FA_METRICS_BIND` | host port of `docker/README.md` (compose `0.0.0.0:9100`; `off` disables) |

## Owns
Venue adapters and exec algos transport.

## Governing specs
- [Execution & risk (ex spec 40)](../../../bins/risk-engine/README.md#execution-and-risk-lasting-content-of-the-retired-spec-40)
- [41-backtest-and-validation](../../../specs/41-backtest-and-validation.md) §6.3
- [fa-ledger README](../../../crates/fa-ledger/README.md) (retired spec 42, Ledger & accounting)
