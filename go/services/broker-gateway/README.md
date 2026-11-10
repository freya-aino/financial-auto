# broker-gateway

Execution adapters: order intents → venues; paper mode first.

**Status:** implemented (paper only, D-051) — fills `order.approved.>` intents against the live tape with the cost + slippage model of spec 41 §6.3 (`costs.yaml`), publishes `OrderEvent` on `order.event.<account>` with `fee`, `arrival_price` and the realised `slippage_bps`, idempotent by `intent_id`. Limit orders fill when marketable, otherwise rest as `ACK` until a trade crosses the limit (passive fill at the limit, commission only) or they are cancelled. Live mode is rejected; `VenueAdapter` is the interface the paper engine implements and a live adapter will share.

## Cost model (`costs.yaml`, `-costs` / `FA_COSTS`)
Per venue / asset class fee schedule, looked up in this order: `venues.<venue>.classes.<class>`, `venues.<venue>.default`, `default.classes.<class>`, `default.default` (first table found wins). The venue is the one of the latest `md.trade.<venue>.<token>` for the instrument; the class comes from the `instruments:` map (`default_class` otherwise — the gateway has no reference data).

| Term | Rule |
|---|---|
| half spread | `(ask − bid)/2` from the latest `md.quote.>` for the instrument, else `price · spread_bps / 2e4` |
| impact | square-root impact `price · k · σ_day · sqrt(notional / ADV)`; `k` = `impact_k` of the schedule or `default_k`; σ_day = EWMA of trade-to-trade log returns per second × √86400 (`default_sigma` until 5 returns are seen); ADV = last hour of traded notional extrapolated to 24 h (`default_adv_eur` when nothing observed) |
| fill | `last ± (half spread + impact)` (+ buy, − sell); passive limit fills: at the limit price, no spread, no impact |
| fee | `max(commission_min_eur, |qty · fill| · commission_bps / 1e4)`, EUR |
| `slippage_bps` | signed cost vs the arrival price (last trade when the intent was accepted), positive = paid |
| funding / borrow | `funding_bps_day`, `borrow_bps_day` carried in the schedule, not yet charged (placeholder) |

Without a costs file the built-in default applies (5 bps commission, 1 EUR minimum, 5 bps spread, k 0.1, σ 2 %/day, ADV 10 M EUR). The repo `costs.yaml` makes the synthetic `sim` venue costless so the baseline paper P&L stays readable.

`testdata/cost_fixture.json` is exported from `costs.yaml` by `TestFixtureMatchesModel` (`UPDATE_FIXTURE=1 go test ./services/broker-gateway` regenerates it): the model as JSON plus input → expected `fee`, `fill_price`, `half_spread`, `impact`, `slippage_bps` cases, so `crates/fa-backtest` can assert it prices fills identically.

## Subjects
| Subject | Direction | Payload |
|---|---|---|
| `order.approved.>` | in | `OrderIntent`; non-paper → `REJECTED`, no price seen → `REJECTED`, unmarketable limit → `ACK` (resting) |
| `md.trade.>` | in | `Envelope{Trade}`: mark, σ, volume; fills resting limits that became marketable |
| `md.quote.>` | in | `Envelope{Quote}`: spread for the next fills |
| `order.cancel.<account>` | in | `OrderIntent` carrying the `intent_id` to cancel (provisional: there is no cancel message in `fa.v1`); `CANCELED` event, or ignored when nothing rests |
| `order.event.<account>` | out | `OrderEvent` (`FILL` / `ACK` / `CANCELED` / `REJECTED`) |

Heartbeat counters: `events.<KIND>`, `resting`, `quotes`, `intents.duplicate`, `cancels.unknown`, `errors.decode`.

## Run
```sh
go/bin/broker-gateway -nats nats://127.0.0.1:4222 -costs go/services/broker-gateway/costs.yaml
go test ./services/broker-gateway        # cost model, fixture, limit/cancel paths, estimators
```

## Owns
Venue adapters and exec algos transport.

## Governing specs
- [40-execution-and-risk](../../../specs/40-execution-and-risk.md)
- [41-backtest-and-validation](../../../specs/41-backtest-and-validation.md) §6.3
- [42-ledger-and-accounting](../../../specs/42-ledger-and-accounting.md)
