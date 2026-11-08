# broker-gateway

Execution adapters: order intents → venues; paper mode first.

**Status:** implemented (paper only) — fills `order.approved.>` intents at the last `md.trade.>` price, publishes `OrderEvent` on `order.event.<account>`, idempotent by `intent_id`; fees come from a `FeeModel` hook (spec 41 cost model) — `ZeroFee` by default, `BpsFee` when `-fee-bps`/`-fee-min` are set; slippage is still zero. Live mode is rejected; `VenueAdapter` interface stubbed.

## Run
```sh
go/bin/broker-gateway -nats nats://127.0.0.1:4222            # zero fees
go/bin/broker-gateway -fee-bps 5 -fee-min 1                  # 5 bps of notional, min 1 EUR per fill
```

## Owns
Venue adapters and exec algos transport.

## Governing specs
- [40-execution-and-risk](../../../specs/40-execution-and-risk.md)
- [42-ledger-and-accounting](../../../specs/42-ledger-and-accounting.md)
