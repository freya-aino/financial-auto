# broker-gateway

Execution adapters: order intents → venues; paper mode first.

**Status:** implemented (paper only) — fills `order.approved.>` intents at the last `md.trade.>` price, publishes `OrderEvent` on `order.event.<account>`, idempotent by `intent_id`; zero fees until the cost model (spec 41). Live mode is rejected; `VenueAdapter` interface stubbed.

## Owns
Venue adapters and exec algos transport.

## Governing specs
- [40-execution-and-risk](../../../specs/40-execution-and-risk.md)
- [42-ledger-and-accounting](../../../specs/42-ledger-and-accounting.md)
