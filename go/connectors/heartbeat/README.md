# heartbeat

Synthetic connector that proves the ingestion path end to end (D-018). It emits one random-walk `Trade` per second for the fixed simulated instrument `01900000-0000-7000-8000-000000000001` on venue `sim`, so events land on `md.trade.sim.01900000000070008000000000000001`.

**Status:** implemented. It isn't a catalog source, so the ingestor runs it as a built-in.

## Governing specs
- [10-ingestion](../../../specs/10-ingestion.md)
- [21-bus-and-contracts](../../../specs/21-bus-and-contracts.md)
