# tsdb

Writes canonical market events to VictoriaMetrics through the JSON line import API (`POST /api/v1/import`).

**Status:** implemented — `Writer` batches `Point`s (500 points or 1 s, bounded buffer with drop counter) and `TradePoints` maps a `Trade` envelope to `fa_trade_price` / `fa_trade_size` with labels `inst` (UUID without dashes), `venue`, `src` at `event_time` in ms (spec 20 §5.3). Quotes, bars and books are not written yet.

## Owns
Go side of the VictoriaMetrics write path (the Rust read side is `crates/fa-tsdb`).

## Governing specs
- [20-storage](../../../specs/20-storage.md)
