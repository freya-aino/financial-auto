# tsdb

Writes canonical market events to VictoriaMetrics through the JSON line import API (`POST /api/v1/import`).

**Status:** implemented — `Writer` batches `Point`s (500 points or 1 s, bounded buffer with drop counter); `Points(env, venue)` maps every market-data envelope with the spec 20 §5.3 names and labels `inst` (UUID without dashes), `venue`, `src` at `event_time` in ms:

| Payload | Metrics | Extra label |
|---|---|---|
| `Trade` | `fa_trade_price`, `fa_trade_size` | |
| `Quote` | `fa_quote_{bid,ask}_{px,sz}` | |
| `Bar` | `fa_bar_{open,high,low,close,volume}` (+ `fa_bar_vwap` when set) | `tf` |
| `BookSnapshot` | `fa_book_{bid,ask}_{px,sz}` per level (≤ 20) | `lvl` |

## Owns
Go side of the VictoriaMetrics write path (the Rust read side is `crates/fa-tsdb`).

## Governing specs
- [20-storage](../../../specs/20-storage.md)
