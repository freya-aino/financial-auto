# tsdb

Writes canonical market events to VictoriaMetrics through the JSON line import API (`POST /api/v1/import`).

**Status:** implemented — `Writer` batches `Point`s (500 points or 1 s, bounded buffer with drop counter); `Points(env, venue)` maps every market-data envelope with the spec 20 §5.3 names and labels `inst` (UUID without dashes), `venue`, `src` at `event_time` in ms:

| Payload | Metrics | Extra label |
|---|---|---|
| `Trade` | `fa_trade_price` (VWAP), `fa_trade_size` (Σ), `fa_trade_count` | |
| `Quote` | `fa_quote_{bid,ask}_{px,sz}` | |
| `Bar` | `fa_bar_{open,high,low,close,volume}` (+ `fa_bar_vwap` when set) | `tf` |
| `BookSnapshot` | `fa_book_{bid,ask}_{px,sz}` per level (≤ 20) | `lvl` |

**Same-millisecond trades (D-038):** `AddEnvelope` routes trades through `AddTrade`, which keeps the newest millisecond of each `(inst, venue, src)` series open (until a newer millisecond arrives or `Options.TradeHold`, default the flush interval, passes) and writes it as one sample: price = VWAP Σp·s/Σs (last price when Σs = 0), size = Σs, count = trades. A trade older than the newest written millisecond of its series is written unaggregated. `Stats.Aggregated` / `Stats.Late` → ingestor heartbeat `tsdb.aggregated` / `tsdb.late`.

**Tests:** unit tests with an `httptest` VictoriaMetrics; `FA_VM_TEST=1 FA_VM_URL=http://victoriametrics:8428 go test ./pkg/tsdb -run Live` round-trips three same-millisecond trades through a real VictoriaMetrics (unique `src` label, deleted again afterwards).

## Owns
Go side of the VictoriaMetrics write path (the Rust read side is `crates/fa-tsdb`).

## Governing specs
- [20-storage](../../../specs/20-storage.md)
