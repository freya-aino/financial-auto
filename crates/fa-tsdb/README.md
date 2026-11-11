# fa-tsdb

VictoriaMetrics read/write: series naming, Prometheus-line import, `query_range`, trade-price reader for decoder evaluation.

**Status:** implemented — label-whitelisted import, `query_range`, `trade_prices` (used by `evo-worker` as the primary evaluation data source, spec 41 §5), `health`, `force_flush`. Formatting/parsing unit-tested; **live round trip tested against VictoriaMetrics** with `FA_VM_TEST=1`.

## Conventions (match `go/pkg/tsdb`, the ingestor's writer)
- Metrics `fa_trade_price`, `fa_trade_size`, `fa_trade_count` (trades folded into one same-millisecond VWAP sample by the Go writer, D-066), `fa_bar_*`, `fa_feat_<name>`; labels only from `ALLOWED_LABELS` (`inst`, `venue`, `tf`, `src`, `lvl`, `feat_set`).
- `inst` = instrument UUID as D-017 subject token: lowercase, no dashes — `inst_token()` is `bus.SubjectToken` of `go/pkg/bus`. `trade_prices` accepts a dashed UUID or a token.
- Timescale step strings via `step_of(Timescale)` (`1s`, `1m`, `1h`, `1d`, `1w`).

## API
| Call | What |
|---|---|
| `Tsdb::new(url)` | client with 3 s connect / 20 s request timeout (unreachable VM fails fast) |
| `write(&[Sample])` | `/api/v1/import/prometheus` |
| `query_range(query, start_s, end_s, step)` | raw matrix result |
| `trade_prices(inst, start_s, end_s, step)` | `last_over_time(fa_trade_price{inst="…"}[step])` → `TradeSeries { inst, venue, src, points: (ts_ms, price) }`, longest first; buckets without trades are absent (no stale fill) |
| `trade_prices_ms(inst, start_ms, end_ms, bar)` | `trade_prices` over the buckets containing both ends, merged across venues into one ascending `(ts_ms, price)` series (strategy-runtime analog outcomes, D-063) |
| `bars(inst, tf, start_s, end_s)` | connector bars: `last_over_time(fa_bar_<x>{inst="…",tf="<tf>"}[tf])` at `tf` resolution for open/high/low/close/volume/vwap, joined per bucket (`join_bar_series`) → `BarSeries { inst, venue, src, tf, points: BarPoint{ts_ms, open, high, low, close, volume, vwap?} }`, longest first; a bucket needs O/H/L/C (volume defaults to 0); `inst = None` reads every instrument (the trainer's `instruments: "all"`) |
| `health()`, `force_flush()` | `/health`, `/internal/force_flush` (tests) |
| `bar_window_s(start_ms, end_ms, bar)` | whole-second window covering the bar buckets that contain both ends |

`MAX_POINTS_PER_SERIES` (20 000) is the window cap callers should respect (VM rejects > 30 000 points per series).

**Not here (spec 20 §9 Remaining):** reads have no `as_of` — VictoriaMetrics stores event time only, and backtests shift series by the per-source `knowledge_delay` (`fa_backtest::knowledge`); there is no active-series budget or per-timescale retention (VM runs with `-retentionPeriod=100y`). The label whitelist is the only cardinality guard.

## Tests
```sh
cargo test -p fa-tsdb                         # unit tests; the live test prints "skipped"
FA_VM_TEST=1 cargo test -p fa-tsdb            # + live: writes a series under a fresh dashed UUID, reads it back by UUID and by token
FA_VM_URL=http://host:8428 FA_VM_TEST=1 ...   # non-default VM
```
`live_vm_bars_write_read` writes 30 one-minute OHLCV+vwap bars under its own venue, reads them back by UUID and by token through `bars` (values, spacing, label mapping of the ingestor's Yahoo bars) and deletes them. The live trade test also asserts every `fa_trade_price` series the ingestor wrote carries a 32-hex dashless `inst`, and deletes its own test series afterwards.

## Owns
Series naming rules, import, range queries, trade-price and bar readers.

## Governing specs
- [20-storage](../../specs/20-storage.md), [41-backtest-and-validation](../../specs/41-backtest-and-validation.md) §5; decision D-038 (ingestor writes `fa_trade_price`).
