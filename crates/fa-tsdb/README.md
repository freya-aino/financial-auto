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

## Point-in-time reads (`asof`, spec 20 §6.3, Q4)
VictoriaMetrics stores event time only. A sample counts as known at `event time + knowledge delay` of its source (the delays measured into `config/sources.yaml`, `fa_backtest::knowledge::KnowledgeDelays`). `AsOf { as_of_ms, delays_ms, default_delay_ms }` (delay matched on the series' `src`, then `venue`) drives `trade_prices_as_of`, `bars_as_of` and `trade_prices_ms_as_of`: the query end is clamped to the cut and every bucket whose time + delay is after it is dropped — a `last_over_time(..[step])` bucket only holds samples with event time ≤ its timestamp, so nothing published after the cut leaks in. There is deliberately no availability index: exact per-event availability lives in the raw archive and the recorded bus (`fa_bus::replay`). Property test `as_of_never_returns_late_publications` (200 random fixtures, probes around every publication instant); live `live_as_of_hides_unpublished_bars`.

## Retention and disk budget (`retention`, spec 20 §6.5, §8)
Open-source single-node VictoriaMetrics has one global `-retentionPeriod` (compose: 100 y); `-retentionFilter` is enterprise-only and `delete_series` removes whole series. `retention::enforce` therefore **rewrites** each series holding samples older than its class horizon: export the samples at or after the cutoff, spool them to a file, delete the series, import the spool, remove it (a spool left by a failed import is imported first on the next run, `replay_spool`). Samples written to a series between its export and delete (seconds) are lost from VM but stay in the raw archive — run it off-hours.

| Class | Selector | Horizon | Tightening priority / floor |
|---|---|---|---|
| `book` | `fa_book_*` | 30 d | 0 (first) / 3 d |
| `seconds` | `fa_trade_*`, `fa_quote_*`, `fa_bar_*{tf=~"[0-9]+s"}` | 90 d | 1 / 14 d |
| `minute_bars` | `fa_bar_*{tf=~"[0-9]+m"}` | 2 y | 2 / 90 d |
| hour/day/week bars, `fa_feat_*`, ops metrics | — | VM global (100 y) | never |

`tighten(policy, used, budget)` (§8 "retention tightened automatically"): over budget, halve the horizon of the lowest-priority class still above its floor — one step per run, so a daily job keeps tightening while VM stays over budget. `Tsdb::{series, export, import_jsonl, delete_series, vm_metric, active_series, data_size_bytes}` are the admin calls it uses. Job: `cargo run -p fa-tsdb --example vm_retention [-- --apply] [--policy book=30d,seconds=90d,minute_bars=730d] [--disk-budget-gib 50] [--url …]` (dry run by default; JSON report with `active_series`, `data_bytes`, `tightened`, per-class series / samples dropped / kept). Live dry run 2026-11-11: nothing past any horizon (data starts 2026-11-10).

**Series budget:** enforced by the ingestor's writer (`go/pkg/tsdb`, the only bulk market writer); this crate's `write` is used by tests and tools only. `active_series()` reads VM's hour count (`vm_cache_entries{type="storage/hour_metric_ids"}`) — 7 045 on 2026-11-11 (12 444 series in total, 82 of them market series) against the 5 M budget.

## Tests
```sh
cargo test -p fa-tsdb                         # unit tests; the live test prints "skipped"
FA_VM_TEST=1 cargo test -p fa-tsdb            # + live: writes a series under a fresh dashed UUID, reads it back by UUID and by token
FA_VM_URL=http://host:8428 FA_VM_TEST=1 ...   # non-default VM
# retention rewrites series, so it runs only against a scratch VM (refuses port 8428):
docker run -d --rm --name fa-vm-scratch -p 127.0.0.1:18429:8428 --tmpfs /storage victoriametrics/victoria-metrics:v1.155.0 -storageDataPath=/storage -retentionPeriod=100y
FA_VM_SCRATCH_URL=http://127.0.0.1:18429 cargo test -p fa-tsdb scratch_retention
```
`live_vm_bars_write_read` writes 30 one-minute OHLCV+vwap bars under its own venue, reads them back by UUID and by token through `bars` (values, spacing, label mapping of the ingestor's Yahoo bars) and deletes them. The live trade test also asserts every `fa_trade_price` series the ingestor wrote carries a 32-hex dashless `inst`, and deletes its own test series afterwards.

## Owns
Series naming rules, import, range queries, trade-price and bar readers, as-of reads, VM retention / disk-budget job.

## Governing specs
- [20-storage](../../specs/20-storage.md), [41-backtest-and-validation](../../specs/41-backtest-and-validation.md) §5; decision D-038 (ingestor writes `fa_trade_price`).
