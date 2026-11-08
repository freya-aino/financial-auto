# fa-allocator

Pluggable meta-allocators and cross-strategy netting.

**Status:** implemented (v1) — equal-weight control, risk-parity-capped and fractional-Kelly allocators (`by_name`), uncertainty-shrunk netting, online per-strategy performance posterior (`PerformanceTracker`) (tested). Used live by `bins/risk-engine`. Allocator tournament and evolved allocators pending.

## Owns
`Allocator` trait and implementations.

## Governing specs
- [40-execution-and-risk](../../specs/40-execution-and-risk.md)
