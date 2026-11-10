# fa-allocator

Pluggable meta-allocators, cross-strategy netting and the evolvable allocator genome.

**Status:** implemented (v1) — equal-weight control, risk-parity-capped and fractional-Kelly allocators (`by_name`), the parametric **`WeightedBlend`** family, uncertainty-shrunk netting (`net_targets`, `net_targets_shrunk`), the online per-strategy performance posterior (`PerformanceTracker`) and the **evolvable `genome::AllocatorGenome`** (tested). Used live by `bins/risk-engine` (fixed allocator from `risk.toml`). The **allocator tournament** runs in evo-worker (`evo.eval.allocator` / `evo.vary.allocator`, replay in `fa_backtest::allocator_replay`); runtime selection of a promoted genome by risk-engine is pending on its side (payload below).

## Allocators
| Name | Type | Weights |
|---|---|---|
| `equal_weight` | `EqualWeight` | `min(1/n, cap)` for every strategy with a signal (control, spec 40 §6.2) |
| `risk_parity` | `RiskParityCapped` | inverse volatility over strategies with positive mean, capped |
| `fractional_kelly` | `FractionalKelly` | `f · μ/σ²`, clipped to `[0, cap]`, scaled to Σ ≤ 1 |
| `weighted_blend` | `WeightedBlend` | uncapped legs of the three above mixed `w_eq : w_rp : w_k` (normalised), then capped and scaled to Σ ≤ 1 |

`net_targets(dists, alloc, deployable)` nets `w · deployable / n_assets · mean / (1 + stddev)` per instrument (D-040); `net_targets_shrunk(…, shrink)` uses `(1 + stddev)^shrink` (`0` ignores the uncertainty, `1` = live).

## Allocator genome (`genome`)
Text form (one line, canonical gene order, ≤ 6 decimals; `parse` ↔ `Display` round trip; id `alloc-<fnv64 of the text>`):
```
alloc:name=risk_parity;cap=0.4;shrink=1;kelly=0.25;hl=240;pv=0.01;w_eq=0;w_rp=1;w_k=0
```
| Gene | Meaning | Bounds |
|---|---|---|
| `name` | `equal_weight` \| `risk_parity` \| `fractional_kelly` \| `weighted_blend` | — |
| `cap` | max capital weight per strategy | 0.05 … 1 |
| `shrink` | exponent of the exposure-uncertainty shrink in netting | 0 … 3 |
| `kelly` | Kelly fraction (Kelly allocator and the blend's Kelly leg) | 0.01 … 1 |
| `hl` | performance-tracker half-life in steps (`decay = 0.5^(1/hl)`) | 2 … 100 000 |
| `pv` | prior per-step volatility of a strategy without observations | 1e-5 … 1 |
| `w_eq`, `w_rp`, `w_k` | blend weights (`weighted_blend`) | 0 … 1 |

Missing genes take the defaults of the named allocator (`AllocatorGenome::named`); values are clamped. `build()` returns the allocator, `tracker()` a fresh `PerformanceTracker`, `random(rng)` an immigrant. `AllocatorOps` implements `fa_evolution::vary::Operators` (uniform crossover over name and genes; mutation = one Gaussian gene `gaussian_gene`, all genes `gaussian_all`, or a name switch `name_switch`, σ 0.2 relative); `AllocatorGenome: Evolvable` with one family, so any two allocators may cross.

### Selection payload (risk-engine hand-off)
`AllocatorGenome::spec()` / `from_spec()` convert to `AllocatorSpec {name, params}`, the value risk-engine should accept as `ctl.risk.limits` arg `set` = `{"allocator": {"name": "...", "params": {...}}}` (all params optional; unknown names/params rejected):
```json
{"allocator": {"name": "weighted_blend", "params": {"cap": 0.4, "shrink": 1, "kelly": 0.25, "hl": 240, "pv": 0.01, "w_eq": 0.2, "w_rp": 0.5, "w_k": 0.3}}}
```
The orchestrator writes the final front of an allocator generation to `fa-artifacts/allocators/<generation>.toml` (`[active]` + one `[[allocator]]` per front member with `genome`, `name`, `params`) and prints the `set=` payload of the active member in its header.

## Owns
`Allocator` trait and implementations, netting, `PerformanceTracker`, `AllocatorGenome` / `AllocatorOps` / `AllocatorSpec`.

## Governing specs
- [40-execution-and-risk](../../specs/40-execution-and-risk.md) §3, §6.2, §6.3; decisions D-040
