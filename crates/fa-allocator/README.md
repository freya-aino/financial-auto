# fa-allocator

Pluggable meta-allocators, cross-strategy netting and the evolvable allocator genome.

**Status:** implemented (v1) — equal-weight control, risk-parity-capped, fractional-Kelly and the Bayesian bandit **`ThompsonRiskParity`** (`thompson_rp`) allocators (`by_name`), the parametric **`WeightedBlend`** family, uncertainty-shrunk netting (`net_targets`, `net_targets_shrunk`), the online per-strategy performance posterior (`PerformanceTracker`), the **evolvable `genome::AllocatorGenome`** and the **live paper tournament** (`tournament`) (tested). Used live by `bins/risk-engine`: the allocator starts from `risk.toml` (`equal_weight`) and can be switched at runtime to a promoted genome (payload below); every tick the engine also runs the tournament. The offline **allocator tournament** runs in evo-worker (`evo.eval.allocator` / `evo.vary.allocator`, replay in `fa_backtest::allocator_replay`).

## Allocators
| Name | Type | Weights |
|---|---|---|
| `equal_weight` | `EqualWeight` | `min(1/n, cap)` for every strategy with a signal (control, spec 40 §6.2) |
| `risk_parity` | `RiskParityCapped` | inverse volatility over strategies with positive mean, capped |
| `fractional_kelly` | `FractionalKelly` | `f · μ/σ²`, clipped to `[0, cap]`, scaled to Σ ≤ 1 |
| `weighted_blend` | `WeightedBlend` | uncapped legs of the three above mixed `w_eq : w_rp : w_k` (normalised), then capped and scaled to Σ ≤ 1 |
| `thompson_rp` | `ThompsonRiskParity` | one Thompson draw per strategy from `N(mean, vol²/max(n, 1))`; positive draws share capital by inverse volatility, capped (Bayesian bandit + risk parity, spec 40 §6.2); deterministic per seed and call sequence (the genome form seeds from its text) |

`net_targets(dists, alloc, deployable)` nets `w · deployable / n_assets · mean / (1 + stddev)` per instrument (D-040); `net_targets_shrunk(…, shrink)` uses `(1 + stddev)^shrink` (`0` ignores the uncertainty, `1` = live).

**Fill attribution of netted orders** (spec 40 §10 Q3, causal pro-rata): `attribution_weights(Δ, [(c_s, h_s)])` weights each contributor by `max(0, sign(Δ)·(c_s − h_s))` — its own signed EUR contribution to the target minus what its sub-book already holds, in the order's direction — falling back to `|c_s|`, then to an even split; `attribution_shares(n, weights)` normalises them (even when missing). The risk-engine attributes fills to sub-books with them and the ledger stores the shares (`PgLedger::record_fill_weighted`).

## Allocator genome (`genome`)
Text form (one line, canonical gene order, ≤ 6 decimals; `parse` ↔ `Display` round trip; id `alloc-<fnv64 of the text>`):
```
alloc:name=risk_parity;cap=0.4;shrink=1;kelly=0.25;hl=240;pv=0.01;w_eq=0;w_rp=1;w_k=0
```
| Gene | Meaning | Bounds |
|---|---|---|
| `name` | `equal_weight` \| `risk_parity` \| `fractional_kelly` \| `weighted_blend` \| `thompson_rp` | — |
| `cap` | max capital weight per strategy | 0.05 … 1 |
| `shrink` | exponent of the exposure-uncertainty shrink in netting | 0 … 3 |
| `kelly` | Kelly fraction (Kelly allocator and the blend's Kelly leg) | 0.01 … 1 |
| `hl` | performance-tracker half-life in steps (`decay = 0.5^(1/hl)`) | 2 … 100 000 |
| `pv` | prior per-step volatility of a strategy without observations | 1e-5 … 1 |
| `w_eq`, `w_rp`, `w_k` | blend weights (`weighted_blend`) | 0 … 1 |

Missing genes take the defaults of the named allocator (`AllocatorGenome::named`); values are clamped. `build()` returns the allocator, `tracker()` a fresh `PerformanceTracker`, `random(rng)` an immigrant. `AllocatorOps` implements `fa_evolution::vary::Operators` (uniform crossover over name and genes; mutation = one Gaussian gene `gaussian_gene`, all genes `gaussian_all`, or a name switch `name_switch`, σ 0.2 relative); `AllocatorGenome: Evolvable` with one family, so any two allocators may cross.

### Selection payload (risk-engine hand-off)
`AllocatorGenome::spec()` / `from_spec()` convert to `AllocatorSpec {name, params}`, the value risk-engine accepts as `ctl.risk.limits` arg `set` = `{"allocator": {"name": "...", "params": {...}}}` (all params optional; unknown names/params rejected):
```json
{"allocator": {"name": "weighted_blend", "params": {"cap": 0.4, "shrink": 1, "kelly": 0.25, "hl": 240, "pv": 0.01, "w_eq": 0.2, "w_rp": 0.5, "w_k": 0.3}}}
```
risk-engine also takes `{"allocator": {"genome": "alloc:…"}}`; the switch is audited (`allocator.changed`) and not persisted (a restart returns to `risk.toml`). The orchestrator writes the final front of an allocator generation to `fa-artifacts/allocators/<generation>.toml` (`[active]` + one `[[allocator]]` per front member with `genome`, `name`, `params`) and prints the `set=` payload of the active member in its header.

## Live tournament (`tournament`)
`Tournament` allocates every entry on the same strategy posteriors each tick; an entry's virtual portfolio earns `Σ_s w_s(t−1) · r_s(t)` with `r_s` the strategy's sleeve return (`PerformanceTracker::step_return`) and pays `cost_bps` on weight turnover (a stand-in for the netted execution cost the offline replay models exactly). The active allocator can be fed in as an external entry. `EntryReport`: cumulative return, Sharpe, max drawdown, turnover, costs, weights. risk-engine runs `equal_weight`, `risk_parity`, `fractional_kelly`, `thompson_rp` and the active allocator (`tournament_cost_bps` 5) and reports it on `ctl.risk.allocators`.

## Owns
`Allocator` trait and implementations, netting, `PerformanceTracker`, `AllocatorGenome` / `AllocatorOps` / `AllocatorSpec`, live `Tournament`.

## Governing specs
- Retired spec 40 → [risk-engine README](../../bins/risk-engine/README.md#execution-and-risk-lasting-content-of-the-retired-spec-40) (decisions, netting attribution); decisions D-040, D-064, D-071
