# 35 — Strategies (Evolving Traversal Algorithms)

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium-low |
| **Owners (code)** | `crates/fa-strategies`, `crates/fa-evolution`, `crates/fa-backtest`, `bins/evo-worker`, `bins/model-runtime`, `go/services/evo-orchestrator` |
| **Depends on** | 32 Expert spaces, 33 MCMC graph, 34 Decoder population, 40 Execution & risk, 41 Backtest & validation, 42 Ledger |

## 1. Purpose
Strategies are the "evolving algorithms traversing the complex-indicator landscape". They read the MCMC state graph, the expert spaces, decoder outputs and their own portfolio state, and emit a **distribution over target exposure** per asset. Two representations, **neural policies** and **genetic programs**, compete in one population under the same Pareto fitness. Promoted strategies feed the meta-allocator (spec 40).

## 2. Scope
**In scope:**
- Strategy representations (GP DSL, neural policy).
- Their inputs and outputs.
- Horizon and asset-set declaration.
- The shared fitness net of costs.
- Population evolution (on the shared engine from spec 34).
- Promotion hand-off to paper/shadow.

**Out of scope:**
- Allocation across strategies, risk limits, execution (spec 40).
- Validation protocol details and promotion gates (spec 41 / 40).
- Decoders (spec 34).

## 3. Decisions
- **Neural policies and genetic programs compete** in one population, selected by the same Pareto fitness.
- **GP primitives** operate on:
  - **graph ops** (current region, neighbours, transition probabilities, k-step walks/samples, path probabilities);
  - **decoder outputs** (forecast quantiles, retrieved analogs, text/visual scores);
  - **portfolio state** (position, P&L, exposure, time-in-trade).
  **No classic TA primitives.**
- **Neural policy output:** **distribution over target exposure** (mean + uncertainty) per asset in its set. The allocator consumes the uncertainty (spec 40).
- Strategies are **multi-horizon** with matched scope: each declares a timescale/horizon (lookback ≈ horizon scale).
- Asset sets are **evolved or MCMC-proposed** (spec 33 §6.5).
- Fitness is **net of costs** (fees, spread, slippage, funding, borrow) and corrected for multiple testing (**deflated Sharpe / PBO**). No walk-forward inside evolution. Locked holdout + live paper shadow before promotion (spec 41).
- Long + short allowed, within the leverage caps enforced downstream (spec 40). Strategies may express negative exposure.

## 4. Interfaces

### 4.1 Common strategy trait (proposal)
```rust
pub trait Strategy: Send + Sync {
    fn id(&self) -> StrategyId;
    fn spec(&self) -> &StrategySpec;           // timescale, horizon, asset set, representation
    /// Called once per timescale step with point-in-time inputs.
    fn decide(&mut self, obs: &StrategyObs) -> ExposureIntent;
}

pub struct StrategySpec {
    pub timescale: Timescale,
    pub horizon: Duration,                     // matched scope with lookback
    pub asset_set: AssetSetId,
    pub representation: Representation,        // Gp | Neural
    pub snapshot_id: SnapshotId,               // encoder snapshot it was evolved on
}

pub struct StrategyObs<'a> {
    pub t: Timestamp,
    pub views: &'a [ExpertView],               // spec 32 (per asset in set)
    pub graph: &'a dyn GraphQuery,             // spec 33 query surface
    pub decoders: &'a dyn DecoderOutputs,      // elite decoder outputs (spec 34), cached per step
    pub portfolio: &'a StrategyPortfolio,      // this strategy's virtual book (spec 42)
}

pub struct ExposureIntent {
    pub t: Timestamp,
    pub strategy: StrategyId,
    pub targets: Vec<ExposureDist>,            // per asset
}

pub struct ExposureDist {
    pub instrument: InstrumentId,
    pub mean: f32,                             // target exposure in [-1, 1] of strategy capital
    pub std: f32,                              // uncertainty (allocator input)
    pub horizon: Duration,
}
```
Published on `signal.exposure.<strategy_id>` as `fa.v1.ExposureIntent` (spec 21), consumed by the meta-allocator (spec 40).

### 4.2 GP DSL (proposal)
Typed, strongly-typed GP (STGP) over expression trees. Programs evaluate once per step and return `(mean, std)` per asset.

**Types:** `Num`, `Prob` (distribution over ℝ or [0,1]), `Region`, `RegionSet`, `Path`, `Bool`, `Dur`, `Asset`.

**Graph-op primitives (spec 33):**
| Primitive | Signature |
|---|---|
| `cur_region(expert)` | `→ Region` |
| `neighbors(r, n)` | `Region → RegionSet` |
| `p_trans(a, b, k)` | `Region, Region, Dur → Prob` |
| `p_to_set(r, S, k)` | `Region, RegionSet, Dur → Prob` |
| `sample_walk(r, k, n)` | `Region, Dur → [Path]` |
| `path_value(paths, fn)` | `[Path], (Region→Num) → Prob` |
| `hitting_time(r, S)` | `Region, RegionSet → Prob` |
| `dwell(r)` | `Region → Dur` (time in current region) |
| `novelty(expert)` | `→ Num` |
| `region_feature(r, f)` | region stats: mass, mean forward return, vol |

**Decoder-output primitives (spec 34):**
| Primitive | Signature |
|---|---|
| `fc_quantile(asset, q, k)` | elite RNN-prior / future-chart forecast quantile |
| `fc_dist(asset, k)` | `→ Prob` |
| `analog_outcome(asset, k)` | retrieval decoder outcome distribution |
| `analog_age(asset)` | mean time distance of analogs |
| `text_score(topic)` | news reconstruction / forecast narrative similarity score |
| `visual_score(kind)` | visual decoder score (future chart direction etc.) |

**Portfolio-state primitives:** `pos(asset)`, `pnl_open(asset)`, `exposure_gross()`, `time_in_trade(asset)`, `drawdown()`.

**Combinators:** arithmetic, `min/max`, `clip`, `if(Bool, a, b)`, comparisons, `mean/std/quantile(Prob)`, `prob_gt(Prob, Num)`, constants (ephemeral random constants). Classic TA (moving averages, RSI, …) is **not** provided.

**Output node:** `exposure(mean: Num, std: Num)` per asset.

Constraints: max depth, max nodes, evaluation-time budget per step (cost objective).

#### 4.2.1 Implemented subset and textual form (D-040)
`crates/fa-strategies` implements the DSL as an AST + interpreter with an **S-expression** text form (`sexpr`) so programs can live in `strategies.toml` files: grammar `atom | (op arg...)`; operators `add sub mul div neg tanh min max if_gt`; primitives `stay_prob`, `(path_up k)`, `region_entropy`, `(quantile q)`, `analog_forecast`, `(score family)`, `position`, `unrealised_pnl`, `time_in_trade`. A strategy entry declares `id`, `genome_id`, `timescale`, `steps` (horizon in bars), `lookback_bars`, optional `step_secs`, and two programs `signal` (→ mean) and `uncertainty` (→ std). Outputs are clipped to `mean ∈ [-1, 1]`, `std ∈ [0, 1]`; NaN → 0 exposure (§8).

**Baseline population:** `crates/fa-strategies/baselines.toml` holds hand-written programs (`genome_id = "baseline"`, **not evolved**) so the signal → allocation → risk → paper-order chain runs before evolution promotes anything (§6.4). `bins/strategy-runtime` runs them live.

**Fallback observation:** until regions, transitions and decoder outputs are consumed, `features::FallbackContext` maps the primitive surface onto empirical price statistics of the strategy's own bar window: `analog_forecast` = mean log return per bar; `(quantile q)` = normal-approximation quantile of lookback returns; `(path_up k)` = share of positive k-bar windows; `stay_prob` = sign persistence (or latent region dwell when `latent.state.>` is present); `region_entropy` = latent posterior std if known, else `std / (|mean| + std)`; transition and decoder-score primitives return 0. This is a placeholder, not the semantics above. Portfolio primitives read the strategy's virtual book (last emitted exposure).

**Since D-053** the primitives read real inputs: `fa_strategies::graph::LatentGraph` (from `latent.state|region|transition.>`, point-in-time, rebuilt per expert space on `fa_mcmc::MarkovGraph`) and `fa_strategies::decoders::DecoderOutputs` (from `decoder.output.<family>`), behind `features::LiveContext`, which falls back to the D-040 price statistics only where data is missing and counts the source per call (`primitives.graph|decoder|fallback` in the runtime heartbeat). The typed spec primitives (`cur_region`, `neighbors`, `p_trans`, `p_to_set`, `hitting_time`, `dwell`, `novelty`, `region_feature`, `path_value`, `fc_quantile`, `fc_dist`, `analog_outcome`, `analog_age`, `text_score`, `visual_score`) are parsed and type-checked by `sexpr`; `text_score`/`visual_score` answer 0 until a producer exists. **Since D-052** a strategy genome (`fa_strategies::genome`, one-line TOML inline table, id = phenotype hash) can be generated at random (`random`), varied (`ops`: typed subtree crossover, point/subtree/hoist mutation, constant jitter) and evaluated by `evo.eval.strategy`.

### 4.3 Neural policy (proposal)
- **Input:** concatenation of expert latents (masked by genome), region membership embeddings, a graph summary vector (k-step predictive region distribution embedded via region embeddings), cached elite decoder outputs (quantiles, analog stats), portfolio state.
- **Architecture genes:** MLP / small transformer over asset tokens (set-equivariant across the asset set), width, depth.
- **Output:** per-asset `(mean = tanh(·), std = softplus(·))`.
- **Training:** hybrid ES + SGD (same as decoders): topology evolved; weights tuned by a differentiable surrogate objective (proposal: maximise a cost-adjusted, risk-penalised log-utility of returns on the training window, using a differentiable cost model) and/or ES on the non-differentiable fitness.

## 5. Data model
| Entity | Store | Fields |
|---|---|---|
| `StrategyGenome` | Helix node | representation, program tree (GP) or topology (neural), spec, generation, status (candidate / holdout-passed / shadow / paper / live / retired), objectives |
| `CHILD_OF`, `READS` (Expert), `USES` (DecoderGenome), `TRADES` (Asset) | Helix edges | |
| Weights (neural) | Garage | |
| Exposure intents | NATS stream `signal.exposure.*`; archived to Garage for replay | |
| Strategy virtual books, P&L series | Postgres ledger (spec 42) + VictoriaMetrics series | |

## 6. Behaviour & algorithms

### 6.1 Evaluation (backtest, spec 41)
- Strategies are evaluated by `fa-backtest` replaying point-in-time inputs (expert views, graph state, decoder outputs) of the **same snapshot** over the evaluation window, at the strategy's timescale.
- **Execution simulation:** exposure intent → target position (strategy capital) → simulated fills with the spread + slippage model, fees, funding/borrow (cost model per venue/asset class from spec 41/42).
- Deterministic given seed + snapshot + data window.
- **Today:** no strategy evaluation job exists. `evo-worker` evaluates decoders only (spec 34 §4.4); strategies run live in `bins/strategy-runtime` from the baseline file and are not scored or evolved.

### 6.2 Fitness (shared Pareto, proposal)
| Objective | Definition | Direction |
|---|---|---|
| Risk-adjusted return | **Deflated Sharpe ratio** net of costs (corrected for the number of trials in the generation and their correlation) | max |
| Drawdown | Max drawdown / CVaR(5%) of daily returns | min |
| Turnover / cost sensitivity | Sharpe degradation under 2× costs | min |
| Calibration | How well `std` predicts realised P&L dispersion (allocator relies on it) | max |
| Compute cost | Evaluation time per step | min |
| Diversification | Correlation of strategy returns with the current elite/live set | min |
Both representations share these objectives, so they compete directly. **PBO** is estimated per generation over the candidate set (combinatorially symmetric cross-validation on evaluation sub-periods). A generation with PBO > `pbo_max` doesn't promote anyone.

### 6.3 Evolution
- Same engine as spec 34 (`fa-evolution`, NSGA-II/III, successive halving by evaluation length, budget-driven).
- GP variation: subtree crossover, point mutation, hoist/shrink mutation, constant perturbation, type-safe.
- Neural variation: topology mutation, weight warm-start + SGD fine-tune, expert-mask flips.
- Cross-representation: populations are separate breeding pools sharing one selection front (no crossover between GP and neural). Quotas keep both alive (`min_share_per_repr`).
- Asset-set genes: mutate by add/remove/swap using MCMC pair proposals as the mutation distribution.

### 6.4 Lifecycle and promotion hand-off
`candidate → holdout-passed → shadow → paper → live → retired`
1. Elite candidates are evaluated **once** on the **locked holdout** (spec 41); failures are retired and the holdout-use counter increments.
2. Passing strategies run in **live paper shadow** for the minimum calendar time (spec 40 gate).
3. Promotion to paper/live is decided by the gate (spec 40). Retirement on gate failure, decay of live performance, or snapshot incompatibility.

**Implemented today (D-052, D-053):** the final Pareto front of a strategy generation is written to Garage as `fa-artifacts/strategies/<generation>.toml` and announced as `artifact_key`; `strategy-runtime` loads it on `ctl.strategy.reload` (validated, swapped atomically, surviving ids keep their virtual books) and honours `ctl.strategy.pause|resume`. Lifecycle status and the promotion gate live in risk-engine (spec 40 §6.6, D-051).

### 6.5 Snapshot dependency
Strategies are tied to the encoder snapshot they were evolved on. On a snapshot change:
- GP programs with region literals are remapped via region lineage (spec 30);
- neural policies are re-fine-tuned;
- both re-enter shadow before trading on the new snapshot.
Live strategies keep running on their snapshot until the replacement passes shadow (runtime serves multiple snapshots, spec 36).

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `gp_max_depth` | 8 | |
| `gp_max_nodes` | 64 | |
| `step_eval_budget` | 1 ms | per strategy per step |
| `min_share_per_repr` | 0.3 | Population share floor for GP / neural |
| `pbo_max` | 0.2 | |
| `dsr_min_candidate` | 0.95 (prob. Sharpe > 0) | for holdout eligibility |
| `cost_stress_mult` | 2.0 | |
| `exposure_clip` | [-1, 1] | per asset, of strategy capital |

## 8. Failure modes
| Failure | Response |
|---|---|
| Overfitting (huge population) | Deflated Sharpe, PBO gate, locked holdout used once per strategy, live shadow |
| Look-ahead via decoder outputs or graph state | All inputs pulled through point-in-time replay. Decoder outputs cached at publish time. Leak tests in spec 41 |
| Strategy emits NaN / out-of-range exposure | Clipped to 0 exposure; strategy flagged; repeated → retired |
| Step time exceeds budget live | Strategy skipped for that step (exposure unchanged); flagged |
| Snapshot or region lineage breaks | Strategy suspended to shadow until revalidated |

## 9. Acceptance criteria
1. GP DSL interpreter evaluates programs type-safely; random program generation and all variation operators produce valid typed programs (property tests).
2. Neural policies and GP programs both implement `Strategy` and emit `ExposureIntent` with mean and std.
3. Backtest of a strategy is deterministic and reproduces identical P&L given snapshot, window and seed.
4. Fitness computation includes costs, and the deflated Sharpe implementation matches reference values on known test vectors.
5. PBO computation runs per generation and blocks promotion when above `pbo_max`.
6. The lifecycle states and transitions are persisted in Helix and visible in the evolution lab and trading desk UI.

## 10. Open questions
1. GP program semantics: should programs be stateful (memory registers across steps) or purely functional on the current observation?
2. Neural policy training signal: differentiable surrogate vs pure ES vs RL (e.g. policy gradients on simulated fills)?
3. How does one fitness compare GP and neural fairly when their compute costs differ by orders of magnitude (normalisation of the cost objective)?
4. Should strategies consume decoder outputs from **specific** elite decoders (genome references) or from an aggregated "best decoder per family" feed?
5. Strategy capital unit: are exposures relative to a notional sleeve the allocator later scales (default) or absolute?
6. Region literals in GP programs depend on spec 30 open question 1 (region identity).

## 11. Deferred
- Market-making / order-book-level strategies (sub-second horizons).
- Strategies consuming raw text directly (only via decoder outputs for now).
- Cross-strategy communication / hierarchical strategies.
