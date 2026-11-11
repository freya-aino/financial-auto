# 35 — Strategies (Evolving Traversal Algorithms)

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium-low |
| **Owners (code)** | `crates/fa-strategies`, `crates/fa-evolution`, `crates/fa-backtest`, `bins/strategy-runtime`, `bins/evo-worker`, `bins/risk-engine` (lifecycle pipeline, `fa_risk::pipeline`), `bins/model-runtime`, `go/services/evo-orchestrator` |
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

**Implemented form:** `fa_strategies::Strategy { id, horizon, step(as_of_ms, assets: &[(InstrumentId, &dyn StrategyContext)]) -> ExposureDistribution, ops_per_asset, genome_id }` with two implementations: `GpStrategy` (`mean = tanh(signal)`, `stddev = sigmoid(uncertainty)`) and `neural::NeuralStrategy` (§4.3). The spec genes (`timescale`, `steps`, `lookback_bars`, `step_secs`, optional `snapshot`, the **asset set** `assets`, and `policy` for a neural genome) live in `population::StrategyDef` / `genome::StrategyGenome` (`StrategyDef::build_dyn` builds either representation); the bus message is `fa.v1.ExposureDistribution` (`proto/fa/v1/strategy.proto`).

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

**Fallback observation:** where no graph or decoder data exists for the step, `features::FallbackContext` maps the primitive surface onto empirical price statistics of the strategy's own bar window: `analog_forecast` = mean log return per bar; `(quantile q)` = normal-approximation quantile of lookback returns; `(path_up k)` = share of positive k-bar windows; `stay_prob` = sign persistence (or latent region dwell when `latent.state.>` is present); `region_entropy` = latent posterior std if known, else `std / (|mean| + std)`; transition and decoder-score primitives return 0. This is a placeholder, not the semantics above; since D-053 it answers only where real data is missing. Portfolio primitives read the strategy's virtual book (last emitted exposure).

**Since D-053** the primitives read real inputs: `fa_strategies::graph::LatentGraph` (from `latent.state|region|transition.>`, point-in-time, rebuilt per expert space on `fa_mcmc::MarkovGraph`) and `fa_strategies::decoders::DecoderOutputs` (from `decoder.output.<family>`), behind `features::LiveContext`, which falls back to the D-040 price statistics only where data is missing and counts the source per call (`primitives.graph|decoder|fallback` in the runtime heartbeat). The typed spec primitives (`cur_region`, `neighbors`, `p_trans`, `p_to_set`, `hitting_time`, `dwell`, `novelty`, `region_feature`, `path_value`, `fc_quantile`, `fc_dist`, `analog_outcome`, `analog_age`, `text_score`, `visual_score`) are parsed and type-checked by `sexpr`; `text_score`/`visual_score` read model-runtime's `text.*` / `visual.*` decoders since D-063 (0 when nothing fresh). The portfolio primitives are `position`, `unrealised_pnl`, `time_in_trade`, `exposure_gross` (Σ |exposure| over the strategy's set) and `drawdown` (its virtual equity's drawdown, sleeve-return units; `features::PortfolioSummary` / `StrategyEquity`, the same in the runtime and the replay). Combinators: the D-040 operator set plus `(clip x lo hi)`, the `Bool`-typed `(gt a b)`, `(lt a b)`, `(and p q)`, `(or p q)`, `(not p)` with the conditional `(if p a b)` (STGP: a `Bool` is only valid as a condition, a program must be `Num`; crossover, mutation and hoist preserve types), and `(prob_gt P x)` over the `Prob` forms `(fc_dist k)` (piecewise-linear CDF through the forecast quantiles) and `(analog_outcome k)` (normal around the analog outcome with the forecast's inter-quartile spread). **Since D-052** a strategy genome (`fa_strategies::genome`, one-line TOML inline table, id = phenotype hash) can be generated at random (`random`), varied (`ops`: typed subtree crossover, point/subtree/hoist mutation, constant jitter) and evaluated by `evo.eval.strategy`.

### 4.3 Neural policy (proposal)
- **Input:** concatenation of expert latents (masked by genome), region membership embeddings, a graph summary vector (k-step predictive region distribution embedded via region embeddings), cached elite decoder outputs (quantiles, analog stats), portfolio state.
- **Architecture genes:** MLP / small transformer over asset tokens (set-equivariant across the asset set), width, depth.
- **Output:** per-asset `(mean = tanh(·), std = softplus(·))`.
- **Training:** hybrid ES + SGD (same as decoders): topology evolved; weights tuned by a differentiable surrogate objective (proposal: maximise a cost-adjusted, risk-penalised log-utility of returns on the training window, using a differentiable cost model) and/or ES on the non-differentiable fitness.

**Implemented (`fa_strategies::neural`):** `PolicyNet` = a shared-weight MLP applied per asset (set-equivariant; the strategy-wide state reaches it through `exposure_gross` / `drawdown`), inputs = a genome-masked subset of 20 observation features (the GP primitives: graph, decoder, portfolio, `prob_up`) plus the first `latent_dims` (≤ 8) latent-mean dims of the masked experts (`LatentGraph` keeps them per state; `StrategyContext::latent`), 1–3 hidden layers of 2–64 units (`tanh` / `relu`), heads `mean = tanh(o₀)`, `std = min(1, softplus(o₁))`. Inference is plain `f32` Rust (live and replay); the text form `mlp:f=…;e=…;l=…;h=16x8;a=tanh;w=<base64 f32>` stores the weights exactly (genome id `nn-<hash>` of the phenotype). Training (feature `train`, candle CPU, forward pinned to the plain one by a test): `neural::train::fine_tune` maximises `mean[ln(1 + L·m·r) − c·|Δm|] − w_s·mean(s − ŝ)²` on recorded inputs and horizon returns, keeping the weights with the best held-out (newest 25 %) utility. Variation: ES weight perturbation, function-preserving widening / input switches, narrowing, layer insert/drop, arithmetic crossover within one topology, spec genes, and the SGD fine-tune (evo-worker `neural_policy`: replay of a training window that ends one embargo before the selection window, holdout-checked). Weights travel in the genome text, not Garage (a few KB).

## 5. Data model
| Entity | Store | Fields |
|---|---|---|
| `StrategyGenome` | Helix node | representation, program tree (GP) or topology (neural), spec, generation, status (candidate / holdout-passed / shadow / paper / live / retired), objectives |
| `CHILD_OF`, `READS` (Expert), `USES` (DecoderGenome), `TRADES` (Asset) | Helix edges | |
| Weights (neural) | Garage | |
| Exposure intents | NATS stream `signal.exposure.*`; archived to Garage for replay | |
| Strategy virtual books, P&L series | Postgres ledger (spec 42) + VictoriaMetrics series | |

**Implemented (Helix status mirror):** evolved strategies are Helix `Genome{kind: strategy, family: gp}` nodes (written by the evo-worker); the risk-engine pipeline mirrors each one's lifecycle status from `ledger.strategy_state` onto it at the end of every round (`fa_hub::set_genome_status`: properties `status` = `candidate|holdout_passed|shadow|active|paused|flattening|retired`, `strategy_id`, `status_reason`, `status_since_ms`, `status_updated_at`; a genome the worker never wrote is created with the genome text as `params`). Postgres stays the source of truth: the mirror is best-effort, reconciles every evolved row against the last successful write each round (operator actions and drawdown pauses included, within one `interval_secs`), stops trying for the round once Helix is unreachable and is reported in `ctl.risk.pipeline` status (`helix_mirror`) and the heartbeat (`helix.mirror_written|failed`, `helix.mirrored`). Hand-written baselines are not genomes (D-040) and get no node. Paper/live map onto `shadow`/`active` (D-029).

**Implemented (exposure intents):** replays read them from the retained `SIGNAL` JetStream stream (`fa_bus::replay`, the allocator replay and the decision replay); they are not archived to Garage.

## 6. Behaviour & algorithms

### 6.1 Evaluation (backtest, spec 41)
- Strategies are evaluated by `fa-backtest` replaying point-in-time inputs (expert views, graph state, decoder outputs) of the **same snapshot** over the evaluation window, at the strategy's timescale.
- **Execution simulation:** exposure intent → target position (strategy capital) → simulated fills with the spread + slippage model, fees, funding/borrow (cost model per venue/asset class from spec 41/42).
- Deterministic given seed + snapshot + data window.
- **Live execution of strategy signals is cost-aware (spec 40 §6.1):** a strategy's exposure changes reach the venue only when the edge it has earned (tracked step returns, shrunk toward a 20 bp prior) pays for the full trade cost, its orders are at least the fee-minimum notional, and its traded notional stays within 20 × its sleeve per UTC day (over budget its last distribution is held). A 1 s strategy whose targets jitter by a few hundred EUR therefore holds its position instead of paying 30–50 bp per rebalance. **Fitness sees the live rule:** the per-genome strategy replay behind `evo.eval.strategy` (and the holdout replay) and the allocator tournament (`evo.eval.allocator`) trade with the same `Rebalancer` (`fa_backtest::trade_rule::TradeRule::CostAware`), parsed by evo-worker from the risk-engine's `risk.toml` (`no_trade_band_frac`, `min_order_eur`, `performance_decay`, `[rebalance]`) and priced with `costs.yaml`. In the strategy replay the genome is the only contributor of each sleeve: its edge comes from its own tracked step returns, the turnover budget is against the whole replay capital, and `confirm_secs` counts as met by one bar when the bar is at least that long (live re-decides every second). `FA_EVO_TRADE_RULE=fraction` / envelope `"trade_rule": "fraction"` reproduce the old rule (any move above `min_trade_fraction` of the sleeve); trials scored before and after the switch are not comparable (fewer trades, lower cost, different Sharpe), so a running generation should be restarted rather than mixed. Event-time replays shift late-publishing sources by their catalog knowledge delay (spec 41 §6.2), so fitness is not computed on prices the live strategy could not yet see; strategy decisions → intents are reproducible from the retained streams (spec 41 §9).
- **Implemented (D-052, D-059):** `evo.eval.strategy` replays a genome through `fa_backtest::replay` with the recorded latent graph and decoder outputs and the shared cost model; every evaluation is a `validation.trial` row. Strategies run live in `bins/strategy-runtime`: the hand-written baselines plus the managed live set of evolved strategies that passed the locked holdout (§6.4).

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

**Implemented (both representations, `fa_backtest::replay::DIRECTIONS` + evo-worker):** `deflated_sharpe` (max), `sharpe` (max), `max_drawdown` (min), `cvar` = CVaR(5 %) of daily returns as a positive loss (per step when the window spans < 20 days, `replay::cvar`) (min), `turnover` (min), `cost_sensitivity` = Sharpe − Sharpe under 2× costs (min), `calibration` = corr(emitted std, next |return|) (max), `compute_cost` = log10 estimated operations per step (min, §10 Q3), `diversification` = mean correlation of the candidate's 64 block returns with the generation's current elites (best 8 other candidates by deflated Sharpe whose trial window overlaps ≥ 90 %; `validation::diversification`, 0 without references) (min).

### 6.3 Evolution
- Same engine as spec 34 (`fa-evolution`, NSGA-II/III, successive halving by evaluation length, budget-driven).
- GP variation: subtree crossover, point mutation, hoist/shrink mutation, constant perturbation, type-safe.
- Neural variation: topology mutation, weight warm-start + SGD fine-tune, expert-mask flips.
- Cross-representation: populations are separate breeding pools sharing one selection front (no crossover between GP and neural). Quotas keep both alive (`min_share_per_repr`).
- Asset-set genes: mutate by add/remove/swap using MCMC pair proposals as the mutation distribution.

**Implemented (D-052, breeding pools):** `evo.vary.strategy` (protocol v1) splits the parents by representation and breeds each pool through `fa_evolution::vary::vary_with` — GP with `fa_strategies::ops`, neural with the §4.3 operators — never across pools; the orchestrator's NSGA-II front stays shared. Pool sizes follow the parents' shares clamped to `[min_share_per_repr, 1 − min_share_per_repr]` (0.3), a pool without parents is refilled with random immigrants of its representation (`random::random_neural_genome`), plus `immigrant_frac` immigrants per pool; `sgd_prob` (0.3) of the neural children are fine-tuned. **Asset-set genes:** `StrategyGenome::assets` (sorted UUIDs, empty = every instrument; part of the phenotype id), mutated with `asset_mutation_prob` (0.15) by `fa_strategies::assets::mutate_assets` — unrestricted → a proposed pair ∝ score, add a partner ∝ pair support, remove the least-supported member, swap a member for a partner — over the latest `latent.crossprior.<ts>` proposals (spec 33 §6.5); the replay (`ReplaySpec.instruments`), the holdout replay and strategy-runtime trade only the set.

### 6.4 Lifecycle and promotion hand-off
`candidate → holdout-passed → shadow → paper → live → retired`
1. Elite candidates are evaluated **once** on the **locked holdout** (spec 41); failures are retired and the holdout-use counter increments.
2. Passing strategies run in **live paper shadow** for the minimum calendar time (spec 40 gate).
3. Promotion to paper/live is decided by the gate (spec 40). Retirement on gate failure, decay of live performance, or snapshot incompatibility.

**Implemented today (D-052, D-053):** the final Pareto front of a strategy generation is written to Garage as `fa-artifacts/strategies/<generation>.toml` and announced as `artifact_key`; `strategy-runtime` loads it on `ctl.strategy.reload` (validated, swapped atomically, surviving ids keep their virtual books) and honours `ctl.strategy.pause|resume`. Lifecycle status and the promotion gate live in risk-engine (spec 40 §6.6, D-051).

**Since D-063:** `text_score` / `visual_score` read model-runtime's text and visual decoders, forecasts are chosen by horizon (`h:<steps>` refs), analog outcomes outside the in-memory history are priced from VictoriaMetrics between steps, and the runtime enforces spec 35 §8: step-budget and NaN streaks auto-pause a strategy, starved primitives are flagged. The population can be loaded from Garage (`latest`) at start.

#### 6.4.1 Closed loop (implemented)
The lifecycle runs automatically in risk-engine (`fa_risk::pipeline` decides, `bins/risk-engine/src/pipeline.rs` does the I/O), every `[pipeline] interval_secs` (60) and on `ctl.risk.pipeline {action: run}`; status on `ctl.risk.pipeline {action: status}`. Paper only (D-029).

| Step | Rule | State |
|---|---|---|
| Discovery | newest `generations_scanned` files `fa-artifacts/strategies/<generation>.toml` (the orchestrator's final fronts); each member gets its highest-rung selection trial (`validation.trial`) and the generation's PBO (`validation.generation_stats`, latest loop) | — |
| Selection | generation PBO known and ≤ `pbo_max`, member deflated Sharpe ≥ `dsr_min_candidate`, best `max_candidates_per_generation` | `candidate` |
| Holdout | exactly one `evo.eval.strategy.holdout` job on the newest locked holdout of its timescale **that its selection windows never touched** (spec 41 §6.4); a job that writes no trial row is retried up to `holdout_max_attempts`, then the candidate is retired | `candidate` |
| Verdict | holdout trial: deflated Sharpe ≥ `holdout_min_dsr`, Sharpe > 0, ≥ `holdout_min_obs` observations; failure retires | `holdout_passed` / `retired` |
| Admission | while fewer than `max_live_evolved` evolved strategies are live: the managed live set `fa-artifacts/live/strategies.toml` is rewritten and strategy-runtime reloaded (`ctl.strategy.reload {artifact_key: managed}`); its default population is baselines + that file | `shadow` |
| Promotion | spec 40 §6.6 gate green (every item blocking, incl. PBO and shadow-vs-backtest) and `auto_promote` | `active` |
| Demotion | active: drawdown pause, gate decay (its own live book inconsistent with the backtest), or a strategy-runtime auto-pause (NaN / step budget) → `shadow`; after `max_demotions` → retired | `shadow` / `retired` |
| Retirement | shadow drawdown breach, definitive gate failure after the minimum shadow time, `max_shadow_days` exceeded, runtime auto-pause, operator `retire` | `retired` (terminal, removed from the live set) |

Every transition is persisted in `ledger.strategy_state` (`status` incl. `candidate|holdout_passed|retired`, `pipeline` JSONB with genome, generation, selection / holdout evidence, attempts, demotions, `last_reason`; migration `20261111000050`) and audited (`strategy.candidate|holdout_passed|admitted|promote|demoted|retired`, `strategy.live_set`, `validation.holdout.evaluate`, actor `system:pipeline`). `GET /api/trading/strategies` shows `evolved` and `pipeline` per row. Hand-written baselines are never touched by the pipeline.

### 6.5 Snapshot dependency
Strategies are tied to the encoder snapshot they were evolved on. On a snapshot change:
- GP programs with region literals are remapped via region lineage (spec 30);
- neural policies are re-fine-tuned;
- both re-enter shadow before trading on the new snapshot.
Live strategies keep running on their snapshot until the replacement passes shadow (runtime serves multiple snapshots, spec 36).

**Implemented (region literals and remapping):** the DSL has a region literal `(region expert ordinal)` (type `Region`, `RegionRef::Lit`), meaning region `<prefix>-e<expert>-r<nnn>` of the snapshot the strategy is bound to — `snapshot` in the `strategies.toml` entry / genome text (part of the phenotype id when set; unbound genomes keep their ids). A literal resolves only while its timescale serves the bound snapshot (else it answers the fallback, counted); `(dwell (region e k))` is the dwell when `k` is the current region, else 0. strategy-runtime collects the swap lineage model-runtime publishes on `latent.lifecycle.<ts>` (D-069: new regions with `carry|split|merge|partial` ancestors and their probe overlap, `reason = snapshot`; unmatched parents die) live and from the last 7 days of the `LATENT` stream at start (`fa_strategies::remap::LineageBook`). When a bound strategy's timescale starts serving another snapshot, `fa_strategies::remap::remap_def` rewrites every literal: a `carry` successor, else the successor with the largest overlap (split / merge / partial, recorded with kind and overlap); no descendant (`died`, `no_descendant`) or no observed lineage (`no_lineage`) is unmappable. A fully mapped strategy has its programs replaced in place (books kept; `ctl.strategy.status` `snapshot{bound, live, remap, remapped_ms}`, heartbeat `snapshot.remapped`); an unmappable one is suspended (pause code 4: flat, not evaluated; `snapshot.suspended`) until a later lineage maps it. The risk-engine pipeline sends an **active** strategy that was remapped after its promotion, or suspended, back to `shadow` (`strategy.snapshot_shadow`, audited) so it re-enters the gate on the new snapshot; a suspended shadow strategy is not retired for it (code 4 is not a runtime auto-pause). **Snapshot binding of evolved genomes:** `evo.vary.strategy` binds every child to the generation's snapshot (`Job.snapshot_id`) unless already bound, and random GP programs of bound genomes may carry literals `(region e k)`, `k < 8`; `evo.eval.strategy` / the holdout replay score a bound genome on its own snapshot. Strategies without a binding are not touched on a swap.

**Parallel old-snapshot serving (strategy-runtime):** `LatentGraph` keeps states per snapshot; `latent.shadow.<ts>.<inst>` states (model-runtime serving a second snapshot, e.g. `ctl.model.shadow` of the previous one after a swap) are ingested with `on_parallel_state` and never change the live snapshot. A bound strategy whose snapshot is no longer live but still served in parallel (newest state within one bar of the live snapshot's and ≤ max(5 bars, 120 s) old) keeps trading on it (`graph.at_snapshot`), and its remapped replacement runs next to it on the live snapshot in shadow: published on `signal.shadow.<id>` (retained in `SIGNAL`, never traded), compared step by step (`ctl.strategy.status` `snapshot.twin {steps, mean_abs_diff, sign_agreement}`, heartbeat `snapshot.parallel|twins`). When the old snapshot stops being published the remap above takes over (and the pipeline's shadow re-entry). Neural policies (no literals) are rebound the same way; their latent inputs then come from the new snapshot.

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `gp_max_depth` | 8 | code default 5 (`fa_strategies::random::RandomSpec`) |
| `gp_max_nodes` | 64 | code default 48 (`RandomSpec`) |
| `step_eval_budget` | 1 ms | per strategy per step; running value `--step-budget-ms` 50 (debug builds, many instruments) |
| `min_share_per_repr` | 0.3 | Population share floor for GP / neural |
| `pbo_max` | 0.2 | `risk.toml [pipeline]` |
| `dsr_min_candidate` | 0.95 (prob. Sharpe > 0) | for holdout eligibility, `risk.toml [pipeline]` |
| `max_candidates_per_generation` | 3 | |
| `holdout_min_dsr` / `holdout_min_obs` | 0.95 / 100 | single holdout trial verdict |
| `max_live_evolved` | 8 | evolved strategies live at once |
| `max_shadow_days` / `max_demotions` | 90 / 2 | retirement |
| `cost_stress_mult` | 2.0 | |
| `exposure_clip` | [-1, 1] | per asset, of strategy capital |

## 8. Failure modes
| Failure | Response |
|---|---|
| Overfitting (huge population) | Deflated Sharpe, PBO gate, locked holdout used once per strategy, live shadow |
| Look-ahead via decoder outputs or graph state | All inputs pulled through point-in-time replay. Decoder outputs cached at publish time. Leak tests in spec 41 |
| Strategy emits NaN / out-of-range exposure | Clipped to 0 exposure; strategy flagged; repeated → auto-paused by strategy-runtime (`--nan-limit`); the pipeline demotes an active evolved strategy and retires a shadow one (§6.4.1) |
| Step time exceeds budget live | Strategy skipped for that step (exposure unchanged); flagged; `--breach-limit` consecutive breaches → auto-pause, handled like NaN |
| Snapshot or region lineage breaks | Strategy suspended to shadow until revalidated |

## 9. Acceptance criteria
**Remaining:**
- **Evolution lab UI** (AC 6): lifecycle is shown on `/trading/pipeline`, not in the evolution lab (frontend; outside this round's code owners).

1. GP DSL interpreter evaluates programs type-safely; random program generation and all variation operators produce valid typed programs (property tests). **Met for GP** (`programs_respect_limits_and_round_trip_through_sexpr`, `operators_are_type_safe_bounded_and_deterministic`).
2. Neural policies and GP programs both implement `Strategy` and emit `ExposureIntent` with mean and std. **Met** (`GpStrategy`, `neural::NeuralStrategy` → `fa.v1.ExposureDistribution`; `neural_policies_and_asset_sets_replay_through_the_same_engine`, `neural_policies_run_live_on_their_asset_set`, `breeding_pools_keep_both_representations_alive`).
3. Backtest of a strategy is deterministic and reproduces identical P&L given snapshot, window and seed. **Met** (`replay_is_deterministic_and_pays_costs`, recorded-latent fixture).
4. Fitness computation includes costs, and the deflated Sharpe implementation matches reference values on known test vectors. **Met:** costs are in every replay (`cost_sensitivity` at 2×); `deflated_sharpe_matches_the_published_example` reproduces Bailey & López de Prado (2014)'s worked example (T = 1250, SR 2.5 annualised, skew −3, kurtosis 10, N = 100, V[SR] = ½: SR₀ ≈ 0.1132, DSR ≈ 0.9004).
5. PBO computation runs per generation and blocks promotion when above `pbo_max`. **Met:** a generation with PBO > `pbo_max` (or unknown) yields no candidate, and PBO is a blocking gate item.
6. The lifecycle states and transitions are persisted in Helix and visible in the evolution lab and trading desk UI. **Partly met:** persisted and audited in Postgres (`ledger.strategy_state`, `audit.log`), served by `GET /api/trading/strategies` and mirrored onto the Helix `Genome` nodes of evolved strategies every pipeline round (§5); the console's `/trading/pipeline` page shows the pipeline state per strategy (D-077); the evolution lab does not show the lifecycle.

## 10. Open questions
1. GP program semantics: should programs be stateful (memory registers across steps) or purely functional on the current observation? *(Implemented answer: purely functional; the only state is the strategy's virtual book, read through the portfolio primitives.)*
2. Neural policy training signal: differentiable surrogate vs pure ES vs RL (e.g. policy gradients on simulated fills)? *(Implemented answer: hybrid — ES (weight perturbation, topology and input-mask mutation, crossover) selected on the non-differentiable replay fitness, plus an SGD fine-tune operator on the differentiable surrogate `mean[ln(1 + L·m·r) − c·|Δm|]` (+ a std-head dispersion term) over a training window that precedes the selection window by an embargo; no RL. The surrogate's inputs are those of a replay with the parent's weights (its own book), so it is exact for one step and approximate beyond; held-out selection of the weights keeps it from overfitting the window.)*
3. How does one fitness compare GP and neural fairly when their compute costs differ by orders of magnitude (normalisation of the cost objective)? *(Implemented answer: one unit — estimated floating-point operations per asset and step (`fa_strategies::op_cost`: arithmetic 1, constant 0.5, primitive read 10, k-step kernel 10·k, sampled walks 10·k·n, `prob_gt` 50; a net: 2·MACs + 10 per observation feature + 1 per latent input), compared on a log10 scale, so a net pays for the primitives it reads like a program does; `compute_cost` is a Pareto objective, not a penalty, and `min_share_per_repr` keeps the cheaper representation from crowding the other out. GP trials scored before this change used log10 nodes and are not comparable on `compute_cost`.)*
4. Should strategies consume decoder outputs from **specific** elite decoders (genome references) or from an aggregated "best decoder per family" feed? *(Implemented answer: the latest fresh output per exact family and instrument, horizon-matched for forecasts; no genome references.)*
5. Strategy capital unit: are exposures relative to a notional sleeve the allocator later scales (default) or absolute? *(Implemented answer: relative to the sleeve, `mean ∈ [-1, 1]`; risk-engine scales by allocated capital.)*
6. Region literals in GP programs depend on spec 30 open question 1 (region identity). *(Answered by D-069 lineage: literals are bound to a snapshot and remapped through it, §6.5.)*

## 11. Deferred
- Market-making / order-book-level strategies (sub-second horizons).
- Strategies consuming raw text directly (only via decoder outputs for now).
- Cross-strategy communication / hierarchical strategies.
