# 33 — MCMC State Graph

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Low-medium |
| **Owners (code)** | `crates/fa-mcmc`, `crates/fa-hub`, `bins/model-runtime`, `bins/evo-worker` |
| **Depends on** | 30 Latent regions, 32 Expert spaces, 20 Storage, 34 Decoder population, 35 Strategies, 36 Training lifecycle |

## 1. Purpose
The MCMC state graph is the "complex indicator landscape" that strategies traverse. It is a **Markov state graph** whose nodes are latent regions (spec 30) of each expert space and whose edges are **Bayesian transition kernels**. The kernels are updated online on live data, and their posteriors are sampled by **pluggable MCMC/SMC samplers**. The graph also carries **cross-asset priors**: covariance and predictive relationships between assets, used to propose asset sets and RNN-decoder pairs.

## 2. Scope
**In scope:**
- State definition, transition kernels and their Bayesian parameterisation.
- The pluggable `Sampler` abstraction and how samplers compete.
- Online updating from live region memberships.
- Calibration scoring.
- Cross-asset covariance/predictive priors.
- Storage of the graph in Helix and the query surface strategies and decoders use.

**Out of scope:**
- Region discovery and identity (spec 30).
- Strategy logic (spec 35).
- Decoder internals (spec 34).

## 3. Decisions
- The MCMC role is **both**: a Markov chain over discretised latent states **and** MCMC/SMC sampling of the kernel posteriors. Transitions are uncertainty-aware.
- Nodes = latent regions per (snapshot, timescale, expert). Edges = transitions.
- **Samplers are pluggable and compete.** SMC/particle filter, HMC/NUTS and others are scored on **calibration + compute cost**. The sampler choice is a gene/config.
- **Cross-asset priors** model covariance and predictive quality between two or more assets. They feed asset selection (spec 31 §6.4) and **propose source→target pairs for RNN-prior decoders** (spec 34). Evolution then selects.
- The graph lives in **HelixDB**.

## 4. Interfaces

### 4.1 Kernel and sampler traits (proposal)
```rust
/// A Bayesian transition model over one region set (one expert, one timescale).
pub trait TransitionKernel: Send + Sync {
    /// Observe a transition (soft memberships allowed) with covariates (horizon, asset set, context).
    fn observe(&mut self, from: &Membership, to: &Membership, ctx: &KernelCtx);
    /// Posterior predictive over next region(s), k steps ahead.
    fn predict(&self, from: &Membership, k: u32, ctx: &KernelCtx) -> RegionDist;
    /// Expose posterior parameters for samplers.
    fn posterior(&self) -> &dyn PosteriorParams;
}

pub trait Sampler: Send + Sync {
    const NAME: &'static str;
    type Config: Genome;
    /// Draw posterior samples of kernel parameters (or latent paths) given observations.
    fn sample(&mut self, model: &dyn KernelModel, obs: &ObsWindow, n: usize, rng: &mut Rng) -> PosteriorSamples;
    /// Incremental update for online samplers (SMC). Batch samplers may refit instead.
    fn update(&mut self, obs: &Observation) -> Option<PosteriorSamples>;
    fn cost(&self) -> CostEstimate;
}
```
Initial samplers (proposal): `ConjugateExact` (Dirichlet closed form, baseline), `SmcParticle` (online), `HmcNuts` (batch refit for non-conjugate kernels), `Gibbs` (hierarchical kernels).

### 4.2 Query surface (used by spec 35 GP primitives and spec 34 decoders)
- `current_state(expert, asset) -> Membership`
- `neighbors(region) -> [(region, p_transition, uncertainty)]`
- `transition_prob(a, b, horizon) -> Dist<f64>` (posterior distribution, not a point)
- `sample_paths(from, k, n) -> [Path]` (Monte Carlo walks under posterior draws)
- `path_prob(path) -> Dist<f64>`
- `hitting_time(from, target_set) -> Dist<f64>`
- `stationary(region_set) -> Dist<Vec<f64>>`
- `asset_prior(a, b) -> CrossAssetPrior` (covariance, lead-lag, predictive gain)

**Implemented today (D-053):** `fa_strategies::graph::GraphAt` is the point-in-time query surface over `fa_mcmc::MarkovGraph` rows rebuilt from the published posterior summaries (`set_row_from_summary` recovers the Dirichlet concentration from the 90 % interval): k-step probabilities by matrix power (`TransitionMatrix::k_step`), neighbours by mass, posterior-predictive walks seeded per step, hitting times from sampled paths, dwell in event-time bars, `novelty = max(posterior std, unknown mass)`.

### 4.3 Bus
- `latent.mcmc.<timescale>.<expert>`: posterior summary updates (`fa.v1.KernelUpdate`).
- `jobs.mcmc.refit`: batch sampler jobs (work queue, spec 21/60).

## 5. Data model
| Entity | Store | Fields |
|---|---|---|
| `Region` (node) | Helix (spec 30) | |
| `TRANSITION` | Helix edge Region → Region | alpha (Dirichlet param or summary), count, posterior mean, credible interval, horizon, last_update, sampler |
| `KernelModel` | Helix node | expert, timescale, snapshot, kernel type, covariates, sampler, calibration scores |
| Posterior samples (full) | Garage `mcmc/<kernel_id>/<ts>.bin` | for heavy samplers |
| `Asset` ↔ `Asset` `CROSS_PRIOR` | Helix edge | covariance (shrunk), lead/lag, predictive gain (Δ CRPS), posterior uncertainty, timescale |
| Calibration time series | VictoriaMetrics | `mcmc_log_score{kernel}`, `mcmc_pit_ks{kernel}` |

## 6. Behaviour & algorithms

### 6.1 Base kernel: Dirichlet-multinomial (proposal)
For each source region `i`, a categorical transition distribution `θ_i ~ Dirichlet(α_i)` over successor regions (including self and `Unknown`).
- **Prior** `α_i = α_0 · m_i + ε`, where `m_i` is a structural prior from latent geometry: neighbours in latent space are more likely successors, `m_ij ∝ exp(−d(i,j)/τ)`. This lets sparse regions borrow strength.
- **Update:** soft counts `α_ij += w_from(i) · w_to(j)` per observed step (membership weights).
- **Forgetting:** exponential decay `α ← γ·(α − α_prior) + α_prior` per step so kernels track non-stationarity. `γ` is per timescale.
- **k-step prediction:** posterior-averaged matrix powers via Monte Carlo over Dirichlet draws (not the power of the mean, which hides uncertainty).

### 6.2 Richer kernels (proposal, compete with the base)
- **Covariate-conditioned:** logistic-normal transitions with covariates (horizon, volatility state, region of other experts, coarse-timescale region from `REFINES` edges). Non-conjugate → HMC/NUTS batch or SMC online.
- **Hierarchical across assets:** asset-specific kernels sharing a global kernel via a hierarchical Dirichlet. Gibbs sampling.
- **Semi-Markov:** explicit duration distribution per region (dwell-time). Useful for strategies' time-in-trade logic.
- **Higher-order:** state = (previous, current) region pair where support allows.

### 6.3 Online update loop
1. `model-runtime` receives region memberships (spec 30) per step.
2. Each active kernel `observe`s the transition. Online samplers (SMC) update their particles. Conjugate kernels update in closed form.
3. Posterior summaries are written to Helix at a throttled cadence (`helix_flush_interval`) and published on `latent.mcmc.*`.
4. Batch samplers (HMC/NUTS, Gibbs) run as jobs on the GPU/CPU queue (spec 60, priority below live inference), at `batch_refit_interval`.

### 6.4 Sampler competition and calibration
- Every kernel × sampler pair produces k-step predictive distributions over regions. Scoring happens on a rolling evaluation window:
  - **Log score** (predictive log-likelihood of the realised region).
  - **Calibration:** PIT histogram uniformity / reliability diagrams over predicted probabilities (binned), expected calibration error (ECE).
  - **Brier score** for multi-class.
  - **Cost:** CPU/GPU time per update and per query.
- These form a Pareto front (spec 34 shared NSGA machinery). The front's chosen point per (timescale, expert) becomes the **serving kernel**. Others keep running in shadow if budget allows.

### 6.5 Cross-asset priors (proposal)
- For pairs/sets of assets at each timescale, maintain:
  - shrunk covariance (Ledoit-Wolf style) of returns and of expert residuals;
  - **lead-lag** structure (cross-correlation at lags, transfer-entropy proxy);
  - **predictive gain:** Δ CRPS when asset A's expert view is added to asset B's next-step model.
- Posterior uncertainty is tracked (bootstrap or Bayesian linear regression).
- **Proposals:** the top-N pairs by `predictive_gain / uncertainty` are emitted as candidates for:
  1. asset sets in encoder cross-attention (spec 31);
  2. RNN-prior decoder source→target pairs (spec 34);
  3. strategy asset sets (spec 35).
  Evolution selects and mutates. Fitness results flow back as evidence (a pair that keeps winning raises its prior).

### 6.6 Snapshot changes
On a new encoder snapshot, region lifecycle events (spec 30 §6.4) arrive:
- carried regions keep their kernel posteriors (with inflated uncertainty: `α ← α · carry_discount`);
- splits distribute counts proportionally to probe overlap; merges sum counts;
- new regions start from the geometric prior.

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `alpha_0` | 1.0 | Prior strength |
| `tau_geo` | median NN distance | Geometric prior temperature |
| `gamma_forget` | per timescale: s 0.999, m 0.9995, h 0.999, d 0.995 | |
| `k_max` | 2 × horizon in steps | Max prediction steps |
| `mc_draws` | 256 | For k-step predictive |
| `smc_particles` | 1024 | |
| `batch_refit_interval` | 1 h (minutes ts), 1 d (hours/days ts) | |
| `helix_flush_interval` | 10 s | |
| `carry_discount` | 0.5 | Uncertainty inflation on snapshot change |
| `cross_prior_top_n` | 50 per timescale | Pair proposals per generation |

## 8. Failure modes
| Failure | Response |
|---|---|
| Sparse regions → overconfident transitions | Geometric prior + hierarchical sharing. Calibration monitor flags ECE > threshold |
| Sampler divergence (HMC) | Divergence diagnostics (R-hat, ESS, divergent transitions) → reject the refit, keep the previous posterior |
| Non-stationarity shock | Forgetting factor + novelty-driven `Unknown` state. Uncertainty grows, so strategies see wider distributions |
| Helix lag / outage | Kernels keep running in memory with JetStream-backed observation log. Replay on reconnect |
| Graph growth (regions × experts × timescales × assets) | Asset-hierarchical kernels instead of per-asset kernels. Prune edges below support |

## 9. Acceptance criteria
1. On synthetic data generated from a known Markov chain over known regions, the base kernel recovers transition probabilities within credible intervals at the nominal rate (e.g. 90% CI covers truth ≥ 85% of edges).
2. At least two samplers (conjugate + SMC) implement `Sampler` and are scored by the calibration suite.
3. All query-surface functions in §4.2 return distributions (not points) and are exposed to strategies (spec 35) and the frontend graph explorer.
4. Online update per step per kernel ≤ 1 ms for the conjugate kernel (placeholder).
5. Snapshot change handling preserves kernels for carried regions (tested with synthetic lifecycle events).
6. Cross-asset priors produce a ranked proposal list per generation consumable by the evo-orchestrator.

## 10. Open questions
1. Does region identity hold across snapshots (spec 30 open question 1 / core blocker #1)? Kernel carry-over depends on it entirely.
2. Soft vs hard memberships in kernel updates (spec 30 open question 3).
3. Which covariates condition kernels in v1: horizon only, or also other experts' regions / coarse-timescale region?
4. Is the "Unknown" state one per region set or per region (novel neighbourhoods)?
5. How do cross-asset priors scale to a universe of thousands of instruments: pre-filter by asset class / correlation clusters, or let evolution propose pairs randomly?
6. Should transitions across experts (joint state) exist at all, given experts are decorrelated by design?

## 11. Deferred
- Continuous-state alternatives (switching linear dynamical systems over expert latents).
- Variational inference samplers (as an additional competing `Sampler`).
- Causal discovery between assets (beyond predictive gain).
