# 33 — MCMC State Graph

| | |
|---|---|
| **Status** | draft |
| **Maturity** | High (implemented on the conjugate/SMC baselines; see §12) |
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

**Implemented:** `fa_mcmc::kernel::KernelSampler` (`observe`, `predict` = one-step posterior predictive, `summarize` = mean + central interval, `graph` = the Dirichlet evidence, `replace_graph`) with `ConjugateExact` and `SmcParticle` (per source row `n` categorical particles with log weights; reweight by `θ[to]` with unseen successors scored as `unknown`; weights tempered by the forgetting factor; resample + exact Gibbs move from the row's Dirichlet posterior when ESS < n/2 or the row's successor set grew; weighted particle mean and quantiles). model-runtime's `TransitionModel` is `regions::ArenaTransitions`: both samplers in a calibration arena (§6.4), the serving one publishes. `HmcNuts` / `Gibbs` need non-conjugate or hierarchical kernels, which do not exist yet (§6.2).

### 4.2 Query surface (used by spec 35 GP primitives and spec 34 decoders)
- `current_state(expert, asset) -> Membership`
- `neighbors(region) -> [(region, p_transition, uncertainty)]`
- `transition_prob(a, b, horizon) -> Dist<f64>` (posterior distribution, not a point)
- `sample_paths(from, k, n) -> [Path]` (Monte Carlo walks under posterior draws)
- `path_prob(path) -> Dist<f64>`
- `hitting_time(from, target_set) -> Dist<f64>`
- `stationary(region_set) -> Dist<Vec<f64>>`
- `asset_prior(a, b) -> CrossAssetPrior` (covariance, lead-lag, predictive gain)

**Implemented (graph-level inference, `fa_mcmc::inference::Posterior`):** `n` full kernels drawn from the row posteriors (successor-only states absorbing); every query is evaluated per draw and summarised as `Dist{mean, sd, lower, upper, median, level, n}`: `transition_prob(a, b, k)` (k-step by vector–matrix iteration per draw — §6.1's posterior-averaged matrix powers, not the power of the mean), `k_step(from, k)`, `neighbors(region, n)` with uncertainty, `sample_paths(from, k, n)` (path *i* walks draw *i mod n*), `path_prob(path)`, `hitting(from, targets, horizon)` (exact first passage per draw: `P(τ ≤ H)`, `E[min(τ, H)]`, `E[τ | hit]`, mean pmf), `stationary()` over the recurrent part (observed sources, rows renormalised, lazy power iteration) and `relaxation()` = `1/(1 − |λ₂|)` by deflated power iteration. control-api serves all of them from the persisted kernel (§4.4). `asset_prior(a, b)` is §6.5. The 90 % intervals cover a known synthetic kernel on ≥ 85 % of edges (test `credible_intervals_cover_the_true_kernel_at_the_nominal_rate`, acceptance 1).

**Strategy-side point-in-time surface (D-053):** `fa_strategies::graph::GraphAt` is the point-in-time query surface over `fa_mcmc::MarkovGraph` rows rebuilt from the published posterior summaries (`set_row_from_summary` recovers the Dirichlet concentration from the 90 % interval): k-step probabilities by matrix power (`TransitionMatrix::k_step`), neighbours by mass, posterior-predictive walks seeded per step, hitting times from sampled paths, dwell in event-time bars, `novelty = max(posterior std, unknown mass)`.

### 4.3 Bus
- `latent.mcmc.<timescale>.<expert>`: posterior summary updates (`fa.v1.KernelUpdate`). **Implemented:** per expert space every `--kernel-every-secs` (60 s, event time, at bar closes): serving sampler, every sampler's standing (log score, Brier, ECE, PIT KS, cost µs, evals, Pareto, serving), stationary distribution (top 32, with intervals), relaxation time with interval, rows, observation mass, live regions, drift KL / `refit_due` (spec 30 §6.6), draws (32). The per-row posteriors stay on `latent.transition.<timescale>` (`TransitionUpdate`, D-039, `sampler_id` = serving sampler).
- `latent.crossprior.<timescale>`: ranked cross-asset proposals (`fa.v1.CrossAssetPriors`, §6.5), every `--graph-every-secs`.
- `jobs.mcmc.refit`: batch sampler jobs (work queue, spec 21/60). *Not used:* both implemented samplers are online and cheap; the subject is reserved for HMC/Gibbs refits of richer kernels (§6.2).

### 4.4 Control API (spec 50)
`bins/control-api/src/mcmc_routes.rs`, from the exact persisted kernel (`RegionSet.kernel_state`, cached 30 s; `draws` default 128, `seed` for reproducible samples, `level`): `GET /api/mcmc/{snapshot}/transition?expert=&from=&to=&k=`, `/k-step?expert=&from=&k=`, `/neighbors?expert=&region=`, `/paths?expert=&from=&k=&n=&seed=`, `/path-prob?expert=&path=a,b,c`, `/hitting?expert=&from=&targets=a,b&horizon=`, `/stationary?expert=`, `/kernels` (the `KernelModel` rows), `/hierarchy` (spec 30 §6.5), `GET /api/mcmc/cross-priors?timescale=&limit=`, `GET /api/latent/regions/{region_id}/lineage?depth=`. 503 when Helix is down, 404 for unknown snapshot / space / region or `no_kernel` before the first persist, 400 for missing parameters.

## 5. Data model
| Entity | Store | Fields |
|---|---|---|
| `Region` (node) | Helix (spec 30) | |
| `TRANSITION` | Helix edge Region → Region | alpha (Dirichlet param or summary), count, posterior mean, credible interval, horizon, last_update, sampler (**implemented** D-048; the exact α of every row is in `RegionSet.kernel_state`) |
| `KernelModel` | Helix node | expert, timescale, snapshot, kernel type, covariates, sampler, calibration scores (**implemented**: `kernel_id = expert_space_id`, `serving_sampler`, `scores` JSON, `stationary` JSON, `relaxation_steps`, `drift_kl`, `n_rows`, `observation_mass`, `live_regions`; `ExpertSpace -HasKernel->`) |
| Kernel state | Helix `RegionSet.kernel_state` | the serving sampler's full `MarkovGraph` (counts, prior, support, centroids, τ); restored on restart |
| Posterior samples (full) | Garage `mcmc/<kernel_id>/<ts>.bin` | for heavy samplers (*none yet*: the conjugate posterior is its own sufficient statistic, SMC particles are regenerated from it) |
| `Asset` ↔ `Asset` `CROSS_PRIOR` | Helix edge | covariance (shrunk), lead/lag, predictive gain (Δ CRPS), posterior uncertainty, timescale (**implemented** as `CrossPrior` nodes keyed `<ts>:<source>:<target>`, so no `Instrument` node is required) |
| Calibration time series | VictoriaMetrics | `mcmc_log_score{kernel}`, `mcmc_pit_ks{kernel}` (**implemented** as heartbeat gauges `mcmc.<ts>.e<k>.<sampler>.log_score|ece|brier|pit_ks|cost_us|serving`, mirrored to metrics) |

## 6. Behaviour & algorithms

### 6.1 Base kernel: Dirichlet-multinomial (proposal)
For each source region `i`, a categorical transition distribution `θ_i ~ Dirichlet(α_i)` over successor regions (including self and `Unknown`).
- **Prior** `α_i = α_0 · m_i + ε`, where `m_i` is a structural prior from latent geometry: neighbours in latent space are more likely successors, `m_ij ∝ exp(−d(i,j)/τ)`. This lets sparse regions borrow strength.
- **Update:** soft counts `α_ij += w_from(i) · w_to(j)` per observed step (membership weights).
- **Forgetting:** exponential decay `α ← γ·(α − α_prior) + α_prior` per step so kernels track non-stationarity. `γ` is per timescale.
- **k-step prediction:** posterior-averaged matrix powers via Monte Carlo over Dirichlet draws (not the power of the mean, which hides uncertainty).

**Implemented:** `MarkovGraph::set_centroid` turns on the geometric prior per row: the row's total prior mass `alpha_0 · |successors|` is spread as `ε + rest · m_ij` (`ε` = 0.1 · `alpha_0`, `unknown` and centroid-less keys get the mean affinity), `τ` = `tau_geo` or the median nearest-neighbour distance. model-runtime registers every region's centroid at birth/split/load and refreshes them with each kernel summary. Forgetting multiplies the counts (`α − α_prior`) by `gamma_forget`. `observe_soft` implements the membership-weighted update; the live kernel observes hard transitions.

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

**Implemented:** `fa_mcmc::calibration::{Calibration, Arena}`: prequential scoring (predict, score, then observe) with exponential weighting (0.999 per evaluation): log score, Brier, ECE over 10 bins (reliability diagram available), KS distance of the randomised categorical PIT over the last 512 evaluations, EWMA cost per predict+observe. Pareto front over (log score ↑, ECE ↓, cost ↓); the serving sampler is the front member with the best log score among ECE ≤ 0.1, switched after 200 evaluations of every sampler and only for a 0.01-nat advantage. Standings are in every `KernelUpdate`, the `KernelModel` row and the heartbeat. A unit test scores calibrated vs over-confident predictions and runs both samplers on a synthetic chain (acceptance 2).

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

**Implemented:** `fa_mcmc::cross_asset::CrossAssetPriors` per timescale over the last 500 closed-bar log returns per instrument (model-runtime, every served timescale), aligned on the union of both instruments' bar closes inside their overlap (a close missing on one side is a bar without trades, return 0; ≥ 60 aligned bars and ≥ 20 real ones per side): Ledoit–Wolf shrunk correlation of the standardised pair, lead-lag `ℓ ∈ 1..5` maximising `|corr(a_{t−ℓ}, b_t)|`, predictive gain `Δ CRPS = (σ₀ − σ₁)/√π` of `b_t ~ 1 + b_{t−1}` vs `+ a_{t−ℓ}` (OLS; a calibrated Gaussian has `E[CRPS] = σ/√π`), uncertainty = gain / |t| of the added coefficient. The top 50 positive-gain pairs by `gain / uncertainty` are published on `latent.crossprior.<ts>`, persisted as `CrossPrior` rows and served by `GET /api/mcmc/cross-priors` (acceptance 6: the ranked list the evo-orchestrator consumes). Returns of residuals (instead of raw returns) and the evidence feedback from evolved pairs are open (§10).

### 6.6 Snapshot changes
On a new encoder snapshot, region lifecycle events (spec 30 §6.4) arrive:
- carried regions keep their kernel posteriors (with inflated uncertainty: `α ← α · carry_discount`);
- splits distribute counts proportionally to probe overlap; merges sum counts;
- new regions start from the geometric prior.

**Implemented (D-062 + this round):** `MarkovGraph::transfer` (scores = Jaccard overlaps, `carry_discount` 0.5; splits share by score, merges sum). A map flagged `regime_break` (spec 30 §8) skips the transfer for that expert: the kernel restarts from the (geometric) prior. Online splits register the child with the kernel (prior row, geometric neighbour of its parent); merged/retired regions leave the support and the geometry.

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

**Status of the criteria:** 1 — coverage test (20 synthetic chains, 90 % intervals cover ≥ 85 % of edges). 2 — `ConjugateExact` + `SmcParticle` scored by the calibration suite, live in model-runtime. 3 — every §4.2 function returns a `Dist` and is exposed (`/api/mcmc/{snapshot}/…`; strategies keep the D-053 point-in-time surface). 4 — not benchmarked against the 1 ms placeholder (the arena runs two samplers per observation). 5 — transfer tests incl. splits/merges and regime break. 6 — `latent.crossprior.<ts>` / `CrossPrior` / `GET /api/mcmc/cross-priors`.

## 10. Open questions
1. Does region identity hold across snapshots (spec 30 open question 1 / core blocker #1)? Kernel carry-over depends on it entirely. *Resolved for the baselines: Hungarian probe alignment + lineage ids + regime-break reset (spec 30 §6.4).*
2. Soft vs hard memberships in kernel updates (spec 30 open question 3). *Hard in the serving kernel; `observe_soft` available.*
3. Which covariates condition kernels in v1: horizon only, or also other experts' regions / coarse-timescale region?
4. Is the "Unknown" state one per region set or per region (novel neighbourhoods)? *One per row (`unknown` successor); out-of-region latents are flagged in memberships.*
5. How do cross-asset priors scale to a universe of thousands of instruments: pre-filter by asset class / correlation clusters, or let evolution propose pairs randomly?
6. Should transitions across experts (joint state) exist at all, given experts are decorrelated by design?

## 11. Deferred
- Continuous-state alternatives (switching linear dynamical systems over expert latents).
- Variational inference samplers (as an additional competing `Sampler`).
- Causal discovery between assets (beyond predictive gain).

## 12. Implementation map
| Piece | Code |
|---|---|
| Dirichlet graph, geometric prior, soft update, transfer | `crates/fa-mcmc/src/lib.rs` |
| Graph-level posterior inference | `crates/fa-mcmc/src/inference.rs` |
| Samplers, calibration suite, arena | `crates/fa-mcmc/src/{kernel,calibration}.rs` |
| Cross-asset priors | `crates/fa-mcmc/src/cross_asset.rs` |
| Live kernel stage, `KernelUpdate`, cross-scale trackers | `bins/model-runtime/src/{pipeline,regions}.rs` |
| Helix rows | `crates/fa-hub/src/regions.rs` (`KernelModel`, `RegionSet.kernel_state`, `CrossPrior`) |
| Query routes | `bins/control-api/src/mcmc_routes.rs` |

Still open: covariate-conditioned / hierarchical / semi-Markov / higher-order kernels and their HMC/Gibbs samplers on `jobs.mcmc.refit` (§6.2), Garage posterior dumps for heavy samplers, residual-based cross priors with evolution feedback.
