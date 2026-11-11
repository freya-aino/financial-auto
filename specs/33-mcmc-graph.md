# 33 — MCMC State Graph

| | |
|---|---|
| **Status** | draft |
| **Maturity** | High (six competing kernels incl. NUTS/Gibbs refits on `jobs.mcmc.refit`; see §12) |
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

**Implemented:** `fa_mcmc::kernel::KernelSampler` (`observe`, `predict` = one-step posterior predictive, `summarize` = mean + central interval, `graph` = the Dirichlet evidence, `replace_graph`) with `ConjugateExact` and `SmcParticle` (per source row `n` categorical particles with log weights; reweight by `θ[to]` with unseen successors scored as `unknown`; weights tempered by the forgetting factor; resample + exact Gibbs move from the row's Dirichlet posterior when ESS < n/2 or the row's successor set grew; weighted particle mean and quantiles). model-runtime's `TransitionModel` is `regions::ArenaTransitions`: all kernels in a calibration arena (§6.4), the serving one publishes.

**Implemented (2026-11-11, richer kernels):** `KernelSampler` gained `observe_ctx / predict_ctx / summarize_ctx` with `fa_mcmc::rich::KernelCtx{instrument, prev, dwell, x}` (the §4.1 `KernelCtx`: asset, region before `from`, closed bars spent in `from`, covariates `[ln dwell, ln(|r|/EWMA|r|), novelty, router weight]`), `apply_refit` and `export_state / import_state`. Context-free kernels ignore the context. Four richer kernels (§6.2) join `conjugate` and `smc` in every arena: `second_order`, `semi_markov`, `hier_asset` (refitted by **Gibbs**) and `logit_cov` (refitted by **NUTS**, `fa_mcmc::nuts`: Hoffman–Gelman slice NUTS with dual-averaging step size, split R-hat, Geyer ESS, divergence count; the `HmcNuts` sampler of the proposal). All share the Dirichlet evidence, so transfer, persistence and §4.2 inference are unchanged whichever kernel serves; their own state (`second_order` pair rows, `semi_markov` hazards, `hier_asset` asset rows, the Garage key of the last refit) is persisted inside `RegionSet.kernel_state` under `arena_states` (ignored by `MarkovGraph` readers such as control-api) and restored on start, refits reloaded from Garage.

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
- `jobs.mcmc.refit`: batch sampler jobs (work queue `JOBS_MCMC`, spec 21/60). **Implemented:** model-runtime keeps per expert space a window of the last 20 000 contextual transitions (`fa_mcmc::refit::ObsRecord`). The first refit is due once a window holds 500 transitions, later ones every `batch_refit_interval` (`--mcmc-refit-secs`, 3600, event time) with ≥ 125 fresh transitions, one in flight per space. **Enqueuer:** model-runtime uploads `ObsDump{kernel_id, graph, records}` to Garage `fa-artifacts/mcmc/<kernel_id>/obs/<job_id>.json` and publishes `Job{kind: "mcmc.refit", spec: RefitSpec, input_refs}` (`Nats-Msg-Id = job_id`). **Runner:** model-runtime's `mcmc` worker task, a durable pull consumer `model-runtime-mcmc` on `JOBS_MCMC` (one job at a time; the CPU work runs on the blocking pool with 2 threads, so the async workers serving live inference are not taken; in-progress acks every 20 s). It runs `fa_mcmc::refit::run` (pure: NUTS per row for `logit_cov`, Gibbs for `hier_asset`), writes the posterior samples to `fa-artifacts/mcmc/<kernel_id>/<created_ms>.bin` and publishes `JobResult{output_refs: [key], result: summary}` on `jobs.result.mcmc.refit`; the result listener loads the samples and the main loop hands them to the arena of the snapshot/expert they were fitted for (stale snapshots are ignored). Undecodable jobs are terminated (→ `DLQ`, spec 21 §8), Garage errors retried. Because the hand-off is Garage + JetStream, the runner can move to evo-worker without contract change (chosen in model-runtime because no other process needs these posteriors and evo-worker is GPU/CPU-busy with evaluations). Heartbeat `mcmc_refits_applied.<ts>`, `mcmc_refit_window.<ts>`, `mcmc.<ts>.e<k>.refit_samplers`; metric `fa_mcmc_refit_jobs_total{outcome}`.

### 4.4 Control API (spec 50)
`bins/control-api/src/mcmc_routes.rs`, from the exact persisted kernel (`RegionSet.kernel_state`, cached 30 s; `draws` default 128, `seed` for reproducible samples, `level`): `GET /api/mcmc/{snapshot}/transition?expert=&from=&to=&k=`, `/k-step?expert=&from=&k=`, `/neighbors?expert=&region=`, `/paths?expert=&from=&k=&n=&seed=`, `/path-prob?expert=&path=a,b,c`, `/hitting?expert=&from=&targets=a,b&horizon=`, `/stationary?expert=`, `/kernels` (the `KernelModel` rows), `/hierarchy` (spec 30 §6.5), `GET /api/mcmc/cross-priors?timescale=&limit=`, `GET /api/latent/regions/{region_id}/lineage?depth=`. 503 when Helix is down, 404 for unknown snapshot / space / region or `no_kernel` before the first persist, 400 for missing parameters.

## 5. Data model
| Entity | Store | Fields |
|---|---|---|
| `Region` (node) | Helix (spec 30) | |
| `TRANSITION` | Helix edge Region → Region | alpha (Dirichlet param or summary), count, posterior mean, credible interval, horizon, last_update, sampler (**implemented** D-048; the exact α of every row is in `RegionSet.kernel_state`) |
| `KernelModel` | Helix node | expert, timescale, snapshot, kernel type, covariates, sampler, calibration scores (**implemented**: `kernel_id = expert_space_id`, `serving_sampler`, `scores` JSON, `stationary` JSON, `relaxation_steps`, `drift_kl`, `n_rows`, `observation_mass`, `live_regions`; `ExpertSpace -HasKernel->`) |
| Kernel state | Helix `RegionSet.kernel_state` | the serving sampler's full `MarkovGraph` (counts, prior, support, centroids, τ); restored on restart |
| Posterior samples (full) | Garage `fa-artifacts/mcmc/<kernel_id>/<ts>.bin` | **implemented** for the batch refits: `b"FAMCMC01"`, u32 header length, JSON header (`RefitOutcome`: rows, keys, covariate standardisation, R-hat/ESS/divergences, Gibbs concentration draws, shapes), then the draws as f32 LE (`logit_cov`: per row `draws × K·(1+4)`; `hier_asset`: per row `β` draws). `fa_mcmc::refit::{encode, decode}`; the conjugate posterior stays its own sufficient statistic and SMC particles are regenerated from it |
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

**Implemented (2026-11-11):** all four, in `fa_mcmc::rich`, competing in the arena:
- **Covariate-conditioned** `logit_cov`: `p(j | i, x) = softmax_j(W_i x̃)` over the row's 8 most observed successors plus a pooled `~other` (spread by the Dirichlet posterior over the remaining successors), `x̃` = standardised `[1, ln dwell, vol ratio, novelty, router weight]`; prior `N(ln p̄_ij, 1)` on biases (`p̄` = the Dirichlet posterior mean, so the prior predictive is the base kernel) and `N(0, 1)` on weights. Non-conjugate → per-row NUTS on the refit job (2 chains × 100 warm-up + 100 draws, tree depth ≤ 5, ≤ 400 most recent observations, rows with ≥ 30); online prediction = Monte Carlo average over 32 of the kept draws; rows without an accepted fit fall back to the Dirichlet posterior. The "region of other experts / coarse-timescale region" covariates of the proposal are not used (open question 3: v1 = dwell, volatility, novelty, routing).
- **Hierarchical across assets** `hier_asset`: `θ_{a,i} ~ Dir(c·β_i)`, `β_i ~ Dir(γ·p̄_i)`, `c ~ Gamma(1, 0.1)`; online predictive `(n_{a,ij} + c·β_ij)/(n_{a,i} + c)` with decayed per-asset counts. Gibbs on the refit job: per sweep `c | β, n` by random-walk Metropolis on `ln c` with `θ` integrated out (Dirichlet-multinomial likelihood, 8 steps), Antoniak table counts `m | c, β, n`, `β | m`; 2 chains from `c = 1` and `c = 50`, 200 burn-in + 400 sweeps; accepted when R-hat (on `ln c` and the largest `β` of each row) ≤ 1.1. Before the first refit `β` = the pooled posterior mean and `c = 10` (empirical Bayes).
- **Semi-Markov** `semi_markov`: per region and dwell bin (1, 2, 3, 4, 5–6, 7–9, 10–14, 15–24, 25–49, 50+ bars) a Beta hazard of leaving with prior `Beta(2(1 − p_ii), 2 p_ii)` centred on the first-order self-transition; exits by the first-order row without self. Conjugate, online; `SemiMarkov::survival(region, horizon)` gives `P(dwell ≥ d)` for time-in-trade logic.
- **Higher-order** `second_order`: Dirichlet rows per (previous, current) pair with the first-order posterior mean as prior (`α = 4·p₁ + n₂`), so pairs without support back off to first order exactly. Conjugate, online.

### 6.3 Online update loop
1. `model-runtime` receives region memberships (spec 30) per step.
2. Each active kernel `observe`s the transition. Online samplers (SMC) update their particles. Conjugate kernels update in closed form.
3. Posterior summaries are written to Helix at a throttled cadence (`helix_flush_interval`) and published on `latent.mcmc.*`.
4. Batch samplers (HMC/NUTS, Gibbs) run as jobs on the GPU/CPU queue (spec 60, priority below live inference), at `batch_refit_interval`. **Implemented** as `jobs.mcmc.refit` (§4.3).

### 6.4 Sampler competition and calibration
- Every kernel × sampler pair produces k-step predictive distributions over regions. Scoring happens on a rolling evaluation window:
  - **Log score** (predictive log-likelihood of the realised region).
  - **Calibration:** PIT histogram uniformity / reliability diagrams over predicted probabilities (binned), expected calibration error (ECE).
  - **Brier score** for multi-class.
  - **Cost:** CPU/GPU time per update and per query.
- These form a Pareto front (spec 34 shared NSGA machinery). The front's chosen point per (timescale, expert) becomes the **serving kernel**. Others keep running in shadow if budget allows.

**Implemented:** `fa_mcmc::calibration::{Calibration, Arena}`: prequential scoring (predict, score, then observe) with exponential weighting (0.999 per evaluation): log score, Brier, ECE over 10 bins (reliability diagram available), KS distance of the randomised categorical PIT over the last 512 evaluations, EWMA cost per predict+observe. Pareto front over (log score ↑, ECE ↓, cost ↓); the serving sampler is the front member with the best log score among ECE ≤ 0.1, switched after 200 evaluations of every sampler and only for a 0.01-nat advantage. Standings are in every `KernelUpdate`, the `KernelModel` row and the heartbeat. A unit test scores calibrated vs over-confident predictions and runs both samplers on a synthetic chain (acceptance 2). Since 2026-11-11 the arena holds six kernels, observes with the step's context (`observe_ctx`) and re-selects every 16 evaluations on log score, ECE and cost (the PIT KS sort is left to the published standings).

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

**Implemented:** `fa_mcmc::cross_asset::CrossAssetPriors` per timescale over the last 500 closed-bar log returns per instrument (model-runtime, every served timescale), aligned on the union of both instruments' bar closes inside their overlap (a close missing on one side is a bar without trades, return 0; ≥ 60 aligned bars and ≥ 20 real ones per side): Ledoit–Wolf shrunk correlation of the standardised pair, lead-lag `ℓ ∈ 1..5` maximising `|corr(a_{t−ℓ}, b_t)|`, predictive gain `Δ CRPS = (σ₀ − σ₁)/√π` of `b_t ~ 1 + b_{t−1}` vs `+ a_{t−ℓ}` (OLS; a calibrated Gaussian has `E[CRPS] = σ/√π`), uncertainty = gain / |t| of the added coefficient. The top 50 positive-gain pairs by `gain / uncertainty` are published on `latent.crossprior.<ts>`, persisted as `CrossPrior` rows and served by `GET /api/mcmc/cross-priors` (acceptance 6: the ranked list the evo-orchestrator consumes).

**Expert residuals and persistence (2026-11-11):** model-runtime pushes, next to every closed-bar return, its **expert residual** `r_t − Σ_k w_k · μ_k(region_k,t−1)`: `w` the router weights and regions of the previous close, `μ_k(region)` the region-conditional mean next-bar return (running mean over ≤ 200 observations, shrunk by `n/(n+5)`; learned after the residual is taken, so it is prequential). With residuals on both sides the shrunk residual correlation is computed next to the return correlation (`CrossPrior.residual_correlation`, `basis = residual`) and the lead-lag, `Δ CRPS` gain, uncertainty and score are computed on the residuals, i.e. what `a`'s view adds to `b`'s **expert** next-step model (the proto `CrossAssetPair.correlation` remains the return correlation). The tracker (series, residuals, last closes, region means) is written to Garage `fa-artifacts/mcmc/crossprior/state.json` every `--hub-state-secs` and restored on start, so restarts no longer reset it. The 60-aligned-bar / 20-own-bar minimum is kept (it is what makes the OLS t-statistic and the LW shrinkage meaningful; §6.5 gives no lower bound to relax to): at 1m bars a pair needs one hour of joint trading. The evidence feedback from evolved pairs is open (§10).

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
| `batch_refit_interval` | 1 h (minutes ts), 1 d (hours/days ts) | implemented: `--mcmc-refit-secs` 3600 for every timescale (event time; refits also need 125 fresh transitions, so slow timescales refit less often on their own), first refit at 500 transitions; window 20 000 |
| `helix_flush_interval` | 10 s | |
| `carry_discount` | 0.5 | Uncertainty inflation on snapshot change |
| `cross_prior_top_n` | 50 per timescale | Pair proposals per generation |

## 8. Failure modes
| Failure | Response |
|---|---|
| Sparse regions → overconfident transitions | Geometric prior + hierarchical sharing. Calibration monitor flags ECE > threshold |
| Sampler divergence (HMC) | Divergence diagnostics (R-hat, ESS, divergent transitions) → reject the refit, keep the previous posterior. **Implemented:** per NUTS row split R-hat ≤ 1.05, divergences ≤ 1 % of draws, min ESS ≥ 50; Gibbs R-hat ≤ 1.1; rejected rows / fits are not applied (their diagnostics are in the `.bin` header and `JobResult.result`) |
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

**Status of the criteria:** 1 — coverage test (20 synthetic chains, 90 % intervals cover ≥ 85 % of edges). 2 — six kernels (`conjugate`, `smc`, `second_order`, `semi_markov`, `hier_asset`, `logit_cov`) scored by the calibration suite, live in model-runtime; NUTS recovers a correlated Gaussian (R-hat < 1.05, ESS > 200, no divergences) and the refit recovers a covariate effect and asset heterogeneity on synthetic data (`refit::tests`). 3 — every §4.2 function returns a `Dist` and is exposed (`/api/mcmc/{snapshot}/…`; strategies keep the D-053 point-in-time surface). 4 — **benchmarked** (`cargo test -p fa-mcmc --release -- --ignored --nocapture bench_`, 64 live regions with 8-d geometry, 6 assets, 20 000 timed updates, predict + score + observe through the arena; 12-core workstation under a load average of ~21–27 from parallel builds): `conjugate` 27–34 µs, `smc` (64 particles) 76–85 µs, `second_order` 38–60 µs, `semi_markov` 34–94 µs, `hier_asset` 40–41 µs, `logit_cov` 35–68 µs unfitted / 9–18 µs fitted (32 draws), the full six-kernel arena 337–683 µs per step: ≤ 1 ms per kernel with ≥ 10× margin for the conjugate one. A refit job over 20 000 observations / 24 rows costs 24–31 s of CPU (2 threads). 5 — transfer tests incl. splits/merges and regime break. 6 — `latent.crossprior.<ts>` / `CrossPrior` / `GET /api/mcmc/cross-priors`.

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
| Richer kernels (§6.2), `KernelCtx` | `crates/fa-mcmc/src/rich.rs` |
| NUTS, R-hat / ESS / divergences | `crates/fa-mcmc/src/nuts.rs` |
| `jobs.mcmc.refit` computation, Gibbs, `.bin` format, observation window | `crates/fa-mcmc/src/refit.rs` |
| Refit enqueue / worker / result listener, cross-asset persistence | `bins/model-runtime/src/mcmc_jobs.rs` |
| Cross-asset priors | `crates/fa-mcmc/src/cross_asset.rs` |
| Live kernel stage, `KernelUpdate`, cross-scale trackers | `bins/model-runtime/src/{pipeline,regions}.rs` |
| Helix rows | `crates/fa-hub/src/regions.rs` (`KernelModel`, `RegionSet.kernel_state`, `CrossPrior`) |
| Query routes | `bins/control-api/src/mcmc_routes.rs` |

Still open: other experts' / coarse-timescale regions as covariates (§10 q3), the evidence feedback from evolved cross-asset pairs (§6.5), a GPU or evo-worker runner for refits once they outgrow model-runtime's CPU budget.
