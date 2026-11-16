# 32 — Expert Spaces

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium (D-099 objective, health, identity and `ExpertView` implemented; item 2 passes 12/12 on synthetic factor data at the tightened D-102 thresholds; live-data evaluation deferred to the final live round, §6.3.1) |
| **Owners (code)** | `crates/fa-expert-space`, `crates/fa-encoders`, `bins/trainer`, `bins/model-runtime` |
| **Depends on** | 31 Expert encoders, 30 Latent regions, 33 MCMC graph, 34 Decoder population, 36 Training lifecycle |

## 1. Purpose
Expert spaces are the low-dimensional manifolds the encoders produce through a **learned mixture-of-experts router**. Each expert specialises in a part of the predictive structure. The system optimises for two things at once (since D-099):
- **diversity:** the experts are distinct representations, not copies (pairwise latent CKA driven down);
- **generality:** the spaces are trained end to end on the family's reconstruction/SSL target plus next step, so each expert's latent stays reconstructable by many decoder families (past events, future trajectories, text, images).

This spec defines the router, the objective and the **decoder-facing contract** that everything downstream reads.

## 2. Scope
**In scope:**
- MoE router design and how the expert count emerges.
- Per-expert manifold projection (dimensionality reduction).
- The end-to-end objective (reconstruction/SSL + next step + expert diversity, D-099) and its balance against generalisation.
- Expert identity across generations.
- The `ExpertView` contract consumed by decoders, MCMC and strategies.

**Out of scope:**
- Encoder backbones and modality fusion (spec 31).
- Region discovery on each expert space (spec 30).
- Decoder evolution mechanics (spec 34).

## 3. Decisions
- Expert partitioning comes from a **learned MoE router**. The number and content of experts **emerge from training**. Decorrelation pressure forces specialisation.
- The objective is **end-to-end reconstruction/SSL + next step** with an **expert-diversity** term (pairwise latent CKA of the experts' latents, D-099). ~~The decorrelation target is the **cross-expert covariance of next-step prediction errors (residuals)**, driven to ≈ 0.~~ Residual decorrelation is an optional regulariser (`λ_dec` 0 by default) and an informational diagnostic, no longer a requirement or promotion gate (D-099; the pre-D-099 form is kept in §6.3 for the record).
- This is balanced against **generalisation**: each expert must stay reconstructable by the decoder population (spec 34). From gen 1, this pressure comes from elite-decoder feedback.
- Expert spaces exist **per timescale** (one model per timescale, spec 31) and per encoder snapshot.
- Asset selection may be expressed through the router (experts specialising on asset subsets). Otherwise it comes from MCMC priors or evolution.

## 4. Interfaces

**Implemented today (D-008 in code; D-070, D-073, D-080):** `fa_expert_space::moe::MoeExpertSpace` is the learned expert space of the neural encoders (candle): a linear router over `K_max` experts (softmax, seeded Gaussian logit noise during training, top-k gates renormalised at inference and published as `router_weights`; the runtime keeps them and only routes encoders without a learned gate), per-expert projectors to `(mu, logvar)` with `mu = tanh(·)` (bounded like the statistical latents so region radii stay comparable; FSQ on the leading dims for `fsq_hybrid`), per-expert linear next-step predictors. Losses (`moe::decorrelation_loss_gated` — the form training uses, see §6.3 — `decorrelation_loss`, `load_balance_loss`, `gaussian_kl`) are differentiable and trained jointly with the encoder (spec 31 §6.5). `health::HealthReport::compute_gated` measures §6.4 on the validation windows; `health::linear_cka` + `match_experts` implement §6.5. Experts with validation usage below `prune_usage_below` (0.05) are pruned after training, at least 2 kept (emergent count); the snapshot stores the surviving expert ids. The statistical family keeps its 3 fixed experts and the `EvidenceSoftmaxRouter`. Health of a fit lives in the snapshot manifest (`metrics.k_eff`, `resid_corr_max`, `skill_e<k>`, `usage_e<k>`, `crpss_e<k>`, `pit_ks_e<k>`, `decodability_e<k>`, `collapsed`, `params.health` with the full matrices), in Garage (`fa-models/snapshots/<id>/health.json`), as VictoriaMetrics series (§5) and on the `Expert` nodes (§5).

**Since D-080 (round 4, 2026-11-11):** *balancing* — `moe.balance: "bias"` is the auxiliary-loss-free form of §6.1 (a per-expert routing bias `moe.route_bias` added to the router logits, excluded from the gradient, moved after every step by `bias_rate` (1e-3) toward the mean top-k load; the Switch loss is then off); `aux` (default) keeps the Switch loss. *Split* — `split_resid_ratio` > 0: after pruning, an expert whose routed scalarised residual variance exceeds that multiple of the experts' mean and whose routed latents are bimodal (Sarle's coefficient of the first principal component > 5/9) is cloned with seeded noise into a free slot (router column copied, both router biases −ln 2, family SSL head copied), then `split_steps` (100) more steps run; `metrics.experts_split`, `params.split`. *`cov_ema`* — `moe.cov_ema` α > 0: the gated D-008 term reads residual moments blended `α·running + (1 − α)·minibatch` (running moments detached and checkpointed, so resume stays bit-exact). *Decorrelation hold-out* — `decor_holdout` > 0: the latest share of each series' training windows (after an embargo) is reserved and the D-008 term is computed on a minibatch of them while the prediction losses use the rest. *Calibration* — after training each expert's Gaussian predictive spread per target is calibrated on the training windows (`pred_sd`, stored in the weights) and the validation health adds per-expert CRPS, CRPS skill vs the naive `N(0, 1)` and the PIT Kolmogorov–Smirnov distance; live, `ExpertView.next_step.log_var` carries the spread. *Decodability* — per expert and probe family (`forecast`: next 4 returns, `future`: realised |return| over 8 bars, `retrieval`: last 8 returns) the validation R² of a ridge readout from that expert's latent alone (`health.decodability`, `metrics.decodability_min`). *Head solver `disagree`* — §6.3.1 round 4.

**§4.2–§4.3 on the bus (implemented):** `fa.v1.ExpertView` (`proto/fa/v1/latent.proto`: `ExpertViewLatent{expert_id, mu, logvar, gate, region: ExpertMembership, next_step: PredictiveDist, codes}`, `expert_mask`, `McmcStateRef{region_ids, kernel_subject}`, `encoder_family`, `prior_score` / `has_prior_score`) is published by model-runtime on `latent.expert.<timescale>.<instrument token>` for every step of the active pipeline (`bins/model-runtime/src/expert_view.rs`): the `LatentState` experts joined with their soft membership and router gate, plus — for learned encoders, through `EncoderFamily::encode_with_extras` — each expert's own next-step prediction, FSQ code and the family prior's novelty (families 2 and 4). `fa.v1.ExpertHealth` (usage, skill, row-major residual-correlation matrix, `K_eff`, `resid_corr_max`, `source = live`) goes to `latent.expert.health`: on every closed bar the previous closed window's per-expert predictions are scored against the realised normalised targets, and every 5 min (≥ 30 rows, last 2000) the §6.4 measures are recomputed with `HealthReport::compute_gated` and mirrored as heartbeat gauges `expert.<ts>.k_eff|resid_corr_max|skill_min` — the §8 live health signal. The statistical family has no per-expert predictor, so its views carry latents, gates and regions only and no live health is published for it. Rust and Go read the same bytes (`go/pkg/bus/expert_view_test.go` decodes the fixture written by the Rust test; subjects `subjects::latent_expert` / `bus.LatentExpert`, `LATENT_EXPERT_HEALTH`). The consumer-side expert mask (decoders select experts, spec 34) stays in the consumers; the published view has every expert selected.

### 4.1 Router and experts (proposal)
```rust
pub struct ExpertSpaceConfig {
    pub max_experts: usize,          // capacity K_max; active count emerges
    pub expert_dim: usize,           // manifold dim per expert (e.g. 8–32)
    pub top_k: usize,                // experts active per window
    pub prune_usage_below: f32,      // usage threshold for expert death
    pub decor: DecorConfig,          // §6.3
}

pub struct ExpertSpaceModule<B: Backend> {
    router: Router<B>,               // pre-router latent tokens -> gate logits [batch, K_max]
    projectors: Vec<Projector<B>>,   // per expert: tokens -> (mu, logvar) in R^expert_dim
    predictors: Vec<NextStep<B>>,    // per expert: latent -> next-step target distribution
}
```

### 4.2 Decoder-facing contract (decided shape, fields proposal)
```rust
/// The one structure decoders, MCMC and strategies consume. Serialized as fa.v1.ExpertView.
pub struct ExpertView {
    pub snapshot_id: SnapshotId,
    pub timescale: Timescale,
    pub t: Timestamp,                       // window end (point-in-time)
    pub instrument_set: AssetSetId,         // instrument or asset set the window covers
    pub experts: Vec<ExpertLatent>,         // only experts selected by the mask below
    pub expert_mask: BitVec,                // which experts the consumer requested / receives
    pub mcmc_state: Option<McmcStateRef>,   // current region per expert + kernel posterior ref (spec 33)
}

pub struct ExpertLatent {
    pub expert_id: ExpertId,                // stable across generations where aligned (§6.5)
    pub mu: Vec<f32>,                       // expert_dim
    pub logvar: Vec<f32>,                   // uncertainty
    pub gate: f32,                          // router weight for this window
    pub region: Membership,                 // spec 30
    pub next_step: PredictiveDist,          // the expert's own next-step prediction
}
```
Decoder genomes select an **expert subset mask** (spec 34). Strategies read `ExpertView` through GP primitives or as policy input (spec 35).

### 4.3 Bus
- `latent.expert.<timescale>.<asset_set>`: `fa.v1.ExpertView` per step.
- `latent.expert.health`: per-expert usage, residual covariance matrix, calibration.

## 5. Data model
| Entity | Store | Fields |
|---|---|---|
| `Expert` | Helix node | expert_uid (`<snapshot>/e<expert_id>`), expert_id, snapshot_id, timescale, dim, usage, skill, specialisation summary (gate mass per instrument), born_gen, died_gen |
| `EXPERT_OF` | Helix edge `ExpertOf` (Expert → Snapshot) | |
| `ALIGNED_TO` | Helix edge `AlignedTo` (Expert → Expert prev gen) | cka, kind |
| Expert latents (sampled) | Helix vectors | for retrieval decoders (spec 34) |
| Residual covariance matrices | VictoriaMetrics (pairwise series) + Garage (full matrices per eval) | `expert_resid_corr{ts,i,j}` — *implemented:* `fa_expert_resid_corr{tf, expert="<i>_<j>", src}` (`src` = `validation` per fit, written by the trainer; `live` every live health report, written by model-runtime), plus `fa_expert_skill{tf, expert, src}` and `fa_expert_k_eff{tf, src}` (the label whitelist of spec 20 has no `ts`/`i`/`j`); full matrices in `fa-models/snapshots/<id>/health.json` per fit |
| Usage / gate stats | VictoriaMetrics | `expert_usage{ts,expert}` — *implemented:* `fa_expert_usage{tf, expert, src}` |

## 6. Behaviour & algorithms

### 6.1 Router (proposal)
- A token-pooled router gives gate logits over `K_max` experts. Experts are selected by top-k with noisy gating during training.
- **Load-balancing:** use auxiliary-loss-free balancing (per-expert bias adjustment) to avoid fighting the decorrelation loss. A small auxiliary loss is the fallback.
- **Emergent count:** experts start at `K_max`. Experts with usage below `prune_usage_below` over `prune_window` are pruned at generation boundaries. An expert can **split** (clone + perturb) when its own residual variance is high and its specialisation is multimodal. The active count is reported per generation.

### 6.2 Projection to the expert manifold
Each expert has a projector (small MLP / attention pool) → Gaussian `(mu, logvar)` in `R^expert_dim`. This is the dimensionality reduction from the large encoder latent to the "expert space". For discrete families (spec 31 family 1), the expert also carries its code subset.

### 6.3 Objective (decided 2026-11-13, D-099)
**Owner decision (2026-11-13):** *"These spaces need to be end to end trained on some sort of reconstruction target, so it makes sense that the latent space does not portray residual correlation."* Residual decorrelation is therefore **no longer an objective requirement nor a promotion gate**; the expert spaces are trained end to end (backbone → router + experts → heads, one optimiser) on the family's reconstruction / SSL target plus next step:

`L_32 = λ_ssl L_ssl + λ_next Σ_k L_pred,k + λ_mix L_mix + λ_div L_div + λ_bal L_balance + β KL [+ λ_dec L_decor]`

- `L_ssl` — reconstruction of the window's last `recon_steps` (8) bars from the gated expert latents `Σ_k p_k mu_k H_k (+ b)` for the reconstructing families (`fsq_hybrid`, `hier_vae_flow` — with the top latent —, `latent_diffusion`), the EMA target encoder's embedding of the window `jepa_horizon` steps ahead for `jepa_vae`; `λ_ssl` 1 (default for every family).
- `L_pred,k`, `L_mix` — per-expert and mixture next-step losses on `[next return, |next return|, next range]` as before (`gated_pred` 0.5).
- `L_div` — **expert diversity**: `Σ_{i<j} CKA_lin(mu_i, mu_j)` over the minibatch (`moe::latent_cka_loss`), `λ_div` 0.3. Without it end-to-end training makes every expert encode the same dominant directions (validation latent CKA 0.99–1.0 on synthetic factor data, round 5), i.e. copies; a mean over pairs instead of the sum left the 8-expert day ladder at CKA ≈ 1, and `λ_div` 1 (sum) collapsed the router there.
- `L_balance` — Switch load balancing, `λ_bal` 0.1 (0.01 let the router collapse onto one expert once the experts differ).
- `L_decor` — the D-008 residual decorrelation below, **`λ_dec` 0 by default** (optional regulariser; the genes `lambda_dec`, `decor_scalar`, `decor_dual_lr`/`decor_eps`, `decor_holdout`, `cov_ema`, `lambda_ncl`, `lambda_skill`, the head refit and the `head_solver`s stay available, none is on by default). Residual correlation stays a reported diagnostic (§6.4).

The rest of this section is the pre-D-099 proposal and its implementation, kept for the record.

*Pre-D-099 proposal (core blocker #2):*
Per expert `k`, a next-step predictor gives residuals `r_k = y_{t+1} − ŷ_k`, where `y` holds next-step targets (returns, vol, next latent) of the timescale. Over a batch, let `R ∈ R^{N×K}` (scalarised residual per expert, or residual vectors per target dimension).

`L_32 = Σ_k L_pred,k  +  λ_dec · L_decor(R)  +  λ_gen · L_gen  +  λ_bal · L_balance  +  β · KL_k`

with:
- **Decorrelation** (proposal, default): squared off-diagonal of the residual **correlation** matrix,
  `L_decor = Σ_{i≠j} corr(r_i, r_j)²`, computed over gated windows (both experts active) with an EMA of the covariance across batches for stability.
  Alternatives: HSIC between residuals (captures non-linear dependence); negative-correlation learning (NCL) penalty `Σ_k (ŷ_k − ȳ)(Σ_{j≠k} ŷ_j − ȳ)`.
- **Generalisation** `L_gen`:
  - gen 0: reconstruction + multi-horizon prediction from each expert alone (each expert must carry information);
  - gen ≥ 1: loss of the **elite decoder set** evaluated on this expert's latent (spec 34 §6.6).
- **Collapse guard:** the decorrelation objective can be trivially satisfied by experts predicting nothing (residual = y, all equal → correlated, so not entirely trivial, but degenerate solutions exist). `L_pred,k` per expert, plus a minimum-skill constraint (each active expert must beat the naive baseline by `min_skill`), prevent this.

The weights `λ_dec, λ_gen` are tunable and may be **evolved** as encoder genes (spec 34 Pareto then decides the trade-off).

*Implemented form (pre-D-099):* residuals are vector-valued over the three next-step targets of spec 31 §6.5 (`[next return, |next return|, next range]`, normalised); `L_decor = mean_targets Σ_{i≠j} corr_w(r_i, r_j)²` over the minibatch with each window weighted by the **detached** router probabilities `p_i·p_j` (`moe::decorrelation_loss_gated`: the "gated windows" of the proposal in soft form — experts only have to disagree where both are routed, and the router cannot lower the penalty by separating them alone; no EMA of the covariance), `L_pred,k` is half ungated (every expert on every window, the gen-0 "each expert carries information" term) and half gate-weighted (`gated_pred` 0.5), `L_balance` is the Switch form `K · Σ f_i p_i` (auxiliary-loss balancing, not the bias-adjustment variant), `L_gen` at gen 0 is per-expert next-step prediction plus the family's SSL head, from gen 1 the D-009 frozen-decoder term. Defaults then `λ_dec 1.0`, `λ_bal 0.01`, `β 1e-3` (since D-099: `λ_dec 0`, `λ_div 0.3`, `λ_bal 0.1`). Further genes of the objective (spec 31 `NeuralConfig`): `decor_scalar` (decorrelate the scalarised residual the §6.4 health measures instead of each target — both forms are allowed above), `decor_dual_lr` / `decor_eps` (open question 1 option (e): `λ_dec` by projected dual ascent on `L_decor ≤ ε`, checkpointed), `lambda_skill` / `skill_floor` (the `min_skill` collapse guard as a hinge on each expert's routed error ratio), `lambda_ncl` (option (c), negative-correlation learning on the scalarised predictions) and `head_refit_steps` (a full-batch refit of the per-expert next-step heads on frozen latents with exact correlations instead of minibatch estimates). Observed on live 1m bars (9 instruments, 12 h): `λ_dec = 1` reaches `K_eff ≈ 2.6` of 4 with residual correlations up to 0.64; `λ_dec = 0.1` gives better per-expert skill (≈ 0.25) but `K_eff ≈ 1.3` — the independence/skill trade-off of open question 1 is real on market data, and the 0.1 health target is not met by either.

#### 6.3.1 Can the checklist be met on the live data? (evidence, 2026-11-11)
**Bound.** Write an expert's prediction as `ŷ_k = f + d_k` with `f` the conditional mean and `d_k` any function of the window, so `d_k ⟂ ε` out of sample. The scalarised residual is `ē − d̄_k`; for two experts sharing windows its correlation is smallest when `d̄_j = −d̄_i`, giving `(V − D)/(V + D)` with `V = var(ē)` and `D ≤` the mean per-target deviation energy, while each expert's skill is at most `R̄² − D`. Both `|corr| ≤ c` and `skill ≥ s` are therefore reachable only if `R̄² − s ≥ V (1 − c)/(1 + c)`; with three experts sharing every window (needed, since two experts give `K_eff = 2/(1 + corr²) < 2`) the pairwise version needs `D ≥ 1.5 V`. A residual correlation near 0 is **not** reachable for skilled experts when the irreducible noise dominates — it requires the experts to disagree, symmetrically, by about as much as the noise. `trainer sweep` reports the margin `R̄² − s − V(1 − c)/(1 + c)` estimated by the trained mixture.

**Data.** 1m connector bars of the 9 live instruments (3 Kraken, 5 Yahoo, the synthetic heartbeat), 48 h ending 2026-11-11 03:20 UTC: 7068 bars, 5878 training / 1037 validation windows. The mixture reaches `R̄² ≈ 0.36` (per target: next return ≈ −0.01, |next return| ≈ 0.55, next range ≈ 0.60 — volatility is predictable, direction is not) with `V ≈ 0.12`, so the margin is ≈ +0.25 for two experts and ≈ +0.15 for three: **not ruled out by the bound**.

**Sweeps** (`trainer sweep`, release build, `jepa_vae`, 600–1500 steps, 42 genomes): default genome `K_eff 1.36`, corr 0.69 (router collapse); balancing (`λ_bal ≥ 0.3`) and `decor_scalar` with all experts active lift `K_eff` to 2.8–2.95 and bring the correlation to 0.13–0.30, but then one expert's skill turns negative (−0.1 … −1.6); the skill hinge, batch 256, dual ascent, NCL (0.5–0.9) and longer training move along the same front (best pairs: corr 0.13 at min skill −0.79; min skill +0.035 at corr 0.58). The in-sample health of the same fits is no better (train `K_eff` 1.5–2.7, corr 0.53–0.75), so the gap is **optimisation, not generalisation**: the minibatch D-008 objective does not find the symmetric-disagreement solution the bound allows. A full-batch refit of the linear next-step heads on the frozen latents (exact correlations, `λ_dec` up to 100, skill hinge) gets closest — `K_eff 2.96`, correlation 0.12, but min skill −0.28 (in sample 0.28 / −0.25): each expert alone reaches far less than the mixture's `R̄²` (single-expert skills 0.1–0.26), which leaves the per-expert margin of the three-expert bound (`R̄²_k − s − 1.5 V`) near zero.

**Conclusion (first sweeps; revised by round 3 below).** No snapshot passing `K_eff ≥ 2`, residual correlation ≤ 0.1 and min skill ≥ 0.02 together was found on this data; the thresholds were not changed. The live data is too short (≈ 2 days after the 2026-11-10 reset) to tell whether more history helps. Next steps: a constrained solver for the heads (the residual covariance is quadratic in the linear heads given the latents), experts with disjoint routing (top-1) where the gated correlation only compares shared windows, and revisiting whether the 0.1 target should apply to the scalarised residual of predictable targets only (a spec decision, not taken here).

**Round 3 — constrained head solver and top-1 routing (2026-11-11).** `head_solver: "constrained"` (`fa_encoders::neural::heads`, runs after training whenever selected — in `trainer sweep` and in a snapshot fit alike, whatever `head_refit_steps`; `NeuralConfig::refits_heads`): with the latents frozen, each expert's linear next-step head minimises its gate-weighted squared error subject to **equality constraints** `cov_S(r_k, r_j) = ±ρ*·sd_k·sd_j` for every other expert `j` (`S` = the shared routed windows, all windows when < 30 — exactly the health report's rule; linear in `θ_k` with the others fixed, solved exactly by the KKT system; dependent rows dropped). Block-coordinate sweeps start from the least-squares heads pushed apart by a seeded disagreement of the bound's size `(K−1)·V` (from identical least-squares residuals no single expert can decorrelate alone — it would have to predict the noise); afterwards every pair is feasible and the objective never increases. On synthetic experts reading one backbone it reaches exact in-sample decorrelation with every expert skilled, for shared and top-1 routing (`heads::tests`). **Live sweep** (same data recipe: 1m, 48 h ending 2026-11-11 ≈ 05:50 UTC, 9 instruments, 7 601 bars; `jepa_vae`, 600–1 200 steps, 12 genomes, `trainer sweep --grid`, rows in the round-3 report): the solver closes the **in-sample** gap — train residual correlation 0.02–0.07 with train `K_eff` 2.0–3.0 for top-1 + balancing — but the decorrelation does **not generalise**: on the validation split the correlation returns to 0.18–0.38 and one expert's skill turns negative (top-1, `λ_bal` 0.3: K_eff 1.91, corr 0.21, skills 0.07 / −0.15; shared top-2: corr 0.004 with only 2 experts left and skill −2.0). The closest genome — `head_solver_corr` 0.08, top-1, `λ_bal` 1.0, 3 experts — meets two of the three items on validation (`K_eff` 2.79, every skill positive, min 0.065 ≥ 0.02) and misses the third (max residual correlation 0.32 > 0.1). So the earlier diagnosis "optimisation, not generalisation" now inverts (D-080): an exact in-sample optimum exists and is found, but the disagreement directions it uses are fitted to noise in the training windows (top-1's routed sets also shift between splits). **No snapshot passes; nothing was promoted or switched, thresholds unchanged.** What remains is a spec decision (target the scalarised residual of the predictable targets only, or measure decorrelation on the routed windows also for top-1) or more history.

**Round 4 — principled generalisation fixes (2026-11-11, D-0xx).** Four fixes aimed at the round-3 finding, all now genes: (1) **decorrelation on held-out minibatches** (`decor_holdout` 0.2: the D-008 term is computed on windows the prediction losses never see); (2) **`cov_ema`** across batches (straight-through: the running estimate's value, the minibatch's gradient — a plain blend scaled the penalty's gradient by `1 − α` and simply switched it off, observed at α 0.99: corr 0.99); (3) **stronger shrinkage** of the constrained solver (`head_solver_ridge` 0.1 and 1.0); (4) a new solver **`head_solver: "disagree"`** that plants the bound's symmetric disagreement from moments that contain no target noise (least-squares residual covariance + cross-covariance of the experts' latents, cross terms set to their out-of-sample value 0, joint penalty with continuation so the cost is shared), which decorrelates *on fresh windows* in the synthetic test (`heads::tests::planted_disagreement_generalises`: validation corr 0.04 vs 0.06 constrained, all experts skilled); plus fewer / more experts (K 3 vs 4), top-1 vs top-2, bias balancing and split. **Data:** the cached 48 h of 1m bars (9 instruments) **minus the locked 1m holdout and its embargo** (`[1794318540000, 1794365340000]`, 4 905 bars removed — rounds 1–3 had unknowingly trained on that period; `trainer sweep` now excludes holdouts like a fit, `--holdout-ms` when `validation.holdout` is unreadable). 19 genomes, `jepa_vae`, 600 steps (rows: scratchpad `r5/sweep_p*.jsonl` of the round-4 report). **Result: no genome passes the checklist on validation.**
- The round-3 best (constrained 0.08, top-1, `λ_bal` 1, K 3) on clean data: `K_eff` 2.64, corr 0.37, min skill −0.08 (in sample corr 0.09) — it no longer even meets min skill once the holdout period is removed. Ridge 0.1: same (corr 0.38); ridge 1.0 with top-2 diverges (skill −400).
- `disagree`: in sample corr 0.11–0.34 but live latents share little cross-covariance, so the planted disagreement needs large heads — skills −1.8 … −40 (top-2), K 2 after pruning with bias balancing (corr 0.12, min skill −0.03). The synthetic success does not transfer: market experts' latents are too unlike for a cheap shared disagreement direction.
- Decorrelation hold-out (SGD): the experts give up decorrelating and converge — best mixture skill (0.37) with all skills ≥ 0.03 but corr 0.65 (K_eff 1.90); combined with `cov_ema`/λ_dec 10 one expert collapses (flagged by the collapse monitor).
- `cov_ema` 0.9 (straight-through), λ_dec 1: corr 0.52, min skill −0.68; with bias balancing K drops to 2 (corr 0.17, min −0.13).
- K 4 + split (top-1, constrained): `K_eff` 3.78, corr 0.28, skills 0.34 / 0.79 / 0.29 / −0.59 — more experts lower the correlation but one expert always pays.
The pattern of rounds 3–4 is consistent: every method that reaches correlation ≤ 0.1 in sample gets it from directions that do not hold on validation, and every method restricted to generalising directions (hold-out, planted disagreement from noise-free moments) either stops decorrelating or costs an expert its skill. With ≈ 35 h of history the 0.1 / 0.02 / 2 checklist is **not met**; thresholds unchanged, nothing promoted.

**Round 5 — end-to-end reconstruction, new item 2, synthetic factor data (2026-11-13, D-099).** Data: `trainer sweep --synthetic 6:3 --synthetic-bars 3000` (`generation::synthetic_factor_bars`: 6 instruments on 3 independent factors, calm/volatile regimes redrawn every 40 steps, vol 0.001/0.006, drift 0.0003/−0.0005, idiosyncratic 0.0005, loadings `1 + 0.2·N(0,1)` on factor `k mod 3`, `0.2·N(0,1)` elsewhere — the twin of `go/connectors/synthetic`), default genomes (`NeuralConfig::for_timescale`, 600 steps, 1m; `latent_diffusion` at 1d per D-010), release build, 3 threads per run, 1–3 min per fit; 15 % time-ordered validation split. Item 2 on validation (usage `K_eff` ≥ 2, latent CKA ≤ 0.9, SSL skill ≥ 0.05, min skill ≥ 0.02, no collapse):

| family | seed | K | usage K_eff | latent CKA max | SSL skill | min skill | resid corr max | residual K_eff | pass |
|---|---|---|---|---|---|---|---|---|---|
| fsq_hybrid | 1 / 2 / 3 | 4 | 3.96 / 3.84 / 3.60 | 0.10 / 0.10 / 0.08 | 0.40 / 0.39 / 0.40 | 0.21 / 0.14 / 0.19 | 0.99–1.00 | 1.03–1.12 | ✓ ✓ ✓ |
| hier_vae_flow | 1 / 2 / 3 | 4 | 3.96 / 3.86 / 3.47 | 0.06 / 0.25 / 0.46 | 0.40 / 0.39 / 0.40 | 0.19 / 0.14 / 0.17 | 1.00 | 1.01–1.02 | ✓ ✓ ✓ |
| jepa_vae | 1 / 2 / 3 | 4 | 3.76 / 3.80 / 3.69 | 0.09 / 0.07 / 0.10 | 0.59 / 0.69 / 0.60 | 0.19 / 0.16 / 0.17 | 1.00 | 1.02 | ✓ ✓ ✓ |
| latent_diffusion (1d) | 1 / 2 / 3 | 8 | 7.48 / 6.62 / 7.72 | 0.19 / 0.11 / 0.09 | 0.41 / 0.42 / 0.42 | 0.19 / 0.14 / 0.15 | 1.00 | 1.02–1.04 | ✓ ✓ ✓ |

Ablations (seed 1): **without the diversity term** (`λ_div` 0) every family fails on diversity — latent CKA 0.995 (fsq), 0.999 (jepa, hvf), 1.000 (ldm): end-to-end reconstruction alone makes the experts copies. `λ_div` as a *mean* over pairs at 0.3 (fsq: CKA 0.99) and at 1 (8-expert ldm: 0.999) is too weak; as a sum at 1 the 8-expert ldm router collapses to 2 experts (usage `K_eff` 1.95), at 3–10 with `λ_bal` 0.01 fsq's router collapses (usage `K_eff` 1.0–1.5). `λ_bal` 0.1 / 0.3 / 1 all pass with diversity on; bias balancing (`bias_rate` 0.01) does not (usage `K_eff` 1.03). Router entropy (normalised, seed 1): fsq 0.94 (near-uniform soft routing), hvf 0.64, jepa 0.63, ldm 0.68 — routing is input-dependent for three families; in-sample CKA equals validation CKA within 0.03 (the diversity is structural and generalises, unlike the residual decorrelation of rounds 1–4). As expected under D-099 the experts' next-step residuals stay almost identical (correlation 0.99–1.00, residual `K_eff` ≈ 1), which no longer gates. Every default genome passes the new item 2 on this data; **real-data evaluation is left for the final live round.**

**Round 6 — tightened thresholds, item-2 drift trigger (2026-11-14, D-102).** Checklist defaults tightened: `latent_cka_max` ≤ **0.7** (was 0.9), `ssl_skill` ≥ **0.1** (was 0.05); usage `K_eff` ≥ 2 and min skill ≥ 0.02 unchanged (trainer `ExpertTargets`, control-api `manifest_health_checks`). The trainer drift trigger (`FA_TRAIN_DRIFT_TRIGGERS=1`, `watch.rs`) no longer reads `resid_corr_max` (≈ 0.99 by design since D-099; `FA_TRIGGER_RESID_CORR` removed): it fires when a live `latent.expert.health` report fails the same `ExpertTargets` on 3 consecutive reports (fields a live report leaves unset are skipped). Re-run of the round-5 sweep (`--synthetic 6:3 --synthetic-bars 3000`, 4 families × seeds 1–3): **12/12 pass at the new limits** (usage `K_eff` 3.5–7.7, latent CKA max 0.06–0.46, SSL skill 0.39–0.69, min skill 0.14–0.21), so the tighter limits cost nothing on planted factors. Still provisional until checked on real bars.

### 6.4 Measured quantities (health)
**Since D-099** the checklist (spec 36 §6.4 item 2) reads, on the validation windows of each fit:
- **Usage `K_eff`** `k_eff_usage = (Σu)² / Σu²` over the experts' usage `u` (mean router probability): how many experts the router actually uses.
- **Expert diversity** `latent_cka [K][K]`: linear CKA between the experts' posterior-mean latents over the validation windows, and `latent_cka_max` (largest off-diagonal) — experts must be distinct representations, not copies.
- **SSL skill** `ssl_skill = 1 − SSE / SSE_naive` of the family's SSL target, the naive predictor being its per-dimension training mean (over ≤ 2 048 training windows): reconstruction of the last bars (`recon_skill`, reconstructing families) or the JEPA target embedding (`jepa_vae`).
- Per-expert next-step skill vs naive (below), collapse monitor (spec 31 §8).
- Diagnostics: `router_entropy` (mean normalised router entropy per window: 1 = the router spreads every window uniformly), the residual correlation matrix and the **residual `K_eff`** (field `k_eff`, name kept for compatibility).
Manifests carry `k_eff_usage`, `latent_cka_max`, `ssl_skill`, `recon_skill`, `router_entropy` next to the older metrics; `HealthReport::vm_series` adds `fa_expert_k_eff_usage`, `fa_expert_latent_cka_max`, `fa_expert_ssl_skill` (`src=validation`); model-runtime's live health adds the heartbeat gauge `expert.<ts>.k_eff_usage` and the `fa_expert_k_eff_usage{src=live}` series (latent CKA and SSL skill are not computed live: the `ExpertHealth` proto has no field for them).

*Pre-D-099 list:*
- Residual correlation matrix per timescale on a held-out validation window (not the locked holdout, spec 41).
- **Residual `K_eff`** (effective number of independent experts): `K_eff = (Σλ)² / Σλ²` over eigenvalues of the residual correlation matrix.
- Per-expert skill vs naive baseline, calibration (PIT / CRPS) of `next_step`.
- Per-expert decodability: best elite decoder score per family (spec 34).

*Implemented (calibration, decodability, round 4):* each expert's `next_step` is a Gaussian with the spread calibrated on the training windows; CRPS, CRPS skill vs `N(0, 1)` and PIT KS distance per expert (`health.crps|crpss|pit_ks`, `metrics.crpss_e<k>|pit_ks_e<k>`); decodability per expert = validation R² of ridge readouts of the expert's latent alone for the probe families `forecast`, `future`, `retrieval` (the linear stand-ins of the spec 34 output families; elite decoders read the concatenated latent and do not decompose per expert). Observed live: CRPS skill 0.05–0.6 for skilled experts, PIT KS 0.1–0.5 (the homoscedastic Gaussian is mis-calibrated on volatility-clustered targets).
*Implemented:* on the validation windows of each fit; residual correlation of a pair over the windows where both experts are in the top-k (all windows when fewer than 30 are shared), skill of an expert over the windows routed to it (top-k gate weighted, naive = the training mean of the normalised target), `K_eff` from the correlation matrix (`tr(C)² / ‖C‖_F²`), usage = mean router probability. They feed the promotion checklist (spec 36 §6.4 item 2).

### 6.5 Expert identity across generations (proposal)
- Experts are matched between generations by **CKA similarity** of their latents on the fixed probe set (same probe set as spec 30 §6.4) plus overlap of specialisation summaries.
- Matched experts keep `expert_id`. Unmatched ones are new. Decoder genomes referencing dead experts are remapped through `ALIGNED_TO` or masked out (spec 34).

### 6.6 Live inference
`model-runtime` runs the router + projectors on each step after the encoder and emits `ExpertView`. Region membership (spec 30) and MCMC state (spec 33) are attached before publishing, or published separately and joined by `(snapshot, t)`.

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `max_experts` | 16 | Per timescale |
| `top_k` | 2 | |
| `expert_dim` | 16 | |
| `lambda_dec` | 0 (was 1.0; D-099) | Evolvable; optional regulariser |
| `lambda_div` | 0.3 | Expert diversity (latent CKA, sum over pairs); D-099 |
| `lambda_ssl` | 1.0 | Reconstruction / SSL weight (spec 31 `NeuralConfig`) |
| `lambda_gen` | 1.0 | Evolvable |
| `lambda_bal` | 0.1 (was 0.01; D-099) | If aux-loss fallback used |
| `cov_ema` | 0.99 | Residual covariance EMA |
| `min_skill` | 2% better than naive (CRPS) | Collapse guard |
| `prune_usage_below` | 0.01 | |
| `prune_window` | 1 generation | |
| `target_resid_corr_max` | 0.1 | Informational since D-099 (`FA_PROMOTE_MAX_RESID_CORR`) |
| min usage `K_eff` | 2 | Checklist, provisional (`FA_PROMOTE_MIN_K_EFF`) |
| max expert latent CKA | 0.7 | Checklist, provisional (D-102) (`FA_PROMOTE_MAX_EXPERT_CKA`) |
| min SSL skill | 0.1 | Checklist, provisional (D-102) (`FA_PROMOTE_MIN_SSL_SKILL`) |

## 8. Failure modes
| Failure | Response |
|---|---|
| Router collapse (one expert takes everything) | Balancing kicks in; if usage `K_eff < 2` after training → snapshot not promotable (spec 36) |
| Experts are copies (same representation) | Diversity term `λ_div`; `latent_cka_max > 0.7` → not promotable (D-099, tightened D-102) |
| Degenerate experts (no skill) | `min_skill` constraint fails → expert pruned or snapshot rejected |
| Residual correlation spikes live (regime change) | Health signal on bus; strategies/allocator can widen uncertainty; can trigger refresh (spec 36). Since D-099 end-to-end trained experts share their next-step residuals (validation correlation ≈ 0.99 on synthetic data), so residual correlation no longer triggers a refresh: the trainer's drift trigger (off by default) fires when live expert health fails the item-2 targets (usage K_eff, latent CKA, SSL skill, min skill) on 3 consecutive reports (D-102) |
| Expert count changes between generations | Lifecycle events. Decoders/strategies remap or re-evaluate (spec 34/35) |

*Implemented:* a neural generation encodes the probe windows with parent and child, computes linear CKA per expert pair, matches greedily (threshold 0.5) and moves matched experts to the parent's index; the matrix and matches are in `manifest.feedback.expert_alignment`. The trainer then writes one Helix **`Expert`** node per surviving expert (`expert_uid = <snapshot>/e<expert_id>`, `expert_id` = the stable index, `dim`, validation `usage` / `skill`, `specialisation` = gate mass per instrument on the probe windows, `born_gen` inherited along the chain), **`ExpertOf`** → its `Snapshot`, **`AlignedTo{cka, kind: carry}`** → the parent-generation expert it continues, and `died_gen` on parent experts no child continues (`fa_hub::experts`: `upsert_expert`, `align_expert`, `retire_expert`, `snapshot_experts`, `expert_parents`, `expert_alignments`; catalog in `helix/schema.hx` / `queries.hx`). The `ExpertSpace <snapshot>-e<k>` nodes stay for regions and vectors.

## 9. Acceptance criteria
**Remaining (why this spec stays):**
- **Item 2 is met on real market data** (2026-11-16, first live generation `gen-1m-cycle-01m7k5ss…`,
  9 687 bars of 9 instruments at 1m, 8 102 train / 1 432 validation windows, three candidate
  families): usage `K_eff` **3.99** ≥ 2, latent CKA **0.033** ≤ 0.7, SSL skill **0.547** ≥ 0.1, min
  expert skill **0.328** ≥ 0.02, no collapse — every candidate family passed, not only the selected
  one (`fsq_hybrid` 0.032/0.225/0.324, `hier_vae_flow` 0.023/0.250/0.324, `jepa_vae`
  0.033/0.547/0.328 for CKA / SSL skill / min skill). The D-102 thresholds were set on synthetic
  data and real bars clear them with room, so they stand rather than needing re-tuning.
- The generation was still **rejected**, by two checks that are not item 2: `carried_mass` 0.0 < 0.5
  (a neural generation whose parent is the gen-0 *statistical* snapshot carries no region mass — the
  check wants a comparable parent) and `mcmc_ece` 0.0375 > 0.02 (the fresh graph's calibration).
  Both need a second neural generation, with a neural parent, before they mean anything. Until then
  the active snapshots stay statistical.
- Promotion itself remains the owner's (`POST /api/latent/snapshots/{id}/promote`), and the shadow
  checks (`shadow_served`, `shadow_minutes`, `shadow_region_agreement`) are only evaluated there.

*Superseded (owner decision 2026-11-13, D-099):* rounds 1–4 (§6.3.1) showed that `K_eff ≥ 2`, residual correlation ≤ 0.1 and min skill ≥ 0.02 are not met together on validation; residual decorrelation is no longer a criterion.

*Done in round 4 (2026-11-11):* bias-adjustment balancing and expert split (§6.1), `cov_ema` (§6.3), PIT/CRPS calibration and per-expert decodability (§6.4), the `fa_expert_resid_corr` / `fa_expert_usage` series and per-eval matrices (§5), questions 1, 3, 5 (§10). *Round 5 (2026-11-13):* end-to-end objective with the diversity term (§6.3), usage `K_eff` / latent CKA / SSL skill (§6.4), `trainer sweep --synthetic N:K`.

*Status:* 1 holds on synthetic data for all four families, three seeds each, at the tightened D-102 limits (rounds 5–6); 2 measured per snapshot (`skill_e<k>`), enforced by the checklist, met on synthetic data; 3 holds (`ExpertView` with every §4.2 field, Rust → Go fixture round trip); 4 implemented with threshold 0.5 (CKA placeholder 0.7 not used), `AlignedTo` edges in Helix; 5 emitted per generation in the manifest and `Expert` nodes, live on `latent.expert.health` for learned active snapshots (live: usage `K_eff` gauge; latent CKA and SSL skill only from fits), kept per snapshot by control-api and shown on the console's Encoder snapshots page (its promotion checklist includes the D-099/D-102 items `expert_k_eff`, `expert_diversity`, `ssl_skill`, `expert_min_skill` — `models.ts NEURAL_CHECKS` — with missing ones flagged).
1. On a synthetic dataset built from K independent latent factors, the default genome of each family passes item 2 on validation: usage `K_eff` ≥ 2 (≥ K−1), `latent_cka_max` ≤ 0.7, SSL skill ≥ 0.1 (D-102), every expert's skill ≥ `min_skill`, no collapse. (Residual correlation is reported, not required.)
2. Every active expert beats the naive baseline by ≥ `min_skill` on the validation window.
3. The `ExpertView` proto round-trips between Rust and Go and contains every field listed in §4.2.
4. Expert identity alignment keeps ids for experts with CKA ≥ 0.7 (placeholder) across two consecutive generations.
5. Health metrics (§6.4) are emitted per generation and visible in the frontend latent explorer.

## 10. Open questions
*Decided 2026-11-13 (owner, D-099):* **Q1** — the spaces are trained end to end on reconstruction / SSL + next step with an expert-diversity term (latent CKA); residual decorrelation (options a–e below) is an optional regulariser, `λ_dec` 0 by default. **Q2** — λ weights are fixed genome defaults (evolvable genes). **Q6** — moot (no decorrelation requirement).

*Decided 2026-11-11 (provisional, superseded for Q1 by D-099):* **Q1** — the loss is (a), the gated correlation penalty on next-step residuals (vector-valued per target, or scalarised as the health measures it: gene `decor_scalar`), with (e) available as dual ascent (`decor_dual_lr`, `decor_eps`), (c) as `lambda_ncl`, `cov_ema` and the decorrelation hold-out as genes, and an exact post-hoc head solve (`head_solver` `constrained` | `disagree`); (b) HSIC and (d) orthogonal gradients are not pursued (round 4 shows the obstacle is generalisation of any residual-dependence estimate, which neither changes). Targets: `[next return, |next return|, next range]` per timescale; the checklist measures the scalarised residual. Which genome wins is the Pareto selection's job (spec 36), not a fixed default. **Q3** — decoders see every expert with its gate (gated-off experts carry gate 0 and stay readable); selection is the decoder genome's expert mask, as implemented in `ExpertView`. **Q5** — asset sets are handled **outside** the router: cross-asset attention before the router (spec 31 §6.4, `asset_attn`) and MCMC priors; experts may specialise on assets emergently (the `Expert.specialisation` gate mass per instrument shows it) but are not forced to.

1. **[Core blocker #2] The exact expert-space loss.** Blocks 31 (training loop), 34 (feedback form), 36. Options:
   - (a) Correlation penalty on residuals (default proposal, §6.3).
   - (b) HSIC / distance-correlation penalty (non-linear independence; more expensive).
   - (c) Negative-correlation learning on predictions.
   - (d) **Orthogonal-gradient formulation**: penalise alignment of experts' residual gradients instead of residual values.
   - (e) Constrained optimisation (Lagrangian, decorrelation as a constraint ≤ ε rather than a weighted term). Avoids hand-tuning λ.
   Sub-questions: what are the "next-step" targets per timescale (returns only, or also vol / latent / cross-asset)? Should residuals be scalarised or vector-valued?
2. Should λ weights be fixed config, evolved genes, or Lagrange multipliers (option e)?
3. Top-k routing vs soft routing for decoders: should a decoder see gated-off experts (with gate ≈ 0) or only active ones?
4. Is the expert count shared across timescales or independent per timescale? (Default: independent.)
5. How does asset selection interact with routing: experts as asset specialists, or asset sets handled outside the router?
6. Does the decorrelation hold within each timescale only, or also across timescales?

## 11. Deferred
- Cross-timescale decorrelation constraints.
- Expert-level online adapters (vs backbone-level, spec 36).
- Interpretability tooling (automatic naming of experts from specialisation summaries).
