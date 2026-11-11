# 32 — Expert Spaces

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Low |
| **Owners (code)** | `crates/fa-expert-space`, `crates/fa-encoders`, `bins/trainer`, `bins/model-runtime` |
| **Depends on** | 31 Expert encoders, 30 Latent regions, 33 MCMC graph, 34 Decoder population, 36 Training lifecycle |

## 1. Purpose
Expert spaces are the low-dimensional manifolds the encoders produce through a **learned mixture-of-experts router**. Each expert specialises in a part of the predictive structure. The system optimises for two things at once:
- **independence:** the experts' next-step prediction errors are ~uncorrelated;
- **generality:** each expert's latent stays reconstructable by many decoder families (past events, future trajectories, text, images).

This spec defines the router, the objective and the **decoder-facing contract** that everything downstream reads.

## 2. Scope
**In scope:**
- MoE router design and how the expert count emerges.
- Per-expert manifold projection (dimensionality reduction).
- The residual-decorrelation objective and its balance against generalisation.
- Expert identity across generations.
- The `ExpertView` contract consumed by decoders, MCMC and strategies.

**Out of scope:**
- Encoder backbones and modality fusion (spec 31).
- Region discovery on each expert space (spec 30).
- Decoder evolution mechanics (spec 34).

## 3. Decisions
- Expert partitioning comes from a **learned MoE router**. The number and content of experts **emerge from training**. Decorrelation pressure forces specialisation.
- The decorrelation target is the **cross-expert covariance of next-step prediction errors (residuals)**, driven to ≈ 0 (ensemble diversity). The latent codes themselves are **not** required to be decorrelated.
- This is balanced against **generalisation**: each expert must stay reconstructable by the decoder population (spec 34). From gen 1, this pressure comes from elite-decoder feedback.
- Expert spaces exist **per timescale** (one model per timescale, spec 31) and per encoder snapshot.
- Asset selection may be expressed through the router (experts specialising on asset subsets). Otherwise it comes from MCMC priors or evolution.

## 4. Interfaces

**Implemented today (D-008 in code; neural-family round):** `fa_expert_space::moe::MoeExpertSpace` is the learned expert space of the neural encoders (candle): a linear router over `K_max` experts (softmax, seeded Gaussian logit noise during training, top-k gates renormalised at inference and published as `router_weights`; the runtime keeps them and only routes encoders without a learned gate), per-expert projectors to `(mu, logvar)` with `mu = tanh(·)` (bounded like the statistical latents so region radii stay comparable; FSQ on the leading dims for `fsq_hybrid`), per-expert linear next-step predictors. Losses (`moe::decorrelation_loss`, `load_balance_loss`, `gaussian_kl`) are differentiable and trained jointly with the encoder (spec 31 §6.5). `health::HealthReport::compute_gated` measures §6.4 on the validation windows; `health::linear_cka` + `match_experts` implement §6.5. Experts with validation usage below `prune_usage_below` (0.05) are pruned after training, at least 2 kept (emergent count); the snapshot stores the surviving expert ids. The statistical family keeps its 3 fixed experts and the `EvidenceSoftmaxRouter`. Health of a fit lives in the snapshot manifest (`metrics.k_eff`, `resid_corr_max`, `skill_e<k>`, `usage_e<k>`, `params.health` with the full matrix) and on the `Expert` nodes (§5).

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
| Residual covariance matrices | VictoriaMetrics (pairwise series) + Garage (full matrices per eval) | `expert_resid_corr{ts,i,j}` |
| Usage / gate stats | VictoriaMetrics | `expert_usage{ts,expert}` |

## 6. Behaviour & algorithms

### 6.1 Router (proposal)
- A token-pooled router gives gate logits over `K_max` experts. Experts are selected by top-k with noisy gating during training.
- **Load-balancing:** use auxiliary-loss-free balancing (per-expert bias adjustment) to avoid fighting the decorrelation loss. A small auxiliary loss is the fallback.
- **Emergent count:** experts start at `K_max`. Experts with usage below `prune_usage_below` over `prune_window` are pruned at generation boundaries. An expert can **split** (clone + perturb) when its own residual variance is high and its specialisation is multimodal. The active count is reported per generation.

### 6.2 Projection to the expert manifold
Each expert has a projector (small MLP / attention pool) → Gaussian `(mu, logvar)` in `R^expert_dim`. This is the dimensionality reduction from the large encoder latent to the "expert space". For discrete families (spec 31 family 1), the expert also carries its code subset.

### 6.3 Objective (proposal — core blocker #2)
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

*Implemented form:* residuals are vector-valued over the three next-step targets of spec 31 §6.5 (`[next return, |next return|, next range]`, normalised); `L_decor = mean_targets Σ_{i≠j} corr(r_i, r_j)²` over every window of the minibatch (soft routing trains all predictors; no EMA of the covariance), `L_balance` is the Switch form `K · Σ f_i p_i` (auxiliary-loss balancing, not the bias-adjustment variant), `L_gen` at gen 0 is per-expert next-step prediction plus the family's SSL head, from gen 1 the D-009 frozen-decoder term. Defaults `λ_dec 1.0`, `λ_bal 0.01`, `β 1e-3`. Further genes of the objective (spec 31 `NeuralConfig`): `decor_scalar` (decorrelate the scalarised residual the §6.4 health measures instead of each target — both forms are allowed above), `decor_dual_lr` / `decor_eps` (open question 1 option (e): `λ_dec` by projected dual ascent on `L_decor ≤ ε`, checkpointed), `lambda_skill` / `skill_floor` (the `min_skill` collapse guard as a hinge on each expert's routed error ratio), `lambda_ncl` (option (c), negative-correlation learning on the scalarised predictions) and `head_refit_steps` (a full-batch refit of the per-expert next-step heads on frozen latents with exact correlations instead of minibatch estimates). Observed on live 1m bars (9 instruments, 12 h): `λ_dec = 1` reaches `K_eff ≈ 2.6` of 4 with residual correlations up to 0.64; `λ_dec = 0.1` gives better per-expert skill (≈ 0.25) but `K_eff ≈ 1.3` — the independence/skill trade-off of open question 1 is real on market data, and the 0.1 health target is not met by either.

#### 6.3.1 Can the checklist be met on the live data? (evidence, 2026-11-11)
**Bound.** Write an expert's prediction as `ŷ_k = f + d_k` with `f` the conditional mean and `d_k` any function of the window, so `d_k ⟂ ε` out of sample. The scalarised residual is `ē − d̄_k`; for two experts sharing windows its correlation is smallest when `d̄_j = −d̄_i`, giving `(V − D)/(V + D)` with `V = var(ē)` and `D ≤` the mean per-target deviation energy, while each expert's skill is at most `R̄² − D`. Both `|corr| ≤ c` and `skill ≥ s` are therefore reachable only if `R̄² − s ≥ V (1 − c)/(1 + c)`; with three experts sharing every window (needed, since two experts give `K_eff = 2/(1 + corr²) < 2`) the pairwise version needs `D ≥ 1.5 V`. A residual correlation near 0 is **not** reachable for skilled experts when the irreducible noise dominates — it requires the experts to disagree, symmetrically, by about as much as the noise. `trainer sweep` reports the margin `R̄² − s − V(1 − c)/(1 + c)` estimated by the trained mixture.

**Data.** 1m connector bars of the 9 live instruments (3 Kraken, 5 Yahoo, the synthetic heartbeat), 48 h ending 2026-11-11 03:20 UTC: 7068 bars, 5878 training / 1037 validation windows. The mixture reaches `R̄² ≈ 0.36` (per target: next return ≈ −0.01, |next return| ≈ 0.55, next range ≈ 0.60 — volatility is predictable, direction is not) with `V ≈ 0.12`, so the margin is ≈ +0.25 for two experts and ≈ +0.15 for three: **not ruled out by the bound**.

**Sweeps** (`trainer sweep`, release build, `jepa_vae`, 600–1500 steps, 42 genomes): default genome `K_eff 1.36`, corr 0.69 (router collapse); balancing (`λ_bal ≥ 0.3`) and `decor_scalar` with all experts active lift `K_eff` to 2.8–2.95 and bring the correlation to 0.13–0.30, but then one expert's skill turns negative (−0.1 … −1.6); the skill hinge, batch 256, dual ascent, NCL (0.5–0.9) and longer training move along the same front (best pairs: corr 0.13 at min skill −0.79; min skill +0.035 at corr 0.58). The in-sample health of the same fits is no better (train `K_eff` 1.5–2.7, corr 0.53–0.75), so the gap is **optimisation, not generalisation**: the minibatch D-008 objective does not find the symmetric-disagreement solution the bound allows. A full-batch refit of the linear next-step heads on the frozen latents (exact correlations, `λ_dec` up to 100, skill hinge) gets closest — `K_eff 2.96`, correlation 0.12, but min skill −0.28 (in sample 0.28 / −0.25): each expert alone reaches far less than the mixture's `R̄²` (single-expert skills 0.1–0.26), which leaves the per-expert margin of the three-expert bound (`R̄²_k − s − 1.5 V`) near zero.

**Conclusion.** No snapshot passing `K_eff ≥ 2`, residual correlation ≤ 0.1 and min skill ≥ 0.02 together was found on this data; the thresholds were not changed. The live data is too short (≈ 2 days after the 2026-11-10 reset) to tell whether more history helps. Next steps: a constrained solver for the heads (the residual covariance is quadratic in the linear heads given the latents), experts with disjoint routing (top-1) where the gated correlation only compares shared windows, and revisiting whether the 0.1 target should apply to the scalarised residual of predictable targets only (a spec decision, not taken here).

### 6.4 Measured quantities (health)
- Residual correlation matrix per timescale on a held-out validation window (not the locked holdout, spec 41).
- **Effective number of independent experts:** `K_eff = (Σλ)² / Σλ²` over eigenvalues of the residual correlation matrix.
- Per-expert skill vs naive baseline, calibration (PIT / CRPS) of `next_step`.
- Per-expert decodability: best elite decoder score per family (spec 34).

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
| `lambda_dec` | 1.0 | Evolvable |
| `lambda_gen` | 1.0 | Evolvable |
| `lambda_bal` | 0.01 | If aux-loss fallback used |
| `cov_ema` | 0.99 | Residual covariance EMA |
| `min_skill` | 2% better than naive (CRPS) | Collapse guard |
| `prune_usage_below` | 0.01 | |
| `prune_window` | 1 generation | |
| `target_resid_corr_max` | 0.1 | Health target |

## 8. Failure modes
| Failure | Response |
|---|---|
| Router collapse (one expert takes everything) | Balancing kicks in; if `K_eff < 2` after training → snapshot not promotable (spec 36) |
| Degenerate decorrelation (experts with no skill) | `min_skill` constraint fails → expert pruned or snapshot rejected |
| Residual correlation spikes live (regime change) | Health signal on bus; strategies/allocator can widen uncertainty; can trigger refresh (spec 36) |
| Expert count changes between generations | Lifecycle events. Decoders/strategies remap or re-evaluate (spec 34/35) |

*Implemented:* a neural generation encodes the probe windows with parent and child, computes linear CKA per expert pair, matches greedily (threshold 0.5) and moves matched experts to the parent's index; the matrix and matches are in `manifest.feedback.expert_alignment`. The trainer then writes one Helix **`Expert`** node per surviving expert (`expert_uid = <snapshot>/e<expert_id>`, `expert_id` = the stable index, `dim`, validation `usage` / `skill`, `specialisation` = gate mass per instrument on the probe windows, `born_gen` inherited along the chain), **`ExpertOf`** → its `Snapshot`, **`AlignedTo{cka, kind: carry}`** → the parent-generation expert it continues, and `died_gen` on parent experts no child continues (`fa_hub::experts`: `upsert_expert`, `align_expert`, `retire_expert`, `snapshot_experts`, `expert_parents`, `expert_alignments`; catalog in `helix/schema.hx` / `queries.hx`). The `ExpertSpace <snapshot>-e<k>` nodes stay for regions and vectors.

## 9. Acceptance criteria
*Status:* 1 in miniature (`moe::tests::two_experts_learn_decorrelated_specialisations`: two independent factors, two experts, both beat naive, decorrelation term < 0.2), not at `target_resid_corr_max` on market data (§6.3.1); 2 measured per snapshot (`skill_e<k>`), enforced by the checklist; 3 holds (`ExpertView` with every §4.2 field, Rust → Go fixture round trip); 4 implemented with threshold 0.5 (CKA placeholder 0.7 not used), `AlignedTo` edges in Helix; 5 emitted per generation in the manifest and `Expert` nodes, live on `latent.expert.health` for learned active snapshots; the frontend latent explorer does not show them yet.
1. On a synthetic dataset built from K independent latent factors, the router recovers ≥ K−1 experts, and the residual correlation off-diagonal max is ≤ `target_resid_corr_max`.
2. Every active expert beats the naive baseline by ≥ `min_skill` on the validation window.
3. The `ExpertView` proto round-trips between Rust and Go and contains every field listed in §4.2.
4. Expert identity alignment keeps ids for experts with CKA ≥ 0.7 (placeholder) across two consecutive generations.
5. Health metrics (§6.4) are emitted per generation and visible in the frontend latent explorer.

## 10. Open questions
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
