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
| `Expert` | Helix node | expert_id, snapshot_id, timescale, dim, usage, specialisation summary (top assets / modalities / regimes), born_gen, died_gen |
| `EXPERT_OF` | Helix edge (Expert → EncoderSnapshot) | |
| `ALIGNED_TO` | Helix edge (Expert → Expert prev gen) | similarity (CKA), kind |
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

### 6.4 Measured quantities (health)
- Residual correlation matrix per timescale on a held-out validation window (not the locked holdout, spec 41).
- **Effective number of independent experts:** `K_eff = (Σλ)² / Σλ²` over eigenvalues of the residual correlation matrix.
- Per-expert skill vs naive baseline, calibration (PIT / CRPS) of `next_step`.
- Per-expert decodability: best elite decoder score per family (spec 34).

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

## 9. Acceptance criteria
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
