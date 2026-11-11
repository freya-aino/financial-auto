# fa-expert-space

MoE router, expert spaces, residual-decorrelation objective, expert health, LoRA, decoder-facing contract.

**Status:** implemented — `ExpertView` contract, `ExpertSpaceConfig`, the `Router` stage trait with the baseline `EvidenceSoftmaxRouter` (softmax over per-expert energy or confidence, used by the statistical family), and the **learned MoE expert space on candle** (`moe`, D-008, D-070) that the neural encoder families of `fa-encoders` train end to end. All tested; feature `cuda` passes through to candle (off by default).

## Modules
| Module | What |
|---|---|
| `lib` | `ExpertView` / `ExpertLatent` (in-process form of `fa.v1.ExpertView`), `ExpertSpaceConfig`, host-side `DecorrelationLoss` / `ResidualCorrelationPenalty` (D-008), `pearson`, `router_load_balance` |
| `router` | `Router` trait (stage boundary in model-runtime), `EvidenceSoftmaxRouter`; a neural encoder fills `router_weights` with its learned top-k gates, which the runtime keeps |
| `moe` | `MoeConfig` (K_max 4, `expert_dim` 8, top-2, `fsq_levels`, `λ_dec` 1.0, `λ_bal` 0.01, `β` 1e-3, router noise, `prune_usage_below` 0.05), `MoeExpertSpace` (noisy linear router, per-expert `(mu, logvar)` projectors with `tanh`-bounded `mu`, per-expert next-step predictors; `forward_w` reads any `Weights`), `fsq` (straight-through FSQ), `decorrelation_loss` / `decorrelation_loss_gated` (pairwise residual correlation weighted by detached `p_i·p_j`, the form training uses), `load_balance_loss` (Switch), `gaussian_kl`, `top_k_gates`, seeded `router_noise` |
| `health` | spec 32 §6.4–§6.5: `HealthReport::compute_gated` (residual correlation over windows where both experts are routed, all windows when < 30 are shared; `K_eff`; routed per-expert skill vs naive; usage), `k_eff`, `linear_cka`, greedy `match_experts` (threshold set by the trainer, 0.5) |
| `lora` | rank-r adapters `W' = W + (α/r)·A·B` (`B = 0`, so a fresh adapter is the identity), `LoraView` (adapted forward passes without code changes), `LoraSet::merge_into`, `delta_norm` drift guard (spec 36 §6.5) |
| `nn` | seeded `ParamStore` (initialised from `Rng(seed, name)`, safetensors round trip) and `Adam` with named moments, so fits are bit-reproducible and checkpoints resume exactly |

## Owns
`ExpertView` contract consumed by decoders and strategies; `Router` stage boundary; the learned expert space and its health / identity measures.

## Governing specs
- [32-expert-spaces](../../specs/32-expert-spaces.md)
