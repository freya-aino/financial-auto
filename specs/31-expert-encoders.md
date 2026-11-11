# 31 — Expert Encoders

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Low-medium |
| **Owners (code)** | `crates/fa-encoders`, `bins/trainer`, `bins/model-runtime` |
| **Depends on** | 10 Ingestion, 12 Reference data, 32 Expert spaces, 30 Latent regions, 36 Training lifecycle, 60 Ops |

## 1. Purpose
Expert encoders turn the multimodal, multi-asset input stream into the latent representation that the rest of the system runs on. They are a modernised VAE family, trained per timescale. Through the MoE router of spec 32, they produce the **expert spaces**. Several encoder families compete. The family is itself a gene evaluated under the shared Pareto fitness.

## 2. Scope
**In scope:**
- The `EncoderFamily` abstraction and the four initial families.
- Modality front-ends (market series, text, images/charts, video frames, audio).
- The per-timescale model ladder (capacity, context and prior sized per scale).
- Asset conditioning and end-to-end asset selection.
- burn vs candle usage boundary.
- Inference contract for the live runtime.

**Out of scope:**
- The expert decomposition, router and decorrelation objective (spec 32).
- Region discovery internals (spec 30); families only declare their method.
- The generation schedule, snapshots and adapters (spec 36).
- Decoders (spec 34).

## 3. Decisions
- **ML is pure Rust.** **burn** (wgpu / CUDA backend) handles all training. **candle** loads and runs pretrained weights (text encoders, vision encoders, Whisper-class ASR). Python isn't used anywhere.
- **All modern families are provided and compete:**
  1. Hybrid **FSQ/VQ discrete + continuous residual** latent.
  2. **Hierarchical VAE + flow-matching prior** (NVAE-style multi-scale continuous latents).
  3. **JEPA backbone + VAE head** (predictive non-reconstructive backbone, decodable head).
  4. **Latent diffusion** (autoencoder + diffusion model in latent space).
  The set can be extended via the `EncoderFamily` trait.
- Each family **declares its region discovery method and its prior/context mechanism** (spec 30).
- **One custom-scaled model per timescale.** Timescale ladder: seconds, minutes, hours, days, weeks. Capacity, context length and prior are sized per scale. The ladder is first-class config.
- **Modalities:** market series, text (news, filings, transcripts), images/charts, video frames (broadcasts), audio (earnings calls, pressers; via ASR + prosody).
- **Asset selection is part of the model:** end-to-end via router/attention over assets where trainable. Otherwise selection comes from evolution (spec 34/35) or MCMC covariance priors (spec 33).
- **Learning over time:** generation snapshots + online adapters (spec 36).
- **Gen 0** is self-supervised (reconstruction + next-step). Decoder-fitness feedback starts at gen 1 (spec 36).

## 4. Interfaces

**Implemented today (D-058, D-062, D-065, and the neural-family round):** two encoder lines serve the same `EncoderFamily` / snapshot contract.

* **Statistical family** (`fa_encoders::StatisticalEncoder`): reads connector bars (OHLCV) where a venue publishes them and trades elsewhere, is fitted by the trainer into Garage snapshots (`weights.bin` = folded normalisation statistics and gains), served by model-runtime with shadow serving and online adapters, and tuned per generation by a stated proxy search.
* **Neural families** (`fa_encoders::neural`, candle, CPU by default): `fsq_hybrid` (family 1) and `jepa_vae` (family 3) on a market front-end of 7 per-bar features (`ret, gap, range, body, wick, ln(1+volume), ln(1+trades)`, z-scored with an `InputNorm` fitted on the training windows and stored with the weights). Backbone is a family gene: `tcn` (causal dilated temporal convolutions, kernel 3, dilation 2^l, residual + RMS norm, pooled `[last, mean]`) or `gru`. On top sits the learned expert space of spec 32 (`fa_expert_space::moe`). `fsq_hybrid` quantises the leading `len(fsq_levels)` dims of every expert latent with FSQ (straight-through, default `[8,5,5,5]`, code index exported) and keeps the rest a continuous Gaussian residual; its SSL term reconstructs the last `recon_steps` bars from the gated expert latents. `jepa_vae` has Gaussian expert heads and predicts the **EMA target encoder's** embedding of the window `jepa_horizon` steps ahead (no input reconstruction). Both train the spec 32 objective (§6.5 below). Weights are safetensors in `weights.bin` (`p.<param>`, metadata `format=fa-neural-v1`, `family`, `config` = genome JSON, `norm`, `active` experts). Initialisation and minibatches are seeded (`ParamStore` / `Rng`), so the same data + genome + seed gives bit-identical weights on CPU and a resumed job continues bit for bit (parameters, Adam moments, EMA target and step are checkpointed). `NeuralEncoder` serves a snapshot in model-runtime and fills `ExpertView.router_weights` with its learned top-k gates.
* **`hier_vae_flow` (family 2, D-070 follow-up):** a top latent `z_top ∈ R^top_dim` (default 4, Gaussian posterior from the pooled backbone state, reparameterised in training) conditions the expert latents **top-down** (`h + gelu(W·z_top)` feeds the router and projectors: a two-level hierarchy), the SSL head reconstructs the last `recon_steps` bars from both levels, both levels carry a KL to `N(0, I)`, and a **flow-matching prior** (`fa_encoders::neural::prior`, velocity MLP, conditional flow-matching loss on the detached top latent) models the aggregate posterior; `prior::flow_log_density` integrates the probability-flow ODE backwards (Euler, divergence by central differences) for the exact density used as novelty (`NeuralEncoder::prior_score` = `−log p(z_top)`).
* **`latent_diffusion` (family 4):** the autoencoder (backbone + expert space + reconstruction) plus a **conditional DDPM** (ε-prediction, cosine schedule, `diffusion_steps` 50) over the full expert latent (expert means at their expert-id slots, `K_max · expert_dim`, stable under pruning), trained on the latent of the window `jepa_horizon` steps ahead given the current one (detached; 20 % of rows unconditioned, so the same net is the unconditional prior). `NeuralEncoder::sample_futures` draws deterministic DDIM samples of the future expert latents; `prior_score` is the unconditional denoising error. **D-010 is enforced:** the trainer refuses the family below days (`config_for`), `NeuralEncoder::from_manifest_on` refuses a manifest at another timescale.
* **Region discovery as declared:** families 2 and 4 declare `density`; `fa_encoders::neural::regions::DensityModes` (fixed-bandwidth mean shift, seeds strided over the canonically sorted points, modes merged within `bandwidth/2`, minimum share 1 %, centroids rounded and sorted, so the fit is order-free) is what the trainer fits for them (bandwidth = `region_radius`) and passes `fa_patterns::conformance::assert_conformance`; `fsq_hybrid` / `jepa_vae` get their declared `codebook` / `mapper` discovery through `fa_patterns::discovery_for_method` (round 3; the codebook merges codes on the experts' next-step residual statistics, spec 30 §6.2), both also passing it.
* **Per-window extras** (`EncoderFamily::encode_with_extras`, default none): a neural encoder also returns each expert's next-step prediction, FSQ codes, the prior score and the realised (normalised) targets of the previous window; model-runtime turns them into `ExpertView` and live expert health (spec 32 §4.3).
* **Media towers (D-010):** `fa_encoders::media` defines `MediaTower`, the pretrained-tower registry (`PretrainedTowerSpec`: Garage key under `fa-models/pretrained/<name>/`, SHA-256, licence; Whisper-small, bge-small, SigLIP-base listed with empty hashes) and `select_tower`, which falls back to `HashedEmbeddingTower` (signed feature hashing of tokens / byte 4-grams, L2-normalised) when weights are absent or do not verify. No pretrained weights are downloaded and no candle runner for them is built in; `point_in_time` / `embed_window` implement the §6.3 publish-time guard. The neural families are market-only: media embeddings are not fused yet.
* **Device:** `FA_ML_DEVICE` / `--device` (`cpu` default, `cuda[:n]`); CUDA is the `cuda` cargo feature of `fa-expert-space`, `fa-encoders`, `trainer` and `model-runtime`, off by default (no GPU on the dev host); without it a CUDA request resolves to the CPU with a logged note.
* **candle, not burn, trains** (deviation from §3 / §6.6 and ADR 0003's split): candle's autograd is sufficient for these model sizes and keeps one tensor library; burn is not a dependency. candle is pinned to 0.10.x because 0.11's AVX512-VNNI kernels do not compile with the host toolchain.

### 4.1 Family trait (proposal)
```rust
pub trait EncoderFamily: Send + Sync + 'static {
    const NAME: &'static str;
    type Config: Serialize + DeserializeOwned + Genome;   // evolvable hyperparameters
    type Model<B: Backend>: Module<B>;

    fn build<B: Backend>(cfg: &Self::Config, ts: &TimescaleSpec, dev: &B::Device) -> Self::Model<B>;

    /// Encode a batch of multimodal windows into the pre-router latent (spec 32 splits into experts).
    fn encode<B: Backend>(m: &Self::Model<B>, x: &MultimodalBatch<B>) -> EncoderOutput<B>;

    /// Self-supervised training loss (gen 0) + hooks for decoder-feedback losses (gen ≥ 1).
    fn ssl_loss<B: AutodiffBackend>(m: &Self::Model<B>, x: &MultimodalBatch<B>) -> LossTerms<B>;

    /// Declares how latent regions are found for this family (spec 30).
    fn region_discovery(&self) -> Box<dyn RegionDiscovery>;

    /// Declares the prior used for density, sampling and novelty.
    fn prior(&self) -> PriorKind;          // Codebook | FlowMatching | Gaussian | Diffusion

    /// Cost model used as a Pareto objective (FLOPs, latency, VRAM) for this timescale.
    fn cost(cfg: &Self::Config, ts: &TimescaleSpec) -> CostEstimate;
}

pub struct EncoderOutput<B: Backend> {
    pub z: Tensor<B, 3>,            // [batch, tokens, d] pre-router latent tokens
    pub z_disc: Option<Tensor<B, 3, Int>>,  // discrete codes if family has them
    pub uncertainty: Tensor<B, 3>,  // per-dim log-variance or equivalent
    pub asset_attn: Option<Tensor<B, 3>>,   // attention over assets (for asset selection)
}
```

### 4.2 Modality front-ends (proposal)
```rust
pub trait ModalityEncoder<B: Backend>: Module<B> {
    fn modality(&self) -> Modality;            // Market | Text | Image | Video | Audio
    fn tokens(&self, raw: &ModalityInput<B>) -> TokenBatch<B>;  // [batch, n_tok, d_model] + timestamps
}
```
Front-ends output timestamped token sequences in a shared `d_model`. The family backbone fuses them (see §6.3).

### 4.3 Runtime
- `model-runtime` loads the **active snapshot** per timescale (spec 36) and consumes canonical events (spec 10). It emits:
  - `latent.z.<timescale>.<instrument>` (`fa.v1.LatentFrame`: snapshot id, expert latents after spec 32 router, uncertainty)
  - region memberships (spec 30).
- Weights are stored in Garage as safetensors (`fa-models/snapshots/<snapshot_id>/weights.bin`, D-058 layout; neural families) or `FASTAT01` (statistical); metadata goes to Helix (`Snapshot` node, `encoder_family` = `statistical | fsq_hybrid | hier_vae_flow | jepa_vae | latent_diffusion`).

## 5. Data model
| Entity | Store | Fields |
|---|---|---|
| `EncoderSnapshot` | Helix node | id, family, timescale, generation, config (genome json), seed, data window, parent snapshot, fitness vector, weights_uri |
| Weights / optimizer state | Garage | `models/encoders/<timescale>/<snapshot_id>/{model.safetensors,opt.bin,config.json}` |
| `TimescaleSpec` | config (`config/platform.toml` → Postgres mirror) | name, step, context length, horizon, backfill window |
| Pretrained weights (text/vision/ASR) | Garage `models/pretrained/` | name, license, hash |
| Latent frames | NATS stream + Helix vectors (sampled) + VM summary features | see spec 20 |

## 6. Behaviour & algorithms

### 6.1 Timescale ladder (proposal defaults)
| Timescale | Step | Context (lookback) | Horizon (matched scope) | Relative capacity | Modalities |
|---|---|---|---|---|---|
| seconds | 1 s | 10 min | 20 min | XS | market, (live ASR text) |
| minutes | 1 min | 1 h | 2 h | S | market, text, audio |
| hours | 1 h | 1 d → 2 d | 2 d | M | all |
| days | 1 d | 30 d → 60 d | 60 d | L | all |
| weeks | 1 w | 1 y | 2 y | M | market, macro, text |

Lookback ≈ horizon scale (spec 00: 1h→2h, 1d→2d). The exact numbers are tunables.

### 6.2 Families (initial specifications, proposal)
1. **Hybrid FSQ + continuous residual.** Transformer backbone → FSQ quantizer (no codebook collapse, no commitment loss) gives a discrete code per token. A small Gaussian residual latent captures the rest. Prior: autoregressive or count-based over codes. Natural fit for Markov states.
2. **Hierarchical VAE + flow-matching prior.** Multi-scale latent hierarchy (top-down inference). The flow-matching prior on the top latent gives exact density for novelty and clustering.
3. **JEPA + VAE head.** Context and target encoders (EMA target) predict masked/future latent tokens in embedding space. This makes them robust to noise in reconstruction targets. A VAE head on the predictor output provides decodable, probabilistic latents for spec 32.
4. **Latent diffusion.** Compact autoencoder plus a diffusion/flow model over latents. It has the strongest generative quality and is used for sampling futures and density. It is the most expensive and probably only viable at the hours/days timescales on one GPU.

### 6.3 Multimodal fusion (proposal)
- Every modality front-end emits tokens with `(event_time, publish_time)`. **Only tokens with `publish_time ≤ window end`** enter a window (point-in-time correctness, spec 41).
- Fusion: a time-aligned perceiver-style cross-attention compresses a variable number of modality tokens into a fixed latent token set per window. Missing modalities are masked, not imputed.
- Text / image / ASR front-ends start from **frozen pretrained candle models**. Their embeddings are projected by trainable burn adapters (a gradient boundary at the candle→burn interface). Making them trainable is open question 3.

### 6.4 Asset conditioning and selection (proposal)
- Every market token carries an **asset embedding** (learned, from instrument UUID) plus static features from reference data (spec 12: class, venue, currency, sector).
- The backbone runs **cross-asset attention** over a candidate set of assets. The attention weights (`asset_attn`) are exported. A sparsity penalty (entropy / L1 on attention) pushes selection.
- Candidate sets come from MCMC cross-asset priors (spec 33) or evolution (spec 34/35) when the end-to-end route doesn't scale (universe of thousands of instruments).

### 6.5 Training objective (gen 0, SSL)
*Implemented form (neural families):* `L = λ_next Σ_k L_pred,k + λ_mix L_mix + λ_dec L_decor + λ_bal L_balance + β KL + λ_ssl L_ssl (+ λ_anchor L_anchor + λ_fb L_feedback from gen 1)`, with `L_pred,k` the per-expert next-step MSE on the normalised targets `[next return, |next return|, next range]`, `L_mix` the MSE of the gate-weighted mixture prediction (trains the router), `L_ssl` the family's reconstruction / JEPA term, and the spec 32 terms. Defaults (CPU-sized): context 64 (1s/1m), 48 (1h), 32 (1d/1w) bars, `d_model` 32, 4 layers, `K_max` 4, `expert_dim` 8, top-2, 600 Adam steps of batch 64, lr 3e-3, 15 % validation split with an embargo of the horizon.

`L_enc = L_recon(modalities) + β·L_KL_or_quant + λ_next·L_next_step + L_spec32(router, decorrelation)`
- `L_recon`: per-modality reconstruction (market series: NLL under predicted distribution; embedding-level reconstruction for text/vision/audio tokens. Pixel/waveform reconstruction is not done at gen 0).
- `L_next_step`: predict the next-step latent and the next-step market targets (returns, volatility) per timescale. Supplies the residuals used by spec 32.
- From gen 1, decoder-feedback terms are added (spec 34 §6.6, spec 36).

### 6.6 burn vs candle boundary
| Use | Library |
|---|---|
| All trainable encoder/decoder/router modules | burn |
| Pretrained text encoder (embeddings) | candle (frozen), output → burn tensor |
| Pretrained vision encoder (chart/frame embeddings) | candle (frozen) |
| ASR (Whisper-class), audio features | candle |
| Online adapters (LoRA-like) on burn backbones | burn |
Tensor handoff happens via host memory or a shared CUDA buffer (proposal: start with host copies; optimise later).

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `timescales` | seconds, minutes, hours, days, weeks | Ladder in config |
| `d_model` per timescale | 128 / 256 / 384 / 512 / 384 | XS…L |
| `latent_tokens` per window | 8 / 16 / 32 / 32 / 16 | |
| `fsq_levels` | [8,5,5,5] | Family 1 |
| `beta_kl` | 1.0 (annealed) | Families 2–4 |
| `lambda_next` | 1.0 | Next-step weight |
| `asset_set_size` | 32 | Assets per cross-attention set |
| `asset_attn_sparsity` | 1e-3 | |
| `families_enabled` | all 4 at hours/days; 1–3 at seconds/minutes | GPU budget |
| `top_dim` | 4 | Family 2 top latent |
| `prior_hidden` | 64 | Flow / denoiser width |
| `lambda_prior` | 1.0 | Flow-matching / diffusion loss weight |
| `diffusion_steps` | 50 | Family 4, cosine schedule |
| `cond_drop` | 0.2 | Family 4 unconditional share |

## 8. Failure modes
| Failure | Response |
|---|---|
| Posterior collapse / codebook collapse | Detected via KL / code-usage monitors. Candidate is marked unfit (it receives a bad Pareto score, not a crash) |
| Modality outage (e.g. no news feed) | Tokens masked. Model must stay calibrated. Acceptance test covers it |
| VRAM OOM | Job preempted and retried with a smaller batch (spec 60 GPU queue). Repeated OOM → config gene penalised |
| Look-ahead leak (token with publish_time > window end) | Hard assertion in the batch builder. Training aborts |
| Pretrained weights unavailable / license unclear | Family falls back to a learned-from-scratch front-end. Logged |

## 9. Acceptance criteria
*Status:* 1 holds for all four families (`families_train_save_load_and_resume_deterministically` iterates `NeuralFamily::ALL`: shapes, loss decreases, seed determinism, save/load, bit-exact resume, wrong-family refusal; `hierarchical_flow_and_latent_diffusion_priors` adds the prior losses, the density / novelty ordering, seeded future sampling and the D-010 gate; their `density` discovery passes `fa_patterns::conformance`), on synthetic data; 2 holds (one snapshot per timescale, trained and loaded independently); 3 holds for the media path (`media::point_in_time`), the market path has no publish-time lag; 4 not measured (no reference GPU; CPU serving of the default genome keeps up with live 1m bars of 9 instruments, not benchmarked); 5 open (no media fused); 6 holds bitwise on CPU.
1. Each of the 4 families implements `EncoderFamily`, trains on a synthetic multimodal dataset, and passes the conformance suite (shapes, determinism with seed, save/load round-trip, cost estimate).
2. One model per configured timescale can be trained and loaded independently.
3. The batch builder provably excludes tokens with `publish_time` after window end (unit + property tests).
4. Live inference for the minutes timescale meets the latency budget (placeholder: ≤ 50 ms per step for the active asset set on the reference GPU).
5. Masking any single modality degrades the next-step loss gracefully (no NaN, bounded increase).
6. Snapshots are reproducible: same data window + genome + seed → bitwise-equal weights on the same backend (or documented nondeterminism tolerance for GPU kernels).

## 10. Open questions
1. **[Core blocker #4] Pure-Rust media feasibility.** Blocks the video/audio modalities and the latent-diffusion family. Options:
   - (a) Spike first: Whisper-class ASR + CLIP/SigLIP-class vision encoder in candle, plus a small latent diffusion trained in burn on one GPU. Measure throughput.
   - (b) Restrict v1 media to **frozen embeddings** (candle inference only) and train only light adapters in burn. Defer trainable vision/audio.
   - (c) Restrict latent diffusion to the days/weeks timescales.
   - (d) Allow an exception to pure Rust for offline media preprocessing (contradicts ADR 0003; needs explicit approval).
   - Recommended: (a) → then (b) + (c).
2. How are the modalities fused (perceiver cross-attention vs early concat vs late fusion per expert)? Interacts with the router in spec 32.
3. Should pretrained text/vision front-ends be fine-tuned (more capacity, more VRAM) or stay frozen?
4. Asset selection at scale: how big a candidate set can end-to-end attention handle on one GPU, and what is the hand-off rule to MCMC/evolutionary selection?
5. Do all families share the same modality front-ends (fair comparison) or can front-ends be family genes?
6. Is the latent-diffusion family's cost acceptable for live inference at all, or is it offline-only (sampling futures for decoders)?

## 11. Deferred
- Pixel- or waveform-level reconstruction losses in encoders (handled by visual/audio decoders in spec 34 instead).
- Order-book-depth modality (depends on the storage modeling rule in spec 20).
- Multi-GPU / distributed data-parallel training (spec 60 horizontal scaling).
