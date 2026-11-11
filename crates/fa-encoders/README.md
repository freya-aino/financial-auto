# fa-encoders

Expert encoder families on candle (training and serving; D-070 departs from ADR 0003's burn/candle split) and the media-tower contract.

**Status:** implemented — `EncoderFamily` trait (`encode`, `encode_with_extras`), the deterministic **`StatisticalEncoder`** (`baseline.rs`), parameter-free (gen-0 baseline) or with fitted normalisation statistics from a snapshot artifact (`snapshot.rs`, spec 36, D-058), and the four **neural families** of spec 31 (`neural`: `fsq_hybrid`, `hier_vae_flow`, `jepa_vae`, `latent_diffusion`; D-070, D-073) with safetensors snapshots, LoRA adapters and their declared region discovery. CPU by default; feature `cuda` (with `FA_ML_DEVICE=cuda[:n]`) moves training and serving to a GPU (unverified: no GPU on the dev host). `media.rs` is the D-010 tower interface with a hashed fallback only (no pretrained weights downloaded; media are not fused into the neural families yet).

## Statistical baseline
- Input: `EncoderInput::from_bars(as_of_ms, &[MarketBar])` (rows `[open, high, low, close, volume, trade_count]`), the last `context_steps` bars are used.
- Output: `ExpertView` with 3 experts (`EXPERT_NAMES`: returns, volatility, activity), each `EXPERT_DIM = 8` features bounded to (−1, 1) via `tanh(z/2)` of in-window z-scores and correlations; `log_var = −ln(n_bars)` per dim (short history = wide posterior). `router_weights` are left to the router stage.
- Snapshot id `stat-v1-<timescale>`; bump `StatisticalEncoder::VERSION` when feature definitions change.

## Snapshots (`snapshot.rs`, spec 36 §5)
- `SnapshotManifest` (snapshot_id, encoder_family, timescale, generation, parent_snapshot_id, created_at, params, probe-set reference, data window, metrics, feedback, seed) → `manifest.json`; `NormStats` (per-(expert, dim) mean/std over the training windows, `to_bytes`/`from_bytes`, magic `FASTAT01`) → `weights.bin`; constants `MANIFEST_FILE`, `WEIGHTS_FILE`, `REGIONS_FILE`, `REGION_MAP_FILE`, `STATISTICAL_FAMILY`.
- `NormStats.gain` (per expert, default 1) multiplies the z-score before the squash; it is how the trainer's feedback search sets the squash scale and per-expert feature weights (`gain = w_e / s`). `to_bytes` **folds** gains into σ (`σ / g`) and always writes `FASTAT01`, so binaries built before gains existed load the same encoder; `from_bytes` also reads `FASTAT02` (gains appended). `NormStats::pool(parts, equal_weight)` merges per-instrument statistics (by window count = one Welford pass over all windows; or equal weight per instrument for `per_instrument_norm`).
- `DataRef.instruments[]` (`InstrumentDataRef{instrument, source: bars|trades, n_bars, start_ms, end_ms}`) records the series of a multi-instrument fit; `AdapterGuard{max_drift, window_hours, checkpoint_interval_secs}` lives in `params.adapter_guard` (`SnapshotManifest::adapter_guard()`, default 1.0 / 1 h / 900 s); `SnapshotManifest::metric(name)`.
- `StatisticalEncoder::fit(ts, bars, &FitConfig)` fits the statistics (one window per `stride` bars, resumable in chunks with `fit_window_ends` + `fit_into` on a serialisable `NormAccumulator`); `StatisticalEncoder::from_manifest(&manifest, &weights)` rebuilds the encoder of a snapshot, which z-scores every raw feature against the fitted distribution before the squash. `StatisticalEncoder::new` (no stats) stays the gen-0 baseline.

## Neural families (`neural/`, spec 31 §6.2, D-070, D-073)
- `model`: `NeuralConfig` (the genome: family, backbone `tcn`|`gru`, `context_steps`, `d_model` 32, `layers` 4, `moe: fa_expert_space::moe::MoeConfig` (K_max 4, `expert_dim` 8, top-2), SSL / prior / objective weights, head-solver genes, 600 Adam steps of batch 64 at lr 3e-3, 15 % validation) and `NeuralModel` (7-feature market front-end → backbone → learned expert space → family head). Family 1 FSQ codes + residual with reconstruction, family 2 top latent conditioning the experts + flow-matching prior, family 3 JEPA against an EMA target, family 4 autoencoder + conditional DDPM (day/week only, D-010: `NeuralFamily::allows`). Snapshot id prefixes `fsq-v1`, `hvf-v1`, `jepa-v1`, `ldm-v1`.
- `data`: per-bar features `[ret, gap, range, body, wick, ln(1+volume), ln(1+trades)]`, next-step targets `[ret, |ret|, range]`, `InputNorm` (fitted, stored with the weights; re-fitted by the input adapter), `Dataset` with time-ordered train / validation windows.
- `train`: `NeuralTrainer` — the spec 31 §6.5 / spec 32 §6.3 loss, `warm_start`, D-007 `Anchors`, the D-009 `DecoderFeedback` hook (implemented by the trainer's elite / probe feedback), `evaluate` (spec 32 §6.4 health), `prune`, `refit_heads` (penalised full-batch refit or the `heads` KKT solver), `checkpoint` / `from_checkpoint` (bit-exact resume), seeded throughout.
- `heads`: constrained least squares for the per-expert next-step heads (pairwise residual covariance as equality constraints, spec 32 §6.3.1).
- `prior`: flow matching (`flow_log_density`, `flow_sample`) and latent diffusion (`diffusion_nll`, `ddim_sample`).
- `regions`: `DensityModes` (mean shift) for families 2/4; `discovery_for(family, …)` routes `fsq_hybrid` → codebook and `jepa_vae` → mapper through `fa_patterns::discovery_for_method` (D-080).
- `adapter`: LoRA online adapters (`LoraConfig` rank 8, α 8, lr 1e-4, 100 steps, L2 1e-3; `train_lora` with the SSL next-step objective only, `LoraReport{delta_norm, latent_shift}`, `merged`), guarded by `AdapterGuard.max_delta_norm` (0.1) / `max_latent_shift` (0.25).
- `NeuralEncoder` (`from_manifest[_on]`, `with_lora`, `encode_windows`, `prior_score`, `sample_futures`) implements `EncoderFamily`; `encode_with_extras` adds per-expert next-step predictions, FSQ codes, prior novelty and the realised targets for live health. `weights.bin` is safetensors (`format = fa-neural-v1`, `family`, `config`, `norm`, `active`). `device_from_env()` reads `FA_ML_DEVICE`.

## Media towers (`media.rs`, D-010)
`MediaTower`, the pretrained registry (`PretrainedTowerSpec`, `pretrained_catalog`: Garage key under `fa-models/pretrained/<name>/`, SHA-256, licence), `select_tower` (pretrained when present and verified, else `HashedEmbeddingTower`) and the publish-time guard `point_in_time` / `embed_window`.

## Owns
`EncoderFamily` trait and the family implementations (statistical and neural), snapshot manifest / weight formats, media-tower contract.

## Governing specs
- [31-expert-encoders](../../specs/31-expert-encoders.md)
