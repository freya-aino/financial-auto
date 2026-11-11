# ADR 0003 — Pure-Rust ML (burn for training, candle for pretrained weights)

**Status:** accepted · **Date:** 2026-11-06

## Context
The user chose pure Rust for all ML, including training of encoders (four competing VAE-family variants per timescale), decoders (evolved ES + SGD), online adapters, and media models (ASR, vision, text).

## Decision
- **burn** for model definition and training (wgpu / CUDA backends), autodiff, checkpointing.
- **candle** allowed for loading and running pretrained weights (text encoders, vision encoders, Whisper-class ASR) where burn lacks importers.
- Weights and snapshots stored as safetensors (or burn records) in Garage (ADR 0009).
- No Python in runtime or training pipelines.

## Consequences
- Single language for the whole model stack; inference and training share code.
- Thinner ecosystem than PyTorch: some architectures (latent diffusion, video models, flow matching) must be implemented by us; research iteration slower.
- Two frameworks (burn + candle) need a tensor bridge or a clear split (candle only for frozen feature extractors).
- **Feasibility risk:** training latent diffusion / video encoders on one GPU in burn is unproven → spike required (core blocker 4 in specs/README).

## Alternatives considered
- **Python (PyTorch) train, Rust serve (ONNX/ort):** fastest research loop; rejected by user.
- **tch-rs (libtorch bindings):** strong coverage but drags in libtorch (C++), not "pure Rust".
- **candle only:** good inference, weaker training ergonomics than burn.

## Implementation note (2026-11-11, D-068, D-070)
The implementation departs from the burn/candle split above: **candle is used for both training and inference**, burn is not used. The workspace pins `candle-core` / `candle-nn` 0.10.x and `safetensors` 0.7; `fa-encoders` and `fa-decoders` depend on candle unconditionally (CPU, opt-in `cuda` feature; the D-004 `burn` / `candle` features are now no-op markers). Weights are safetensors (`fa-neural-v1` encoder snapshots, decoder bundles with embedded metadata) in Garage. The two-framework consequence above therefore no longer applies; the feasibility risk remains open, now for candle, because the development host has no GPU (spec 31 §6.6, [D-070](../DECISIONS.md)). The decision to use pure-Rust ML stands.
