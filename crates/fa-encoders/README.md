# fa-encoders

Expert encoder families (burn) and pretrained modality towers (candle).

**Status:** baseline — `EncoderFamily` trait, descriptors for the 4 neural families (no models; features `burn`/`candle` off, D-004) plus the implemented **`StatisticalEncoder`** (`baseline.rs`): a deterministic, parameter-free encoder used by `model-runtime` until neural snapshots exist.

## Statistical baseline
- Input: `EncoderInput::from_bars(as_of_ms, &[MarketBar])` (rows `[open, high, low, close, volume, trade_count]`), the last `context_steps` bars are used.
- Output: `ExpertView` with 3 experts (`EXPERT_NAMES`: returns, volatility, activity), each `EXPERT_DIM = 8` features bounded to (−1, 1) via `tanh(z/2)` of in-window z-scores and correlations; `log_var = −ln(n_bars)` per dim (short history = wide posterior). `router_weights` are left to the router stage.
- Snapshot id `stat-v1-<timescale>`; bump `StatisticalEncoder::VERSION` when feature definitions change.

## Owns
`EncoderFamily` trait and the family implementations.

## Governing specs
- [31-expert-encoders](../../specs/31-expert-encoders.md)
