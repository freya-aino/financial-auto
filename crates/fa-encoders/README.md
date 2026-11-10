# fa-encoders

Expert encoder families (burn) and pretrained modality towers (candle).

**Status:** baseline — `EncoderFamily` trait, descriptors for the 4 neural families (no models; features `burn`/`candle` off, D-004) plus the implemented **`StatisticalEncoder`** (`baseline.rs`): a deterministic encoder used by `model-runtime` until neural snapshots exist, either parameter-free (gen-0 baseline) or with fitted normalisation statistics loaded from a snapshot artifact (`snapshot.rs`, spec 36, D-058).

## Statistical baseline
- Input: `EncoderInput::from_bars(as_of_ms, &[MarketBar])` (rows `[open, high, low, close, volume, trade_count]`), the last `context_steps` bars are used.
- Output: `ExpertView` with 3 experts (`EXPERT_NAMES`: returns, volatility, activity), each `EXPERT_DIM = 8` features bounded to (−1, 1) via `tanh(z/2)` of in-window z-scores and correlations; `log_var = −ln(n_bars)` per dim (short history = wide posterior). `router_weights` are left to the router stage.
- Snapshot id `stat-v1-<timescale>`; bump `StatisticalEncoder::VERSION` when feature definitions change.

## Snapshots (`snapshot.rs`, spec 36 §5)
- `SnapshotManifest` (snapshot_id, encoder_family, timescale, generation, parent_snapshot_id, created_at, params, probe-set reference, data window, metrics, feedback, seed) → `manifest.json`; `NormStats` (per-(expert, dim) mean/std over the training windows, `to_bytes`/`from_bytes`, magic `FASTAT01`) → `weights.bin`; constants `MANIFEST_FILE`, `WEIGHTS_FILE`, `REGIONS_FILE`, `REGION_MAP_FILE`, `STATISTICAL_FAMILY`.
- `NormStats.gain` (per expert, default 1) multiplies the z-score before the squash; it is how the trainer's feedback search sets the squash scale and per-expert feature weights (`gain = w_e / s`). `to_bytes` **folds** gains into σ (`σ / g`) and always writes `FASTAT01`, so binaries built before gains existed load the same encoder; `from_bytes` also reads `FASTAT02` (gains appended). `NormStats::pool(parts, equal_weight)` merges per-instrument statistics (by window count = one Welford pass over all windows; or equal weight per instrument for `per_instrument_norm`).
- `DataRef.instruments[]` (`InstrumentDataRef{instrument, source: bars|trades, n_bars, start_ms, end_ms}`) records the series of a multi-instrument fit; `AdapterGuard{max_drift, window_hours, checkpoint_interval_secs}` lives in `params.adapter_guard` (`SnapshotManifest::adapter_guard()`, default 1.0 / 1 h / 900 s); `SnapshotManifest::metric(name)`.
- `StatisticalEncoder::fit(ts, bars, &FitConfig)` fits the statistics (one window per `stride` bars, resumable in chunks with `fit_window_ends` + `fit_into` on a serialisable `NormAccumulator`); `StatisticalEncoder::from_manifest(&manifest, &weights)` rebuilds the encoder of a snapshot, which z-scores every raw feature against the fitted distribution before the squash. `StatisticalEncoder::new` (no stats) stays the gen-0 baseline.

## Owns
`EncoderFamily` trait and the family implementations.

## Governing specs
- [31-expert-encoders](../../specs/31-expert-encoders.md)
