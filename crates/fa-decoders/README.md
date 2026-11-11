# fa-decoders

Decoder families: visual, retrieval, text, RNN-prior.

**Status:** partial — `Decoder` trait and family enum; CRPS (ensemble + quantile), pinball loss, time-discounted analog weighting; **baseline genome encoding** (`genome`: parse, canonical `Display`, numeric `params()` / `with_params()` for evolution) and two **CPU point-in-time series decoders** (`baseline`: empirical quantile forecaster, analog retrieval with leakage guard and optional **latent conditioning** through `DecodeCtx`), all tested; the **baseline text and visual decoders** served live by model-runtime (`text_baseline`, `visual_baseline`, D-062); and six **trainable neural families on candle** (`neural`: `nn_forecast` MLP/GRU quantile prior, `nn_retrieval` learned metric, `nn_chart` / `nn_future` raster decoders over `render`, `nn_news` / `nn_narrative` news-embedding decoders), with safetensors bundles and the D-009 frozen-elite feedback API (`neural::feedback`). CPU by default; feature `cuda` moves training/serving onto CUDA device 0 when present.

## Neural families (`neural`, spec 34 §4.5–4.6)
- `neural::train(&genome, &series, Arc<DecodeCtx>, Arc<TextTimeline>, &TrainOptions)` → `NeuralDecoder`: point-in-time training before the cutoff (`train` gene), AdamW, seeded init, early stopping on a held-out tail, optional warm start from a parent bundle.
- `NeuralDecoder` implements `SeriesDecoder` (forecasts only after the cutoff, batched + cached per series), `recon_eval` (reconstruction skill vs the training-mean target), `serve` (last step with an explicit latent / news context: quantiles or a [-1, 1] score + components), `to_bundle` / `from_bundle` / `meta_of`.
- `neural::data`: input layout `[return window | extra | latent | flag]`, targets, `TextTimeline` (hashed news embeddings by knowledge time), `sentiment_direction`.
- `neural::feedback`: `select_top_k`, `quota_weights`, `FeedbackSet::{load, prepare, aux_loss}` — the auxiliary loss for encoder training, differentiable w.r.t. the latents only.
- `render`: column-distribution chart rasters (lookback window, future window), `column_quantiles`, `prob_up`, `direction`.

## Genome format (`genome::DecoderGenome::parse`)
| Form | Example | Notes |
|---|---|---|
| key/value | `forecast:h=5;lb=32;kernel=16;levels=0.1/0.5/0.9` · `retrieval:k=20;tau=30;gap=5;metric=cos` | genes separated by `;` (or space). Commas work only when the string reaches the worker intact; the `ctl.evo.start` transport splits genome lists on `,`. |
| JSON | `{"id":"g1","family":"retrieval","params":{"k":20,"levels":[0.2,0.8],"ts":"1m"}}` | only via `Job.spec` (contains commas) |
| bare id | `g1`, a ULID | deterministic genome derived from an FNV hash of the id: family by parity, genes jittered around defaults. A bare string containing `=` is rejected as a split fragment. |

Families: `forecast` (aliases `quantile`, `rnn_prior`; published on `decoder.output.rnn_prior`) and `retrieval` (`analog`). Common genes `ts` (timescale, default `1s`), `h` horizon, `lb` lookback, `levels`, `inst`. Forecast: `kernel` (EW half-life, steps), `scale`, `shrink`. Retrieval: `k`, `tau` (days), `gap` (≥ `h`, leakage guard), `metric` (`cos`|`euclid`), `lat` (latent-similarity weight in [0, 1], default 0.5).

### Canonical form and evolution hooks
- `DecoderGenome::to_string()` (`Display`) prints the canonical kv form with **every** gene — `forecast:ts=1s;h=3;lb=16;levels=0.05/0.25/0.5/0.75/0.95;kernel=16;scale=1;shrink=0.5;k=20;tau=30;gap=5;metric=cos;lat=0.5` — using shortest round-trip floats; `parse(g.to_string())` yields an equal genome (tested, including derived ones). Children produced by `evo.vary.decoder` travel in this form.
- `params()` → `BTreeMap<name, f64>` of the numeric genes (`NUMERIC_GENES`: `h lb kernel scale shrink k tau gap lat`); `with_params(&map)` applies them (rounding integers, clamping into the valid ranges, `gap ≥ h`) and sets `id` to the canonical string, so mutated children are always evaluable.

## Baseline decoders (`baseline`)
- `Series::resample` builds a log-price series from `(ts_ms, price)` trades (last price per timescale bucket).
- `QuantileForecaster`: EW empirical distribution of past overlapping `h`-step returns, shrunk drift, `scale` spread.
- `AnalogRetriever`: z-scored return-shape query, candidates whose outcome window ends ≥ `gap` before now, top-`k` by `sim·exp(−Δt/τ)`, weighted outcome quantiles + `Analog` list.
- `SeriesDecoder::forecast(series, t)` reads only `series[..=t]` (tested).
- `build(genome)` / `build_with(genome, Arc<DecodeCtx>)`: the context holds `LatentPoint`s (`as_of_ms`, concatenated expert means of a `LatentState`) per instrument token. When latents exist for the series' instrument, the retriever's similarity becomes `(1 − lat)·shape + lat·cos(latent as of query, latent as of candidate)`; both lookups are **as of** the respective window end (`latent_as_of`), so a later latent never leaks in (tested: truncating the future leaves the forecast unchanged; other instruments' latents are ignored). An empty context reproduces the unconditioned decoder exactly.

## Baseline text decoder (`text_baseline`, decoder id `text-stat-v1`)
Pure and deterministic; model-runtime feeds it `news.item.>` and publishes `text.news` / `text.summary` (contract in [model-runtime](../../bins/model-runtime/README.md)).
- `tokenize`: lower-case alphanumeric tokens (`n't` → `not`), length ≥ 2.
- `score_text`: embedded finance lexicon (`LEXICON_POS`, `LEXICON_NEG`, ~100 words each); a negator (`NEGATORS`) flips the polarity of the next `NEGATION_SCOPE = 3` tokens; polarity `(pos − neg)/(pos + neg)`, confidence `1 − exp(−hits/2)`.
- `TextWindow::score_at(now, W)`: `text.news = Σ d·w·s / (Σ d·w + SHRINK)` over items in `[now − W, now]` with recency decay `d = exp(−age/(W/2))`, confidence `w`, polarity `s`, `SHRINK = 0.5` (volume normalisation: one weak item cannot saturate the score); `text.summary` novelty `= 1 − cos(tf_current, tf_trailing)` of FNV-1a hashed term frequencies (`HASH_DIM = 1024`) of the current window vs `[now − 2W, now − W)` (0 when either is empty). Items after `now` are invisible (point-in-time).
- `InstrumentIndex::match_doc`: case-sensitive ticker tokens (`SPY`, `$SPY`, pair base `BTC` of `BTC-USD`), name phrases (lower-cased, legal suffixes and `(…)` dropped, word-bounded; ticker-only names are matched as symbols only) and `entities_raw` values (`query:Apple stock`).

## Baseline visual decoder (`visual_baseline`, decoder id `visual-stat-v1`)
`chart_scores(bars)` over ≥ `MIN_BARS = 8` OHLCV bars (model-runtime uses the last 60 closed bars):
- `breakout = tanh(4·(close − R)/W)` above the resistance `R`, `tanh(4·(close − S)/W)` below the support `S` (band of the bars before the last one, `W = R − S`), else 0;
- `trend = tanh(t/3)` with `t` the OLS t-statistic of `ln close` on the bar index;
- `compression = 1 − min(1, (range_k/range_n)·√(n/k))`, `k = max(3, n/5)` (recent range squeezed inside the window, corrected for the random-walk √span growth);
- `future = clamp(0.5·breakout + 0.35·trend + 0.15·sign(trend)·compression, −1, 1)`.
Tested on synthetic shapes: flat → all 0, box breakouts up/down mirrored, ramps up/down, a squeeze.

## Owns
`Decoder` trait and family implementations (incl. the baseline text / visual scorers and the neural families); genome encoding of all families; decoder bundles and the decoder → encoder feedback API; `DecodeCtx` / `LatentPoint` / `instrument_token` / `cosine` conditioning primitives.

## Governing specs
- [34-decoder-population](../../specs/34-decoder-population.md)
