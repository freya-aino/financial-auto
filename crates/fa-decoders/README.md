# fa-decoders

Decoder families: visual, retrieval, text, RNN-prior.

**Status:** partial — one object-safe **`Decoder` trait** implemented by every genome family (`decoder`, D-074; conformance test over all eight); CRPS (ensemble + quantile), pinball loss, time-discounted analog weighting; **baseline genome encoding** (`genome`: parse, canonical `Display`, numeric `params()` / `with_params()` for evolution) and two **CPU point-in-time series decoders** (`baseline`: empirical quantile forecaster, analog retrieval with leakage guard and optional **latent conditioning** through `DecodeCtx`), all tested; the **baseline text and visual decoders** served live by model-runtime (`text_baseline`, `visual_baseline`, D-062); and eight **trainable neural families on candle** (`neural`: `nn_forecast` MLP/GRU quantile prior, `nn_retrieval` learned metric, `nn_chart` / `nn_future` raster decoders over `render`, `nn_news` / `nn_narrative` news-embedding decoders, `nn_summary` templated event summaries (`summary`), `nn_frame` broadcast-frame embeddings), with safetensors bundles (versioned latent normaliser, expert layout, transfer records) and the D-009 frozen-elite feedback API (`neural::feedback`), so all eight output families of the population exist. CPU by default; feature `cuda` moves training/serving onto CUDA device 0 when present.

## Unified decoder interface (`decoder`, spec 34 §4.1, D-074)
- `Decoder` (`Send + Sync`): `id`, `family` (output family), `genome`, `expert_mask` (`0` = all), `forward(&DecodeView)` → `DecodeOutput` (quantiles + levels / score + components), `target(series, t, text)` → `(DecodeTarget::{Return, Recon}, known_at_ms)`, `loss(out, target)` (CRPS of the quantiles, raster CE, `1 − cos`; NaN on a mismatch), `cost(series_len)` (ops per forward).
- `DecodeView::new(series, t, text)` (+ `with_experts(&ExpertView)`, spec 32): a neural decoder reads the experts of its mask from the view (`select_latent`; `DecodeError::MissingExperts` when they are absent); `decoder::Baseline` wraps a `SeriesDecoder` (`build_decoder(genome, ctx)`), which conditions on its own point-in-time `DecodeCtx` masked at construction (the retriever needs past candidates' latents too).
- `DecodeCtx::masked(mask)` / `masked_latent` / `split_latent`: per-instrument expert layouts (`(expert_index, len)`, from `LatentState.experts`), mask bit `i` = expert `i` (`EXPERT_MASK_BITS` = 16).
- `decoder::tests::every_family_conforms`: all ten genome families (eight output families) through forward / target / loss / cost, the bundle round trip and the mask, on synthetic data.

## Neural families (`neural`, spec 34 §4.5–4.6)
- `neural::train(&genome, &series, Arc<DecodeCtx>, Arc<TextTimeline>, &TrainOptions)` → `NeuralDecoder`: point-in-time training before the cutoff (`train` gene), AdamW, seeded init, early stopping on a held-out tail, optional warm start from a parent bundle.
- `NeuralDecoder` implements `Decoder` and `SeriesDecoder` (forecasts only after the cutoff, batched + cached per series), `recon_eval` (reconstruction skill vs the training-mean target), `serve` (last step with an explicit latent / news context: quantiles or a [-1, 1] score + components), `select_latent` (the experts of its mask), `to_bundle` / `from_bundle` / `meta_of`.
- Bundles: safetensors (`net.*` + `__meta__` = `BundleMeta` JSON, `format = fa-decoder-bundle/v1`: canonical genome, families, input layout, latent normaliser, `norm_version`, `latent_layout` of the full pre-mask latent, training-mean target, cutoff, train report, objectives, score). `NORM_VERSION = 2` (latents standardised with std ≥ 1e-3 and clipped ±8 everywhere); bundles without it are version 1: `from_bundle` refuses them (`StaleNormaliser`), `from_bundle_any` still reads them (warm starts).
- `neural::data`: input layout `[return window | extra | latent | flag]`, targets, `TextTimeline` (hashed news embeddings by knowledge time; `push_tagged` + `embedding_for(Some(instrument), …)` for instrument-scoped `ninst=1` genomes), `sentiment_direction`.
- `neural::feedback`: `BUCKET`/`PREFIX`/`CARD_SUFFIX` (`fa-models/decoders/<gen>/<family>-<hash>-<job>.safetensors` + `….card.json`, `bundle_key` / `card_key`, `EliteCard`), `select_top_k` (Pareto rank, then score, per output family; only `score > 0`, current normaliser and, given a latent dim, full-latent cards of that width without `src`), `stale_cards` (old-normaliser re-train list), `quota_weights`, `FeedbackSet::{load, prepare, aux_loss}` — the auxiliary loss for encoder training (masked elites read only their experts' columns), differentiable w.r.t. the latents only.
- `neural::net`: the network bodies (MLP, or a GRU whose initial state is `tanh(W·[extra | latent | flag])`) and family heads / losses: monotone quantiles + `pinball_loss`, L2-normalised embeddings + `soft_knn_crps` / `cosine_loss`, per-column `raster_logp` + `raster_ce`; `seeded_init`.
- `render`: column-distribution chart rasters (lookback window, future window), `column_quantiles`, `prob_up`, `direction`.

## Genome format (`genome::DecoderGenome::parse`)
| Form | Example | Notes |
|---|---|---|
| key/value | `forecast:h=5;lb=32;kernel=16;levels=0.1/0.5/0.9` · `retrieval:k=20;tau=30;gap=5;metric=cos` | genes separated by `;` (or space). Commas work only when the string reaches the worker intact; the `ctl.evo.start` transport splits genome lists on `,`. |
| JSON | `{"id":"g1","family":"retrieval","params":{"k":20,"levels":[0.2,0.8],"ts":"1m"}}` | only via `Job.spec` (contains commas) |
| bare id | `g1`, a ULID | deterministic genome derived from an FNV hash of the id: family by parity, genes jittered around defaults. A bare string containing `=` is rejected as a split fragment. |

Families: baselines `forecast` (aliases `quantile`, `rnn_prior`; published on `decoder.output.rnn_prior`) and `retrieval` (`analog`); neural `nn_forecast` (`nn_quantile`, `nn_rnn`), `nn_retrieval` (`nn_metric`), `nn_chart`, `nn_future`, `nn_news`, `nn_narrative`, `nn_summary`, `nn_frame` (table below). Bare ids derive baseline genomes only. Common genes `ts` (timescale, default `1s`), `h` horizon, `lb` lookback, `levels`, `inst`. Forecast: `kernel` (EW half-life, steps), `scale`, `shrink`. Retrieval: `k`, `tau` (days), `gap` (≥ `h`, leakage guard), `metric` (`cos`|`euclid`), `lat` (latent-similarity weight in [0, 1], default 0.5). Neural (`NEURAL_GENES`): `width` 4–256, `depth` 1–4, `gru` 0/1, `steps`, `lr`, `wd`, `bs`, `train` (training share of the window, 0.2–0.9), `emb`, `temp`, `pool`, `res` 4–32, `band`, `tdim` 16–1024, `tw` (minutes), `ninst`; `seed` is a gene but not mutated. Structural (D-074): `mask` (expert bitmask, decimal or `0x…`, ≤ 16 bits, `0` = all), `src` (cross-asset source instrument: latents from `src`, series from `inst`), `ninst=1` (text families read only news matched to the instrument).

### Canonical form and evolution hooks
- `DecoderGenome::to_string()` (`Display`) prints the canonical kv form with **every** numeric gene (the neural genes only for neural families; `ninst`, `mask`, `inst`, `src` only when set, so older strings keep their form) — `forecast:ts=1s;h=3;lb=16;levels=0.05/0.25/0.5/0.75/0.95;kernel=16;scale=1;shrink=0.5;k=20;tau=30;gap=5;metric=cos;lat=0.5` — using shortest round-trip floats; `parse(g.to_string())` yields an equal genome (tested, including derived ones). Children produced by `evo.vary.decoder` travel in this form.
- `params()` → `BTreeMap<name, f64>` of the numeric genes (`NUMERIC_GENES`: `h lb kernel scale shrink k tau gap lat`, plus `NEURAL_GENES` for neural families; the mask is bred separately by evo-worker); `with_params(&map)` applies them (rounding integers, clamping into the valid ranges, `gap ≥ h`) and sets `id` to the canonical string, so mutated children are always evaluable.

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

## Decoder population (lasting content of the retired spec 34)
Spec 34 (decoder population) was retired on 2026-11-11 once every acceptance criterion held and its open questions were decided; its last version is `specs/34-decoder-population.md` in git history ("spec 34 §x" in code comments refers to it). The orchestration half lives in the [evo-orchestrator README](../../go/services/evo-orchestrator/README.md) (loop, ranking, budget), the job protocols in the [evo-worker README](../../bins/evo-worker/README.md), serving in the [model-runtime README](../../bins/model-runtime/README.md). What remains authoritative here:

**Purpose.** A large, evolving population of decoders tests whether the expert spaces are reconstructable: decoders reconstruct charts, frames and text, retrieve analogs and unroll forecasts (also for *other* assets) from the latents, are evolved under a multi-objective Pareto fitness (hybrid ES + SGD: evolution picks topology, hyper-parameters and the expert mask, SGD fits the weights, warm-started from parents), and the elite decoders' losses feed back into encoder training (D-009). Population size is budget-driven with successive halving; no walk-forward inside evolution (overfitting control = deflated Sharpe / PBO + locked holdout + paper shadow, spec 41).

**Families** (genome family → output family on `decoder.output.<family>`):

| Genome family | Output family | Input | Target / loss |
|---|---|---|---|
| `forecast` (baseline) · `nn_forecast` | `rnn_prior` | return window (+ latent → GRU initial state); `src` = latents of another asset (cross-asset prior, pairs from the MCMC cross-asset priors) | `z = r_{t→t+h}/(σ_t√h)`; monotone quantiles, pinball; CRPS + trade value |
| `retrieval` (baseline) · `nn_retrieval` | `retrieval` | return window + latent → embedding | soft-kNN CRPS over analogs whose outcome ended ≥ `gap` before the query, weights `softmax(cos/temp − Δt/τ)` (time distance is an explicit feature) |
| `nn_chart` | `visual.chart` | latent only (return window when unconditioned) | raster of the lookback window (`res`×`res`), per-column CE |
| `nn_future` | `visual.future` | return window + latent | raster of the next `h` steps (±`band·σ√h`), per-column CE; the last column is a predictive distribution |
| `nn_frame` | `visual.frame` | return window + latent | folded embedding of the media frames of `(t − tw, t]` (frame timeline: `media.>` image/video chunks → Garage blob → `data::frame_features`, hashed byte 4-grams; no pretrained vision tower, D-010), `1 − cos` |
| `nn_news` | `text.news` | return window + latent | hashed news embedding of `(t − tw, t]`, `1 − cos` |
| `nn_summary` | `text.summary` | latent only (return window when unconditioned) | templated event summary of the lookback window (below), mean per-slot CE |
| `nn_narrative` | `text.narrative` | return window + current news embedding + latent | news embedding of `(t, t + tw]`, `1 − cos` |

Visual decoders stay in low-resolution raster / embedding space (no pixel generation); text decoders reconstruct hashed bag-of-words embeddings or summary slots, not tokens (LLM text generation, audio reconstruction and multi-GPU population training are deferred). The chart "perceptual loss on a frozen vision encoder" of the original design is replaced by the raster CE: no pretrained vision tower is fetched (D-010; out of scope like spec 31's pretrained towers). Retrieval searches the series' own history in memory, not Helix vectors.

**Event-summary targets (decision on spec 34 open question 2, 2026-11-11, provisional).** There are no human-written summaries, so the ground truth is a **templated summary generated from the window itself** (`summary`): five categorical slots — move (fell sharply · fell · was flat · rose · rose sharply, by `z = Σr/(σ√lb)` at ±0.5/±1.5), volatility (calmed · held steady · turned volatile, second-half / first-half σ at 0.7/1.4), close (near the low · mid-range · near the high, thirds of the window range), path (choppy · two-way · one-way, efficiency `|Σr|/Σ|r|` at 0.2/0.5) and news (no news · negative · neutral · positive, lexicon tone of the window's news folded at 4096 dims, dead band ±0.01). The decoder predicts one distribution per slot; `summary_text` turns the argmax phrases into a sentence ("rose sharply and turned volatile, closing near the high on one-way trade; positive news"), the served score is the expected event intensity `Σ p·|move|` in [0, 1] (the scale of the baseline's vocabulary novelty on the same subject; components: `1 − loss`, the signed expected move, the slot classes). A run without a news source (empty timeline, e.g. the encoder trainer) has no target and sits out. Revisit when transcripts / curated summaries exist.

**Retrieval history (decision on open question 4, 2026-11-11, provisional): no stored re-embedding.** Retrieval decoders embed candidate windows on demand from the latents of the snapshot being read — inside the job's data window during evaluation, the last `pool` windows of the live series while serving, the in-batch rows in feedback — so a new snapshot costs nothing up front and the cost per query is bounded by `pool` (2048). Historical windows of an older snapshot are never mixed in. A Helix vector index over re-embedded history is only worth it once retrieval should reach beyond the in-memory pools (days of history at 1 s); revisit then, re-embedding a sampled subset per snapshot.

**Objectives (spec 34 §6.3 as implemented).** `crps` (min), `crps_skill` = `1 − crps/crps_naive` vs a zero-return forecast (max), `coverage_err` (min), `hit_rate` (max), `trade_sharpe` / `trade_psr` of the naive sign policy net of the `costs.yaml` venue cost model (max), `compute_cost` = log10 ops per forecast (min), `recon_skill` = `1 − loss/loss(training-mean target)` (max, reconstruction families). Families without an objective get the naive value (`crps_skill` 0, `hit_rate` 0.5, `trade_sharpe` 0, `trade_psr` 0.5, `recon_skill` 0); undefined `crps` / `coverage_err` are missing (= worst). The family score is `crps_skill` (forecasting families) or `recon_skill`.

**Cross-family normalisation (decision on open question 6, 2026-11-11, provisional).** (1) Skills are measured against each family's own naive baseline, so they are dimensionless and comparable; (2) selection fronts are computed **per niche** (decoder output family; baseline and neural genomes of one output family compete) and interleaved by relative position, so a family never crowds out another during halving; (3) inside a niche, objectives are min–max normalised over the ranked set before the hypervolume (reference 1.1, only the objectives every front member defines) and before NSGA-III's ideal/nadir scaling; (4) the cross-family elite choice for feedback is per family with equal quotas (`quota_weights`), never one cross-family front. Raw `crps` stays ranked only within a niche, where units agree.

**Training invariants.** Point-in-time: a row trains only when its target is known by the cutoff (`outcome_end ≤ cutoff`); decoders forecast only after it. Validation is the last 10 % of the training rows, **purged** of training rows whose target window reaches into the first validation row's (López de Prado; `TrainReport.purged_rows`). Reconstruction families (chart, news, narrative, summary, frame) **start at their baseline**: output weights 0, output bias = the training-mean target in output space, so SGD learns a residual, weight decay shrinks towards the baseline and early stopping keeps the baseline when nothing beats it (D-085). Elites need `score > MIN_ELITE_SCORE` (1e-3), so a decoder that kept its baseline (score ±1e-7) never qualifies.

**Live status (2026-11-11, probes on VictoriaMetrics / `LATENT` / `TEXT`, read-only).** Before D-085 the live text and chart decoders scored below their training-mean baseline (`recon_skill` 1 s / 2 h: chart −0.003, news −0.022, narrative −0.115; 1 m / 24 h: news −0.036, narrative −0.277, chart +0.007): they started from a random output, and the unpurged validation tail (overlapping 10–60 min news windows) let early stopping keep weights that only memorised the training period. After: chart 1 s 0.000, news 0.000 / 0.000, narrative 0.000 / 0.000, chart 1 m **+0.008**; `nn_summary` **+0.041** (1 s / 2 h) and **+0.050** (1 m / 24 h) from the latent alone (+0.58 unconditioned, where the slots are computable from the input). The text families now sit exactly at naive: market-wide news windows are dominated by a common vocabulary (training-mean `1 − cos` 0.036 at 1 h) and the 24-wide statistical latent carries no news information — more history (24 h, 3 383 items) did not change that; it needs a text expert in the encoder (spec 31 media front-ends), not a decoder change. `visual.frame` cannot train live: the `MEDIA` stream is empty (no capture producer). Elite transfer, re-evaluated frozen on the last 2 h (15 h after training): 6 of 6 elites within 10 % (ratios 0.96–1.01).

**Data model.**

| Entity | Store | Fields / writer |
|---|---|---|
| `Genome{genome_id, kind = decoder, family, params}` | Helix | `evo.vary.decoder` (parents, children), `evo.eval.decoder` (evaluated genome), `evo.front.decoder` (front members) |
| `ParentOf{op}` / `Scored` → `FitnessRecord{generation, objectives, pareto_rank}` | Helix | `evo.vary.decoder` |
| `TrainedOn` Genome → Snapshot | Helix | `evo.vary.decoder` (parents → the job's snapshot) |
| `EvaluatedOn{generation, gen_index, rung, score, objectives, job_id}` Genome → Snapshot | Helix | every `evo.eval.decoder` with a `snapshot_id` (upsert per pair) |
| `Reads{snapshot_id, expert_id}` Genome → `Expert` | Helix | same, one per expert the mask selects (mask 0: every expert of the snapshot) |
| `ParetoFront{front_id = <generation>/<loop>, generation_id, gen_index, kind, snapshot_id, keys, members, front_size, hypervolume, final}` + `InFront{rank, hv, crowding, score}` → Genome | Helix | `evo.front.decoder` at every loop end (`fa_hub::evolution`) |
| `FeedbackFrom{weight}` Snapshot → Genome | Helix | trainer (D-009) |
| Bundle + card | Garage `fa-models/decoders/<generation>/<family>-<hash>-<job>.safetensors` + `….card.json` | every trained neural decoder; cards carry `meta.transfer[]` |
| Inspection samples | Garage `fa-artifacts/decoder-out/<decoder_id>/<job>.json` (`decoder-out/v1`: up to 6 out-of-sample outputs — quantiles + realised, raster / embedding + target, summary sentence + target sentence) | `evo.eval.decoder` (neural) |
| Eval report | Garage `fa-artifacts/evals/<generation>/<job>.json` | `evo.eval.decoder` |
| Pareto front per generation | `generation.summary` on `jobs.result.evo.generation` (+ `pareto.*` gauges), the `ParetoFront` node, control-api `GET /api/evolution/populations/{id}/pareto` | orchestrator |

**Elite re-evaluation and transfer (spec 34 §8).** `evo.reeval.decoder` (evo-worker) re-scores the elite cards (`feedback::reeval_candidates`, `k` per family, as if never re-evaluated) frozen on a later window after each card's cutoff, with the latents (and news / frames) of the snapshot serving then; each card gains `meta.transfer[] = {snapshot_id, score, base_score, ratio = score/base_score, n_eval, window, at_ms, job_id}`. `BundleMeta::transfer_factor` = latest ratio clipped to [0, 1]: `quota_weights` multiplies feedback weights by it (renormalised), `select_top_k` drops elites whose factor is 0 (no longer beat naive). `feedback::transfer_check(cards, k, snapshot, tol)` = spec 36 §6.4 item 5 (`TRANSFER_TOL` 0.1: re-evaluated elites lose ≤ 10 %). The orchestrator sends the job on `ctl.evo.reeval` and two hours after a snapshot swap.

**Tunables.**

| Name | Default | Where |
|---|---|---|
| successive-halving `eta` / rungs | 3 / from `min_budget_ms`…`max_budget_ms` | `ctl.evo.start` |
| elites per loop | `ceil(N/η)` | orchestrator |
| feedback elites per family | 4 | trainer (`select_top_k`) |
| `family_quota` | equal | `quota_weights` |
| `MIN_ELITE_SCORE` | 1e-3 | `feedback` |
| `TRANSFER_TOL` | 0.10 | `neural` |
| retrieval `k` / `gap` | 20 / ≥ `h` (leakage guard) | genome |
| `mcmc_pairs` (cross-asset `nn_forecast` per loop) | 4 | `FA_EVO_MCMC_PAIRS` |
| re-evaluation delay after a swap | 120 min | `FA_EVO_REEVAL_DELAY_MIN` |
| neural training share of the job budget | 0.6 | evo-worker `TRAIN_BUDGET_SHARE` |
| budget overrun tolerance | 10 % | orchestrator |
| decoder trade notional | 10 000 EUR | evo-worker |
| evolvable expert-mask bits | 8 | evo-worker `MASK_GENES` |
| HV reference / Monte-Carlo samples | 1.1 / 8192 | `fa_evolution::pareto` |

**Failure modes.** Convergence to one family → per-niche ranking, crowding, immigrants (mcmc pairs). Retrieval leakage → `gap` enforced in training masks, serving and feedback (tested with planted leaks). Decoders exploiting snapshot quirks → re-evaluation on the next snapshot + transfer decay. Worker crash → JetStream redelivery (ack window). Budget overrun → rung truncation, `budget.truncated`. Feedback degrading the encoder → `λ_gen` cap and the promotion checklist (trainer README).

**Acceptance (all hold).** 1: all eight output families implement `Decoder` and pass `every_family_conforms` on synthetic data. 2: kv / JSON round trip and clamped mutation are tested (protobuf genomes do not exist: genomes travel as strings, D-044). 3: budget within 10 % (`TestBudgetTruncatesRungs`), fronts stored in Helix (`ParetoFront`) with lineage. 4: retrieval leakage gap tested for baseline and learned retrieval. 5: RNN-prior CRPS + trade value through the spec 35 cost model. 6: elite bundles load in the trainer (`trainer::feedback` integration test). 7: redelivery (D-044).

## Owns
`Decoder` trait and family implementations (incl. the baseline text / visual scorers and the neural families); genome encoding of all families; decoder bundles and the decoder → encoder feedback API; `DecodeCtx` / `LatentPoint` / `instrument_token` / `cosine` conditioning primitives.

## Governing specs
- Spec 34 (retired; content above), [32-expert-spaces](../../specs/32-expert-spaces.md) (expert view), decisions D-009, D-044, D-049, D-062, D-068, D-074, D-085 (proposed)
