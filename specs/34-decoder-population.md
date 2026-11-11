# 34 — Decoder Population

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium-high |
| **Owners (code)** | `crates/fa-decoders`, `crates/fa-evolution`, `bins/evo-worker`, `go/services/evo-orchestrator` |
| **Depends on** | 32 Expert spaces, 33 MCMC graph, 30 Latent regions, 36 Training lifecycle, 41 Backtest & validation, 60 Ops |

## 1. Purpose
Expert spaces are only valuable if many kinds of models can reconstruct useful things from them. A large, evolving **population of decoders** tests this. Decoders reconstruct images, retrieve past analog events, produce text, and unroll forecasts for other assets. They are evolved under a multi-objective Pareto fitness. The **elite decoders' losses feed back into encoder training**, so expert spaces become reconstructable by design.

## 2. Scope
**In scope:**
- Decoder families and their targets.
- Genome encoding.
- Hybrid ES + SGD evolution loop.
- Pareto objectives and selection.
- Budget-driven population sizing with successive halving.
- The feedback interface to encoders.
- Lineage storage.

**Out of scope:**
- Strategy evolution (spec 35; it shares the evolution engine in `fa-evolution`).
- Encoder families (spec 31).
- Job transport details (spec 21, 60).

## 3. Decisions
- **Families:**
  - **Visual:** chart renderings, broadcast-frame reconstruction, future chart (visual forecast).
  - **Retrieval (analog search):** vector search in Helix for historical windows most similar/covarying with the current state, scored on **what followed**. **Time distance is an explicit feature.**
  - **Text:** news reconstruction, event summary, forecast narrative.
  - **RNN-prior:** latent → RNN prior for a **different asset**, unrolled over the **timescale's horizon** (24h is the hour/day instance). Source→target pairs **proposed by MCMC cross-asset priors, selected/mutated by evolution**. Scored by **CRPS/NLL + tradeable value**.
- Evolution is **hybrid ES + SGD**: topology, hyperparameters and expert-subspace selection are evolved; weights are warm-started from parents and briefly fine-tuned by SGD.
- Fitness is **multi-objective Pareto**: reconstruction quality, forecast skill, downstream trading value, compute cost (NSGA-II/III style).
- **Decoder fitness shapes encoders:** elite decoders' losses periodically flow back into encoder retraining (spec 36).
- Population size is **budget-driven**: GPU-hours per generation; the number of candidates is derived from the per-candidate cost, with successive halving.
- **No walk-forward inside evolution.** Overfitting control is deflated Sharpe / PBO + locked holdout + live paper shadow (spec 41).
- The design is horizontally scalable: Go orchestrator + stateless Rust workers over NATS work queues.

## 4. Interfaces

### 4.1 Decoder trait (proposal)
```rust
pub trait Decoder<B: Backend>: Module<B> {
    fn family(&self) -> DecoderFamily;
    /// Consumes the expert-space contract (spec 32), restricted to the genome's expert mask.
    fn forward(&self, view: &ExpertViewBatch<B>, ctx: &DecodeCtx<B>) -> DecoderOutput<B>;
    /// Family-specific loss vs target (reconstruction / CRPS / retrieval outcome loss).
    fn loss(&self, out: &DecoderOutput<B>, target: &DecodeTarget<B>) -> Tensor<B, 1>;
}

pub enum DecoderFamily {
    VisualChart, VisualFrame, VisualFutureChart,
    Retrieval,
    TextNews, TextEventSummary, TextForecastNarrative,
    RnnPrior,
}
```

### 4.2 Genome (proposal)
```rust
pub struct DecoderGenome {
    pub id: GenomeId,
    pub parents: Vec<GenomeId>,
    pub family: DecoderFamily,
    pub timescale: Timescale,
    pub expert_mask: BitVec,               // which experts it reads (spec 32)
    pub use_mcmc_state: bool,              // reads region/kernel context (spec 33)
    pub topology: TopologyGene,            // layer list: kind, width, depth, attention heads, recurrence
    pub hyper: HyperGene,                  // lr, wd, steps budget, dropout, loss weights
    pub family_genes: FamilyGenes,         // see §6.1
    pub warm_start: Option<WeightsRef>,    // parent weights (Garage)
    pub seed: u64,
}

pub enum FamilyGenes {
    Retrieval { k: u16, metric: Metric, time_decay: f32, min_gap: Duration, weight_by: WeightBy },
    RnnPrior  { source: AssetSetId, target: InstrumentId, cell: RnnCell, unroll_steps: u32, dist: OutDist },
    Visual    { target: VisualTarget, resolution: u16, renderer: RendererCfg },
    Text      { target: TextTarget, vocab: TokenizerRef, max_len: u16, decode: DecodeCfg },
}
```

### 4.3 Jobs (spec 21)
- `jobs.evo.decoder.eval` → `fa.v1.EvalJob { genome, snapshot_id, data_window, budget_steps, rung }`
- `jobs.evo.decoder.result` → `fa.v1.EvalResult { genome_id, objectives[], weights_uri, diagnostics }`

**Implemented today (D-044, `bins/evo-worker`, `crates/fa-decoders`):** the generic `fa.v1.Job` / `fa.v1.JobResult` of spec 21 are used instead of dedicated messages. Job kind is `evo.eval.decoder` on `jobs.evo.eval.decoder` (work queue `JOBS_EVO`), results on `jobs.result.evo.eval.decoder`, progress on `jobs.progress.<job_id>`.

### 4.4 Baseline genome encoding (implemented, D-044)
The typed `DecoderGenome` above is the target. Until the neural families exist, a genome is a **string** parsed by `fa_decoders::genome::DecoderGenome::parse`, carried in `Job.spec` (UTF-8) or, failing that, `Job.genome_id`:

| Form | Example | Notes |
|---|---|---|
| key/value | `forecast:h=5;lb=32;kernel=16;levels=0.1/0.5/0.9` · `retrieval:k=20;tau=30;gap=5;metric=cos` | genes separated by `;` (never `,`: `ctl.evo.start` splits genome lists on commas) |
| JSON | `{"id":"g1","family":"retrieval","params":{"k":20,"levels":[0.2,0.8],"ts":"1m"}}` | only via `Job.spec` |
| bare id | `g1`, a ULID | deterministic genome derived from a hash of the id (family by parity, genes jittered around defaults) |

Families: `forecast` (aliases `quantile`, `rnn_prior`; outputs on `decoder.output.rnn_prior`) and `retrieval` (`analog`; outputs on `decoder.output.retrieval`). Common genes: `ts` (timescale, default `1s`), `h` horizon, `lb` lookback, `levels`, `inst`. Forecast: `kernel`, `scale`, `shrink`. Retrieval: `k`, `tau` (days), `gap` (≥ `h`, leakage guard), `metric` (`cos` | `euclid`).

Result payload (`JobResult.result` JSON, protocol `evo.eval.decoder/v0`): `score` (= `crps_skill`, the orchestrator's provisional scalar, D-025), `objectives` and `directions` (keys in §6.3), `genome`, `data` (source + instruments), `budget {budget_ms, elapsed_ms, truncated}`, `stop`, `report_ref {planned, persisted}`.

### 4.5 Neural genome families (implemented, `fa_decoders::neural`, candle)
The same string genome carries six trainable families (aliases in parentheses); bare ids still derive baseline genomes only.

| Genome family | Output family | Network input | Target / loss |
|---|---|---|---|
| `nn_forecast` (`nn_quantile`, `nn_rnn`) | `rnn_prior` | return window (+ latent) — MLP, or with `gru=1` a GRU over the window whose initial state is `tanh(W·[latent, flag])` | `z = r_{t→t+h}/(σ_t√h)`; monotone quantiles (cumulative softplus), pinball loss |
| `nn_retrieval` (`nn_metric`) | `retrieval` | return window + latent → L2-normalised embedding (`emb`) | soft-kNN CRPS: `w = softmax(cos/temp − Δt/τ)` over candidates whose outcome ended ≥ `gap` before the query; serving takes the top `k` of the last `pool` windows |
| `nn_chart` | `visual.chart` | latent only (the return window when trained without latents) | raster of the lookback window (`res`×`res`, `fa_decoders::render`), per-column cross-entropy |
| `nn_future` | `visual.future` | return window + latent | raster of the next `h` steps (rows ±`band·σ√h`), per-column CE; the last column is a predictive distribution → quantiles / `P(up)` |
| `nn_news` | `text.news` | return window + latent | hashed news embedding (`tdim`, stop-worded, `1 + ln tf`) of `(t − tw, t]`, `1 − cos` |
| `nn_narrative` | `text.narrative` | return window + current news embedding + latent | news embedding of `(t, t + tw]`, `1 − cos` |

Latents are the instrument's `latent.state` as of the step (concatenated expert means, standardised with the training mean / std — std floored at 1e-3, values clipped to ±8 in training, serving and feedback alike — plus a presence flag; `lat=0` trains unconditioned). A latent-conditioned `nn_chart` without enough latents before the cutoff falls back to the unconditioned decoder (return window → chart). Neural genes (all numeric, mutated by `evo.vary.decoder`, clamped by `with_params`): `width` 4–256, `depth` 1–4, `gru` 0/1, `steps` 1–20000, `lr`, `wd`, `bs`, `train` (share of the window by time used for training, 0.2–0.9), `emb`, `temp`, `pool`, `res` 4–32, `band`, `tdim` 16–1024, `tw` (minutes), `seed` (not mutated). Example: `nn_forecast:h=5;lb=32;width=32;depth=2;gru=1;steps=300;lr=0.003`. The canonical kv form prints the neural genes only for neural families, so baseline genome strings are unchanged.

**Training (point-in-time).** `neural::train` uses only rows whose target is known by the **train cutoff** (`outcome_end ≤ cutoff`), AdamW minibatch SGD, deterministic seeded initialisation, early stopping on the last 10 % of the training rows (best validation weights kept). The decoder then forecasts only at steps after the cutoff, so every objective is out of sample. Optional warm start copies parent variables with matching name and shape (§6.2.3).

**Bundles.** One safetensors file per trained decoder: the network variables (`net.*`) plus a `__meta__` u8 tensor with the `BundleMeta` JSON (`format = fa-decoder-bundle/v1`, canonical genome, families, input layout, latent normaliser, training-mean target, cutoff, train report, objectives, score). Garage: `fa-models/decoders/<generation>/<family>-<genome hash>-<job_id>.safetensors` and a JSON **card** `….card.json` (`EliteCard {meta, bucket, key, created_ms}`) next to it; cards are what elites are selected from (listing, no index object, so concurrent workers never race).

### 4.6 Decoder → encoder feedback API (implemented, D-009, `fa_decoders::neural::feedback`)
For the encoder trainer (spec 36); all types are re-exported candle types (`fa_decoders::neural::{Tensor, Device, DType}`).

| Item | Purpose |
|---|---|
| `feedback::{BUCKET, PREFIX, CARD_SUFFIX}` | `fa-models`, `decoders/`, `.card.json`: list the cards with `fa_blob::BlobStore::list(BUCKET, PREFIX)` and parse each as `EliteCard` |
| `select_top_k(&cards, k, Some(latent_dim)) -> Vec<EliteCard>` | top `k` per output family: Pareto rank over the §6.3 objectives, then family score, newest first; only decoders with `score > 0` (better than naive) and, with `Some(d)`, latent-conditioned with exactly that latent dimension |
| `quota_weights(&elites) -> Vec<(EliteCard, w)>` | §6.6 weights: equal family quota, `1/(1+rank)` inside a family, Σ w = 1 |
| `FeedbackSet::load(&[(bundle bytes, w)], latent_dim, &device)` | frozen elites (weights are constants, not `Var`s); bundles of another latent dimension or unconditioned ones are skipped (`skipped`), weights renormalised |
| `set.prepare(&[(&Series, step)], &TextTimeline) -> FeedbackBatch` | per elite: context (return window / news embedding) and target (z, raster, embedding) of each step; steps without context or target are left out for that elite only |
| `set.aux_loss(&latents [n, latent_dim], &batch) -> FeedbackLoss {total, parts}` | `L_gen = Σ_d w_d · L_d(decoder_d([context, (latent − μ)/σ, 1]), target_d)` with the family loss of training (pinball, in-batch soft-kNN with the leakage gap, raster CE, 1 − cos); `total` is a scalar differentiable w.r.t. `latents` only; `parts` = `(decoder id, family, loss, weight)` |

`n` rows of `latents` correspond to the `n` prepared steps; the encoder trainer adds `λ_gen · total` to its loss (spec 32 `L_gen`). A conformance test checks that the gradient reaches the latents and that no elite weight is a variable.

## 5. Data model
| Entity | Store | Fields |
|---|---|---|
| `DecoderGenome` | Helix node | genome json, family, timescale, generation, status (alive/elite/dead), objectives vector, rung reached |
| `CHILD_OF` | Helix edge | operator (mutate/crossover), mutation summary |
| `READS` | Helix edge Genome → Expert | from expert_mask |
| `EVALUATED_ON` | Helix edge Genome → EncoderSnapshot | |
| Weights | Garage `models/decoders/<gen>/<genome_id>.safetensors` | only for candidates beyond rung 1 or elites |
| Outputs for inspection | Garage `decoder-out/<genome_id>/…` | sample images, texts, forecasts for the evolution lab UI |
| Pareto front per generation | Helix node `ParetoFront` + VM metrics | |

## 6. Behaviour & algorithms

### 6.1 Family specifications
**Visual**
- *Chart rendering:* target = deterministic raster of price/volume of the window (rendered by a Rust renderer, `fa-decoders::render`). Loss: perceptual loss on the frozen candle vision encoder's embedding + pixel L1 at low resolution.
- *Broadcast frames:* target = embedding of co-occurring broadcast frames (frozen vision encoder). Loss: cosine / InfoNCE against negatives. Pixel-space generation is deferred.
- *Future chart:* target = rendered chart of the **next horizon**. Scored like chart rendering, plus a forecast term on the underlying series extracted from the chart.

**Retrieval (analog search)**
- Query = expert-masked latent (+ optional MCMC state). Helix vector search over historical latents of the **same snapshot** (re-embedding history is part of snapshot creation, spec 36).
- Scoring of a retrieved neighbour set: the weighted outcome distribution of "what followed" each analog over the horizon is compared to the realised outcome (CRPS).
- **Time distance as feature:** weights `w_i ∝ sim_i · f(Δt_i)` with an evolved decay `f`. `min_gap` excludes analogs inside the leakage window (analog's outcome window overlapping now).
- "Large covariance": the similarity metric may include a covariance term between analog and current multi-asset trajectories (metric gene).
- The learnable part is a small query projector (trained by SGD) plus genes for k, metric and decay.

**Text**
- *News reconstruction:* target = embeddings (or tokens) of headlines that co-occurred with the window. v1: embedding-level contrastive retrieval/reconstruction. Token generation is optional.
- *Event summary:* target = a summary text of the window. Needs targets: derived from co-occurring news/transcripts (proposal) or generated templates from market stats.
- *Forecast narrative:* target = news/transcripts published **after** the window within the horizon (embedding level). This scores whether the latent anticipates the narrative.
- v1 generation uses small burn decoders over a frozen tokenizer. Large-LM text generation is deferred.

**RNN-prior**
- Latent (source asset set) → initial state + per-step prior of an RNN/SSM over the **target** asset's series, unrolled `unroll_steps = horizon / step` (spec 31 timescale ladder).
- Output: predictive distribution per step (mixture density or quantiles).
- Scores: **CRPS / NLL** over the trajectory; **tradeable value** = return of a fixed naive policy trading the forecast's sign/quantiles, net of costs (spec 35 cost model).

**Implemented today (D-044, D-062):** CPU baselines exist for `rnn_prior` (quantile forecaster) and `retrieval` (analog retriever) as evolvable genomes in evo-worker, and for `text.news` / `text.summary` (lexicon sentiment and vocabulary novelty over `news.item.>`) and `visual.chart` / `visual.future` (breakout, trend t-statistic and range compression over bars) as fixed decoders in model-runtime. All publish `DecoderOutput` with `quantiles[0]` as the score and `inst:` / `topic:` / `kind:` / `src:` / `h:` refs.

**Neural families (implemented, §4.5):** trainable candle decoders for `rnn_prior` (MLP / GRU quantile prior), `retrieval` (learned metric), `visual.chart`, `visual.future`, `text.news` and `text.narrative`, evolved through `evo.eval.decoder` like the baselines and served by model-runtime (§6.7). Differences to the text above: visual decoders stay in **embedding/raster space at low resolution** (open question 3 answered "raster, no pixels"); the chart "perceptual loss on the frozen vision encoder" is replaced by the column CE on the raster (no frozen vision encoder exists); text decoders reconstruct **hashed bag-of-words embeddings**, not tokens; retrieval searches the series' own history in memory, not Helix vectors; the RNN prior forecasts the cumulative horizon return of the **same** instrument (no cross-asset source → target pair yet). `visual.frame` (needs broadcast frames, an external source) and `text.summary` (open question 2) have no neural family.

### 6.2 Evolution loop (per generation, per timescale)
1. **Seed:** elites of the previous generation + MCMC-proposed RNN pairs (spec 33 §6.5) + random immigrants (`immigrant_frac`).
2. **Variation:** mutation (topology add/remove/resize, hyperparameter perturbation, expert-mask bit flips, family-gene perturbation) and crossover (within family; layer-block crossover; expert-mask uniform crossover).
3. **Warm start:** children inherit compatible parent weights (shape-matching; new layers initialised near identity / zero-residual).
4. **Successive halving:** rung r trains `steps_r = steps_0 · η^r` and keeps the top `1/η` by a scalarised rung score (hypervolume contribution on the current front) → next rung.
5. **Full evaluation** of survivors on the evaluation window. Objectives are computed (§6.3).
6. **Selection:** NSGA-II (≤ 3 objectives) or NSGA-III (≥ 4 objectives, reference directions). Elitist archive of size `elite_size`.
7. **Feedback export:** the elite set per family is exported to the encoder trainer (§6.6).
8. **Persist:** genomes, lineage, objectives, front → Helix. Weights of elites → Garage.

### 6.3 Objectives (Pareto)
| Objective | Definition | Direction |
|---|---|---|
| Reconstruction | Family loss normalised per family (z-score vs family baseline) | min |
| Forecast skill | CRPS skill vs naive baseline on next horizon (families with a forward target) | max |
| Trading value | Deflated Sharpe of the naive policy using the decoder output, net of costs (spec 41) | max |
| Compute cost | Inference FLOPs + VRAM + latency (normalised) | min |
Families without a meaningful value for an objective get a neutral value. Fronts are computed **per family** for selection, and across families for the feedback-elite choice. Overfitting control: PBO estimated across the candidate set per generation (spec 41).

**Implemented today (`fa_backtest::eval`, D-044):** the baseline series decoders report `crps` (min), `crps_skill` = `1 − crps/crps_naive` vs a zero-return forecast (max), `coverage` / `coverage_err` = `|coverage − nominal|` (min), `hit_rate` (max), `trade_sharpe` and `trade_psr` of the naive sign policy net of a bp cost (max), `compute_cost` = log10 arithmetic ops per forecast (min), `n_eval`. "Reconstruction" has no value for these families. The orchestrator ranks rungs with NSGA-II over `objectives` (Go port kept in lock-step with `fa-evolution` by a shared fixture, D-049); `score` is only a fallback.

### 6.4 Budget-driven sizing
- Input: `gpu_hours_per_generation` (per timescale share from spec 60), measured cost per candidate per rung (EMA by family/size).
- `N_0` (initial candidates) solves `Σ_r N_0 · η^{-r} · cost_r ≤ budget`. Typical outcome: hundreds to thousands of tiny candidates at rung 0.
- Preemptible: evaluation jobs checkpoint at rung boundaries (spec 60 priority queue: evolution eval is below live inference and online adapters).

### 6.5 Orchestration (Go ↔ Rust)
- `evo-orchestrator` (Go) owns the generation state machine, variation operators (or delegates them to Rust via a job), rung scheduling, budget accounting and the NSGA selection call. `fa-evolution` exposes NSGA as a job so both decoders and strategies share it.
- `evo-worker` (Rust) pulls `jobs.evo.decoder.eval`, loads the snapshot + data window, trains/evaluates and returns `EvalResult`. Workers are stateless. Scaling = more workers.
- **Today (D-044):** the worker scores the genome **sequentially point-in-time** over an in-memory buffer of `md.trade.>` (last 50 000 trades per instrument, stopgap until VictoriaMetrics reads via `fa-tsdb` land), under `budget_gpu_ms`, holding a `ctl.gpu.*` lease (class `evolution_eval`) and returning `PREEMPTED` with a `checkpoint_ref` on `ctl.gpu.preempt`. Since D-049 the data comes from VictoriaMetrics (`fa_tsdb::trade_prices`, default 6 h window) with the buffer as fallback, decoders take a point-in-time `DecodeCtx` of `latent.state.>` latents, and multi-generation runs carry Pareto elites over and breed children through `evo.vary.decoder` (`fa_evolution::vary`). Eval reports persist to Garage (D-045).
- **Neural genomes** (`nn_*`, §4.5) take the same job: the worker loads the window, the latents (the recorded `LATENT` stream over the whole window whenever the recorder is enabled — training needs the past, which the live buffer only holds since the worker started —, else the live buffer) and, for the text families, the `TEXT` stream (`news.item.>` over `[start − tw, end]` by receive time); trains on the CPU for at most 60 % of the job budget (preemption / cancellation stop training early, best validation weights kept), then scores out of sample after the train cutoff with `fa_backtest::eval` (forecasting families) and `recon_eval` (reconstruction families). The bundle + card go to Garage (`neural.weights` in the payload), the trial registry gets the naive-policy P&L, `rnn_prior` / `retrieval` outputs are published like the baselines. Payload extras: `neural {decoder_id, output_family, params, layout, cutoff_ms, train {rows, steps, best_step, initial/train/val loss, latent_dim, latent_coverage, interrupted, warm_started, device}, recon, weights}`, `data.text`, `budget.train_ms`. Envelope key `warm_start: "<bucket>/<key>"` warm-starts from a parent bundle. Small genomes train and evaluate in seconds on the CPU (live check: 1 h of 1 s data, ~1 300 training rows, 150 steps ≈ 0.1–1.7 s training).

### 6.6 Feedback to encoders (core blocker #3; mechanism implemented in `fa-decoders`, D-009)
At generation end, the elite set `E` (per family, top by front rank and diversity) is handed to the encoder trainer (spec 36) as a feedback term:
`L_gen = Σ_{d∈E} w_d · L_d(decoder_d(expert_view(x)), target_d)`
Default proposal: **backprop through frozen elite decoders** into encoder + expert projectors for `feedback_steps`. Elite weights are frozen during encoder updates and re-fine-tuned on the next generation's warm start.
Weights `w_d` ∝ family quota × front rank. Family quotas ensure no single family dominates the representation.

**Implemented:** the elite selection, weighting, frozen loading and the differentiable auxiliary loss are the §4.6 API; the trainer side (when to call it, `λ_gen`, `feedback_steps`) belongs to spec 36.

### 6.7 Serving (model-runtime, implemented)
`bins/model-runtime` lists the cards every `FA_NEURAL_RELOAD_SECS` (300 s) and on `ctl.decoders.reload`, ranks them per (output family, timescale) with `select_top_k` and loads the best three bundles. The best decoder of a slot **serves** (`decoder.output.<family>`), the next one runs in **shadow** (`decoder.shadow.<family>`, not consumed by strategies), both at every closed bar of the decoder's timescale, with the series' closes, the latent the pipeline just produced for the instrument and the live news timeline as inputs. Contract: `rnn_prior` / `retrieval` publish horizon-return quantiles (+ analogs, `h:`), `visual.*` / `text.*` publish a score in [-1, 1] as `quantiles[0]` (future chart `2·P(up) − 1`, chart direction of the reconstructed raster, news / narrative sentiment of the decoded embedding normalised by the training targets) followed by components; refs add `src:neural`, `role:serving|shadow`, `card:<key>`. `FA_NEURAL_REPLACE_BASELINE=1` makes a serving neural decoder replace the baseline output of its family and timescale (default: alongside). Forecasts are scored online when their horizon closes (CRPS), chart / news outputs immediately (raster CE, `1 − cos`); a shadow that beats the serving decoder by `FA_NEURAL_PROMOTE_MARGIN` (2 %) after `FA_NEURAL_PROMOTE_MIN` (200) scored outputs each and `FA_NEURAL_PROMOTE_MINUTES` (30) of shadow time is promoted (roles swap, the promotion becomes a pin). `ctl.decoders.{status, reload, promote, pin, unpin, off, on}` (CtlRequest / CtlReply; `ctl.model.>` stays with the encoder snapshot listener). Heartbeat gauges `neural.*` (cards, loaded, slots, serving per slot, outputs per family and role, online n / loss per role, promotions, load errors, inference ms).

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `gpu_hours_per_generation` | 8 h (all timescales) | Single-GPU budget |
| `eta` | 3 | Successive-halving factor |
| `steps_0` | 200 | Rung-0 SGD steps |
| `rungs` | 4 | |
| `elite_size` | 64 per family | |
| `immigrant_frac` | 0.1 | |
| `feedback_elites` | 8 per family | |
| `family_quota` | equal | Weight share in feedback |
| `retrieval_k_range` | 5–200 | |
| `retrieval_min_gap` | ≥ horizon | Leakage guard |
| `rnn_pairs_from_mcmc` | 50 per generation | spec 33 |
| neural training share of the job budget | 0.6 | `evo-worker` (`TRAIN_BUDGET_SHARE`) |
| `FA_NEURAL_RELOAD_SECS` / `_PROMOTE_MIN` / `_PROMOTE_MARGIN` / `_PROMOTE_MINUTES` / `_REPLACE_BASELINE` | 300 / 200 / 0.02 / 30 / 0 | model-runtime serving (§6.7) |

## 8. Failure modes
| Failure | Response |
|---|---|
| Population converges to one family / one topology | Family quotas + crowding distance + immigrants. Diversity metrics on the evolution-lab dashboard |
| Retrieval leakage (analog outcome overlaps evaluation target) | `min_gap` enforced in the query; acceptance test with planted leaks |
| Decoder exploits encoder artefacts (e.g. snapshot-specific quirks) | Elites re-evaluated on the next snapshot; feedback weight decays for decoders failing transfer |
| Worker crash mid-eval | Job redelivered by JetStream (ack timeout); rung checkpoint resumes |
| Evaluation cost exceeds budget | Orchestrator truncates rungs and records partial results; budget overrun alert |
| Feedback degrades encoder SSL metrics | Feedback term capped by λ_gen (spec 32); snapshot not promoted if SSL/calibration regress (spec 36) |

## 9. Acceptance criteria
1. All eight families implement `Decoder` and pass a conformance test on synthetic data (forward, loss, save/load, cost estimate).
2. Genomes serialize to/from JSON and protobuf identically. Mutation and crossover produce valid genomes (property tests).
3. A generation run with a fixed budget respects the budget within 10% and produces a Pareto front stored in Helix with full lineage.
4. Retrieval decoders never return analogs within `min_gap` of the query time (property test).
5. RNN-prior decoders produce CRPS and tradeable-value scores using the shared cost model from spec 35.
6. The elite feedback export produces a loadable bundle consumed by the encoder trainer (spec 36) in an integration test.
7. Killing a worker mid-generation doesn't lose results (redelivery test).

**Status of the criteria (implementation):**
1. Six of eight families have a trainable decoder with conformance tests on SYNTHETIC data (forward, loss decrease, out-of-sample skill vs naive / mean target, save/load bundle round trip, deterministic training, cost estimate, point-in-time truncation, leakage gap); `visual.frame` and `text.summary` have none (external frames, no summary targets).
2. kv / JSON round trip and clamped mutation of the neural genes are tested; protobuf genomes do not exist (genomes travel as strings, D-044).
3. Budget, NSGA front, lineage: as before (D-049); neural training respects the job budget share.
4. Retrieval leakage gap: tested for the baseline and the learned retrieval (training masks, serving, feedback).
5. RNN-prior CRPS + tradeable value: via `fa_backtest::eval` (naive sign policy net of 1 bp, not yet the spec 35 venue cost model).
6. The feedback bundle is loadable and gives gradients on latents (unit test); the trainer integration test belongs to the spec 36 owner.
7. Redelivery: unchanged (JetStream ack window, D-044).

## 10. Open questions
1. **[Core blocker #3] Decoder→encoder feedback path.** Blocks spec 36 training loop and 32 `L_gen`. Options:
   - (a) Backprop through frozen elite decoders (default proposal §6.6). Simple, but couples encoders to specific decoders.
   - (b) **Distillation:** elites produce soft targets; encoders get an auxiliary loss to make latents predictive of those targets via a fresh probe head.
   - (c) **Sample reweighting:** windows where elites fail get higher weight in encoder SSL training (no gradient through decoders).
   - (d) **Fitness-only:** encoder genomes are selected by how well the decoder population does on them (evolution at the encoder level, no gradient feedback).
   - (e) Combination: (a) for differentiable families, (d) for retrieval/non-differentiable parts.
2. Targets for the **event summary** text decoder: where do ground-truth summaries come from?
3. ~~Should visual decoders generate pixels (expensive) or stay in embedding space for v1?~~ v1: low-resolution rasters (≤ 32×32, column distributions), no pixel generation (§4.5).
4. Is retrieval history re-embedded on every snapshot (cost!) or only a sampled subset?
5. Does the orchestrator implement variation operators in Go, or call Rust for them (single genome implementation)?
6. How are objectives normalised across families so the cross-family front is meaningful?

## 11. Deferred
- Large-language-model text generation decoders.
- Audio reconstruction decoders.
- Multi-GPU population training.
- Decoders as direct strategy inputs beyond GP primitives (spec 35 consumes decoder outputs via primitives for now).
