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

### 4.1 Decoder trait (original proposal; implemented form below)
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

**Implemented (`fa_decoders::decoder`):** one object-safe `Decoder` trait for every genome family — the two CPU baselines (`decoder::Baseline` wrapping a `SeriesDecoder`, built by `build_decoder`) and the six candle families (`neural::NeuralDecoder`): `id`, `family` (output family), `genome`, `expert_mask`, `forward(&DecodeView)` (series up to step `t`, an optional spec 32 `ExpertView`, the news timeline), `target(series, t, text)` → `(DecodeTarget::{Return, Recon}, known_at_ms)`, `loss(out, target)` (CRPS of the quantiles, raster CE, `1 − cos`) and `cost(series_len)` (ops per forward). A neural decoder reads the experts of its genome's **expert mask** from the view (`select_latent`; a view lacking a selected expert is `DecodeError::MissingExperts`); baselines condition on their own point-in-time `DecodeCtx`, masked at construction, because the retrieval baseline also needs the latents of past candidate windows. `decoder::tests::every_family_conforms` runs all eight genome families (six output families) through forward / target / loss / cost, the bundle round trip and the mask on SYNTHETIC data. `Module`/save-load is the safetensors bundle (neural) or the genome string (baselines).

### 4.2 Genome (original proposal; the implemented genome is the string of §4.4–4.5)
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

**Implemented genes beyond D-044:** `mask` (expert mask, bit `i` = expert index `i`, `0` = all; decimal or `0x…`, ≤ 16 bits), `src` (cross-asset source instrument: latents come from `src`, the series / `inst` is the target — the RNN-prior pair of §6.1), `ninst` (text families: read only the news matched to the series' instrument). The three print in the canonical kv form only when set, so older genome strings keep their canonical form. `DecodeCtx` records per-instrument expert layouts (`(expert_index, len)`, from `LatentState.experts`) and `DecodeCtx::masked(mask)` gives a decoder only its experts. `warm_start` is not a gene: the orchestrator puts the parent bundle ref in the eval envelope (§6.5).

**Implemented today (D-044, `bins/evo-worker`, `crates/fa-decoders`):** the generic `fa.v1.Job` / `fa.v1.JobResult` of spec 21 are used instead of dedicated messages. Job kind is `evo.eval.decoder` on `jobs.evo.eval.decoder` (work queue `JOBS_EVO`), results on `jobs.result.evo.eval.decoder`, progress on `jobs.progress.<job_id>`.

### 4.4 Baseline genome encoding (implemented, D-044)
The typed `DecoderGenome` above was the target; the implemented genome — baselines and the neural families of §4.5 alike — is a **string** parsed by `fa_decoders::genome::DecoderGenome::parse`, carried in `Job.spec` (UTF-8) or, failing that, `Job.genome_id`:

| Form | Example | Notes |
|---|---|---|
| key/value | `forecast:h=5;lb=32;kernel=16;levels=0.1/0.5/0.9` · `retrieval:k=20;tau=30;gap=5;metric=cos` | genes separated by `;` (never `,`: `ctl.evo.start` splits genome lists on commas) |
| JSON | `{"id":"g1","family":"retrieval","params":{"k":20,"levels":[0.2,0.8],"ts":"1m"}}` | only via `Job.spec` |
| bare id | `g1`, a ULID | deterministic genome derived from a hash of the id (family by parity, genes jittered around defaults) |

Baseline families: `forecast` (aliases `quantile`, `rnn_prior`; outputs on `decoder.output.rnn_prior`) and `retrieval` (`analog`; outputs on `decoder.output.retrieval`); the neural families are §4.5, the structural genes `mask` / `src` / `ninst` §4.3. Common genes: `ts` (timescale, default `1s`), `h` horizon, `lb` lookback, `levels`, `inst`. Forecast: `kernel`, `scale`, `shrink`. Retrieval: `k`, `tau` (days), `gap` (≥ `h`, leakage guard), `metric` (`cos` | `euclid`).

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

News items are tagged with the instruments they mention (`TextTimeline::push_tagged`; evo-worker and model-runtime both use `text_baseline::InstrumentIndex` over the `ref.*` snapshot — ticker tokens, cleaned names, `entities_raw`); `ninst=1` genomes read `embedding_for(Some(instrument), …)` in training, evaluation, serving and feedback alike. A cross-asset genome (`src`) reads the source's latents everywhere the own instrument's would be read.

Latents are the instrument's `latent.state` as of the step (concatenated expert means, standardised with the training mean / std — std floored at 1e-3, values clipped to ±8 in training, serving and feedback alike — plus a presence flag; `lat=0` trains unconditioned). A latent-conditioned `nn_chart` without enough latents before the cutoff falls back to the unconditioned decoder (return window → chart). Neural genes (all numeric, mutated by `evo.vary.decoder`, clamped by `with_params`): `width` 4–256, `depth` 1–4, `gru` 0/1, `steps` 1–20000, `lr`, `wd`, `bs`, `train` (share of the window by time used for training, 0.2–0.9), `emb`, `temp`, `pool`, `res` 4–32, `band`, `tdim` 16–1024, `tw` (minutes), `seed` (not mutated). Example: `nn_forecast:h=5;lb=32;width=32;depth=2;gru=1;steps=300;lr=0.003`. The canonical kv form prints the neural genes only for neural families, so baseline genome strings are unchanged.

**Training (point-in-time).** `neural::train` uses only rows whose target is known by the **train cutoff** (`outcome_end ≤ cutoff`), AdamW minibatch SGD, deterministic seeded initialisation, early stopping on the last 10 % of the training rows (best validation weights kept). The decoder then forecasts only at steps after the cutoff, so every objective is out of sample. Optional warm start copies parent variables with matching name and shape (§6.2.3).

**Bundles.** One safetensors file per trained decoder: the network variables (`net.*`) plus a `__meta__` u8 tensor with the `BundleMeta` JSON (`format = fa-decoder-bundle/v1`, canonical genome, families, input layout, latent normaliser and its version `norm_version`, the full latent's expert layout `latent_layout`, training-mean target, cutoff, train report, objectives, score). **Normaliser versions:** `NORM_VERSION = 2` (std floor + ±8 clip everywhere); cards / bundles without the field are version 1, the first generation trained before the fix (D-068 open item). `NeuralDecoder::from_bundle` refuses them (`StaleNormaliser`; `from_bundle_any` still reads them, e.g. for warm starts), `select_top_k` skips them, `feedback::stale_cards` lists them for re-training (§6.7 `ctl.decoders.retrain`). Garage: `fa-models/decoders/<generation>/<family>-<genome hash>-<job_id>.safetensors` and a JSON **card** `….card.json` (`EliteCard {meta, bucket, key, created_ms}`) next to it; cards are what elites are selected from (listing, no index object, so concurrent workers never race).

### 4.6 Decoder → encoder feedback API (implemented, D-009, `fa_decoders::neural::feedback`)
For the encoder trainer (spec 36); all types are re-exported candle types (`fa_decoders::neural::{Tensor, Device, DType}`).

| Item | Purpose |
|---|---|
| `feedback::{BUCKET, PREFIX, CARD_SUFFIX}` | `fa-models`, `decoders/`, `.card.json`: list the cards with `fa_blob::BlobStore::list(BUCKET, PREFIX)` and parse each as `EliteCard` |
| `select_top_k(&cards, k, Some(latent_dim)) -> Vec<EliteCard>` | top `k` per output family: Pareto rank over the §6.3 objectives, then family score, newest first; only decoders with `score > 0` (better than naive) and the current normaliser and, with `Some(d)`, latent-conditioned with a **full** (pre-mask) latent of exactly that dimension and no cross-asset `src` |
| `stale_cards(&cards)` | qualifying cards with an old latent normaliser (re-train list) |
| `quota_weights(&elites) -> Vec<(EliteCard, w)>` | §6.6 weights: equal family quota, `1/(1+rank)` inside a family, Σ w = 1 |
| `FeedbackSet::load(&[(bundle bytes, w)], latent_dim, &device)` | frozen elites (weights are constants, not `Var`s); bundles of another latent dimension or unconditioned ones are skipped (`skipped`), weights renormalised |
| `set.prepare(&[(&Series, step)], &TextTimeline) -> FeedbackBatch` | per elite: context (return window / news embedding) and target (z, raster, embedding) of each step; steps without context or target are left out for that elite only |
| `set.aux_loss(&latents [n, latent_dim], &batch) -> FeedbackLoss {total, parts}` | an expert-masked elite reads only its experts' columns of `latents` (`mask_columns` over `latent_layout`); `L_gen = Σ_d w_d · L_d(decoder_d([context, (latent − μ)/σ, 1]), target_d)` with the family loss of training (pinball, in-batch soft-kNN with the leakage gap, raster CE, 1 − cos); `total` is a scalar differentiable w.r.t. `latents` only; `parts` = `(decoder id, family, loss, weight)` |

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

**Implemented today:** Helix `Genome{genome_id, kind = decoder, family, params}` nodes with `ParentOf{op}` lineage edges and `FitnessRecord{generation, objectives, pareto_rank}` via `Scored` (written best effort by `evo.vary.decoder`, `fa_hub::{upsert_genome, link_parent, add_fitness_record}`); `READS`, `EVALUATED_ON` and a `ParetoFront` node are not written (the mask is in the genome string, the snapshot in the job; fronts are recomputed by control-api `GET /api/evolution/populations/{id}/pareto`). Weights: every trained neural decoder, not only elites, at `fa-models/decoders/<gen>/<family>-<hash>-<job>.safetensors` with its card (§4.5). `decoder-out/` inspection samples are not written; eval reports go to `fa-artifacts/evals/<generation>/<job>.json`.

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

**Neural families (implemented, §4.5):** trainable candle decoders for `rnn_prior` (MLP / GRU quantile prior), `retrieval` (learned metric), `visual.chart`, `visual.future`, `text.news` and `text.narrative`, evolved through `evo.eval.decoder` like the baselines and served by model-runtime (§6.7). Differences to the text above: visual decoders stay in **embedding/raster space at low resolution** (open question 3 answered "raster, no pixels"); the chart "perceptual loss on the frozen vision encoder" is replaced by the column CE on the raster (no frozen vision encoder exists); text decoders reconstruct **hashed bag-of-words embeddings**, not tokens; retrieval searches the series' own history in memory, not Helix vectors; the RNN prior forecasts the cumulative horizon return of the target (a cross-asset pair is the `src` gene: the source's latent conditions the target's prior; pairs come from the MCMC cross-asset priors, §6.5). `visual.frame` (needs broadcast frames, an external source) and `text.summary` (open question 2) have no neural family.

### 6.2 Evolution loop (per generation, per timescale)
1. **Seed:** elites of the previous generation + MCMC-proposed RNN pairs (spec 33 §6.5) + random immigrants (`immigrant_frac`).
2. **Variation:** mutation (topology add/remove/resize, hyperparameter perturbation, expert-mask bit flips, family-gene perturbation) and crossover (within family; layer-block crossover; expert-mask uniform crossover).
3. **Warm start:** children inherit compatible parent weights (shape-matching; new layers initialised near identity / zero-residual).
4. **Successive halving:** rung r trains `steps_r = steps_0 · η^r` and keeps the top `1/η` by a scalarised rung score (hypervolume contribution on the current front) → next rung.
5. **Full evaluation** of survivors on the evaluation window. Objectives are computed (§6.3).
6. **Selection:** NSGA-II (≤ 3 objectives) or NSGA-III (≥ 4 objectives, reference directions). Elitist archive of size `elite_size`.
7. **Feedback export:** the elite set per family is exported to the encoder trainer (§6.6).
8. **Persist:** genomes, lineage, objectives, front → Helix. Weights of elites → Garage.

**Implemented (orchestrator + worker, §6.5):** step 1 seeds every decoder loop with up to `mcmc_pairs` `nn_forecast` src→target genomes from the latest `latent.crossprior.<ts>` proposals (best score first — the score already weighs the pair's evolution evidence: model-runtime counts every evaluated card with `src`/`inst` genes as a win (`crps_skill > 0`) or loss and scales the pair's `|t|` by `2·(1 + wins)/(2 + wins + losses)`, spec 33 §6.5, D-080 — built from the population's first `nn_forecast` genome, else a small GRU template; strategy / allocator kinds never get pairs); step 2 breeds numeric genes and the expert mask (8 evolvable bits, flips + uniform crossover; `inst`/`src` kept from the first parent); step 3 hands each child its first parent's bundle and each survivor / elite its own previous bundle as `warm_start`; step 4 trains rung `r` of `R` with `η^-(R−1−r)` of the genome's SGD steps (`train_steps_fraction`, full fidelity at the last rung) and ranks with NSGA-II (non-dominated front, then crowding — the hypervolume-contribution scalarisation is not implemented); the per-loop budget truncates rungs (§6.4).

### 6.3 Objectives (Pareto)
| Objective | Definition | Direction |
|---|---|---|
| Reconstruction | Family loss normalised per family (z-score vs family baseline) | min |
| Forecast skill | CRPS skill vs naive baseline on next horizon (families with a forward target) | max |
| Trading value | Deflated Sharpe of the naive policy using the decoder output, net of costs (spec 41) | max |
| Compute cost | Inference FLOPs + VRAM + latency (normalised) | min |
Families without a meaningful value for an objective get a neutral value. Fronts are computed **per family** for selection, and across families for the feedback-elite choice. Overfitting control: PBO estimated across the candidate set per generation (spec 41).

**Implemented today (`fa_backtest::eval`, D-044):** the baseline series decoders report `crps` (min), `crps_skill` = `1 − crps/crps_naive` vs a zero-return forecast (max), `coverage` / `coverage_err` = `|coverage − nominal|` (min), `hit_rate` (max), `trade_sharpe` and `trade_psr` of the naive sign policy net of the **spec 35 venue cost model** (max; per unit of position change the cost of a 10 000 EUR taker fill on the series' venue / asset class from `costs.yaml`: half spread + square-root impact with the series' own daily σ + commission incl. its minimum — `evo_worker::news::trade_cost_bp`, reported per instrument as `cost_bp` and as `trade_cost` in the payload), `compute_cost` = log10 arithmetic ops per forecast (min), `n_eval`. "Reconstruction" has no value for these families. The orchestrator ranks rungs with NSGA-II over `objectives` (Go port kept in lock-step with `fa-evolution` by the shared fixture `go/services/evo-orchestrator/testdata/nsga_fixture.json`, D-049 — under `go/` so `just go-test`, which mounts only `go/`, runs it); `score` is only a fallback.

### 6.4 Budget-driven sizing
- Input: `gpu_hours_per_generation` (per timescale share from spec 60), measured cost per candidate per rung (EMA by family/size).
- `N_0` (initial candidates) solves `Σ_r N_0 · η^{-r} · cost_r ≤ budget`. Typical outcome: hundreds to thousands of tiny candidates at rung 0.
- Preemptible: evaluation jobs checkpoint at rung boundaries (spec 60 priority queue: evolution eval is below live inference and online adapters).

**Implemented (budget accounting, evo-orchestrator):** every `JobResult.used_gpu_ms` is added to the loop and generation totals; with `total_budget_ms` set, a rung whose survivors × per-candidate budget would take the loop beyond 110 % of it is not started — the loop ends with the partial results (summary note, `budget.truncated`, counter `budget.truncations`). Summaries carry `budget {total_budget_ms, loop_used_ms, used_ms, planned_ms, truncated, truncations}`.

### 6.5 Orchestration (Go ↔ Rust)
- `evo-orchestrator` (Go) owns the generation state machine, variation operators (or delegates them to Rust via a job), rung scheduling, budget accounting and the NSGA selection call. `fa-evolution` exposes NSGA as a job so both decoders and strategies share it.
- `evo-worker` (Rust) pulls `jobs.evo.decoder.eval`, loads the snapshot + data window, trains/evaluates and returns `EvalResult`. Workers are stateless. Scaling = more workers.
- **Today (D-044):** the worker scores the genome **sequentially point-in-time** over an in-memory buffer of `md.trade.>` (last 50 000 trades per instrument, stopgap until VictoriaMetrics reads via `fa-tsdb` land), under `budget_gpu_ms`, holding a `ctl.gpu.*` lease (class `evolution_eval`) and returning `PREEMPTED` with a `checkpoint_ref` on `ctl.gpu.preempt`. Since D-049 the data comes from VictoriaMetrics (`fa_tsdb::trade_prices`, default 6 h window) with the buffer as fallback, decoders take a point-in-time `DecodeCtx` of `latent.state.>` latents, and multi-generation runs carry Pareto elites over and breed children through `evo.vary.decoder` (`fa_evolution::vary`). Eval reports persist to Garage (D-045).
- **Neural genomes** (`nn_*`, §4.5) take the same job: the worker loads the window, the latents (the recorded `LATENT` stream over the whole window whenever the recorder is enabled — training needs the past, which the live buffer only holds since the worker started —, else the live buffer) and, for the text families, the `TEXT` stream (`news.item.>` over `[start − tw, end]` by receive time); trains on the CPU for at most 60 % of the job budget (preemption / cancellation stop training early, best validation weights kept), then scores out of sample after the train cutoff with `fa_backtest::eval` (forecasting families) and `recon_eval` (reconstruction families). The bundle + card go to Garage (`neural.weights` in the payload), the trial registry gets the naive-policy P&L, `rnn_prior` / `retrieval` outputs are published like the baselines. Payload extras: `neural {decoder_id, output_family, params, layout, cutoff_ms, train {rows, steps, best_step, initial/train/val loss, latent_dim, latent_coverage, interrupted, warm_started, device}, recon, weights}`, `data.text`, `budget.train_ms`. Envelope key `warm_start: "<bucket>/<key>"` warm-starts from a parent bundle; `train_steps_fraction` scales the SGD steps of a rung (payload `warm_start {ref, loaded, variables}`, `budget.train_steps`).
- **Orchestrator side of the neural loop (implemented):** the orchestrator remembers each genome's latest bundle (`neural.weights` of its eval result), sends it as `warm_start` to the genome's next rung and next loop, and after `evo.vary.decoder` sends each child its first parent's bundle (the vary result's `lineage[].parents_as_sent` are the orchestrator's own genome ids; `parents` are canonical). It subscribes to `latent.crossprior.>` (counters `crossprior.messages|proposals`) and seeds the pairs of §6.2; `ctl.evo.start` takes `mcmc_pairs` (default `-mcmc-pairs` / `FA_EVO_MCMC_PAIRS`, 4). Summaries add `warm_started`, `pairs_seeded`, `budget`; heartbeat `warm_start.jobs`, `pairs.seeded`, `budget.*`. Small genomes train and evaluate in seconds on the CPU (live check: 1 h of 1 s data, ~1 300 training rows, 150 steps ≈ 0.1–1.7 s training).

### 6.6 Feedback to encoders (core blocker #3; mechanism implemented in `fa-decoders`, D-009)
At generation end, the elite set `E` (per family, top by front rank and diversity) is handed to the encoder trainer (spec 36) as a feedback term:
`L_gen = Σ_{d∈E} w_d · L_d(decoder_d(expert_view(x)), target_d)`
Default proposal: **backprop through frozen elite decoders** into encoder + expert projectors for `feedback_steps`. Elite weights are frozen during encoder updates and re-fine-tuned on the next generation's warm start.
Weights `w_d` ∝ family quota × front rank. Family quotas ensure no single family dominates the representation.

**Implemented:** the elite selection, weighting, frozen loading and the differentiable auxiliary loss are the §4.6 API. The trainer uses it in neural `generation` jobs (`EliteFeedback`, D-073): the top 4 cards per output family with the fit's step, the current normaliser and the trained latent width, quota-weighted, loss into the expert means with `λ = 0.5`; news-reading elites sit out (no news timeline in the trainer); without eligible elites it falls back to a frozen linear probe of the parent. The statistical family has no loss and uses the decoder-skill-penalised hyper-parameter search instead (spec 36, D-065). Details: spec 36.

### 6.7 Serving (model-runtime, implemented)
`bins/model-runtime` lists the cards every `FA_NEURAL_RELOAD_SECS` (300 s) and on `ctl.decoders.reload`, ranks them per (output family, timescale) with `select_top_k` and loads the best three bundles. The best decoder of a slot **serves** (`decoder.output.<family>`), the next one runs in **shadow** (`decoder.shadow.<family>`, not consumed by strategies), both at every closed bar of the decoder's timescale, with the series' closes, the latent the pipeline just produced for the instrument and the live news timeline as inputs. Contract: `rnn_prior` / `retrieval` publish horizon-return quantiles (+ analogs, `h:`), `visual.*` / `text.*` publish a score in [-1, 1] as `quantiles[0]` (future chart `2·P(up) − 1`, chart direction of the reconstructed raster, news / narrative sentiment of the decoded embedding normalised by the training targets) followed by components; refs add `src:neural`, `role:serving|shadow`, `card:<key>`. `FA_NEURAL_REPLACE_BASELINE=1` makes a serving neural decoder replace the baseline output of its family and timescale (default: alongside). Forecasts are scored online when their horizon closes (CRPS), chart / news outputs immediately (raster CE, `1 − cos`); a shadow that beats the serving decoder by `FA_NEURAL_PROMOTE_MARGIN` (2 %) after `FA_NEURAL_PROMOTE_MIN` (200) scored outputs each and `FA_NEURAL_PROMOTE_MINUTES` (30) of shadow time is promoted (roles swap, the promotion becomes a pin). `ctl.decoders.{status, reload, promote, pin, unpin, off, on}` (CtlRequest / CtlReply; `ctl.model.>` stays with the encoder snapshot listener). Heartbeat gauges `neural.*` (cards, loaded, slots, serving per slot, outputs per family and role, online n / loss per role, promotions, load errors, inference ms, `stale_cards`, `narrative_unscored`). Each decoder gets its own latent (`select_latent`: the experts of its mask, from the **source** instrument's latest state for a cross-asset genome), text decoders with `ninst=1` the news model-runtime matched to the instrument. `text.narrative` outputs are scored online when their leading window `(t, t + tw]` has passed: `1 − cos` against the embedding of the news that actually arrived (unscored when none did, as in training). Stale-normaliser cards are never loaded; `status` lists them and `ctl.decoders.retrain {min_budget_ms?, max_budget_ms?, max_generations?, eta?}` re-submits their genomes to `ctl.evo.start` (one halving loop, no MCMC pairs).

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
| `mcmc_pairs` (implemented `rnn_pairs_from_mcmc`) | 4 per loop | `FA_EVO_MCMC_PAIRS` / `-mcmc-pairs`, `ctl.evo.start mcmc_pairs` |
| budget overrun tolerance | 10 % | rung truncation (§6.4) |
| decoder trade notional | 10 000 EUR | `evo_worker::news::TRADE_NOTIONAL_EUR` (cost per unit of position change) |
| evolvable expert-mask bits | 8 | `evo_worker::vary::MASK_GENES` |
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
1. Every genome family (two baselines, six neural) implements the unified `Decoder` trait and passes `every_family_conforms` (forward, target, loss, cost, bundle round trip, expert mask) on SYNTHETIC data, plus the per-family tests (loss decrease, out-of-sample skill vs naive / mean target, deterministic training, point-in-time truncation, leakage gap); six of the eight output families are covered — `visual.frame` and `text.summary` have no decoder (external frames, no summary targets).
2. kv / JSON round trip and clamped mutation of the neural genes are tested; protobuf genomes do not exist (genomes travel as strings, D-044).
3. Budget: per-loop accounting from `used_gpu_ms` with rung truncation beyond 110 % (`TestBudgetTruncatesRungs`); NSGA front, lineage: as before (D-049); neural training respects the job budget share.
4. Retrieval leakage gap: tested for the baseline and the learned retrieval (training masks, serving, feedback).
5. RNN-prior CRPS + tradeable value: via `fa_backtest::eval`, the naive sign policy paying the spec 35 venue cost model (`costs.yaml`, mounted into evo-worker as `FA_COSTS`).
6. The feedback bundle is loadable and gives gradients on latents (unit test); `trainer::feedback::elite_decoders_feed_back_into_a_neural_fit` trains an `nn_forecast` elite on an encoder's latents, plugs it into a neural fit and checks a finite non-zero auxiliary loss and the step / latent-width eligibility (SYNTHETIC data).
7. Redelivery: unchanged (JetStream ack window, D-044).

## 10. Open questions
1. **[Core blocker #3, decided for the neural families] Decoder→encoder feedback path.** Implemented as (a) for the neural encoder families (§4.6, §6.6, D-009/D-068/D-073) and as a (d)-like decoder-skill term in the statistical family's search (D-065). Options were:
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
