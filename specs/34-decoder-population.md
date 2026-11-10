# 34 — Decoder Population

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium |
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

**Implemented today (D-044, D-062):** CPU baselines exist for `rnn_prior` (quantile forecaster) and `retrieval` (analog retriever) as evolvable genomes in evo-worker, and for `text.news` / `text.summary` (lexicon sentiment and vocabulary novelty over `news.item.>`) and `visual.chart` / `visual.future` (breakout, trend t-statistic and range compression over bars) as fixed decoders in model-runtime. All publish `DecoderOutput` with `quantiles[0]` as the score and `inst:` / `topic:` / `kind:` / `src:` / `h:` refs. `text.narrative`, `visual.frame` and every neural decoder remain open.

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
- **Today (D-044):** the worker scores the genome **sequentially point-in-time** over an in-memory buffer of `md.trade.>` (last 50 000 trades per instrument, stopgap until VictoriaMetrics reads via `fa-tsdb` land), under `budget_gpu_ms`, holding a `ctl.gpu.*` lease (class `evolution_eval`) and returning `PREEMPTED` with a `checkpoint_ref` on `ctl.gpu.preempt`. Since D-049 the data comes from VictoriaMetrics (`fa_tsdb::trade_prices`, default 6 h window) with the buffer as fallback, decoders take a point-in-time `DecodeCtx` of `latent.state.>` latents, and multi-generation runs carry Pareto elites over and breed children through `evo.vary.decoder` (`fa_evolution::vary`). Eval reports persist to Garage (D-045). There is still no training step.

### 6.6 Feedback to encoders (proposal — core blocker #3)
At generation end, the elite set `E` (per family, top by front rank and diversity) is handed to the encoder trainer (spec 36) as a feedback term:
`L_gen = Σ_{d∈E} w_d · L_d(decoder_d(expert_view(x)), target_d)`
Default proposal: **backprop through frozen elite decoders** into encoder + expert projectors for `feedback_steps`. Elite weights are frozen during encoder updates and re-fine-tuned on the next generation's warm start.
Weights `w_d` ∝ family quota × front rank. Family quotas ensure no single family dominates the representation.

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

## 10. Open questions
1. **[Core blocker #3] Decoder→encoder feedback path.** Blocks spec 36 training loop and 32 `L_gen`. Options:
   - (a) Backprop through frozen elite decoders (default proposal §6.6). Simple, but couples encoders to specific decoders.
   - (b) **Distillation:** elites produce soft targets; encoders get an auxiliary loss to make latents predictive of those targets via a fresh probe head.
   - (c) **Sample reweighting:** windows where elites fail get higher weight in encoder SSL training (no gradient through decoders).
   - (d) **Fitness-only:** encoder genomes are selected by how well the decoder population does on them (evolution at the encoder level, no gradient feedback).
   - (e) Combination: (a) for differentiable families, (d) for retrieval/non-differentiable parts.
2. Targets for the **event summary** text decoder: where do ground-truth summaries come from?
3. Should visual decoders generate pixels (expensive) or stay in embedding space for v1?
4. Is retrieval history re-embedded on every snapshot (cost!) or only a sampled subset?
5. Does the orchestrator implement variation operators in Go, or call Rust for them (single genome implementation)?
6. How are objectives normalised across families so the cross-family front is meaningful?

## 11. Deferred
- Large-language-model text generation decoders.
- Audio reconstruction decoders.
- Multi-GPU population training.
- Decoders as direct strategy inputs beyond GP primitives (spec 35 consumes decoder outputs via primitives for now).
