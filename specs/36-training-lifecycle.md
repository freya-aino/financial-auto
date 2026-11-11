# 36 — Training Lifecycle

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium |
| **Owners (code)** | `bins/trainer`, `bins/model-runtime`, `bins/evo-worker`, `crates/fa-encoders`, `crates/fa-expert-space`, `go/services/evo-orchestrator` |
| **Depends on** | 31 Expert encoders, 32 Expert spaces, 30 Latent regions, 33 MCMC graph, 34 Decoder population, 35 Strategies, 41 Backtest & validation, 60 Ops |

## 1. Purpose
This spec defines how the model stack evolves over time: bootstrapping (gen 0), the generation cycle (encoder training → regions → MCMC → decoder evolution → feedback → strategy evolution), frozen snapshots and their promotion/rollback, online adapters on the live stream, reproducibility, and how this work shares a single GPU.

## 2. Scope
**In scope:**
- Gen-0 bootstrap.
- Generation state machine and cadence.
- Snapshot creation, versioning, promotion, rollback, retirement.
- Online adapters.
- Reproducibility requirements.
- GPU priority classes and preemption.
- Data windows (train / validation / evaluation / locked holdout boundaries).

**Out of scope:**
- The model internals (31–35).
- Infrastructure of the job queue (spec 60/21).
- Holdout and gate statistics (spec 41/40).

## 3. Decisions
- **Gen 0: self-supervised pretraining** (reconstruction + next-step objectives, spec 31 §6.5). Decoder feedback starts at gen 1.
- **Generations + online adapters:** each generation produces **frozen backbone snapshots** (per timescale), stored in Garage with metadata in Helix. Small **online adapters** (LoRA-like) update on the live stream between generations.
- **Elite decoder losses feed back** into encoder retraining each generation (spec 34 §6.6).
- **GPU priority queue:** live inference > online adapters > evolution eval > generation training. Jobs are **preemptible and checkpointed**.
- Single-GPU workstation now; everything is distributable over NATS work queues.
- **No walk-forward inside evolution.** A **locked holdout period** is never used for training or selection, only for promotion gates (spec 41).

## 4. Interfaces

### 4.1 Generation state machine (proposal)
```rust
pub enum GenPhase {
    Planned,
    DataFreeze,          // fix data windows, probe set version, seeds
    EncoderTrain,        // per timescale, per family candidates (+ feedback from prev gen elites)
    EncoderSelect,       // Pareto over encoder candidates (spec 34 machinery)
    Reembed,             // embed history with selected snapshot (retrieval index, regions)
    RegionFit,           // spec 30 fit + align to previous snapshot
    McmcInit,            // carry/transfer kernels (spec 33 §6.6)
    DecoderEvolve,       // spec 34
    StrategyEvolve,      // spec 35
    Validate,            // snapshot promotion checks (§6.4)
    Promoted | Rejected,
}
```
*Not implemented as a state machine:* encoder generations are `train.generation` jobs (trainer CLI, `trainer serve`'s planner or a direct enqueue), decoder / strategy / allocator generations are evo-orchestrator halving loops, and MCMC refits run in model-runtime; they share the work queues, the GPU arbiter and Helix lineage but no `GenPhase` record or `Generation` node.
Owned by `evo-orchestrator` (Go). Each phase dispatches work-queue jobs (`jobs.train.*`, `jobs.evo.*`) to Rust workers. Phase transitions are persisted in Postgres (`generation` table) and Helix (`Generation` node).

### 4.2 Control API (spec 50)
- `POST /api/generations` (start), `GET /api/generations/:id`, `POST /api/generations/:id/abort`
- `POST /api/snapshots/:id/promote`, `POST /api/snapshots/:id/rollback`
- `GET /api/gpu/queue`
*Implemented (control-api):* `GET /api/latent/snapshots`, `GET /api/latent/snapshots/{id}` (manifest, checklist, shadow gauges, latest adapter), `POST /api/latent/snapshots/{id}/promote|rollback`, `POST|DELETE …/{id}/shadow` (`POST …/shadow/stop`), `GET /api/gpu`, `GET /api/jobs`, `POST /api/jobs/{id}/cancel`; evolution generations under `/api/evolution/generations` (spec 34).

### 4.3 Bus
- `jobs.train.encoder`, `jobs.train.adapter`, `jobs.evo.*`, `jobs.mcmc.refit`
- `lifecycle.generation` (phase events), `lifecycle.snapshot` (promote/rollback/retire)
*Implemented:* `jobs.train.pretrain|generation|adapter` (`JOBS_TRAIN`, results on `jobs.result.train.<kind>`), `ctl.train.cancel`, `jobs.mcmc.refit` (`JOBS_MCMC`, spec 33), `ctl.model.reload|shadow|adapter` to model-runtime; no `lifecycle.*` subjects — promote / rollback are audited by control-api (`latent.snapshot.<action>`) and visible as Helix status changes.

## 5. Data model
| Entity | Store | Fields |
|---|---|---|
| `Generation` | Postgres + Helix node | id, timescales, phase, started_at, data windows, probe_set_version, budget, seeds, git commit, config hash |
| `EncoderSnapshot` | Helix (spec 31) | + status: candidate / selected / active / shadow / retired |
| `AdapterState` | Helix node + Garage weights | snapshot_id, timescale, version, updated_at, drift metrics (implemented, D-065/D-073: `HasAdapter` edge, artifact `fa-models/adapters/<snapshot>/<ms>.json` + `.lora.safetensors`) |
| `DataWindow` | Postgres | gen, kind (train/val/eval/holdout), start, end, timescale |
| Checkpoints | Garage `checkpoints/<job_id>/` (implemented: `fa-ckpt/<job_id>/state.json`, + `neural.safetensors` for neural fits) | for preemption resume |
| Run manifest | Garage `runs/<gen>/manifest.json` | everything needed for reproduction (§6.6) |

## 6. Behaviour & algorithms

**Implemented today (D-058):** the lifecycle runs with the statistical encoder standing in for a neural model. Snapshots are Garage folders (`manifest.json`, `weights.bin` = fitted normalisation statistics, `regions.json`, `region_map.json`) mirrored as Helix `Snapshot` nodes with `SnapshotParent` edges; `bins/trainer` serves `jobs.train.pretrain|generation|adapter` from the `JOBS_TRAIN` queue with GPU leases, checkpoints and preempt/resume; region identity is carried through probe-set alignment (`lineage_id`, `RegionMatch` edges, `carried_mass`); model-runtime loads the active snapshot per timescale at start and hot-swaps on `ctl.model.reload`, which promote/rollback trigger. The items D-058 left open — kernel transfer on swap, shadow serving, bar inputs, the neural families — have since been delivered (D-062, D-065, D-070, D-073; below).

**Neural families (D-070, D-073):** `pretrain` / `generation` jobs and CLI fits take `family: fsq_hybrid | hier_vae_flow | jepa_vae | latent_diffusion` (default `statistical`) and `neural: {..}` genome overrides (`--family`, `--neural '<json>'`, `--steps`, `--device`); `bins/trainer/src/neural_fit.rs` trains the spec 31 model end to end on CPU from the same VictoriaMetrics bars (connector bars first, trades fallback) and writes the same D-058 folder (`weights.bin` = safetensors) and Helix mirror (`Snapshot.encoder_family` = the family, `ExpertSpace` per surviving expert with `dim = expert_dim`). A 12 h × 9-instrument 1m fit (≈ 4.6 k bars, 3.8 k training windows, 19 k parameters, 600 steps) takes ≈ 45 s on 12 cores. Gen ≥ 1 from a neural parent of the same family: **warm start** from the parent's weights (pruned experts may revive), **D-007 anchoring** — `λ_anchor` (0.1 when a parent exists) times the soft-min squared distance of each expert's latents to the parent's region centroids of the matched expert (`regions.json`) — in addition to the probe-set alignment fallback, and the **D-009 hook** `fa_encoders::neural::DecoderFeedback` (frozen elite decoders as auxiliary losses on the expert latents, `λ_feedback` 0.5). **The real elites are wired (D-073, through D-068's API):** a `generation` lists the decoder cards in Garage (`fa-models/decoders/**.card.json`), keeps those of the fit's step whose decoders read a latent of exactly the trained width (`n_experts × expert_dim`, the concatenated expert means of `LatentState`) with the current normaliser and positive score, takes `select_top_k` (4 per output family) with `quota_weights`, loads them as a frozen `FeedbackSet` and, per minibatch, builds each row's decoder context and target from the row's own price series (`FeedbackBatch.windows`: the latent of window `bars[..end]` is as of bar `end − 1`) and back-propagates `FeedbackSet::aux_loss` into the expert means only (`trainer::feedback::EliteFeedback`; `manifest.feedback.source = elite_decoders`, the elites, weights and skipped bundles in `feedback.decoders`). News-reading elites sit out (the trainer has no news timeline). Without eligible elites a neural parent of the same expert dimension supplies the **stand-in** `FrozenProbeFeedback` (ridge readouts of the parent's experts, `source = frozen_linear_probe`). Live: all 82 cards in Garage are 1s decoders on the 24-wide statistical latent, so a 1m fit falls back, while a 1s `generation` with 3 experts × 8 loaded 12 elites (retrieval, forecast, future families) and its feedback term fell from 1.00 to 0.85 over 150 steps. The snapshot's Helix node gets a `FeedbackFrom{weight}` edge to each elite's decoder genome, and evo-worker links the parents of every `evo.vary.decoder` job to the snapshot they were evaluated on with `TrainedOn` (D-084). Experts are matched to the parent by CKA and keep their index (spec 32 §6.5). Validation metrics recorded per snapshot (`metrics.k_eff`, `resid_corr_max`, `skill_min`, `skill_e<k>`, `usage_e<k>`, `mix_skill`, and with a neural parent `parent_mix_skill` / `ssl_regression` on the same windows) make §6.4 items 1 and 2 checkable: control-api adds `ssl_regression` (≤ `FA_PROMOTE_REGRESS_TOL` 0.02), `expert_k_eff` (≥ `FA_PROMOTE_MIN_K_EFF` 2), `expert_resid_corr` (≤ `FA_PROMOTE_MAX_RESID_CORR` 0.1) and `expert_min_skill` (≥ `FA_PROMOTE_MIN_SKILL` 0.02) to the checklist whenever the manifest carries them. Preemption / budget / cancel are honoured every 20 SGD steps; checkpoints are `fa-ckpt/<job_id>/neural.safetensors` (parameters, Adam moments, EMA target, step) + `state.json` (spec, data pins, snapshot id), and a resumed job continues bit for bit on CPU. Online adapters of a neural snapshot have two layers: the **input adapter** (the market front-end's normalisation re-fitted on the last window, drift guard `max_drift` on the inputs) and, on top of it, a **LoRA adapter** (§6.5; `fa_expert_space::lora`, `fa_encoders::neural::adapter`): rank-8 factors `W' = W + (α/r)·A·B` (`A` seeded `N(0, 1/in)`, `B = 0`, `α = 8`) on the backbone output projection `bb.out` and every active expert projector `moe.proj<e>`, trained for 100 Adam steps (lr 1e-4, batch 64) on all windows of the adapter window (the replay buffer) with the **SSL next-step objective only** (per-expert next-step + mixture + family SSL; decorrelation, balancing, KL, prior, anchor and feedback off) plus an L2 anchor of the factors (1e-3); the snapshot weights stay frozen (forward passes read `LoraView`). Guards from `params.adapter_guard`: `max_delta_norm` (‖ΔW‖_F/‖W‖_F, default 0.1) and `max_latent_shift` (mean expert-latent displacement relative to the latent spread on the windows, default 0.25); beyond either the LoRA layer is `frozen` and not served. The factors go to `fa-models/adapters/<snapshot>/<ms>.lora.safetensors` (`fa-lora-v1`), the artifact JSON gains `lora{key, rank, alpha, config, seed, report{loss_before, loss_after, delta_norm, latent_shift}, frozen}`, and model-runtime folds a non-frozen LoRA into the serving encoder's weights (`NeuralEncoder::with_lora`) together with the input normalisation. `--no-lora` / `no_lora` keeps the input adapter only. Live on `hvf-v1-1m-g0-1794370490`: SSL loss 4.54 → 4.43 on 343 recent windows, `delta_norm` 0.019, `latent_shift` 0.11 (served); a 1e-3 learning rate moved the latents by 1.0 spread and was frozen by the guard. model-runtime resolves the active snapshot across families and serves neural snapshots (active and shadow) with their learned router gates.

**Families 2 and 4 (D-073):** `family: hier_vae_flow | latent_diffusion` train through the same paths (`--family`, jobs, checkpoints, warm start, anchors, feedback, adapters); `latent_diffusion` is refused below days (D-010). Their regions are fitted with the `density` discovery they declare. Every neural fit writes the spec 32 `Expert` / `ExpertOf` / `AlignedTo` nodes. `trainer sweep --family … --grid '[…]' [--bars-cache f]` trains a list of genome overrides on one cached history and prints the §6.4 health, the checklist verdict, the in-sample health and the spec 32 §6.3.1 feasibility margin per genome without writing anything; `head_refit_steps` adds a full-batch refit of the next-step heads to a fit, and `head_solver: "constrained"` adds the exact constrained solve to a snapshot fit as to a sweep, even at `head_refit_steps` 0. Live: a 48 h × 9-instrument `hier_vae_flow` pretrain (24 k parameters, 96 s) wrote `hvf-v1-1m-g0-1794370490` (2 experts after pruning, `K_eff` 1.71, corr 0.42, min skill 0.22 — not promotable, §6.4 item 2) and it was shadow-served next to the statistical active snapshot (shadows live in model-runtime memory and are lost on restart, D-080). D-080 routes each neural family to its declared region discovery (`fsq_hybrid` codebook with the residual merge, `jepa_vae` mapper, families 2/4 density).

**Since D-062 and D-065:** candidates are shadow-served next to the active snapshot (`ctl.model.shadow`, gauges `shadow.<ts>.*`), promotion is refused unless carried mass, shadow minutes and shadow region agreement pass (or an audited force), swaps transfer the Dirichlet kernel through the region map, rollback restores exactly the replaced snapshot, adapters are applied automatically and recorded as `AdapterState` nodes, and `trainer serve` schedules adapter and generation jobs.

### 6.1 Data windows (proposal)
For each timescale, time is split (from oldest to newest):
`[ train | validation | evaluation | embargo | LOCKED HOLDOUT | live ]`
- **Train:** encoder + decoder SGD.
- **Validation:** early stopping, expert-space health (spec 32 §6.4).
- **Evaluation:** Pareto fitness for decoders/strategies (spec 34/35).
- **Embargo:** ≥ max horizon between evaluation and holdout (prevents overlap leakage).
- **Locked holdout:** fixed calendar period. Only spec 41 promotion gates can read it, and access is logged (each read counted per strategy).
- Windows roll forward per generation by `window_roll`. The holdout is either fixed for a long period or rolled with explicit re-locking (open question 2).

### 6.2 Gen 0 bootstrap
1. Backfill per timescale (spec 10: seconds ≈ weeks, minutes ≈ 1y, hours ≈ 5y, days = max).
2. Train each enabled encoder family per timescale with SSL only (`L_recon + KL/quant + L_next + L_32` with `L_gen` = reconstruction/multi-horizon from each expert alone).
3. Encoder selection by Pareto (SSL loss, next-step skill, expert-space health, cost).
4. Region fit, MCMC init with geometric priors, decoder seed population (one simple decoder per family + random genomes), then strategy population.
5. No strategy can be promoted from gen 0 without the full gate (spec 40/41).

### 6.3 Generation cycle (gen ≥ 1)
- Encoder candidates warm-start from the active snapshot (and online-adapter merge, optional) and train with the **feedback term from previous-generation elites** (spec 34 §6.6).
- Cadence (proposal): days/weeks timescales weekly; hours daily; minutes/seconds daily or on drift trigger. Generations can run per timescale independently.
- **Drift triggers** (early generation): sustained novelty (spec 30 §6.3), residual-correlation breakdown (spec 32), MCMC calibration degradation (spec 33), adapter drift beyond bounds.

### 6.4 Snapshot promotion and rollback
A selected snapshot becomes **active** only if, on the validation + evaluation windows:
1. SSL / next-step skill is not worse than the active snapshot by more than `regress_tol`;
2. expert-space health holds (`K_eff ≥ 2`, residual correlations ≤ target, `min_skill` satisfied);
3. region alignment carries ≥ `min_carry_mass` of mass, or the generation is explicitly marked as a representation break;
4. MCMC calibration (ECE) is not worse than `ece_tol`;
5. decoder elites transfer (re-evaluated elites lose ≤ `transfer_tol`).

The runtime can serve **multiple snapshots** simultaneously (active + shadow), so strategies tied to the old snapshot keep trading until their replacements pass shadow (spec 35 §6.5).
**Rollback:** one API call reverts the active pointer to the previous snapshot; adapters reset to that snapshot's last adapter state.
Retired snapshots are kept for `snapshot_retention` (reproducibility, audit).

*Enforced today (control-api `POST /api/latent/snapshots/{id}/promote`, 409 with the checklist otherwise):* item 3 for every snapshot (manifest `carried_mass` ≥ `FA_PROMOTE_MIN_CARRIED_MASS` 0.5, waived by `{representation_break: true}`); shadow serving as the precondition of the "serve both" rule (model-runtime's fresh heartbeat shows the candidate in shadow for ≥ `FA_PROMOTE_MIN_SHADOW_MINUTES` 30 with region agreement ≥ `FA_PROMOTE_MIN_REGION_AGREEMENT` 0.3); items 1 and 2 when the manifest carries the neural metrics (`ssl_regression` ≤ `FA_PROMOTE_REGRESS_TOL` 0.02, `K_eff` ≥ `FA_PROMOTE_MIN_K_EFF` 2, residual correlation ≤ `FA_PROMOTE_MAX_RESID_CORR` 0.1, min skill ≥ `FA_PROMOTE_MIN_SKILL` 0.02) — reported `n/a` for the statistical family; items 4 (no ECE producer) and 5 (elites not re-evaluated on candidates) are listed as `not_evaluated`. `{force: true, note}` overrides, audited. Promote stores the replaced snapshot as `previous_active`; rollback restores exactly that one; the reload serves the restored snapshot **without** an adapter until its next adapter is applied (`ctl.model.adapter` or the next adapter job) instead of restoring its last adapter state. On the live data **no neural snapshot meets items 1–2 together** (spec 32 §6.3.1), so none is promotable without `force`.

### 6.5 Online adapters
- LoRA-like low-rank adapters on attention/projection weights of the active backbone and on expert projectors (spec 32), per timescale.
- Trained continuously on the live stream with the SSL next-step objective only (no decoder feedback online), with a small LR, a replay buffer of recent windows and an L2 anchor toward zero (stays close to the snapshot).
- **Guards:** adapter drift (norm of ΔW, change in region occupancy, KL of latent distribution) bounded by `adapter_max_drift`; beyond it, adapters freeze and a drift trigger fires.
- Adapter updates are versioned (`AdapterState`) every `adapter_checkpoint_interval`; strategies see a consistent adapter version per step.
- Region membership is computed against the snapshot's region set (spec 30 §6.6); adapters must not move latents across region boundaries en masse (monitored).

### 6.6 Reproducibility
Every generation writes a run manifest containing: git commit, `Cargo.lock`/`go.sum` hashes, config hash, seeds per job, data window boundaries, source data version (raw-archive object hashes or VM query snapshots), probe-set version, backend + device info.
Required: re-running any job from its manifest on the same backend reproduces results within the documented nondeterminism tolerance (GPU kernels).

### 6.7 GPU scheduling (shared with spec 60)
*Device side today:* jobs still lease in their class (`generation_training`, `online_adapter`) with `--gpu-lease-mb`; the neural fit records `metrics.mem_estimate_mb` (parameters × 4 bytes × (weights + grads + 2 Adam moments) + one batch of backbone activations) and `params.device`. Training and serving run on the device named by `FA_ML_DEVICE` / `--device` / `spec.device` (`cpu` default; `cuda[:n]` only with the `cuda` cargo feature, otherwise a logged fallback to CPU). No GPU exists on the dev host, so CUDA paths are compiled out and unverified.
| Priority class | Workloads | Preemptible |
|---|---|---|
| P0 live inference | encoders + experts + decoders needed by live strategies | never |
| P1 online adapters | adapter training steps | yes (pauses) |
| P2 evolution eval | decoder/strategy candidate training & evaluation (spec 34/35) | yes, at rung checkpoints |
| P3 generation training | encoder candidate training, re-embedding, batch MCMC refits | yes, at checkpoints |
A single local GPU scheduler (in `evo-orchestrator` or a dedicated scheduler service, spec 60) grants leases. VRAM is budgeted per class (`vram_reserve_p0`). Jobs checkpoint every `checkpoint_interval` and resume on any worker.

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `generation_cadence` | weekly (d/w), daily (h, m, s) | per timescale |
| `window_roll` | = cadence | |
| `holdout_length` | days ts: 1 y; hours: 6 mo; minutes: 2 mo; seconds: 2 wk | |
| `embargo` | ≥ max horizon of timescale | |
| `regress_tol` | 2% | |
| `min_carry_mass` | 0.5 | |
| `ece_tol` | +0.02 | |
| `transfer_tol` | 10% | |
| `adapter_rank` | 8 | |
| `adapter_lr` | 1e-5 | LoRA default 1e-4 × 100 steps (CPU, 1-hour-scale windows); 1e-3 trips the latent-shift guard |
| `adapter_max_drift` | KL 0.05 | |
| `adapter_checkpoint_interval` | 15 min | |
| `checkpoint_interval` | 10 min | |
| `vram_reserve_p0` | 4 GB | |
| `snapshot_retention` | 12 generations | |

## 8. Failure modes
| Failure | Response |
|---|---|
| Generation phase fails | Phase retried up to N times; then generation rejected; active snapshot unchanged |
| Promotion checks fail | Snapshot rejected; diagnostics in UI; next generation proceeds from the current active snapshot |
| Live degradation after promotion | Health monitors (spec 40 halts) + one-click rollback; automatic rollback if `auto_rollback` enabled and health halts fire within `rollback_window` |
| Adapter runaway drift | Adapters frozen; drift trigger; strategies notified via health signal |
| GPU contention starves training | Budget accounting shows starvation; generation cadence stretches rather than preempting P0/P1 |
| Holdout contamination (accidental read) | Holdout access only through gate API; any other access path fails a test; counter + audit log in Postgres |

## 9. Acceptance criteria
**Remaining (why this spec stays):**
- No generation state machine exists (§4.1): there is no `GenPhase`, no `Generation` row or node, no `lifecycle.*` subjects and no `/api/generations` for encoder generations. Gen-0 encoder selection by Pareto over family candidates (§6.2 step 3) and the end-to-end bootstrap of acceptance 1 (encoders → regions → MCMC → decoders → strategies in one run) are not built. The pieces run as separate jobs and loops.
- Promotion (§6.4): items 4 (MCMC ECE) and 5 (elite transfer) have no producer (`not_evaluated`). Items 1–2 are evaluated but no neural snapshot passes them on live data (spec 32 §6.3.1).
- Rollback does not restore the previous snapshot's last adapter; it serves without one until the next adapter. Shadows are not restored after a model-runtime restart.
- Online adapters (§6.5, acceptance 4) are hourly jobs, not continuous training. The planner schedules adapters and generations for **statistical** snapshots only; neural ones are enqueued by hand or by the region-drift trigger. The other §6.3 drift triggers (residual-correlation breakdown, MCMC calibration degradation, adapter drift) do not start an early generation.
- Reproducibility (§6.6, acceptance 6): the snapshot manifest pins seed, data window, instruments and genome, but there is no run manifest with git commit, `Cargo.lock` / `go.sum` hashes, config hash or device info.
- Holdout (§6.1, acceptance 7): evolution paths respect the locked holdout (spec 41, `fa_backtest::holdout`), but the encoder trainer does not consult `validation.holdout`. Its fit windows can cover a locked period, and no test asserts otherwise. There is no `DataWindow` table.
- §8: there is no `auto_rollback` on health halts.
- Open questions 2, 3 and 4 are undecided. Q1 is decided (spec 34 §10 Q1), Q5 is decided (D-006: the GPU arbiter lives in evo-orchestrator) and Q6 is decided (spec 30 §10 Q1).

*Status (neural lifecycle):* 3 holds (single API calls, hot swap with kernel transfer, active + shadow served concurrently; the shadow is not restored after a model-runtime restart); 2 holds for neural fits — a `generation` consumes the D-068 elite decoders as frozen feedback when elites of its step and latent width exist (verified live at 1s; `trainer::feedback` test on synthetic data) and the candidate is evaluated by the checklist, which no neural snapshot passes yet (spec 32 §6.3.1); 4 holds for neural snapshots with input + LoRA adapters (versioned `AdapterState`, guard-frozen LoRA, runtime folding), adapters are not yet trained continuously between hourly jobs; 5 holds bit for bit on CPU for all four families.
1. A gen-0 bootstrap runs end-to-end on a small synthetic + recorded dataset: encoders → regions → MCMC → decoders → strategies, persisting all artefacts.
2. A gen-1 run consumes gen-0 elites as feedback and produces a candidate snapshot evaluated by the promotion checks.
3. Promotion and rollback are single API calls; the runtime switches snapshots without restarting and serves active + shadow concurrently.
4. Online adapters update live, are versioned, and freeze automatically when drift exceeds bounds.
5. Preempting a P3 job and resuming it on another worker yields the same final result as an uninterrupted run (within tolerance).
6. A run manifest is sufficient to reproduce any training/eval job.
7. The holdout window can't be read by training or evolution code paths (integration test).

## 10. Open questions
1. Feedback form from elites (spec 34 open question 1 / core blocker #3): determines the gen ≥ 1 encoder training loop.
2. Locked holdout policy: fixed period for a long time (purer, but stale) vs rolling with re-locking (fresher, but each roll exposes old holdout to training)?
3. Do generations run per timescale independently, or must all timescales advance together (cross-timescale `REFINES` edges in spec 30 need consistent snapshots)?
4. Should online adapters be merged into the backbone at generation start (warm start) or discarded?
5. Who owns GPU scheduling: `evo-orchestrator` or a separate scheduler service (spec 60)? *Decided (D-006): the GPU arbiter in `evo-orchestrator` (`ctl.gpu.acquire|release|preempt`).*
6. Region identity strategy (spec 30 open question 1) affects step `RegionFit` and promotion check 3.

## 11. Deferred
- Multi-GPU data/model parallel training.
- Automatic hyperparameter search for the lifecycle tunables themselves.
- Continual-learning methods beyond adapters (e.g. EWC-style regularisation across generations).
