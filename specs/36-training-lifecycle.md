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
Owned by `evo-orchestrator` (Go). Each phase dispatches work-queue jobs (`jobs.train.*`, `jobs.evo.*`) to Rust workers. Phase transitions are persisted in Postgres (`generation` table) and Helix (`Generation` node).

### 4.2 Control API (spec 50)
- `POST /api/generations` (start), `GET /api/generations/:id`, `POST /api/generations/:id/abort`
- `POST /api/snapshots/:id/promote`, `POST /api/snapshots/:id/rollback`
- `GET /api/gpu/queue`

### 4.3 Bus
- `jobs.train.encoder`, `jobs.train.adapter`, `jobs.evo.*`, `jobs.mcmc.refit`
- `lifecycle.generation` (phase events), `lifecycle.snapshot` (promote/rollback/retire)

## 5. Data model
| Entity | Store | Fields |
|---|---|---|
| `Generation` | Postgres + Helix node | id, timescales, phase, started_at, data windows, probe_set_version, budget, seeds, git commit, config hash |
| `EncoderSnapshot` | Helix (spec 31) | + status: candidate / selected / active / shadow / retired |
| `AdapterState` | Helix node + Garage weights | snapshot_id, timescale, version, updated_at, drift metrics |
| `DataWindow` | Postgres | gen, kind (train/val/eval/holdout), start, end, timescale |
| Checkpoints | Garage `checkpoints/<job_id>/` | for preemption resume |
| Run manifest | Garage `runs/<gen>/manifest.json` | everything needed for reproduction (§6.6) |

## 6. Behaviour & algorithms

**Implemented today (D-058):** the lifecycle runs with the statistical encoder standing in for a neural model. Snapshots are Garage folders (`manifest.json`, `weights.bin` = fitted normalisation statistics, `regions.json`, `region_map.json`) mirrored as Helix `Snapshot` nodes with `SnapshotParent` edges; `bins/trainer` serves `jobs.train.pretrain|generation|adapter` from the `JOBS_TRAIN` queue with GPU leases, checkpoints and preempt/resume; region identity is carried through probe-set alignment (`lineage_id`, `RegionMatch` edges, `carried_mass`); model-runtime loads the active snapshot per timescale at start and hot-swaps on `ctl.model.reload`, which promote/rollback trigger. Open: kernel transfer on swap (spec 33 §6.6), shadow serving of two snapshots, bar metrics as training input, and of course the neural families themselves.

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
| `adapter_lr` | 1e-5 | |
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
5. Who owns GPU scheduling: `evo-orchestrator` or a separate scheduler service (spec 60)?
6. Region identity strategy (spec 30 open question 1) affects step `RegionFit` and promotion check 3.

## 11. Deferred
- Multi-GPU data/model parallel training.
- Automatic hyperparameter search for the lifecycle tunables themselves.
- Continual-learning methods beyond adapters (e.g. EWC-style regularisation across generations).
