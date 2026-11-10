# trainer

Encoder training lifecycle (spec 36) with the **statistical family standing in for a neural model** (D-004): the plumbing — snapshot artifacts, Helix mirror, work-queue jobs, GPU leases, checkpoints, preemption, probe-set region alignment, promotion → hot reload — is real end to end; when a neural `EncoderFamily` lands only the fit inside `src/fit.rs` changes.

**Status:** implemented (D-058). CLI and `JOBS_TRAIN` worker; verified against the compose stack.

## Jobs / subcommands
| Kind (`Job.kind`, CLI) | What it fits | GPU class | Artifact |
|---|---|---|---|
| `train.pretrain` / `trainer pretrain` | gen-0 snapshot of one timescale from VictoriaMetrics history (`fa_trade_price` over `--window-hours`, price-only bars) | `generation_training` | snapshot folder, Helix `Snapshot{status: candidate}` |
| `train.generation` / `trainer generation` | gen N+1 from the parent (`--parent`, default: the active snapshot); records the latest `validation.generation_stats` row as `manifest.feedback` (the statistical family only re-fits statistics) | `generation_training` | snapshot folder + `SnapshotParent{op}` edge |
| `train.adapter` / `trainer adapter` | re-fit of the normalisation statistics on the last `--window-hours` (default 1 h) of a snapshot; drift = mean \|Δmean\|/σ vs the snapshot, `frozen` beyond `max_drift` (spec 36 §6.5 guard) | `online_adapter` | `fa-models/adapters/<snapshot>/<unix_ms>.json` |
| `trainer serve` | work-queue worker for `jobs.train.>` | — | — |

A snapshot folder `fa-models/snapshots/<snapshot_id>/` (and `--out/<snapshot_id>/`) holds `manifest.json` (`fa_encoders::SnapshotManifest`: id, family, timescale, generation, parent, created_at, params, probe-set reference, data window, metrics, feedback, seed), `weights.bin` (`NormStats`: per-feature mean/std, the "weights" of this family), `regions.json` (region set per expert fitted on the probe windows) and `region_map.json` (`fa_patterns::RegionMap`, the alignment to the parent). Snapshot ids are `stat-v1-<ts>-g<gen>-<unix_s>`.

### Region identity (spec 30 §6.4, D-007)
The probe set is up to `--probe-max` (512) window ends spread over the fit windows. The parent's assignment of the same windows comes from its stored `LatentVec` observations in Helix (`kind=observation`, `ref_id=region_id`, `ts` = bar close) — i.e. the live runtime's real region ids — and the new snapshot's assignment from leader clustering on the new latents. `fa_patterns::align_expert` matches regions by Jaccard overlap ≥ `--carry-threshold` (0.5): a matched region **inherits the parent's `lineage_id`** (region ids stay unique per snapshot, `<snapshot>-e<k>-r<nnn>`, as the Helix index requires) and gets a `RegionMatch{score}` edge; unmatched regions open a new lineage. `metrics.carried_mass` is spec 36 §6.4 check 3's input (a fitted snapshot vs the un-normalised baseline carries little mass: a representation break, as expected).

## Worker (`trainer serve`)
`fa_bus::streams::ensure_work_queue("train", "trainer")` → stream `JOBS_TRAIN` on `jobs.train.>`, one job at a time. Per job: GPU lease from the arbiter (`--gpu-lease-mb`, default 256; 0 disables), `JobProgress` on `jobs.progress.<job_id>`, in-progress acks every 20 s, checkpoint of the fit state (Welford accumulator + data window) to `fa-ckpt/<job_id>/state.json` every `--checkpoint-secs` (default 30), `ctl.gpu.preempt` for the job's lease → checkpoint + `STATUS_PREEMPTED`, message NAKed so JetStream redelivers it and the run **resumes from the checkpoint** (`Job.checkpoint_ref` overrides the key; the resumed fit equals the uninterrupted one), `budget_gpu_ms` → `STATUS_BUDGET_EXCEEDED` (0 = unlimited), result JSON (`protocol: train/v0`) on `jobs.result.train.<kind>` with `output_refs = [manifest key]`. `Job.spec` is JSON: `{"timescale":"1m","window_hours":6,"instrument"?,"parent_snapshot_id"?,"seed"?,"fit":{"stride","min_window"}?,"region_radius"?,"max_regions"?,"carry_threshold"?,"probe_max"?}` for fits, `{"snapshot_id"?,"timescale"?,"window_hours"?,"instrument"?,"max_drift"?}` for adapters.

## Run
```sh
cargo run -p trainer -- pretrain --timescale 1m --window-hours 6 --out ./snap        # Garage + Helix + ./snap/<id>/
cargo run -p trainer -- --local-only pretrain --timescale 1m --window-hours 6 --out ./snap   # no Garage/Helix
cargo run -p trainer -- generation --timescale 1m --parent stat-v1-1m-g0-1794292853
cargo run -p trainer -- adapter --timescale 1m --window-hours 1
cargo run -p trainer -- serve --checkpoint-secs 30
cargo run -p trainer --example enqueue -- adapter '{"timescale":"1m","window_hours":1}'   # prints the job id
cargo run -p trainer --example enqueue -- preempt <job_id>                               # exercise checkpoint + resume
curl 'http://127.0.0.1:7700/api/jobs?kind=train.adapter'
curl -X POST http://127.0.0.1:7700/api/latent/snapshots/<id>/promote                     # Helix status + ctl.model.reload
```
Garage credentials come from `config/secrets.toml` (D-045); without them snapshots are only written to `--out`. Postgres is read only for `generation` feedback. The trainer runs on the host (compose profile `tools`).

## Owns
`src/fit.rs` (resumable fit: statistics → regions → alignment → bundle), `src/data.rs` (VM bars), `src/artifact.rs` (Garage / local / Helix mirror, checkpoints), `src/run.rs` (job semantics), `src/worker.rs` (queue loop), `src/gpu.rs` (lease client), `src/feedback.rs`.

## Governing specs
- [36-training-lifecycle](../../specs/36-training-lifecycle.md), [30-latent-regions](../../specs/30-latent-regions.md) §6.4, [31-expert-encoders](../../specs/31-expert-encoders.md) §4.3, [21-bus-and-contracts](../../specs/21-bus-and-contracts.md) §4.3, [60-ops](../../specs/60-ops.md) §6.2; decisions D-004, D-006, D-007, D-045, D-048
