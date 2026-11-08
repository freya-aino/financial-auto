# evo-worker

Stateless worker evaluating decoder genomes pulled from the `JOBS_EVO` NATS work queue (`jobs.evo.>`).

**Status:** evaluates `evo.eval.decoder` jobs end to end on the live trade feed with baseline CPU decoders — genome parsing, point-in-time CRPS/coverage/trading-value scoring, multi-objective `JobResult`, `JobProgress`, GPU-arbiter lease with preemption handling, `DecoderOutput` publishing. Other job kinds still get `STATUS_FAILED` ("not implemented"). Evaluation data is an in-memory stopgap (see below).

## What it does per job
1. Decodes `Job`; genome = `Job.spec` (UTF-8 JSON) if present, else `Job.genome_id`. Format: [`crates/fa-decoders/README.md`](../../crates/fa-decoders/README.md) (`forecast:h=5;lb=32;…`, `retrieval:k=20;tau=30;…`, or a bare id that is hash-derived). Separate genes with `;`, never `,` — the orchestrator splits the genome list on commas.
2. Acquires a GPU lease from the arbiter (`ctl.gpu.acquire`, class `evolution_eval`, `--gpu-lease-mb`, default 512; `0` disables). Queued leases wait for `ctl.gpu.granted` until the job deadline; an unreachable arbiter is tolerated (CPU work).
3. Takes the buffered trades of every instrument (or the genome's `inst`), resamples them to the genome's timescale (last price per bucket) and runs `fa_backtest::eval::evaluate`: forecast at every step using only past data, scored against the realised horizon return. Stops at `budget_gpu_ms` (partial evaluation, `budget.truncated=true`), on `ctl.gpu.preempt` for its lease (→ `STATUS_PREEMPTED` with `checkpoint_ref`; redelivered jobs re-evaluate from scratch because evaluation is cheap) or when the window is exhausted.
4. Publishes `JobProgress` on `jobs.progress.<job_id>` every 500 ms, in-progress JetStream acks every 20 s (`ack_wait` 60 s, `max_deliver` 5), one message in flight per worker.
5. Publishes `DecoderOutput` on `decoder.output.rnn_prior` (forecast quantiles) / `decoder.output.retrieval` (analogs with similarity, window end, time distance) for up to 4 instruments, then the `JobResult` on `jobs.result.evo.eval.decoder`, then acks. If the result cannot be published the message is left unacked for redelivery.

### Result payload (`evo.eval.decoder/v0`, D-025 compatible)
```json
{"protocol":"evo.eval.decoder/v0","score":0.39,
 "objectives":{"crps":6e-4,"crps_skill":0.39,"coverage":0.94,"coverage_err":0.04,"hit_rate":0.56,
               "trade_sharpe":0.13,"trade_psr":0.69,"compute_cost":2.5,"n_eval":18},
 "directions":{"crps":"min","crps_skill":"max","coverage_err":"min","hit_rate":"max","trade_sharpe":"max","trade_psr":"max","compute_cost":"min"},
 "genome":{...},"data":{"source":"md.trade.> in-memory buffer (stopgap; not VictoriaMetrics)","instruments":[...]},
 "budget":{"budget_ms":2000,"elapsed_ms":11,"truncated":false},"stop":"complete",
 "report_ref":{"planned":"fa-artifacts/evals/<generation>/<job>.json","persisted":false}}
```
`score` = `crps_skill` (the orchestrator's provisional scalar); control-api shows `objectives` in the Evolution Lab. Objective definitions: `crates/fa-backtest/src/eval.rs`.

## Evaluation data (stopgap)
The worker subscribes to `md.trade.>` and keeps the last 50 000 trades per instrument (64 instruments) in RAM. A fresh worker has nothing to score until enough trades arrived: a genome needs `warmup + horizon + 1` resampled steps (forecast: `lb + 2h + 1`; retrieval: `2·lb + gap + 2h + 3`), i.e. ~1–3 minutes on the 1 trade/s heartbeat feed. Until then jobs fail with a message stating the buffer size. The intended source is VictoriaMetrics through `fa-tsdb` (spec 41 §5), which nothing populates yet. No `latent.state.*` conditioning yet (model-runtime does not publish latents). Eval reports are not persisted to Garage: `fa-blob` has no S3 client and `config/secrets.toml` has no credentials; the result carries the planned key.

## Run
```sh
cargo run -p evo-worker                # host, against compose NATS; FA_CONFIG=config/platform.toml
cargo run -p evo-worker -- --gpu-lease-mb 0
# then start a generation through control-api:
curl -X POST http://127.0.0.1:7700/api/evolution/generations -H 'content-type: application/json' \
  -d '{"kind":"evo.eval.decoder","snapshot_id":"snap-test","genomes":["forecast:h=3;lb=16","retrieval:h=3;lb=8;k=10","g1"],"eta":2,"min_budget_ms":2000}'
curl http://127.0.0.1:7700/api/evolution/jobs ; curl http://127.0.0.1:7700/api/decoders/outputs
```
Only one consumer may read a work queue, so stop the compose `evo-worker` (`docker compose stop evo-worker`) when running one on the host, or they compete for jobs.

## Owns
`bins/evo-worker/src/{main,job,market,gpu}.rs`: JetStream consumer loop, job evaluation, trade buffer, arbiter lease client.

## Governing specs
- [21-bus-and-contracts](../../specs/21-bus-and-contracts.md), [34-decoder-population](../../specs/34-decoder-population.md), [36-training-lifecycle](../../specs/36-training-lifecycle.md), [41-backtest-and-validation](../../specs/41-backtest-and-validation.md), [60-ops](../../specs/60-ops.md); decisions D-006, D-025.
