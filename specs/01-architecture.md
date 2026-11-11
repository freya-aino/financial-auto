# 01 — Architecture

| | |
|---|---|
| **Status** | draft |
| **Maturity** | High |
| **Owners (code)** | all crates, bins, go services |
| **Depends on** | [00](00-vision-and-scope.md) |

## 1. Purpose
Defines the components, the boundary between Rust and Go, how data flows, and which store each component writes to.

## 2. Scope
**In scope:** process topology, data flow, store responsibilities, language split.
**Out of scope:** internals of each component (see the component specs).

## 3. Decisions
- Rust: core types, ML (candle, CPU with an opt-in `cuda` feature; D-068, D-070 depart from burn), model runtime, strategy runtime, risk engine (allocator + limits + paper book), training, evolution workers, ledger, backtest, control API. The live path is split into three processes (`model-runtime` → `strategy-runtime` → `risk-engine`, D-040) rather than one; one `model-runtime` serves all timescales (D-005).
- Go: connectors + `ingestor`, `broker-gateway`, `evo-orchestrator`, and `healthprobe` (the compose healthcheck binary of the distroless Go images); VictoriaMetrics and `vmalert` are Go as well.
- Contracts: protobuf `fa.v1.*` ([21](21-bus-and-contracts.md)); the bus is NATS JetStream (live streams + work queues).
- Stores: HelixDB (graph/vectors/lineage), VictoriaMetrics (time series), PostgreSQL (static records, ledger, audit), Garage S3 (blobs). See [20](20-storage.md).
- Local-only, no auth.
- Deployment: everything runs in `compose.yaml` (D-026): infra and ops services by default, the platform services under profile `app`, the `trainer` CLI under profile `tools`. control-api starts/stops/restarts app services through the Docker Engine API on the compose socket (D-060).

## 4. Interfaces
```
                ┌──────────────────── Go ─────────────────────┐
  ~70 APIs ───► │ connectors ─► ingestor ──┐                   │
                │                          │ canonical events  │
                │ broker-gateway ◄─────────┼──── order intents │
                │ evo-orchestrator ─ jobs ─┼─────────┐         │
                └──────────────────────────┼─────────┼─────────┘
                                           ▼         ▼
                                   ┌──── NATS JetStream ────┐
                                   │ md.* news.* media.*    │
                                   │ latent.* signal.*      │
                                   │ order.* jobs.*         │
                                   │ decoder.* risk.* ops.* │
                                   └──┬─────────┬───────┬───┘
                ┌──────────── Rust ───┼─────────┼───────┼────────────┐
                │ model-runtime ◄─────┘         │       │            │
                │   encoders → expert spaces → regions → latent.*    │
                │ strategy-runtime: latent.* + md.* → GP → signal.*  │
                │ risk-engine: signal.* → allocator → risk ─────────┼─► order.*
                │ trainer (gen-0, generations, adapters)             │
                │ evo-worker ◄── jobs.evo.* ────┘       │            │
                │ control-api ◄─────────────────────────┘ (WS tap)   │
                └─────┬───────────────┬──────────────┬───────────────┘
                      ▼               ▼              ▼
                  HelixDB       VictoriaMetrics   PostgreSQL     Garage S3
                (graph/vec)     (series)          (static/ledger) (blobs)
                                                    ▲
                                   frontend (SvelteKit) ──► control-api
```

| Process | Lang | Reads | Writes |
|---|---|---|---|
| ingestor | Go | APIs, NATS `ctl.ingest.*`, `JOBS_INGEST`, `ref.changed.*` | NATS (`md.*`, `news.*`, `macro.*`, `filing.*`, `ingest.*`, `risk.halt.source:*`), VM, Postgres (catalog, reference, ingest, `wh`), Garage (raw) |
| model-runtime | Rust | NATS `md.*`, `news.*` (`TEXT` backlog replayed at start), Helix, Garage (snapshots, decoder bundles, adapters) | NATS (`latent.*`, `decoder.output.*`, `decoder.shadow.*`, `jobs.mcmc.refit`), Helix (regions, transitions, lineage), Garage (MCMC draws) |
| strategy-runtime | Rust | NATS `md.*`, `latent.*`, `decoder.*`, VM (analog pricing), Garage (populations) | NATS `signal.exposure.*` (D-040) |
| risk-engine | Rust | NATS `signal.*`, `order.*.<account>`, `md.trade.*`, `risk.halt.*`, `risk.resume.*`, `ingest.health.*`, `latent.mcmc.*` (drift halts), `exec.costs.calibrated`, Postgres (reference, ledger), Garage (strategy fronts) | NATS `order.intent.*`, `order.approved.*`, `risk.state`, `risk.halt.*`, `risk.event.*`, holdout jobs; Postgres (ledger, `ref.fx_rate`, strategy state); Garage (live strategy set) (D-040..D-042, D-061, D-071, D-072) |
| trainer | Rust | VM, Garage, Helix, Postgres, NATS `JOBS_TRAIN` | Garage (snapshots, checkpoints, adapters), Helix (snapshots, lineage, experts) |
| evo-orchestrator | Go | NATS `jobs.result.*`, `ctl.evo.*`, `ctl.gpu.*`, `latent.crossprior.*` | NATS `jobs.evo.*`, `ctl.gpu.*`, generation summaries; Garage (strategy/allocator fronts) |
| evo-worker | Rust | NATS jobs, Garage, VM, Helix, Postgres (trial registry), `LATENT` stream | NATS results, Garage (eval reports, decoder bundles), Helix, Postgres (validation) |
| broker-gateway | Go | NATS `order.approved.*`, `order.cancel.*`, `md.trade.*`, `md.quote.*`, risk-engine heartbeat | paper venue, NATS `order.event.<account>`, `exec.costs.calibrated`, local state volume (D-051, D-076, D-079) |
| control-api | Rust | everything; vmalert notifications; Docker socket | config, commands on NATS, `ops.alert`, `dlq.*`, Postgres (audit, `ops`, operator ledger actions) |
| vmalert | Go | VM | `ALERTS` series → VM, notifications → control-api (D-066) |
| ui | nginx (static SvelteKit) | control-api `/api`, `/ws` | — (D-035) |

## 5. Data model
See [20](20-storage.md) and [21](21-bus-and-contracts.md).

## 6. Behaviour & algorithms
- **Live path** (per timescale): event → feature windows → encoder snapshot + online adapter → expert views → region assignment → MCMC kernel update → decoders → strategies → exposure distributions → allocator → risk → order intent → broker-gateway.
- **Learning path:** trainer runs generations (gen 0 self-supervised, then decoder feedback) and publishes snapshots. evo-orchestrator schedules decoder/strategy candidates on evo-workers within a GPU-hour budget.
- **Replay path:** the backtest drives the *same* live path from the Garage archive and VM ([41](41-backtest-and-validation.md)).
- **GPU arbitration:** priority queue, live inference > online adapters > evolution eval > generation training, held by the arbiter in `evo-orchestrator` (`ctl.gpu.*`, D-006) ([60](60-ops.md)).

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `runtime.timescales_per_process` | all | may split per timescale for latency |

## 8. Failure modes
- Bus down → every producer buffers locally up to a bound, then halts; risk halts trading ([40](40-execution-and-risk.md)).
- A store is unavailable → the services that depend on it degrade or halt as specified per service in [60](60-ops.md).

## 9. Acceptance criteria
- Every process in the table above exists as a directory with a README that links back here.
- No Rust↔Go communication bypasses protobuf contracts. *Known exceptions:* JSON inside `Job.spec` / `JobResult.result`, the broker-gateway's `exec.costs.calibrated` report, and the JSON ingest-frame fallback (`FA_INGEST_HEALTH_JSON=1`, D-067).
- Backtest and live share one code path for decisions.

## 10. Open questions
1. ~~One `model-runtime` process for all timescales, or one per timescale?~~ **Decided (D-005):** one process for all timescales in v1; subjects carry the timescale, so a split needs no contract change.
2. ~~Where do decoder outputs live for UI inspection (Garage vs Helix)?~~ **Decided:** outputs are bus messages (`decoder.output.*` / `decoder.shadow.*`, `LATENT` stream); trained bundles and their `.card.json` live in Garage `fa-models/decoders/` (D-068), eval reports in `fa-artifacts/evals/`; control-api serves decoder status and cards (D-077).

## 11. Deferred
- Multi-machine topology (NATS cluster, remote workers).
