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
- Rust: core types, ML (candle, CPU with an opt-in `cuda` feature; D-068, D-070 depart from burn), model runtime, strategy runtime, risk engine (allocator + limits + paper book), training, evolution workers, ledger, organisational accounting (`fa-accounting`, D-104), backtest, control API. The live path is split into three processes (`model-runtime` → `strategy-runtime` → `risk-engine`, D-040) rather than one; one `model-runtime` serves all timescales (D-005).
- Go: connectors + `ingestor`, `broker-gateway`, `evo-orchestrator`, and `healthprobe` (the compose healthcheck binary of the distroless Go images); VictoriaMetrics and `vmalert` are Go as well.
- Contracts: protobuf `fa.v1.*` ([21](21-bus-and-contracts.md)); the bus is NATS JetStream (live streams + work queues).
- Stores: Qdrant (vectors, D-101), PostgreSQL (static records, ledger, audit, accounting `acct` D-104, model-state graph `graph` — the default `GraphStore` backend, D-108), VictoriaMetrics (time series), Garage S3 (blobs); HelixDB still holds genomes/fitness/lineage records, experts and the graph explorer until the full cutover (ADR 0010). All graph/vector access goes through `fa-hub`. See [20](20-storage.md).
- Local-only, no auth.
- Deployment: everything runs in `compose.yaml` (D-026) with no profiles, so `docker compose up -d` starts every service — infra, ops and the platform (D-123); `just infra-up` names the stores alone. control-api starts/stops/restarts app services through the Docker Engine API on the compose socket (D-060).

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
                 Qdrant    PostgreSQL              VictoriaMetrics  HelixDB    Garage S3
                (vectors) (static/ledger/graph)    (series)        (genomes)   (blobs)
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
| control-api | Rust | everything; vmalert notifications; Docker socket | config, commands on NATS, `ops.alert`, `dlq.*`, Postgres (audit, `ops`, operator ledger actions, `wh.decoder_output` D-098, `acct` daily close D-104/D-106) |
| vmalert | Go | VM | `ALERTS` series → VM, notifications → control-api (D-066) |
| ui | nginx (static SvelteKit) | control-api `/api`, `/ws` | — (D-035) |

"Helix" in the rows above means the hub crate `fa-hub`: since D-101/D-108 vectors go to Qdrant and the model-state graph to Postgres schema `graph` (default; `[graph] backend` / `FA_GRAPH_BACKEND`), while genome records, experts and the graph explorer stay HelixDB-resident until the full cutover.

## 5. Data model
See [20](20-storage.md) and [21](21-bus-and-contracts.md).

## 6. Behaviour & algorithms
- **Live path** (per timescale): event → feature windows → encoder snapshot + online adapter → expert views → region assignment → MCMC kernel update → decoders → strategies → exposure distributions → allocator → risk → order intent → broker-gateway.
- **Learning path:** trainer runs generations (gen 0 self-supervised, then decoder feedback) and publishes snapshots. evo-orchestrator schedules decoder/strategy candidates on evo-workers within a GPU-hour budget.
- **Replay path:** the backtest drives the *same* live path from the Garage archive and VM ([41](41-backtest-and-validation.md)).
- **Market-data modes (D-103):** `FA_MARKET_DATA` (`real` default | `synthetic` | `all`) gates what the decision path consumes — in `real` the synthetic `sim` venues are dropped by model-runtime, strategy-runtime, risk-engine and the paper gateway — while ingestion stays neutral. Ephemeral test stacks (D-096) run `synthetic`.
- **GPU arbitration:** priority queue, live inference > online adapters > evolution eval > generation training, held by the arbiter in `evo-orchestrator` (`ctl.gpu.*`, D-006) ([60, retired](../docker/README.md#reference-spec-60-ops-retired-2026-11-11)).

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `runtime.timescales_per_process` | all | may split per timescale for latency |

## 8. Failure modes
- Bus down → every producer buffers locally up to a bound, then halts; risk halts trading ([40](../bins/risk-engine/README.md#execution-and-risk-lasting-content-of-the-retired-spec-40)).
- A store is unavailable → the services that depend on it degrade or halt as specified per service in [60, retired](../docker/README.md#reference-spec-60-ops-retired-2026-11-11).

## 9. Acceptance criteria
- Every process in the table above exists as a directory with a README that links back here.
- No Rust↔Go communication bypasses protobuf contracts. *Known exceptions:* JSON inside `Job.spec` / `JobResult.result`, the broker-gateway's `exec.costs.calibrated` report, and the JSON ingest-frame fallback (`FA_INGEST_HEALTH_JSON=1`, D-067).
- Backtest and live share one code path for decisions.

## 10. Open questions
1. ~~One `model-runtime` process for all timescales, or one per timescale?~~ **Decided (D-005):** one process for all timescales in v1; subjects carry the timescale, so a split needs no contract change.
2. ~~Where do decoder outputs live for UI inspection (Garage vs Helix)?~~ **Decided:** outputs are bus messages (`decoder.output.*` / `decoder.shadow.*`, `LATENT` stream); trained bundles and their `.card.json` live in Garage `fa-models/decoders/` (D-068), eval reports in `fa-artifacts/evals/`; control-api serves decoder status and cards (D-077).

## 11. Deferred
- Multi-machine topology (NATS cluster, remote workers).
