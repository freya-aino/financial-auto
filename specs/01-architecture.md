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
- Rust: core types, ML (burn/candle), model runtime, strategy runtime, risk engine (allocator + limits + paper book), training, evolution workers, ledger, backtest, control API. The live path is split into three processes (`model-runtime` → `strategy-runtime` → `risk-engine`, D-040) rather than one.
- Go: connectors + `ingestor`, `broker-gateway`, `evo-orchestrator`; VictoriaMetrics is Go as well.
- Contracts: protobuf `fa.v1.*` ([21](21-bus-and-contracts.md)); the bus is NATS JetStream (live streams + work queues).
- Stores: HelixDB (graph/vectors/lineage), VictoriaMetrics (time series), PostgreSQL (static records, ledger, audit), Garage S3 (blobs). See [20](20-storage.md).
- Local-only, no auth.

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
| ingestor | Go | APIs | NATS, VM, Postgres (reference), Garage (raw) |
| model-runtime | Rust | NATS, VM, Helix, Garage (snapshots) | NATS (`latent.*`, `decoder.*`), Helix |
| strategy-runtime | Rust | NATS `md.*`, `latent.*`, `decoder.*` | NATS `signal.exposure.*` (D-040) |
| risk-engine | Rust | NATS `signal.*`, `order.event.*`, `md.trade.*`, `risk.halt.*` | NATS `order.intent.*`, `order.approved.*`, `risk.state`, `risk.halt.*` (D-040..D-042) |
| trainer | Rust | VM, Garage, Helix, Postgres | Garage (snapshots), Helix (lineage) |
| evo-orchestrator | Go | Helix (populations) | NATS `jobs.evo.*` |
| evo-worker | Rust | NATS jobs, Garage, VM, Helix | NATS results, Garage, Helix |
| broker-gateway | Go | NATS `order.*` | venues, NATS `order.fill`, Postgres ledger |
| control-api | Rust | everything | config, commands on NATS |
| ui | nginx (static SvelteKit) | control-api `/api`, `/ws` | — (D-035) |

## 5. Data model
See [20](20-storage.md) and [21](21-bus-and-contracts.md).

## 6. Behaviour & algorithms
- **Live path** (per timescale): event → feature windows → encoder snapshot + online adapter → expert views → region assignment → MCMC kernel update → decoders → strategies → exposure distributions → allocator → risk → order intent → broker-gateway.
- **Learning path:** trainer runs generations (gen 0 self-supervised, then decoder feedback) and publishes snapshots. evo-orchestrator schedules decoder/strategy candidates on evo-workers within a GPU-hour budget.
- **Replay path:** the backtest drives the *same* live path from the Garage archive and VM ([41](41-backtest-and-validation.md)).
- **GPU arbitration:** priority queue, live inference > online adapters > evolution eval > generation training ([60](60-ops.md)).

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `runtime.timescales_per_process` | all | may split per timescale for latency |

## 8. Failure modes
- Bus down → every producer buffers locally up to a bound, then halts; risk halts trading ([40](40-execution-and-risk.md)).
- A store is unavailable → the services that depend on it degrade or halt as specified per service in [60](60-ops.md).

## 9. Acceptance criteria
- Every process in the table above exists as a directory with a README that links back here.
- No Rust↔Go communication bypasses protobuf contracts.
- Backtest and live share one code path for decisions.

## 10. Open questions
1. One `model-runtime` process for all timescales, or one per timescale? Blocks the deploy layout.
2. Where do decoder outputs live for UI inspection (Garage vs Helix)? Blocks 50/51.

## 11. Deferred
- Multi-machine topology (NATS cluster, remote workers).
