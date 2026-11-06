# Specifications

Specs are the source of truth until code exists. Each one follows [`_template.md`](_template.md). Stack choices are recorded as ADRs in [`adr/`](adr/).

**Workflow:** draft → review → accepted. Specs are refined one at a time; the ML core (30–33) comes first because it carries the most risk.

## Index & maturity

**High** = can be accepted as written. **Medium** = structure settled, key definitions or numbers missing. **Low** = intent settled, needs another design round before it can be implemented.

| Spec | Status | Maturity | Main open items |
|---|---|---|---|
| [00 Vision & scope](00-vision-and-scope.md) | draft | High | Success metrics, starting capital |
| [01 Architecture](01-architecture.md) | draft | High | One runtime process vs one per timescale |
| [02 Glossary](02-glossary.md) | draft | Medium | "Complex indicator"; exact wording for space / region / state |
| [10 Ingestion](10-ingestion.md) | draft | Medium-high | Stream vs poll per source, ASR/frame pipeline, news dedup |
| [11 Source catalog](11-source-catalog.md) | draft | Medium | Per-source facts not yet verified |
| [12 Reference data](12-reference-data.md) | draft | Medium | Authoritative source per field, adjustment policy |
| [20 Storage](20-storage.md) | draft | Medium | Order books / cardinality in VictoriaMetrics, retention budget, Helix schema |
| [21 Bus & contracts](21-bus-and-contracts.md) | draft | Medium-high | Retention per stream, versioning, backpressure |
| [30 Latent regions](30-latent-regions.md) | draft | Low-medium | **Region identity across retraining** |
| [31 Expert encoders](31-expert-encoders.md) | draft | Low-medium | Modality fusion, router mechanics, **pure-Rust media feasibility** |
| [32 Expert spaces](32-expert-spaces.md) | draft | Low | **Exact decorrelation/generalization loss**, expert-count stability |
| [33 MCMC graph](33-mcmc-graph.md) | draft | Low-medium | Transitions across snapshots, calibration scoring |
| [34 Decoder population](34-decoder-population.md) | draft | Medium | Genome encoding, **decoder→encoder feedback path** |
| [35 Strategies](35-strategies.md) | draft | Medium-low | GP DSL semantics, comparing policies and programs |
| [36 Training lifecycle](36-training-lifecycle.md) | draft | Medium | Generation cadence, adapter drift/reset |
| [40 Execution & risk](40-execution-and-risk.md) | draft | Medium-high | Numeric limits, payout mechanics, override policy |
| [41 Backtest & validation](41-backtest-and-validation.md) | draft | Medium-high | Holdout length, cost model per venue |
| [42 Ledger & accounting](42-ledger-and-accounting.md) | draft | Medium | Fee schedules, margin/funding, tax lots |
| [50 Control API](50-control-api.md) | draft | Medium | Endpoint catalog detail |
| [51 Frontend](51-frontend.md) | draft | Medium | Rendering large graphs / latent views |
| [60 Ops](60-ops.md) | draft | Medium | Backups, recovery order, GPU memory partitioning |

## Core blockers (next refinement round)
1. **Region identity across retraining:** every generation shifts the latent space. The Markov graph, the GP graph operations and retrieval all need regions with stable identity. → [30](30-latent-regions.md), [33](33-mcmc-graph.md)
2. **Expert-space loss:** the formula and weighting for decorrelating residuals while keeping every space decodable. → [32](32-expert-spaces.md)
3. **Decoder → encoder feedback path:** backprop through elite decoders, distillation, or reweighting. → [34](34-decoder-population.md), [36](36-training-lifecycle.md)
4. **Pure-Rust media feasibility:** ASR and pretrained towers in candle are realistic. Training diffusion/video models in burn on one GPU needs a spike. → [31](31-expert-encoders.md)

## ADRs
| ADR | Decision | Status |
|---|---|---|
| [0001](adr/0001-rust-go-split.md) | Rust core, Go for parallel services | accepted |
| [0002](adr/0002-helixdb-hub.md) | HelixDB as graph/vector hub | accepted |
| [0003](adr/0003-pure-rust-ml.md) | Pure-Rust ML (burn + candle) | accepted |
| [0004](adr/0004-nats-jetstream.md) | NATS JetStream bus + work queues | proposed |
| [0005](adr/0005-protobuf-contracts.md) | Protobuf contracts | proposed |
| [0006](adr/0006-local-only-no-auth.md) | Local-only, no auth | accepted |
| [0007](adr/0007-victoriametrics.md) | VictoriaMetrics for time series | accepted |
| [0008](adr/0008-postgres-warehouse.md) | PostgreSQL warehouse + ledger | accepted |
| [0009](adr/0009-garage-s3.md) | Garage S3 for blobs | accepted |

## Decision traceability checklist
Every design decision made so far, and the spec that records it.

- [ ] Regions = regions of latent space, family-specific discovery → 30
- [ ] MCMC = Markov state graph + Bayesian kernels, pluggable samplers → 33
- [ ] Learned MoE router, emergent expert count → 32
- [ ] Decorrelate next-step prediction residuals → 32
- [ ] Multi-horizon, matched scope → 00, 34, 35
- [ ] Media: text, video (frames + ASR), images/charts, audio → 10, 31
- [ ] All encoder families compete; one model per timescale → 31
- [ ] Generations + online adapters; gen 0 self-supervised → 36
- [ ] Asset selection learned / evolved / MCMC-prior-driven → 31, 33, 35
- [ ] Decoder families (visual ×3, retrieval with time distance, text ×3, RNN-prior) → 34
- [ ] ES + SGD; Pareto fitness; budget-driven population → 34
- [ ] Decoder fitness shapes encoders → 34, 36
- [ ] GP + neural strategies compete; GP primitives = graph ops, decoder outputs, portfolio state → 35
- [ ] Neural policy outputs an exposure distribution → 35
- [ ] Pluggable meta-allocator, pluggable exec algos → 40
- [ ] Long + short, leverage cap per asset class → 40
- [ ] Hard limits incl. liquidity cap net of payout quota → 40
- [ ] Promotion = min shadow time + stats → 40, 41
- [ ] Locked holdout, deflated Sharpe/PBO, shadow; no walk-forward in evolution → 41
- [ ] Canonical proto events, three timestamps, raw archive → 10, 21
- [ ] Internal UUID + FIGI/ISIN/LEI → 12
- [ ] Helix / VictoriaMetrics / Postgres / Garage responsibilities → 20
- [ ] NATS work queues for scaling; GPU priority queue → 21, 60
- [ ] Backfill depth per timescale → 10
- [ ] EUR base currency → 00, 42
- [ ] Frontend v1: ops, latent/graph explorer, evolution lab, trading desk → 51
