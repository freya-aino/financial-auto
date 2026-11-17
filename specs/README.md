# Specifications

Specs were the source of truth until code existed; now the code is, and the specs describe it. Each one follows [`_template.md`](_template.md). Stack choices are recorded as ADRs in [`adr/`](adr/). Implementation decisions (D-001..D-108) are logged in [`DECISIONS.md`](DECISIONS.md) and folded into the specs they affect.

**Retirement:** a feature spec is removed once the code meets every in-scope acceptance criterion and its open questions are decided; its lasting reference content (data model, tunables, invariants) moves into the owning component README first. Out-of-scope items (live broker, keyed sources, off-machine backup, GPU hardware) do not block retirement. As of 2026-11-11 specs 10, 11, 12, 30, 31, 33, 34, 36, 42, 50 and 60 are retired, and 35 followed on 2026-11-15 (their content lives in the ingestor, catalog, fa-warehouse, fa-patterns, fa-encoders, fa-mcmc, fa-decoders, fa-strategies, evo-orchestrator, evo-worker, trainer, model-runtime, risk-engine, strategy-runtime, fa-ledger, control-api and docker READMEs; the last versions are in git history); every other feature spec lists what keeps it at the top of its §9 ("Remaining"). Overview documents (00–02), ADRs and the decision log are not retired.

**Workflow:** draft → review → accepted. Specs are refined one at a time; the ML core (30–33) came first because it carries the most risk. Every spec is still `draft` (no owner review yet), and most D-entries are `provisional`.

## Index & maturity

**High** = can be accepted as written. **Medium** = structure settled, key definitions or numbers missing. **Low** = intent settled, needs another design round before it can be implemented.

| Spec | Status | Maturity | Main open items (feature specs: what keeps them from retirement) |
|---|---|---|---|
| [00 Vision & scope](00-vision-and-scope.md) | draft | High | Success metrics (target return, max drawdown), live starting capital and q.2 latency budget; the revenue split and opex prices are owner-confirmed (D-106) |
| [01 Architecture](01-architecture.md) | draft | High | None blocking (one runtime process, D-005); multi-machine topology deferred |
| [02 Glossary](02-glossary.md) | draft | Medium | "Complex indicator"; exact wording for space / region / state |
| 10 Ingestion | **retired** 2026-11-11 | — | Content in [ingestor README](../go/services/ingestor/README.md#ingestion-reference). Out of scope: media capture, keyed first-wave sources, a 24 h soak against external APIs (harness runs against a local fake) |
| 11 Source catalog | **retired** 2026-11-11 | — | Content in [catalog README](../go/pkg/catalog/README.md#source-catalog-reference). Out of scope: verifying the 62 planned entries, a CCXT adapter (external venues) |
| 12 Reference data | **retired** 2026-11-11 | — | Content in [fa-warehouse README](../crates/fa-warehouse/README.md#reference-data-reference). Out of scope: OpenFIGI/CoinGecko lookups, corporate-action feed, calendar seeding from venues (external) |
| [20 Storage](20-storage.md) | draft | High | Vectors in Qdrant (D-101, k-NN 10.9 ms p95 at 200 k) and the model-state graph in Postgres (D-108). Remaining: full Helix cutover (genomes/fitness/pareto/evaluations, graph explorer, experts, retention aggregates — then the `helix` service itself), k-NN re-bench at 1 M–10 M, Qdrant volume cap/alerting |
| [21 Bus & contracts](21-bus-and-contracts.md) | draft | High | Durable `LiveConsumer`s, `jobs::publish_result` and contract checks adopted everywhere (D-097); MD keeps 48 h (D-100). Remaining: `max_mib` re-measure after a week of market hours (Q1) |
| 30 Latent regions | **retired** 2026-11-11 | — | Content in [fa-patterns README](../crates/fa-patterns/README.md#region-model-lasting-content-of-the-retired-spec-30) and [model-runtime README](../bins/model-runtime/README.md) |
| 31 Expert encoders | **retired** 2026-11-11 | — | Content in [fa-encoders README](../crates/fa-encoders/README.md#encoder-model-lasting-content-of-the-retired-spec-31). Out of scope: media front-ends / fusion with pretrained towers (external weights), the GPU spike (no hardware) |
| [32 Expert spaces](32-expert-spaces.md) | draft | Medium | Objective reframed to end-to-end reconstruction + CKA diversity (D-099); 12/12 synthetic runs pass (D-102) **and item 2 now holds on real bars** (D-118). Remaining: a second neural generation with a neural parent, so `carried_mass` and `mcmc_ece` become meaningful; then promotion (owner's) |
| 33 MCMC graph | **retired** 2026-11-11 | — | Content in [fa-mcmc README](../crates/fa-mcmc/README.md#graph-model-lasting-content-of-the-retired-spec-33) and [model-runtime README](../bins/model-runtime/README.md) |
| 34 Decoder population | **retired** 2026-11-11 | — | Content in [fa-decoders README](../crates/fa-decoders/README.md#decoder-population-lasting-content-of-the-retired-spec-34), [evo-orchestrator](../go/services/evo-orchestrator/README.md) and [evo-worker](../bins/evo-worker/README.md) READMEs. Out of scope: the chart perceptual loss on a pretrained vision tower (external weights), live `visual.frame` training (no media producer). Live: text decoders at naive, not above (no text expert in the latents) |
| 35 Strategies | **retired** 2026-11-15 | — | Content in the [fa-strategies README](../crates/fa-strategies/README.md#strategy-model-lasting-content-of-the-retired-spec-35) (strategy model), [risk-engine README](../bins/risk-engine/README.md) (lifecycle pipeline) and [strategy-runtime README](../bins/strategy-runtime/README.md) (live serving). All six ACs met (AC 6 by D-098), Q1–Q6 answered |
| 36 Training lifecycle | **retired** 2026-11-11 | — | Content in [trainer README](../bins/trainer/README.md#training-lifecycle-lasting-content-of-the-retired-spec-36) and [model-runtime README](../bins/model-runtime/README.md). No neural snapshot passes checklist item 2 on live data (spec 32) |
| 40 Execution & risk | **retired** 2026-11-11 | — | Content in [risk-engine README](../bins/risk-engine/README.md#execution-and-risk-lasting-content-of-the-retired-spec-40) and the [broker-gateway README](../go/services/broker-gateway/README.md). Out of scope (live adapter): paper/live shared `Venue` tests, `trading.mode.live_enabled`, the initial live sleeve, the first live allocator (Q6) |
| [41 Backtest & validation](41-backtest-and-validation.md) | draft | High | Full-day live-replay equivalence only: deployed replays run from the retained streams on an ephemeral stack (D-096, D-100); the risk-engine's 20 % full-day failure was fixed by deterministic ticks (D-105). Remaining: the 24 h re-verification on a fresh ephemeral day run, then once more on real feeds |
| 42 Ledger & accounting | **retired** 2026-11-11 | — | Content in [fa-ledger README](../crates/fa-ledger/README.md) (same section numbers); organisational books above it in [fa-accounting](../crates/fa-accounting/README.md) (D-104, D-106). The pending FX restatement was resolved by D-107 (fresh book). Left to the operator/user: the tax jurisdiction and its rules (§10 Q2) |
| 50 Control API | **retired 2026-11-11** | — | Reference moved to [`bins/control-api/README.md`](../bins/control-api/README.md#reference-spec-50-retired-2026-11-11) |
| [51 Frontend](51-frontend.md) | draft | High | Rounds through 9 built (`/infra`, `/growth`, region detail, lifecycle view, organisational books, info popovers; D-098, D-104). Remaining: the 50 000-point projection verified interactive on the workstation's real browser (measured headless only) |
| 60 Ops | **retired 2026-11-11** | — | Reference moved to [`docker/README.md`](../docker/README.md#reference-spec-60-ops-retired-2026-11-11) (+ `scripts/README.md`) |

## Core blockers (status after round 7)
1. **Region identity across retraining:** every generation shifts the latent space. *Implemented:* probe-set alignment with Hungarian matching, lineage ids, split/merge/death and kernel transfer (D-007, D-069); anchored training for the neural families (D-070). → [fa-patterns](../crates/fa-patterns/README.md#region-model-lasting-content-of-the-retired-spec-30), [fa-mcmc](../crates/fa-mcmc/README.md#graph-model-lasting-content-of-the-retired-spec-33) (specs 30, 33 retired)
2. **Expert-space loss:** *met on real data 2026-11-16* (D-118: usage K_eff 3.99, latent CKA 0.033, SSL skill 0.547, min skill 0.328 on the first live 1m generation — every candidate family passed), after being *reframed by D-099* (owner): the spaces train end to end on reconstruction/JEPA + next step + Switch balancing + KL + a latent-CKA expert-diversity term; residual decorrelation no longer gates (`lambda_dec` 0, the D-008 machinery stays as optional genes). Checklist item 2 reads usage K_eff, latent CKA, SSL skill and min expert skill at the D-102 thresholds; 12/12 synthetic runs pass and the live 1m generation passes too. → [32](32-expert-spaces.md)
3. **Decoder → encoder feedback path:** *implemented* through the D-009 feedback API (D-068) and `EliteFeedback` over elite decoder cards, `FrozenProbeFeedback` as fallback (D-073); the statistical family uses a hyper-parameter search stand-in (D-065). → [fa-decoders](../crates/fa-decoders/README.md#decoder-population-lasting-content-of-the-retired-spec-34) (spec 34 retired), [trainer](../bins/trainer/README.md#training-lifecycle-lasting-content-of-the-retired-spec-36) (spec 36 retired)
4. **Pure-Rust media feasibility:** training runs on **candle**, not burn (D-070, departs from ADR 0003). Media towers are a contract with a hashed-embedding fallback; pretrained weights are not fetched (D-010). Latent diffusion is restricted to day/week (D-073). Media front-ends with pretrained towers and the GPU spike are out of scope (external weights, no hardware); spec 31 Q1 decided (b)+(c). → [fa-encoders](../crates/fa-encoders/README.md#encoder-model-lasting-content-of-the-retired-spec-31) (spec 31 retired)

## ADRs
| ADR | Decision | Status |
|---|---|---|
| [0001](adr/0001-rust-go-split.md) | Rust core, Go for parallel services | accepted |
| [0002](adr/0002-helixdb-hub.md) | HelixDB as graph/vector hub | superseded: vectors by 0010 (D-101), graph by 0010/0011 (D-108, D-110, D-112) |
| [0003](adr/0003-pure-rust-ml.md) | Pure-Rust ML (burn + candle); implemented on candle only (D-070) | accepted |
| [0004](adr/0004-nats-jetstream.md) | NATS JetStream bus + work queues | accepted (D-001) |
| [0005](adr/0005-protobuf-contracts.md) | Protobuf contracts | accepted (D-001) |
| [0006](adr/0006-local-only-no-auth.md) | Local-only, no auth | accepted |
| [0007](adr/0007-victoriametrics.md) | VictoriaMetrics for time series | accepted |
| [0008](adr/0008-postgres-warehouse.md) | PostgreSQL warehouse + ledger | accepted |
| [0009](adr/0009-garage-s3.md) | Garage S3 for blobs | accepted |
| [0010](adr/0010-vector-and-graph-store.md) | Qdrant for vectors, PostgreSQL for the model-state graph | vectors accepted (D-101); graph half back in force after the D-112 rollback of 0011 |
| [0011](adr/0011-dgraph-graph-store.md) | Dgraph as the single graph store | **rolled back** 2026-11-16 (D-112): the copy was faithful, the store could not stay up under 898 k version rows; the graph is in Postgres again and the decision is reopened |
| [0012](adr/0012-llm-suite-and-mcp-boundary.md) | The LLM suite, and what an agent inside this platform may do | accepted (D-117, D-119, D-120): the agent acts only through control-api, mutations need per-call approval, and an external MCP server grants it no new authority |

## Decision traceability checklist
Every design decision made so far, and the spec that records it.

- [ ] Regions = regions of latent space, family-specific discovery → 30 (retired → fa-patterns README)
- [ ] MCMC = Markov state graph + Bayesian kernels, pluggable samplers → 33 (retired → fa-mcmc README)
- [ ] Learned MoE router, emergent expert count → 32
- [ ] Decorrelate next-step prediction residuals → 32 (superseded by D-099: end-to-end reconstruction + CKA expert diversity)
- [ ] Multi-horizon, matched scope → 00, 34, 35 (retired → fa-strategies README)
- [ ] Media: text, video (frames + ASR), images/charts, audio → 10 (retired → ingestor README; media capture out of scope: external), 31 (retired → fa-encoders README; media front-ends out of scope: external weights)
- [ ] All encoder families compete; one model per timescale → 31 (retired → fa-encoders README)
- [ ] Generations + online adapters; gen 0 self-supervised → 36
- [ ] Asset selection learned / evolved / MCMC-prior-driven → 31 (retired → fa-encoders README), 33, 35 (retired → fa-strategies README)
- [ ] Decoder families (visual ×3, retrieval with time distance, text ×3, RNN-prior) → 34
- [ ] ES + SGD; Pareto fitness; budget-driven population → 34
- [ ] Decoder fitness shapes encoders → 34, 36
- [ ] GP + neural strategies compete; GP primitives = graph ops, decoder outputs, portfolio state → 35 (retired → fa-strategies README)
- [ ] Neural policy outputs an exposure distribution → 35 (retired → fa-strategies README)
- [ ] Pluggable meta-allocator, pluggable exec algos → 40
- [ ] Long + short, leverage cap per asset class → 40
- [ ] Hard limits incl. liquidity cap net of payout quota → 40
- [ ] Promotion = min shadow time + stats → 40, 41
- [ ] Locked holdout, deflated Sharpe/PBO, shadow; no walk-forward in evolution → 41
- [ ] Canonical proto events, three timestamps, raw archive → 10 (retired → ingestor README), 21
- [ ] Internal UUID + FIGI/ISIN/LEI → 12 (retired → fa-warehouse README)
- [ ] Dgraph / Qdrant / VictoriaMetrics / Postgres / Garage responsibilities → 20, ADR 0011, ADR 0010
- [ ] NATS work queues for scaling; GPU priority queue → 21, 60
- [ ] Backfill depth per timescale → 10
- [ ] EUR base currency → 00, 42 (retired → fa-ledger README)
- [ ] Frontend v1: ops, latent/graph explorer, evolution lab, trading desk → 51
