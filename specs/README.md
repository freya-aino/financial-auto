# Specifications

Specs were the source of truth until code existed; now the code is, and the specs describe it. Each one follows [`_template.md`](_template.md). Stack choices are recorded as ADRs in [`adr/`](adr/). Implementation decisions (D-001..D-084) are logged in [`DECISIONS.md`](DECISIONS.md) and folded into the specs they affect.

**Retirement:** a feature spec is removed once the code meets every in-scope acceptance criterion and its open questions are decided; its lasting reference content (data model, tunables, invariants) moves into the owning component README first. Out-of-scope items (live broker, keyed sources, off-machine backup, GPU hardware) do not block retirement. As of 2026-11-11 no feature spec qualifies; each lists what keeps it at the top of its §9 ("Remaining"). Overview documents (00–02), ADRs and the decision log are not retired.

**Workflow:** draft → review → accepted. Specs are refined one at a time; the ML core (30–33) came first because it carries the most risk. Every spec is still `draft` (no owner review yet), and most D-entries are `provisional`.

## Index & maturity

**High** = can be accepted as written. **Medium** = structure settled, key definitions or numbers missing. **Low** = intent settled, needs another design round before it can be implemented.

| Spec | Status | Maturity | Main open items (feature specs: what keeps them from retirement) |
|---|---|---|---|
| [00 Vision & scope](00-vision-and-scope.md) | draft | High | Success metrics, starting capital |
| [01 Architecture](01-architecture.md) | draft | High | None blocking (one runtime process, D-005); multi-machine topology deferred |
| [02 Glossary](02-glossary.md) | draft | Medium | "Complex indicator"; exact wording for space / region / state |
| [10 Ingestion](10-ingestion.md) | draft | Medium-high | Media capture (no connector, no `MediaChunk` producer), 24 h rate-limit soak, gap-fill, backfill lanes, NATS-outage disk buffer, clock-skew monitor, `ctl.ingest.reload`; Q1–Q6 |
| [11 Source catalog](11-source-catalog.md) | draft | Medium | PATCH for rate limits/config, reload-on-change and auto-disable, ccxt/SDMX/RSS adapters, fixed-id catalog test, facts of 62 planned sources; Q1–Q5 |
| [12 Reference data](12-reference-data.md) | draft | Medium | OpenFIGI/CoinGecko lookup, intraday FX, scheduled calendar refresh, corporate-action feed, materialised adjusted series, `IN_SECTOR`; Q1–Q5 |
| [20 Storage](20-storage.md) | draft | Medium | VictoriaMetrics series budget and per-timescale retention, disk-budget tightening, genome-weight GC, VM as-of, Helix k-NN benchmark at 10 M, VM rebuild drill; Q1–Q4 |
| [21 Bus & contracts](21-bus-and-contracts.md) | draft | Medium-high | Crash exactly-once and two-worker throughput tests, durable consumers with backpressure / quote coalescing, proto major-version check, `buf breaking` in CI; Q2–Q4 |
| [30 Latent regions](30-latent-regions.md) | draft | High | Region-collapse guard, Helix write buffering/replay, predictive coherence outside the FSQ merge, `region_weight` series and episodes; Q2, Q5, Q6 |
| [31 Expert encoders](31-expert-encoders.md) | draft | Low-medium | Media front-ends and fusion (no pretrained towers run), asset embeddings / cross-asset attention, per-timescale sizing beyond context, collapse monitor, OOM retry, GPU spike; all Qs |
| [32 Expert spaces](32-expert-spaces.md) | draft | Low | **No snapshot meets the checklist on live data** (§6.3.1); bias balancing, expert split, `cov_ema`, PIT/CRPS calibration, expert VM series; Q1, Q3, Q5 |
| [33 MCMC graph](33-mcmc-graph.md) | draft | High | Helix-outage observation log and replay, support-based edge pruning, cross-asset priors at universe scale (O(n²)); Q5, Q6 |
| [34 Decoder population](34-decoder-population.md) | draft | Medium-high | Text/chart decoders below naive on live data, `visual.frame` / `text.summary` decoders, hypervolume rung scores / NSGA-III, elite re-evaluation, `READS`/`EVALUATED_ON`/`ParetoFront` rows; Q2, Q4, Q6 |
| [35 Strategies](35-strategies.md) | draft | Medium-low | Neural policies (trait only), asset-set genes, CVaR/diversification objectives, DSL subset gaps, deflated-Sharpe reference test, parallel old-snapshot serving; Q2, Q3 |
| [36 Training lifecycle](36-training-lifecycle.md) | draft | Medium | Generation state machine, Pareto encoder selection, gen-0 bootstrap, checklist items 4–5, adapter restore on rollback, persistent shadows, continuous adapters, run manifest, holdout check in training, auto-rollback; Q2–Q4 |
| [40 Execution & risk](40-execution-and-risk.md) | draft | Medium-high | Randomised risk-chain property test, paper borrow/funding accrual, persisted allocation snapshots / targets / promotion reviews, session-based daily-loss day; Q3, Q4 (live adapter out of scope) |
| [41 Backtest & validation](41-backtest-and-validation.md) | draft | Medium-high | Full-day live-replay equivalence, missing-data coverage cut, holdout-access audit, stale-cost-model alert; Q1, Q3, Q5 |
| [42 Ledger & accounting](42-ledger-and-accounting.md) | draft | High | Tax reporting (Q2), intraday FX, per-currency stale-FX block; the pending FX restatement keeps criterion 6 failing on `paper` (operator action) |
| [50 Control API](50-control-api.md) | draft | Medium | Audit rows for generation start, job cancel and ingest start/stop; `/api/metrics/query`; Q2 (`/api/v1` prefix) |
| [51 Frontend](51-frontend.md) | draft | Medium | 50k-point pan/zoom charts (Q1), service control, lineage tree, pre-submit risk check, 3-D latent explorer, per-subject sparklines, API-only routes without UI |
| [60 Ops](60-ops.md) | draft | Medium | Secret scanner / pre-commit hook, JSON logs and log store (Q3), OTel tracing, evo-worker checkpoint-resume drill, raw-media disk budget (Q5) |

## Core blockers (status after round 7)
1. **Region identity across retraining:** every generation shifts the latent space. *Implemented:* probe-set alignment with Hungarian matching, lineage ids, split/merge/death and kernel transfer (D-007, D-069); anchored training for the neural families (D-070). → [30](30-latent-regions.md), [33](33-mcmc-graph.md)
2. **Expert-space loss:** *implemented* as D-008 in `fa_expert_space::moe` (D-070), with a constrained head solver (D-080). Open: no snapshot meets the promotion checklist on live data; it decorrelates in sample but not on validation. → [32](32-expert-spaces.md)
3. **Decoder → encoder feedback path:** *implemented* through the D-009 feedback API (D-068) and `EliteFeedback` over elite decoder cards, `FrozenProbeFeedback` as fallback (D-073); the statistical family uses a hyper-parameter search stand-in (D-065). → [34](34-decoder-population.md), [36](36-training-lifecycle.md)
4. **Pure-Rust media feasibility:** training runs on **candle**, not burn (D-070, departs from ADR 0003). Media towers are a contract with a hashed-embedding fallback; pretrained weights are not fetched (D-010). Latent diffusion is restricted to day/week (D-073). Open: a GPU spike (the host has none). → [31](31-expert-encoders.md)

## ADRs
| ADR | Decision | Status |
|---|---|---|
| [0001](adr/0001-rust-go-split.md) | Rust core, Go for parallel services | accepted |
| [0002](adr/0002-helixdb-hub.md) | HelixDB as graph/vector hub | accepted |
| [0003](adr/0003-pure-rust-ml.md) | Pure-Rust ML (burn + candle); implemented on candle only (D-070) | accepted |
| [0004](adr/0004-nats-jetstream.md) | NATS JetStream bus + work queues | accepted (D-001) |
| [0005](adr/0005-protobuf-contracts.md) | Protobuf contracts | accepted (D-001) |
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
