# 00 — Vision and Scope

| | |
|---|---|
| **Status** | draft |
| **Maturity** | High |
| **Owners (code)** | whole repository |
| **Depends on** | — |

## 1. Purpose
Build a locally operated, largely autonomous trading platform. It learns a shared multimodal representation of markets (prices, macro, on-chain, news, broadcast media), models how that representation moves as a Bayesian Markov graph, and evolves large populations of decoders and strategies over it. Only strategies that survive strict validation reach real money.

## 2. Scope
**In scope:**
- Ingestion from every API we can access (brokers, exchanges, market data, FX, macro/statistics, on-chain, filings, news, broadcast video/audio). See [Source catalog reference (ex spec 11)](../go/pkg/catalog/README.md#source-catalog-reference).
- Pure-Rust ML stack: expert encoders, expert spaces, latent regions, MCMC graph, decoders, strategies.
- Evolutionary search over decoders and strategies under multi-objective Pareto fitness.
- Paper trading, gated promotion to live trading, meta-allocation, hard risk limits, ledger.
- A local Svelte operator UI for running the whole system by hand.

**Out of scope (for now):**
- Authentication, multi-user access, remote deployment hardening ([ADR 0006](adr/0006-local-only-no-auth.md)).
- Jurisdiction-specific tax reporting (the jurisdiction-neutral realised-gains report exists; jurisdiction and rules are the user's decision: [42, retired → fa-ledger](../crates/fa-ledger/README.md) §10 Q2).
- Sub-second / HFT latency targets.

## 3. Decisions
- **Assets:** everything reachable via API; the **asset universe is selected by the model** ([31, retired → fa-encoders](../crates/fa-encoders/README.md#encoder-model-lasting-content-of-the-retired-spec-31), [33, retired → fa-mcmc](../crates/fa-mcmc/README.md#graph-model-lasting-content-of-the-retired-spec-33), [35](35-strategies.md)).
- **Horizons:** multiple; each strategy has its own, with **matched scope**: lookback scale ≈ forecast scale (1h → 2h, 1d → 2d).
- **Timescales:** one custom-sized model per timescale (seconds, minutes, hours, days, weeks…).
- **Base currency:** EUR for P&L, risk and fitness.
- **Path to money:** paper first; promotion requires minimum shadow calendar time plus statistical criteria ([40](../bins/risk-engine/README.md#execution-and-risk-lasting-content-of-the-retired-spec-40), [41](41-backtest-and-validation.md)).
- **Languages:** Rust for most of the system including ML; Go for parallel/scalable services ([ADR 0001](adr/0001-rust-go-split.md)).
- **Compute:** one GPU workstation as the target. The current development host has no GPU: the neural families train and serve on CPU through candle, with an opt-in `cuda` feature (D-068, D-070). Everything must scale horizontally later without redesign ([60, retired](../docker/README.md#reference-spec-60-ops-retired-2026-11-11)).

## 4. Interfaces
Not applicable: see [01](01-architecture.md).

## 5. Data model
Not applicable: see [20](20-storage.md).

## 6. Behaviour & algorithms
System loop, end to end:
1. **Ingest** canonical events from all sources → NATS, warehouses, archive.
2. **Encode** per timescale into expert spaces via a learned MoE router.
3. **Discover regions** of latent space → nodes of the Markov state graph.
4. **Model transitions** with Bayesian kernels updated online by pluggable samplers.
5. **Decode** expert spaces with an evolving decoder population (visual, retrieval, text, RNN-prior); elite decoders shape the next encoder generation.
6. **Trade**: evolving strategies (GP programs + neural policies) emit exposure distributions → meta-allocator → risk → execution.
7. **Validate & promote**: holdout, deflated Sharpe/PBO, paper shadow, then live.

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `base_currency` | EUR | fixed |
| `timescale_ladder` | 1s, 1m, 1h, 1d, 1w | `timescales` in `config/platform.toml` (D-011); see [31, retired → fa-encoders](../crates/fa-encoders/README.md#encoder-model-lasting-content-of-the-retired-spec-31) |
| `initial_live_capital` | TBD | open question |

## 8. Failure modes
System-level halts are owned by [40](../bins/risk-engine/README.md#execution-and-risk-lasting-content-of-the-retired-spec-40) and [60, retired](../docker/README.md#reference-spec-60-ops-retired-2026-11-11).

## 9. Acceptance criteria
- Every decision above is traceable to a component spec (checklist in [README](README.md)).
- The system can run end to end in paper mode on a single GPU workstation. *Met in paper mode on a CPU-only host: every service runs in compose on live keyless feeds with statistical baselines behind every stage and the neural families on CPU; no live trading.*

## 10. Open questions
1. **Success metrics:** target return, max acceptable drawdown, starting capital. Blocks the tunables in 40.
2. **Latency budget per timescale.** No longer blocks the process split in 01 (one `model-runtime` for all timescales, D-005); measured so far: the MCMC 1m step at p99 1.3–3.6 ms against a 5 ms budget (D-075).

## 11. Deferred
- Scaling beyond one workstation (the design stays open for it).
- Keyword/web-search-driven media discovery.
