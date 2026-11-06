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
- Ingestion from every API we can access (brokers, exchanges, market data, FX, macro/statistics, on-chain, filings, news, broadcast video/audio). See [11](11-source-catalog.md).
- Pure-Rust ML stack: expert encoders, expert spaces, latent regions, MCMC graph, decoders, strategies.
- Evolutionary search over decoders and strategies under multi-objective Pareto fitness.
- Paper trading, gated promotion to live trading, meta-allocation, hard risk limits, ledger.
- A local Svelte operator UI for running the whole system by hand.

**Out of scope (for now):**
- Authentication, multi-user access, remote deployment hardening ([ADR 0006](adr/0006-local-only-no-auth.md)).
- Tax reporting (open question in [42](42-ledger-and-accounting.md)).
- Sub-second / HFT latency targets.

## 3. Decisions
- **Assets:** everything reachable via API; the **asset universe is selected by the model** ([31](31-expert-encoders.md), [33](33-mcmc-graph.md), [35](35-strategies.md)).
- **Horizons:** multiple; each strategy has its own, with **matched scope**: lookback scale ≈ forecast scale (1h → 2h, 1d → 2d).
- **Timescales:** one custom-sized model per timescale (seconds, minutes, hours, days, weeks…).
- **Base currency:** EUR for P&L, risk and fitness.
- **Path to money:** paper first; promotion requires minimum shadow calendar time plus statistical criteria ([40](40-execution-and-risk.md), [41](41-backtest-and-validation.md)).
- **Languages:** Rust for most of the system including ML; Go for parallel/scalable services ([ADR 0001](adr/0001-rust-go-split.md)).
- **Compute:** one GPU workstation today. Everything must scale horizontally later without redesign ([60](60-ops.md)).

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
| `timescale_ladder` | 1s, 1m, 1h, 1d, 1w | see [31](31-expert-encoders.md) |
| `initial_live_capital` | TBD | open question |

## 8. Failure modes
System-level halts are owned by [40](40-execution-and-risk.md) and [60](60-ops.md).

## 9. Acceptance criteria
- Every decision above is traceable to a component spec (checklist in [README](README.md)).
- The system can run end to end in paper mode on a single GPU workstation.

## 10. Open questions
1. **Success metrics:** target return, max acceptable drawdown, starting capital. Blocks the tunables in 40.
2. **Latency budget per timescale.** Blocks the process split in 01.

## 11. Deferred
- Scaling beyond one workstation (the design stays open for it).
- Keyword/web-search-driven media discovery.
