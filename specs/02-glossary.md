# 02 — Glossary

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium |
| **Owners (code)** | — |
| **Depends on** | all |

Terms are binding across the specs. If a spec uses a term differently, the spec is wrong.

| Term | Definition |
|---|---|
| **Timescale** | A resolution level of the data and models (seconds, minutes, hours, days, weeks). Each timescale has its own custom-sized encoder. |
| **Horizon** | How far ahead a decoder or strategy predicts or holds. With matched scope, horizon ≈ the lookback scale of its timescale (1h → 2h). |
| **Canonical event** | Normalized protobuf message produced by ingestion (Trade, Quote, Bar, BookSnapshot, NewsItem, MediaChunk, MacroObservation, OnchainEvent, FilingEvent). |
| **Encoder family** | One modern VAE-type architecture (hybrid FSQ/VQ+continuous, hierarchical VAE + flow prior, JEPA + VAE head, latent diffusion). Families compete. |
| **Expert encoder** | A trained model of one family at one timescale, which maps the multimodal input through the MoE router into expert spaces. |
| **MoE router** | Learned gating that assigns inputs to experts; the number of experts emerges from training. |
| **Expert space** | The latent manifold of one expert. Spaces are trained so their **next-step prediction residuals** are mutually ~uncorrelated while each stays decodable by many decoders. |
| **Expert view** | The contract a decoder or strategy receives: snapshot id, timescale, expert subset mask, latent + uncertainty, current region / MCMC state ([32](32-expert-spaces.md)). |
| **Latent region** | A meaningful area of an expert space (basin, attractor, regime). The discovery method depends on the encoder family. Regions are the nodes of the Markov state graph. *(Formerly "regional pattern".)* |
| **Complex indicator** | The whole landscape strategies traverse: latent regions, their transition kernels and decoder outputs. It replaces classic technical indicators. *(Definition still open, see [35](35-strategies.md).)* |
| **Markov state graph / MCMC graph** | Graph of latent regions with Bayesian transition kernels; the kernels are updated and sampled by pluggable samplers (SMC, HMC/NUTS, …). |
| **Sampler** | A pluggable MCMC/SMC implementation; samplers compete on calibration and cost. |
| **Decoder** | A model that reconstructs or predicts from expert views: visual, retrieval (analog search), text, or RNN-prior. |
| **RNN-prior decoder** | Decoder that seeds an RNN from the expert view of one asset and unrolls a timescale-scaled horizon for a different target asset. |
| **Genome** | Evolvable description of a decoder or strategy (topology, hyperparameters, expert subset, assets, horizon). |
| **ES + SGD** | Hybrid evolution: topology evolves, while weights are warm-started from parents and briefly gradient-trained. |
| **Pareto fitness** | Multi-objective fitness: reconstruction, forecast skill, trading value, compute cost; selection is NSGA-style. |
| **Elite** | Candidates on or near the Pareto front. Elite decoders' losses feed into the next encoder generation. |
| **Generation (encoder)** | One retraining cycle that produces a frozen backbone **snapshot**. |
| **Online adapter** | A small LoRA-like module updated on the live stream on top of a frozen snapshot. |
| **Strategy** | A GP program or neural policy that emits a distribution over target exposure for its asset set and horizon. |
| **Meta-allocator** | Pluggable layer that turns strategy exposure distributions into capital allocation, with cross-strategy netting. |
| **Hard limit** | A non-evolvable risk constraint enforced before every order ([40](40-execution-and-risk.md)). |
| **Payout quota** | Capital reserved daily for withdrawals; deployable capital = equity − quota. |
| **Locked holdout** | A time period never used for selection, used only for promotion gates. |
| **Shadow (paper shadow)** | Running a candidate on live data with simulated fills before promotion. |
| **Deflated Sharpe / PBO** | Sharpe ratio corrected for multiple testing / Probability of Backtest Overfitting. |
| **Point-in-time** | Using only data whose `publish_time` / availability time is at or before the decision time. |
