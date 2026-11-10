# fa-strategies

Strategy population: GP programs and neural policies producing exposure distributions.

**Status:** partial — GP DSL AST + interpreter, `GpStrategy`, `Strategy`/`NeuralPolicy` traits, proto conversion; S-expression program syntax (`sexpr`) with the typed spec 35 §4.2 graph/decoder primitives; `strategies.toml` population files and the hand-written **baseline** set (`baselines.toml`, not evolved); the live observation `features::LiveContext` (real latent graph + decoder outputs with price-only fallbacks and per-source counters); the point-in-time latent graph `graph::LatentGraph` over `fa_mcmc::MarkovGraph`; the decoder-output cache `decoders::DecoderOutputs`; the **strategy genome** (`genome`), **typed random program generation** (`random`) and the **GP variation operators** (`ops`) used by `evo-worker` for `evo.vary.strategy` (all tested). Neural policies are pending.

## DSL (`sexpr`)
Grammar `atom | (op arg...)`; operators `add sub mul div neg tanh min max if_gt`.

| Type | Forms |
|---|---|
| `Region` | `(cur_region expert)` |
| `RegionSet` | `(neighbors r n)` |
| `[Path]` | `(sample_walk r k n)` |
| `Num` (graph, spec 33) | `stay_prob`, `(path_up k)`, `region_entropy`, `(p_trans a b k)`, `(p_to_set r S k)`, `(hitting_time r S)`, `(dwell r)`, `(novelty expert)`, `(region_feature r mass\|mean_fwd_return\|vol)`, `(path_value (sample_walk r k n) up\|mass\|mean_fwd_return\|vol)` |
| `Num` (decoders, spec 34) | `(quantile q)`, `analog_forecast`, `(score family)`, `(fc_quantile q k)`, `(fc_dist k)`, `(analog_outcome k)`, `analog_age`, `(text_score topic)`, `(visual_score kind)` |
| `Num` (portfolio) | `position`, `unrealised_pnl`, `time_in_trade` |

Typed forms are only valid as arguments (`(add (cur_region 0) 1)` is a `ParseError::Type`); arities are checked. The asset of the decoder primitives is the instrument being evaluated. `Display` round-trips (`parse(&e.to_string()) == e`).

## Live observation (`features::LiveContext`)
Wraps the price-only `FallbackContext` and, when given, a `graph::GraphAt` view (one timescale, fixed at the step time) and a `decoders::DecodersAt` view (one instrument, recency window). Each primitive answers from the first source with data and counts the source (`Sources { graph, decoder, fallback }`), which `bins/strategy-runtime` sums into its heartbeat (`primitives.graph|decoder|fallback`). Point-in-time: views only see messages with event time ≤ the step. The fallback table is in the module docs.

## Latent graph (`graph::LatentGraph`)
Fed with `LatentState` (per-instrument region per expert, posterior std, history → `cur_region`, `dwell`, `novelty`, occupancy), `RegionEvent` (registry: lineage, `live|merged|split|retired`), `TransitionUpdate` (bounded **versions** of each row's posterior mean + 90 % interval) and realised bar closes (`record_close` → per-region return statistics for `region_feature`). `GraphAt` rebuilds each expert space's kernel as an `fa_mcmc::MarkovGraph` from the summaries as of the view time (`set_row_from_summary` recovers the Dirichlet concentration from the interval), so `p_trans(a, b, k)` is the matrix power of the posterior-mean kernel (`TransitionMatrix::k_step`), `neighbors` ranks by mass, `sample_walk` / `hitting_time` / `path_value` use posterior-predictive walks seeded from `(timescale, expert, region, k, n, at_ms)` (deterministic per step).

## Decoder outputs (`decoders::DecoderOutputs`)
`decoder.output.<family>` keyed by `(family, instrument)` (instrument from `output_refs` `inst:<uuid>`), bounded, event-time ordered. `fc_quantile` interpolates the latest `rnn_prior` quantile vector (levels `0.05 0.25 0.5 0.75 0.95` for five values), `fc_dist` is its mean, `analog_outcome(k)` is the similarity-weighted realised k-bar return after each `retrieval` analog (needs the caller's price history), `analog_age` the mean time distance, `score(family)` the output's freshness. `text_score` / `visual_score` answer 0 (no `text.*` / `visual.*` producer yet).

## Genome, random programs, operators (`genome`, `random`, `ops`)
`genome::StrategyGenome` = spec genes (`timescale`, `steps`, `lookback_bars`, `step_secs`) + the two programs + lineage (`parents`, `generation`). Its id is `gp-<16 hex>` of the phenotype (same programs and spec → same id, whatever the lineage). **Text form** = a `strategies.toml` entry: `to_string()` renders one TOML inline table on a single line (`{ id = "gp-…", genome_id = "gp-…", timescale = "1s", steps = 5, lookback_bars = 32, signal = "…", uncertainty = "…" }`) so genomes travel in the newline-separated `genomes` of `ctl.evo.start`; `parse` accepts that, a bare `key = value` body or a `[[strategy]]` entry; `to_toml_entry` / `population_toml` write the multi-line file the promoted `strategies.<generation>.toml` artifact uses. `parse(&g.to_string()).id() == g.id()`.

`random::RandomSpec` (depth / node limits, ERC range, `max_k`, experts, score families, `extended` = include the spec 35 §4.2 primitives) drives grow-method generation: every node is `Num`-typed, literal arguments (`k`, `q` from `LEVELS`, expert index, feature names, family / topic words) are drawn from their own domains, so every program parses back through `sexpr`. `random_genome` / `random_population(seed, n, timescale, spec)` produce immigrants.

`ops`: pre-order tree helpers (`subtree`, `replace`, `rebuild`, `depth`), `subtree_crossover` (node-limited), `point_mutation` (same-arity swap), `subtree_mutation`, `hoist_mutation`, `constant_jitter` (ERCs × `1 + σN`, quantile levels and `k` steps move to neighbours), `mutate_genome` (one tree mutation or a spec-gene mutation, lineage set) and `crossover_genome`. `OpKind::as_str` names the operator for Helix `ParentOf{op}` edges.

## Owns
GP DSL interpreter, live observation, latent graph query surface, decoder cache, strategy genome + random generation + variation operators, policy nets, `Strategy` trait.

## Governing specs
- [35-strategies](../../specs/35-strategies.md), [33-mcmc-graph](../../specs/33-mcmc-graph.md) §4.2, [34-decoder-population](../../specs/34-decoder-population.md)
