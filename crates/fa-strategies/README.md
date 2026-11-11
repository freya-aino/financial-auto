# fa-strategies

Strategy population: GP programs and neural policies producing exposure distributions.

**Status:** partial (the GP side implements the spec 35 §4.2 graph / decoder primitives, D-063, but not `exposure_gross`, `drawdown`, `clip`, `Bool` comparisons or `prob_gt`; neural policies and asset-set genes are pending) — GP DSL AST + interpreter, `GpStrategy`, `Strategy`/`NeuralPolicy` traits, proto conversion; S-expression program syntax (`sexpr`) with the typed spec 35 §4.2 graph/decoder primitives; `strategies.toml` population files and the hand-written **baseline** set (`baselines.toml`, not evolved); the live observation `features::LiveContext` (real latent graph + decoder outputs with price-only fallbacks and per-source counters); the point-in-time latent graph `graph::LatentGraph` over `fa_mcmc::MarkovGraph`; the decoder-output cache `decoders::DecoderOutputs`; the **strategy genome** (`genome`), **typed random program generation** (`random`) and the **GP variation operators** (`ops`) used by `evo-worker` for `evo.vary.strategy`; snapshot-bound **region literals** and their remapping through region lineage on an encoder swap (`remap`, spec 35 §6.5) (all tested). Neural policies are pending (`NeuralPolicy` is a trait without an implementation).

## DSL (`sexpr`)
Grammar `atom | (op arg...)`; operators `add sub mul div neg tanh min max if_gt`.

| Type | Forms |
|---|---|
| `Region` | `(cur_region expert)`, literal `(region expert ordinal)` (region `<prefix>-e<expert>-r<nnn>` of the strategy's bound `snapshot`) |
| `RegionSet` | `(neighbors r n)` |
| `[Path]` | `(sample_walk r k n)` |
| `Num` (graph, spec 33) | `stay_prob`, `(path_up k)`, `region_entropy`, `(p_trans a b k)`, `(p_to_set r S k)`, `(hitting_time r S)`, `(dwell r)`, `(novelty expert)`, `(region_feature r mass\|mean_fwd_return\|vol)`, `(path_value (sample_walk r k n) up\|mass\|mean_fwd_return\|vol)` |
| `Num` (decoders, spec 34) | `(quantile q)`, `analog_forecast`, `(score family)`, `(fc_quantile q k)`, `(fc_dist k)`, `(analog_outcome k)`, `analog_age`, `(text_score topic)`, `(visual_score kind)` |
| `Num` (portfolio) | `position`, `unrealised_pnl`, `time_in_trade` |

Typed forms are only valid as arguments (`(add (cur_region 0) 1)` is a `ParseError::Type`); arities are checked. The asset of the decoder primitives is the instrument being evaluated. `(score family)` takes dotted family names (`(score text.news)`); `(text_score topic)` / `(visual_score kind)` take a bare word. `Display` round-trips (`parse(&e.to_string()) == e`).

Evaluation is total: a non-finite node value becomes 0. `fa_strategies::take_non_finite()` returns (and resets) how many such replacements happened on the calling thread, which the runtime reads around a step to apply spec 35 §8 (NaN → 0 exposure, repeated → auto-pause).

## Live observation (`features::LiveContext`)
Wraps the price-only `FallbackContext` and, when given, a `graph::GraphAt` view (one timescale, fixed at the step time) and a `decoders::DecodersAt` view (one instrument, recency window). Each primitive answers from the first source with data and counts the source (`Sources { graph, decoder, fallback }`), which `bins/strategy-runtime` sums into its heartbeat (`primitives.graph|decoder|fallback`); `primitive_sources()` names the primitives per source (the runtime's `starved` flag). `with_bar(ts)` sets the unit of decoder `k` arguments. `analog_outcome` reports the analogs it could not price (`take_analog_requests()`) and those it priced (`take_analog_resolved()`) so a caller can prefetch prices. Point-in-time: views only see messages with event time ≤ the step. The fallback table is in the module docs.

## Latent graph (`graph::LatentGraph`)
Fed with `LatentState` (per-instrument region per expert, posterior std, history → `cur_region`, `dwell`, `novelty`, occupancy), `RegionEvent` (registry: lineage, `live|merged|split|retired`), `TransitionUpdate` (bounded **versions** of each row's posterior mean + 90 % interval) and realised bar closes (`record_close` → per-region return statistics for `region_feature`). `GraphAt` rebuilds each expert space's kernel as an `fa_mcmc::MarkovGraph` from the summaries as of the view time (`set_row_from_summary` recovers the Dirichlet concentration from the interval), so `p_trans(a, b, k)` is the matrix power of the posterior-mean kernel (`TransitionMatrix::k_step`), `neighbors` ranks by mass, `sample_walk` / `hitting_time` / `path_value` use posterior-predictive walks seeded from `(timescale, expert, region, k, n, at_ms)` (deterministic per step).

## Decoder outputs (`decoders::DecoderOutputs`)
`decoder.output.<family>` keyed by `(exact family, instrument)` (instrument from `output_refs` `inst:<uuid>`; `inst:*` or no `inst:` ref = market-wide), bounded, event-time ordered; every query is point-in-time (`as_of ≤ t`) within a recency window, and a fresh instrument-specific output always wins over a market-wide one.

| Primitive | Source |
|---|---|
| `fc_quantile(q, k)`, `fc_dist(k)` | interpolated `rnn_prior` quantile vector (levels `0.05 0.25 0.5 0.75 0.95` for five values) / its mean, of the fresh output whose **horizon** is closest to `k` (ties → latest) |
| `analog_outcome(k)` | similarity-weighted realised `k`-bar log return after each `retrieval` analog's `window_end`, from a caller `PriceLookup`; only outcome windows completed at or before `t` count; `analog_outcome_detail` lists resolved / pending / unpriced analogs |
| `analog_age` | mean `time_distance_days` |
| `score(family)` | freshness `1 − age / max_age` of the latest output of that exact family |
| `text_score(topic)` / `visual_score(kind)` | `quantiles[0]` (clamped to `[-1, 1]`) of the latest fresh `text.<topic>` / `visual.<kind>` output (model-runtime: `text.news`, `text.summary`, `text.narrative`, `visual.chart`, `visual.future`); `None` → fallback 0 |

**Horizon.** `DecoderOutput` has no horizon field. `Output::horizon` is the `output_refs` entry `h:<steps>`, else `h=<n>` / `"h": n` in a `genome:<text>` ref or in `genome_id` when that is a decoder genome text (`forecast:h=3;lb=8`). evo-worker writes `output_refs = [inst:<uuid>, job:<id>, h:<horizon bars>]` (D-064) and model-runtime's neural serving adds `h:<horizon>` too, so the horizon is exact for both producers; with no known horizon the latest output wins (the pre-D-063 behaviour).

## Baselines (`baselines.toml`)
Hand-written, `genome_id = "baseline"`: momentum / reversion / path skew / quantile asymmetry / regime (price-era names), two graph programs (`baseline-graph-neighbors-1s`, `baseline-graph-regime-1m`) and two decoder-score programs: `baseline-text-visual-1s` (`(mul 3 (mul (text_score news) (visual_score future)))`, uncertainty `1 − (score text.news)`) and `baseline-narrative-chart-1m` (narrative + summary consensus gated by the chart score's sign). All evaluate in tests on price-only, synthetic-graph and synthetic text/visual decoder inputs.

## Genome, random programs, operators (`genome`, `random`, `ops`)
`genome::StrategyGenome` = spec genes (`timescale`, `steps`, `lookback_bars`, `step_secs`, optional `snapshot`) + the two programs + lineage (`parents`, `generation`). Its id is `gp-<16 hex>` of the phenotype (same programs and spec → same id, whatever the lineage; a `snapshot` binding is part of the phenotype, unbound genomes keep their ids). **Text form** = a `strategies.toml` entry: `to_string()` renders one TOML inline table on a single line (`{ id = "gp-…", genome_id = "gp-…", timescale = "1s", steps = 5, lookback_bars = 32, signal = "…", uncertainty = "…" }`) so genomes travel in the newline-separated `genomes` of `ctl.evo.start`; `parse` accepts that, a bare `key = value` body or a `[[strategy]]` entry; `to_toml_entry` / `population_toml` write the multi-line file the promoted `strategies.<generation>.toml` artifact uses. `parse(&g.to_string()).id() == g.id()`.

`random::RandomSpec` (depth / node limits, ERC range, `max_k`, experts, score families, `extended` = include the spec 35 §4.2 primitives) drives grow-method generation: every node is `Num`-typed, literal arguments (`k`, `q` from `LEVELS`, expert index, feature names, family / topic words) are drawn from their own domains, so every program parses back through `sexpr`. `random_genome` / `random_population(seed, n, timescale, spec)` produce immigrants.

`ops`: pre-order tree helpers (`subtree`, `replace`, `rebuild`, `depth`), `subtree_crossover` (node-limited), `point_mutation` (same-arity swap), `subtree_mutation`, `hoist_mutation`, `constant_jitter` (ERCs × `1 + σN`, quantile levels and `k` steps move to neighbours), `mutate_genome` (one tree mutation or a spec-gene mutation, lineage set) and `crossover_genome`. `OpKind::as_str` names the operator for Helix `ParentOf{op}` edges.

## Region remapping (`remap`)
`LineageBook` collects model-runtime's snapshot-swap lineage (`RegionLifecycleEvent` on `latent.lifecycle.<ts>`, `reason = snapshot`: ancestors `carry|split|merge|partial` with probe overlap; unmatched parents die). `remap_def` rewrites a bound strategy's literals into the new snapshot: a `carry` successor, else the successor with the largest overlap; no descendant or no observed lineage → unmappable (`RemapReport`). strategy-runtime swaps a fully mapped strategy in place and suspends an unmappable one; the risk-engine pipeline sends a remapped active strategy back to shadow. Random generation does not emit literals.

## Owns
GP DSL interpreter, live observation, latent graph query surface, decoder cache, strategy genome + random generation + variation operators, region-literal remapping, `Strategy` / `NeuralPolicy` traits.

## Governing specs
- [35-strategies](../../specs/35-strategies.md), [33-mcmc-graph](../../specs/33-mcmc-graph.md) §4.2, [34-decoder-population](../../specs/34-decoder-population.md)
