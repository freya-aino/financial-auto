# fa-decoders

Decoder families: visual, retrieval, text, RNN-prior.

**Status:** partial — `Decoder` trait and family enum; CRPS (ensemble + quantile), pinball loss, time-discounted analog weighting; **baseline genome encoding** (`genome`: parse, canonical `Display`, numeric `params()` / `with_params()` for evolution) and two **CPU point-in-time series decoders** (`baseline`: empirical quantile forecaster, analog retrieval with leakage guard and optional **latent conditioning** through `DecodeCtx`), all tested. No ML-backend models; `ExpertView` conditioning is limited to the latent-similarity term below.

## Genome format (`genome::DecoderGenome::parse`)
| Form | Example | Notes |
|---|---|---|
| key/value | `forecast:h=5;lb=32;kernel=16;levels=0.1/0.5/0.9` · `retrieval:k=20;tau=30;gap=5;metric=cos` | genes separated by `;` (or space). Commas work only when the string reaches the worker intact; the `ctl.evo.start` transport splits genome lists on `,`. |
| JSON | `{"id":"g1","family":"retrieval","params":{"k":20,"levels":[0.2,0.8],"ts":"1m"}}` | only via `Job.spec` (contains commas) |
| bare id | `g1`, a ULID | deterministic genome derived from an FNV hash of the id: family by parity, genes jittered around defaults. A bare string containing `=` is rejected as a split fragment. |

Families: `forecast` (aliases `quantile`, `rnn_prior`; published on `decoder.output.rnn_prior`) and `retrieval` (`analog`). Common genes `ts` (timescale, default `1s`), `h` horizon, `lb` lookback, `levels`, `inst`. Forecast: `kernel` (EW half-life, steps), `scale`, `shrink`. Retrieval: `k`, `tau` (days), `gap` (≥ `h`, leakage guard), `metric` (`cos`|`euclid`), `lat` (latent-similarity weight in [0, 1], default 0.5).

### Canonical form and evolution hooks
- `DecoderGenome::to_string()` (`Display`) prints the canonical kv form with **every** gene — `forecast:ts=1s;h=3;lb=16;levels=0.05/0.25/0.5/0.75/0.95;kernel=16;scale=1;shrink=0.5;k=20;tau=30;gap=5;metric=cos;lat=0.5` — using shortest round-trip floats; `parse(g.to_string())` yields an equal genome (tested, including derived ones). Children produced by `evo.vary.decoder` travel in this form.
- `params()` → `BTreeMap<name, f64>` of the numeric genes (`NUMERIC_GENES`: `h lb kernel scale shrink k tau gap lat`); `with_params(&map)` applies them (rounding integers, clamping into the valid ranges, `gap ≥ h`) and sets `id` to the canonical string, so mutated children are always evaluable.

## Baseline decoders (`baseline`)
- `Series::resample` builds a log-price series from `(ts_ms, price)` trades (last price per timescale bucket).
- `QuantileForecaster`: EW empirical distribution of past overlapping `h`-step returns, shrunk drift, `scale` spread.
- `AnalogRetriever`: z-scored return-shape query, candidates whose outcome window ends ≥ `gap` before now, top-`k` by `sim·exp(−Δt/τ)`, weighted outcome quantiles + `Analog` list.
- `SeriesDecoder::forecast(series, t)` reads only `series[..=t]` (tested).
- `build(genome)` / `build_with(genome, Arc<DecodeCtx>)`: the context holds `LatentPoint`s (`as_of_ms`, concatenated expert means of a `LatentState`) per instrument token. When latents exist for the series' instrument, the retriever's similarity becomes `(1 − lat)·shape + lat·cos(latent as of query, latent as of candidate)`; both lookups are **as of** the respective window end (`latent_as_of`), so a later latent never leaks in (tested: truncating the future leaves the forecast unchanged; other instruments' latents are ignored). An empty context reproduces the unconditioned decoder exactly.

## Owns
`Decoder` trait and family implementations; genome encoding of the baseline families; `DecodeCtx` / `LatentPoint` / `instrument_token` / `cosine` conditioning primitives.

## Governing specs
- [34-decoder-population](../../specs/34-decoder-population.md)
