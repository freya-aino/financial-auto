# fa-decoders

Decoder families: visual, retrieval, text, RNN-prior.

**Status:** partial — `Decoder` trait and family enum; CRPS (ensemble + quantile), pinball loss, time-discounted analog weighting; **baseline genome encoding** (`genome`) and two **CPU point-in-time series decoders** (`baseline`: empirical quantile forecaster, analog retrieval with leakage guard), all tested. No ML-backend models, no `ExpertView` conditioning yet.

## Genome format (`genome::DecoderGenome::parse`)
| Form | Example | Notes |
|---|---|---|
| key/value | `forecast:h=5;lb=32;kernel=16;levels=0.1/0.5/0.9` · `retrieval:k=20;tau=30;gap=5;metric=cos` | genes separated by `;` (or space). Commas work only when the string reaches the worker intact; the `ctl.evo.start` transport splits genome lists on `,`. |
| JSON | `{"id":"g1","family":"retrieval","params":{"k":20,"levels":[0.2,0.8],"ts":"1m"}}` | only via `Job.spec` (contains commas) |
| bare id | `g1`, a ULID | deterministic genome derived from an FNV hash of the id: family by parity, genes jittered around defaults. A bare string containing `=` is rejected as a split fragment. |

Families: `forecast` (aliases `quantile`, `rnn_prior`; published on `decoder.output.rnn_prior`) and `retrieval` (`analog`). Common genes `ts` (timescale, default `1s`), `h` horizon, `lb` lookback, `levels`, `inst`. Forecast: `kernel` (EW half-life, steps), `scale`, `shrink`. Retrieval: `k`, `tau` (days), `gap` (≥ `h`, leakage guard), `metric` (`cos`|`euclid`).

## Baseline decoders (`baseline`)
- `Series::resample` builds a log-price series from `(ts_ms, price)` trades (last price per timescale bucket).
- `QuantileForecaster`: EW empirical distribution of past overlapping `h`-step returns, shrunk drift, `scale` spread.
- `AnalogRetriever`: z-scored return-shape query, candidates whose outcome window ends ≥ `gap` before now, top-`k` by `sim·exp(−Δt/τ)`, weighted outcome quantiles + `Analog` list.
- `SeriesDecoder::forecast(series, t)` reads only `series[..=t]` (tested).

## Owns
`Decoder` trait and family implementations; genome encoding of the baseline families.

## Governing specs
- [34-decoder-population](../../specs/34-decoder-population.md)
