# synthetic

Local multi-instrument market generator for ephemeral test stacks (D-096). `FA_SYNTHETIC_INSTRUMENTS` instruments (default 8) on venue `sim` follow `FA_SYNTHETIC_FACTORS` latent factors (default 3) with calm / volatile regimes redrawn every 40 ticks (the shape of the trainer's `generation::synthetic_bars`). Instrument `k` loads mainly on factor `k mod K`, so the factor structure is recoverable by the encoders and regions. Every `FA_SYNTHETIC_INTERVAL` (default `1s`) it emits a `Trade` and a `Quote` per instrument; every `FA_SYNTHETIC_NEWS_EVERY` ticks (default 60, 0 = off) a templated `NewsItem`. `FA_SYNTHETIC_SEED` makes runs reproducible.

Ids are fixed: instrument `k` is `01900000-0000-7000-8000-<100+k, 12 digits>` (symbol `SYN<k>`, zero-padded to two digits), next to the heartbeat's `…0001`. Like the heartbeat it is not a catalog source; the ingestor runs it as a built-in. Needs no network, so `just eph-up` runs it by default (`FA_CONNECTORS=synthetic,heartbeat`).

**Status:** implemented.

## Governing specs
- [Ingestion reference (ex spec 10)](../../services/ingestor/README.md#ingestion-reference)
- [21-bus-and-contracts](../../../specs/21-bus-and-contracts.md)
