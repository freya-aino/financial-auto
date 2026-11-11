# vm-rebuild

Rebuilds VictoriaMetrics market series (`fa_trade_*`, `fa_quote_*`, `fa_bar_*`, `fa_book_*`) from the Garage raw archive into a **scratch** VictoriaMetrics and compares them with the live one — the drill behind `scripts/vm-rebuild-drill.sh` (spec 20 §6.6 / criterion 5: VM is not backed up, it is rebuilt from the raw archive).

**Path per source:** `archive.Replay` (the reader behind `ctl.ingest.replay`, index sidecars required) → the connector's own `Normalizer` with fresh in-memory cursors (as the replay does) → `tsdb.Writer.AddEnvelope` (the ingestor's write call, same-millisecond trade VWAP included) → `-target`. Then `/api/v1/export` of both sides for `{src="<source>"}` over the window, diffed per series and timestamp: matched / missing (live only) / extra (rebuilt only) / value mismatches, per metric, with examples.

**Read-only against live:** nothing is published on NATS (unlike `ctl.ingest.replay mode=publish`), nothing is written to Garage; symbols resolve through `refdata.Resolver.Lookup` only (never provisioning) on a Postgres session with `default_transaction_read_only=on`; the live VM is only exported. The tool refuses a `-target` that equals `-compare` or is port 8428 on localhost / `victoriametrics`.

Linked connectors: `kraken`, `yahoo` (the sources that write market series today; add a blank import for others).

| Flag | Default | |
|---|---|---|
| `-sources` | `kraken,yahoo` | |
| `-from` / `-to` | `-2h` / `-10m` | RFC 3339 or negative duration; samples compared by event time in `[from, to)` |
| `-tail` | `5m` | archive payloads received up to this long after `-to` are also replayed (events inside the window arrive later by up to the knowledge delay, e.g. Kraken bars) |
| `-target` | `FA_VM_DRILL_URL` / `http://127.0.0.1:18428` | scratch VM |
| `-compare` | `FA_TSDB_URL` / `http://127.0.0.1:8428` | live VM (read only; empty skips the diff) |
| `-garage`, `-secrets`, `-catalog`, `-postgres` | as the ingestor | `[garage]` credentials from `config/secrets.toml` |
| `-min-match` | `0.99` | exit 2 when matched < this share of live samples or any value differs |
| `-json` | off | machine-readable report |

```sh
scripts/vm-rebuild-drill.sh                      # scratch VM container + this tool + cleanup
go -C go run ./services/vm-rebuild -target http://127.0.0.1:18428 -from -6h   # against your own scratch VM
```

**Known differences against live (2026-11-11 drill):** samples written live in the last seconds before an ingestor restart are missing from the archive (the open minute object is not uploaded on shutdown, and `archive.raw_missing` does not count it); archive objects older than the index sidecars (D-067) are skipped (`no index`); `fa_trade_count` exists only since D-038's writer change, so older windows show it as rebuilt-only.

## Governing specs
- [20-storage](../../../specs/20-storage.md) §6.6, criterion 5; [Ingestion reference (ex spec 10)](../ingestor/README.md#ingestion-reference) §9 (replay); [Ops reference (ex spec 60)](../../../docker/README.md#reference-spec-60-ops-retired-2026-11-11) §6.6
