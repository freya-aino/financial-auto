# Venue fixtures (D-121)

Response bodies for the shared venue conformance suite (`../../venue_conformance_test.go`).

**These are reference fixtures, not captures.** Each file is the response shape from the venue's
published API reference — Kraken's REST docs, lemon.markets' docs, OANDA's v20 reference — with
`{{PLACEHOLDER}}` fields the stub server fills from the case being run. No live venue has answered
any of them, because no trading credentials exist yet (`[venues.<id>]` in `config/secrets.toml` is
empty).

That distinction matters for what the suite proves. It proves the adapters parse the documented
shapes, map them onto `OrderEvent` correctly, and obey the contract every adapter must obey
(`conformanceCases`): no double-counted fill, one terminal event, idempotent resubmission, fees
with a currency, nothing filled beyond the intent. It does **not** prove any venue actually answers
this way. When the owner provides a key, the first step is to replace these with captures from the
venue's sandbox and re-run the same suite — the cases do not change, only what answers them.
