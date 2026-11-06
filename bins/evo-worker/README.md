# evo-worker

Stateless worker evaluating decoder/strategy genomes pulled from NATS work queues.

**Status:** skeleton running — JetStream work-queue consumer on `jobs.evo.>`, replies FAILED/not-implemented, heartbeat.

## Owns
Job handler for `jobs.evo.*`.

## Governing specs
- [34-decoder-population](../../specs/34-decoder-population.md)
- [35-strategies](../../specs/35-strategies.md)
- [60-ops](../../specs/60-ops.md)
