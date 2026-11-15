# Code review — 2026-11-15, whole stack

Reviewed at HEAD `9930172` (79 commits). Read-only survey of all crates, bins, Go services,
frontend, specs and ops scripts (~129k lines Rust, ~34.5k Go, ~17k frontend TS/Svelte), focused on
the five owner questions: internal finance / infrastructure growth as a revenue split, a dashboard
that captures the whole system, cloud entrypoint and bootstrapping, the required tech stack, and
readiness/maturity.

## 1. Verdict

The platform is a broad, well-tested **single-machine paper-trading system** that is close to
feature-complete for that role and deliberately not ready for anything beyond it. Specifically:

- **Readiness.** ~1,035 Rust + ~207 Go test functions, zero TODO/FIXME in source, drills for
  backup/restore/alerts/worker-kill. But: the risk-engine failed its full-day determinism replay
  (20 % match), spec 32 has never been evaluated on live data, live broker execution is a stub,
  and every decision D-001..D-103 is still `provisional` — nothing is owner-confirmed.
- **Internal finance.** The trading book is exact (event-sourced ledger, FIFO lots, FX split,
  per-strategy attribution, payout quota). The system's *own economics* do not exist: no cost of
  compute/storage/data in money, no revenue-split or reinvestment policy, no treasury. §4 proposes
  the design.
- **Dashboard.** Operating the system is well covered (services, streams, sources, evolution,
  trading, ledger). *Observing* the system is not: every store is a TCP up/down badge, backups,
  disk, DLQ and maintenance are visible only when an alert fires, and nothing shows growth over
  time. §5 lists the gaps; most need no new backend.
- **Cloud.** Single-host by explicit design (ADR 0006 localhost-only/no-auth; ex-spec 60 lists
  cloud as out of scope). No k8s/Helm/Terraform/CI/registry anywhere. Bootstrap from nothing is 7
  manual steps with the Garage init host-only. §6 gives the single-host inventory and a phased
  path; none of it requires redesigning the data plane, which is the good news — the bus/work-queue
  architecture scales as intended.

## 2. Required tech stack

| Layer | Component | Version / notes |
|---|---|---|
| Bus | NATS JetStream | 2.12-alpine, single node, 9 streams + work queues, caps Σ 69 GiB |
| Relational | PostgreSQL | 17-alpine; schemas `ledger ref ingest catalog validation ops wh audit`; WAL archiving on |
| Time series | VictoriaMetrics + vmalert | v1.155.0; 50 GiB budget with auto-tightening retention; ~40 alert rules |
| Blobs | Garage S3 | v2.5.0, replication_factor 1; buckets fa-raw/media/models/ckpt/artifacts/backup |
| Graph | HelixDB | v0.0.14 — **replacement under decision** (ADR 0010 graph half open; Postgres/CloudNativePG recommended) |
| Vectors | Qdrant | v1.20.0 (D-101, accepted) |
| Core + ML | Rust | 1.98; candle (CPU; `cuda` feature unverified — no GPU); 19 crates, 6 bins |
| Parallel services | Go | 1.25; ingestor, broker-gateway, evo-orchestrator, healthprobe, vm-rebuild; 13 connectors |
| Contracts | protobuf `fa.v1` | buf lint/breaking (`scripts/buf-check.sh` — the only "CI") |
| UI | SvelteKit 5 | static build served by unprivileged nginx; control-api is the only backend |
| Ops | Docker Compose | profiles default/app/tools; backup sidecar (crond + rclone); nats/postgres exporters |

Everything is local images tagged `:dev`; there is no registry, no CI pipeline, no versioning. A
cloud target adds (at minimum): a registry + CI, k8s manifests/Helm, an operator-managed Postgres
(CloudNativePG fits the open graph-store recommendation), NATS clustering, and an auth/TLS story
(revisiting ADR 0006).

## 3. Readiness and maturity

### 3.1 Component verdicts

| Verdict | Components |
|---|---|
| Mature | fa-core, fa-proto, fa-bus, fa-blob, fa-tsdb, fa-warehouse, fa-metrics, fa-ledger (paper), fa-risk, fa-evolution, fa-patterns, fa-mcmc, fa-strategies, fa-backtest, control-api, strategy-runtime, ingestor, evo-orchestrator, healthprobe, go/pkg/*, proto, keyless connectors |
| Usable | fa-hub (graph decision open), fa-allocator (thin tests for its scope), fa-encoders (CUDA unverified), fa-decoders (text decoders only reach naive skill), model-runtime, evo-worker, trainer (live promotion round pending), vm-rebuild, frontend, broker-gateway (paper) |
| Incomplete | risk-engine (replay determinism, §3.2), fa-expert-space / spec 32 (no live validation), Kraken trade de-dup after restart, lemon/oanda (implemented, never run — no keys) |
| Stub | live broker venue adapters (`paper.go:321` rejects everything but paper) |

Spec status: 12 of 20 retired into READMEs; active 00/01/02/20/21/32/35/41/51 (35 is retirement
bookkeeping only). ADRs 0001–0009 accepted; 0010 split — vectors accepted, graph proposed.

### 3.2 Defects found by this review

Ordered by severity; items 1–2 were already known, the rest are new findings.

1. **risk-engine is not deterministic under Postgres stalls** (spec 41, 2026-11-15). 24 h replay:
   strategy-runtime passed (~43k identical signals/strategy), risk-engine matched 1,037/5,176
   approvals (20 %). Cause: the decision loop awaits a Postgres write per intent
   (`bins/risk-engine/src/main.rs:104-130`, tick at `:1100` with `MissedTickBehavior::Skip`); a
   1.3–3.4 s stall makes a tick keep grid time `t` but act on inputs published after `t`; the
   replay's exact-quantity fill fingerprint then unravels (5,803/6,840 order events unmapped). Fix
   proposed in spec 41 (publish before persisting, drain only inputs ≤ t, quantity-free fill
   mapping, feed halt subjects to the replay) — awaiting owner decision.
2. **Spec 32 unproven on live data.** The D-099/D-102 checklist passes 12/12 on synthetic runs but
   has never been run on real bars; D-103 adds a gap: the trainer still fits on everything in
   VictoriaMetrics, including the synthetic heartbeat instrument.
3. **control-api serves a second, divergent book.** `bins/control-api/src/live.rs:1341-1349`
   applies every `order.event.*` fill at `fx_to_eur: 1.0` ("until FX conversion lands, D-028") into
   its own in-memory `Ledger`, and `GET /api/trading/{positions,orders,fills,risk}` serve *that*,
   not the Postgres ledger. For USD-quoted instruments the dashboard's positions/P&L disagree with
   the authoritative book. The trading desk's headline numbers run through this path.
4. **risk-engine silently falls back to an in-memory paper book** when Postgres is unreachable at
   start (`bins/risk-engine/src/main.rs:537-539`, no retry, 3 s timeout). Under compose
   `depends_on` this is masked; after a host reboot (a recorded incident — the daemon ignores
   ordering) or in k8s it means trading state that resets on restart. Should fail closed or retry.
5. **Kraken connector re-sends trades after an ingestor restart.** `Init` restores the `since`
   time cursor but not the `lastTrade` id filter (`go/connectors/kraken/kraken.go:~116-130`).
   Known in D-094, still open.
6. **Payout quota is not enforced and is global.** `PgLedger::record_payout`
   (`crates/fa-ledger/src/pg_accounting.rs:1094`) never checks the amount against
   `ledger.payout_quota`; the table is keyed by `day` only, not `(book, day)`; the risk-engine only
   reads `reserved_eur`, nothing writes it. The reserve that shrinks deployable capital is honored,
   the quota itself is advisory.
7. **`runtime.toml` is written into the image.** `PUT /api/config/runtime` writes next to the
   platform file — `/app/config` inside the container (`fa-core/src/config.rs:263`,
   `control-api/src/config_routes.rs:43-52`), owned by root while the process is uid 10001: the
   write likely fails, and would be lost on recreate anyway. Runtime overrides belong in Postgres.
8. **A 16 MB compiled binary is committed:** `go/evo-orchestrator` (commit 56c11fd). `go/bin/` is
   gitignored; this path is not. Should be removed from the tree.
9. **Committed shared secrets.** `docker/garage.toml` carries `rpc_secret` and `admin_token`
   (secret-scan allowlisted); Postgres `fa/fa` appears in compose, `platform.docker.toml` and the
   exporter DSN. Acceptable under ADR 0006 (localhost), fatal if ever deployed as-is.
10. **Legacy `GET /api/trading/pnl` is approximate** (replays only in-range fills into an empty
    average-cost book; positions opened before the range have no basis). The correct route is
    `/api/trading/pnl/breakdown`; the legacy one should redirect or be removed.
11. **Allocator performance stats are signal-implied, not realized.** `PerformanceTracker` uses
    `exposure_mean × instrument return` (`crates/fa-allocator/src/lib.rs:221-301`), ignoring fees,
    slippage and carry; ledger-attributed P&L exists but doesn't feed the allocators (or
    evolution). Fine for equal-weight (the live default); misleading the day Kelly/Thompson is
    switched on.
12. Smaller: FX restatement still pending (owner-gated, D-095 — until applied the
    strategy-vs-book identity fails on `paper` by −214.37 EUR); unrealised P&L in ledger reports is
    last-fill-marked, not market-marked (`pg_accounting.rs:396-408`); `BookSnapshot`,
    `OnchainEvent`, `MediaChunk` protos have no producer; the tax report endpoint has no UI caller;
    7 `#[ignore]` tests are all accounted for (replay/live-stack gated).

### 3.3 Maturity staircase

1. **Paper on one host (current):** done except items 1–2 above. This is the level the acceptance
   criterion in spec 00 §9 claims, and the claim holds.
2. **Real keyed data:** code exists (lemon, oanda fail `Init` without keys); blocked on keys only.
3. **Live trading:** a stub by design (D-029); needs venue adapters, `trading.mode.live_enabled`,
   the live sleeve, and a revisit of ADR 0006.
4. **Cloud / multi-machine:** §6.

## 4. Internal finance: infrastructure growth as a revenue split

### 4.1 What exists

- An exact, event-sourced trading ledger (`crates/fa-ledger`, Postgres `ledger.*`): fills + cash
  movements as the only event types, `NUMERIC(38,18)`, EUR base with native amounts and the FX P&L
  split preserved per lot, FIFO/avg lots, per-strategy causal pro-rata attribution, orphan
  handling, restatement machinery, statements and a tax report. The identity
  `equity − external flows = net P&L` is tested exactly. This is a solid foundation.
- **The only profit-allocation primitive is the daily payout quota**
  (`crates/fa-ledger/src/payout.rs`, `risk.toml:191-202`): `fixed_eur` (default **0**),
  `pct_equity` or `pct_profit` of the previous day's positive net P&L, capped at 5 % of equity;
  the reserve shrinks deployable capital (`fa-risk/src/lib.rs:273-285`); booking is a manual
  operator action.
- **Every budget for the system itself is a resource, never money:** VM 50 GiB disk + 5 M series,
  stream byte caps, the evo GPU-hour budget (`used_gpu_ms` per job in `ops.job_run` — the only
  resource meter), a 50 ms CPU step budget, and the `compute_cost` fitness objective (an op count,
  not a price).
- **Nothing records what the system costs to run.** No `cash_movement` kind for operating
  expenses, no cost table in any schema, no price on any meter, no notion of revenue split,
  reinvestment or treasury anywhere in code, specs or decisions (verified by repo-wide search).
  Spec 00's open question 1 (success metrics, starting capital) is still unresolved.

### 4.2 Proposed design: close the loop profit → infra budget → capacity

The owner's framing — infrastructure growth as the method of revenue split — fits the existing
ledger with three additions, in dependency order:

1. **Price the meters (opex accounting).** A config price table
   (`[costs] gpu_eur_per_hour, storage_eur_per_gib_month, egress_eur_per_gib, api_call_eur, …`)
   and a monthly/daily accrual job (control-api already owns scheduled maintenance,
   `bins/control-api/src/maintenance.rs`, and already reads `vm_data_size_bytes`, `used_gpu_ms`,
   stream bytes). It books `cash_movement kind='opex'` rows with a `ref` naming the meter and
   period — idempotent via the existing `(book, venue, kind, ref)` uniqueness. A new
   `CashClass::Opex` in `accounting.rs:146-167` makes net P&L *after own costs* fall out of the
   existing `PnlBreakdown` for free. On this host the prices are notional (electricity + wear);
   in the cloud they become real invoices, same rows.
2. **Generalize the payout rule into a split policy.** Extend `payout.rs` from one quota to a
   split of the previous day's positive net P&L (now net of opex): fractions for
   **operator payout** / **infra budget** / **retained trading capital**, each booked as its own
   cash movement (`payout`, `transfer:infra`, nothing — retained is the default residual). The
   infra budget accumulates in its own balance (a `transfer`-class movement already exists and is
   excluded from P&L, so no accounting change). Fix the enforcement gaps first: check amounts
   against the quota in `record_payout`, key `payout_quota` by `(book, day)`.
3. **Let the infra balance set capacity.** The infra budget becomes the input to the resource
   budgets instead of hand-set constants: VM disk budget, stream caps (`FA_STREAM_SCALE` already
   exists), evo GPU-hours, and — once §6 lands — replica counts / node-pool size. Growth is then
   literally purchased by trading profit: the system expands data retention, evolution compute and
   (in cloud) nodes only when its net P&L after its own costs funds it. A `ledger.infra_budget`
   view plus one control-api route makes it auditable; the dashboard work in §5 makes it visible.

Prerequisite from §3.2: one source of truth for money on screen — retire or fix the control-api
in-memory book (item 3) before any finance view grows, or the new finance numbers will sit next to
wrong ones.

## 5. Dashboard: capturing the entire system

### 5.1 Covered today

Services (heartbeats, container state, start/stop/restart), NATS streams + consumer lag, live bus
rates, sources with limiter saturation and backfills, jobs/GPU, audit, config snapshots, alerts
(vmalert), the full evolution lab (generations, Pareto fronts, lineage, validation, lifecycle),
snapshots with promotion checklist and expert-health heatmaps, MCMC posterior queries, trading
desk, risk desk, ledger (P&L breakdown, lots, statements, orphans, payouts, restatement), refdata.

### 5.2 Not captured (the gap list)

1. **Stores are TCP up/down badges only** (`control-api/src/routes.rs:116-148` — 300 ms connect):
   no Postgres size/connections, VM series/ingest rate, Garage buckets/bytes, Qdrant
   collections/vectors, Helix node/edge counts — although nats/postgres exporters already scrape
   into VM.
2. **Backups, WAL archive, restore drills:** metrics are pushed to VM (`fa_backup_*`,
   `fa_restore_drill_*`, `fa_wal_archive_*`) but no page or endpoint reads them; they surface only
   as alerts.
3. **Disk usage:** only the DiskSpaceLow / JetStreamStorageHigh alerts; `vm_data_size_bytes` vs
   budget is used inside maintenance but never exposed.
4. **DLQ:** stream rows + an alert; no browser, no payload inspection, no replay/purge.
5. **Maintenance runs** (VM retention, genome GC, media downsampling): raw audit rows only — no
   last-run/next-run/reclaimed-bytes panel.
6. **Encoder training generations:** `/api/generations` exists and is unused by any page.
7. **No time series of anything about the system itself** — and no growth-over-time view at all:
   data volume, series counts, genome/population counts, snapshot cadence, promotions, ingest
   volume per source. The trading sparklines are client-side buffers.
8. Unused backend: `/api/metrics/query{,_range}` (a full VM proxy — only `Projection.svelte` uses
   it), `/api/trading/fills`, `/api/trading/tax/realised`.

### 5.3 Proposal

Two pages close most of this with almost no new backend, because the VM proxy exists:

- **/ops/infra:** store usage cards (from the exporters + VM self-metrics + one new cheap
  control-api aggregate for Garage/Qdrant/Helix counts), backup/WAL/drill status, disk vs budgets,
  maintenance last/next runs, a DLQ browser (list + replay is the only genuinely new endpoint).
- **/growth:** time-series charts over `query_range` — data volumes, series count, per-source
  ingest, genome and snapshot counts, promotions, and (after §4) equity vs opex vs infra budget:
  the system's own growth curve next to the money that funds it. Record the few missing counters
  (genomes, snapshots, promotions) as VM series where they change — one `fa_metrics` gauge each.

## 6. Cloud deployment: entrypoint and bootstrapping

### 6.1 Current state

Single host by design and by document: ADR 0006 (localhost-only, no auth; exposing any port
"requires revisiting this ADR"), docker/README (ex-spec 60) lists cloud as out of scope, specs
01/21 defer NATS clustering. There are **no** k8s/Helm/Terraform/kustomize files, no CI, no image
registry (all `:dev` local tags). The data-plane *design* is cloud-friendly — stateless pull
workers over JetStream work queues, idempotent stream/schema ensure, checkpoints in S3 — the
packaging is not.

**Bootstrap from nothing today** (manual): write `config/secrets.toml` → write `.env`
(`FA_DOCKER_SOCK`, `FA_DOCKER_GID`) → `docker compose up -d` (infra) → `docker/garage-init.sh`
(host-only: layout, buckets, key — paste the printed key back into secrets.toml) →
`compose --profile app build && up -d` (control-api migrates Postgres once, no retry; streams
ensured with retry; Helix/Qdrant lazily) → `scripts/healthwait.sh`. `scripts/eph.sh up` scripts
this for ephemeral runs and is the closest thing to an entrypoint.

**Single-host assumptions** (full inventory in the survey; the load-bearing ones): the Docker
socket supervisor (D-060); repo bind mounts for secrets/sources/risk.toml/costs.yaml; 127.0.0.1
baked into platform.toml, Go flag defaults, `garage.public_endpoint` (browser presigned URLs),
`rpc_public_addr`, all ops/backup/restore scripts; volumes shared between containers
(helix-data ↔ backup, pg-wal-archive ↔ postgres); backups to a local `./backups` dir, off-site
hook disabled; fixed host ports (one stack per host); no leader election anywhere; config file-only
with only scattered env overrides (no general `FA_NATS_URL`/`FA_POSTGRES_URL`); hard-coded bucket
names; single-node NATS/Garage/Postgres.

### 6.2 Phased path

**Phase A — make the code cloud-agnostic (valuable now, no cloud required):**
env overrides for every endpoint in `PlatformConfig`; initial-connect retry in `Bus::connect` and
`PgLedger` (fixes §3.2 item 4 — fail closed, never in-memory); move Garage layout/bucket/key init
from `garage-init.sh` into an idempotent code path (buckets are already hard-coded constants — an
`ensure_buckets` beside `ensure_streams`); `runtime.toml` → Postgres (item 7); backup target
becomes S3 (the rclone off-site hook exists, enable and point it at a bucket); drop the committed
binary and rotate the committed Garage secrets out of git. Result: `eph.sh`-style bootstrap becomes
`compose up` with zero manual steps — the real entrypoint, testable on this machine.

**Phase B — packaging:** registry + versioned tags + CI (buf-check, the Docker test commands,
image build/push); Helm chart or kustomize with a `bootstrap` Job (migrations + ensure
streams/buckets/collections) and init-container gates; **CloudNativePG for Postgres** — which also
answers the open ADR 0010 graph half if the Postgres recommendation is confirmed, making Postgres
the one clustered store for ledger + warehouse + graph; NATS via its Helm chart (3 nodes, stream
`replicas: 3` — `fa-bus/streams.rs:178` currently leaves replicas at the server default); Qdrant
chart; Garage multi-node or a managed S3 (bucket names must become config); supervisor: replace the
Docker socket with the k8s API + RBAC or ship disabled (it already degrades to `501
no_supervisor`). Singletons (risk-engine, broker-gateway, control-api, evo-orchestrator,
model-runtime) run as replicas=1 StatefulSets; only evo-worker/trainer scale out, which matches the
owner's stated scaling goal.

**Phase C — security (blocking any real exposure):** revisit ADR 0006 — auth on control-api,
ingress TLS, NATS accounts, store credentials as k8s Secrets. Until then nothing may leave
localhost/cluster-internal networking.

## 7. Recommended order of work

1. **risk-engine determinism fix + replay hardening** (§3.2 item 1 — already specified in spec 41,
   awaiting go-ahead; verify on the retained `day1` data before it expires ~2026-11-16 02:52 UTC).
2. **Correctness batch:** control-api book (item 3), risk-engine fail-closed (4), Kraken de-dup
   (5), payout enforcement (6), runtime.toml (7), committed binary (8).
3. **Finance §4 steps 1–2** (opex accrual + split policy) — small, pure-ledger work with outsized
   payoff for the owner's stated goal.
4. **Dashboard §5.3** (/ops/infra, /growth) — mostly frontend over existing endpoints.
5. **Cloud Phase A** — doubles as reliability hardening for the current host.
6. **Real keyed data → spec 32 on real bars → live-trading groundwork**, per the existing plan's
   Phase 9; cloud Phases B/C when the owner schedules them.

Items 1, 3's policy fractions, FX restatement, and the graph-store/CloudNativePG choice are owner
decisions; everything else is implementable under standing rules.
