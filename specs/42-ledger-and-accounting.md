# 42 — Ledger & Accounting

| | |
|---|---|
| **Status** | draft |
| **Maturity** | High |
| **Owners (code)** | `crates/fa-ledger`, `go/services/broker-gateway` (event source), `migrations/` |
| **Depends on** | 12 Reference data, 20 Storage, 40 Execution & risk |

## 1. Purpose
Be the single source of truth for what the system intended, ordered, filled and holds — per venue, per strategy, in native currency and in EUR. It enables risk checks, P&L, fitness attribution, payout-quota tracking and reconciliation with brokers, and provides an append-only audit log of every consequential action.

## 2. Scope
**In scope:** intents, orders, child orders, fills, positions, cash balances, fees, funding/borrow, FX conversion to EUR, strategy attribution, idempotency, broker reconciliation, payout quota, audit log.
**Out of scope:** tax reporting (open question), risk decisions (40), fitness computation (35/41).

## 3. Decisions
- **PostgreSQL** holds the ledger (ADR 0008); append-only event tables, positions derived (and cached) from events.
- **EUR** is the base currency; native-currency amounts are always stored alongside EUR conversions with the FX rate and its source/time.
- **Idempotent order IDs**: every intent/child order has a deterministic client order id; replays and redeliveries never double-book.
- **Broker reconciliation** on startup and periodically; trading on a venue is blocked until reconciled.
- **Daily payout quota** is tracked in the ledger and consumed by the liquidity/payout limit (40).
- Paper and live books are separate ledgers (`book = paper | live`) with identical schema.

## 4. Interfaces
- Consumes `order.intent.*`, `order.event.*` (21) and manual adjustments from the control API.
- Provides (Rust crate `fa-ledger`):
```rust
pub trait Ledger {
    fn record(&self, ev: LedgerEvent) -> Result<Recorded, LedgerError>; // idempotent on ev.key()
    fn positions(&self, book: Book, at: Option<Timestamp>) -> Vec<Position>;
    fn equity_eur(&self, book: Book, at: Option<Timestamp>) -> Decimal;
    fn strategy_pnl(&self, strategy: StrategyId, range: TimeRange) -> PnlBreakdown;
    fn payout_quota(&self, day: Date) -> PayoutQuota;
}
```
  **Implemented as** inherent methods of `fa_ledger::pg::PgLedger` (no trait): `record(LedgerEvent)` / `record_fill_event`, `positions(book, at)`, `equity_eur(book, at, marks)`, `strategy_pnl(book, strategy, from, to)` / `strategy_pnls`, `book_pnl`, `payout_quota(day)`, plus the in-memory `fa_ledger::Ledger` the risk-engine books into.
- Control API: `GET /api/trading/positions`, `/api/trading/orders`, `/api/trading/fills`, `/api/trading/pnl`, `/api/trading/reconciliation`, `/api/audit`.
- Money is `fa_ledger::Decimal` (`rust_decimal`) at the ledger scale of 18 places (`fa_ledger::money`); wire values (`f64` in `fa.v1`) are converted once by `money::dec` (shortest round-tripping decimal, banker's rounding to 18 places).

## 5. Data model (Postgres sketch)
```sql
-- all money as NUMERIC(38,18); timestamps timestamptz (UTC)
create table order_intent (
  id uuid primary key, book text not null, portfolio_target_id uuid,
  instrument_id uuid not null, side text not null, qty numeric not null,
  exec_algo text not null, params jsonb, idempotency_key text unique not null,
  created_at timestamptz not null);

create table venue_order (
  client_order_id text primary key,          -- deterministic, derived from intent id + seq
  intent_id uuid references order_intent, venue_id uuid not null,
  venue_order_id text, status text not null, qty numeric, limit_price numeric,
  created_at timestamptz, updated_at timestamptz);

create table fill (
  id uuid primary key, client_order_id text references venue_order,
  venue_fill_id text not null, qty numeric not null, price numeric not null,
  ccy text not null, fee numeric not null, fee_ccy text not null,
  fx_rate_eur numeric not null, fx_source text not null,
  arrival_price numeric, filled_at timestamptz not null,
  unique (client_order_id, venue_fill_id));

create table fill_attribution (fill_id uuid references fill, strategy_id uuid, qty numeric, fee_share numeric);

create table cash_movement (                  -- deposits, withdrawals, payouts, funding, borrow, dividends
  id uuid primary key, book text, venue_id uuid, kind text, amount numeric, ccy text,
  fx_rate_eur numeric, ref text, occurred_at timestamptz, unique (venue_id, kind, ref));

create table position_snapshot (book text, venue_id uuid, instrument_id uuid, qty numeric,
  avg_price numeric, as_of timestamptz, primary key (book, venue_id, instrument_id, as_of));

create table payout_quota (day date primary key, quota_eur numeric, reserved_eur numeric, paid_eur numeric);

create table reconciliation_run (id uuid primary key, venue_id uuid, started_at timestamptz,
  status text, diffs jsonb);

create table audit_log (id bigserial primary key, at timestamptz not null default now(),
  actor text not null,           -- 'system:<service>' | 'ui'
  action text not null, subject text, payload jsonb);
```
Strategy virtual sub-positions are derived from `fill_attribution`. Beyond this sketch the migrations add `orphan_fill`, the lot cache (`lot`, `lot_closing`, `lot_rebuild`), view `fill_valued`, `fill_restatement`, `venue_settlement`, `payout_quota.rule`, `strategy_state` and `risk_state` (see `migrations/README.md`); the audit log is `audit.log`.

## 6. Behaviour & algorithms
- **Recording**: events are applied in a transaction; uniqueness constraints implement idempotency. Late/out-of-order venue events are accepted and positions recomputed from the affected point: every derived view (lots, positions, P&L, statements) is a pure replay of the events sorted by `(time, id)` and de-duplicated by id, so arrival order never matters.
- **Position valuation**: mark-to-market with last price from VictoriaMetrics; EUR via latest FX (ECB reference for daily close, live mid for intraday, source recorded). Today both use the ECB daily rate (`ecb_daily`); no intraday FX source is wired.
- **P&L**: realised (FIFO lots per venue/instrument; average cost selectable), unrealised, fees, funding, borrow, FX P&L separated. EUR P&L of a lot splits as `q·(P₁X₁ − P₀X₀) = q·(P₁ − P₀)·X₁` (price part, at the closing/mark rate) `+ q·P₀·(X₁ − X₀)` (FX part), plus the revaluation of non-EUR cash; with that split `equity − external flows = net P&L` holds exactly.
- **Attribution**: netted fills split to strategies per 40 §6.3; strategy P&L = attributed fills + virtual mark-to-market.
- **Reconciliation**: fetch venue positions, open orders, cash, recent fills; diff against ledger; small cash diffs within tolerance auto-booked as `adjustment` with audit; otherwise venue `halted` (40) until manual resolution in the UI.
- **Payout quota**: each day a quota (rule TBD, see open questions) is reserved; deployable capital = equity − reserved; payouts are recorded as `cash_movement(kind='payout')`.
- **Audit log**: every status change, manual action, limit change, promotion, kill switch, reconciliation adjustment and config reload is appended; never updated or deleted.

**First implementation (D-047; the halting, FX and reconciliation points are superseded below):** `fa-ledger::pg::PgLedger` persists intents, fills (with even strategy attribution), cash movements, position snapshots, reconciliation runs and payout quota to `ledger.*`; idempotency is DB-enforced (`ON CONFLICT DO NOTHING`, cash movements keyed by `(book, venue_id, kind, ref)`). risk-engine is the single writer, opens the `paper` book with a one-time 100 000 EUR deposit, reloads it on start, snapshots every 60 s and on halts, and reconciles the in-memory book against Postgres every 300 s into `reconciliation_run`. A diff was logged and audited (it now halts `broker`, round 2); FX was fixed at 1.0 (ECB since D-061).

**Since D-061:** accounting FX is the ECB daily reference rate, fetched by risk-engine into `ref.fx_rate` (`source_id = ecb_daily`) and applied as of the event time to marks, fills and fees; stale rates are flagged and can raise a reduce-only `fx` halt. Fills persist currency, rate, source, arrival price and slippage. Each strategy's attributed sub-book is persisted in `ledger.strategy_state.book` and restored on start. The payout reserve is the larger of the configured quota and today's `ledger.payout_quota`; booking the payout itself is still manual (open question 1).

**Since the spec-42 completion pass (2026-11-11):** `fa_ledger::accounting` derives tax lots per `(book, venue, instrument)` with signed quantities (short lots included) under `ledger.lot_method` = `fifo` (default) or `average` (`PgLedger::with_lot_method`), lot closings with realised P&L in the native currency and in EUR (price and FX parts), as-of positions and equity, and `PnlBreakdown {realised, unrealised, fx_realised, fx_unrealised, fx_cash, fees, funding, borrow, other, fx, net}` from `ledger.fill` + `ledger.cash_movement`; cash kinds are classified as external (deposit, withdrawal, transfer, payout — not P&L), funding, borrow, fee and other. Fills settle in EUR at the fill's rate (the paper venue's and in-memory `Ledger`'s model); non-EUR cash movements are held as native balances and revalued. `PgLedger` gains `positions(book, at)`, `open_lots`, `equity_eur(book, at, marks)`, `book_pnl(book, from, to, marks)`, `strategy_pnl(book, strategy, from, to)`, `strategy_pnls` (per strategy UUID plus `unattributed`, which carries unattributed trades and book-level cash items, so the values sum to the book), `statement(book, from, to, marks)` / `daily_statement(book, day)` (typed `Statement`: opening/closing cash, foreign cash, positions and equity, fills with native and EUR values, cash movements, lot closings, P&L, payouts, external flows and an `unexplained_eur` check; JSON via serde and `to_csv()`), `rebuild_lots` (replaces the derived cache `ledger.lot`/`lot_closing`/`lot_rebuild` for a book wholesale; deterministic), `snapshot_check` (derived quantities as of the latest snapshot batch vs that batch), `fill_valuation_gaps` (view `ledger.fill_valued`: native + EUR values, FX source and the rate's validity time per fill), `record(LedgerEvent)` / `record_fill_event` (a fill for an unknown client order id is parked in `ledger.orphan_fill` and audited instead of failing; redelivery after the order is known books it and closes the orphan), `orphan_fills`, `open_orphan_count(venue)`, `resolve_orphan(id, Book | Dismiss{reason}, actor)` (audited), and `record_payout(book, day, eur)` (one `cash_movement(kind='payout', ref=day)` per book and day, sets `payout_quota.paid_eur` to that day's payouts across books, audited). Valuation without explicit marks uses the latest fill price per instrument and the latest `ref.fx_rate` at or before the cutoff (else the fill's rate); the ledger has no market-data access, so live mark-to-market is the caller's `Marks`. Migration `20261111000030_ledger_lots_orphans`. Verified on the live `paper` book (6 575 fills, read-only): derived positions equal the latest snapshot, no fill lacks a valuation, per-strategy 24 h P&L sums to the book's, the statement identity holds to < 1e-6 EUR. Since wired: control-api routes for statements/orphans/lots (below) and the risk-engine's orphan halt (round 2).

**Ledger ↔ risk-engine interface and exact money (2026-11-11, round 2):**
- *Root cause of the recurring reconciliation diffs* (19:53, 20:57, 21:10, 01:48 and 02:54 UTC; each raised a persisted `broker` halt): not fees, FX, duplicates or the restart rebuild, but a snapshot race. The reconciliation cloned the in-memory book, then read Postgres (cash, then ~7 000 fills); the sim instrument fills about once a second, so a fill applied and persisted between the clone and the read appeared in Postgres only. The 02:54:33.68 diff is exactly the 02:54:33.548 buy of 1.118 974 724 at 100.778 5 (cash +112.77 + fee, quantity +1.118 97); the next pass was always clean. It clustered after restarts only because the first pass ran while the allocator was rebalancing. A second, persistent cause appeared at 03:32–04:14 UTC: risk-engine subscribed to `order.>` and booked every account's fills into the paper book, while Postgres files them under their own book (a `paper-test` harness traded on `order.*.paper-test`). It now gates and books only `order.*.<account>` (`recon::own_account`, tested). The new reconciliation named the 60 foreign fills in `fills_missing_in_postgres`. Two latent sources of real divergence also existed: payouts booked through control-api (`cash_movement` written by another writer) never reached the in-memory cash, and fills for intents the engine did not know (pending at a restart, evicted from the intent cache) were dropped from both books.
- *Fix, by construction*: `PgLedger::reconcile_cut(book, ours, in_flight, cash_tol, qty_tol)` compares at the memory's cut. It compares additive totals (cash, fees, quantity per instrument — order-independent) in exact decimals: ours minus the fills whose write is still in flight, against Postgres restricted to the fills and cash movements ours has applied. Rows Postgres has and memory has not are returned (`fills_only_theirs`, `cash_only_theirs`) and the risk-engine applies them (`Engine::sync_from_ledger`, audited `ledger.synced`). Fills memory booked that Postgres lacks after the in-flight grace (60 s) are reported (`fills_missing_in_postgres`). Tolerances are 0: both sides hold the same decimals. `reconciliation_run.diffs` now also records the cut (`compared_fills`, `in_flight`) on `ok` runs. Regression tests: `pg::tests::reconcile_cut_has_no_snapshot_race_and_finds_real_gaps` (replays the 02:54 race: the pre-fix comparison reports a diff, `reconcile_cut` does not, a write that never landed is still found) and `tests::totals_are_order_independent_and_exact`.
- *Live verification*: after the restart at 04:26 UTC, the runs at 04:31, 04:36 and 04:41 were `ok` with 7 260 fills compared. The persisted `broker` halt was left in place; re-arming it is the operator's call.
- *Known data issue* (restatement tool since round 3, dry run below): instrument `9daa8cea…` has 1 080 fills booked as `EUR` at rate 1.0 (until 01:48, before its USD quote currency was known) and 106 as `USD` at 0.8917. The book total is unaffected, but per-strategy residual lots carry mixed currency labels: on 2026-11-11 the 24 h strategy sum differs from the book by −214.37 EUR (FX realised), and `verify_existing_book_read_only` fails on `paper` (also at the previous commit). Restating those fills would rewrite the book of record, so it is left to the operator.
- *Fill persistence*: risk-engine books every fill through `PgLedger::record_fill_event` (`bins/risk-engine/src/recon.rs`). A fill stays *unconfirmed* in the engine until Postgres has it (new or duplicate) and is retried every 5 s. An own fill parked as an orphan (its intent write had failed) gets the intent re-recorded from memory (`exec_algo = recovered`) and is booked, closing the orphan. A fill event for an unknown intent is recovered from `ledger.order_intent` (instrument, side, contributors) and booked in both books. Without an intent anywhere it is parked in `ledger.orphan_fill` without assumptions (round 3, below).
- *Orphan halt*: at start and on every reconciliation pass (and right after an orphan is parked), `open_orphan_count(PAPER_VENUE) > 0` halts scope `broker` (persisted, flattening). The halt is only raised, never cleared: the operator resolves the orphan (`POST /api/trading/orphans/{id}/resolve`) and publishes `risk.resume.broker`; an orphan still open re-halts on the next pass.
- *Exact money*: `Ledger` holds `ExactBook` (cash, fees, payouts, positions with average cost and realised P&L as `Decimal`). Its `f64` fields are read-only views refreshed after each change, kept for the risk arithmetic and JSON. `ExactFill::of` is the canonical decimal of a wire fill, and its `cash_flow()` is shared by booking and reconciliation. Every Postgres write in `pg.rs`/`pg_accounting.rs` (intents, venue orders, fills, attribution, cash movements, snapshots, payout quota, orphans, payouts) binds `NUMERIC` directly; no `float8` casts remain. `load_book` rebuilds from the stored decimals (`cash_rows`, `exact_fills`). Existing rows were written through `float8::numeric` (15 significant digits). They are read as stored, so no migration is needed: the in-memory book is seeded from them. Strategy sub-books (attributed estimates, not reconciled) keep `f64` via `Position::apply_f64`.
- *Settlement currency per venue (Q5, decided)*: default EUR settlement for every venue (`fa_ledger::settlement`). `ledger.venue_settlement` (migration `20261112000010`; the paper venue is seeded `eur`) can mark a venue `native`. `PgLedger::with_stored_settlement` then flags its trade events `settle_native`, and `BookState` books their cash in a native balance that is revalued as `fx_cash_eur`, so FX P&L covers trading balances too. Fees stay EUR. `margin_model` (`cash`, `cross`, `isolated`) is recorded but not interpreted (Q3).
- *Payout rule (Q1, decided)*: `fa_ledger::payout::PayoutRule` (`[payout]` in `risk.toml`): `fixed_eur` (`amount_eur`), `pct_equity` or `pct_profit` (of the previous day's positive net P&L), optional `rollover` of yesterday's unpaid quota, capped at `max_equity_frac` (0.05) of the previous close, rounded down to the cent. At the first 30 s check of each UTC day the risk-engine calls `PgLedger::apply_payout_rule`, which raises today's `ledger.payout_quota.quota_eur` (never lowers it) and stores the rule and its inputs in `payout_quota.rule`. The quota is reserved (deployable = equity − reserve). Booking the payout stays an operator action (`POST /api/trading/payouts`, `record_payout`), because moving money off the account is an external transfer. The risk-engine picks the payout cash movement up on its next reconciliation. Default: `fixed_eur` 0 (nothing reserved, as D-041).

**Round 3 (2026-11-11): exact derived analytics, unique fill ids, orphans without assumptions, FX restatement.**
- *Exact derived analytics*: `fa_ledger::accounting` (`TradeEvent`, `CashEvent`, `Lot`, `LotClosing`, `PnlBreakdown`, `BookPosition`, `CashBalance`, `Marks`, `BookState`), `statement` and the `pg_accounting` reads compute in `Decimal` (products rounded to 18 places by `money::mul`; a fill's EUR cash uses the formula of `ExactFill::cash_flow`, so the derived cash equals the book of record digit for digit — `verify_existing_book_read_only` asserts it on `paper`). Reads bind `NUMERIC` (no `::float8` left in the ledger crate; attribution shares come as `numeric::text` and are applied as absolute quantity/fee shares, so shares and the unattributed remainder sum to the fill exactly); the lot cache is written and read as `numeric[]`; `FillRow`, `SnapshotRow` and `OrphanFill` are decimals. API JSON keeps numbers (`money::json`, `json_opt`, `json_map` serialise a decimal as its nearest `f64`), so `/api/trading/statement|lots|pnl/breakdown|orphans|pnl` keep their shapes; the statement CSV prints exact decimals. Dust is no longer hidden: the former `QTY_EPS` (1e-12) is gone, so residues of pre-decimal `float8` rows that the book of record really holds (e.g. 4.91e-14 units; five instruments on `paper`) appear as open positions with their exact quantity, as in the in-memory book. On the live `paper` book the 24 h statement's `unexplained_eur` is 0.000000 and FX P&L, realised parts and fees are exact; `PnlBreakdown::max_abs_diff` and `quantities()` return decimals.
- *Unique fill ids*: `fa.v1.OrderEvent` gains `venue_fill_id = 12`, `instrument_id = 13`, `venue_symbol = 14` (buf lint/breaking pass). The paper broker-gateway stamps every fill `pf-<unix ns>-<seq>` (process-wide sequence) plus the intent's instrument and subject token. risk-engine's `fill_id_of` is `intent:venue_order:vf:<venue_fill_id>` when present, else the legacy `intent:venue_order:<event ms>`; stored rows and replayed old events keep the legacy id, so idempotency is unchanged for them, and two partial fills of one order in the same millisecond are now two fills (tests `same_millisecond_partials_are_distinct_fills`, Go `TestFillIdsAreUniqueWithinOneMillisecond`).
- *Orphans without assumptions* (migration `20261113000010`): `ledger.orphan_fill.instrument_id`, `ccy`, `fx_rate_eur` are nullable; `instrument_status` is `resolved | unresolved | ambiguous`, `venue_symbol` and `fee_eur` are new, the fee is kept in `fee_ccy` as reported. `record_orphan` takes the event's instrument id, else resolves `venue_symbol` through `ref.listing` as of the fill time (one instrument → resolved, several → ambiguous, none → unresolved); the currency is the event's, else the instrument's `quote_ccy`, else unknown; rate and fee EUR value come from `ref.fx_rate` (valid at the fill time) or stay NULL (`fx_source` `unknown`/`missing`). `FillEvent` gains `venue_symbol` and `fee_native {amount, ccy}`; risk-engine parks orphans with the event's instrument id / venue symbol and native fee. `resolve_orphan(Book)` takes the instrument from the now-known order (refusing a mismatch), completes currency, rate and EUR fee from `ref` — refusing (409) instead of assuming when a rate is missing — and signs the venue's unsigned quantity by the order side. Test `orphans_never_assume_instrument_or_fee`; the live orphan tests use a throwaway venue so they never trip the risk-engine's orphan halt.
- *FX restatement* (`fa_ledger::restate`, migration `20261113000011`): `plan_fx_restatement` (dry run, writes nothing) selects a book's fills of one instrument still booked in `from_ccy` (EUR), assigns the instrument's quote currency and the valid-time rate (latest `ref.fx_rate.ts ≤ filled_at`, `ecb_daily` preferred), and replays the book with and without it (cash, equity, inception P&L, window book net vs per-strategy sum). `apply_fx_restatement` rewrites exactly the planned rows in one transaction (optimistic on the old values; a stale plan is refused), keeps the old values in `ledger.fill_restatement`, audits `ledger.restatement` and rebuilds the lot cache; quantities, prices and fees never change. Operator entry points: `cargo run -p fa-ledger --example restate_fx -- --instrument <id> [--apply --reason …]` and `POST /api/trading/restatements/fx` (dry run by default; `apply` needs `confirm: true` → else 428, and a reason). After an apply the risk-engine must be restarted so its in-memory book reloads (else reconciliation reports the cash change and halts `broker`).
  - *Dry run for instrument `9daa8cea…` on `paper` (not applied, operator decision)*: 1 080 fills (626 `fixed`, 454 `eur`, 2026-11-10 19:48:59 → 2026-11-11 01:48:18) EUR → USD at 0.891742464776173 (`ecb_daily`, valid 2026-11-10; recorded 2026-11-11 02:09); 0 without a rate. Cash and equity +8.054251 EUR, inception realised +8.054251, 24 h book net −13 194.715886 → −13 186.661635, 24 h strategy sum −13 409.088950 → −13 186.661635: the strategy − book gap goes from −214.373064 to 0.000000.

**Control API (2026-11-11):** `bins/control-api/src/refdata_routes.rs` serves the derived accounting: `GET /api/trading/statement?book=&from=&to=&format=json|csv`, `GET /api/trading/lots?book=&at=`, `GET /api/trading/pnl/breakdown?book=&from=&to=` (book plus per-strategy breakdown summing to it), `GET /api/trading/orphans?status=`, `POST /api/trading/orphans/{id}/resolve` (`{"action":"book"}` | `{"action":"dismiss","reason":…}`) and `POST /api/trading/payouts {book?, day, amount_eur}` (audited). Since round 2 the risk-engine books through `record_fill_event` and halts on open orphans (above).

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `recon.interval` | 300 s (`reconcile_secs`) | orphan check also at startup |
| `recon.cash_tolerance_eur` | 0 (exact) | paper book vs Postgres hold identical decimals; a broker connector will need a tolerance + auto-adjust |
| `recon.in_flight_grace` | 60 s | unconfirmed fill writes younger than this are excluded from the cut |
| `fill.retry_after` | 5 s | unconfirmed fill writes are retried |
| `ledger.settlement` | `eur` per venue | `ledger.venue_settlement`; `native` books fill cash in the fill currency |
| `ledger.lot_method` | FIFO | `fifo` \| `average`; `PgLedger::with_lot_method` |
| `fx.intraday_source` | broker/venue mid | not implemented: `ecb_daily` is used intraday too |
| `fx.daily_source` | ECB reference rate | `ecb_daily`, `risk.toml [fx] source` |
| `payout.rule` | `fixed_eur`, 0 EUR | `[payout]` in `risk.toml`: `fixed_eur` \| `pct_equity` \| `pct_profit`, `rollover`, `max_equity_frac` 0.05 |

## 8. Failure modes
| Failure | Response |
|---|---|
| Duplicate fill event | rejected by unique constraint, no-op |
| Fill for unknown client order id | stored in `orphan_fill` (audited; `open_orphan_count` for the halt), venue halted, alert |
| Postgres unavailable | broker-gateway stops sending new orders (fail-closed); events buffered in JetStream. Today the gateway has no Postgres check: it fails closed on a missing risk-engine heartbeat (15 s, D-079), and unconfirmed fill writes are retried by the risk-engine |
| Reconciliation diff | venue halted until resolved; diff shown in UI (`fills_missing_in_postgres` names the fills) |
| Fill write fails | fill stays unconfirmed in the engine, retried every 5 s; after 60 s it counts as missing (diff → halt) |
| Fill event for an intent unknown to the engine | intent recovered from Postgres and the fill booked in both books; no intent anywhere → orphan → halt |
| Ledger row written by another service (payout, operator-booked orphan) | applied to the paper book on the next reconciliation (audited `ledger.synced`), not a diff |
| Missing FX rate | use last known rate flagged `stale`; block risk-increasing orders in that currency after threshold. **Today:** the latest `ref.fx_rate` at or before the event is used; a currency older than `fx.max_age_hours` (72) is flagged stale (`fx.stale_now`) and, only with `fx.halt_on_stale` (default `false`), raises a book-wide reduce-only `fx` halt that clears when rates are fresh — not a per-currency block |

## 9. Acceptance criteria
**Remaining (2026-11-11, code checked at e25e9b5):**
- **Live `paper` data until the FX restatement is applied** (operator action, tool exists — §6 round 3): instrument `9daa8cea…`'s 1 080 fills booked as `EUR` at 1.0 leave the 24 h per-strategy sum −214.37 EUR off the book's net, so criterion 6 holds in tests but not on the current `paper` book, and `verify_existing_book_read_only` fails there. No code change is needed; apply the planned restatement (`POST /api/trading/restatements/fx` with `apply` + `confirm`, then restart the risk-engine) and re-run the verification.
- **Tax reporting** (open question 2): undecided — whether a jurisdiction-specific report (e.g. Dutch box 3) is required, or tax reporting is explicitly deferred.
- **Intraday FX** (§6 valuation, `fx.intraday_source`): not built; `ecb_daily` is used intraday too. A live mid needs a keyed FX source (OANDA) or a decision that daily ECB rates suffice for a EUR paper book.
- **Missing FX rate** (§8): a book-wide, opt-in reduce-only halt instead of a per-currency block of risk-increasing orders.
- **Audit criterion** (5): ledger-side manual actions are audited; whole-API coverage is tracked in spec 50.
- Out of scope here: margin model per venue (Q3, needs a live broker connector), reconciliation against a real venue (needs a broker connector).

- Replaying the same venue event stream twice yields identical ledger state. — **met** (`accounting::tests::replay_is_order_independent_and_idempotent`, live `derived_accounting_round_trip`; `verify_existing_book_read_only`, which also checks replay identity on `paper`, fails there until the FX restatement is applied — see Remaining).
- Positions derived from events equal the latest position snapshot. — **met** (`PgLedger::snapshot_check`; 0 diffs on `paper`).
- Every fill has native and EUR values with FX source and timestamp. — **met** (`ledger.fill_valued`, `fill_valuation_gaps` = 0 on `paper`; the timestamp is the `ref.fx_rate` validity time, or the fill time for EUR/fixed).
- Reconciliation detects an injected mismatch and halts the venue. — **met for the paper venue** (`reconcile_cut` reports a fill missing in Postgres; risk-engine halts scope `broker`; open orphans halt too); false positives from the snapshot race are gone by construction. A real venue needs a broker connector.
- Audit log contains an entry for every manual API action. — ledger-side manual actions (orphan resolution, payout) are audited; API coverage is spec 50's.
- Sum of strategy-attributed P&L equals portfolio P&L (minus unattributed manual trades). — **met in code** (`strategy_pnls` incl. `unattributed` sums to `book_pnl`; unit test over FIFO/average and several ranges); on the live `paper` book only after the pending FX restatement (Remaining).

## 10. Open questions
1. ~~Payout quota rule~~ — **decided** (round 2): `fa_ledger::payout`, default `fixed_eur` 0; reserved automatically each UTC day, booked by the operator. Open only: the real amounts (user's call).
2. Is tax-lot / tax reporting (e.g. Dutch box 3 or other jurisdiction) required? — affects lot method and retention. FIFO and average-cost lots exist; jurisdiction-specific reports do not.
3. Margin accounting model per venue (cross vs isolated) — affects equity and leverage definitions.
4. ~~Ledger for paper strategies in the allocator tournament~~ — **answered by the implementation** (D-064, spec 40 §6 "Live paper tournament"): virtual books. Each allocator's weights earn the same per-tick strategy sleeve returns inside the risk-engine (`fa_allocator::tournament`), records persist in `ledger.risk_state`; there is one ledger book (`paper`) and no per-allocator ledger.
5. ~~Settlement currency per venue~~ — **decided** (round 2): EUR by default, `native` per venue via `ledger.venue_settlement`; the in-memory paper book stays EUR-settled (only the paper venue exists).
6. ~~Money as `f64`~~ — **resolved** (round 2 for the book of record, round 3 for the derived analytics): lots, P&L breakdown, statements, the lot cache, positions/equity and the reporting reads are exact decimals; only the wire (`fa.v1`) and the risk arithmetic's read-only views are `f64`.

## 11. Deferred
- Fee schedules per venue (filled per connector spec).
- Multi-entity / multi-account consolidation.
- Export to accounting software.
