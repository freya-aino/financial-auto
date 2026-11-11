# 42 — Ledger & Accounting

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Medium |
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
- Control API: `GET /api/trading/positions`, `/api/trading/orders`, `/api/trading/fills`, `/api/trading/pnl`, `/api/trading/reconciliation`, `/api/audit`.

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
Strategy virtual sub-positions are derived from `fill_attribution`.

## 6. Behaviour & algorithms
- **Recording**: events are applied in a transaction; uniqueness constraints implement idempotency. Late/out-of-order venue events are accepted and positions recomputed from the affected point: every derived view (lots, positions, P&L, statements) is a pure replay of the events sorted by `(time, id)` and de-duplicated by id, so arrival order never matters.
- **Position valuation**: mark-to-market with last price from VictoriaMetrics; EUR via latest FX (ECB reference for daily close, live mid for intraday, source recorded).
- **P&L**: realised (FIFO lots per venue/instrument; average cost selectable), unrealised, fees, funding, borrow, FX P&L separated. EUR P&L of a lot splits as `q·(P₁X₁ − P₀X₀) = q·(P₁ − P₀)·X₁` (price part, at the closing/mark rate) `+ q·P₀·(X₁ − X₀)` (FX part), plus the revaluation of non-EUR cash; with that split `equity − external flows = net P&L` holds exactly.
- **Attribution**: netted fills split to strategies per 40 §6.3; strategy P&L = attributed fills + virtual mark-to-market.
- **Reconciliation**: fetch venue positions, open orders, cash, recent fills; diff against ledger; small cash diffs within tolerance auto-booked as `adjustment` with audit; otherwise venue `halted` (40) until manual resolution in the UI.
- **Payout quota**: each day a quota (rule TBD, see open questions) is reserved; deployable capital = equity − reserved; payouts are recorded as `cash_movement(kind='payout')`.
- **Audit log**: every status change, manual action, limit change, promotion, kill switch, reconciliation adjustment and config reload is appended; never updated or deleted.

**Implemented today (D-047):** `fa-ledger::pg::PgLedger` persists intents, fills (with even strategy attribution), cash movements, position snapshots, reconciliation runs and payout quota to `ledger.*`; idempotency is DB-enforced (`ON CONFLICT DO NOTHING`, cash movements keyed by `(book, venue_id, kind, ref)`). risk-engine is the single writer, opens the `paper` book with a one-time 100 000 EUR deposit, reloads it on start, snapshots every 60 s and on halts, and reconciles the in-memory book against Postgres every 300 s into `reconciliation_run`. A diff is logged and audited; halting on diff is the pending step. FX is fixed at 1.0.

**Since D-061:** accounting FX is the ECB daily reference rate, fetched by risk-engine into `ref.fx_rate` (`source_id = ecb_daily`) and applied as of the event time to marks, fills and fees; stale rates are flagged and can raise a reduce-only `fx` halt. Fills persist currency, rate, source, arrival price and slippage. Each strategy's attributed sub-book is persisted in `ledger.strategy_state.book` and restored on start. The payout reserve is the larger of the configured quota and today's `ledger.payout_quota`; booking the payout itself is still manual (open question 1).

**Since the spec-42 completion pass (2026-11-11):** `fa_ledger::accounting` derives tax lots per `(book, venue, instrument)` with signed quantities (short lots included) under `ledger.lot_method` = `fifo` (default) or `average` (`PgLedger::with_lot_method`), lot closings with realised P&L in the native currency and in EUR (price and FX parts), as-of positions and equity, and `PnlBreakdown {realised, unrealised, fx_realised, fx_unrealised, fx_cash, fees, funding, borrow, other, fx, net}` from `ledger.fill` + `ledger.cash_movement`; cash kinds are classified as external (deposit, withdrawal, transfer, payout — not P&L), funding, borrow, fee and other. Fills settle in EUR at the fill's rate (the paper venue's and in-memory `Ledger`'s model); non-EUR cash movements are held as native balances and revalued. `PgLedger` gains `positions(book, at)`, `open_lots`, `equity_eur(book, at, marks)`, `book_pnl(book, from, to, marks)`, `strategy_pnl(book, strategy, from, to)`, `strategy_pnls` (per strategy UUID plus `unattributed`, which carries unattributed trades and book-level cash items, so the values sum to the book), `statement(book, from, to, marks)` / `daily_statement(book, day)` (typed `Statement`: opening/closing cash, foreign cash, positions and equity, fills with native and EUR values, cash movements, lot closings, P&L, payouts, external flows and an `unexplained_eur` check; JSON via serde and `to_csv()`), `rebuild_lots` (replaces the derived cache `ledger.lot`/`lot_closing`/`lot_rebuild` for a book wholesale; deterministic), `snapshot_check` (derived quantities as of the latest snapshot batch vs that batch), `fill_valuation_gaps` (view `ledger.fill_valued`: native + EUR values, FX source and the rate's validity time per fill), `record(LedgerEvent)` / `record_fill_event` (a fill for an unknown client order id is parked in `ledger.orphan_fill` and audited instead of failing; redelivery after the order is known books it and closes the orphan), `orphan_fills`, `open_orphan_count(venue)`, `resolve_orphan(id, Book | Dismiss{reason}, actor)` (audited), and `record_payout(book, day, eur)` (one `cash_movement(kind='payout', ref=day)` per book and day, sets `payout_quota.paid_eur` to that day's payouts across books, audited). Valuation without explicit marks uses the latest fill price per instrument and the latest `ref.fx_rate` at or before the cutoff (else the fill's rate); the ledger has no market-data access, so live mark-to-market is the caller's `Marks`. Migration `20261111000030_ledger_lots_orphans`. Verified on the live `paper` book (6 575 fills, read-only): derived positions equal the latest snapshot, no fill lacks a valuation, per-strategy 24 h P&L sums to the book's, the statement identity holds to < 1e-6 EUR. Still not wired: control-api routes for statements/orphans/lots and risk-engine halting a venue on open orphans (`open_orphan_count`) belong to their owners.

**Control API (2026-11-11):** `bins/control-api/src/refdata_routes.rs` serves the derived accounting: `GET /api/trading/statement?book=&from=&to=&format=json|csv`, `GET /api/trading/lots?book=&at=`, `GET /api/trading/pnl/breakdown?book=&from=&to=` (book plus per-strategy breakdown summing to it), `GET /api/trading/orphans?status=`, `POST /api/trading/orphans/{id}/resolve` (`{"action":"book"}` | `{"action":"dismiss","reason":…}`) and `POST /api/trading/payouts {book?, day, amount_eur}` (audited). Not yet wired: risk-engine still records fills through `record_fill_with` and does not halt on open orphans (risk-engine owner).

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `recon.interval` | 15 min | plus on startup and reconnect |
| `recon.cash_tolerance_eur` | 1.00 | auto-adjust threshold |
| `ledger.lot_method` | FIFO | `fifo` \| `average`; `PgLedger::with_lot_method` |
| `fx.intraday_source` | broker/venue mid | |
| `fx.daily_source` | ECB reference rate | |
| `payout.rule` | TBD | see §10 |

## 8. Failure modes
| Failure | Response |
|---|---|
| Duplicate fill event | rejected by unique constraint, no-op |
| Fill for unknown client order id | stored in `orphan_fill` (audited; `open_orphan_count` for the halt), venue halted, alert |
| Postgres unavailable | broker-gateway stops sending new orders (fail-closed); events buffered in JetStream |
| Reconciliation diff | venue halted until resolved; diff shown in UI |
| Missing FX rate | use last known rate flagged `stale`; block risk-increasing orders in that currency after threshold |

## 9. Acceptance criteria
- Replaying the same venue event stream twice yields identical ledger state. — **met** (`accounting::tests::replay_is_order_independent_and_idempotent`, live `derived_accounting_round_trip`, `verify_existing_book_read_only` on `paper`).
- Positions derived from events equal the latest position snapshot. — **met** (`PgLedger::snapshot_check`; 0 diffs on `paper`).
- Every fill has native and EUR values with FX source and timestamp. — **met** (`ledger.fill_valued`, `fill_valuation_gaps` = 0 on `paper`; the timestamp is the `ref.fx_rate` validity time, or the fill time for EUR/fixed).
- Reconciliation detects an injected mismatch and halts the venue. — ledger side **met** (an injected fill shows in `snapshot_check` and `diff_books`); the halt is risk-engine's scope `broker` (D-051); a real venue needs a broker connector.
- Audit log contains an entry for every manual API action. — ledger-side manual actions (orphan resolution, payout) are audited; API coverage is spec 50's.
- Sum of strategy-attributed P&L equals portfolio P&L (minus unattributed manual trades). — **met** (`strategy_pnls` incl. `unattributed` sums to `book_pnl`; unit test over FIFO/average and several ranges, and on `paper`).

## 10. Open questions
1. Payout quota rule (fixed EUR, % equity, % realised profit, rollover) — shared with 40 Q1. Booking exists (`record_payout`, one per book and day); what triggers it and how much is still open.
2. Is tax-lot / tax reporting (e.g. Dutch box 3 or other jurisdiction) required? — affects lot method and retention. FIFO and average-cost lots exist; jurisdiction-specific reports do not.
3. Margin accounting model per venue (cross vs isolated) — affects equity and leverage definitions.
4. Ledger for paper strategies in the allocator tournament: one paper book per allocator, or virtual books? — affects schema.
5. Settlement currency per venue: fills settle in EUR today (paper model); a live venue with native-currency cash accounts needs per-venue settlement so FX P&L on cash covers trading balances too.
6. Money is `f64` in Rust (`NUMERIC` in Postgres); move to a decimal type before live money (spec §4 sketches `Decimal`).

## 11. Deferred
- Fee schedules per venue (filled per connector spec).
- Multi-entity / multi-account consolidation.
- Export to accounting software.
