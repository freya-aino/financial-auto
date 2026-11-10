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
- **Recording**: events are applied in a transaction; uniqueness constraints implement idempotency. Late/out-of-order venue events are accepted and positions recomputed from the affected point **(proposal)**.
- **Position valuation**: mark-to-market with last price from VictoriaMetrics; EUR via latest FX (ECB reference for daily close, live mid for intraday, source recorded).
- **P&L**: realised (FIFO lots per venue/instrument **(proposal)**), unrealised, fees, funding, borrow, FX P&L separated.
- **Attribution**: netted fills split to strategies per 40 §6.3; strategy P&L = attributed fills + virtual mark-to-market.
- **Reconciliation**: fetch venue positions, open orders, cash, recent fills; diff against ledger; small cash diffs within tolerance auto-booked as `adjustment` with audit; otherwise venue `halted` (40) until manual resolution in the UI.
- **Payout quota**: each day a quota (rule TBD, see open questions) is reserved; deployable capital = equity − reserved; payouts are recorded as `cash_movement(kind='payout')`.
- **Audit log**: every status change, manual action, limit change, promotion, kill switch, reconciliation adjustment and config reload is appended; never updated or deleted.

**Implemented today (D-047):** `fa-ledger::pg::PgLedger` persists intents, fills (with even strategy attribution), cash movements, position snapshots, reconciliation runs and payout quota to `ledger.*`; idempotency is DB-enforced (`ON CONFLICT DO NOTHING`, cash movements keyed by `(book, venue_id, kind, ref)`). risk-engine is the single writer, opens the `paper` book with a one-time 100 000 EUR deposit, reloads it on start, snapshots every 60 s and on halts, and reconciles the in-memory book against Postgres every 300 s into `reconciliation_run`. A diff is logged and audited; halting on diff is the pending step. FX is fixed at 1.0.

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `recon.interval` | 15 min | plus on startup and reconnect |
| `recon.cash_tolerance_eur` | 1.00 | auto-adjust threshold |
| `ledger.lot_method` | FIFO | |
| `fx.intraday_source` | broker/venue mid | |
| `fx.daily_source` | ECB reference rate | |
| `payout.rule` | TBD | see §10 |

## 8. Failure modes
| Failure | Response |
|---|---|
| Duplicate fill event | rejected by unique constraint, no-op |
| Fill for unknown client order id | stored in `orphan_fill`, venue halted, alert |
| Postgres unavailable | broker-gateway stops sending new orders (fail-closed); events buffered in JetStream |
| Reconciliation diff | venue halted until resolved; diff shown in UI |
| Missing FX rate | use last known rate flagged `stale`; block risk-increasing orders in that currency after threshold |

## 9. Acceptance criteria
- Replaying the same venue event stream twice yields identical ledger state.
- Positions derived from events equal the latest position snapshot.
- Every fill has native and EUR values with FX source and timestamp.
- Reconciliation detects an injected mismatch and halts the venue.
- Audit log contains an entry for every manual API action.
- Sum of strategy-attributed P&L equals portfolio P&L (minus unattributed manual trades).

## 10. Open questions
1. Payout quota rule (fixed EUR, % equity, % realised profit, rollover) — shared with 40 Q1.
2. Is tax-lot / tax reporting (e.g. Dutch box 3 or other jurisdiction) required? — affects lot method and retention.
3. Margin accounting model per venue (cross vs isolated) — affects equity and leverage definitions.
4. Ledger for paper strategies in the allocator tournament: one paper book per allocator, or virtual books? — affects schema.

## 11. Deferred
- Fee schedules per venue (filled per connector spec).
- Multi-entity / multi-account consolidation.
- Export to accounting software.
