# fa-ledger

Orders/fills/positions/cash ledger, EUR conversion, reconciliation, payout quota, derived accounting (lots, P&L, statements) and the realised-gains report.

This README is the reference for the ledger. It replaces spec 42 (Ledger & Accounting), retired on 2026-11-11 under the rule in [specs/README.md](../../specs/README.md) once every in-scope acceptance criterion was met in code. The numbered sections below keep that spec's numbering, so code comments citing "spec 42 §n" resolve here. The decision history is in [specs/DECISIONS.md](../../specs/DECISIONS.md) (D-047, D-061, D-072, D-078 and later).

**Status:** implemented. The in-memory `Ledger` is the paper book of `bins/risk-engine` (which persists it) and `control-api` (which seeds it from Postgres at startup). `pg::PgLedger` persists to the `ledger.*` tables. Tested in memory and against the live DB (`DATABASE_URL`). `LEDGER_VERIFY_BOOK=paper DATABASE_URL=… cargo test -p fa-ledger verify_existing_book -- --nocapture` runs a read-only acceptance pass over a live book.

## 1. Purpose
The ledger is the single source of truth for what the system intended, ordered, filled and holds, per venue and per strategy, in native currency and in EUR. Risk checks, P&L, fitness attribution, payout-quota tracking, reconciliation with brokers and tax reporting all read from it. It also keeps the append-only audit log of every consequential action.

## 2. Scope
**In scope:** intents, orders, child orders, fills, positions, cash balances, fees, funding/borrow, FX conversion to EUR, strategy attribution, idempotency, broker reconciliation, payout quota, audit log, the jurisdiction-neutral realised-gains report.
**Out of scope:** jurisdiction-specific tax rules (user decision, §10 Q2), risk decisions (spec 40), fitness computation (specs 35/41), margin model per venue and reconciliation against a real venue (both need a live broker connector).

## 3. Decisions
- **PostgreSQL** holds the ledger (ADR 0008). Event tables are append-only; positions are derived from events (and cached).
- **EUR** is the base currency. Native-currency amounts are always stored alongside their EUR conversion, with the FX rate and its source and time.
- **Idempotent order IDs**: every intent and child order has a deterministic client order id, so replays and redeliveries never double-book.
- **Broker reconciliation** runs on startup and periodically; trading on a venue is blocked until it is reconciled.
- **Daily payout quota** is tracked in the ledger and consumed by the liquidity/payout limit (spec 40).
- Paper and live books are separate ledgers (`book = paper | live`) with identical schema.
- **Money** is `fa_ledger::Decimal` (`rust_decimal`) at 18 places (`money`). Wire values (`f64` in `fa.v1`) are converted once by `money::dec` (shortest round-tripping decimal, banker's rounding to 18 places). Postgres stores `NUMERIC(38,18)`.

## 4. Interfaces
Inherent methods of `pg::PgLedger` (there is no trait):
- recording: `record(LedgerEvent)` / `record_fill_event`, `record_fill_with(fill, at, strategies, &FillDetails {ccy, fx_source, arrival_price, slippage_bps})`, `record_fill` (the EUR/`fixed` shorthand)
- book: `positions(book, at)`, `open_lots`, `equity_eur(book, at, marks)`, `book_pnl(book, from, to, marks)`, `strategy_pnl(book, strategy, from, to)` / `strategy_pnls`, `statement(book, from, to, marks)` / `daily_statement(book, day)`
- tax: `realised_gains(book, &TaxReportOptions)` → `tax::TaxReport` (JSON via serde / `to_json`, CSV via `to_csv`)
- payout: `payout_quota(day)`, `apply_payout_rule`, `record_payout(book, day, eur)`
- reconciliation and checks: `reconcile_cut`, `snapshot_check`, `fill_valuation_gaps`, `rebuild_lots` / `cached_lots` / `cached_lot_closings`
- orphans: `orphan_fills`, `open_orphan_count(venue)`, `resolve_orphan(id, Book | Dismiss{reason}, actor)`, `resolve_orphan_instrument`
- restatement: `plan_fx_restatement`, `apply_fx_restatement`
- `load_book(book)` rebuilds an in-memory `Ledger` from the stored decimals.

Consumers: risk-engine books every fill (`record_fill_event`, `bins/risk-engine/src/recon.rs`). Control API: `GET /api/trading/positions|orders|fills|pnl|reconciliation`, `/api/audit` (`ledger_routes.rs`); `GET /api/trading/statement?book=&from=&to=&format=json|csv`, `/api/trading/lots?book=&at=`, `/api/trading/pnl/breakdown?book=&from=&to=`, `/api/trading/orphans?status=`, `POST /api/trading/orphans/{id}/resolve`, `POST /api/trading/payouts`, `POST /api/trading/restatements/fx`, `GET /api/trading/tax/realised?book=&year=&method=&year_start=&format=json|csv` (`refdata_routes.rs`).

Examples (operator tools): `cargo run -p fa-ledger --example restate_fx -- --instrument <id> [--apply --reason …]` and `cargo run -p fa-ledger --example tax_report -- [--book paper] [--year 2026] [--method fifo|lifo|average] [--year-start MM-DD] [--format summary|csv|json] [--out file]`.

## 5. Data model
All money is `NUMERIC(38,18)`; timestamps are `timestamptz` (UTC). The schema is in `migrations/` (see `migrations/README.md`):
- `ledger.order_intent` (book, instrument, side, qty, exec algo, `idempotency_key` unique), `ledger.venue_order` (deterministic `client_order_id`), `ledger.fill` (qty, price, `ccy`, fee, `fx_rate_eur`, `fx_source`, arrival price, slippage, `filled_at`; unique `(client_order_id, venue_fill_id)`), `ledger.fill_attribution` (strategy, qty, fee share).
- `ledger.cash_movement` (deposits, withdrawals, payouts, funding, borrow, dividends, adjustments; unique `(book, venue_id, kind, ref)`), `ledger.position_snapshot`, `ledger.payout_quota` (+ `rule`), `ledger.reconciliation_run` (diffs incl. the cut), `audit.log`.
- Derived and support tables: `ledger.orphan_fill`, the lot cache (`ledger.lot`, `lot_closing`, `lot_rebuild`), view `ledger.fill_valued` (native + EUR values, FX source and the rate's validity time per fill), `ledger.fill_restatement`, `ledger.venue_settlement`, `ledger.strategy_state`, `ledger.risk_state`.
- FX rates live in `ref.fx_rate` (`ccy`, `ts`, `rate_to_eur`, `source_id` → `catalog.source`): `ecb_daily` rows and intraday rows under the feed's source id (`yahoo`).

Strategy sub-positions are derived from `fill_attribution`.

## 6. Behaviour
- **Recording**: events are applied in a transaction; uniqueness constraints make it idempotent (`ON CONFLICT DO NOTHING`). Every derived view (lots, positions, P&L, statements, the tax report) is a pure replay of the events sorted by `(time, id)` and de-duplicated by id. Arrival order never matters, so late or out-of-order events land where they belong.
- **Valuation and FX**: an event converts to EUR at the latest `ref.fx_rate` at or before its time. The ECB daily reference rate (`ecb_daily`, fetched by risk-engine) is the daily rate. Intraday, the risk-engine turns trades of EUR FX pairs in reference data (`fa_warehouse::fx::EurCross` / `load_eur_crosses`, today Yahoo `EURUSD=X`, keyless, per minute) into rates under the feed's source id. It drops an intraday rate more than `fx.intraday_max_dev` (5 %) away from the latest daily rate as a bad tick. It values and books fills with these rates and writes them to `ref.fx_rate` every 60 s, so `fill_valued` (which joins on the fill's own `fx_source`) finds the rate a fill used.
- **FX source policy** (`pg_accounting::ACCOUNTING_FX_SOURCE` = `ecb_daily`): a fill is booked at the rate the risk-engine used at fill time (intraday allowed) and keeps that rate and source on its row; nothing re-derives it. Every rate the *ledger* reads from `ref.fx_rate` is the ECB daily rate only: valuation marks for statements, equity and P&L (`fx_rates_at` / `marks_for`), orphan booking (`rate_to_eur`) and FX restatements (`plan_fx_restatement`). Statements and revaluations are therefore reproducible and never switch to intraday rows. Test: `restate::tests::ledger_fx_reads_use_the_accounting_source_only` (a newer `yahoo` row is ignored). Currencies without an EUR pair in reference data use the daily rate intraday; adding a Yahoo `EURxxx=X` symbol to `config/sources.yaml` covers another one without code. The ledger has no market-data access: live mark-to-market prices are the caller's `Marks`; without marks the latest fill price per instrument is used.
- **P&L**: realised (lots per venue/instrument: FIFO by default, average cost selectable), unrealised, fees, funding, borrow, other, and FX P&L, each reported separately. The EUR P&L of a lot splits as `q·(P₁X₁ − P₀X₀) = q·(P₁ − P₀)·X₁` (price part, at the closing or mark rate) `+ q·P₀·(X₁ − X₀)` (FX part), plus the revaluation of non-EUR cash. With that split, `equity − external flows = net P&L` holds exactly.
- **Attribution**: netted fills are split to strategies per spec 40 §6.3. Strategy P&L is attributed fills plus virtual mark-to-market. `strategy_pnls` returns each strategy UUID plus `unattributed`, which carries unattributed trades and book-level cash items, so the values sum to the book.
- **Reconciliation**: `reconcile_cut(book, ours, in_flight, cash_tol, qty_tol)` compares additive totals (cash, fees, quantity per instrument) at the in-memory book's cut, in exact decimals: ours minus the fills whose write is still in flight, against Postgres restricted to the rows ours has applied. This removes the snapshot race behind the 2026-11-10/11 `broker` halts. Rows only Postgres has (payouts, operator-booked orphans) are returned and applied by the risk-engine (`ledger.synced`). Fills the memory booked that are missing in Postgres after the 60 s grace are reported and halt `broker`. Risk-engine books only `order.*.<account>`. Small cash diffs on a real venue will need a tolerance and auto-adjustment (broker connector).
- **Orphans**: a fill for an unknown client order id is parked in `ledger.orphan_fill` and audited instead of failing. It takes its instrument and currency from the event or from `ref.listing` as of the fill time (`resolved | unresolved | ambiguous`), and keeps the fee in its own currency. Booking refuses (409) instead of assuming a missing rate. Redelivery after the order is known books it. Any open orphan halts `broker` (raised, never auto-cleared).
- **Payout quota** (`payout::PayoutRule`, `[payout]` in `risk.toml`): `fixed_eur` (default 0), `pct_equity` or `pct_profit`, optional `rollover`, capped at `max_equity_frac` (0.05) of the previous close, rounded down to the cent. Each UTC day the risk-engine raises today's `payout_quota.quota_eur` (never lowers it) and the quota is reserved (deployable = equity − reserve). Booking the payout (`record_payout`, one `cash_movement(kind='payout', ref=day)` per book and day) is an operator action.
- **Settlement currency**: EUR by default per venue (`settlement`). `ledger.venue_settlement = native` books a venue's trade cash in the fill currency (`TradeEvent.settle_native`), revalued as `fx_cash_eur`; fees stay in EUR. `margin_model` is recorded, not interpreted.
- **Statements**: `statement::Statement` (opening/closing cash, foreign cash, positions and equity, fills with native and EUR values, cash movements, lot closings, P&L, payouts, external flows and an `unexplained_eur` check; JSON and `to_csv`).
- **Realised-gains report** (`tax`): per disposal (lot closing) the tax year, venue, instrument, side (long/short), quantity, acquisition and disposal time, holding days, native prices and rates, proceeds, cost basis, fees and gain in EUR at the rates booked on the fills. Aggregates per tax year and instrument and per tax year (gains, losses, net), plus non-capital cash movements per tax year (funding, borrow, fees, dividends/interest/other; deposits, withdrawals and payouts are excluded). Lots are matched per `(venue, instrument)` over the whole history by `TaxLotMethod` (`fifo` default, `lifo`, `average`), so a year's disposals carry their real cost basis. The tax year starts on a configurable `MM-DD` (UTC dates). Fees are allocated pro rata by quantity: the opening fee goes into the cost basis and the closing fee is set against the proceeds; `gain = proceeds − cost − fees`. A short lot is realised at the buy-back (proceeds = the opening sale). Before fees the gain equals the ledger's realised price + FX parts under the same lot method (tested, and on the live book).
- **FX restatement** (`restate`): `plan_fx_restatement` (a dry run that writes nothing) selects a book's fills of one instrument still booked in `from_ccy`, assigns the instrument's quote currency and the valid-time rate, and replays the book with and without the change. `apply_fx_restatement` rewrites exactly the planned rows in one transaction (optimistic; a stale plan is refused), keeps the old values in `ledger.fill_restatement`, audits `ledger.restatement` and rebuilds the lot cache; quantities, prices and fees never change. After an apply the risk-engine must be restarted so its in-memory book reloads.
- **Exact money**: the in-memory `Ledger` holds `ExactBook` (its `f64` fields are read-only views for the risk arithmetic and JSON). `ExactFill::of` / `cash_flow()` are shared by booking and reconciliation. The derived analytics compute in `Decimal` (products rounded to 18 places by `money::mul`), so the derived cash equals the book of record digit for digit. JSON keeps numbers via the `money::json*` adapters. Rows written before the decimal switch went through `float8::numeric` (15 significant digits) and are read as stored.
- **Audit log**: every status change, manual action, limit change, promotion, kill switch, reconciliation adjustment, payout, orphan resolution, restatement and config reload is appended; it is never updated or deleted.

## 7. Tunables
| Name | Default | Notes |
|---|---|---|
| `recon.interval` | 300 s (`reconcile_secs`) | the orphan check also runs at startup |
| `recon.cash_tolerance_eur` | 0 (exact) | a broker connector will need a tolerance and auto-adjustment |
| `recon.in_flight_grace` | 60 s | unconfirmed fill writes younger than this are excluded from the cut |
| `fill.retry_after` | 5 s | unconfirmed fill writes are retried |
| `ledger.settlement` | `eur` per venue | `ledger.venue_settlement`; `native` books fill cash in the fill currency |
| `ledger.lot_method` | `fifo` | `fifo` \| `average`; `PgLedger::with_lot_method` |
| tax report lot method | `fifo` | `fifo` \| `lifo` \| `average` (`TaxReportOptions.method`, `?method=`) |
| tax year start | `01-01` | `MM-DD` (`TaxReportOptions.year_start`, `?year_start=`) |
| `fx.daily_source` | `ecb_daily` | `risk.toml [fx] source` |
| `fx.intraday` | on | `risk.toml [fx] intraday`: EUR FX pairs in reference data (Yahoo `EURUSD=X`) |
| `fx.intraday_max_dev` | 0.05 | intraday rate dropped beyond this deviation from the daily rate |
| `fx.max_age_hours` | 72 | older rate → `stale` |
| `fx.block_stale_ccy` | on | stale currency → its instruments are reduce-only (§8) |
| `fx.halt_on_stale` | off | additionally the book-wide reduce-only `fx` halt |
| `payout.rule` | `fixed_eur`, 0 EUR | `[payout]` in `risk.toml` |

## 8. Failure modes
| Failure | Response |
|---|---|
| Duplicate fill event | rejected by the unique constraint, no-op |
| Fill for an unknown client order id | parked in `orphan_fill` (audited), venue halted (`broker`), alert |
| Postgres unavailable | the broker-gateway fails closed on a missing risk-engine heartbeat (15 s, D-079); unconfirmed fill writes are retried by the risk-engine; events buffered in JetStream |
| Reconciliation diff | venue halted until resolved; the diff is shown in the UI (`fills_missing_in_postgres` names the fills) |
| Fill write fails | the fill stays unconfirmed in the engine and is retried every 5 s; after 60 s it counts as missing (diff → halt) |
| Fill event for an intent unknown to the engine | intent recovered from Postgres and the fill booked in both books; no intent anywhere → orphan → halt |
| Ledger row written by another service (payout, operator-booked orphan) | applied to the paper book on the next reconciliation (audited `ledger.synced`), not a diff |
| Missing or stale FX rate | the latest known rate is used and flagged `stale` (older than `fx.max_age_hours`; `fx.stale_now`). With `fx.block_stale_ccy` (default) risk-increasing orders in that currency are blocked: allocator targets on instruments quoted in it are held reduce-only (`risk.event fx.stale_block`, counter `fx.blocked`) and foreign orders adding risk in it are rejected pre-trade; other currencies trade on. The block lifts as soon as a fresh daily or intraday rate arrives. `fx.halt_on_stale` additionally raises the book-wide reduce-only `fx` halt |
| Bad intraday FX tick | dropped when more than `fx.intraday_max_dev` from the daily rate (`fx.intraday_rejected`) |

## 9. Acceptance criteria (all met in code)
- Replaying the same venue event stream twice yields identical ledger state: `accounting::tests::replay_is_order_independent_and_idempotent`, live `derived_accounting_round_trip`, and `verify_existing_book_read_only`.
- Positions derived from events equal the latest position snapshot: `PgLedger::snapshot_check` (0 diffs on `paper`).
- Every fill has native and EUR values with FX source and timestamp: `ledger.fill_valued`, `fill_valuation_gaps` = 0 on `paper` (the timestamp is the `ref.fx_rate` validity time, or the fill time for EUR/`fixed`; intraday rates are written to `ref.fx_rate` for the same reason).
- Reconciliation detects an injected mismatch and halts the venue: `pg::tests::reconcile_cut_has_no_snapshot_race_and_finds_real_gaps`; the risk-engine halts scope `broker`, and open orphans halt too (paper venue; a real venue needs a broker connector).
- The audit log contains an entry for every manual API action: ledger-side manual actions (orphan resolution, payouts, restatements) are audited; whole-API coverage is spec 50's.
- The sum of strategy-attributed P&L equals portfolio P&L (minus unattributed manual trades): `strategy_pnls` incl. `unattributed` sums to `book_pnl` (unit test over FIFO/average and several ranges).

**Operator action, not implementation:** on the live `paper` book the last criterion (and `verify_existing_book_read_only`) holds only after the planned FX restatement of instrument `9daa8cea…` is applied. Its 1 080 fills from 2026-11-10 19:48:59 to 2026-11-11 01:48:18 were booked as `EUR` at 1.0 before the instrument's USD quote currency was known. The dry run moves them EUR → USD at 0.891742464776173 (`ecb_daily`, valid 2026-11-10): cash and equity +8.054251 EUR, and the 24 h strategy − book gap goes from −214.373064 to 0.000000. To apply: `POST /api/trading/restatements/fx` with `apply` + `confirm` + a reason (or the `restate_fx` example with `--apply`), then restart the risk-engine and re-run the verification.

## 10. Open questions and user decisions
1. Payout quota rule: **decided** (`payout`, default `fixed_eur` 0). The real amounts are the user's call.
2. Tax reporting: the jurisdiction-neutral realised-gains report exists (§6). **User decision:** which jurisdiction applies and its rules (loss pools and carry-forward, holding-period exemptions, wash-sale / bed-and-breakfast matching, cross-venue pooling, deemed-return regimes such as Dutch box 3, record retention). They are applied on top of the report's per-disposal facts, not in the ledger.
3. Margin accounting model per venue (cross vs isolated): out of scope until a live broker connector exists (`margin_model` is recorded only).
4. Ledger for paper strategies in the allocator tournament: answered by virtual books (D-064); there is one ledger book (`paper`).
5. Settlement currency per venue: **decided** (EUR by default, `native` per venue).
6. Money as `f64`: **resolved** (exact decimals end to end; only the wire and the risk arithmetic's read-only views are `f64`).

## 11. Deferred
- Fee schedules per venue (filled per connector spec).
- Multi-entity / multi-account consolidation.
- Export to accounting software (the statement and tax report CSV/JSON are the hand-off formats).

## Owns
Ledger repositories, derived accounting, the tax report and reconciliation logic.
