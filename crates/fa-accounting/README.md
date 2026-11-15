# fa-accounting — organisational books above the trading ledger

The double-entry ledger of the organisation around the trading system (D-104; design from the
2026-11-15 code review §4.2, since retired). [`fa-ledger`](../fa-ledger) stays
the book of record for trading events; this crate records where the money *goes*: revenue
recognition, the system's own operating costs, the revenue split, and the segmentation and
allocation of funds. All amounts are EUR at the ledger scale (`fa_ledger::money`, 18 dp).

## Model

- **Chart of accounts** (`Chart`, default in `Chart::default_chart()`): accounts of kind
  asset / liability / equity / income / expense, each in an organisational **segment**
  (`trading`, `infrastructure`, `operations`, `owner`, `treasury`) and optionally carrying an
  **external reference** naming the real-world balance it mirrors (`ledger:paper`, a bank IBAN,
  a cloud account id).
- **Journal** (`Journal` / `pg::PgAccounting`): append-only balanced entries (debit +, credit −,
  postings sum to zero exactly), **idempotent by `entry_ref`** — the same ref with the same
  postings is skipped, with different postings refused. Invariant: the trial balance is zero.
  Balances per account and net funds per segment fall out of the postings.
- **Revenue split** (`split::SplitPolicy`): fractions of a period's positive net income to the
  owner, the infrastructure fund and reserves; the residual stays retained, so rounding never
  creates or destroys money. Default: everything retained (splitting is an owner decision, like
  the payout quota's `fixed_eur = 0`).
- **Period close** (`close::close_period`): for one period, deterministic `close:<period>:*`
  entries book the trading book's net P&L as revenue, accrue priced opex **against the
  infrastructure fund**, close income/expenses to retained earnings and apply the split (owner
  share → `liabilities:owner-payable`; infra/reserve shares move out of the trading-book
  mirror). `close::payout_paid` mirrors the operator's payout cash movement in `fa-ledger`.
  Re-running a close posts nothing twice.

## Storage

Schema `acct` (`migrations/20261115100000_accounting.sql`): `acct.account`, `acct.entry`
(unique `entry_ref`), `acct.posting` (`NUMERIC(38,18)`, nonzero). control-api's embedded
migrations apply it. An account's `kind` is immutable in `sync_chart` — changing it would flip
the orientation of its history.

## Wiring (control-api)

- **Daily close** (`bins/control-api/src/accounting.rs`): every `FA_ACCT_CLOSE_SECS` (default
  86 400; `0` disables) control-api closes the previous UTC day of the `paper` book — net P&L
  from `PgLedger::book_pnl` as revenue, the day's metered costs priced by `[accounting]`
  (`fa_core::config::AccountingConfig`, all zero by default): flat `base_eur_per_day`
  (operations), GPU ms summed from `ops.job_run` (`SUM(bigint)` is `NUMERIC` — cast back to
  `bigint`) at `gpu_eur_per_hour`, VictoriaMetrics + Postgres bytes at 1/30 of
  `storage_eur_per_gib_month` per day (both to `expenses:infrastructure`); zero-priced or
  zero-usage meters produce no item. The first ever run seeds `equity:contributed` with the
  book's external capital (deposits − withdrawals), so the trading-book mirror tracks equity,
  not just P&L since the job existed. Each run is audited `ops.accounting_close`; a failed
  close is retried next interval.
- **Routes**: `GET /api/accounting/balances` (accounts, segment funds, trial balance),
  `GET /api/accounting/entries`, `POST /api/accounting/close` (the scheduled job's code path,
  so closing an already-closed day posts nothing twice). The frontend Infra page shows the
  Accounting panel.
- **Live numbers** (D-106, owner-confirmed): `[accounting]` in both platform tomls —
  `owner_frac 0.2`, `infra_frac 0.3`, `reserve_frac 0.0` (50 % retained); notional opex prices
  `base_eur_per_day 1.0`, `gpu_eur_per_hour 0.5`, `storage_eur_per_gib_month 0.02`
  (cloud-equivalent estimates, to be replaced by real invoices when they exist).

## Not here (by design)

- **Pricing the meters**: opex items arrive priced (`close::OpexItem`); the price table and the
  accrual queries live in the caller (control-api, above), per §4.2 step 1 of the review.
- `close::payout_paid` is not yet called by any service: an operator payout
  (`POST /api/trading/payouts`) is not mirrored into these books automatically yet.
- Jurisdiction, external tax, multi-currency books (EUR only, like the trading ledger's base).

## Tests

`just cargo test -p fa-accounting`. The Postgres round trip runs only with `DATABASE_URL` set
(migrations applied); it uses throwaway refs and deletes its own rows.
