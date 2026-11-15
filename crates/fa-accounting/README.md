# fa-accounting — organisational books above the trading ledger

The double-entry ledger of the organisation around the trading system (D-104;
[code_review_15_11_26.md](../../code_review_15_11_26.md) §4). [`fa-ledger`](../fa-ledger) stays
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

## Not here (by design)

- **Pricing the meters**: opex items arrive priced (`close::OpexItem`); the price table and the
  accrual job (reading `used_gpu_ms`, `vm_data_size_bytes`, stream bytes, …) belong to the
  caller (control-api), per §4.2 step 1 of the review.
- **Wiring**: no service calls this crate yet; the close job, config (`[accounting]` split
  fractions) and API/dashboard views are the follow-up step.
- Jurisdiction, external tax, multi-currency books (EUR only, like the trading ledger's base).

## Tests

`just cargo test -p fa-accounting`. The Postgres round trip runs only with `DATABASE_URL` set
(migrations applied); it uses throwaway refs and deletes its own rows.
