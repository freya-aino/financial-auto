-- Organisational accounting (fa-accounting, D-104): the double-entry books above the trading
-- ledger. Append-only journal; balance checks live in the crate (entries are validated balanced
-- before insert, and the insert of an entry with its postings is one transaction).
CREATE SCHEMA IF NOT EXISTS acct;

CREATE TABLE acct.account (
    id           text PRIMARY KEY,
    name         text NOT NULL,
    kind         text NOT NULL CHECK (kind IN ('asset', 'liability', 'equity', 'income', 'expense')),
    segment      text NOT NULL,
    external_ref text,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE acct.entry (
    id         uuid PRIMARY KEY,
    ts         timestamptz NOT NULL,
    entry_ref  text NOT NULL UNIQUE,
    memo       text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE acct.posting (
    entry_id uuid NOT NULL REFERENCES acct.entry (id) ON DELETE CASCADE,
    seq      smallint NOT NULL,
    account  text NOT NULL REFERENCES acct.account (id),
    amount   numeric(38, 18) NOT NULL CHECK (amount <> 0),
    PRIMARY KEY (entry_id, seq)
);

CREATE INDEX posting_account_idx ON acct.posting (account);
CREATE INDEX entry_ts_idx ON acct.entry (ts);
