//! Organisational accounting above the trading book (code_review_15_11_26.md §4, D-104).
//!
//! `fa-ledger` stays the book of record for trading events (fills, cash movements); this crate is
//! the double-entry ledger of the *organisation around it*: where revenue goes, what running the
//! system costs, and how funds are segmented and allocated. It answers "who owns which euro"
//! where the trading ledger answers "what did trading do".
//!
//! - **Chart of accounts** ([`Chart`]): asset / liability / equity / income / expense accounts,
//!   each in an organisational [`segment`] (trading, infrastructure, operations, owner,
//!   treasury) and optionally carrying an external reference (`ledger:paper`, a bank IBAN, a
//!   cloud account id) that says which real-world balance the account mirrors.
//! - **Journal** ([`Journal`], [`Entry`]): append-only balanced entries (debit +, credit −; an
//!   entry's postings sum to zero), idempotent by `entry_ref`, with per-account and per-segment
//!   balances and a zero trial balance as an invariant.
//! - **Revenue split** ([`split::SplitPolicy`]): fractions of a period's positive net income to
//!   the owner (payable), the infrastructure fund and reserves; the remainder stays retained.
//! - **Period close** ([`close::close_period`]): books trading revenue from the ledger's net
//!   P&L, accrues priced operating costs against the infrastructure fund, closes income and
//!   expenses to retained earnings and allocates by the policy — all as deterministic entries
//!   (`close:<period>:<step>` refs), so a re-run posts nothing twice.
//! - **Postgres** ([`pg::PgAccounting`]): the same journal persisted in the `acct` schema
//!   (migration `20261115100000`), amounts `NUMERIC(38,18)` like the trading ledger.
//!
//! Amounts are EUR at the ledger scale (`fa_ledger::money`, 18 decimal places). Jurisdiction
//! and external tax treatment stay out of scope, as for `fa-ledger`.

pub mod close;
pub mod pg;
pub mod split;

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
pub use fa_ledger::money::{self, Decimal, dec};
use uuid::Uuid;

/// Namespace for deterministic entry ids: `Uuid::new_v5(NAMESPACE, entry_ref)`.
pub const NAMESPACE: Uuid = Uuid::from_bytes(*b"fa-accounting\0\0\0");

/// Organisational segments of the default chart. A chart may add its own.
pub mod segment {
    pub const TRADING: &str = "trading";
    pub const INFRASTRUCTURE: &str = "infrastructure";
    pub const OPERATIONS: &str = "operations";
    pub const OWNER: &str = "owner";
    pub const TREASURY: &str = "treasury";
}

/// Account ids of the default chart.
pub mod accounts {
    /// Mirror of the trading book's equity (external ref `ledger:paper`).
    pub const TRADING_BOOK: &str = "assets:trading-book";
    /// Funds allocated to run and grow the infrastructure; opex accrues against it.
    pub const INFRA_FUND: &str = "assets:infra-fund";
    /// Allocated but undeployed reserves.
    pub const RESERVES: &str = "assets:reserves";
    pub const OPERATING_CASH: &str = "assets:operating-cash";
    /// Declared but unpaid owner distributions.
    pub const OWNER_PAYABLE: &str = "liabilities:owner-payable";
    pub const CONTRIBUTED: &str = "equity:contributed";
    pub const RETAINED: &str = "equity:retained";
    pub const TRADING_INCOME: &str = "income:trading";
    pub const INFRA_EXPENSE: &str = "expenses:infrastructure";
    pub const DATA_EXPENSE: &str = "expenses:data";
    pub const OPERATIONS_EXPENSE: &str = "expenses:operations";
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Asset,
    Liability,
    Equity,
    Income,
    Expense,
}

impl Kind {
    /// +1 where a debit (positive raw) balance is the normal state, −1 where a credit is.
    pub fn normal_sign(self) -> i64 {
        match self {
            Kind::Asset | Kind::Expense => 1,
            Kind::Liability | Kind::Equity | Kind::Income => -1,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Asset => "asset",
            Kind::Liability => "liability",
            Kind::Equity => "equity",
            Kind::Income => "income",
            Kind::Expense => "expense",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "asset" => Kind::Asset,
            "liability" => Kind::Liability,
            "equity" => Kind::Equity,
            "income" => Kind::Income,
            "expense" => Kind::Expense,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Account {
    pub id: String,
    pub name: String,
    pub kind: Kind,
    pub segment: String,
    /// A real-world balance this account mirrors (`ledger:paper`, `bank:<iban>`, a provider
    /// account id); reconciliation compares against it.
    pub external_ref: Option<String>,
}

/// The chart of accounts: validated, append-only (removing an account would orphan postings).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Chart {
    accounts: BTreeMap<String, Account>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AcctError {
    #[error("account {0}: {1}")]
    Account(String, &'static str),
    #[error("unknown account {0}")]
    UnknownAccount(String),
    #[error("entry {0}: {1}")]
    Entry(String, &'static str),
}

impl Chart {
    pub fn add(&mut self, a: Account) -> Result<(), AcctError> {
        if a.id.is_empty() || a.segment.is_empty() {
            return Err(AcctError::Account(a.id, "empty id or segment"));
        }
        if self.accounts.contains_key(&a.id) {
            return Err(AcctError::Account(a.id, "duplicate id"));
        }
        self.accounts.insert(a.id.clone(), a);
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&Account> {
        self.accounts.get(id)
    }

    pub fn accounts(&self) -> impl Iterator<Item = &Account> {
        self.accounts.values()
    }

    /// The default organisational chart (account ids in [`accounts`]).
    pub fn default_chart() -> Self {
        let mut c = Chart::default();
        let mut add = |id: &str, name: &str, kind, seg: &str, ext: Option<&str>| {
            c.add(Account {
                id: id.to_owned(),
                name: name.to_owned(),
                kind,
                segment: seg.to_owned(),
                external_ref: ext.map(str::to_owned),
            })
            .expect("default chart is valid");
        };
        use segment::*;
        add(accounts::TRADING_BOOK, "Trading book equity", Kind::Asset, TRADING, Some("ledger:paper"));
        add(accounts::INFRA_FUND, "Infrastructure fund", Kind::Asset, INFRASTRUCTURE, None);
        add(accounts::RESERVES, "Reserves", Kind::Asset, TREASURY, None);
        add(accounts::OPERATING_CASH, "Operating cash", Kind::Asset, OPERATIONS, None);
        add(accounts::OWNER_PAYABLE, "Owner distributions payable", Kind::Liability, OWNER, None);
        add(accounts::CONTRIBUTED, "Contributed capital", Kind::Equity, TREASURY, None);
        add(accounts::RETAINED, "Retained earnings", Kind::Equity, TREASURY, None);
        add(accounts::TRADING_INCOME, "Trading revenue", Kind::Income, TRADING, None);
        add(accounts::INFRA_EXPENSE, "Infrastructure opex", Kind::Expense, INFRASTRUCTURE, None);
        add(accounts::DATA_EXPENSE, "Data and API opex", Kind::Expense, INFRASTRUCTURE, None);
        add(accounts::OPERATIONS_EXPENSE, "Operations opex", Kind::Expense, OPERATIONS, None);
        c
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Posting {
    pub account: String,
    /// Debit positive, credit negative, EUR at the ledger scale.
    #[serde(with = "fa_ledger::money::json")]
    pub amount: Decimal,
}

/// A balanced journal entry. `entry_ref` is the idempotency key and determines `id`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Entry {
    pub id: Uuid,
    pub ts: DateTime<Utc>,
    pub entry_ref: String,
    pub memo: String,
    pub postings: Vec<Posting>,
}

impl Entry {
    /// Validates: a ref, at least two postings, no zero posting, debits = credits exactly.
    pub fn new(
        ts: DateTime<Utc>,
        entry_ref: &str,
        memo: &str,
        postings: Vec<(&str, Decimal)>,
    ) -> Result<Self, AcctError> {
        if entry_ref.is_empty() {
            return Err(AcctError::Entry(entry_ref.into(), "empty ref"));
        }
        if postings.len() < 2 {
            return Err(AcctError::Entry(entry_ref.into(), "fewer than two postings"));
        }
        if postings.iter().any(|(_, a)| a.is_zero()) {
            return Err(AcctError::Entry(entry_ref.into(), "zero posting"));
        }
        if !money::sum(postings.iter().map(|(_, a)| *a)).is_zero() {
            return Err(AcctError::Entry(entry_ref.into(), "postings do not balance"));
        }
        Ok(Self {
            id: Uuid::new_v5(&NAMESPACE, entry_ref.as_bytes()),
            ts,
            entry_ref: entry_ref.to_owned(),
            memo: memo.to_owned(),
            postings: postings
                .into_iter()
                .map(|(acc, amount)| Posting { account: acc.to_owned(), amount: money::round(amount) })
                .collect(),
        })
    }
}

/// Balances of one account in its normal orientation, plus the raw (debit-positive) value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Balance {
    #[serde(with = "fa_ledger::money::json")]
    pub raw: Decimal,
    /// `raw × normal_sign`: what a statement prints (an asset's holdings, a liability owed…).
    #[serde(with = "fa_ledger::money::json")]
    pub normal: Decimal,
}

/// In-memory journal over a chart: the unit the Postgres layer persists and rebuilds.
#[derive(Debug, Clone, Default)]
pub struct Journal {
    chart: Chart,
    entries: Vec<Entry>,
    refs: BTreeMap<String, Uuid>,
    raw: BTreeMap<String, Decimal>,
}

impl Journal {
    pub fn new(chart: Chart) -> Self {
        Self { chart, ..Default::default() }
    }

    pub fn chart(&self) -> &Chart {
        &self.chart
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Posts an entry; `Ok(false)` when its ref was already posted (idempotent re-run).
    /// A ref collision with different content is an error, never a silent skip.
    pub fn post(&mut self, e: Entry) -> Result<bool, AcctError> {
        if let Some(prev) = self.refs.get(&e.entry_ref) {
            let same = self.entries.iter().any(|x| x.id == *prev && x.postings == e.postings);
            return if same {
                Ok(false)
            } else {
                Err(AcctError::Entry(e.entry_ref, "ref reused with different postings"))
            };
        }
        for p in &e.postings {
            if self.chart.get(&p.account).is_none() {
                return Err(AcctError::UnknownAccount(p.account.clone()));
            }
        }
        for p in &e.postings {
            *self.raw.entry(p.account.clone()).or_default() += p.amount;
        }
        self.refs.insert(e.entry_ref.clone(), e.id);
        self.entries.push(e);
        Ok(true)
    }

    pub fn balance(&self, account: &str) -> Balance {
        let raw = self.raw.get(account).copied().unwrap_or_default();
        let sign = self.chart.get(account).map_or(1, |a| a.kind.normal_sign());
        Balance { raw, normal: raw * Decimal::from(sign) }
    }

    /// Normal balances of every account that has postings.
    pub fn balances(&self) -> BTreeMap<String, Balance> {
        self.raw.keys().map(|id| (id.clone(), self.balance(id))).collect()
    }

    /// Net funds (assets − liabilities) per segment: the organisational split of the money.
    pub fn segment_funds(&self) -> BTreeMap<String, Decimal> {
        let mut out = BTreeMap::new();
        for (id, raw) in &self.raw {
            if let Some(a) = self.chart.get(id)
                && matches!(a.kind, Kind::Asset | Kind::Liability)
            {
                *out.entry(a.segment.clone()).or_default() += *raw;
            }
        }
        out
    }

    /// Sum of raw balances: zero for every well-formed journal.
    pub fn trial_balance(&self) -> Decimal {
        money::sum(self.raw.values().copied())
    }

    /// Income − expenses still open (not yet closed to retained earnings).
    pub fn open_net_income(&self) -> Decimal {
        let mut net = Decimal::ZERO;
        for (id, raw) in &self.raw {
            match self.chart.get(id).map(|a| a.kind) {
                Some(Kind::Income) => net -= *raw,
                Some(Kind::Expense) => net -= *raw,
                _ => {}
            }
        }
        net
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts() -> DateTime<Utc> {
        chrono::DateTime::parse_from_rfc3339("2026-11-15T00:00:00Z").unwrap().into()
    }

    #[test]
    fn chart_rejects_duplicates_and_empty_ids() {
        let mut c = Chart::default_chart();
        let dup = c.get(accounts::RESERVES).unwrap().clone();
        assert_eq!(c.add(dup), Err(AcctError::Account(accounts::RESERVES.into(), "duplicate id")));
        let bad =
            Account { id: String::new(), name: "x".into(), kind: Kind::Asset, segment: "t".into(), external_ref: None };
        assert!(c.add(bad).is_err());
        assert_eq!(Chart::default_chart().accounts().count(), 11);
    }

    #[test]
    fn entries_must_balance_exactly() {
        assert!(Entry::new(ts(), "r", "m", vec![(accounts::TRADING_BOOK, dec(1.0))]).is_err());
        assert!(
            Entry::new(ts(), "r", "m", vec![(accounts::TRADING_BOOK, dec(0.1)), (accounts::TRADING_INCOME, dec(-0.2))])
                .is_err()
        );
        assert!(
            Entry::new(ts(), "", "m", vec![(accounts::TRADING_BOOK, dec(1.0)), (accounts::TRADING_INCOME, dec(-1.0))])
                .is_err()
        );
        // 0.1 + 0.2 = 0.3 exactly in decimal: the f64 classic must balance
        let e = Entry::new(
            ts(),
            "r",
            "m",
            vec![
                (accounts::TRADING_BOOK, dec(0.1)),
                (accounts::TRADING_BOOK, dec(0.2)),
                (accounts::TRADING_INCOME, dec(-0.3)),
            ],
        )
        .unwrap();
        assert_eq!(e.id, Uuid::new_v5(&NAMESPACE, b"r"));
    }

    #[test]
    fn journal_posts_idempotently_and_keeps_the_trial_balance() {
        let mut j = Journal::new(Chart::default_chart());
        let e = Entry::new(
            ts(),
            "seed",
            "initial capital",
            vec![(accounts::TRADING_BOOK, dec(100_000.0)), (accounts::CONTRIBUTED, dec(-100_000.0))],
        )
        .unwrap();
        assert!(j.post(e.clone()).unwrap());
        assert!(!j.post(e.clone()).unwrap(), "same ref, same postings: skipped");
        let mut tampered = e.clone();
        tampered.postings[0].amount = dec(1.0);
        tampered.postings[1].amount = dec(-1.0);
        assert!(j.post(tampered).is_err(), "same ref, different postings: refused");
        assert_eq!(j.balance(accounts::TRADING_BOOK).normal, dec(100_000.0));
        assert_eq!(j.balance(accounts::CONTRIBUTED).normal, dec(100_000.0), "equity in its normal orientation");
        assert_eq!(j.trial_balance(), Decimal::ZERO);
        assert_eq!(j.segment_funds()[segment::TRADING], dec(100_000.0));
    }

    #[test]
    fn unknown_accounts_are_refused() {
        let mut j = Journal::new(Chart::default_chart());
        let e = Entry::new(ts(), "x", "m", vec![("assets:nope", dec(1.0)), (accounts::RETAINED, dec(-1.0))]).unwrap();
        assert_eq!(j.post(e), Err(AcctError::UnknownAccount("assets:nope".into())));
        assert_eq!(j.trial_balance(), Decimal::ZERO, "a refused entry changes nothing");
    }
}
