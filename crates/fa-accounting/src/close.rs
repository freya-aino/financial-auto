//! Period close (code_review_15_11_26.md §4.2): the bridge from the trading ledger's numbers to
//! the organisational books. For one period (a day `2026-11-14` or a month `2026-11`) it books,
//! as deterministic entries idempotent by ref:
//!
//! 1. `close:<p>:revenue` — the trading book's net P&L for the period as trading revenue
//!    (a loss books the same entry with the signs flipped);
//! 2. `close:<p>:opex:<account>` — each priced operating cost, accrued **against the
//!    infrastructure fund** (the fund pays for the system; a fund driven negative is visible
//!    underfunding, not an error);
//! 3. `close:<p>:net` — income and expenses close to retained earnings;
//! 4. on positive net income, the [`SplitPolicy`] allocation:
//!    - `close:<p>:distribution` — the owner's share declared as a payable
//!      (paid later via [`payout_paid`], matching the operator's payout booking in `fa-ledger`);
//!    - `close:<p>:funding` — the infra and reserve shares moved out of the trading book.
//!
//! The caller feeds the revenue number from `fa_ledger` (`PnlBreakdown::net_eur` over the
//! period) and the opex items from priced meters (§4.2 step 1); pricing itself lives with the
//! caller, not here.

use chrono::{DateTime, Utc};

use crate::{
    AcctError, Decimal, Entry, accounts,
    split::{Allocation, SplitPolicy},
};

/// A priced operating cost: `account` is an expense account of the chart
/// (e.g. [`accounts::INFRA_EXPENSE`]), `amount` a positive EUR cost.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpexItem {
    pub account: String,
    #[serde(with = "fa_ledger::money::json")]
    pub amount: Decimal,
    pub memo: String,
}

/// The close's result: the entries to post (any subset may already exist — posting is
/// idempotent by ref) and the allocation that produced them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeriodClose {
    pub period: String,
    pub entries: Vec<Entry>,
    pub allocation: Allocation,
}

/// Books one period. `revenue_eur` is the trading book's net P&L for the period (already net of
/// trading fees, funding and borrow — those live inside the trading ledger's breakdown; opex here
/// is the cost of the system itself, which the trading book does not know about).
pub fn close_period(
    period: &str,
    ts: DateTime<Utc>,
    revenue_eur: Decimal,
    opex: &[OpexItem],
    policy: &SplitPolicy,
) -> Result<PeriodClose, AcctError> {
    if period.is_empty() || period.contains(':') {
        return Err(AcctError::Entry(period.into(), "period must be non-empty and colon-free"));
    }
    policy.validate().map_err(|_| AcctError::Entry(period.into(), "invalid split policy"))?;
    let mut entries = Vec::new();
    let mut net = Decimal::ZERO;

    if !revenue_eur.is_zero() {
        entries.push(Entry::new(
            ts,
            &format!("close:{period}:revenue"),
            &format!("trading net P&L {period}"),
            vec![(accounts::TRADING_BOOK, revenue_eur), (accounts::TRADING_INCOME, -revenue_eur)],
        )?);
        net += revenue_eur;
    }

    for item in opex {
        if item.amount <= Decimal::ZERO {
            return Err(AcctError::Entry(item.account.clone(), "opex must be positive"));
        }
        entries.push(Entry::new(
            ts,
            &format!("close:{period}:opex:{}", item.account),
            &item.memo,
            vec![(item.account.as_str(), item.amount), (accounts::INFRA_FUND, -item.amount)],
        )?);
        net -= item.amount;
    }

    if !net.is_zero() {
        // Close exactly what this period booked above, so an unrelated open balance on the
        // income accounts (an aborted earlier run) can never leak into this period's net.
        let mut postings = vec![(accounts::RETAINED, -net)];
        if !revenue_eur.is_zero() {
            postings.push((accounts::TRADING_INCOME, revenue_eur));
        }
        for item in opex {
            postings.push((item.account.as_str(), -item.amount));
        }
        entries.push(Entry::new(ts, &format!("close:{period}:net"), &format!("close {period} to retained"), postings)?);
    }

    let allocation = policy.allocate(net);
    if allocation.owner > Decimal::ZERO {
        entries.push(Entry::new(
            ts,
            &format!("close:{period}:distribution"),
            &format!("owner distribution declared for {period}"),
            vec![(accounts::RETAINED, allocation.owner), (accounts::OWNER_PAYABLE, -allocation.owner)],
        )?);
    }
    let funded = allocation.infra + allocation.reserves;
    if funded > Decimal::ZERO {
        let mut postings = vec![(accounts::TRADING_BOOK, -funded)];
        if allocation.infra > Decimal::ZERO {
            postings.push((accounts::INFRA_FUND, allocation.infra));
        }
        if allocation.reserves > Decimal::ZERO {
            postings.push((accounts::RESERVES, allocation.reserves));
        }
        entries.push(Entry::new(
            ts,
            &format!("close:{period}:funding"),
            &format!("infra and reserve funding for {period}"),
            postings,
        )?);
    }

    Ok(PeriodClose { period: period.to_owned(), entries, allocation })
}

/// The entry for an owner payout actually paid out of the trading book: the counterpart of the
/// operator's `POST /api/trading/payouts` cash movement in `fa-ledger`. `payout_ref` should name
/// that movement's ref so the two books reconcile row for row.
pub fn payout_paid(ts: DateTime<Utc>, payout_ref: &str, amount: Decimal) -> Result<Entry, AcctError> {
    if amount <= Decimal::ZERO {
        return Err(AcctError::Entry(payout_ref.into(), "payout must be positive"));
    }
    Entry::new(
        ts,
        &format!("payout:{payout_ref}"),
        "owner payout paid from the trading book",
        vec![(accounts::OWNER_PAYABLE, amount), (accounts::TRADING_BOOK, -amount)],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Chart, Journal, dec, money};

    fn ts() -> DateTime<Utc> {
        chrono::DateTime::parse_from_rfc3339("2026-11-15T00:00:00Z").unwrap().into()
    }

    fn opex(amount: f64) -> Vec<OpexItem> {
        vec![
            OpexItem { account: accounts::INFRA_EXPENSE.into(), amount: dec(amount), memo: "compute+storage".into() },
            OpexItem { account: accounts::DATA_EXPENSE.into(), amount: dec(1.5), memo: "api calls".into() },
        ]
    }

    fn post_all(j: &mut Journal, c: &PeriodClose) {
        for e in &c.entries {
            j.post(e.clone()).unwrap();
        }
    }

    #[test]
    fn a_profitable_period_splits_and_the_books_tie_out() {
        let policy = SplitPolicy::new(0.2, 0.5, 0.1).unwrap();
        let mut j = Journal::new(Chart::default_chart());
        let c = close_period("2026-11-14", ts(), dec(1000.0), &opex(8.5), &policy).unwrap();
        post_all(&mut j, &c);

        let net = dec(990.0); // 1000 − 8.5 − 1.5
        assert_eq!(c.allocation.owner, dec(198.0));
        assert_eq!(c.allocation.infra, dec(495.0));
        assert_eq!(c.allocation.reserves, dec(99.0));
        assert_eq!(c.allocation.retained, net - dec(198.0) - dec(495.0) - dec(99.0));
        assert_eq!(j.trial_balance(), money::Decimal::ZERO);
        assert_eq!(j.open_net_income(), money::Decimal::ZERO, "income and expenses are closed");
        // The infra fund received its share and paid the period's opex.
        assert_eq!(j.balance(accounts::INFRA_FUND).normal, dec(495.0) - dec(10.0));
        assert_eq!(j.balance(accounts::RESERVES).normal, dec(99.0));
        assert_eq!(j.balance(accounts::OWNER_PAYABLE).normal, dec(198.0));
        // The trading book mirror: revenue in, funding out; the payable is not yet paid.
        assert_eq!(j.balance(accounts::TRADING_BOOK).normal, dec(1000.0) - dec(594.0));
        // Retained = net − declared distribution.
        assert_eq!(j.balance(accounts::RETAINED).normal, net - dec(198.0));

        // Re-running the close posts nothing new.
        let again = close_period("2026-11-14", ts(), dec(1000.0), &opex(8.5), &policy).unwrap();
        for e in &again.entries {
            assert!(!j.post(e.clone()).unwrap());
        }

        // Paying the payout clears the payable out of the trading book.
        j.post(payout_paid(ts(), "payout/2026-11-15", dec(198.0)).unwrap()).unwrap();
        assert_eq!(j.balance(accounts::OWNER_PAYABLE).normal, money::Decimal::ZERO);
        assert_eq!(j.balance(accounts::TRADING_BOOK).normal, dec(1000.0) - dec(594.0) - dec(198.0));
        assert_eq!(j.trial_balance(), money::Decimal::ZERO);
    }

    #[test]
    fn a_loss_closes_to_retained_and_allocates_nothing() {
        let policy = SplitPolicy::new(0.5, 0.5, 0.0).unwrap();
        let mut j = Journal::new(Chart::default_chart());
        let c = close_period("2026-11-14", ts(), dec(-50.0), &opex(8.5), &policy).unwrap();
        post_all(&mut j, &c);
        assert_eq!(c.allocation.retained, dec(-60.0));
        assert_eq!(c.entries.len(), 4, "revenue, two opex, net close; no distribution or funding");
        assert_eq!(j.balance(accounts::RETAINED).normal, dec(-60.0));
        assert_eq!(j.balance(accounts::INFRA_FUND).normal, dec(-10.0), "visible underfunding");
        assert_eq!(j.trial_balance(), money::Decimal::ZERO);
    }

    #[test]
    fn empty_periods_and_bad_inputs() {
        let p = SplitPolicy::default();
        let c = close_period("2026-11-14", ts(), Decimal::ZERO, &[], &p).unwrap();
        assert!(c.entries.is_empty());
        assert!(close_period("", ts(), dec(1.0), &[], &p).is_err());
        assert!(close_period("a:b", ts(), dec(1.0), &[], &p).is_err(), "a colon would collide with the ref scheme");
        let neg = [OpexItem { account: accounts::INFRA_EXPENSE.into(), amount: dec(-1.0), memo: String::new() }];
        assert!(close_period("p", ts(), dec(1.0), &neg, &p).is_err());
        assert!(payout_paid(ts(), "r", Decimal::ZERO).is_err());
    }

    #[test]
    fn opex_only_period_still_closes() {
        let mut j = Journal::new(Chart::default_chart());
        let c = close_period("2026-11", ts(), Decimal::ZERO, &opex(8.5), &SplitPolicy::default()).unwrap();
        post_all(&mut j, &c);
        assert_eq!(j.balance(accounts::RETAINED).normal, dec(-10.0));
        assert_eq!(j.open_net_income(), Decimal::ZERO);
        assert_eq!(j.trial_balance(), Decimal::ZERO);
    }
}
