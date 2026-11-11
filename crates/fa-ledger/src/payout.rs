//! Daily payout quota rule (spec 42 §10 Q1, spec 40 Q1).
//!
//! **Decided default (2026-11-11):** the quota for UTC day `D` is computed once, at the first
//! check on `D`, from the book as of the start of `D`:
//!
//! ```text
//! base(D)  = fixed_eur                                         rule = fixed_eur
//!          | pct × equity(D−1 close)                           rule = pct_equity
//!          | pct × max(0, net P&L of D−1)                      rule = pct_profit
//! carry(D) = rollover ? max(0, quota(D−1) − paid(D−1)) : 0
//! quota(D) = min(base(D) + carry(D), max_equity_frac × equity(D−1 close))
//! ```
//!
//! rounded down to the cent. The quota is *reserved* (deployable capital = equity − reserve,
//! spec 40 §6.5); booking the payout itself (`cash_movement(kind = 'payout')`) stays an
//! operator action (`POST /api/trading/payouts`), because money leaving the account is an
//! external transfer the system cannot perform on its own. Writing the quota only ever raises
//! the stored row (tighten-only). Default rule: `fixed_eur` with 0 EUR — no payout reserved
//! until the operator sets a rule (D-041's `daily_payout_quota_eur = 0`).

use rust_decimal::{Decimal, RoundingStrategy};
use serde::{Deserialize, Serialize};

use crate::money::dec;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PayoutBase {
    /// A fixed EUR amount per day (`amount_eur`).
    #[default]
    FixedEur,
    /// `pct` of the previous day's closing equity.
    PctEquity,
    /// `pct` of the previous day's positive net P&L (realised + unrealised − costs).
    PctProfit,
}

/// The payout rule (`[payout]` in `risk.toml`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PayoutRule {
    pub rule: PayoutBase,
    /// EUR per day for `fixed_eur`.
    pub amount_eur: f64,
    /// Fraction (0.1 = 10 %) for `pct_equity` / `pct_profit`.
    pub pct: f64,
    /// Carry yesterday's unpaid quota into today.
    pub rollover: bool,
    /// Cap as a fraction of the previous close's equity.
    pub max_equity_frac: f64,
}

impl Default for PayoutRule {
    fn default() -> Self {
        Self { rule: PayoutBase::FixedEur, amount_eur: 0.0, pct: 0.0, rollover: false, max_equity_frac: 0.05 }
    }
}

/// What the rule needs from the ledger for day `D`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PayoutInputs {
    /// Equity at the close of `D−1` (EUR).
    pub prev_equity_eur: Decimal,
    /// Net P&L of `D−1` (EUR).
    pub prev_net_pnl_eur: Decimal,
    /// `quota(D−1)` and `paid(D−1)`.
    pub prev_quota_eur: Decimal,
    pub prev_paid_eur: Decimal,
}

impl PayoutRule {
    pub fn validate(&self) -> Result<(), String> {
        let ok = |x: f64| x.is_finite() && x >= 0.0;
        if !ok(self.amount_eur) || !ok(self.pct) || !ok(self.max_equity_frac) {
            return Err("payout: amount_eur, pct and max_equity_frac must be finite and ≥ 0".into());
        }
        if self.pct > 1.0 || self.max_equity_frac > 1.0 {
            return Err("payout: pct and max_equity_frac are fractions (≤ 1)".into());
        }
        Ok(())
    }

    /// `quota(D)` in EUR, rounded down to the cent, never negative.
    pub fn quota(&self, i: &PayoutInputs) -> Decimal {
        let zero = Decimal::ZERO;
        let base = match self.rule {
            PayoutBase::FixedEur => dec(self.amount_eur),
            PayoutBase::PctEquity => dec(self.pct) * i.prev_equity_eur.max(zero),
            PayoutBase::PctProfit => dec(self.pct) * i.prev_net_pnl_eur.max(zero),
        };
        let carry = if self.rollover { (i.prev_quota_eur - i.prev_paid_eur).max(zero) } else { zero };
        let cap = dec(self.max_equity_frac) * i.prev_equity_eur.max(zero);
        (base + carry).min(cap).max(zero).round_dp_with_strategy(2, RoundingStrategy::ToZero)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(eq: f64, pnl: f64, q: f64, paid: f64) -> PayoutInputs {
        PayoutInputs {
            prev_equity_eur: dec(eq),
            prev_net_pnl_eur: dec(pnl),
            prev_quota_eur: dec(q),
            prev_paid_eur: dec(paid),
        }
    }

    #[test]
    fn default_reserves_nothing() {
        assert_eq!(PayoutRule::default().quota(&inputs(100_000.0, 500.0, 0.0, 0.0)), Decimal::ZERO);
    }

    #[test]
    fn rules_rollover_and_cap() {
        let fixed = PayoutRule { amount_eur: 50.0, ..PayoutRule::default() };
        assert_eq!(fixed.quota(&inputs(100_000.0, -10.0, 0.0, 0.0)), dec(50.0));
        let pct_eq = PayoutRule { rule: PayoutBase::PctEquity, pct: 0.001, ..PayoutRule::default() };
        assert_eq!(pct_eq.quota(&inputs(99_246.137, 0.0, 0.0, 0.0)), dec(99.24));
        let profit = PayoutRule { rule: PayoutBase::PctProfit, pct: 0.5, rollover: true, ..PayoutRule::default() };
        assert_eq!(profit.quota(&inputs(100_000.0, -300.0, 0.0, 0.0)), Decimal::ZERO, "no profit, no payout");
        assert_eq!(profit.quota(&inputs(100_000.0, 300.0, 40.0, 10.0)), dec(180.0), "150 + 30 carried");
        let capped = PayoutRule { amount_eur: 10_000.0, max_equity_frac: 0.01, ..PayoutRule::default() };
        assert_eq!(capped.quota(&inputs(100_000.0, 0.0, 0.0, 0.0)), dec(1_000.0));
        assert!(PayoutRule { pct: 2.0, ..PayoutRule::default() }.validate().is_err());
        assert!(PayoutRule::default().validate().is_ok());
    }
}
