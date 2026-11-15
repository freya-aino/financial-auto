//! The revenue-split policy (code_review_15_11_26.md §4.2 step 2): fractions of a period's
//! positive net income to the owner (a declared distribution), the infrastructure fund and
//! reserves; whatever is left stays retained in the trading book. A period with zero or
//! negative net income allocates nothing — the loss stays in retained earnings.

use crate::{Decimal, dec, money};

/// Fractions in `[0, 1]` summing to at most 1. The default allocates nothing (all retained),
/// mirroring the payout quota's `fixed_eur = 0` default: splitting is an owner decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SplitPolicy {
    #[serde(with = "fa_ledger::money::json")]
    pub owner_frac: Decimal,
    #[serde(with = "fa_ledger::money::json")]
    pub infra_frac: Decimal,
    #[serde(with = "fa_ledger::money::json")]
    pub reserve_frac: Decimal,
}

impl Default for SplitPolicy {
    fn default() -> Self {
        Self { owner_frac: Decimal::ZERO, infra_frac: Decimal::ZERO, reserve_frac: Decimal::ZERO }
    }
}

/// EUR allocation of one period's net income; `owner + infra + reserves + retained` is exactly
/// the net income (the residual goes to retained, so rounding never creates or destroys money).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct Allocation {
    #[serde(with = "fa_ledger::money::json")]
    pub owner: Decimal,
    #[serde(with = "fa_ledger::money::json")]
    pub infra: Decimal,
    #[serde(with = "fa_ledger::money::json")]
    pub reserves: Decimal,
    #[serde(with = "fa_ledger::money::json")]
    pub retained: Decimal,
}

impl SplitPolicy {
    pub fn new(owner_frac: f64, infra_frac: f64, reserve_frac: f64) -> Result<Self, String> {
        let p = Self { owner_frac: dec(owner_frac), infra_frac: dec(infra_frac), reserve_frac: dec(reserve_frac) };
        p.validate()?;
        Ok(p)
    }

    pub fn validate(&self) -> Result<(), String> {
        let fs =
            [("owner_frac", self.owner_frac), ("infra_frac", self.infra_frac), ("reserve_frac", self.reserve_frac)];
        for (name, f) in fs {
            if f < Decimal::ZERO || f > Decimal::ONE {
                return Err(format!("{name} = {f} outside [0, 1]"));
            }
        }
        let sum = self.owner_frac + self.infra_frac + self.reserve_frac;
        if sum > Decimal::ONE {
            return Err(format!("fractions sum to {sum} > 1"));
        }
        Ok(())
    }

    /// Splits `net_income`. Zero or negative income: nothing is allocated, all stays retained.
    pub fn allocate(&self, net_income: Decimal) -> Allocation {
        if net_income <= Decimal::ZERO {
            return Allocation { retained: net_income, ..Default::default() };
        }
        let owner = money::mul(net_income, self.owner_frac);
        let infra = money::mul(net_income, self.infra_frac);
        let reserves = money::mul(net_income, self.reserve_frac);
        Allocation { owner, infra, reserves, retained: net_income - owner - infra - reserves }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_bounds_the_fractions() {
        assert!(SplitPolicy::new(0.2, 0.5, 0.3).is_ok(), "sum exactly 1 is allowed");
        assert!(SplitPolicy::new(0.5, 0.5, 0.1).is_err());
        assert!(SplitPolicy::new(-0.1, 0.0, 0.0).is_err());
        assert!(SplitPolicy::new(1.1, 0.0, 0.0).is_err());
        assert!(SplitPolicy::default().validate().is_ok());
    }

    #[test]
    fn allocation_is_exact_with_the_residual_retained() {
        let p = SplitPolicy::new(0.25, 0.5, 0.1).unwrap();
        let a = p.allocate(dec(1234.56));
        assert_eq!(a.owner, dec(308.64));
        assert_eq!(a.infra, dec(617.28));
        assert_eq!(a.reserves, dec(123.456));
        assert_eq!(a.owner + a.infra + a.reserves + a.retained, dec(1234.56));
        assert!(a.retained > Decimal::ZERO);
    }

    #[test]
    fn losses_and_the_default_policy_allocate_nothing() {
        let p = SplitPolicy::new(0.5, 0.5, 0.0).unwrap();
        assert_eq!(p.allocate(dec(-10.0)), Allocation { retained: dec(-10.0), ..Default::default() });
        assert_eq!(p.allocate(Decimal::ZERO), Allocation::default());
        let a = SplitPolicy::default().allocate(dec(100.0));
        assert_eq!(a.retained, dec(100.0));
        assert_eq!(a.owner + a.infra + a.reserves, Decimal::ZERO);
    }
}
