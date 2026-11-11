//! Settlement currency per venue (spec 42 §10 Q5).
//!
//! **Default (decided 2026-11-11):** every venue settles in EUR. A fill's cash effect is booked
//! to EUR cash at the fill's FX rate — the paper venue's model and that of the in-memory
//! [`crate::Ledger`] — so FX P&L on trading balances is zero by construction and only non-EUR
//! cash movements carry a native balance. A venue that keeps native-currency cash accounts (a
//! live broker with a USD sub-account, say) is listed in `ledger.venue_settlement` with
//! `settlement = 'native'` (migration `20261112000010`): its fills then move the native cash
//! balance of the fill's currency, which [`crate::accounting::BookState`] revalues like any
//! foreign cash, so FX P&L on that cash (`fx_cash_eur`) covers trading balances too. Fees are
//! always booked in EUR (`ledger.fill.fee_ccy = 'EUR'`). The risk-engine's paper book keeps the
//! EUR model (it books only the paper venue); a native-settling venue needs a connector, which
//! is out of scope until one exists.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// How a venue settles fills.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Settlement {
    /// Fill cash converts to EUR at the fill's rate (default).
    #[default]
    Eur,
    /// Fill cash stays in the fill's currency (a native cash balance, revalued to EUR).
    Native,
}

impl Settlement {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Eur => "eur",
            Self::Native => "native",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "eur" | "base" => Some(Self::Eur),
            "native" | "quote" => Some(Self::Native),
            _ => None,
        }
    }
}

/// Settlement per venue; venues not listed use `default`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SettlementPolicy {
    #[serde(default)]
    pub default: Settlement,
    #[serde(default)]
    pub venues: BTreeMap<Uuid, Settlement>,
}

impl SettlementPolicy {
    pub fn of(&self, venue: &Uuid) -> Settlement {
        self.venues.get(venue).copied().unwrap_or(self.default)
    }

    pub fn with(mut self, venue: Uuid, s: Settlement) -> Self {
        self.venues.insert(venue, s);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_eur_and_venues_override() {
        let v = Uuid::from_u128(7);
        let p = SettlementPolicy::default();
        assert_eq!(p.of(&v), Settlement::Eur);
        let p = p.with(v, Settlement::Native);
        assert_eq!(p.of(&v), Settlement::Native);
        assert_eq!(p.of(&Uuid::nil()), Settlement::Eur);
        assert_eq!(Settlement::parse(" Native "), Some(Settlement::Native));
        assert_eq!(Settlement::parse("x"), None);
    }
}
