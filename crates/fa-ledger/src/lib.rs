//! Ledger: positions, cash and P&L in EUR, fees, idempotent intents, reconciliation (spec 42).
//! This in-memory model mirrors the Postgres ledger tables; persistence is added in the
//! ledger service.

use std::collections::{BTreeMap, HashSet};

use fa_core::InstrumentId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    /// Unique per venue fill; used for idempotency.
    pub fill_id: String,
    pub intent_id: String,
    pub instrument: InstrumentId,
    /// Signed quantity (+ buy, − sell).
    pub quantity: f64,
    /// Price in instrument currency.
    pub price: f64,
    /// FX rate instrument currency → EUR at fill time.
    pub fx_to_eur: f64,
    pub fee_eur: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub quantity: f64,
    /// Average cost in EUR per unit.
    pub avg_cost_eur: f64,
    pub realised_eur: f64,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum LedgerError {
    #[error("duplicate intent id {0}")]
    DuplicateIntent(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Ledger {
    pub cash_eur: f64,
    pub fees_eur: f64,
    pub positions: BTreeMap<InstrumentId, Position>,
    intents: HashSet<String>,
    fills: HashSet<String>,
    /// Paid out to date (payout quota tracking).
    pub paid_out_eur: f64,
}

impl Ledger {
    pub fn with_cash(cash_eur: f64) -> Self {
        Self {
            cash_eur,
            ..Self::default()
        }
    }

    /// Registers an order intent; intent ids are idempotency keys.
    pub fn register_intent(&mut self, intent_id: &str) -> Result<(), LedgerError> {
        if !self.intents.insert(intent_id.to_owned()) {
            return Err(LedgerError::DuplicateIntent(intent_id.to_owned()));
        }
        Ok(())
    }

    /// Applies a fill. Replays of the same `fill_id` are ignored (returns false).
    pub fn apply_fill(&mut self, f: &Fill) -> bool {
        if !self.fills.insert(f.fill_id.clone()) {
            return false;
        }
        let px = f.price * f.fx_to_eur;
        self.cash_eur -= f.quantity * px + f.fee_eur;
        self.fees_eur += f.fee_eur;
        let p = self.positions.entry(f.instrument).or_default();
        let q0 = p.quantity;
        let q1 = q0 + f.quantity;
        if q0 == 0.0 || q0.signum() == f.quantity.signum() {
            // opening / adding
            p.avg_cost_eur = (p.avg_cost_eur * q0.abs() + px * f.quantity.abs()) / q1.abs();
        } else {
            // reducing / closing / flipping
            let closed = f.quantity.abs().min(q0.abs());
            p.realised_eur += closed * (px - p.avg_cost_eur) * q0.signum();
            if q1 != 0.0 && q1.signum() != q0.signum() {
                p.avg_cost_eur = px; // flipped: remainder opened at fill price
            }
        }
        p.quantity = q1;
        if p.quantity.abs() < 1e-12 {
            p.quantity = 0.0;
            p.avg_cost_eur = 0.0;
        }
        true
    }

    pub fn record_payout(&mut self, eur: f64) {
        self.cash_eur -= eur;
        self.paid_out_eur += eur;
    }

    /// Equity in EUR given marks (EUR per unit).
    pub fn equity(&self, marks_eur: &BTreeMap<InstrumentId, f64>) -> f64 {
        self.cash_eur
            + self
                .positions
                .iter()
                .map(|(i, p)| p.quantity * marks_eur.get(i).copied().unwrap_or(p.avg_cost_eur))
                .sum::<f64>()
    }
}

/// One mismatch between our ledger and the broker's view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReconDiff {
    pub instrument: InstrumentId,
    pub ours: f64,
    pub broker: f64,
}

/// Compares positions with the broker's statement; differences above `tol` are reported.
pub fn reconcile(ours: &Ledger, broker: &BTreeMap<InstrumentId, f64>, tol: f64) -> Vec<ReconDiff> {
    let mut keys: Vec<InstrumentId> = ours
        .positions
        .keys()
        .chain(broker.keys())
        .copied()
        .collect();
    keys.sort();
    keys.dedup();
    keys.into_iter()
        .filter_map(|i| {
            let o = ours.positions.get(&i).map_or(0.0, |p| p.quantity);
            let b = broker.get(&i).copied().unwrap_or(0.0);
            ((o - b).abs() > tol).then_some(ReconDiff {
                instrument: i,
                ours: o,
                broker: b,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(id: &str, inst: InstrumentId, q: f64, px: f64) -> Fill {
        Fill {
            fill_id: id.into(),
            intent_id: "i".into(),
            instrument: inst,
            quantity: q,
            price: px,
            fx_to_eur: 1.0,
            fee_eur: 1.0,
        }
    }

    #[test]
    fn round_trip_realises_pnl_net_of_fees() {
        let i = InstrumentId::new();
        let mut l = Ledger::with_cash(10_000.0);
        assert!(l.apply_fill(&fill("f1", i, 10.0, 100.0)));
        assert!(
            !l.apply_fill(&fill("f1", i, 10.0, 100.0)),
            "replayed fill ignored"
        );
        l.apply_fill(&fill("f2", i, -10.0, 110.0));
        assert_eq!(l.positions[&i].quantity, 0.0);
        assert!((l.positions[&i].realised_eur - 100.0).abs() < 1e-9);
        assert!((l.cash_eur - 10_098.0).abs() < 1e-9);
        assert_eq!(l.fees_eur, 2.0);
    }

    #[test]
    fn short_and_flip() {
        let i = InstrumentId::new();
        let mut l = Ledger::with_cash(0.0);
        l.apply_fill(&fill("a", i, -5.0, 100.0));
        l.apply_fill(&fill("b", i, 8.0, 90.0));
        let p = &l.positions[&i];
        assert_eq!(p.quantity, 3.0);
        assert!((p.realised_eur - 50.0).abs() < 1e-9);
        assert_eq!(p.avg_cost_eur, 90.0);
    }

    #[test]
    fn fx_conversion_and_equity() {
        let i = InstrumentId::new();
        let mut l = Ledger::with_cash(1_000.0);
        let mut f = fill("a", i, 1.0, 100.0);
        f.fx_to_eur = 0.9;
        f.fee_eur = 0.0;
        l.apply_fill(&f);
        assert!((l.cash_eur - 910.0).abs() < 1e-9);
        assert!((l.equity(&BTreeMap::from([(i, 95.0)])) - 1_005.0).abs() < 1e-9);
    }

    #[test]
    fn idempotent_intents_and_reconciliation() {
        let mut l = Ledger::default();
        l.register_intent("x").unwrap();
        assert_eq!(
            l.register_intent("x"),
            Err(LedgerError::DuplicateIntent("x".into()))
        );
        let (a, b) = (InstrumentId::new(), InstrumentId::new());
        l.apply_fill(&fill("f", a, 2.0, 1.0));
        let d = reconcile(&l, &BTreeMap::from([(a, 2.0), (b, 1.0)]), 1e-9);
        assert_eq!(
            d,
            vec![ReconDiff {
                instrument: b,
                ours: 0.0,
                broker: 1.0
            }]
        );
    }
}
