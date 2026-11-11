//! Paper funding / borrow carry (spec 40 §6.5 shorts): the broker-gateway accrues the
//! `costs.yaml` carry of every open position and announces one charge per (instrument, kind,
//! period) on `exec.carry.<account>` (JSON, `go/services/broker-gateway/carry.go`); the
//! risk-engine books each one once as a `ledger.cash_movement` of kind `funding` / `borrow`
//! (reference = the charge's reference, so a re-announced charge is a duplicate in both books)
//! and applies it to the paper book. Carry is book-level cash (the accounting puts it in the
//! unattributed bucket, spec 42 §6); it lowers equity and thus counts toward the drawdown and
//! daily-loss limits.
use chrono::{DateTime, Utc};
use fa_core::InstrumentId;
use fa_ledger::{
    cash_key,
    money::{dec, mul, round},
    pg::{CashMovement, PAPER_VENUE},
};
use serde::{Deserialize, Serialize};

use super::Engine;

/// One charge as the gateway publishes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CarryCharge {
    pub account: String,
    pub instrument_id: String,
    /// `funding` | `borrow`
    pub kind: String,
    /// In `ccy` (empty = the instrument's quote currency); negative = paid.
    pub amount: f64,
    #[serde(default)]
    pub ccy: String,
    #[serde(default)]
    pub bps_day: f64,
    #[serde(default)]
    pub notional_days: f64,
    pub period_end: DateTime<Utc>,
    pub reference: String,
}

/// A charge ready to book: the cash movement for Postgres and its in-memory key.
#[derive(Debug, Clone)]
pub struct CarryBooking {
    pub movement: CashMovement,
    pub key: String,
}

impl Engine {
    /// The booking of a carry charge for this engine's book; `None` for another account, an
    /// unknown kind, a non-finite amount or a charge already booked.
    pub fn carry_booking(&mut self, c: &CarryCharge, book: &str) -> Option<CarryBooking> {
        if c.account != self.settings.account
            || !matches!(c.kind.as_str(), "funding" | "borrow")
            || !c.amount.is_finite()
            || c.reference.trim().is_empty()
        {
            self.bump("carry.ignored");
            return None;
        }
        let key = cash_key(&PAPER_VENUE, &c.kind, &c.reference);
        if self.ledger.cash_refs().contains(&key) {
            self.bump("carry.duplicates");
            return None;
        }
        let at_ms = c.period_end.timestamp_millis();
        let (ccy, rate) = match c.ccy.trim() {
            "" => {
                let inst = uuid::Uuid::parse_str(&c.instrument_id).ok().map(InstrumentId);
                let ccy = inst
                    .and_then(|i| self.quote_ccy(&i).map(str::to_owned))
                    .unwrap_or_else(|| fa_core::BASE_CURRENCY.to_owned());
                let rate = match inst {
                    Some(i) => self.fx_for(&i, at_ms).rate_to_eur,
                    None => 1.0,
                };
                (ccy, rate)
            }
            x if x.eq_ignore_ascii_case(fa_core::BASE_CURRENCY) => (fa_core::BASE_CURRENCY.to_owned(), 1.0),
            x => (x.to_ascii_uppercase(), self.fx.convert(x, c.period_end, self.settings.fx.max_age()).rate_to_eur),
        };
        Some(CarryBooking {
            movement: CashMovement {
                book: book.to_owned(),
                kind: c.kind.clone(),
                amount: c.amount,
                ccy,
                fx_rate_eur: rate,
                reference: c.reference.clone(),
                occurred_at: c.period_end,
            },
            key,
        })
    }

    /// Applies a booking to the paper book (after Postgres has it, or without Postgres), with
    /// the same exact EUR amount reconciliation derives from the stored row. `false` when it was
    /// applied before.
    pub fn apply_carry(&mut self, b: &CarryBooking) -> bool {
        let eur = mul(round(dec(b.movement.amount)), round(dec(b.movement.fx_rate_eur)));
        if !self.ledger.apply_cash(&b.key, &b.movement.kind, eur) {
            return false;
        }
        *self.counters.entry("carry.booked").or_insert(0.0) += 1.0;
        let k = if b.movement.kind == "borrow" { "carry.borrow_eur" } else { "carry.funding_eur" };
        *self.counters.entry(k).or_insert(0.0) += fa_ledger::money::f(eur);
        self.risk_dirty = true;
        true
    }
}
