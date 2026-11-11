//! Exact money (spec 42 §4, §10 Q6): every ledger amount is a [`Decimal`] at a fixed scale of
//! [`SCALE`] decimal places, the scale of the `NUMERIC(38, 18)` columns, so the in-memory book
//! and the book rebuilt from Postgres hold the same numbers digit for digit and sums do not
//! depend on the order fills are applied in.
//!
//! Wire values (`fa.v1.OrderEvent` prices, quantities, fees) are `f64`; [`dec`] converts them
//! once, at the ledger boundary, through the shortest decimal that round-trips the `f64` (what
//! `Display` prints), rounded to [`SCALE`]. Postgres stores exactly that decimal, so a fill's
//! numbers are identical whichever side computes with them. Rows written before the decimal
//! change were cast `float8 → numeric` by Postgres (15 significant digits); they are read back
//! as stored, and since the in-memory book is seeded from Postgres both sides agree on them too.

use std::str::FromStr;

pub use rust_decimal::Decimal;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

/// Decimal places of every ledger amount (`NUMERIC(38, 18)`).
pub const SCALE: u32 = 18;

/// Canonical decimal of a wire `f64`: the shortest round-tripping decimal, at most [`SCALE`]
/// places (banker's rounding). Non-finite input maps to zero; callers validate before booking.
pub fn dec(x: f64) -> Decimal {
    if !x.is_finite() {
        return Decimal::ZERO;
    }
    let d = Decimal::from_str(&format!("{x}")).ok().or_else(|| Decimal::from_f64(x)).unwrap_or(Decimal::ZERO);
    round(d)
}

/// Rounds to the ledger scale.
pub fn round(d: Decimal) -> Decimal {
    d.round_dp(SCALE).normalize()
}

/// A product (price × quantity × rate) at the ledger scale.
pub fn mul(a: Decimal, b: Decimal) -> Decimal {
    round(
        a.checked_mul(b)
            .unwrap_or_else(|| if a.is_sign_negative() == b.is_sign_negative() { Decimal::MAX } else { Decimal::MIN }),
    )
}

/// A quotient at the ledger scale (zero when dividing by zero).
pub fn div(a: Decimal, b: Decimal) -> Decimal {
    if b.is_zero() { Decimal::ZERO } else { round(a.checked_div(b).unwrap_or(Decimal::ZERO)) }
}

/// The nearest `f64` (reporting, risk arithmetic).
pub fn f(d: Decimal) -> f64 {
    d.to_f64().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dec_is_the_shortest_round_trip_and_symmetric() {
        assert_eq!(dec(0.1).to_string(), "0.1");
        assert_eq!(dec(2.85021203500556).to_string(), "2.85021203500556");
        assert_eq!(dec(-101.80316189408), -dec(101.80316189408));
        assert_eq!(dec(1e-30), Decimal::ZERO);
        assert_eq!(dec(f64::NAN), Decimal::ZERO);
        assert_eq!(dec(123_456_789.0).to_string(), "123456789");
        // 0.1 + 0.2 is exact in decimal
        assert_eq!(dec(0.1) + dec(0.2), dec(0.3));
        assert!((f(dec(99_359.901_147_101_93)) - 99_359.901_147_101_93).abs() < 1e-9);
    }

    #[test]
    fn products_are_rounded_to_the_ledger_scale() {
        let p = mul(dec(2.85021203500556), dec(101.80316189408));
        assert!(p.scale() <= SCALE);
        assert_eq!(div(dec(1.0), Decimal::ZERO), Decimal::ZERO);
        assert_eq!(div(dec(1.0), dec(3.0)).scale(), SCALE);
    }
}
