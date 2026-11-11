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

/// Exact sum of decimals.
pub fn sum<I: IntoIterator<Item = Decimal>>(it: I) -> Decimal {
    it.into_iter().fold(Decimal::ZERO, |a, b| a + b)
}

/// Serde adapter: a [`Decimal`] as a JSON number (its nearest `f64`), read back through [`dec`].
/// Keeps the API JSON shapes of the derived analytics (numbers, not strings) while the
/// computation is exact; use `#[serde(with = "crate::money::json")]`.
pub mod json {
    use serde::{Deserialize, Deserializer, Serializer};

    use super::{Decimal, dec, f};

    pub fn serialize<S: Serializer>(d: &Decimal, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_f64(f(*d))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Decimal, D::Error> {
        Ok(dec(f64::deserialize(d)?))
    }
}

/// [`json`] for `Option<Decimal>` (`null` when absent).
pub mod json_opt {
    use serde::{Deserialize, Deserializer, Serializer};

    use super::{Decimal, dec, f};

    pub fn serialize<S: Serializer>(d: &Option<Decimal>, s: S) -> Result<S::Ok, S::Error> {
        match d {
            Some(d) => s.serialize_some(&f(*d)),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Decimal>, D::Error> {
        Ok(Option::<f64>::deserialize(d)?.filter(|x| x.is_finite()).map(dec))
    }
}

/// [`json`] for maps of decimals (`{"key": number}`).
pub mod json_map {
    use std::collections::BTreeMap;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::{Decimal, dec, f};

    pub fn serialize<K: Serialize, S: Serializer>(m: &BTreeMap<K, Decimal>, s: S) -> Result<S::Ok, S::Error> {
        s.collect_map(m.iter().map(|(k, v)| (k, f(*v))))
    }

    pub fn deserialize<'de, K, D>(d: D) -> Result<BTreeMap<K, Decimal>, D::Error>
    where
        K: Deserialize<'de> + Ord,
        D: Deserializer<'de>,
    {
        let m = BTreeMap::<K, f64>::deserialize(d)?;
        Ok(m.into_iter().map(|(k, v)| (k, dec(v))).collect())
    }
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
    fn json_adapters_write_numbers() {
        #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
        struct T {
            #[serde(with = "json")]
            a: Decimal,
            #[serde(with = "json_opt")]
            b: Option<Decimal>,
            #[serde(with = "json_map")]
            m: std::collections::BTreeMap<String, Decimal>,
        }
        let t = T { a: dec(0.1) + dec(0.2), b: None, m: [("x".to_owned(), dec(2.5))].into() };
        let s = serde_json::to_string(&t).unwrap();
        assert_eq!(s, r#"{"a":0.3,"b":null,"m":{"x":2.5}}"#);
        assert_eq!(serde_json::from_str::<T>(&s).unwrap(), t);
        assert_eq!(sum([dec(0.1), dec(0.2)]), dec(0.3));
    }

    #[test]
    fn products_are_rounded_to_the_ledger_scale() {
        let p = mul(dec(2.85021203500556), dec(101.80316189408));
        assert!(p.scale() <= SCALE);
        assert_eq!(div(dec(1.0), Decimal::ZERO), Decimal::ZERO);
        assert_eq!(div(dec(1.0), dec(3.0)).scale(), SCALE);
    }
}
