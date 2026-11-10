//! Timescales and horizons (specs 00, 31; D-011).
use std::{fmt, str::FromStr, time::Duration};

use serde::{Deserialize, Serialize};

/// A resolution level; each timescale has its own custom-sized model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum Timescale {
    Second,
    Minute,
    Hour,
    Day,
    Week,
}

impl Timescale {
    pub const LADDER: [Timescale; 5] = [Self::Second, Self::Minute, Self::Hour, Self::Day, Self::Week];

    pub fn duration(self) -> Duration {
        Duration::from_secs(match self {
            Self::Second => 1,
            Self::Minute => 60,
            Self::Hour => 3_600,
            Self::Day => 86_400,
            Self::Week => 604_800,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Second => "1s",
            Self::Minute => "1m",
            Self::Hour => "1h",
            Self::Day => "1d",
            Self::Week => "1w",
        }
    }
}

impl fmt::Display for Timescale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, thiserror::Error)]
#[error("unknown timescale: {0}")]
pub struct UnknownTimescale(String);

impl FromStr for Timescale {
    type Err = UnknownTimescale;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::LADDER.into_iter().find(|t| t.as_str() == s).ok_or_else(|| UnknownTimescale(s.to_owned()))
    }
}

impl TryFrom<String> for Timescale {
    type Error = UnknownTimescale;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        s.parse()
    }
}

impl From<Timescale> for String {
    fn from(t: Timescale) -> Self {
        t.as_str().to_owned()
    }
}

/// Forecast/holding horizon. With matched scope, a strategy at timescale `t`
/// with lookback `n·t` forecasts roughly `n·t` ahead (1h → 2h, 1d → 2d).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Horizon {
    pub timescale: Timescale,
    pub steps: u32,
}

impl Horizon {
    pub fn duration(self) -> Duration {
        self.timescale.duration() * self.steps
    }

    /// The matched-scope forecast horizon for a lookback (spec 00).
    pub fn matched(lookback: Horizon) -> Horizon {
        lookback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timescale_roundtrip() {
        for t in Timescale::LADDER {
            assert_eq!(t.as_str().parse::<Timescale>().unwrap(), t);
        }
    }

    #[test]
    fn horizon_duration() {
        let h = Horizon { timescale: Timescale::Hour, steps: 2 };
        assert_eq!(h.duration(), Duration::from_secs(7_200));
    }
}
