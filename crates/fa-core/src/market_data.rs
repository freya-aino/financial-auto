//! Which market data the paper-trading path acts on (D-103), from `FA_MARKET_DATA`:
//!
//! - `real` (default): data from the real connectors only. The synthetic venues
//!   ([`SYNTHETIC_VENUES`]: the `heartbeat` and `synthetic` connectors) are ignored by
//!   model-runtime, strategy-runtime, risk-engine and the paper broker, so paper orders are only
//!   priced and filled on real market data.
//! - `synthetic`: only the synthetic venues (ephemeral test stacks, D-096).
//! - `all`: both (the behaviour before D-103).
//!
//! Mirrors `go/pkg/bus/marketdata.go`.
use std::{fmt, str::FromStr};

/// Venues of locally generated market data.
pub const SYNTHETIC_VENUES: &[&str] = &["sim"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MarketDataMode {
    #[default]
    Real,
    Synthetic,
    All,
}

pub fn is_synthetic_venue(venue: &str) -> bool {
    SYNTHETIC_VENUES.contains(&venue)
}

impl MarketDataMode {
    /// `FA_MARKET_DATA` (unset or empty = `real`); an unknown value is an error, so a service
    /// with a typo fails at start instead of trading on the wrong data.
    pub fn from_env() -> Result<Self, String> {
        std::env::var("FA_MARKET_DATA").unwrap_or_default().parse()
    }

    pub fn admits_venue(self, venue: &str) -> bool {
        match self {
            Self::All => true,
            Self::Synthetic => is_synthetic_venue(venue),
            Self::Real => !is_synthetic_venue(venue),
        }
    }

    /// [`Self::admits_venue`] of an `md.<kind>.<venue>.…` subject (`md.bar.<tf>.<venue>.…` for
    /// bars); other subjects pass.
    pub fn admits_subject(self, subject: &str) -> bool {
        let p: Vec<&str> = subject.split('.').collect();
        match p.as_slice() {
            ["md", "bar", _, venue, ..] => self.admits_venue(venue),
            ["md", kind, venue, ..] if *kind != "bar" => self.admits_venue(venue),
            _ => true,
        }
    }
}

impl FromStr for MarketDataMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "real" => Ok(Self::Real),
            "synthetic" => Ok(Self::Synthetic),
            "all" => Ok(Self::All),
            other => Err(format!("FA_MARKET_DATA must be real, synthetic or all, not `{other}`")),
        }
    }
}

impl fmt::Display for MarketDataMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Real => "real",
            Self::Synthetic => "synthetic",
            Self::All => "all",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_is_the_default_and_drops_synthetic_venues() {
        assert_eq!("".parse::<MarketDataMode>(), Ok(MarketDataMode::Real));
        assert!("live".parse::<MarketDataMode>().is_err());
        let real = MarketDataMode::default();
        assert!(real.admits_subject("md.trade.kraken.0190abc"));
        assert!(!real.admits_subject("md.trade.sim.01900000000070008000000000000001"));
        assert!(!real.admits_subject("md.bar.1m.sim.x"));
        assert!(real.admits_subject("md.bar.1m.kraken.x"));
        assert!(real.admits_subject("signal.exposure.s1"));
        let syn = MarketDataMode::Synthetic;
        assert!(syn.admits_subject("md.quote.sim.x") && !syn.admits_subject("md.trade.yahoo.x"));
        assert!(MarketDataMode::All.admits_subject("md.trade.sim.x"));
        assert_eq!(MarketDataMode::Synthetic.to_string(), "synthetic");
    }
}
