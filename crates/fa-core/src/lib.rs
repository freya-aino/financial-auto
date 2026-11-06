//! Shared domain types used across the platform (specs 01, 10, 12).

pub mod config;
pub mod ids;
pub mod rng;
pub mod time;

pub use ids::{EventId, InstrumentId, JobId};
pub use time::{Horizon, Timescale};

/// Asset classes; leverage caps and concentration limits are keyed by these (spec 40).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetClass {
    Equity,
    Etf,
    Fx,
    Crypto,
    Commodity,
    Rates,
    Index,
    Derivative,
    PredictionMarket,
}

/// Paper / shadow / live trading mode (spec 40).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TradingMode {
    Paper,
    Shadow,
    Live,
}

/// Base currency for P&L, risk and fitness (spec 00).
pub const BASE_CURRENCY: &str = "EUR";
