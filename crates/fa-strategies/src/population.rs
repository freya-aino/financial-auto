//! Strategy definitions loadable from a TOML file (`strategies.toml`) and the built-in
//! baseline population. Baselines are hand-written placeholders, not evolved genomes.
use fa_core::{Horizon, Timescale};
use serde::{Deserialize, Serialize};

use crate::{GpStrategy, sexpr};

/// Genome id that marks a hand-written, non-evolved program.
pub const BASELINE_GENOME: &str = "baseline";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyDef {
    pub id: String,
    #[serde(default = "default_genome")]
    pub genome_id: String,
    pub timescale: Timescale,
    /// Horizon in bars of `timescale`.
    #[serde(default = "default_steps")]
    pub steps: u32,
    /// Bars of history the fallback features are computed from.
    #[serde(default = "default_lookback")]
    pub lookback_bars: usize,
    /// Decision cadence in seconds; defaults to one bar of `timescale`.
    #[serde(default)]
    pub step_secs: Option<u64>,
    /// GP program for the exposure mean (S-expression).
    pub signal: String,
    /// GP program for the exposure uncertainty (S-expression).
    #[serde(default = "default_uncertainty")]
    pub uncertainty: String,
}

fn default_genome() -> String {
    BASELINE_GENOME.into()
}
fn default_steps() -> u32 {
    1
}
fn default_lookback() -> usize {
    32
}
fn default_uncertainty() -> String {
    "region_entropy".into()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StrategyFile {
    /// Instrument UUIDs to trade; empty = every instrument seen.
    #[serde(default)]
    pub instruments: Vec<String>,
    #[serde(default)]
    pub strategy: Vec<StrategyDef>,
}

#[derive(Debug, thiserror::Error)]
pub enum PopulationError {
    #[error("strategy `{id}` {field}: {source}")]
    Program {
        id: String,
        field: &'static str,
        source: sexpr::ParseError,
    },
    #[error("duplicate strategy id `{0}`")]
    Duplicate(String),
    #[error("parsing strategy file: {0}")]
    Toml(#[from] toml::de::Error),
}

impl StrategyDef {
    pub fn horizon(&self) -> Horizon {
        Horizon {
            timescale: self.timescale,
            steps: self.steps.max(1),
        }
    }

    pub fn step_secs(&self) -> u64 {
        self.step_secs
            .unwrap_or_else(|| self.timescale.duration().as_secs())
            .max(1)
    }

    pub fn is_baseline(&self) -> bool {
        self.genome_id == BASELINE_GENOME
    }

    pub fn build(&self) -> Result<GpStrategy, PopulationError> {
        let prog = |field: &'static str, src: &str| {
            sexpr::parse(src).map_err(|source| PopulationError::Program {
                id: self.id.clone(),
                field,
                source,
            })
        };
        Ok(GpStrategy {
            id: self.id.clone(),
            genome_id: self.genome_id.clone(),
            horizon: self.horizon(),
            signal: prog("signal", &self.signal)?,
            uncertainty: prog("uncertainty", &self.uncertainty)?,
        })
    }
}

impl StrategyFile {
    pub fn parse(toml_src: &str) -> Result<Self, PopulationError> {
        let f: StrategyFile = toml::from_str(toml_src)?;
        let mut ids: Vec<&str> = f.strategy.iter().map(|s| s.id.as_str()).collect();
        ids.sort_unstable();
        if let Some(d) = ids.windows(2).find(|w| w[0] == w[1]) {
            return Err(PopulationError::Duplicate(d[0].into()));
        }
        for s in &f.strategy {
            s.build()?;
        }
        Ok(f)
    }

    /// The built-in baseline population (`crates/fa-strategies/baselines.toml`).
    pub fn baselines() -> Self {
        Self::parse(BASELINES_TOML).expect("built-in baselines must parse")
    }
}

/// Source of the built-in baseline population.
pub const BASELINES_TOML: &str = include_str!("../baselines.toml");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Strategy,
        features::{FallbackContext, VirtualBook},
    };
    use fa_core::InstrumentId;

    #[test]
    fn baselines_parse_and_are_labelled() {
        let f = StrategyFile::baselines();
        assert!(f.strategy.len() >= 4);
        assert!(f.strategy.iter().all(StrategyDef::is_baseline));
        assert_eq!(f.strategy[0].step_secs(), 1);
        assert_eq!(
            f.strategy
                .iter()
                .find(|s| s.id == "baseline-regime-1m")
                .unwrap()
                .step_secs(),
            10
        );
    }

    #[test]
    fn baselines_emit_bounded_exposures_on_price_only_features() {
        let closes: Vec<f64> = (0..40)
            .map(|i| 100.0 + (i as f64 * 0.7).sin() + i as f64 * 0.05)
            .collect();
        let ctx = FallbackContext::new(&closes, None, VirtualBook::default());
        let inst = InstrumentId::new();
        for def in StrategyFile::baselines().strategy {
            let mut s = def.build().unwrap();
            let d = s.step(1_000, &[(inst, &ctx)]);
            let e = d.exposures[0];
            assert!(
                (-1.0..=1.0).contains(&e.mean),
                "{}: mean {}",
                def.id,
                e.mean
            );
            assert!(
                (0.0..=1.0).contains(&e.stddev),
                "{}: std {}",
                def.id,
                e.stddev
            );
        }
    }

    #[test]
    fn file_errors_are_reported() {
        let bad = "[[strategy]]\nid='x'\ntimescale='1s'\nsignal='(rsi 14)'\n";
        assert!(matches!(
            StrategyFile::parse(bad),
            Err(PopulationError::Program {
                field: "signal",
                ..
            })
        ));
        let dup = "[[strategy]]\nid='x'\ntimescale='1s'\nsignal='1'\n[[strategy]]\nid='x'\ntimescale='1s'\nsignal='1'\n";
        assert!(matches!(
            StrategyFile::parse(dup),
            Err(PopulationError::Duplicate(_))
        ));
    }
}
