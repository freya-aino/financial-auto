//! Strategy definitions loadable from a TOML file (`strategies.toml`) and the built-in
//! baseline population. Baselines are hand-written placeholders, not evolved genomes.
use fa_core::{Horizon, Timescale};
use serde::{Deserialize, Serialize};

use crate::{
    GpStrategy, Strategy,
    neural::{NetError, NeuralStrategy, PolicyNet},
    sexpr,
};

/// Genome id that marks a hand-written, non-evolved program.
pub const BASELINE_GENOME: &str = "baseline";

/// Key (bucket `fa-artifacts`) of the **managed live set**: the evolved strategies the
/// risk-engine pipeline has admitted (shadow, active, paused), rewritten on every admission
/// and retirement (spec 35 §6.4). strategy-runtime's default population is the baselines plus
/// this file ([`StrategyFile::merged`]).
pub const LIVE_SET_KEY: &str = "live/strategies.toml";

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
    /// GP program for the exposure mean (S-expression); unused by a neural policy.
    #[serde(default = "default_signal")]
    pub signal: String,
    /// GP program for the exposure uncertainty (S-expression).
    #[serde(default = "default_uncertainty")]
    pub uncertainty: String,
    /// Encoder snapshot the programs' region literals `(region e k)` refer to (spec 35 §6.5).
    /// Bound strategies are remapped through the region lineage when the live snapshot of their
    /// timescale changes; unbound literals resolve against whatever snapshot is live.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<String>,
    /// Asset set (instrument UUIDs) the strategy trades (spec 35 §3, §6.3); empty = every
    /// instrument it sees.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assets: Vec<String>,
    /// Neural policy (`neural::PolicyNet` text form, spec 35 §4.3); `None` = the GP programs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<String>,
}

fn default_signal() -> String {
    "0".into()
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
    Program { id: String, field: &'static str, source: sexpr::ParseError },
    #[error("strategy `{id}` policy: {source}")]
    Policy { id: String, source: NetError },
    #[error("duplicate strategy id `{0}`")]
    Duplicate(String),
    #[error("parsing strategy file: {0}")]
    Toml(#[from] toml::de::Error),
}

impl StrategyDef {
    pub fn horizon(&self) -> Horizon {
        Horizon { timescale: self.timescale, steps: self.steps.max(1) }
    }

    pub fn step_secs(&self) -> u64 {
        self.step_secs.unwrap_or_else(|| self.timescale.duration().as_secs()).max(1)
    }

    pub fn is_baseline(&self) -> bool {
        self.genome_id == BASELINE_GENOME
    }

    /// Representation of the definition.
    pub fn is_neural(&self) -> bool {
        self.policy.is_some()
    }

    /// Whether `instrument` (UUID text) is in the asset set (an empty set holds everything).
    pub fn trades(&self, instrument: &str) -> bool {
        self.assets.is_empty() || self.assets.iter().any(|a| a == instrument)
    }

    /// The neural policy of a neural definition.
    pub fn policy_net(&self) -> Result<Option<PolicyNet>, PopulationError> {
        self.policy
            .as_deref()
            .map(|p| PolicyNet::parse(p).map_err(|source| PopulationError::Policy { id: self.id.clone(), source }))
            .transpose()
    }

    /// The runnable strategy: a [`NeuralStrategy`] for a neural definition, else the GP programs.
    pub fn build_dyn(&self) -> Result<Box<dyn Strategy>, PopulationError> {
        Ok(match self.policy_net()? {
            Some(net) => Box::new(NeuralStrategy {
                id: self.id.clone(),
                genome_id: self.genome_id.clone(),
                horizon: self.horizon(),
                net,
                recorder: None,
            }),
            None => Box::new(self.build()?),
        })
    }

    /// The GP programs (a neural definition has the placeholder program `0`).
    pub fn build(&self) -> Result<GpStrategy, PopulationError> {
        let prog = |field: &'static str, src: &str| {
            sexpr::parse(src).map_err(|source| PopulationError::Program { id: self.id.clone(), field, source })
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
            s.policy_net()?;
        }
        Ok(f)
    }

    /// `self` plus every strategy of `other` whose id is not already present (the first file
    /// wins on an id clash; evolved `gp-…` ids never clash with baselines). `instruments` are
    /// united (empty = every instrument stays empty only when both are).
    pub fn merged(mut self, other: StrategyFile) -> Self {
        for s in other.strategy {
            if !self.strategy.iter().any(|x| x.id == s.id) {
                self.strategy.push(s);
            }
        }
        if !self.instruments.is_empty() && !other.instruments.is_empty() {
            for i in other.instruments {
                if !self.instruments.contains(&i) {
                    self.instruments.push(i);
                }
            }
        } else {
            self.instruments.clear();
        }
        self
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
        assert!(f.strategy.len() >= 9);
        assert!(f.strategy.iter().any(|s| s.id == "baseline-text-visual-1s"));
        assert!(f.strategy.iter().any(|s| s.id == "baseline-narrative-chart-1m"));
        assert!(f.strategy.iter().all(StrategyDef::is_baseline));
        assert_eq!(f.strategy[0].step_secs(), 1);
        assert_eq!(f.strategy.iter().find(|s| s.id == "baseline-regime-1m").unwrap().step_secs(), 10);
    }

    #[test]
    fn baselines_emit_bounded_exposures_on_price_only_features() {
        let closes: Vec<f64> = (0..40).map(|i| 100.0 + (i as f64 * 0.7).sin() + i as f64 * 0.05).collect();
        let ctx = FallbackContext::new(&closes, None, VirtualBook::default());
        let inst = InstrumentId::new();
        for def in StrategyFile::baselines().strategy {
            let mut s = def.build().unwrap();
            let d = s.step(1_000, &[(inst, &ctx)]);
            let e = d.exposures[0];
            assert!((-1.0..=1.0).contains(&e.mean), "{}: mean {}", def.id, e.mean);
            assert!((0.0..=1.0).contains(&e.stddev), "{}: std {}", def.id, e.stddev);
        }
    }

    #[test]
    fn merged_keeps_the_first_definition_and_adds_new_ids() {
        let live = StrategyFile::parse(
            "[[strategy]]\nid='gp-1'\ngenome_id='gp-1'\ntimescale='1s'\nsignal='1'\n[[strategy]]\nid='baseline-momentum-1s'\ntimescale='1s'\nsignal='0'\n",
        )
        .unwrap();
        let n = StrategyFile::baselines().strategy.len();
        let m = StrategyFile::baselines().merged(live);
        assert_eq!(m.strategy.len(), n + 1);
        assert!(m.strategy.iter().any(|s| s.id == "gp-1" && !s.is_baseline()));
        assert_ne!(m.strategy.iter().find(|s| s.id == "baseline-momentum-1s").unwrap().signal, "0");
        assert!(m.instruments.is_empty());
    }

    #[test]
    fn file_errors_are_reported() {
        let bad = "[[strategy]]\nid='x'\ntimescale='1s'\nsignal='(rsi 14)'\n";
        assert!(matches!(StrategyFile::parse(bad), Err(PopulationError::Program { field: "signal", .. })));
        let dup =
            "[[strategy]]\nid='x'\ntimescale='1s'\nsignal='1'\n[[strategy]]\nid='x'\ntimescale='1s'\nsignal='1'\n";
        assert!(matches!(StrategyFile::parse(dup), Err(PopulationError::Duplicate(_))));
    }
}
