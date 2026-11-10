//! Strategy genome (spec 35 §5, §6.3): the strategy spec, its two programs and its lineage.
//!
//! The **text form** is a `strategies.toml` entry. `to_string` renders it as one TOML inline
//! table on a single line so genomes travel in the newline-separated `genomes` list of
//! `ctl.evo.start` (D-044); [`StrategyGenome::parse`] accepts that inline form, a bare
//! `key = value` body and a full `[[strategy]]` array-of-tables entry. The canonical form is
//! unique per program (`parse(&g.to_string()) == g`), so string equality identifies a genome
//! in the orchestrator, and [`StrategyGenome::to_toml_entry`] writes the multi-line entry used
//! in the promoted `strategies.<generation>.toml` artifact.

use std::fmt;

use fa_core::{Horizon, Timescale};
use serde::{Deserialize, Serialize};

use crate::{
    Expr, GpStrategy,
    population::{BASELINE_GENOME, StrategyDef, StrategyFile},
};

/// Numeric knobs of a strategy spec shared by random generation and mutation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrategyGenome {
    pub timescale: Timescale,
    /// Horizon in bars.
    pub steps: u32,
    pub lookback_bars: usize,
    pub step_secs: Option<u64>,
    pub signal: Expr,
    pub uncertainty: Expr,
    /// Genome ids of the parents (empty for immigrants / baselines).
    #[serde(default)]
    pub parents: Vec<String>,
    #[serde(default)]
    pub generation: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum GenomeError {
    #[error("toml: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("{0}")]
    Population(#[from] crate::population::PopulationError),
    #[error("no strategy entry in genome text")]
    Empty,
    #[error("genome text holds {0} strategies, expected one")]
    Several(usize),
}

/// Id prefix of evolved strategies.
pub const GP_ID_PREFIX: &str = "gp-";

/// Stable 64-bit FNV-1a hash of a string (same as `fa_decoders::genome::stable_hash`).
pub fn stable_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

impl StrategyGenome {
    /// Canonical id: `gp-<16 hex>` of the phenotype (timescale, steps, lookback, programs).
    /// Two genomes with the same programs and spec share an id whatever their lineage.
    pub fn id(&self) -> String {
        let key = format!(
            "{}|{}|{}|{}|{}|{}",
            self.timescale,
            self.steps,
            self.lookback_bars,
            self.step_secs.map_or(String::new(), |s| s.to_string()),
            self.signal,
            self.uncertainty
        );
        format!("{GP_ID_PREFIX}{:016x}", stable_hash(&key))
    }

    pub fn horizon(&self) -> Horizon {
        Horizon { timescale: self.timescale, steps: self.steps.max(1) }
    }

    /// Program nodes (both programs).
    pub fn size(&self) -> usize {
        self.signal.size() + self.uncertainty.size()
    }

    /// The `strategies.toml` definition (id and genome_id = [`Self::id`]).
    pub fn def(&self) -> StrategyDef {
        let id = self.id();
        StrategyDef {
            id: id.clone(),
            genome_id: id,
            timescale: self.timescale,
            steps: self.steps.max(1),
            lookback_bars: self.lookback_bars.max(2),
            step_secs: self.step_secs,
            signal: self.signal.to_string(),
            uncertainty: self.uncertainty.to_string(),
        }
    }

    pub fn build(&self) -> GpStrategy {
        GpStrategy {
            id: self.id(),
            genome_id: self.id(),
            horizon: self.horizon(),
            signal: self.signal.clone(),
            uncertainty: self.uncertainty.clone(),
        }
    }

    /// From a parsed definition (programs are re-parsed; `genome_id` is recomputed unless the
    /// entry is a baseline, whose id is kept in `parents` as provenance).
    pub fn from_def(def: &StrategyDef) -> Result<Self, GenomeError> {
        let s = def.build()?;
        let parents = if def.genome_id == BASELINE_GENOME { vec![def.id.clone()] } else { Vec::new() };
        Ok(Self {
            timescale: def.timescale,
            steps: def.steps.max(1),
            lookback_bars: def.lookback_bars.max(2),
            step_secs: def.step_secs,
            signal: s.signal,
            uncertainty: s.uncertainty,
            parents,
            generation: 0,
        })
    }

    /// Parses the text form: an inline table `{ id = "…", … }`, a `[[strategy]]` entry, or a
    /// bare `key = value` body. Exactly one strategy must be present.
    pub fn parse(text: &str) -> Result<Self, GenomeError> {
        let t = text.trim();
        if t.is_empty() {
            return Err(GenomeError::Empty);
        }
        let doc = if t.starts_with('{') {
            format!("strategy = [{t}]")
        } else if t.contains("[[strategy]]") {
            t.to_owned()
        } else {
            format!("[[strategy]]\n{t}\n")
        };
        let file = StrategyFile::parse(&doc)?;
        match file.strategy.len() {
            0 => Err(GenomeError::Empty),
            1 => Self::from_def(&file.strategy[0]),
            n => Err(GenomeError::Several(n)),
        }
    }

    /// Multi-line `[[strategy]]` entry for a population file.
    pub fn to_toml_entry(&self) -> String {
        let d = self.def();
        let mut s = format!(
            "[[strategy]]\nid = {}\ngenome_id = {}\ntimescale = {}\nsteps = {}\nlookback_bars = {}\n",
            toml_str(&d.id),
            toml_str(&d.genome_id),
            toml_str(d.timescale.as_str()),
            d.steps,
            d.lookback_bars
        );
        if let Some(sec) = d.step_secs {
            s.push_str(&format!("step_secs = {sec}\n"));
        }
        s.push_str(&format!("signal = {}\nuncertainty = {}\n", toml_str(&d.signal), toml_str(&d.uncertainty)));
        s
    }

    /// Whole population file text (`instruments = []` + one entry per genome).
    pub fn population_toml(genomes: &[StrategyGenome], header: &str) -> String {
        let mut out = String::new();
        for line in header.lines() {
            out.push_str("# ");
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("instruments = []\n");
        for g in genomes {
            out.push('\n');
            out.push_str(&g.to_toml_entry());
        }
        out
    }
}

/// TOML basic string (JSON escapes are a subset of TOML's).
fn toml_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| format!("\"{s}\""))
}

/// Single-line inline-table form (the transport form).
impl fmt::Display for StrategyGenome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let d = self.def();
        write!(
            f,
            "{{ id = {}, genome_id = {}, timescale = {}, steps = {}, lookback_bars = {}",
            toml_str(&d.id),
            toml_str(&d.genome_id),
            toml_str(d.timescale.as_str()),
            d.steps,
            d.lookback_bars
        )?;
        if let Some(sec) = d.step_secs {
            write!(f, ", step_secs = {sec}")?;
        }
        write!(f, ", signal = {}, uncertainty = {} }}", toml_str(&d.signal), toml_str(&d.uncertainty))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sexpr;

    fn sample() -> StrategyGenome {
        StrategyGenome {
            timescale: Timescale::Minute,
            steps: 3,
            lookback_bars: 20,
            step_secs: Some(10),
            signal: sexpr::parse("(if_gt stay_prob 0.55 (mul 400 analog_forecast) 0)").unwrap(),
            uncertainty: sexpr::parse("(sub 1 stay_prob)").unwrap(),
            parents: vec!["p1".into()],
            generation: 2,
        }
    }

    #[test]
    fn text_form_round_trips_on_one_line() {
        let g = sample();
        let text = g.to_string();
        assert!(!text.contains('\n') && text.starts_with("{ id = \"gp-"), "{text}");
        let back = StrategyGenome::parse(&text).unwrap();
        assert_eq!((back.signal.clone(), back.uncertainty.clone()), (g.signal.clone(), g.uncertainty.clone()));
        assert_eq!((back.timescale, back.steps, back.lookback_bars, back.step_secs), (g.timescale, 3, 20, Some(10)));
        assert_eq!(back.id(), g.id());
        assert_eq!(back.to_string(), text, "canonical");
        // lineage is not part of the text form
        assert!(back.parents.is_empty());
        // multi-line entry and population file parse too
        let entry = g.to_toml_entry();
        assert!(entry.starts_with("[[strategy]]\n"));
        assert_eq!(StrategyGenome::parse(&entry).unwrap().id(), g.id());
        let file = StrategyGenome::population_toml(&[g.clone(), sample()], "promoted\nby test");
        let parsed = StrategyFile::parse(&file);
        assert!(matches!(parsed, Err(crate::population::PopulationError::Duplicate(_))), "same id twice");
        let mut other = sample();
        other.steps = 4;
        let file = StrategyGenome::population_toml(&[g.clone(), other.clone()], "promoted");
        let parsed = StrategyFile::parse(&file).unwrap();
        assert_eq!(parsed.strategy.len(), 2);
        assert_ne!(other.id(), g.id());
        assert!(file.starts_with("# promoted\ninstruments = []\n"));
    }

    #[test]
    fn parses_bare_bodies_and_baselines() {
        let g = StrategyGenome::parse("id = 'x'\ntimescale = '1s'\nsignal = '(mul 2 analog_forecast)'").unwrap();
        assert_eq!(g.uncertainty.to_string(), "region_entropy");
        assert_eq!(g.lookback_bars, 32);
        for def in StrategyFile::baselines().strategy {
            let g = StrategyGenome::from_def(&def).unwrap();
            assert_eq!(g.parents, vec![def.id.clone()], "baseline provenance");
            assert!(g.id().starts_with(GP_ID_PREFIX));
            assert_eq!(g.build().horizon, def.horizon());
        }
        assert!(matches!(StrategyGenome::parse(""), Err(GenomeError::Empty)));
        assert!(matches!(StrategyGenome::parse("instruments = []"), Err(GenomeError::Population(_))));
        assert!(matches!(
            StrategyGenome::parse(
                "[[strategy]]\nid='a'\ntimescale='1s'\nsignal='1'\n[[strategy]]\nid='b'\ntimescale='1s'\nsignal='1'\n"
            ),
            Err(GenomeError::Several(2))
        ));
        assert!(StrategyGenome::parse("{ id = \"a\", timescale = \"1s\", signal = \"(rsi 14)\" }").is_err());
    }
}
