//! Evolvable allocator genomes (spec 40 §6.2 "evolved allocator", §3 "pluggable
//! meta-allocator"): an [`AllocatorGenome`] names one of the fixed allocators or the parametric
//! [`WeightedBlend`](crate::WeightedBlend) family and carries the numeric genes every allocator
//! reads — per-strategy cap, exposure-uncertainty shrink, Kelly fraction, performance-tracker
//! half-life and prior volatility, blend weights. It travels as one line of text
//! (`alloc:name=risk_parity;cap=0.4;shrink=1;kelly=0.25;hl=240;pv=0.01;w_eq=0;w_rp=1;w_k=0`),
//! parses back to the same value ([`AllocatorGenome::parse`] / `Display` round trip, canonical
//! key order) and is bred through the shared `fa_evolution::vary` loop with [`AllocatorOps`]
//! (uniform crossover over the genes, Gaussian mutation of one or all numeric genes, an
//! occasional name switch), clamped back into [`GENE_BOUNDS`].

use std::{collections::BTreeMap, fmt, str::FromStr};

use fa_core::rng::Rng;
use fa_evolution::vary::{Evolvable, Operators};
use serde::{Deserialize, Serialize};

use crate::{Allocator, EqualWeight, FractionalKelly, PerformanceTracker, RiskParityCapped, WeightedBlend};

/// Text prefix of the genome form.
pub const PREFIX: &str = "alloc:";

/// Allocator families a genome can name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AllocatorName {
    EqualWeight,
    RiskParity,
    FractionalKelly,
    /// Parametric mix of the three above (`w_eq`, `w_rp`, `w_k`).
    WeightedBlend,
}

impl AllocatorName {
    pub const ALL: [AllocatorName; 4] =
        [Self::EqualWeight, Self::RiskParity, Self::FractionalKelly, Self::WeightedBlend];

    /// Whether this allocator reads gene `key` (`kelly` only Kelly and blend, `w_*` only the
    /// blend; the other genes always).
    pub fn reads(self, key: &str) -> bool {
        match key {
            "kelly" => matches!(self, Self::FractionalKelly | Self::WeightedBlend),
            "w_eq" | "w_rp" | "w_k" => self == Self::WeightedBlend,
            _ => true,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::EqualWeight => "equal_weight",
            Self::RiskParity => "risk_parity",
            Self::FractionalKelly => "fractional_kelly",
            Self::WeightedBlend => "weighted_blend",
        }
    }
}

impl FromStr for AllocatorName {
    type Err = GenomeError;
    fn from_str(s: &str) -> Result<Self, GenomeError> {
        Ok(match s.trim() {
            "equal_weight" | "equal" => Self::EqualWeight,
            "risk_parity" | "risk_parity_capped" => Self::RiskParity,
            "fractional_kelly" | "kelly" => Self::FractionalKelly,
            "weighted_blend" | "blend" => Self::WeightedBlend,
            other => return Err(GenomeError::Name(other.to_owned())),
        })
    }
}

impl fmt::Display for AllocatorName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GenomeError {
    #[error("allocator genome must start with `{PREFIX}`: `{0}`")]
    Prefix(String),
    #[error("unknown allocator `{0}` (equal_weight | risk_parity | fractional_kelly | weighted_blend)")]
    Name(String),
    #[error("gene `{0}`: expected `key=value`")]
    Pair(String),
    #[error("gene `{key}`: bad number `{value}`")]
    Number { key: String, value: String },
    #[error("unknown gene `{0}`")]
    Gene(String),
    #[error("gene `{0}` given twice")]
    Duplicate(String),
}

/// `(key, min, max)` of every numeric gene, in canonical text order.
pub const GENE_BOUNDS: [(&str, f64, f64); 8] = [
    ("cap", 0.05, 1.0),
    ("shrink", 0.0, 3.0),
    ("kelly", 0.01, 1.0),
    ("hl", 2.0, 100_000.0),
    ("pv", 1e-5, 1.0),
    ("w_eq", 0.0, 1.0),
    ("w_rp", 0.0, 1.0),
    ("w_k", 0.0, 1.0),
];

/// One allocator candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AllocatorGenome {
    pub name: AllocatorName,
    /// Max capital weight per strategy.
    pub cap: f64,
    /// Exponent of the exposure-uncertainty shrink `mean / (1 + stddev)^shrink` (1 = D-040 netting).
    pub shrink: f64,
    /// Kelly fraction (`fractional_kelly`, and the Kelly leg of a blend).
    pub kelly: f64,
    /// Half-life of the performance tracker in steps (`decay = 0.5^(1/hl)`).
    pub half_life: f64,
    /// Prior per-step volatility of a strategy without observations.
    pub prior_vol: f64,
    /// Blend weights of the equal-weight, risk-parity and Kelly legs (`weighted_blend`).
    pub w_eq: f64,
    pub w_rp: f64,
    pub w_k: f64,
}

impl Default for AllocatorGenome {
    /// The live default of `risk.toml` (D-040): equal weight, cap 0.5, Kelly 0.25.
    fn default() -> Self {
        Self {
            name: AllocatorName::EqualWeight,
            cap: 0.5,
            shrink: 1.0,
            kelly: 0.25,
            half_life: 240.0,
            prior_vol: 0.01,
            w_eq: 1.0,
            w_rp: 0.0,
            w_k: 0.0,
        }
    }
}

/// Clamps into the gene's bounds and quantises to the text precision (6 decimals), so a
/// genome always equals its own text form.
fn clamp_gene(key: &str, v: f64) -> f64 {
    let (_, lo, hi) = GENE_BOUNDS.iter().find(|(k, _, _)| *k == key).copied().unwrap_or((key, f64::MIN, f64::MAX));
    let v = if v.is_finite() { v.clamp(lo, hi) } else { lo };
    ((v * 1e6).round() / 1e6).clamp(lo, hi)
}

/// Canonical number text: shortest round-trippable form with at most 6 significant decimals.
fn fmt_num(v: f64) -> String {
    let s = format!("{:.6}", v);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() || s == "-0" { "0".into() } else { s.to_owned() }
}

impl AllocatorGenome {
    /// Named allocator with the default genes.
    pub fn named(name: AllocatorName) -> Self {
        let mut g = Self { name, ..Self::default() };
        match name {
            AllocatorName::EqualWeight => (g.w_eq, g.w_rp, g.w_k) = (1.0, 0.0, 0.0),
            AllocatorName::RiskParity => (g.w_eq, g.w_rp, g.w_k) = (0.0, 1.0, 0.0),
            AllocatorName::FractionalKelly => (g.w_eq, g.w_rp, g.w_k) = (0.0, 0.0, 1.0),
            AllocatorName::WeightedBlend => (g.w_eq, g.w_rp, g.w_k) = (0.34, 0.33, 0.33),
        }
        g
    }

    /// Parses the text form (`alloc:name=…;cap=…;…`); missing genes take the defaults of the
    /// named allocator, values are clamped into [`GENE_BOUNDS`].
    pub fn parse(text: &str) -> Result<Self, GenomeError> {
        let t = text.trim();
        let Some(body) = t.strip_prefix(PREFIX) else { return Err(GenomeError::Prefix(t.to_owned())) };
        let mut name: Option<AllocatorName> = None;
        let mut genes: BTreeMap<&str, f64> = BTreeMap::new();
        for part in body.split(';').map(str::trim).filter(|p| !p.is_empty()) {
            let Some((k, v)) = part.split_once('=') else { return Err(GenomeError::Pair(part.to_owned())) };
            let (k, v) = (k.trim(), v.trim());
            if k == "name" {
                if name.is_some() {
                    return Err(GenomeError::Duplicate("name".into()));
                }
                name = Some(v.parse()?);
                continue;
            }
            let Some((key, _, _)) = GENE_BOUNDS.iter().find(|(g, _, _)| *g == k) else {
                return Err(GenomeError::Gene(k.to_owned()));
            };
            let x: f64 = v.parse().map_err(|_| GenomeError::Number { key: k.to_owned(), value: v.to_owned() })?;
            if genes.insert(key, x).is_some() {
                return Err(GenomeError::Duplicate(k.to_owned()));
            }
        }
        let mut g = Self::named(name.unwrap_or(AllocatorName::EqualWeight));
        for (k, v) in genes {
            g.set(k, v);
        }
        Ok(g.clamped())
    }

    /// Numeric genes as a map (canonical keys).
    pub fn params(&self) -> BTreeMap<String, f64> {
        BTreeMap::from([
            ("cap".to_owned(), self.cap),
            ("shrink".to_owned(), self.shrink),
            ("kelly".to_owned(), self.kelly),
            ("hl".to_owned(), self.half_life),
            ("pv".to_owned(), self.prior_vol),
            ("w_eq".to_owned(), self.w_eq),
            ("w_rp".to_owned(), self.w_rp),
            ("w_k".to_owned(), self.w_k),
        ])
    }

    /// Sets one gene by key (unknown keys are ignored); not clamped.
    pub fn set(&mut self, key: &str, v: f64) {
        match key {
            "cap" => self.cap = v,
            "shrink" => self.shrink = v,
            "kelly" => self.kelly = v,
            "hl" => self.half_life = v,
            "pv" => self.prior_vol = v,
            "w_eq" => self.w_eq = v,
            "w_rp" => self.w_rp = v,
            "w_k" => self.w_k = v,
            _ => {}
        }
    }

    pub fn get(&self, key: &str) -> Option<f64> {
        self.params().get(key).copied()
    }

    /// Every gene clamped into [`GENE_BOUNDS`] and quantised to the text precision; genes the
    /// named allocator does not read are reset to its defaults (so a genome is its phenotype:
    /// two texts that build the same allocator have the same id); blend weights of a
    /// `weighted_blend` that sum to zero fall back to equal thirds.
    pub fn clamped(mut self) -> Self {
        let defaults = Self::named(self.name);
        for (k, _, _) in GENE_BOUNDS {
            let v = if self.name.reads(k) { self.get(k).unwrap_or(0.0) } else { defaults.get(k).unwrap_or(0.0) };
            self.set(k, clamp_gene(k, v));
        }
        if self.name == AllocatorName::WeightedBlend && self.w_eq + self.w_rp + self.w_k <= 0.0 {
            (self.w_eq, self.w_rp, self.w_k) = (0.34, 0.33, 0.33);
        }
        self
    }

    /// Stable identity: `alloc-<hash of the canonical text>`.
    pub fn id(&self) -> String {
        format!("alloc-{:016x}", stable_hash(&self.to_string()))
    }

    /// The allocator this genome describes.
    pub fn build(&self) -> Box<dyn Allocator> {
        match self.name {
            AllocatorName::EqualWeight => Box::new(EqualWeight { max_weight: self.cap }),
            AllocatorName::RiskParity => Box::new(RiskParityCapped { max_weight: self.cap }),
            AllocatorName::FractionalKelly => Box::new(FractionalKelly { fraction: self.kelly, max_weight: self.cap }),
            AllocatorName::WeightedBlend => Box::new(WeightedBlend {
                max_weight: self.cap,
                kelly_fraction: self.kelly,
                w_equal: self.w_eq,
                w_risk_parity: self.w_rp,
                w_kelly: self.w_k,
            }),
        }
    }

    /// EWMA decay of the performance tracker: `0.5^(1/half_life)`.
    pub fn decay(&self) -> f64 {
        (0.5f64).powf(1.0 / self.half_life.max(1.0))
    }

    /// A fresh performance tracker with this genome's memory and prior.
    pub fn tracker(&self) -> PerformanceTracker {
        PerformanceTracker::new(self.decay(), self.prior_vol)
    }

    /// A random genome (uniform in the log-ish ranges of each gene).
    pub fn random(rng: &mut Rng) -> Self {
        let name = AllocatorName::ALL[rng.below(AllocatorName::ALL.len())];
        let mut g = Self::named(name);
        g.cap = 0.1 + 0.9 * rng.uniform();
        g.shrink = 2.0 * rng.uniform();
        g.kelly = 0.05 + 0.75 * rng.uniform();
        g.half_life = (1.0 + 4.0 * rng.uniform()).exp2() * 10.0; // 20 … 320 steps
        g.prior_vol = 10f64.powf(-4.0 + 2.5 * rng.uniform()); // 1e-4 … 0.03
        if name == AllocatorName::WeightedBlend {
            g.w_eq = rng.uniform();
            g.w_rp = rng.uniform();
            g.w_k = rng.uniform();
        }
        g.clamped()
    }
}

/// The allocator selection risk-engine accepts at runtime (`ctl.risk.limits {set: {allocator:
/// AllocatorSpec}}`) and the orchestrator writes into `fa-artifacts/allocators/<generation>.toml`:
/// `{"name": "weighted_blend", "params": {"cap": 0.4, "shrink": 1, "kelly": 0.25, "hl": 240,
/// "pv": 0.01, "w_eq": 0.2, "w_rp": 0.5, "w_k": 0.3}}`. Missing params take the named defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AllocatorSpec {
    pub name: String,
    #[serde(default)]
    pub params: BTreeMap<String, f64>,
}

impl AllocatorGenome {
    /// The `{name, params}` selection payload of this genome.
    pub fn spec(&self) -> AllocatorSpec {
        AllocatorSpec { name: self.name.to_string(), params: self.params() }
    }

    /// Genome from a `{name, params}` payload (unknown names / params are errors, values clamped).
    pub fn from_spec(spec: &AllocatorSpec) -> Result<Self, GenomeError> {
        let mut g = Self::named(spec.name.parse()?);
        for (k, v) in &spec.params {
            if !GENE_BOUNDS.iter().any(|(g, _, _)| g == k) {
                return Err(GenomeError::Gene(k.clone()));
            }
            g.set(k, *v);
        }
        Ok(g.clamped())
    }
}

impl fmt::Display for AllocatorGenome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{PREFIX}name={}", self.name)?;
        for (k, _, _) in GENE_BOUNDS {
            write!(f, ";{k}={}", fmt_num(self.get(k).unwrap_or(0.0)))?;
        }
        Ok(())
    }
}

impl FromStr for AllocatorGenome {
    type Err = GenomeError;
    fn from_str(s: &str) -> Result<Self, GenomeError> {
        Self::parse(s)
    }
}

/// FNV-1a over the text (same recipe as `fa_decoders::genome::stable_hash`).
pub fn stable_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl Evolvable for AllocatorGenome {
    fn genome_id(&self) -> String {
        self.id()
    }
    /// One family: every allocator shares the gene set, so any two may cross.
    fn family(&self) -> &str {
        "allocator"
    }
}

/// Variation operators for allocator genomes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AllocatorOps {
    /// Relative Gaussian scale of a gene mutation.
    pub sigma: f64,
    /// Probability that a mutation switches the allocator name instead of a gene.
    pub name_switch_prob: f64,
    /// Probability that every gene (instead of one) is perturbed.
    pub all_genes_prob: f64,
}

impl Default for AllocatorOps {
    fn default() -> Self {
        Self { sigma: 0.2, name_switch_prob: 0.1, all_genes_prob: 0.25 }
    }
}

impl Operators<AllocatorGenome> for AllocatorOps {
    /// Uniform crossover: every gene (and the name) from either parent with p = 0.5.
    fn crossover(&self, a: &AllocatorGenome, b: &AllocatorGenome, rng: &mut Rng) -> AllocatorGenome {
        let mut c = a.clone();
        if rng.uniform() < 0.5 {
            c.name = b.name;
        }
        for (k, _, _) in GENE_BOUNDS {
            if rng.uniform() < 0.5 {
                c.set(k, b.get(k).unwrap_or(0.0));
            }
        }
        c.clamped()
    }

    fn mutate(&self, g: &AllocatorGenome, rng: &mut Rng) -> (AllocatorGenome, &'static str) {
        let mut c = g.clone();
        if rng.uniform() < self.name_switch_prob {
            let others: Vec<AllocatorName> = AllocatorName::ALL.iter().copied().filter(|n| *n != g.name).collect();
            c.name = others[rng.below(others.len())];
            return (c.clamped(), "name_switch");
        }
        let perturb = |rng: &mut Rng, v: f64, lo: f64, hi: f64| {
            let scale = if v.abs() > 1e-9 { v.abs() } else { (hi - lo) * 0.05 };
            v + self.sigma * rng.normal() * scale
        };
        // only genes the allocator reads: a change elsewhere would be a phenotypic duplicate
        let active: Vec<(&str, f64, f64)> = GENE_BOUNDS.iter().copied().filter(|(k, _, _)| g.name.reads(k)).collect();
        if rng.uniform() < self.all_genes_prob {
            for (k, lo, hi) in &active {
                let v = c.get(k).unwrap_or(0.0);
                c.set(k, perturb(rng, v, *lo, *hi));
            }
            (c.clamped(), "gaussian_all")
        } else {
            let (k, lo, hi) = active[rng.below(active.len())];
            let v = c.get(k).unwrap_or(0.0);
            c.set(k, perturb(rng, v, lo, hi));
            (c.clamped(), "gaussian_gene")
        }
    }
}

#[cfg(test)]
mod tests {
    use fa_evolution::vary::vary_with;

    use super::*;
    use crate::StrategyStats;

    #[test]
    fn text_round_trip_is_canonical_and_clamped() {
        let g = AllocatorGenome::parse("alloc:name=risk_parity;cap=0.4;hl=100").unwrap();
        assert_eq!(g.name, AllocatorName::RiskParity);
        assert_eq!((g.cap, g.half_life, g.shrink, g.kelly), (0.4, 100.0, 1.0, 0.25));
        assert_eq!((g.w_eq, g.w_rp, g.w_k), (0.0, 1.0, 0.0), "named defaults fill the missing genes");
        let text = g.to_string();
        assert_eq!(text, "alloc:name=risk_parity;cap=0.4;shrink=1;kelly=0.25;hl=100;pv=0.01;w_eq=0;w_rp=1;w_k=0");
        assert_eq!(AllocatorGenome::parse(&text).unwrap(), g, "Display → parse round trip");
        assert_eq!(AllocatorGenome::parse(&text).unwrap().to_string(), text);
        assert_eq!(g.id(), AllocatorGenome::parse(&text).unwrap().id());
        // order and spacing do not matter; values are clamped
        let h = AllocatorGenome::parse(" alloc: hl=100 ; cap=7 ; name=risk_parity ").unwrap();
        assert_eq!(h.cap, 1.0);
        assert_eq!(h.to_string(), text.replace("cap=0.4", "cap=1"));
        assert_eq!(AllocatorGenome::parse("alloc:").unwrap(), AllocatorGenome::default());
        // genes the allocator does not read are normalised away: same phenotype, same id
        let ew = AllocatorGenome::parse("alloc:name=equal_weight;cap=0.5;w_k=0.007668;kelly=0.9").unwrap();
        assert_eq!(ew, AllocatorGenome::named(AllocatorName::EqualWeight));
        assert_eq!(ew.id(), AllocatorGenome::named(AllocatorName::EqualWeight).id());
        assert_eq!(AllocatorGenome::parse("alloc:name=fractional_kelly;kelly=0.9").unwrap().kelly, 0.9);
        assert!(AllocatorName::WeightedBlend.reads("w_k") && !AllocatorName::RiskParity.reads("kelly"));
        assert_eq!(
            AllocatorGenome::default().to_string(),
            AllocatorGenome::named(AllocatorName::EqualWeight).to_string()
        );
        // errors are specific
        assert!(matches!(AllocatorGenome::parse("forecast:h=3"), Err(GenomeError::Prefix(_))));
        assert!(matches!(AllocatorGenome::parse("alloc:name=evolved"), Err(GenomeError::Name(_))));
        assert!(matches!(AllocatorGenome::parse("alloc:cap"), Err(GenomeError::Pair(_))));
        assert!(matches!(AllocatorGenome::parse("alloc:cap=x"), Err(GenomeError::Number { .. })));
        assert!(matches!(AllocatorGenome::parse("alloc:foo=1"), Err(GenomeError::Gene(_))));
        assert!(matches!(AllocatorGenome::parse("alloc:cap=1;cap=2"), Err(GenomeError::Duplicate(_))));
        // JSON too
        let j = serde_json::to_string(&g).unwrap();
        assert!(j.contains("\"name\":\"risk_parity\""));
        assert_eq!(serde_json::from_str::<AllocatorGenome>(&j).unwrap(), g);
    }

    #[test]
    fn selection_payload_round_trips() {
        let g = AllocatorGenome::parse("alloc:name=weighted_blend;cap=0.4;w_eq=0.2;w_rp=0.5;w_k=0.3").unwrap();
        let spec = g.spec();
        let j = serde_json::to_value(&spec).unwrap();
        assert_eq!(j["name"], "weighted_blend");
        assert_eq!(j["params"]["cap"], 0.4);
        assert_eq!(j["params"].as_object().unwrap().len(), 8);
        let back: AllocatorSpec = serde_json::from_value(j).unwrap();
        assert_eq!(AllocatorGenome::from_spec(&back).unwrap(), g);
        let partial: AllocatorSpec =
            serde_json::from_str(r#"{"name":"fractional_kelly","params":{"kelly":0.1}}"#).unwrap();
        let k = AllocatorGenome::from_spec(&partial).unwrap();
        assert_eq!((k.name, k.kelly, k.cap), (AllocatorName::FractionalKelly, 0.1, 0.5));
        assert!(AllocatorGenome::from_spec(&AllocatorSpec { name: "x".into(), params: BTreeMap::new() }).is_err());
        let bad = AllocatorSpec { name: "risk_parity".into(), params: BTreeMap::from([("foo".into(), 1.0)]) };
        assert!(matches!(AllocatorGenome::from_spec(&bad), Err(GenomeError::Gene(_))));
    }

    #[test]
    fn builds_the_named_allocator_and_tracker() {
        let stats = vec![
            StrategyStats { strategy_id: "a".into(), mean_return: 0.01, volatility: 0.1 },
            StrategyStats { strategy_id: "b".into(), mean_return: 0.01, volatility: 0.2 },
        ];
        for name in AllocatorName::ALL {
            let g = AllocatorGenome::named(name);
            let a = g.build().allocate(&stats);
            assert!(!a.is_empty(), "{name}");
            assert!(a.values().all(|w| *w >= 0.0 && *w <= g.cap + 1e-12), "{name}: {a:?}");
            assert!(a.values().sum::<f64>() <= 1.0 + 1e-9, "{name}");
            assert_eq!(format!("{}", name.to_string().parse::<AllocatorName>().unwrap()), name.to_string());
        }
        assert_eq!(AllocatorGenome::named(AllocatorName::FractionalKelly).build().id(), "fractional_kelly");
        let blend = AllocatorGenome::parse("alloc:name=weighted_blend;w_eq=1;w_rp=1;w_k=0;cap=1").unwrap();
        let a = blend.build().allocate(&stats);
        // equal weight (0.5 / 0.5) and risk parity (2/3 / 1/3) averaged
        assert!((a["a"] - (0.5 + 2.0 / 3.0) / 2.0).abs() < 1e-9, "{a:?}");
        assert!((a["b"] - (0.5 + 1.0 / 3.0) / 2.0).abs() < 1e-9);
        let g = AllocatorGenome::parse("alloc:hl=10;pv=0.02").unwrap();
        assert!((g.decay() - 0.5f64.powf(0.1)).abs() < 1e-12);
        assert_eq!(g.tracker().prior_volatility, 0.02);
        assert_eq!(g.tracker().stats_for(["x"])[0].volatility, 0.02);
    }

    #[test]
    fn operators_are_deterministic_bounded_and_change_something() {
        let parents = vec![
            AllocatorGenome::named(AllocatorName::RiskParity),
            AllocatorGenome::parse("alloc:name=fractional_kelly;kelly=0.5;cap=0.3").unwrap(),
            AllocatorGenome::named(AllocatorName::WeightedBlend),
        ];
        let ops = AllocatorOps::default();
        let a = vary_with(&parents, 40, 7, 0.5, &ops);
        let b = vary_with(&parents, 40, 7, 0.5, &ops);
        assert_eq!(a, b, "deterministic in seed");
        assert_ne!(a, vary_with(&parents, 40, 8, 0.5, &ops));
        let mut ops_seen = std::collections::BTreeSet::new();
        for c in &a {
            let g = &c.genome;
            for (k, lo, hi) in GENE_BOUNDS {
                let v = g.get(k).unwrap();
                assert!((lo..=hi).contains(&v), "{k}={v} out of bounds");
            }
            assert_eq!(AllocatorGenome::parse(&g.to_string()).unwrap(), *g, "child text round trips");
            ops_seen.insert(c.op.clone());
            assert!(!c.parents.is_empty());
        }
        assert!(ops_seen.iter().any(|o| o.starts_with("crossover+")), "{ops_seen:?}");
        assert!(ops_seen.iter().any(|o| o.ends_with("gaussian_gene")), "{ops_seen:?}");
        assert!(ops_seen.iter().any(|o| o.ends_with("name_switch")), "{ops_seen:?}");
        let changed = a.iter().filter(|c| !parents.contains(&c.genome)).count();
        assert!(changed > 30, "{changed} of 40 children differ from every parent");
        // a gene mutation of a non-blend allocator never lands on a gene it does not read
        let ew = vec![AllocatorGenome::named(AllocatorName::EqualWeight)];
        let only_gene = AllocatorOps { name_switch_prob: 0.0, ..ops };
        for c in vary_with(&ew, 50, 1, 0.0, &only_gene) {
            assert_ne!(c.genome, ew[0], "every mutation of an equal-weight genome changes its phenotype");
        }
        // random genomes are valid and diverse
        let mut rng = Rng::seed(3);
        let rs: Vec<AllocatorGenome> = (0..20).map(|_| AllocatorGenome::random(&mut rng)).collect();
        let names: std::collections::BTreeSet<_> = rs.iter().map(|g| g.name).collect();
        assert!(names.len() >= 3, "{names:?}");
        assert!(rs.iter().all(|g| AllocatorGenome::parse(&g.to_string()).unwrap() == *g));
    }
}
