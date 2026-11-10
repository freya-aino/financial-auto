//! Concrete genome encoding for the **baseline decoder families** (spec 34 §4.2, provisional).
//!
//! The evo-orchestrator forwards whatever the UI/API sent as `Job.genome_id`; this module
//! turns that string into a deterministic [`DecoderGenome`]. Three forms are accepted:
//!
//! 1. **Key/value form** — `<family>[:key=value[;key=value]...]`, e.g.
//!    `forecast:h=5;lb=32;kernel=16;levels=0.1/0.5/0.9` or `retrieval:k=20;tau=30;gap=5`.
//!    Genes are separated by `;` (or whitespace). A comma also works when the string
//!    reaches the worker intact, but the `ctl.evo.start` transport joins/splits the genome
//!    list on commas, so **never use commas in genomes sent through the API**.
//! 2. **JSON form** — `{"id":"g1","family":"retrieval","params":{"k":20,"tau":30}}`
//!    (`params` values are numbers; `levels` may be an array; `ts` is a string). JSON
//!    contains commas, so it only survives in `Job.spec`, not in the `genomes` list.
//! 3. **Bare id** — anything else (`g1`, `abc-123`, a ULID). The genome is *derived* from a
//!    stable hash of the id: the family alternates with the hash parity and every numeric
//!    gene is jittered around its default. This keeps plain ids from the Evolution Lab
//!    form usable and reproducible.
//!
//! Families (aliases in parentheses): `forecast` (`quantile`, `rnn_prior`) → empirical
//! quantile forecaster; `retrieval` (`analog`) → time-discounted analog retrieval.
//!
//! Common genes: `ts` timescale of the resampled series (default `1s`, the only live feed),
//! `h` forecast horizon in steps, `lb` lookback in steps, `levels` quantile levels, `inst`
//! instrument UUID filter (empty = every buffered instrument).
//! Forecast genes: `kernel` (exponential half-life in steps), `scale` (spread multiplier),
//! `shrink` (drift shrinkage in [0, 1]).
//! Retrieval genes: `k` (analogs kept), `tau` (time-decay in days), `gap` (leakage guard in
//! steps, ≥ `h`), `metric` (`cos` | `euclid`), `lat` (weight of the latent-similarity term in
//! [0, 1], used only when a [`crate::DecodeCtx`] carries latents for the instrument).
//!
//! [`DecoderGenome`] implements `Display`: the canonical kv form, which [`DecoderGenome::parse`]
//! reads back to an equal genome (round-trip tested). [`DecoderGenome::params`] /
//! [`DecoderGenome::with_params`] expose the numeric genes as a `name → f64` map for the
//! `fa-evolution` operators (clamped back into their valid ranges).

use std::{collections::BTreeMap, fmt, hash::Hasher};

use fa_core::Timescale;
use serde::{Deserialize, Serialize};

use crate::Family;

/// Baseline family the genome instantiates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BaselineFamily {
    /// Empirical quantile forecaster; occupies the `rnn_prior` output family slot (the
    /// family whose output contract is forecast quantiles) until a real RNN prior exists.
    Forecast,
    /// Analog retrieval with time distance as an explicit feature.
    Retrieval,
}

impl BaselineFamily {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "forecast" | "quantile" | "rnn_prior" => Some(Self::Forecast),
            "retrieval" | "analog" => Some(Self::Retrieval),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Forecast => "forecast",
            Self::Retrieval => "retrieval",
        }
    }

    /// Family string used on `decoder.output.<family>` (proto `DecoderOutput.family`).
    pub fn output_family(self) -> Family {
        match self {
            Self::Forecast => Family::RnnPrior,
            Self::Retrieval => Family::Retrieval,
        }
    }
}

/// Similarity metric of the retrieval family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    Cosine,
    Euclid,
}

/// Decoded, validated genome of a baseline decoder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecoderGenome {
    /// The raw string the job carried (reported back unchanged).
    pub id: String,
    pub family: BaselineFamily,
    pub timescale: Timescale,
    /// Forecast horizon in steps of `timescale`.
    pub horizon: usize,
    /// Lookback window in steps.
    pub lookback: usize,
    /// Quantile levels in (0, 1), strictly increasing.
    pub levels: Vec<f64>,
    /// Instrument UUID filter; empty = all instruments.
    pub instrument: Option<String>,
    /// Forecast: exponential half-life (steps) of the return weighting.
    pub kernel: f64,
    /// Forecast: multiplicative spread scale.
    pub scale: f64,
    /// Forecast: drift shrinkage in [0, 1] (0 = no drift, 1 = full EW mean).
    pub shrink: f64,
    /// Retrieval: analogs kept.
    pub k: usize,
    /// Retrieval: time-decay constant in days.
    pub tau_days: f64,
    /// Retrieval: minimum gap (steps) between an analog's outcome window and now.
    pub min_gap: usize,
    pub metric: Metric,
    /// Retrieval: weight of the latent-similarity term in [0, 1] (`lat` gene).
    #[serde(default = "default_latent_weight")]
    pub latent_weight: f64,
    /// How the genome was obtained: `kv`, `json` or `derived`.
    pub encoding: String,
}

fn default_latent_weight() -> f64 {
    0.5
}

/// Numeric gene names exposed to the evolution operators (`params()` keys).
pub const NUMERIC_GENES: [&str; 9] = ["h", "lb", "kernel", "scale", "shrink", "k", "tau", "gap", "lat"];

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum GenomeError {
    #[error("unknown decoder family `{0}`")]
    UnknownFamily(String),
    #[error("gene `{key}`: {msg}")]
    Gene { key: String, msg: String },
    #[error("empty genome")]
    Empty,
}

pub const DEFAULT_LEVELS: [f64; 5] = [0.05, 0.25, 0.5, 0.75, 0.95];

impl DecoderGenome {
    /// Defaults for a family (horizon 5, lookback 32, 1 s timescale).
    pub fn default_for(id: &str, family: BaselineFamily) -> Self {
        Self {
            id: id.to_owned(),
            family,
            timescale: Timescale::Second,
            horizon: 5,
            lookback: 32,
            levels: DEFAULT_LEVELS.to_vec(),
            instrument: None,
            kernel: 16.0,
            scale: 1.0,
            shrink: 0.5,
            k: 20,
            tau_days: 30.0,
            min_gap: 5,
            metric: Metric::Cosine,
            latent_weight: 0.5,
            encoding: "default".into(),
        }
    }

    /// Numeric genes as `name → value` (see [`NUMERIC_GENES`]); integers are exact.
    pub fn params(&self) -> BTreeMap<String, f64> {
        BTreeMap::from([
            ("h".to_owned(), self.horizon as f64),
            ("lb".to_owned(), self.lookback as f64),
            ("kernel".to_owned(), self.kernel),
            ("scale".to_owned(), self.scale),
            ("shrink".to_owned(), self.shrink),
            ("k".to_owned(), self.k as f64),
            ("tau".to_owned(), self.tau_days),
            ("gap".to_owned(), self.min_gap as f64),
            ("lat".to_owned(), self.latent_weight),
        ])
    }

    /// Copy of this genome with the numeric genes replaced by `params` (missing keys keep
    /// their value, unknown keys are ignored). Values are rounded/clamped into the valid
    /// range instead of rejected, so mutated children are always evaluable. The id becomes
    /// the canonical kv string and `encoding` is `kv`.
    pub fn with_params(&self, params: &BTreeMap<String, f64>) -> Result<Self, GenomeError> {
        let mut g = self.clone();
        let int = |v: f64, lo: usize| if v.is_finite() { (v.round().max(lo as f64)) as usize } else { lo };
        let pos = |v: f64, cur: f64| if v.is_finite() && v > 0.0 { v.max(1e-3) } else { cur };
        for (k, v) in params {
            match k.as_str() {
                "h" => g.horizon = int(*v, 1),
                "lb" => g.lookback = int(*v, 2),
                "kernel" => g.kernel = pos(*v, g.kernel),
                "scale" => g.scale = pos(*v, g.scale),
                "shrink" => g.shrink = if v.is_finite() { v.clamp(0.0, 1.0) } else { g.shrink },
                "k" => g.k = int(*v, 1),
                "tau" => g.tau_days = pos(*v, g.tau_days),
                "gap" => g.min_gap = int(*v, 0),
                "lat" => g.latent_weight = if v.is_finite() { v.clamp(0.0, 1.0) } else { g.latent_weight },
                _ => {}
            }
        }
        g.validate()?;
        g.encoding = "kv".into();
        g.id = g.to_string();
        Ok(g)
    }

    /// Parses any of the three accepted forms (see module docs). Never fails for a bare id.
    pub fn parse(raw: &str) -> Result<Self, GenomeError> {
        let raw = raw.trim();
        if raw.is_empty() {
            return Err(GenomeError::Empty);
        }
        if raw.starts_with('{') {
            return Self::parse_json(raw);
        }
        let (fam, rest) = match raw.split_once(':') {
            Some((f, r)) => (f, Some(r)),
            None => (raw, None),
        };
        if let Some(family) = BaselineFamily::parse(fam) {
            let mut g = Self::default_for(raw, family);
            g.encoding = "kv".into();
            if let Some(rest) = rest {
                for kv in rest.split([';', ',', ' ']).map(str::trim).filter(|s| !s.is_empty()) {
                    let (k, v) = kv
                        .split_once('=')
                        .ok_or_else(|| GenomeError::Gene { key: kv.to_owned(), msg: "expected key=value".into() })?;
                    g.set_gene(k.trim(), v.trim())?;
                }
            }
            g.validate()?;
            return Ok(g);
        }
        if rest.is_some() {
            return Err(GenomeError::UnknownFamily(fam.to_owned()));
        }
        if raw.contains('=') {
            // A bare id never contains `=`; this is a gene fragment, most likely a kv genome
            // that was split on commas by the orchestrator transport.
            return Err(GenomeError::Gene {
                key: raw.to_owned(),
                msg: "gene fragment without a family (genomes must use `;` between genes; commas are split by the transport)".into(),
            });
        }
        Ok(Self::derive(raw))
    }

    fn parse_json(raw: &str) -> Result<Self, GenomeError> {
        #[derive(Deserialize)]
        struct J {
            id: Option<String>,
            family: String,
            #[serde(default)]
            params: BTreeMap<String, serde_json::Value>,
        }
        let j: J =
            serde_json::from_str(raw).map_err(|e| GenomeError::Gene { key: "json".into(), msg: e.to_string() })?;
        let family = BaselineFamily::parse(&j.family).ok_or_else(|| GenomeError::UnknownFamily(j.family.clone()))?;
        let mut g = Self::default_for(j.id.as_deref().unwrap_or(raw), family);
        g.encoding = "json".into();
        for (k, v) in &j.params {
            let s = match v {
                serde_json::Value::Array(xs) => xs
                    .iter()
                    .map(|x| x.as_f64().map(|f| f.to_string()).unwrap_or_default())
                    .collect::<Vec<_>>()
                    .join("/"),
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            g.set_gene(k, &s)?;
        }
        g.validate()?;
        Ok(g)
    }

    /// Deterministic genome derived from an opaque id (hash-jittered defaults).
    pub fn derive(id: &str) -> Self {
        let h = stable_hash(id);
        let family = if h.is_multiple_of(2) { BaselineFamily::Forecast } else { BaselineFamily::Retrieval };
        let mut g = Self::default_for(id, family);
        g.encoding = "derived".into();
        // Each gene takes a different byte of the hash; ranges are deliberately modest so
        // every derived genome is evaluable on a short live window.
        let b = |i: u32| ((h >> (8 * i)) & 0xff) as f64 / 255.0;
        g.horizon = 2 + (b(1) * 7.0) as usize; // 2..=8
        g.lookback = 16 + (b(2) * 48.0) as usize; // 16..=64
        g.kernel = 4.0 + b(3) * 28.0; // 4..32
        g.scale = 0.6 + b(4) * 0.9; // 0.6..1.5
        g.shrink = b(5);
        g.k = 5 + (b(6) * 35.0) as usize; // 5..=40
        g.tau_days = 1.0 + b(7) * 59.0; // 1..60
        g.min_gap = g.horizon;
        g.metric = if (h >> 60) & 1 == 0 { Metric::Cosine } else { Metric::Euclid };
        g
    }

    fn set_gene(&mut self, key: &str, v: &str) -> Result<(), GenomeError> {
        let err = |msg: &str| GenomeError::Gene { key: key.to_owned(), msg: msg.to_owned() };
        let num = |v: &str| v.parse::<f64>().map_err(|_| err("not a number"));
        let int = |v: &str| {
            v.parse::<f64>()
                .ok()
                .filter(|x| *x >= 0.0)
                .map(|x| x as usize)
                .ok_or_else(|| err("not a non-negative integer"))
        };
        match key {
            "ts" | "timescale" => self.timescale = v.parse().map_err(|_| err("unknown timescale"))?,
            "h" | "horizon" => self.horizon = int(v)?,
            "lb" | "lookback" => self.lookback = int(v)?,
            "levels" => {
                self.levels = v.split(['/', ';']).map(|s| num(s.trim())).collect::<Result<_, _>>()?;
            }
            "inst" | "instrument" => self.instrument = Some(v.to_owned()).filter(|s| !s.is_empty()),
            "kernel" => self.kernel = num(v)?,
            "scale" => self.scale = num(v)?,
            "shrink" => self.shrink = num(v)?,
            "k" => self.k = int(v)?,
            "tau" | "tau_days" => self.tau_days = num(v)?,
            "gap" | "min_gap" => self.min_gap = int(v)?,
            "metric" => {
                self.metric = match v.to_ascii_lowercase().as_str() {
                    "cos" | "cosine" => Metric::Cosine,
                    "euclid" | "l2" => Metric::Euclid,
                    _ => return Err(err("expected cos|euclid")),
                }
            }
            "lat" | "latent" => self.latent_weight = num(v)?,
            _ => return Err(err("unknown gene")),
        }
        Ok(())
    }

    fn validate(&mut self) -> Result<(), GenomeError> {
        let gene = |key: &str, msg: &str| GenomeError::Gene { key: key.into(), msg: msg.into() };
        if self.horizon == 0 {
            return Err(gene("h", "must be ≥ 1"));
        }
        if self.lookback < 2 {
            return Err(gene("lb", "must be ≥ 2"));
        }
        if self.levels.is_empty() || self.levels.iter().any(|l| !(*l > 0.0 && *l < 1.0)) {
            return Err(gene("levels", "levels must lie in (0, 1)"));
        }
        self.levels.sort_by(f64::total_cmp);
        self.levels.dedup();
        if self.kernel.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(gene("kernel", "must be > 0"));
        }
        if self.scale.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(gene("scale", "must be > 0"));
        }
        if !(0.0..=1.0).contains(&self.shrink) {
            return Err(gene("shrink", "must be in [0, 1]"));
        }
        if self.k == 0 {
            return Err(gene("k", "must be ≥ 1"));
        }
        if self.tau_days.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(gene("tau", "must be > 0"));
        }
        if !(0.0..=1.0).contains(&self.latent_weight) {
            return Err(gene("lat", "must be in [0, 1]"));
        }
        // Leakage guard (spec 34 §6.1): an analog's outcome window must end before now.
        self.min_gap = self.min_gap.max(self.horizon);
        Ok(())
    }

    /// Short human-readable summary (`forecast h=5 lb=32 ...`).
    pub fn summary(&self) -> String {
        match self.family {
            BaselineFamily::Forecast => format!(
                "forecast ts={} h={} lb={} kernel={:.1} scale={:.2} shrink={:.2}",
                self.timescale, self.horizon, self.lookback, self.kernel, self.scale, self.shrink
            ),
            BaselineFamily::Retrieval => format!(
                "retrieval ts={} h={} lb={} k={} tau={:.1}d gap={} metric={:?}",
                self.timescale, self.horizon, self.lookback, self.k, self.tau_days, self.min_gap, self.metric
            ),
        }
    }
}

impl fmt::Display for DecoderGenome {
    /// Canonical kv form: `<family>:ts=..;h=..;lb=..;levels=a/b/c;kernel=..;scale=..;shrink=..;
    /// k=..;tau=..;gap=..;metric=..;lat=..[;inst=..]`. Every gene is printed (also the other
    /// family's, so a child keeps them when its family gene flips) and floats print in Rust's
    /// shortest round-trip form, so `parse(to_string())` is lossless.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let levels: Vec<String> = self.levels.iter().map(|l| l.to_string()).collect();
        write!(
            f,
            "{}:ts={};h={};lb={};levels={};kernel={};scale={};shrink={};k={};tau={};gap={};metric={};lat={}",
            self.family.as_str(),
            self.timescale,
            self.horizon,
            self.lookback,
            levels.join("/"),
            self.kernel,
            self.scale,
            self.shrink,
            self.k,
            self.tau_days,
            self.min_gap,
            match self.metric {
                Metric::Cosine => "cos",
                Metric::Euclid => "euclid",
            },
            self.latent_weight
        )?;
        if let Some(inst) = &self.instrument {
            write!(f, ";inst={inst}")?;
        }
        Ok(())
    }
}

/// FNV-1a 64 over the id: stable across runs, platforms and Rust versions
/// (unlike `DefaultHasher`).
pub fn stable_hash(s: &str) -> u64 {
    struct Fnv(u64);
    impl Hasher for Fnv {
        fn finish(&self) -> u64 {
            self.0
        }
        fn write(&mut self, bytes: &[u8]) {
            for b in bytes {
                self.0 ^= u64::from(*b);
                self.0 = self.0.wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
    }
    let mut h = Fnv(0xcbf2_9ce4_8422_2325);
    h.write(s.as_bytes());
    // splitmix64 finaliser: plain FNV barely spreads a one-character difference into the
    // high bytes, which made `g1`..`g5` derive near-identical genomes.
    let mut z = h.finish();
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_form_parses_and_validates() {
        let g = DecoderGenome::parse("forecast:h=3;lb=20;levels=0.1/0.5/0.9;kernel=8").unwrap();
        assert_eq!(DecoderGenome::parse("forecast:h=3,lb=20,levels=0.1/0.5/0.9,kernel=8").unwrap().levels, g.levels);
        assert_eq!(g.family, BaselineFamily::Forecast);
        assert_eq!((g.horizon, g.lookback), (3, 20));
        assert_eq!(g.levels, vec![0.1, 0.5, 0.9]);
        assert_eq!(g.kernel, 8.0);
        assert_eq!(g.encoding, "kv");

        let r = DecoderGenome::parse("retrieval:k=7;tau=10;gap=1;h=4;metric=euclid").unwrap();
        assert!(
            matches!(DecoderGenome::parse("lb=16"), Err(GenomeError::Gene { .. })),
            "split fragment must not derive"
        );
        assert_eq!(r.family, BaselineFamily::Retrieval);
        assert_eq!(r.k, 7);
        assert_eq!(r.min_gap, 4, "gap is raised to the horizon (leakage guard)");
        assert_eq!(r.metric, Metric::Euclid);

        assert!(matches!(DecoderGenome::parse("forecast:h=0"), Err(GenomeError::Gene { .. })));
        assert!(matches!(DecoderGenome::parse("forecast:bogus=1"), Err(GenomeError::Gene { .. })));
        assert!(matches!(DecoderGenome::parse("visual:h=1"), Err(GenomeError::UnknownFamily(_))));
        assert_eq!(DecoderGenome::parse("  "), Err(GenomeError::Empty));
    }

    #[test]
    fn json_form_parses() {
        let g = DecoderGenome::parse(r#"{"id":"j1","family":"analog","params":{"k":12,"levels":[0.2,0.8],"ts":"1m"}}"#)
            .unwrap();
        assert_eq!(g.id, "j1");
        assert_eq!(g.family, BaselineFamily::Retrieval);
        assert_eq!(g.k, 12);
        assert_eq!(g.levels, vec![0.2, 0.8]);
        assert_eq!(g.timescale, Timescale::Minute);
    }

    /// Equality modulo the bookkeeping fields (`id`, `encoding`).
    fn same_genes(a: &DecoderGenome, b: &DecoderGenome) -> bool {
        let mut x = a.clone();
        let mut y = b.clone();
        x.id = String::new();
        y.id = String::new();
        x.encoding = String::new();
        y.encoding = String::new();
        x == y
    }

    #[test]
    fn to_string_round_trips() {
        let samples = [
            "forecast:h=3;lb=20;levels=0.1/0.5/0.9;kernel=8.25;scale=1.7;shrink=0.33",
            "retrieval:k=7;tau=10.5;gap=1;h=4;metric=euclid;lat=0.2;ts=1m",
            "retrieval:k=7;inst=01900000-0000-7000-8000-000000000001",
            "g1",
            "g2",
            "forecast",
        ];
        for raw in samples {
            let g = DecoderGenome::parse(raw).unwrap();
            let s = g.to_string();
            assert!(!s.contains(','), "{s}");
            let back = DecoderGenome::parse(&s).unwrap();
            assert!(same_genes(&g, &back), "{raw}: {s} → {back:?}");
            assert_eq!(back.id, s);
            assert_eq!(back.encoding, "kv");
            assert_eq!(back.to_string(), s, "canonical form is a fixed point");
        }
        // derived genomes have non-trivial floats
        let d = DecoderGenome::derive("ulid-like-id");
        assert!(same_genes(&d, &DecoderGenome::parse(&d.to_string()).unwrap()));
    }

    #[test]
    fn params_round_trip_and_clamp() {
        let g = DecoderGenome::parse("retrieval:k=7;tau=10;gap=2;h=4;lb=12").unwrap();
        let p = g.params();
        assert_eq!(p.len(), NUMERIC_GENES.len());
        assert_eq!(p["k"], 7.0);
        assert!(same_genes(&g, &g.with_params(&p).unwrap()));
        let mut m = p.clone();
        m.insert("k".into(), 0.2); // → 1
        m.insert("h".into(), 6.4); // → 6, gap raised to 6
        m.insert("lat".into(), 3.0); // → 1
        m.insert("tau".into(), -5.0); // invalid → unchanged
        m.insert("bogus".into(), 1.0); // ignored
        let c = g.with_params(&m).unwrap();
        assert_eq!((c.k, c.horizon, c.min_gap, c.latent_weight, c.tau_days), (1, 6, 6, 1.0, 10.0));
        assert_eq!(c.id, c.to_string());
        assert_eq!(c.family, g.family);
        assert!(matches!(DecoderGenome::parse("retrieval:lat=2"), Err(GenomeError::Gene { .. })));
    }

    #[test]
    fn bare_ids_derive_deterministically() {
        let a = DecoderGenome::parse("g1").unwrap();
        let b = DecoderGenome::parse("g1").unwrap();
        assert_eq!(a, b);
        assert_eq!(a.encoding, "derived");
        assert_ne!(DecoderGenome::parse("g1").unwrap().horizon, 0);
        // Different ids spread over both families.
        let fams: std::collections::HashSet<_> =
            (0..40).map(|i| DecoderGenome::derive(&format!("g{i}")).family).collect();
        assert_eq!(fams.len(), 2);
        assert_eq!(stable_hash("g1"), stable_hash("g1"));
        // Neighbouring ids must not collapse onto the same genes.
        let distinct: std::collections::HashSet<(usize, usize, u64)> = (1..=8)
            .map(|i| DecoderGenome::derive(&format!("g{i}")))
            .map(|g| (g.horizon, g.lookback, g.kernel.to_bits()))
            .collect();
        assert!(distinct.len() >= 6, "derived genomes collide: {distinct:?}");
    }
}
