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

/// Genome family the genome instantiates: the two CPU baselines and the six trainable neural
/// families of [`crate::neural`] (the name predates the neural families).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BaselineFamily {
    /// Empirical quantile forecaster; occupies the `rnn_prior` output family slot (the
    /// family whose output contract is forecast quantiles).
    Forecast,
    /// Analog retrieval with time distance as an explicit feature.
    Retrieval,
    /// Neural quantile head (MLP or GRU prior over the return window, latent → initial state),
    /// pinball loss; output family `rnn_prior`.
    NnForecast,
    /// Learned retrieval metric: a projector trained with a soft-kNN CRPS loss over
    /// time-decayed analogs; output family `retrieval`.
    NnRetrieval,
    /// Chart reconstruction: latent (+ return window) → low-resolution raster of the lookback
    /// window ([`crate::render`]), per-column cross-entropy; output family `visual.chart`.
    NnChart,
    /// Future chart: → raster of the next horizon, whose last column is a predictive
    /// distribution of the horizon return; output family `visual.future`.
    NnFuture,
    /// News reconstruction: latent (+ return window) → hashed embedding of the trailing news
    /// window, cosine loss; output family `text.news`.
    NnNews,
    /// Forecast narrative: latent + return window + current news embedding → embedding of the
    /// news published in the next text window; output family `text.narrative`.
    NnNarrative,
}

impl BaselineFamily {
    pub const ALL: [Self; 8] = [
        Self::Forecast,
        Self::Retrieval,
        Self::NnForecast,
        Self::NnRetrieval,
        Self::NnChart,
        Self::NnFuture,
        Self::NnNews,
        Self::NnNarrative,
    ];
    pub const NEURAL: [Self; 6] =
        [Self::NnForecast, Self::NnRetrieval, Self::NnChart, Self::NnFuture, Self::NnNews, Self::NnNarrative];

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "forecast" | "quantile" | "rnn_prior" => Some(Self::Forecast),
            "retrieval" | "analog" => Some(Self::Retrieval),
            "nn_forecast" | "nn_quantile" | "nn_rnn" | "neural_forecast" => Some(Self::NnForecast),
            "nn_retrieval" | "nn_metric" | "neural_retrieval" => Some(Self::NnRetrieval),
            "nn_chart" | "neural_chart" => Some(Self::NnChart),
            "nn_future" | "neural_future" => Some(Self::NnFuture),
            "nn_news" | "neural_news" => Some(Self::NnNews),
            "nn_narrative" | "neural_narrative" => Some(Self::NnNarrative),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Forecast => "forecast",
            Self::Retrieval => "retrieval",
            Self::NnForecast => "nn_forecast",
            Self::NnRetrieval => "nn_retrieval",
            Self::NnChart => "nn_chart",
            Self::NnFuture => "nn_future",
            Self::NnNews => "nn_news",
            Self::NnNarrative => "nn_narrative",
        }
    }

    /// Trainable (candle) family.
    pub fn is_neural(self) -> bool {
        !matches!(self, Self::Forecast | Self::Retrieval)
    }

    /// Families whose output is a forecast of the horizon return (scored with `fa_backtest::eval`).
    pub fn forecasts(self) -> bool {
        matches!(self, Self::Forecast | Self::Retrieval | Self::NnForecast | Self::NnRetrieval | Self::NnFuture)
    }

    /// Families with a reconstruction target (`recon` / `recon_skill` objectives).
    pub fn reconstructs(self) -> bool {
        matches!(self, Self::NnChart | Self::NnFuture | Self::NnNews | Self::NnNarrative)
    }

    /// Families that read the news timeline.
    pub fn uses_text(self) -> bool {
        matches!(self, Self::NnNews | Self::NnNarrative)
    }

    /// Family string used on `decoder.output.<family>` (proto `DecoderOutput.family`).
    pub fn output_family(self) -> Family {
        match self {
            Self::Forecast | Self::NnForecast => Family::RnnPrior,
            Self::Retrieval | Self::NnRetrieval => Family::Retrieval,
            Self::NnChart => Family::VisualChart,
            Self::NnFuture => Family::VisualFuture,
            Self::NnNews => Family::TextNews,
            Self::NnNarrative => Family::TextNarrative,
        }
    }
}

/// Genes of the neural families (spec 34 §4.2 `TopologyGene` / `HyperGene` / family genes),
/// all numeric so the `fa-evolution` operators can mutate them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NeuralGenes {
    /// Hidden width (`width`, 4..=256).
    pub width: usize,
    /// Hidden layers of the MLP / head (`depth`, 1..=4).
    pub depth: usize,
    /// Recurrent cell: GRU over the return window when true (`gru` ≥ 0.5), else MLP.
    pub gru: bool,
    /// SGD (AdamW) steps (`steps`, 1..=20000).
    pub steps: usize,
    /// Learning rate (`lr`).
    pub lr: f64,
    /// Weight decay (`wd`).
    pub wd: f64,
    /// Minibatch size (`bs`, 4..=1024).
    pub batch: usize,
    /// Share of the window (by time) used for training; the rest is the evaluation window
    /// (`train`, 0.2..=0.9).
    pub train_frac: f64,
    /// Retrieval: embedding dimension (`emb`).
    pub emb: usize,
    /// Retrieval: softmax temperature of the soft-kNN weights (`temp`).
    pub temp: f64,
    /// Retrieval: candidate windows kept per query at inference (`pool`).
    pub pool: usize,
    /// Visual: raster resolution (`res`, 4..=32; columns = rows = res).
    pub res: usize,
    /// Visual future: half-height of the raster in units of `σ·√h` (`band`).
    pub band: f64,
    /// Text: folded hash-embedding dimension (`tdim`, 16..=1024).
    pub tdim: usize,
    /// Text: news window in minutes (`tw`), trailing (news) or leading (narrative).
    pub tw_min: f64,
    /// Init / minibatch seed (`seed`; not mutated).
    pub seed: u64,
    /// Text: read only the news matched to the series' own instrument (`ninst` ≥ 0.5; symbol,
    /// name or entity match, `text_baseline::InstrumentIndex`) instead of the market-wide stream.
    pub news_inst: bool,
}

impl Default for NeuralGenes {
    fn default() -> Self {
        Self {
            width: 32,
            depth: 2,
            gru: false,
            steps: 300,
            lr: 3e-3,
            wd: 1e-4,
            batch: 64,
            train_frac: 0.5,
            emb: 16,
            temp: 0.1,
            pool: 2048,
            res: 12,
            band: 3.0,
            tdim: 128,
            tw_min: 60.0,
            seed: 7,
            news_inst: false,
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
    /// Neural-family genes (ignored by the baselines).
    #[serde(default)]
    pub nn: NeuralGenes,
    /// Experts the decoder reads (`mask` gene, bit `i` = expert index `i`, spec 34 §4.2
    /// `expert_mask`); `0` = every expert (the concatenated latent).
    #[serde(default)]
    pub expert_mask: u32,
    /// Cross-asset source (`src` gene, spec 34 §6.1 RNN prior): latents are read from this
    /// instrument while the series (`inst`) is the target. `None` = the series' own latents.
    #[serde(default)]
    pub source: Option<String>,
}

fn default_latent_weight() -> f64 {
    0.5
}

/// Numeric gene names exposed to the evolution operators (`params()` keys of the baselines).
pub const NUMERIC_GENES: [&str; 9] = ["h", "lb", "kernel", "scale", "shrink", "k", "tau", "gap", "lat"];
/// Additional numeric genes of the neural families (`params()` adds them for those).
pub const NEURAL_GENES: [&str; 16] = [
    "width", "depth", "gru", "steps", "lr", "wd", "bs", "train", "emb", "temp", "pool", "res", "band", "tdim", "tw",
    "ninst",
];

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
            nn: NeuralGenes::default(),
            expert_mask: 0,
            source: None,
        }
    }

    /// Numeric genes as `name → value` (see [`NUMERIC_GENES`], plus [`NEURAL_GENES`] for the
    /// neural families); integers are exact.
    pub fn params(&self) -> BTreeMap<String, f64> {
        let mut m = self.baseline_params();
        if self.family.is_neural() {
            let n = &self.nn;
            m.extend([
                ("width".to_owned(), n.width as f64),
                ("depth".to_owned(), n.depth as f64),
                ("gru".to_owned(), if n.gru { 1.0 } else { 0.0 }),
                ("steps".to_owned(), n.steps as f64),
                ("lr".to_owned(), n.lr),
                ("wd".to_owned(), n.wd),
                ("bs".to_owned(), n.batch as f64),
                ("train".to_owned(), n.train_frac),
                ("emb".to_owned(), n.emb as f64),
                ("temp".to_owned(), n.temp),
                ("pool".to_owned(), n.pool as f64),
                ("res".to_owned(), n.res as f64),
                ("band".to_owned(), n.band),
                ("tdim".to_owned(), n.tdim as f64),
                ("tw".to_owned(), n.tw_min),
                ("ninst".to_owned(), if n.news_inst { 1.0 } else { 0.0 }),
            ]);
        }
        m
    }

    fn baseline_params(&self) -> BTreeMap<String, f64> {
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
                "width" => g.nn.width = int(*v, 4).min(256),
                "depth" => g.nn.depth = int(*v, 1).min(4),
                "gru" => g.nn.gru = v.is_finite() && *v >= 0.5,
                "steps" => g.nn.steps = int(*v, 1).min(20_000),
                "lr" => g.nn.lr = if v.is_finite() && *v > 0.0 { v.clamp(1e-5, 0.1) } else { g.nn.lr },
                "wd" => g.nn.wd = if v.is_finite() { v.clamp(0.0, 0.1) } else { g.nn.wd },
                "bs" => g.nn.batch = int(*v, 4).min(1024),
                "train" => g.nn.train_frac = if v.is_finite() { v.clamp(0.2, 0.9) } else { g.nn.train_frac },
                "emb" => g.nn.emb = int(*v, 2).min(128),
                "temp" => g.nn.temp = if v.is_finite() && *v > 0.0 { v.clamp(0.01, 10.0) } else { g.nn.temp },
                "pool" => g.nn.pool = int(*v, 16).min(65_536),
                "res" => g.nn.res = int(*v, 4).min(32),
                "band" => g.nn.band = if v.is_finite() && *v > 0.0 { v.clamp(0.5, 10.0) } else { g.nn.band },
                "tdim" => g.nn.tdim = int(*v, 16).min(1024),
                "tw" => g.nn.tw_min = if v.is_finite() && *v > 0.0 { v.clamp(1.0, 10_080.0) } else { g.nn.tw_min },
                "ninst" => g.nn.news_inst = v.is_finite() && *v >= 0.5,
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
            "width" => self.nn.width = int(v)?,
            "depth" => self.nn.depth = int(v)?,
            "gru" | "cell" => {
                self.nn.gru = match v.to_ascii_lowercase().as_str() {
                    "gru" | "rnn" | "true" => true,
                    "mlp" | "false" => false,
                    _ => num(v)? >= 0.5,
                }
            }
            "steps" => self.nn.steps = int(v)?,
            "lr" => self.nn.lr = num(v)?,
            "wd" => self.nn.wd = num(v)?,
            "bs" | "batch" => self.nn.batch = int(v)?,
            "train" => self.nn.train_frac = num(v)?,
            "emb" => self.nn.emb = int(v)?,
            "temp" => self.nn.temp = num(v)?,
            "pool" => self.nn.pool = int(v)?,
            "res" => self.nn.res = int(v)?,
            "band" => self.nn.band = num(v)?,
            "tdim" => self.nn.tdim = int(v)?,
            "tw" => self.nn.tw_min = num(v)?,
            "seed" => self.nn.seed = v.parse::<u64>().map_err(|_| err("not an unsigned integer"))?,
            "ninst" | "news_inst" => {
                self.nn.news_inst = match v.to_ascii_lowercase().as_str() {
                    "inst" | "true" => true,
                    "all" | "market" | "false" => false,
                    _ => num(v)? >= 0.5,
                }
            }
            "mask" | "expert_mask" => {
                let m = match v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")) {
                    Some(hex) => u32::from_str_radix(hex, 16).map_err(|_| err("not a hex mask"))?,
                    None => v.parse::<u32>().map_err(|_| err("not an unsigned integer mask"))?,
                };
                if m >> crate::EXPERT_MASK_BITS != 0 {
                    return Err(err("mask uses bits beyond the expert range"));
                }
                self.expert_mask = m;
            }
            "src" | "source" => self.source = Some(v.to_owned()).filter(|s| !s.is_empty()),
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
        if self.family.is_neural() {
            let n = &self.nn;
            let bad = |k: &str, m: &str| Err(gene(k, m));
            if !(4..=256).contains(&n.width) {
                return bad("width", "must be in 4..=256");
            }
            if !(1..=4).contains(&n.depth) {
                return bad("depth", "must be in 1..=4");
            }
            if !(1..=20_000).contains(&n.steps) {
                return bad("steps", "must be in 1..=20000");
            }
            if !(n.lr > 0.0 && n.lr <= 0.1) {
                return bad("lr", "must be in (0, 0.1]");
            }
            if !(0.0..=0.1).contains(&n.wd) {
                return bad("wd", "must be in [0, 0.1]");
            }
            if !(4..=1024).contains(&n.batch) {
                return bad("bs", "must be in 4..=1024");
            }
            if !(0.2..=0.9).contains(&n.train_frac) {
                return bad("train", "must be in [0.2, 0.9]");
            }
            if !(2..=128).contains(&n.emb) {
                return bad("emb", "must be in 2..=128");
            }
            if !(n.temp > 0.0 && n.temp.is_finite()) {
                return bad("temp", "must be > 0");
            }
            if n.pool < 16 {
                return bad("pool", "must be ≥ 16");
            }
            if !(4..=32).contains(&n.res) {
                return bad("res", "must be in 4..=32");
            }
            if !(n.band > 0.0 && n.band.is_finite()) {
                return bad("band", "must be > 0");
            }
            if !(16..=1024).contains(&n.tdim) {
                return bad("tdim", "must be in 16..=1024");
            }
            if !(n.tw_min > 0.0 && n.tw_min.is_finite()) {
                return bad("tw", "must be > 0");
            }
        }
        // Leakage guard (spec 34 §6.1): an analog's outcome window must end before now.
        self.min_gap = self.min_gap.max(self.horizon);
        Ok(())
    }

    /// Short human-readable summary (`forecast h=5 lb=32 ...`).
    pub fn summary(&self) -> String {
        let mut s = self.summary_genes();
        if self.expert_mask != 0 {
            s.push_str(&format!(" mask={:#x}", self.expert_mask));
        }
        if let Some(src) = &self.source {
            s.push_str(&format!(" src={src}"));
        }
        s
    }

    fn summary_genes(&self) -> String {
        match self.family {
            BaselineFamily::Forecast => format!(
                "forecast ts={} h={} lb={} kernel={:.1} scale={:.2} shrink={:.2}",
                self.timescale, self.horizon, self.lookback, self.kernel, self.scale, self.shrink
            ),
            BaselineFamily::Retrieval => format!(
                "retrieval ts={} h={} lb={} k={} tau={:.1}d gap={} metric={:?}",
                self.timescale, self.horizon, self.lookback, self.k, self.tau_days, self.min_gap, self.metric
            ),
            f => format!(
                "{} ts={} h={} lb={} {}×{}{} steps={} lr={} lat={}{}",
                f.as_str(),
                self.timescale,
                self.horizon,
                self.lookback,
                self.nn.width,
                self.nn.depth,
                if self.nn.gru { " gru" } else { "" },
                self.nn.steps,
                self.nn.lr,
                self.latent_weight,
                match f {
                    BaselineFamily::NnRetrieval => format!(" k={} emb={} temp={}", self.k, self.nn.emb, self.nn.temp),
                    BaselineFamily::NnChart | BaselineFamily::NnFuture => format!(" res={}", self.nn.res),
                    BaselineFamily::NnNews | BaselineFamily::NnNarrative => {
                        format!(" tdim={} tw={}m", self.nn.tdim, self.nn.tw_min)
                    }
                    _ => String::new(),
                }
            ),
        }
    }
}

impl fmt::Display for DecoderGenome {
    /// Canonical kv form: `<family>:ts=..;h=..;lb=..;levels=a/b/c;kernel=..;scale=..;shrink=..;
    /// k=..;tau=..;gap=..;metric=..;lat=..[;<neural genes>][;ninst=1][;mask=..][;inst=..][;src=..]`.
    /// Every numeric gene is printed (also the other family's, so a child keeps them when its
    /// family gene flips) and floats print in Rust's shortest round-trip form, so
    /// `parse(to_string())` is lossless. `ninst`, `mask`, `inst` and `src` print only when set, so
    /// genome strings written before those genes existed keep their canonical form.
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
        if self.family.is_neural() {
            let n = &self.nn;
            write!(
                f,
                ";width={};depth={};gru={};steps={};lr={};wd={};bs={};train={};emb={};temp={};pool={};res={};band={};tdim={};tw={};seed={}",
                n.width,
                n.depth,
                u8::from(n.gru),
                n.steps,
                n.lr,
                n.wd,
                n.batch,
                n.train_frac,
                n.emb,
                n.temp,
                n.pool,
                n.res,
                n.band,
                n.tdim,
                n.tw_min,
                n.seed
            )?;
        }
        if self.family.is_neural() && self.nn.news_inst {
            write!(f, ";ninst=1")?;
        }
        if self.expert_mask != 0 {
            write!(f, ";mask={}", self.expert_mask)?;
        }
        if let Some(inst) = &self.instrument {
            write!(f, ";inst={inst}")?;
        }
        if let Some(src) = &self.source {
            write!(f, ";src={src}")?;
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
            "nn_news:tdim=64;ninst=1;mask=5",
            "nn_forecast:h=3;inst=01900000-0000-7000-8000-000000000001;src=01900000-0000-7000-8000-000000000002;mask=0x3",
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
    fn neural_genomes_parse_round_trip_and_clamp() {
        let g =
            DecoderGenome::parse("nn_forecast:h=4;lb=16;width=24;depth=3;gru=gru;steps=120;lr=0.002;bs=32").unwrap();
        assert_eq!(g.family, BaselineFamily::NnForecast);
        assert!(g.family.is_neural() && g.family.forecasts() && !g.family.reconstructs());
        assert_eq!(g.family.output_family(), Family::RnnPrior);
        assert_eq!((g.nn.width, g.nn.depth, g.nn.gru, g.nn.steps, g.nn.batch), (24, 3, true, 120, 32));
        for fam in BaselineFamily::NEURAL {
            let g =
                DecoderGenome::parse(&format!("{}:h=3;lb=8;res=10;tdim=64;tw=30;emb=6;seed=5", fam.as_str())).unwrap();
            let s = g.to_string();
            assert!(!s.contains(','), "{s}");
            let back = DecoderGenome::parse(&s).unwrap();
            assert!(same_genes(&g, &back), "{s}");
            assert_eq!(back.to_string(), s);
            let p = g.params();
            assert_eq!(p.len(), NUMERIC_GENES.len() + NEURAL_GENES.len());
            assert!(same_genes(&g, &g.with_params(&p).unwrap()));
        }
        let fams: std::collections::HashSet<_> = BaselineFamily::ALL.iter().map(|f| f.output_family()).collect();
        assert_eq!(fams.len(), 6, "rnn_prior, retrieval, visual.chart/future, text.news/narrative");
        let mut m = g.params();
        m.insert("width".into(), 1000.0);
        m.insert("lr".into(), 5.0);
        m.insert("gru".into(), 0.2);
        m.insert("train".into(), 0.0);
        let c = g.with_params(&m).unwrap();
        assert_eq!((c.nn.width, c.nn.lr, c.nn.gru, c.nn.train_frac), (256, 0.1, false, 0.2));
        assert!(matches!(DecoderGenome::parse("nn_chart:res=64"), Err(GenomeError::Gene { .. })));
        assert!(matches!(DecoderGenome::parse("nn_chart:mask=0x10000"), Err(GenomeError::Gene { .. })));
        let x = DecoderGenome::parse("nn_narrative:ninst=inst;mask=0x6;src=abc").unwrap();
        assert_eq!((x.nn.news_inst, x.expert_mask, x.source.as_deref()), (true, 6, Some("abc")));
        assert!(x.to_string().ends_with(";ninst=1;mask=6;src=abc"), "{x}");
        let mut m = x.params();
        m.insert("ninst".into(), 0.1);
        let y = x.with_params(&m).unwrap();
        assert!(
            !y.nn.news_inst && y.expert_mask == 6 && y.source.as_deref() == Some("abc"),
            "mask / src survive variation"
        );
        assert!(matches!(DecoderGenome::parse("nn_news:tdim=4"), Err(GenomeError::Gene { .. })));
        let j = DecoderGenome::parse(r#"{"family":"nn_retrieval","params":{"emb":12,"temp":0.3,"gru":1}}"#).unwrap();
        assert_eq!((j.nn.emb, j.nn.temp, j.nn.gru), (12, 0.3, true));
        // baseline strings are unchanged by the neural genes
        assert!(!DecoderGenome::parse("forecast").unwrap().to_string().contains("width"));
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
