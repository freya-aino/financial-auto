//! Snapshot artifacts (spec 36 §5, spec 31 §4.3): the manifest every trained encoder ships
//! with, and the fitted normalisation statistics that stand in for the weights of the
//! statistical family.
//!
//! A snapshot folder in Garage (`fa-models/snapshots/<snapshot_id>/`, keys from `fa_blob`)
//! holds [`MANIFEST_FILE`] (this manifest as JSON), [`WEIGHTS_FILE`] (family-specific bytes;
//! [`NormStats::to_bytes`] for the statistical family), [`REGIONS_FILE`] (the region set fitted on
//! the probe set) and [`REGION_MAP_FILE`] (the alignment to the parent, `fa_patterns::RegionMap`).
//! The Helix `Snapshot` node mirrors `snapshot_id`, `timescale`, `encoder_family`, `generation`,
//! `status` and the Garage key of the manifest.

use std::collections::BTreeMap;

use fa_core::Timescale;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{EXPERT_DIM, EXPERT_NAMES, EncoderError};

pub const MANIFEST_FILE: &str = "manifest.json";
pub const WEIGHTS_FILE: &str = "weights.bin";
pub const REGIONS_FILE: &str = "regions.json";
pub const REGION_MAP_FILE: &str = "region_map.json";

/// `encoder_family` of the statistical baseline in manifests and Helix (`Snapshot.encoder_family`).
pub const STATISTICAL_FAMILY: &str = "statistical";

/// The fixed probe set a snapshot was aligned with (spec 30 §6.4): the window ends re-encoded
/// under the parent and the new snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ProbeSetRef {
    /// `probe-<timescale>-<n>-<hash>`; the same windows give the same version.
    pub version: String,
    pub instrument: String,
    pub n_windows: usize,
    /// Unix ms of the first and last probe window end.
    pub start_ms: i64,
    pub end_ms: i64,
    /// FNV-1a over the window ends, hex.
    pub hash: String,
}

/// Where the training data came from (spec 36 §6.6 reproducibility).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DataRef {
    /// e.g. `victoriametrics:fa_trade_price`, `victoriametrics:fa_bar`, or `mixed` for a
    /// multi-instrument fit whose instruments differ (see `instruments`).
    pub source: String,
    /// The (first) instrument; `n` instruments → `multi:<n>`.
    pub instrument: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub n_bars: usize,
    /// Per-instrument source and size of a multi-instrument fit (empty for legacy manifests).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub instruments: Vec<InstrumentDataRef>,
}

/// One instrument of a fit: which series fed it and how many bars.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct InstrumentDataRef {
    /// VictoriaMetrics instrument token (dashless UUID).
    pub instrument: String,
    /// `bars` (`fa_bar_*` OHLCV + vwap) or `trades` (price-only bars built from `fa_trade_price`).
    pub source: String,
    pub n_bars: usize,
    pub start_ms: i64,
    pub end_ms: i64,
}

/// Online-adapter drift guard of a snapshot (spec 36 §6.5), stored in `manifest.params.adapter_guard`
/// so adapter jobs of this snapshot freeze at the thresholds the generation chose.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterGuard {
    /// Mean |Δmean|/σ over features beyond which an adapter is frozen.
    pub max_drift: f64,
    /// History the adapter re-fits on.
    pub window_hours: f64,
    /// `adapter_checkpoint_interval`: how often an `AdapterState` version is expected.
    pub checkpoint_interval_secs: u64,
}

impl Default for AdapterGuard {
    fn default() -> Self {
        Self { max_drift: 1.0, window_hours: 1.0, checkpoint_interval_secs: 900 }
    }
}

impl AdapterGuard {
    pub const PARAM: &'static str = "adapter_guard";

    /// Reads `params.adapter_guard`, `None` when absent or malformed.
    pub fn from_params(params: &Value) -> Option<Self> {
        serde_json::from_value(params.get(Self::PARAM)?.clone()).ok()
    }

    /// Writes `params.adapter_guard` (params becomes an object when it was not one).
    pub fn write_params(&self, params: &mut Value) {
        if !params.is_object() {
            *params = Value::Object(Default::default());
        }
        if let (Some(obj), Ok(v)) = (params.as_object_mut(), serde_json::to_value(self)) {
            obj.insert(Self::PARAM.into(), v);
        }
    }
}

/// Everything needed to rebuild and audit one snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotManifest {
    pub snapshot_id: String,
    pub encoder_family: String,
    pub timescale: Timescale,
    pub generation: u32,
    #[serde(default)]
    pub parent_snapshot_id: Option<String>,
    /// RFC 3339.
    pub created_at: String,
    /// Family-specific hyperparameters and fit settings.
    #[serde(default)]
    pub params: Value,
    #[serde(default)]
    pub probe_set: ProbeSetRef,
    #[serde(default)]
    pub data: DataRef,
    /// Fit metrics (`n_windows`, per-expert latent spread, regions, carried mass, ...).
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    /// Decoder-feedback summary consumed by a `generation` job (spec 34 §6.6), when any.
    #[serde(default)]
    pub feedback: Option<Value>,
    /// Seed of the fit.
    #[serde(default)]
    pub seed: u64,
}

impl SnapshotManifest {
    /// The adapter drift guard of this snapshot (`params.adapter_guard`), default when unset.
    pub fn adapter_guard(&self) -> AdapterGuard {
        AdapterGuard::from_params(&self.params).unwrap_or_default()
    }

    /// `metrics[name]`.
    pub fn metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }

    pub fn to_json(&self) -> Result<Vec<u8>, EncoderError> {
        serde_json::to_vec_pretty(self).map_err(|e| EncoderError::Manifest(e.to_string()))
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, EncoderError> {
        serde_json::from_slice(bytes).map_err(|e| EncoderError::Manifest(e.to_string()))
    }
}

/// Fitted per-feature normalisation statistics of the statistical family: mean and standard
/// deviation of every (expert, dim) feature over the training windows. They are the
/// "weights" of this family: an encoder built from them maps `gain_e · (raw − mean) / std`
/// through the bounded squash, so a snapshot is reproducible from its artifact alone.
///
/// `gain` (one factor per expert, empty = 1 everywhere) is the only free hyper-parameter the
/// family exposes to decoder feedback: the `generation` search of `bins/trainer` chooses a
/// squash scale `s` and per-expert feature weights `w_e` and bakes `gain_e = w_e / s` in here,
/// so `StatisticalEncoder::from_manifest` reproduces the chosen encoder without any code path
/// knowing about the search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormStats {
    pub n_windows: u64,
    /// `[expert][dim]`
    pub mean: Vec<Vec<f32>>,
    /// `[expert][dim]`, population standard deviation.
    pub std: Vec<Vec<f32>>,
    /// `[expert]` multiplier applied to the z-score before the squash; empty = 1.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gain: Vec<f32>,
}

const MAGIC: &[u8; 8] = b"FASTAT01";
/// Version 2 appends one `f32` gain per expert after the stds.
const MAGIC_V2: &[u8; 8] = b"FASTAT02";

impl NormStats {
    pub fn n_experts(&self) -> usize {
        self.mean.len()
    }

    pub fn dim(&self) -> usize {
        self.mean.first().map_or(0, Vec::len)
    }

    /// Gain of `expert` (1 when unset).
    pub fn gain_of(&self, expert: usize) -> f32 {
        self.gain.get(expert).copied().unwrap_or(1.0)
    }

    /// The same statistics with per-expert gains (`[]` clears them).
    pub fn with_gain(mut self, gain: Vec<f32>) -> Self {
        self.gain = if gain.iter().all(|g| (*g - 1.0).abs() < 1e-9) { Vec::new() } else { gain };
        self
    }

    /// Merges Welford-style statistics of several instruments into one. `equal_weight`
    /// gives every part the same effective weight (per-instrument normalisation: a long
    /// history does not dominate); otherwise parts are pooled by their window counts.
    pub fn pool(parts: &[NormStats], equal_weight: bool) -> Option<NormStats> {
        let first = parts.iter().find(|p| p.n_windows > 0)?;
        let (e, d) = (first.n_experts(), first.dim());
        let weights: Vec<f64> = parts
            .iter()
            .map(|p| {
                if p.n_windows == 0 {
                    0.0
                } else if equal_weight {
                    1.0
                } else {
                    p.n_windows as f64
                }
            })
            .collect();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return None;
        }
        let mut mean = vec![vec![0.0f64; d]; e];
        let mut m2 = vec![vec![0.0f64; d]; e];
        let at = |rows: &[Vec<f32>], k: usize, j: usize| {
            f64::from(rows.get(k).and_then(|r| r.get(j)).copied().unwrap_or(0.0))
        };
        for (p, w) in parts.iter().zip(&weights).filter(|(_, w)| **w > 0.0) {
            for (k, row) in mean.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    *cell += w / total * at(&p.mean, k, j);
                }
            }
        }
        for (p, w) in parts.iter().zip(&weights).filter(|(_, w)| **w > 0.0) {
            for (k, row) in m2.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    let (pm, ps) = (at(&p.mean, k, j), at(&p.std, k, j));
                    *cell += w / total * (ps * ps + (pm - mean[k][j]).powi(2));
                }
            }
        }
        Some(NormStats {
            n_windows: parts.iter().map(|p| p.n_windows).sum(),
            mean: mean.iter().map(|r| r.iter().map(|m| *m as f32).collect()).collect(),
            std: m2.iter().map(|r| r.iter().map(|v| v.sqrt() as f32).collect()).collect(),
            gain: first.gain.clone(),
        })
    }

    /// The same encoder without a `gain` vector: every gain is folded into the std of its
    /// expert (`g · (x − μ) / σ = (x − μ) / (σ / g)`), so the bytes stay `FASTAT01` and any
    /// runtime built before gains existed reproduces the encoder. Features with `σ ≤ 1e-6`
    /// (passed through raw by [`Self::apply`]) keep their σ, so only their gain is lost.
    pub fn folded(&self) -> Self {
        if self.gain.is_empty() {
            return self.clone();
        }
        let std = self
            .std
            .iter()
            .enumerate()
            .map(|(k, row)| {
                let g = self.gain_of(k);
                row.iter().map(|s| if *s > 1e-6 && g > 0.0 { s / g } else { *s }).collect()
            })
            .collect();
        Self { n_windows: self.n_windows, mean: self.mean.clone(), std, gain: Vec::new() }
    }

    /// Compact little-endian `FASTAT01` encoding: magic, `u32` experts, `u32` dim, `u64` windows,
    /// means, stds — with any gains [folded](Self::folded) into the stds, so every reader of
    /// the format reproduces the encoder. ([`Self::from_bytes`] also accepts `FASTAT02`, which
    /// appends one gain per expert, for forward compatibility; nothing writes it.)
    pub fn to_bytes(&self) -> Vec<u8> {
        let f = self.folded();
        let (e, d) = (f.n_experts(), f.dim());
        let mut out = Vec::with_capacity(24 + 8 * e * d);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(e as u32).to_le_bytes());
        out.extend_from_slice(&(d as u32).to_le_bytes());
        out.extend_from_slice(&f.n_windows.to_le_bytes());
        for row in f.mean.iter().chain(&f.std) {
            for x in row {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, EncoderError> {
        let bad = |m: &str| EncoderError::Manifest(format!("weights.bin: {m}"));
        if b.len() < 24 || (&b[..8] != MAGIC && &b[..8] != MAGIC_V2) {
            return Err(bad("missing FASTAT01/FASTAT02 header"));
        }
        let v2 = &b[..8] == MAGIC_V2;
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().expect("4 bytes")) as usize;
        let (e, d) = (u32_at(8), u32_at(12));
        let n_windows = u64::from_le_bytes(b[16..24].try_into().expect("8 bytes"));
        let need = 24 + 8 * e * d + if v2 { 4 * e } else { 0 };
        if b.len() != need {
            return Err(bad(&format!("expected {need} bytes for {e}x{d}, got {}", b.len())));
        }
        let mut floats = b[24..].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c));
        let mut take = || -> Vec<Vec<f32>> { (0..e).map(|_| floats.by_ref().take(d).collect()).collect() };
        let mean = take();
        let std = take();
        let gain = if v2 { floats.by_ref().take(e).collect() } else { Vec::new() };
        Ok(Self { n_windows, mean, std, gain })
    }

    /// Normalised feature: `gain_e ·` z-score when the feature varied in training, else the
    /// raw value (times the gain).
    pub fn apply(&self, expert: usize, dim: usize, raw: f64) -> f64 {
        let g = f64::from(self.gain_of(expert));
        match (self.mean.get(expert).and_then(|r| r.get(dim)), self.std.get(expert).and_then(|r| r.get(dim))) {
            (Some(m), Some(s)) if *s > 1e-6 => g * (raw - f64::from(*m)) / f64::from(*s),
            _ => g * raw,
        }
    }
}

/// Online (Welford) accumulator for [`NormStats`]; serialisable so a preempted fit resumes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormAccumulator {
    pub n: u64,
    pub mean: Vec<Vec<f64>>,
    pub m2: Vec<Vec<f64>>,
}

impl Default for NormAccumulator {
    fn default() -> Self {
        Self::new(EXPERT_NAMES.len(), EXPERT_DIM)
    }
}

impl NormAccumulator {
    pub fn new(n_experts: usize, dim: usize) -> Self {
        Self { n: 0, mean: vec![vec![0.0; dim]; n_experts], m2: vec![vec![0.0; dim]; n_experts] }
    }

    /// Adds one window's raw features (`[expert][dim]`).
    pub fn push(&mut self, features: &[Vec<f32>]) {
        self.n += 1;
        let n = self.n as f64;
        for (e, row) in features.iter().enumerate() {
            for (d, x) in row.iter().enumerate() {
                let x = f64::from(*x);
                let delta = x - self.mean[e][d];
                self.mean[e][d] += delta / n;
                self.m2[e][d] += delta * (x - self.mean[e][d]);
            }
        }
    }

    pub fn finish(&self) -> NormStats {
        let n = self.n.max(1) as f64;
        NormStats {
            n_windows: self.n,
            mean: self.mean.iter().map(|r| r.iter().map(|m| *m as f32).collect()).collect(),
            std: self.m2.iter().map(|r| r.iter().map(|v| (v / n).sqrt() as f32).collect()).collect(),
            gain: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_stats_roundtrip_bytes_and_welford() {
        let mut acc = NormAccumulator::new(2, 3);
        acc.push(&[vec![1.0, 2.0, 3.0], vec![0.0, 0.0, 1.0]]);
        acc.push(&[vec![3.0, 2.0, 5.0], vec![0.0, 0.0, 3.0]]);
        let s = acc.finish();
        assert_eq!(s.n_windows, 2);
        assert_eq!(s.mean[0], vec![2.0, 2.0, 4.0]);
        assert_eq!(s.std[0], vec![1.0, 0.0, 1.0]);
        let b = s.to_bytes();
        assert_eq!(b.len(), 24 + 8 * 6);
        assert_eq!(NormStats::from_bytes(&b).unwrap(), s);
        assert!(NormStats::from_bytes(&b[..20]).is_err());
        assert!(NormStats::from_bytes(&[b.clone(), vec![0]].concat()).is_err());
        // constant features pass through, varying ones are z-scored
        assert_eq!(s.apply(0, 1, 7.0), 7.0);
        assert_eq!(s.apply(0, 0, 3.0), 1.0);
        assert_eq!(s.apply(5, 0, 0.5), 0.5);
        // gains scale the z-score and are folded into the stds on write (still FASTAT01)
        let g = s.clone().with_gain(vec![2.0, 0.5]);
        assert_eq!(g.apply(0, 0, 3.0), 2.0);
        assert_eq!(g.apply(0, 1, 7.0), 14.0, "constant feature: raw × gain in memory");
        let b2 = g.to_bytes();
        assert_eq!(&b2[..8], b"FASTAT01");
        assert_eq!(b2.len(), 24 + 8 * 6);
        let back = NormStats::from_bytes(&b2).unwrap();
        assert_eq!(back, g.folded());
        assert!(back.gain.is_empty());
        for (k, j, x) in [(0, 0, 3.0), (0, 2, -1.5), (1, 2, 3.0)] {
            assert!((back.apply(k, j, x) - g.apply(k, j, x)).abs() < 1e-6, "folded encoder is the same at ({k},{j})");
        }
        assert_eq!(back.apply(1, 2, 3.0), 0.5, "expert 1 gain 0.5: z = (3-2)/1");
        // FASTAT02 (gains appended) is still readable
        let mut v2 = b"FASTAT02".to_vec();
        v2.extend_from_slice(&s.to_bytes()[8..]);
        v2.extend_from_slice(&2.0f32.to_le_bytes());
        v2.extend_from_slice(&0.5f32.to_le_bytes());
        assert_eq!(NormStats::from_bytes(&v2).unwrap(), g);
        assert!(s.clone().with_gain(vec![1.0, 1.0]).gain.is_empty(), "unit gains are not stored");
        assert_eq!(NormStats::from_bytes(&s.to_bytes()).unwrap().gain, Vec::<f32>::new());
    }

    #[test]
    fn pooling_by_count_and_equal_weight() {
        let a = NormStats { n_windows: 3, mean: vec![vec![0.0]], std: vec![vec![1.0]], gain: vec![] };
        let b = NormStats { n_windows: 1, mean: vec![vec![4.0]], std: vec![vec![0.0]], gain: vec![] };
        let pooled = NormStats::pool(&[a.clone(), b.clone()], false).unwrap();
        assert_eq!(pooled.n_windows, 4);
        assert!((pooled.mean[0][0] - 1.0).abs() < 1e-6, "count-weighted mean (0·3 + 4·1)/4");
        // var = Σ w (σ² + (μ−m)²) = 0.75·(1 + 1) + 0.25·(0 + 9) = 3.75
        assert!((pooled.std[0][0] - 3.75f32.sqrt()).abs() < 1e-5);
        let equal = NormStats::pool(&[a, b], true).unwrap();
        assert!((equal.mean[0][0] - 2.0).abs() < 1e-6, "equal weight: (0 + 4)/2");
        // var = 0.5·(1 + 4) + 0.5·(0 + 4) = 4.5
        assert!((equal.std[0][0] - 4.5f32.sqrt()).abs() < 1e-5);
        assert!(NormStats::pool(&[], true).is_none());
        let empty = NormStats { n_windows: 0, mean: vec![], std: vec![], gain: vec![] };
        assert!(NormStats::pool(&[empty], false).is_none());
    }

    #[test]
    fn adapter_guard_and_instrument_refs_in_params() {
        let mut params = serde_json::json!({ "context_steps": 64 });
        assert!(AdapterGuard::from_params(&params).is_none());
        AdapterGuard { max_drift: 0.4, window_hours: 2.0, checkpoint_interval_secs: 600 }.write_params(&mut params);
        assert_eq!(params["adapter_guard"]["max_drift"], 0.4);
        assert_eq!(params["context_steps"], 64);
        let m = SnapshotManifest {
            snapshot_id: "s".into(),
            encoder_family: STATISTICAL_FAMILY.into(),
            timescale: Timescale::Minute,
            generation: 1,
            parent_snapshot_id: None,
            created_at: String::new(),
            params,
            probe_set: Default::default(),
            data: DataRef {
                source: "victoriametrics:fa_bar".into(),
                instrument: "multi:2".into(),
                instruments: vec![InstrumentDataRef {
                    instrument: "a".into(),
                    source: "bars".into(),
                    n_bars: 3,
                    ..Default::default()
                }],
                ..Default::default()
            },
            metrics: BTreeMap::from([("carried_mass".to_owned(), 0.7)]),
            feedback: None,
            seed: 0,
        };
        assert_eq!(m.adapter_guard().max_drift, 0.4);
        assert_eq!(m.metric("carried_mass"), Some(0.7));
        let back = SnapshotManifest::from_json(&m.to_json().unwrap()).unwrap();
        assert_eq!(back, m);
        // legacy manifests without `instruments` still parse
        let legacy: DataRef =
            serde_json::from_str(r#"{"source":"x","instrument":"i","start_ms":0,"end_ms":1,"n_bars":2}"#).unwrap();
        assert!(legacy.instruments.is_empty());
    }

    #[test]
    fn manifest_json_roundtrip() {
        let m = SnapshotManifest {
            snapshot_id: "stat-v1-1m-g0-1".into(),
            encoder_family: STATISTICAL_FAMILY.into(),
            timescale: Timescale::Minute,
            generation: 0,
            parent_snapshot_id: Some("stat-v1-1m".into()),
            created_at: "2026-11-10T00:00:00Z".into(),
            params: serde_json::json!({"context_steps": 480}),
            probe_set: ProbeSetRef { version: "probe-1m-3-ab".into(), n_windows: 3, ..Default::default() },
            data: DataRef::default(),
            metrics: BTreeMap::from([("n_windows".to_owned(), 3.0)]),
            feedback: None,
            seed: 7,
        };
        let back = SnapshotManifest::from_json(&m.to_json().unwrap()).unwrap();
        assert_eq!(back, m);
        assert!(SnapshotManifest::from_json(b"{").is_err());
    }
}
