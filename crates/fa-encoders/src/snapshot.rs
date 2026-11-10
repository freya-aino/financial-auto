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
    /// e.g. `victoriametrics:fa_trade_price`
    pub source: String,
    pub instrument: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub n_bars: usize,
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
    pub fn to_json(&self) -> Result<Vec<u8>, EncoderError> {
        serde_json::to_vec_pretty(self).map_err(|e| EncoderError::Manifest(e.to_string()))
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, EncoderError> {
        serde_json::from_slice(bytes).map_err(|e| EncoderError::Manifest(e.to_string()))
    }
}

/// Fitted per-feature normalisation statistics of the statistical family: mean and standard
/// deviation of every (expert, dim) feature over the training windows. They are the
/// "weights" of this family: an encoder built from them maps `(raw − mean) / std` through the
/// bounded squash, so a snapshot is reproducible from its artifact alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormStats {
    pub n_windows: u64,
    /// `[expert][dim]`
    pub mean: Vec<Vec<f32>>,
    /// `[expert][dim]`, population standard deviation.
    pub std: Vec<Vec<f32>>,
}

const MAGIC: &[u8; 8] = b"FASTAT01";

impl NormStats {
    pub fn n_experts(&self) -> usize {
        self.mean.len()
    }

    pub fn dim(&self) -> usize {
        self.mean.first().map_or(0, Vec::len)
    }

    /// Compact little-endian encoding: magic, `u32` experts, `u32` dim, `u64` windows, means, stds.
    pub fn to_bytes(&self) -> Vec<u8> {
        let (e, d) = (self.n_experts(), self.dim());
        let mut out = Vec::with_capacity(24 + 8 * e * d);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(e as u32).to_le_bytes());
        out.extend_from_slice(&(d as u32).to_le_bytes());
        out.extend_from_slice(&self.n_windows.to_le_bytes());
        for row in self.mean.iter().chain(&self.std) {
            for x in row {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, EncoderError> {
        let bad = |m: &str| EncoderError::Manifest(format!("weights.bin: {m}"));
        if b.len() < 24 || &b[..8] != MAGIC {
            return Err(bad("missing FASTAT01 header"));
        }
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().expect("4 bytes")) as usize;
        let (e, d) = (u32_at(8), u32_at(12));
        let n_windows = u64::from_le_bytes(b[16..24].try_into().expect("8 bytes"));
        let need = 24 + 8 * e * d;
        if b.len() != need {
            return Err(bad(&format!("expected {need} bytes for {e}x{d}, got {}", b.len())));
        }
        let mut floats = b[24..].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c));
        let mut take = || -> Vec<Vec<f32>> { (0..e).map(|_| floats.by_ref().take(d).collect()).collect() };
        let mean = take();
        let std = take();
        Ok(Self { n_windows, mean, std })
    }

    /// Normalised feature: z-score when the feature varied in training, else the raw value.
    pub fn apply(&self, expert: usize, dim: usize, raw: f64) -> f64 {
        match (self.mean.get(expert).and_then(|r| r.get(dim)), self.std.get(expert).and_then(|r| r.get(dim))) {
            (Some(m), Some(s)) if *s > 1e-6 => (raw - f64::from(*m)) / f64::from(*s),
            _ => raw,
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
