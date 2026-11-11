//! Expert encoder families and per-timescale sizing (spec 31; D-004, D-010).
//!
//! Families compete; each declares its latent type, region-discovery method and prior.
//! The four neural families are implemented on candle in [`neural`] (CPU by default, `cuda`
//! feature for a GPU; D-070, D-073). [`baseline`] holds the deterministic statistical encoder
//! ([`StatisticalEncoder`]); every family implements the same [`EncoderFamily`] contract.

pub mod baseline;
pub mod media;
pub mod neural;
pub mod snapshot;

pub use baseline::{EXPERT_DIM, EXPERT_NAMES, FitConfig, MARKET_ROW_LEN, MarketBar, StatisticalEncoder};
use fa_core::Timescale;
use serde::{Deserialize, Serialize};
pub use snapshot::{
    AdapterGuard, DataRef, InstrumentDataRef, MANIFEST_FILE, NormAccumulator, NormStats, ProbeSetRef, REGION_MAP_FILE,
    REGIONS_FILE, STATISTICAL_FAMILY, SnapshotManifest, WEIGHTS_FILE,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FamilyId {
    /// FSQ/VQ discrete codes + continuous residual.
    FsqHybrid,
    /// Hierarchical VAE with flow-matching prior.
    HierVaeFlow,
    /// JEPA backbone + VAE head.
    JepaVae,
    /// Autoencoder + latent diffusion (day/week only, D-010).
    LatentDiffusion,
    /// Deterministic statistical features, no learned parameters ([`baseline`]).
    StatBaseline,
}

impl FamilyId {
    /// The four neural families of spec 31 plus the statistical baseline.
    pub const ALL: [FamilyId; 5] =
        [Self::FsqHybrid, Self::HierVaeFlow, Self::JepaVae, Self::LatentDiffusion, Self::StatBaseline];
    /// The learned families (candle, [`neural`]).
    pub const NEURAL: [FamilyId; 4] = [Self::FsqHybrid, Self::HierVaeFlow, Self::JepaVae, Self::LatentDiffusion];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::FsqHybrid => "fsq_hybrid",
            Self::HierVaeFlow => "hier_vae_flow",
            Self::JepaVae => "jepa_vae",
            Self::LatentDiffusion => "latent_diffusion",
            Self::StatBaseline => "stat_baseline",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatentKind {
    Discrete,
    Continuous,
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorKind {
    Codebook,
    FlowMatching,
    Gaussian,
    Diffusion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Market,
    Text,
    Image,
    Video,
    Audio,
}

/// Static description of a family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FamilyDescriptor {
    pub id: FamilyId,
    pub latent: LatentKind,
    pub prior: PriorKind,
    /// Region discovery method name (matches `fa_patterns::DiscoveryMethod`).
    pub region_discovery: &'static str,
    pub timescales: Vec<Timescale>,
}

pub fn descriptors() -> Vec<FamilyDescriptor> {
    use Timescale::*;
    vec![
        FamilyDescriptor {
            id: FamilyId::FsqHybrid,
            latent: LatentKind::Hybrid,
            prior: PriorKind::Codebook,
            region_discovery: "codebook",
            timescales: Timescale::LADDER.to_vec(),
        },
        FamilyDescriptor {
            id: FamilyId::HierVaeFlow,
            latent: LatentKind::Continuous,
            prior: PriorKind::FlowMatching,
            region_discovery: "density",
            timescales: Timescale::LADDER.to_vec(),
        },
        FamilyDescriptor {
            id: FamilyId::JepaVae,
            latent: LatentKind::Continuous,
            prior: PriorKind::Gaussian,
            region_discovery: "mapper",
            timescales: Timescale::LADDER.to_vec(),
        },
        FamilyDescriptor {
            id: FamilyId::LatentDiffusion,
            latent: LatentKind::Continuous,
            prior: PriorKind::Diffusion,
            region_discovery: "density",
            timescales: vec![Day, Week],
        },
        FamilyDescriptor {
            id: FamilyId::StatBaseline,
            latent: LatentKind::Continuous,
            prior: PriorKind::Gaussian,
            region_discovery: "leader",
            timescales: Timescale::LADDER.to_vec(),
        },
    ]
}

/// Per-timescale model sizing (placeholders; spec 31 tunables).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ScaleSizing {
    pub timescale: Timescale,
    /// Context length in steps of the timescale.
    pub context_steps: u32,
    pub d_model: u32,
    pub layers: u32,
    pub max_experts: u32,
    pub latent_dim: u32,
}

pub fn default_sizing(ts: Timescale) -> ScaleSizing {
    let (context_steps, d_model, layers) = match ts {
        Timescale::Second => (600, 128, 4),
        Timescale::Minute => (480, 192, 6),
        Timescale::Hour => (336, 256, 8),
        Timescale::Day => (512, 256, 8),
        Timescale::Week => (260, 192, 6),
    };
    ScaleSizing { timescale: ts, context_steps, d_model, layers, max_experts: 16, latent_dim: 32 }
}

/// An input window for one encoding step (modality embeddings are produced by frozen
/// pretrained towers, D-010).
#[derive(Debug, Clone, Default)]
pub struct EncoderInput {
    pub as_of_ms: i64,
    /// `[step][feature]` market features.
    pub market: Vec<Vec<f32>>,
    /// Pre-computed text/image/video/audio embeddings in the window.
    pub media: Vec<(Modality, Vec<f32>)>,
}

#[derive(Debug, thiserror::Error)]
pub enum EncoderError {
    #[error("ML backend not enabled (build with feature `burn`/`candle`)")]
    BackendDisabled,
    #[error("input shape mismatch: {0}")]
    Shape(String),
    #[error("snapshot manifest: {0}")]
    Manifest(String),
}

/// A trained encoder of one family at one timescale.
pub trait EncoderFamily: Send + Sync {
    fn descriptor(&self) -> &FamilyDescriptor;
    fn sizing(&self) -> ScaleSizing;
    fn snapshot_id(&self) -> &str;
    /// Encodes one window into per-expert latents. `router_weights` and region fields are
    /// left for the router and region stages (spec 32 §6.6).
    fn encode(&self, input: &EncoderInput) -> Result<fa_expert_space::ExpertView, EncoderError>;

    /// [`Self::encode`] plus what only learned encoders know (spec 32 §4.2 `ExpertView`): each
    /// expert's own next-step prediction, FSQ codes, the prior novelty score, and the realised
    /// (normalised) next-step targets of the previous window for live health. Default: none.
    fn encode_with_extras(
        &self,
        input: &EncoderInput,
    ) -> Result<(fa_expert_space::ExpertView, Option<ExpertExtras>), EncoderError> {
        Ok((self.encode(input)?, None))
    }
}

/// Per-window extras of a learned encoder ([`EncoderFamily::encode_with_extras`]).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ExpertExtras {
    /// Names of the next-step targets (`next_ret`, `next_abs_ret`, `next_range`).
    pub targets: Vec<String>,
    /// `[expert][target]` next-step prediction on the normalised targets.
    pub next_step: Vec<Vec<f32>>,
    /// FSQ code per expert (`fsq_hybrid`).
    pub codes: Option<Vec<u32>>,
    /// Novelty under the family prior (families 2 and 4).
    pub prior_score: Option<f64>,
    /// Normalised targets realised by the window's last bar: the outcome of the previous
    /// window's `next_step` (live residuals, spec 32 §8).
    pub realised_prev: Option<Vec<f32>>,
    /// Full router probabilities (before top-k).
    pub probs: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_family_has_a_descriptor_and_diffusion_is_restricted() {
        let d = descriptors();
        assert_eq!(d.len(), FamilyId::ALL.len());
        for f in FamilyId::ALL {
            assert!(d.iter().any(|x| x.id == f), "{f:?}");
        }
        let diff = d.iter().find(|f| f.id == FamilyId::LatentDiffusion).unwrap();
        assert_eq!(diff.timescales, vec![Timescale::Day, Timescale::Week]);
    }

    #[test]
    fn sizing_covers_ladder() {
        for ts in Timescale::LADDER {
            assert_eq!(default_sizing(ts).timescale, ts);
        }
    }
}
