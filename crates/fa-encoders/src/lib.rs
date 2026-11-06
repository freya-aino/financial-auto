//! Expert encoder families and per-timescale sizing (spec 31; D-004, D-010).
//!
//! Families compete; each declares its latent type, region-discovery method and prior.
//! Model code lands behind the `burn` / `candle` features.

use fa_core::Timescale;
use serde::{Deserialize, Serialize};

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
}

impl FamilyId {
    pub const ALL: [FamilyId; 4] = [Self::FsqHybrid, Self::HierVaeFlow, Self::JepaVae, Self::LatentDiffusion];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::FsqHybrid => "fsq_hybrid",
            Self::HierVaeFlow => "hier_vae_flow",
            Self::JepaVae => "jepa_vae",
            Self::LatentDiffusion => "latent_diffusion",
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
        FamilyDescriptor { id: FamilyId::FsqHybrid, latent: LatentKind::Hybrid, prior: PriorKind::Codebook, region_discovery: "codebook", timescales: Timescale::LADDER.to_vec() },
        FamilyDescriptor { id: FamilyId::HierVaeFlow, latent: LatentKind::Continuous, prior: PriorKind::FlowMatching, region_discovery: "density", timescales: Timescale::LADDER.to_vec() },
        FamilyDescriptor { id: FamilyId::JepaVae, latent: LatentKind::Continuous, prior: PriorKind::Gaussian, region_discovery: "mapper", timescales: Timescale::LADDER.to_vec() },
        FamilyDescriptor { id: FamilyId::LatentDiffusion, latent: LatentKind::Continuous, prior: PriorKind::Diffusion, region_discovery: "density", timescales: vec![Day, Week] },
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
}

/// A trained encoder of one family at one timescale.
pub trait EncoderFamily: Send + Sync {
    fn descriptor(&self) -> &FamilyDescriptor;
    fn sizing(&self) -> ScaleSizing;
    fn snapshot_id(&self) -> &str;
    /// Encodes one window into per-expert latents + router weights.
    fn encode(&self, input: &EncoderInput) -> Result<fa_expert_space::ExpertView, EncoderError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_families_and_diffusion_restricted() {
        let d = descriptors();
        assert_eq!(d.len(), 4);
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
