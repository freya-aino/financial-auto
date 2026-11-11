//! Media towers (spec 31 §6.3, D-010): frozen front-ends that turn text, images, video frames and
//! audio into fixed-size embeddings feeding [`crate::EncoderInput::media`].
//!
//! D-010 calls for frozen **pretrained** candle towers (Whisper-class ASR, a text embedding
//! model, a vision embedding model). Their weights are external downloads (hundreds of MB, with
//! licences to record), which this platform does not fetch on its own. The contract is
//! therefore split:
//!
//! * [`MediaTower`]: what every tower provides (modality, dimension, id, `embed`);
//! * [`PretrainedTowerSpec`]: the registry entry of a pretrained tower — its Garage key under
//!   `fa-models/pretrained/<name>/`, expected SHA-256, licence and dimension. Loading verifies
//!   the hash; a tower whose weights are absent is **unavailable**, never silently replaced;
//! * [`HashedEmbeddingTower`]: the always-available fallback (spec 31 §8 "pretrained weights
//!   unavailable → learned-from-scratch front-end"): signed feature hashing of tokens (text,
//!   ASR transcripts, captions, alt text) and of byte n-grams (raw image/audio bytes), L2
//!   normalised. It carries no semantics beyond lexical/byte overlap; it exists so the media
//!   path, masking (spec 31 §9.5) and point-in-time filtering are exercised end to end.
//! * [`select_tower`]: picks the pretrained tower when its weights are present and valid,
//!   otherwise the hashed fallback, and says which and why.
//!
//! [`point_in_time`] is the batch-builder guard of spec 31 §6.3 / §9.3: only items with
//! `publish_time ≤ window end` enter a window.

use serde::{Deserialize, Serialize};

use crate::Modality;

/// One media item before embedding.
#[derive(Debug, Clone, PartialEq)]
pub enum MediaPayload {
    /// Text, a transcript, a caption.
    Text(String),
    /// Raw bytes (image, audio, a video frame).
    Bytes(Vec<u8>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MediaItem {
    pub modality: Modality,
    pub payload: MediaPayload,
    pub event_time_ms: i64,
    pub publish_time_ms: i64,
}

/// A frozen media front-end.
pub trait MediaTower: Send + Sync {
    /// `hashed-v1-text-256`, `whisper-small@<sha8>`, …
    fn id(&self) -> &str;
    fn modality(&self) -> Modality;
    fn dim(&self) -> usize;
    fn embed(&self, item: &MediaPayload) -> Vec<f32>;
}

/// Feature-hashing fallback tower.
#[derive(Debug, Clone)]
pub struct HashedEmbeddingTower {
    modality: Modality,
    dim: usize,
    id: String,
    /// Byte n-gram length for [`MediaPayload::Bytes`].
    ngram: usize,
}

impl HashedEmbeddingTower {
    pub fn new(modality: Modality, dim: usize) -> Self {
        let dim = dim.max(8);
        let m = format!("{modality:?}").to_ascii_lowercase();
        Self { modality, dim, id: format!("hashed-v1-{m}-{dim}"), ngram: 4 }
    }
}

fn fnv(bytes: &[u8], salt: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ salt;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

impl MediaTower for HashedEmbeddingTower {
    fn id(&self) -> &str {
        &self.id
    }

    fn modality(&self) -> Modality {
        self.modality
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, item: &MediaPayload) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];
        let mut add = |tok: &[u8]| {
            let h = fnv(tok, 0x51);
            let sign = if fnv(tok, 0xA7) & 1 == 0 { 1.0 } else { -1.0 };
            v[(h % self.dim as u64) as usize] += sign;
        };
        match item {
            MediaPayload::Text(t) => {
                let lower = t.to_lowercase();
                let toks: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).filter(|s| !s.is_empty()).collect();
                for w in &toks {
                    add(w.as_bytes());
                }
                for pair in toks.windows(2) {
                    add(format!("{} {}", pair[0], pair[1]).as_bytes());
                }
            }
            MediaPayload::Bytes(b) => {
                for g in b.windows(self.ngram.min(b.len().max(1))) {
                    add(g);
                }
            }
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            v.iter_mut().for_each(|x| *x /= n);
        }
        v
    }
}

/// Registry entry of a pretrained (frozen) tower (D-010). Weights live in Garage under
/// `fa-models/pretrained/<name>/<file>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PretrainedTowerSpec {
    pub name: String,
    pub modality: Modality,
    pub dim: usize,
    pub file: String,
    /// Hex SHA-256 the weights must have.
    pub sha256: String,
    pub licence: String,
}

/// The towers D-010 names, with no weights shipped. Hashes are left empty until an operator
/// registers downloaded weights (an empty hash never validates).
pub fn pretrained_catalog() -> Vec<PretrainedTowerSpec> {
    let spec = |name: &str, modality, dim, file: &str, licence: &str| PretrainedTowerSpec {
        name: name.into(),
        modality,
        dim,
        file: file.into(),
        sha256: String::new(),
        licence: licence.into(),
    };
    vec![
        spec("whisper-small", Modality::Audio, 768, "model.safetensors", "MIT"),
        spec("bge-small-en-v1.5", Modality::Text, 384, "model.safetensors", "MIT"),
        spec("siglip-base-patch16-224", Modality::Image, 768, "model.safetensors", "Apache-2.0"),
    ]
}

/// Which tower serves a modality, and why.
pub struct TowerChoice {
    pub tower: Box<dyn MediaTower>,
    pub pretrained: bool,
    pub note: String,
}

/// Picks the pretrained tower of `modality` when `weights` (the bytes found in Garage, if any)
/// match its registered hash; otherwise the hashed fallback of `fallback_dim`. Running a
/// pretrained network is not implemented in this build (no weights are available to test it), so
/// even valid weights currently yield the fallback with a note saying so.
pub fn select_tower(
    modality: Modality,
    weights: Option<(&PretrainedTowerSpec, &[u8], &str)>,
    fallback_dim: usize,
) -> TowerChoice {
    let fallback = |note: String| TowerChoice {
        tower: Box::new(HashedEmbeddingTower::new(modality, fallback_dim)),
        pretrained: false,
        note,
    };
    match weights {
        None => fallback(format!("no pretrained {modality:?} tower weights in fa-models/pretrained/; hashed fallback")),
        Some((spec, _, sha)) if spec.sha256.is_empty() || !spec.sha256.eq_ignore_ascii_case(sha) => {
            fallback(format!("{} weights do not match the registered sha256; hashed fallback", spec.name))
        }
        Some((spec, _, _)) => {
            fallback(format!("{} weights verified but the candle runner is not built in; hashed fallback", spec.name))
        }
    }
}

/// Spec 31 §6.3: items published after the window end never enter the window.
pub fn point_in_time(items: &[MediaItem], window_end_ms: i64) -> Vec<&MediaItem> {
    items.iter().filter(|i| i.publish_time_ms <= window_end_ms).collect()
}

/// Embeds the admissible items of a window (`EncoderInput::media`).
pub fn embed_window(towers: &[&dyn MediaTower], items: &[MediaItem], window_end_ms: i64) -> Vec<(Modality, Vec<f32>)> {
    point_in_time(items, window_end_ms)
        .into_iter()
        .filter_map(|i| towers.iter().find(|t| t.modality() == i.modality).map(|t| (i.modality, t.embed(&i.payload))))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashed_tower_is_deterministic_normalised_and_lexical() {
        let t = HashedEmbeddingTower::new(Modality::Text, 64);
        let a = t.embed(&MediaPayload::Text("Fed raises rates".into()));
        let b = t.embed(&MediaPayload::Text("fed RAISES rates!".into()));
        let c = t.embed(&MediaPayload::Text("earnings beat expectations".into()));
        assert_eq!(a, b);
        assert!((a.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-5);
        let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f32>();
        assert!(dot(&a, &b) > dot(&a, &c));
        let img = HashedEmbeddingTower::new(Modality::Image, 32).embed(&MediaPayload::Bytes(vec![1, 2, 3, 4, 5, 6]));
        assert_eq!(img.len(), 32);
        assert_eq!(t.id(), "hashed-v1-text-64");
    }

    #[test]
    fn point_in_time_and_tower_selection() {
        let item = |p| MediaItem {
            modality: Modality::Text,
            payload: MediaPayload::Text("x".into()),
            event_time_ms: 0,
            publish_time_ms: p,
        };
        let items = [item(10), item(20), item(30)];
        assert_eq!(point_in_time(&items, 20).len(), 2);
        let tower = HashedEmbeddingTower::new(Modality::Text, 16);
        let emb = embed_window(&[&tower], &items, 25);
        assert_eq!(emb.len(), 2);
        let cat = pretrained_catalog();
        assert!(cat.iter().any(|s| s.modality == Modality::Audio));
        let c = select_tower(Modality::Text, None, 16);
        assert!(!c.pretrained && c.tower.dim() == 16);
        let c = select_tower(Modality::Text, Some((&cat[1], &[], "abc")), 16);
        assert!(c.note.contains("sha256"));
    }
}
