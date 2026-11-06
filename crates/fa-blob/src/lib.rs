//! Garage S3 layout (spec 20 §5.5, spec 10 §4.2) and a storage trait.
use chrono::{DateTime, Datelike, Timelike, Utc};
use fa_core::Timescale;

pub mod buckets {
    pub const RAW: &str = "fa-raw";
    pub const MEDIA: &str = "fa-media";
    pub const MODELS: &str = "fa-models";
    pub const CKPT: &str = "fa-ckpt";
    pub const ARTIFACTS: &str = "fa-artifacts";
}

/// A fully qualified object location.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObjectKey {
    pub bucket: &'static str,
    pub key: String,
}

/// Raw payload: `fa-raw/<source>/<yyyy>/<mm>/<dd>/<hh>/<ulid>.zst`.
pub fn raw_payload(source: &str, received: DateTime<Utc>, id: ulid::Ulid) -> ObjectKey {
    ObjectKey {
        bucket: buckets::RAW,
        key: format!(
            "{source}/{:04}/{:02}/{:02}/{:02}/{id}.zst",
            received.year(),
            received.month(),
            received.day(),
            received.hour()
        ),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaPart {
    Video,
    Audio,
    Frames,
    Transcript,
}

impl MediaPart {
    fn as_str(self) -> &'static str {
        match self {
            Self::Video => "video",
            Self::Audio => "audio",
            Self::Frames => "frames",
            Self::Transcript => "transcript",
        }
    }
}

/// Media: `fa-media/<source>/<media_id>/<part>/<name>`.
pub fn media(source: &str, media_id: &str, part: MediaPart, name: &str) -> ObjectKey {
    ObjectKey { bucket: buckets::MEDIA, key: format!("{source}/{media_id}/{}/{name}", part.as_str()) }
}

/// Encoder snapshot: `fa-models/snapshots/<timescale>/<family>/<gen>/<name>`.
pub fn snapshot(ts: Timescale, family: &str, generation: u32, name: &str) -> ObjectKey {
    ObjectKey { bucket: buckets::MODELS, key: format!("snapshots/{ts}/{family}/{generation}/{name}") }
}

/// Online adapter state: `fa-models/adapters/<snapshot_id>/<version>`.
pub fn adapter(snapshot_id: &str, version: u64) -> ObjectKey {
    ObjectKey { bucket: buckets::MODELS, key: format!("adapters/{snapshot_id}/{version}") }
}

/// Genome weights: `fa-models/genomes/<genome_id>/weights`.
pub fn genome_weights(genome_id: &str) -> ObjectKey {
    ObjectKey { bucket: buckets::MODELS, key: format!("genomes/{genome_id}/weights") }
}

/// Job checkpoint: `fa-ckpt/<job_id>/<seq>`.
pub fn checkpoint(job_id: &str, seq: u32) -> ObjectKey {
    ObjectKey { bucket: buckets::CKPT, key: format!("{job_id}/{seq:06}") }
}

/// Decoder output artifact: `fa-artifacts/decoders/<family>/<decoder_id>/<name>`.
pub fn decoder_artifact(family: &str, decoder_id: &str, name: &str) -> ObjectKey {
    ObjectKey { bucket: buckets::ARTIFACTS, key: format!("decoders/{family}/{decoder_id}/{name}") }
}

#[derive(Debug, thiserror::Error)]
pub enum BlobError {
    #[error("not found: {0}/{1}")]
    NotFound(&'static str, String),
    #[error("blob store not configured")]
    NotConfigured,
    #[error("backend: {0}")]
    Backend(String),
}

/// Object storage abstraction. The Garage S3 implementation is added when the
/// first producer needs it; `MemoryStore` serves tests and local dry runs.
pub trait BlobStore: Send + Sync {
    fn put(&self, key: &ObjectKey, bytes: Vec<u8>) -> Result<(), BlobError>;
    fn get(&self, key: &ObjectKey) -> Result<Vec<u8>, BlobError>;
}

#[derive(Default)]
pub struct MemoryStore {
    inner: std::sync::Mutex<std::collections::HashMap<ObjectKey, Vec<u8>>>,
}

impl BlobStore for MemoryStore {
    fn put(&self, key: &ObjectKey, bytes: Vec<u8>) -> Result<(), BlobError> {
        self.inner.lock().map_err(|e| BlobError::Backend(e.to_string()))?.insert(key.clone(), bytes);
        Ok(())
    }

    fn get(&self, key: &ObjectKey) -> Result<Vec<u8>, BlobError> {
        self.inner
            .lock()
            .map_err(|e| BlobError::Backend(e.to_string()))?
            .get(key)
            .cloned()
            .ok_or_else(|| BlobError::NotFound(key.bucket, key.key.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn raw_layout() {
        let t = Utc.with_ymd_and_hms(2026, 11, 6, 9, 5, 0).unwrap();
        let id = ulid::Ulid::from_string("01JC0000000000000000000000").unwrap();
        let k = raw_payload("lemon", t, id);
        assert_eq!(k.bucket, "fa-raw");
        assert_eq!(k.key, "lemon/2026/11/06/09/01JC0000000000000000000000.zst");
    }

    #[test]
    fn model_layouts() {
        assert_eq!(snapshot(Timescale::Hour, "fsq_hybrid", 3, "weights.mpk").key, "snapshots/1h/fsq_hybrid/3/weights.mpk");
        assert_eq!(checkpoint("job1", 7).key, "job1/000007");
        assert_eq!(media("youtube", "m1", MediaPart::Frames, "0001.jpg").key, "youtube/m1/frames/0001.jpg");
    }

    #[test]
    fn memory_store_roundtrip() {
        let s = MemoryStore::default();
        let k = genome_weights("g1");
        s.put(&k, vec![1, 2]).unwrap();
        assert_eq!(s.get(&k).unwrap(), vec![1, 2]);
        assert!(s.get(&genome_weights("g2")).is_err());
    }
}
