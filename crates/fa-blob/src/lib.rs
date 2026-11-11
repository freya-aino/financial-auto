//! Garage S3 layout (spec 20 §5.5, spec 10 §4.2), a storage trait, an in-memory store and
//! the SigV4 S3 client (`S3BlobStore`) for Garage.
use chrono::{DateTime, Datelike, Timelike, Utc};
use fa_core::Timescale;

pub mod gc;
pub mod s3;
pub mod sigv4;

pub use s3::S3BlobStore;

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

/// A file of a trained snapshot addressed by id: `fa-models/snapshots/<snapshot_id>/<name>`
/// (spec 36 §5; `manifest.json`, `weights.bin`, `regions.json`, `region_map.json`).
pub fn snapshot_file(snapshot_id: &str, name: &str) -> ObjectKey {
    ObjectKey { bucket: buckets::MODELS, key: format!("snapshots/{snapshot_id}/{name}") }
}

/// Prefix of every file of one snapshot (for listings).
pub fn snapshot_prefix(snapshot_id: &str) -> String {
    format!("snapshots/{snapshot_id}/")
}

/// Versioned online-adapter artifact: `fa-models/adapters/<snapshot_id>/<unix_ms>.json` (spec 36 §6.5).
pub fn adapter_state(snapshot_id: &str, ts_ms: i64) -> ObjectKey {
    ObjectKey { bucket: buckets::MODELS, key: format!("adapters/{snapshot_id}/{ts_ms}.json") }
}

/// Resumable job state: `fa-ckpt/<job_id>/state.json` (spec 36 §6.7; the trainer's
/// checkpoint, overwritten on every save so `checkpoint_ref` stays one key).
pub fn job_state(job_id: &str) -> ObjectKey {
    ObjectKey { bucket: buckets::CKPT, key: format!("{job_id}/state.json") }
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

/// Evaluation report of one evo job: `fa-artifacts/evals/<generation_id>/<job_id>.json`
/// (`generation_id` = `Job.parent_job_id`, or `adhoc` when empty).
pub fn eval_report(generation_id: &str, job_id: &str) -> ObjectKey {
    let generation = if generation_id.is_empty() { "adhoc" } else { generation_id };
    ObjectKey { bucket: buckets::ARTIFACTS, key: format!("evals/{generation}/{job_id}.json") }
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

/// One entry of a prefix listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectMeta {
    pub key: String,
    pub size: u64,
    /// RFC 3339 as reported by the server.
    pub last_modified: Option<String>,
    pub etag: Option<String>,
}

pub const OCTET_STREAM: &str = "application/octet-stream";
pub const JSON: &str = "application/json";

/// Object storage abstraction. [`S3BlobStore`] talks to Garage; `MemoryStore` serves tests
/// and local dry runs. Methods return `Send` futures so stores can be shared across tasks.
pub trait BlobStore: Send + Sync {
    fn put(
        &self,
        key: &ObjectKey,
        bytes: Vec<u8>,
        content_type: &str,
    ) -> impl Future<Output = Result<(), BlobError>> + Send;
    fn get(&self, key: &ObjectKey) -> impl Future<Output = Result<Vec<u8>, BlobError>> + Send;
    /// HEAD: `Ok(false)` when the object does not exist.
    fn exists(&self, key: &ObjectKey) -> impl Future<Output = Result<bool, BlobError>> + Send;
    /// Deleting a missing object is not an error.
    fn delete(&self, key: &ObjectKey) -> impl Future<Output = Result<(), BlobError>> + Send;
    /// All objects under `prefix` in `bucket` (follows pagination).
    fn list(
        &self,
        bucket: &'static str,
        prefix: &str,
    ) -> impl Future<Output = Result<Vec<ObjectMeta>, BlobError>> + Send;
}

/// `(content_type, bytes)` per key.
type MemoryMap = std::collections::BTreeMap<ObjectKey, (String, Vec<u8>)>;

#[derive(Default)]
pub struct MemoryStore {
    inner: std::sync::Mutex<MemoryMap>,
}

impl MemoryStore {
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, MemoryMap>, BlobError> {
        self.inner.lock().map_err(|e| BlobError::Backend(e.to_string()))
    }

    /// Content type recorded with `put`.
    pub fn content_type(&self, key: &ObjectKey) -> Option<String> {
        self.inner.lock().ok()?.get(key).map(|(ct, _)| ct.clone())
    }
}

impl std::cmp::PartialOrd for ObjectKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Ord for ObjectKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.bucket, &self.key).cmp(&(other.bucket, &other.key))
    }
}

impl BlobStore for MemoryStore {
    async fn put(&self, key: &ObjectKey, bytes: Vec<u8>, content_type: &str) -> Result<(), BlobError> {
        self.lock()?.insert(key.clone(), (content_type.to_owned(), bytes));
        Ok(())
    }

    async fn get(&self, key: &ObjectKey) -> Result<Vec<u8>, BlobError> {
        self.lock()?.get(key).map(|(_, b)| b.clone()).ok_or_else(|| BlobError::NotFound(key.bucket, key.key.clone()))
    }

    async fn exists(&self, key: &ObjectKey) -> Result<bool, BlobError> {
        Ok(self.lock()?.contains_key(key))
    }

    async fn delete(&self, key: &ObjectKey) -> Result<(), BlobError> {
        self.lock()?.remove(key);
        Ok(())
    }

    async fn list(&self, bucket: &'static str, prefix: &str) -> Result<Vec<ObjectMeta>, BlobError> {
        Ok(self
            .lock()?
            .iter()
            .filter(|(k, _)| k.bucket == bucket && k.key.starts_with(prefix))
            .map(|(k, (_, b))| ObjectMeta { key: k.key.clone(), size: b.len() as u64, last_modified: None, etag: None })
            .collect())
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
        assert_eq!(
            snapshot(Timescale::Hour, "fsq_hybrid", 3, "weights.mpk").key,
            "snapshots/1h/fsq_hybrid/3/weights.mpk"
        );
        assert_eq!(checkpoint("job1", 7).key, "job1/000007");
        assert_eq!(snapshot_file("stat-v2-1m-01J", "manifest.json").key, "snapshots/stat-v2-1m-01J/manifest.json");
        assert_eq!(snapshot_file("s", "weights.bin").bucket, "fa-models");
        assert_eq!(snapshot_prefix("s"), "snapshots/s/");
        assert_eq!(adapter_state("stat-v1-1m", 1_700_000_000_000).key, "adapters/stat-v1-1m/1700000000000.json");
        assert_eq!(job_state("j1").key, "j1/state.json");
        assert_eq!(job_state("j1").bucket, "fa-ckpt");
        assert_eq!(eval_report("", "j1").key, "evals/adhoc/j1.json");
        assert_eq!(eval_report("gen1", "j1").bucket, "fa-artifacts");
        assert_eq!(media("youtube", "m1", MediaPart::Frames, "0001.jpg").key, "youtube/m1/frames/0001.jpg");
    }

    #[test]
    fn memory_store_roundtrip() {
        let s = MemoryStore::default();
        let k = genome_weights("g1");
        futures_lite_block_on(async {
            s.put(&k, vec![1, 2], OCTET_STREAM).await.unwrap();
            assert_eq!(s.get(&k).await.unwrap(), vec![1, 2]);
            assert!(s.exists(&k).await.unwrap());
            assert!(!s.exists(&genome_weights("g2")).await.unwrap());
            assert!(matches!(s.get(&genome_weights("g2")).await, Err(BlobError::NotFound(..))));
            s.put(&eval_report("g", "j"), b"{}".to_vec(), JSON).await.unwrap();
            assert_eq!(s.content_type(&eval_report("g", "j")).as_deref(), Some(JSON));
            let l = s.list(buckets::MODELS, "genomes/").await.unwrap();
            assert_eq!(l.len(), 1);
            assert_eq!(l[0].key, "genomes/g1/weights");
            assert!(s.list(buckets::ARTIFACTS, "nothing/").await.unwrap().is_empty());
            s.delete(&k).await.unwrap();
            s.delete(&k).await.unwrap();
            assert!(!s.exists(&k).await.unwrap());
        });
    }

    fn futures_lite_block_on<F: Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(f)
    }
}
