//! News embeddings for the encoder's text expert (D-117).
//!
//! Every `news.item.>` envelope is embedded **once, when it arrives**, and the vector is written to
//! the store with the item's own knowledge time. The encoder then reads stored vectors as of a
//! window end and never calls an API on the model path. That ordering is not an optimisation, it
//! is what keeps the platform replayable: spec 41's equivalence tests re-run encoders over the
//! retained streams, and an endpoint cannot be asked to return the same vector twice — it can be
//! rate-limited, re-versioned or simply gone.
//!
//! When no endpoint is configured, or a call fails, the item is embedded with the deterministic
//! hashed fallback instead and the row records which path produced it (`provider`), so a latent
//! built on fallback vectors is visible rather than quietly different. The fallback is the same
//! byte-4-gram hashing the decoders already use for text, projected to the configured width.
use std::collections::VecDeque;

use fa_hub::{HubClient, vectors::NewEmbedding};
use fa_proto::v1::{Envelope, envelope::Payload};
use sha2::{Digest, Sha256};

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

/// How many items are embedded in one request / written in one batch.
pub const BATCH: usize = 32;

/// How long a partial batch waits for company before it is flushed.
pub const LINGER_MS: u64 = 2_000;

/// Longest text sent to the endpoint per item. News bodies run to tens of kilobytes and the first
/// few hundred words carry the signal; the cap also bounds spend per item.
pub const TEXT_LIMIT: usize = 4_000;

/// One item waiting to be embedded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pending {
    /// Stable id of the item: its `dedup_key` when the source set one, else a digest of the text.
    pub doc_id: String,
    pub source: String,
    pub ts_ms: i64,
    pub text: String,
}

/// Which path produced a vector, recorded with the row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// The configured endpoint's embedding model.
    Endpoint,
    /// The deterministic hashed fallback (no endpoint, or the call failed).
    Hashed,
}

impl Provider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Endpoint => "endpoint",
            Self::Hashed => "hashed",
        }
    }
}

/// The text and identity of one news envelope, or `None` when there is nothing to embed.
///
/// `body_ref`-only items (long bodies parked in Garage) carry no text on the bus; they are
/// embedded from title and outlet alone rather than skipped, because a headline is most of the
/// signal and fetching the blob would put object storage on the ingest path.
pub fn pending_of(subject: &str, env: &Envelope) -> Option<Pending> {
    let Some(Payload::News(n)) = env.payload.as_ref() else { return None };
    let title = n.title.trim();
    let body = n.body.trim();
    if title.is_empty() && body.is_empty() {
        return None;
    }
    let source = subject.rsplit('.').next().unwrap_or("unknown").to_owned();
    let mut text = String::new();
    if !title.is_empty() {
        text.push_str(title);
    }
    if !body.is_empty() {
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        text.push_str(body);
    }
    if text.len() > TEXT_LIMIT {
        // Cut on a char boundary: `text` is UTF-8 and slicing mid-codepoint would panic.
        let mut end = TEXT_LIMIT;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
    }
    let ts_ms = env
        .event_time
        .as_ref()
        .or(env.publish_time.as_ref())
        .map(|t| t.seconds * 1_000 + i64::from(t.nanos) / 1_000_000)
        .unwrap_or_else(now_ms);
    let doc_id = if n.dedup_key.trim().is_empty() { digest_id(&source, &text) } else { n.dedup_key.trim().to_owned() };
    Some(Pending { doc_id, source, ts_ms, text })
}

/// `news-<16 hex>` from source and text: the id of an item whose source set no dedup key.
///
/// Content-addressed on purpose — the same item republished by the same source embeds once, and a
/// restart cannot create a second vector for it.
pub fn digest_id(source: &str, text: &str) -> String {
    let mut h = Sha256::new();
    h.update(source.as_bytes());
    h.update([0]);
    h.update(text.as_bytes());
    format!("news-{}", hex::encode(&h.finalize()[..8]))
}

/// Deterministic hashed embedding: byte 4-grams into `dim` buckets, signed by a second hash, then
/// L2-normalised. The same text always gives the same vector, which is what makes it usable as a
/// replay-safe fallback.
pub fn hashed_embedding(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0f32; dim.max(1)];
    let b = text.as_bytes();
    if b.len() >= 4 {
        for w in b.windows(4) {
            let mut h = Sha256::new();
            h.update(w);
            let d = h.finalize();
            let idx = u32::from_le_bytes([d[0], d[1], d[2], d[3]]) as usize % v.len();
            let sign = if d[4] & 1 == 0 { 1.0 } else { -1.0 };
            v[idx] += sign;
        }
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// A bounded queue of items waiting to be embedded.
///
/// Bounded because the bus can outrun an API: on overflow the **oldest** item is dropped and
/// counted, so a backlog costs old headlines rather than the whole pipeline. The durable consumer
/// has already acked them, which is the right trade — a news vector that arrives late is worth
/// less than the live path staying responsive.
#[derive(Debug)]
pub struct Queue {
    items: VecDeque<Pending>,
    seen: std::collections::HashSet<String>,
    cap: usize,
    pub dropped: u64,
    pub duplicates: u64,
}

impl Queue {
    pub fn new(cap: usize) -> Self {
        Self { items: VecDeque::new(), seen: Default::default(), cap: cap.max(1), dropped: 0, duplicates: 0 }
    }

    /// Queues an item unless its `doc_id` is already queued or was embedded in this process.
    pub fn push(&mut self, p: Pending) -> bool {
        if !self.seen.insert(p.doc_id.clone()) {
            self.duplicates += 1;
            return false;
        }
        if self.items.len() >= self.cap
            && let Some(old) = self.items.pop_front()
        {
            self.seen.remove(&old.doc_id);
            self.dropped += 1;
        }
        self.items.push_back(p);
        true
    }

    /// Up to `n` items, oldest first.
    pub fn take(&mut self, n: usize) -> Vec<Pending> {
        let n = n.min(self.items.len());
        self.items.drain(..n).collect()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Rows already embedded, so a restart does not pay for them twice.
///
/// The queue's in-process dedupe set dies with the process; this is what survives it. A missing
/// or unreachable Postgres is not fatal — the worst case is re-embedding a few items — so every
/// query here degrades to "unknown" rather than stopping the pipeline.
#[derive(Clone)]
pub struct Store {
    pg: Option<sqlx::PgPool>,
}

impl Store {
    pub fn new(pg: Option<sqlx::PgPool>) -> Self {
        Self { pg }
    }

    /// Which of `ids` are already embedded.
    pub async fn known(&self, ids: &[String]) -> std::collections::HashSet<String> {
        let Some(pg) = &self.pg else { return Default::default() };
        if ids.is_empty() {
            return Default::default();
        }
        match sqlx::query_scalar::<_, String>("SELECT doc_id FROM llm.news_embedding WHERE doc_id = ANY($1)")
            .bind(ids)
            .fetch_all(pg)
            .await
        {
            Ok(rows) => rows.into_iter().collect(),
            Err(e) => {
                tracing::warn!(error = %e, "reading embedded ids failed; may re-embed");
                Default::default()
            }
        }
    }

    /// Records what was embedded and by which path.
    pub async fn record(&self, batch: &[Pending], provider: Provider, model: &str, dim: usize) {
        let Some(pg) = &self.pg else { return };
        for p in batch {
            let ts = chrono::DateTime::from_timestamp_millis(p.ts_ms).unwrap_or_else(chrono::Utc::now);
            if let Err(e) = sqlx::query(
                "INSERT INTO llm.news_embedding (doc_id, source, ts, provider, model, dim) \
                 VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (doc_id) DO NOTHING",
            )
            .bind(&p.doc_id)
            .bind(&p.source)
            .bind(ts)
            .bind(provider.as_str())
            .bind(model)
            .bind(dim as i32)
            .execute(pg)
            .await
            {
                tracing::warn!(error = %e, doc = %p.doc_id, "recording an embedding failed");
            }
        }
    }
}

/// Embeds one batch and writes it, returning the rows written and which path produced them.
///
/// A failed endpoint call is not an error for the caller: the batch is embedded with the hashed
/// fallback and written anyway, because the alternative is a gap in the text expert's input for
/// exactly the period when the endpoint was unavailable.
pub async fn embed_and_store(
    hub: &HubClient,
    llm: Option<&fa_llm::LlmClient>,
    dim: usize,
    batch: &[Pending],
) -> anyhow::Result<(usize, Provider)> {
    if batch.is_empty() {
        return Ok((0, Provider::Hashed));
    }
    let (vectors, provider) = match llm {
        Some(c) => {
            let inputs: Vec<String> = batch.iter().map(|p| p.text.clone()).collect();
            match c.embed(&inputs).await {
                Ok(v) => (v, Provider::Endpoint),
                Err(e) => {
                    tracing::warn!(error = %e, n = batch.len(), "embedding call failed; using the hashed fallback");
                    (batch.iter().map(|p| hashed_embedding(&p.text, dim)).collect(), Provider::Hashed)
                }
            }
        }
        None => (batch.iter().map(|p| hashed_embedding(&p.text, dim)).collect(), Provider::Hashed),
    };
    let docs: Vec<NewEmbedding> = batch
        .iter()
        .zip(vectors)
        .map(|(p, vector)| NewEmbedding {
            id: p.doc_id.clone(),
            source: p.source.clone(),
            ts_ms: p.ts_ms,
            vector,
        })
        .collect();
    hub.add_news_docs(&docs).await?;
    Ok((docs.len(), provider))
}

#[cfg(test)]
mod tests {
    use fa_proto::v1::NewsItem;

    use super::*;

    fn env(title: &str, body: &str, dedup: &str, secs: i64) -> Envelope {
        Envelope {
            event_time: Some(fa_proto::Timestamp { seconds: secs, nanos: 0 }),
            payload: Some(Payload::News(NewsItem {
                title: title.into(),
                body: body.into(),
                dedup_key: dedup.into(),
                ..Default::default()
            })),
            ..Default::default()
        }
    }

    #[test]
    fn the_item_id_is_its_dedup_key_or_a_content_digest() {
        let p = pending_of("news.item.hn", &env("Title", "Body", "hn:42", 100)).unwrap();
        assert_eq!(p.doc_id, "hn:42");
        assert_eq!(p.source, "hn");
        assert_eq!(p.ts_ms, 100_000);
        assert_eq!(p.text, "Title\n\nBody");

        let a = pending_of("news.item.gdelt", &env("Same", "Text", "", 1)).unwrap();
        let b = pending_of("news.item.gdelt", &env("Same", "Text", "", 9)).unwrap();
        assert_eq!(a.doc_id, b.doc_id, "content-addressed: a republished item embeds once");
        let c = pending_of("news.item.hn", &env("Same", "Text", "", 1)).unwrap();
        assert_ne!(a.doc_id, c.doc_id, "but the source is part of the identity");
    }

    #[test]
    fn an_item_with_no_text_is_skipped_and_a_long_one_is_cut_on_a_boundary() {
        assert!(pending_of("news.item.hn", &env("", "", "k", 1)).is_none());
        assert!(pending_of("news.item.hn", &Envelope::default()).is_none());
        // A multi-byte character straddling the limit must not panic or split a codepoint.
        let long = "é".repeat(TEXT_LIMIT);
        let p = pending_of("news.item.hn", &env("", &long, "k", 1)).unwrap();
        assert!(p.text.len() <= TEXT_LIMIT);
        assert!(p.text.chars().all(|c| c == 'é'));
    }

    #[test]
    fn the_hashed_fallback_is_deterministic_normalised_and_width_correct() {
        let a = hashed_embedding("the quick brown fox", 64);
        let b = hashed_embedding("the quick brown fox", 64);
        assert_eq!(a, b, "a replay must produce the same vector");
        assert_eq!(a.len(), 64);
        let norm = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "normalised, so cosine distance behaves: {norm}");
        assert_ne!(a, hashed_embedding("a different headline entirely", 64));
        // Degenerate inputs stay finite and the right width.
        assert_eq!(hashed_embedding("", 8), vec![0.0; 8]);
        assert!(hashed_embedding("ab", 8).iter().all(|x| x.is_finite()));
    }

    #[test]
    fn the_queue_dedupes_and_drops_the_oldest_when_full() {
        let mut q = Queue::new(2);
        let p = |id: &str| Pending { doc_id: id.into(), source: "hn".into(), ts_ms: 1, text: id.into() };
        assert!(q.push(p("a")));
        assert!(!q.push(p("a")), "the same id is not queued twice");
        assert_eq!(q.duplicates, 1);
        assert!(q.push(p("b")));
        assert!(q.push(p("c")));
        assert_eq!(q.dropped, 1, "over capacity the oldest goes, not the newest");
        assert_eq!(q.len(), 2);
        let taken = q.take(5);
        assert_eq!(taken.iter().map(|x| x.doc_id.as_str()).collect::<Vec<_>>(), ["b", "c"]);
        assert!(q.is_empty());
    }
}
