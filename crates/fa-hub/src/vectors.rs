//! Vector plumbing (spec 20 §4 `knn`, §5.2 "vectors indexed per ExpertSpace … plus text/media
//! embedding indexes", D-024, D-043).
//!
//! - **Latents.** One label `LatentVec`; a `d`-dimensional vector lives in property `vec<d>`;
//!   one cosine index per `d`, tenant-partitioned by `expert_space_id`, created on demand and
//!   remembered per client (the DDL is not re-sent on every write). The registered
//!   `ExpertSpace.dim` is the authority: [`HubClient::add_latents_checked`] and
//!   [`HubClient::knn`] reject vectors of another length with [`HubError::DimMismatch`].
//! - **Text / media.** `NewsDoc` / `MediaSegment` embeddings use the same `vec<d>` convention with
//!   one un-partitioned cosine index per `(label, d)` ([`crate::schema::embedding_index`]).
//! - Every vector is validated before it is sent: empty, non-finite (NaN would serialize as
//!   `null`) and zero-norm vectors (the cosine index answers 400 `zero_norm_cosine_vector`) fail
//!   with [`HubError::InvalidVector`].

use std::collections::BTreeSet;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::ast::{Batch, Input, Op, Pred, Val};
use crate::queries::{LatentHit, NewLatent};
use crate::schema::{LATENT_TENANT, embedding_index, latent_vec_property, node};
use crate::{HubClient, HubError, Result, lock};

/// How long the `*_checked` writers wait for a freshly created vector index.
pub const INDEX_READY_WAIT: Duration = Duration::from_secs(60);
/// Upper bound on the candidates a filtered [`HubClient::knn`] asks the index for.
pub const MAX_KNN_FETCH: usize = 4096;
/// Default over-fetch factor of a filtered k-NN (the filter is applied after the index top-k).
pub const DEFAULT_OVERFETCH: usize = 4;

/// Reject vectors the server cannot store or search.
pub fn validate_vector(v: &[f32]) -> Result<()> {
    if v.is_empty() {
        return Err(HubError::InvalidVector("empty vector".into()));
    }
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(HubError::InvalidVector(format!("component {i} is not finite ({})", v[i])));
    }
    if v.iter().all(|x| *x == 0.0) {
        return Err(HubError::InvalidVector("zero-norm vector (cosine index)".into()));
    }
    Ok(())
}

fn latent_props(l: &NewLatent) -> Vec<(String, Input)> {
    vec![
        (LATENT_TENANT.to_owned(), Input::Lit(Val::Str(l.expert_space_id.clone()))),
        ("kind".into(), Input::Lit(Val::Str(l.kind.clone()))),
        ("ref_id".into(), Input::Lit(Val::Str(l.ref_id.clone()))),
        ("instrument_id".into(), Input::Lit(Val::Str(l.instrument_id.clone()))),
        ("ts".into(), Input::Lit(Val::DateTime(l.ts_ms))),
        ("uncertainty".into(), l.uncertainty.into()),
        (latent_vec_property(l.vector.len()), l.vector.clone().into()),
    ]
}

/// The write batch of `addLatent` for several rows (returns `n0`, `n1`, …). Validates every
/// vector. Used by [`HubClient::add_latents`] and for queuing in a [`crate::WriteBuffer`].
pub fn add_latents_batch(latents: &[NewLatent]) -> Result<Batch> {
    let mut batch = Batch::write().named("add_latents");
    for (i, l) in latents.iter().enumerate() {
        validate_vector(&l.vector)?;
        if l.expert_space_id.is_empty() {
            return Err(HubError::BadParam("latent without expert_space_id (index tenant)".into()));
        }
        let props = latent_props(l);
        let props = props.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        batch = batch.ret(&format!("n{i}"), Op::add_n(node::LATENT_VEC, props));
    }
    Ok(batch)
}

/// Filter of [`HubClient::knn`] (spec 20 §4 `KnnFilter`). Empty = plain top-k in the space.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KnnFilter {
    /// `centroid | observation | window | aggregate`.
    pub kind: Option<String>,
    pub instrument_id: Option<String>,
    /// As-of rule (spec 20 §6.3): only rows with `ts <= as_of_ms`.
    pub as_of_ms: Option<i64>,
    /// Only rows with `ts >= since_ms`.
    pub since_ms: Option<i64>,
    /// Drop these `ref_id`s (e.g. the query window itself).
    pub exclude_ref_ids: Vec<String>,
    /// Drop hits farther than this cosine distance.
    pub max_distance: Option<f64>,
    /// Candidates fetched per requested hit when a property filter is set (default
    /// [`DEFAULT_OVERFETCH`]); the server filters after the index top-k, so a selective filter
    /// needs a larger factor to still return `k` hits.
    pub overfetch: Option<usize>,
}

impl KnnFilter {
    fn predicate(&self) -> Option<Pred> {
        let mut p = Vec::new();
        if let Some(k) = self.kind.as_deref().filter(|k| !k.is_empty()) {
            p.push(Pred::eq("kind", k));
        }
        if let Some(i) = self.instrument_id.as_deref().filter(|i| !i.is_empty()) {
            p.push(Pred::eq("instrument_id", i));
        }
        if let Some(t) = self.as_of_ms {
            p.push(Pred::lte("ts", Input::Lit(Val::DateTime(t))));
        }
        if let Some(t) = self.since_ms {
            p.push(Pred::gte("ts", Input::Lit(Val::DateTime(t))));
        }
        for r in &self.exclude_ref_ids {
            p.push(Pred::negate(Pred::eq("ref_id", r.as_str())));
        }
        (!p.is_empty()).then(|| Pred::and(p))
    }

    /// Number of index candidates to request for `k` hits.
    pub fn fetch_k(&self, k: usize) -> usize {
        if self.predicate().is_none() {
            return k;
        }
        k.saturating_mul(self.overfetch.unwrap_or(DEFAULT_OVERFETCH).max(1)).clamp(k, MAX_KNN_FETCH.max(k))
    }
}

/// Write payload of a `NewsDoc` / `MediaSegment` embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct NewEmbedding {
    /// `doc_id` (NewsDoc) or `segment_id` (MediaSegment).
    pub id: String,
    /// `source_id` (NewsDoc) or `media_id` (MediaSegment).
    pub source: String,
    /// Publication / segment time, unix ms.
    pub ts_ms: i64,
    pub vector: Vec<f32>,
}

/// A text / media vector-search hit.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct EmbeddingHit {
    #[serde(rename = "$id")]
    pub node: u64,
    #[serde(rename = "$distance")]
    pub distance: f64,
    #[serde(alias = "doc_id", alias = "segment_id", default)]
    pub id: String,
    #[serde(alias = "source_id", alias = "media_id", default)]
    pub source: String,
    #[serde(default)]
    pub ts: Option<String>,
}

/// `(id property, source property)` of an embedding label.
fn embedding_keys(label: &str) -> Result<(&'static str, &'static str)> {
    match label {
        node::NEWS_DOC => Ok(("doc_id", "source_id")),
        node::MEDIA_SEGMENT => Ok(("segment_id", "media_id")),
        other => Err(HubError::BadParam(format!("`{other}` is not an embedding label"))),
    }
}

fn index_missing(e: &HubError) -> bool {
    e.server_code() == Some("index_not_found")
}

impl HubClient {
    /// `ExpertSpace.dim` of a registered space (cached per client). [`HubError::NotFound`] when
    /// the space is not registered.
    pub async fn space_dim(&self, expert_space_id: &str) -> Result<u32> {
        if let Some(d) = lock(&self.cache.space_dims).get(expert_space_id) {
            return Ok(*d);
        }
        let space = self
            .space_by_id(expert_space_id)
            .await?
            .ok_or_else(|| HubError::NotFound { label: node::EXPERT_SPACE.into(), id: expert_space_id.into() })?;
        lock(&self.cache.space_dims).insert(expert_space_id.to_owned(), space.dim);
        Ok(space.dim)
    }

    /// Validate `vector` against the registered dimension of `expert_space_id`.
    pub async fn check_space_dim(&self, expert_space_id: &str, vector: &[f32]) -> Result<()> {
        validate_vector(vector)?;
        let expected = self.space_dim(expert_space_id).await?;
        if expected as usize != vector.len() {
            return Err(HubError::DimMismatch { expert_space_id: expert_space_id.into(), expected, got: vector.len() });
        }
        Ok(())
    }

    /// `addLatent` with the full contract: every vector valid and of its space's registered
    /// `dim` (spaces must exist), the `vec<dim>` index ensured once per client and dimension,
    /// then one write batch. Returns the node ids in input order.
    pub async fn add_latents_checked(&self, latents: &[NewLatent]) -> Result<Vec<u64>> {
        if latents.is_empty() {
            return Ok(Vec::new());
        }
        let mut dims = BTreeSet::new();
        for l in latents {
            self.check_space_dim(&l.expert_space_id, &l.vector).await?;
            dims.insert(l.vector.len());
        }
        for d in dims {
            self.ensure_latent_index_ready(d, INDEX_READY_WAIT).await?;
        }
        self.add_latents(latents).await
    }

    /// k nearest `LatentVec`s of one expert space (spec 20 §4 `knn(space, v, k, filter)`), cosine
    /// distance ascending. The vector must match the space's registered `dim`; a dimension whose
    /// index does not exist yet has no vectors, so it answers an empty list. Filters run on the
    /// server after the index top-k (over-fetched by [`KnnFilter::fetch_k`]); `max_distance` and
    /// the final cut to `k` run here.
    pub async fn knn(
        &self,
        expert_space_id: &str,
        vector: &[f32],
        k: usize,
        filter: &KnnFilter,
    ) -> Result<Vec<LatentHit>> {
        self.check_space_dim(expert_space_id, vector).await?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let prop = latent_vec_property(vector.len());
        let mut op = Op::vector_search_nodes(
            node::LATENT_VEC,
            &prop,
            vector.to_vec(),
            filter.fetch_k(k),
            Some(Input::Lit(Val::Str(expert_space_id.to_owned()))),
        );
        if let Some(p) = filter.predicate() {
            op = op.where_(p);
        }
        let op = op.value_map(Some(&[
            "$id",
            "$distance",
            LATENT_TENANT,
            "kind",
            "ref_id",
            "instrument_id",
            "ts",
            "uncertainty",
        ]));
        let mut hits: Vec<LatentHit> = match self.batch(&Batch::read().named("knn").ret("hits", op)).await {
            Ok(mut resp) => resp.take("hits")?,
            Err(e) if index_missing(&e) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        if let Some(max) = filter.max_distance {
            hits.retain(|h| h.distance <= max);
        }
        hits.truncate(k);
        Ok(hits)
    }

    // ── text / media embeddings ─────────────────────────────────────────────

    async fn add_embeddings(&self, label: &'static str, rows: &[NewEmbedding]) -> Result<Vec<u64>> {
        let (id_key, source_key) = embedding_keys(label)?;
        let Some(first) = rows.first() else { return Ok(Vec::new()) };
        let dim = first.vector.len();
        for r in rows {
            validate_vector(&r.vector)?;
            if r.vector.len() != dim {
                return Err(HubError::InvalidVector(format!(
                    "mixed dimensions in one {label} batch ({dim} and {})",
                    r.vector.len()
                )));
            }
        }
        let prop = latent_vec_property(dim);
        self.ensure_vector_index_ready(embedding_index(label, dim), label, &prop, INDEX_READY_WAIT).await?;
        let mut batch = Batch::write().named(format!("add_{}", label.to_lowercase()));
        for (i, r) in rows.iter().enumerate() {
            let props = vec![
                (id_key, Input::Lit(Val::Str(r.id.clone()))),
                (source_key, Input::Lit(Val::Str(r.source.clone()))),
                ("ts", Input::Lit(Val::DateTime(r.ts_ms))),
                (prop.as_str(), r.vector.clone().into()),
            ];
            batch = batch.ret(&format!("n{i}"), Op::add_n(label, props));
        }
        let mut resp = self.batch(&batch).await?;
        (0..rows.len())
            .map(|i| {
                let ids: Vec<serde_json::Value> = resp.take(&format!("n{i}"))?;
                ids.first()
                    .and_then(|v| v["$id"].as_u64())
                    .ok_or_else(|| HubError::Shape { name: format!("n{i}"), detail: "no $id".into() })
            })
            .collect()
    }

    async fn search_embeddings(
        &self,
        label: &'static str,
        vector: &[f32],
        k: usize,
        as_of_ms: Option<i64>,
    ) -> Result<Vec<EmbeddingHit>> {
        let (id_key, source_key) = embedding_keys(label)?;
        validate_vector(vector)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let prop = latent_vec_property(vector.len());
        let fetch = if as_of_ms.is_some() { k.saturating_mul(DEFAULT_OVERFETCH).min(MAX_KNN_FETCH.max(k)) } else { k };
        let mut op = Op::vector_search_nodes(label, &prop, vector.to_vec(), fetch, None);
        if let Some(t) = as_of_ms {
            op = op.where_(Pred::lte("ts", Input::Lit(Val::DateTime(t))));
        }
        let op = op.value_map(Some(&["$id", "$distance", id_key, source_key, "ts"]));
        let mut hits: Vec<EmbeddingHit> =
            match self.batch(&Batch::read().named(format!("search_{}", label.to_lowercase())).ret("hits", op)).await {
                Ok(mut resp) => resp.take("hits")?,
                Err(e) if index_missing(&e) => return Ok(Vec::new()),
                Err(e) => return Err(e),
            };
        hits.truncate(k);
        Ok(hits)
    }

    /// `addNewsDoc`: text embeddings (one dimension per call), index `NewsDoc.vec<dim>` ensured
    /// once per client. Returns node ids.
    pub async fn add_news_docs(&self, docs: &[NewEmbedding]) -> Result<Vec<u64>> {
        self.add_embeddings(node::NEWS_DOC, docs).await
    }

    /// `searchNews`: k nearest `NewsDoc`s, optionally as of `ts <= as_of_ms`.
    pub async fn search_news_docs(&self, vector: &[f32], k: usize, as_of_ms: Option<i64>) -> Result<Vec<EmbeddingHit>> {
        self.search_embeddings(node::NEWS_DOC, vector, k, as_of_ms).await
    }

    /// `addMediaSegment`: media embeddings, index `MediaSegment.vec<dim>` ensured once per client.
    pub async fn add_media_segments(&self, segments: &[NewEmbedding]) -> Result<Vec<u64>> {
        self.add_embeddings(node::MEDIA_SEGMENT, segments).await
    }

    /// `searchMedia`: k nearest `MediaSegment`s, optionally as of `ts <= as_of_ms`.
    pub async fn search_media_segments(
        &self,
        vector: &[f32],
        k: usize,
        as_of_ms: Option<i64>,
    ) -> Result<Vec<EmbeddingHit>> {
        self.search_embeddings(node::MEDIA_SEGMENT, vector, k, as_of_ms).await
    }

    /// Ensure (and wait for) the latent index of every dimension in `dims`; cached per client.
    pub async fn ensure_latent_indexes_ready(
        &self,
        dims: impl IntoIterator<Item = usize>,
        timeout: Duration,
    ) -> Result<()> {
        for d in dims.into_iter().collect::<BTreeSet<_>>() {
            self.ensure_latent_index_ready(d, timeout).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn latent(v: Vec<f32>) -> NewLatent {
        NewLatent {
            vector: v,
            expert_space_id: "sp".into(),
            kind: "observation".into(),
            ref_id: "r".into(),
            instrument_id: "i".into(),
            ts_ms: 1,
            uncertainty: 0.0,
        }
    }

    #[test]
    fn vectors_are_validated() {
        assert!(validate_vector(&[0.5, 0.0]).is_ok());
        assert!(matches!(validate_vector(&[]), Err(HubError::InvalidVector(_))));
        assert!(matches!(validate_vector(&[0.0, 0.0]), Err(HubError::InvalidVector(_))));
        assert!(matches!(validate_vector(&[1.0, f32::NAN]), Err(HubError::InvalidVector(_))));
        assert!(matches!(add_latents_batch(&[latent(vec![1.0]), latent(vec![])]), Err(HubError::InvalidVector(_))));
        assert!(HubError::InvalidVector("x".into()).is_client_error());
    }

    #[test]
    fn latent_batch_uses_the_dim_property() {
        let b = add_latents_batch(&[latent(vec![1.0, 0.0, 0.0]), latent(vec![1.0; 5])]).unwrap().to_json();
        let e = b["query"]["write"]["entries"].as_array().unwrap();
        let props = |i: usize| e[i]["query"]["root"]["add_n"]["properties"].as_array().unwrap().clone();
        assert!(props(0).iter().any(|p| p[0] == "vec3"));
        assert!(props(1).iter().any(|p| p[0] == "vec5"));
        assert_eq!(b["query"]["write"]["returns"], json!(["n0", "n1"]));
    }

    #[test]
    fn knn_filter_predicate_and_fetch() {
        let none = KnnFilter::default();
        assert!(none.predicate().is_none());
        assert_eq!(none.fetch_k(10), 10);
        let f = KnnFilter {
            kind: Some("window".into()),
            as_of_ms: Some(1000),
            exclude_ref_ids: vec!["w0".into()],
            ..KnnFilter::default()
        };
        let p = f.predicate().unwrap().into_json();
        let preds = p["and"]["predicates"].as_array().unwrap();
        assert_eq!(preds.len(), 3);
        assert_eq!(preds[1]["lte"]["right"], json!({ "constant": { "date_time": 1000 } }));
        assert!(preds[2].get("not").is_some());
        assert_eq!(f.fetch_k(10), 40);
        assert_eq!(f.fetch_k(5000), 5000, "never below k");
        assert_eq!(KnnFilter { overfetch: Some(1000), ..f }.fetch_k(10), MAX_KNN_FETCH);
    }

    #[test]
    fn embedding_hits_accept_either_label() {
        let n: EmbeddingHit = serde_json::from_value(
            json!({ "$id": 1, "$distance": 0.1, "doc_id": "d", "source_id": "rss", "ts": "2026-01-01T00:00:00.000Z" }),
        )
        .unwrap();
        assert_eq!((n.id.as_str(), n.source.as_str()), ("d", "rss"));
        let m: EmbeddingHit =
            serde_json::from_value(json!({ "$id": 2, "$distance": 0.2, "segment_id": "s", "media_id": "m" })).unwrap();
        assert_eq!((m.id.as_str(), m.source.as_str()), ("s", "m"));
        assert!(embedding_keys("LatentVec").is_err());
    }
}
