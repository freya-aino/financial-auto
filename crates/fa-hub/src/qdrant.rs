//! Qdrant vector store (ADR 0010, D-101): latents, news and media embeddings live in Qdrant;
//! the graph (snapshots, spaces, regions, transitions, lineage, genomes, reference mirror) stays
//! in HelixDB. A [`HubClient`] with a [`QdrantVectors`] attached routes every vector operation
//! here, so callers keep one client and one API.
//!
//! - **Collections per dimension**, cosine: `latent_d<dim>`, `news_d<dim>`, `media_d<dim>`.
//!   Created on first write with payload indexes — `expert_space_id` keyword marked `is_tenant`
//!   (one HNSW partition per expert space), `ts` integer range (unix ms), keyword `kind`,
//!   `instrument_id`, `ref_id` — and remembered per client.
//! - **Filters run inside the index** (`kind`, `instrument_id`, `ts` as-of / since, excluded
//!   `ref_id`s, `max_distance` as `score_threshold`), so a filtered k-NN returns `k` hits
//!   without the Helix over-fetch.
//! - **Point ids** are a 64-bit FNV-1a hash of the business key: `(space, kind, ref_id,
//!   instrument_id, ts)` for observations / windows (a redelivered write is idempotent),
//!   `(space, kind, ref_id)` for the one-per-region `centroid` / `aggregate` rows, `(label, id)`
//!   for embeddings.
//! - **Raw vectors.** Qdrant stores cosine vectors normalised, so the payload keeps the original
//!   `norm` and reads multiply it back (projection and retention means need the raw vector).
//! - Rows come back in the Helix shape (`$id`, `$distance` = 1 − cosine, `ts` as RFC 3339,
//!   vector in `vec<dim>`), so [`LatentHit`], [`LatentRow`] and [`EmbeddingHit`] are shared.
//!
//! The backend is chosen per process: [`configure_vectors`] from `[qdrant]` in
//! `config/platform.toml` (`vector_backend = "qdrant" | "helix"`), overridden by
//! `FA_VECTOR_BACKEND` / `FA_QDRANT_URL`; without either, clients use the Helix indexes.

use std::collections::{BTreeMap, HashSet};
use std::sync::{Mutex, RwLock};
use std::time::Duration;

use serde_json::{Map, Value as Json, json};

use crate::queries::{LatentHit, LatentRow, NewLatent};
use crate::retention::{AGGREGATE_KIND, AggRow, PRUNE_PAGE, PruneReport, ms_rfc3339, plan_folds};
use crate::schema::{LATENT_TENANT, latent_vec_property, node};
use crate::vectors::{EmbeddingHit, KnnFilter, NewEmbedding, validate_vector};
use crate::{HubClient, HubError, Result, lock};

/// Environment variable with the Qdrant base URL (also enables the backend).
pub const URL_ENV: &str = "FA_QDRANT_URL";
/// Environment variable forcing the vector backend: `qdrant` or `helix`.
pub const BACKEND_ENV: &str = "FA_VECTOR_BACKEND";
/// Points per upsert request.
pub const UPSERT_CHUNK: usize = 1000;

/// Qdrant URL clients get by default (`None` = Helix vector indexes). Set by
/// [`configure_vectors`], else resolved once from the environment.
static DEFAULT_URL: RwLock<Option<Option<String>>> = RwLock::new(None);

fn env_backend(configured: Option<String>) -> Option<String> {
    let url = std::env::var(URL_ENV).ok().filter(|u| !u.is_empty()).or(configured);
    match std::env::var(BACKEND_ENV).ok().as_deref() {
        Some("helix") => None,
        _ => url,
    }
}

/// Select the vector backend of every [`HubClient`] created afterwards in this process from the
/// platform config (`[qdrant]`), with the environment overrides. Returns the Qdrant URL in use,
/// `None` for Helix.
pub fn configure_vectors(cfg: &fa_core::config::PlatformConfig) -> Option<String> {
    let configured = (cfg.qdrant.vector_backend == "qdrant").then(|| cfg.qdrant.url.clone());
    let url = env_backend(configured);
    *DEFAULT_URL.write().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(url.clone());
    url
}

/// The process default (see [`configure_vectors`]).
pub(crate) fn default_url() -> Option<String> {
    if let Some(u) = DEFAULT_URL.read().unwrap_or_else(std::sync::PoisonError::into_inner).as_ref() {
        return u.clone();
    }
    let u = env_backend(None);
    *DEFAULT_URL.write().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(u.clone());
    u
}

/// FNV-1a 64 over the key parts (separated by `0x1f`).
pub fn point_id(parts: &[&str]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            h ^= 0x1f;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        for b in p.bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// Point id of a latent row (see the module docs).
pub fn latent_point_id(l: &NewLatent) -> u64 {
    if l.kind == "centroid" || l.kind == AGGREGATE_KIND {
        point_id(&["latent", &l.expert_space_id, &l.kind, &l.ref_id])
    } else {
        point_id(&["latent", &l.expert_space_id, &l.kind, &l.ref_id, &l.instrument_id, &l.ts_ms.to_string()])
    }
}

pub fn latent_collection(dim: usize) -> String {
    format!("latent_d{dim}")
}

/// Collection of a `NewsDoc` / `MediaSegment` embedding of `dim`.
pub fn embedding_collection(label: &str, dim: usize) -> Result<String> {
    match label {
        node::NEWS_DOC => Ok(format!("news_d{dim}")),
        node::MEDIA_SEGMENT => Ok(format!("media_d{dim}")),
        other => Err(HubError::BadParam(format!("`{other}` is not an embedding label"))),
    }
}

fn norm(v: &[f32]) -> f64 {
    v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt()
}

fn eq(key: &str, value: &str) -> Json {
    json!({ "key": key, "match": { "value": value } })
}

fn ts_range(gte: Option<i64>, lte: Option<i64>, lt: Option<i64>) -> Option<Json> {
    let mut r = Map::new();
    if let Some(t) = gte {
        r.insert("gte".into(), t.into());
    }
    if let Some(t) = lte {
        r.insert("lte".into(), t.into());
    }
    if let Some(t) = lt {
        r.insert("lt".into(), t.into());
    }
    (!r.is_empty()).then(|| json!({ "key": "ts", "range": r }))
}

/// Qdrant filter of a k-NN in one expert space.
pub fn knn_filter(expert_space_id: &str, f: &KnnFilter) -> Json {
    let mut must = vec![eq(LATENT_TENANT, expert_space_id)];
    if let Some(k) = f.kind.as_deref().filter(|k| !k.is_empty()) {
        must.push(eq("kind", k));
    }
    if let Some(i) = f.instrument_id.as_deref().filter(|i| !i.is_empty()) {
        must.push(eq("instrument_id", i));
    }
    must.extend(ts_range(f.since_ms, f.as_of_ms, None));
    let mut filter = json!({ "must": must });
    if !f.exclude_ref_ids.is_empty() {
        filter["must_not"] = json!([{ "key": "ref_id", "match": { "any": f.exclude_ref_ids } }]);
    }
    filter
}

/// Filter of the space-scoped scans (`space_latents`, `space_latents_as_of`, counts).
pub fn space_filter(expert_space_id: &str, kind: Option<&str>, since_ms: Option<i64>, as_of_ms: Option<i64>) -> Json {
    let mut must = vec![eq(LATENT_TENANT, expert_space_id)];
    if let Some(k) = kind.filter(|k| !k.is_empty()) {
        must.push(eq("kind", k));
    }
    must.extend(ts_range(since_ms, as_of_ms, None));
    json!({ "must": must })
}

/// The Qdrant point of a latent row.
pub fn latent_point(l: &NewLatent) -> Json {
    json!({
        "id": latent_point_id(l),
        "vector": l.vector,
        "payload": {
            LATENT_TENANT: l.expert_space_id, "kind": l.kind, "ref_id": l.ref_id,
            "instrument_id": l.instrument_id, "ts": l.ts_ms, "uncertainty": l.uncertainty,
            "norm": norm(&l.vector),
        },
    })
}

/// A Qdrant point (search hit or scroll row) in the Helix row shape: `$id`, `$distance` (when
/// scored), the payload with `ts` rendered as RFC 3339, and the raw vector in `vec<dim>`.
pub fn point_to_row(p: &Json) -> Json {
    let mut row = Map::new();
    row.insert("$id".into(), p.get("id").cloned().unwrap_or(Json::Null));
    if let Some(score) = p.get("score").and_then(Json::as_f64) {
        row.insert("$distance".into(), json!(1.0 - score));
    }
    let payload = p.get("payload").and_then(Json::as_object).cloned().unwrap_or_default();
    let scale = payload.get("norm").and_then(Json::as_f64).unwrap_or(1.0);
    for (k, v) in payload {
        match k.as_str() {
            "norm" => {}
            "ts" => {
                row.insert(k, v.as_i64().map_or(v.clone(), |ms| Json::String(ms_rfc3339(ms))));
            }
            _ => {
                row.insert(k, v);
            }
        }
    }
    if let Some(vec) = p.get("vector").and_then(Json::as_array) {
        let raw: Vec<Json> = vec.iter().map(|x| json!((x.as_f64().unwrap_or(0.0) * scale) as f32)).collect();
        row.insert(latent_vec_property(raw.len()), Json::Array(raw));
    }
    Json::Object(row)
}

/// REST client for one Qdrant server. Shared (behind an `Arc`) by clones of a [`HubClient`].
#[derive(Debug)]
pub struct QdrantVectors {
    base: String,
    http: reqwest::Client,
    /// Collections known to exist with their payload indexes.
    ensured: Mutex<HashSet<String>>,
}

impl QdrantVectors {
    pub fn new(base_url: impl Into<String>, timeout: Option<Duration>) -> Self {
        let http = match timeout {
            Some(t) => reqwest::Client::builder()
                .timeout(t)
                .connect_timeout(t.min(Duration::from_secs(2)))
                .build()
                .unwrap_or_default(),
            None => reqwest::Client::new(),
        };
        Self { base: base_url.into().trim_end_matches('/').to_owned(), http, ensured: Mutex::default() }
    }

    pub fn base_url(&self) -> &str {
        &self.base
    }

    /// Whether this client created / verified collection `name`.
    pub fn is_ensured(&self, name: &str) -> bool {
        lock(&self.ensured).contains(name)
    }

    /// Forget the ensured-collection cache (e.g. after the server was wiped).
    pub fn forget_collections(&self) {
        lock(&self.ensured).clear();
    }

    /// One request; returns the `result` field. Non-2xx → [`HubError::Qdrant`].
    async fn call(&self, method: reqwest::Method, path: &str, body: Option<Json>) -> Result<Json> {
        let mut req = self.http.request(method, format!("{}{path}", self.base));
        if let Some(b) = body {
            req = req.json(&b);
        }
        let resp = req.send().await?;
        let status = resp.status().as_u16();
        let text = resp.text().await?;
        let v: Json = serde_json::from_str(&text).unwrap_or(Json::String(text));
        if !(200..300).contains(&status) {
            let msg = v.pointer("/status/error").and_then(Json::as_str).map_or_else(|| v.to_string(), str::to_owned);
            return Err(HubError::Qdrant { status, msg });
        }
        Ok(v.get("result").cloned().unwrap_or(Json::Null))
    }

    /// `GET /readyz` answered 2xx.
    pub async fn ready(&self) -> Result<bool> {
        let resp = self.http.get(format!("{}/readyz", self.base)).send().await?;
        Ok(resp.status().is_success())
    }

    /// Create `name` (cosine, `dim`) with the payload indexes unless known; cached per client.
    pub async fn ensure_collection(&self, name: &str, dim: usize, latent: bool) -> Result<()> {
        if dim == 0 {
            return Err(HubError::InvalidVector("dimension 0".into()));
        }
        if lock(&self.ensured).contains(name) {
            return Ok(());
        }
        let exists = self.call(reqwest::Method::GET, &format!("/collections/{name}/exists"), None).await?;
        if !exists["exists"].as_bool().unwrap_or(false) {
            let body = json!({ "vectors": { "size": dim, "distance": "Cosine" } });
            match self.call(reqwest::Method::PUT, &format!("/collections/{name}"), Some(body)).await {
                Ok(_) => {}
                Err(HubError::Qdrant { msg, .. }) if msg.contains("already exists") => {}
                Err(e) => return Err(e),
            }
        }
        let mut indexes = vec![("ts", json!({ "type": "integer", "lookup": false, "range": true }))];
        if latent {
            indexes.push((LATENT_TENANT, json!({ "type": "keyword", "is_tenant": true })));
            indexes.extend(["kind", "instrument_id", "ref_id"].map(|k| (k, json!("keyword"))));
        }
        for (field, schema) in indexes {
            let body = json!({ "field_name": field, "field_schema": schema });
            self.call(reqwest::Method::PUT, &format!("/collections/{name}/index?wait=true"), Some(body)).await?;
        }
        lock(&self.ensured).insert(name.to_owned());
        Ok(())
    }

    /// Upsert points (chunked, `wait=true` so reads see them).
    pub async fn upsert(&self, collection: &str, points: Vec<Json>) -> Result<()> {
        for chunk in points.chunks(UPSERT_CHUNK) {
            let body = json!({ "points": chunk });
            self.call(reqwest::Method::PUT, &format!("/collections/{collection}/points?wait=true"), Some(body)).await?;
        }
        Ok(())
    }

    /// Top-k by cosine; `None` when the collection does not exist (nothing stored yet).
    pub async fn query(
        &self,
        collection: &str,
        vector: &[f32],
        k: usize,
        filter: Option<Json>,
        max_distance: Option<f64>,
    ) -> Result<Option<Vec<Json>>> {
        let mut body = json!({ "query": vector, "limit": k, "with_payload": true });
        if let Some(f) = filter {
            body["filter"] = f;
        }
        if let Some(d) = max_distance {
            body["score_threshold"] = json!(1.0 - d);
        }
        match self.call(reqwest::Method::POST, &format!("/collections/{collection}/points/query"), Some(body)).await {
            Ok(r) => Ok(Some(r["points"].as_array().cloned().unwrap_or_default())),
            Err(HubError::Qdrant { status: 404, .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Rows matching `filter` ordered by `ts` (newest first when `desc`), at most `limit`, with
    /// vectors. Empty when the collection does not exist.
    pub async fn scroll_by_ts(&self, collection: &str, filter: Json, limit: usize, desc: bool) -> Result<Vec<Json>> {
        let body = json!({
            "filter": filter, "limit": limit.max(1), "with_payload": true, "with_vector": true,
            "order_by": { "key": "ts", "direction": if desc { "desc" } else { "asc" } },
        });
        match self.call(reqwest::Method::POST, &format!("/collections/{collection}/points/scroll"), Some(body)).await {
            Ok(r) => Ok(r["points"].as_array().cloned().unwrap_or_default()),
            Err(HubError::Qdrant { status: 404, .. }) => Ok(Vec::new()),
            Err(e) => Err(e),
        }
    }

    /// Exact count of the points matching `filter` (0 when the collection does not exist).
    pub async fn count(&self, collection: &str, filter: Json) -> Result<u64> {
        let body = json!({ "filter": filter, "exact": true });
        match self.call(reqwest::Method::POST, &format!("/collections/{collection}/points/count"), Some(body)).await {
            Ok(r) => Ok(r["count"].as_u64().unwrap_or(0)),
            Err(HubError::Qdrant { status: 404, .. }) => Ok(0),
            Err(e) => Err(e),
        }
    }

    /// Delete by ids or by filter (`selector` = `{"points": [...]}` or `{"filter": ...}`); a
    /// missing collection has nothing to delete.
    pub async fn delete(&self, collection: &str, selector: Json) -> Result<()> {
        match self
            .call(reqwest::Method::POST, &format!("/collections/{collection}/points/delete?wait=true"), Some(selector))
            .await
        {
            Ok(_) | Err(HubError::Qdrant { status: 404, .. }) => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Merge `payload` into one point.
    pub async fn set_payload(&self, collection: &str, id: u64, payload: Json) -> Result<()> {
        let body = json!({ "payload": payload, "points": [id] });
        self.call(reqwest::Method::POST, &format!("/collections/{collection}/points/payload?wait=true"), Some(body))
            .await
            .map(drop)
    }

    /// Every collection name (for backups and diagnostics).
    pub async fn collections(&self) -> Result<Vec<String>> {
        let r = self.call(reqwest::Method::GET, "/collections", None).await?;
        Ok(r["collections"]
            .as_array()
            .map(|a| a.iter().filter_map(|c| c["name"].as_str().map(str::to_owned)).collect())
            .unwrap_or_default())
    }
}

fn rows<T: serde::de::DeserializeOwned>(points: &[Json]) -> Result<Vec<T>> {
    points.iter().map(|p| serde_json::from_value(point_to_row(p)).map_err(HubError::from)).collect()
}

/// The latent operations of [`HubClient`] on Qdrant (called by the public methods when a
/// [`QdrantVectors`] is attached).
impl HubClient {
    pub(crate) async fn q_add_latents(&self, q: &QdrantVectors, latents: &[NewLatent]) -> Result<Vec<u64>> {
        let mut by_dim: BTreeMap<usize, Vec<Json>> = BTreeMap::new();
        for l in latents {
            validate_vector(&l.vector)?;
            if l.expert_space_id.is_empty() {
                return Err(HubError::BadParam("latent without expert_space_id (index tenant)".into()));
            }
            by_dim.entry(l.vector.len()).or_default().push(latent_point(l));
        }
        for (dim, points) in by_dim {
            let c = latent_collection(dim);
            q.ensure_collection(&c, dim, true).await?;
            q.upsert(&c, points).await?;
        }
        Ok(latents.iter().map(latent_point_id).collect())
    }

    pub(crate) async fn q_knn(
        &self,
        q: &QdrantVectors,
        expert_space_id: &str,
        vector: &[f32],
        k: usize,
        filter: &KnnFilter,
    ) -> Result<Vec<LatentHit>> {
        let f = knn_filter(expert_space_id, filter);
        let Some(points) = q.query(&latent_collection(vector.len()), vector, k, Some(f), filter.max_distance).await?
        else {
            return Ok(Vec::new());
        };
        let mut hits: Vec<LatentHit> = rows(&points)?;
        if let Some(max) = filter.max_distance {
            hits.retain(|h| h.distance <= max + 1e-6);
        }
        hits.truncate(k);
        Ok(hits)
    }

    /// Collections holding a space's latents: the one of its registered `dim`, or — for a space
    /// not (or no longer) registered in the graph — every `latent_d*` collection.
    async fn q_space_collections(&self, q: &QdrantVectors, expert_space_id: &str) -> Result<Vec<String>> {
        match self.space_dim(expert_space_id).await {
            Ok(d) => Ok(vec![latent_collection(d as usize)]),
            Err(HubError::NotFound { .. }) => {
                Ok(q.collections().await?.into_iter().filter(|c| c.starts_with("latent_d")).collect())
            }
            Err(e) => Err(e),
        }
    }

    pub(crate) async fn q_space_latents(
        &self,
        q: &QdrantVectors,
        expert_space_id: &str,
        kind: Option<&str>,
        since_ms: Option<i64>,
        as_of_ms: Option<i64>,
        limit: usize,
    ) -> Result<Vec<LatentRow>> {
        let f = space_filter(expert_space_id, kind, since_ms, as_of_ms);
        let mut out: Vec<LatentRow> = Vec::new();
        for c in self.q_space_collections(q, expert_space_id).await? {
            out.extend(rows(&q.scroll_by_ts(&c, f.clone(), limit, true).await?)?);
        }
        // ts renders as fixed-width RFC 3339, so the string order is the time order
        out.sort_by(|a, b| b.ts.cmp(&a.ts));
        out.truncate(limit.max(1));
        Ok(out)
    }

    pub(crate) async fn q_count_space_latents(&self, q: &QdrantVectors, expert_space_id: &str) -> Result<u64> {
        let mut n = 0;
        for c in self.q_space_collections(q, expert_space_id).await? {
            n += q.count(&c, space_filter(expert_space_id, None, None, None)).await?;
        }
        Ok(n)
    }

    /// Drop the centroid row(s) of a region in the collection of `dim`.
    pub(crate) async fn q_drop_centroid(&self, q: &QdrantVectors, dim: usize, region_id: &str) -> Result<()> {
        let f = json!({ "must": [eq("ref_id", region_id), eq("kind", "centroid")] });
        q.delete(&latent_collection(dim), json!({ "filter": f })).await
    }

    pub(crate) async fn q_add_embeddings(
        &self,
        q: &QdrantVectors,
        label: &'static str,
        id_key: &str,
        source_key: &str,
        rows: &[NewEmbedding],
    ) -> Result<Vec<u64>> {
        let Some(first) = rows.first() else { return Ok(Vec::new()) };
        let c = embedding_collection(label, first.vector.len())?;
        q.ensure_collection(&c, first.vector.len(), false).await?;
        let ids: Vec<u64> = rows.iter().map(|r| point_id(&[label, &r.id])).collect();
        let points = rows
            .iter()
            .zip(&ids)
            .map(|(r, id)| {
                json!({ "id": id, "vector": r.vector,
                        "payload": { id_key: r.id, source_key: r.source, "ts": r.ts_ms, "norm": norm(&r.vector) } })
            })
            .collect();
        q.upsert(&c, points).await?;
        Ok(ids)
    }

    pub(crate) async fn q_search_embeddings(
        &self,
        q: &QdrantVectors,
        label: &'static str,
        vector: &[f32],
        k: usize,
        as_of_ms: Option<i64>,
    ) -> Result<Vec<EmbeddingHit>> {
        let c = embedding_collection(label, vector.len())?;
        let filter = ts_range(None, as_of_ms, None).map(|r| json!({ "must": [r] }));
        let Some(points) = q.query(&c, vector, k, filter, None).await? else { return Ok(Vec::new()) };
        rows(&points)
    }

    /// One page of the observation fold (see [`crate::retention`]) on Qdrant: read the oldest
    /// [`PRUNE_PAGE`] observations before the cut-off and the aggregates of their regions, upsert
    /// the merged aggregates, then delete the folded observations. Not one transaction: a crash
    /// between the upsert and the delete folds that page twice on the next run.
    pub(crate) async fn q_prune_page(
        &self,
        q: &QdrantVectors,
        expert_space_id: &str,
        cutoff_ms: i64,
    ) -> Result<PruneReport> {
        let mut report = PruneReport::default();
        for c in self.q_space_collections(q, expert_space_id).await? {
            let r = self.q_prune_collection(q, &c, expert_space_id, cutoff_ms).await?;
            report.pruned += r.pruned;
            report.aggregates_created += r.aggregates_created;
            report.aggregates_updated += r.aggregates_updated;
        }
        Ok(report)
    }

    async fn q_prune_collection(
        &self,
        q: &QdrantVectors,
        c: &str,
        expert_space_id: &str,
        cutoff_ms: i64,
    ) -> Result<PruneReport> {
        let old = json!({ "must": [eq(LATENT_TENANT, expert_space_id), eq("kind", "observation"),
                                    ts_range(None, None, Some(cutoff_ms))] });
        let points = q.scroll_by_ts(c, old, PRUNE_PAGE, false).await?;
        if points.is_empty() {
            return Ok(PruneReport::default());
        }
        let obs: Vec<LatentRow> = rows(&points)?;
        let refs: Vec<&str> = obs.iter().map(|r| r.ref_id.as_str()).collect::<HashSet<_>>().into_iter().collect();
        let agg_filter = json!({ "must": [eq(LATENT_TENANT, expert_space_id), eq("kind", AGGREGATE_KIND),
                                          { "key": "ref_id", "match": { "any": refs } }] });
        let aggs: Vec<AggRow> = rows(&q.scroll_by_ts(c, agg_filter, refs.len(), true).await?)?;

        let mut report = PruneReport { pruned: obs.len() as u64, ..PruneReport::default() };
        let mut upserts = Vec::new();
        for w in plan_folds(&obs, &aggs, cutoff_ms) {
            let payload = json!({ "obs_count": w.n, "uncertainty": w.unc, "ts": w.ts_ms });
            match (w.prev, w.mean) {
                (Some(prev), None) => {
                    q.set_payload(c, prev.id, payload).await?;
                    report.aggregates_updated += 1;
                }
                (None, None) => {} // zero-norm mean of a new aggregate: not storable as a cosine point
                (prev, Some(mean)) => {
                    let l = NewLatent {
                        vector: mean,
                        expert_space_id: expert_space_id.to_owned(),
                        kind: AGGREGATE_KIND.into(),
                        ref_id: w.ref_id.to_owned(),
                        instrument_id: w.instrument_id.clone(),
                        ts_ms: w.ts_ms,
                        uncertainty: w.unc,
                    };
                    let mut p = latent_point(&l);
                    p["payload"]["obs_count"] = json!(w.n);
                    if let Some(a) = prev {
                        p["id"] = json!(a.id);
                        report.aggregates_updated += 1;
                    } else {
                        report.aggregates_created += 1;
                    }
                    upserts.push(p);
                }
            }
        }
        q.upsert(c, upserts).await?;
        let ids: Vec<u64> = obs.iter().map(|r| r.id).collect();
        q.delete(c, json!({ "points": ids })).await?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn latent(kind: &str, ts: i64) -> NewLatent {
        NewLatent {
            vector: vec![3.0, 4.0],
            expert_space_id: "sp".into(),
            kind: kind.into(),
            ref_id: "r1".into(),
            instrument_id: "i".into(),
            ts_ms: ts,
            uncertainty: 0.5,
        }
    }

    #[test]
    fn point_ids_are_stable_and_keyed_by_business_id() {
        assert_eq!(point_id(&[]), 0xcbf2_9ce4_8422_2325);
        assert_eq!(point_id(&["a"]), 0xaf63_dc4c_8601_ec8c, "FNV-1a reference value of `a`");
        assert_ne!(point_id(&["ab", "c"]), point_id(&["a", "bc"]), "parts are separated");
        assert_eq!(latent_point_id(&latent("observation", 1)), latent_point_id(&latent("observation", 1)));
        assert_ne!(latent_point_id(&latent("observation", 1)), latent_point_id(&latent("observation", 2)));
        assert_eq!(latent_point_id(&latent("centroid", 1)), latent_point_id(&latent("centroid", 2)), "one per region");
        assert_eq!(latent_point_id(&latent(AGGREGATE_KIND, 1)), latent_point_id(&latent(AGGREGATE_KIND, 9)));
        assert_eq!(latent_collection(8), "latent_d8");
        assert_eq!(embedding_collection(node::NEWS_DOC, 4).unwrap(), "news_d4");
        assert_eq!(embedding_collection(node::MEDIA_SEGMENT, 4).unwrap(), "media_d4");
        assert!(embedding_collection(node::LATENT_VEC, 4).is_err());
    }

    #[test]
    fn knn_filter_runs_inside_the_index() {
        assert_eq!(knn_filter("sp", &KnnFilter::default()), json!({ "must": [eq(LATENT_TENANT, "sp")] }));
        let f = KnnFilter {
            kind: Some("window".into()),
            instrument_id: Some("BTC".into()),
            as_of_ms: Some(2_000),
            since_ms: Some(1_000),
            exclude_ref_ids: vec!["w0".into(), "w1".into()],
            max_distance: Some(0.1),
            overfetch: Some(9),
        };
        assert_eq!(
            knn_filter("sp", &f),
            json!({
                "must": [eq(LATENT_TENANT, "sp"), eq("kind", "window"), eq("instrument_id", "BTC"),
                         { "key": "ts", "range": { "gte": 1_000, "lte": 2_000 } }],
                "must_not": [{ "key": "ref_id", "match": { "any": ["w0", "w1"] } }],
            })
        );
        assert_eq!(
            space_filter("sp", Some(""), None, Some(5)),
            json!({ "must": [eq(LATENT_TENANT, "sp"), { "key": "ts", "range": { "lte": 5 } }] })
        );
    }

    #[test]
    fn points_round_trip_to_helix_rows() {
        let l = latent("observation", 1_762_620_000_000);
        let p = latent_point(&l);
        assert_eq!(p["payload"]["norm"], json!(5.0));
        // Qdrant returns the normalised vector and a cosine score.
        let hit = json!({ "id": p["id"], "score": 0.75, "payload": p["payload"], "vector": [0.6, 0.8] });
        let row = point_to_row(&hit);
        assert_eq!(row["ts"], json!("2025-11-08T16:40:00.000Z"));
        assert!(row.get("norm").is_none());
        let h: LatentHit = serde_json::from_value(row.clone()).unwrap();
        assert_eq!((h.id, h.kind.as_str(), h.ref_id.as_str()), (latent_point_id(&l), "observation", "r1"));
        assert!((h.distance - 0.25).abs() < 1e-12);
        let r: LatentRow = serde_json::from_value(row).unwrap();
        let v = r.vector().unwrap();
        assert!((v[0] - 3.0).abs() < 1e-5 && (v[1] - 4.0).abs() < 1e-5, "raw vector restored: {v:?}");
    }
}
