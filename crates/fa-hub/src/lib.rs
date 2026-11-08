//! HelixDB client for the standalone server (`ghcr.io/helixdb/helixdb`, spec 20 §5.2, ADR 0002,
//! D-033): a typed Rust layer over `POST /v2/query`, the operation-tree contract shared by the
//! Helix SDKs.
//!
//! - [`ast`]: builder for operation trees ([`Op`], [`Pred`], [`Batch`], [`IndexSpec`]).
//! - [`schema`]: labels, property names and index DDL mirroring `helix/schema.hx`.
//! - [`queries`]: typed functions on [`HubClient`] implementing every query of
//!   `helix/queries.hx` (same names, snake_case), plus the graph reads spec 50 §4.4/§4.5 need.
//!
//! The server is schemaless: labels and properties are inferred on write. What must be
//! declared are indexes — unique/equality lookups and vector indexes — which
//! [`HubClient::ensure_schema`] and [`HubClient::ensure_latent_index`] create idempotently.
//! Index builds are asynchronous; wait for them with [`HubClient::wait_for_index`] (or the
//! `*_ready` helpers) before relying on uniqueness or vector search.

pub mod ast;
pub mod queries;
pub mod schema;

use std::time::{Duration, Instant};

use serde::{Deserialize, de::DeserializeOwned};
use serde_json::Value as Json;

pub use ast::{Batch, Cond, EdgeRef, IndexSpec, Input, Metric, NodeRef, Op, ParamType, Pred, Val, param};
pub use queries::*;

/// Default base URL (D-014: host port 6969 maps to the container's 8080).
pub const DEFAULT_URL: &str = "http://127.0.0.1:6969";
/// Environment variable overriding the base URL (also gates the live integration tests).
pub const URL_ENV: &str = "FA_HELIX_URL";

#[derive(Debug, thiserror::Error)]
pub enum HubError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    /// A non-2xx response. `code` is the server's static error code (`unique_constraint_violation`,
    /// `index_not_found`, `invalid_query_json`, ...); `retryable` is its hint when present.
    #[error("helix {status} {code}: {msg}")]
    Server { status: u16, code: String, msg: String, retryable: Option<bool> },
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("response has no return variable `{0}`")]
    MissingReturn(String),
    #[error("unexpected response shape for `{name}`: {detail}")]
    Shape { name: String, detail: String },
    #[error("index operation {operation_id} blocked ({blocker}): {message}")]
    IndexBlocked { operation_id: String, blocker: String, message: String },
    #[error("index operation {0} aborted")]
    IndexAborted(String),
    #[error("timed out waiting for index operation {0}")]
    IndexTimeout(String),
}

impl HubError {
    pub fn server_code(&self) -> Option<&str> {
        match self {
            HubError::Server { code, .. } => Some(code),
            _ => None,
        }
    }
    /// `true` for `unique_constraint_violation` (HTTP 409).
    pub fn is_unique_violation(&self) -> bool {
        self.server_code() == Some("unique_constraint_violation")
    }
}

pub type Result<T, E = HubError> = std::result::Result<T, E>;

/// `GET /healthz` and `/readyz` body.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Health {
    pub ready: bool,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub index_runtime: Option<String>,
    #[serde(default)]
    pub blocked_index_entity_count: Option<u64>,
}

/// Receipt of `create_index` / `drop_index`. `kind` is `accepted` (with `operation_id`),
/// `already_active`, or `existing_operation`.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct IndexReceipt {
    pub kind: String,
    #[serde(default)]
    pub operation_id: Option<String>,
    #[serde(default)]
    pub index_id: Option<String>,
    #[serde(default)]
    pub generation: Option<String>,
}

impl IndexReceipt {
    /// The index is usable right away (no pending operation).
    pub fn is_active(&self) -> bool {
        self.kind == "already_active"
    }
}

/// `get_index_operation` row. `status` is `queued | running | blocked | succeeded | aborted`.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct IndexOperation {
    pub operation_id: String,
    pub status: String,
    #[serde(default)]
    pub stage: Option<String>,
    #[serde(default)]
    pub operation_kind: Option<String>,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub blocker_code: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
}

/// A `/v2/query` response: one JSON value per returned variable. Node/edge streams are arrays
/// of objects (`{"$id": n}` bare, or full rows after `value_map` / `edge_properties`); `count`
/// is a number, `exists` a bool, index DDL an [`IndexReceipt`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Response(serde_json::Map<String, Json>);

impl Response {
    pub fn get(&self, name: &str) -> Option<&Json> {
        self.0.get(name)
    }
    /// Deserialize one returned variable.
    pub fn take<T: DeserializeOwned>(&mut self, name: &str) -> Result<T> {
        let v = self.0.remove(name).ok_or_else(|| HubError::MissingReturn(name.to_owned()))?;
        Ok(serde_json::from_value(v)?)
    }
    /// Deserialize a stream variable that must hold exactly one row.
    pub fn take_one<T: DeserializeOwned>(&mut self, name: &str) -> Result<T> {
        let mut rows: Vec<T> = self.take(name)?;
        match rows.len() {
            1 => Ok(rows.pop().expect("len checked")),
            n => Err(HubError::Shape { name: name.to_owned(), detail: format!("expected 1 row, got {n}") }),
        }
    }
    /// Deserialize a stream variable that holds zero or one row.
    pub fn take_opt<T: DeserializeOwned>(&mut self, name: &str) -> Result<Option<T>> {
        let mut rows: Vec<T> = self.take(name)?;
        match rows.len() {
            0 => Ok(None),
            1 => Ok(rows.pop()),
            n => Err(HubError::Shape { name: name.to_owned(), detail: format!("expected ≤1 row, got {n}") }),
        }
    }
    pub fn into_inner(self) -> serde_json::Map<String, Json> {
        self.0
    }
}

/// Client for one HelixDB server.
#[derive(Clone, Debug)]
pub struct HubClient {
    base: String,
    http: reqwest::Client,
}

impl HubClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self { base: base_url.into().trim_end_matches('/').to_owned(), http: reqwest::Client::new() }
    }

    /// `FA_HELIX_URL` or [`DEFAULT_URL`].
    pub fn from_env() -> Self {
        Self::new(std::env::var(URL_ENV).unwrap_or_else(|_| DEFAULT_URL.to_owned()))
    }

    pub fn base_url(&self) -> &str {
        &self.base
    }

    pub fn query_url(&self) -> String {
        format!("{}/v2/query", self.base)
    }

    async fn get(&self, path: &str) -> Result<(u16, Json)> {
        let resp = self.http.get(format!("{}{path}", self.base)).send().await?;
        let status = resp.status().as_u16();
        let body = resp.text().await?;
        let json = serde_json::from_str(&body).unwrap_or(Json::String(body));
        Ok((status, json))
    }

    /// `GET /healthz`: liveness plus the index runtime state.
    pub async fn health(&self) -> Result<Health> {
        let (status, body) = self.get("/healthz").await?;
        if !(200..300).contains(&status) {
            return Err(server_error(status, body));
        }
        Ok(serde_json::from_value(body)?)
    }

    /// `GET /readyz`: `true` on 200, `false` on 503 (indexes still catching up).
    pub async fn ready(&self) -> Result<bool> {
        let (status, body) = self.get("/readyz").await?;
        match status {
            200..=299 => Ok(true),
            503 => Ok(false),
            _ => Err(server_error(status, body)),
        }
    }

    /// `GET /v2/diagnostics`: read-only setup checks (storage, memory, index lag, query hygiene).
    pub async fn diagnostics(&self) -> Result<Json> {
        let (status, body) = self.get("/v2/diagnostics").await?;
        if !(200..300).contains(&status) {
            return Err(server_error(status, body));
        }
        Ok(body)
    }

    /// `POST /v2/query` with a full batch.
    pub async fn batch(&self, batch: &Batch) -> Result<Response> {
        let resp = self.http.post(self.query_url()).json(&batch.to_json()).send().await?;
        let status = resp.status().as_u16();
        let body = resp.text().await?;
        if !(200..300).contains(&status) {
            let json = serde_json::from_str(&body).unwrap_or(Json::String(body));
            return Err(server_error(status, json));
        }
        match serde_json::from_str::<Json>(&body)? {
            Json::Object(map) => Ok(Response(map)),
            other => Err(HubError::Shape { name: "<response>".into(), detail: format!("not an object: {other}") }),
        }
    }

    /// Run a single operation tree (read or write batch chosen from the op) and return its value.
    pub async fn query(&self, op: Op) -> Result<Json> {
        let mut resp = self.batch(&Batch::for_op(&op).ret("r", op)).await?;
        resp.take("r")
    }

    /// Poll an index operation until it succeeds, or fail on `blocked`/`aborted`/timeout.
    pub async fn wait_for_index(&self, operation_id: &str, timeout: Duration) -> Result<IndexOperation> {
        let start = Instant::now();
        let mut delay = Duration::from_millis(25);
        loop {
            let op: IndexOperation = serde_json::from_value(self.query(Op::get_index_operation(operation_id)).await?)?;
            match op.status.as_str() {
                "succeeded" => return Ok(op),
                "aborted" => return Err(HubError::IndexAborted(operation_id.to_owned())),
                "blocked" => {
                    return Err(HubError::IndexBlocked {
                        operation_id: operation_id.to_owned(),
                        blocker: op.blocker_code.unwrap_or_default(),
                        message: op.message.unwrap_or_default(),
                    });
                }
                _ => {}
            }
            if start.elapsed() >= timeout {
                return Err(HubError::IndexTimeout(operation_id.to_owned()));
            }
            tokio::time::sleep(delay).await;
            delay = (delay * 2).min(Duration::from_millis(500));
        }
    }

    /// Wait for every `accepted` receipt.
    pub async fn wait_for_receipts(&self, receipts: &[IndexReceipt], timeout: Duration) -> Result<()> {
        for r in receipts {
            if let Some(id) = r.operation_id.as_deref() {
                self.wait_for_index(id, timeout).await?;
            }
        }
        Ok(())
    }

    /// Create indexes in one DDL batch (`if_not_exists`), returning one receipt per spec.
    pub async fn ensure_indexes(&self, specs: Vec<IndexSpec>) -> Result<Vec<IndexReceipt>> {
        let mut batch = Batch::write().named("ensure_indexes");
        let names: Vec<String> = (0..specs.len()).map(|i| format!("i{i}")).collect();
        for (name, spec) in names.iter().zip(specs) {
            batch = batch.ret(name, Op::create_index(spec, true));
        }
        let mut resp = self.batch_retrying(&batch, 8).await?;
        names.iter().map(|n| resp.take::<IndexReceipt>(n)).collect()
    }

    /// [`Self::batch`], retrying server errors flagged `retryable` (e.g. 409 `transaction_conflict`
    /// when several processes run DDL at once) with exponential backoff. Only for idempotent
    /// batches: a retried data write whose first attempt did commit would run twice.
    pub async fn batch_retrying(&self, batch: &Batch, max_attempts: u32) -> Result<Response> {
        let mut delay = Duration::from_millis(50);
        let mut attempt = 1;
        loop {
            match self.batch(batch).await {
                Err(HubError::Server { retryable: Some(true), .. }) if attempt < max_attempts => {
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(Duration::from_secs(2));
                    attempt += 1;
                }
                other => return other,
            }
        }
    }

    /// Idempotently create every static index of `helix/schema.hx` ([`schema::static_indexes`]).
    pub async fn ensure_schema(&self) -> Result<Vec<IndexReceipt>> {
        self.ensure_indexes(schema::static_indexes()).await
    }

    /// [`Self::ensure_schema`] and wait for the builds to activate.
    pub async fn ensure_schema_ready(&self, timeout: Duration) -> Result<()> {
        let receipts = self.ensure_schema().await?;
        self.wait_for_receipts(&receipts, timeout).await
    }

    /// Idempotently create the latent vector index for one dimension (partitioned by expert space).
    pub async fn ensure_latent_index(&self, dim: usize) -> Result<IndexReceipt> {
        let mut receipts = self.ensure_indexes(vec![schema::latent_vec_index(dim)]).await?;
        Ok(receipts.pop().expect("one spec, one receipt"))
    }

    /// [`Self::ensure_latent_index`] and wait for the build to activate.
    pub async fn ensure_latent_index_ready(&self, dim: usize, timeout: Duration) -> Result<()> {
        let r = self.ensure_latent_index(dim).await?;
        self.wait_for_receipts(&[r], timeout).await
    }
}

fn server_error(status: u16, body: Json) -> HubError {
    let code = body.get("error").and_then(Json::as_str).unwrap_or("unknown").to_owned();
    let msg = body.get("msg").and_then(Json::as_str).map(str::to_owned).unwrap_or_else(|| body.to_string());
    let retryable = body.get("retryable").and_then(Json::as_bool);
    HubError::Server { status, code, msg, retryable }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn query_url_strips_trailing_slash() {
        let c = HubClient::new("http://127.0.0.1:6969/");
        assert_eq!(c.query_url(), "http://127.0.0.1:6969/v2/query");
    }

    #[test]
    fn server_error_parses_envelope() {
        let e = server_error(409, json!({ "error": "unique_constraint_violation", "msg": "dup", "retryable": false }));
        assert!(e.is_unique_violation());
        assert!(matches!(e, HubError::Server { status: 409, retryable: Some(false), .. }));
        let e = server_error(502, Json::String("bad gateway".into()));
        assert_eq!(e.server_code(), Some("unknown"));
    }

    #[test]
    fn response_take_helpers() {
        let mut r = Response(json!({ "one": [{ "$id": 3 }], "none": [], "n": 7 }).as_object().unwrap().clone());
        #[derive(Deserialize)]
        struct Row {
            #[serde(rename = "$id")]
            id: u64,
        }
        assert_eq!(r.take_one::<Row>("one").unwrap().id, 3);
        assert!(r.take_opt::<Row>("none").unwrap().is_none());
        assert_eq!(r.take::<u64>("n").unwrap(), 7);
        assert!(matches!(r.take::<u64>("n"), Err(HubError::MissingReturn(_))));
    }

    #[test]
    fn receipts_and_operations_deserialize() {
        let r: IndexReceipt =
            serde_json::from_value(json!({ "generation": "1", "index_id": "1", "kind": "already_active" })).unwrap();
        assert!(r.is_active());
        let op: IndexOperation = serde_json::from_value(json!({
            "attempt": 3, "family": "vector", "generation": "1", "index_id": "2",
            "operation_id": "6993", "operation_kind": "build", "progress": { "entities": "1" },
            "stage": "activate", "status": "succeeded"
        }))
        .unwrap();
        assert_eq!(op.status, "succeeded");
    }
}
