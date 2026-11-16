//! Dgraph HTTP/JSON transport (ADR 0011, D-110): the graph store that replaces HelixDB and the
//! Postgres `graph` schema.
//!
//! Dgraph speaks DQL over four endpoints on alpha's HTTP port — `/alter` (schema), `/query`,
//! `/mutate` and `/commit`. We talk to them with `reqwest` + `serde_json` exactly as
//! [`crate::qdrant`] does, and for the same reason: the Rust gRPC clients are unmaintained
//! (`dgraph-tonic` last published 2023-03), so a thin HTTP layer we own is the smaller liability.
//! `examples/graph_bench.rs` (ADR 0010) drives Dgraph the same way.
//!
//! Atomicity: every call here commits on its own (`commitNow=true`). Read-modify-write is expressed
//! as an **upsert block** — one request carrying a query and the mutations conditioned on it —
//! which is Dgraph's equivalent of the `ON CONFLICT … DO UPDATE` that the Postgres backend used for
//! every idempotent writer. Multi-request transactions (`/commit` with a start timestamp) are not
//! needed by `GraphStore` and are deliberately not wired up.
use std::collections::BTreeMap;

use serde_json::{Map, Value as Json, json};

use crate::{HubError, Result};

/// Uniqueness is enforced by upsert conditions rather than by server-side constraints, so the
/// backend raises this itself. The shape matches what the Postgres backend produced from SQLSTATE
/// 23505, which `examples/graph_migrate.rs` and the conformance suite both branch on.
pub fn unique_violation(what: &str) -> HubError {
    HubError::Server {
        status: 409,
        code: "unique_constraint_violation".into(),
        msg: format!("{what} already exists"),
        retryable: Some(false),
    }
}

/// A Dgraph alpha endpoint. Cheap to clone (the inner `reqwest::Client` is an `Arc`).
#[derive(Clone, Debug)]
pub struct DgraphClient {
    http: reqwest::Client,
    base: String,
}

impl DgraphClient {
    /// `url` is alpha's HTTP base, e.g. `http://dgraph-alpha:8080`; a trailing slash is ignored.
    pub fn new(url: &str) -> Self {
        Self { http: reqwest::Client::new(), base: url.trim_end_matches('/').to_owned() }
    }

    /// `FA_DGRAPH_URL`, else the compose default.
    pub fn from_env() -> Self {
        let url = std::env::var("FA_DGRAPH_URL").ok().filter(|u| !u.is_empty());
        Self::new(url.as_deref().unwrap_or("http://127.0.0.1:18180"))
    }

    pub fn url(&self) -> &str {
        &self.base
    }

    /// POST, then fold Dgraph's `errors` array into [`HubError::Server`]. Dgraph answers 200 with
    /// an `errors` member for most query faults, so the status alone is not enough.
    async fn post(&self, path: &str, content_type: &str, body: String) -> Result<Json> {
        let resp = self
            .http
            .post(format!("{}{path}", self.base))
            .header("Content-Type", content_type)
            .body(body)
            .send()
            .await?;
        let status = resp.status().as_u16();
        let text = resp.text().await?;
        let v: Json = serde_json::from_str(&text).unwrap_or_else(|_| json!({ "errors": [{ "message": text }] }));
        if let Some(errs) = v.get("errors").and_then(Json::as_array).filter(|e| !e.is_empty()) {
            let msg = errs
                .iter()
                .filter_map(|e| e.get("message").and_then(Json::as_str))
                .collect::<Vec<_>>()
                .join("; ");
            let code = errs
                .first()
                .and_then(|e| e.pointer("/extensions/code"))
                .and_then(Json::as_str)
                .unwrap_or("dgraph_error")
                .to_owned();
            // Dgraph retries are safe for transport/leader-election faults; `is_transient` in
            // buffer.rs keys off `status >= 500`, so map those two classes onto 503.
            let status = if status >= 500 || msg.contains("Please retry") || msg.contains("not ready") {
                503
            } else if status >= 400 {
                status
            } else {
                400
            };
            return Err(HubError::Server { status, code, msg, retryable: Some(status == 503) });
        }
        Ok(v)
    }

    /// Apply a DQL schema (`/alter`). Idempotent: re-applying the same predicates is a no-op.
    pub async fn alter(&self, schema: &str) -> Result<()> {
        self.post("/alter", "application/dql", schema.to_owned()).await.map(|_| ())
    }

    /// Run a DQL read block and return its `data` object.
    pub async fn query(&self, dql: &str) -> Result<Json> {
        let v = self.post("/query", "application/dql", dql.to_owned()).await?;
        Ok(v.get("data").cloned().unwrap_or(Json::Null))
    }

    /// Run a DQL read block with GraphQL± variables (`$name` → value, values are always strings).
    pub async fn query_vars(&self, dql: &str, vars: &BTreeMap<String, String>) -> Result<Json> {
        let body = json!({ "query": dql, "variables": vars }).to_string();
        let v = self.post("/query", "application/json", body).await?;
        Ok(v.get("data").cloned().unwrap_or(Json::Null))
    }

    /// Set nodes, committing immediately. Returns the blank-node → uid map Dgraph assigned.
    pub async fn mutate(&self, set: Vec<Json>) -> Result<BTreeMap<String, String>> {
        if set.is_empty() {
            return Ok(BTreeMap::new());
        }
        let v = self.post("/mutate?commitNow=true", "application/json", json!({ "set": set }).to_string()).await?;
        Ok(uids(&v))
    }

    /// Delete nodes or predicates (`{"uid": …}` deletes the whole node).
    pub async fn delete(&self, del: Vec<Json>) -> Result<()> {
        if del.is_empty() {
            return Ok(());
        }
        self.post("/mutate?commitNow=true", "application/json", json!({ "delete": del }).to_string()).await.map(|_| ())
    }

    /// An upsert block: `query` binds variables, then `set` / `del` run against them atomically.
    /// This is how every get-or-create and update-by-business-key is expressed — the replacement
    /// for `INSERT … ON CONFLICT … DO UPDATE`.
    pub async fn upsert(&self, query: &str, set: Vec<Json>, del: Vec<Json>) -> Result<BTreeMap<String, String>> {
        let mut body = Map::new();
        body.insert("query".into(), json!(query));
        if !set.is_empty() {
            body.insert("set".into(), json!(set));
        }
        if !del.is_empty() {
            body.insert("delete".into(), json!(del));
        }
        let v = self.post("/mutate?commitNow=true", "application/json", Json::Object(body).to_string()).await?;
        Ok(uids(&v))
    }

    /// `GET /health` — alpha reports `"healthy"` once it has joined the group.
    pub async fn healthy(&self) -> bool {
        let Ok(r) = self.http.get(format!("{}/health", self.base)).send().await else { return false };
        r.status().is_success()
    }
}

/// The `data.uids` map of a mutation response.
fn uids(v: &Json) -> BTreeMap<String, String> {
    v.pointer("/data/uids")
        .and_then(Json::as_object)
        .map(|m| m.iter().filter_map(|(k, x)| x.as_str().map(|s| (k.clone(), s.to_owned()))).collect())
        .unwrap_or_default()
}

/// Dgraph returns uids as `"0x1a"`; the `GraphStore` API hands them to callers as `u64` node ids,
/// which is exactly what Dgraph allocates them as.
pub fn uid_to_u64(uid: &str) -> Option<u64> {
    u64::from_str_radix(uid.trim_start_matches("0x"), 16).ok()
}

/// The inverse of [`uid_to_u64`], for addressing a known node in a mutation or query.
pub fn u64_to_uid(id: u64) -> String {
    format!("0x{id:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uid_round_trips() {
        assert_eq!(uid_to_u64("0x1a"), Some(26));
        assert_eq!(uid_to_u64(&u64_to_uid(123_456)), Some(123_456));
        assert_eq!(uid_to_u64("nonsense"), None);
    }

    #[test]
    fn unique_violation_matches_the_postgres_contract() {
        let HubError::Server { status, code, retryable, .. } = unique_violation("snapshot `s1`") else {
            panic!("wrong variant");
        };
        assert_eq!((status, code.as_str(), retryable), (409, "unique_constraint_violation", Some(false)));
    }

    #[test]
    fn uids_are_read_from_the_mutation_response() {
        let v = json!({"data": {"code": "Success", "uids": {"blank-0": "0x2a"}}});
        assert_eq!(uids(&v).get("blank-0").map(String::as_str), Some("0x2a"));
        assert!(uids(&json!({"data": {}})).is_empty());
    }

    #[test]
    fn base_url_drops_a_trailing_slash() {
        assert_eq!(DgraphClient::new("http://dgraph-alpha:8080/").url(), "http://dgraph-alpha:8080");
    }
}
