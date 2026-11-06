//! HelixDB client: typed calls to the HTTP query endpoints `POST /<query_name>` (spec 20, D-012).
use serde::{Serialize, de::DeserializeOwned};

#[derive(Debug, thiserror::Error)]
pub enum HubError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("helix returned {status}: {body}")]
    Status { status: u16, body: String },
}

/// Thin client over HelixDB's query endpoints. Query names mirror `helix/queries.hx`.
#[derive(Clone)]
pub struct HubClient {
    base: String,
    http: reqwest::Client,
}

impl HubClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self { base: base_url.into().trim_end_matches('/').to_owned(), http: reqwest::Client::new() }
    }

    pub fn query_url(&self, query: &str) -> String {
        format!("{}/{}", self.base, query)
    }

    /// Calls a named query with JSON arguments and decodes the JSON result.
    pub async fn query<A: Serialize + ?Sized, R: DeserializeOwned>(&self, query: &str, args: &A) -> Result<R, HubError> {
        let resp = self.http.post(self.query_url(query)).json(args).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(HubError::Status { status: status.as_u16(), body });
        }
        Ok(resp.json().await?)
    }
}

/// Names of queries defined in `helix/queries.hx`.
pub mod queries {
    pub const ADD_INSTRUMENT: &str = "add_instrument";
    pub const ADD_SNAPSHOT: &str = "add_snapshot";
    pub const ADD_REGION: &str = "add_region";
    pub const UPSERT_TRANSITION: &str = "upsert_transition";
    pub const ADD_GENOME: &str = "add_genome";
    pub const LINK_PARENT: &str = "link_parent";
    pub const SEARCH_WINDOWS: &str = "search_windows";
    pub const REGION_GRAPH: &str = "region_graph";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_url_strips_trailing_slash() {
        let c = HubClient::new("http://127.0.0.1:6969/");
        assert_eq!(c.query_url(queries::ADD_REGION), "http://127.0.0.1:6969/add_region");
    }
}
