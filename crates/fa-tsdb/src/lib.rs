//! VictoriaMetrics access (spec 20 §5.3): series naming, Prometheus-text import, range queries.
use std::collections::BTreeMap;

use serde::Deserialize;

#[derive(Debug, thiserror::Error)]
pub enum TsdbError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("victoriametrics returned {status}: {body}")]
    Status { status: u16, body: String },
    #[error("invalid metric name: {0}")]
    InvalidName(String),
    #[error("label {0} is not allowed (unbounded cardinality)")]
    ForbiddenLabel(String),
}

/// Labels allowed on market series; anything else is rejected to protect cardinality.
pub const ALLOWED_LABELS: &[&str] = &["inst", "venue", "tf", "src", "lvl", "feat_set"];

/// One sample to import.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub metric: String,
    pub labels: BTreeMap<String, String>,
    pub value: f64,
    pub ts_ms: i64,
}

impl Sample {
    pub fn new(metric: impl Into<String>, value: f64, ts_ms: i64) -> Self {
        Self { metric: metric.into(), labels: BTreeMap::new(), value, ts_ms }
    }

    pub fn label(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.labels.insert(k.into(), v.into());
        self
    }

    /// Validates the metric name (`fa_` prefix, `[a-z0-9_]`) and the label whitelist.
    pub fn validate(&self) -> Result<(), TsdbError> {
        let ok = self.metric.starts_with("fa_")
            && self.metric.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
        if !ok {
            return Err(TsdbError::InvalidName(self.metric.clone()));
        }
        if let Some(k) = self.labels.keys().find(|k| !ALLOWED_LABELS.contains(&k.as_str())) {
            return Err(TsdbError::ForbiddenLabel(k.clone()));
        }
        Ok(())
    }

    /// Prometheus exposition line: `name{k="v",...} value ts_ms`.
    pub fn to_line(&self) -> String {
        let mut s = self.metric.clone();
        if !self.labels.is_empty() {
            s.push('{');
            let parts: Vec<String> =
                self.labels.iter().map(|(k, v)| format!("{k}=\"{}\"", escape_label(v))).collect();
            s.push_str(&parts.join(","));
            s.push('}');
        }
        s.push_str(&format!(" {} {}", self.value, self.ts_ms));
        s
    }
}

fn escape_label(v: &str) -> String {
    v.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

/// Metric names (spec 20 §5.3).
pub mod metrics {
    pub const TRADE_PRICE: &str = "fa_trade_price";
    pub const TRADE_SIZE: &str = "fa_trade_size";
    pub const BAR_OPEN: &str = "fa_bar_open";
    pub const BAR_HIGH: &str = "fa_bar_high";
    pub const BAR_LOW: &str = "fa_bar_low";
    pub const BAR_CLOSE: &str = "fa_bar_close";
    pub const BAR_VOLUME: &str = "fa_bar_volume";

    pub fn feature(name: &str) -> String {
        format!("fa_feat_{name}")
    }
}

#[derive(Debug, Deserialize)]
struct QueryRangeResponse {
    data: QueryRangeData,
}

#[derive(Debug, Deserialize)]
struct QueryRangeData {
    result: Vec<RangeSeries>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RangeSeries {
    pub metric: BTreeMap<String, String>,
    /// `[unix_seconds, "value"]` pairs as returned by the Prometheus API.
    pub values: Vec<(f64, String)>,
}

#[derive(Clone)]
pub struct Tsdb {
    base: String,
    http: reqwest::Client,
}

impl Tsdb {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self { base: base_url.into().trim_end_matches('/').to_owned(), http: reqwest::Client::new() }
    }

    /// Imports samples via `/api/v1/import/prometheus`.
    pub async fn write(&self, samples: &[Sample]) -> Result<(), TsdbError> {
        let mut body = String::new();
        for s in samples {
            s.validate()?;
            body.push_str(&s.to_line());
            body.push('\n');
        }
        let resp = self.http.post(format!("{}/api/v1/import/prometheus", self.base)).body(body).send().await?;
        check(resp).await.map(|_| ())
    }

    /// Range query via `/api/v1/query_range` (`start`/`end` in unix seconds, `step` e.g. "1m").
    pub async fn query_range(&self, query: &str, start: i64, end: i64, step: &str) -> Result<Vec<RangeSeries>, TsdbError> {
        let resp = self
            .http
            .get(format!("{}/api/v1/query_range", self.base))
            .query(&[("query", query), ("start", &start.to_string()), ("end", &end.to_string()), ("step", step)])
            .send()
            .await?;
        let resp = check(resp).await?;
        let parsed: QueryRangeResponse = resp.json().await?;
        Ok(parsed.data.result)
    }
}

async fn check(resp: reqwest::Response) -> Result<reqwest::Response, TsdbError> {
    let status = resp.status();
    if status.is_success() {
        Ok(resp)
    } else {
        let body = resp.text().await.unwrap_or_default();
        Err(TsdbError::Status { status: status.as_u16(), body })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_format() {
        let s = Sample::new(metrics::TRADE_PRICE, 101.5, 1_700_000_000_000).label("venue", "sim").label("inst", "abc");
        assert_eq!(s.to_line(), r#"fa_trade_price{inst="abc",venue="sim"} 101.5 1700000000000"#);
        s.validate().unwrap();
    }

    #[test]
    fn escapes_label_values() {
        let s = Sample::new("fa_x", 1.0, 0).label("src", "a\"b");
        assert_eq!(s.to_line(), r#"fa_x{src="a\"b"} 1 0"#);
    }

    #[test]
    fn rejects_unbounded_labels_and_bad_names() {
        assert!(matches!(Sample::new("fa_x", 1.0, 0).label("trade_id", "1").validate(), Err(TsdbError::ForbiddenLabel(_))));
        assert!(matches!(Sample::new("price", 1.0, 0).validate(), Err(TsdbError::InvalidName(_))));
    }
}
