//! VictoriaMetrics access (spec 20 §5.3): series naming, Prometheus-text import, range queries
//! and the trade-price reader used for decoder evaluation (spec 41 §5).
//!
//! Label conventions match `go/pkg/tsdb` (the ingestor's writer): `inst` is the instrument
//! UUID as a D-017 subject token (lowercase, no dashes — [`inst_token`]), `venue` the
//! connector's venue token, `src` the source id. The live tests (`FA_VM_TEST=1`, VM at
//! `FA_VM_URL` or `http://127.0.0.1:8428`) write with this crate and read back with the
//! same token mapping, and check the ingestor's `fa_trade_price` series obey it.
use std::{collections::BTreeMap, time::Duration};

use fa_core::Timescale;
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
            let parts: Vec<String> = self.labels.iter().map(|(k, v)| format!("{k}=\"{}\"", escape_label(v))).collect();
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

/// `inst` label value of an instrument id: lowercase UUID without dashes, exactly
/// `bus.SubjectToken` in `go/pkg/bus` (D-017). Already-tokenised ids pass through.
pub fn inst_token(instrument_id: &str) -> String {
    instrument_id.trim().to_ascii_lowercase().replace('-', "")
}

/// PromQL step / range for a timescale (`1s`, `1m`, `1h`, `1d`, `1w`).
pub fn step_of(ts: Timescale) -> &'static str {
    ts.as_str()
}

/// Escapes a label value for a PromQL matcher.
fn matcher_value(v: &str) -> String {
    escape_label(v)
}

/// One instrument's price series as returned by [`Tsdb::trade_prices`].
#[derive(Debug, Clone, PartialEq)]
pub struct TradeSeries {
    /// `inst` label (instrument token, no dashes).
    pub inst: String,
    pub venue: String,
    pub src: String,
    /// `(ts_ms, price)` ascending; one point per step bucket that had a trade.
    pub points: Vec<(i64, f64)>,
}

/// VictoriaMetrics refuses `query_range` requests with more points than
/// `-search.maxPointsPerTimeseries` (default 30 000); callers shrink the window to this.
pub const MAX_POINTS_PER_SERIES: i64 = 20_000;

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
    /// Client with a 3 s connect / 20 s request timeout so an unreachable VM fails fast
    /// (evo-worker falls back to its in-memory buffer).
    pub fn new(base_url: impl Into<String>) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(20))
            .build()
            .unwrap_or_default();
        Self { base: base_url.into().trim_end_matches('/').to_owned(), http }
    }

    pub fn base_url(&self) -> &str {
        &self.base
    }

    /// `GET /health` → `Ok(())` when VM answers 200.
    pub async fn health(&self) -> Result<(), TsdbError> {
        let resp = self.http.get(format!("{}/health", self.base)).send().await?;
        check(resp).await.map(|_| ())
    }

    /// Makes just-written samples searchable (`/internal/force_flush`); tests only.
    pub async fn force_flush(&self) -> Result<(), TsdbError> {
        let resp = self.http.get(format!("{}/internal/force_flush", self.base)).send().await?;
        check(resp).await.map(|_| ())
    }

    /// Last trade price per `step` bucket over `[start_s, end_s]` for one instrument (or every
    /// instrument when `instrument` is `None`): `last_over_time(fa_trade_price{inst="…"}[step])`
    /// at `step` resolution, so buckets without a trade are absent instead of filled.
    /// `instrument` may be a dashed UUID or a token. Series are returned longest first.
    pub async fn trade_prices(
        &self,
        instrument: Option<&str>,
        start_s: i64,
        end_s: i64,
        step: &str,
    ) -> Result<Vec<TradeSeries>, TsdbError> {
        let selector = match instrument {
            Some(i) => format!("{}{{inst=\"{}\"}}", metrics::TRADE_PRICE, matcher_value(&inst_token(i))),
            None => metrics::TRADE_PRICE.to_owned(),
        };
        let query = format!("last_over_time({selector}[{step}])");
        let series = self.query_range(&query, start_s, end_s, step).await?;
        let mut out: Vec<TradeSeries> = series
            .into_iter()
            .map(|s| {
                let label = |k: &str| s.metric.get(k).cloned().unwrap_or_default();
                let mut points: Vec<(i64, f64)> = s
                    .values
                    .iter()
                    .filter_map(|(t, v)| v.parse::<f64>().ok().map(|p| ((*t * 1000.0) as i64, p)))
                    .filter(|(_, p)| p.is_finite() && *p > 0.0)
                    .collect();
                points.sort_by_key(|p| p.0);
                points.dedup_by_key(|p| p.0);
                TradeSeries { inst: label("inst"), venue: label("venue"), src: label("src"), points }
            })
            .filter(|s| !s.points.is_empty())
            .collect();
        out.sort_by(|a, b| b.points.len().cmp(&a.points.len()).then_with(|| a.inst.cmp(&b.inst)));
        Ok(out)
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
    pub async fn query_range(
        &self,
        query: &str,
        start: i64,
        end: i64,
        step: &str,
    ) -> Result<Vec<RangeSeries>, TsdbError> {
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
        assert!(matches!(
            Sample::new("fa_x", 1.0, 0).label("trade_id", "1").validate(),
            Err(TsdbError::ForbiddenLabel(_))
        ));
        assert!(matches!(Sample::new("price", 1.0, 0).validate(), Err(TsdbError::InvalidName(_))));
    }

    #[test]
    fn inst_token_matches_go_subject_token() {
        // go/pkg/bus/subjects_test.go: SubjectToken strips dashes and lowercases
        assert_eq!(inst_token("01900000-0000-7000-8000-000000000001"), "01900000000070008000000000000001");
        assert_eq!(inst_token(" 0190AB-CD "), "0190abcd");
        assert_eq!(inst_token("01900000000070008000000000000001"), "01900000000070008000000000000001");
        assert_eq!(step_of(Timescale::Minute), "1m");
    }

    #[test]
    fn parses_query_range_matrix() {
        let body = r#"{"status":"success","data":{"resultType":"matrix","result":[{"metric":{"__name__":"fa_trade_price","inst":"abc","src":"heartbeat","venue":"sim"},"values":[[1794285112,"102.18"],[1794285113,"101.9"]]}]}}"#;
        let parsed: QueryRangeResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.data.result.len(), 1);
        assert_eq!(parsed.data.result[0].values[1], (1794285113.0, "101.9".to_owned()));
        assert_eq!(parsed.data.result[0].metric["inst"], "abc");
    }

    /// Live VictoriaMetrics round trip; run with `FA_VM_TEST=1 cargo test -p fa-tsdb`.
    #[tokio::test]
    async fn live_vm_write_read_with_token_mapping() {
        if std::env::var("FA_VM_TEST").ok().as_deref() != Some("1") {
            eprintln!("skipped: set FA_VM_TEST=1 (VictoriaMetrics at FA_VM_URL or http://127.0.0.1:8428)");
            return;
        }
        let url = std::env::var("FA_VM_URL").unwrap_or_else(|_| "http://127.0.0.1:8428".into());
        let vm = Tsdb::new(url);
        vm.health().await.expect("VM /health");

        // 1. Write with a dashed UUID → `inst` label is the dashless token (as go/pkg/tsdb).
        let uuid = fa_core::InstrumentId::new().to_string();
        let token = inst_token(&uuid);
        assert!(uuid.contains('-') && !token.contains('-') && token.len() == 32);
        let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64;
        // well behind `-search.latencyOffset` (30 s) so query_range sees them
        let base_ms = now_ms - 600_000;
        let samples: Vec<Sample> = (0..120)
            .map(|i| {
                Sample::new(metrics::TRADE_PRICE, 100.0 + i as f64 * 0.01, base_ms + i * 1000)
                    .label("inst", token.clone())
                    .label("venue", "fa-tsdb-test")
                    .label("src", "fa-tsdb-test")
            })
            .collect();
        vm.write(&samples).await.expect("import");
        vm.force_flush().await.expect("force_flush");

        // 2. Read back by dashed UUID (mapped) and by token: identical, one point per second.
        let start = base_ms / 1000 - 5;
        let end = base_ms / 1000 + 125;
        let by_uuid = vm.trade_prices(Some(&uuid), start, end, "1s").await.expect("query_range");
        let by_token = vm.trade_prices(Some(&token), start, end, "1s").await.unwrap();
        assert_eq!(by_uuid, by_token);
        assert_eq!(by_uuid.len(), 1, "exactly the test series: {by_uuid:?}");
        let s = &by_uuid[0];
        assert_eq!(
            (s.inst.as_str(), s.venue.as_str(), s.src.as_str()),
            (token.as_str(), "fa-tsdb-test", "fa-tsdb-test")
        );
        assert_eq!(s.points.len(), 120, "one bucket per written second");
        assert!((s.points[0].1 - 100.0).abs() < 1e-9 && (s.points[119].1 - 101.19).abs() < 1e-9);
        assert!(s.points.windows(2).all(|w| w[1].0 - w[0].0 == 1000));
        // 1m buckets keep the last price of each minute
        let per_min = vm.trade_prices(Some(&uuid), start, end, "1m").await.unwrap();
        assert!(per_min[0].points.len() <= 3 && per_min[0].points.len() >= 2, "{:?}", per_min[0].points);

        // 3. The ingestor's series (go/pkg/tsdb) use the same mapping: every `inst` is a
        //    32-char dashless lowercase token.
        let all = vm.trade_prices(None, end - 86_400, end, "1h").await.unwrap();
        for s in &all {
            assert_eq!(s.inst, inst_token(&s.inst), "inst label {} is not a subject token", s.inst);
            assert!(s.inst.chars().all(|c| c.is_ascii_hexdigit()), "{}", s.inst);
        }
        assert!(all.iter().any(|s| s.inst == token));

        // 4. Clean up the test series (best effort) so evaluation jobs do not see it.
        let _ = vm
            .http
            .post(format!("{}/api/v1/admin/tsdb/delete_series", vm.base))
            .query(&[("match[]", r#"{venue="fa-tsdb-test"}"#)])
            .send()
            .await;
    }
}
