//! VictoriaMetrics passthrough with safety limits (spec 50 §5):
//!
//! | Route | Answer |
//! |---|---|
//! | `GET /api/metrics/query?query=&time=` | instant MetricsQL/PromQL query: VictoriaMetrics' `/api/v1/query` JSON, verbatim |
//! | `GET /api/metrics/query_range?query=&start=&end=&step=` | range query: VictoriaMetrics' `/api/v1/query_range` JSON, verbatim |
//!
//! Times are RFC3339, unix seconds, `now` or a duration back from now (`-6h`, `6h`); `start`
//! defaults to `end − 1h`, `end` to now, `step` to `range / 500` (at least 1 s). Limits (400 /
//! 422 otherwise): query ≤ 4 KiB, range ≤ 31 days, ≤ 11 000 points per series, ≤ 500 result
//! series, answer ≤ 8 MiB; VictoriaMetrics gets `timeout=10s` and control-api gives up after
//! 15 s (504). Read-only: only the two query endpoints are proxied (no admin, delete or import
//! API). 503 when VictoriaMetrics is unreachable.
use std::time::Duration;

use axum::{
    Json, Router,
    extract::{Query, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::state::AppState;

pub const MAX_QUERY_LEN: usize = 4096;
pub const MAX_RANGE_SECS: f64 = 31.0 * 86_400.0;
pub const MAX_POINTS: f64 = 11_000.0;
pub const MAX_SERIES: usize = 500;
pub const MAX_BYTES: usize = 8 * 1024 * 1024;
const VM_TIMEOUT: &str = "10s";
const HTTP_TIMEOUT: Duration = Duration::from_secs(15);

pub fn router() -> Router<AppState> {
    Router::new().route("/api/metrics/query", get(query)).route("/api/metrics/query_range", get(query_range))
}

#[derive(Debug, Default, Deserialize)]
pub struct MetricsQuery {
    pub query: Option<String>,
    pub time: Option<String>,
    pub start: Option<String>,
    pub end: Option<String>,
    pub step: Option<String>,
}

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({ "error": msg.into() }))).into_response()
}

/// `30s`, `5m`, `6h`, `2d`, `1w` or plain seconds → seconds.
pub fn parse_duration(s: &str) -> Option<f64> {
    let s = s.trim();
    if let Ok(v) = s.parse::<f64>() {
        return (v.is_finite() && v >= 0.0).then_some(v);
    }
    let (num, unit) = s.split_at(s.find(|c: char| c.is_ascii_alphabetic())?);
    let n: f64 = num.parse().ok()?;
    let mult = match unit {
        "ms" => 0.001,
        "s" => 1.0,
        "m" => 60.0,
        "h" => 3600.0,
        "d" => 86_400.0,
        "w" => 604_800.0,
        _ => return None,
    };
    (n.is_finite() && n >= 0.0).then_some(n * mult)
}

/// A time parameter → unix seconds (`now` = `now_s`).
pub fn parse_time(s: &str, now_s: f64) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() || s == "now" {
        return Some(now_s);
    }
    if let Ok(v) = s.parse::<f64>() {
        return v.is_finite().then_some(v);
    }
    if let Ok(t) = DateTime::parse_from_rfc3339(s) {
        return Some(t.timestamp_millis() as f64 / 1000.0);
    }
    let back = s.strip_prefix("now-").or_else(|| s.strip_prefix('-')).unwrap_or(s);
    parse_duration(back).map(|d| now_s - d)
}

fn check_query(q: &MetricsQuery) -> Result<String, String> {
    let query = q.query.as_deref().map(str::trim).unwrap_or_default();
    if query.is_empty() {
        return Err("query is required".into());
    }
    if query.len() > MAX_QUERY_LEN {
        return Err(format!("query longer than {MAX_QUERY_LEN} bytes"));
    }
    Ok(query.to_owned())
}

/// `(start, end, step)` in seconds for a range query, or the refusal.
pub fn range_params(q: &MetricsQuery, now_s: f64) -> Result<(f64, f64, f64), String> {
    let end = match q.end.as_deref() {
        Some(e) => parse_time(e, now_s).ok_or_else(|| format!("bad end {e:?}"))?,
        None => now_s,
    };
    let start = match q.start.as_deref() {
        Some(s) => parse_time(s, now_s).ok_or_else(|| format!("bad start {s:?}"))?,
        None => end - 3600.0,
    };
    if end <= start {
        return Err("end must be after start".into());
    }
    let range = end - start;
    if range > MAX_RANGE_SECS {
        return Err(format!("range {:.1} d exceeds 31 d", range / 86_400.0));
    }
    let step = match q.step.as_deref() {
        Some(s) => parse_duration(s).filter(|v| *v > 0.0).ok_or_else(|| format!("bad step {s:?}"))?,
        None => (range / 500.0).max(1.0).ceil(),
    };
    if range / step > MAX_POINTS {
        return Err(format!(
            "{:.0} points per series exceed {MAX_POINTS}; use step ≥ {:.0}s",
            range / step,
            (range / MAX_POINTS).ceil()
        ));
    }
    Ok((start, end, step))
}

fn now_s() -> f64 {
    Utc::now().timestamp_millis() as f64 / 1000.0
}

async fn query(State(s): State<AppState>, Query(q): Query<MetricsQuery>) -> Response {
    let query = match check_query(&q) {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_REQUEST, e),
    };
    let mut params = vec![("query", query), ("timeout", VM_TIMEOUT.into())];
    if let Some(t) = q.time.as_deref() {
        match parse_time(t, now_s()) {
            Some(v) => params.push(("time", format!("{v:.3}"))),
            None => return err(StatusCode::BAD_REQUEST, format!("bad time {t:?}")),
        }
    }
    forward(&s, "/api/v1/query", params).await
}

async fn query_range(State(s): State<AppState>, Query(q): Query<MetricsQuery>) -> Response {
    let query = match check_query(&q) {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_REQUEST, e),
    };
    let (start, end, step) = match range_params(&q, now_s()) {
        Ok(v) => v,
        Err(e) => return err(StatusCode::UNPROCESSABLE_ENTITY, e),
    };
    let params = vec![
        ("query", query),
        ("start", format!("{start:.3}")),
        ("end", format!("{end:.3}")),
        ("step", format!("{step}s")),
        ("timeout", VM_TIMEOUT.into()),
    ];
    forward(&s, "/api/v1/query_range", params).await
}

/// Number of result series in a VictoriaMetrics answer.
pub fn series_count(v: &Value) -> usize {
    v["data"]["result"].as_array().map_or(0, Vec::len)
}

async fn forward(s: &AppState, path: &str, params: Vec<(&str, String)>) -> Response {
    let url = format!("{}{path}", s.cfg.victoriametrics.url.trim_end_matches('/'));
    let client = match reqwest::Client::builder().timeout(HTTP_TIMEOUT).build() {
        Ok(c) => c,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let mut resp = match client.get(&url).query(&params).send().await {
        Ok(r) => r,
        Err(e) if e.is_timeout() => return err(StatusCode::GATEWAY_TIMEOUT, format!("victoriametrics: {e}")),
        Err(e) => return err(StatusCode::SERVICE_UNAVAILABLE, format!("victoriametrics unreachable: {e}")),
    };
    let status = resp.status();
    let mut body = Vec::new();
    loop {
        match resp.chunk().await {
            Ok(Some(c)) => {
                body.extend_from_slice(&c);
                if body.len() > MAX_BYTES {
                    return err(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        format!("answer larger than {} MiB; narrow the query", MAX_BYTES >> 20),
                    );
                }
            }
            Ok(None) => break,
            Err(e) => return err(StatusCode::BAD_GATEWAY, format!("victoriametrics: {e}")),
        }
    }
    let Ok(v) = serde_json::from_slice::<Value>(&body) else {
        return err(StatusCode::BAD_GATEWAY, format!("victoriametrics answered {status} without JSON"));
    };
    if !status.is_success() {
        // VM's own error object (`{status:"error", errorType, error}`), 422 for a bad query
        let code = if status.is_client_error() { StatusCode::UNPROCESSABLE_ENTITY } else { StatusCode::BAD_GATEWAY };
        return (code, Json(v)).into_response();
    }
    let n = series_count(&v);
    if n > MAX_SERIES {
        return err(StatusCode::UNPROCESSABLE_ENTITY, format!("{n} series exceed {MAX_SERIES}; aggregate or filter"));
    }
    ([(header::CACHE_CONTROL, "no-store")], Json(v)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_and_times_parse() {
        assert_eq!(parse_duration("90"), Some(90.0));
        assert_eq!(parse_duration("5m"), Some(300.0));
        assert_eq!(parse_duration("2d"), Some(172_800.0));
        assert_eq!(parse_duration("5x"), None);
        assert_eq!(parse_duration("-5m"), None);
        let now = 1_800_000_000.0;
        assert_eq!(parse_time("now", now), Some(now));
        assert_eq!(parse_time("-1h", now), Some(now - 3600.0));
        assert_eq!(parse_time("now-1h", now), Some(now - 3600.0));
        assert_eq!(parse_time("6h", now), Some(now - 21_600.0));
        assert_eq!(parse_time("1700000000", now), Some(1_700_000_000.0));
        assert_eq!(parse_time("2026-11-11T00:00:00Z", now), Some(1_794_355_200.0));
        assert_eq!(parse_time("yesterday", now), None);
    }

    #[test]
    fn range_queries_are_bounded() {
        let now = 1_800_000_000.0;
        let q = |start: &str, step: Option<&str>| MetricsQuery {
            query: Some("up".into()),
            start: Some(start.into()),
            step: step.map(Into::into),
            ..Default::default()
        };
        let (s, e, st) = range_params(&MetricsQuery::default(), now).unwrap();
        assert_eq!((e - s, st), (3600.0, 8.0), "default: last hour, ~500 points");
        assert!(range_params(&q("-40d", None), now).unwrap_err().contains("31 d"));
        assert!(range_params(&q("-1d", Some("1s")), now).unwrap_err().contains("points"));
        assert!(range_params(&q("-1d", Some("10s")), now).is_ok());
        assert!(range_params(&q("now", None), now).is_err(), "empty range");
        assert!(range_params(&q("-1h", Some("0")), now).is_err());
    }

    #[test]
    fn query_is_required_and_bounded() {
        assert!(check_query(&MetricsQuery::default()).is_err());
        let long = MetricsQuery { query: Some("x".repeat(MAX_QUERY_LEN + 1)), ..Default::default() };
        assert!(check_query(&long).is_err());
        let ok = MetricsQuery { query: Some(" up ".into()), ..Default::default() };
        assert_eq!(check_query(&ok).unwrap(), "up");
        assert_eq!(series_count(&json!({"data": {"result": [{}, {}]}})), 2);
    }
}
