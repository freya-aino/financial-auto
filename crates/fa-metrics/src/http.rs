//! Tower layer recording request counts and latency per method and matched route.

use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
    time::Instant,
};

use axum::{
    extract::MatchedPath,
    http::{Request, Response},
};
use tower::{Layer, Service};

/// `fa_http_requests_total{method,path,status}` + `fa_http_request_duration_seconds{method,path}`.
#[derive(Debug, Clone, Default)]
pub struct HttpMetricsLayer;

/// Builds the layer; use as `router.layer(fa_metrics::http::layer())`.
pub fn layer() -> HttpMetricsLayer {
    HttpMetricsLayer
}

impl<S> Layer<S> for HttpMetricsLayer {
    type Service = HttpMetrics<S>;
    fn layer(&self, inner: S) -> Self::Service {
        HttpMetrics { inner }
    }
}

#[derive(Debug, Clone)]
pub struct HttpMetrics<S> {
    inner: S,
}

/// Route label: the matched axum pattern when the router set one, otherwise the raw path with
/// id-looking segments collapsed to `{id}` (control-api chains routers with `fallback_service`,
/// which hides `MatchedPath` from an outer layer).
fn path_label<B>(req: &Request<B>) -> String {
    if let Some(m) = req.extensions().get::<MatchedPath>() {
        return m.as_str().to_owned();
    }
    normalize_path(req.uri().path())
}

/// Collapses numeric, UUID, ULID and long-hex segments to `{id}` and caps the depth at six
/// segments so the label set stays bounded.
pub fn normalize_path(path: &str) -> String {
    let mut out = String::new();
    for (i, seg) in path.split('/').filter(|s| !s.is_empty()).enumerate() {
        if i >= 6 {
            out.push_str("/*");
            break;
        }
        out.push('/');
        out.push_str(if looks_like_id(seg) { "{id}" } else { seg });
    }
    if out.is_empty() { "/".into() } else { out }
}

fn looks_like_id(seg: &str) -> bool {
    let n = seg.len();
    let all = |f: fn(char) -> bool| seg.chars().all(f);
    (n > 0 && all(|c| c.is_ascii_digit()))
        || (n == 36 && seg.chars().all(|c| c == '-' || c.is_ascii_hexdigit()) && seg.matches('-').count() == 4)
        || (n == 26 && all(|c| c.is_ascii_alphanumeric()) && seg.chars().any(|c| c.is_ascii_digit()))
        || (n >= 16 && all(|c| c.is_ascii_hexdigit()))
        || (seg.contains(['-', '_', ':']) && seg.chars().filter(|c| c.is_ascii_digit()).count() >= 6)
        // Generated identities carry a readable name before the digest (D-111), e.g.
        // `gp-nimble-elephant-6433eb2ddeeddf72`. Rule 4 no longer sees those as all-hex, and a
        // digest with few decimal digits would slip past the rule above too, so match the
        // trailing digest directly — otherwise every genome id becomes its own metric label.
        || seg.rsplit_once('-').is_some_and(|(head, digest)| {
            !head.is_empty() && digest.len() >= 8 && digest.chars().all(|c| c.is_ascii_hexdigit())
        })
}

pub(crate) fn record(method: &str, path: &str, status: u16, secs: f64) {
    crate::counter("fa_http_requests_total", "HTTP requests by method, matched route and status").inc_with(&[
        ("method", method),
        ("path", path),
        ("status", &status.to_string()),
    ]);
    crate::histogram(
        "fa_http_request_duration_seconds",
        "HTTP request latency by method and matched route",
        crate::LATENCY_BUCKETS,
    )
    .observe_with(&[("method", method), ("path", path)], secs);
}

impl<S, B, R> Service<Request<B>> for HttpMetrics<S>
where
    S: Service<Request<B>, Response = Response<R>>,
    S::Future: Send + 'static,
    S::Error: Send + 'static,
    R: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<B>) -> Self::Future {
        let method = req.method().as_str().to_owned();
        let path = path_label(&req);
        let start = Instant::now();
        let fut = self.inner.call(req);
        Box::pin(async move {
            let res = fut.await;
            if let Ok(r) = &res {
                record(&method, &path, r.status().as_u16(), start.elapsed().as_secs_f64());
            }
            res
        })
    }
}

#[cfg(test)]
mod tests {
    use axum::{Router, body::Body, http::Request, routing::get};
    use tower::ServiceExt;

    #[test]
    fn normalize_collapses_ids_and_caps_depth() {
        let n = super::normalize_path;
        assert_eq!(n("/api/trading/strategies/42/pause"), "/api/trading/strategies/{id}/pause");
        assert_eq!(n("/api/graph/lineage/01M721FCXSVATJ6CQM9BEBFAKB"), "/api/graph/lineage/{id}");
        assert_eq!(
            n("/api/instruments/quarantine/0d7f2c4a-1b2c-4d5e-8f90-abcdef123456/resolve"),
            "/api/instruments/quarantine/{id}/resolve"
        );
        assert_eq!(n("/api/latent/stat-v1-1794292563/regions"), "/api/latent/{id}/regions");
        assert_eq!(n("/api/health"), "/api/health");
        assert_eq!(n("/"), "/");
        assert_eq!(n("/a/b/c/d/e/f/g/h"), "/a/b/c/d/e/f/*");
    }

    #[tokio::test]
    async fn layer_counts_requests_by_matched_route() {
        let app = Router::new().route("/api/items/{id}", get(|| async { "ok" })).layer(super::layer());
        for id in ["1", "2"] {
            let res = app
                .clone()
                .oneshot(Request::get(format!("/api/items/{id}")).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(res.status(), 200);
        }
        let res = app.clone().oneshot(Request::get("/api/nope").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(res.status(), 404);
        let out = crate::render();
        assert!(
            out.contains("fa_http_requests_total{method=\"GET\",path=\"/api/items/{id}\",status=\"200\"} 2\n"),
            "{out}"
        );
        assert!(out.contains("fa_http_requests_total{method=\"GET\",path=\"/api/nope\",status=\"404\"} 1\n"), "{out}");
        assert!(out.contains("fa_http_request_duration_seconds_count{method=\"GET\",path=\"/api/items/{id}\"} 2\n"));
    }
}

#[cfg(test)]
mod id_shape_tests {
    use super::looks_like_id;

    /// D-111 ids must still be redacted from metric labels, including a digest with no decimal
    /// digits at all (which every other rule misses).
    #[test]
    fn named_ids_are_redacted() {
        assert!(looks_like_id("gp-nimble-elephant-6433eb2ddeeddf72"));
        assert!(looks_like_id("alloc-quick-fox-abcdefabcdefabcd"));
        assert!(looks_like_id("baseline-trend-silent-ibex-deadbeef"));
    }

    #[test]
    fn ordinary_route_segments_are_not_redacted() {
        for seg in ["genomes", "evolution", "api", "graph", "health", "e-mail"] {
            assert!(!looks_like_id(seg), "{seg} must not be treated as an id");
        }
    }
}
