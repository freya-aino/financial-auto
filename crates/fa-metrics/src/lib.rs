//! Tiny dependency-free Prometheus text exporter (spec 60 §4, §6.5).
//!
//! One process-wide [`Registry`] holds counters, gauges and fixed-bucket histograms, each
//! keyed by name + label set. [`render`] produces the Prometheus text exposition format and
//! [`serve`] exposes it on `GET /metrics`. Collectors registered with [`register_collector`]
//! run right before every render (used by [`process`] for uptime / RSS).

pub mod heartbeat;
pub mod http;
pub mod process;

use std::{
    collections::BTreeMap,
    fmt::Write as _,
    net::SocketAddr,
    sync::{LazyLock, Mutex, MutexGuard},
};

/// Label set as given at the call site (order preserved, part of the series key).
pub type Labels = Vec<(String, String)>;

/// Default latency buckets in seconds.
pub const LATENCY_BUCKETS: &[f64] = &[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0];

#[derive(Debug, Clone, PartialEq)]
enum Kind {
    Counter,
    Gauge,
    Histogram(Vec<f64>),
}

impl Kind {
    fn text(&self) -> &'static str {
        match self {
            Kind::Counter => "counter",
            Kind::Gauge => "gauge",
            Kind::Histogram(_) => "histogram",
        }
    }
}

#[derive(Debug, Clone)]
enum Series {
    Value(f64),
    Hist { counts: Vec<u64>, sum: f64, count: u64 },
}

#[derive(Debug, Clone)]
struct Family {
    help: String,
    kind: Kind,
    series: BTreeMap<Labels, Series>,
}

type Collector = Box<dyn Fn() + Send + Sync>;

/// A metrics registry. Services use the process-wide one through the free functions.
#[derive(Default)]
pub struct Registry {
    families: BTreeMap<String, Family>,
    collectors: Vec<Collector>,
}

static GLOBAL: LazyLock<Mutex<Registry>> = LazyLock::new(|| Mutex::new(Registry::default()));

fn global() -> MutexGuard<'static, Registry> {
    GLOBAL.lock().unwrap_or_else(|p| p.into_inner())
}

fn labels_of(labels: &[(&str, &str)]) -> Labels {
    labels.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect()
}

impl Registry {
    fn family(&mut self, name: &str, help: &str, kind: Kind) -> &mut Family {
        let fam = self.families.entry(name.to_owned()).or_insert_with(|| Family {
            help: help.to_owned(),
            kind: kind.clone(),
            series: BTreeMap::new(),
        });
        if fam.kind != kind {
            tracing::warn!(metric = name, "metric re-registered with a different type; keeping the first");
        }
        fam
    }

    fn value(&mut self, name: &str, help: &str, kind: Kind, labels: Labels) -> &mut f64 {
        let fam = self.family(name, help, kind);
        match fam.series.entry(labels).or_insert(Series::Value(0.0)) {
            Series::Value(v) => v,
            Series::Hist { sum, .. } => sum, // type clash already warned above
        }
    }

    /// Renders every family in the Prometheus text format.
    pub fn render_to(&self, out: &mut String) {
        for (name, fam) in &self.families {
            let _ = writeln!(out, "# HELP {name} {}", escape_help(&fam.help));
            let _ = writeln!(out, "# TYPE {name} {}", fam.kind.text());
            for (labels, series) in &fam.series {
                match (series, &fam.kind) {
                    (Series::Value(v), _) => {
                        let _ = writeln!(out, "{name}{} {}", fmt_labels(labels, None), fmt_f64(*v));
                    }
                    (Series::Hist { counts, sum, count }, Kind::Histogram(buckets)) => {
                        let mut cum = 0u64;
                        for (b, c) in buckets.iter().zip(counts) {
                            cum += c;
                            let le = fmt_f64(*b);
                            let _ = writeln!(out, "{name}_bucket{} {cum}", fmt_labels(labels, Some(&le)));
                        }
                        let _ = writeln!(out, "{name}_bucket{} {count}", fmt_labels(labels, Some("+Inf")));
                        let _ = writeln!(out, "{name}_sum{} {}", fmt_labels(labels, None), fmt_f64(*sum));
                        let _ = writeln!(out, "{name}_count{} {count}", fmt_labels(labels, None));
                    }
                    (Series::Hist { .. }, _) => {}
                }
            }
        }
    }
}

fn fmt_f64(v: f64) -> String {
    if v.is_nan() {
        "NaN".into()
    } else if v.is_infinite() {
        if v > 0.0 { "+Inf".into() } else { "-Inf".into() }
    } else {
        format!("{v}")
    }
}

fn escape_label(v: &str) -> String {
    v.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

fn escape_help(v: &str) -> String {
    v.replace('\\', "\\\\").replace('\n', "\\n")
}

fn fmt_labels(labels: &Labels, le: Option<&str>) -> String {
    if labels.is_empty() && le.is_none() {
        return String::new();
    }
    let mut parts: Vec<String> = labels.iter().map(|(k, v)| format!("{k}=\"{}\"", escape_label(v))).collect();
    if let Some(le) = le {
        parts.push(format!("le=\"{le}\""));
    }
    format!("{{{}}}", parts.join(","))
}

/// Handle to a counter family (monotonic).
#[derive(Debug, Clone)]
pub struct Counter {
    name: String,
    help: String,
}

impl Counter {
    pub fn inc(&self) {
        self.add_with(&[], 1.0);
    }
    pub fn add(&self, v: f64) {
        self.add_with(&[], v);
    }
    pub fn inc_with(&self, labels: &[(&str, &str)]) {
        self.add_with(labels, 1.0);
    }
    pub fn add_with(&self, labels: &[(&str, &str)], v: f64) {
        if v < 0.0 {
            return;
        }
        *global().value(&self.name, &self.help, Kind::Counter, labels_of(labels)) += v;
    }
}

/// Handle to a gauge family.
#[derive(Debug, Clone)]
pub struct Gauge {
    name: String,
    help: String,
}

impl Gauge {
    pub fn set(&self, v: f64) {
        self.set_with(&[], v);
    }
    pub fn set_with(&self, labels: &[(&str, &str)], v: f64) {
        *global().value(&self.name, &self.help, Kind::Gauge, labels_of(labels)) = v;
    }
    pub fn add_with(&self, labels: &[(&str, &str)], v: f64) {
        *global().value(&self.name, &self.help, Kind::Gauge, labels_of(labels)) += v;
    }
    /// Drops one series (e.g. a consumer that no longer exists).
    pub fn remove_with(&self, labels: &[(&str, &str)]) {
        if let Some(f) = global().families.get_mut(&self.name) {
            f.series.remove(&labels_of(labels));
        }
    }
}

/// Handle to a histogram family with fixed buckets.
#[derive(Debug, Clone)]
pub struct Histogram {
    name: String,
    help: String,
    buckets: Vec<f64>,
}

impl Histogram {
    pub fn observe(&self, v: f64) {
        self.observe_with(&[], v);
    }
    pub fn observe_with(&self, labels: &[(&str, &str)], v: f64) {
        let mut g = global();
        let n = self.buckets.len();
        let fam = g.family(&self.name, &self.help, Kind::Histogram(self.buckets.clone()));
        let s = fam.series.entry(labels_of(labels)).or_insert_with(|| Series::Hist {
            counts: vec![0; n],
            sum: 0.0,
            count: 0,
        });
        if let Series::Hist { counts, sum, count } = s {
            if let Some(i) = self.buckets.iter().position(|b| v <= *b) {
                counts[i] += 1;
            }
            *sum += v;
            *count += 1;
        }
    }
}

/// Registers (or looks up) a counter family.
pub fn counter(name: &str, help: &str) -> Counter {
    global().family(name, help, Kind::Counter);
    Counter { name: name.to_owned(), help: help.to_owned() }
}

/// Registers (or looks up) a gauge family.
pub fn gauge(name: &str, help: &str) -> Gauge {
    global().family(name, help, Kind::Gauge);
    Gauge { name: name.to_owned(), help: help.to_owned() }
}

/// Registers (or looks up) a histogram family; buckets are sorted and deduplicated.
pub fn histogram(name: &str, help: &str, buckets: &[f64]) -> Histogram {
    let mut b: Vec<f64> = buckets.iter().copied().filter(|x| x.is_finite()).collect();
    b.sort_by(|a, c| a.total_cmp(c));
    b.dedup();
    global().family(name, help, Kind::Histogram(b.clone()));
    Histogram { name: name.to_owned(), help: help.to_owned(), buckets: b }
}

/// Registers a function run before every render (to refresh gauges lazily).
pub fn register_collector(f: impl Fn() + Send + Sync + 'static) {
    global().collectors.push(Box::new(f));
}

/// Runs the collectors and renders the whole registry.
pub fn render() -> String {
    let collectors: Vec<Collector> = std::mem::take(&mut global().collectors);
    for c in &collectors {
        c();
    }
    let mut out = String::new();
    {
        let mut g = global();
        g.render_to(&mut out);
        let mut cs = collectors;
        cs.append(&mut g.collectors);
        g.collectors = cs;
    }
    out
}

/// Host-side default bind per service (spec 60 §4 metrics ports, D-056). Unknown → `:9100`.
pub fn default_bind(service: &str) -> &'static str {
    match service {
        "control-api" => "127.0.0.1:9101",
        "model-runtime" => "127.0.0.1:9102",
        "strategy-runtime" => "127.0.0.1:9103",
        "risk-engine" => "127.0.0.1:9104",
        "ingestor" => "127.0.0.1:9105",
        "broker-gateway" => "127.0.0.1:9106",
        "evo-orchestrator" => "127.0.0.1:9107",
        "evo-worker" => "127.0.0.1:9110",
        _ => "127.0.0.1:9100",
    }
}

/// Resolves the bind from `FA_METRICS_BIND` (`off` / `0` / empty disables) or the service default.
pub fn bind_from_env(service: &str) -> Option<SocketAddr> {
    resolve_bind(service, std::env::var("FA_METRICS_BIND").ok().as_deref())
}

/// Pure part of [`bind_from_env`]: `env` is the raw `FA_METRICS_BIND` value, if any.
pub fn resolve_bind(service: &str, env: Option<&str>) -> Option<SocketAddr> {
    let raw = env.map(str::trim).filter(|v| !v.is_empty()).unwrap_or_else(|| default_bind(service));
    if matches!(raw, "off" | "0" | "none" | "false") {
        return None;
    }
    match raw.parse() {
        Ok(a) => Some(a),
        Err(e) => {
            tracing::warn!(bind = raw, error = %e, "FA_METRICS_BIND unparsable; metrics disabled");
            None
        }
    }
}

/// Starts `GET /metrics` on `bind` in a background task (needs a tokio runtime).
pub fn serve(bind: SocketAddr) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let app = axum::Router::new()
            .route("/metrics", axum::routing::get(metrics_handler))
            .route("/", axum::routing::get(|| async { "fa-metrics: see /metrics\n" }));
        match tokio::net::TcpListener::bind(bind).await {
            Ok(l) => {
                tracing::info!(%bind, "metrics endpoint listening");
                if let Err(e) = axum::serve(l, app).await {
                    tracing::warn!(error = %e, "metrics server stopped");
                }
            }
            Err(e) => tracing::warn!(%bind, error = %e, "metrics endpoint not started"),
        }
    })
}

async fn metrics_handler() -> axum::response::Response {
    use axum::response::IntoResponse;
    ([(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")], render()).into_response()
}

/// The one-line service hook: registers process metrics and `fa_build_info`, then serves on
/// `FA_METRICS_BIND` or the per-service host default. Returns `None` when disabled.
pub fn serve_from_env(service: &str, version: &str) -> Option<tokio::task::JoinHandle<()>> {
    process::register(service, version);
    bind_from_env(service).map(serve)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_counter_gauge_histogram() {
        let c = counter("t_requests_total", "requests");
        c.inc_with(&[("method", "GET"), ("path", "/a\"b")]);
        c.inc_with(&[("method", "GET"), ("path", "/a\"b")]);
        let g = gauge("t_gauge", "a gauge");
        g.set(2.5);
        let h = histogram("t_lat", "latency", &[0.1, 0.5, 0.5, 1.0]);
        h.observe(0.05);
        h.observe(0.7);
        h.observe(5.0);
        let out = render();
        assert!(out.contains("# TYPE t_requests_total counter\n"));
        assert!(out.contains("t_requests_total{method=\"GET\",path=\"/a\\\"b\"} 2\n"), "{out}");
        assert!(out.contains("t_gauge 2.5\n"));
        assert!(out.contains("t_lat_bucket{le=\"0.1\"} 1\n"));
        assert!(out.contains("t_lat_bucket{le=\"0.5\"} 1\n"));
        assert!(out.contains("t_lat_bucket{le=\"1\"} 2\n"));
        assert!(out.contains("t_lat_bucket{le=\"+Inf\"} 3\n"));
        assert!(out.contains("t_lat_sum 5.75\n"));
        assert!(out.contains("t_lat_count 3\n"));
    }

    #[test]
    fn counters_ignore_negative_and_gauges_remove() {
        let c = counter("t_neg_total", "x");
        c.add(-1.0);
        c.add(3.0);
        let g = gauge("t_rm", "x");
        g.set_with(&[("k", "a")], 1.0);
        g.set_with(&[("k", "b")], 2.0);
        g.remove_with(&[("k", "a")]);
        let out = render();
        assert!(out.contains("t_neg_total 3\n"));
        assert!(!out.contains("t_rm{k=\"a\"}"));
        assert!(out.contains("t_rm{k=\"b\"} 2\n"));
    }

    #[test]
    fn collectors_run_before_render_and_persist() {
        let g = gauge("t_collected", "x");
        register_collector(move || g.add_with(&[], 1.0));
        render();
        let out = render();
        assert!(out.contains("t_collected 2\n"), "{out}");
    }

    #[test]
    fn special_floats_render_prometheus_style() {
        assert_eq!(fmt_f64(f64::INFINITY), "+Inf");
        assert_eq!(fmt_f64(f64::NEG_INFINITY), "-Inf");
        assert_eq!(fmt_f64(f64::NAN), "NaN");
        assert_eq!(fmt_f64(1.0), "1");
    }

    #[test]
    fn bind_defaults_and_env_off() {
        assert_eq!(default_bind("risk-engine"), "127.0.0.1:9104");
        assert_eq!(default_bind("nope"), "127.0.0.1:9100");
        assert!(resolve_bind("risk-engine", Some("off")).is_none());
        assert!(resolve_bind("risk-engine", Some("not an addr")).is_none());
        assert_eq!(resolve_bind("x", Some(" 0.0.0.0:9100 ")).unwrap().port(), 9100);
        assert_eq!(resolve_bind("evo-worker", None).unwrap().port(), 9110);
        assert_eq!(resolve_bind("evo-worker", Some("")).unwrap().port(), 9110);
    }
}
