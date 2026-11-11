//! Spec 33 §4.2 query surface over the **persisted** kernels, plus region lineage / hierarchy /
//! cross-asset priors (spec 30 §5, spec 33 §5, §6.5).
//!
//! Every answer is a posterior **distribution** (`{mean, sd, lower, upper, median, level, n}`),
//! computed from the exact Dirichlet kernel model-runtime persists in the expert space's
//! `RegionSet.kernel_state` (geometric prior and forgetting included), by Monte Carlo over
//! `draws` kernel draws (`fa_mcmc::inference::Posterior`). Parsed kernels are cached 30 s per
//! expert space. `seed` makes a sampled answer reproducible.
//!
//! | Route | Answer |
//! |---|---|
//! | `GET /api/mcmc/{snapshot}/kernels` | `KernelModel` rows: serving sampler, calibration standings, stationary, relaxation, drift |
//! | `GET /api/mcmc/{snapshot}/transition?expert=&from=&to=&k=` | `transition_prob(a, b, k)` |
//! | `GET /api/mcmc/{snapshot}/k-step?expert=&from=&k=&n=` | k-step distribution over regions |
//! | `GET /api/mcmc/{snapshot}/neighbors?expert=&region=&n=` | one-step successors with uncertainty |
//! | `GET /api/mcmc/{snapshot}/paths?expert=&from=&k=&n=&seed=` | `sample_paths` (one kernel draw per path) |
//! | `GET /api/mcmc/{snapshot}/path-prob?expert=&path=a,b,c` | `path_prob` |
//! | `GET /api/mcmc/{snapshot}/hitting?expert=&from=&targets=a,b&horizon=` | `hitting_time`: P(hit), E[steps], pmf |
//! | `GET /api/mcmc/{snapshot}/stationary?expert=` | `stationary` + relaxation time |
//! | `GET /api/mcmc/{snapshot}/hierarchy` | `Hier` (REFINES) edges of the snapshot's regions |
//! | `GET /api/mcmc/cross-priors?timescale=&limit=` | ranked `CrossPrior` proposals |
//! | `GET /api/latent/regions/{region_id}/lineage?depth=` | ancestors, descendants, lineage members |
//! | `GET /api/latent/regions/{region_id}/reconstructions?limit=&max_bars=` | the region's stored `RegionEpisode`s (latest exit first) each with the 1 m market path over `[t_enter, t_exit]` from VictoriaMetrics (connector bars, else last trade price per minute); per-region decoder outputs are not recorded |
//!
//! Helix unreachable → 503 `helix_unavailable`; unknown snapshot / expert space / region → 404;
//! no persisted kernel yet → 404 `no_kernel`; bad parameters → 400.

// Handlers return ready `Response`s as their error type (axum style).
#![allow(clippy::result_large_err)]

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use fa_core::rng::Rng;
use fa_hub::HubError;
use fa_mcmc::{MarkovGraph, inference::Posterior};
use serde::Deserialize;
use serde_json::{Value, json};
use tower_http::cors::CorsLayer;

use crate::state::AppState;

const CACHE_TTL: Duration = Duration::from_secs(30);
const DEFAULT_DRAWS: usize = 128;
const MAX_DRAWS: usize = 1000;
const MAX_K: u32 = 512;
const MAX_PATHS: usize = 2000;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/mcmc/cross-priors", get(cross_priors))
        .route("/api/mcmc/{snapshot}/kernels", get(kernels))
        .route("/api/mcmc/{snapshot}/transition", get(transition))
        .route("/api/mcmc/{snapshot}/k-step", get(k_step))
        .route("/api/mcmc/{snapshot}/neighbors", get(neighbors))
        .route("/api/mcmc/{snapshot}/paths", get(paths))
        .route("/api/mcmc/{snapshot}/path-prob", get(path_prob))
        .route("/api/mcmc/{snapshot}/hitting", get(hitting))
        .route("/api/mcmc/{snapshot}/stationary", get(stationary))
        .route("/api/mcmc/{snapshot}/hierarchy", get(hierarchy))
        .route("/api/latent/regions/{region_id}/lineage", get(lineage))
        .route("/api/latent/regions/{region_id}/reconstructions", get(reconstructions))
        .layer(CorsLayer::permissive())
}

fn err(status: StatusCode, error: &str, detail: impl Into<String>) -> Response {
    (status, Json(json!({ "error": error, "detail": detail.into() }))).into_response()
}

fn hub_err(e: HubError) -> Response {
    if e.is_unreachable() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "helix_unavailable", e.to_string());
    }
    match e {
        HubError::NotFound { label, id } => {
            (StatusCode::NOT_FOUND, Json(json!({ "error": "not_found", "label": label, "id": id }))).into_response()
        }
        other => err(StatusCode::BAD_GATEWAY, "helix_error", other.to_string()),
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct Q {
    pub expert: Option<u32>,
    pub from: Option<String>,
    pub to: Option<String>,
    pub region: Option<String>,
    pub k: Option<u32>,
    pub n: Option<usize>,
    pub draws: Option<usize>,
    pub seed: Option<u64>,
    pub path: Option<String>,
    pub targets: Option<String>,
    pub horizon: Option<u32>,
    pub level: Option<f64>,
}

type Cache = Mutex<HashMap<String, (Instant, Arc<MarkovGraph>)>>;

fn cache() -> &'static Cache {
    static C: OnceLock<Cache> = OnceLock::new();
    C.get_or_init(Mutex::default)
}

/// The persisted kernel of `<snapshot>-e<expert>`.
async fn kernel(s: &AppState, snapshot: &str, expert: u32) -> Result<Arc<MarkovGraph>, Response> {
    let id = format!("{snapshot}-e{expert}");
    if let Some((t, g)) = cache().lock().expect("cache").get(&id)
        && t.elapsed() < CACHE_TTL
    {
        return Ok(g.clone());
    }
    let row = match s.hub.region_set(&id).await {
        Ok(Some(r)) => r,
        Ok(None) => {
            return Err(match s.hub.space_by_id(&id).await {
                Ok(Some(_)) => err(
                    StatusCode::NOT_FOUND,
                    "no_kernel",
                    format!(
                        "expert space {id} has no persisted kernel yet (model-runtime writes it every --hub-state-secs)"
                    ),
                ),
                Ok(None) => {
                    (StatusCode::NOT_FOUND, Json(json!({ "error": "not_found", "label": "ExpertSpace", "id": id })))
                        .into_response()
                }
                Err(e) => hub_err(e),
            });
        }
        Err(e) => return Err(hub_err(e)),
    };
    let g: MarkovGraph = serde_json::from_str(&row.kernel_state)
        .map_err(|e| err(StatusCode::BAD_GATEWAY, "bad_kernel_state", e.to_string()))?;
    let g = Arc::new(g);
    cache().lock().expect("cache").insert(id, (Instant::now(), g.clone()));
    Ok(g)
}

fn seed_of(q: &Q, salt: &str) -> u64 {
    q.seed.unwrap_or_else(|| {
        salt.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
    })
}

async fn posterior(s: &AppState, snapshot: &str, q: &Q, salt: &str) -> Result<(Posterior, Rng), Response> {
    let g = kernel(s, snapshot, q.expert.unwrap_or(0)).await?;
    let mut rng = Rng::seed(seed_of(q, salt));
    let draws = q.draws.unwrap_or(DEFAULT_DRAWS).clamp(8, MAX_DRAWS);
    let level = q.level.unwrap_or(0.9).clamp(0.5, 0.999);
    let post = Posterior::draw(&g, draws, level, &mut rng);
    Ok((post, rng))
}

fn required<'a>(v: &'a Option<String>, name: &str) -> Result<&'a str, Response> {
    v.as_deref()
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, "bad_request", format!("`{name}` is required")))
}

fn list(v: &str) -> Vec<String> {
    v.split(',').map(str::trim).filter(|x| !x.is_empty()).map(str::to_owned).collect()
}

fn unknown_region(r: &str) -> Response {
    (StatusCode::NOT_FOUND, Json(json!({ "error": "not_found", "label": "LatentRegion", "id": r }))).into_response()
}

fn ctx(snapshot: &str, q: &Q, post: &Posterior) -> Value {
    json!({ "snapshot": snapshot, "expert": q.expert.unwrap_or(0), "draws": post.n_draws(), "level": post.level, "states": post.index.len() })
}

async fn transition(State(s): State<AppState>, Path(snapshot): Path<String>, Query(q): Query<Q>) -> Response {
    let (from, to) = match (required(&q.from, "from"), required(&q.to, "to")) {
        (Ok(a), Ok(b)) => (a.to_owned(), b.to_owned()),
        (Err(e), _) | (_, Err(e)) => return e,
    };
    let k = q.k.unwrap_or(1).min(MAX_K);
    let (post, _) = match posterior(&s, &snapshot, &q, &format!("t{from}{to}{k}")).await {
        Ok(p) => p,
        Err(r) => return r,
    };
    match post.transition_prob(&from, &to, k) {
        Some(d) => Json(json!({ "from": from, "to": to, "k": k, "prob": d, "context": ctx(&snapshot, &q, &post) }))
            .into_response(),
        None => unknown_region(if post.position(&from).is_none() { &from } else { &to }),
    }
}

async fn k_step(State(s): State<AppState>, Path(snapshot): Path<String>, Query(q): Query<Q>) -> Response {
    let from = match required(&q.from, "from") {
        Ok(f) => f.to_owned(),
        Err(r) => return r,
    };
    let k = q.k.unwrap_or(1).min(MAX_K);
    let (post, _) = match posterior(&s, &snapshot, &q, &format!("k{from}{k}")).await {
        Ok(p) => p,
        Err(r) => return r,
    };
    if post.position(&from).is_none() {
        return unknown_region(&from);
    }
    let dist: Vec<Value> = post
        .k_step(&from, k)
        .into_iter()
        .take(q.n.unwrap_or(64).max(1))
        .map(|(r, d)| json!({ "region_id": r, "prob": d }))
        .collect();
    Json(json!({ "from": from, "k": k, "distribution": dist, "context": ctx(&snapshot, &q, &post) })).into_response()
}

async fn neighbors(State(s): State<AppState>, Path(snapshot): Path<String>, Query(q): Query<Q>) -> Response {
    let region = match required(&q.region, "region") {
        Ok(f) => f.to_owned(),
        Err(r) => return r,
    };
    let (post, _) = match posterior(&s, &snapshot, &q, &format!("n{region}")).await {
        Ok(p) => p,
        Err(r) => return r,
    };
    if post.position(&region).is_none() {
        return unknown_region(&region);
    }
    let nb: Vec<Value> = post
        .neighbors(&region, q.n.unwrap_or(10).max(1))
        .into_iter()
        .map(|(r, d)| json!({ "region_id": r, "p_transition": d.mean, "uncertainty": d.sd, "prob": d }))
        .collect();
    Json(json!({ "region": region, "neighbors": nb, "context": ctx(&snapshot, &q, &post) })).into_response()
}

async fn paths(State(s): State<AppState>, Path(snapshot): Path<String>, Query(q): Query<Q>) -> Response {
    let from = match required(&q.from, "from") {
        Ok(f) => f.to_owned(),
        Err(r) => return r,
    };
    let k = q.k.unwrap_or(5).min(MAX_K) as usize;
    let n = q.n.unwrap_or(100).clamp(1, MAX_PATHS);
    let (post, mut rng) = match posterior(&s, &snapshot, &q, &format!("p{from}{k}{n}")).await {
        Ok(p) => p,
        Err(r) => return r,
    };
    if post.position(&from).is_none() {
        return unknown_region(&from);
    }
    let paths = post.sample_paths(&from, k, n, &mut rng);
    let mut ends: HashMap<&str, usize> = HashMap::new();
    for p in &paths {
        *ends.entry(p.last().map_or("", String::as_str)).or_default() += 1;
    }
    let mut ends: Vec<(&str, usize)> = ends.into_iter().collect();
    ends.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let ends: Vec<Value> =
        ends.into_iter().map(|(r, c)| json!({ "region_id": r, "share": c as f64 / n as f64 })).collect();
    Json(
        json!({ "from": from, "k": k, "paths": paths, "end_distribution": ends, "context": ctx(&snapshot, &q, &post) }),
    )
    .into_response()
}

async fn path_prob(State(s): State<AppState>, Path(snapshot): Path<String>, Query(q): Query<Q>) -> Response {
    let path = match required(&q.path, "path") {
        Ok(p) => list(p),
        Err(r) => return r,
    };
    if path.len() < 2 {
        return err(StatusCode::BAD_REQUEST, "bad_request", "`path` needs at least two regions (comma separated)");
    }
    let (post, _) = match posterior(&s, &snapshot, &q, &format!("pp{}", path.join(","))).await {
        Ok(p) => p,
        Err(r) => return r,
    };
    match post.path_prob(&path) {
        Some(d) => Json(json!({ "path": path, "prob": d, "context": ctx(&snapshot, &q, &post) })).into_response(),
        None => unknown_region(path.iter().find(|r| post.position(r).is_none()).map_or("", String::as_str)),
    }
}

async fn hitting(State(s): State<AppState>, Path(snapshot): Path<String>, Query(q): Query<Q>) -> Response {
    let (from, targets) = match (required(&q.from, "from"), required(&q.targets, "targets")) {
        (Ok(a), Ok(b)) => (a.to_owned(), list(b)),
        (Err(e), _) | (_, Err(e)) => return e,
    };
    let horizon = q.horizon.unwrap_or(20).clamp(1, MAX_K);
    let (post, _) = match posterior(&s, &snapshot, &q, &format!("h{from}{}{horizon}", targets.join(","))).await {
        Ok(p) => p,
        Err(r) => return r,
    };
    if post.position(&from).is_none() {
        return unknown_region(&from);
    }
    match post.hitting(&from, &targets, horizon) {
        Some(h) => {
            Json(json!({ "from": from, "targets": targets, "hitting": h, "context": ctx(&snapshot, &q, &post) }))
                .into_response()
        }
        None => unknown_region(&targets.join(",")),
    }
}

async fn stationary(State(s): State<AppState>, Path(snapshot): Path<String>, Query(q): Query<Q>) -> Response {
    let (post, _) = match posterior(&s, &snapshot, &q, "stationary").await {
        Ok(p) => p,
        Err(r) => return r,
    };
    let st: Vec<Value> = post
        .stationary()
        .into_iter()
        .take(q.n.unwrap_or(256).max(1))
        .map(|(r, d)| json!({ "region_id": r, "prob": d }))
        .collect();
    let relax = post.relaxation(1e4);
    Json(json!({ "stationary": st, "relaxation_steps": relax, "context": ctx(&snapshot, &q, &post) })).into_response()
}

async fn kernels(State(s): State<AppState>, Path(snapshot): Path<String>) -> Response {
    match s.hub.snapshot_kernels(&snapshot).await {
        Ok(rows) => {
            let rows: Vec<Value> = rows
                .into_iter()
                .map(|r| {
                    let mut p = Value::Object(r.properties);
                    // JSON columns come back as strings: inline them
                    for key in ["scores", "stationary"] {
                        if let Some(v) =
                            p.get(key).and_then(Value::as_str).and_then(|x| serde_json::from_str::<Value>(x).ok())
                        {
                            p[key] = v;
                        }
                    }
                    p
                })
                .collect();
            Json(json!({ "snapshot": snapshot, "kernels": rows })).into_response()
        }
        Err(e) => hub_err(e),
    }
}

async fn hierarchy(State(s): State<AppState>, Path(snapshot): Path<String>) -> Response {
    match s.hub.snapshot_by_id(&snapshot).await {
        Ok(Some(_)) => {}
        Ok(None) => {
            return (StatusCode::NOT_FOUND, Json(json!({ "error": "not_found", "label": "Snapshot", "id": snapshot })))
                .into_response();
        }
        Err(e) => return hub_err(e),
    }
    match s.hub.snapshot_hierarchy(&snapshot).await {
        Ok(edges) => {
            let edges: Vec<Value> = edges.into_iter().map(|e| Value::Object(e.properties)).collect();
            Json(json!({ "snapshot": snapshot, "edges": edges })).into_response()
        }
        Err(e) => hub_err(e),
    }
}

#[derive(Debug, Deserialize)]
pub struct CrossQ {
    pub timescale: Option<String>,
    pub limit: Option<usize>,
}

async fn cross_priors(State(s): State<AppState>, Query(q): Query<CrossQ>) -> Response {
    let ts = q.timescale.as_deref().filter(|t| !t.is_empty()).unwrap_or("1m");
    match s.hub.cross_priors(ts, q.limit.unwrap_or(50).clamp(1, 500)).await {
        Ok(rows) => {
            let rows: Vec<Value> = rows.into_iter().map(|r| Value::Object(r.properties)).collect();
            Json(json!({ "timescale": ts, "proposals": rows })).into_response()
        }
        Err(e) => hub_err(e),
    }
}

#[derive(Debug, Deserialize)]
pub struct LineageQ {
    pub depth: Option<usize>,
}

async fn lineage(State(s): State<AppState>, Path(region_id): Path<String>, Query(q): Query<LineageQ>) -> Response {
    match s.hub.region_lineage(&region_id, q.depth.unwrap_or(4).clamp(1, 16)).await {
        Ok(Some(l)) => Json(json!(l)).into_response(),
        Ok(None) => unknown_region(&region_id),
        Err(e) => hub_err(e),
    }
}

#[derive(Debug, Deserialize)]
pub struct ReconQ {
    /// Episodes (default 20, max 100).
    pub limit: Option<usize>,
    /// Bars per episode (default 240, max 1 440; the newest are kept).
    pub max_bars: Option<usize>,
}

/// Unix ms of an episode time property (Helix returns RFC 3339 text or a number).
fn episode_ms(v: Option<&Value>) -> Option<i64> {
    match v? {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(t) => chrono::DateTime::parse_from_rfc3339(t).ok().map(|d| d.timestamp_millis()),
        _ => None,
    }
}

/// `GET /api/latent/regions/{region_id}/reconstructions`: what the market did while instruments
/// sat in the region. `{region_id, episodes[{episode (the RegionEpisode row), instrument_id,
/// t_enter_ms, t_exit_ms, source: "bars"|"trades"|"none", bars[{ts_ms, open, high, low, close,
/// volume}], truncated, error?}], decoder_outputs: null, note}`. Bars are 1 m buckets: connector
/// `fa_bar_*` when the instrument has them, else the last trade price per minute (OHLC equal).
/// Decoder outputs per region are not recorded anywhere, so none are returned. 503 when Helix
/// is unreachable; a VictoriaMetrics failure is reported per episode.
async fn reconstructions(State(s): State<AppState>, Path(region_id): Path<String>, Query(q): Query<ReconQ>) -> Response {
    let limit = q.limit.unwrap_or(20).clamp(1, 100);
    let max_bars = q.max_bars.unwrap_or(240).clamp(1, 1_440);
    let rows = match s.hub.region_episodes(&region_id, limit).await {
        Ok(r) => r,
        Err(e) => return hub_err(e),
    };
    let tsdb = fa_tsdb::Tsdb::new(s.cfg.victoriametrics.url.clone());
    let mut episodes = Vec::with_capacity(rows.len());
    for row in rows {
        let p = &row.properties;
        let inst = p.get("instrument_id").and_then(Value::as_str).unwrap_or_default().to_owned();
        let (t0, t1) = (episode_ms(p.get("t_enter")), episode_ms(p.get("t_exit")));
        let mut ep = json!({ "episode": row, "instrument_id": inst, "t_enter_ms": t0, "t_exit_ms": t1,
            "source": "none", "bars": [], "truncated": false });
        let (Some(t0), Some(t1)) = (t0, t1) else {
            ep["error"] = json!("episode without t_enter / t_exit");
            episodes.push(ep);
            continue;
        };
        if inst.is_empty() {
            episodes.push(ep);
            continue;
        }
        // whole minutes covering the episode
        let (start_s, end_s) = (t0.div_euclid(60_000) * 60, (t1 + 59_999).div_euclid(60_000) * 60);
        let mut bars: Vec<Value> = Vec::new();
        match tsdb.bars(Some(&inst), "1m", start_s, end_s).await {
            Ok(series) => {
                if let Some(sr) = series.into_iter().next() {
                    ep["source"] = json!("bars");
                    bars = sr.points.iter().map(|b| json!({ "ts_ms": b.ts_ms, "open": b.open, "high": b.high,
                        "low": b.low, "close": b.close, "volume": b.volume })).collect();
                }
            }
            Err(e) => ep["error"] = json!(format!("victoriametrics: {e}")),
        }
        if bars.is_empty() && ep.get("error").is_none() {
            match tsdb.trade_prices_ms(&inst, t0, t1, fa_core::Timescale::Minute).await {
                Ok(pts) if !pts.is_empty() => {
                    ep["source"] = json!("trades");
                    bars = pts.iter().map(|(t, p)| json!({ "ts_ms": t, "open": p, "high": p, "low": p,
                        "close": p, "volume": null })).collect();
                }
                Ok(_) => {}
                Err(e) => ep["error"] = json!(format!("victoriametrics: {e}")),
            }
        }
        if bars.len() > max_bars {
            bars.drain(..bars.len() - max_bars);
            ep["truncated"] = json!(true);
        }
        ep["bars"] = Value::Array(bars);
        episodes.push(ep);
    }
    Json(json!({
        "region_id": region_id,
        "episodes": episodes,
        "decoder_outputs": null,
        "note": "per-region decoder outputs are not recorded; episodes come from RegionEpisode rows (model-runtime), market paths from VictoriaMetrics 1m bars or trade prices",
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use fa_core::config::PlatformConfig;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    fn app() -> Router {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let mut cfg = PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        cfg.helix.url = "http://127.0.0.1:1".into();
        let state = AppState::new(cfg, format!("{root}/sources.yaml").into(), None);
        let state = AppState { hub: fa_hub::HubClient::new("http://127.0.0.1:1"), ..state };
        router().with_state(state)
    }

    async fn get_json(uri: &str) -> (StatusCode, Value) {
        let res = app().oneshot(Request::get(uri).body(Body::empty()).unwrap()).await.unwrap();
        let status = res.status();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&body).unwrap())
    }

    #[tokio::test]
    async fn bad_params_are_400_before_helix_and_helix_down_is_503() {
        for uri in [
            "/api/mcmc/s/transition?from=a",
            "/api/mcmc/s/k-step",
            "/api/mcmc/s/neighbors",
            "/api/mcmc/s/paths",
            "/api/mcmc/s/path-prob?path=a",
            "/api/mcmc/s/hitting?from=a",
        ] {
            let (st, v) = get_json(uri).await;
            assert_eq!(st, StatusCode::BAD_REQUEST, "{uri}: {v}");
        }
        for uri in [
            "/api/mcmc/s/transition?from=a&to=b",
            "/api/mcmc/s/stationary",
            "/api/mcmc/s/kernels",
            "/api/mcmc/s/hierarchy",
            "/api/mcmc/cross-priors?timescale=1m",
            "/api/latent/regions/r1/lineage",
        ] {
            let (st, v) = get_json(uri).await;
            assert_eq!(st, StatusCode::SERVICE_UNAVAILABLE, "{uri}: {v}");
            assert_eq!(v["error"], "helix_unavailable");
        }
    }

    #[test]
    fn seeds_are_stable_and_lists_parse() {
        let q = Q::default();
        assert_eq!(seed_of(&q, "x"), seed_of(&q, "x"));
        assert_ne!(seed_of(&q, "x"), seed_of(&q, "y"));
        assert_eq!(seed_of(&Q { seed: Some(3), ..Q::default() }, "x"), 3);
        assert_eq!(list(" a, b,,c "), vec!["a", "b", "c"]);
    }
}
