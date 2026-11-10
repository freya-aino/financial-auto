//! Latent routes of spec 50 §4.5 that were still 501: experts of a snapshot, the 2D/3D
//! projection of its stored latents, snapshot promote/rollback and the `/ws/latent` stream.
//!
//! | Route | Source |
//! |---|---|
//! | `GET /api/latent/{snapshot}/experts` | Helix `ExpertSpace` rows (+ region counts) joined with the bus-fed routing stats and norm covariance of `live::SnapshotStats` |
//! | `GET /api/latent/{snapshot}/projection?expert=&range=&n=&dims=&kind=` | Helix `LatentVec` rows of the expert space, PCA (power iteration) server-side, cached 30 s |
//! | `POST /api/latent/snapshots/{id}/promote\|rollback` | `HubClient::set_snapshot_status` |
//! | `WS /ws/latent?snapshot=&instruments=` | `LatentState` changes from `live`, latest value per instrument, ≤ `runtime.ws.latent_hz` |
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

use axum::{
    Json, Router,
    extract::{Path, Query, State, ws::WebSocketUpgrade},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use chrono::Utc;
use fa_hub::HubError;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    config_routes::audit,
    ledger_routes::parse_since,
    state::AppState,
    ws_routes::{self, Coalescer},
};

pub const DEFAULT_POINTS: usize = 2000;
pub const MAX_POINTS: usize = 20_000;
const CACHE_TTL: Duration = Duration::from_secs(30);

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/latent/{snapshot}/experts", get(experts))
        .route("/api/latent/{snapshot}/projection", get(projection))
        .route("/api/latent/snapshots/{id}/{action}", post(snapshot_action))
        .route("/ws/latent", get(ws_latent))
}

fn hub_error(e: HubError) -> Response {
    let (status, body) = if e.is_unreachable() {
        (StatusCode::SERVICE_UNAVAILABLE, json!({ "error": "helix_unavailable", "detail": e.to_string() }))
    } else {
        (StatusCode::BAD_GATEWAY, json!({ "error": "helix_error", "detail": e.to_string() }))
    };
    (status, Json(body)).into_response()
}

fn not_found(label: &str, id: &str) -> Response {
    (StatusCode::NOT_FOUND, Json(json!({ "error": "not_found", "label": label, "id": id }))).into_response()
}

fn bad_request(detail: impl Into<String>) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({ "error": "bad_request", "detail": detail.into() }))).into_response()
}

/// Expert-space business id as model-runtime writes it (`<snapshot>-e<k>`).
pub fn expert_space_id(snapshot: &str, expert: u32) -> String {
    format!("{snapshot}-e{expert}")
}

// ------------------------------------------------------------- experts

async fn experts(State(s): State<AppState>, Path(snapshot): Path<String>) -> Response {
    let live_stats = s.live().snapshot_stats(&snapshot).map(|st| st.view());
    match s.hub.snapshot_graph(&snapshot).await {
        Ok(Some(g)) => {
            let mut regions_per_space: HashMap<&str, (usize, usize)> = HashMap::new();
            for r in &g.regions {
                let e = regions_per_space.entry(r.expert_space_id.as_str()).or_default();
                e.0 += 1;
                if r.status.as_deref().unwrap_or("live") == "live" {
                    e.1 += 1;
                }
            }
            let live_experts = live_stats.as_ref().and_then(|v| v["experts"].as_array().cloned()).unwrap_or_default();
            let mut spaces: Vec<_> = g.spaces.iter().collect();
            spaces.sort_by_key(|sp| sp.index);
            let experts: Vec<Value> = spaces
                .iter()
                .map(|sp| {
                    let (regions, live) = regions_per_space.get(sp.expert_space_id.as_str()).copied().unwrap_or((0, 0));
                    let routing = live_experts
                        .iter()
                        .find(|e| e["expert_index"].as_u64() == Some(u64::from(sp.index)))
                        .cloned()
                        .unwrap_or(Value::Null);
                    json!({
                        "expert_index": sp.index,
                        "expert_space_id": sp.expert_space_id,
                        "dim": sp.dim,
                        "helix_id": sp.id,
                        "regions": regions,
                        "regions_live": live,
                        "routing": routing,
                    })
                })
                .collect();
            Json(json!({
                "snapshot": g.snapshot,
                "experts": experts,
                "covariance": live_stats.as_ref().map(|v| v["covariance"].clone()),
                "live": live_stats.as_ref().map(|v| json!({ "messages": v["messages"], "last_as_of": v["last_as_of"] })),
                "source": "helix+live",
            }))
            .into_response()
        }
        Ok(None) => match live_stats {
            Some(v) => Json(json!({
                "snapshot": { "snapshot_id": snapshot, "timescale": v["timescale"] },
                "experts": v["experts"].as_array().map(|a| a.iter().map(|e| json!({
                    "expert_index": e["expert_index"], "expert_space_id": expert_space_id(&snapshot, e["expert_index"].as_u64().unwrap_or(0) as u32),
                    "dim": e["dims"], "regions": e["regions_seen"], "routing": e,
                })).collect::<Vec<_>>()),
                "covariance": v["covariance"],
                "live": { "messages": v["messages"], "last_as_of": v["last_as_of"] },
                "source": "live",
                "helix": "snapshot not persisted",
            }))
            .into_response(),
            None => not_found("Snapshot", &snapshot),
        },
        Err(e) => match live_stats {
            Some(v) if e.is_unreachable() => Json(json!({
                "snapshot": { "snapshot_id": snapshot, "timescale": v["timescale"] },
                "experts": v["experts"],
                "covariance": v["covariance"],
                "live": { "messages": v["messages"], "last_as_of": v["last_as_of"] },
                "source": "live",
                "helix": { "error": "helix_unavailable", "detail": e.to_string() },
            }))
            .into_response(),
            _ => hub_error(e),
        },
    }
}

// ---------------------------------------------------------- projection

#[derive(Debug, Deserialize)]
pub struct ProjectionQuery {
    /// Expert index (default 0) or a full `expert_space_id`.
    pub expert: Option<String>,
    /// `<n>m|h|d`, RFC 3339 or `all` (default) — lower bound on `ts`.
    pub range: Option<String>,
    /// Points fetched (newest first); default 2000, max 20 000.
    pub n: Option<usize>,
    /// 2 or 3 (default 2).
    pub dims: Option<usize>,
    /// `centroid | observation | window`; default all.
    pub kind: Option<String>,
}

/// Result of a PCA: `components[j]` is the j-th principal axis (unit length, largest-magnitude
/// coordinate positive), `explained[j]` its share of total variance.
#[derive(Debug, Clone, PartialEq)]
pub struct Pca {
    pub mean: Vec<f64>,
    pub components: Vec<Vec<f64>>,
    pub explained: Vec<f64>,
    pub total_variance: f64,
}

impl Pca {
    pub fn project(&self, x: &[f32]) -> Vec<f64> {
        self.components
            .iter()
            .map(|c| c.iter().zip(x).enumerate().map(|(i, (ci, xi))| ci * (f64::from(*xi) - self.mean[i])).sum())
            .collect()
    }
}

/// Deterministic PCA by power iteration with deflation (pure Rust, no dependencies): builds the
/// `d × d` covariance of `points`, then extracts `k` eigenvectors from a fixed start vector.
/// Points shorter than `d` are zero-padded. Returns `None` for fewer than two points.
pub fn pca(points: &[Vec<f32>], k: usize) -> Option<Pca> {
    let n = points.len();
    let d = points.iter().map(Vec::len).max().unwrap_or(0);
    if n < 2 || d == 0 {
        return None;
    }
    let k = k.min(d);
    let get = |p: &Vec<f32>, i: usize| p.get(i).copied().map_or(0.0, f64::from);
    let mean: Vec<f64> = (0..d).map(|i| points.iter().map(|p| get(p, i)).sum::<f64>() / n as f64).collect();
    let mut cov = vec![vec![0.0f64; d]; d];
    for p in points {
        let dev: Vec<f64> = (0..d).map(|i| get(p, i) - mean[i]).collect();
        for (row, di) in cov.iter_mut().zip(&dev) {
            for (cell, dj) in row.iter_mut().zip(&dev) {
                *cell += di * dj;
            }
        }
    }
    let denom = (n - 1) as f64;
    for row in &mut cov {
        for cell in row.iter_mut() {
            *cell /= denom;
        }
    }
    let total_variance: f64 = (0..d).map(|i| cov[i][i]).sum();
    let mut components = Vec::with_capacity(k);
    let mut explained = Vec::with_capacity(k);
    for _ in 0..k {
        // fixed, non-symmetric start so no eigenvector is orthogonal to it by construction
        let mut v: Vec<f64> = (0..d).map(|i| 1.0 + i as f64 / d as f64).collect();
        normalise(&mut v);
        let mut lambda = 0.0;
        for _ in 0..500 {
            let mut w: Vec<f64> = (0..d).map(|i| (0..d).map(|j| cov[i][j] * v[j]).sum()).collect();
            let norm = normalise(&mut w);
            if norm < 1e-300 {
                break;
            }
            let delta: f64 = w.iter().zip(&v).map(|(a, b)| (a - b).abs()).sum();
            v = w;
            lambda = norm;
            if delta < 1e-12 {
                break;
            }
        }
        // sign convention: largest-magnitude coordinate positive
        if let Some(m) = v.iter().copied().max_by(|a, b| a.abs().total_cmp(&b.abs()))
            && m < 0.0
        {
            v.iter_mut().for_each(|x| *x = -*x);
        }
        for (row, vi) in cov.iter_mut().zip(&v) {
            for (cell, vj) in row.iter_mut().zip(&v) {
                *cell -= lambda * vi * vj;
            }
        }
        explained.push(if total_variance > 0.0 { lambda / total_variance } else { 0.0 });
        components.push(v);
    }
    Some(Pca { mean, components, explained, total_variance })
}

fn normalise(v: &mut [f64]) -> f64 {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    norm
}

type Cache = Mutex<HashMap<String, (Instant, Value)>>;

fn cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(Mutex::default)
}

fn cache_get(key: &str) -> Option<Value> {
    let mut c = cache().lock().expect("projection cache");
    c.retain(|_, (at, _)| at.elapsed() < CACHE_TTL);
    c.get(key).map(|(_, v)| v.clone())
}

fn cache_put(key: String, v: Value) {
    cache().lock().expect("projection cache").insert(key, (Instant::now(), v));
}

async fn projection(
    State(s): State<AppState>,
    Path(snapshot): Path<String>,
    Query(q): Query<ProjectionQuery>,
) -> Response {
    let expert = q.expert.as_deref().map(str::trim).filter(|e| !e.is_empty()).unwrap_or("0");
    let space_id = if let Ok(k) = expert.parse::<u32>() { expert_space_id(&snapshot, k) } else { expert.to_owned() };
    let range = q.range.as_deref().unwrap_or("all");
    let since = match parse_since(range, Utc::now()) {
        Ok(t) => t,
        Err(e) => return bad_request(e),
    };
    let n = q.n.unwrap_or(DEFAULT_POINTS).clamp(1, MAX_POINTS);
    let dims = q.dims.unwrap_or(2);
    if !(2..=3).contains(&dims) {
        return bad_request("dims must be 2 or 3");
    }
    let kind = q.kind.as_deref().map(str::trim).filter(|k| !k.is_empty());
    let key = format!("{snapshot}|{space_id}|{range}|{n}|{dims}|{}", kind.unwrap_or(""));
    if let Some(mut v) = cache_get(&key) {
        v["cached"] = json!(true);
        return Json(v).into_response();
    }
    let space = match s.hub.space_by_id(&space_id).await {
        Ok(Some(sp)) => sp,
        Ok(None) => return not_found("ExpertSpace", &space_id),
        Err(e) => return hub_error(e),
    };
    let since_ms = since.map(|t| t.timestamp_millis());
    let (rows, total) =
        match tokio::try_join!(s.hub.space_latents(&space_id, kind, since_ms, n), s.hub.count_space_latents(&space_id))
        {
            Ok(x) => x,
            Err(e) => return hub_error(e),
        };
    let vectors: Vec<Vec<f32>> = rows.iter().filter_map(fa_hub::LatentRow::vector).collect();
    let p = pca(&vectors, dims);
    let points: Vec<Value> = rows
        .iter()
        .filter_map(|r| {
            let v = r.vector()?;
            let coords = p.as_ref().map(|p| p.project(&v)).unwrap_or_else(|| vec![0.0; dims]);
            Some(json!({
                "id": r.id, "coords": coords, "region_id": r.ref_id, "kind": r.kind,
                "instrument_id": r.instrument_id, "ts": r.ts, "uncertainty": r.uncertainty,
            }))
        })
        .collect();
    let body = json!({
        "snapshot_id": snapshot,
        "expert_space_id": space_id,
        "expert_index": space.index,
        "dim": space.dim,
        "range": { "from": since, "spec": range },
        "requested": n,
        "fetched": rows.len(),
        "total_in_space": total,
        "dims": dims,
        "pca": p.as_ref().map(|p| json!({
            "method": "power iteration with deflation (deterministic)",
            "mean": p.mean, "components": p.components, "explained_variance_ratio": p.explained,
            "total_variance": p.total_variance,
        })),
        "points": points,
        "cached": false,
        "cache_ttl_secs": CACHE_TTL.as_secs(),
    });
    cache_put(key, body.clone());
    Json(body).into_response()
}

// ----------------------------------------------------- promote/rollback

/// `promote`: this snapshot → `active`, every other active snapshot of the same timescale →
/// `retired`. `rollback`: this snapshot → `retired`; the newest other retired snapshot of the
/// timescale (by `created_at`) → `active` when there is one.
async fn snapshot_action(State(s): State<AppState>, Path((id, action)): Path<(String, String)>) -> Response {
    if !matches!(action.as_str(), "promote" | "rollback") {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "unknown action", "action": action, "use": "promote|rollback" })),
        )
            .into_response();
    }
    let snaps = match s.hub.list_snapshots().await {
        Ok(v) => v,
        Err(e) => return hub_error(e),
    };
    let Some(target) = snaps.iter().find(|sn| sn.snapshot_id == id) else {
        return not_found("Snapshot", &id);
    };
    let mut changes: Vec<(u64, String, &str)> = Vec::new();
    if action == "promote" {
        if target.status == "active" {
            return (
                StatusCode::CONFLICT,
                Json(json!({ "error": "conflict", "detail": "snapshot is already active" })),
            )
                .into_response();
        }
        changes.push((target.id, target.snapshot_id.clone(), "active"));
        for other in
            snaps.iter().filter(|o| o.id != target.id && o.timescale == target.timescale && o.status == "active")
        {
            changes.push((other.id, other.snapshot_id.clone(), "retired"));
        }
    } else {
        if target.status == "retired" {
            return (
                StatusCode::CONFLICT,
                Json(json!({ "error": "conflict", "detail": "snapshot is already retired" })),
            )
                .into_response();
        }
        changes.push((target.id, target.snapshot_id.clone(), "retired"));
        if target.status == "active"
            && let Some(prev) = snaps
                .iter()
                .filter(|o| o.id != target.id && o.timescale == target.timescale && o.status == "retired")
                .max_by(|a, b| a.created_at.cmp(&b.created_at))
        {
            changes.push((prev.id, prev.snapshot_id.clone(), "active"));
        }
    }
    let mut applied = Vec::new();
    for (hid, sid, status) in &changes {
        match s.hub.set_snapshot_status(*hid, status).await {
            Ok(row) => applied.push(json!({ "snapshot_id": sid, "status": row.status })),
            Err(e) => return hub_error(e),
        }
    }
    let audited = audit(&s, &format!("latent.snapshot.{action}"), &id, json!({ "changes": applied })).await;
    Json(json!({ "ok": true, "snapshot_id": id, "action": action, "changes": applied, "audited": audited }))
        .into_response()
}

// ------------------------------------------------------------ ws/latent

#[derive(Debug, Deserialize)]
pub struct LatentWsQuery {
    pub snapshot: Option<String>,
    /// Comma-separated instrument ids; empty = all.
    pub instruments: Option<String>,
}

async fn ws_latent(ws: WebSocketUpgrade, State(state): State<AppState>, Query(q): Query<LatentWsQuery>) -> Response {
    let snapshot = q.snapshot.unwrap_or_default().trim().to_owned();
    let instruments: Vec<String> = q
        .instruments
        .iter()
        .flat_map(|s| s.split(','))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    ws.on_upgrade(move |socket| async move {
        let (initial, events, hz) = {
            let live = state.live();
            let initial = if snapshot.is_empty() { Vec::new() } else { live.snapshot_states(&snapshot, &instruments) };
            (initial, live.subscribe(), live.runtime().ws.latent_hz)
        };
        let source = ws_routes::spawn_live_source(events, "latent", move |ev| {
            let Some(rest) = ev.topic.strip_prefix("state.") else {
                return snapshot.is_empty() || ev.topic.contains(&format!(".{snapshot}."));
            };
            let Some((snap, inst)) = rest.split_once('.') else { return false };
            (snapshot.is_empty() || snap == snapshot)
                && (instruments.is_empty() || instruments.iter().any(|i| i == inst))
        });
        ws_routes::serve(socket, "latent", initial, source, Coalescer::latest_value(), ws_routes::flush_interval(hz))
            .await;
    })
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    /// Synthetic anisotropic cloud: variance 9 along `u`, 1 along `w`, 0.01 along the rest.
    fn cloud(n: usize, d: usize) -> (Vec<Vec<f32>>, Vec<f64>, Vec<f64>) {
        let mut u: Vec<f64> = (0..d).map(|i| ((i * 7 + 3) % 11) as f64 - 5.0).collect();
        normalise(&mut u);
        let mut w: Vec<f64> = (0..d).map(|i| if i % 2 == 0 { 1.0 } else { -0.5 }).collect();
        let dot: f64 = u.iter().zip(&w).map(|(a, b)| a * b).sum();
        w.iter_mut().zip(&u).for_each(|(wi, ui)| *wi -= dot * ui);
        normalise(&mut w);
        let mut rng = fa_core::rng::Rng::seed(42);
        let mut pts = Vec::with_capacity(n);
        for _ in 0..n {
            let a = 3.0 * rng.normal();
            let b = rng.normal();
            let p: Vec<f32> = (0..d).map(|i| (10.0 + a * u[i] + b * w[i] + 0.1 * rng.normal()) as f32).collect();
            pts.push(p);
        }
        (pts, u, w)
    }

    #[test]
    fn pca_recovers_principal_axes_of_an_anisotropic_cloud() {
        let (pts, u, w) = cloud(3000, 6);
        let p = pca(&pts, 3).unwrap();
        assert_eq!(p.components.len(), 3);
        let cos = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>().abs();
        assert!(cos(&p.components[0], &u) > 0.99, "first axis {:?} vs {:?}", p.components[0], u);
        assert!(cos(&p.components[1], &w) > 0.98, "second axis {:?} vs {:?}", p.components[1], w);
        assert!(cos(&p.components[0], &p.components[1]) < 1e-6, "orthogonal");
        assert!(p.explained[0] > 0.85 && p.explained[0] < 0.95, "{:?}", p.explained);
        assert!(p.explained[1] > 0.08 && p.explained[1] < 0.12, "{:?}", p.explained);
        assert!(p.explained[2] < 0.01);
        for (i, m) in p.mean.iter().enumerate() {
            assert!((m - 10.0).abs() < 0.2, "mean[{i}] = {m}");
        }
        // deterministic
        assert_eq!(pca(&pts, 3).unwrap(), p);
        // projection of the mean is the origin; sign convention is stable
        let origin = p.project(&p.mean.iter().map(|m| *m as f32).collect::<Vec<_>>());
        assert!(origin.iter().all(|c| c.abs() < 1e-4), "f32 round-trip of the mean: {origin:?}");
        for c in &p.components {
            let m = c.iter().copied().max_by(|a, b| a.abs().total_cmp(&b.abs())).unwrap();
            assert!(m > 0.0);
        }
    }

    #[test]
    fn pca_edge_cases() {
        assert!(pca(&[], 2).is_none());
        assert!(pca(&[vec![1.0, 2.0]], 2).is_none());
        // ragged input is zero-padded; k is clamped to the dimension
        let p = pca(&[vec![1.0], vec![2.0, 1.0], vec![3.0, 2.0]], 5).unwrap();
        assert_eq!(p.components.len(), 2);
        assert_eq!(p.mean.len(), 2);
        // constant cloud: zero variance, no NaNs
        let p = pca(&[vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]], 2).unwrap();
        assert_eq!(p.total_variance, 0.0);
        assert!(p.explained.iter().all(|e| *e == 0.0));
        assert!(p.components.iter().flatten().all(|x| x.is_finite()));
    }

    fn app() -> Router {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let cfg = fa_core::config::PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        let state = AppState::new(cfg, format!("{root}/sources.yaml").into(), None);
        let state = AppState { hub: fa_hub::HubClient::new("http://127.0.0.1:1"), ..state };
        router().with_state(state)
    }

    async fn send(req: Request<Body>) -> (StatusCode, Value) {
        let res = app().oneshot(req).await.unwrap();
        let status = res.status();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
    }

    #[tokio::test]
    async fn helix_down_and_bad_params() {
        let (s, v) = send(Request::get("/api/latent/stat-v1-1m/experts").body(Body::empty()).unwrap()).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{v}");
        assert_eq!(v["error"], "helix_unavailable");
        let (s, v) =
            send(Request::get("/api/latent/stat-v1-1m/projection?expert=0").body(Body::empty()).unwrap()).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{v}");
        let (s, v) =
            send(Request::get("/api/latent/stat-v1-1m/projection?range=yesterday").body(Body::empty()).unwrap()).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        let (s, _) = send(Request::get("/api/latent/stat-v1-1m/projection?dims=4").body(Body::empty()).unwrap()).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, _) = send(Request::post("/api/latent/snapshots/x/promote").body(Body::empty()).unwrap()).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
        let (s, _) = send(Request::post("/api/latent/snapshots/x/explode").body(Body::empty()).unwrap()).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert_eq!(expert_space_id("stat-v1-1m", 2), "stat-v1-1m-e2");
    }
}
