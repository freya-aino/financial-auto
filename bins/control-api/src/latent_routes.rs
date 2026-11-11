//! Latent routes of spec 50 §4.5 that were still 501: experts of a snapshot, the 2D/3D
//! projection of its stored latents, snapshot promote/rollback and the `/ws/latent` stream.
//!
//! | Route | Source |
//! |---|---|
//! | `GET /api/latent/{snapshot}/experts` | Helix `ExpertSpace` rows (+ region counts) joined with the bus-fed routing stats and norm covariance of `live::SnapshotStats` |
//! | `GET /api/latent/{snapshot}/projection?expert=&range=&n=&dims=&kind=` | Helix `LatentVec` rows of the expert space, PCA (power iteration) server-side, cached 30 s |
//! | `POST /api/latent/snapshots/{id}/promote\|rollback` | `promote` first evaluates the spec 36 §6.4 checklist ([`promotion_checklist`]: manifest carried mass, shadow serving minutes, shadow region agreement) and answers 409 with it unless every item passes or the body says `{"force": true}`; then `HubClient::set_snapshot_status` and `ctl.model.reload` (`timescale`, `snapshot_id` = the new active) so model-runtime hot-swaps the encoder |
//! | `POST\|DELETE /api/latent/snapshots/{id}/shadow`, `POST …/shadow/stop` | `ctl.model.shadow {timescale, snapshot_id \| "none"}`: model-runtime serves the candidate next to the active snapshot (`latent.shadow.<ts>.<inst>`, heartbeat gauges `shadow.<ts>.*`) |
//! | `GET /api/latent/snapshots/{id}` | Helix row + manifest (Helix mirror, else Garage) + the promotion checklist + latest `AdapterState` |
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
use bytes::Bytes;
use chrono::Utc;
use fa_hub::HubError;
use fa_proto::{
    Message,
    v1::{CtlReply, CtlRequest},
};
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
        .route("/api/latent/snapshots/{id}", get(snapshot_detail))
        .route("/api/latent/snapshots/{id}/shadow", post(shadow_start).delete(shadow_stop))
        .route("/api/latent/snapshots/{id}/shadow/stop", post(shadow_stop))
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
    let mut res = experts_inner(&s, &snapshot).await;
    // spec 32 §4.3: the live expert health model-runtime publishes on `latent.expert.health`
    let health = s.live().expert_health(&snapshot);
    if res.status() == StatusCode::OK
        && let Some(h) = health
    {
        let (parts, body) = res.into_parts();
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap_or_default();
        let mut v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        if let Some(o) = v.as_object_mut() {
            o.insert("health".into(), h);
        }
        res = (parts, Json(v)).into_response();
    }
    res
}

async fn experts_inner(s: &AppState, snapshot: &str) -> Response {
    let snapshot = snapshot.to_owned();
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

// ------------------------------------------------- promotion checklist

/// Thresholds of the enforced spec 36 §6.4 checks, from the environment (fa-core's `[runtime]`
/// config has no promotion section yet): `FA_PROMOTE_MIN_CARRIED_MASS` (default 0.5, spec 36
/// §7 `min_carry_mass`), `FA_PROMOTE_MIN_SHADOW_MINUTES` (30) and
/// `FA_PROMOTE_MIN_REGION_AGREEMENT` (0.3).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct PromotionThresholds {
    pub min_carried_mass: f64,
    pub min_shadow_minutes: f64,
    pub min_region_agreement: f64,
}

impl Default for PromotionThresholds {
    fn default() -> Self {
        Self { min_carried_mass: 0.5, min_shadow_minutes: 30.0, min_region_agreement: 0.3 }
    }
}

impl PromotionThresholds {
    pub fn from_env() -> Self {
        let d = Self::default();
        let get = |k: &str, dflt: f64| {
            std::env::var(k).ok().and_then(|v| v.trim().parse::<f64>().ok()).filter(|v| v.is_finite()).unwrap_or(dflt)
        };
        Self {
            min_carried_mass: get("FA_PROMOTE_MIN_CARRIED_MASS", d.min_carried_mass),
            min_shadow_minutes: get("FA_PROMOTE_MIN_SHADOW_MINUTES", d.min_shadow_minutes),
            min_region_agreement: get("FA_PROMOTE_MIN_REGION_AGREEMENT", d.min_region_agreement),
        }
    }
}

/// Whether model-runtime's counters say `snapshot_id` is the shadow of `timescale`. Heartbeat
/// counters only carry numbers, so model-runtime publishes the shadow's id as the marker
/// `shadow.<ts>.snapshot_id.<id> = 1` next to `shadow.<ts>.snapshot` (= its generation).
pub fn shadow_matches(counters: &std::collections::BTreeMap<String, f64>, timescale: &str, snapshot_id: &str) -> bool {
    counters.get(&format!("shadow.{timescale}.snapshot_id.{snapshot_id}")).is_some_and(|v| *v > 0.0)
}

/// The id model-runtime currently shadows on `timescale`, from the marker counter.
pub fn shadow_id(counters: &std::collections::BTreeMap<String, f64>, timescale: &str) -> Option<String> {
    let prefix = format!("shadow.{timescale}.snapshot_id.");
    counters.iter().find(|(k, v)| k.starts_with(&prefix) && **v > 0.0).map(|(k, _)| k[prefix.len()..].to_owned())
}

/// One item of the checklist.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Check {
    pub name: &'static str,
    pub ok: bool,
    pub value: Option<f64>,
    pub threshold: Option<f64>,
    pub detail: String,
}

/// The spec 36 §6.4 promotion checklist as enforced today: item 3 (carried mass) and the shadow
/// checks for every snapshot; items 1 (`ssl_regression`) and 2 (`expert_k_eff`,
/// `expert_resid_corr`, `expert_min_skill`) are added when the manifest carries the neural
/// metrics ([`manifest_health_checks`]; item 1 is an `n/a` pass without a neural parent;
/// statistical snapshots report both as `not_evaluated`). Items 4 (ECE) and 5 (elite
/// transfer) have no producer yet and are reported as `not_evaluated` entries in the route body,
/// not as checks.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Checklist {
    pub snapshot_id: String,
    pub timescale: String,
    pub passed: bool,
    pub checks: Vec<Check>,
    pub thresholds: PromotionThresholds,
}

/// Pure checklist: `carried_mass` from the candidate's manifest (`None` = no manifest),
/// `counters` = model-runtime's heartbeat counters (`None` = no heartbeat, or stale),
/// `representation_break` waives the carried-mass item (spec 36 §6.4 check 3).
pub fn promotion_checklist(
    snapshot_id: &str,
    timescale: &str,
    carried_mass: Option<f64>,
    counters: Option<&std::collections::BTreeMap<String, f64>>,
    representation_break: bool,
    th: PromotionThresholds,
) -> Checklist {
    let mut checks = Vec::with_capacity(4);
    checks.push(Check {
        name: "carried_mass",
        ok: representation_break || carried_mass.is_some_and(|m| m >= th.min_carried_mass),
        value: carried_mass,
        threshold: Some(th.min_carried_mass),
        detail: match (carried_mass, representation_break) {
            (_, true) => "waived: marked as a representation break".into(),
            (None, _) => "no manifest.metrics.carried_mass (not a trained snapshot, or manifest unreadable)".into(),
            (Some(m), _) => format!("manifest.metrics.carried_mass = {m:.3}"),
        },
    });
    let g = |k: &str| counters.and_then(|c| c.get(&format!("shadow.{timescale}.{k}")).copied());
    let served = counters.is_some_and(|c| shadow_matches(c, timescale, snapshot_id));
    let current = counters.and_then(|c| shadow_id(c, timescale));
    checks.push(Check {
        name: "shadow_served",
        ok: served,
        value: g("snapshot"),
        threshold: None,
        detail: match (counters.is_some(), served, current) {
            (false, ..) => "no fresh model-runtime heartbeat".into(),
            (true, true, _) => {
                format!("model-runtime serves {snapshot_id} as the {timescale} shadow (value = its generation)")
            }
            (true, false, Some(other)) => format!(
                "the {timescale} shadow is {other}, not {snapshot_id} (POST /api/latent/snapshots/{snapshot_id}/shadow)"
            ),
            (true, false, None) => {
                format!("no {timescale} shadow is served (POST /api/latent/snapshots/{snapshot_id}/shadow)")
            }
        },
    });
    let minutes = if served { g("minutes") } else { None };
    checks.push(Check {
        name: "shadow_minutes",
        ok: minutes.is_some_and(|m| m >= th.min_shadow_minutes),
        value: minutes,
        threshold: Some(th.min_shadow_minutes),
        detail: format!("shadow.{timescale}.minutes (only counted while this snapshot is the shadow)"),
    });
    let agreement = if served { g("region_agreement") } else { None };
    checks.push(Check {
        name: "shadow_region_agreement",
        ok: agreement.is_some_and(|a| a >= th.min_region_agreement),
        value: agreement,
        threshold: Some(th.min_region_agreement),
        detail: format!("shadow.{timescale}.region_agreement (live agreement of shadow vs active region lineages)"),
    });
    Checklist {
        snapshot_id: snapshot_id.into(),
        timescale: timescale.into(),
        passed: checks.iter().all(|c| c.ok),
        checks,
        thresholds: th,
    }
}

/// Heartbeats older than this do not count for the shadow checks.
const HEARTBEAT_FRESH: Duration = Duration::from_secs(60);

/// model-runtime's latest heartbeat counters, when fresh.
fn runtime_counters(s: &AppState) -> Option<std::collections::BTreeMap<String, f64>> {
    let services = s.services.read().expect("services lock");
    let st = services.get("model-runtime")?;
    let age = Utc::now().signed_duration_since(st.last_seen).to_std().unwrap_or_default();
    (age <= HEARTBEAT_FRESH).then(|| st.counters.clone())
}

/// The candidate's manifest: the Helix mirror written by the trainer, else Garage.
async fn snapshot_manifest(s: &AppState, row: &fa_hub::Snapshot) -> (Option<Value>, &'static str) {
    if let Some(m) = row.manifest.as_deref().and_then(|m| serde_json::from_str::<Value>(m).ok()) {
        return (Some(m), "helix");
    }
    if row.garage_key.is_empty() {
        return (None, "none");
    }
    let Ok(secrets) = fa_core::Secrets::load() else { return (None, "none") };
    let Ok(blob) = fa_blob::S3BlobStore::from_config(&s.cfg, &secrets) else { return (None, "none") };
    let key = fa_blob::snapshot_file(&row.snapshot_id, "manifest.json");
    match fa_blob::BlobStore::get(&blob, &key).await {
        Ok(b) => (serde_json::from_slice(&b).ok(), "garage"),
        Err(_) => (None, "none"),
    }
}

fn carried_mass_of(row: &fa_hub::Snapshot, manifest: Option<&Value>) -> Option<f64> {
    manifest.and_then(|m| m["metrics"]["carried_mass"].as_f64()).or(row.carried_mass)
}

async fn checklist_for(
    s: &AppState,
    row: &fa_hub::Snapshot,
    representation_break: bool,
) -> (Checklist, Option<Value>, &'static str) {
    let (manifest, source) = snapshot_manifest(s, row).await;
    let counters = runtime_counters(s);
    let mut c = promotion_checklist(
        &row.snapshot_id,
        &row.timescale,
        carried_mass_of(row, manifest.as_ref()),
        counters.as_ref(),
        representation_break,
        PromotionThresholds::from_env(),
    );
    // Items 1 and 2 for snapshots whose trainer measured them (the neural families).
    if let Some(m) = manifest.as_ref() {
        let extra = manifest_health_checks(m);
        if !extra.is_empty() {
            c.checks.extend(extra);
            c.passed = c.checks.iter().all(|x| x.ok);
        }
    }
    (c, manifest, source)
}

/// Spec 36 §6.4 items 1 (`ssl_regression`: next-step skill not worse than the parent by more than
/// `FA_PROMOTE_REGRESS_TOL`, default 0.02) and 2 (expert-space health, spec 32 §6.4: `K_eff ≥
/// FA_PROMOTE_MIN_K_EFF` (2), max off-diagonal residual correlation ≤ `FA_PROMOTE_MAX_RESID_CORR`
/// (0.1), every expert's skill ≥ `FA_PROMOTE_MIN_SKILL` (0.02)) from the validation metrics the
/// trainer writes into `manifest.metrics` for neural snapshots. A metric that is absent yields no
/// check (the statistical family has neither; a neural child of a statistical parent has no
/// `ssl_regression`).
pub fn manifest_health_checks(manifest: &Value) -> Vec<Check> {
    let env = |k: &str, d: f64| {
        std::env::var(k).ok().and_then(|v| v.trim().parse::<f64>().ok()).filter(|v| v.is_finite()).unwrap_or(d)
    };
    let metric = |k: &str| manifest["metrics"][k].as_f64();
    let mut out = Vec::new();
    let tol = env("FA_PROMOTE_REGRESS_TOL", 0.02);
    // a trained (neural) snapshot always reports item 1; without a neural parent there is no
    // SSL objective to regress against, which the check says instead of being silent
    let trained = metric("k_eff").is_some() || metric("mix_skill").is_some();
    if trained && metric("ssl_regression").is_none() {
        let parent = manifest["parent_snapshot_id"].as_str().unwrap_or("");
        out.push(Check {
            name: "ssl_regression",
            ok: true,
            value: None,
            threshold: Some(tol),
            detail: if parent.is_empty() {
                "no parent snapshot: nothing to regress against (n/a)".into()
            } else {
                format!("parent {parent} has no SSL objective (statistical family): nothing to regress against (n/a)")
            },
        });
    }
    if let Some(r) = metric("ssl_regression") {
        out.push(Check {
            name: "ssl_regression",
            ok: r <= tol,
            value: Some(r),
            threshold: Some(tol),
            detail: format!(
                "parent mixture skill {:.3} − candidate {:.3} on the same validation windows",
                metric("parent_mix_skill").unwrap_or(f64::NAN),
                metric("mix_skill").unwrap_or(f64::NAN)
            ),
        });
    }
    for (name, key, th, at_least, what) in [
        (
            "expert_k_eff",
            "k_eff",
            env("FA_PROMOTE_MIN_K_EFF", 2.0),
            true,
            "effective independent experts (spec 32 §6.4)",
        ),
        (
            "expert_resid_corr",
            "resid_corr_max",
            env("FA_PROMOTE_MAX_RESID_CORR", 0.1),
            false,
            "max |corr| of expert residuals on validation",
        ),
        ("expert_min_skill", "skill_min", env("FA_PROMOTE_MIN_SKILL", 0.02), true, "weakest expert's skill vs naive"),
    ] {
        if let Some(v) = metric(key) {
            out.push(Check {
                name,
                ok: if at_least { v >= th } else { v <= th },
                value: Some(v),
                threshold: Some(th),
                detail: format!("manifest.metrics.{key}: {what}"),
            });
        }
    }
    out
}

const NOT_EVALUATED: [&str; 4] = [
    "1 ssl_regression (statistical family / statistical parent: no SSL loss; neural snapshots add the check)",
    "2 expert_space_health (statistical family; neural snapshots add expert_k_eff / expert_resid_corr / expert_min_skill)",
    "4 mcmc_ece (no ECE producer yet)",
    "5 elite_transfer (elites not re-evaluated on candidates yet)",
];

/// Promote body: `{"force": bool, "note": string, "representation_break": bool}`, all optional.
#[derive(Debug, Default, Deserialize)]
pub struct PromoteBody {
    #[serde(default)]
    pub force: bool,
    #[serde(default)]
    pub note: Option<String>,
    #[serde(default)]
    pub representation_break: bool,
}

fn parse_body(body: &Bytes) -> Result<PromoteBody, String> {
    if body.iter().all(u8::is_ascii_whitespace) {
        return Ok(PromoteBody::default());
    }
    serde_json::from_slice(body).map_err(|e| format!("body: {e}"))
}

async fn snapshot_detail(State(s): State<AppState>, Path(id): Path<String>) -> Response {
    let row = match s.hub.snapshot_by_id(&id).await {
        Ok(Some(r)) => r,
        Ok(None) => return not_found("Snapshot", &id),
        Err(e) => return hub_error(e),
    };
    let (checklist, manifest, source) = checklist_for(&s, &row, false).await;
    let adapter = s.hub.latest_adapter(&id).await.ok().flatten();
    let shadow: serde_json::Map<String, Value> = runtime_counters(&s)
        .unwrap_or_default()
        .into_iter()
        .filter(|(k, _)| k.starts_with(&format!("shadow.{}.", row.timescale)))
        .map(|(k, v)| (k, json!(v)))
        .collect();
    let mut snapshot = serde_json::to_value(&row).unwrap_or(Value::Null);
    if let Some(o) = snapshot.as_object_mut() {
        o.remove("manifest");
    }
    // items reported as checks are no longer "not evaluated"
    let has = |n: &str| checklist.checks.iter().any(|c| c.name == n);
    let not_evaluated: Vec<&str> = NOT_EVALUATED
        .iter()
        .copied()
        .filter(|item| {
            !(item.starts_with("1 ") && has("ssl_regression")) && !(item.starts_with("2 ") && has("expert_k_eff"))
        })
        .collect();
    Json(json!({
        "snapshot": snapshot,
        "manifest": manifest,
        "manifest_source": source,
        "checklist": checklist,
        "not_evaluated": not_evaluated,
        "shadow_gauges": shadow,
        "latest_adapter": adapter,
    }))
    .into_response()
}

// ----------------------------------------------------------- shadow

async fn shadow_start(State(s): State<AppState>, Path(id): Path<String>) -> Response {
    let row = match s.hub.snapshot_by_id(&id).await {
        Ok(Some(r)) => r,
        Ok(None) => return not_found("Snapshot", &id),
        Err(e) => return hub_error(e),
    };
    if row.status == "active" {
        return (
            StatusCode::CONFLICT,
            Json(json!({ "error": "conflict", "detail": "snapshot is active; shadow a candidate" })),
        )
            .into_response();
    }
    shadow_request(&s, &row, &id, "start").await
}

async fn shadow_stop(State(s): State<AppState>, Path(id): Path<String>) -> Response {
    let row = match s.hub.snapshot_by_id(&id).await {
        Ok(Some(r)) => r,
        Ok(None) => return not_found("Snapshot", &id),
        Err(e) => return hub_error(e),
    };
    shadow_request(&s, &row, "none", "stop").await
}

/// `ctl.model.shadow {timescale, snapshot_id}`; on success the Helix status follows
/// (`shadow` on start, back to `candidate` on stop when it was `shadow`).
async fn shadow_request(s: &AppState, row: &fa_hub::Snapshot, target: &str, op: &str) -> Response {
    let args = [("timescale", row.timescale.as_str()), ("snapshot_id", target)];
    let reply = ctl_model(s, "shadow", &args, Duration::from_secs(15)).await;
    let ok = reply["ok"].as_bool().unwrap_or(false);
    let mut status_change = Value::Null;
    if ok {
        let new_status = match (op, row.status.as_str()) {
            ("start", st) if st != "active" => Some("shadow"),
            ("stop", "shadow") => Some("candidate"),
            _ => None,
        };
        if let Some(st) = new_status {
            status_change = match s.hub.set_snapshot_status(row.id, st).await {
                Ok(r) => json!({ "snapshot_id": r.snapshot_id, "status": r.status }),
                Err(e) => json!({ "error": e.to_string() }),
            };
        }
    }
    let audited = audit(
        s,
        &format!("latent.snapshot.shadow.{op}"),
        &row.snapshot_id,
        json!({ "timescale": row.timescale, "reply": reply, "status_change": status_change }),
    )
    .await;
    let code = match reply["error"].as_str() {
        _ if ok => StatusCode::OK,
        Some("nats_unavailable") => StatusCode::SERVICE_UNAVAILABLE,
        Some("runtime_timeout") => StatusCode::GATEWAY_TIMEOUT,
        _ => StatusCode::BAD_GATEWAY,
    };
    (
        code,
        Json(json!({
            "ok": ok, "snapshot_id": row.snapshot_id, "timescale": row.timescale, "op": op,
            "runtime": reply, "status_change": status_change, "audited": audited,
        })),
    )
        .into_response()
}

// ----------------------------------------------------- promote/rollback

/// `promote`: refused with 409 and the checklist unless [`promotion_checklist`] passes or the
/// body says `{"force": true}` (audited with the failing checklist and `note`); then this
/// snapshot → `active`, every other active (or shadow) snapshot of the same timescale →
/// `retired`, and the replaced active is recorded on the target as `previous_active`.
/// `rollback`: this snapshot → `retired`; the snapshot it replaced ([`rollback_target`]) → `active`.
async fn snapshot_action(
    State(s): State<AppState>,
    Path((id, action)): Path<(String, String)>,
    body: Bytes,
) -> Response {
    if !matches!(action.as_str(), "promote" | "rollback") {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "unknown action", "action": action, "use": "promote|rollback|shadow" })),
        )
            .into_response();
    }
    let opts = match parse_body(&body) {
        Ok(b) => b,
        Err(e) => return bad_request(e),
    };
    let snaps = match s.hub.list_snapshots().await {
        Ok(v) => v,
        Err(e) => return hub_error(e),
    };
    let Some(target) = snaps.iter().find(|sn| sn.snapshot_id == id) else {
        return not_found("Snapshot", &id);
    };
    let mut changes: Vec<(u64, String, &str)> = Vec::new();
    let mut gate = Value::Null;
    if action == "promote" {
        if target.status == "active" {
            return (
                StatusCode::CONFLICT,
                Json(json!({ "error": "conflict", "detail": "snapshot is already active" })),
            )
                .into_response();
        }
        let (checklist, _, source) = checklist_for(&s, target, opts.representation_break).await;
        if !checklist.passed && !opts.force {
            let failing: Vec<&str> = checklist.checks.iter().filter(|c| !c.ok).map(|c| c.name).collect();
            let audited = audit(
                &s,
                "latent.snapshot.promote.refused",
                &id,
                json!({ "checklist": checklist, "failing": failing }),
            )
            .await;
            return (
                StatusCode::CONFLICT,
                Json(json!({
                    "error": "promotion_checks_failed",
                    "detail": "spec 36 §6.4 checks failed; shadow the candidate, or send {\"force\": true} to override (audited)",
                    "failing": failing,
                    "checklist": checklist,
                    "manifest_source": source,
                    "not_evaluated": NOT_EVALUATED,
                    "audited": audited,
                })),
            )
                .into_response();
        }
        gate = json!({
            "checklist": checklist,
            "forced": !checklist.passed && opts.force,
            "note": opts.note,
            "representation_break": opts.representation_break,
        });
        changes.push((target.id, target.snapshot_id.clone(), "active"));
        for other in snaps.iter().filter(|o| {
            o.id != target.id && o.timescale == target.timescale && matches!(o.status.as_str(), "active" | "shadow")
        }) {
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
            && let Some(prev) = rollback_target(&snaps, target)
        {
            changes.push((prev.id, prev.snapshot_id.clone(), "active"));
        }
    }
    // The active snapshot this promotion replaces (recorded on the target for a later rollback).
    let replaced = (action == "promote")
        .then(|| {
            snaps
                .iter()
                .find(|o| o.id != target.id && o.timescale == target.timescale && o.status == "active")
                .map(|o| o.snapshot_id.clone())
        })
        .flatten();
    let mut applied = Vec::new();
    for (hid, sid, status) in &changes {
        match s.hub.set_snapshot_status(*hid, status).await {
            Ok(row) => applied.push(json!({ "snapshot_id": sid, "status": row.status })),
            Err(e) => return hub_error(e),
        }
    }
    if let Some(prev) = replaced.as_deref()
        && let Err(e) = s.hub.set_snapshot_previous_active(target.id, prev).await
    {
        return hub_error(e);
    }
    // The runtime follows the status change: reload the timescale onto the new active snapshot
    // (promote: the target; rollback: the previous active when one exists, else the baseline).
    let new_active = changes.iter().find(|(_, _, st)| *st == "active").map(|(_, sid, _)| sid.clone());
    let mut args = vec![("timescale", target.timescale.as_str())];
    if let Some(id) = new_active.as_deref() {
        args.push(("snapshot_id", id));
    }
    let reload = ctl_model(&s, "reload", &args, Duration::from_secs(30)).await;
    let audited = audit(
        &s,
        &format!("latent.snapshot.{action}"),
        &id,
        json!({ "changes": applied, "reload": reload, "gate": gate }),
    )
    .await;
    Json(json!({
        "ok": true, "snapshot_id": id, "action": action, "changes": applied, "reload": reload,
        "gate": gate, "audited": audited,
    }))
    .into_response()
}

/// The snapshot a `rollback` of `target` re-activates: the one `target` replaced on promotion
/// (`previous_active`, D-065) when it still exists in the same timescale; otherwise, for snapshots
/// promoted before that field existed, the newest other retired snapshot of the timescale.
fn rollback_target<'a>(snaps: &'a [fa_hub::Snapshot], target: &fa_hub::Snapshot) -> Option<&'a fa_hub::Snapshot> {
    let same_ts = |o: &&fa_hub::Snapshot| o.id != target.id && o.timescale == target.timescale;
    if let Some(prev_id) = target.previous_active.as_deref()
        && let Some(prev) = snaps.iter().filter(same_ts).find(|o| o.snapshot_id == prev_id)
    {
        return Some(prev);
    }
    snaps.iter().filter(same_ts).filter(|o| o.status == "retired").max_by(|a, b| a.created_at.cmp(&b.created_at))
}

/// `ctl.model.<verb>` request/reply; never fails: `{ok, detail, subject, error?}` where
/// `error` is `nats_unavailable | runtime_timeout | runtime_error | bad_reply`.
async fn ctl_model(s: &AppState, verb: &str, args: &[(&str, &str)], timeout: Duration) -> Value {
    let subject = fa_bus::subjects::ctl("model", verb);
    let Some(client) = s.nats.as_ref().filter(|_| s.nats_connected()) else {
        return json!({ "ok": false, "error": "nats_unavailable", "detail": "NATS not connected; model-runtime keeps its current state", "subject": subject });
    };
    let args: HashMap<String, String> = args.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect();
    let req = CtlRequest { verb: verb.into(), args: args.clone() };
    let fut = client.request(subject.clone(), req.encode_to_vec().into());
    match tokio::time::timeout(timeout, fut).await {
        Ok(Ok(msg)) => match CtlReply::decode(msg.payload.as_ref()) {
            Ok(r) => json!({ "ok": r.ok, "detail": r.message, "subject": subject, "args": args }),
            Err(e) => {
                json!({ "ok": false, "error": "bad_reply", "detail": format!("bad reply: {e}"), "subject": subject })
            }
        },
        Ok(Err(e)) => {
            json!({ "ok": false, "error": "runtime_error", "detail": format!("{e} (is model-runtime running?)"), "subject": subject })
        }
        Err(_) => json!({
            "ok": false, "error": "runtime_timeout", "subject": subject,
            "detail": format!("no reply within {}s (model-runtime without `{verb}` support?)", timeout.as_secs()),
        }),
    }
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
    fn hsnap(id: u64, sid: &str, status: &str, created: &str, prev: Option<&str>) -> fa_hub::Snapshot {
        fa_hub::Snapshot {
            id,
            snapshot_id: sid.into(),
            timescale: "1m".into(),
            encoder_family: "statistical".into(),
            generation: 0,
            garage_key: String::new(),
            status: status.into(),
            created_at: Some(created.into()),
            carried_mass: None,
            manifest: None,
            previous_active: prev.map(str::to_owned),
        }
    }

    #[test]
    fn rollback_restores_the_replaced_snapshot_not_the_newest_retired() {
        let baseline = hsnap(1, "stat-v1-1m", "retired", "2026-11-10T01:00:00Z", None);
        let newer_retired = hsnap(2, "bad-candidate", "retired", "2026-11-10T03:00:00Z", None);
        let active = hsnap(3, "g1", "active", "2026-11-10T02:00:00Z", Some("stat-v1-1m"));
        let snaps = vec![baseline.clone(), newer_retired.clone(), active.clone()];
        assert_eq!(super::rollback_target(&snaps, &active).map(|s| s.snapshot_id.as_str()), Some("stat-v1-1m"));
        // Legacy rows without `previous_active` keep the newest-retired fallback.
        let legacy = hsnap(4, "g0", "active", "2026-11-10T02:30:00Z", None);
        let snaps = vec![baseline, newer_retired, legacy.clone()];
        assert_eq!(super::rollback_target(&snaps, &legacy).map(|s| s.snapshot_id.as_str()), Some("bad-candidate"));
    }

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
        // new routes resolve (Helix down → 503), bad promote body → 400 before any lookup
        for req in [
            Request::get("/api/latent/snapshots/x").body(Body::empty()).unwrap(),
            Request::post("/api/latent/snapshots/x/shadow").body(Body::empty()).unwrap(),
            Request::delete("/api/latent/snapshots/x/shadow").body(Body::empty()).unwrap(),
            Request::post("/api/latent/snapshots/x/shadow/stop").body(Body::empty()).unwrap(),
        ] {
            let (s, v) = send(req).await;
            assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{v}");
        }
        let (s, v) = send(Request::post("/api/latent/snapshots/x/promote").body(Body::from("{nope")).unwrap()).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    }

    fn counters(pairs: &[(&str, f64)]) -> std::collections::BTreeMap<String, f64> {
        pairs.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect()
    }

    #[test]
    fn checklist_logic() {
        let th = PromotionThresholds::default();
        let id = "stat-v1-1m-g1-1794292853";
        let marker = format!("shadow.1m.snapshot_id.{id}");
        let good = counters(&[
            ("shadow.1m.snapshot", 1.0),
            (&marker, 1.0),
            ("shadow.1m.minutes", 45.0),
            ("shadow.1m.region_agreement", 0.6),
            ("shadow.1m.states", 300.0),
        ]);
        let c = promotion_checklist(id, "1m", Some(0.8), Some(&good), false, th);
        assert!(c.passed, "{c:?}");
        assert_eq!(
            c.checks.iter().map(|c| c.name).collect::<Vec<_>>(),
            vec!["carried_mass", "shadow_served", "shadow_minutes", "shadow_region_agreement"]
        );

        // each item fails on its own
        let fail = |c: &Checklist| c.checks.iter().filter(|x| !x.ok).map(|x| x.name).collect::<Vec<_>>();
        assert_eq!(fail(&promotion_checklist(id, "1m", Some(0.2), Some(&good), false, th)), vec!["carried_mass"]);
        assert_eq!(fail(&promotion_checklist(id, "1m", None, Some(&good), false, th)), vec!["carried_mass"]);
        assert!(promotion_checklist(id, "1m", Some(0.2), Some(&good), true, th).passed, "representation break waives");
        let short = counters(&[(&marker, 1.0), ("shadow.1m.minutes", 5.0), ("shadow.1m.region_agreement", 0.6)]);
        assert_eq!(fail(&promotion_checklist(id, "1m", Some(0.8), Some(&short), false, th)), vec!["shadow_minutes"]);
        let disagree = counters(&[(&marker, 1.0), ("shadow.1m.minutes", 50.0), ("shadow.1m.region_agreement", 0.1)]);
        assert_eq!(
            fail(&promotion_checklist(id, "1m", Some(0.8), Some(&disagree), false, th)),
            vec!["shadow_region_agreement"]
        );
        // another snapshot in shadow: its minutes/agreement do not count for this one
        let other = counters(&[
            ("shadow.1m.snapshot", 1.0),
            ("shadow.1m.snapshot_id.stat-v1-1m-g1-1", 1.0),
            ("shadow.1m.minutes", 500.0),
            ("shadow.1m.region_agreement", 0.9),
        ]);
        let c = promotion_checklist(id, "1m", Some(0.8), Some(&other), false, th);
        assert_eq!(fail(&c), vec!["shadow_served", "shadow_minutes", "shadow_region_agreement"]);
        assert_eq!(c.checks[2].value, None);
        assert!(c.checks[1].detail.contains("stat-v1-1m-g1-1"), "{}", c.checks[1].detail);
        assert_eq!(shadow_id(&other, "1m").as_deref(), Some("stat-v1-1m-g1-1"));
        // wrong timescale gauges, and no heartbeat at all
        let c = promotion_checklist(id, "1h", Some(0.8), Some(&good), false, th);
        assert!(!c.passed);
        let c = promotion_checklist(id, "1m", Some(0.8), None, false, th);
        assert_eq!(c.checks[1].detail, "no fresh model-runtime heartbeat");
        // thresholds are inclusive; a zeroed marker does not count
        let edge = counters(&[(&marker, 1.0), ("shadow.1m.minutes", 30.0), ("shadow.1m.region_agreement", 0.3)]);
        assert!(promotion_checklist(id, "1m", Some(0.5), Some(&edge), false, th).passed, "thresholds are inclusive");
        let stopped = counters(&[(&marker, 0.0), ("shadow.1m.minutes", 30.0), ("shadow.1m.region_agreement", 0.3)]);
        assert!(!promotion_checklist(id, "1m", Some(0.5), Some(&stopped), false, th).passed);
    }

    #[test]
    fn thresholds_from_env_fall_back_to_defaults() {
        // Only the defaults are asserted: tests share the process environment.
        let d = PromotionThresholds::default();
        assert_eq!((d.min_carried_mass, d.min_shadow_minutes, d.min_region_agreement), (0.5, 30.0, 0.3));
        if std::env::var("FA_PROMOTE_MIN_CARRIED_MASS").is_err() {
            assert_eq!(PromotionThresholds::from_env().min_carried_mass, 0.5);
        }
    }

    /// Route behaviour with a fake Helix (Snapshot rows) and synthetic model-runtime heartbeat
    /// counters: 409 + checklist, force → 200, then pass with good counters.
    #[tokio::test]
    async fn promote_route_enforces_checklist_with_synthetic_heartbeat() {
        use axum::routing::post as axpost;
        use std::sync::{Arc, Mutex as StdMutex};
        let id = "stat-v1-1m-g1-1794292853";
        let manifest = json!({ "snapshot_id": id, "metrics": { "carried_mass": 0.72 } }).to_string();
        let rows = Arc::new(StdMutex::new(vec![
            json!({ "$id": 1, "snapshot_id": "stat-v1-1m", "timescale": "1m", "encoder_family": "statistical", "generation": 0, "garage_key": "", "status": "active" }),
            json!({ "$id": 2, "snapshot_id": id, "timescale": "1m", "encoder_family": "statistical", "generation": 1, "garage_key": "", "status": "candidate", "carried_mass": 0.72, "manifest": manifest }),
        ]));
        // Fake Helix: answers every batch from `rows`; a write with set_property flips a status.
        let fake = {
            let rows = rows.clone();
            Router::new().route(
                "/v2/query",
                axpost(move |Json(req): Json<Value>| {
                    let rows = rows.clone();
                    async move {
                        let text = req.to_string();
                        let mut rows = rows.lock().unwrap();
                        if text.contains("set_property") {
                            let hid = if text.contains(r#""ids":[2]"#) { 2 } else { 1 };
                            let status = ["active", "retired", "shadow", "candidate"]
                                .into_iter()
                                .find(|s| text.contains(&format!(r#""string":"{s}""#)))
                                .unwrap_or("active");
                            for r in rows.iter_mut() {
                                if r["$id"] == hid {
                                    r["status"] = json!(status);
                                }
                            }
                            let row = rows.iter().find(|r| r["$id"] == hid).cloned().unwrap();
                            return Json(json!({ "row": [row] }));
                        }
                        if text.contains("list_snapshots") {
                            return Json(json!({ "rows": rows.clone() }));
                        }
                        if text.contains("AdapterState") {
                            return Json(json!({ "rows": [] }));
                        }
                        let hit: Vec<Value> = rows
                            .iter()
                            .filter(|r| text.contains(r["snapshot_id"].as_str().unwrap()))
                            .cloned()
                            .collect();
                        let best = hit.iter().max_by_key(|r| r["snapshot_id"].as_str().unwrap().len()).cloned();
                        Json(json!({ "row": best.map(|b| vec![b]).unwrap_or_default() }))
                    }
                }),
            )
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, fake).await.unwrap() });

        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let cfg = fa_core::config::PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        let state = AppState::new(cfg, format!("{root}/sources.yaml").into(), None);
        let state = AppState { hub: fa_hub::HubClient::new(format!("http://{addr}")), ..state };
        let heartbeat = |pairs: &[(&str, f64)]| {
            state.record_heartbeat(fa_proto::v1::Heartbeat {
                service: "model-runtime".into(),
                instance_id: "test".into(),
                at: None,
                version: "test".into(),
                counters: pairs.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect(),
            });
        };
        let app = router().with_state(state.clone());
        let call = |req: Request<Body>| {
            let app = app.clone();
            async move {
                let res = app.oneshot(req).await.unwrap();
                let status = res.status();
                let body = res.into_body().collect().await.unwrap().to_bytes();
                (status, serde_json::from_slice::<Value>(&body).unwrap_or(Value::Null))
            }
        };
        let promote = |body: &str| {
            Request::post(format!("/api/latent/snapshots/{id}/promote")).body(Body::from(body.to_owned())).unwrap()
        };

        // 1. no shadow yet (heartbeat without shadow gauges) → 409 with the checklist
        heartbeat(&[("trades_total", 10.0)]);
        let (s, v) = call(promote("")).await;
        assert_eq!(s, StatusCode::CONFLICT, "{v}");
        assert_eq!(v["error"], "promotion_checks_failed");
        assert_eq!(v["failing"], json!(["shadow_served", "shadow_minutes", "shadow_region_agreement"]));
        assert_eq!(v["checklist"]["checks"][0]["value"], 0.72);
        assert_eq!(rows.lock().unwrap()[1]["status"], "candidate", "nothing changed");

        // 2. detail route shows the same checklist and the shadow gauges
        let marker = format!("shadow.1m.snapshot_id.{id}");
        heartbeat(&[
            ("shadow.1m.snapshot", 1.0),
            (&marker, 1.0),
            ("shadow.1m.minutes", 12.0),
            ("shadow.1m.region_agreement", 0.5),
        ]);
        let (s, v) = call(Request::get(format!("/api/latent/snapshots/{id}")).body(Body::empty()).unwrap()).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["manifest_source"], "helix");
        assert_eq!(v["checklist"]["passed"], false);
        assert_eq!(v["shadow_gauges"]["shadow.1m.minutes"], 12.0);
        let (s, v) = call(promote("")).await;
        assert_eq!(s, StatusCode::CONFLICT);
        assert_eq!(v["failing"], json!(["shadow_minutes"]));

        // 3. force overrides (NATS absent: reload reports nats_unavailable, route still 200)
        let (s, v) = call(promote(r#"{"force": true, "note": "test override"}"#)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["gate"]["forced"], true);
        assert_eq!(v["gate"]["note"], "test override");
        assert_eq!(v["reload"]["error"], "nats_unavailable");
        assert_eq!(rows.lock().unwrap()[1]["status"], "active");
        assert_eq!(rows.lock().unwrap()[0]["status"], "retired");

        // 4. back to candidate; with enough shadow minutes it passes unforced
        rows.lock().unwrap()[1]["status"] = json!("shadow");
        rows.lock().unwrap()[0]["status"] = json!("active");
        heartbeat(&[
            ("shadow.1m.snapshot", 1.0),
            (&marker, 1.0),
            ("shadow.1m.minutes", 31.0),
            ("shadow.1m.region_agreement", 0.5),
        ]);
        let (s, v) = call(promote("")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["gate"]["forced"], false);
        assert_eq!(v["gate"]["checklist"]["passed"], true);

        // 5. shadow start without NATS → 503 with the runtime error
        rows.lock().unwrap()[1]["status"] = json!("candidate");
        let (s, v) =
            call(Request::post(format!("/api/latent/snapshots/{id}/shadow")).body(Body::empty()).unwrap()).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{v}");
        assert_eq!(v["runtime"]["error"], "nats_unavailable");
    }

    #[test]
    fn neural_manifest_adds_ssl_and_health_checks() {
        let stat = json!({ "metrics": { "carried_mass": 0.7 } });
        assert!(manifest_health_checks(&stat).is_empty(), "statistical manifests add nothing");
        let neural = json!({ "metrics": {
            "ssl_regression": 0.01, "parent_mix_skill": 0.3, "mix_skill": 0.29,
            "k_eff": 2.6, "resid_corr_max": 0.4, "skill_min": 0.05
        }});
        let c = manifest_health_checks(&neural);
        let by = |n: &str| c.iter().find(|x| x.name == n).unwrap().ok;
        assert_eq!(c.len(), 4);
        assert!(by("ssl_regression") && by("expert_k_eff") && by("expert_min_skill"));
        assert!(!by("expert_resid_corr"), "0.4 > 0.1 target");
        // a neural child of a statistical parent still reports item 1 (n/a, passing)
        let first = json!({ "parent_snapshot_id": "stat-v1-1m", "metrics": { "mix_skill": 0.3, "k_eff": 1.7 } });
        let c = manifest_health_checks(&first);
        let ssl = c.iter().find(|x| x.name == "ssl_regression").expect("ssl_regression present");
        assert!(ssl.ok && ssl.value.is_none() && ssl.detail.contains("stat-v1-1m"), "{ssl:?}");
    }
}
