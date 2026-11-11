//! Helix-backed graph / latent routes (spec 50 §4.4–4.5, D-048): the **persisted** view served
//! from HelixDB through `fa_hub`, next to the bus-fed in-memory views of `live_routes` (D-032).
//!
//! | Route | Source |
//! |---|---|
//! | `GET /api/graph/neighborhood?node=&depth=` | `HubClient::resolve_node` + `neighborhood` |
//! | `GET /api/graph/lineage/{genome_id}` | `genome_by_id` + `genome_ancestry` + `genome_fitness` + `genome_parents` per node (`edges[{from: parent, to: child, from_id, to_id}]`) |
//! | `GET /api/graph/query` | the named-query catalog (`fa_hub::catalog::NAMED_QUERIES`) |
//! | `POST /api/graph/query` `{name, params}` | `HubClient::run_named` (whitelist; never raw trees) |
//! | `GET /api/latent/snapshots` | `list_snapshots` |
//! | `GET /api/latent/{snapshot}/regions` | `snapshot_graph` |
//! | `GET /api/mcmc/{snapshot}/graph?timescale=` | `snapshot_graph` |
//!
//! Helix unreachable → `503 {"error":"helix_unavailable"}`; unknown ids → 404; bad params → 400;
//! any other server error → `502 {"error":"helix_error"}`.
use std::collections::BTreeMap;

use axum::{
    Json, Router,
    extract::{Path, Query, State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use fa_hub::{HubError, catalog};
use serde::Deserialize;
use serde_json::{Value, json};
use tower_http::cors::CorsLayer;

use crate::state::AppState;

/// Default / maximum hops for `/api/graph/neighborhood`.
const DEFAULT_DEPTH: usize = 1;
/// Ancestry levels fetched for `/api/graph/lineage/{genome_id}`.
const LINEAGE_DEPTH: usize = 16;

pub fn router() -> Router<AppState> {
    Router::new()
        // 4.4 Hub / graph
        .route("/api/graph/neighborhood", get(neighborhood))
        .route("/api/graph/query", get(query_catalog).post(run_query))
        .route("/api/graph/lineage/{genome_id}", get(lineage))
        // 4.5 Latent / MCMC (persisted view; the in-memory `/api/latent/timescales/...` stay in `routes`)
        .route("/api/latent/snapshots", get(snapshots))
        .route("/api/latent/{snapshot}/regions", get(snapshot_regions))
        .route("/api/mcmc/{snapshot}/graph", get(mcmc_graph))
        .layer(CorsLayer::permissive())
}

/// `HubError` → HTTP reply.
struct HubReply(HubError);

impl IntoResponse for HubReply {
    fn into_response(self) -> Response {
        let e = self.0;
        let (status, body) = if e.is_unreachable() {
            (StatusCode::SERVICE_UNAVAILABLE, json!({ "error": "helix_unavailable", "detail": e.to_string() }))
        } else {
            match &e {
                HubError::NotFound { label, id } => {
                    (StatusCode::NOT_FOUND, json!({ "error": "not_found", "label": label, "id": id }))
                }
                HubError::UnknownQuery(name) => (
                    StatusCode::NOT_FOUND,
                    json!({ "error": "unknown_query", "name": name, "known": catalog::names() }),
                ),
                HubError::BadParam(msg) => (StatusCode::BAD_REQUEST, json!({ "error": "bad_request", "detail": msg })),
                _ => (StatusCode::BAD_GATEWAY, json!({ "error": "helix_error", "detail": e.to_string() })),
            }
        };
        (status, Json(body)).into_response()
    }
}

fn not_found(label: &str, id: &str) -> Response {
    HubReply(HubError::NotFound { label: label.into(), id: id.into() }).into_response()
}

type Reply = Result<Json<Value>, HubReply>;

impl From<HubError> for HubReply {
    fn from(e: HubError) -> Self {
        HubReply(e)
    }
}

// ------------------------------------------------------------------ 4.4

#[derive(Debug, Deserialize)]
pub struct NeighborhoodQuery {
    /// Numeric Helix `$id` or any unique business id (region / snapshot / space / genome / instrument).
    pub node: Option<String>,
    pub depth: Option<usize>,
}

async fn neighborhood(State(s): State<AppState>, Query(q): Query<NeighborhoodQuery>) -> Response {
    let Some(node) = q.node.as_deref().map(str::trim).filter(|n| !n.is_empty()) else {
        return (StatusCode::BAD_REQUEST, Json(json!({ "error": "bad_request", "detail": "`node` is required" })))
            .into_response();
    };
    let depth = q.depth.unwrap_or(DEFAULT_DEPTH).clamp(1, catalog::MAX_NEIGHBORHOOD_DEPTH);
    let res: Reply = async {
        let Some(center) = s.hub.resolve_node(node).await? else {
            return Err(HubError::NotFound { label: "node".into(), id: node.into() }.into());
        };
        let nb = s.hub.neighborhood(center.id, depth).await?;
        Ok(Json(json!({ "center": center, "depth": depth, "nodes": nb.nodes, "edges": nb.edges })))
    }
    .await;
    res.into_response()
}

async fn lineage(State(s): State<AppState>, Path(genome_id): Path<String>) -> Response {
    let res: Reply = async {
        let Some(genome) = s.hub.genome_by_id(&genome_id).await? else {
            return Err(HubError::NotFound { label: "Genome".into(), id: genome_id.clone() }.into());
        };
        let (ancestry, fitness) =
            tokio::try_join!(s.hub.genome_ancestry(genome.id, LINEAGE_DEPTH), s.hub.genome_fitness(genome.id))?;
        let edges = lineage_edges(&s.hub, &genome, &ancestry).await?;
        Ok(Json(json!({ "genome": genome, "ancestors_by_depth": ancestry, "fitness": fitness, "edges": edges })))
    }
    .await;
    res.into_response()
}

/// Most genomes whose parents are looked up for the lineage edges.
const LINEAGE_MAX_NODES: usize = 500;

/// `ParentOf` edges inside the lineage: `{from, to, from_id, to_id}` with `from` = parent and
/// `to` = child genome id (business ids; `*_id` = Helix `$id`), for the genome and every
/// ancestor (at most [`LINEAGE_MAX_NODES`] children), deduplicated.
async fn lineage_edges(
    hub: &fa_hub::HubClient,
    genome: &fa_hub::Genome,
    ancestry: &[Vec<fa_hub::Genome>],
) -> Result<Vec<Value>, HubError> {
    let mut children: Vec<(u64, String)> = vec![(genome.id, genome.genome_id.clone())];
    let mut seen = std::collections::BTreeSet::from([genome.id]);
    for g in ancestry.iter().flatten() {
        if seen.insert(g.id) {
            children.push((g.id, g.genome_id.clone()));
        }
    }
    children.truncate(LINEAGE_MAX_NODES);
    let mut parents: Vec<((u64, String), Vec<fa_hub::Genome>)> = Vec::with_capacity(children.len());
    for chunk in children.chunks(16) {
        let got = futures::future::try_join_all(chunk.iter().map(|c| hub.genome_parents(c.0))).await?;
        parents.extend(chunk.iter().cloned().zip(got));
    }
    let mut edges = std::collections::BTreeSet::new();
    for ((child_id, child), ps) in &parents {
        for p in ps {
            edges.insert((p.genome_id.clone(), child.clone(), p.id, *child_id));
        }
    }
    Ok(edges
        .into_iter()
        .map(|(from, to, from_id, to_id)| json!({ "from": from, "to": to, "from_id": from_id, "to_id": to_id }))
        .collect())
}

async fn query_catalog() -> impl IntoResponse {
    Json(json!({ "queries": catalog::NAMED_QUERIES }))
}

#[derive(Debug, Deserialize)]
pub struct NamedQueryBody {
    pub name: String,
    #[serde(default)]
    pub params: Value,
}

async fn run_query(State(s): State<AppState>, body: Result<Json<NamedQueryBody>, JsonRejection>) -> Response {
    let body = match body {
        Ok(Json(b)) => b,
        Err(rej) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "bad_request", "detail": format!("JSON body `{{name, params}}` required: {}", rej.body_text()), "known": catalog::names() })),
            )
                .into_response();
        }
    };
    match s.hub.run_named(&body.name, &body.params).await {
        Ok(result) => Json(json!({ "name": body.name, "result": result })).into_response(),
        Err(e) => HubReply(e).into_response(),
    }
}

// ------------------------------------------------------------------ 4.5

async fn snapshots(State(s): State<AppState>) -> Response {
    let res: Reply = async {
        let snaps = s.hub.list_snapshots().await?;
        let mut by_timescale: BTreeMap<String, Vec<&fa_hub::Snapshot>> = BTreeMap::new();
        for sn in &snaps {
            by_timescale.entry(sn.timescale.clone()).or_default().push(sn);
        }
        Ok(Json(json!({ "snapshots": snaps, "by_timescale": by_timescale })))
    }
    .await;
    res.into_response()
}

async fn snapshot_regions(State(s): State<AppState>, Path(snapshot): Path<String>) -> Response {
    let res: Reply = async {
        let Some(g) = s.hub.snapshot_graph(&snapshot).await? else {
            return Err(HubError::NotFound { label: "Snapshot".into(), id: snapshot.clone() }.into());
        };
        let mut status_counts: BTreeMap<String, usize> = BTreeMap::new();
        for r in &g.regions {
            *status_counts.entry(r.status.clone().unwrap_or_else(|| "live".into())).or_default() += 1;
        }
        Ok(Json(json!({
            "snapshot": g.snapshot,
            "spaces": g.spaces,
            "regions": g.regions,
            "status_counts": status_counts,
            "n_transitions": g.transitions.len(),
        })))
    }
    .await;
    res.into_response()
}

#[derive(Debug, Deserialize)]
pub struct GraphQuery {
    pub timescale: Option<String>,
}

async fn mcmc_graph(State(s): State<AppState>, Path(snapshot): Path<String>, Query(q): Query<GraphQuery>) -> Response {
    let g = match s.hub.snapshot_graph(&snapshot).await {
        Ok(Some(g)) => g,
        Ok(None) => return not_found("Snapshot", &snapshot),
        Err(e) => return HubReply(e).into_response(),
    };
    if let Some(ts) = q.timescale.as_deref().filter(|t| !t.is_empty())
        && ts != g.snapshot.timescale
    {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "not_found", "detail": format!("snapshot {snapshot} is at timescale {}, not {ts}", g.snapshot.timescale) })),
        )
            .into_response();
    }
    let edges: Vec<Value> = g
        .transitions
        .iter()
        .map(|t| {
            json!({
                "id": t.id, "from": t.from, "to": t.to,
                "from_region_id": t.from_region_id, "to_region_id": t.to_region_id,
                "sampler_id": t.sampler_id, "count": t.count,
                "mean": t.mean, "lower": t.lower, "upper": t.upper,
                "kernel_params": t.kernel_params, "updated_at": t.updated_at,
            })
        })
        .collect();
    Json(json!({
        "snapshot": g.snapshot,
        "timescale": g.snapshot.timescale,
        "spaces": g.spaces,
        "nodes": g.regions,
        "edges": edges,
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

    /// Router whose Helix URL points at a closed port: every hub call fails as unreachable.
    fn app() -> Router {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let mut cfg = PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        cfg.helix.url = "http://127.0.0.1:1".into();
        // The env override must not re-route the test at a live server.
        let state = AppState::new(cfg, format!("{root}/sources.yaml").into(), None);
        let state = AppState { hub: fa_hub::HubClient::new("http://127.0.0.1:1"), ..state };
        router().with_state(state)
    }

    async fn send(req: Request<Body>) -> (StatusCode, Value) {
        let res = app().oneshot(req).await.unwrap();
        let status = res.status();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&body).unwrap())
    }

    async fn get_json(uri: &str) -> (StatusCode, Value) {
        send(Request::get(uri).body(Body::empty()).unwrap()).await
    }

    async fn post_json(uri: &str, body: Value) -> (StatusCode, Value) {
        send(Request::post(uri).header("content-type", "application/json").body(Body::from(body.to_string())).unwrap())
            .await
    }

    #[tokio::test]
    async fn helix_down_is_503_on_every_read() {
        for uri in [
            "/api/graph/neighborhood?node=1m-e0-r000&depth=2",
            "/api/graph/lineage/g1",
            "/api/latent/snapshots",
            "/api/latent/stat-v1-1m/regions",
            "/api/mcmc/stat-v1-1m/graph?timescale=1m",
        ] {
            let (s, v) = get_json(uri).await;
            assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{uri}: {v}");
            assert_eq!(v["error"], "helix_unavailable", "{uri}");
        }
        let (s, v) = post_json("/api/graph/query", json!({ "name": "listSnapshots" })).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(v["error"], "helix_unavailable");
    }

    #[tokio::test]
    async fn named_queries_are_whitelisted_and_validated_before_any_request() {
        let (s, v) = post_json("/api/graph/query", json!({ "name": "dropEverything", "params": {} })).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert_eq!(v["error"], "unknown_query");
        assert!(v["known"].as_array().unwrap().iter().any(|n| n == "snapshotGraph"));
        let (s, v) = post_json("/api/graph/query", json!({ "name": "getRegion", "params": {} })).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert_eq!(v["error"], "bad_request");
        let (s, _) = post_json("/api/graph/query", json!({ "nope": 1 })).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "missing name");
        let (s, v) = get_json("/api/graph/query").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["queries"].as_array().unwrap().len(), catalog::NAMED_QUERIES.len());
        let (s, v) = get_json("/api/graph/neighborhood").await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert_eq!(v["error"], "bad_request");
    }
}
