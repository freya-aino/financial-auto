//! Handlers for the Latent, Evolution and Trading views (spec 50 §4.5–4.7).
//! Reads come from the in-memory `Live` aggregator; commands go out on NATS.
//! Live trading is impossible from here: every order is forced to PAPER (D-029).
use std::time::Duration;

use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use fa_bus::subjects;
use fa_proto::{
    Message, now,
    v1::{CtlReply, CtlRequest, Halt, OrderIntent, Side, TradingMode},
};
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use crate::state::AppState;

/// GPU priority classes, highest first (spec 60, D-006).
const GPU_CLASSES: [&str; 4] = ["live_inference", "online_adapter", "evolution_eval", "generation_training"];
const CTL_TIMEOUT: Duration = Duration::from_secs(5);

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    ApiError(status, msg.into()).into_response()
}

/// Error from a bus command; small so it can travel in `Result`.
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

#[derive(Debug, Deserialize)]
pub struct ListQuery {
    pub limit: Option<usize>,
    pub family: Option<String>,
    pub kind: Option<String>,
    pub fills: Option<bool>,
}

impl ListQuery {
    fn limit(&self, default: usize) -> usize {
        self.limit.unwrap_or(default).clamp(1, 500)
    }
}

// ------------------------------------------------------------------ latent

pub async fn latent_timescales(State(s): State<AppState>) -> impl IntoResponse {
    let configured: Vec<&str> = s.cfg.timescales.iter().map(|t| t.as_str()).collect();
    Json(json!({ "configured": configured, "seen": s.live().timescales_seen() }))
}

pub async fn latent_view(State(s): State<AppState>, Path((ts, view)): Path<(String, String)>) -> Response {
    let live = s.live();
    match view.as_str() {
        "states" => Json(live.states_view(&ts)).into_response(),
        "regions" => Json(live.regions_view(&ts)).into_response(),
        "transitions" | "graph" => Json(live.transitions_view(&ts)).into_response(),
        _ => err(StatusCode::NOT_FOUND, format!("unknown latent view {view:?}; use states|regions|transitions")),
    }
}

pub async fn decoder_outputs(State(s): State<AppState>, Query(q): Query<ListQuery>) -> impl IntoResponse {
    Json(s.live().decoder_outputs_view(q.family.as_deref(), q.limit(50)))
}

// --------------------------------------------------------------- evolution

pub async fn jobs(State(s): State<AppState>, Query(q): Query<ListQuery>) -> impl IntoResponse {
    Json(s.live().jobs_view(q.kind.as_deref(), q.limit(100)))
}

/// GPU arbiter status, derived from the evo-orchestrator heartbeat counters
/// (the orchestrator has no `ctl.gpu.status` request yet).
pub async fn gpu(State(s): State<AppState>) -> impl IntoResponse {
    let services = s.services.read().expect("services lock");
    let Some(orch) = services.get("evo-orchestrator") else {
        return Json(json!({
            "available": false,
            "reason": "no heartbeat from evo-orchestrator",
            "classes": GPU_CLASSES,
        }));
    };
    let c = |k: &str| orch.counters.get(k).copied();
    Json(json!({
        "available": true,
        "last_seen": orch.last_seen,
        "used_mb": c("gpu.used_mb"),
        "active": c("gpu.active"),
        "queued": c("gpu.queued"),
        "capacity_mb": c("gpu.capacity_mb"),
        "generations_started": c("generations.started"),
        "generations_done": c("generations.done"),
        "classes": GPU_CLASSES,
    }))
}

#[derive(Debug, Deserialize)]
pub struct StartGeneration {
    pub kind: String,
    #[serde(default)]
    pub snapshot_id: String,
    pub genomes: Vec<String>,
    pub eta: Option<u32>,
    pub min_budget_ms: Option<u64>,
    pub max_budget_ms: Option<u64>,
    /// Generation loops (elites + `evo.vary.decoder` children); orchestrator default 1 (D-049).
    pub max_generations: Option<u32>,
    pub total_budget_ms: Option<u64>,
    pub data_window_hours: Option<u32>,
    /// Cross-asset `src→target` genomes seeded per decoder loop from `latent.crossprior.*`
    /// (D-074; orchestrator default `-mcmc-pairs` 4, 0 disables).
    #[serde(default)]
    pub mcmc_pairs: Option<u32>,
}

/// Starts a successive-halving generation via the orchestrator's `ctl.evo.start`.
pub async fn start_generation(State(s): State<AppState>, Json(body): Json<StartGeneration>) -> Response {
    let Some(args) = start_args(body) else {
        return err(StatusCode::BAD_REQUEST, "kind and at least one genome are required");
    };
    let req = CtlRequest { verb: "start".into(), args: args.into_iter().map(|(k, v)| (k.to_owned(), v)).collect() };
    match ctl_request(&s, subjects::ctl("evo", "start"), &req).await {
        Ok(r) if r.ok => Json(json!({ "ok": true, "generation_id": r.message })).into_response(),
        Ok(r) => err(StatusCode::UNPROCESSABLE_ENTITY, r.message),
        Err(e) => e.into_response(),
    }
}

/// `ctl.evo.start` arguments of a start body (`None` without a kind or a genome).
pub fn start_args(body: StartGeneration) -> Option<Vec<(&'static str, String)>> {
    let genomes: Vec<&str> = body.genomes.iter().map(|g| g.trim()).filter(|g| !g.is_empty()).collect();
    if body.kind.trim().is_empty() || genomes.is_empty() {
        return None;
    }
    let mut args =
        // newline-separated so kv (`;`) and JSON (`,`) genomes survive transport (D-044)
        vec![("kind", body.kind.trim().to_owned()), ("snapshot_id", body.snapshot_id), ("genomes", genomes.join("\n"))];
    if let Some(v) = body.eta {
        args.push(("eta", v.to_string()));
    }
    if let Some(v) = body.min_budget_ms {
        args.push(("min_budget_ms", v.to_string()));
    }
    if let Some(v) = body.max_budget_ms {
        args.push(("max_budget_ms", v.to_string()));
    }
    if let Some(v) = body.max_generations {
        args.push(("max_generations", v.to_string()));
    }
    if let Some(v) = body.total_budget_ms {
        args.push(("total_budget_ms", v.to_string()));
    }
    if let Some(v) = body.data_window_hours {
        args.push(("data_window_hours", v.to_string()));
    }
    if let Some(v) = body.mcmc_pairs {
        args.push(("mcmc_pairs", v.to_string()));
    }
    Some(args)
}

async fn ctl_request(s: &AppState, subject: String, req: &CtlRequest) -> Result<CtlReply, ApiError> {
    let Some(client) = s.nats.as_ref().filter(|_| s.nats_connected()) else {
        return Err(ApiError(StatusCode::SERVICE_UNAVAILABLE, "NATS not connected".into()));
    };
    let fut = client.request(subject.clone(), Bytes::from(req.encode_to_vec()));
    match tokio::time::timeout(CTL_TIMEOUT, fut).await {
        Ok(Ok(msg)) => CtlReply::decode(msg.payload.as_ref())
            .map_err(|e| ApiError(StatusCode::BAD_GATEWAY, format!("bad reply on {subject}: {e}"))),
        Ok(Err(e)) => Err(ApiError(StatusCode::BAD_GATEWAY, format!("{subject}: {e} (is the service running?)"))),
        Err(_) => Err(ApiError(StatusCode::GATEWAY_TIMEOUT, format!("{subject}: no reply within {CTL_TIMEOUT:?}"))),
    }
}

async fn publish<M: Message>(s: &AppState, subject: String, m: &M) -> Result<(), ApiError> {
    let Some(client) = s.nats.as_ref().filter(|_| s.nats_connected()) else {
        return Err(ApiError(StatusCode::SERVICE_UNAVAILABLE, "NATS not connected".into()));
    };
    client
        .publish(subject, Bytes::from(m.encode_to_vec()))
        .await
        .map_err(|e| ApiError(StatusCode::BAD_GATEWAY, e.to_string()))?;
    client.flush().await.map_err(|e| ApiError(StatusCode::BAD_GATEWAY, e.to_string()))
}

// ------------------------------------------------------------------- ops

/// Maps a service's `CtlReply` to HTTP. Error messages carry a prefix agreed with the Go
/// services: `not_found:` → 404, `conflict:` → 409, `invalid:` → 400, anything else → 422.
fn ctl_reply_response(r: CtlReply, ok_body: impl FnOnce(String) -> serde_json::Value) -> Response {
    if r.ok {
        return Json(ok_body(r.message)).into_response();
    }
    let status = match r.message.split(':').next().unwrap_or_default() {
        "not_found" => StatusCode::NOT_FOUND,
        "conflict" => StatusCode::CONFLICT,
        "invalid" => StatusCode::BAD_REQUEST,
        _ => StatusCode::UNPROCESSABLE_ENTITY,
    };
    err(status, r.message)
}

/// Starts or stops a connector at runtime via the ingestor's `ctl.ingest.{start|stop}`
/// (spec 10 §4.3). 503 without NATS, 504 when the ingestor does not answer in 5 s.
pub async fn ingest_control(State(s): State<AppState>, Path((source, action)): Path<(String, String)>) -> Response {
    if !matches!(action.as_str(), "start" | "stop") {
        return err(StatusCode::NOT_FOUND, format!("unknown action {action:?}; use start|stop"));
    }
    let req = CtlRequest { verb: action.clone(), args: [("source".to_owned(), source.clone())].into_iter().collect() };
    match ctl_request(&s, subjects::ctl("ingest", &action), &req).await {
        Ok(r) => ctl_reply_response(r, |m| json!({ "ok": true, "source": source, "action": action, "message": m })),
        Err(e) => e.into_response(),
    }
}

/// Cancels a pending evolution job via the orchestrator's `ctl.evo.cancel`.
pub async fn cancel_job(State(s): State<AppState>, Path(id): Path<String>) -> Response {
    let req = CtlRequest { verb: "cancel".into(), args: [("job_id".to_owned(), id.clone())].into_iter().collect() };
    match ctl_request(&s, subjects::ctl("evo", "cancel"), &req).await {
        Ok(r) => ctl_reply_response(r, |m| json!({ "ok": true, "job_id": id, "message": m })),
        Err(e) => e.into_response(),
    }
}

// --------------------------------------------------------------- trading

pub async fn positions(State(s): State<AppState>) -> impl IntoResponse {
    Json(s.live().positions_view())
}

pub async fn orders(State(s): State<AppState>, Query(q): Query<ListQuery>) -> impl IntoResponse {
    Json(s.live().orders_view(q.fills.unwrap_or(false), q.limit(100)))
}

pub async fn fills(State(s): State<AppState>, Query(q): Query<ListQuery>) -> impl IntoResponse {
    Json(s.live().orders_view(true, q.limit(100)))
}

pub async fn risk(State(s): State<AppState>) -> impl IntoResponse {
    Json(s.live().risk_view())
}

pub async fn signals(State(s): State<AppState>) -> impl IntoResponse {
    Json(s.live().signals_view())
}

#[derive(Debug, Default, Deserialize)]
pub struct HaltBody {
    pub reason: Option<String>,
}

/// Manual kill switch: portfolio halt on `risk.halt.portfolio`.
pub async fn halt(State(s): State<AppState>, body: Option<Json<HaltBody>>) -> Response {
    let reason = body.and_then(|Json(b)| b.reason).filter(|r| !r.trim().is_empty());
    let h = Halt {
        scope: "portfolio".into(),
        reason: reason.unwrap_or_else(|| "manual kill switch (operator UI)".into()),
        since: Some(now()),
        manual: true,
    };
    if let Err(e) = publish(&s, subjects::risk_halt("portfolio"), &h).await {
        return e.into_response();
    }
    s.live_mut().on_halt(h);
    Json(json!({ "ok": true, "halted": "portfolio" })).into_response()
}

/// Re-arms trading after a manual halt (`risk.resume.portfolio`, D-030; payload `fa.v1.Resume`,
/// wire-compatible with the `Halt` older consumers decode).
pub async fn resume(State(s): State<AppState>) -> Response {
    let h = fa_proto::v1::Resume {
        scope: "portfolio".into(),
        reason: "manual resume (operator UI)".into(),
        at: Some(now()),
        manual: true,
        actor: "ui".into(),
    };
    if let Err(e) = publish(&s, subjects::risk_resume("portfolio"), &h).await {
        return e.into_response();
    }
    s.live_mut().on_resume("portfolio");
    Json(json!({ "ok": true, "resumed": "portfolio" })).into_response()
}

#[derive(Debug, Deserialize)]
pub struct PaperOrder {
    pub instrument_id: String,
    pub side: String,
    pub quantity: f64,
    pub limit_price: Option<f64>,
    /// Must be "paper" when given; anything else is rejected.
    pub mode: Option<String>,
}

pub const PAPER_ACCOUNT: &str = "paper";

/// Builds a manual paper intent, or explains why it is refused.
pub fn paper_intent(o: &PaperOrder) -> Result<OrderIntent, String> {
    if let Some(mode) = o.mode.as_deref()
        && !mode.eq_ignore_ascii_case("paper")
    {
        return Err(format!("mode {mode:?} rejected: only paper trading is available, no live broker is connected"));
    }
    let side = match o.side.to_ascii_lowercase().as_str() {
        "buy" => Side::Buy,
        "sell" => Side::Sell,
        other => return Err(format!("side must be buy or sell, got {other:?}")),
    };
    if !(o.quantity.is_finite() && o.quantity > 0.0) {
        return Err("quantity must be a positive number".into());
    }
    if o.limit_price.is_some_and(|p| !(p.is_finite() && p > 0.0)) {
        return Err("limit_price must be a positive number".into());
    }
    let inst = Uuid::parse_str(o.instrument_id.trim()).map_err(|e| format!("instrument_id must be a UUID: {e}"))?;
    Ok(OrderIntent {
        intent_id: fa_core::EventId::new().to_string(),
        account_id: PAPER_ACCOUNT.into(),
        instrument_id: inst.to_string(),
        side: side as i32,
        quantity: o.quantity,
        exec_algo: if o.limit_price.is_some() { "limit" } else { "market" }.into(),
        limit_price: o.limit_price,
        mode: TradingMode::Paper as i32,
        strategy_ids: vec!["manual".into()],
        created_at: Some(now()),
    })
}

/// Manual paper order straight to the broker-gateway's paper engine
/// (`order.approved.paper`; there is no risk service in the chain yet).
pub async fn paper_order(State(s): State<AppState>, Json(body): Json<PaperOrder>) -> Response {
    let intent = match paper_intent(&body) {
        Ok(i) => i,
        Err(e) => return err(StatusCode::BAD_REQUEST, e),
    };
    if s.live().is_halted("portfolio") {
        return err(StatusCode::CONFLICT, "portfolio is halted; resume before placing orders");
    }
    if let Err(e) = publish(&s, subjects::order_intent(PAPER_ACCOUNT), &intent).await {
        // gated by risk-engine (D-051)
        return e.into_response();
    }
    let id = intent.intent_id.clone();
    s.live_mut().on_intent(intent, "approved");
    Json(json!({ "ok": true, "intent_id": id, "mode": "paper" })).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_forwards_mcmc_pairs() {
        let body: StartGeneration = serde_json::from_value(json!({
            "kind": "evo.eval.decoder", "genomes": ["nn_forecast:h=5", " "], "eta": 3, "mcmc_pairs": 0
        }))
        .unwrap();
        let args = start_args(body).unwrap();
        let get = |k: &str| args.iter().find(|(a, _)| *a == k).map(|(_, v)| v.as_str());
        assert_eq!(get("mcmc_pairs"), Some("0"), "0 disables pair seeding and must reach the orchestrator");
        assert_eq!(get("genomes"), Some("nn_forecast:h=5"));
        assert_eq!(get("eta"), Some("3"));
        let none: StartGeneration =
            serde_json::from_value(json!({ "kind": "evo.eval.decoder", "genomes": ["g"] })).unwrap();
        assert!(start_args(none).unwrap().iter().all(|(k, _)| *k != "mcmc_pairs"), "absent = orchestrator default");
        let bad: StartGeneration = serde_json::from_value(json!({ "kind": " ", "genomes": ["g"] })).unwrap();
        assert!(start_args(bad).is_none());
    }

    fn order(mode: Option<&str>) -> PaperOrder {
        PaperOrder {
            instrument_id: "01900000-0000-7000-8000-000000000001".into(),
            side: "buy".into(),
            quantity: 1.0,
            limit_price: None,
            mode: mode.map(Into::into),
        }
    }

    #[test]
    fn paper_intent_forces_paper_mode() {
        let i = paper_intent(&order(None)).unwrap();
        assert_eq!(i.mode, TradingMode::Paper as i32);
        assert_eq!(i.account_id, "paper");
        assert_eq!(i.exec_algo, "market");
        assert!(paper_intent(&order(Some("PAPER"))).is_ok());
        assert!(paper_intent(&order(Some("live"))).unwrap_err().contains("only paper"));
        assert!(paper_intent(&order(Some("shadow"))).is_err());
    }

    #[test]
    fn paper_intent_validates_fields() {
        let mut o = order(None);
        o.quantity = 0.0;
        assert!(paper_intent(&o).is_err());
        let mut o = order(None);
        o.side = "hold".into();
        assert!(paper_intent(&o).is_err());
        let mut o = order(None);
        o.instrument_id = "AAPL".into();
        assert!(paper_intent(&o).is_err());
        let mut o = order(None);
        o.limit_price = Some(10.0);
        assert_eq!(paper_intent(&o).unwrap().exec_algo, "limit");
    }
}
