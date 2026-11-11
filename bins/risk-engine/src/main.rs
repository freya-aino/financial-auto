//! risk-engine: meta-allocation + non-evolvable risk limits + paper book (specs 40, 42).
//!
//! Subscribes to `signal.exposure.>`, `order.>`, `risk.halt.>`, `risk.resume.>`,
//! `ingest.health.>` and `md.trade.>`; publishes `order.intent.paper` → `order.approved.paper`
//! (or a rejection on `order.event.paper`), `risk.state`, `risk.halt.<scope>` and
//! `svc.heartbeat.risk-engine`. Answers request/reply on `ctl.risk.strategy`
//! (`{strategy_id, action: pause|resume|flatten|promote|shadow}` or `{action: list}`),
//! `ctl.risk.limits` (`{}` → current limits; `{set: <json>}` → tighten-only change) and
//! `ctl.risk.flatten` with a `CtlReply` whose message is JSON (D-051).
//! Only paper trading exists: settings with any other mode are refused and non-paper intents
//! are rejected (D-029). Starting cash 100 000 EUR (D-028). The halt/resume contract follows
//! D-030. All decision logic is in [`engine`] (pure, tested); this file is I/O.
//!
//! When `[postgres]` is reachable the paper book and the strategy lifecycle rows are loaded at
//! startup and every intent, fill, halt/resume, lifecycle action, limit change (audit) and
//! periodic position / strategy snapshot is persisted ([`persist`]); a reconciliation diff halts
//! scope `broker`. Without Postgres the engine stays in memory and says so once.
//!
//! Reference data and FX (D-061): the instrument table (`fa_warehouse::venues`) is loaded at
//! start, every `refdata_refresh_secs` and (debounced) on `ref.changed.>` / `ingest.unresolved.>`;
//! the ECB daily reference rates are fetched every `fx.ecb_refresh_secs` into `ref.fx_rate` and
//! the in-memory FX table reloaded; today's `ledger.payout_quota` is polled every 30 s.
//!
//! Strategy pipeline and holdouts ([`pipeline`], spec 35 §6.4, spec 41 §6.4): with Postgres a
//! background task keeps a locked holdout per timescale, selects candidates from the newest
//! strategy generations in Garage, requests their single holdout evaluation, admits passing
//! ones to the managed live set (strategy-runtime reload), promotes on a green gate (paper) and
//! demotes / retires; `ctl.risk.pipeline {action: status|run}`. `ctl.risk.allocators` reports
//! the live allocator tournament; `ctl.risk.limits {calibrate: "{}"|"{\"apply\":true}"}` the
//! limit calibration rules (applying only tightens). The halt state machine (kill switch,
//! halts, drawdown peak) is persisted in `ledger.risk_state` and restored at start. Limit
//! clips, pre-trade rejections, feed freezes / bad ticks and allocator fallbacks go out as
//! `fa.v1.RiskEvent{LIMIT_BREACH}` on `risk.event.<account>` and into `audit.log`.
mod engine;
mod persist;
mod pipeline;
mod settings;

use std::{path::PathBuf, sync::Arc, time::Duration};

use anyhow::Context;
use clap::Parser;
use fa_bus::{Bus, subjects};
use fa_core::{InstrumentId, config::PlatformConfig};
use fa_proto::{
    Message,
    v1::{
        CtlReply, CtlRequest, Envelope, ExposureDistribution, Halt, Heartbeat, OrderEvent, OrderIntent,
        envelope::Payload,
    },
};
use fa_risk::strategy::StrategyAction;
use futures::StreamExt;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    engine::{Engine, Evidence, Out},
    persist::Persist,
    pipeline::Pipeline,
    settings::Settings,
};

type Shared = Arc<Mutex<Engine>>;
type Store = Option<Arc<Persist>>;
type Pipe = Option<Arc<Pipeline>>;

const SERVICE: &str = "risk-engine";

#[derive(Debug, Parser)]
#[command(about = "allocator + hard risk limits + paper book → order.approved.paper")]
struct Args {
    /// Platform config (spec 60).
    #[arg(long, env = "FA_CONFIG", default_value = "config/platform.toml")]
    config: PathBuf,
    /// Risk settings (hard limits); defaults to the built-in `risk.toml` when the file is absent.
    #[arg(long, env = "FA_RISK", default_value = "bins/risk-engine/risk.toml")]
    risk: PathBuf,
    /// Decision tick in milliseconds.
    #[arg(long, default_value_t = 1_000)]
    tick_ms: u64,
    /// `risk.state` publish interval in seconds.
    #[arg(long, default_value_t = 2)]
    state_secs: u64,
    #[arg(long, default_value_t = 2)]
    heartbeat_secs: u64,
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

fn ms(t: Option<&fa_proto::Timestamp>) -> Option<i64> {
    t.map(|t| t.seconds * 1000 + i64::from(t.nanos) / 1_000_000)
}

async fn publish_outs(bus: &Bus, store: &Store, engine: &Shared, outs: Vec<Out>) {
    for o in outs {
        if let Some(p) = store {
            match &o {
                Out::Intent(i) => p.intent(i).await,
                Out::Halt(h) => {
                    p.audit(
                        "halt",
                        Some(&h.scope),
                        serde_json::json!({"reason": h.reason, "manual": h.manual, "raised_by": SERVICE}),
                    )
                    .await;
                    let ledger = engine.lock().await.ledger().clone();
                    p.snapshot(&ledger, "halt").await;
                }
                Out::Audit { action, subject, payload } => {
                    p.audit(action, Some(subject), payload.clone()).await;
                }
                Out::Resume(h) => {
                    p.audit("resume", Some(&h.scope), serde_json::json!({"reason": h.reason, "raised_by": SERVICE}))
                        .await;
                }
                Out::RiskEvent { kind, scope, detail } => {
                    p.audit(&format!("risk.event.{kind}"), Some(scope), detail.clone()).await;
                }
                _ => {}
            }
        }
        let r = match &o {
            Out::Audit { action, subject, payload } => {
                tracing::warn!(action, subject, %payload, "automatic lifecycle change");
                Ok(())
            }
            Out::Intent(i) => bus.publish(subjects::order_intent(&i.account_id), i).await,
            Out::Approved(i) => bus.publish(subjects::order_approved(&i.account_id), i).await,
            Out::Rejected { account, event } => {
                // typed decision on risk.event.<account>; the OrderEvent{REJECTED} stays for older consumers
                let _ = bus.publish(subjects::risk_event(account), &rejection_event(account, event)).await;
                bus.publish(subjects::order_event(account), event).await
            }
            Out::RiskEvent { kind, scope, detail } => {
                tracing::info!(%kind, %scope, %detail, "risk event");
                bus.publish(subjects::risk_event(RISK_EVENT_ACCOUNT), &breach_event(kind, scope, detail)).await
            }
            Out::Halt(h) => {
                tracing::warn!(scope = %h.scope, reason = %h.reason, "raising halt");
                let _ = bus.publish(subjects::risk_event(RISK_EVENT_ACCOUNT), &halt_event(h, true)).await;
                bus.publish(subjects::risk_halt(&h.scope), h).await
            }
            Out::Resume(h) => {
                tracing::info!(scope = %h.scope, reason = %h.reason, "clearing own halt");
                let _ = bus.publish(subjects::risk_event(RISK_EVENT_ACCOUNT), &halt_event(h, false)).await;
                // fa.v1 Resume is wire-compatible with Halt (D-030 consumers keep decoding Halt)
                let resume = fa_proto::v1::Resume {
                    scope: h.scope.clone(),
                    reason: h.reason.clone(),
                    at: h.since.or_else(|| Some(fa_proto::now())),
                    manual: h.manual,
                    actor: format!("system:{SERVICE}"),
                };
                bus.publish(subjects::risk_resume(&h.scope), &resume).await
            }
        };
        if let Err(e) = r {
            tracing::warn!(error = %e, "publish failed");
        }
    }
}

/// Account token of `risk.event.<account>` for engine-wide halts (the engine guards the paper book).
const RISK_EVENT_ACCOUNT: &str = "paper";

/// `RiskEvent{INTENT_REJECTED}` for a gated intent (D-042 follow-up).
fn rejection_event(account: &str, ev: &OrderEvent) -> fa_proto::v1::RiskEvent {
    fa_proto::v1::RiskEvent {
        event_id: ulid::Ulid::generate().to_string(),
        kind: fa_proto::v1::risk_event::Kind::IntentRejected as i32,
        at: ev.event_time.or_else(|| Some(fa_proto::now())),
        account: account.to_owned(),
        intent_id: ev.intent_id.clone(),
        reason: ev.reason.clone(),
        ..Default::default()
    }
}

/// `RiskEvent{LIMIT_BREACH}` for an engine event (limit clip, pre-trade rejection, feed freeze
/// or bad tick, allocator fallback): `reason` = `<kind>: <detail JSON>`, `scope` = the kind,
/// `instrument_id` when the scope is an instrument, numeric detail fields as `metrics`.
fn breach_event(kind: &str, scope: &str, detail: &serde_json::Value) -> fa_proto::v1::RiskEvent {
    let metrics = detail
        .as_object()
        .map(|o| o.iter().filter_map(|(k, v)| v.as_f64().map(|x| (k.clone(), x))).collect())
        .unwrap_or_default();
    let instrument = if Uuid::parse_str(scope).is_ok() { scope.to_owned() } else { String::new() };
    fa_proto::v1::RiskEvent {
        event_id: ulid::Ulid::generate().to_string(),
        kind: fa_proto::v1::risk_event::Kind::LimitBreach as i32,
        at: Some(fa_proto::now()),
        account: RISK_EVENT_ACCOUNT.to_owned(),
        intent_id: detail["intent_id"].as_str().unwrap_or_default().to_owned(),
        instrument_id: instrument,
        scope: kind.to_owned(),
        reason: format!("{kind}: {detail}"),
        metrics,
        ..Default::default()
    }
}

/// `RiskEvent{HALT_RAISED|HALT_CLEARED}` mirroring a halt or resume the engine publishes.
fn halt_event(h: &Halt, raised: bool) -> fa_proto::v1::RiskEvent {
    use fa_proto::v1::risk_event::Kind;
    fa_proto::v1::RiskEvent {
        event_id: ulid::Ulid::generate().to_string(),
        kind: if raised { Kind::HaltRaised } else { Kind::HaltCleared } as i32,
        at: h.since.or_else(|| Some(fa_proto::now())),
        account: RISK_EVENT_ACCOUNT.to_owned(),
        scope: h.scope.clone(),
        reason: h.reason.clone(),
        ..Default::default()
    }
}

/// Actor recorded in `audit.log` for a control request (`ui` from control-api, else this service).
fn actor_of(req: &CtlRequest) -> String {
    match req.args.get("actor").map(String::as_str) {
        Some("ui") => "ui".into(),
        Some(a) if a.starts_with("system:") && a.len() < 64 => a.into(),
        _ => persist::ACTOR.into(),
    }
}

/// One control request. Ok → JSON for the reply message; Err → prefixed message
/// (`not_found:` / `conflict:` / `invalid:` / `loosen:`).
async fn handle_ctl(
    verb: &str,
    req: &CtlRequest,
    engine: &Shared,
    store: &Store,
    pipe: &Pipe,
) -> Result<serde_json::Value, String> {
    let now = now_ms();
    let actor = actor_of(req);
    match verb {
        "strategy" => {
            let action = req.args.get("action").map(String::as_str).unwrap_or("");
            if action == "list" {
                let ids: Vec<String> = engine.lock().await.strategy_ids();
                let mut evidence = std::collections::BTreeMap::new();
                if let Some(p) = store {
                    for id in &ids {
                        evidence.insert(id.clone(), evidence_for(engine, p, id).await);
                    }
                }
                return Ok(engine.lock().await.strategies_view(&evidence, now));
            }
            let Some(action) = StrategyAction::parse(action) else {
                return Err(format!(
                    "invalid: action {action:?}; use pause|resume|flatten|promote|shadow|retire|admit|list"
                ));
            };
            let id = req.args.get("strategy_id").cloned().unwrap_or_default();
            if id.is_empty() {
                return Err("invalid: strategy_id is required".into());
            }
            let ev = match store {
                Some(p) => evidence_for(engine, p, &id).await,
                None => Evidence::default(),
            };
            let (state, gate) = {
                let mut e = engine.lock().await;
                let state = e.strategy_action(&id, action, &ev, now)?;
                let gate = e.checklist(&id, &ev, now);
                (state, gate)
            };
            if let Some(p) = store {
                p.strategy_state(&state, None, gate.as_ref()).await;
                p.audit_as(
                    &actor,
                    &format!("strategy.{}", action.as_str()),
                    Some(&id),
                    serde_json::json!({"status": state.status.as_str(), "prev_status": state.prev_status.map(|s| s.as_str()), "gate": gate}),
                )
                .await;
            }
            let view = engine.lock().await.strategy_json(&state, gate.as_ref(), now);
            Ok(view)
        }
        "limits" if req.args.contains_key("calibrate") => {
            // limit calibration rules (spec 40 §7): suggestions; `apply` only ever tightens
            let opts: serde_json::Value =
                serde_json::from_str(req.args.get("calibrate").map(String::as_str).unwrap_or("{}")).unwrap_or_default();
            let rules = fa_risk::calibrate::CalibrationRules::default();
            let cal = engine.lock().await.calibration(&rules);
            let mut v = serde_json::to_value(&cal).unwrap_or_default();
            if opts["apply"].as_bool() == Some(true) && !cal.tightened.is_empty() {
                let patch: serde_json::Map<String, serde_json::Value> = cal
                    .tightened
                    .iter()
                    .filter_map(|k| cal.suggested.get(k).map(|x| (k.clone(), serde_json::json!(x))))
                    .collect();
                let patch = serde_json::Value::Object(patch);
                let (before, after) = {
                    let mut e = engine.lock().await;
                    let before = e.limits().clone();
                    (before, e.set_limits(&patch)?)
                };
                if let Some(p) = store {
                    p.audit_as(
                        &actor,
                        "limits.calibrated",
                        Some("paper"),
                        serde_json::json!({"set": patch, "before": before, "after": after, "calibration": cal}),
                    )
                    .await;
                }
                v["applied"] = patch;
            }
            Ok(v)
        }
        "limits" => {
            let Some(set) = req.args.get("set") else {
                let e = engine.lock().await;
                let mut v = serde_json::to_value(e.limits()).unwrap_or_default();
                v["allocator"] = e.allocator_spec().clone();
                v["payout_reserve_eur"] = e.payout_reserve().into();
                return Ok(v);
            };
            let mut patch: serde_json::Value =
                serde_json::from_str(set).map_err(|e| format!("invalid: set is not JSON: {e}"))?;
            // `allocator` is not a limit: a runtime switch (evolved allocator genomes), allowed
            // in both directions, audited and logged (D-061)
            let alloc = patch.as_object_mut().and_then(|o| o.remove("allocator"));
            let mut reply = serde_json::json!({});
            if let Some(spec) = alloc {
                let (before, after) = {
                    let mut e = engine.lock().await;
                    let before = e.allocator_spec().clone();
                    (before, e.set_allocator(&spec)?)
                };
                if let Some(p) = store {
                    p.audit_as(
                        &actor,
                        "allocator.changed",
                        Some("paper"),
                        serde_json::json!({"before": before, "after": after}),
                    )
                    .await;
                }
                tracing::warn!(%actor, %before, %after, "allocator switched at runtime");
                reply["allocator"] = after;
            }
            if patch.as_object().is_some_and(|o| !o.is_empty()) || !patch.is_object() {
                let (before, after) = {
                    let mut e = engine.lock().await;
                    let before = e.limits().clone();
                    let after = e.set_limits(&patch)?;
                    (before, after)
                };
                // payout quota (spec 40 §6.5, spec 42 §6): the raised quota is also today's
                // ledger.payout_quota row, which may only rise
                if let (Some(p), Some(q)) = (store, patch.get("daily_payout_quota_eur").and_then(|v| v.as_f64())) {
                    match p.raise_payout_quota(q).await {
                        Ok(row) => {
                            engine.lock().await.set_reserved_payout(p.payout_reserve_today().await.max(row.min(q)))
                        }
                        Err(e) => tracing::warn!(error = %e, "payout quota row not written"),
                    }
                }
                if let Some(p) = store {
                    p.audit_as(
                        &actor,
                        "limits.tightened",
                        Some("paper"),
                        serde_json::json!({"set": patch, "before": before, "after": after}),
                    )
                    .await;
                }
                tracing::warn!(%actor, %patch, "limits tightened at runtime");
                let alloc = reply.get("allocator").cloned();
                reply = serde_json::to_value(&after).unwrap_or_default();
                if let Some(a) = alloc {
                    reply["allocator"] = a;
                }
            } else if reply.get("allocator").is_none() {
                return Err("invalid: `set` is empty".into());
            }
            Ok(reply)
        }
        "flatten" => {
            let ids = engine.lock().await.flatten_all(now);
            if let Some(p) = store {
                p.audit_as(&actor, "flatten.portfolio", Some("paper"), serde_json::json!({"strategies": ids})).await;
            }
            tracing::warn!(%actor, ?ids, "portfolio flatten requested");
            Ok(serde_json::json!({"ok": true, "flattening": ids}))
        }
        "allocators" => Ok(engine.lock().await.tournament_view()),
        "pipeline" => {
            let Some(pipe) = pipe else {
                return Err("invalid: the strategy pipeline needs [postgres]".into());
            };
            match req.args.get("action").map(String::as_str).unwrap_or("status") {
                "status" => Ok(pipe.status().await),
                "run" => {
                    if let Some(p) = store {
                        p.audit_as(&actor, "pipeline.run", Some("paper"), serde_json::json!({})).await;
                    }
                    Ok(pipe.round().await)
                }
                other => Err(format!("invalid: pipeline action {other:?}; use status|run")),
            }
        }
        _ => Err(format!("invalid: unknown verb {verb}")),
    }
}

/// Gate evidence of one strategy from the validation registry.
async fn evidence_for(engine: &Shared, p: &Persist, id: &str) -> Evidence {
    let (cands, genome, generation) = {
        let e = engine.lock().await;
        let rec = e.strategy(id).and_then(|s| s.pipeline.clone());
        (e.candidate_ids(id), rec.as_ref().map(|r| r.genome.clone()), rec.map(|r| r.generation))
    };
    let refs: Vec<&str> = cands.iter().map(String::as_str).collect();
    p.evidence(&refs, genome.as_deref(), generation.as_deref()).await
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();
    fa_bus::metrics::serve_from_env("risk-engine", env!("CARGO_PKG_VERSION")); // GET /metrics on FA_METRICS_BIND (spec 60 §4, D-056)
    let cfg = PlatformConfig::load(&args.config).context("loading platform config")?;
    let settings = match std::fs::read_to_string(&args.risk) {
        Ok(src) => Settings::parse(&src).with_context(|| format!("parsing {}", args.risk.display()))?,
        Err(e) => {
            tracing::warn!(path = %args.risk.display(), error = %e, "risk settings file not readable; using built-in risk.toml");
            Settings::parse(include_str!("../risk.toml"))?
        }
    };
    tracing::info!(
        mode = %settings.mode,
        account = %settings.account,
        allocator = %settings.allocator,
        starting_cash_eur = settings.starting_cash_eur,
        max_drawdown = settings.limits.max_drawdown,
        max_daily_loss = settings.limits.max_daily_loss,
        max_position_frac = settings.limits.max_position_frac,
        "risk-engine starting (paper only)"
    );
    let limits = serde_json::to_value(&settings.limits).unwrap_or_default();
    let (snapshot_secs, reconcile_secs) = (settings.snapshot_secs, settings.reconcile_secs);
    let fx_settings = settings.fx.clone();
    let refdata_refresh_secs = settings.refdata_refresh_secs;
    let engine: Shared = Arc::new(Mutex::new(Engine::new(settings.clone())));
    let store: Store = if cfg.postgres.url.is_empty() {
        tracing::warn!("no [postgres] url configured: paper book is in memory only (resets on restart)");
        None
    } else {
        match Persist::open(&cfg.postgres.url, &settings.account, settings.starting_cash_eur).await {
            Ok((p, ledger)) => {
                tracing::info!(
                    book = %p.book(),
                    cash_eur = ledger.cash_eur,
                    positions = ledger.positions.values().filter(|x| x.quantity != 0.0).count(),
                    "paper book loaded from postgres"
                );
                let states = p.load_strategy_states().await;
                let with_books = states.iter().filter(|(_, b)| b.is_some()).count();
                tracing::info!(strategies = states.len(), with_books, "strategy lifecycle rows and sub-books loaded");
                let refs = p.load_refdata().await;
                let fx = p.load_fx(&fx_settings.source, fx_settings.history_days).await;
                let reserve = p.payout_reserve_today().await;
                let risk_state = p.load_risk_state().await;
                let mut e = engine.lock().await;
                e.seed_ledger(ledger);
                e.seed_strategies(states);
                if let Some(r) = risk_state {
                    // kill switch / halts / drawdown peak survive the restart (spec 40 §8)
                    tracing::warn!(halts = ?r.halt_scopes(), "risk state restored from ledger.risk_state");
                    e.restore_risk(r);
                }
                if let Some(r) = refs {
                    tracing::info!(instruments = r.len(), "reference data loaded");
                    e.set_refdata(r);
                }
                if let Some(t) = fx {
                    tracing::info!(rates = t.len(), currencies = ?t.currencies().collect::<Vec<_>>(), "fx table loaded");
                    let _ = e.set_fx(t, now_ms());
                }
                e.set_reserved_payout(reserve);
                drop(e);
                p.audit("limits.loaded", Some(&settings.account), limits).await;
                Some(Arc::new(p))
            }
            Err(e) => {
                tracing::warn!(error = %e, "postgres unavailable: paper book is in memory only (resets on restart)");
                None
            }
        }
    };
    let bus = Bus::connect(&cfg.nats.url).await.context("connecting to NATS")?;
    let instance_id = ulid::Ulid::generate().to_string();

    // heartbeat
    {
        let (bus, engine, instance_id) = (bus.clone(), engine.clone(), instance_id.clone());
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(args.heartbeat_secs));
            loop {
                tick.tick().await;
                let counters = {
                    let e = engine.lock().await;
                    let mut c: std::collections::HashMap<String, f64> =
                        e.counters.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect();
                    c.insert("equity_eur".into(), e.equity());
                    c.insert("cash_eur".into(), e.ledger().cash_eur);
                    c.insert(
                        "positions".into(),
                        e.ledger().positions.values().filter(|p| p.quantity != 0.0).count() as f64,
                    );
                    c.insert("gross_exposure_eur".into(), e.gross_exposure_eur());
                    c.insert("halted".into(), if e.halted() { 1.0 } else { 0.0 });
                    for (venue, eur) in e.exposure_by_venue() {
                        c.insert(format!("exposure_eur.venue.{venue}"), eur);
                    }
                    if let Some(rows) = e.tournament_view()["entries"].as_array() {
                        for r in rows {
                            if let (Some(n), Some(v)) = (r["name"].as_str(), r["cumulative_return"].as_f64()) {
                                c.insert(format!("allocator.{n}.return"), v);
                            }
                        }
                    }
                    c
                };
                let hb = Heartbeat {
                    service: SERVICE.into(),
                    instance_id: instance_id.clone(),
                    at: Some(fa_proto::now()),
                    version: env!("CARGO_PKG_VERSION").into(),
                    counters,
                };
                if let Err(e) = bus.publish(subjects::heartbeat(SERVICE), &hb).await {
                    tracing::warn!(error = %e, "heartbeat publish failed");
                }
            }
        });
    }

    // risk.state
    {
        let (bus, engine) = (bus.clone(), engine.clone());
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(args.state_secs.max(1)));
            loop {
                tick.tick().await;
                let st = engine.lock().await.state(now_ms());
                if let Err(e) = bus.publish(subjects::RISK_STATE.to_owned(), &st).await {
                    tracing::warn!(error = %e, "risk.state publish failed");
                }
            }
        });
    }

    // signals
    {
        let (bus, engine) = (bus.clone(), engine.clone());
        let mut sub = bus.subscribe::<ExposureDistribution>("signal.exposure.>".into()).await?;
        tokio::spawn(async move {
            while let Some((_, m)) = sub.next().await {
                engine.lock().await.on_signal(&m, now_ms());
            }
        });
    }

    // trades: marks + volume (+ equity-driven halts)
    {
        let (bus, engine, store) = (bus.clone(), engine.clone(), store.clone());
        let mut sub = bus.subscribe::<Envelope>(subjects::ALL_MD_TRADES.into()).await?;
        tokio::spawn(async move {
            while let Some((_, env)) = sub.next().await {
                if let Some(Payload::Trade(t)) = &env.payload
                    && let Ok(uuid) = Uuid::parse_str(&env.instrument_id)
                {
                    let at = ms(env.event_time.as_ref()).unwrap_or_else(now_ms);
                    let outs = engine.lock().await.on_trade(InstrumentId(uuid), at, t.price, t.size);
                    publish_outs(&bus, &store, &engine, outs).await;
                }
            }
        });
    }

    // order.> : intents (foreign ones are gated), approvals (remembered), events (booked)
    {
        let (bus, engine, store) = (bus.clone(), engine.clone(), store.clone());
        let mut sub = bus.subscribe_raw("order.>".into()).await?;
        tokio::spawn(async move {
            use fa_proto::Message;
            while let Some(m) = sub.next().await {
                let subject = m.subject.as_str();
                let stage = if subject.starts_with("order.intent.") {
                    Some(false)
                } else if subject.starts_with("order.approved.") {
                    Some(true)
                } else {
                    None
                };
                let outs = match stage {
                    Some(approved) => match OrderIntent::decode(m.payload.as_ref()) {
                        Ok(i) => {
                            if let Some(p) = &store {
                                p.intent(&i).await; // no-op for our own (already recorded) intents
                            }
                            engine.lock().await.on_intent_seen(&i, approved, now_ms())
                        }
                        Err(e) => {
                            tracing::debug!(subject, error = %e, "undecodable intent");
                            continue;
                        }
                    },
                    None if subject.starts_with("order.event.") => match OrderEvent::decode(m.payload.as_ref()) {
                        Ok(ev) => {
                            let (outs, booked) = {
                                let mut e = engine.lock().await;
                                let outs = e.on_order_event(&ev, now_ms());
                                (outs, e.take_booked_fills())
                            };
                            if let Some(p) = &store
                                && !booked.is_empty()
                            {
                                p.fills(booked).await;
                            }
                            outs
                        }
                        Err(e) => {
                            tracing::debug!(subject, error = %e, "undecodable order event");
                            continue;
                        }
                    },
                    None => continue,
                };
                publish_outs(&bus, &store, &engine, outs).await;
            }
        });
    }

    // halts and resumes (incl. the UI kill switch, D-030)
    {
        let mut halts = bus.subscribe::<Halt>("risk.halt.>".into()).await?;
        let (engine_h, store_h) = (engine.clone(), store.clone());
        tokio::spawn(async move {
            while let Some((_, h)) = halts.next().await {
                let ledger = {
                    let mut e = engine_h.lock().await;
                    e.on_halt(&h, now_ms());
                    e.ledger().clone()
                };
                if let Some(p) = &store_h {
                    p.audit(
                        "halt",
                        Some(&h.scope),
                        serde_json::json!({"reason": h.reason, "manual": h.manual, "raised_by": "bus"}),
                    )
                    .await;
                    p.snapshot(&ledger, "halt").await;
                }
            }
        });
        let (engine, store) = (engine.clone(), store.clone());
        let mut resumes = bus.subscribe_raw("risk.resume.>".into()).await?;
        tokio::spawn(async move {
            while let Some(m) = resumes.next().await {
                if let Some(scope) = m.subject.strip_prefix("risk.resume.") {
                    tracing::info!(scope, "resume received");
                    engine.lock().await.on_resume(scope);
                    if let Some(p) = &store {
                        p.audit("resume", Some(scope), serde_json::json!({})).await;
                    }
                }
            }
        });
    }

    // ingest.health.<source> (fa.v1 IngestHealth, or legacy D-054 JSON): `down` halts the source, `ok` clears it
    {
        let (engine, store) = (engine.clone(), store.clone());
        let mut sub = bus.subscribe_raw("ingest.health.>".into()).await?;
        tokio::spawn(async move {
            while let Some(m) = sub.next().await {
                let Some(source) = m.subject.strip_prefix("ingest.health.") else { continue };
                let Some((status, reason)) = fa_bus::ingest::health_status(&m.payload) else { continue };
                let (status, reason) = (status.as_str(), reason.as_str());
                let changed = engine.lock().await.on_source_health(source, status, reason, now_ms());
                if changed {
                    tracing::warn!(source, status, reason, "source health changed the halt set");
                    if let Some(p) = &store {
                        let action = if status == "down" { "halt" } else { "resume" };
                        p.audit(
                            action,
                            Some(&format!("source:{source}")),
                            serde_json::json!({"reason": reason, "manual": false, "raised_by": "ingest.health"}),
                        )
                        .await;
                    }
                }
            }
        });
    }

    // strategy pipeline + holdout locking (spec 35 §6.4, spec 41 §6.4): needs Postgres
    let pipe: Pipe = match &store {
        Some(p) => {
            let blob = fa_core::secrets::Secrets::load()
                .map_err(|e| tracing::warn!(error = %e, "no secrets: the pipeline cannot read Garage"))
                .ok()
                .and_then(|sec| {
                    fa_blob::S3BlobStore::from_config(&cfg, &sec)
                        .map_err(
                            |e| tracing::warn!(error = %e, "Garage unavailable: no generation discovery / live set"),
                        )
                        .ok()
                });
            let tsdb =
                (!cfg.victoriametrics.url.is_empty()).then(|| fa_tsdb::Tsdb::new(cfg.victoriametrics.url.clone()));
            Some(Arc::new(Pipeline::new(bus.clone(), p.clone(), blob, tsdb, engine.clone(), &settings)))
        }
        None => None,
    };
    if let Some(pp) = pipe.clone() {
        let every = Duration::from_secs(settings.pipeline.interval_secs.min(settings.holdout.check_secs).max(5));
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(every);
            loop {
                tick.tick().await; // the first tick fires at once: holdouts are ensured at start
                let st = pp.round().await;
                tracing::debug!(status = %st, "pipeline round");
            }
        });
        let pp = pipe.clone().expect("some");
        let mut results =
            bus.subscribe::<fa_proto::v1::JobResult>(subjects::job_result(pipeline::HOLDOUT_JOB_KIND)).await?;
        tokio::spawn(async move {
            while let Some((_, r)) = results.next().await {
                pp.on_job_result(&r).await;
            }
        });
    }

    // control plane: ctl.risk.strategy | limits | flatten | pipeline | allocators (request/reply, CtlReply)
    for verb in ["strategy", "limits", "flatten", "pipeline", "allocators"] {
        let (bus, engine, store, pipe) = (bus.clone(), engine.clone(), store.clone(), pipe.clone());
        let mut sub = bus.subscribe_raw(subjects::ctl("risk", verb)).await?;
        tokio::spawn(async move {
            while let Some(m) = sub.next().await {
                let Some(reply_to) = m.reply.clone() else { continue };
                let req = CtlRequest::decode(m.payload.as_ref()).unwrap_or_default();
                let (ok, message) = match handle_ctl(verb, &req, &engine, &store, &pipe).await {
                    Ok(v) => (true, v.to_string()),
                    Err(e) => (false, e),
                };
                let reply = CtlReply { ok, message };
                if let Err(e) = bus.client().publish(reply_to, reply.encode_to_vec().into()).await {
                    tracing::warn!(error = %e, "ctl reply failed");
                }
            }
        });
    }

    // persistence timers: position + strategy snapshots, dirty strategy rows, reconciliation (spec 42)
    if let Some(p) = store.clone() {
        let engine_s = engine.clone();
        let p_s = p.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(snapshot_secs.max(1)));
            tick.tick().await; // skip the immediate first tick
            loop {
                tick.tick().await;
                let (ledger, rows) = {
                    let e = engine_s.lock().await;
                    (e.ledger().clone(), e.strategy_rows())
                };
                p_s.snapshot(&ledger, "timer").await;
                for (s, book) in &rows {
                    p_s.strategy_state(s, Some(book), None).await;
                }
                let rec = engine_s.lock().await.risk_record();
                p_s.save_risk_state(&rec).await;
            }
        });
        let (engine_d, p_d) = (engine.clone(), p.clone());
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(1));
            loop {
                tick.tick().await;
                let (dirty, risk) = {
                    let mut e = engine_d.lock().await;
                    let d = e.take_dirty_strategies();
                    let r = e.take_risk_dirty().then(|| e.risk_record());
                    (d, r)
                };
                for (s, book) in &dirty {
                    p_d.strategy_state(s, Some(book), None).await;
                }
                if let Some(r) = risk {
                    p_d.save_risk_state(&r).await;
                }
            }
        });
        // reference data: periodic reload + debounced reload on ref.changed.> / ingest.unresolved.>
        let refresh = Arc::new(tokio::sync::Notify::new());
        for subject in ["ref.changed.>", "ingest.unresolved.>"] {
            let mut sub = bus.subscribe_raw(subject.into()).await?;
            let refresh = refresh.clone();
            tokio::spawn(async move {
                while let Some(m) = sub.next().await {
                    tracing::debug!(subject = %m.subject, "reference data notice");
                    refresh.notify_one();
                }
            });
        }
        let (engine_f, p_f, refresh_f) = (engine.clone(), p.clone(), refresh.clone());
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(refdata_refresh_secs.max(5)));
            tick.tick().await;
            loop {
                tokio::select! {
                    _ = tick.tick() => {}
                    _ = refresh_f.notified() => tokio::time::sleep(Duration::from_secs(2)).await, // debounce bursts
                }
                if let Some(r) = p_f.load_refdata().await {
                    tracing::debug!(instruments = r.len(), "reference data reloaded");
                    engine_f.lock().await.set_refdata(r);
                }
            }
        });
        // ECB daily reference rates → ref.fx_rate → in-memory FX table (spec 12 §6, spec 42 §6)
        let (engine_x, p_x, bus_x, fx_x) = (engine.clone(), p.clone(), bus.clone(), fx_settings.clone());
        tokio::spawn(async move {
            let http = reqwest::Client::builder()
                .timeout(Duration::from_secs(20))
                .user_agent(concat!("financial-auto risk-engine/", env!("CARGO_PKG_VERSION")))
                .build()
                .expect("http client");
            let mut tick = tokio::time::interval(Duration::from_secs(fx_x.ecb_refresh_secs.max(60)));
            loop {
                tick.tick().await;
                match p_x.refresh_ecb(&http).await {
                    Ok((date, n, new)) => {
                        tracing::info!(%date, currencies = n, new_rows = new, "ecb reference rates fetched");
                        engine_x.lock().await.counters.insert("fx.ecb_fetches", 1.0);
                        if new > 0 {
                            p_x.audit(
                                "fx.ecb_refresh",
                                Some(fa_warehouse::fx::ECB_SOURCE),
                                serde_json::json!({"date": date, "currencies": n, "new_rows": new}),
                            )
                            .await;
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %format!("{e:#}"), "ecb reference rates not fetched");
                        *engine_x.lock().await.counters.entry("fx.ecb_errors").or_insert(0.0) += 1.0;
                    }
                }
                if let Some(t) = p_x.load_fx(&fx_x.source, fx_x.history_days).await {
                    let outs = engine_x.lock().await.set_fx(t, now_ms());
                    publish_outs(&bus_x, &Some(p_x.clone()), &engine_x, outs).await;
                }
            }
        });
        // today's payout quota (spec 40 §6.5: deployable = equity − reserved quota)
        let (engine_q, p_q) = (engine.clone(), p.clone());
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(30));
            loop {
                tick.tick().await;
                let r = p_q.payout_reserve_today().await;
                engine_q.lock().await.set_reserved_payout(r);
            }
        });
        let (engine_r, bus_r) = (engine.clone(), bus.clone());
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(reconcile_secs.max(1)));
            tick.tick().await;
            loop {
                tick.tick().await;
                let ledger = engine_r.lock().await.ledger().clone();
                if let Some(d) = p.reconcile(&ledger).await {
                    p.audit("reconciliation.diff", Some(p.book()), serde_json::to_value(&d).unwrap_or_default()).await;
                    // spec 42 §6: a diff halts the venue until resolved (risk.resume.broker re-arms)
                    let h = Halt {
                        scope: "broker".into(),
                        reason: format!(
                            "reconciliation diff: cash memory {:.2} vs postgres {:.2}, {} position diffs",
                            d.cash_ours,
                            d.cash_theirs,
                            d.positions.len()
                        ),
                        since: Some(fa_proto::now()),
                        manual: false,
                    };
                    engine_r.lock().await.on_halt(&h, now_ms());
                    publish_outs(&bus_r, &Some(p.clone()), &engine_r, vec![Out::Halt(h)]).await;
                }
            }
        });
    }

    // decision loop; SIGTERM / SIGINT flush the book snapshot and every strategy row + sub-book
    // before exiting, so a clean stop (docker compose stop) loses no sub-book change (D-061)
    let mut tick = tokio::time::interval(Duration::from_millis(args.tick_ms.max(100)));
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    loop {
        tokio::select! {
            _ = tick.tick() => {
                let outs = engine.lock().await.tick(now_ms());
                publish_outs(&bus, &store, &engine, outs).await;
            }
            _ = term.recv() => break,
            _ = tokio::signal::ctrl_c() => break,
        }
    }
    if let Some(p) = &store {
        let (ledger, rows) = {
            let e = engine.lock().await;
            (e.ledger().clone(), e.strategy_rows())
        };
        p.snapshot(&ledger, "shutdown").await;
        for (s, book) in &rows {
            p.strategy_state(s, Some(book), None).await;
        }
        let rec = engine.lock().await.risk_record();
        p.save_risk_state(&rec).await;
        tracing::info!(strategies = rows.len(), "shutdown flush: book snapshot and strategy sub-books written");
    }
    Ok(())
}
