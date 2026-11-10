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
mod engine;
mod persist;
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
    engine::{Engine, Out},
    persist::Persist,
    settings::Settings,
};

type Shared = Arc<Mutex<Engine>>;
type Store = Option<Arc<Persist>>;

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
            Out::Rejected { account, event } => bus.publish(subjects::order_event(account), event).await,
            Out::Halt(h) => {
                tracing::warn!(scope = %h.scope, reason = %h.reason, "raising halt");
                bus.publish(subjects::risk_halt(&h.scope), h).await
            }
        };
        if let Err(e) = r {
            tracing::warn!(error = %e, "publish failed");
        }
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
async fn handle_ctl(verb: &str, req: &CtlRequest, engine: &Shared, store: &Store) -> Result<serde_json::Value, String> {
    let now = now_ms();
    let actor = actor_of(req);
    match verb {
        "strategy" => {
            let action = req.args.get("action").map(String::as_str).unwrap_or("");
            if action == "list" {
                let candidates: Vec<(String, Vec<String>)> = {
                    let e = engine.lock().await;
                    e.strategy_ids().into_iter().map(|id| (id.clone(), e.candidate_ids(&id))).collect()
                };
                let mut dsr = std::collections::BTreeMap::new();
                if let Some(p) = store {
                    for (id, cands) in &candidates {
                        let refs: Vec<&str> = cands.iter().map(String::as_str).collect();
                        dsr.insert(id.clone(), p.deflated_sharpe(&refs).await);
                    }
                }
                return Ok(engine.lock().await.strategies_view(&dsr, now));
            }
            let Some(action) = StrategyAction::parse(action) else {
                return Err(format!("invalid: action {action:?}; use pause|resume|flatten|promote|shadow|list"));
            };
            let id = req.args.get("strategy_id").cloned().unwrap_or_default();
            if id.is_empty() {
                return Err("invalid: strategy_id is required".into());
            }
            let dsr = match (action, store) {
                (StrategyAction::Promote, Some(p)) => {
                    let cands = engine.lock().await.candidate_ids(&id);
                    let refs: Vec<&str> = cands.iter().map(String::as_str).collect();
                    p.deflated_sharpe(&refs).await
                }
                _ => None,
            };
            let (state, gate) = {
                let mut e = engine.lock().await;
                let state = e.strategy_action(&id, action, dsr, now)?;
                let gate = e.checklist(&id, dsr, now);
                (state, gate)
            };
            if let Some(p) = store {
                p.strategy_state(&state, gate.as_ref()).await;
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
        "limits" => {
            let Some(set) = req.args.get("set") else {
                return Ok(serde_json::to_value(engine.lock().await.limits()).unwrap_or_default());
            };
            let patch: serde_json::Value =
                serde_json::from_str(set).map_err(|e| format!("invalid: set is not JSON: {e}"))?;
            let (before, after) = {
                let mut e = engine.lock().await;
                let before = e.limits().clone();
                let after = e.set_limits(&patch)?;
                (before, after)
            };
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
            Ok(serde_json::to_value(&after).unwrap_or_default())
        }
        "flatten" => {
            let ids = engine.lock().await.flatten_all(now);
            if let Some(p) = store {
                p.audit_as(&actor, "flatten.portfolio", Some("paper"), serde_json::json!({"strategies": ids})).await;
            }
            tracing::warn!(%actor, ?ids, "portfolio flatten requested");
            Ok(serde_json::json!({"ok": true, "flattening": ids}))
        }
        _ => Err(format!("invalid: unknown verb {verb}")),
    }
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
                tracing::info!(strategies = states.len(), "strategy lifecycle rows loaded");
                let mut e = engine.lock().await;
                e.seed_ledger(ledger);
                e.seed_strategies(states);
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

    // ingest.health.<source> (JSON frames from the ingestor): `down` halts the source, `ok` clears it
    {
        let (engine, store) = (engine.clone(), store.clone());
        let mut sub = bus.subscribe_raw("ingest.health.>".into()).await?;
        tokio::spawn(async move {
            while let Some(m) = sub.next().await {
                let Some(source) = m.subject.strip_prefix("ingest.health.") else { continue };
                let Ok(frame) = serde_json::from_slice::<serde_json::Value>(&m.payload) else { continue };
                let status = frame["status"].as_str().unwrap_or("");
                let reason = frame["reason"].as_str().unwrap_or("");
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

    // control plane: ctl.risk.strategy | ctl.risk.limits | ctl.risk.flatten (request/reply, CtlReply)
    for verb in ["strategy", "limits", "flatten"] {
        let (bus, engine, store) = (bus.clone(), engine.clone(), store.clone());
        let mut sub = bus.subscribe_raw(subjects::ctl("risk", verb)).await?;
        tokio::spawn(async move {
            while let Some(m) = sub.next().await {
                let Some(reply_to) = m.reply.clone() else { continue };
                let req = CtlRequest::decode(m.payload.as_ref()).unwrap_or_default();
                let (ok, message) = match handle_ctl(verb, &req, &engine, &store).await {
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
                let (ledger, states) = {
                    let e = engine_s.lock().await;
                    (e.ledger().clone(), e.strategy_states())
                };
                p_s.snapshot(&ledger, "timer").await;
                for s in &states {
                    p_s.strategy_state(s, None).await;
                }
            }
        });
        let (engine_d, p_d) = (engine.clone(), p.clone());
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(1));
            loop {
                tick.tick().await;
                let dirty = engine_d.lock().await.take_dirty_strategies();
                for s in &dirty {
                    p_d.strategy_state(s, None).await;
                }
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

    // decision loop
    let mut tick = tokio::time::interval(Duration::from_millis(args.tick_ms.max(100)));
    loop {
        tick.tick().await;
        let outs = engine.lock().await.tick(now_ms());
        publish_outs(&bus, &store, &engine, outs).await;
    }
}
