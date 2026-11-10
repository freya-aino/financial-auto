//! risk-engine: meta-allocation + non-evolvable risk limits + paper book (specs 40, 42).
//!
//! Subscribes to `signal.exposure.>`, `order.>`, `risk.halt.>`, `risk.resume.>` and
//! `md.trade.>`; publishes `order.intent.paper` → `order.approved.paper` (or a rejection on
//! `order.event.paper`), `risk.state`, `risk.halt.<scope>` and `svc.heartbeat.risk-engine`.
//! Only paper trading exists: settings with any other mode are refused and non-paper intents
//! are rejected (D-029). Starting cash 100 000 EUR (D-028). The halt/resume contract follows
//! D-030. All decision logic is in [`engine`] (pure, tested); this file is I/O.
//!
//! When `[postgres]` is reachable the paper book is loaded from the ledger tables at startup and
//! every intent, fill, halt/resume (audit) and periodic position snapshot is persisted
//! ([`persist`]); without Postgres the engine stays in memory and says so once.
mod engine;
mod persist;
mod settings;

use std::{path::PathBuf, sync::Arc, time::Duration};

use anyhow::Context;
use clap::Parser;
use fa_bus::{Bus, subjects};
use fa_core::{InstrumentId, config::PlatformConfig};
use fa_proto::v1::{Envelope, ExposureDistribution, Halt, Heartbeat, OrderEvent, OrderIntent, envelope::Payload};
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
                _ => {}
            }
        }
        let r = match &o {
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();
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
                engine.lock().await.seed_ledger(ledger);
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

    // persistence timers: position snapshots + reconciliation (spec 42)
    if let Some(p) = store.clone() {
        let engine_s = engine.clone();
        let p_s = p.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(snapshot_secs.max(1)));
            tick.tick().await; // skip the immediate first tick
            loop {
                tick.tick().await;
                let ledger = engine_s.lock().await.ledger().clone();
                p_s.snapshot(&ledger, "timer").await;
            }
        });
        let engine_r = engine.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(reconcile_secs.max(1)));
            tick.tick().await;
            loop {
                tick.tick().await;
                let ledger = engine_r.lock().await.ledger().clone();
                if let Some(d) = p.reconcile(&ledger).await {
                    p.audit("reconciliation.diff", Some(p.book()), serde_json::to_value(&d).unwrap_or_default()).await;
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
