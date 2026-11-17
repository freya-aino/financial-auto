//! Shared application state.
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use chrono::{DateTime, Utc};
use fa_bus::{async_nats, subjects};
use fa_core::config::PlatformConfig;
use fa_proto::{Message, v1::Heartbeat};
use futures::StreamExt;
use serde::Serialize;

use crate::live::{self, Live};

/// Latest heartbeat seen per service (spec 50 §4.1 `/api/services`).
#[derive(Debug, Clone, Serialize)]
pub struct ServiceStatus {
    pub service: String,
    pub instance_id: String,
    pub version: String,
    pub last_seen: DateTime<Utc>,
    pub counters: BTreeMap<String, f64>,
}

#[derive(Clone)]
pub struct AppState {
    pub cfg: Arc<PlatformConfig>,
    pub sources_path: Arc<PathBuf>,
    pub nats: Option<async_nats::Client>,
    /// HelixDB client for the persisted graph/latent routes (`graph_routes`, D-048); `FA_HELIX_URL` overrides `helix.url`.
    pub hub: fa_hub::HubClient,
    pub services: Arc<RwLock<BTreeMap<String, ServiceStatus>>>,
    pub live: Arc<RwLock<Live>>,
    /// Postgres pool, set once by `ledger_routes::install` when Postgres is reachable.
    pub pg: Arc<std::sync::OnceLock<sqlx::PgPool>>,
}

impl AppState {
    pub fn new(cfg: PlatformConfig, sources_path: PathBuf, nats: Option<async_nats::Client>) -> Self {
        Self {
            hub: fa_hub::HubClient::with_timeout(
                fa_hub::HubClient::url_from_env_or(&cfg.helix.url),
                std::time::Duration::from_secs(5),
            ),
            cfg: Arc::new(cfg),
            sources_path: Arc::new(sources_path),
            nats,
            services: Arc::default(),
            live: Arc::default(),
            pg: Arc::default(),
        }
    }

    pub fn nats_connected(&self) -> bool {
        self.nats.as_ref().is_some_and(|c| c.connection_state() == async_nats::connection::State::Connected)
    }

    pub fn live(&self) -> RwLockReadGuard<'_, Live> {
        self.live.read().expect("live lock")
    }

    pub fn live_mut(&self) -> RwLockWriteGuard<'_, Live> {
        self.live.write().expect("live lock")
    }

    /// Feeds latent/decoder/jobs/signal/order/risk/trade traffic into `live`.
    pub fn spawn_live_collector(&self) {
        let Some(client) = self.nats.clone() else {
            return;
        };
        let state = self.clone();
        tokio::spawn(async move {
            let mut subs = Vec::new();
            for subject in live::SUBJECTS {
                match client.subscribe(subject).await {
                    Ok(s) => subs.push(s),
                    Err(e) => tracing::error!(subject, error = %e, "live subscription failed"),
                }
            }
            let mut all = futures::stream::select_all(subs);
            while let Some(msg) = all.next().await {
                state.live_mut().ingest(&msg.subject, &msg.payload);
            }
        });
    }

    /// Keeps the live book's FX view fresh (review 2026-11-15 §3.2 item 3): every
    /// `FA_LIVE_FX_REFRESH_SECS` (default 300; `0` disables), once Postgres is attached, the
    /// latest `ref.fx_rate` rate per currency (newest `ts` across sources, so Yahoo intraday
    /// beats the ECB daily when fresher) and every instrument's `quote_ccy` are loaded and
    /// handed to [`Live::set_fx`]. Fills seen before the first load convert at the counted
    /// 1.0 fallback.
    pub fn spawn_live_fx_refresher(&self) {
        let every = std::env::var("FA_LIVE_FX_REFRESH_SECS").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(300);
        if every == 0 {
            tracing::info!("live FX refresh disabled (FA_LIVE_FX_REFRESH_SECS=0)");
            return;
        }
        let state = self.clone();
        tokio::spawn(async move {
            loop {
                let Some(pool) = state.pg.get().cloned() else {
                    tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                    continue;
                };
                match load_fx_view(&pool).await {
                    Ok((rates, ccys)) => {
                        tracing::debug!(rates = rates.len(), instruments = ccys.len(), "live FX view refreshed");
                        state.live_mut().set_fx(rates, ccys);
                    }
                    Err(e) => tracing::warn!(error = %e, "live FX view not refreshed"),
                }
                tokio::time::sleep(std::time::Duration::from_secs(every)).await;
            }
        });
    }

    /// Keeps the live view's trading mode current (D-121).
    ///
    /// The mode is asked for rather than listened for: it changes rarely and only through
    /// `ctl.risk.mode`, and a control plane that inferred it from the bus could hold a stale
    /// "paper" across a switch — which is the one direction in which being wrong is dangerous.
    /// Until the first successful read it is `None`, reported as `unknown`.
    pub fn spawn_trading_mode_refresher(&self) {
        let every = std::env::var("FA_TRADING_MODE_REFRESH_SECS").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(30);
        if every == 0 {
            tracing::info!("trading mode refresh disabled (FA_TRADING_MODE_REFRESH_SECS=0)");
            return;
        }
        let state = self.clone();
        tokio::spawn(async move {
            loop {
                let (mode, venues) = crate::console_routes::read_trading_mode(&state).await;
                if mode.is_some() {
                    state.live_mut().set_trading_mode(mode, venues);
                }
                tokio::time::sleep(std::time::Duration::from_secs(every)).await;
            }
        });
    }

    pub fn record_heartbeat(&self, hb: Heartbeat) {
        let status = ServiceStatus {
            service: hb.service.clone(),
            instance_id: hb.instance_id,
            version: hb.version,
            last_seen: Utc::now(),
            counters: hb.counters.into_iter().collect(),
        };
        self.services.write().expect("services lock").insert(hb.service, status);
    }

    /// Collects `svc.heartbeat.>` into `services`.
    pub fn spawn_heartbeat_collector(&self) {
        let Some(client) = self.nats.clone() else {
            return;
        };
        let state = self.clone();
        tokio::spawn(async move {
            let mut sub = match client.subscribe(subjects::ALL_HEARTBEATS).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(error = %e, "heartbeat subscription failed");
                    return;
                }
            };
            while let Some(msg) = sub.next().await {
                match Heartbeat::decode(msg.payload.as_ref()) {
                    Ok(hb) => state.record_heartbeat(hb),
                    Err(e) => tracing::warn!(subject = %msg.subject, error = %e, "bad heartbeat"),
                }
            }
        });
    }
}

/// The two maps [`Live::set_fx`] takes: ccy → latest `rate_to_eur`, instrument UUID → quote ccy.
async fn load_fx_view(
    pool: &sqlx::PgPool,
) -> Result<(BTreeMap<String, f64>, std::collections::HashMap<String, String>), sqlx::Error> {
    let rates: Vec<(String, f64)> = sqlx::query_as(
        "SELECT DISTINCT ON (upper(ccy)) upper(ccy), rate_to_eur::float8 \
         FROM ref.fx_rate ORDER BY upper(ccy), ts DESC",
    )
    .fetch_all(pool)
    .await?;
    let ccys: Vec<(String, String)> = sqlx::query_as(
        "SELECT id::text, upper(trim(quote_ccy)) FROM ref.instrument \
         WHERE quote_ccy IS NOT NULL AND trim(quote_ccy) <> ''",
    )
    .fetch_all(pool)
    .await?;
    Ok((rates.into_iter().collect(), ccys.into_iter().collect()))
}
