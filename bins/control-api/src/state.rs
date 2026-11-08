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
    pub services: Arc<RwLock<BTreeMap<String, ServiceStatus>>>,
    pub live: Arc<RwLock<Live>>,
}

impl AppState {
    pub fn new(cfg: PlatformConfig, sources_path: PathBuf, nats: Option<async_nats::Client>) -> Self {
        Self {
            cfg: Arc::new(cfg),
            sources_path: Arc::new(sources_path),
            nats,
            services: Arc::default(),
            live: Arc::default(),
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
