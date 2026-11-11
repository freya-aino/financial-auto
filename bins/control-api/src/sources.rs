//! Source catalog loader for `config/sources.yaml` (spec 11) and the in-memory mirror of the
//! ingestor's `ingest.health.*` / `ingest.gap.*` / `ingest.unresolved.*` frames (D-054).
//!
//! The health frames are JSON published by `go/services/ingestor` (`health.go`); they are kept
//! as `serde_json::Value` so new fields reach the UI without a control-api change. The store is
//! a process-wide `OnceLock` rather than a field of `AppState` so this module stays
//! self-contained; `spawn_health_collector` fills it from NATS.
use std::{
    collections::{BTreeMap, VecDeque},
    path::Path,
    sync::{OnceLock, RwLock},
};

use chrono::{DateTime, Utc};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::state::AppState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Catalog {
    pub version: u32,
    pub sources: Vec<Source>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub id: String,
    pub name: String,
    pub category: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub asset_classes: Vec<String>,
    #[serde(default)]
    pub transport: Vec<String>,
    pub auth: String,
    pub status: String,
    #[serde(default)]
    pub tos_notes: String,
    #[serde(default)]
    pub notes: String,
    /// Optional per-source token bucket (`{per_second, burst}`, D-050).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<Value>,
    /// Optional connector configuration (`symbols`, `queries`, `timeframe`, `extra`; D-050).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
}

pub fn load(path: &Path) -> anyhow::Result<Catalog> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_yaml_ng::from_str(&text)?)
}

/// Recent frames kept per source for `ingest.gap.*` and `ingest.unresolved.*`.
pub const RECENT_KEEP: usize = 50;

/// Latest health frame per source plus recent gap / unresolved frames.
#[derive(Debug, Default)]
pub struct HealthStore {
    health: BTreeMap<String, Value>,
    gaps: BTreeMap<String, VecDeque<Value>>,
    unresolved: BTreeMap<String, VecDeque<Value>>,
    pub frames_seen: u64,
}

impl HealthStore {
    /// Feeds one `ingest.<kind>.<source>` message: fa.v1 `IngestHealth|IngestGap|IngestUnresolved`
    /// or the legacy D-054 JSON, both mapped to the same JSON shape. Undecodable payloads are ignored.
    pub fn ingest(&mut self, subject: &str, payload: &[u8], at: DateTime<Utc>) {
        let Some(rest) = subject.strip_prefix("ingest.") else { return };
        let Some((kind, source)) = rest.split_once('.') else { return };
        let Some(mut v) = fa_bus::ingest::frame_json(kind, payload) else { return };
        if let Value::Object(m) = &mut v {
            m.insert("received_at".into(), json!(at));
        }
        self.frames_seen += 1;
        match kind {
            "health" => {
                self.health.insert(source.to_owned(), v);
            }
            "gap" => push_recent(self.gaps.entry(source.to_owned()).or_default(), v),
            "unresolved" => push_recent(self.unresolved.entry(source.to_owned()).or_default(), v),
            _ => {}
        }
    }

    pub fn health(&self, source: &str) -> Option<&Value> {
        self.health.get(source)
    }

    pub fn all_health(&self) -> &BTreeMap<String, Value> {
        &self.health
    }

    pub fn gaps(&self, source: &str) -> Vec<Value> {
        self.gaps.get(source).map(|d| d.iter().cloned().collect()).unwrap_or_default()
    }

    pub fn unresolved(&self, source: &str) -> Vec<Value> {
        self.unresolved.get(source).map(|d| d.iter().cloned().collect()).unwrap_or_default()
    }

    /// `running` as the ingestor reports it, if a frame was seen.
    pub fn running(&self, source: &str) -> Option<bool> {
        self.health.get(source).and_then(|h| h.get("running")).and_then(Value::as_bool)
    }
}

fn push_recent(q: &mut VecDeque<Value>, v: Value) {
    q.push_front(v);
    q.truncate(RECENT_KEEP);
}

static HEALTH: OnceLock<RwLock<HealthStore>> = OnceLock::new();

/// The process-wide health store.
pub fn health() -> &'static RwLock<HealthStore> {
    HEALTH.get_or_init(RwLock::default)
}

/// Subscribes to `ingest.>` and mirrors the frames into [`health`]. No-op without NATS.
pub fn spawn_health_collector(state: &AppState) {
    let Some(client) = state.nats.clone() else {
        return;
    };
    tokio::spawn(async move {
        let mut sub = match client.subscribe("ingest.>").await {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "ingest.> subscription failed");
                return;
            }
        };
        while let Some(msg) = sub.next().await {
            health().write().expect("health lock").ingest(&msg.subject, &msg.payload, Utc::now());
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_catalog_parses() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/sources.yaml");
        let cat = load(Path::new(path)).expect("config/sources.yaml must parse");
        assert!(cat.sources.len() >= 70, "expected the full catalog");
        let mut ids: Vec<_> = cat.sources.iter().map(|s| s.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), cat.sources.len(), "source ids must be unique");
        let yahoo = cat.sources.iter().find(|s| s.id == "yahoo").unwrap();
        assert_eq!(yahoo.config.as_ref().unwrap()["timeframe"], "1m");
        assert_eq!(yahoo.rate_limit.as_ref().unwrap()["burst"], 3);
    }

    #[test]
    fn health_store_keeps_latest_and_recent() {
        let mut h = HealthStore::default();
        let at = Utc::now();
        h.ingest("ingest.health.yahoo", br#"{"source":"yahoo","status":"ok","running":true}"#, at);
        h.ingest("ingest.health.yahoo", br#"{"source":"yahoo","status":"degraded","running":true}"#, at);
        h.ingest("ingest.gap.yahoo", br#"{"instrument_id":"i","missing_periods":3}"#, at);
        h.ingest("ingest.unresolved.yahoo", br#"{"venue_symbol":"X"}"#, at);
        h.ingest("ingest.health.hn", b"not json", at);
        h.ingest("svc.heartbeat.ingestor", b"{}", at);
        assert_eq!(h.frames_seen, 4);
        // fa.v1 IngestHealth frames map to the same JSON shape
        let proto = fa_proto::v1::IngestHealth {
            source: "gdelt".into(),
            status: fa_proto::v1::HealthStatus::Down as i32,
            reason: "connector exited".into(),
            ..Default::default()
        };
        h.ingest("ingest.health.gdelt", &fa_proto::Message::encode_to_vec(&proto), at);
        assert_eq!(h.frames_seen, 5);
        assert_eq!(h.health("gdelt").unwrap()["status"], "down");
        assert_eq!(h.running("gdelt"), Some(false));
        assert_eq!(h.health("yahoo").unwrap()["status"], "degraded");
        assert!(h.health("yahoo").unwrap()["received_at"].is_string());
        assert_eq!(h.running("yahoo"), Some(true));
        assert_eq!(h.running("hn"), None);
        assert_eq!(h.gaps("yahoo").len(), 1);
        assert_eq!(h.unresolved("yahoo")[0]["venue_symbol"], "X");
        for i in 0..(RECENT_KEEP + 5) {
            h.ingest("ingest.gap.yahoo", format!(r#"{{"n":{i}}}"#).as_bytes(), at);
        }
        let gaps = h.gaps("yahoo");
        assert_eq!(gaps.len(), RECENT_KEEP);
        assert_eq!(gaps[0]["n"], RECENT_KEEP + 4, "newest first");
    }
}
