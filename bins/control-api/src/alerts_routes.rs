//! Alert receiver (spec 60 §6.5): vmalert evaluates `docker/vm-alerts.yaml` against
//! VictoriaMetrics and notifies control-api as if it were an Alertmanager
//! (`-notifier.url=http://control-api:7700/api/alerts/vmalert`, so vmalert POSTs the Alertmanager
//! v2 body to `/api/alerts/vmalert/api/v2/alerts`). A generic Alertmanager-style webhook body
//! (`{"alerts": [...]}`) is accepted on `POST /api/alerts/webhook`.
//!
//! The receiver keeps the active set in memory and acts on *transitions* only, because vmalert
//! re-sends every firing alert on every evaluation:
//! - new firing alert → JSON [`AlertEvent`] on NATS `ops.alert` + `audit.log` row
//!   (`actor = 'system:vmalert'`, `action = 'ops.alert.firing'`, `subject = alertname`);
//! - resolved (vmalert sends `endsAt` in the past) → the same with `status = "resolved"`;
//! - an active alert whose `endsAt` passes without a refresh (vmalert gone) → `resolved` with
//!   `expired = true`, like Alertmanager's resolve timeout.
//!
//! `severity = "none"` alerts (the always-firing `Watchdog`) are never published or audited; their
//! last arrival proves the rule → vmalert → control-api path is alive (`pipeline.ok` in
//! `GET /api/alerts`, and the `fa_alerts_watchdog_age_seconds` gauge).
//!
//! The module also probes the dependencies `/api/health` reports (TCP reachability) every 15 s
//! into `fa_dependency_up{dep}`, so HelixDB, which has no metrics endpoint, can be alerted on.
//!
//! History (spec 20 §5.4 `ops`): every transition is also stored in `ops.alert_event`
//! (`fa_warehouse::ops`). At start the store is restored from it once Postgres is attached:
//! `recent` gets the last 200 transitions and alerts whose latest transition is `firing` become
//! active again (refreshed from now), so a restart neither forgets the history nor re-publishes
//! a firing alert vmalert keeps re-sending. `GET /api/alerts/history?since=&alertname=&limit=`
//! reads the stored history.
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::state::AppState;

/// NATS subject every alert transition is published on (spec 60 §6.5).
pub const SUBJECT: &str = "ops.alert";
/// Transitions kept for `GET /api/alerts` `recent`.
const RECENT_CAP: usize = 200;
/// The Watchdog must have arrived within this window for `pipeline.ok`.
const WATCHDOG_MAX_AGE: chrono::TimeDelta = chrono::TimeDelta::minutes(3);
/// An active alert without `endsAt` expires this long after its last refresh.
const DEFAULT_TTL: chrono::TimeDelta = chrono::TimeDelta::minutes(5);
const SEVERITIES: [&str; 4] = ["critical", "warning", "info", "none"];

/// One alert as sent by vmalert / Alertmanager clients (v2 `PostableAlert`, webhook `Alert`).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IncomingAlert {
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
    #[serde(default)]
    pub annotations: BTreeMap<String, String>,
    #[serde(default)]
    pub starts_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub ends_at: Option<DateTime<Utc>>,
    #[serde(default, rename = "generatorURL")]
    pub generator_url: Option<String>,
    /// Webhook format only; v2 bodies derive it from `endsAt`.
    #[serde(default)]
    pub status: Option<String>,
}

/// What `ops.alert` carries and `GET /api/alerts` lists.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AlertEvent {
    pub status: String, // "firing" | "resolved"
    pub alertname: String,
    pub severity: String,
    pub summary: String,
    pub fingerprint: String,
    pub labels: BTreeMap<String, String>,
    pub annotations: BTreeMap<String, String>,
    pub starts_at: Option<DateTime<Utc>>,
    pub ends_at: Option<DateTime<Utc>>,
    pub generator_url: Option<String>,
    pub at: DateTime<Utc>,
    /// Resolved because vmalert stopped refreshing it, not because the rule cleared.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub expired: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ActiveAlert {
    #[serde(flatten)]
    pub event: AlertEvent,
    pub last_seen: DateTime<Utc>,
    pub refreshes: u64,
}

/// In-memory alert state; pure (no I/O), so the transition logic is unit-tested directly.
#[derive(Debug, Default)]
pub struct AlertStore {
    active: HashMap<String, ActiveAlert>,
    recent: VecDeque<AlertEvent>,
    watchdog_at: Option<DateTime<Utc>>,
    posts: u64,
    last_post_at: Option<DateTime<Utc>>,
}

/// Stable fingerprint of a label set (FNV-1a 64 over sorted `k=v\0`).
pub fn fingerprint(labels: &BTreeMap<String, String>) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (k, v) in labels {
        for b in k.bytes().chain(*b"=").chain(v.bytes()).chain([0]) {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    format!("{h:016x}")
}

impl AlertStore {
    /// Applies one notification batch; returns the transitions to publish (Watchdog excluded).
    pub fn apply(&mut self, alerts: Vec<IncomingAlert>, now: DateTime<Utc>) -> Vec<AlertEvent> {
        self.posts += 1;
        self.last_post_at = Some(now);
        let mut out = Vec::new();
        for a in alerts {
            let severity = a.labels.get("severity").cloned().unwrap_or_else(|| "warning".into());
            let resolved = match a.status.as_deref() {
                Some(s) => s == "resolved",
                None => a.ends_at.is_some_and(|e| e <= now),
            };
            if severity == "none" {
                if !resolved {
                    self.watchdog_at = Some(now);
                }
                continue;
            }
            let fp = fingerprint(&a.labels);
            let event = AlertEvent {
                status: if resolved { "resolved" } else { "firing" }.into(),
                alertname: a.labels.get("alertname").cloned().unwrap_or_else(|| "unnamed".into()),
                severity,
                summary: a.annotations.get("summary").cloned().unwrap_or_default(),
                fingerprint: fp.clone(),
                labels: a.labels,
                annotations: a.annotations,
                starts_at: a.starts_at,
                ends_at: a.ends_at,
                generator_url: a.generator_url,
                at: now,
                expired: false,
            };
            match (resolved, self.active.get_mut(&fp)) {
                (false, Some(cur)) => {
                    cur.last_seen = now;
                    cur.refreshes += 1;
                    cur.event.ends_at = event.ends_at;
                    cur.event.annotations = event.annotations; // `{{ $value }}` changes between evaluations
                    cur.event.summary = event.summary;
                }
                (false, None) => {
                    self.active.insert(fp, ActiveAlert { event: event.clone(), last_seen: now, refreshes: 0 });
                    out.push(event);
                }
                (true, Some(_)) => {
                    self.active.remove(&fp);
                    out.push(event);
                }
                (true, None) => {} // resolution of something we never saw firing (e.g. after a restart)
            }
        }
        for e in &out {
            self.remember(e.clone());
        }
        out
    }

    /// Resolves active alerts whose `endsAt` (or last refresh + 5 min) has passed.
    pub fn expire(&mut self, now: DateTime<Utc>) -> Vec<AlertEvent> {
        let dead: Vec<String> = self
            .active
            .iter()
            .filter(|(_, a)| a.event.ends_at.unwrap_or(a.last_seen + DEFAULT_TTL) <= now)
            .map(|(k, _)| k.clone())
            .collect();
        let mut out = Vec::new();
        for k in dead {
            if let Some(a) = self.active.remove(&k) {
                let mut e = a.event;
                e.status = "resolved".into();
                e.expired = true;
                e.at = now;
                self.remember(e.clone());
                out.push(e);
            }
        }
        out
    }

    /// Restores the store from stored transitions (newest first): `recent` in arrival order, and
    /// every fingerprint whose latest transition is `firing` as active again, refreshed at `now`
    /// (it expires like any active alert if vmalert no longer sends it). Alerts already active
    /// (received since start) are kept as they are.
    pub fn restore(&mut self, newest_first: Vec<AlertEvent>, now: DateTime<Utc>) {
        let mut latest: HashMap<String, AlertEvent> = HashMap::new();
        for e in &newest_first {
            latest.entry(e.fingerprint.clone()).or_insert_with(|| e.clone());
        }
        for (fp, e) in latest {
            if e.status == "firing" && !self.active.contains_key(&fp) {
                self.active.insert(fp, ActiveAlert { event: e, last_seen: now, refreshes: 0 });
            }
        }
        let since_start: Vec<AlertEvent> = self.recent.drain(..).collect();
        for e in newest_first.into_iter().take(RECENT_CAP).rev().chain(since_start) {
            self.remember(e);
        }
    }

    fn remember(&mut self, e: AlertEvent) {
        if self.recent.len() == RECENT_CAP {
            self.recent.pop_front();
        }
        self.recent.push_back(e);
    }

    pub fn active(&self) -> Vec<ActiveAlert> {
        let mut v: Vec<_> = self.active.values().cloned().collect();
        v.sort_by(|a, b| {
            sev_rank(&a.event.severity).cmp(&sev_rank(&b.event.severity)).then(a.event.at.cmp(&b.event.at))
        });
        v
    }

    pub fn watchdog_ok(&self, now: DateTime<Utc>) -> bool {
        self.watchdog_at.is_some_and(|t| now - t <= WATCHDOG_MAX_AGE)
    }

    fn update_gauges(&self, now: DateTime<Utc>) {
        let g = fa_bus::metrics::gauge("fa_alerts_active", "Alerts currently firing, as received from vmalert");
        for s in SEVERITIES {
            let n = self.active.values().filter(|a| a.event.severity == s).count();
            g.set_with(&[("severity", s)], n as f64);
        }
        if let Some(t) = self.watchdog_at {
            fa_bus::metrics::gauge(
                "fa_alerts_watchdog_age_seconds",
                "Seconds since vmalert last delivered the Watchdog alert",
            )
            .set((now - t).num_milliseconds() as f64 / 1000.0);
        }
    }
}

fn sev_rank(s: &str) -> u8 {
    match s {
        "critical" => 0,
        "warning" => 1,
        "info" => 2,
        _ => 3,
    }
}

#[derive(Clone)]
pub struct AlertsState {
    pub app: AppState,
    pub store: Arc<Mutex<AlertStore>>,
}

/// Routes plus the expiry sweep and the dependency probe.
pub fn router(app: AppState) -> Router {
    let st = AlertsState { app, store: Arc::default() };
    spawn_restore(st.clone());
    spawn_sweeper(st.clone());
    spawn_dependency_probe(st.app.clone());
    routes(st)
}

pub fn routes(st: AlertsState) -> Router {
    Router::new()
        .route("/api/alerts", get(list))
        .route("/api/alerts/history", get(history))
        .route("/api/alerts/vmalert/api/v2/alerts", post(receive_v2))
        .route("/api/alerts/webhook", post(receive_webhook))
        .with_state(st)
}

#[derive(Debug, Deserialize)]
struct ListQuery {
    limit: Option<usize>,
}

async fn list(State(st): State<AlertsState>, Query(q): Query<ListQuery>) -> Response {
    let now = Utc::now();
    let store = st.store.lock().expect("alert store");
    let limit = q.limit.unwrap_or(50).min(RECENT_CAP);
    let recent: Vec<_> = store.recent.iter().rev().take(limit).cloned().collect();
    Json(json!({
        "active": store.active(),
        "recent": recent,
        "pipeline": {
            "ok": store.watchdog_ok(now),
            "watchdog_seen_at": store.watchdog_at,
            "posts": store.posts,
            "last_post_at": store.last_post_at,
            "subject": SUBJECT,
        },
    }))
    .into_response()
}

/// vmalert / Alertmanager v2 `POST /api/v2/alerts`: a JSON array.
async fn receive_v2(State(st): State<AlertsState>, Json(alerts): Json<Vec<IncomingAlert>>) -> Response {
    ingest(&st, alerts).await
}

#[derive(Debug, Deserialize)]
struct WebhookBody {
    alerts: Vec<IncomingAlert>,
}

/// Alertmanager webhook body `{"alerts": [...]}` or a bare array.
async fn receive_webhook(State(st): State<AlertsState>, Json(body): Json<Value>) -> Response {
    let alerts = if body.is_array() {
        serde_json::from_value::<Vec<IncomingAlert>>(body)
    } else {
        serde_json::from_value::<WebhookBody>(body).map(|b| b.alerts)
    };
    match alerts {
        Ok(a) => ingest(&st, a).await,
        Err(e) => {
            (StatusCode::BAD_REQUEST, Json(json!({ "error": "bad_alerts", "detail": e.to_string() }))).into_response()
        }
    }
}

async fn ingest(st: &AlertsState, alerts: Vec<IncomingAlert>) -> Response {
    let n = alerts.len();
    fa_bus::metrics::counter("fa_alerts_received_total", "Alerts received from vmalert (incl. re-sends)").add(n as f64);
    let now = Utc::now();
    let events = {
        let mut store = st.store.lock().expect("alert store");
        let ev = store.apply(alerts, now);
        store.update_gauges(now);
        ev
    };
    let published = emit(&st.app, &events).await;
    (StatusCode::OK, Json(json!({ "received": n, "transitions": events.len(), "published": published })))
        .into_response()
}

/// Publishes each transition on `ops.alert` and writes an audit row; returns how many reached NATS.
async fn emit(app: &AppState, events: &[AlertEvent]) -> usize {
    let mut published = 0;
    for e in events {
        let body = serde_json::to_vec(e).unwrap_or_default();
        if let Some(nc) = &app.nats {
            match nc.publish(SUBJECT, body.into()).await {
                Ok(()) => published += 1,
                Err(err) => tracing::warn!(alert = %e.alertname, error = %err, "ops.alert not published"),
            }
        }
        fa_bus::metrics::counter("fa_alerts_transitions_total", "Alert transitions published on ops.alert")
            .inc_with(&[("status", &e.status), ("severity", &e.severity)]);
        tracing::info!(alert = %e.alertname, status = %e.status, severity = %e.severity, summary = %e.summary, "ops.alert");
        audit(app, e).await;
        persist(app, e).await;
    }
    if published > 0
        && let Some(nc) = &app.nats
    {
        let _ = nc.flush().await;
    }
    published
}

async fn audit(app: &AppState, e: &AlertEvent) {
    let Some(pool) = app.pg.get() else {
        return;
    };
    let action = format!("ops.alert.{}", e.status);
    let payload = serde_json::to_value(e).unwrap_or(Value::Null);
    if let Err(err) =
        sqlx::query("INSERT INTO audit.log (actor, action, subject, payload) VALUES ('system:vmalert', $1, $2, $3)")
            .bind(&action)
            .bind(&e.alertname)
            .bind(payload)
            .execute(pool)
            .await
    {
        tracing::warn!(action, error = %err, "alert audit row not written");
    }
}

fn to_record(e: &AlertEvent) -> fa_warehouse::ops::AlertRecord {
    fa_warehouse::ops::AlertRecord {
        status: e.status.clone(),
        alertname: e.alertname.clone(),
        severity: e.severity.clone(),
        summary: e.summary.clone(),
        fingerprint: e.fingerprint.clone(),
        labels: json!(e.labels),
        annotations: json!(e.annotations),
        starts_at: e.starts_at,
        ends_at: e.ends_at,
        generator_url: e.generator_url.clone(),
        at: e.at,
        expired: e.expired,
    }
}

fn from_record(r: fa_warehouse::ops::AlertRecord) -> AlertEvent {
    AlertEvent {
        status: r.status,
        alertname: r.alertname,
        severity: r.severity,
        summary: r.summary,
        fingerprint: r.fingerprint,
        labels: serde_json::from_value(r.labels).unwrap_or_default(),
        annotations: serde_json::from_value(r.annotations).unwrap_or_default(),
        starts_at: r.starts_at,
        ends_at: r.ends_at,
        generator_url: r.generator_url,
        at: r.at,
        expired: r.expired,
    }
}

/// Stores a transition in `ops.alert_event` (spec 20 §5.4).
async fn persist(app: &AppState, e: &AlertEvent) {
    let Some(pool) = app.pg.get() else {
        return;
    };
    if let Err(err) = fa_warehouse::ops::record_alert(pool, &to_record(e)).await {
        tracing::warn!(alert = %e.alertname, error = %err, "alert transition not stored in ops.alert_event");
    }
}

/// Restores `recent` and the active set from `ops.alert_event` once Postgres is attached (it is
/// set by `ledger_routes::install` shortly after start); gives up after 5 minutes.
fn spawn_restore(st: AlertsState) {
    tokio::spawn(async move {
        for _ in 0..150 {
            if let Some(pool) = st.app.pg.get() {
                match fa_warehouse::ops::alerts(pool, None, None, RECENT_CAP as i64).await {
                    Ok(rows) => {
                        let n = rows.len();
                        let events = rows.into_iter().map(from_record).collect();
                        let now = Utc::now();
                        let mut store = st.store.lock().expect("alert store");
                        store.restore(events, now);
                        store.update_gauges(now);
                        tracing::info!(
                            transitions = n,
                            active = store.active.len(),
                            "alert state restored from ops.alert_event"
                        );
                    }
                    Err(e) => tracing::warn!(error = %e, "alert history not restored"),
                }
                return;
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    });
}

#[derive(Debug, Deserialize)]
struct HistoryQuery {
    since: Option<DateTime<Utc>>,
    alertname: Option<String>,
    limit: Option<i64>,
}

/// `GET /api/alerts/history`: stored transitions, newest first (503 without Postgres).
async fn history(State(st): State<AlertsState>, Query(q): Query<HistoryQuery>) -> Response {
    let Some(pool) = st.app.pg.get() else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "postgres_unavailable"}))).into_response();
    };
    let limit = q.limit.unwrap_or(200).clamp(1, 5_000);
    match fa_warehouse::ops::alerts(pool, q.since, q.alertname.as_deref(), limit).await {
        Ok(rows) => Json(json!({"transitions": rows.into_iter().map(from_record).collect::<Vec<_>>()})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response(),
    }
}

fn spawn_sweeper(st: AlertsState) {
    tokio::spawn(async move {
        let mut t = tokio::time::interval(Duration::from_secs(30));
        loop {
            t.tick().await;
            let now = Utc::now();
            let events = {
                let mut store = st.store.lock().expect("alert store");
                let ev = store.expire(now);
                store.update_gauges(now);
                ev
            };
            emit(&st.app, &events).await;
        }
    });
}

/// `fa_dependency_up{dep}` (TCP reachability, same targets as `/api/health`) every 15 s.
fn spawn_dependency_probe(app: AppState) {
    tokio::spawn(async move {
        let g =
            fa_bus::metrics::gauge("fa_dependency_up", "Dependency reachable from control-api (TCP connect, 1 = up)");
        let mut t = tokio::time::interval(Duration::from_secs(15));
        loop {
            t.tick().await;
            let cfg = &app.cfg;
            let targets = [
                ("helix", cfg.helix.url.as_str()),
                ("victoriametrics", cfg.victoriametrics.url.as_str()),
                ("postgres", cfg.postgres.url.as_str()),
                ("garage", cfg.garage.endpoint.as_str()),
                ("nats", cfg.nats.url.as_str()),
            ];
            for (dep, url) in targets {
                let up = match host_port(url) {
                    Some(addr) => {
                        tokio::time::timeout(Duration::from_millis(500), tokio::net::TcpStream::connect(addr))
                            .await
                            .is_ok_and(|r| r.is_ok())
                    }
                    None => false,
                };
                g.set_with(&[("dep", dep)], if up { 1.0 } else { 0.0 });
            }
            g.set_with(&[("dep", "nats_client")], if app.nats_connected() { 1.0 } else { 0.0 });
        }
    });
}

/// `host:port` of `scheme://[user@]host[:port]/...`, with the scheme's default port.
fn host_port(url: &str) -> Option<String> {
    let (scheme, rest) = url.split_once("://")?;
    let authority = rest.split(['/', '?']).next()?;
    let authority = authority.rsplit_once('@').map_or(authority, |(_, a)| a);
    if authority.is_empty() {
        return None;
    }
    if authority.contains(':') {
        return Some(authority.to_owned());
    }
    let port = match scheme {
        "http" => 80,
        "https" => 443,
        "postgres" | "postgresql" => 5432,
        "nats" => 4222,
        _ => return None,
    };
    Some(format!("{authority}:{port}"))
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use fa_core::config::PlatformConfig;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    fn alert(name: &str, sev: &str, ends_in: i64, now: DateTime<Utc>) -> IncomingAlert {
        IncomingAlert {
            labels: [("alertname", name), ("severity", sev), ("service", "risk-engine")]
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v.to_owned()))
                .collect(),
            annotations: [("summary".to_owned(), format!("{name} summary"))].into_iter().collect(),
            starts_at: Some(now - chrono::TimeDelta::minutes(1)),
            ends_at: Some(now + chrono::TimeDelta::seconds(ends_in)),
            generator_url: None,
            status: None,
        }
    }

    #[test]
    fn transitions_only_once_per_firing_and_resolution() {
        let t0 = Utc::now();
        let mut s = AlertStore::default();
        let ev = s.apply(vec![alert("RiskHalted", "critical", 120, t0)], t0);
        assert_eq!(ev.len(), 1);
        assert_eq!(ev[0].status, "firing");
        assert_eq!(ev[0].summary, "RiskHalted summary");
        // re-send on the next evaluation: no transition, refresh counted
        let t1 = t0 + chrono::TimeDelta::seconds(30);
        assert!(s.apply(vec![alert("RiskHalted", "critical", 120, t1)], t1).is_empty());
        assert_eq!(s.active()[0].refreshes, 1);
        // vmalert's resolved notification has endsAt in the past
        let t2 = t1 + chrono::TimeDelta::seconds(30);
        let ev = s.apply(vec![alert("RiskHalted", "critical", -1, t2)], t2);
        assert_eq!(ev.len(), 1);
        assert_eq!(ev[0].status, "resolved");
        assert!(s.active().is_empty());
        // a resolution for something never seen is ignored
        assert!(s.apply(vec![alert("Other", "warning", -1, t2)], t2).is_empty());
        assert_eq!(s.recent.len(), 2);
    }

    #[test]
    fn watchdog_is_tracked_but_never_published() {
        let t0 = Utc::now();
        let mut s = AlertStore::default();
        assert!(!s.watchdog_ok(t0));
        assert!(s.apply(vec![alert("Watchdog", "none", 120, t0)], t0).is_empty());
        assert!(s.active().is_empty());
        assert!(s.watchdog_ok(t0 + chrono::TimeDelta::minutes(2)));
        assert!(!s.watchdog_ok(t0 + chrono::TimeDelta::minutes(4)));
    }

    #[test]
    fn restore_rebuilds_recent_and_active_without_republishing() {
        let now = Utc::now();
        let mut old = AlertStore::default();
        let fired = old.apply(vec![alert("A", "warning", 300, now), alert("B", "critical", 300, now)], now);
        let resolved = old.apply(vec![alert("B", "critical", -1, now)], now + chrono::TimeDelta::seconds(5));
        let mut stored: Vec<AlertEvent> = fired.into_iter().chain(resolved).collect();
        stored.reverse(); // newest first, as ops::alerts returns them
        let mut s = AlertStore::default();
        s.restore(stored, now + chrono::TimeDelta::seconds(10));
        assert_eq!(s.recent.len(), 3);
        assert_eq!(s.recent.back().unwrap().status, "resolved", "arrival order kept");
        let active: Vec<_> = s.active().into_iter().map(|a| a.event.alertname).collect();
        assert_eq!(active, vec!["A"], "only the alert whose latest transition fired");
        // vmalert re-sends A after the restart: a refresh, not a new transition
        assert!(s.apply(vec![alert("A", "warning", 300, now)], now + chrono::TimeDelta::seconds(20)).is_empty());
    }

    #[test]
    fn unrefreshed_alerts_expire() {
        let t0 = Utc::now();
        let mut s = AlertStore::default();
        s.apply(vec![alert("HeartbeatStale", "critical", 60, t0)], t0);
        assert!(s.expire(t0 + chrono::TimeDelta::seconds(30)).is_empty());
        let ev = s.expire(t0 + chrono::TimeDelta::seconds(61));
        assert_eq!(ev.len(), 1);
        assert!(ev[0].expired && ev[0].status == "resolved");
    }

    #[test]
    fn fingerprint_is_label_order_independent_and_distinct() {
        let a: BTreeMap<String, String> = [("a".into(), "1".into()), ("b".into(), "2".into())].into_iter().collect();
        let b: BTreeMap<String, String> = [("b".into(), "2".into()), ("a".into(), "1".into())].into_iter().collect();
        let c: BTreeMap<String, String> = [("a".into(), "12".into())].into_iter().collect();
        assert_eq!(fingerprint(&a), fingerprint(&b));
        assert_ne!(fingerprint(&a), fingerprint(&c));
    }

    #[test]
    fn host_port_defaults() {
        assert_eq!(host_port("postgres://fa:fa@postgres/fa").as_deref(), Some("postgres:5432"));
        assert_eq!(host_port("http://helix:8080").as_deref(), Some("helix:8080"));
        assert_eq!(host_port("nats://nats").as_deref(), Some("nats:4222"));
        assert_eq!(host_port("garbage"), None);
    }

    fn state() -> AlertsState {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let cfg = PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        AlertsState { app: AppState::new(cfg, format!("{root}/sources.yaml").into(), None), store: Arc::default() }
    }

    async fn body(r: Response) -> Value {
        serde_json::from_slice(&r.into_body().collect().await.unwrap().to_bytes()).unwrap()
    }

    #[tokio::test]
    async fn vmalert_v2_and_webhook_bodies_reach_the_store() {
        let st = state();
        let app = routes(st.clone());
        // vmalert's Alertmanager v2 body (what -notifier.url posts)
        let v2 = json!([{
            "labels": {"alertname": "ServiceScrapeDown", "severity": "critical", "service": "ingestor"},
            "annotations": {"summary": "ingestor /metrics not scraped for 1m"},
            "startsAt": "2026-11-11T01:00:00Z", "endsAt": "2999-01-01T00:00:00Z",
            "generatorURL": "http://127.0.0.1:8880/vmalert/alert?group_id=1&alert_id=2"
        }, {
            "labels": {"alertname": "Watchdog", "severity": "none"}, "annotations": {},
            "startsAt": "2026-11-11T01:00:00Z", "endsAt": "2999-01-01T00:00:00Z"
        }]);
        let r = app
            .clone()
            .oneshot(
                Request::post("/api/alerts/vmalert/api/v2/alerts")
                    .header("content-type", "application/json")
                    .body(Body::from(v2.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::OK);
        let b = body(r).await;
        assert_eq!(b["received"], 2);
        assert_eq!(b["transitions"], 1);
        assert_eq!(b["published"], 0, "no NATS in tests");

        // Alertmanager webhook body resolving it
        let wh = json!({"version": "4", "alerts": [{
            "status": "resolved",
            "labels": {"alertname": "ServiceScrapeDown", "severity": "critical", "service": "ingestor"},
            "annotations": {}, "startsAt": "2026-11-11T01:00:00Z", "endsAt": "2026-11-11T01:05:00Z"
        }]});
        let r = app
            .clone()
            .oneshot(
                Request::post("/api/alerts/webhook")
                    .header("content-type", "application/json")
                    .body(Body::from(wh.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(body(r).await["transitions"], 1);

        let r = app.clone().oneshot(Request::get("/api/alerts").body(Body::empty()).unwrap()).await.unwrap();
        let b = body(r).await;
        assert_eq!(b["active"].as_array().unwrap().len(), 0);
        assert_eq!(b["recent"][0]["status"], "resolved");
        assert_eq!(b["recent"][1]["status"], "firing");
        assert_eq!(b["recent"][1]["alertname"], "ServiceScrapeDown");
        assert_eq!(b["pipeline"]["ok"], true);
        assert_eq!(b["pipeline"]["posts"], 2);

        let r = app
            .oneshot(
                Request::post("/api/alerts/webhook")
                    .header("content-type", "application/json")
                    .body(Body::from("{\"nope\": 1}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    }
}
