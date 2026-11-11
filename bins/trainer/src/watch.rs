//! Lifecycle watchers of `trainer serve` (spec 36 §6.3 drift triggers, §8 auto-rollback).
//!
//! **Drift triggers** (`FA_TRAIN_DRIFT_TRIGGERS=1`, default off): an early `train.generation` of
//! the active snapshot of a timescale when
//! * the live expert health (`latent.expert.health`, `source = live`) reports
//!   `resid_corr_max > FA_TRIGGER_RESID_CORR` (0.6) on [`SUSTAIN`] consecutive reports
//!   (residual-correlation breakdown, spec 32 §8);
//! * the serving MCMC sampler's ECE in model-runtime's heartbeat
//!   (`mcmc.<ts>.e<k>.<sampler>.ece` with `.serving = 1`) exceeds `FA_TRIGGER_ECE` (0.15) on
//!   [`SUSTAIN`] consecutive heartbeats (calibration degradation, spec 33);
//! * an adapter job result reports a frozen adapter or a frozen LoRA (adapter drift beyond its
//!   guard, spec 36 §6.5).
//!
//! Region-occupancy drift already requests adapters from model-runtime (spec 30 §6.6). Triggers
//! respect a per-timescale cooldown (`FA_TRIGGER_COOLDOWN`, default 6h).
//!
//! **Auto-rollback** (`FA_AUTO_ROLLBACK=1`, default off): when a risk halt fires
//! (`risk.halt.>`) within `FA_ROLLBACK_WINDOW` (default 6h) of a snapshot's promotion (the
//! promotion time is read from `audit.log`, action `latent.snapshot.promote`), the trainer asks
//! control-api to roll that timescale back (`POST {FA_CONTROL_API_URL}/api/latent/snapshots/{id}/rollback`,
//! default `http://control-api:7700`), so the rollback goes through the same audited route as a
//! manual one (the previous snapshot is restored together with its last adapter). One rollback
//! per promotion.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    time::{Duration, Instant},
};

use fa_bus::{Bus, subjects};
use fa_proto::v1::{ExpertHealth, Heartbeat, JobResult};
use futures::StreamExt;
use serde_json::{Value, json};

use crate::{plan, run::Kind};

pub const SUSTAIN: usize = 3;

fn env_f(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.trim().parse().ok()).filter(|v: &f64| v.is_finite()).unwrap_or(d)
}

fn env_on(k: &str) -> bool {
    std::env::var(k).is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

fn env_dur(k: &str, d: Duration) -> Duration {
    std::env::var(k).ok().and_then(|v| plan::parse_every(&v).ok().flatten()).unwrap_or(d)
}

/// Why a timescale should start an early generation.
#[derive(Debug, Clone, PartialEq)]
pub enum Trigger {
    ResidCorr { ts: String, value: f64 },
    McmcEce { ts: String, value: f64 },
    AdapterFrozen { ts: String, snapshot: String },
}

impl Trigger {
    pub fn timescale(&self) -> &str {
        match self {
            Self::ResidCorr { ts, .. } | Self::McmcEce { ts, .. } | Self::AdapterFrozen { ts, .. } => ts,
        }
    }

    pub fn describe(&self) -> Value {
        match self {
            Self::ResidCorr { ts, value } => {
                json!({ "trigger": "resid_corr_breakdown", "timescale": ts, "value": value })
            }
            Self::McmcEce { ts, value } => json!({ "trigger": "mcmc_calibration", "timescale": ts, "value": value }),
            Self::AdapterFrozen { ts, snapshot } => {
                json!({ "trigger": "adapter_drift", "timescale": ts, "snapshot": snapshot })
            }
        }
    }
}

/// Pure trigger state (sustain counters + cooldown).
#[derive(Debug)]
pub struct TriggerState {
    pub resid_corr_max: f64,
    pub ece_max: f64,
    pub cooldown: Duration,
    streak_corr: HashMap<String, usize>,
    streak_ece: HashMap<String, usize>,
    last_fired: HashMap<String, Instant>,
}

impl TriggerState {
    pub fn new(resid_corr_max: f64, ece_max: f64, cooldown: Duration) -> Self {
        Self {
            resid_corr_max,
            ece_max,
            cooldown,
            streak_corr: HashMap::new(),
            streak_ece: HashMap::new(),
            last_fired: HashMap::new(),
        }
    }

    fn fire(&mut self, t: Trigger, now: Instant) -> Option<Trigger> {
        let ts = t.timescale().to_owned();
        if self.last_fired.get(&ts).is_some_and(|at| now.duration_since(*at) < self.cooldown) {
            return None;
        }
        self.last_fired.insert(ts.clone(), now);
        self.streak_corr.remove(&ts);
        self.streak_ece.remove(&ts);
        Some(t)
    }

    pub fn on_health(&mut self, ts: &str, source: &str, resid_corr_max: f64, now: Instant) -> Option<Trigger> {
        if source != "live" {
            return None;
        }
        let s = self.streak_corr.entry(ts.to_owned()).or_default();
        *s = if resid_corr_max > self.resid_corr_max { *s + 1 } else { 0 };
        (*s >= SUSTAIN)
            .then_some(())
            .and_then(|()| self.fire(Trigger::ResidCorr { ts: ts.into(), value: resid_corr_max }, now))
    }

    /// Worst ECE of a serving sampler per timescale from model-runtime heartbeat counters.
    pub fn serving_ece(counters: &HashMap<String, f64>) -> BTreeMap<String, f64> {
        let mut out: BTreeMap<String, f64> = BTreeMap::new();
        for (k, v) in counters {
            let Some(prefix) = k.strip_suffix(".ece").filter(|p| p.starts_with("mcmc.")) else { continue };
            if counters.get(&format!("{prefix}.serving")).copied().unwrap_or(0.0) < 0.5 {
                continue;
            }
            let ts = prefix.split('.').nth(1).unwrap_or_default().to_owned();
            let e = out.entry(ts).or_insert(0.0);
            *e = e.max(*v);
        }
        out
    }

    pub fn on_heartbeat(&mut self, counters: &HashMap<String, f64>, now: Instant) -> Vec<Trigger> {
        let mut out = Vec::new();
        for (ts, ece) in Self::serving_ece(counters) {
            let s = self.streak_ece.entry(ts.clone()).or_default();
            *s = if ece > self.ece_max { *s + 1 } else { 0 };
            if *s >= SUSTAIN
                && let Some(t) = self.fire(Trigger::McmcEce { ts, value: ece }, now)
            {
                out.push(t);
            }
        }
        out
    }

    pub fn on_adapter_result(&mut self, result: &Value, now: Instant) -> Option<Trigger> {
        let frozen = result["frozen"] == json!(true) || result["lora"]["frozen"] == json!(true);
        let ts = result["timescale"].as_str()?;
        let snap = result["snapshot_id"].as_str()?;
        frozen
            .then_some(())
            .and_then(|()| self.fire(Trigger::AdapterFrozen { ts: ts.into(), snapshot: snap.into() }, now))
    }
}

async fn enqueue_generation(bus: &Bus, hub: &fa_hub::HubClient, t: &Trigger) {
    let active = match plan::active_snapshots(hub).await {
        Ok(a) => a,
        Err(e) => {
            tracing::warn!(error = %e, "drift trigger: active snapshots unreadable");
            return;
        }
    };
    let rows: Vec<fa_hub::Snapshot> = active.into_iter().filter(|s| s.timescale == t.timescale()).collect();
    for p in plan::jobs_for(Kind::Generation, &rows, &plan::PlanConfig::default()) {
        let mut job = p.to_job();
        let mut spec: Value = serde_json::from_slice(&job.spec).unwrap_or_default();
        spec["trigger"] = t.describe();
        job.spec = serde_json::to_vec(&spec).unwrap_or_default();
        match bus.publish(subjects::jobs(&job.kind), &job).await {
            Ok(()) => {
                tracing::warn!(job = %job.job_id, trigger = %t.describe(), "drift trigger: early generation enqueued")
            }
            Err(e) => tracing::warn!(error = %e, "drift trigger enqueue failed"),
        }
    }
}

/// Spawns the enabled watchers.
pub fn spawn(bus: Bus, hub: Option<fa_hub::HubClient>, pg_url: Option<String>) {
    if env_on("FA_TRAIN_DRIFT_TRIGGERS") {
        match hub.clone() {
            Some(h) => {
                tokio::spawn(run_triggers(bus.clone(), h));
            }
            None => tracing::warn!("drift triggers disabled: Helix unavailable"),
        }
    }
    if env_on("FA_AUTO_ROLLBACK") {
        tokio::spawn(run_auto_rollback(bus, hub, pg_url));
    }
}

async fn run_triggers(bus: Bus, hub: fa_hub::HubClient) {
    let mut st = TriggerState::new(
        env_f("FA_TRIGGER_RESID_CORR", 0.6),
        env_f("FA_TRIGGER_ECE", 0.15),
        env_dur("FA_TRIGGER_COOLDOWN", Duration::from_secs(6 * 3600)),
    );
    let (Ok(mut health), Ok(mut beats), Ok(mut results)) = (
        bus.subscribe::<ExpertHealth>(subjects::LATENT_EXPERT_HEALTH.into()).await,
        bus.subscribe::<Heartbeat>(subjects::heartbeat("model-runtime")).await,
        bus.subscribe::<JobResult>(subjects::job_result("train.adapter")).await,
    ) else {
        tracing::warn!("drift triggers: subscriptions failed");
        return;
    };
    tracing::info!(
        resid_corr = st.resid_corr_max,
        ece = st.ece_max,
        cooldown_s = st.cooldown.as_secs(),
        "drift triggers armed"
    );
    loop {
        let fired: Vec<Trigger> = tokio::select! {
            Some((_, h)) = health.next() => st.on_health(&h.timescale, &h.source, h.resid_corr_max, Instant::now()).into_iter().collect(),
            Some((_, b)) = beats.next() => st.on_heartbeat(&b.counters.into_iter().collect(), Instant::now()),
            Some((_, r)) = results.next() => {
                let v: Value = serde_json::from_slice(&r.result).unwrap_or_default();
                st.on_adapter_result(&v, Instant::now()).into_iter().collect()
            }
            else => break,
        };
        for t in fired {
            enqueue_generation(&bus, &hub, &t).await;
        }
    }
}

/// Promotion times of snapshots from `audit.log` (newest first).
async fn promotions(pool: &sqlx::PgPool, since_ms: i64) -> Vec<(String, i64)> {
    sqlx::query_as::<_, (String, f64)>(
        "select subject, extract(epoch from at) * 1000 from audit.log \
         where action = 'latent.snapshot.promote' and subject is not null and at >= to_timestamp($1 / 1000.0) order by at desc",
    )
    .bind(since_ms as f64)
    .fetch_all(pool)
    .await
    .unwrap_or_default()
    .into_iter()
    .map(|(s, t)| (s, t as i64))
    .collect()
}

/// Which promoted snapshots a halt at `now_ms` rolls back: active ones promoted within `window_ms`
/// that were not rolled back already.
pub fn rollback_targets(
    promoted: &[(String, i64)],
    active: &[fa_hub::Snapshot],
    done: &HashSet<String>,
    now_ms: i64,
    window_ms: i64,
) -> Vec<String> {
    let mut out = Vec::new();
    for (id, at) in promoted {
        if now_ms - at > window_ms || done.contains(id) || out.contains(id) {
            continue;
        }
        if active.iter().any(|a| &a.snapshot_id == id && a.status == "active" && a.previous_active.is_some()) {
            out.push(id.clone());
        }
    }
    out
}

async fn run_auto_rollback(bus: Bus, hub: Option<fa_hub::HubClient>, pg_url: Option<String>) {
    let (Some(hub), Some(url)) = (hub, pg_url) else {
        tracing::warn!("auto-rollback disabled: needs Helix and Postgres");
        return;
    };
    let Ok(pool) = sqlx::postgres::PgPoolOptions::new().max_connections(1).connect(&url).await else {
        tracing::warn!("auto-rollback disabled: Postgres unreachable");
        return;
    };
    let window = env_dur("FA_ROLLBACK_WINDOW", Duration::from_secs(6 * 3600));
    let api = std::env::var("FA_CONTROL_API_URL").unwrap_or_else(|_| "http://control-api:7700".into());
    let Ok(mut halts) = bus.client().subscribe(subjects::ALL_RISK_HALTS).await else {
        tracing::warn!("auto-rollback disabled: risk.halt.> subscription failed");
        return;
    };
    let http = reqwest::Client::builder().timeout(Duration::from_secs(30)).build().unwrap_or_default();
    let mut done: HashSet<String> = HashSet::new();
    tracing::warn!(window_s = window.as_secs(), %api, "auto-rollback armed (FA_AUTO_ROLLBACK=1)");
    while let Some(msg) = halts.next().await {
        let now = crate::data::now_ms();
        let window_ms = window.as_millis() as i64;
        let promoted = promotions(&pool, now - window_ms).await;
        let active: Vec<fa_hub::Snapshot> = hub.list_snapshots().await.unwrap_or_default();
        for id in rollback_targets(&promoted, &active, &done, now, window_ms) {
            let body =
                json!({ "note": format!("auto_rollback: {} within {}s of promotion", msg.subject, window.as_secs()) });
            let r = http.post(format!("{api}/api/latent/snapshots/{id}/rollback")).json(&body).send().await;
            match r {
                Ok(resp) if resp.status().is_success() => {
                    tracing::warn!(snapshot = %id, halt = %msg.subject, "auto-rollback done");
                    done.insert(id);
                }
                Ok(resp) => tracing::warn!(snapshot = %id, status = %resp.status(), "auto-rollback refused"),
                Err(e) => tracing::warn!(snapshot = %id, error = %e, "auto-rollback request failed"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triggers_need_sustain_and_respect_the_cooldown() {
        let t0 = Instant::now();
        let mut st = TriggerState::new(0.6, 0.15, Duration::from_secs(3600));
        assert!(st.on_health("1m", "validation", 0.9, t0).is_none(), "fit-time health does not trigger");
        assert!(st.on_health("1m", "live", 0.9, t0).is_none());
        assert!(st.on_health("1m", "live", 0.9, t0).is_none());
        assert!(matches!(st.on_health("1m", "live", 0.9, t0), Some(Trigger::ResidCorr { .. })));
        for _ in 0..5 {
            assert!(st.on_health("1m", "live", 0.9, t0 + Duration::from_secs(60)).is_none(), "cooldown");
        }
        // a dip resets the streak
        let mut st = TriggerState::new(0.6, 0.15, Duration::from_secs(1));
        st.on_health("1s", "live", 0.9, t0);
        st.on_health("1s", "live", 0.1, t0);
        st.on_health("1s", "live", 0.9, t0);
        assert!(st.on_health("1s", "live", 0.9, t0).is_none());
        // ECE of the serving sampler only
        let hb: HashMap<String, f64> = [
            ("mcmc.1m.e0.smc.ece".to_owned(), 0.3),
            ("mcmc.1m.e0.smc.serving".to_owned(), 0.0),
            ("mcmc.1m.e1.exact.ece".to_owned(), 0.2),
            ("mcmc.1m.e1.exact.serving".to_owned(), 1.0),
        ]
        .into();
        assert_eq!(TriggerState::serving_ece(&hb).get("1m"), Some(&0.2));
        let mut fired = Vec::new();
        for _ in 0..SUSTAIN {
            fired.extend(st.on_heartbeat(&hb, t0));
        }
        assert_eq!(fired.len(), 1);
        let frozen = json!({ "timescale": "1h", "snapshot_id": "s", "frozen": false, "lora": { "frozen": true } });
        assert!(matches!(st.on_adapter_result(&frozen, t0), Some(Trigger::AdapterFrozen { .. })));
    }

    #[test]
    fn auto_rollback_targets_recent_promotions_once() {
        let snap = |id: &str, status: &str, prev: Option<&str>| fa_hub::Snapshot {
            id: 1,
            snapshot_id: id.into(),
            timescale: "1m".into(),
            encoder_family: "jepa_vae".into(),
            generation: 1,
            garage_key: String::new(),
            status: status.into(),
            created_at: None,
            carried_mass: None,
            manifest: None,
            previous_active: prev.map(str::to_owned),
        };
        let active =
            vec![snap("new", "active", Some("old")), snap("first", "active", None), snap("gone", "retired", Some("x"))];
        let promoted = vec![
            ("new".to_owned(), 9_000),
            ("first".to_owned(), 9_500),
            ("gone".to_owned(), 9_900),
            ("stale".to_owned(), 1),
        ];
        let mut done = HashSet::new();
        assert_eq!(rollback_targets(&promoted, &active, &done, 10_000, 5_000), vec!["new".to_owned()]);
        done.insert("new".to_owned());
        assert!(rollback_targets(&promoted, &active, &done, 10_000, 5_000).is_empty());
        assert!(rollback_targets(&promoted, &active, &HashSet::new(), 20_000, 5_000).is_empty(), "outside the window");
    }
}
