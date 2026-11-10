//! Periodic planner of `trainer serve` (spec 36 §6.3 cadence, §6.5 adapters, §6.7 classes).
//!
//! Every `--adapter-every` (default 1 h) it enqueues one `train.adapter` job per **active**
//! statistical snapshot (Helix `Snapshot{status: active}`, every timescale), and every
//! `--generation-every` (default off) one `train.generation` job per active snapshot with that
//! snapshot as parent. Jobs go onto the same `JOBS_TRAIN` queue the worker pulls from, so each
//! runs under its kind's GPU arbiter class (`online_adapter` P1 / `generation_training` P3, see
//! [`crate::run::Kind::gpu_class`]); `Job.priority` carries the class number. The planner never
//! runs a job itself. `trainer plan` prints what the next tick would enqueue.

use std::time::Duration;

use fa_encoders::STATISTICAL_FAMILY;
use fa_proto::v1::Job;
use serde::Serialize;

use crate::run::Kind;

/// Parses `90s`, `15m`, `1h`, `2d`, a bare number of seconds, or `off`/`0`/`none` (= disabled).
pub fn parse_every(s: &str) -> Result<Option<Duration>, String> {
    let t = s.trim().to_ascii_lowercase();
    if matches!(t.as_str(), "off" | "none" | "0" | "") {
        return Ok(None);
    }
    let (num, mult) = match t.char_indices().last() {
        Some((i, 's')) => (&t[..i], 1),
        Some((i, 'm')) => (&t[..i], 60),
        Some((i, 'h')) => (&t[..i], 3600),
        Some((i, 'd')) => (&t[..i], 86_400),
        _ => (t.as_str(), 1),
    };
    let n: f64 = num.trim().parse().map_err(|_| format!("bad interval `{s}` (use e.g. 30m, 1h, off)"))?;
    if !(n.is_finite() && n > 0.0) {
        return Ok(None);
    }
    Ok(Some(Duration::from_secs_f64(n * mult as f64)))
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanConfig {
    pub adapter_every: Option<Duration>,
    pub generation_every: Option<Duration>,
    /// History of a generation job.
    pub generation_window_hours: f64,
    /// `instruments` of generation jobs (`"all"` by default).
    pub generation_instruments: String,
    pub adapter_window_hours: f64,
}

impl Default for PlanConfig {
    fn default() -> Self {
        Self {
            adapter_every: Some(Duration::from_secs(3600)),
            generation_every: None,
            generation_window_hours: 6.0,
            generation_instruments: "all".into(),
            adapter_window_hours: 1.0,
        }
    }
}

/// One job the planner would enqueue.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Planned {
    pub kind: String,
    pub gpu_class: &'static str,
    pub priority: u32,
    pub snapshot_id: String,
    pub timescale: String,
    pub spec: serde_json::Value,
}

/// Steps of the timescale a planned window covers at least (a 1-hour window holds no 1d bars).
pub const MIN_PLANNED_STEPS: f64 = 32.0;

/// `hours`, raised to [`MIN_PLANNED_STEPS`] steps of `timescale` (unknown timescales unchanged).
pub fn planned_hours(timescale: &str, hours: f64) -> f64 {
    match timescale.parse::<fa_core::Timescale>() {
        Ok(ts) => hours.max(MIN_PLANNED_STEPS * ts.duration().as_secs_f64() / 3600.0),
        Err(_) => hours,
    }
}

/// Active snapshot rows → the jobs of one tick of `kind`.
pub fn jobs_for(kind: Kind, active: &[fa_hub::Snapshot], cfg: &PlanConfig) -> Vec<Planned> {
    active
        .iter()
        .filter(|s| s.encoder_family == STATISTICAL_FAMILY && s.status == "active")
        .map(|s| {
            let spec = match kind {
                Kind::Adapter => serde_json::json!({
                    "snapshot_id": s.snapshot_id, "timescale": s.timescale, "window_hours": planned_hours(&s.timescale, cfg.adapter_window_hours),
                }),
                _ => serde_json::json!({
                    "timescale": s.timescale, "parent_snapshot_id": s.snapshot_id,
                    "window_hours": planned_hours(&s.timescale, cfg.generation_window_hours),
                    "instruments": if cfg.generation_instruments == "all" { serde_json::json!("all") }
                        else { serde_json::json!(cfg.generation_instruments.split(',').map(str::trim).collect::<Vec<_>>()) },
                }),
            };
            Planned {
                kind: kind.job_kind(),
                gpu_class: kind.gpu_class(),
                priority: if kind == Kind::Adapter { 1 } else { 3 },
                snapshot_id: s.snapshot_id.clone(),
                timescale: s.timescale.clone(),
                spec,
            }
        })
        .collect()
}

impl Planned {
    pub fn to_job(&self) -> Job {
        Job {
            job_id: ulid::Ulid::generate().to_string(),
            kind: self.kind.clone(),
            priority: self.priority,
            snapshot_id: self.snapshot_id.clone(),
            spec: serde_json::to_vec(&self.spec).unwrap_or_default(),
            ..Default::default()
        }
    }
}

/// Active statistical snapshots of every timescale (Helix).
pub async fn active_snapshots(hub: &fa_hub::HubClient) -> anyhow::Result<Vec<fa_hub::Snapshot>> {
    let mut rows: Vec<fa_hub::Snapshot> = hub
        .list_snapshots()
        .await?
        .into_iter()
        .filter(|s| s.status == "active" && s.encoder_family == STATISTICAL_FAMILY)
        .collect();
    rows.sort_by(|a, b| a.timescale.cmp(&b.timescale).then(a.snapshot_id.cmp(&b.snapshot_id)));
    Ok(rows)
}

/// The schedule `trainer plan` prints.
pub fn describe(cfg: &PlanConfig, active: &[fa_hub::Snapshot]) -> serde_json::Value {
    let every = |d: Option<Duration>| d.map_or_else(|| "off".to_owned(), |d| format!("{}s", d.as_secs()));
    serde_json::json!({
        "adapter_every": every(cfg.adapter_every),
        "generation_every": every(cfg.generation_every),
        "active_snapshots": active.iter().map(|s| serde_json::json!({
            "snapshot_id": s.snapshot_id, "timescale": s.timescale, "generation": s.generation,
            "artifact": !s.garage_key.is_empty(),
        })).collect::<Vec<_>>(),
        "next_tick": {
            "adapter": if cfg.adapter_every.is_some() { jobs_for(Kind::Adapter, active, cfg) } else { vec![] },
            "generation": if cfg.generation_every.is_some() { jobs_for(Kind::Generation, active, cfg) } else { vec![] },
        },
        "note": "ticks start one interval after `trainer serve` starts; each job leases the GPU in its class when it runs",
    })
}

/// Planner loop: enqueues one tick of `kind` every `every`.
pub async fn run(bus: fa_bus::Bus, hub: Option<fa_hub::HubClient>, kind: Kind, every: Duration, cfg: PlanConfig) {
    let Some(hub) = hub else {
        tracing::warn!(kind = kind.as_str(), "planner disabled: Helix unavailable (active snapshots unknown)");
        return;
    };
    let mut tick = tokio::time::interval_at(tokio::time::Instant::now() + every, every);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    tracing::info!(kind = kind.as_str(), every_secs = every.as_secs(), "planner running");
    loop {
        tick.tick().await;
        let active = match active_snapshots(&hub).await {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!(error = %e, "planner: active snapshots unreadable; skipping tick");
                continue;
            }
        };
        for p in jobs_for(kind, &active, &cfg) {
            let job = p.to_job();
            match bus.publish(fa_bus::subjects::jobs(&job.kind), &job).await {
                Ok(()) => {
                    tracing::info!(job = %job.job_id, kind = %job.kind, snapshot = %p.snapshot_id, "planner enqueued")
                }
                Err(e) => tracing::warn!(error = %e, kind = %job.kind, "planner enqueue failed"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(id: &str, ts: &str, status: &str, family: &str) -> fa_hub::Snapshot {
        fa_hub::Snapshot {
            id: 1,
            snapshot_id: id.into(),
            timescale: ts.into(),
            encoder_family: family.into(),
            generation: 1,
            garage_key: String::new(),
            status: status.into(),
            created_at: None,
            carried_mass: None,
            manifest: None,
            previous_active: None,
        }
    }

    #[test]
    fn intervals_and_jobs_per_active_snapshot() {
        assert_eq!(parse_every("1h").unwrap(), Some(Duration::from_secs(3600)));
        assert_eq!(parse_every("90s").unwrap(), Some(Duration::from_secs(90)));
        assert_eq!(parse_every("15m").unwrap(), Some(Duration::from_secs(900)));
        assert_eq!(parse_every("120").unwrap(), Some(Duration::from_secs(120)));
        assert_eq!(parse_every("off").unwrap(), None);
        assert!(parse_every("soon").is_err());
        let active = vec![
            snap("a", "1m", "active", "statistical"),
            snap("b", "1h", "active", "statistical"),
            snap("c", "1m", "candidate", "statistical"),
            snap("d", "1m", "active", "fsq_hybrid"),
        ];
        let cfg = PlanConfig::default();
        let a = jobs_for(Kind::Adapter, &active, &cfg);
        assert_eq!(a.iter().map(|p| p.snapshot_id.as_str()).collect::<Vec<_>>(), vec!["a", "b"]);
        assert_eq!((a[0].gpu_class, a[0].priority), ("online_adapter", 1));
        assert_eq!(a[0].spec["snapshot_id"], "a");
        assert_eq!(a[0].spec["window_hours"], 1.0, "1m: 1 h holds 60 bars");
        assert_eq!(a[1].spec["window_hours"], 32.0, "1h: raised to 32 steps");
        assert_eq!(planned_hours("1d", 6.0), 768.0);
        assert_eq!(planned_hours("bogus", 6.0), 6.0);
        let g = jobs_for(Kind::Generation, &active, &cfg);
        assert_eq!((g[0].gpu_class, g[0].priority), ("generation_training", 3));
        assert_eq!(g[0].spec["parent_snapshot_id"], "a");
        assert_eq!(g[0].spec["instruments"], "all");
        let job = g[0].to_job();
        assert_eq!(job.kind, "train.generation");
        assert_eq!(job.snapshot_id, "a");
        let d = describe(&cfg, &active[..2]);
        assert_eq!(d["generation_every"], "off");
        assert_eq!(d["next_tick"]["adapter"].as_array().unwrap().len(), 2);
        assert!(d["next_tick"]["generation"].as_array().unwrap().is_empty());
    }
}
