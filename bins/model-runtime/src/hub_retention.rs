//! Scheduled Helix retention (spec 20 §6.4, §6.5): model-runtime writes the `observation`
//! latents and the transition / region-status versions, so it also prunes them. A background
//! task runs [`fa_hub::HubClient::prune_observations`] with a per-timescale [`RetentionPolicy`]
//! every `--hub-prune-secs` (first run after `--hub-prune-delay-secs`): observations older than
//! the horizon are folded into per-region aggregates, versions older than it are thinned to the
//! one in force at the cut-off. It has its own client and never touches the bus path; a failed
//! run is logged, counted (`hub.retention.errors`) and retried at the next interval.

use std::time::Duration;

use fa_hub::{HubClient, PruneReport, RetentionPolicy};

use crate::Counters;

/// Per-run HTTP timeout (folding pages is heavier than the sink's writes).
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, PartialEq)]
pub struct RetentionConfig {
    pub url: String,
    /// Interval between runs; zero disables the task.
    pub every: Duration,
    /// Delay before the first run (lets the sink create the snapshot rows first).
    pub initial_delay: Duration,
    pub policy: RetentionPolicy,
}

/// `5s`, `90m`, `36h`, `14d`, `2w` (a bare number is seconds).
pub fn parse_duration(s: &str) -> Result<Duration, String> {
    let s = s.trim();
    let (num, unit) = s.split_at(s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len()));
    let n: u64 = num.parse().map_err(|_| format!("bad duration {s:?}"))?;
    let secs = match unit {
        "" | "s" => 1,
        "m" => 60,
        "h" => 3_600,
        "d" => 86_400,
        "w" => 7 * 86_400,
        _ => return Err(format!("bad duration unit in {s:?} (s, m, h, d, w)")),
    };
    Ok(Duration::from_secs(n.saturating_mul(secs)))
}

/// `--hub-retention`: comma-separated `timescale=duration` pairs; `default=duration` sets the
/// horizon of unlisted timescales (without it they are kept forever); `timescale=off` removes an
/// entry (that timescale then follows `default`).
pub fn parse_policy(spec: &str) -> Result<RetentionPolicy, String> {
    let mut p = RetentionPolicy::default();
    for part in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (k, v) = part.split_once('=').ok_or_else(|| format!("expected timescale=duration, got {part:?}"))?;
        let (k, v) = (k.trim(), v.trim());
        let d = if v.eq_ignore_ascii_case("off") { None } else { Some(parse_duration(v)?) };
        if k == "default" {
            p.default = d;
        } else if let Some(d) = d {
            p.horizons.insert(k.to_owned(), d);
        } else {
            p.horizons.remove(k);
        }
    }
    Ok(p)
}

async fn record(counters: &Counters, r: Result<&PruneReport, ()>) {
    let mut c = counters.lock().await;
    *c.entry("hub.retention.runs".into()).or_default() += 1.0;
    match r {
        Ok(r) => {
            *c.entry("hub.retention.pruned".into()).or_default() += r.pruned as f64;
            *c.entry("hub.retention.aggregates".into()).or_default() +=
                (r.aggregates_created + r.aggregates_updated) as f64;
            *c.entry("hub.retention.versions_pruned".into()).or_default() += r.versions_pruned as f64;
        }
        Err(()) => *c.entry("hub.retention.errors".into()).or_default() += 1.0,
    }
}

/// Starts the periodic task (no-op when `every` is zero or the policy keeps everything).
pub fn spawn(cfg: RetentionConfig, counters: Counters) {
    if cfg.every.is_zero() || (cfg.policy.horizons.is_empty() && cfg.policy.default.is_none()) {
        tracing::info!("helix retention disabled");
        return;
    }
    tracing::info!(
        every_secs = cfg.every.as_secs(),
        first_in_secs = cfg.initial_delay.as_secs(),
        policy = ?cfg.policy,
        "helix retention scheduled"
    );
    tokio::spawn(async move {
        let hub = HubClient::with_timeout(cfg.url.clone(), REQUEST_TIMEOUT);
        tokio::time::sleep(cfg.initial_delay).await;
        loop {
            let started = std::time::Instant::now();
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));
            match hub.prune_observations(now_ms, &cfg.policy).await {
                Ok(r) => {
                    tracing::info!(
                        pruned = r.pruned,
                        aggregates_created = r.aggregates_created,
                        aggregates_updated = r.aggregates_updated,
                        versions_pruned = r.versions_pruned,
                        took_ms = started.elapsed().as_millis() as u64,
                        "helix retention run"
                    );
                    record(&counters, Ok(&r)).await;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "helix retention run failed (retried next interval)");
                    record(&counters, Err(())).await;
                }
            }
            // region episodes (spec 30 §10 q6): same horizon as the observation latents
            for ts in ["1s", "1m", "1h", "1d", "1w"] {
                let Some(cut) = cfg.policy.cutoff_ms(ts, now_ms) else { continue };
                match hub.prune_region_episodes(ts, cut).await {
                    Ok(n) => {
                        *counters.lock().await.entry("hub.retention.episodes_pruned".into()).or_default() += n as f64
                    }
                    Err(e) => {
                        tracing::warn!(%ts, error = %e, "region episode prune failed (retried next interval)");
                        record(&counters, Err(())).await;
                    }
                }
            }
            tokio::time::sleep(cfg.every).await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_and_policies_parse() {
        assert_eq!(parse_duration("90m").unwrap(), Duration::from_secs(5_400));
        assert_eq!(parse_duration("14d").unwrap(), Duration::from_secs(14 * 86_400));
        assert_eq!(parse_duration("30").unwrap(), Duration::from_secs(30));
        assert!(parse_duration("3y").is_err() && parse_duration("d").is_err());
        let p = parse_policy("1s=2d, 1m=14d,1h=180d").unwrap();
        assert_eq!(p.horizon_for("1m"), Some(Duration::from_secs(14 * 86_400)));
        assert_eq!(p.horizon_for("1d"), None, "unlisted timescales are kept");
        let p = parse_policy("1m=1h,1h=off,default=30d").unwrap();
        assert_eq!(p.horizon_for("1h"), Some(Duration::from_secs(30 * 86_400)), "off falls back to default");
        assert_eq!(p.horizon_for("1w"), Some(Duration::from_secs(30 * 86_400)));
        assert!(parse_policy("1m").is_err());
        assert_eq!(parse_policy("").unwrap(), RetentionPolicy::default());
    }
}
