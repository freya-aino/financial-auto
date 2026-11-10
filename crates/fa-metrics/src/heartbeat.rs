//! Bridge from a service's heartbeat counters (spec 21 `svc.heartbeat.<service>`) to gauges, so
//! everything already reported in heartbeats is scrapeable without a second instrumentation.

use std::time::{SystemTime, UNIX_EPOCH};

/// Mirrors `counters` into `fa_heartbeat_counter{service,key}` and bumps
/// `fa_heartbeat_published_total{service}` / `fa_heartbeat_last_unix_seconds{service}`.
pub fn record<'a>(service: &str, counters: impl IntoIterator<Item = (&'a str, f64)>) {
    let g = crate::gauge("fa_heartbeat_counter", "Heartbeat counter values mirrored per service and key");
    for (k, v) in counters {
        g.set_with(&[("service", service), ("key", k)], v);
    }
    crate::counter("fa_heartbeat_published_total", "Heartbeats published by this process")
        .inc_with(&[("service", service)]);
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    crate::gauge("fa_heartbeat_last_unix_seconds", "Unix time of the last heartbeat published")
        .set_with(&[("service", service)], now);
}

#[cfg(test)]
mod tests {
    #[test]
    fn mirrors_counters_as_labelled_gauges() {
        super::record("svc-a", [("orders.ok", 3.0), ("lag", 0.5)]);
        super::record("svc-a", [("orders.ok", 4.0)]);
        let out = crate::render();
        assert!(out.contains("fa_heartbeat_counter{service=\"svc-a\",key=\"orders.ok\"} 4\n"), "{out}");
        assert!(out.contains("fa_heartbeat_counter{service=\"svc-a\",key=\"lag\"} 0.5\n"));
        assert!(out.contains("fa_heartbeat_published_total{service=\"svc-a\"} 2\n"));
    }
}
