//! Process metrics: uptime, start time, RSS/VSZ from `/proc/self/statm`, build info.

use std::{
    sync::{LazyLock, Once},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

static START: LazyLock<(Instant, f64)> = LazyLock::new(|| {
    let unix = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    (Instant::now(), unix)
});
static REGISTERED: Once = Once::new();

/// `(resident, virtual)` bytes parsed from `/proc/self/statm` (pages × page size).
pub fn memory_bytes() -> Option<(u64, u64)> {
    parse_statm(&std::fs::read_to_string("/proc/self/statm").ok()?, page_size())
}

fn page_size() -> u64 {
    // Linux x86_64/aarch64 default; matched by every supported image.
    4096
}

pub(crate) fn parse_statm(s: &str, page: u64) -> Option<(u64, u64)> {
    let mut it = s.split_whitespace();
    let size: u64 = it.next()?.parse().ok()?;
    let resident: u64 = it.next()?.parse().ok()?;
    Some((resident * page, size * page))
}

/// Seconds since the process (well, this crate) started.
pub fn uptime_seconds() -> f64 {
    START.0.elapsed().as_secs_f64()
}

/// Registers the process collector and `fa_build_info{service,version} 1`. Idempotent.
pub fn register(service: &str, version: &str) {
    LazyLock::force(&START);
    let (service, version) = (service.to_owned(), version.to_owned());
    REGISTERED.call_once(move || {
        crate::gauge("fa_build_info", "Service name and version")
            .set_with(&[("service", &service), ("version", &version)], 1.0);
        crate::gauge("fa_process_start_time_seconds", "Process start, unix seconds").set(START.1);
        let uptime = crate::gauge("fa_process_uptime_seconds", "Seconds since process start");
        let rss = crate::gauge("fa_process_resident_memory_bytes", "Resident set size from /proc/self/statm");
        let vsz = crate::gauge("fa_process_virtual_memory_bytes", "Virtual memory size from /proc/self/statm");
        crate::register_collector(move || {
            uptime.set(uptime_seconds());
            if let Some((r, v)) = memory_bytes() {
                rss.set(r as f64);
                vsz.set(v as f64);
            }
        });
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn statm_parses_pages() {
        assert_eq!(super::parse_statm("1000 250 100 1 0 50 0\n", 4096), Some((250 * 4096, 1000 * 4096)));
        assert_eq!(super::parse_statm("garbage", 4096), None);
    }

    #[test]
    fn register_exposes_build_info_and_uptime() {
        super::register("test-svc", "1.2.3");
        super::register("other", "9"); // idempotent: first wins
        let out = crate::render();
        assert!(out.contains("fa_build_info{service=\"test-svc\",version=\"1.2.3\"} 1\n"), "{out}");
        assert!(!out.contains("service=\"other\""));
        assert!(out.contains("fa_process_uptime_seconds "));
        if std::path::Path::new("/proc/self/statm").exists() {
            assert!(out.contains("fa_process_resident_memory_bytes "));
        }
    }
}
