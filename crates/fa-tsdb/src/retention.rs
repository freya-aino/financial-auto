//! Per-timescale retention for VictoriaMetrics (spec 20 §6.5) and automatic tightening under a
//! disk budget (spec 20 §8).
//!
//! Single-node VictoriaMetrics (open source) has one global `-retentionPeriod`; per-series or
//! per-sample retention (`-retentionFilter`) is an enterprise feature, and the open-source
//! `delete_series` API removes a whole series, never a time range. The job therefore **rewrites**
//! each series that holds samples older than its class horizon: export the samples at or after
//! the cutoff (`/api/v1/export`), spool them to a local file, delete the series, import the spool
//! back (`/api/v1/import`), remove the spool. A spool left behind by a failed import is imported
//! first on the next run ([`replay_spool`]). Samples the ingestor writes to a series in the
//! seconds between its export and its delete are lost from VictoriaMetrics (they remain in the
//! Garage raw archive, from which VM is rebuildable); schedule the job off-hours.
//!
//! Classes (§6.5): book top-N 30 d, seconds-scale series (trades, quotes, `tf=<n>s` bars) 90 d,
//! minute bars 2 y; hour/day/week bars, features and ops metrics are not touched (VM's global
//! 100 y). Under a disk budget, [`tighten`] shortens the horizons in priority order (book first,
//! then seconds-scale, then minute bars), one halving per class per run, never below a class floor.
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Duration,
};

use crate::{ALLOWED_LABELS, Tsdb, TsdbError, check, escape_label};

const DAY: u64 = 86_400;

/// One retention class: series selectors, horizon, tightening priority and floor.
#[derive(Debug, Clone, PartialEq)]
pub struct RetentionClass {
    pub name: String,
    /// PromQL series selectors (`match[]`), OR-ed.
    pub selectors: Vec<String>,
    /// Samples older than `now − horizon` are removed.
    pub horizon: Duration,
    /// Lower = tightened first under disk pressure.
    pub priority: u8,
    /// [`tighten`] never goes below this.
    pub floor: Duration,
}

/// The classes of spec 20 §6.5, in tightening order.
#[derive(Debug, Clone, PartialEq)]
pub struct RetentionPolicy {
    pub classes: Vec<RetentionClass>,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        let class = |name: &str, selectors: &[&str], days: u64, priority: u8, floor_days: u64| RetentionClass {
            name: name.into(),
            selectors: selectors.iter().map(|s| (*s).to_owned()).collect(),
            horizon: Duration::from_secs(days * DAY),
            priority,
            floor: Duration::from_secs(floor_days * DAY),
        };
        Self {
            classes: vec![
                class("book", &[r#"{__name__=~"fa_book_.+"}"#], 30, 0, 3),
                class(
                    "seconds",
                    &[r#"{__name__=~"fa_(trade|quote)_.+"}"#, r#"{__name__=~"fa_bar_.+",tf=~"[0-9]+s"}"#],
                    90,
                    1,
                    14,
                ),
                class("minute_bars", &[r#"{__name__=~"fa_bar_.+",tf=~"[0-9]+m"}"#], 730, 2, 90),
            ],
        }
    }
}

impl RetentionPolicy {
    /// Overrides horizons from `name=<n>d,...` (e.g. `book=30d,seconds=90d,minute_bars=730d`);
    /// `h` and `d` suffixes are accepted. Unknown names are an error.
    pub fn with_overrides(mut self, spec: &str) -> Result<Self, String> {
        for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (name, v) = part.split_once('=').ok_or_else(|| format!("expected name=<n>d: {part}"))?;
            let d = parse_duration(v.trim()).ok_or_else(|| format!("bad duration {v}"))?;
            let c = self
                .classes
                .iter_mut()
                .find(|c| c.name == name.trim())
                .ok_or_else(|| format!("unknown retention class {name}"))?;
            c.horizon = d;
        }
        Ok(self)
    }
}

fn parse_duration(v: &str) -> Option<Duration> {
    let (n, unit) = v.split_at(v.len().checked_sub(1)?);
    let n: u64 = n.parse().ok()?;
    match unit {
        "d" => Some(Duration::from_secs(n * DAY)),
        "h" => Some(Duration::from_secs(n * 3600)),
        _ => None,
    }
}

/// Spec 20 §8 "retention tightened automatically per priority table": when `used_bytes`
/// exceeds `budget_bytes`, halves the horizon of the lowest-priority class that is still above
/// its floor (clamped to the floor) and returns the new policy with a description of the step.
/// One step per call: the job runs again (daily) and keeps tightening while the store stays
/// over budget. At or under budget, or with every class at its floor, the policy is unchanged.
pub fn tighten(policy: &RetentionPolicy, used_bytes: u64, budget_bytes: u64) -> (RetentionPolicy, Option<String>) {
    let mut out = policy.clone();
    if budget_bytes == 0 || used_bytes <= budget_bytes {
        return (out, None);
    }
    let mut order: Vec<usize> = (0..out.classes.len()).collect();
    order.sort_by_key(|&i| out.classes[i].priority);
    for i in order {
        let c = &mut out.classes[i];
        if c.horizon > c.floor {
            let before = c.horizon;
            c.horizon = (c.horizon / 2).max(c.floor);
            let msg = format!(
                "{}: {}d -> {}d (VictoriaMetrics uses {} MiB > budget {} MiB)",
                c.name,
                before.as_secs() / DAY,
                c.horizon.as_secs() / DAY,
                used_bytes >> 20,
                budget_bytes >> 20
            );
            return (out, Some(msg));
        }
    }
    (out, Some(format!("every class at its floor; VictoriaMetrics still over budget ({} MiB)", used_bytes >> 20)))
}

/// Exact selector of one series: every label equal, every other whitelisted label absent
/// (`k=""`), so a series with an extra `tf`/`lvl` label is not matched by accident.
pub fn exact_selector(labels: &BTreeMap<String, String>) -> String {
    let mut parts: Vec<String> = labels.iter().map(|(k, v)| format!("{k}=\"{}\"", escape_label(v))).collect();
    for k in ALLOWED_LABELS {
        if !labels.contains_key(*k) {
            parts.push(format!("{k}=\"\""));
        }
    }
    format!("{{{}}}", parts.join(","))
}

/// What one class run did (or, dry run, would do).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct ClassReport {
    pub class: String,
    pub horizon_days: u64,
    pub cutoff_ms: i64,
    /// Series holding samples before the cutoff.
    pub series: usize,
    /// Samples before the cutoff (removed, or to be removed).
    pub samples_dropped: u64,
    /// Samples at or after the cutoff re-imported.
    pub samples_kept: u64,
    /// Series rewritten (0 in a dry run).
    pub rewritten: usize,
    pub errors: Vec<String>,
}

/// Options of [`enforce`].
#[derive(Debug, Clone)]
pub struct EnforceOptions {
    pub dry_run: bool,
    /// Where exports wait between delete and import.
    pub spool_dir: PathBuf,
    /// Series rewritten per class per run (bounds one run's work; the rest waits for the next).
    pub max_series: usize,
}

impl Default for EnforceOptions {
    fn default() -> Self {
        Self { dry_run: true, spool_dir: std::env::temp_dir().join("fa-vm-retention"), max_series: 10_000 }
    }
}

impl Tsdb {
    /// Series (label sets) matching `selectors` that have samples in `[start_s, end_s]`
    /// (`/api/v1/series`; VictoriaMetrics resolves the range at day granularity and treats
    /// `start_s = 0` as unset, i.e. the last day — pass 1 for "since the epoch").
    pub async fn series(
        &self,
        selectors: &[String],
        start_s: i64,
        end_s: i64,
    ) -> Result<Vec<BTreeMap<String, String>>, TsdbError> {
        #[derive(serde::Deserialize)]
        struct Resp {
            data: Vec<BTreeMap<String, String>>,
        }
        let mut q: Vec<(&str, String)> = selectors.iter().map(|s| ("match[]", s.clone())).collect();
        q.push(("start", start_s.to_string()));
        q.push(("end", end_s.to_string()));
        let resp = self.http.get(format!("{}/api/v1/series", self.base)).query(&q).send().await?;
        Ok(check(resp).await?.json::<Resp>().await?.data)
    }

    /// Raw JSON-line export (`/api/v1/export`) of `selector` over `[start_ms, end_ms]`.
    pub async fn export(&self, selector: &str, start_ms: i64, end_ms: i64) -> Result<Vec<u8>, TsdbError> {
        let resp = self
            .http
            .get(format!("{}/api/v1/export", self.base))
            .query(&[
                ("match[]", selector.to_owned()),
                ("start", ms_param(start_ms)),
                ("end", ms_param(end_ms)),
            ])
            .send()
            .await?;
        Ok(check(resp).await?.bytes().await?.to_vec())
    }

    /// Imports VictoriaMetrics JSON lines (`/api/v1/import`), as produced by [`Tsdb::export`].
    pub async fn import_jsonl(&self, body: Vec<u8>) -> Result<(), TsdbError> {
        let resp = self.http.post(format!("{}/api/v1/import", self.base)).body(body).send().await?;
        check(resp).await.map(|_| ())
    }

    /// Deletes every series matching `selector` (`/api/v1/admin/tsdb/delete_series`).
    pub async fn delete_series(&self, selector: &str) -> Result<(), TsdbError> {
        let resp = self
            .http
            .post(format!("{}/api/v1/admin/tsdb/delete_series", self.base))
            .query(&[("match[]", selector)])
            .send()
            .await?;
        check(resp).await.map(|_| ())
    }

    /// Reads one gauge from VictoriaMetrics' own `/metrics` (summing every line of `name` whose
    /// labels contain `label_filter`, e.g. `type="storage/hour_metric_ids"`).
    pub async fn vm_metric(&self, name: &str, label_filter: &str) -> Result<f64, TsdbError> {
        let resp = self.http.get(format!("{}/metrics", self.base)).send().await?;
        let text = check(resp).await?.text().await?;
        Ok(sum_metric(&text, name, label_filter))
    }

    /// Series that received samples within the current hour — VictoriaMetrics' active-series
    /// count, the quantity `vm.max_active_series` bounds (spec 20 §5.3).
    pub async fn active_series(&self) -> Result<u64, TsdbError> {
        Ok(self.vm_metric("vm_cache_entries", r#"type="storage/hour_metric_ids""#).await? as u64)
    }

    /// Bytes VictoriaMetrics' data and index take on disk (`vm_data_size_bytes`, all types).
    pub async fn data_size_bytes(&self) -> Result<u64, TsdbError> {
        Ok(self.vm_metric("vm_data_size_bytes", "").await? as u64)
    }
}

fn ms_param(ms: i64) -> String {
    // VictoriaMetrics accepts fractional unix seconds
    format!("{}.{:03}", ms.div_euclid(1000), ms.rem_euclid(1000))
}

fn sum_metric(text: &str, name: &str, label_filter: &str) -> f64 {
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let (series, value) = l.rsplit_once(' ')?;
            let (n, labels) = series.split_once('{').map_or((series, ""), |(n, rest)| (n, rest));
            (n == name && labels.contains(label_filter)).then(|| value.parse::<f64>().ok())?
        })
        .sum()
}

/// Samples in a JSON-line export (sum of each line's `values` length).
pub fn count_samples(jsonl: &[u8]) -> u64 {
    #[derive(serde::Deserialize)]
    struct Line {
        values: Vec<serde_json::Value>,
    }
    jsonl
        .split(|b| *b == b'\n')
        .filter(|l| !l.is_empty())
        .filter_map(|l| serde_json::from_slice::<Line>(l).ok())
        .map(|l| l.values.len() as u64)
        .sum()
}

/// Imports and removes every spool file left by an earlier, interrupted run; returns how many
/// were restored. A file whose import fails stays for the next run.
pub async fn replay_spool(tsdb: &Tsdb, dir: &Path) -> Result<usize, TsdbError> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Ok(0) };
    let mut n = 0;
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "jsonl")
            && let Ok(body) = std::fs::read(&p)
        {
            tsdb.import_jsonl(body).await?;
            let _ = std::fs::remove_file(&p);
            n += 1;
        }
    }
    Ok(n)
}

/// Enforces `policy` at `now_ms`: for every class, every series with samples before
/// `now − horizon` is rewritten without them (module docs). A dry run only counts.
pub async fn enforce(
    tsdb: &Tsdb,
    policy: &RetentionPolicy,
    now_ms: i64,
    opts: &EnforceOptions,
) -> Result<Vec<ClassReport>, TsdbError> {
    if !opts.dry_run {
        std::fs::create_dir_all(&opts.spool_dir).map_err(|e| TsdbError::Io(e.to_string()))?;
        replay_spool(tsdb, &opts.spool_dir).await?;
    }
    let mut reports = Vec::new();
    for class in &policy.classes {
        let cutoff_ms = now_ms - class.horizon.as_millis() as i64;
        let mut rep = ClassReport {
            class: class.name.clone(),
            horizon_days: class.horizon.as_secs() / DAY,
            cutoff_ms,
            ..Default::default()
        };
        if cutoff_ms <= 0 {
            reports.push(rep);
            continue;
        }
        // start=1: VictoriaMetrics treats start=0 as unset (= the last day only)
        let series = tsdb.series(&class.selectors, 1, cutoff_ms / 1000).await?;
        for (i, labels) in series.iter().enumerate().take(opts.max_series) {
            let sel = exact_selector(labels);
            let old = tsdb.export(&sel, 0, cutoff_ms - 1).await?;
            let dropped = count_samples(&old);
            if dropped == 0 {
                continue; // day-granular series index: nothing actually before the cutoff
            }
            rep.series += 1;
            rep.samples_dropped += dropped;
            if opts.dry_run {
                continue;
            }
            match rewrite(tsdb, &sel, cutoff_ms, &opts.spool_dir.join(format!("{}-{i}.jsonl", class.name))).await {
                Ok(kept) => {
                    rep.samples_kept += kept;
                    rep.rewritten += 1;
                }
                Err(e) => rep.errors.push(format!("{sel}: {e}")),
            }
        }
        reports.push(rep);
    }
    Ok(reports)
}

/// Export the kept samples, spool them, delete the series, import, drop the spool.
async fn rewrite(tsdb: &Tsdb, sel: &str, cutoff_ms: i64, spool: &Path) -> Result<u64, TsdbError> {
    let far_future = 4_102_444_800_000; // 2100-01-01
    let kept = tsdb.export(sel, cutoff_ms, far_future).await?;
    let n = count_samples(&kept);
    std::fs::write(spool, &kept).map_err(|e| TsdbError::Io(e.to_string()))?;
    tsdb.delete_series(sel).await?;
    if n > 0 {
        tsdb.import_jsonl(kept).await?; // on failure the spool stays for replay_spool
    }
    let _ = std::fs::remove_file(spool);
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: Duration = Duration::from_secs(DAY);

    #[test]
    fn default_policy_is_spec_20_table() {
        let p = RetentionPolicy::default();
        let h: Vec<(&str, u64)> = p.classes.iter().map(|c| (c.name.as_str(), c.horizon.as_secs() / DAY)).collect();
        assert_eq!(h, [("book", 30), ("seconds", 90), ("minute_bars", 730)]);
        let p = p.with_overrides("book=7d, minute_bars=365d").unwrap();
        assert_eq!(p.classes[0].horizon, 7 * D);
        assert_eq!(p.classes[2].horizon, 365 * D);
        assert!(RetentionPolicy::default().with_overrides("hourly=1d").is_err());
        assert!(RetentionPolicy::default().with_overrides("book=7w").is_err());
    }

    #[test]
    fn tightening_follows_priority_and_floors() {
        let p = RetentionPolicy::default();
        assert_eq!(tighten(&p, 10, 100).1, None, "under budget: unchanged");
        assert_eq!(tighten(&p, 10, 0).1, None, "no budget: unchanged");
        let mut cur = p.clone();
        let mut steps = Vec::new();
        for _ in 0..20 {
            let (next, msg) = tighten(&cur, 200 << 20, 100 << 20);
            steps.push(msg.unwrap());
            cur = next;
        }
        // book halves to its 3 d floor first, then seconds-scale to 14 d, then minute bars to 90 d
        let days: Vec<u64> = cur.classes.iter().map(|c| c.horizon.as_secs() / DAY).collect();
        assert_eq!(days, [3, 14, 90]);
        assert!(steps[0].starts_with("book: 30d -> 15d"), "{}", steps[0]);
        assert!(steps.iter().position(|s| s.starts_with("seconds")) > steps.iter().rposition(|s| s.starts_with("book")));
        assert!(steps.last().unwrap().starts_with("every class at its floor"));
    }

    #[test]
    fn exact_selector_pins_absent_labels() {
        let labels = BTreeMap::from([
            ("__name__".to_owned(), "fa_trade_price".to_owned()),
            ("inst".to_owned(), "ab".to_owned()),
            ("venue".to_owned(), "k\"x".to_owned()),
        ]);
        let s = exact_selector(&labels);
        assert!(s.starts_with(r#"{__name__="fa_trade_price",inst="ab",venue="k\"x",tf="",src="",lvl="""#), "{s}");
        for k in ALLOWED_LABELS.iter().filter(|k| !["inst", "venue"].contains(k)) {
            assert!(s.contains(&format!(",{k}=\"\"")), "{k} not pinned: {s}");
        }
        assert!(!s.contains(r#"inst="""#) && s.ends_with("\"}"));
    }

    #[test]
    fn parses_metrics_and_counts_export() {
        let text = "# HELP x\nvm_data_size_bytes{type=\"storage/small\"} 100\nvm_data_size_bytes{type=\"indexdb/file\"} 20\nvm_cache_entries{type=\"storage/hour_metric_ids\"} 7045\nvm_cache_entries{type=\"other\"} 1\n";
        assert_eq!(sum_metric(text, "vm_data_size_bytes", ""), 120.0);
        assert_eq!(sum_metric(text, "vm_cache_entries", r#"type="storage/hour_metric_ids""#), 7045.0);
        let jsonl = b"{\"metric\":{\"__name__\":\"a\"},\"values\":[1,2,3],\"timestamps\":[1,2,3]}\n{\"metric\":{\"__name__\":\"a\"},\"values\":[4],\"timestamps\":[4]}\n";
        assert_eq!(count_samples(jsonl), 4);
        assert_eq!(ms_param(1_700_000_000_123), "1700000000.123");
        assert_eq!(ms_param(-1), "-1.999");
    }

    /// Against a SCRATCH VictoriaMetrics only (it deletes and rewrites series):
    /// `docker run -d --rm --name fa-vm-scratch -p 127.0.0.1:18428:8428 --tmpfs /storage
    ///  victoriametrics/victoria-metrics:v1.155.0 -storageDataPath=/storage -retentionPeriod=100y`
    /// then `FA_VM_SCRATCH_URL=http://127.0.0.1:18428 cargo test -p fa-tsdb scratch_retention`.
    #[tokio::test]
    async fn scratch_retention_rewrites_only_old_samples() {
        let Ok(url) = std::env::var("FA_VM_SCRATCH_URL") else {
            eprintln!("skipped: set FA_VM_SCRATCH_URL to a scratch VictoriaMetrics (never the live one)");
            return;
        };
        assert!(!url.contains(":8428"), "refusing to rewrite series on the live VictoriaMetrics port");
        let vm = Tsdb::new(url);
        vm.health().await.unwrap();
        // real time: VictoriaMetrics drops samples more than 2 days in the future
        let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64;
        let day = 86_400_000i64;
        let mut samples = Vec::new();
        let s = |m: &str, v: f64, ts: i64| {
            crate::Sample::new(m, v, ts).label("inst", "aa").label("venue", "scratch").label("src", "scratch")
        };
        for d in [200, 120, 95, 60, 10, 1] {
            samples.push(s("fa_trade_price", d as f64, now_ms - d * day));
            samples.push(s("fa_bar_close", d as f64, now_ms - d * day).label("tf", "1m"));
            samples.push(s("fa_bar_close", d as f64, now_ms - d * day).label("tf", "1h"));
            samples.push(s("fa_book_bid_px", d as f64, now_ms - d * day).label("lvl", "0"));
        }
        vm.write(&samples).await.unwrap();
        vm.force_flush().await.unwrap();
        let dir = std::env::temp_dir().join(format!("fa-vm-retention-test-{}", std::process::id()));
        let policy = RetentionPolicy::default();
        let dry = enforce(&vm, &policy, now_ms, &EnforceOptions { dry_run: true, spool_dir: dir.clone(), max_series: 100 })
            .await
            .unwrap();
        let by = |r: &[ClassReport], c: &str| r.iter().find(|x| x.class == c).unwrap().clone();
        assert_eq!((by(&dry, "seconds").series, by(&dry, "seconds").samples_dropped), (1, 3), "{dry:?}");
        assert_eq!(by(&dry, "book").samples_dropped, 4, "60/95/120/200 d > 30 d");
        assert_eq!(by(&dry, "minute_bars").samples_dropped, 0, "nothing older than 2 y");
        let real = enforce(&vm, &policy, now_ms, &EnforceOptions { dry_run: false, spool_dir: dir.clone(), max_series: 100 })
            .await
            .unwrap();
        assert_eq!(by(&real, "seconds").rewritten, 1);
        assert_eq!(by(&real, "seconds").samples_kept, 3);
        assert!(real.iter().all(|r| r.errors.is_empty()), "{real:?}");
        vm.force_flush().await.unwrap();
        let count = |sel: &str| {
            let vm = vm.clone();
            let sel = sel.to_owned();
            async move { count_samples(&vm.export(&sel, 0, now_ms + day).await.unwrap()) }
        };
        assert_eq!(count(r#"{__name__="fa_trade_price",venue="scratch"}"#).await, 3, "60/10/1 d kept");
        assert_eq!(count(r#"{__name__="fa_book_bid_px",venue="scratch"}"#).await, 2, "10/1 d kept");
        assert_eq!(count(r#"{__name__="fa_bar_close",venue="scratch",tf="1m"}"#).await, 6, "minute bars untouched");
        assert_eq!(count(r#"{__name__="fa_bar_close",venue="scratch",tf="1h"}"#).await, 6, "hour bars kept forever");
        // idempotent: a second run finds nothing
        let again = enforce(&vm, &policy, now_ms, &EnforceOptions { dry_run: true, spool_dir: dir.clone(), max_series: 100 })
            .await
            .unwrap();
        assert!(again.iter().all(|r| r.samples_dropped == 0), "{again:?}");
        assert!(std::fs::read_dir(&dir).map(|d| d.count() == 0).unwrap_or(true), "spool emptied");
        vm.delete_series(r#"{venue="scratch"}"#).await.unwrap();
        let _ = std::fs::remove_dir_all(dir);
    }
}
