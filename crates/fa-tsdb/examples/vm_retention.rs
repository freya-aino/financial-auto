//! VictoriaMetrics per-timescale retention job (spec 20 §6.5, §8; `fa_tsdb::retention`).
//!
//! ```sh
//! cargo run -p fa-tsdb --example vm_retention                       # dry run against 127.0.0.1:8428
//! cargo run -p fa-tsdb --example vm_retention -- --url http://127.0.0.1:18428 --apply
//! cargo run -p fa-tsdb --example vm_retention -- --policy book=7d --disk-budget-gib 50
//! ```
//! Without `--apply` nothing is changed: it lists, per class, the series and samples older than
//! the horizon. `--disk-budget-gib` compares VictoriaMetrics' `vm_data_size_bytes` with the
//! budget and, when over, tightens the policy one priority step ([`fa_tsdb::retention::tighten`])
//! for this run. Output: one JSON object (`active_series`, `data_bytes`, `tightened`, `classes`).
use std::time::{SystemTime, UNIX_EPOCH};

use fa_tsdb::{
    Tsdb,
    retention::{EnforceOptions, RetentionPolicy, enforce, tighten},
};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut url = std::env::var("FA_VM_URL").unwrap_or_else(|_| "http://127.0.0.1:8428".into());
    let (mut apply, mut policy, mut budget_gib, mut spool) =
        (false, RetentionPolicy::default(), None::<f64>, EnforceOptions::default().spool_dir);
    let mut max_series = EnforceOptions::default().max_series;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--url" => url = args.next().ok_or("--url needs a value")?,
            "--apply" => apply = true,
            "--policy" => policy = policy.with_overrides(&args.next().ok_or("--policy needs a value")?)?,
            "--disk-budget-gib" => budget_gib = Some(args.next().ok_or("--disk-budget-gib needs a value")?.parse()?),
            "--spool" => spool = args.next().ok_or("--spool needs a value")?.into(),
            "--max-series" => max_series = args.next().ok_or("--max-series needs a value")?.parse()?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let vm = Tsdb::new(url);
    vm.health().await?;
    let data_bytes = vm.data_size_bytes().await?;
    let active_series = vm.active_series().await?;
    let mut tightened = None;
    if let Some(gib) = budget_gib {
        let (p, msg) = tighten(&policy, data_bytes, (gib * (1u64 << 30) as f64) as u64);
        policy = p;
        tightened = msg;
    }
    let now_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as i64;
    let opts = EnforceOptions { dry_run: !apply, spool_dir: spool, max_series };
    let classes = enforce(&vm, &policy, now_ms, &opts).await?;
    let out = serde_json::json!({
        "vm": vm.base_url(), "dry_run": !apply, "active_series": active_series, "data_bytes": data_bytes,
        "tightened": tightened, "classes": classes,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
