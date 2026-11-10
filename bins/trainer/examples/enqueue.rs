//! Enqueue one training job on the `JOBS_TRAIN` work queue (what the orchestrator will do):
//!
//! ```sh
//! cargo run -p trainer --example enqueue -- adapter '{"timescale":"1m","window_hours":1}'
//! cargo run -p trainer --example enqueue -- pretrain '{"timescale":"1m","window_hours":6}' 60000
//! ```
//! Arguments: kind (`pretrain | generation | adapter`), spec JSON (default `{}`), budget ms
//! (default 0 = unlimited). Prints the job id; follow it with `GET /api/jobs?kind=train.<kind>`.
//! `enqueue preempt <job_id>` instead publishes a `ctl.gpu.preempt` notice for that job's lease
//! (what the arbiter does when a higher class needs the GPU), to exercise checkpoint + resume.

use fa_core::config::PlatformConfig;
use fa_proto::v1::Job;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut argv = std::env::args().skip(1);
    let kind = argv.next().unwrap_or_else(|| "adapter".into());
    let spec = argv.next().unwrap_or_else(|| "{}".into());
    let budget: u64 = argv.next().and_then(|b| b.parse().ok()).unwrap_or(0);
    if kind != "preempt" {
        serde_json::from_str::<serde_json::Value>(&spec)?;
    }
    let cfg = PlatformConfig::load(std::env::var("FA_CONFIG").unwrap_or_else(|_| "config/platform.toml".into()))?;
    let bus = fa_bus::Bus::connect(&cfg.nats.url).await?;
    if kind == "preempt" {
        let req = fa_proto::v1::CtlRequest {
            verb: "preempt".into(),
            args: [("lease_id".to_owned(), spec.clone())].into_iter().collect(),
        };
        bus.publish(fa_bus::subjects::CTL_GPU_PREEMPT.into(), &req).await?;
        bus.client().flush().await?;
        println!("preempt sent for lease {spec}");
        return Ok(());
    }
    // The queue is owned by `trainer serve`; creating it here only matters when no worker ran yet.
    fa_bus::streams::ensure_work_queue(&bus.jetstream(), "train", "trainer").await?;
    let job = Job {
        job_id: ulid::Ulid::generate().to_string(),
        kind: format!("train.{kind}"),
        priority: if kind == "adapter" { 1 } else { 3 },
        spec: spec.into_bytes(),
        budget_gpu_ms: budget,
        ..Default::default()
    };
    bus.publish(fa_bus::subjects::jobs(&job.kind), &job).await?;
    bus.client().flush().await?;
    println!("{}", job.job_id);
    Ok(())
}
