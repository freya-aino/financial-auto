//! evo-worker kill drill (spec 60 §9 "killing an evo-worker mid-job results in the job
//! completing on another worker from its checkpoint"). Driven by `scripts/worker-kill-drill.sh`,
//! which starts a throwaway NATS; nothing here touches the compose stack.
//!
//! 1. starts worker A (GPU lease on) and answers its `ctl.gpu.acquire` with `queued`, so A holds
//!    the job mid-run (progress published, message unacked, in-progress acks every 20 s);
//! 2. publishes a synthetic random-walk trade feed (`md.trade.sim.<inst>`) and one
//!    `evo.eval.decoder` job on `JOBS_EVO`;
//! 3. once A has reported progress, starts worker B (no GPU lease) and lets it buffer trades;
//! 4. SIGKILLs A; JetStream redelivers the job after the ack wait (60 s) and B finishes it;
//! 5. checks: the job was received once by A and once by B (worker logs), exactly one `JobResult`,
//!    status OK, from a worker other than A, nothing left pending or unacked. The JSON summary goes to stdout; exit 1 on failure.
//!
//! evo-worker evaluations are cheap and deterministic, so a redelivered job re-evaluates from the
//! start (`checkpoint_ref` is a progress marker, `job.rs`); the trainer's checkpoint/resume of
//! long fits is verified separately (D-058, D-070).
//!
//! cargo run -p control-api --example worker_kill_drill -- <nats-url> <evo-worker-binary> <platform.toml>
use std::{
    process::Stdio,
    time::{Duration, Instant},
};

use anyhow::{Context, bail};
use fa_bus::{async_nats, subjects};
use fa_proto::{
    Message, now,
    v1::{CtlReply, Envelope, Job, JobProgress, JobResult, Trade, envelope::Payload, job_result::Status},
};
use futures::StreamExt;
use serde_json::json;
use tokio::process::{Child, Command};

const STREAM: &str = "JOBS_EVO";
const CONSUMER: &str = "evo-worker";
const KIND: &str = "evo.eval.decoder";
const GENOME: &str = "forecast:ts=1s;h=2;lb=8";

fn worker(bin: &str, config: &str, lease_mb: u32, name: &str) -> anyhow::Result<Child> {
    let log = std::fs::File::create(format!("{}/{name}.log", std::env::var("FA_DRILL_DIR").unwrap_or(".".into())))?;
    Command::new(bin)
        .env("FA_CONFIG", config)
        .env("FA_EVO_GPU_LEASE_MB", lease_mb.to_string())
        .env("FA_EVO_NO_TSDB", "true")
        .env("FA_EVO_NO_REGISTRY", "true")
        .env("FA_METRICS_BIND", "off")
        .env("FA_SECRETS", "/nonexistent/secrets.toml")
        .env("FA_HELIX_URL", "http://127.0.0.1:9")
        .env("RUST_LOG", "info")
        .env_remove("FA_GARAGE_ACCESS_KEY_ID")
        .env_remove("FA_GARAGE_SECRET_ACCESS_KEY")
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("starting {bin}"))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let (Some(url), Some(bin), Some(config)) = (args.next(), args.next(), args.next()) else {
        bail!("usage: worker_kill_drill <nats-url> <evo-worker-binary> <platform.toml>");
    };
    // rootless Docker accepts on the published port before nats-server is up: retry for 30 s
    let mut tries = 0;
    let client = loop {
        match async_nats::connect(&url).await {
            Ok(c) => break c,
            Err(e) if tries < 60 => {
                tries += 1;
                tracing::debug!(error = %e, "nats not ready");
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            Err(e) => return Err(e.into()),
        }
    };
    let js = async_nats::jetstream::new(client.clone());

    // fake GPU arbiter: every lease stays queued, so worker A blocks mid-job
    let mut acquire = client.subscribe(subjects::ctl("gpu", "acquire")).await?;
    let mut release = client.subscribe(subjects::ctl("gpu", "release")).await?;
    let arb = client.clone();
    tokio::spawn(async move {
        loop {
            let (m, msg) = tokio::select! {
                Some(m) = acquire.next() => (m, "queued"),
                Some(m) = release.next() => (m, "released"),
                else => break,
            };
            if let Some(reply) = m.reply {
                let r = CtlReply { ok: true, message: msg.into() };
                let _ = arb.publish(reply, r.encode_to_vec().into()).await;
            }
        }
    });

    // synthetic trade feed (10 Hz random walk)
    let feed = client.clone();
    let inst = fa_core::InstrumentId::new();
    tokio::spawn(async move {
        let mut price = 100.0_f64;
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        loop {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            price *= 1.0 + ((seed % 2001) as f64 / 1000.0 - 1.0) * 0.001;
            let env = Envelope {
                event_id: fa_core::EventId::new().to_string(),
                source_id: "drill".into(),
                instrument_id: inst.to_string(),
                event_time: Some(now()),
                publish_time: Some(now()),
                ingest_time: Some(now()),
                schema_version: 1,
                payload: Some(Payload::Trade(Trade { price, size: 1.0, ..Default::default() })),
                ..Default::default()
            };
            let _ = feed.publish(subjects::md_trade("sim", &inst), env.encode_to_vec().into()).await;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    });

    let started = Instant::now();
    let mut a = worker(&bin, &config, 512, "worker-a")?;
    // wait for A's durable consumer
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Ok(s) = js.get_stream(STREAM).await
            && s.consumer_info(CONSUMER).await.is_ok()
        {
            break;
        }
        if Instant::now() > deadline {
            bail!("worker A did not create {STREAM}/{CONSUMER} within 60 s (see worker-a.log)");
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    let job_id = fa_core::JobId::new().to_string();
    let mut progress = client.subscribe(subjects::job_progress(&job_id)).await?;
    let mut results = client.subscribe(subjects::job_result(KIND)).await?;
    let job = Job {
        job_id: job_id.clone(),
        kind: KIND.into(),
        genome_id: GENOME.into(),
        spec: GENOME.as_bytes().to_vec(),
        deadline: Some(fa_proto::Timestamp { seconds: now().seconds + 900, nanos: 0 }),
        seed: 7,
        ..Default::default()
    };
    js.publish(subjects::jobs(KIND), job.encode_to_vec().into()).await?.await.context("job publish ack")?;

    let first = tokio::time::timeout(Duration::from_secs(60), progress.next())
        .await
        .context("worker A reported no progress within 60 s")?
        .context("progress subscription closed")?;
    let first = JobProgress::decode(first.payload.as_ref())?;
    let mid_job_at = started.elapsed().as_secs_f64();

    // B buffers trades while A holds the job, then A dies without acking
    let mut b = worker(&bin, &config, 0, "worker-b")?;
    tokio::time::sleep(Duration::from_secs(45)).await;
    let pending_before = js.get_stream(STREAM).await?.consumer_info(CONSUMER).await?.num_ack_pending;
    a.start_kill()?;
    let _ = a.wait().await;
    let killed_at = Instant::now();

    let result = tokio::time::timeout(Duration::from_secs(150), async {
        while let Some(m) = results.next().await {
            if let Ok(r) = JobResult::decode(m.payload.as_ref())
                && r.job_id == job_id
            {
                return Some(r);
            }
        }
        None
    })
    .await
    .context("no JobResult within 150 s of the kill")?
    .context("result subscription closed")?;
    let finished_after = killed_at.elapsed().as_secs_f64();
    // a second result for the same job would mean double work
    let dup = tokio::time::timeout(Duration::from_secs(3), async {
        while let Some(m) = results.next().await {
            if JobResult::decode(m.payload.as_ref()).is_ok_and(|r| r.job_id == job_id) {
                return true;
            }
        }
        false
    })
    .await
    .unwrap_or(false);
    let info = js.get_stream(STREAM).await?.consumer_info(CONSUMER).await?.clone();
    let _ = b.start_kill();

    let status = Status::try_from(result.status).unwrap_or(Status::Failed);
    // which worker did what, from their logs (`pulling jobs` names the worker id, `job received` the job)
    let dir = std::env::var("FA_DRILL_DIR").unwrap_or(".".into());
    let log = |name: &str| std::fs::read_to_string(format!("{dir}/{name}.log")).unwrap_or_default();
    let (log_a, log_b) = (log("worker-a"), log("worker-b"));
    let worker_a = log_a
        .split_whitespace()
        .find_map(|w| w.find("evo-worker-").map(|i| w[i..].chars().take(11 + 26).collect::<String>()))
        .unwrap_or_default();
    let received = |l: &str| l.lines().filter(|x| x.contains("job received") && x.contains(&job_id)).count();
    let (recv_a, recv_b) = (received(&log_a), received(&log_b));
    let ok = status == Status::Ok
        && !dup
        && pending_before == 1
        && recv_a == 1
        && recv_b == 1
        && !worker_a.is_empty()
        && result.worker_id != worker_a
        && info.num_ack_pending == 0
        && info.num_pending == 0;
    let summary = json!({
        "ok": ok,
        "job_id": job_id,
        "genome": GENOME,
        "mid_job": { "at_s": mid_job_at, "progress": first.message, "ack_pending_before_kill": pending_before },
        "result": {
            "status": format!("{status:?}"),
            "worker_id": result.worker_id,
            "error": result.error,
            "checkpoint_ref": result.checkpoint_ref,
            "seconds_after_kill": finished_after,
            "duplicate_result": dup,
        },
        "deliveries": { "worker_a": worker_a, "received_by_a": recv_a, "received_by_b": recv_b },
        "consumer": {
            "num_redelivered": info.num_redelivered,
            "num_ack_pending": info.num_ack_pending,
            "num_pending": info.num_pending,
        },
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    if !ok {
        bail!("drill failed (see worker-a.log / worker-b.log)");
    }
    Ok(())
}
