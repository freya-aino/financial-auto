//! GPU lease client for the arbiter in `evo-orchestrator` (spec 60, spec 36 §6.7, D-006).
//! Same protocol as `bins/evo-worker/src/gpu.rs`, with the class chosen per job kind:
//! `generation_training` (P3) for `pretrain`/`generation`, `online_adapter` (P1) for `adapter`.
//! No GPU exists on the baseline machine (D-004): the lease makes the trainer's activity
//! visible to the arbiter and exercises preemption; `--gpu-lease-mb 0` disables it.

use std::time::{Duration, Instant};

use fa_bus::{Bus, subjects};
use fa_proto::{
    Message,
    v1::{CtlReply, CtlRequest},
};
use futures::StreamExt;
use tokio::sync::broadcast;

pub const GENERATION_TRAINING: &str = "generation_training";
pub const ONLINE_ADAPTER: &str = "online_adapter";
const CTL_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Notice {
    Granted(String),
    Preempt(String),
}

/// Fans `ctl.gpu.granted` / `ctl.gpu.preempt` out to job handlers.
pub async fn run_notices(bus: Bus, tx: broadcast::Sender<Notice>) {
    let granted = bus.subscribe::<CtlRequest>(subjects::ctl("gpu", "granted")).await;
    let preempt = bus.subscribe::<CtlRequest>(subjects::ctl("gpu", "preempt")).await;
    let (Ok(granted), Ok(preempt)) = (granted, preempt) else {
        tracing::error!("cannot subscribe to ctl.gpu notifications");
        return;
    };
    let mut merged = futures::stream::select(granted, preempt);
    while let Some((subject, req)) = merged.next().await {
        let lease = req.args.get("lease_id").cloned().unwrap_or_default();
        let n = if subject.ends_with(".preempt") { Notice::Preempt(lease) } else { Notice::Granted(lease) };
        let _ = tx.send(n);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lease {
    Granted,
    /// Arbiter unreachable or leasing disabled: proceed without a lease (CPU work).
    Unavailable,
}

async fn ctl(bus: &Bus, verb: &str, args: Vec<(&str, String)>) -> Result<CtlReply, String> {
    let req = CtlRequest { verb: verb.into(), args: args.into_iter().map(|(k, v)| (k.to_owned(), v)).collect() };
    let fut = bus.client().request(subjects::ctl("gpu", verb), req.encode_to_vec().into());
    match tokio::time::timeout(CTL_TIMEOUT, fut).await {
        Ok(Ok(m)) => CtlReply::decode(m.payload.as_ref()).map_err(|e| e.to_string()),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err(format!("no reply within {CTL_TIMEOUT:?}")),
    }
}

/// Acquires `mem_mb` in `class` for `lease_id`, waiting for a grant until `deadline`.
pub async fn acquire(
    bus: &Bus,
    notices: &mut broadcast::Receiver<Notice>,
    lease_id: &str,
    class: &str,
    mem_mb: u32,
    deadline: Instant,
) -> Result<Lease, String> {
    if mem_mb == 0 {
        return Ok(Lease::Unavailable);
    }
    let reply = match ctl(
        bus,
        "acquire",
        vec![("lease_id", lease_id.into()), ("class", class.into()), ("mem_mb", mem_mb.to_string())],
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(error = %e, "gpu arbiter unreachable; running without a lease");
            return Ok(Lease::Unavailable);
        }
    };
    if !reply.ok {
        return Err(format!("gpu lease refused: {}", reply.message));
    }
    if reply.message == "granted" {
        return Ok(Lease::Granted);
    }
    tracing::info!(%lease_id, class, "gpu lease queued; waiting for grant");
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            release(bus, lease_id).await;
            return Err("gpu lease not granted before the job deadline".into());
        }
        match tokio::time::timeout(remaining, notices.recv()).await {
            Ok(Ok(Notice::Granted(id))) if id == lease_id => return Ok(Lease::Granted),
            Ok(Ok(_)) => continue,
            Ok(Err(broadcast::error::RecvError::Lagged(_))) => continue,
            Ok(Err(broadcast::error::RecvError::Closed)) => return Ok(Lease::Unavailable),
            Err(_) => continue,
        }
    }
}

pub async fn release(bus: &Bus, lease_id: &str) {
    if let Err(e) = ctl(bus, "release", vec![("lease_id", lease_id.into())]).await {
        tracing::debug!(error = %e, %lease_id, "gpu release failed");
    }
}
