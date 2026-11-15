//! The bounded background Postgres writer (spec 41 §9, D-105). The decision loop publishes its
//! outputs to NATS first and hands every Postgres write to this task through a bounded queue,
//! so a slow or stalled Postgres can no longer hold a tick past its grid slot (the 2026-11-14
//! full-day replay failure: 1.3–3.4 s `p.intent` awaits inside the loop). The queue bound is
//! the backpressure: when it is full the enqueue awaits, which is the deliberate stop — better
//! a late tick than unbounded memory. Order is preserved (one FIFO queue, one consumer), and
//! the ledger tolerates late writes by construction: it is event-sourced and replay-ordered by
//! `(time, id)`, fills stay "unconfirmed" in the engine until the write lands (retried by the
//! 1 s `retry_unconfirmed` task after 5 s), and reconciliation excludes writes younger than 60 s
//! (`recon::IN_FLIGHT_GRACE_MS`), so a writer backlog does not raise spurious diffs.

use std::sync::{
    Arc,
    atomic::{AtomicI64, Ordering},
};

use fa_bus::Bus;
use fa_ledger::Ledger;
use fa_proto::v1::{OrderEvent, OrderIntent};
use serde_json::Value;

use crate::{Shared, engine::BookedFill, persist::Persist, recon};

/// Queue capacity (`FA_PERSIST_QUEUE`); at ~10 writes per decision tick, 1024 is ~100 s of
/// Postgres outage before backpressure reaches the loop.
fn queue_cap() -> usize {
    std::env::var("FA_PERSIST_QUEUE").ok().and_then(|v| v.parse().ok()).filter(|n| *n > 0).unwrap_or(1024)
}

pub enum WriteJob {
    Intent(OrderIntent),
    Audit {
        action: String,
        subject: Option<String>,
        payload: Value,
    },
    /// Book snapshot; the ledger is cloned at enqueue so it is the state at the triggering
    /// event, not at write time.
    Snapshot {
        ledger: Box<Ledger>,
        label: &'static str,
    },
    /// Booked fills → `recon::persist_fills` (confirm / orphan handling as before).
    Fills(Vec<BookedFill>),
    /// Fill events for unknown intents → `recon::resolve_unknown`.
    Unknown(Vec<OrderEvent>),
}

#[derive(Clone)]
pub struct Writer {
    tx: tokio::sync::mpsc::Sender<WriteJob>,
    depth: Arc<AtomicI64>,
}

impl Writer {
    /// Spawns the consumer. Outs the recovery paths raise (orphan/broker halts) go back through
    /// the direct publish path: the writer is off the decision loop, so awaiting there is fine.
    pub fn spawn(bus: Bus, p: Arc<Persist>, engine: Shared) -> Writer {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<WriteJob>(queue_cap());
        let depth = Arc::new(AtomicI64::new(0));
        let d = depth.clone();
        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                d.fetch_sub(1, Ordering::Relaxed);
                let outs = match job {
                    WriteJob::Intent(i) => {
                        p.intent(&i).await;
                        Vec::new()
                    }
                    WriteJob::Audit { action, subject, payload } => {
                        p.audit(&action, subject.as_deref(), payload).await;
                        Vec::new()
                    }
                    WriteJob::Snapshot { ledger, label } => {
                        p.snapshot(&ledger, label).await;
                        Vec::new()
                    }
                    WriteJob::Fills(fills) => recon::persist_fills(&p, &engine, fills).await,
                    WriteJob::Unknown(events) => recon::resolve_unknown(&p, &engine, events).await,
                };
                if !outs.is_empty() {
                    crate::publish_outs_direct(&bus, &Some(p.clone()), &engine, outs).await;
                }
            }
        });
        Writer { tx, depth }
    }

    /// Enqueues; awaits only when the queue is full (bounded backpressure, see module docs).
    pub async fn send(&self, job: WriteJob) {
        self.depth.fetch_add(1, Ordering::Relaxed);
        if self.tx.send(job).await.is_err() {
            self.depth.fetch_sub(1, Ordering::Relaxed);
            tracing::error!("persist writer task is gone; write dropped");
        }
    }

    /// Jobs enqueued and not yet written (heartbeat gauge `persist.queue_depth`).
    pub fn depth(&self) -> i64 {
        self.depth.load(Ordering::Relaxed)
    }
}
