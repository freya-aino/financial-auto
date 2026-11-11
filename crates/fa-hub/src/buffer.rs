//! Bounded write buffer for "Helix unavailable → buffers writes" (spec 20 §8).
//!
//! [`WriteBuffer`] queues prepared write [`Batch`]es (e.g. [`crate::vectors::add_latents_batch`])
//! and sends them in order on [`WriteBuffer::flush`]. It is bounded: when full, the **oldest**
//! batch is dropped and counted ([`BufferStats::dropped`]), so a long outage costs the oldest
//! writes, never memory. A flush stops at the first transport failure, 5xx or retryable error
//! and keeps the rest queued; a batch the server rejects outright (4xx, not retryable) is
//! discarded and counted as `rejected` so one bad write cannot block the queue.
//!
//! Only queue **idempotent** batches (get-or-create, upserts, `set_property`) or accept
//! duplicates: a request that timed out after the server committed is sent again.
//! The buffer is a plain `&mut self` structure; share it behind a `tokio::sync::Mutex` or own it
//! in one writer task.

use std::collections::VecDeque;

use crate::ast::Batch;
use crate::{HubClient, HubError, Result};

/// Counters since the buffer was created.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BufferStats {
    pub queued: u64,
    pub sent: u64,
    /// Dropped because the buffer was full (oldest first).
    pub dropped: u64,
    /// Discarded because the server rejected them (non-retryable 4xx).
    pub rejected: u64,
}

/// Result of one [`WriteBuffer::flush`].
#[derive(Debug, Default)]
pub struct FlushReport {
    pub sent: usize,
    pub rejected: usize,
    /// Still queued (Helix unavailable or erroring).
    pub remaining: usize,
    /// The error that stopped the flush, if any.
    pub stopped_by: Option<HubError>,
}

#[derive(Debug)]
pub struct WriteBuffer {
    client: HubClient,
    queue: VecDeque<Batch>,
    capacity: usize,
    stats: BufferStats,
}

/// Errors after which a flush keeps the batch and stops (try again later).
fn is_transient(e: &HubError) -> bool {
    match e {
        HubError::Http(_) => true,
        HubError::Server { status, retryable, .. } => *status >= 500 || *retryable == Some(true) || *status == 429,
        _ => false,
    }
}

impl WriteBuffer {
    /// A buffer holding at most `capacity` batches (at least 1).
    pub fn new(client: HubClient, capacity: usize) -> Self {
        Self { client, queue: VecDeque::new(), capacity: capacity.max(1), stats: BufferStats::default() }
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn stats(&self) -> BufferStats {
        self.stats
    }

    /// Queue a batch; returns `true` when an older batch had to be dropped to make room.
    pub fn push(&mut self, batch: Batch) -> bool {
        let mut dropped = false;
        while self.queue.len() >= self.capacity {
            self.queue.pop_front();
            self.stats.dropped += 1;
            dropped = true;
        }
        self.queue.push_back(batch);
        self.stats.queued += 1;
        dropped
    }

    /// Send queued batches in order (see the module docs for what stops a flush).
    pub async fn flush(&mut self) -> FlushReport {
        let mut report = FlushReport::default();
        while let Some(batch) = self.queue.front() {
            match self.client.batch(batch).await {
                Ok(_) => {
                    self.queue.pop_front();
                    self.stats.sent += 1;
                    report.sent += 1;
                }
                Err(e) if is_transient(&e) => {
                    report.stopped_by = Some(e);
                    break;
                }
                Err(e) => {
                    self.queue.pop_front();
                    self.stats.rejected += 1;
                    report.rejected += 1;
                    report.stopped_by = Some(e);
                }
            }
        }
        report.remaining = self.queue.len();
        report
    }

    /// Try the batch right away when the queue is empty; on a transient failure (or when older
    /// batches are still waiting, to keep order) queue it instead. Returns `Ok(true)` when it was
    /// sent now, `Ok(false)` when queued; a non-transient rejection is returned as the error.
    pub async fn send_or_queue(&mut self, batch: Batch) -> Result<bool> {
        if !self.queue.is_empty() {
            let r = self.flush().await;
            if r.remaining > 0 {
                self.push(batch);
                return Ok(false);
            }
        }
        match self.client.batch(&batch).await {
            Ok(_) => {
                self.stats.sent += 1;
                Ok(true)
            }
            Err(e) if is_transient(&e) => {
                self.push(batch);
                Ok(false)
            }
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Op, Pred};

    fn b(i: i64) -> Batch {
        Batch::write().named(format!("b{i}")).ret("n", Op::nodes_with_label_where("X", Pred::eq("i", i)).count())
    }

    #[test]
    fn bounded_drop_oldest() {
        let mut w = WriteBuffer::new(HubClient::new("http://127.0.0.1:1"), 2);
        assert!(!w.push(b(1)));
        assert!(!w.push(b(2)));
        assert!(w.push(b(3)));
        assert_eq!(w.len(), 2);
        assert_eq!(w.stats(), BufferStats { queued: 3, sent: 0, dropped: 1, rejected: 0 });
        assert_eq!(w.queue.front().map(|x| x.to_json()["query_name"].clone()), Some("b2".into()));
    }

    #[tokio::test]
    async fn unreachable_server_keeps_everything_queued() {
        // Port 1 never listens: connection refused is transient.
        let mut w = WriteBuffer::new(HubClient::new("http://127.0.0.1:1"), 8);
        assert!(!w.send_or_queue(b(1)).await.expect("queued, not an error"));
        w.push(b(2));
        let r = w.flush().await;
        assert_eq!((r.sent, r.rejected, r.remaining), (0, 0, 2));
        assert!(r.stopped_by.as_ref().is_some_and(HubError::is_unreachable));
        assert!(!w.is_empty());
    }

    #[test]
    fn transient_classification() {
        let e = |status, retryable| HubError::Server { status, code: "x".into(), msg: String::new(), retryable };
        assert!(is_transient(&e(503, None)));
        assert!(is_transient(&e(409, Some(true))));
        assert!(!is_transient(&e(409, Some(false))));
        assert!(!is_transient(&e(400, None)));
        assert!(!is_transient(&HubError::BadParam("x".into())));
    }
}
