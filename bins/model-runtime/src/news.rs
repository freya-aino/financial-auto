//! The news embeddings the text expert reads (D-119).
//!
//! `llm-suite` embeds every news item once on arrival and stores the vector with the item's
//! knowledge time (D-117). This module keeps the recent ones in memory so the synchronous
//! encode path can take the window's news without an await, and — the point of doing it this
//! way — **live and replay read the same source**: the store. A replay of a past window calls
//! [`NewsCache::window`] after filling the cache from the same `news_docs_in_window` read the
//! poller uses, so it sees exactly the items whose knowledge time was inside that window.
//!
//! What this module deliberately does not do is subscribe to `news.item.>` and embed. The text
//! on the bus is not a vector, and a second embedder would double the spend and produce vectors
//! the store does not have.
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fa_hub::HubClient;

/// How far back the cache keeps items. The coarsest timescale encodes windows of days, but a text
/// expert that reads a week of headlines at a 1-week step would mostly read noise, and the memory
/// is linear in this: 6 hours of one item a minute at 1 536 floats is ~2 MB.
pub const RETAIN: Duration = Duration::from_secs(6 * 3600);

/// Most items held, whatever `RETAIN` says — the bound that keeps a news storm from growing the
/// process (1 024 × 1 536 f32 ≈ 6 MB).
pub const MAX_ITEMS: usize = 1024;

/// How often the poller asks the store for new items.
pub const POLL: Duration = Duration::from_secs(10);

/// One embedded item: its knowledge time and vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub ts_ms: i64,
    pub doc_id: String,
    pub vector: Vec<f32>,
}

/// Recent news embeddings, newest last.
#[derive(Debug, Default)]
pub struct NewsCache {
    items: VecDeque<Item>,
    /// Newest `ts_ms` seen, so a poll only asks for what it does not have.
    high_water_ms: i64,
    pub polls: u64,
    pub fetched: u64,
}

impl NewsCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts items, keeping the deque ordered by `ts_ms` and free of duplicates.
    ///
    /// Duplicates matter: a poll overlaps the previous one on purpose (an item can be stored with
    /// a knowledge time slightly behind the newest one already seen, because items arrive out of
    /// order), so the same vector is offered more than once and must be counted once.
    pub fn insert(&mut self, new: impl IntoIterator<Item = Item>) -> usize {
        let mut added = 0;
        for item in new {
            if item.vector.is_empty() || self.items.iter().any(|i| i.doc_id == item.doc_id) {
                continue;
            }
            self.high_water_ms = self.high_water_ms.max(item.ts_ms);
            let at = self.items.partition_point(|i| i.ts_ms <= item.ts_ms);
            self.items.insert(at, item);
            added += 1;
        }
        self.fetched += added as u64;
        while self.items.len() > MAX_ITEMS {
            self.items.pop_front();
        }
        added
    }

    /// Drops items older than `now_ms - RETAIN`.
    pub fn prune(&mut self, now_ms: i64) {
        let cutoff = now_ms - RETAIN.as_millis() as i64;
        while self.items.front().is_some_and(|i| i.ts_ms < cutoff) {
            self.items.pop_front();
        }
    }

    /// The vectors whose knowledge time is in `(from_ms, to_ms]`, **newest first** — the order the
    /// text expert's `MAX_ITEMS` cut keeps the newest news when a window holds more than it reads.
    pub fn window(&self, from_ms: i64, to_ms: i64) -> Vec<Vec<f32>> {
        self.items
            .iter()
            .rev()
            .filter(|i| i.ts_ms > from_ms && i.ts_ms <= to_ms)
            .map(|i| i.vector.clone())
            .collect()
    }

    /// Where the next poll should start: a little before the newest item seen, so items stored
    /// out of order are still picked up.
    pub fn poll_from_ms(&self, now_ms: i64) -> i64 {
        let overlap = 2 * POLL.as_millis() as i64;
        if self.high_water_ms == 0 { now_ms - RETAIN.as_millis() as i64 } else { self.high_water_ms - overlap }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn newest_ms(&self) -> Option<i64> {
        self.items.back().map(|i| i.ts_ms)
    }
}

/// Keeps a shared cache filled from the store.
///
/// Failures are logged and retried on the next tick rather than propagated: with no news the text
/// expert reports "no news" through a wide posterior, which is a correct latent, and stalling the
/// model path because an embedding store is briefly unreachable would be the wrong trade.
pub fn spawn_poller(hub: HubClient, dim: usize, cache: Arc<Mutex<NewsCache>>) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(POLL);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            let now_ms = now_ms();
            let from = {
                let mut c = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                c.prune(now_ms);
                c.polls += 1;
                c.poll_from_ms(now_ms)
            };
            match hub.news_docs_in_window(dim, from, now_ms, MAX_ITEMS).await {
                Ok(rows) => {
                    let items: Vec<Item> = rows
                        .into_iter()
                        .map(|(hit, vector)| Item {
                            ts_ms: hit.ts.as_deref().and_then(parse_ms).unwrap_or(now_ms),
                            doc_id: hit.id,
                            vector,
                        })
                        .collect();
                    let n = items.len();
                    let added = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(items);
                    if added > 0 {
                        tracing::debug!(added, seen = n, dim, "news embeddings cached for the text expert");
                    }
                }
                Err(e) => tracing::warn!(error = %e, dim, "reading news embeddings failed; text expert sees none"),
            }
        }
    });
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

/// RFC 3339 → unix ms (the store renders `ts` as a timestamp string).
fn parse_ms(s: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(s).ok().map(|t| t.timestamp_millis())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(ts_ms: i64, id: &str) -> Item {
        Item { ts_ms, doc_id: id.into(), vector: vec![1.0, 0.0, 0.0, 0.0] }
    }

    #[test]
    fn items_are_ordered_deduped_and_capped() {
        let mut c = NewsCache::new();
        assert_eq!(c.insert([item(300, "c"), item(100, "a"), item(200, "b")]), 3);
        assert_eq!(c.insert([item(100, "a")]), 0, "the same doc is cached once");
        assert_eq!(c.newest_ms(), Some(300));
        assert_eq!(c.window(0, 400).len(), 3);
        assert_eq!(c.fetched, 3);

        for i in 0..MAX_ITEMS as i64 {
            c.insert([item(1000 + i, &format!("x{i}"))]);
        }
        assert_eq!(c.len(), MAX_ITEMS, "bounded");
        assert!(c.window(0, 400).is_empty(), "the oldest went first");
    }

    #[test]
    fn a_window_is_half_open_and_newest_first() {
        let mut c = NewsCache::new();
        c.insert([item(100, "a"), item(200, "b"), item(300, "c")]);
        // (100, 300]: 100 excluded, 300 included.
        let w = c.window(100, 300);
        assert_eq!(w.len(), 2);
        c.insert([Item { ts_ms: 250, doc_id: "d".into(), vector: vec![0.5, 0.5, 0.0, 0.0] }]);
        let w = c.window(100, 300);
        assert_eq!(w.len(), 3);
        assert_eq!(w[0], vec![1.0, 0.0, 0.0, 0.0], "newest first: c at 300");
        assert_eq!(w[1], vec![0.5, 0.5, 0.0, 0.0], "then d at 250");
        assert!(c.window(300, 400).is_empty());
        assert!(c.window(0, 50).is_empty());
    }

    #[test]
    fn pruning_keeps_the_retention_window() {
        let mut c = NewsCache::new();
        let now = 10 * 3600 * 1000;
        c.insert([item(now - 7 * 3600 * 1000, "old"), item(now - 60_000, "recent")]);
        c.prune(now);
        assert_eq!(c.len(), 1, "older than RETAIN is dropped");
        assert_eq!(c.window(0, now)[0], vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn a_poll_overlaps_so_late_arrivals_are_not_missed() {
        let mut c = NewsCache::new();
        let now = 1_000_000;
        // Empty cache: ask for the whole retention window.
        assert_eq!(c.poll_from_ms(now), now - RETAIN.as_millis() as i64);
        c.insert([item(now - 5_000, "a")]);
        let from = c.poll_from_ms(now);
        assert!(from < now - 5_000, "the next poll starts before the newest item seen: {from}");
        assert_eq!(from, now - 5_000 - 2 * POLL.as_millis() as i64);
    }

    #[test]
    fn an_empty_vector_is_not_cached() {
        let mut c = NewsCache::new();
        assert_eq!(c.insert([Item { ts_ms: 1, doc_id: "e".into(), vector: vec![] }]), 0);
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn timestamps_parse_from_the_stores_rendering() {
        assert_eq!(parse_ms("2026-11-17T00:00:00Z"), Some(1_794_873_600_000));
        assert_eq!(parse_ms("not a time"), None);
    }
}
