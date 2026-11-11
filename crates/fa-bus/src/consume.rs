//! Durable live consumers with backpressure and quote coalescing (spec 21 §5 consumers, §6
//! "Backpressure", §7 `consumer.max_ack_pending`, open question 2).
//!
//! A [`LiveConsumer`] is a durable, explicit-ack pull consumer on a limits stream that starts at
//! new messages and resumes where it left off after a restart. `max_ack_pending` bounds what is
//! delivered but not yet acked, so a slow consumer stops the server from pushing more instead of
//! growing a client-side queue; the backlog stays in the stream. When the backlog behind the
//! batch exceeds `behind_threshold`, subjects of the [`CoalescePolicy`] are reduced to their
//! latest message per subject within the batch (the superseded ones are acked unprocessed).
//! Coalescing is allowed only for `md.quote.*` and `md.book.*` — a top-of-book consumer needs
//! only the current quote — and never for trades, bars, latents, signals, `order.*` or `risk.*`
//! (decision for Q2): [`CoalescePolicy::new`] refuses such prefixes.
use std::{collections::HashMap, time::Duration};

use async_nats::jetstream::{
    self,
    consumer::{AckPolicy, DeliverPolicy, pull},
};
use futures::StreamExt;

/// Subjects whose messages may be coalesced to latest-only.
pub const COALESCABLE: &[&str] = &["md.quote.", "md.book."];
/// Subject roots that must never be coalesced.
pub const NEVER_COALESCE: &[&str] = &["order.", "risk.", "md.trade.", "md.bar.", "jobs.", "signal.", "latent."];

/// Default `max_ack_pending` for market-data consumers (spec 21 §7).
pub const MD_MAX_ACK_PENDING: i64 = 10_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoalescePolicy {
    prefixes: Vec<String>,
}

impl Default for CoalescePolicy {
    fn default() -> Self {
        Self { prefixes: COALESCABLE.iter().map(|s| (*s).to_owned()).collect() }
    }
}

impl CoalescePolicy {
    /// A policy over `prefixes`; refuses any prefix that could match a never-coalesce subject.
    /// (A prefix may carry a namespace in front, e.g. a test stream's `t1.md.quote.`.)
    pub fn new(prefixes: &[&str]) -> Result<Self, String> {
        for p in prefixes {
            let tail_ok = COALESCABLE.iter().any(|c| p.ends_with(c) || p.contains(c));
            if !tail_ok || NEVER_COALESCE.iter().any(|n| p.ends_with(n)) {
                return Err(format!("coalescing {p} is not allowed (spec 21 Q2: md.quote/md.book only)"));
            }
        }
        Ok(Self { prefixes: prefixes.iter().map(|s| (*s).to_owned()).collect() })
    }

    /// No coalescing at all.
    pub fn none() -> Self {
        Self { prefixes: Vec::new() }
    }

    pub fn applies(&self, subject: &str) -> bool {
        self.prefixes.iter().any(|p| subject.starts_with(p.as_str()))
    }
}

/// Splits a batch into (kept, superseded): for coalescable subjects only the last message per
/// subject is kept; everything else is kept as is. Order of the kept items is preserved.
pub fn coalesce<T>(items: Vec<T>, policy: &CoalescePolicy, subject: impl Fn(&T) -> &str) -> (Vec<T>, Vec<T>) {
    let mut last: HashMap<String, usize> = HashMap::new();
    for (i, it) in items.iter().enumerate() {
        let s = subject(it);
        if policy.applies(s) {
            last.insert(s.to_owned(), i);
        }
    }
    let (mut kept, mut dropped) = (Vec::new(), Vec::new());
    for (i, it) in items.into_iter().enumerate() {
        let s = subject(&it);
        if policy.applies(s) && last.get(s) != Some(&i) {
            dropped.push(it);
        } else {
            kept.push(it);
        }
    }
    (kept, dropped)
}

/// Settings of a [`LiveConsumer`].
#[derive(Debug, Clone)]
pub struct LiveSpec {
    pub stream: String,
    pub durable: String,
    pub filter_subjects: Vec<String>,
    pub max_ack_pending: i64,
    pub ack_wait: Duration,
    /// Messages pulled per batch (capped at `max_ack_pending`).
    pub batch: usize,
    /// Coalesce when more than this many messages wait behind the batch.
    pub behind_threshold: u64,
    pub coalesce: CoalescePolicy,
}

impl LiveSpec {
    /// Market-data defaults: `max_ack_pending` 10 000, batches of 500, coalesce quotes/books
    /// when more than 1 000 messages are waiting.
    pub fn market_data(stream: &str, durable: &str, filters: &[&str]) -> Self {
        Self {
            stream: stream.into(),
            durable: durable.into(),
            filter_subjects: filters.iter().map(|s| (*s).to_owned()).collect(),
            max_ack_pending: MD_MAX_ACK_PENDING,
            ack_wait: Duration::from_secs(30),
            batch: 500,
            behind_threshold: 1_000,
            coalesce: CoalescePolicy::default(),
        }
    }

    pub fn config(&self) -> pull::Config {
        pull::Config {
            durable_name: Some(self.durable.clone()),
            deliver_policy: DeliverPolicy::New,
            ack_policy: AckPolicy::Explicit,
            ack_wait: self.ack_wait,
            max_ack_pending: self.max_ack_pending,
            filter_subjects: self.filter_subjects.clone(),
            ..Default::default()
        }
    }
}

/// Counters of a [`LiveConsumer`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LiveStats {
    pub delivered: u64,
    pub coalesced: u64,
}

pub struct LiveConsumer {
    spec: LiveSpec,
    consumer: jetstream::consumer::Consumer<pull::Config>,
    pub stats: LiveStats,
}

#[derive(Debug, thiserror::Error)]
#[error("live consumer {durable}: {msg}")]
pub struct LiveError {
    pub durable: String,
    pub msg: String,
}

impl LiveConsumer {
    /// Creates (or updates) the durable consumer.
    pub async fn ensure(js: &jetstream::Context, spec: LiveSpec) -> Result<Self, LiveError> {
        let err = |msg: String| LiveError { durable: spec.durable.clone(), msg };
        let stream = js.get_stream(&spec.stream).await.map_err(|e| err(e.to_string()))?;
        let consumer = stream.create_consumer(spec.config()).await.map_err(|e| err(e.to_string()))?;
        Ok(Self { spec, consumer, stats: LiveStats::default() })
    }

    pub fn consumer(&self) -> &jetstream::consumer::Consumer<pull::Config> {
        &self.consumer
    }

    /// Pulls up to one batch (waiting at most `wait`), coalesces it when behind and acks the
    /// superseded messages. The caller acks every returned message after handling it.
    pub async fn next_batch(&mut self, wait: Duration) -> Result<Vec<jetstream::Message>, LiveError> {
        let n = self.spec.batch.min(self.spec.max_ack_pending.max(1) as usize);
        let err = |msg: String| LiveError { durable: self.spec.durable.clone(), msg };
        let mut batch = self
            .consumer
            .fetch()
            .max_messages(n)
            .expires(wait)
            .messages()
            .await
            .map_err(|e| err(e.to_string()))?;
        let mut msgs = Vec::new();
        while let Some(m) = batch.next().await {
            match m {
                Ok(m) => msgs.push(m),
                Err(e) => return Err(err(e.to_string())),
            }
        }
        let behind = msgs.last().and_then(|m| m.info().ok()).map(|i| i.pending).unwrap_or(0);
        let (kept, superseded) = if behind > self.spec.behind_threshold {
            coalesce(msgs, &self.spec.coalesce, |m| m.subject.as_str())
        } else {
            (msgs, Vec::new())
        };
        for m in &superseded {
            let _ = m.ack().await;
        }
        self.stats.coalesced += superseded.len() as u64;
        self.stats.delivered += kept.len() as u64;
        Ok(kept)
    }
}

#[cfg(test)]
mod tests {
    use async_nats::jetstream::stream;

    use super::*;

    #[test]
    fn coalesces_quotes_only_and_keeps_order() {
        let items = vec![
            "md.quote.k.a",
            "md.trade.k.a",
            "md.quote.k.b",
            "md.quote.k.a",
            "order.event.paper",
            "md.book.k.a",
            "md.quote.k.a",
            "md.trade.k.a",
            "md.book.k.a",
        ]
        .into_iter()
        .enumerate()
        .collect::<Vec<_>>();
        let (kept, dropped) = coalesce(items, &CoalescePolicy::default(), |(_, s)| s);
        let kept: Vec<usize> = kept.iter().map(|(i, _)| *i).collect();
        assert_eq!(kept, [1, 2, 4, 6, 7, 8], "latest quote per subject, every trade and order");
        assert_eq!(dropped.len(), 3);
        let (all, none) = coalesce(vec!["md.quote.k.a", "md.quote.k.a"], &CoalescePolicy::none(), |s| s);
        assert_eq!((all.len(), none.len()), (2, 0));
    }

    #[test]
    fn policy_refuses_orders_and_trades() {
        assert!(CoalescePolicy::new(&["order."]).is_err());
        assert!(CoalescePolicy::new(&["md.trade."]).is_err());
        assert!(CoalescePolicy::new(&["md."]).is_err());
        assert!(CoalescePolicy::new(&["md.quote.", "t1.md.book."]).is_ok());
        let c = LiveSpec::market_data("MD", "x", &["md.quote.>"]).config();
        assert_eq!((c.max_ack_pending, c.ack_policy, c.deliver_policy), (10_000, AckPolicy::Explicit, DeliverPolicy::New));
    }

    /// `FA_NATS_TEST=1 cargo test -p fa-bus live_backpressure -- --nocapture`: a scratch stream
    /// with 2 000 quotes on 5 subjects and 50 trades; a slow durable consumer with
    /// `max_ack_pending` 100 never holds more than 100 unacked, sees every trade in order and the
    /// final quote of every subject, and skips most intermediate quotes.
    #[tokio::test]
    async fn live_backpressure_and_coalescing() {
        if std::env::var("FA_NATS_TEST").is_err() {
            eprintln!("skipping live JetStream test (set FA_NATS_TEST=1)");
            return;
        }
        let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
        let client = async_nats::connect(&url).await.unwrap();
        let js = jetstream::new(client.clone());
        let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() % 1_000_000_000;
        let (name, root) = (format!("FABUSBP{n}"), format!("fabustest.bp{n}"));
        js.create_stream(stream::Config { name: name.clone(), subjects: vec![format!("{root}.>")], ..Default::default() })
            .await
            .unwrap();
        let mut spec = LiveSpec::market_data(&name, "slow", &[&format!("{root}.md.>")]);
        spec.max_ack_pending = 100;
        spec.batch = 100;
        spec.behind_threshold = 50;
        spec.coalesce = CoalescePolicy::new(&[&format!("{root}.md.quote.")]).unwrap();
        let mut live = LiveConsumer::ensure(&js, spec).await.unwrap();
        let mut acks = Vec::new();
        for i in 0..2_000u32 {
            let s = format!("{root}.md.quote.k.i{}", i % 5);
            acks.push(js.publish(s, i.to_string().into()).await.unwrap());
            if i % 40 == 0 {
                acks.push(js.publish(format!("{root}.md.trade.k.i0"), (i / 40).to_string().into()).await.unwrap());
            }
        }
        for a in acks {
            a.await.unwrap();
        }
        let mut trades = Vec::new();
        let mut last_quote: HashMap<String, u32> = HashMap::new();
        let mut max_ack_pending_seen = 0;
        loop {
            let batch = live.next_batch(Duration::from_millis(500)).await.unwrap();
            if batch.is_empty() {
                break;
            }
            let info = live.consumer.info().await.unwrap();
            max_ack_pending_seen = max_ack_pending_seen.max(info.num_ack_pending);
            for m in batch {
                let v: u32 = std::str::from_utf8(&m.payload).unwrap().parse().unwrap();
                if m.subject.contains(".md.trade.") {
                    trades.push(v);
                } else {
                    let prev = last_quote.insert(m.subject.to_string(), v);
                    assert!(prev.is_none_or(|p| p < v), "quotes stay in order");
                }
                tokio::time::sleep(Duration::from_micros(200)).await; // slow handler
                m.ack().await.unwrap();
            }
        }
        eprintln!("stats {:?}, max ack pending seen {max_ack_pending_seen}", live.stats);
        assert!(max_ack_pending_seen <= 100, "{max_ack_pending_seen}");
        assert_eq!(trades, (0..50).collect::<Vec<_>>(), "every trade, in order");
        for k in 0..5 {
            assert_eq!(last_quote[&format!("{root}.md.quote.k.i{k}")], 1_995 + k, "final quote of every subject");
        }
        assert!(live.stats.coalesced > 1_000, "{:?}", live.stats);
        assert_eq!(live.stats.coalesced + live.stats.delivered, 2_050);
        js.delete_stream(&name).await.unwrap();
    }
}
