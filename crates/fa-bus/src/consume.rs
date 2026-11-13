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
    /// Resume an existing consumer only when it acked something within this window; after a
    /// longer outage the backlog is stale for a live decision path, so the consumer is recreated
    /// at new messages instead (`None` = always resume, e.g. `order.*` booking).
    pub resume_within: Option<Duration>,
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
            resume_within: None,
        }
    }

    /// Defaults for a durable consumer of a non-market-data stream (latents, signals, orders):
    /// `max_ack_pending` 1 000, batches of 100, never coalesced.
    pub fn durable(stream: &str, durable: &str, filters: &[&str]) -> Self {
        Self {
            max_ack_pending: 1_000,
            batch: 100,
            coalesce: CoalescePolicy::none(),
            ..Self::market_data(stream, durable, filters)
        }
    }

    /// The stream holding `subject` in the spec 21 §5 table (e.g. `md.trade.>` → `MD`).
    pub fn stream_of(subject: &str) -> Option<String> {
        crate::streams::table()
            .streams
            .into_iter()
            .find(|s| s.subjects.iter().any(|f| crate::streams::subjects_overlap(f, subject)))
            .map(|s| s.name)
    }

    pub fn with_resume_within(mut self, window: Duration) -> Self {
        self.resume_within = Some(window);
        self
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
        if let Some(window) = spec.resume_within
            && let Ok(mut old) = stream.get_consumer::<pull::Config>(&spec.durable).await
            && let Ok(info) = old.info().await
            && info.num_pending > 0
            && !recently_active(
                info.ack_floor.last_active.or(info.delivered.last_active).map(|t| t.unix_timestamp()),
                window,
            )
        {
            tracing::warn!(durable = %spec.durable, backlog = info.num_pending, "consumer idle longer than resume_within; skipping its backlog");
            stream.delete_consumer(&spec.durable).await.map_err(|e| err(e.to_string()))?;
        }
        let consumer = stream.create_consumer(spec.config()).await.map_err(|e| err(e.to_string()))?;
        crate::streams::spawn_consumer_lag(consumer.clone());
        Ok(Self { spec, consumer, stats: LiveStats::default() })
    }

    pub fn consumer(&self) -> &jetstream::consumer::Consumer<pull::Config> {
        &self.consumer
    }

    /// Pulls up to one batch — the messages available now, or else the first one to arrive
    /// within `wait` — coalesces it when behind and acks the superseded messages. The caller
    /// acks every returned message after handling it.
    pub async fn next_batch(&mut self, wait: Duration) -> Result<Vec<jetstream::Message>, LiveError> {
        let n = self.spec.batch.min(self.spec.max_ack_pending.max(1) as usize);
        let err = |msg: String| LiveError { durable: self.spec.durable.clone(), msg };
        // what is there now, without waiting (`fetch` is no-wait) …
        let batch = self.consumer.fetch().max_messages(n).messages().await.map_err(|e| err(e.to_string()))?;
        let mut msgs = collect(batch).await.map_err(err)?;
        if msgs.is_empty() {
            // … else block for the first message, so an idle consumer neither spins nor waits
            // for a full batch while its messages' ack_wait runs
            let batch =
                self.consumer.batch().max_messages(1).expires(wait).messages().await.map_err(|e| err(e.to_string()))?;
            msgs = collect(batch).await.map_err(err)?;
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

async fn collect(mut batch: pull::Batch) -> Result<Vec<jetstream::Message>, String> {
    let mut msgs = Vec::new();
    while let Some(m) = batch.next().await {
        msgs.push(m.map_err(|e| e.to_string())?);
    }
    Ok(msgs)
}

/// `last_unix_s` (seconds) lies within `window` of now; unknown counts as idle.
fn recently_active(last_unix_s: Option<i64>, window: Duration) -> bool {
    last_unix_s.is_some_and(|t| chrono::Utc::now().timestamp() - t <= window.as_secs() as i64)
}

/// A durable consumer as a stream of raw messages, contract-checked (spec 21 §5, §6). A message
/// is acked when the caller asks for the **next** one — i.e. after its handler finished — so a
/// crash mid-handling redelivers it after the restart (at least once; handlers must be
/// idempotent). Fetch errors are logged and retried with a 1 s backoff; the stream never ends.
pub fn raw(live: LiveConsumer) -> futures::stream::BoxStream<'static, async_nats::Message> {
    use std::collections::VecDeque;
    struct State {
        live: LiveConsumer,
        queue: VecDeque<jetstream::Message>,
        prev: Option<jetstream::Message>,
    }
    let state = State { live, queue: VecDeque::new(), prev: None };
    futures::stream::unfold(state, |mut st| async move {
        loop {
            if let Some(done) = st.prev.take() {
                let _ = done.ack().await;
            }
            let Some(m) = st.queue.pop_front() else {
                match st.live.next_batch(Duration::from_secs(5)).await {
                    Ok(b) => st.queue.extend(b),
                    Err(e) => {
                        tracing::warn!(error = %e, "live consumer fetch failed; retrying");
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
                continue;
            };
            let accepted = crate::contract::accept(&m.subject, m.headers.as_ref());
            let msg = m.message.clone();
            st.prev = Some(m);
            if accepted {
                return Some((msg, st));
            }
        }
    })
    .boxed()
}

/// [`raw`] decoded as `M`, with the same shape as [`crate::Bus::subscribe`]: undecodable
/// messages are logged, acked and skipped.
pub fn typed<M: fa_proto::Message + Default + Send + 'static>(
    live: LiveConsumer,
) -> futures::stream::BoxStream<'static, (String, M)> {
    raw(live)
        .filter_map(|m| {
            futures::future::ready(match M::decode(m.payload.as_ref()) {
                Ok(v) => Some((m.subject.to_string(), v)),
                Err(e) => {
                    tracing::warn!(subject = %m.subject, error = %e, "dropping undecodable message");
                    None
                }
            })
        })
        .boxed()
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
        assert_eq!(
            (c.max_ack_pending, c.ack_policy, c.deliver_policy),
            (10_000, AckPolicy::Explicit, DeliverPolicy::New)
        );
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
        js.create_stream(stream::Config {
            name: name.clone(),
            subjects: vec![format!("{root}.>")],
            ..Default::default()
        })
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

    /// `FA_NATS_TEST=1 cargo test -p fa-bus durable_typed -- --nocapture`: a message whose handler
    /// was interrupted (stream dropped before the next `next()`) is redelivered to the restarted
    /// consumer; nothing handled is redelivered; an idle consumer with `resume_within` skips its
    /// stale backlog.
    #[tokio::test]
    async fn durable_typed_redelivers_after_crash_and_skips_stale_backlog() {
        use fa_proto::Message;
        if std::env::var("FA_NATS_TEST").is_err() {
            eprintln!("skipping live JetStream test (set FA_NATS_TEST=1)");
            return;
        }
        let url = std::env::var("FA_NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
        let js = jetstream::new(async_nats::connect(&url).await.unwrap());
        let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() % 1_000_000_000;
        let (name, root) = (format!("FABUSDT{n}"), format!("fabustest.dt{n}"));
        js.create_stream(stream::Config {
            name: name.clone(),
            subjects: vec![format!("{root}.>")],
            ..Default::default()
        })
        .await
        .unwrap();
        let publish = |i: u32| {
            let hb = fa_proto::v1::Heartbeat { service: i.to_string(), ..Default::default() };
            let js = js.clone();
            let subject = format!("{root}.order.event.paper");
            async move { js.publish(subject, hb.encode_to_vec().into()).await.unwrap().await.unwrap() }
        };
        let spec =
            || LiveSpec { ack_wait: Duration::from_secs(2), ..LiveSpec::durable(&name, "dt", &[&format!("{root}.>")]) };
        // created before publishing: DeliverPolicy::New starts at the next message
        let live = LiveConsumer::ensure(&js, spec()).await.unwrap();
        for i in 0..5 {
            publish(i).await;
        }
        let mut sub = typed::<fa_proto::v1::Heartbeat>(live);
        let mut seen = Vec::new();
        for _ in 0..3 {
            seen.push(sub.next().await.unwrap().1.service);
        }
        drop(sub); // "crash" while handling the third message: it was never acked
        assert_eq!(seen, ["0", "1", "2"]);
        let mut sub = typed::<fa_proto::v1::Heartbeat>(LiveConsumer::ensure(&js, spec()).await.unwrap());
        let mut after = Vec::new();
        for _ in 0..3 {
            after.push(sub.next().await.unwrap().1.service);
        }
        assert_eq!(after, ["2", "3", "4"], "the interrupted message comes back, handled ones don't");
        drop(sub);
        // stale backlog: a short resume window and a consumer idle longer than it
        let stale = spec().with_resume_within(Duration::from_millis(1));
        publish(5).await;
        tokio::time::sleep(Duration::from_millis(1_200)).await;
        let live = LiveConsumer::ensure(&js, stale).await.unwrap();
        assert_eq!(live.consumer.clone().info().await.unwrap().num_pending, 0, "backlog skipped");
        js.delete_stream(&name).await.unwrap();
    }
}
