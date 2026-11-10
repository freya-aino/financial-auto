//! Shared WebSocket plumbing for `/ws/bus`, `/ws/latent`, `/ws/evolution` and `/ws/trading`
//! (spec 50 §4.8), plus the `/ws/trading` handler itself.
//!
//! Every socket runs the same loop ([`serve`]): a source task feeds `(topic, data)` items into
//! a channel, a [`Coalescer`] holds them between flushes (latest value per topic for the state
//! streams, a bounded queue for the bus tap), and every flush tick sends the pending frames as
//! `{stream, seq, ts, topic, data}` envelopes followed by `{op: "lagged", dropped}` when
//! anything was dropped since the last notice. Clients may send
//! `{op: "subscribe" | "unsubscribe", topics: [...]}` to filter topics (prefix match on `.`
//! boundaries, NATS-style `>` wildcard); no filter = everything.
use std::{collections::HashMap, time::Duration};

use axum::{
    Router,
    extract::{
        State,
        ws::{Message as WsMessage, WebSocket, WebSocketUpgrade},
    },
    response::Response,
    routing::get,
};
use chrono::Utc;
use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{broadcast, mpsc};

use crate::{live::LiveEvent, state::AppState};

/// Capacity of the per-socket source channel.
const SOURCE_CAPACITY: usize = 1024;
/// Flush rate of the bus tap (its per-second budget is spread over these ticks).
pub const BUS_TICKS_PER_SEC: u32 = 20;

/// `/ws/trading` (the other sockets are registered by their own modules).
pub fn router() -> Router<AppState> {
    Router::new().route("/ws/trading", get(ws_trading))
}

/// One item from a socket's source.
#[derive(Debug, Clone, PartialEq)]
pub enum SourceItem {
    Frame(String, Value),
    /// The source itself skipped `n` messages (e.g. a lagging broadcast receiver).
    Lagged(u64),
}

/// Holds frames between flushes. Pure: no clock, no I/O.
#[derive(Debug, Clone)]
pub struct Coalescer {
    latest: bool,
    budget: usize,
    pending: Vec<(String, Value)>,
    index: HashMap<String, usize>,
    dropped: u64,
}

impl Coalescer {
    /// Latest value per topic (state streams): a newer frame replaces the pending one for the
    /// same topic, counted as one dropped frame.
    pub fn latest_value() -> Self {
        Self { latest: true, budget: usize::MAX, pending: Vec::new(), index: HashMap::new(), dropped: 0 }
    }

    /// Bounded queue (bus tap): at most `budget_per_flush` frames per flush, the rest dropped.
    pub fn queue(budget_per_flush: usize) -> Self {
        Self { latest: false, budget: budget_per_flush.max(1), pending: Vec::new(), index: HashMap::new(), dropped: 0 }
    }

    pub fn push(&mut self, item: SourceItem) {
        match item {
            SourceItem::Lagged(n) => self.dropped += n,
            SourceItem::Frame(topic, data) => {
                if self.latest {
                    if let Some(&i) = self.index.get(&topic) {
                        self.pending[i].1 = data;
                        self.dropped += 1;
                    } else {
                        self.index.insert(topic.clone(), self.pending.len());
                        self.pending.push((topic, data));
                    }
                } else if self.pending.len() >= self.budget {
                    self.dropped += 1;
                } else {
                    self.pending.push((topic, data));
                }
            }
        }
    }

    /// Pending frames in arrival order and the drop count since the previous flush.
    pub fn flush(&mut self) -> (Vec<(String, Value)>, u64) {
        self.index.clear();
        (std::mem::take(&mut self.pending), std::mem::take(&mut self.dropped))
    }
}

/// The pure step on its own: feeds `items` and flushes once.
#[cfg(test)]
pub fn coalesce(mut c: Coalescer, items: impl IntoIterator<Item = SourceItem>) -> (Vec<(String, Value)>, u64) {
    for i in items {
        c.push(i);
    }
    c.flush()
}

/// Client-side topic filter.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Topics {
    filters: Vec<String>,
}

impl Topics {
    pub fn allows(&self, topic: &str) -> bool {
        self.filters.is_empty() || self.filters.iter().any(|f| topic_matches(f, topic))
    }

    /// Applies a client op; returns the acknowledgement frame, or an error frame for bad input.
    pub fn apply(&mut self, text: &str) -> Value {
        #[derive(Deserialize)]
        struct ClientOp {
            op: String,
            #[serde(default)]
            topics: Vec<String>,
        }
        let Ok(op) = serde_json::from_str::<ClientOp>(text) else {
            return json!({ "op": "error", "error": "expected {op: subscribe|unsubscribe, topics: [...]}" });
        };
        match op.op.as_str() {
            "subscribe" => {
                for t in op.topics.into_iter().map(|t| t.trim().to_owned()).filter(|t| !t.is_empty()) {
                    if !self.filters.contains(&t) {
                        self.filters.push(t);
                    }
                }
            }
            "unsubscribe" => self.filters.retain(|f| !op.topics.contains(f)),
            other => return json!({ "op": "error", "error": format!("unknown op {other:?}") }),
        }
        json!({ "op": "subscribed", "topics": self.filters })
    }
}

/// `filter` matches `topic` when equal, when `topic` continues it after a `.`, or when
/// `filter` ends in `>` and `topic` starts with the part before it.
pub fn topic_matches(filter: &str, topic: &str) -> bool {
    if filter == ">" || filter == "*" || filter == topic {
        return true;
    }
    if let Some(prefix) = filter.strip_suffix('>') {
        return topic.starts_with(prefix);
    }
    topic.len() > filter.len() && topic.starts_with(filter) && topic.as_bytes()[filter.len()] == b'.'
}

/// The frame envelope of spec 50 §4.8 (`topic` added so clients can route coalesced streams).
pub fn envelope(stream: &str, seq: u64, topic: &str, data: &Value) -> String {
    json!({ "stream": stream, "seq": seq, "ts": Utc::now().to_rfc3339(), "topic": topic, "data": data }).to_string()
}

/// Runs one socket: `initial` frames go out on the first flush, then items from `source` until
/// the source closes or the client leaves. `flush_every` bounds the frame rate.
pub async fn serve(
    socket: WebSocket,
    stream: &'static str,
    initial: Vec<(String, Value)>,
    mut source: mpsc::Receiver<SourceItem>,
    mut coalescer: Coalescer,
    flush_every: Duration,
) {
    let (mut tx, mut rx) = socket.split();
    let mut topics = Topics::default();
    let mut tick = tokio::time::interval(flush_every);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut seq = 0u64;
    for (topic, data) in initial {
        coalescer.push(SourceItem::Frame(topic, data));
    }
    let hello = json!({ "op": "hello", "stream": stream, "flush_ms": flush_every.as_millis() as u64 });
    if tx.send(WsMessage::text(hello.to_string())).await.is_err() {
        return;
    }
    loop {
        tokio::select! {
            item = source.recv() => {
                match item {
                    None => break,
                    Some(SourceItem::Frame(topic, data)) => {
                        if topics.allows(&topic) {
                            coalescer.push(SourceItem::Frame(topic, data));
                        }
                    }
                    Some(lag) => coalescer.push(lag),
                }
            }
            _ = tick.tick() => {
                let (frames, dropped) = coalescer.flush();
                for (topic, data) in frames {
                    seq += 1;
                    if tx.send(WsMessage::text(envelope(stream, seq, &topic, &data))).await.is_err() {
                        return;
                    }
                }
                if dropped > 0 {
                    let lag = json!({ "op": "lagged", "dropped": dropped });
                    if tx.send(WsMessage::text(lag.to_string())).await.is_err() {
                        return;
                    }
                }
            }
            incoming = rx.next() => {
                match incoming {
                    None | Some(Err(_)) | Some(Ok(WsMessage::Close(_))) => break,
                    Some(Ok(WsMessage::Text(t))) => {
                        let ack = topics.apply(t.as_str());
                        if tx.send(WsMessage::text(ack.to_string())).await.is_err() {
                            return;
                        }
                    }
                    Some(Ok(_)) => {}
                }
            }
        }
    }
    let _ = tx.close().await;
}

/// A source channel of the standard capacity.
pub fn source_channel() -> (mpsc::Sender<SourceItem>, mpsc::Receiver<SourceItem>) {
    mpsc::channel(SOURCE_CAPACITY)
}

/// Forwards `Live` change events of one `stream` that pass `keep` into a source channel; ends
/// when the socket (receiver) is gone.
pub fn spawn_live_source(
    mut events: broadcast::Receiver<LiveEvent>,
    stream: &'static str,
    keep: impl Fn(&LiveEvent) -> bool + Send + 'static,
) -> mpsc::Receiver<SourceItem> {
    let (tx, rx) = source_channel();
    tokio::spawn(async move {
        loop {
            let item = match events.recv().await {
                Ok(ev) if ev.stream == stream && keep(&ev) => SourceItem::Frame(ev.topic, ev.data),
                Ok(_) => continue,
                Err(broadcast::error::RecvError::Lagged(n)) => SourceItem::Lagged(n),
                Err(broadcast::error::RecvError::Closed) => break,
            };
            if tx.send(item).await.is_err() {
                break;
            }
        }
    });
    rx
}

/// Flush interval for a rate in Hz (at least 1 Hz).
pub fn flush_interval(hz: u32) -> Duration {
    Duration::from_millis(1000 / u64::from(hz.max(1)))
}

/// `WS /ws/trading`: positions, risk state, halts, orders and fills as a latest-value stream
/// (spec 50 §4.7), coalesced to `runtime.ws.trading_hz` (default 5 Hz).
pub async fn ws_trading(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| async move {
        let (initial, events, hz) = {
            let live = state.live();
            (live.trading_frames(), live.subscribe(), live.runtime().ws.trading_hz)
        };
        let source = spawn_live_source(events, "trading", |_| true);
        serve(socket, "trading", initial, source, Coalescer::latest_value(), flush_interval(hz)).await;
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(topic: &str, v: i64) -> SourceItem {
        SourceItem::Frame(topic.into(), json!(v))
    }

    #[test]
    fn latest_value_keeps_one_frame_per_topic_in_arrival_order() {
        let items = [frame("a", 1), frame("b", 1), frame("a", 2), SourceItem::Lagged(3), frame("a", 3), frame("c", 1)];
        let (frames, dropped) = coalesce(Coalescer::latest_value(), items);
        assert_eq!(frames, vec![("a".into(), json!(3)), ("b".into(), json!(1)), ("c".into(), json!(1))]);
        assert_eq!(dropped, 2 + 3, "two superseded frames plus the source lag");
        // a second flush starts clean
        let mut c = Coalescer::latest_value();
        c.push(frame("a", 1));
        c.flush();
        c.push(frame("a", 2));
        assert_eq!(c.flush(), (vec![("a".into(), json!(2))], 0));
    }

    #[test]
    fn queue_mode_enforces_a_budget_per_flush() {
        let items = (0..7).map(|i| frame("md.trade.x", i));
        let (frames, dropped) = coalesce(Coalescer::queue(5), items);
        assert_eq!(frames.len(), 5);
        assert_eq!(frames[4].1, json!(4), "earliest frames win, later ones are dropped");
        assert_eq!(dropped, 2);
        assert_eq!(Coalescer::queue(0).budget, 1);
    }

    #[test]
    fn topic_filters_follow_dot_boundaries_and_wildcards() {
        assert!(topic_matches("state", "state.snap.inst"));
        assert!(!topic_matches("state", "statement.x"));
        assert!(topic_matches("md.>", "md.trade.sim.a"));
        assert!(topic_matches(">", "anything"));
        assert!(topic_matches("risk", "risk"));
        assert!(!topic_matches("risk.state", "risk"));
        let mut t = Topics::default();
        assert!(t.allows("whatever"));
        let ack = t.apply(r#"{"op":"subscribe","topics":["positions"," fill.>","positions"]}"#);
        assert_eq!(ack["topics"], json!(["positions", "fill.>"]));
        assert!(t.allows("positions") && t.allows("fill.i1") && !t.allows("risk"));
        t.apply(r#"{"op":"unsubscribe","topics":["positions"]}"#);
        assert!(!t.allows("positions") && t.allows("fill.x"));
        assert_eq!(t.apply("nonsense")["op"], "error");
        assert_eq!(t.apply(r#"{"op":"dance"}"#)["op"], "error");
    }

    /// Real socket round trip: `/ws/trading` and `/ws/latent` served on an ephemeral port, read
    /// with `tokio-tungstenite`.
    #[tokio::test]
    async fn sockets_send_hello_initial_frames_and_live_updates() {
        use fa_proto::{
            Message,
            v1::{ExpertLatent, Halt, LatentState},
        };
        use futures::StreamExt;
        use tokio_tungstenite::tungstenite::Message as TMsg;

        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        let cfg = fa_core::config::PlatformConfig::load(format!("{root}/platform.toml")).unwrap();
        let state = AppState::new(cfg, format!("{root}/sources.yaml").into(), None);
        let app = router()
            .merge(crate::latent_routes::router())
            .merge(crate::evolution_routes::router())
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        async fn next_json(
            ws: &mut (impl StreamExt<Item = Result<TMsg, tokio_tungstenite::tungstenite::Error>> + Unpin),
        ) -> Value {
            loop {
                match tokio::time::timeout(Duration::from_secs(5), ws.next()).await.expect("frame in time") {
                    Some(Ok(TMsg::Text(t))) => return serde_json::from_str(&t).unwrap(),
                    Some(Ok(_)) => continue,
                    other => panic!("socket ended: {other:?}"),
                }
            }
        }

        // trading: hello, then the initial positions/risk/halts frames, then a live halt update
        let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws/trading")).await.unwrap();
        let hello = next_json(&mut ws).await;
        assert_eq!(hello["op"], "hello");
        assert_eq!(hello["stream"], "trading");
        assert_eq!(hello["flush_ms"], 200);
        let mut topics = Vec::new();
        for _ in 0..3 {
            let f = next_json(&mut ws).await;
            assert_eq!(f["stream"], "trading");
            topics.push(f["topic"].as_str().unwrap().to_owned());
        }
        assert_eq!(topics, ["positions", "risk", "halts"]);
        state.live_mut().on_halt(Halt { scope: "portfolio".into(), reason: "test".into(), ..Default::default() });
        let f = next_json(&mut ws).await;
        assert_eq!(f["topic"], "halts");
        assert_eq!(f["data"]["halts"][0]["scope"], "portfolio");
        assert!(f["seq"].as_u64().unwrap() >= 4);

        // latent: filtered by snapshot + instrument; a non-matching state is not forwarded
        let (mut ws, _) =
            tokio_tungstenite::connect_async(format!("ws://{addr}/ws/latent?snapshot=snapA&instruments=i1"))
                .await
                .unwrap();
        assert_eq!(next_json(&mut ws).await["op"], "hello");
        let st = |snap: &str, inst: &str| LatentState {
            snapshot_id: snap.into(),
            timescale: "1h".into(),
            instrument_id: inst.into(),
            experts: vec![ExpertLatent {
                expert_index: 0,
                mean: vec![1.0],
                region_id: "r".into(),
                ..Default::default()
            }],
            router_weights: vec![1.0],
            ..Default::default()
        };
        state.live_mut().ingest("latent.state.1h.i2", &st("snapA", "i2").encode_to_vec());
        state.live_mut().ingest("latent.state.1h.i1", &st("snapB", "i1").encode_to_vec());
        state.live_mut().ingest("latent.state.1h.i1", &st("snapA", "i1").encode_to_vec());
        let f = next_json(&mut ws).await;
        assert_eq!(f["topic"], "state.snapA.i1");
        assert_eq!(f["data"]["experts"][0]["region_id"], "r");
        // client-side subscribe narrows further and is acknowledged
        use futures::SinkExt;
        ws.send(TMsg::text(r#"{"op":"subscribe","topics":["region"]}"#)).await.unwrap();
        let ack = next_json(&mut ws).await;
        assert_eq!(ack["op"], "subscribed");
        assert_eq!(ack["topics"], json!(["region"]));
        let _ = ws.close(None).await;
    }

    /// Live smoke against a running control-api: `FA_WS_SMOKE=ws://127.0.0.1:7703 cargo test -p
    /// control-api live_sockets -- --nocapture`. Prints the first frames of each socket.
    #[tokio::test]
    async fn live_sockets_smoke() {
        use futures::StreamExt;
        use tokio_tungstenite::tungstenite::Message as TMsg;
        let Ok(base) = std::env::var("FA_WS_SMOKE") else { return };
        for path in ["/ws/bus?subjects=md.%3E", "/ws/trading", "/ws/latent?snapshot=stat-v1-1s", "/ws/evolution"] {
            let (mut ws, _) = tokio_tungstenite::connect_async(format!("{base}{path}")).await.expect(path);
            let mut n = 0;
            while n < 3 {
                match tokio::time::timeout(Duration::from_secs(8), ws.next()).await {
                    Ok(Some(Ok(TMsg::Text(t)))) => {
                        let v: Value = serde_json::from_str(&t).unwrap();
                        let short = if v["op"].is_string() {
                            v.to_string()
                        } else {
                            format!(
                                "stream={} seq={} topic={} data_keys={:?}",
                                v["stream"],
                                v["seq"],
                                v["topic"],
                                v["data"]
                                    .as_object()
                                    .map(|o| o.keys().cloned().collect::<Vec<_>>())
                                    .unwrap_or_default()
                            )
                        };
                        println!("{path} <- {short}");
                        n += 1;
                    }
                    Ok(Some(Ok(_))) => {}
                    other => {
                        println!("{path} ended/timeout: {other:?}");
                        break;
                    }
                }
            }
            let _ = ws.close(None).await;
        }
    }

    #[test]
    fn envelope_shape_and_flush_interval() {
        let v: Value = serde_json::from_str(&envelope("latent", 7, "state.s.i", &json!({"x": 1}))).unwrap();
        assert_eq!(v["stream"], "latent");
        assert_eq!(v["seq"], 7);
        assert_eq!(v["topic"], "state.s.i");
        assert_eq!(v["data"]["x"], 1);
        assert!(v["ts"].as_str().unwrap().contains('T'));
        assert_eq!(flush_interval(10), Duration::from_millis(100));
        assert_eq!(flush_interval(0), Duration::from_secs(1));
    }
}
